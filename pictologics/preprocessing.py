"""
Image Preprocessing Module
==========================

This module provides a collection of preprocessing functions essential for radiomics
analysis. These functions are designed to be IBSI-compliant where applicable.

Key Features:
-------------
- **Resampling**: Voxel resampling using 'Align grid centers' (IBSI compliant).
- **Discretisation**: Fixed Bin Number (FBN) and Fixed Bin Size (FBS) algorithms.
- **Filtering**: Outlier filtering (mean +/- sigma).
- **Mask Operations**: Resegmentation (thresholding), ROI extraction, Largest Connected Component.
- **Utilities**: Rounding intensities, applying masks.
"""

from __future__ import annotations

import math
import warnings
import weakref
from dataclasses import replace
from functools import lru_cache
from typing import Any, Literal, Optional, cast

import numpy as np
from numba import jit, prange
from numpy import typing as npt
from scipy.ndimage import affine_transform, label

from .features._utils import (
    _SAME_SIZE,
    PRANGE_ONLY,
    compute_nonzero_bbox,
    roi_min_max,
    serial_twin,
    sized,
)
from .filters._passes import fma as _fma
from .loader import Image, _direction_matrix, _row_order, _validate_geometry
from .threads import get_num_threads


def _row_order_copy(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """`array` in row order: a large column-order array of a loader type takes the tiled
    parallel copy of the loader (see _row_order; a bool source mask as its uint8 bytes),
    the others np.ascontiguousarray (no copy of a row-order array)."""
    if array.dtype == np.bool_:
        return _row_order_copy(array.view(np.uint8)).view(np.bool_)
    array = _row_order(array)
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


# Common sentinel values used in medical imaging to denote "no data" or "background"
COMMON_SENTINEL_VALUES: tuple[float, ...] = (
    -2048.0,
    -3024.0,
    -1024.0,
    -1000.0,
    0.0,
    -32768.0,
)

# Below this size the numba parallel-launch overhead exceeds the scan itself;
# the numpy fallback paths are kept for small arrays (same convention as
# features._utils). filter_outliers runs its parallel kernels from this size on and
# makes the new mask with a serial kernel below it; the kernels below also use it
# for arrays that share no memory order (they need a copy); both paths are
# bit-identical, so the switch is purely a performance decision. round_intensities
# also uses it when more than one thread runs: with new memory pages, the kernel is
# faster from about 0.5 million voxels; with pages in the cache, below 1.5 million.
_KERNEL_MIN_SIZE = 1 << 20

# Output voxels from which the resample kernels run in parallel; below, their serial twins
# (measured on 10 threads: starting the threads costs about 70 us).
_NEAREST_PARALLEL_MIN = 160_000
_LINEAR_PARALLEL_MIN = 50_000
_MASKED_PARALLEL_MIN = 36_000
# scipy's affine_transform(order=3, mode='nearest') pads 12 edge voxels before the spline
# prefilter (_prepad_for_spline_filter); the pole of the cubic B-spline filter
_SPLINE_NPAD = 12
_SPLINE_POLE = -0.267949192431122706472553658494127633
# Measured crossovers (14-core M4 Pro): output voxels from which the tricubic kernel runs in
# parallel; padded voxels from which the pad and the prefilter sweeps do
_CUBIC_PARALLEL_MIN = 5_000
_PREFILTER_PARALLEL_MIN = 1 << 17

# Measured crossover (14-core M4 Pro), in voxels: below this size the sentinel kernel's
# start-up cost (0.07-0.2 ms) is more than the numpy code's whole run.
_SENTINEL_KERNEL_MIN_SIZE = 200_000
# Voxels from which the != kernel of a sentinel mask runs in parallel; below, its serial
# twin (measured on 10 threads; the serial twin is faster than numpy at every size)
_NOT_EQUAL_PARALLEL_MIN = 1_500_000
# The parallel gather of ROI values (_threaded_roi_values) needs this many threads. With fewer,
# numpy's gather is faster (measured on 1 M voxels: x1.9 at 1 thread, x1.1 at 2, x0.85 at 4).
_GATHER_MIN_THREADS = 4
# Voxels from which the resegment kernel runs in parallel; below, its serial twin
# (measured on 10 threads: the two take the same time at 0.8 to 1.2 million voxels)
_RESEGMENT_PARALLEL_MIN = 1 << 20


def _shared_order(*arrays: npt.NDArray[Any]) -> Optional[Literal["C", "F"]]:
    """The memory order that all arrays share: "C" (row) or "F" (column), else None.
    In a shared order, ravel and reshape copy nothing."""
    if all(a.flags.c_contiguous for a in arrays):
        return "C"
    if all(a.flags.f_contiguous for a in arrays):
        return "F"
    return None


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------
# Single-pass parallel replacements for the memory-bound numpy chains below.
# The discretise kernels apply the exact same per-element operations (in the
# same order) as the numpy code they shadow, and the resegment kernel keeps or
# zeroes the bits of each mask value, so their output is bit-identical;
# uncommon dtypes fall back to numpy. All signatures used by
# the dispatch code are compiled eagerly in warmup.warmup_jit().


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _nonfinite_blocks_numba(flat: npt.NDArray[np.float64], found: npt.NDArray[np.uint8]) -> None:
    """found[b] = 1 when block b (65,536 values) of `flat` holds a NaN or an infinite value."""
    n = flat.size
    for b in prange(found.size):
        for i in range(b * 65536, min(n, (b + 1) * 65536)):
            if not math.isfinite(flat[i]):
                found[b] = 1
                break


# From this many values on, with 4 threads or more, the finite check of a float64 array runs
# in threads (measured on the M4 Pro). One thread of the kernel reads a value in about 3 times
# the time of np.sum, and the start of the threads costs about 70 us. So np.sum is faster below
# 2^21 values on 10 threads, and at all sizes on 1 to 3 threads.
_FINITE_PARALLEL_MIN = 1 << 21
_FINITE_PARALLEL_THREADS = 4

# The result of _all_finite for each array that lives (key: the id of the array). A weak
# finalizer removes the result when the array dies, so an id names one array.
_FINITE_RESULTS: dict[int, bool] = {}


def _all_finite(array: npt.NDArray[Any]) -> bool:
    """Whether every value of a float array is finite: one parallel pass for a large
    contiguous float64 array on 4 threads or more, else numpy (the sum of an array with a
    NaN or an infinite value is not finite; a sum too large for float64 only costs the
    caller one more check).

    The result stays while the array lives, so the next run on the same image does not read
    it again. This memo assumes that the caller does not change the array in place between
    runs. The package itself never changes an array in place."""
    key = id(array)
    if key in _FINITE_RESULTS:
        return _FINITE_RESULTS[key]
    contiguous = array.flags.c_contiguous or array.flags.f_contiguous
    if (
        array.dtype == np.float64
        and array.size >= _FINITE_PARALLEL_MIN
        and contiguous
        and get_num_threads() >= _FINITE_PARALLEL_THREADS
    ):
        flat = array.ravel(order="K")
        found = np.zeros((flat.size + 65535) // 65536, dtype=np.uint8)
        _nonfinite_blocks_numba(flat, found)
        finite = not found.any()
    else:
        with np.errstate(over="ignore", invalid="ignore"):  # inf + -inf, or a sum too large
            finite = math.isfinite(float(np.sum(array)))
    _FINITE_RESULTS[key] = finite
    weakref.finalize(array, _FINITE_RESULTS.pop, key, None)
    return finite


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _discretise_fbn_numba(
    flat: npt.NDArray[np.float64],
    n_bins: float,
    current_min: float,
    current_max: float,
    out: npt.NDArray[np.int32],
) -> None:
    """FBN: floor(N_g * (x - min) / (max - min)) + 1, clipped to [1, N_g]; NaN -> 0."""
    denom = current_max - current_min
    for i in prange(flat.size):
        t = n_bins * (flat[i] - current_min)
        t = t / denom
        t = np.floor(t) + 1.0
        if t < 1.0:
            t = 1.0
        elif t > n_bins:
            t = n_bins
        if np.isnan(t):
            out[i] = 0
        else:
            out[i] = np.int32(t)


_discretise_fbn_numba_serial = serial_twin(_discretise_fbn_numba)


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _discretise_fbs_numba(
    flat: npt.NDArray[np.float64],
    bin_width: float,
    current_min: float,
    out: npt.NDArray[np.int32],
) -> None:
    """FBS: floor((x - min) / w_b) + 1, clamped to >= 1; NaN and +inf -> 0."""
    for i in prange(flat.size):
        t = (flat[i] - current_min) / bin_width
        t = np.floor(t) + 1.0
        if t < 1.0:
            t = 1.0
        if not np.isfinite(t):
            out[i] = 0
        else:
            out[i] = np.int32(t)


_discretise_fbs_numba_serial = serial_twin(_discretise_fbs_numba)


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _discretise_cutoffs_numba(
    flat: npt.NDArray[np.float64],
    cutoffs: npt.NDArray[np.float64],
    increasing: bool,
    out: npt.NDArray[np.int32],
) -> None:
    """Fixed-cutoff bins, the values of `np.digitize(x, cutoffs) + 1` (NaN -> 0).

    A binary search per voxel: for increasing cutoffs the bin counts the cutoffs <= x
    (digitize's side="right"), for decreasing ones the cutoffs > x. No int64 array,
    no boolean temporary, and no int32 copy.
    """
    n = cutoffs.size
    for i in prange(flat.size):
        x = flat[i]
        if x != x:  # NaN
            out[i] = 0
            continue
        lo = 0
        hi = n
        if increasing:
            while lo < hi:
                mid = (lo + hi) // 2
                if cutoffs[mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if cutoffs[mid] > x:
                    lo = mid + 1
                else:
                    hi = mid
        out[i] = lo + 1


_discretise_cutoffs_numba_serial = serial_twin(_discretise_cutoffs_numba)
# Voxels from which the discretise kernels run in parallel; below, their serial twins
# (measured on 10 threads: the two FBN forms take the same time at 160,000 voxels, the
# FBS forms at about 200,000)
_DISCRETISE_PARALLEL_MIN = 160_000


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _normalise_map_numba(
    flat: npt.NDArray[np.float64], center: float, scale: float, out: npt.NDArray[np.float64]
) -> None:
    """out = (flat - center) / scale: the subtraction and the division of np.subtract and
    np.divide in one pass, so each value is the same."""
    for i in prange(flat.size):
        out[i] = (flat[i] - center) / scale


_normalise_map_numba_serial = serial_twin(_normalise_map_numba)
# Voxels from which the normalise map runs in parallel; below, its serial twin (measured on
# 10 threads: the two take the same time at about 450,000 voxels)
_NORMALISE_PARALLEL_MIN = 1 << 19


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _resegment_numba(
    img_flat: npt.NDArray[np.float64],
    mask_flat: npt.NDArray[Any],
    range_min: float,
    range_max: float,
    out: npt.NDArray[Any],
) -> None:
    """Copy mask, zeroing voxels whose intensity lies outside [range_min, range_max].

    A NaN intensity is in no range, so its voxel leaves the mask, as in the numpy
    fallback. Missing bounds are passed as +/-inf. The mask and out are the integer
    view of the mask (see resegment_mask): a multiply by 1 or 0 keeps the bits or
    gives 0. The compiler vectorises the multiply; it cannot vectorise a branch.
    """
    for i in prange(img_flat.size):
        v = img_flat[i]
        out[i] = mask_flat[i] * ((v >= range_min) & (v <= range_max))


_resegment_numba_serial = serial_twin(_resegment_numba)


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _rint_numba(flat: npt.NDArray[np.float64], out: npt.NDArray[np.float64]) -> None:
    """out = np.round(flat): the nearest whole number, a half to the even number, in
    threads. np.rint gives the same bits as np.round (NaN, infinite values and -0.0 stay)."""
    for i in prange(flat.size):
        out[i] = np.rint(flat[i])


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _resample_nearest_numba(
    src: npt.NDArray[Any],
    scale: npt.NDArray[np.float64],
    shift: npt.NDArray[np.float64],
    start: npt.NDArray[np.int64],
    out: npt.NDArray[Any],
) -> None:
    """Nearest-neighbour resampling for a diagonal transform, 'nearest' boundary.

    Replicates scipy's zoom_shift path (used by affine_transform for 1-D
    matrices) exactly: the coordinate is (idx + shift) * scale with
    shift = offset / scale, clipped to the volume, then floor(coord + 0.5)
    selects the voxel. scipy adds the value to a sum that starts at 0.0, so a
    -0.0 comes out as +0.0. Same expressions and order -> identical output.
    `out` is the region of the output grid that starts at index `start` (the
    index is the grid index, so a region has the values of the whole grid).
    """
    nz, ny, nx = src.shape
    oz, oy, ox = out.shape
    for k in prange(oz):
        zc = ((k + start[0]) + shift[0]) * scale[0]
        if zc < 0.0:
            zc = 0.0
        elif zc > nz - 1:
            zc = float(nz - 1)
        iz = int(np.floor(zc + 0.5))
        for j in range(oy):
            yc = ((j + start[1]) + shift[1]) * scale[1]
            if yc < 0.0:
                yc = 0.0
            elif yc > ny - 1:
                yc = float(ny - 1)
            iy = int(np.floor(yc + 0.5))
            for i in range(ox):
                xc = ((i + start[2]) + shift[2]) * scale[2]
                if xc < 0.0:
                    xc = 0.0
                elif xc > nx - 1:
                    xc = float(nx - 1)
                ix = int(np.floor(xc + 0.5))
                out[k, j, i] = 0.0 + src[iz, iy, ix]


_resample_nearest_numba_serial = serial_twin(_resample_nearest_numba)


@jit(nopython=True, cache=True)  # type: ignore
def _linear_axis_numba(
    n_out: int, n_in: int, shift: float, scale: float, start: int
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Support indices and weights of trilinear resampling along one axis, as scipy's
    zoom_shift computes them in 'nearest' mode.

    The coordinate (idx + shift) * scale is not clipped: the weights come from the
    coordinate itself (w0 = 1 - t, and w1 = 1 - w0 as scipy sets the last weight), and
    only the two support indices are clamped to the volume. idx is the grid index of
    the n_out outputs from `start`.
    """
    lo = np.empty(n_out, dtype=np.int64)
    hi = np.empty(n_out, dtype=np.int64)
    w0 = np.empty(n_out, dtype=np.float64)
    w1 = np.empty(n_out, dtype=np.float64)
    for k in range(n_out):
        c = ((k + start) + shift) * scale
        f = np.floor(c)
        w0[k] = 1.0 - (c - f)
        w1[k] = 1.0 - w0[k]
        lo[k] = min(max(int(f), 0), n_in - 1)
        hi[k] = min(max(int(f) + 1, 0), n_in - 1)
    return lo, hi, w0, w1


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _trilinear_value(
    src: npt.NDArray[Any],
    z0: int,
    z1: int,
    wz0: float,
    wz1: float,
    y0: int,
    y1: int,
    wy0: float,
    wy1: float,
    x0: int,
    x1: int,
    wx0: float,
    wx1: float,
) -> Any:
    """The trilinear value of one output voxel from its 8 support voxels and weights.

    scipy starts the sum at 0.0 (so -0.0 terms give +0.0) and multiplies each coefficient
    by the per-axis weights sequentially (value * wz * wy * wx, left-associated); keep that
    exact order so the result is bit-identical.
    """
    acc = 0.0
    acc += src[z0, y0, x0] * wz0 * wy0 * wx0
    acc += src[z0, y0, x1] * wz0 * wy0 * wx1
    acc += src[z0, y1, x0] * wz0 * wy1 * wx0
    acc += src[z0, y1, x1] * wz0 * wy1 * wx1
    acc += src[z1, y0, x0] * wz1 * wy0 * wx0
    acc += src[z1, y0, x1] * wz1 * wy0 * wx1
    acc += src[z1, y1, x0] * wz1 * wy1 * wx0
    acc += src[z1, y1, x1] * wz1 * wy1 * wx1
    return acc


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _resample_trilinear_numba(
    src: npt.NDArray[Any],
    scale: npt.NDArray[np.float64],
    shift: npt.NDArray[np.float64],
    start: npt.NDArray[np.int64],
    unsigned_out: bool,
    threshold: float,
    rounded: bool,
    out: npt.NDArray[Any],
) -> None:
    """Trilinear resampling for a diagonal transform, 'nearest' boundary.

    Replicates scipy's zoom_shift path bit for bit: the per-axis indices and weights
    of `_linear_axis_numba`, and the 8 support terms added in scipy's point order
    (`_trilinear_value`). A `threshold` (not NaN) stores 1 where the value rounds to 1
    at it, else 0 (see `_mask_threshold`). Else, with `unsigned_out`, the value is
    stored as scipy stores it in an unsigned integer output: t + 0.5 for t > 0, else 0,
    truncated. Else, with `rounded`, the value is rounded to the nearest integer (half
    to even: the bits of np.round). Each row chooses its store rule once, so the loop
    over the row has no branch on the rule.
    """
    nz, ny, nx = src.shape
    oz, oy, ox = out.shape
    zlo, zhi, zw0, zw1 = _linear_axis_numba(oz, nz, shift[0], scale[0], start[0])
    ylo, yhi, yw0, yw1 = _linear_axis_numba(oy, ny, shift[1], scale[1], start[1])
    xlo, xhi, xw0, xw1 = _linear_axis_numba(ox, nx, shift[2], scale[2], start[2])
    for k in prange(oz):
        z0, z1, wz0, wz1 = zlo[k], zhi[k], zw0[k], zw1[k]
        for j in range(oy):
            y0, y1, wy0, wy1 = ylo[j], yhi[j], yw0[j], yw1[j]
            if threshold == threshold:
                for i in range(ox):
                    acc = _trilinear_value(
                        src, z0, z1, wz0, wz1, y0, y1, wy0, wy1, xlo[i], xhi[i], xw0[i], xw1[i]
                    )
                    out[k, j, i] = 1 if acc + (1.0 - threshold) >= 1.0 else 0
            elif unsigned_out:
                for i in range(ox):
                    acc = _trilinear_value(
                        src, z0, z1, wz0, wz1, y0, y1, wy0, wy1, xlo[i], xhi[i], xw0[i], xw1[i]
                    )
                    out[k, j, i] = acc + 0.5 if acc > 0 else 0.0
            elif rounded:
                for i in range(ox):
                    out[k, j, i] = np.rint(
                        _trilinear_value(
                            src, z0, z1, wz0, wz1, y0, y1, wy0, wy1, xlo[i], xhi[i], xw0[i], xw1[i]
                        )
                    )
            else:
                for i in range(ox):
                    out[k, j, i] = _trilinear_value(
                        src, z0, z1, wz0, wz1, y0, y1, wy0, wy1, xlo[i], xhi[i], xw0[i], xw1[i]
                    )


_resample_trilinear_numba_serial = serial_twin(_resample_trilinear_numba)


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _resample_trilinear_masked_numba(
    src: npt.NDArray[np.float64],
    valid: npt.NDArray[np.bool_],
    scale: npt.NDArray[np.float64],
    shift: npt.NDArray[np.float64],
    start: npt.NDArray[np.int64],
    weight_threshold: float,
    out: npt.NDArray[np.float64],
    out_valid: npt.NDArray[np.bool_],
) -> None:
    """Fused normalized (validity-weighted) trilinear resampling.

    Computes in one pass what _resample_with_source_mask obtains from two
    affine_transform calls: per output voxel, the trilinear weights of the 8
    neighbours are accumulated only over valid source voxels; the value is the
    weight-normalized sum, and the voxel is valid iff the accumulated weight
    reaches weight_threshold. Coordinates and weights follow scipy's zoom_shift
    path (see _resample_trilinear_numba).
    """
    nz, ny, nx = src.shape
    oz, oy, ox = out.shape
    zlo, zhi, zw0, zw1 = _linear_axis_numba(oz, nz, shift[0], scale[0], start[0])
    ylo, yhi, yw0, yw1 = _linear_axis_numba(oy, ny, shift[1], scale[1], start[1])
    xlo, xhi, xw0, xw1 = _linear_axis_numba(ox, nx, shift[2], scale[2], start[2])
    for k in prange(oz):
        z0, z1, wz0, tz = zlo[k], zhi[k], zw0[k], zw1[k]
        for j in range(oy):
            y0, y1, wy0, ty = ylo[j], yhi[j], yw0[j], yw1[j]
            for i in range(ox):
                x0, x1, wx0, tx = xlo[i], xhi[i], xw0[i], xw1[i]

                # Accumulation mirrors the two affine_transform calls of the
                # scipy path bit-for-bit: values multiply the per-axis weights
                # sequentially (left-associated), the weight sum matches the
                # transform of the 0/1 mask, and invalid voxels contribute
                # exactly 0 to both sums (as the zeroed image does there).
                acc = 0.0
                wacc = 0.0
                if valid[z0, y0, x0]:
                    acc += src[z0, y0, x0] * wz0 * wy0 * wx0
                    wacc += wz0 * wy0 * wx0
                if valid[z0, y0, x1]:
                    acc += src[z0, y0, x1] * wz0 * wy0 * tx
                    wacc += wz0 * wy0 * tx
                if valid[z0, y1, x0]:
                    acc += src[z0, y1, x0] * wz0 * ty * wx0
                    wacc += wz0 * ty * wx0
                if valid[z0, y1, x1]:
                    acc += src[z0, y1, x1] * wz0 * ty * tx
                    wacc += wz0 * ty * tx
                if valid[z1, y0, x0]:
                    acc += src[z1, y0, x0] * tz * wy0 * wx0
                    wacc += tz * wy0 * wx0
                if valid[z1, y0, x1]:
                    acc += src[z1, y0, x1] * tz * wy0 * tx
                    wacc += tz * wy0 * tx
                if valid[z1, y1, x0]:
                    acc += src[z1, y1, x0] * tz * ty * wx0
                    wacc += tz * ty * wx0
                if valid[z1, y1, x1]:
                    acc += src[z1, y1, x1] * tz * ty * tx
                    wacc += tz * ty * tx

                if wacc >= weight_threshold:
                    out[k, j, i] = acc / wacc
                    out_valid[k, j, i] = True
                else:
                    out[k, j, i] = 0.0
                    out_valid[k, j, i] = False


_resample_trilinear_masked_numba_serial = serial_twin(_resample_trilinear_masked_numba)


# Cubic spline resampling: scipy's affine_transform(order=3, mode='nearest') bit for bit.
# scipy's wheels compile with contraction on, so the C code fuses a * b + c into one rounding
# (FMA) in an expression. numba never fuses, so the kernels fuse with _fma at the sites where
# scipy's C code fuses. Every other operation keeps scipy's order.


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _pad_edge_numba(src: npt.NDArray[Any], npad: int, out: npt.NDArray[np.float64]) -> None:
    """out = np.pad(src, npad, mode='edge'), as float64."""
    nz, ny, nx = src.shape
    oz, oy, ox = out.shape
    for k in prange(oz):
        kk = min(max(k - npad, 0), nz - 1)
        for j in range(oy):
            jj = min(max(j - npad, 0), ny - 1)
            for i in range(ox):
                ii = min(max(i - npad, 0), nx - 1)
                out[k, j, i] = src[kk, jj, ii]


_pad_edge_numba_serial = serial_twin(_pad_edge_numba)


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _pad_edge_masked_numba(
    src: npt.NDArray[np.float64],
    valid: npt.NDArray[np.bool_],
    npad: int,
    out: npt.NDArray[np.float64],
    out_valid: npt.NDArray[np.float64],
) -> None:
    """out = np.pad(np.where(valid, src, 0.0), npad, mode='edge') and out_valid = the same pad
    of valid as float64: the two arrays of the normalized interpolation, in one pass."""
    nz, ny, nx = src.shape
    oz, oy, ox = out.shape
    for k in prange(oz):
        kk = min(max(k - npad, 0), nz - 1)
        for j in range(oy):
            jj = min(max(j - npad, 0), ny - 1)
            for i in range(ox):
                ii = min(max(i - npad, 0), nx - 1)
                if valid[kk, jj, ii]:
                    out[k, j, i] = src[kk, jj, ii]
                    out_valid[k, j, i] = 1.0
                else:
                    out[k, j, i] = 0.0
                    out_valid[k, j, i] = 0.0


_pad_edge_masked_numba_serial = serial_twin(_pad_edge_masked_numba)


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _prefilter_axis0_numba(
    c: npt.NDArray[np.float64], z: float, gain: float, z_n: float, n_chunks: int
) -> None:
    """scipy's spline_filter1d (order 3, the reflect start of mode 'nearest') along axis 0,
    in place: the lines of a slab of rows advance together, so the inner loop is contiguous."""
    nz, ny, nx = c.shape
    step = (ny + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        j0 = ch * step
        j1 = min(ny, (ch + 1) * step)
        if j0 >= j1:
            continue
        c0 = np.empty((j1 - j0, nx))
        for k in range(nz):
            for j in range(j0, j1):
                for i in range(nx):
                    c[k, j, i] *= gain
        for j in range(j0, j1):
            for i in range(nx):
                c0[j - j0, i] = c[0, j, i]
                c[0, j, i] = _fma(z_n, c[nz - 1, j, i], c[0, j, i])
        z_i = z
        for q in range(1, nz):
            for j in range(j0, j1):
                for i in range(nx):
                    c[0, j, i] = _fma(z_i, _fma(z_n, c[nz - 1 - q, j, i], c[q, j, i]), c[0, j, i])
            z_i *= z
        scale0 = z / _fma(-z_n, z_n, 1.0)
        for j in range(j0, j1):
            for i in range(nx):
                c[0, j, i] = c[0, j, i] * scale0
                c[0, j, i] += c0[j - j0, i]
        for q in range(1, nz):
            for j in range(j0, j1):
                for i in range(nx):
                    c[q, j, i] = _fma(z, c[q - 1, j, i], c[q, j, i])
        scale1 = z / (z - 1.0)
        for j in range(j0, j1):
            for i in range(nx):
                c[nz - 1, j, i] = c[nz - 1, j, i] * scale1
        for q in range(nz - 2, -1, -1):
            for j in range(j0, j1):
                for i in range(nx):
                    c[q, j, i] = z * (c[q + 1, j, i] - c[q, j, i])


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _prefilter_axis1_numba(c: npt.NDArray[np.float64], z: float, gain: float, z_n: float) -> None:
    """As _prefilter_axis0_numba, along axis 1 (one plane per thread)."""
    nz, ny, nx = c.shape
    scale0 = z / _fma(-z_n, z_n, 1.0)
    scale1 = z / (z - 1.0)
    for k in prange(nz):
        c0 = np.empty(nx)
        for j in range(ny):
            for i in range(nx):
                c[k, j, i] *= gain
        for i in range(nx):
            c0[i] = c[k, 0, i]
            c[k, 0, i] = _fma(z_n, c[k, ny - 1, i], c[k, 0, i])
        z_i = z
        for q in range(1, ny):
            for i in range(nx):
                c[k, 0, i] = _fma(z_i, _fma(z_n, c[k, ny - 1 - q, i], c[k, q, i]), c[k, 0, i])
            z_i *= z
        for i in range(nx):
            c[k, 0, i] = c[k, 0, i] * scale0
            c[k, 0, i] += c0[i]
        for q in range(1, ny):
            for i in range(nx):
                c[k, q, i] = _fma(z, c[k, q - 1, i], c[k, q, i])
        for i in range(nx):
            c[k, ny - 1, i] = c[k, ny - 1, i] * scale1
        for q in range(ny - 2, -1, -1):
            for i in range(nx):
                c[k, q, i] = z * (c[k, q + 1, i] - c[k, q, i])


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _prefilter_axis2_numba(c: npt.NDArray[np.float64], z: float, gain: float, z_n: float) -> None:
    """As _prefilter_axis0_numba, along axis 2 (the lines of one plane per thread)."""
    nz, ny, nx = c.shape
    scale0 = z / _fma(-z_n, z_n, 1.0)
    scale1 = z / (z - 1.0)
    for k in prange(nz):
        for j in range(ny):
            line = c[k, j]
            for i in range(nx):
                line[i] *= gain
            z_i = z
            c0 = line[0]
            line[0] = _fma(z_n, line[nx - 1], line[0])
            for i in range(1, nx):
                line[0] = _fma(z_i, _fma(z_n, line[nx - 1 - i], line[i]), line[0])
                z_i *= z
            line[0] = line[0] * scale0
            line[0] += c0
            for i in range(1, nx):
                line[i] = _fma(z, line[i - 1], line[i])
            line[nx - 1] = line[nx - 1] * scale1
            for i in range(nx - 2, -1, -1):
                line[i] = z * (line[i + 1] - line[i])


_prefilter_axis0_numba_serial = serial_twin(_prefilter_axis0_numba)
_prefilter_axis1_numba_serial = serial_twin(_prefilter_axis1_numba)
_prefilter_axis2_numba_serial = serial_twin(_prefilter_axis2_numba)


def _prefilter(c: npt.NDArray[np.float64]) -> None:
    """scipy's spline_filter(order 3, mode 'nearest') of the padded float64 array `c`, in
    place and bit for bit: one sweep per axis."""
    z = _SPLINE_POLE
    gain = 1.0
    gain *= (1.0 - z) * (1.0 - 1.0 / z)
    sweeps = (
        (_prefilter_axis0_numba, _prefilter_axis0_numba_serial),
        (_prefilter_axis1_numba, _prefilter_axis1_numba_serial),
        (_prefilter_axis2_numba, _prefilter_axis2_numba_serial),
    )
    for axis, (kernel, twin) in enumerate(sweeps):
        sweep = sized(kernel, twin, c.size, _PREFILTER_PARALLEL_MIN)
        if axis == 0:
            sweep(c, z, gain, math.pow(z, c.shape[0]), 4 * get_num_threads())
        else:
            sweep(c, z, gain, math.pow(z, c.shape[axis]))


def _padded(shape: tuple[int, ...]) -> npt.NDArray[np.float64]:
    """An empty float64 array of `shape` with _SPLINE_NPAD more voxels on each side."""
    return np.empty(tuple(n + 2 * _SPLINE_NPAD for n in shape), dtype=np.float64)


def _spline_coefficients(array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The cubic B-spline coefficients of `array` padded by _SPLINE_NPAD edge voxels: what
    scipy's affine_transform(order=3, mode='nearest') interpolates, bit for bit."""
    c = _padded(array.shape)
    pad = sized(_pad_edge_numba, _pad_edge_numba_serial, c.size, _PREFILTER_PARALLEL_MIN)
    pad(array, _SPLINE_NPAD, c)
    _prefilter(c)
    return c


def _spline_coefficients_masked(
    array: npt.NDArray[np.float64], valid: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The coefficients of the zeroed image (invalid voxels 0) and of the validity mask as
    float64: the two arrays _resample_with_source_mask transforms, bit for bit."""
    c = _padded(array.shape)
    w = _padded(array.shape)
    pad = sized(
        _pad_edge_masked_numba, _pad_edge_masked_numba_serial, c.size, _PREFILTER_PARALLEL_MIN
    )
    pad(array, valid, _SPLINE_NPAD, c, w)
    _prefilter(c)
    _prefilter(w)
    return c, w


@jit(nopython=True, cache=True)  # type: ignore
def _cubic_axis_numba(
    n_out: int, n_pad: int, shift: float, scale: float, start: int, npad: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Support indices (clamped to the padded array) and cubic B-spline weights of one axis,
    as scipy's zoom_shift computes them (the coordinate is not clamped in mode 'nearest')."""
    idx = np.empty((n_out, 4), np.int64)
    w = np.empty((n_out, 4), np.float64)
    for k in range(n_out):
        cc = float(k + start)
        cc += shift
        cc *= scale
        cc += npad
        fl = np.floor(cc)
        s = int(fl) - 1
        y = cc - fl
        z = 1.0 - y
        w1 = _fma(y * y * (y - 2.0), 3.0, 4.0) / 6.0
        w2 = _fma(z * z * (z - 2.0), 3.0, 4.0) / 6.0
        w0 = z * z * z / 6.0
        w3 = 1.0 - w0
        w3 -= w1
        w3 -= w2
        w[k, 0] = w0
        w[k, 1] = w1
        w[k, 2] = w2
        w[k, 3] = w3
        for a in range(4):
            idx[k, a] = min(max(s + a, 0), n_pad - 1)
    return idx, w


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _resample_tricubic_numba(
    coef: npt.NDArray[np.float64],
    scale: npt.NDArray[np.float64],
    shift: npt.NDArray[np.float64],
    start: npt.NDArray[np.int64],
    npad: float,
    out: npt.NDArray[np.float64],
) -> None:
    """Tricubic spline resampling of the coefficients `coef` (see _spline_coefficients) for a
    diagonal transform, 'nearest' boundary: scipy's zoom_shift, order 3, bit for bit (the 64
    support terms added in scipy's order, each value times the weights left to right)."""
    nz, ny, nx = coef.shape
    oz, oy, ox = out.shape
    zi, zw = _cubic_axis_numba(oz, nz, shift[0], scale[0], start[0], npad)
    yi, yw = _cubic_axis_numba(oy, ny, shift[1], scale[1], start[1], npad)
    xi, xw = _cubic_axis_numba(ox, nx, shift[2], scale[2], start[2], npad)
    for k in prange(oz):
        for j in range(oy):
            for i in range(ox):
                t = 0.0
                for a in range(4):
                    wz = zw[k, a]
                    za = zi[k, a]
                    for b in range(4):
                        wy = yw[j, b]
                        yb = yi[j, b]
                        for c in range(4):
                            t += coef[za, yb, xi[i, c]] * wz * wy * xw[i, c]
                out[k, j, i] = t


_resample_tricubic_numba_serial = serial_twin(_resample_tricubic_numba)


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _resample_tricubic_masked_numba(
    coef: npt.NDArray[np.float64],
    wcoef: npt.NDArray[np.float64],
    scale: npt.NDArray[np.float64],
    shift: npt.NDArray[np.float64],
    start: npt.NDArray[np.int64],
    npad: float,
    weight_threshold: float,
    out: npt.NDArray[np.float64],
    out_valid: npt.NDArray[np.bool_],
) -> None:
    """Normalized tricubic resampling: the two sums of _resample_with_source_mask (the zeroed
    image and the validity weights, see _spline_coefficients_masked) in one pass, each as
    _resample_tricubic_numba computes it. The value is their quotient where the weight
    reaches weight_threshold, else 0 and invalid."""
    nz, ny, nx = coef.shape
    oz, oy, ox = out.shape
    zi, zw = _cubic_axis_numba(oz, nz, shift[0], scale[0], start[0], npad)
    yi, yw = _cubic_axis_numba(oy, ny, shift[1], scale[1], start[1], npad)
    xi, xw = _cubic_axis_numba(ox, nx, shift[2], scale[2], start[2], npad)
    for k in prange(oz):
        for j in range(oy):
            for i in range(ox):
                t = 0.0
                u = 0.0
                for a in range(4):
                    wz = zw[k, a]
                    za = zi[k, a]
                    for b in range(4):
                        wy = yw[j, b]
                        yb = yi[j, b]
                        for c in range(4):
                            xc = xi[i, c]
                            wx = xw[i, c]
                            t += coef[za, yb, xc] * wz * wy * wx
                            u += wcoef[za, yb, xc] * wz * wy * wx
                if u >= weight_threshold:
                    out[k, j, i] = t / u
                    out_valid[k, j, i] = True
                else:
                    out[k, j, i] = 0.0
                    out_valid[k, j, i] = False


_resample_tricubic_masked_numba_serial = serial_twin(_resample_tricubic_masked_numba)


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _sentinel_counts_numba(
    flat: npt.NDArray[np.float64],
    roi_flat: npt.NDArray[Any],
    candidates: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Voxels equal to each candidate, in the image and in the ROI (roi != 0), in one
    pass. An empty `roi_flat` means no ROI mask (the ROI counts stay 0)."""
    n = flat.size
    n_c = candidates.size
    n_chunks = 64
    step = (n + n_chunks - 1) // n_chunks
    total = np.zeros((n_chunks, n_c), dtype=np.int64)
    inside = np.zeros((n_chunks, n_c), dtype=np.int64)
    has_roi = roi_flat.size == n
    for c in prange(n_chunks):
        for i in range(c * step, min(n, (c + 1) * step)):
            v = flat[i]
            for k in range(n_c):
                if v == candidates[k]:
                    total[c, k] += 1
                    if has_roi and roi_flat[i] != 0:
                        inside[c, k] += 1
    return total.sum(axis=0), inside.sum(axis=0)


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _sentinel_counts6_numba(
    flat: npt.NDArray[np.float64],
    roi_flat: npt.NDArray[Any],
    c: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """_sentinel_counts_numba for exactly 6 candidates, with scalar counters: the compiler
    keeps them in registers and vectorizes the pass."""
    n = flat.size
    n_chunks = 64
    step = (n + n_chunks - 1) // n_chunks
    total = np.zeros((n_chunks, 6), dtype=np.int64)
    inside = np.zeros((n_chunks, 6), dtype=np.int64)
    has_roi = roi_flat.size == n
    c0, c1, c2, c3, c4, c5 = c[0], c[1], c[2], c[3], c[4], c[5]
    for ch in prange(n_chunks):
        t0 = 0
        t1 = 0
        t2 = 0
        t3 = 0
        t4 = 0
        t5 = 0
        i0 = 0
        i1 = 0
        i2 = 0
        i3 = 0
        i4 = 0
        i5 = 0
        lo = ch * step
        hi = min(n, (ch + 1) * step)
        if has_roi:
            for i in range(lo, hi):
                v = flat[i]
                r = roi_flat[i] != 0
                e0 = v == c0
                e1 = v == c1
                e2 = v == c2
                e3 = v == c3
                e4 = v == c4
                e5 = v == c5
                t0 += e0
                t1 += e1
                t2 += e2
                t3 += e3
                t4 += e4
                t5 += e5
                i0 += e0 & r
                i1 += e1 & r
                i2 += e2 & r
                i3 += e3 & r
                i4 += e4 & r
                i5 += e5 & r
        else:
            for i in range(lo, hi):
                v = flat[i]
                t0 += v == c0
                t1 += v == c1
                t2 += v == c2
                t3 += v == c3
                t4 += v == c4
                t5 += v == c5
        total[ch, 0] = t0
        total[ch, 1] = t1
        total[ch, 2] = t2
        total[ch, 3] = t3
        total[ch, 4] = t4
        total[ch, 5] = t5
        inside[ch, 0] = i0
        inside[ch, 1] = i1
        inside[ch, 2] = i2
        inside[ch, 3] = i3
        inside[ch, 4] = i4
        inside[ch, 5] = i5
    return total.sum(axis=0), inside.sum(axis=0)


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _not_equal_numba(
    flat: npt.NDArray[np.float64], value: float, out: npt.NDArray[np.bool_]
) -> None:
    """out = flat != value, in parallel."""
    for i in prange(flat.size):
        out[i] = flat[i] != value


_not_equal_numba_serial = serial_twin(_not_equal_numba)


def _not_sentinel(array: npt.NDArray[Any], value: float) -> npt.NDArray[np.bool_]:
    """array != value (True = valid voxel), in the memory order of `array`. A float64 array
    in row or column order takes the kernel: one pass, in parallel from
    _NOT_EQUAL_PARALLEL_MIN voxels, with the bits of numpy."""
    order = _shared_order(array)
    if array.dtype != np.float64 or order is None:
        return cast(npt.NDArray[np.bool_], array != value)
    out = np.empty(array.size, dtype=np.bool_)
    kernel = sized(_not_equal_numba, _not_equal_numba_serial, array.size, _NOT_EQUAL_PARALLEL_MIN)
    kernel(array.ravel(order), float(value), out)
    return out.reshape(array.shape, order=order)


def detect_sentinel_value(
    image: Image,
    candidate_values: tuple[float, ...] = COMMON_SENTINEL_VALUES,
    min_presence_fraction: float = 0.05,
    roi_mask: Optional[Image] = None,
) -> Optional[float]:
    """
    Detect if image contains a common sentinel value outside the ROI.

    A candidate is eligible as a sentinel if:
    1. It occupies a significant fraction of the whole image (>= min_presence_fraction).
    2. If roi_mask is provided, it appears primarily outside the ROI (ratio > 2:1
       outside vs inside). This guard distinguishes a padding/fill value from a
       legitimate low-density tissue value (e.g. real air at ~-1000 HU in a raw,
       un-masked CT) that would otherwise be misdetected on proportion alone.

    When several candidates are eligible, the one occupying the largest fraction
    of the image is returned; exact ties are broken by candidate_values order.

    This is used by the pipeline's AUTO source_mode to automatically detect
    images that have been pre-masked with sentinel values.

    Args:
        image: Input Image object.
        candidate_values: Values to check for sentinel patterns.
            Defaults to common medical imaging sentinels: -2048, -3024, -1024,
            -1000, 0, -32768.
        min_presence_fraction: Minimum fraction of voxels that must equal
            the candidate to consider it a sentinel. Default is 5%.
        roi_mask: Optional ROI mask. If provided, checks that sentinel values
            are primarily outside the mask (ratio > 2:1 outside vs inside).

    Returns:
        The detected sentinel value, or None if not detected.

    Example:
        ```python
        from pictologics.preprocessing import detect_sentinel_value
        from pictologics.loader import load_image

        image = load_image("image_with_sentinel.nii.gz")
        sentinel = detect_sentinel_value(image)

        if sentinel is not None:
            print(f"Detected sentinel value: {sentinel}")
        ```
    """
    return _sentinel_search(image, candidate_values, min_presence_fraction, roi_mask)[0]


def _sentinel_search(
    image: Image,
    candidate_values: tuple[float, ...] = COMMON_SENTINEL_VALUES,
    min_presence_fraction: float = 0.05,
    roi_mask: Optional[Image] = None,
) -> tuple[Optional[float], int]:
    """detect_sentinel_value with the count of the image voxels equal to the found value
    (0 when no value is found)."""
    array = image.array
    total_voxels = array.size
    roi_arr: Optional[npt.NDArray[np.bool_]] = None
    if roi_mask is not None:
        _validate_geometry(roi_mask, image, "ROI mask", "image")

    # Large images: one parallel pass counts every candidate (and its ROI voxels),
    # instead of one pass per candidate. The counts are the same.
    counts: Optional[npt.NDArray[np.int64]] = None
    inside_counts: Optional[npt.NDArray[np.int64]] = None
    order = _shared_order(array) if roi_mask is None else _shared_order(array, roi_mask.array)
    if (
        array.dtype == np.float64
        and array.size >= (_SENTINEL_KERNEL_MIN_SIZE if order else _KERNEL_MIN_SIZE)
        and (roi_mask is None or roi_mask.array.dtype in (np.float64, np.uint8, np.bool_))
    ):
        roi_flat = (
            roi_mask.array.ravel(order or "C") if roi_mask is not None else np.empty(0, np.uint8)
        )
        candidates = np.asarray(candidate_values, dtype=np.float64)
        kernel = _sentinel_counts6_numba if candidates.size == 6 else _sentinel_counts_numba
        counts, inside_counts = kernel(array.ravel(order or "C"), roi_flat, candidates)
    elif roi_mask is not None:
        roi_arr = roi_mask.array != 0

    best_candidate: Optional[float] = None
    best_fraction = 0.0
    best_count = 0

    for k, candidate in enumerate(candidate_values):
        if counts is None:
            sentinel_mask = array == candidate
            count = np.count_nonzero(sentinel_mask)
        else:
            count = counts[k]
        fraction = count / total_voxels

        if fraction < min_presence_fraction:
            continue

        # If ROI mask provided, verify sentinel is primarily outside (>= 2:1)
        if roi_mask is not None:
            if inside_counts is None:
                inside_count = np.count_nonzero(sentinel_mask & roi_arr)
            else:
                inside_count = inside_counts[k]
            outside_count = count - inside_count
            if outside_count <= inside_count * 2:
                continue

        # Eligible: keep the candidate covering the largest fraction. Strict >
        # means an earlier (higher-priority) candidate wins an exact tie.
        if fraction > best_fraction:
            best_fraction = fraction
            best_candidate = candidate
            best_count = int(count)

    return best_candidate, best_count


def create_source_mask_from_sentinel(
    image: Image,
    sentinel_value: float,
    tolerance: float = 0.0,
) -> Image:
    """
    Create a source validity mask by marking sentinel voxels as invalid.

    The returned mask has value 1 for valid (non-sentinel) voxels and 0 for
    invalid (sentinel) voxels. This mask can be used with the Image.source_mask
    attribute to exclude sentinel voxels from resampling and filtering.

    Args:
        image: Input Image object.
        sentinel_value: The value considered as sentinel (invalid data).
        tolerance: Values within this tolerance of sentinel_value are also
            considered invalid. Default 0 means exact match only.

    Returns:
        Image object with binary mask (1 = valid, 0 = sentinel).

    Example:
        ```python
        from pictologics.preprocessing import create_source_mask_from_sentinel
        from pictologics.loader import load_image

        image = load_image("ct_with_background.nii.gz")

        # Create mask excluding -2048 sentinel values
        source_mask = create_source_mask_from_sentinel(image, sentinel_value=-2048)

        # Apply to image for sentinel-aware processing
        image_with_mask = image.with_source_mask(source_mask)
        ```
    """
    if tolerance == 0:
        valid = _not_sentinel(image.array, sentinel_value)
    else:
        valid = np.abs(image.array - sentinel_value) > tolerance

    return Image(
        array=valid.view(np.uint8),  # the bytes of a bool array are 0 and 1
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
        modality="SOURCE_MASK",
    )


def _resample_with_source_mask(
    image_array: npt.NDArray[np.floating[Any]],
    source_mask: npt.NDArray[np.bool_],
    matrix: npt.NDArray[np.floating[Any]],
    offset: npt.NDArray[np.floating[Any]],
    output_shape: tuple[int, ...],
    order: int,
    mode: str,
    weight_threshold: float = 0.5,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]:
    """
    Resample image using normalized interpolation that excludes invalid (sentinel) voxels.

    This implements normalized convolution for resampling, where only valid source
    voxels contribute to the interpolated output values.

    Algorithm:
        1. Zero out invalid voxels in image (image * mask)
        2. Resample zeroed image -> weighted_sum
        3. Resample mask (as float) -> weight_sum
        4. Normalize: result = weighted_sum / weight_sum
        5. Mark output voxels with low weight as invalid

    Args:
        image_array: Input image array.
        source_mask: Boolean mask where True = valid voxel.
        matrix: Affine transform matrix (diagonal elements).
        offset: Affine transform offset.
        output_shape: Shape of output array.
        order: Interpolation order.
        mode: Boundary mode.
        weight_threshold: Minimum weight to consider output voxel valid.
            Output voxels with weight < threshold are marked as invalid.
            Default 0.5 means the majority of the interpolation weight must
            come from valid voxels; lower values keep more boundary voxels
            at the cost of heavier extrapolation.

    Returns:
        Tuple of (resampled_image, resampled_source_mask).
    """
    # Step 1: Zero out invalid voxels
    valid_image = np.where(source_mask, image_array, 0.0).astype(np.float64)

    # Step 2: Resample zeroed image - this gives weighted sum where invalid=0
    weighted_sum = affine_transform(
        valid_image,
        matrix=matrix,
        offset=offset,
        output_shape=output_shape,
        order=order,
        mode=mode,
    )

    # Step 3: Resample mask as weights
    weight_sum = affine_transform(
        source_mask.astype(np.float64),
        matrix=matrix,
        offset=offset,
        output_shape=output_shape,
        order=order,
        mode=mode,
    )

    # Step 4: Normalize (avoid division by zero)
    # Voxels with weight_sum < threshold are considered invalid
    valid_output = weight_sum >= weight_threshold

    result = np.zeros_like(weighted_sum)
    result[valid_output] = weighted_sum[valid_output] / weight_sum[valid_output]

    # Output source mask based on interpolation weights
    output_source_mask = valid_output

    return result, output_source_mask


def _mask_threshold(values: npt.NDArray[Any], threshold: float, rounded: bool) -> npt.NDArray[Any]:
    """1 where a mask value reaches `threshold`, else 0.

    An interpolated value (`rounded`: a linear or cubic resample, of every mask type) is
    rounded at the threshold, value + (1 - threshold) >= 1: at 0.5 this is the rounding
    that uint8 masks always had, which keeps a value that is one half up to rounding
    error. A value of a nearest resample compares itself.
    """
    if rounded:
        return (values + (1.0 - threshold) >= 1.0).astype(np.uint8)
    return (values >= threshold).astype(np.uint8)


_NEAREST_DTYPES = (np.float64, np.uint8, np.bool_)


_Grid = tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]


def _grid(shape: tuple[int, ...], spacing: tuple[Any, ...], new_spacing: tuple[Any, ...]) -> _Grid:
    """The resampled grid shape, the diagonal transform (matrix, offset) from an output
    index to an input coordinate, x_in = matrix * x_out + offset ('align grid centers'),
    and the shift of the origin, as tuples."""
    original_spacing = np.array(spacing)
    target_spacing = np.array(new_spacing)
    # IBSI: nb = ceil(na * sa / sb). Round to 9 decimals before ceil: floating-point noise
    # can push an exact integer product (e.g. 110.0) to 110.00000000000001, yielding a
    # spurious extra voxel along that axis.
    new_shape = np.ceil(np.round(shape * (original_spacing / target_spacing), 9)).astype(int)
    matrix = target_spacing / original_spacing
    offset = (np.array(shape) - 1) / 2.0 - matrix * ((new_shape - 1) / 2.0)
    # O_new = O_old + 0.5 * ((N_old - 1) * S_old - (N_new - 1) * S_new), along the axes
    extent_orig = (np.array(shape) - 1) * original_spacing
    extent_new = (new_shape - 1) * target_spacing
    origin_shift = 0.5 * (extent_orig - extent_new)
    return (
        tuple(new_shape.tolist()),
        tuple(matrix.tolist()),
        tuple(offset.tolist()),
        tuple(origin_shift.tolist()),
    )


_kept_grid = lru_cache(maxsize=256)(_grid)


def _output_grid(shape: tuple[int, ...], spacing: Any, new_spacing: Any) -> _Grid:
    """`_grid` of a resample. The grid of positive spacings stays for each input: equal
    positive numbers have the same bits, so a kept grid is the grid of the input."""
    key = (tuple(shape), tuple(spacing), tuple(new_spacing))
    if all(isinstance(s, (int, float)) and s > 0 for s in key[1] + key[2]):
        return _kept_grid(*key)
    return _grid(*key)


def _roi_region(
    box: tuple[slice, slice, slice],
    shape: tuple[int, ...],
    spacing: tuple[float, float, float],
    new_spacing: tuple[float, float, float],
    margin: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    """The region of the resampled grid that can hold voxels of an ROI in the input box
    `box`, grown by `margin` output voxels. An output voxel takes its value from input
    voxels within 1 of its input coordinate (linear, nearest), so the region holds every
    output whose coordinate is within 2 of the box; a box at the input edge reaches the
    grid edge (the kernels clamp there)."""
    new_shape, matrix, offset, _ = _output_grid(shape, spacing, new_spacing)
    region = []
    for b, n_in, n_out, m, o, g in zip(box, shape, new_shape, matrix, offset, margin, strict=True):
        lo = 0 if b.start == 0 else max(math.floor((b.start - 2 - o) / m), 0)
        hi = n_out if b.stop == n_in else min(math.ceil((b.stop + 1 - o) / m) + 1, n_out)
        region.append(slice(max(lo - g, 0), min(hi + g, int(n_out))))
    return region[0], region[1], region[2]


def _region_origin(
    origin: tuple[float, ...],
    direction: Any,
    spacing: tuple[float, float, float],
    region: tuple[slice, slice, slice],
) -> tuple[float, float, float]:
    """The world position of the first voxel of `region` of a grid."""
    step = np.array([r.start for r in region]) * np.array(spacing)
    x, y, z = np.array(origin) + _direction_matrix(direction) @ step
    return float(x), float(y), float(z)


def _region_of(image: Image, region: tuple[slice, slice, slice]) -> Image:
    """The part of `image` (and of its source mask) in `region`, as its own Image (copies:
    a view would keep the whole arrays alive)."""
    return Image(
        array=np.array(image.array[region], order="C"),
        spacing=image.spacing,
        origin=_region_origin(image.origin, image.direction, image.spacing, region),
        direction=image.direction,
        modality=image.modality,
        source_mask=(
            None if image.source_mask is None else np.array(image.source_mask[region], order="C")
        ),
    )


def resample_image(
    image: Image,
    new_spacing: tuple[float, float, float],
    interpolation: str = "linear",
    boundary_mode: str = "nearest",
    round_intensities: bool = False,
    mask_threshold: Optional[float] = None,
    source_mask: Optional[Image | npt.NDArray[np.bool_]] = None,
    weight_threshold: float = 0.5,
    region: Optional[tuple[slice, slice, slice]] = None,
) -> Image:
    """
    Resample image to new voxel spacing using IBSI-compliant 'Align grid centers' method.

    The common cases (3D float64 or uint8 image, 'nearest' boundary, nearest/linear/cubic
    interpolation) run on parallel numba kernels that give scipy's output bit for bit;
    everything else uses scipy.ndimage.affine_transform.

    Args:
        image: Input Image object.
        new_spacing: Target spacing (x, y, z). Must be positive.
        interpolation: Interpolation method.
            'nearest': Nearest neighbour (order 0).
            'linear': Trilinear (order 1).
            'cubic': Tricubic spline (order 3), with the spline prefilter of scipy.
        boundary_mode: Padding mode for extrapolation.
            'nearest' (default): Replicates edge values (aaaa|abcd|dddd).
            'constant': Pads with constant value (0).
            'reflect': Reflects at boundary.
            'wrap': Wraps around.
        round_intensities: If True, round resulting intensities to nearest integer.
        mask_threshold: If provided, treat output as a binary mask.
                        Interpolated values >= threshold become 1, others 0.
                        An interpolated value of every mask type is rounded at
                        the threshold (value + 1 - threshold >= 1), as a uint8
                        mask always did at 0.5.
                        Commonly 0.5 for partial volume correction.
        source_mask: Optional source validity mask. If provided (or if image.source_mask
                     is set), only valid voxels are used for interpolation. This prevents
                     sentinel values (e.g., -2048 in CT) from contaminating the resampled
                     output. Can be an Image object or a boolean numpy array.
        weight_threshold: Only used with a source mask. Minimum fraction of
                     interpolation weight that must come from valid voxels for an
                     output voxel to be considered valid. Default 0.5 (majority).
        region: Optional slices of the new grid. The output then holds only this region,
                     with the values of the whole grid bit for bit (its origin is the
                     region's first voxel), and the rest of the grid is not computed.

    Returns:
        Resampled Image object. If source_mask was used, the output Image will have
        its source_mask attribute set to the resampled validity mask.

    Note:
        When source_mask is active, the function uses normalized interpolation:
        the contribution of each input voxel is weighted by its validity, and the
        result is normalized by the sum of valid weights. This ensures that sentinel
        voxels do not affect the output. A float64 image with linear or cubic
        interpolation runs this on the numba kernels.

    Raises:
        ValueError: If any element of `new_spacing` is not positive, if
            `interpolation` is not one of 'nearest', 'linear', 'cubic', or if
            `source_mask` does not match the image geometry (when passed as an
            Image) or array shape (when passed as a numpy array).

    Example:
        Resample image to isotropic 1mm spacing using linear interpolation:

        ```python
        from pictologics.preprocessing import resample_image

        # Resample to 1x1x1 mm
        resampled_img = resample_image(
            image,
            new_spacing=(1.0, 1.0, 1.0),
            interpolation="linear"
        )
        ```

        Resample with sentinel-value exclusion:

        ```python
        # Image has -2048 outside ROI
        image_with_sentinel = image.with_source_mask(roi_mask)
        resampled = resample_image(
            image_with_sentinel,
            new_spacing=(1.0, 1.0, 1.0)
        )
        # Sentinel voxels were excluded from interpolation
        ```
    """
    if any(s <= 0 for s in new_spacing):
        raise ValueError(f"New spacing must be positive, got {new_spacing}")

    # Determine effective source mask
    effective_source: Optional[npt.NDArray[np.bool_]] = None

    if source_mask is not None:
        if isinstance(source_mask, Image):
            _validate_geometry(source_mask, image, "source mask", "image")
            arr: npt.NDArray[Any] = source_mask.array
            effective_source = arr if arr.dtype == np.bool_ else arr > 0
        else:
            effective_source = source_mask.astype(bool, copy=False)
    elif image.has_source_mask:
        effective_source = image.source_mask

    if effective_source is not None and effective_source.shape != image.array.shape:
        raise ValueError(
            f"Source mask shape {effective_source.shape} must match image shape {image.array.shape}"
        )

    # An all-valid mask is equivalent to no mask; skip the normalized
    # interpolation path, which costs a second affine_transform.
    source_all_valid = effective_source is not None and bool(effective_source.all())
    if source_all_valid:
        effective_source = None

    # Map interpolation string to spline order
    interpolation_map = {
        "nearest": 0,
        "linear": 1,
        "cubic": 3,
    }

    if interpolation not in interpolation_map:
        raise ValueError(
            f"Unknown interpolation method: {interpolation}. "
            f"Supported: {list(interpolation_map.keys())}"
        )

    order = interpolation_map[interpolation]

    # The new grid, and the transform from an output index to an input coordinate:
    # x_in = matrix * x_out + offset ('align grid centers', see _grid)
    out_shape, scale, shift, origin_shift = _output_grid(
        image.array.shape, image.spacing, new_spacing
    )
    matrix = np.array(scale)
    offset = np.array(shift)

    # Perform resampling
    resampled_array: npt.NDArray[Any]
    new_source_mask: Optional[npt.NDArray[np.bool_]] = None
    start = np.zeros(3, dtype=np.int64)

    # The kernels cover the hot path (3D, 'nearest' boundary, common dtypes) at every
    # size, with serial twins below the parallel gates; anything else falls back to
    # scipy.ndimage. The kernels take the shift form scipy's zoom_shift uses internally
    # (coordinate is (idx + offset/scale) * scale) and follow its steps, so their output
    # is bit for bit the scipy output.
    kernel_ok = image.array.ndim == 3 and boundary_mode == "nearest"
    kernel_shift = offset / matrix
    if region is not None:
        dtype = image.array.dtype
        if not (
            kernel_ok
            and (
                (order == 1 and dtype == np.float64)
                or (effective_source is None and order == 1 and dtype == np.uint8)
                or (effective_source is None and order == 0 and dtype in _NEAREST_DTYPES)
                or (order == 3 and dtype == np.float64)
            )
        ):
            # No kernel for this case: cut the whole grid
            whole = resample_image(
                image,
                new_spacing,
                interpolation,
                boundary_mode,
                round_intensities,
                mask_threshold,
                source_mask,
                weight_threshold,
            )
            return _region_of(whole, region)
        start = np.array([r.start for r in region], dtype=np.int64)
        out_shape = tuple(r.stop - r.start for r in region)
    # A mask threshold applies to the interpolated value: a uint8 or bool mask would
    # otherwise round or truncate the value first and lose the threshold.
    thresholded = False
    # The linear kernel rounds the values itself (no second pass over the output)
    rounded = False

    if effective_source is None:
        if kernel_ok and order == 1 and image.array.dtype in (np.float64, np.uint8):
            src = _row_order_copy(image.array)
            thresholded = mask_threshold is not None
            rounded = bool(round_intensities)
            resampled_array = np.empty(
                out_shape, dtype=np.uint8 if thresholded else image.array.dtype
            )
            linear = sized(
                _resample_trilinear_numba,
                _resample_trilinear_numba_serial,
                resampled_array.size,
                _LINEAR_PARALLEL_MIN,
            )
            linear(
                src,
                matrix,
                kernel_shift,
                start,
                image.array.dtype == np.uint8,
                math.nan if mask_threshold is None else float(mask_threshold),
                rounded,
                resampled_array,
            )
        elif kernel_ok and order == 0 and image.array.dtype in _NEAREST_DTYPES:
            src = _row_order_copy(image.array)
            resampled_array = np.empty(out_shape, dtype=image.array.dtype)
            nearest = sized(
                _resample_nearest_numba,
                _resample_nearest_numba_serial,
                resampled_array.size,
                _NEAREST_PARALLEL_MIN,
            )
            nearest(src, matrix, kernel_shift, start, resampled_array)
        elif kernel_ok and order == 3 and image.array.dtype == np.float64:
            coef = _spline_coefficients(_row_order_copy(image.array))
            resampled_array = np.empty(out_shape, dtype=np.float64)
            cubic = sized(
                _resample_tricubic_numba,
                _resample_tricubic_numba_serial,
                resampled_array.size,
                _CUBIC_PARALLEL_MIN,
            )
            cubic(coef, matrix, kernel_shift, start, float(_SPLINE_NPAD), resampled_array)
            del coef
        else:
            # A mask threshold needs the interpolated value, so a non-float64 mask is
            # interpolated in float64. A uint8 mask at the threshold 0.5 keeps scipy's
            # uint8 output, whose rounding gives that same result.
            source = image.array
            if (
                mask_threshold is not None
                and order > 0
                and not (source.dtype == np.uint8 and mask_threshold == 0.5)
            ):
                source = source.astype(np.float64, copy=False)
            resampled_array = affine_transform(
                source,
                matrix=matrix,
                offset=offset,
                output_shape=out_shape,
                order=order,
                mode=boundary_mode,
            )
        if source_all_valid:
            # Preserve the output contract: a resampled image keeps a source mask
            new_source_mask = np.ones(out_shape, dtype=bool)
    elif kernel_ok and order == 1 and image.array.dtype == np.float64:
        # Masked resampling: one fused pass instead of two affine_transforms
        src = _row_order_copy(image.array)
        valid = _row_order_copy(effective_source)
        resampled_array = np.empty(out_shape, dtype=np.float64)
        new_source_mask = np.empty(out_shape, dtype=np.bool_)
        masked = sized(
            _resample_trilinear_masked_numba,
            _resample_trilinear_masked_numba_serial,
            resampled_array.size,
            _MASKED_PARALLEL_MIN,
        )
        masked(
            src,
            valid,
            matrix,
            kernel_shift,
            start,
            float(weight_threshold),
            resampled_array,
            new_source_mask,
        )
    elif kernel_ok and order == 3 and image.array.dtype == np.float64:
        # Masked cubic resampling: the two sums of the normalized interpolation in one pass
        src = _row_order_copy(image.array)
        valid = _row_order_copy(effective_source)
        coef, wcoef = _spline_coefficients_masked(src, valid)
        resampled_array = np.empty(out_shape, dtype=np.float64)
        new_source_mask = np.empty(out_shape, dtype=np.bool_)
        masked_cubic = sized(
            _resample_tricubic_masked_numba,
            _resample_tricubic_masked_numba_serial,
            resampled_array.size,
            _CUBIC_PARALLEL_MIN,
        )
        masked_cubic(
            coef,
            wcoef,
            matrix,
            kernel_shift,
            start,
            float(_SPLINE_NPAD),
            float(weight_threshold),
            resampled_array,
            new_source_mask,
        )
        del coef, wcoef
    else:
        # Masked resampling using normalized interpolation
        resampled_array, new_source_mask = _resample_with_source_mask(
            image.array,
            effective_source,
            matrix=matrix,
            offset=offset,
            output_shape=out_shape,
            order=order,
            mode=boundary_mode,
            weight_threshold=weight_threshold,
        )

    # Post-processing
    if mask_threshold is not None and not thresholded:
        # Binarize mask
        resampled_array = _mask_threshold(resampled_array, mask_threshold, order > 0)
    elif round_intensities and not rounded:
        # Round intensities, in place: the resampled array is always a new one
        np.round(resampled_array, out=resampled_array)

    # Update origin to maintain center alignment (the shift of _grid, in world axes)
    new_origin = tuple(
        np.array(image.origin) + _direction_matrix(image.direction) @ np.array(origin_shift)
    )
    if region is not None:
        new_origin = _region_origin(new_origin, image.direction, new_spacing, region)

    return Image(
        array=resampled_array,
        spacing=new_spacing,
        origin=new_origin,
        direction=image.direction,
        modality=image.modality,
        source_mask=new_source_mask,
    )


def discretise_image(
    image: Image | npt.NDArray[Any],
    method: str,
    roi_mask: Image | npt.NDArray[Any] | None = None,
    n_bins: Optional[int] = None,
    bin_width: Optional[float] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    cutoffs: Optional[list[float]] = None,
) -> Image | npt.NDArray[np.int32]:
    """
    Discretise image intensities.

    Supports IBSI-compliant Fixed Bin Number (FBN) and Fixed Bin Size (FBS).

    Args:
        image: Input Image object or numpy array.
        method: 'FBN' (Fixed Bin Number), 'FBS' (Fixed Bin Size), or 'FIXED_CUTOFFS'.
        roi_mask: Optional mask to define the ROI for determining min/max values.
        n_bins: Number of bins (required for FBN).
        bin_width: Bin width (required for FBS).
        min_val: Minimum value for discretisation.
                 For FBS, defaults to ROI minimum (or global minimum). To compare
                 images, give FBS the same min_val for every image, as IBSI
                 recommends (the lower bound of the resegmentation range); the
                 pipeline always does.
                 For FBN, defaults to ROI minimum.
        max_val: Maximum value for discretisation (FBN only).
                 Defaults to ROI maximum.
        cutoffs: List of cutoffs (required for FIXED_CUTOFFS). Values below the
                 first cutoff map to bin 1; values >= the last cutoff map to
                 bin len(cutoffs) + 1.

    Returns:
        Discretised Image object or numpy array (depending on input).
        Values are 1-based int32 indices; 0 marks NaN (invalid) voxels.

    Raises:
        ValueError: If `roi_mask` shape does not match `image` shape (or, when
            both are Image objects, if their geometry otherwise differs); if
            `method` is 'FBN' and `n_bins` is missing or not positive; if
            `method` is 'FBS' and `bin_width` is missing or not positive; if
            `method` is 'FIXED_CUTOFFS' and `cutoffs` is missing; or if
            `method` is not one of 'FBN', 'FBS', 'FIXED_CUTOFFS'.

    Example:
        Discretise image into 32 fixed bins (FBN):

        ```python
        from pictologics.preprocessing import discretise_image

        # FBN with 32 bins
        disc_image = discretise_image(
            image,
            method="FBN",
            n_bins=32
        )
        ```
    """
    # Handle input type
    array = image.array if isinstance(image, Image) else image

    mask_arr: Optional[npt.NDArray[Any]] = None
    if roi_mask is not None:
        if isinstance(roi_mask, Image):
            if isinstance(image, Image):
                _validate_geometry(roi_mask, image, "ROI mask", "image")
            mask_arr = roi_mask.array
        else:
            mask_arr = roi_mask

        if mask_arr.shape != array.shape:
            raise ValueError(f"Shape mismatch: Image {array.shape} vs Mask {mask_arr.shape}")

    roi_range: Optional[tuple[Any, ...]] = None  # (min, max) of the finite ROI values

    def _default_bound(index: int, global_reduce: Any) -> Any:
        """Default bin bound (index 0: min, 1: max) over the ROI, falling back to the
        global one. The ROI range is found at the first call, so given bounds skip the
        search. A float64 row-order image with a float64 or uint8 row-order mask takes
        one fused pass; other inputs gather the ROI values. Both skip NaN and infinite
        values."""
        nonlocal roi_range
        if roi_range is None and mask_arr is not None:
            if (
                array.ndim == 3
                and array.dtype == np.float64
                and mask_arr.dtype in (np.float64, np.uint8)
                and array.flags.c_contiguous
                and mask_arr.flags.c_contiguous
            ):
                found = roi_min_max(array, mask_arr)
                roi_range = () if found is None else (np.float64(found[0]), np.float64(found[1]))
            else:
                values = array[mask_arr != 0]
                values = values[np.isfinite(values)]
                roi_range = (values.min(), values.max()) if values.size else ()
        if roi_range:
            return roi_range[index]
        # nanmin/nanmax of an all-NaN image is NaN (warning suppressed); it
        # propagates through the bin math so every voxel ends up invalid (0).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return global_reduce(array)

    discretised: npt.NDArray[Any]

    if array.size == 0:
        # int32 halves memory vs the default int64; bin counts in radiomics
        # never approach the int32 limit
        discretised = np.zeros(array.shape, dtype=np.int32)
        if isinstance(image, Image):
            # Create new Image with discretised array
            return Image(
                array=discretised,
                spacing=image.spacing,
                origin=image.origin,
                direction=image.direction,
                modality=image.modality,
            )
        return discretised

    # The bin maths below run densely on the full array (no boolean
    # gather/scatter): NaN voxels propagate through the float ops and are
    # mapped to bin 0 (invalid) at the end. The kernels read and write a row- or
    # column-order float64 array in its own order, with no copy (another layout from
    # _KERNEL_MIN_SIZE voxels on, as a copy); the numpy chain takes the other arrays.
    order = _shared_order(array)
    use_kernel = array.dtype == np.float64 and (order is not None or array.size >= _KERNEL_MIN_SIZE)
    if method == "FBN":
        if n_bins is None:
            raise ValueError("n_bins required for FBN")
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")

        # Determine min/max
        current_min = min_val if min_val is not None else _default_bound(0, np.nanmin)
        current_max = max_val if max_val is not None else _default_bound(1, np.nanmax)

        if current_max <= current_min:
            # Edge case: flat region or invalid range
            discretised = (~np.isnan(array)).astype(np.int32)
        elif use_kernel:
            # Single-pass kernel; bit-identical to the numpy chain below
            flat = array.ravel(order or "C")
            binned: npt.NDArray[Any] = np.empty(flat.size, dtype=np.int32)
            fbn = sized(
                _discretise_fbn_numba,
                _discretise_fbn_numba_serial,
                flat.size,
                _DISCRETISE_PARALLEL_MIN,
            )
            fbn(flat, float(n_bins), float(current_min), float(current_max), binned)
            discretised = binned.reshape(array.shape, order=order or "C")
        else:
            # IBSI FBN: floor(N_g * (X - X_min) / (X_max - X_min)) + 1
            # (same operation order as before, so bins are bit-identical)
            temp = n_bins * (array - current_min)
            if temp.dtype.kind == "f":
                np.divide(temp, current_max - current_min, out=temp)
            else:
                # integer input: out-of-place division promotes to float
                temp = temp / (current_max - current_min)
            np.floor(temp, out=temp)
            temp += 1

            # Clip to [1, N_g]: the max value falls into N_g + 1 with this
            # formula, and outliers are clipped to the boundary bins
            np.clip(temp, 1, n_bins, out=temp)

            # NaN voxels propagated through the maths; map them to bin 0
            temp[np.isnan(temp)] = 0
            discretised = temp.astype(np.int32)

    elif method == "FBS":
        if bin_width is None:
            raise ValueError("bin_width required for FBS")
        if bin_width <= 0:
            raise ValueError("bin_width must be positive")

        current_min = min_val if min_val is not None else _default_bound(0, np.nanmin)

        if use_kernel:
            # Single-pass kernel; bit-identical to the numpy chain below
            flat = array.ravel(order or "C")
            fbs_binned: npt.NDArray[Any] = np.empty(flat.size, dtype=np.int32)
            fbs = sized(
                _discretise_fbs_numba,
                _discretise_fbs_numba_serial,
                flat.size,
                _DISCRETISE_PARALLEL_MIN,
            )
            fbs(flat, float(bin_width), float(current_min), fbs_binned)
            discretised = fbs_binned.reshape(array.shape, order=order or "C")
        else:
            # IBSI FBS: floor((X - X_min) / w_b) + 1
            temp = array - current_min
            if temp.dtype.kind == "f":
                np.divide(temp, bin_width, out=temp)
            else:
                # integer input: out-of-place division promotes to float
                temp = temp / bin_width
            np.floor(temp, out=temp)
            temp += 1

            # Ensure minimum bin is 1 (NaN propagates through np.maximum)
            np.maximum(temp, 1, out=temp)

            # NaN and +inf voxels propagated through the maths; map them to bin 0
            temp[~np.isfinite(temp)] = 0
            discretised = temp.astype(np.int32)

    elif method == "FIXED_CUTOFFS":
        if cutoffs is None:
            raise ValueError("cutoffs required for FIXED_CUTOFFS")

        # +1 so bins are 1-based: values below the first cutoff map to bin 1,
        # keeping 0 reserved for invalid (NaN) voxels.
        edges = np.asarray(cutoffs, dtype=np.float64)
        steps = np.diff(edges)
        increasing = bool(np.all(steps >= 0))
        if use_kernel and (increasing or bool(np.all(steps <= 0))):
            # The bins of np.digitize in one pass, with no int64 array and no copy
            flat = array.ravel(order or "C")
            cut_binned: npt.NDArray[Any] = np.empty(flat.size, dtype=np.int32)
            cut = sized(
                _discretise_cutoffs_numba,
                _discretise_cutoffs_numba_serial,
                flat.size,
                _DISCRETISE_PARALLEL_MIN,
            )
            cut(flat, edges, increasing, cut_binned)
            discretised = cut_binned.reshape(array.shape, order=order or "C")
        else:  # other arrays, and cutoffs that are not monotonic (digitize raises)
            temp_int = np.digitize(array, bins=np.array(cutoffs)) + 1
            temp_int[np.isnan(array)] = 0
            discretised = temp_int.astype(np.int32)

    else:
        raise ValueError(f"Unknown discretisation method: {method}")

    if isinstance(image, Image):
        return Image(
            array=discretised,
            spacing=image.spacing,
            origin=image.origin,
            direction=image.direction,
            modality=image.modality,
        )
    return discretised


def apply_mask(
    image: Image | npt.NDArray[Any],
    mask: Image | npt.NDArray[Any],
    mask_values: int | list[int] | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """
    Apply mask to image and return flattened array of voxel values.

    A large 3D array (from 2^20 voxels, row order, a float64 or int32 image, a uint8, bool
    or float64 mask) is gathered in parallel when 4 or more threads are set. The values and
    their order are the same.

    Args:
        image: Image object or numpy array.
        mask: Image object (mask) or numpy array.
        mask_values: Optional value(s) in the mask to consider as ROI. When
                     omitted, all nonzero mask values are considered ROI
                     membership. Can be a single integer or a list of integers.

    Returns:
        1D numpy array of values within the mask.

    Raises:
        ValueError: If the image and mask arrays have different shapes.

    Example:
        Extract intensities within a spherical mask:

        ```python
        import numpy as np
        from pictologics.preprocessing import apply_mask

        image = np.arange(27.0).reshape(3, 3, 3)
        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        mask[1, 1, 1] = 1

        values = apply_mask(image, mask)
        # array([13.])
        ```
    """
    # Handle inputs
    img_arr = image.array if isinstance(image, Image) else image
    mask_arr = mask.array if isinstance(mask, Image) else mask

    if isinstance(image, Image) and isinstance(mask, Image):
        _validate_geometry(mask, image, "mask", "image")

    # Ensure shapes match
    if img_arr.shape != mask_arr.shape:
        raise ValueError(
            f"Image shape {img_arr.shape} and mask shape {mask_arr.shape} do not match"
        )

    # Handle mask values
    if mask_values is None:
        # A large array takes the parallel gather where it can
        values = (
            None if img_arr.size < _KERNEL_MIN_SIZE else _threaded_roi_values(img_arr, mask_arr)
        )
        if values is not None:
            return values
        roi_mask = mask_arr != 0
    else:
        if isinstance(mask_values, int):
            mask_values = [mask_values]
        roi_mask = np.isin(mask_arr, mask_values)

    # Apply mask (an all-False mask yields an empty array of the image dtype)
    return cast(npt.NDArray[np.floating[Any]], img_arr[roi_mask])


def extract_roi(
    image: Image,
    mask: Image,
    mask_values: int | list[int] | None = None,
) -> Image:
    """
    Extract ROI from image. Voxels outside the mask are set to NaN.
    IBSI 'ROI extraction'.

    Args:
        image: Image object.
        mask: Image object (mask).
        mask_values: Optional value(s) in the mask to consider as ROI. When
            omitted, all nonzero mask values are considered ROI membership.

    Returns:
        New Image object with non-ROI voxels set to NaN.

    Raises:
        ValueError: If `image` and `mask` have different shapes, or otherwise
            occupy different physical geometry (spacing, origin, or direction).

    Example:
        Set voxels outside a mask to NaN:

        ```python
        import numpy as np
        from pictologics.loader import Image
        from pictologics.preprocessing import extract_roi

        array = np.arange(27.0).reshape(3, 3, 3)
        mask_array = np.zeros((3, 3, 3), dtype=np.uint8)
        mask_array[1, 1, 1] = 1

        image = Image(array=array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
        mask = Image(array=mask_array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))

        roi_image = extract_roi(image, mask)
        # roi_image.array is all NaN except at [1, 1, 1], which is 13.0
        ```
    """
    try:
        _validate_geometry(mask, image, "mask", "image")
    except ValueError as exc:
        if "Dimension mismatch" in str(exc):
            raise ValueError("Image and mask must have the same shape.") from exc
        raise

    # Handle mask values
    if mask_values is None:
        roi_mask = mask.array != 0
    else:
        if isinstance(mask_values, int):
            mask_values = [mask_values]
        roi_mask = np.isin(mask.array, mask_values)

    # Preserve floating dtype (NaN-capable); only integer inputs need upcasting. One pass
    # writes the ROI values and the NaN values (a 0-d NaN array keeps the dtype).
    if np.issubdtype(image.array.dtype, np.floating):
        dtype = image.array.dtype
    else:
        dtype = np.dtype(np.float64)
    new_array = np.where(roi_mask, image.array, np.array(np.nan, dtype=dtype))

    return Image(
        array=new_array,
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
        modality=image.modality,
    )


def resegment_mask(
    image: Image,
    mask: Image,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
) -> Image:
    """
    Update mask to exclude voxels where image intensity is outside the specified range.
    Used for IBSI re-segmentation (e.g. [-1000, 400] HU). A NaN intensity is in no
    range, so its voxel always leaves the mask.

    Args:
        image: Image object.
        mask: Image object (mask).
        range_min: Minimum intensity value (inclusive). If None, no lower bound.
        range_max: Maximum intensity value (inclusive). If None, no upper bound.

    Returns:
        Updated Image object (mask) with re-segmentation applied.

    Example:
        Resegment mask to keep only values between -1000 and 400 (e.g. HU range):

        ```python
        from pictologics.preprocessing import resegment_mask

        # Keep voxels in range [-1000, 400]
        new_mask = resegment_mask(
            image,
            mask,
            range_min=-1000,
            range_max=400
        )
        ```
    """
    try:
        _validate_geometry(mask, image, "mask", "image")
    except ValueError as exc:
        if "Dimension mismatch" in str(exc):
            raise ValueError(
                "Image and mask must have the same shape for re-segmentation."
            ) from exc
        raise

    order = _shared_order(image.array, mask.array)
    # The kernel reads and writes the mask as the integer type of its size (the same
    # bits), so a float mask keeps its NaN, infinite and negative values, and 0 is +0.0
    bits = _SAME_SIZE.get(mask.array.dtype.itemsize) if mask.array.dtype.kind in "biuf" else None
    if (
        image.array.dtype == np.float64
        and bits is not None
        and (order is not None or image.array.size >= _KERNEL_MIN_SIZE)
    ):
        # Single-pass kernel; bit-identical to the numpy chain below. It keeps the
        # memory order that the image and the mask share.
        lo = float("-inf") if range_min is None else float(range_min)
        hi = float("inf") if range_max is None else float(range_max)
        flat_out = np.empty(mask.array.size, dtype=mask.array.dtype)
        resegment = sized(
            _resegment_numba, _resegment_numba_serial, flat_out.size, _RESEGMENT_PARALLEL_MIN
        )
        resegment(
            image.array.ravel(order or "C"),
            mask.array.ravel(order or "C").view(bits),
            lo,
            hi,
            flat_out.view(bits),
        )
        new_mask_array = flat_out.reshape(mask.array.shape, order=order or "C")
    else:
        new_mask_array = mask.array.copy()

        # Keep the voxels in [range_min, range_max]; NaN compares False on both bounds
        lo = -np.inf if range_min is None else range_min
        hi = np.inf if range_max is None else range_max
        new_mask_array[~((image.array >= lo) & (image.array <= hi))] = 0

    return Image(
        array=new_mask_array,
        spacing=mask.spacing,
        origin=mask.origin,
        direction=mask.direction,
        modality=mask.modality,
    )


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _roi_values_numba(
    image_flat: npt.NDArray[np.float64], mask_flat: npt.NDArray[Any]
) -> npt.NDArray[np.float64]:
    """Image values where mask != 0, in array order (as image[mask != 0] gives them).

    Each chunk counts its ROI voxels, a running sum gives each chunk its start, and
    the chunks then copy their values in parallel.
    """
    n = image_flat.size
    n_chunks = 64
    step = (n + n_chunks - 1) // n_chunks
    starts = np.zeros(n_chunks + 1, dtype=np.int64)
    for c in prange(n_chunks):
        count = 0
        for i in range(c * step, min(n, (c + 1) * step)):
            if mask_flat[i] != 0:
                count += 1
        starts[c + 1] = count
    for c in range(n_chunks):
        starts[c + 1] += starts[c]
    values = np.empty(starts[n_chunks], dtype=np.float64)
    for c in prange(n_chunks):
        k = starts[c]
        for i in range(c * step, min(n, (c + 1) * step)):
            if mask_flat[i] != 0:
                values[k] = image_flat[i]
                k += 1
    return values


@jit(nopython=True, nogil=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _box_values_numba(
    image: npt.NDArray[Any],
    mask: npt.NDArray[Any],
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> npt.NDArray[Any]:
    """image[box][mask[box] != 0] for the box z0:z1, y0:y1, x0:x1, in row order and in the
    image type (as numpy gives them), read in place. Each row of the box counts its ROI
    voxels, a running sum gives each row its start, and the rows then copy their values
    in parallel. Without the GIL: the morphology worker thread runs at the same time."""
    ny = y1 - y0
    rows = (z1 - z0) * ny
    starts = np.zeros(rows + 1, dtype=np.int64)
    for r in prange(rows):
        z = z0 + r // ny
        y = y0 + r % ny
        count = 0
        for x in range(x0, x1):
            if mask[z, y, x] != 0:
                count += 1
        starts[r + 1] = count
    for r in range(rows):
        starts[r + 1] += starts[r]
    values = np.empty(starts[rows], dtype=image.dtype)
    for r in prange(rows):
        z = z0 + r // ny
        y = y0 + r % ny
        k = starts[r]
        for x in range(x0, x1):
            if mask[z, y, x] != 0:
                values[k] = image[z, y, x]
                k += 1
    return values


def _threaded_roi_values(
    image: npt.NDArray[Any],
    mask: npt.NDArray[Any],
    box: Optional[tuple[slice, slice, slice]] = None,
) -> Optional[npt.NDArray[Any]]:
    """image[box][mask[box] != 0] (box None: the whole arrays) by the parallel gather of
    _box_values_numba: the same values in the same order, with no bool mask and no copy of
    the box. None when numpy's gather is faster (a box of fewer than _KERNEL_MIN_SIZE
    voxels, or fewer than _GATHER_MIN_THREADS threads) or when the kernel does not take
    the arrays (not 3D, other shapes, types or memory order)."""
    if image.ndim != 3 or image.shape != mask.shape:
        return None
    if box is None:
        box = (slice(0, image.shape[0]), slice(0, image.shape[1]), slice(0, image.shape[2]))
    (z0, z1), (y0, y1), (x0, x1) = ((b.start, b.stop) for b in box)
    if (
        (z1 - z0) * (y1 - y0) * (x1 - x0) >= _KERNEL_MIN_SIZE
        and image.dtype in (np.float64, np.int32)
        and mask.dtype in (np.float64, np.uint8, np.bool_)
        and image.flags.c_contiguous
        and mask.flags.c_contiguous
        and get_num_threads() >= _GATHER_MIN_THREADS
    ):
        return cast(npt.NDArray[Any], _box_values_numba(image, mask, z0, z1, y0, y1, x0, x1))
    return None


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _filter_outliers_numba(
    image_flat: npt.NDArray[np.float64],
    mask_flat: npt.NDArray[Any],
    lower: float,
    upper: float,
    out: npt.NDArray[Any],
) -> None:
    """Copy the mask, zeroing ROI voxels whose intensity lies outside [lower, upper].

    NaN intensities compare False on both bounds and are zeroed, as in the numpy path.
    """
    for i in prange(image_flat.size):
        m = mask_flat[i]
        if m != 0:
            v = image_flat[i]
            if v >= lower and v <= upper:
                out[i] = m
            else:
                out[i] = 0
        else:
            out[i] = m


@jit(nopython=True, cache=True)  # type: ignore
def _filter_outliers_serial_numba(
    image_flat: npt.NDArray[np.float64],
    mask_flat: npt.NDArray[Any],
    lower: float,
    upper: float,
    out: npt.NDArray[Any],
) -> None:
    """Serial form of _filter_outliers_numba with the same result. Each voxel reads its
    intensity, and one select (no branch) keeps the mask value when it is 0 or when the
    intensity is in [lower, upper]. A NaN intensity compares False and is zeroed."""
    for i in range(image_flat.size):
        m = mask_flat[i]
        v = image_flat[i]
        if ((v >= lower) & (v <= upper)) | (m == 0):
            out[i] = m
        else:
            out[i] = 0


def filter_outliers(image: Image, mask: Image, sigma: float = 3.0) -> Image:
    """
    Exclude outliers from the mask based on mean +/- sigma * std: the intensity outlier
    filtering of IBSI 1 re-segmentation.

    Args:
        image: Image object.
        mask: Image object (mask).
        sigma: Number of standard deviations.

    Returns:
        New Image object (mask) with outliers removed.

    Example:
        Remove outliers beyond 3 standard deviations from the mean:

        ```python
        from pictologics.preprocessing import filter_outliers

        # Remove outliers > 3 sigma
        clean_mask = filter_outliers(
            image,
            mask,
            sigma=3.0
        )
        ```
    """
    _validate_geometry(mask, image, "mask", "image")

    # Large images use two parallel passes (the ROI values, then the new mask); they
    # give the same values in the same order, so the mean and the std stay the same.
    # Smaller images in a shared memory order make the new mask with a serial kernel.
    size = image.array.size
    order = _shared_order(image.array, mask.array)
    kernel = image.array.dtype == np.float64 and mask.array.dtype in (
        np.float64,
        np.uint8,
        np.bool_,
    )
    large = kernel and size >= _KERNEL_MIN_SIZE

    # Gather intensities within the mask once; all thresholding below runs on
    # this ROI-sized array instead of the full volume
    if large:
        roi_values = _roi_values_numba(image.array.ravel(), mask.array.ravel())
    else:
        roi = mask.array != 0
        roi_values = image.array[roi]

    # Exclude NaNs (e.g. from extract_roi) which would otherwise poison the
    # mean/std and wipe the whole mask; with no NaN, the ROI values are used as they
    # are (no copy)
    nan = np.isnan(roi_values)
    finite_values = roi_values[~nan] if nan.any() else roi_values

    if finite_values.size == 0:
        return mask

    mean_val = np.mean(finite_values)
    # IBSI uses population std (no bias correction, ddof=0)
    std_val = np.std(finite_values, ddof=0)

    lower_bound = mean_val - sigma * std_val
    upper_bound = mean_val + sigma * std_val

    if large or (kernel and order is not None):
        flat_out = np.empty(mask.array.size, dtype=mask.array.dtype)
        zero = sized(_filter_outliers_numba, _filter_outliers_serial_numba, size, _KERNEL_MIN_SIZE)
        zero(
            image.array.ravel(order or "C"),
            mask.array.ravel(order or "C"),
            float(lower_bound),
            float(upper_bound),
            flat_out,
        )
        new_mask_array = flat_out.reshape(mask.array.shape, order=order or "C")
    else:
        # Keep values within [lower, upper]; NaN compares False and is excluded
        keep = (roi_values >= lower_bound) & (roi_values <= upper_bound)

        # Update original mask (zeroing outliers preserves dtype and label values)
        new_mask_array = mask.array.copy()
        roi_mask_values = new_mask_array[roi]
        roi_mask_values[~keep] = 0
        new_mask_array[roi] = roi_mask_values

    return Image(
        array=new_mask_array,
        spacing=mask.spacing,
        origin=mask.origin,
        direction=mask.direction,
        modality=mask.modality,
    )


def round_intensities(image: Image) -> Image:
    """
    Round image intensities to the nearest integer (a half goes to the even integer,
    as with np.round).

    When more than one thread runs, a float64 image of 2^20 voxels or more in row or
    column order is rounded in threads. The result is the same, bit for bit.
    """
    array = image.array
    order = _shared_order(array)
    if (
        array.dtype == np.float64
        and array.size >= _KERNEL_MIN_SIZE
        and order is not None
        and get_num_threads() > 1
    ):
        new_array = np.empty(array.shape, dtype=np.float64, order=order)
        _rint_numba(array.ravel(order), new_array.ravel(order))
    else:
        new_array = np.round(array)
    return Image(
        array=new_array,
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
        modality=image.modality,
    )


def keep_largest_component(mask: Image) -> Image:
    """
    Keep only the largest connected component in the mask.

    Uses full (26-)connectivity in 3D, i.e. voxels touching at a face, edge, or
    vertex belong to the same component. This matches the connectivity IBSI
    prescribes for 3D zones (GLSZM/GLDZM) and the convention used elsewhere in
    this library, rather than scipy's face-only (6-connectivity) default. A mask
    that fills its bounding box is one component, so the function does not label it.
    """
    mask_array = mask.array
    new_array: npt.NDArray[Any] = np.zeros(mask_array.shape, dtype=np.uint8)

    # Label only inside the nonzero bounding box: for a compact ROI in a
    # CT-sized volume this replaces the full-volume connected-component pass
    # with one fast bbox scan plus a small labeling. Cropping removes whole
    # all-zero slabs, so components and their raster-scan label order (which
    # breaks size ties) are unchanged.
    bbox: Optional[tuple[slice, ...]]
    if mask_array.ndim == 3:
        bbox = compute_nonzero_bbox(mask_array)
    else:
        bbox = tuple(slice(None) for _ in mask_array.shape)

    if bbox is not None:
        inside = mask_array[bbox] != 0
        if inside.all():  # A full box is one component: labeling is not necessary.
            new_array[bbox] = 1
        else:
            # Full connectivity: a block of 3 voxels on each axis, all True, connects
            # all diagonal neighbours.
            structure = np.ones((3,) * inside.ndim, dtype=bool)
            labeled_sub, num_features = label(inside, structure=structure)
            if num_features <= 1:
                new_array[bbox] = inside
            else:
                del inside  # Release it before np.bincount makes its intp copy of the labels.
                counts = np.bincount(labeled_sub.ravel())
                counts[0] = 0  # ignore background
                max_label = int(counts.argmax())
                new_array[bbox] = labeled_sub == max_label

    return Image(
        array=new_array,
        spacing=mask.spacing,
        origin=mask.origin,
        direction=mask.direction,
        modality=mask.modality,
    )


# The distance passes of grow_mask and _nearest_roi_map: one pass for each axis, in the
# order 0, 1, 2. A voxel gets ((d0 * s0)^2 + (d1 * s1)^2) + (d2 * s2)^2 for its nearest
# feature voxel: the sum of scipy's distance_transform_edt, in the same order. The passes
# along axes 1 and 2 are the Voronoi step of scipy's feature transform.


@jit(nopython=True, cache=True)  # type: ignore
def _edt_line(
    f: npt.NDArray[np.float64],
    s: float,
    out: npt.NDArray[np.float64],
    src: npt.NDArray[np.int64],
    stack: npt.NDArray[np.int64],
) -> None:
    """One line of a pass with the spacing s. f holds the squared distance (mm^2) of each
    voxel in the axes before (inf: no feature voxel yet). Puts into src[q] the p of the
    least f[p] + ((p - q) * s)^2 and into out[q] that sum: the Voronoi step of scipy's
    feature transform (_VoronoiFT, Maurer et al.), with its removal test, its scan and
    the same operations in the same order. At a tie, the lower p. With no finite f[p], out
    is inf and src is -1. stack is a work line."""
    n = f.size
    top = -1
    for q in range(n):
        wr = f[q]
        if wr == np.inf:
            continue
        while top >= 1:
            p1 = stack[top]
            p0 = stack[top - 1]
            a = (p1 - p0) * s
            b = (q - p1) * s
            c = a + b
            if c * f[p1] - b * f[p0] - a * wr - a * b * c <= 0.0:
                break
            top -= 1
        top += 1
        stack[top] = q
    if top < 0:
        for q in range(n):
            out[q] = np.inf
            src[q] = -1
        return
    k = 0
    for q in range(n):
        step = (stack[k] - q) * s
        d1 = f[stack[k]] + step * step
        while k < top:
            step = (stack[k + 1] - q) * s
            d2 = f[stack[k + 1]] + step * step
            if d1 <= d2:
                break
            d1 = d2
            k += 1
        out[q] = d1
        src[q] = stack[k]


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _edt_axis0_numba(
    feature: npt.NDArray[np.bool_],
    s: float,
    d2: npt.NDArray[np.float64],
    lab: npt.NDArray[np.int32],
    carry: bool,
) -> None:
    """The pass along axis 0: into d2, the squared distance (mm^2, spacing s) to the
    nearest voxel of the axis-0 line where feature is True (inf when the line has none;
    at a tie, the lower one). With carry, each voxel of lab gets the lab value of that
    voxel. One row (axis 1) of each plane for each thread."""
    nz, ny, nx = feature.shape
    for j in prange(ny):
        near = np.empty(nx, np.int64)
        for i in range(nx):
            near[i] = -1
        for k in range(nz):
            for i in range(nx):
                if feature[k, j, i]:
                    near[i] = k
                d2[k, j, i] = k - near[i] if near[i] >= 0 else np.inf
                if carry and near[i] >= 0:
                    lab[k, j, i] = lab[near[i], j, i]
        for i in range(nx):
            near[i] = -1
        for k in range(nz - 1, -1, -1):
            for i in range(nx):
                if feature[k, j, i]:
                    near[i] = k
                if near[i] >= 0 and near[i] - k < d2[k, j, i]:
                    d2[k, j, i] = near[i] - k
                    if carry:
                        lab[k, j, i] = lab[near[i], j, i]
                if d2[k, j, i] != np.inf:
                    step = d2[k, j, i] * s
                    d2[k, j, i] = step * step


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _edt_axis1_numba(
    d2: npt.NDArray[np.float64], s: float, lab: npt.NDArray[np.int32], carry: bool
) -> None:
    """The pass along axis 1: each axis-1 line of d2 becomes the least d2 + (distance
    along axis 1)^2 (see _edt_line); with carry, lab follows. One plane for each thread."""
    nz, ny, nx = d2.shape
    for k in prange(nz):
        f = np.empty(ny)
        out = np.empty(ny)
        src = np.empty(ny, np.int64)
        stack = np.empty(ny, np.int64)
        line = np.empty(ny, np.int32)
        for i in range(nx):
            for j in range(ny):
                f[j] = d2[k, j, i]
            _edt_line(f, s, out, src, stack)
            if src[0] < 0:
                continue
            for j in range(ny):
                d2[k, j, i] = out[j]
            if carry:
                for j in range(ny):
                    line[j] = lab[k, j, i]
                for j in range(ny):
                    lab[k, j, i] = line[src[j]]


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _edt_axis2_numba(
    d2: npt.NDArray[np.float64], s: float, lab: npt.NDArray[np.int32], carry: bool
) -> None:
    """As _edt_axis1_numba, along axis 2 (the rows of one plane for each thread)."""
    nz, ny, nx = d2.shape
    for k in prange(nz):
        f = np.empty(nx)
        out = np.empty(nx)
        src = np.empty(nx, np.int64)
        stack = np.empty(nx, np.int64)
        line = np.empty(nx, np.int32)
        for j in range(ny):
            for i in range(nx):
                f[i] = d2[k, j, i]
            _edt_line(f, s, out, src, stack)
            if src[0] < 0:
                continue
            for i in range(nx):
                d2[k, j, i] = out[i]
            if carry:
                for i in range(nx):
                    line[i] = lab[k, j, i]
                for i in range(nx):
                    lab[k, j, i] = line[src[i]]


_edt_axis0_numba_serial = serial_twin(_edt_axis0_numba)
_edt_axis1_numba_serial = serial_twin(_edt_axis1_numba)
_edt_axis2_numba_serial = serial_twin(_edt_axis2_numba)

# From this many voxels on, the distance passes run in parallel: below, the start of three
# parallel regions costs more than it saves (about 27,000 voxels on 10 threads)
_EDT_PARALLEL_MIN = 30_000


def _squared_distances(
    feature: npt.NDArray[np.bool_],
    spacing: tuple[float, ...],
    lab: Optional[npt.NDArray[np.int32]] = None,
) -> npt.NDArray[np.float64]:
    """The squared distance (mm^2) of each voxel to the nearest True voxel of the 3D bool
    array `feature`, with the spacing of each axis (inf when there is none). These are the
    squares of scipy's distance_transform_edt(~feature, sampling=spacing), bit for bit
    where the scipy build does not fuse a multiply-add. With `lab` (int32, row order, the
    shape of `feature`), each voxel of `lab` gets the value of that nearest voxel, in
    place."""
    feature = np.ascontiguousarray(feature)
    d2 = np.empty(feature.shape)
    work = np.zeros((1, 1, 1), np.int32) if lab is None else lab
    carry = lab is not None
    size = feature.size
    axis0 = sized(_edt_axis0_numba, _edt_axis0_numba_serial, size, _EDT_PARALLEL_MIN)
    axis0(feature, float(spacing[0]), d2, work, carry)
    axis1 = sized(_edt_axis1_numba, _edt_axis1_numba_serial, size, _EDT_PARALLEL_MIN)
    axis1(d2, float(spacing[1]), work, carry)
    axis2 = sized(_edt_axis2_numba, _edt_axis2_numba_serial, size, _EDT_PARALLEL_MIN)
    axis2(d2, float(spacing[2]), work, carry)
    return d2


# A distance within this much (mm) of a grow_mask limit counts as at the limit: 3 steps of
# 0.4 mm are 1.2000000000000002 mm in floating point.
_GROW_TOLERANCE_MM = 1e-6


def _grow_problem(to_mm: Any, from_mm: Any = None) -> Optional[str]:
    """Why grow_mask cannot take these distances, or None when it can."""
    for name, value in (("to_mm", to_mm), ("from_mm", from_mm)):
        if (name == "to_mm" or value is not None) and not (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            return f"{name} must be a finite number (mm), not {value!r}"
    if from_mm is not None and from_mm >= to_mm:
        return f"from_mm ({from_mm}) must be below to_mm ({to_mm})"
    return None


def grow_mask(mask: Image, to_mm: float, from_mm: Optional[float] = None) -> Image:
    """
    Grow or shrink a mask by a distance in mm, or keep a ring at its edge.

    Outside the mask, the distance of a voxel is the distance from its center to the
    nearest center of a mask voxel. Inside the mask, it is the distance to the nearest
    center of a voxel outside the mask (the voxels past the image edge count as outside).
    The distances are in mm, with the spacing of each axis, so the mask changes by the
    same amount in every direction, straight out from its surface.

    A positive `to_mm` adds the voxels up to `to_mm` from the mask. A negative `to_mm`
    removes the mask voxels up to -`to_mm` from the outside. With `from_mm`, the result
    leaves out the mask grown (or shrunk) by `from_mm`, which leaves a ring. On a grid, the
    mask changes by whole voxels: a grow by 1 mm on 0.4 mm voxels adds 2 voxels along an
    axis.

    Args:
        mask: A 3D mask. Every voxel that is not 0 is in the mask.
        to_mm: The distance (mm) to grow the mask by. A negative distance shrinks it.
        from_mm: None, or a distance (mm) below `to_mm`: the result leaves out the mask
            grown (or shrunk, when negative) by this distance.

    Returns:
        The new mask (uint8: 1 in the mask, 0 outside), with the geometry of `mask`.

    Raises:
        ValueError: If a distance is not a finite number, or `from_mm` is not below
            `to_mm`.

    Example:
        ```python
        from pictologics.preprocessing import grow_mask

        ring = grow_mask(mask, to_mm=3, from_mm=0)  # the 3 mm around the mask
        core = grow_mask(mask, to_mm=-1)  # the mask without its outer 1 mm
        rim = grow_mask(mask, to_mm=0, from_mm=-1)  # the outer 1 mm of the mask
        ```
    """
    problem = _grow_problem(to_mm, from_mm)
    if problem:
        raise ValueError(problem)
    array = mask.array
    new_array: npt.NDArray[Any] = np.zeros(array.shape, dtype=np.uint8)
    bbox = compute_nonzero_bbox(array)
    if bbox is not None:
        spacing = tuple(float(s) for s in mask.spacing)
        reach = max(float(to_mm), 0.0) + _GROW_TOLERANCE_MM
        box = tuple(
            slice(max(b.start - int(reach // s), 0), min(b.stop + int(reach // s), n))
            for b, s, n in zip(bbox, spacing, array.shape, strict=True)
        )
        # One voxel more on each side, outside the mask (also past the image edge)
        inside = np.pad(array[box] != 0, 1)
        distances: dict[bool, npt.NDArray[np.float64]] = {}

        def grown(x: float) -> npt.NDArray[np.bool_]:
            """The voxels of the mask grown by x mm (shrunk by -x mm when x < 0)."""
            if x == 0:
                return inside
            outside = x > 0
            if outside not in distances:  # to the nearest voxel in (out of) the mask
                squares = _squared_distances(inside if outside else ~inside, spacing)
                distances[outside] = np.sqrt(squares, out=squares)
            if outside:
                return distances[outside] <= x + _GROW_TOLERANCE_MM
            return distances[outside] > _GROW_TOLERANCE_MM - x

        result = grown(float(to_mm))
        if from_mm is not None:
            result = result & ~grown(float(from_mm))
        new_array[box] = result[1:-1, 1:-1, 1:-1]
    return Image(
        array=new_array,
        spacing=mask.spacing,
        origin=mask.origin,
        direction=mask.direction,
        modality=mask.modality,
    )


def _nearest_roi_map(labels: Image, reach_mm: float) -> Optional[Image]:
    """The label of the nearest ROI voxel of each voxel up to `reach_mm` (and 2 voxels)
    from the box of the ROIs of the label map `labels` (0: background). Each voxel has one
    nearest voxel (the same Voronoi step as scipy's feature transform, in plain float64: a
    voxel at the same distance from two ROIs goes to the same one on every machine and
    thread count), so the voxels of the map have one ROI each. None for a map without
    ROIs."""
    box = compute_nonzero_bbox(labels.array)
    if box is None:
        return None
    margins = [math.ceil(reach_mm / s) + 2 for s in labels.spacing]
    region = cast(
        tuple[slice, slice, slice],
        tuple(
            slice(max(b.start - m, 0), min(b.stop + m, n))
            for b, m, n in zip(box, margins, labels.array.shape, strict=True)
        ),
    )
    # A row-order int32 copy (the labels are at most 1,048,576, see run_rois): the passes
    # write the label of the nearest ROI voxel into it
    nearest: npt.NDArray[Any] = labels.array[region].astype(np.int32, order="C")
    _squared_distances(nearest != 0, labels.spacing, nearest)
    return Image(
        array=nearest,
        spacing=labels.spacing,
        origin=_region_origin(labels.origin, labels.direction, labels.spacing, region),
        direction=labels.direction,
        modality=labels.modality,
    )


def _nearest_roi_part(grown: Image, mask: Image, nearest: Image, label: int) -> Image:
    """`grown` (a mask grown from `mask`, on its grid) without the added voxels whose
    nearest ROI in the map `nearest` (see _nearest_roi_map; the map voxel nearest to each
    voxel center) is not `label`. Changes the array of `grown`."""
    added = np.argwhere((grown.array != 0) & (mask.array == 0))
    to_world = _direction_matrix(grown.direction) * np.asarray(grown.spacing)
    to_map = np.linalg.inv(_direction_matrix(nearest.direction) * np.asarray(nearest.spacing))
    shift = np.asarray(grown.origin) - np.asarray(nearest.origin)
    where = np.rint((added @ to_world.T + shift) @ to_map.T).astype(np.intp)
    np.clip(where, 0, np.array(nearest.array.shape) - 1, out=where)
    other = added[nearest.array[tuple(where.T)] != label]
    grown.array[tuple(other.T)] = 0
    return grown


_NORMALISE_METHODS = ("zscore", "percentile")


def _normalise_problem(percentiles: Any, range_min: Any, range_max: Any) -> Optional[str]:
    """Why normalise_image cannot take these percentiles and value range, or None."""
    numbers = (int, float, np.integer, np.floating)
    if not (
        isinstance(percentiles, (list, tuple))
        and len(percentiles) == 2
        and all(isinstance(p, numbers) and not isinstance(p, bool) for p in percentiles)
        and 0 <= percentiles[0] < percentiles[1] <= 100
    ):
        return (
            f"percentiles must be two numbers from 0 to 100, the first lower, not {percentiles!r}"
        )
    for name, value in (("range_min", range_min), ("range_max", range_max)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, numbers)):
            return f"{name} must be a number, not {value!r}"
    if range_min is not None and range_max is not None and range_min > range_max:
        return f"range_min ({range_min}) must not be above range_max ({range_max})"
    return None


def normalise_image(
    image: Image,
    method: str,
    mask: Optional[Image] = None,
    percentiles: tuple[float, float] = (1.0, 99.0),
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
) -> Image:
    """
    Normalise the intensities of an image with one linear map, (x - center) / scale.

    For MR images and other images without fixed units. The method sets the center and
    the scale from the statistics of a region: `"zscore"` takes the mean and the
    standard deviation (over N), so the region gets mean 0 and standard deviation 1;
    `"percentile"` takes the lower percentile and the distance to the upper one, so
    that range becomes 0 to 1 (`(0, 100)`: the minimum and the maximum). The region is
    the voxels of `mask` (every voxel that is not 0; default: the whole image) with a
    finite value from `range_min` to `range_max` (each optional, for example to leave out
    the background). The map changes every voxel of the image.

    Args:
        image: The image.
        method: `"zscore"` or `"percentile"`.
        mask: The voxels of the statistics, for example a reference tissue. None: the
            whole image.
        percentiles: The lower and the upper percentile of the `"percentile"` method.
        range_min: Leave voxels below this value out of the statistics.
        range_max: Leave voxels above this value out of the statistics.

    Returns:
        The normalised image (float64), with the geometry of `image`.

    Raises:
        ValueError: If an option is not valid, the mask is not on the grid of the image,
            the region holds no voxel, or its values do not spread (a scale of 0).

    Example:
        ```python
        from pictologics.preprocessing import normalise_image

        normalised = normalise_image(mr_image, "zscore", mask=muscle_mask)
        ```
    """
    problem = (
        f"method must be one of {_NORMALISE_METHODS}, not {method!r}"
        if method not in _NORMALISE_METHODS
        else _normalise_problem(percentiles, range_min, range_max)
    )
    if problem:
        raise ValueError(problem)
    if mask is not None:
        _validate_geometry(mask, image, "mask", "image")
    select = None if mask is None else mask.array != 0
    return _normalised(image, method, select, percentiles, range_min, range_max)[0]


def _normalised(
    image: Image,
    method: str,
    select: Optional[npt.NDArray[np.bool_]],
    percentiles: tuple[float, float],
    range_min: Optional[float],
    range_max: Optional[float],
) -> tuple[Image, float, float]:
    """`image` normalised by the statistics of the voxels in `select` (None: all) with a
    finite value in the range, and the center and scale of the map."""
    array = image.array
    values: npt.NDArray[Any]
    if (
        select is None
        and range_min is None
        and range_max is None
        and array.flags.c_contiguous
        and _all_finite(array)
    ):
        # Every voxel counts. numpy reduces a row-order array in the order of its gathered
        # copy, so the statistics of the array are those of the copy, with no copy.
        values = array
    else:
        keep = np.isfinite(array) if select is None else select & np.isfinite(array)
        if range_min is not None:
            keep &= array >= range_min
        if range_max is not None:
            keep &= array <= range_max
        values = array[keep]
        del keep  # free the selection before the map makes its output
    if values.size == 0:
        raise ValueError("The normalise region holds no voxel with a finite value in its range.")
    if method == "zscore":
        center, scale = float(values.mean()), float(values.std())
    else:
        low, high = np.percentile(values, percentiles)
        center, scale = float(low), float(high - low)
    if not scale > 0:
        raise ValueError(
            f"The values of the normalise region do not spread (scale {scale:g}), so they "
            "cannot set the scale."
        )
    order = _shared_order(array)
    if array.dtype == np.float64 and order is not None:
        # One pass in the memory order of the array, with no copy
        flat = array.ravel(order)
        mapped = np.empty(flat.size, dtype=np.float64)
        sized(
            _normalise_map_numba, _normalise_map_numba_serial, flat.size, _NORMALISE_PARALLEL_MIN
        )(flat, center, scale, mapped)
        normalised = mapped.reshape(array.shape, order=order)
    else:
        normalised = np.subtract(array, center, dtype=np.float64)
        normalised /= scale
    return replace(image, array=normalised), center, scale
