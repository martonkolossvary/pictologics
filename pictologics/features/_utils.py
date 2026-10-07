"""
Internal Array Utilities for Feature Extraction
================================================

This module provides shared array manipulation utilities used by texture and morphology
feature calculation modules. These are internal functions not intended for external use.

Note: The underscore prefix (_utils) indicates this is a private module.
"""

from __future__ import annotations

import math
import types
from typing import Any, Optional

import numpy as np
from numba import get_num_threads, jit, prange
from numba.core.cpu_options import ParallelOptions
from numpy import typing as npt

# Parallel options of the kernels that make arrays: prange is the only parallel loop. With
# numba's default, every np.zeros, np.full or np.max in a parallel kernel is a parallel
# region of its own, which starts the threads once more for a few small arrays. (A
# ParallelOptions object: numba empties a plain dict at the first compile.)
PRANGE_ONLY = ParallelOptions(
    {
        "comprehension": False,
        "reduction": False,
        "inplace_binop": False,
        "setitem": False,
        "numpy": False,
        "stencil": False,
        "fusion": False,
        "prange": True,
    }
)


def serial_twin(kernel: Any) -> Any:
    """A serial copy of a parallel numba kernel: the same Python code with parallel=False,
    so prange runs as range and each iteration computes as in the kernel. It serves work
    that is too small to pay the start of a parallel region (about 70 us on 10 threads,
    47 us on 1 thread). The copy has its own name, so the numba cache keeps both. With the
    compiler off, the kernel is Python code that runs serially, so it is its own twin."""
    code = getattr(kernel, "py_func", None)
    if code is None:
        return kernel
    twin = types.FunctionType(
        code.__code__, code.__globals__, code.__name__ + "_serial", code.__defaults__
    )
    twin.__qualname__ = code.__qualname__ + "_serial"
    twin.__module__ = code.__module__
    twin.__doc__ = code.__doc__
    return jit(**{**kernel.targetoptions, "parallel": False}, cache=True)(twin)


def sized(kernel: Any, twin: Any, size: int, parallel_min: int) -> Any:
    """`kernel` for work of `size` items from `parallel_min` on, with more than one thread;
    else its serial `twin` (see serial_twin)."""
    return kernel if size >= parallel_min and get_num_threads() > 1 else twin


# The box and range scans read an integer or bool mask as the integer type of its size: the
# same bits, so the same nonzero voxels. So these scans compile for six mask types (see
# warmup.py).
_SAME_SIZE = {1: np.uint8, 2: np.uint16, 4: np.int32, 8: np.int64}


def _nonzero_form(mask: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """`mask` as the box and range scans read it: an integer or bool mask as the integer
    type of its size (see _SAME_SIZE), with no copy; a float16 mask as a float32 copy (the
    same values, so the same nonzero voxels)."""
    if mask.dtype.kind in "biu":
        return mask.view(_SAME_SIZE[mask.dtype.itemsize])
    if mask.dtype == np.float16:
        return mask.astype(np.float32)
    return mask


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _bbox_scan_numba(
    mask: npt.NDArray[Any],
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Single parallel pass over the mask collecting per-slice nonzero extents.

    Avoids the `mask != 0` boolean temporary and the three separate axis
    reductions of the pure-numpy approach. The whole-row `!= 0` OR-reduction is
    branch-free and SIMD-vectorizable, and runs at memory bandwidth (measured
    ~1.3x over a blocked early-exit scan at CT row widths on float64, and equal
    to a dedicated uint8 max-reduction); the scalar locate loops only touch
    non-empty rows. `!= 0` is correct for every mask dtype: negative values
    count, and NaN counts as nonzero, matching `mask != 0`.
    """
    depth, height, width = mask.shape
    z_any = np.zeros(depth, dtype=np.uint8)
    y_min = np.full(depth, height, dtype=np.int64)
    y_max = np.full(depth, -1, dtype=np.int64)
    x_min = np.full(depth, width, dtype=np.int64)
    x_max = np.full(depth, -1, dtype=np.int64)

    for z in prange(depth):
        for y in range(height):
            # Vectorized any-nonzero test for the whole row.
            hit = False
            for x in range(width):
                hit |= mask[z, y, x] != 0
            if not hit:
                continue
            # First nonzero from the left (row is known non-empty).
            first = 0
            for x in range(width):
                if mask[z, y, x] != 0:
                    first = x
                    break
            # Last nonzero from the right (never left of `first`).
            last = first
            for x in range(width - 1, first, -1):
                if mask[z, y, x] != 0:
                    last = x
                    break

            z_any[z] = 1
            if y < y_min[z]:
                y_min[z] = y
            if y > y_max[z]:
                y_max[z] = y
            if first < x_min[z]:
                x_min[z] = first
            if last > x_max[z]:
                x_max[z] = last

    return z_any, y_min, y_max, x_min, x_max


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _roi_min_max_numba(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[Any],
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Per-slice min/max of the finite `data` values of the `mask != 0` voxels, in a
    single fused pass."""
    depth, height, width = data.shape
    found = np.zeros(depth, dtype=np.uint8)
    mins = np.full(depth, np.inf, dtype=np.float64)
    maxs = np.full(depth, -np.inf, dtype=np.float64)

    for z in prange(depth):
        lo = np.inf
        hi = -np.inf
        hit = False
        for y in range(height):
            for x in range(width):
                if mask[z, y, x] != 0:
                    v = data[z, y, x]
                    if not math.isfinite(v):  # NaN or infinite: no ROI intensity
                        continue
                    hit = True
                    if v < lo:
                        lo = v
                    if v > hi:
                        hi = v
        if hit:
            found[z] = 1
            mins[z] = lo
            maxs[z] = hi

    return found, mins, maxs


@jit(nopython=True, cache=True)  # type: ignore
def _roi_min_max_serial_numba(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[Any],
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Serial variant of `_roi_min_max_numba` for small volumes, where the
    parallel threading-dispatch cost exceeds the scan itself."""
    depth, height, width = data.shape
    found = np.zeros(depth, dtype=np.uint8)
    mins = np.full(depth, np.inf, dtype=np.float64)
    maxs = np.full(depth, -np.inf, dtype=np.float64)

    for z in range(depth):
        lo = np.inf
        hi = -np.inf
        hit = False
        for y in range(height):
            for x in range(width):
                if mask[z, y, x] != 0:
                    v = data[z, y, x]
                    if not math.isfinite(v):  # NaN or infinite: no ROI intensity
                        continue
                    hit = True
                    if v < lo:
                        lo = v
                    if v > hi:
                        hi = v
        if hit:
            found[z] = 1
            mins[z] = lo
            maxs[z] = hi

    return found, mins, maxs


def _column_order(array: npt.NDArray[Any]) -> bool:
    """Whether an array is in column order and not in row order (so not 1-D)."""
    return bool(array.flags.f_contiguous and not array.flags.c_contiguous)


def _row_order_pair(
    data: npt.NDArray[Any], mask: npt.NDArray[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """`data` and `mask` in the layouts that the range scans compile for at the import:
    two column-order arrays as their transposes (no copy), else row-order copies of the
    arrays that are not in row order. One exception stays as it is: a strided `data` with a
    row-order uint8 `mask`, the grey-level crop of the texture features, which the import
    warms. The scans read every voxel once, so the layout does not change their result."""
    if _column_order(data) and _column_order(mask):
        return data.T, mask.T
    if not mask.flags.c_contiguous:
        mask = np.ascontiguousarray(mask)
    if not data.flags.c_contiguous and (_column_order(data) or mask.dtype != np.uint8):
        data = np.ascontiguousarray(data)
    return data, mask


def roi_min_max(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[Any],
) -> Optional[tuple[float, float]]:
    """Min and max of the finite `data` values of the ROI voxels (`mask != 0`, as
    everywhere in the package). NaN and infinite values are no ROI intensities.

    Equivalent to the min and max of `v[np.isfinite(v)]` with `v = data[mask != 0]`, but
    in a single fused pass, without the boolean-mask and gathered-copy temporaries.

    Args:
        data: 3D array of values.
        mask: 3D array where nonzero values indicate ROI membership. Same shape as data.

    Returns:
        (min, max) over the ROI, or None if no ROI voxel has a finite value.
    """
    if data.ndim != 3 or data.shape != mask.shape:
        raise ValueError(
            f"Expected two 3D arrays of equal shape, got {data.shape!r} vs {mask.shape!r}"
        )
    data, mask = _row_order_pair(data, _nonzero_form(mask))
    # Measured crossover: below ~2^19 voxels the parallel launch overhead dominates.
    if data.size < 1 << 19:
        found, mins, maxs = _roi_min_max_serial_numba(data, mask)
    else:
        found, mins, maxs = _roi_min_max_numba(data, mask)
    idx = np.flatnonzero(found)
    if idx.size == 0:
        return None
    return float(mins[idx].min()), float(maxs[idx].max())


_bbox_scan_numba_serial = serial_twin(_bbox_scan_numba)
# Voxels from which the box scan runs in parallel; below, its serial twin (measured: the twin
# takes 0.37 to 0.97 of the numpy reductions of before up to 2^20 voxels)
_BBOX_PARALLEL_MIN = 1 << 20


def compute_nonzero_bbox(
    mask: npt.NDArray[Any],
) -> Optional[tuple[slice, slice, slice]]:
    """Compute the tight bounding box of non-zero voxels in a 3D mask.

    Args:
        mask: 3D array where non-zero indicates ROI.

    Returns:
        A tuple of slices (z, y, x) covering the non-zero region, or None if the mask is empty.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape={mask.shape!r}")

    mask = _nonzero_form(mask)
    if not mask.flags.c_contiguous:
        if mask.flags.f_contiguous:  # column order: the box of the row-order transpose
            box = compute_nonzero_bbox(mask.T)
            return None if box is None else (box[2], box[1], box[0])
        mask = np.ascontiguousarray(mask)
    scan = sized(_bbox_scan_numba, _bbox_scan_numba_serial, mask.size, _BBOX_PARALLEL_MIN)
    z_any, y_min, y_max, x_min, x_max = scan(mask)
    nz = np.flatnonzero(z_any)
    if nz.size == 0:
        return None

    z0, z1 = int(nz[0]), int(nz[-1])
    y0 = int(y_min[nz].min())
    y1 = int(y_max[nz].max())
    x0 = int(x_min[nz].min())
    x1 = int(x_max[nz].max())

    return slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1)


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _label_extents_numba(
    labels: npt.NDArray[Any],
    box: npt.NDArray[np.int64],
    ext: npt.NDArray[np.int64],
    top: npt.NDArray[np.int64],
) -> None:
    """ext[c, l] = the first and last z, y and x of label l (0 < l < ext.shape[1]) in chunk
    c: the slabs of axis 0 of the box (z0, z1, y0, y1, x0, x1), split into ext.shape[0]
    chunks (a first above the last: label l is not in chunk c). top[c] = the largest label
    of chunk c, or -1 for a label below 0 (a large unsigned label read as a signed one). A
    row without a label costs one vectorised test, as in _bbox_scan_numba."""
    n_chunks, size = ext.shape[0], ext.shape[1]
    z0, z1, y0, y1, x0, x1 = box[0], box[1], box[2], box[3], box[4], box[5]
    for c in prange(n_chunks):
        for k in range(size):
            ext[c, k, 0] = z1
            ext[c, k, 1] = -1
            ext[c, k, 2] = y1
            ext[c, k, 3] = -1
            ext[c, k, 4] = x1
            ext[c, k, 5] = -1
        largest = np.int64(0)
        for z in range(z0 + (z1 - z0) * c // n_chunks, z0 + (z1 - z0) * (c + 1) // n_chunks):
            for y in range(y0, y1):
                hit = False
                for x in range(x0, x1):
                    hit |= labels[z, y, x] != 0
                if not hit:
                    continue
                for x in range(x0, x1):
                    label = np.int64(labels[z, y, x])
                    if label == 0:
                        continue
                    if label < 0:
                        largest = np.int64(-1)
                        break
                    if largest >= 0 and label > largest:
                        largest = label
                    if label < size:
                        ext[c, label, 0] = min(ext[c, label, 0], z)
                        ext[c, label, 1] = max(ext[c, label, 1], z)
                        ext[c, label, 2] = min(ext[c, label, 2], y)
                        ext[c, label, 3] = max(ext[c, label, 3], y)
                        ext[c, label, 4] = min(ext[c, label, 4], x)
                        ext[c, label, 5] = max(ext[c, label, 5], x)
        top[c] = largest


# The largest label of the tables of label_boxes: 4 chunks for each thread of 4,097 x 6
# int64 values (8 MB at 10 threads). A map with a larger label takes find_objects.
_LABEL_TABLE_MAX = 4096


def label_boxes(
    labels: npt.NDArray[Any], box: tuple[slice, slice, slice]
) -> Optional[list[Optional[tuple[slice, slice, slice]]]]:
    """The box of each label 1, 2, ... of a 3D map of whole numbers of 0 or above in row
    order, with `box` its nonzero box (None for a label that is not in the map), as
    `scipy.ndimage.find_objects` gives them: one parallel pass with a table of each label
    for each slab. None for a label above _LABEL_TABLE_MAX (find_objects then)."""
    bounds = np.array([box[0].start, box[0].stop, box[1].start, box[1].stop, box[2].start, box[2].stop])  # fmt: skip
    n_chunks = min(4 * get_num_threads(), box[0].stop - box[0].start)
    size = 256
    while True:
        ext = np.empty((n_chunks, size, 6), dtype=np.int64)
        top = np.empty(n_chunks, dtype=np.int64)
        _label_extents_numba(_nonzero_form(labels), bounds.astype(np.int64), ext, top)
        largest = int(top.max())
        if (top < 0).any() or largest > _LABEL_TABLE_MAX:
            return None
        if largest < size:
            break
        size = largest + 1  # one more pass with a row for each label
    first = ext[:, 1 : largest + 1, 0::2].min(axis=0)
    last = ext[:, 1 : largest + 1, 1::2].max(axis=0)
    return [
        None
        if last[k, 0] < 0
        else (
            slice(int(first[k, 0]), int(last[k, 0]) + 1),
            slice(int(first[k, 1]), int(last[k, 1]) + 1),
            slice(int(first[k, 2]), int(last[k, 2]) + 1),
        )
        for k in range(largest)
    ]


@jit(nopython=True, nogil=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _copy_numba(source: npt.NDArray[np.float64], out: npt.NDArray[np.float64]) -> None:
    """out = source, the planes of axis 0 in parallel."""
    for i in prange(out.shape[0]):
        for j in range(out.shape[1]):
            for k in range(out.shape[2]):
                out[i, j, k] = source[i, j, k]


# From this many float64 values on, with more than one thread, region_copy copies in
# parallel (measured on regions of a 512 x 512 x 200 image: np.array takes 24 us at 50^3,
# 165 us at 64^3 and 494 us at 86 x 86 x 90; the parallel copy about 115 and 130 us)
_COPY_PARALLEL_MIN = 1 << 18


def region_copy(view: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """np.array(view, order="C"): a large float64 3-D view in row order or strided is copied
    in parallel, other arrays by numpy."""
    if (
        view.dtype == np.float64
        and view.ndim == 3
        and view.size >= _COPY_PARALLEL_MIN
        and get_num_threads() > 1
        and (view.flags.c_contiguous or not view.flags.f_contiguous)
    ):
        out = np.empty(view.shape, dtype=np.float64)
        _copy_numba(view, out)
        return out
    return np.array(view, order="C")


def merge_bboxes(
    a: Optional[tuple[slice, slice, slice]],
    b: Optional[tuple[slice, slice, slice]],
) -> Optional[tuple[slice, slice, slice]]:
    """Merge two nonzero bounding boxes per axis.

    The bbox of a union of two nonzero sets is the per-axis merge of their bboxes,
    so no union array needs to be materialised. None (empty mask) is the identity.
    """
    if a is None:
        return b
    if b is None:
        return a
    return (
        slice(min(a[0].start, b[0].start), max(a[0].stop, b[0].stop)),
        slice(min(a[1].start, b[1].start), max(a[1].stop, b[1].stop)),
        slice(min(a[2].start, b[2].start), max(a[2].stop, b[2].stop)),
    )
