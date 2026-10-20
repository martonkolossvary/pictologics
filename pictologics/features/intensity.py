"""
Intensity Feature Extraction Module
===================================

This module provides functions for calculating First Order Statistics (Intensity)
features from medical images. It implements the Image Biomarker Standardisation
Initiative (IBSI) compliant algorithms.

Key Features:
-------------
- **First Order Statistics**: Mean, Variance, Skewness, Kurtosis, Percentiles, etc.
- **Intensity Histogram**: Features based on discretised intensity histograms.
- **Intensity-Volume Histogram (IVH)**: Volume fractions and intensity fractions, AUC.
- **Spatial Intensity**: Moran's I and Geary's C (spatial autocorrelation) [Optimized].
- **Local Intensity**: Local and Global Intensity Peaks.

Optimization:
-------------
Uses `numba` for JIT compilation, with parallel execution for computationally intensive
spatial feature calculations.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import numpy as np
import scipy.fft
from numba import get_num_threads, jit, prange
from numpy import typing as npt

from ..threads import _memory_budget
from ._utils import PRANGE_ONLY, compute_nonzero_bbox, serial_twin, sized

if TYPE_CHECKING:
    from ..loader import Image


@jit(nopython=True, fastmath=True, cache=True)  # type: ignore
def _sum_sq_centered(values: npt.NDArray[np.floating[Any]], mean_val: float) -> float:
    """Compute sum of squared deviations from mean (for Moran's I denominator)."""
    total = 0.0
    for i in range(values.size):
        d = float(values[i]) - mean_val
        total += d * d
    return total


@jit(nopython=True, cache=True)  # type: ignore
def _sum_of_squares(values: npt.NDArray[np.floating[Any]]) -> float:
    """The sum of the squares of `values` in a fixed order: blocks of 4096 values, each
    summed in eight accumulators (no fastmath, so the order is the source order). The
    same values give the same bits in every buffer; np.dot (BLAS) blocks by the address
    of the buffer and gives values one ulp apart."""
    n = values.shape[0]
    total = 0.0
    start = 0
    while start < n:
        stop = min(start + 4096, n)
        a0 = 0.0
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        a4 = 0.0
        a5 = 0.0
        a6 = 0.0
        a7 = 0.0
        i = start
        while i + 8 <= stop:
            a0 += values[i] * values[i]
            a1 += values[i + 1] * values[i + 1]
            a2 += values[i + 2] * values[i + 2]
            a3 += values[i + 3] * values[i + 3]
            a4 += values[i + 4] * values[i + 4]
            a5 += values[i + 5] * values[i + 5]
            a6 += values[i + 6] * values[i + 6]
            a7 += values[i + 7] * values[i + 7]
            i += 8
        block = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))
        while i < stop:
            block += values[i] * values[i]
            i += 1
        total += block
        start = stop
    return total


@jit(nopython=True, fastmath=True, cache=True)  # type: ignore
def _central_moments_2_3_4(
    values: npt.NDArray[np.floating[Any]], mean_val: float
) -> tuple[float, float, float]:
    """Compute 2nd, 3rd, and 4th central moments in a single pass (for skewness/kurtosis)."""
    n = values.size
    if n == 0:
        return 0.0, 0.0, 0.0

    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for i in range(n):
        d = float(values[i]) - mean_val
        d2 = d * d
        m2 += d2
        m3 += d2 * d
        m4 += d2 * d2

    inv_n = 1.0 / n
    return m2 * inv_n, m3 * inv_n, m4 * inv_n


@jit(nopython=True, fastmath=True, cache=True)  # type: ignore
def _mean_abs_dev(values: npt.NDArray[np.floating[Any]], center: float) -> float:
    """Compute mean absolute deviation from a center value (for MAD features)."""
    n = values.size
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        total += abs(float(values[i]) - center)
    return float(total / n)


@jit(nopython=True, fastmath=True, cache=True)  # type: ignore
def _robust_mean_abs_dev(
    values: npt.NDArray[np.floating[Any]], lower: float, upper: float
) -> float:
    """Compute robust MAD using only values in [lower, upper] range (two-pass, no allocation)."""
    n = values.size
    if n == 0:
        return 0.0

    count = 0
    total = 0.0
    for i in range(n):
        v = float(values[i])
        if v >= lower and v <= upper:
            total += v
            count += 1

    if count == 0:
        return 0.0

    mean_val = total / count
    dev_total = 0.0
    for i in range(n):
        v = float(values[i])
        if v >= lower and v <= upper:
            dev_total += abs(v - mean_val)

    return dev_total / count


@jit(nopython=True, parallel=PRANGE_ONLY, fastmath=True, cache=True)  # type: ignore
def _calculate_spatial_features_numba(
    x_idx: npt.NDArray[np.integer[Any]],
    y_idx: npt.NDArray[np.integer[Any]],
    z_idx: npt.NDArray[np.integer[Any]],
    intensities: npt.NDArray[np.floating[Any]],
    mean_int: float,
    sx: float,
    sy: float,
    sz: float,
) -> tuple[float, float, float]:
    """
    Calculate Moran's I and Geary's C components using Numba with Parallelization.

    This feature is O(N^2) complexity where N is the number of ROI voxels.
    Parallel execution significantly speeds up the outer loop.

    Voxel coordinates must be pairwise distinct (guaranteed when they come from
    np.where on a mask); duplicate coordinates would divide by zero.

    Args:
        x_idx: (N,) x indices for ROI voxels.
        y_idx: (N,) y indices for ROI voxels.
        z_idx: (N,) z indices for ROI voxels.
        intensities: (N,) array of voxel intensities.
        mean_int: Mean intensity of the ROI.
        sx: Voxel spacing in x (mm).
        sy: Voxel spacing in y (mm).
        sz: Voxel spacing in z (mm).

    Returns:
        Tuple containing:
        - numer_moran: Numerator for Moran's I: sum of w_ij * diff_i * diff_j.
        - numer_geary: Numerator for Geary's C: sum of w_ij * (x_i - x_j)^2.
        - sum_weights: Sum of all weights (inverse distances).
    """
    n = intensities.size

    # Reduction arrays to avoid Numba parallel reduction cycle issues
    # Allocate arrays to store partial results for each voxel
    local_moran_arr = np.zeros(n, dtype=np.float64)
    local_geary_arr = np.zeros(n, dtype=np.float64)
    local_w_sum_arr = np.zeros(n, dtype=np.float64)

    # Pre-cast coordinates to physical (mm) float64 once (O(N)) so the O(N^2)
    # inner loop avoids per-iteration int->float conversion, spacing multiplies,
    # and repeated mean subtraction, which lets the compiler vectorize it.
    xf = x_idx.astype(np.float64) * sx
    yf = y_idx.astype(np.float64) * sy
    zf = z_idx.astype(np.float64) * sz
    difff = intensities.astype(np.float64) - mean_int

    # The weight w_ij = 1/dist(i, j) is symmetric, so only the upper-triangular
    # pairs (j > i) need to be visited; each pair's symmetric contribution is
    # accumulated for both endpoints. This halves the O(N^2) inner work.
    #
    # Upper-triangular rows have inner length (n-1-i), which decreases with i.
    # Under static parallel scheduling that starves later threads, so map the
    # loop index k to a folded row index that interleaves long and short rows,
    # keeping each chunk balanced. Every row i is still visited exactly once and
    # results are stored by true row index, so the reduction is unchanged.
    for k in prange(n):
        if k % 2 == 0:
            i = k // 2
        else:
            i = n - 1 - (k // 2)

        local_moran = 0.0
        local_geary = 0.0
        local_w_sum = 0.0

        diff_i = difff[i]

        xi = xf[i]
        yi = yf[i]
        zi = zf[i]

        # Inner loop over j > i only. Distinct ROI voxels always have distance
        # > 0, so no divide-by-zero guard is needed (removing it enables SIMD).
        for j in range(i + 1, n):
            dx = xi - xf[j]
            dy = yi - yf[j]
            dz = zi - zf[j]
            d_sq = dx * dx + dy * dy + dz * dz

            w = 1.0 / np.sqrt(d_sq)
            diff_j = difff[j]
            # (x_i - x_j) == (diff_i - diff_j): the mean cancels.
            d_v = diff_i - diff_j

            local_w_sum += w
            local_moran += w * diff_i * diff_j
            local_geary += w * d_v * d_v

        # Store partial (upper-triangular) sums for this row.
        local_moran_arr[i] = local_moran
        local_geary_arr[i] = local_geary
        local_w_sum_arr[i] = local_w_sum

    # Full sums: each unordered pair contributes twice to the ordered-pair sums. The rows
    # add in one pass (PRANGE_ONLY: numba does not split these sums between the threads),
    # so the sums do not depend on the number of threads.
    numer_moran = 2.0 * np.sum(local_moran_arr)
    numer_geary = 2.0 * np.sum(local_geary_arr)
    sum_weights = 2.0 * np.sum(local_w_sum_arr)

    return numer_moran, numer_geary, sum_weights


# The serial twin of the pair loop (see serial_twin). Measured on 10 threads: below this
# many ROI voxels, starting the threads costs more than the pairs.
_calculate_spatial_features_numba_serial = serial_twin(_calculate_spatial_features_numba)
_PAIRS_PARALLEL_MIN = 450


@jit(nopython=True, parallel=PRANGE_ONLY, fastmath=True, cache=True)  # type: ignore
def _calculate_local_mean_numba(
    data: npt.NDArray[np.floating[Any]],
    mask_indices: npt.NDArray[np.integer[Any]],
    offsets: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Calculate local mean intensity in sphere neighborhood for each ROI voxel (parallel)."""
    n_voxels = mask_indices.shape[0]
    means = np.zeros(n_voxels, dtype=np.float64)

    for i in prange(n_voxels):
        x = mask_indices[i, 0]
        y = mask_indices[i, 1]
        z = mask_indices[i, 2]

        sum_val = 0.0
        count = 0

        for j in range(offsets.shape[0]):
            nx = x + offsets[j, 0]
            ny = y + offsets[j, 1]
            nz = z + offsets[j, 2]

            if nx < 0 or ny < 0 or nz < 0:
                continue
            if nx >= data.shape[0] or ny >= data.shape[1] or nz >= data.shape[2]:
                continue

            sum_val += float(data[nx, ny, nz])
            count += 1

        if count > 0:
            means[i] = sum_val / count

    return means


# The serial twin of the exact kernel, for few (ROI voxel, sphere offset) pairs (see serial_twin)
_calculate_local_mean_numba_serial = serial_twin(_calculate_local_mean_numba)
# Pairs from which the exact kernel and the approximate kernels run in threads
_LOCAL_MEAN_PARALLEL_MIN = 1 << 17


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _roi_count_numba(
    roi: npt.NDArray[np.bool_], voxels: npt.NDArray[np.int64], runs: npt.NDArray[np.int64]
) -> None:
    """voxels[x], runs[x]: the ROI voxels and the z-runs (ROI voxels that follow each other
    along z) of slab x of the bool array `roi`."""
    nx, ny, nz = roi.shape
    for x in prange(nx):
        nv = 0
        nr = 0
        for y in range(ny):
            prev = False
            for z in range(nz):
                inside = roi[x, y, z]
                if inside:
                    nv += 1
                    if not prev:
                        nr += 1
                prev = inside
        voxels[x] = nv
        runs[x] = nr


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _roi_fill_numba(
    roi: npt.NDArray[np.bool_],
    data: npt.NDArray[np.floating[Any]],
    voxel_start: npt.NDArray[np.int64],
    run_start: npt.NDArray[np.int64],
    indices: npt.NDArray[np.int32],
    values: npt.NDArray[np.float64],
    runs: npt.NDArray[np.int64],
) -> None:
    """The ROI voxels of slab x from voxel_start[x] on, in row order: indices[i] = (x, y, z),
    values[i] = data[x, y, z]; its z-runs from run_start[x] on: runs[r] = (x, y, first z,
    last z + 1, first voxel index)."""
    nx, ny, nz = roi.shape
    for x in prange(nx):
        i = voxel_start[x]
        r = run_start[x]
        for y in range(ny):
            prev = False
            for z in range(nz):
                inside = roi[x, y, z]
                if inside:
                    indices[i, 0] = x
                    indices[i, 1] = y
                    indices[i, 2] = z
                    values[i] = data[x, y, z]
                    if not prev:
                        runs[r, 0] = x
                        runs[r, 1] = y
                        runs[r, 2] = z
                        runs[r, 4] = i
                        r += 1
                    runs[r - 1, 3] = z + 1
                    i += 1
                prev = inside


_roi_count_numba_serial = serial_twin(_roi_count_numba)
_roi_fill_numba_serial = serial_twin(_roi_fill_numba)
# Crop voxels from which the ROI scan and the prefix sums run in threads
_CROP_PARALLEL_MIN = 1 << 18


def _roi_scan(
    roi: npt.NDArray[np.bool_], data: npt.NDArray[np.floating[Any]]
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """The ROI voxels of the row-order bool array `roi` in row order: indices (n, 3) int32,
    values (n) float64 and the z-runs (m, 5) int64 (see _roi_fill_numba)."""
    nx = roi.shape[0]
    voxels = np.empty(nx, dtype=np.int64)
    nruns = np.empty(nx, dtype=np.int64)
    count = sized(_roi_count_numba, _roi_count_numba_serial, roi.size, _CROP_PARALLEL_MIN)
    count(roi, voxels, nruns)
    voxel_start = np.cumsum(voxels) - voxels
    run_start = np.cumsum(nruns) - nruns
    n = int(voxel_start[-1] + voxels[-1])
    indices = np.empty((n, 3), dtype=np.int32)
    values = np.empty(n, dtype=np.float64)
    runs = np.empty((int(run_start[-1] + nruns[-1]), 5), dtype=np.int64)
    if n:
        fill = sized(_roi_fill_numba, _roi_fill_numba_serial, roi.size, _CROP_PARALLEL_MIN)
        fill(roi, data, voxel_start, run_start, indices, values, runs)
    return indices, values, runs


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _prefix_pad_numba(
    data: npt.NDArray[np.floating[Any]],
    pad: int,
    out: npt.NDArray[np.float64],
    lo: npt.NDArray[np.float64],
    hi: npt.NDArray[np.float64],
) -> None:
    """out[x, y, pad + k] = the sum of data[x, y, :k] (float64), with `pad` zeros before and
    `pad` copies of the row total after, so a sphere row clipped to the image reads the
    same two sums as an unclipped one. lo[x], hi[x]: the smallest and the largest value of
    slab x (a NaN value goes into the totals, not into lo and hi)."""
    nx, ny, nz = data.shape
    for x in prange(nx):
        mn = np.inf
        mx = -np.inf
        for y in range(ny):
            s = 0.0
            for k in range(pad + 1):
                out[x, y, k] = 0.0
            for z in range(nz):
                v = float(data[x, y, z])
                s += v
                out[x, y, pad + z + 1] = s
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
            for k in range(pad):
                out[x, y, pad + nz + 1 + k] = s
        lo[x] = mn
        hi[x] = mx


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _approximate_local_means_numba(
    prefix: npt.NDArray[np.float64],
    mask_indices: npt.NDArray[np.integer[Any]],
    row_dx: npt.NDArray[np.int64],
    row_dy: npt.NDArray[np.int64],
    row_dz: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """The sphere mean of each ROI voxel from the row prefix sums: one subtraction per
    sphere row (dx, dy, -dz..dz), clipped to the image. Close to the exact local mean;
    `_local_peaks_two_stage` bounds the difference."""
    nx, ny, nz1 = prefix.shape
    nz = nz1 - 1
    out = np.empty(mask_indices.shape[0], dtype=np.float64)
    for i in prange(mask_indices.shape[0]):
        x = mask_indices[i, 0]
        y = mask_indices[i, 1]
        z = mask_indices[i, 2]
        s = 0.0
        count = 0
        for r in range(row_dx.shape[0]):
            xx = x + row_dx[r]
            yy = y + row_dy[r]
            if xx < 0 or yy < 0 or xx >= nx or yy >= ny:
                continue
            lo = max(z - row_dz[r], 0)
            hi = min(z + row_dz[r] + 1, nz)
            s += prefix[xx, yy, hi] - prefix[xx, yy, lo]
            count += hi - lo
        out[i] = s / count
    return out


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _approx_runs_numba(
    prefix: npt.NDArray[np.float64],
    pad: int,
    nz: int,
    runs: npt.NDArray[np.int64],
    row_dx: npt.NDArray[np.int64],
    row_dy: npt.NDArray[np.int64],
    row_dz: npt.NDArray[np.int64],
    out: npt.NDArray[np.float64],
) -> None:
    """out[i] = the sphere mean of ROI voxel i from the padded row prefix sums, z-run by
    z-run: for each sphere row (dx, dy, -dz..dz) one subtraction per voxel of the run, read
    from the row (xx, yy) of `prefix` in one piece. The terms of each voxel are those of
    `_approximate_local_means_numba`, in the same order."""
    nx, ny = prefix.shape[0], prefix.shape[1]
    for q in prange(runs.shape[0]):
        x = runs[q, 0]
        y = runs[q, 1]
        z0 = runs[q, 2]
        length = runs[q, 3] - z0
        i0 = runs[q, 4]
        s = np.zeros(length, dtype=np.float64)
        cnt = np.zeros(length, dtype=np.int64)
        for r in range(row_dx.shape[0]):
            xx = x + row_dx[r]
            yy = y + row_dy[r]
            if xx < 0 or yy < 0 or xx >= nx or yy >= ny:
                continue
            h = row_dz[r]
            line = prefix[xx, yy]
            base_hi = pad + z0 + h + 1
            base_lo = pad + z0 - h
            for k in range(length):
                s[k] += line[base_hi + k] - line[base_lo + k]
            for k in range(length):
                z = z0 + k
                cnt[k] += min(z + h + 1, nz) - max(z - h, 0)
        for k in range(length):
            out[i0 + k] = s[k] / cnt[k]


_prefix_pad_numba_serial = serial_twin(_prefix_pad_numba)
_approximate_local_means_numba_serial = serial_twin(_approximate_local_means_numba)
_approx_runs_numba_serial = serial_twin(_approx_runs_numba)
# Mean z-run length (ROI voxels per run) from which the run kernel beats the voxel kernel
_RUN_MEAN_LENGTH_MIN = 8


@jit(nopython=True, fastmath=True, cache=True)  # type: ignore
def _calculate_local_peaks_numba(
    data: npt.NDArray[np.floating[Any]],
    mask_indices: npt.NDArray[np.integer[Any]],
    roi_means: npt.NDArray[np.floating[Any]],
) -> tuple[float, float]:
    """Compute global/local intensity peaks from pre-computed local means (IBSI 4.5)."""
    global_peak = -1.0e308
    max_intensity = -1.0e308
    local_peak = -1.0e308

    n = mask_indices.shape[0]
    for i in range(n):
        x = mask_indices[i, 0]
        y = mask_indices[i, 1]
        z = mask_indices[i, 2]

        mean_val = float(roi_means[i])
        if mean_val > global_peak:
            global_peak = mean_val

        v = float(data[x, y, z])
        if v > max_intensity:
            max_intensity = v
            local_peak = mean_val
        elif v == max_intensity and mean_val > local_peak:
            local_peak = mean_val

    return global_peak, local_peak


@lru_cache(maxsize=32)
def _sphere_offsets_for_radius(
    spacing: tuple[float, float, float], radius_mm: float
) -> npt.NDArray[np.int32]:
    """Generate voxel offsets for a sphere of given radius (cached for reuse)."""
    sx, sy, sz = spacing
    rx = int(np.ceil(radius_mm / sx))
    ry = int(np.ceil(radius_mm / sy))
    rz = int(np.ceil(radius_mm / sz))

    radius_sq = float(radius_mm * radius_mm)
    offsets = []
    for dx in range(-rx, rx + 1):
        px = float(dx) * sx
        for dy in range(-ry, ry + 1):
            py = float(dy) * sy
            for dz in range(-rz, rz + 1):
                pz = float(dz) * sz
                if (px * px + py * py + pz * pz) <= radius_sq:
                    offsets.append((dx, dy, dz))

    return np.ascontiguousarray(np.array(offsets, dtype=np.int32))


@jit(nopython=True, cache=True)  # type: ignore
def _percentile_ranks(n: int) -> npt.NDArray[np.int64]:
    """0-based ranks of P10, P25, P75 and P90 among n sorted values.

    This is the index rule of `np.percentile(..., method="inverted_cdf")`, with the same
    float64 steps (numpy divides the percentiles by 100 in float64 for values of every
    type), so the ranks are the same as numpy's.
    """
    ranks = np.empty(4, dtype=np.int64)
    q = (10.0 / 100.0, 25.0 / 100.0, 75.0 / 100.0, 90.0 / 100.0)
    for j in range(4):
        index = n * q[j] - 1.0
        below = math.floor(index)
        ranks[j] = below if index - below == 0.0 else below + 1
    return ranks


# From this many float64 values on, the order statistics come from a radix select: the key
# range from the smallest and the largest value, one parallel count of about 65,536 key
# buckets over that range, then a partition of the few values in the buckets of the ranks.
# Below it, one partition is faster (measured on CT-like, mixed-sign, LoG-like and bimodal
# values: from 100,000 values on the radix select wins for each; at 80,000, not for all).
_RADIX_SELECT_MIN = 100_000
_SIGN_BIT = np.uint64(1 << 63)
_ALL_BITS = np.uint64(0xFFFFFFFFFFFFFFFF)


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _float_key(bits: np.uint64) -> np.uint64:
    """The float64 order as an unsigned order: flip the sign bit of a positive value and
    every bit of a negative one."""
    return bits ^ _ALL_BITS if bits >> np.uint64(63) else bits ^ _SIGN_BIT


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _bucket_counts_numba(
    bits: npt.NDArray[np.uint64], shift: int, base: np.uint64, counts: npt.NDArray[np.int64]
) -> None:
    """counts[t, h]: the values of chunk t in bucket h = (key >> shift) - base. Each chunk
    sets its row to 0 first."""
    n = bits.size
    n_chunks = counts.shape[0]
    size = (n + n_chunks - 1) // n_chunks
    for t in prange(n_chunks):
        for h in range(counts.shape[1]):
            counts[t, h] = 0
        for i in range(t * size, min(n, (t + 1) * size)):
            counts[t, (_float_key(bits[i]) >> np.uint64(shift)) - base] += 1


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _bucket_values_numba(
    bits: npt.NDArray[np.uint64],
    values: npt.NDArray[np.float64],
    shift: int,
    base: np.uint64,
    wanted: npt.NDArray[np.bool_],
    chunk_counts: npt.NDArray[np.int64],
    out: npt.NDArray[np.float64],
) -> None:
    """Copy the values whose bucket h has wanted[h] into `out`, chunk by chunk in parallel.
    chunk_counts[t] holds the number of them in chunk t (the chunks of
    `_bucket_counts_numba`); the order inside `out` does not matter."""
    n = bits.size
    n_chunks = chunk_counts.size
    size = (n + n_chunks - 1) // n_chunks
    for t in prange(n_chunks):
        c = 0
        for k in range(t):
            c += chunk_counts[k]
        for i in range(t * size, min(n, (t + 1) * size)):
            if wanted[(_float_key(bits[i]) >> np.uint64(shift)) - base]:
                out[c] = values[i]
                c += 1


def _key(value: float) -> int:
    """The key of `_float_key` of one float64 value, in Python."""
    bits = int(np.float64(value).view(np.uint64))
    return bits ^ int(_ALL_BITS) if bits >> 63 else bits ^ int(_SIGN_BIT)


def _radix_select(
    values: npt.NDArray[np.float64], ranks: npt.NDArray[np.intp], bounds: tuple[Any, Any]
) -> Optional[npt.NDArray[np.float64]]:
    """The values at the sorted 0-based `ranks` (np.partition's values), or None when the
    array holds a NaN (then a bound, the smallest or the largest value, is NaN).

    About 65,536 buckets split the key range of the data evenly (keys order like the
    floats), so close values spread over many buckets. The bounds give the key range; a
    bound of 0 takes the key of either zero, as -0.0 and +0.0 have two keys. The table holds
    every bucket from the smallest key to the largest: up to 65,537 buckets, because these
    two keys can be 65,536 buckets apart. The values of the buckets of the ranks are copied
    out and partitioned at the ranks inside them: a partition, not a sort, so data that
    crowd into few buckets cost about one partition of all values.
    """
    smallest, largest = bounds
    if smallest != smallest or largest != largest:
        return None
    values = np.ascontiguousarray(values)
    bits = values.view(np.uint64)
    threads = get_num_threads()
    low = _key(-0.0 if smallest == 0 else smallest)
    high = _key(0.0 if largest == 0 else largest)
    shift = max(0, (high - low).bit_length() - 16)
    base = np.uint64(low >> shift)
    counts = np.empty((threads, (high >> shift) - (low >> shift) + 1), dtype=np.int64)
    _bucket_counts_numba(bits, shift, base, counts)
    per_bucket = counts.sum(axis=0)
    below = np.concatenate(([0], np.cumsum(per_bucket)))  # values in lower buckets
    buckets = np.searchsorted(below[1:], ranks, side="right")
    used = np.unique(buckets)  # the buckets of the ranks
    wanted = np.zeros(per_bucket.size, dtype=np.bool_)
    wanted[used] = True
    kept = per_bucket[used]
    candidates = np.empty(int(kept.sum()), dtype=np.float64)
    _bucket_values_numba(bits, values, shift, base, wanted, counts[:, used].sum(axis=1), candidates)
    # A rank sits in its bucket after the candidates of the lower buckets of the ranks
    lower = (np.cumsum(kept) - kept)[np.searchsorted(used, buckets)]
    at = lower + ranks - below[buckets]
    return cast(npt.NDArray[np.float64], np.partition(candidates, np.unique(at))[at])


# From this many float64 values on, the order statistics take a linear select first: buckets
# of equal width from the smallest to the largest value, a count, and a partition of the keys
# of the buckets of the ranks only (measured on pools of new CT-like arrays: x0.65 of the
# partition at 1,000 values, x0.09 to x0.25 from 5,000 to 50,000 values, x0.17 and x0.41 of
# the radix select at 100,000 and 268,000 values; at 500 values it does not win). Below it,
# _small_order_statistics_numba partitions in numba (measured on pools of new arrays: x0.14
# of the numpy partition at 5 values, x0.46 at 200, x0.70 at 500, x0.86 to x0.90 at 999; a
# numba sort loses from about 550 values).
_LINEAR_SELECT_MIN = 1_000
# From this many values on, with more than one thread, the count and the copy of the linear
# select run in threads (measured: the serial passes lose to the radix select at 2 million).
_PARALLEL_SELECT_MIN = 1 << 20
_NEGATIVE_ZERO_KEY = np.uint64(0x7FFFFFFFFFFFFFFF)  # the key of -0.0 (see _float_key)
_POSITIVE_ZERO_KEY = np.uint64(0x8000000000000000)  # the key of +0.0


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _key_bits(key: np.uint64) -> np.uint64:
    """The float64 bits of a key of `_float_key`."""
    return key ^ _SIGN_BIT if key >> np.uint64(63) else key ^ _ALL_BITS


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _linear_bucket(value: float, low: float, scale: float, last: int) -> int:
    """The bucket of `value` among buckets of equal width from `low`: (value - low) * scale,
    at most `last`. It never falls when the value grows, so the buckets keep the order."""
    return min(int((value - low) * scale), last)


@jit(nopython=True, cache=True)  # type: ignore
def _crowded_numba(
    values: npt.NDArray[np.float64],
    low: float,
    scale: float,
    ranks: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
    n_blocks: int,
    block: int,
) -> bool:
    """Whether the buckets of the sorted `ranks` hold more than a quarter of a sample of the
    values (`counts`: zeros, one for each bucket): n_blocks blocks of `block` values, spread
    evenly from the first value to the last. For such crowded values the linear select is
    slower than the other paths."""
    n = values.size
    last = counts.size - 1
    m = 0
    for k in range(n_blocks):
        first = (n - block) * k // (n_blocks - 1)
        for i in range(first, first + block):
            counts[_linear_bucket(values[i], low, scale, last)] += 1
            m += 1
    kept = 0
    run = 0
    j = 0
    for b in range(counts.size):
        c = counts[b]
        if j < ranks.size and ranks[j] * m < (run + c) * n:
            kept += c
            while j < ranks.size and ranks[j] * m < (run + c) * n:
                j += 1
        if j == ranks.size:
            break
        run += c
    return 4 * kept > m


@jit(nopython=True, cache=True)  # type: ignore
def _linear_select_numba(
    values: npt.NDArray[np.float64],
    low: float,
    scale: float,
    ranks: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
    out: npt.NDArray[np.uint64],
) -> int:
    """out[j] = the bits of the value at the sorted 0-based rank ranks[j] (in ascending
    order), in the order of `_float_key` (-0.0 before +0.0). One pass counts the values of
    each bucket (`counts`: zeros); the running count gives the buckets of the ranks; a
    second pass copies the keys of their values; a partition of these keys gives the ranks.
    Returns 1 (no result) when these buckets hold more than a quarter of the values, 2 when
    a rank is a zero and the values hold -0.0 and +0.0 (np.partition can give either zero
    there), else 0."""
    n = values.size
    last = counts.size - 1
    for i in range(n):
        counts[_linear_bucket(values[i], low, scale, last)] += 1
    m = ranks.size
    first = np.empty(m, dtype=np.int64)  # the bucket of each rank
    below = np.empty(m, dtype=np.int64)  # the values of the lower buckets
    run = 0
    j = 0
    for b in range(counts.size):
        while j < m and ranks[j] < run + counts[b]:
            first[j] = b
            below[j] = run
            j += 1
        if j == m:
            break
        run += counts[b]
    slot = np.full(counts.size, -1, dtype=np.int64)  # the place of a bucket of the ranks
    start = np.zeros(m + 1, dtype=np.int64)  # where the keys of each such bucket start
    k = 0
    for j in range(m):
        if slot[first[j]] < 0:
            slot[first[j]] = k
            start[k + 1] = start[k] + counts[first[j]]
            k += 1
    if 4 * start[k] > n + 256:
        return 1
    fill = start[:k].copy()
    keys = np.empty(start[k], dtype=np.uint64)
    bits = values.view(np.uint64)
    for i in range(n):
        s = slot[_linear_bucket(values[i], low, scale, last)]
        if s >= 0:
            keys[fill[s]] = _float_key(bits[i])
            fill[s] += 1
    at = np.empty(m, dtype=np.int64)  # the place of each rank among the keys (ascending)
    kth = np.empty(m, dtype=np.int64)
    n_kth = 0
    for j in range(m):
        at[j] = start[slot[first[j]]] + ranks[j] - below[j]
        if n_kth == 0 or kth[n_kth - 1] != at[j]:
            kth[n_kth] = at[j]
            n_kth += 1
    picked = np.partition(keys, kth[:n_kth])
    zero = False
    for j in range(m):
        key = picked[at[j]]
        zero |= (key == _NEGATIVE_ZERO_KEY) | (key == _POSITIVE_ZERO_KEY)
        out[j] = _key_bits(key)
    if zero:
        negative = False
        positive = False
        for q in range(keys.size):  # the zeros share one bucket, a bucket of the ranks
            negative |= keys[q] == _NEGATIVE_ZERO_KEY
            positive |= keys[q] == _POSITIVE_ZERO_KEY
        if negative and positive:
            return 2
    return 0


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _linear_counts_numba(
    values: npt.NDArray[np.float64], low: float, scale: float, counts: npt.NDArray[np.int64]
) -> None:
    """counts[t, b]: the values of chunk t in bucket b of the linear select. Each chunk
    sets its row to 0 first."""
    n = values.size
    n_chunks, size_b = counts.shape
    size = (n + n_chunks - 1) // n_chunks
    for t in prange(n_chunks):
        for b in range(size_b):
            counts[t, b] = 0
        for i in range(t * size, min(n, (t + 1) * size)):
            counts[t, _linear_bucket(values[i], low, scale, size_b - 1)] += 1


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _linear_keys_numba(
    values: npt.NDArray[np.float64],
    low: float,
    scale: float,
    slot: npt.NDArray[np.int64],
    offsets: npt.NDArray[np.int64],
    keys: npt.NDArray[np.uint64],
) -> None:
    """Copy the keys of the values of the buckets b with slot[b] >= 0 into `keys`: chunk t
    (the chunks of _linear_counts_numba) from offsets[t, slot[b]] on."""
    n = values.size
    n_chunks = offsets.shape[0]
    size = (n + n_chunks - 1) // n_chunks
    bits = values.view(np.uint64)
    last = slot.size - 1
    for t in prange(n_chunks):
        fill = offsets[t].copy()
        for i in range(t * size, min(n, (t + 1) * size)):
            s = slot[_linear_bucket(values[i], low, scale, last)]
            if s >= 0:
                keys[fill[s]] = _float_key(bits[i])
                fill[s] += 1


def _linear_select_threads(
    values: npt.NDArray[np.float64],
    low: float,
    scale: float,
    ranks: npt.NDArray[np.int64],
    n_buckets: int,
) -> Optional[npt.NDArray[np.float64]]:
    """`_linear_select_numba` with the count and the copy in threads (one row of counts for
    each thread); None for the cases where it returns 1 or 2."""
    counts = np.empty((get_num_threads(), n_buckets), dtype=np.int64)
    _linear_counts_numba(values, low, scale, counts)
    per_bucket = counts.sum(axis=0)
    below = np.concatenate(([0], np.cumsum(per_bucket)))
    buckets = np.searchsorted(below[1:], ranks, side="right")
    used = np.unique(buckets)  # the buckets of the ranks
    kept = per_bucket[used]
    if 4 * int(kept.sum()) > values.size + 256:
        return None
    slot = np.full(per_bucket.size, -1, dtype=np.int64)
    slot[used] = np.arange(used.size)
    start = np.concatenate(([0], np.cumsum(kept)))
    chunks = counts[:, used]
    keys = np.empty(int(start[-1]), dtype=np.uint64)
    _linear_keys_numba(values, low, scale, slot, start[:-1] + np.cumsum(chunks, axis=0) - chunks, keys)  # fmt: skip
    at = start[np.searchsorted(used, buckets)] + ranks - below[buckets]
    picked = np.partition(keys, np.unique(at))[at]
    zeros = (picked == _NEGATIVE_ZERO_KEY) | (picked == _POSITIVE_ZERO_KEY)
    if zeros.any() and (keys == _NEGATIVE_ZERO_KEY).any() and (keys == _POSITIVE_ZERO_KEY).any():
        return None
    return np.where(picked >> np.uint64(63) != 0, picked ^ _SIGN_BIT, picked ^ _ALL_BITS).view(np.float64)  # fmt: skip


def _linear_select(
    values: npt.NDArray[np.float64], ranks: npt.NDArray[np.int64], bounds: tuple[Any, Any]
) -> Optional[npt.NDArray[np.float64]]:
    """The values at the sorted 0-based `ranks` (in ascending order) by a linear select, or
    None where the other paths give the bits: a NaN or an infinite bound, equal values, a
    span too small for the bucket scale, crowded values (a count of a sample says so
    first), and a zero rank with both zeros in the values (see _linear_select_numba)."""
    low = float(bounds[0])
    span = float(bounds[1]) - low
    if not 0.0 < span < math.inf:
        return None
    values = np.ascontiguousarray(values)
    n = values.size
    n_buckets = 4096
    scale = n_buckets / span
    if not math.isfinite(scale):
        return None
    # The sample: every 16th value of a small array, 32 blocks of 32 values from 16,384
    # values on (contiguous reads: on a large array each value of a spread sample is one
    # read from memory)
    n_blocks, block = (32, 32) if n >= 1 << 14 else (n // 16, 1)
    if _crowded_numba(
        values, low, scale, ranks, np.zeros(n_buckets, dtype=np.int64), n_blocks, block
    ):
        return None
    if n >= _PARALLEL_SELECT_MIN and get_num_threads() > 1:
        return _linear_select_threads(values, low, scale, ranks, n_buckets)
    out = np.empty(ranks.size, dtype=np.uint64)
    if _linear_select_numba(values, low, scale, ranks, np.zeros(n_buckets, dtype=np.int64), out):
        return None
    return out.view(np.float64)


@jit(nopython=True, cache=True)  # type: ignore
def _small_order_statistics_numba(
    values: npt.NDArray[np.float64],
) -> tuple[bool, float, float, float, float, float]:
    """P10, P25, P75, P90 and the median of float64 `values` by one partition in numba (no
    numpy call), after a flag: False (no result, before the partition) when a rank is a
    zero and the values hold -0.0 and +0.0 (np.partition can give either zero there). A
    NaN makes each of them NaN, as np.partition puts NaNs last. The median is numpy's mean
    of the middle value or values (a zero mean is +0.0, as np.mean starts its sum at +0.0)."""
    n = values.size
    negatives = 0
    negative_zeros = 0
    positive_zeros = 0
    for i in range(n):
        v = values[i]
        if v != v:
            return True, np.nan, np.nan, np.nan, np.nan, np.nan
        if v < 0.0:
            negatives += 1
        elif v == 0.0:
            if math.copysign(1.0, v) < 0.0:
                negative_zeros += 1
            else:
                positive_zeros += 1
    half = n // 2
    low = half - 1 + n % 2
    percentiles = _percentile_ranks(n)
    # The ranks in ascending order: P10, P25, the middle ranks, P75, P90
    ranks = np.array([percentiles[0], percentiles[1], low, half, percentiles[2], percentiles[3]])
    if negative_zeros > 0 and positive_zeros > 0:
        for j in range(6):  # the zeros sit at the ranks from `negatives` on, in sorted order
            if negatives <= ranks[j] < negatives + negative_zeros + positive_zeros:
                return False, 0.0, 0.0, 0.0, 0.0, 0.0
    part = np.partition(values, ranks)
    median = (part[low] + part[half]) / 2.0 if n % 2 == 0 else part[low]
    if median == 0.0:
        median = 0.0  # np.mean starts its sum at +0.0, so a zero mean of -0.0 values is +0.0
    return True, part[ranks[0]], part[ranks[1]], part[ranks[4]], part[ranks[5]], median


def _order_statistics(
    values: npt.NDArray[Any], bounds: Optional[tuple[Any, Any]] = None
) -> tuple[Any, ...]:
    """P10, P25, P75, P90 and the median of a non-empty array.

    `np.percentile(..., method="inverted_cdf")` and `np.median` partition the array one
    time each. One partition at all their ranks gives the same values: the median is
    numpy's mean of the middle value or values. A NaN makes all of them NaN, as in numpy
    (a partition puts NaNs last). `bounds`: the smallest and the largest value (np.min
    and np.max), when the caller has them.

    Float64 values take the linear select first (from _LINEAR_SELECT_MIN values), then
    the radix select (from _RADIX_SELECT_MIN values), then one partition: each gives the
    bits of the next one, or gives way to it. Below _LINEAR_SELECT_MIN float64 values, one
    partition in numba gives them (no numpy call), or gives way when the values hold both
    zeros at a rank.
    """
    n = values.size
    if values.dtype == np.float64 and 0 < n < _LINEAR_SELECT_MIN:
        found, p10, p25, p75, p90, median = _small_order_statistics_numba(
            np.ascontiguousarray(values)
        )
        if found:
            return p10, p25, p75, p90, median
    ranks = _percentile_ranks(n)
    half = n // 2
    low = half - 1 + n % 2  # lower middle rank (the middle rank when n is odd)
    if values.dtype == np.float64 and n >= _LINEAR_SELECT_MIN:
        if bounds is None:
            bounds = (np.min(values), np.max(values))
        # The ranks in ascending order: P10, P25, the middle ranks, P75, P90
        order = np.array([ranks[0], ranks[1], low, half, ranks[2], ranks[3]], dtype=np.int64)
        found = _linear_select(values, order, bounds)
        if found is not None:
            middle = found[2:4] if n % 2 == 0 else found[2:3]
            return found[0], found[1], found[4], found[5], np.mean(middle)
    if values.dtype == np.float64 and n >= _RADIX_SELECT_MIN:
        if bounds is None:
            bounds = (np.min(values), np.max(values))
        picked = _radix_select(values, np.concatenate((ranks, [low, half])), bounds)
        if picked is not None:
            return (
                picked[0],
                picked[1],
                picked[2],
                picked[3],
                np.mean(picked[4:6] if n % 2 == 0 else picked[4:5]),
            )
    part = np.partition(values, np.unique(np.concatenate((ranks, [low, half, n - 1]))))
    if part.dtype.kind == "f" and np.isnan(part[-1]):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    p10, p25, p75, p90 = part[ranks]
    return p10, p25, p75, p90, np.mean(part[low : half + 1])


def _range_error(n_bins: int, low: int, high: int) -> str:
    """The error of discretised values outside [1, n_bins]."""
    return f"discretised values must lie in [1, n_bins={n_bins}]; got range [{low}, {high}]"


# The histogram features, in the order of calculate_intensity_histogram_features
_HISTOGRAM_FEATURES = (
    "mean_discretised_intensity_X6K6",
    "discretised_intensity_variance_CH89",
    "discretised_intensity_skewness_88K1",
    "discretised_intensity_kurtosis_C3I7",
    "median_discretised_intensity_WIFQ",
    "minimum_discretised_intensity_1PR8",
    "10th_discretised_intensity_percentile_1PR",
    "90th_discretised_intensity_percentile_GPMT",
    "maximum_discretised_intensity_3NCY",
    "intensity_histogram_mode_AMMC",
    "discretised_intensity_interquartile_range_WR0O",
    "discretised_intensity_range_5Z3W",
    "intensity_histogram_mean_absolute_deviation_D2ZX",
    "intensity_histogram_robust_mean_absolute_deviation_WRZB",
    "intensity_histogram_median_absolute_deviation_4RNL",
    "intensity_histogram_coefficient_of_variation_CWYJ",
    "intensity_histogram_quartile_coefficient_of_dispersion_SLWD",
    "discretised_intensity_entropy_TLU2",
    "discretised_intensity_uniformity_BJ5W",
    "maximum_histogram_gradient_12CE",
    "maximum_histogram_gradient_intensity_8E6O",
    "minimum_histogram_gradient_VQB3",
    "minimum_histogram_gradient_intensity_RHQZ",
)


@jit(nopython=True, cache=True)  # type: ignore
def _level_counts_numba(
    values: npt.NDArray[Any], n_bins: int
) -> tuple[npt.NDArray[np.int64], Any, Any, bool]:
    """The counts of the levels 0 to n_bins of the 1-D `values` (a value below 0 counts as
    level 0, one above n_bins as level n_bins), the smallest and the largest value, and
    whether every value is a whole number (an integer always is, a NaN never): the count
    stops at the first value that is not, and then the counts and the range are partial.
    A NaN counts as level 0 until then (max keeps its first argument), so int() never
    reads one. Four tables take the values in turn, so a run of equal values does not wait
    on one table."""
    n = values.size
    tables = np.zeros((4, n_bins + 1), dtype=np.int64)
    low = values[0]
    high = values[0]
    whole = True
    i = 0
    while whole and i + 4 <= n:
        v0 = values[i]
        v1 = values[i + 1]
        v2 = values[i + 2]
        v3 = values[i + 3]
        k0 = int(min(max(0, v0), n_bins))
        k1 = int(min(max(0, v1), n_bins))
        k2 = int(min(max(0, v2), n_bins))
        k3 = int(min(max(0, v3), n_bins))
        whole &= (math.floor(v0) == v0) & (math.floor(v1) == v1)
        whole &= (math.floor(v2) == v2) & (math.floor(v3) == v3)
        low = min(low, min(min(v0, v1), min(v2, v3)))
        high = max(high, max(max(v0, v1), max(v2, v3)))
        tables[0, k0] += 1
        tables[1, k1] += 1
        tables[2, k2] += 1
        tables[3, k3] += 1
        i += 4
    while whole and i < n:
        v0 = values[i]
        k0 = int(min(max(0, v0), n_bins))
        whole &= math.floor(v0) == v0
        low = min(low, v0)
        high = max(high, v0)
        tables[0, k0] += 1
        i += 1
    counts = np.empty(n_bins + 1, dtype=np.int64)
    for k in range(n_bins + 1):
        counts[k] = tables[0, k] + tables[1, k] + tables[2, k] + tables[3, k]
    return counts, low, high, whole


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _level_counts_threads_numba(
    values: npt.NDArray[Any],
    n_bins: int,
    counts: npt.NDArray[np.int64],
    lows: npt.NDArray[Any],
    highs: npt.NDArray[Any],
    wholes: npt.NDArray[np.bool_],
) -> None:
    """_level_counts_numba of each of counts.shape[0] parts of `values`, in threads: the
    counts of part c into counts[c], its smallest and largest value into lows[c] and
    highs[c], its whole-number flag into wholes[c] (the tables of a part are its own)."""
    n = values.size
    parts = counts.shape[0]
    for c in prange(parts):
        start = c * n // parts
        stop = (c + 1) * n // parts
        tables = np.zeros((4, n_bins + 1), dtype=np.int64)
        low = values[start]
        high = values[start]
        whole = True
        i = start
        while whole and i + 4 <= stop:
            v0 = values[i]
            v1 = values[i + 1]
            v2 = values[i + 2]
            v3 = values[i + 3]
            k0 = int(min(max(0, v0), n_bins))
            k1 = int(min(max(0, v1), n_bins))
            k2 = int(min(max(0, v2), n_bins))
            k3 = int(min(max(0, v3), n_bins))
            whole &= (math.floor(v0) == v0) & (math.floor(v1) == v1)
            whole &= (math.floor(v2) == v2) & (math.floor(v3) == v3)
            low = min(low, min(min(v0, v1), min(v2, v3)))
            high = max(high, max(max(v0, v1), max(v2, v3)))
            tables[0, k0] += 1
            tables[1, k1] += 1
            tables[2, k2] += 1
            tables[3, k3] += 1
            i += 4
        while whole and i < stop:
            v0 = values[i]
            k0 = int(min(max(0, v0), n_bins))
            whole &= math.floor(v0) == v0
            low = min(low, v0)
            high = max(high, v0)
            tables[0, k0] += 1
            i += 1
        for k in range(n_bins + 1):
            counts[c, k] = tables[0, k] + tables[1, k] + tables[2, k] + tables[3, k]
        lows[c] = low
        highs[c] = high
        wholes[c] = whole


# From this many values on, the level count runs in threads (measured: below, one thread is
# as fast, and the start of the threads costs about 70 us)
_PARALLEL_LEVEL_MIN = 1 << 19


def _level_counts(
    values: npt.NDArray[Any], n_bins: int
) -> tuple[npt.NDArray[np.int64], Any, Any, bool]:
    """The counts of the levels 0 to n_bins of the integer or float64 `values` (see
    _level_counts_numba), the smallest and the largest value, and whether every value is a
    whole number. int32 (the pipeline), int64 and float64 values go as they are; other
    integers as an int64 copy."""
    data = values if values.dtype in (np.int32, np.int64, np.float64) else values.astype(np.int64)
    data = np.ascontiguousarray(data)
    if data.size >= _PARALLEL_LEVEL_MIN and get_num_threads() > 1:
        parts = get_num_threads()
        counts = np.empty((parts, n_bins + 1), dtype=np.int64)
        lows = np.empty(parts, dtype=data.dtype)
        highs = np.empty(parts, dtype=data.dtype)
        wholes = np.empty(parts, dtype=np.bool_)
        _level_counts_threads_numba(data, n_bins, counts, lows, highs, wholes)
        return counts.sum(axis=0), lows.min(), highs.max(), bool(wholes.all())
    return cast(tuple[npt.NDArray[np.int64], Any, Any, bool], _level_counts_numba(data, n_bins))


@jit(nopython=True, cache=True)  # type: ignore
def _pairwise_sum_numba(terms: npt.NDArray[np.float64], start: int, stop: int) -> float:
    """The sum of terms[start:stop] in the order of np.sum (numpy's pairwise summation: a
    plain loop below 8 terms, 8 partial sums up to 128 terms, above that the sum of the two
    halves, the first a multiple of 8), so the bits of np.sum. The halves wait on a stack of
    frames (a cached kernel cannot call a recursive one): a frame holds the range of a
    half and, once its first half is summed, that sum."""
    starts = np.empty(64, dtype=np.int64)
    stops = np.empty(64, dtype=np.int64)
    firsts = np.empty(64, dtype=np.float64)
    summed = np.zeros(64, dtype=np.bool_)  # the first half of the frame is summed
    depth = 0
    starts[0] = start
    stops[0] = stop
    while True:
        a = starts[depth]
        b = stops[depth]
        n = b - a
        if n > 128:
            half = n // 2
            half -= half % 8
            depth += 1
            if summed[depth - 1]:
                starts[depth] = a + half
                stops[depth] = b
            else:
                starts[depth] = a
                stops[depth] = a + half
            summed[depth] = False
            continue
        if n < 8:
            value = 0.0
            for i in range(a, b):
                value += terms[i]
        else:
            r0 = terms[a]
            r1 = terms[a + 1]
            r2 = terms[a + 2]
            r3 = terms[a + 3]
            r4 = terms[a + 4]
            r5 = terms[a + 5]
            r6 = terms[a + 6]
            r7 = terms[a + 7]
            i = a + 8
            end = b - n % 8
            while i < end:
                r0 += terms[i]
                r1 += terms[i + 1]
                r2 += terms[i + 2]
                r3 += terms[i + 3]
                r4 += terms[i + 4]
                r5 += terms[i + 5]
                r6 += terms[i + 6]
                r7 += terms[i + 7]
                i += 8
            value = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i < b:
                value += terms[i]
                i += 1
        # The value goes up: a frame whose first half this is keeps it and takes its second
        # half next; a frame with both halves summed passes their sum up
        while True:
            depth -= 1
            if depth < 0:
                return value
            if summed[depth]:
                value = firsts[depth] + value
            else:
                firsts[depth] = value
                summed[depth] = True
                break


@jit(nopython=True, cache=True)  # type: ignore
def _rank_numba(n: int, q: float) -> int:
    """The 0-based rank of the q quantile among n sorted values: the index rule of
    np.percentile(..., method="inverted_cdf"), as _percentile_ranks computes it."""
    index = n * q - 1.0
    below = math.floor(index)
    return int(below) if index - below == 0.0 else int(below) + 1


@jit(nopython=True, cache=True)  # type: ignore
def _counted_statistics_numba(
    counts: npt.NDArray[np.int64], origin: int, out: npt.NDArray[np.float64]
) -> None:
    """The histogram features (_HISTOGRAM_FEATURES, in that order) of integer values from
    their bin counts (bin i counts the value origin + i), into `out`. The sums of terms
    over the bins run in the order of np.sum (_pairwise_sum_numba), the sums of integers
    in int64, so the values are the bits of the numpy steps that this kernel replaces."""
    nb = counts.size
    n = 0
    total = 0
    for i in range(nb):
        n += counts[i]
        total += (origin + i) * counts[i]
    mean = total / n
    terms = np.empty(nb, dtype=np.float64)
    for i in range(nb):
        d = (origin + i) - mean
        terms[i] = counts[i] * (d * d)
    var = _pairwise_sum_numba(terms, 0, nb) / n
    if var == 0.0:  # IBSI: skewness and kurtosis 0 when the variance is 0
        skew = 0.0
        kurt = 0.0
    else:
        inv_n = 1.0 / n
        for i in range(nb):
            d = (origin + i) - mean
            d2 = d * d
            terms[i] = counts[i] * (d2 * d)
        m3 = _pairwise_sum_numba(terms, 0, nb) * inv_n
        for i in range(nb):
            d = (origin + i) - mean
            d2 = d * d
            terms[i] = counts[i] * (d2 * d2)
        m4 = _pairwise_sum_numba(terms, 0, nb) * inv_n
        skew = m3 / var**1.5
        kurt = (m4 / (var * var)) - 3.0
    # The order statistics: the value at a rank is the value of the first bin whose running
    # count is above the rank. The ranks in ascending order: P10, P25, the middle rank or
    # ranks, P75, P90; the median is the float64 mean of the middle value or values.
    half = n // 2
    ranks = (
        _rank_numba(n, 0.1),
        _rank_numba(n, 0.25),
        half - 1 + n % 2,
        half,
        _rank_numba(n, 0.75),
        _rank_numba(n, 0.9),
    )
    picked = np.empty(6, dtype=np.int64)
    j = 0
    running = 0
    for i in range(nb):
        running += counts[i]
        while j < 6 and running > ranks[j]:
            picked[j] = origin + i
            j += 1
    p10 = picked[0]
    p25 = picked[1]
    median = (float(picked[2]) + float(picked[3])) / 2.0
    p75 = picked[4]
    p90 = picked[5]
    # The smallest and the largest value, and the mode: of the most frequent values the one
    # closest to the mean (the lower one of two at the same distance)
    low = origin
    high = origin
    largest = 0
    for i in range(nb):
        if counts[i] > 0:
            if largest == 0:
                low = origin + i
            high = origin + i
            largest = max(largest, counts[i])
    mode = origin
    best = math.inf
    for i in range(nb):
        if counts[i] == largest:
            distance = abs((origin + i) - mean)
            if distance < best:
                best = distance
                mode = origin + i
    # The mean, the median and the robust (values from P10 to P90) absolute deviations
    for i in range(nb):
        terms[i] = counts[i] * abs((origin + i) - mean)
    mad = _pairwise_sum_numba(terms, 0, nb) / n
    for i in range(nb):
        terms[i] = counts[i] * abs((origin + i) - median)
    median_ad = _pairwise_sum_numba(terms, 0, nb) / n
    sub_n = 0
    sub_total = 0
    for i in range(nb):
        level = origin + i
        if level >= p10 and level <= p90:
            sub_n += counts[i]
            sub_total += level * counts[i]
    sub_mean = sub_total / sub_n
    k = 0
    for i in range(nb):
        level = origin + i
        if level >= p10 and level <= p90:
            terms[k] = counts[i] * abs(level - sub_mean)
            k += 1
    robust = _pairwise_sum_numba(terms, 0, k) / sub_n
    # The entropy and the uniformity over the occupied bins
    k = 0
    for i in range(nb):
        if counts[i] > 0:
            p = counts[i] / float(n)
            terms[k] = p * np.log2(p)
            k += 1
    entropy = -_pairwise_sum_numba(terms, 0, k)
    k = 0
    for i in range(nb):
        if counts[i] > 0:
            p = counts[i] / float(n)
            terms[k] = p * p
            k += 1
    uniformity = _pairwise_sum_numba(terms, 0, k)
    # The histogram gradient (np.gradient: one-sided at the ends, central inside), its
    # largest and smallest value and their first bins
    if nb < 2:
        max_gradient = np.nan
        max_gradient_at = np.nan
        min_gradient = np.nan
        min_gradient_at = np.nan
    else:
        max_gradient = -math.inf
        min_gradient = math.inf
        max_gradient_at = 0.0
        min_gradient_at = 0.0
        for i in range(nb):
            if i == 0:
                gradient = float(counts[1]) - float(counts[0])
            elif i == nb - 1:
                gradient = float(counts[nb - 1]) - float(counts[nb - 2])
            else:
                gradient = (float(counts[i + 1]) - float(counts[i - 1])) / 2.0
            if gradient > max_gradient:
                max_gradient = gradient
                max_gradient_at = float(origin + i)
            if gradient < min_gradient:
                min_gradient = gradient
                min_gradient_at = float(origin + i)
    out[0] = mean
    out[1] = var
    out[2] = skew
    out[3] = kurt
    out[4] = median
    out[5] = low
    out[6] = p10
    out[7] = p90
    out[8] = high
    out[9] = mode
    out[10] = p75 - p25
    out[11] = float(high) - float(low)
    out[12] = mad
    out[13] = robust
    out[14] = median_ad
    out[15] = math.sqrt(var) / mean if mean != 0 else np.nan
    out[16] = (p75 - p25) / (p75 + p25) if (p75 + p25) != 0 else np.nan
    out[17] = entropy
    out[18] = uniformity
    out[19] = max_gradient
    out[20] = max_gradient_at
    out[21] = min_gradient
    out[22] = min_gradient_at


def calculate_intensity_features(
    values: npt.NDArray[np.floating[Any]],
) -> dict[str, float]:
    """
    Calculate intensity-based features (First Order Statistics) as defined in IBSI 4.1.

    Computes 18 statistical features from the intensity values within the ROI:
    mean, variance, skewness, kurtosis, median, min/max, percentiles (10th, 90th),
    interquartile range, range, MAD variants, coefficient of variation, energy, RMS.

    Args:
        values: 1D array of intensity values from the ROI (after mask application).
            The sums run in float64, also for float32 values.

    Returns:
        Dictionary mapping feature names (with IBSI codes) to computed values.
        Empty dict if input is empty.

    Example:
        Calculate features from an ROI:

        ```python
        from pictologics.features.intensity import calculate_intensity_features
        from pictologics.preprocessing import apply_mask

        # Get values within ROI
        roi_values = apply_mask(image, mask)

        # Calculate features
        features = calculate_intensity_features(roi_values)
        print(features["mean_intensity_Q4LE"])
        ```
    """
    if len(values) == 0:
        return {}
    # float32 values (for example filter responses) would sum in float32: with its rounding
    # errors, and in an order that depends on the numpy version.
    values = np.asarray(values, dtype=np.float64)

    features: dict[str, float] = {}

    # 4.1.1 Mean intensity (Q4LE)
    mean_val = np.mean(values)
    features["mean_intensity_Q4LE"] = float(mean_val)

    # 4.1.2 Intensity variance (ECT3): the second central moment. The pass that gives it
    # also gives the third and the fourth, so np.var does not read the values again.
    m2, m3, m4 = _central_moments_2_3_4(values, float(mean_val))
    var_val = m2
    features["intensity_variance_ECT3"] = float(var_val)

    # 4.1.3 Intensity skewness (KE2A) and 4.1.4 kurtosis (IPH6). IBSI defines both as 0
    # when the variance is 0. Equal values can give a variance of about 1e-34 (a mean of
    # 0.1 values is not exact), so equal values count as 0 variance here.
    min_val = np.min(values)
    max_val = np.max(values)
    if var_val == 0.0 or min_val == max_val:
        features["intensity_skewness_KE2A"] = 0.0
        features["intensity_kurtosis_IPH6"] = 0.0
    else:
        denom = m2**1.5
        if denom != 0.0:
            features["intensity_skewness_KE2A"] = float(m3 / denom)
            features["intensity_kurtosis_IPH6"] = float((m4 / (m2 * m2)) - 3.0)
        else:  # pragma: no cover  # unreachable: denom==0 iff var_val==0, caught above
            features["intensity_skewness_KE2A"] = np.nan
            features["intensity_kurtosis_IPH6"] = np.nan

    # IBSI percentiles use the nearest-rank convention (the smallest value with at
    # least p% of the data at or below it), i.e. numpy's 'inverted_cdf'. This
    # reproduces the IBSI benchmark (e.g. P90 = 4), whereas linear interpolation
    # would give an interpolated 4.2. The median (Y12H) is the conventional
    # sample median. One partition gives both.
    p10, p25, p75, p90, median_val = _order_statistics(values, (min_val, max_val))

    # 4.1.5 Median intensity (Y12H)
    features["median_intensity_Y12H"] = float(median_val)

    # 4.1.6 Minimum intensity (1GSF)
    features["minimum_intensity_1GSF"] = float(min_val)

    features["10th_intensity_percentile_QG58"] = float(p10)
    features["90th_intensity_percentile_8DWT"] = float(p90)

    # 4.1.9 Maximum intensity (84IY)
    features["maximum_intensity_84IY"] = float(max_val)

    # 4.1.10 Intensity interquartile range (SALO)
    features["intensity_interquartile_range_SALO"] = float(p75 - p25)

    # 4.1.11 Intensity range (2OJQ)
    features["intensity_range_2OJQ"] = float(max_val - min_val)

    # 4.1.12 Mean absolute deviation (4FUA)
    features["intensity_mean_absolute_deviation_4FUA"] = float(
        _mean_abs_dev(values, float(mean_val))
    )

    # 4.1.13 Robust mean absolute deviation (1128)
    features["intensity_robust_mean_absolute_deviation_1128"] = float(
        _robust_mean_abs_dev(values, float(p10), float(p90))
    )

    # 4.1.14 Median absolute deviation (N72L)
    features["intensity_median_absolute_deviation_N72L"] = float(
        _mean_abs_dev(values, float(median_val))
    )

    # 4.1.15 Coefficient of variation (7TET)
    if mean_val != 0:
        features["intensity_coefficient_of_variation_7TET"] = float(np.sqrt(var_val) / mean_val)
    else:
        features["intensity_coefficient_of_variation_7TET"] = np.nan

    # 4.1.16 Quartile coefficient of dispersion (9S40)
    if (p75 + p25) != 0:
        features["intensity_quartile_coefficient_of_dispersion_9S40"] = float(
            (p75 - p25) / (p75 + p25)
        )
    else:
        features["intensity_quartile_coefficient_of_dispersion_9S40"] = np.nan

    # 4.1.17 Energy (N8CA)
    energy = float(_sum_of_squares(values))  # a fixed order: the same bits in every buffer
    features["intensity_energy_N8CA"] = energy

    # 4.1.18 Root mean square (5ZWQ)
    features["root_mean_square_intensity_5ZWQ"] = float(np.sqrt(energy / len(values)))

    return features


def calculate_intensity_histogram_features(
    discretised_values: npt.NDArray[np.floating[Any]],
    n_bins: Optional[int] = None,
) -> dict[str, float]:
    """
    Calculate intensity histogram features as defined in IBSI 4.2.

    Computes features from the discretised intensity histogram including
    mean, variance, skewness, kurtosis, mode, entropy, uniformity, and gradient features.

    Args:
        discretised_values: 1D array of discretised intensity values (after binning).
        n_bins: Total number of bins N_g used at discretisation. When given, the
            histogram spans the full IBSI range [1, N_g], including bins that are
            empty because the data does not reach them (this only affects the
            gradient features). When None, the histogram spans the observed value
            range.

    Returns:
        Dictionary mapping feature names (with IBSI codes) to computed values.
        Empty dict if input is empty.

    Raises:
        ValueError: If n_bins is given and the values do not lie in [1, n_bins].

    Example:
        ```python
        import numpy as np
        from pictologics.features.intensity import calculate_intensity_histogram_features

        discretised_values = np.array([1, 1, 2, 2, 3, 4, 4, 4], dtype=np.float64)
        features = calculate_intensity_histogram_features(discretised_values, n_bins=4)
        print(features["mean_discretised_intensity_X6K6"])
        # 2.625
        print(features["intensity_histogram_mode_AMMC"])
        # 4.0
        ```
    """
    if len(discretised_values) == 0:
        return {}

    disc = np.asarray(discretised_values)
    if disc.ndim != 1:
        raise ValueError(f"discretised_values must be a 1-D array; got shape {disc.shape}")
    n = disc.size
    # Integer values (but uint64, which bincount does not read) take every feature from
    # their bin counts, and so do float64 values that are whole numbers in [1, n_bins]:
    # the levels themselves. Other values are read by the kernels.
    counted = disc.dtype.kind in "iu" and disc.dtype != np.uint64

    counts: Optional[npt.NDArray[np.int64]] = None
    if n_bins is not None and (counted or disc.dtype == np.float64):
        # IBSI: histogram over the full discretisation range [1, N_g]
        counts, low, high, whole = _level_counts(disc, n_bins)
        if not whole:
            counts = None
        elif low < 1 or high > n_bins:
            raise ValueError(_range_error(n_bins, int(low), int(high)))
        else:
            counts, hist_origin = counts[1:], 1
    elif counted:
        # Observed value range; shifting also supports negative values for bincount
        hist_origin = int(np.min(disc))
        counts = np.bincount(
            disc.astype(np.int64) - hist_origin, minlength=int(np.max(disc)) - hist_origin + 1
        )
    if counts is not None:
        out = np.empty(len(_HISTOGRAM_FEATURES))
        _counted_statistics_numba(counts, hist_origin, out)
        return dict(zip(_HISTOGRAM_FEATURES, out.tolist(), strict=True))

    # Discretised values are whole numbers (a float array holds them too); a value that is
    # not whole, NaN or infinite is a caller error (a pipeline passes int32)
    if not (np.all(np.isfinite(disc)) and np.array_equal(disc, np.rint(disc))):
        raise ValueError(
            "discretised_values must be finite whole numbers (bin numbers); a value "
            "that is not whole, NaN or infinite was given."
        )
    # One float64 copy of another type (float32, float16, uint64, bool): the mean, the
    # variance, the moments, the percentiles and the deviations then sum in float64
    if disc.dtype != np.float64:
        disc = disc.astype(np.float64)
    features: dict[str, float] = {}
    min_val_i = int(np.min(disc))
    max_val_i = int(np.max(disc))
    if n_bins is not None:
        if min_val_i < 1 or max_val_i > n_bins:
            raise ValueError(_range_error(n_bins, min_val_i, max_val_i))
        hist_origin = 1
        counts_full = np.bincount(disc.astype(np.int64) - 1, minlength=n_bins)
    else:
        # Observed value range; shifting also supports negative values
        # for bincount compatibility.
        hist_origin = min_val_i
        counts_full = np.bincount(
            disc.astype(np.int64) - min_val_i, minlength=(max_val_i - min_val_i + 1)
        )
    total = float(n)
    p = counts_full[counts_full > 0].astype(np.float64) / total

    # 4.2.1 Mean discretised intensity (X6K6) and 4.2.2 variance (CH89)
    mean_disc = float(np.mean(disc))
    var_disc = float(np.var(disc, ddof=0))
    features["mean_discretised_intensity_X6K6"] = float(mean_disc)
    features["discretised_intensity_variance_CH89"] = float(var_disc)

    # 4.2.3 Discretised intensity skewness (88K1) and 4.2.4 kurtosis (C3I7): 0 when the
    # variance is 0 (IBSI)
    if var_disc == 0.0:
        features["discretised_intensity_skewness_88K1"] = 0.0
        features["discretised_intensity_kurtosis_C3I7"] = 0.0
    else:
        m2, m3, m4 = _central_moments_2_3_4(disc, float(mean_disc))
        denom = m2**1.5
        if denom != 0.0:
            features["discretised_intensity_skewness_88K1"] = float(m3 / denom)
            features["discretised_intensity_kurtosis_C3I7"] = float((m4 / (m2 * m2)) - 3.0)
        else:  # pragma: no cover  # unreachable: denom==0 iff var_val==0, caught above
            features["discretised_intensity_skewness_88K1"] = np.nan
            features["discretised_intensity_kurtosis_C3I7"] = np.nan

    # IBSI nearest-rank percentiles (see calculate_intensity_features); the
    # discretised median (WIFQ) is the conventional sample median.
    p10, p25, p75, p90, median_val = _order_statistics(disc)

    features["median_discretised_intensity_WIFQ"] = float(median_val)
    features["minimum_discretised_intensity_1PR8"] = float(min_val_i)
    features["10th_discretised_intensity_percentile_1PR"] = float(p10)
    features["90th_discretised_intensity_percentile_GPMT"] = float(p90)
    features["maximum_discretised_intensity_3NCY"] = float(max_val_i)

    # IBSI: with multiple modes, select the one closest to the mean discretised
    # intensity; if two modes are equidistant from the mean, select the lower one
    # (argmin returns the first, i.e. lowest, candidate on ties).
    mode_candidates = np.flatnonzero(counts_full == np.max(counts_full)) + hist_origin
    best_mode = int(np.argmin(np.abs(mode_candidates - mean_disc)))
    features["intensity_histogram_mode_AMMC"] = float(mode_candidates[best_mode])

    features["discretised_intensity_interquartile_range_WR0O"] = float(p75 - p25)

    features["discretised_intensity_range_5Z3W"] = float(
        features["maximum_discretised_intensity_3NCY"]
        - features["minimum_discretised_intensity_1PR8"]
    )

    mad = _mean_abs_dev(disc, float(mean_disc))
    robust_mad = _robust_mean_abs_dev(disc, float(p10), float(p90))
    median_ad = _mean_abs_dev(disc, float(median_val))
    features["intensity_histogram_mean_absolute_deviation_D2ZX"] = float(mad)
    features["intensity_histogram_robust_mean_absolute_deviation_WRZB"] = float(robust_mad)
    features["intensity_histogram_median_absolute_deviation_4RNL"] = float(median_ad)

    if mean_disc != 0:
        features["intensity_histogram_coefficient_of_variation_CWYJ"] = float(
            np.sqrt(var_disc) / mean_disc
        )
    else:
        features["intensity_histogram_coefficient_of_variation_CWYJ"] = np.nan

    if (p75 + p25) != 0:
        features["intensity_histogram_quartile_coefficient_of_dispersion_SLWD"] = float(
            (p75 - p25) / (p75 + p25)
        )
    else:
        features["intensity_histogram_quartile_coefficient_of_dispersion_SLWD"] = np.nan

    # Vectorized entropy/uniformity
    features["discretised_intensity_entropy_TLU2"] = float(-np.sum(p * np.log2(p)))
    features["discretised_intensity_uniformity_BJ5W"] = float(np.sum(p * p))

    hist_counts = counts_full.astype(np.float64)
    if len(hist_counts) < 2:
        features["maximum_histogram_gradient_12CE"] = np.nan
        features["maximum_histogram_gradient_intensity_8E6O"] = np.nan
        features["minimum_histogram_gradient_VQB3"] = np.nan
        features["minimum_histogram_gradient_intensity_RHQZ"] = np.nan
    else:
        gradient = np.gradient(hist_counts)
        features["maximum_histogram_gradient_12CE"] = float(np.max(gradient))
        max_grad_idx = int(np.argmax(gradient))
        features["maximum_histogram_gradient_intensity_8E6O"] = float(hist_origin + max_grad_idx)
        features["minimum_histogram_gradient_VQB3"] = float(np.min(gradient))
        min_grad_idx = int(np.argmin(gradient))
        features["minimum_histogram_gradient_intensity_RHQZ"] = float(hist_origin + min_grad_idx)

    return features


class _Grid:
    """The candidate values base + (i + shift) * step, i in range(n), one at a time: the
    float64 operations of the np.arange grid without an array of the whole value range."""

    __slots__ = ("n", "base", "step", "shift")

    def __init__(self, n: int, base: Any, step: Any, shift: float) -> None:
        # Python floats: a float32 base or step would otherwise round the candidates
        self.n, self.base, self.step, self.shift = n, float(base), float(step), shift

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> float:
        f = float(i if i >= 0 else self.n + i)
        return self.base + (f + self.shift) * self.step


def calculate_ivh_features(
    discretised_values: npt.NDArray[np.floating[Any]],
    bin_width: Optional[float] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    target_range_min: Optional[float] = None,
    target_range_max: Optional[float] = None,
) -> dict[str, float]:
    """
    Calculate Intensity-Volume Histogram (IVH) features as defined in IBSI 4.3.

    Computes volume fractions at intensity thresholds, intensity values at volume
    fractions, and Area Under the IVH Curve (AUC).

    Args:
        discretised_values: 1D array of discretised intensity values.
        bin_width: Optional bin width used in discretisation (for physical units).
        min_val: Optional minimum value used in discretisation.
        max_val: Optional maximum value used in discretisation.
        target_range_min: Optional target range minimum for fraction calculations.
        target_range_max: Optional target range maximum for fraction calculations.

    Returns:
        Dictionary mapping feature names (with IBSI codes) to computed values.
        Empty dict if input is empty.

    Raises:
        ValueError: If both min_val and max_val are given and max_val < min_val.

    Example:
        ```python
        import numpy as np
        from pictologics.features.intensity import calculate_ivh_features

        discretised_values = np.arange(1, 11, dtype=np.float64)
        features = calculate_ivh_features(
            discretised_values, bin_width=1.0, min_val=0.0, max_val=10.0
        )
        print(round(features["area_under_the_ivh_curve_9CMM"], 2))
        # 4.95
        ```
    """
    if len(discretised_values) == 0:
        return {}

    if min_val is not None and max_val is not None and max_val < min_val:
        raise ValueError(f"max_val ({max_val}) must be >= min_val ({min_val})")

    features: dict[str, float] = {}
    N = len(discretised_values)

    vals = np.ascontiguousarray(discretised_values)  # one layout for the numba kernels
    count_below, kth, unique_vals, below_unique = _sorted_view(vals)

    # -------------------------------------------------------------------------
    # 1. Volume Fractions
    # -------------------------------------------------------------------------
    t_min = target_range_min if target_range_min is not None else min_val
    t_max = target_range_max if target_range_max is not None else max_val

    # Fallback to data min/max if still None
    if t_min is None or t_max is None:
        t_min_idx = float(kth(0))
        t_max_idx = float(kth(N - 1))
        val_range_idx = t_max_idx - t_min_idx

        def get_volume_fraction_at_intensity_fraction_indices(frac: float) -> float:
            threshold_idx = t_min_idx + frac * val_range_idx
            count = N - count_below(threshold_idx)
            return float(count / N)

        features["volume_at_intensity_fraction_0.10_BC2M_10"] = (
            get_volume_fraction_at_intensity_fraction_indices(0.10)
        )
        features["volume_at_intensity_fraction_0.90_BC2M_90"] = (
            get_volume_fraction_at_intensity_fraction_indices(0.90)
        )

    else:
        # We have physical units for the target range.
        t_range = t_max - t_min

        def get_volume_fraction_at_intensity_fraction_physical(frac: float) -> float:
            threshold_val = t_min + frac * t_range  # type: ignore
            if bin_width is not None and min_val is not None:
                # Convert to index based on FBS
                mv = min_val
                bw = bin_width
                threshold_idx = np.floor((threshold_val - mv) / bw) + 1  # type: ignore
            else:
                threshold_idx = threshold_val

            count = N - count_below(threshold_idx)
            return float(count / N)

        features["volume_at_intensity_fraction_0.10_BC2M_10"] = (
            get_volume_fraction_at_intensity_fraction_physical(0.10)
        )
        features["volume_at_intensity_fraction_0.90_BC2M_90"] = (
            get_volume_fraction_at_intensity_fraction_physical(0.90)
        )

    features["volume_fraction_difference_between_intensity_0.10_and_0.90_fractions_DDTU"] = float(
        features["volume_at_intensity_fraction_0.10_BC2M_10"]
        - features["volume_at_intensity_fraction_0.90_BC2M_90"]
    )

    # -------------------------------------------------------------------------
    # 2. Intensity Fractions
    # -------------------------------------------------------------------------
    def get_intensity_at_volume_fraction(vol_frac: float) -> float:
        # Fast path for standard integer bins (step=1)
        if (
            bin_width is not None
            and min_val is None
            and bin_width > 0
            and float(bin_width) == 1.0
            and np.issubdtype(vals.dtype, np.integer)
        ):
            target_count = int(np.floor(vol_frac * N))
            if target_count <= 0:
                return float(kth(N - 1))

            # Smallest integer threshold t such that count(vals >= t) <= target_count.
            # Let k = N - target_count. We need searchsorted(t) >= k.
            k = N - target_count
            v = int(kth(k - 1))
            t = v + 1
            vmax = int(kth(N - 1))
            if t > vmax:
                t = vmax
            return float(t)

        # Determine candidates
        if bin_width is not None:
            g_min = min_val if min_val is not None else np.min(discretised_values)
            if max_val is not None:
                g_max = max_val
            elif min_val is not None:
                g_max = min_val + np.max(discretised_values) * bin_width
            else:
                g_max = np.max(discretised_values)

            if bin_width > 0:
                # Degenerate ranges (g_max within half a bin of g_min) would yield
                # an empty candidate grid; keep at least one candidate.
                num_steps = max(int(np.round((g_max - g_min) / bin_width)), 1)
                if min_val is not None:
                    # Candidates are bin centers
                    candidates: _Grid | npt.NDArray[Any] = _Grid(num_steps, g_min, bin_width, 0.5)
                else:
                    candidates = _Grid(num_steps + 1, g_min, bin_width, 0.0)
            else:
                # The values that occur: the first one that meets the condition below is
                # the first such value of all the sorted values
                candidates = unique_vals.astype(np.float64)
        else:
            candidates = unique_vals

        target_count = int(np.floor(vol_frac * N))

        # Binary search
        low = 0
        high = len(candidates) - 1
        ans_idx = -1

        while low <= high:
            mid = (low + high) // 2
            val = candidates[mid]

            # Convert physical value to index if in discrete mode
            if bin_width is not None and min_val is not None and bin_width > 0:
                check_val = np.floor((val - min_val) / bin_width) + 1
            else:
                check_val = val

            count = N - count_below(check_val)

            if count <= target_count:
                ans_idx = mid
                high = mid - 1
            else:
                low = mid + 1

        if ans_idx != -1:
            return float(candidates[ans_idx])
        else:
            return float(candidates[-1])

    features["intensity_at_volume_fraction_0.10_GBPN_10"] = get_intensity_at_volume_fraction(0.10)
    features["intensity_at_volume_fraction_0.90_GBPN_90"] = get_intensity_at_volume_fraction(0.90)

    features["intensity_fraction_difference_between_volume_0.10_and_0.90_fractions_CNV2"] = float(
        features["intensity_at_volume_fraction_0.10_GBPN_10"]
        - features["intensity_at_volume_fraction_0.90_GBPN_90"]
    )

    # -------------------------------------------------------------------------
    # 3. Area Under the IVH Curve (AUC)
    # -------------------------------------------------------------------------
    # IVH Curve: Volume Fraction (phi) vs Intensity (I)
    # We construct the curve points from the unique values in the data.
    if len(unique_vals) == 1:
        # If there is only one discretised intensity, AUC is 0 by definition.
        features["area_under_the_ivh_curve_9CMM"] = 0.0
    else:
        # P(X >= i)
        # For each unique value, calculate fraction >= value

        # If we have physical mapping, map unique_vals to physical intensities
        if bin_width is not None and min_val is not None:
            # Map index to physical center: min_val + (idx - 0.5) * w ?
            # Standard FBS mapping: index k corresponds to [min + (k-1)w, min + kw)
            # center = min_val + (k - 1 + 0.5) * w
            # k is the value in unique_vals
            intensities_arr = min_val + (unique_vals.astype(np.float64) - 0.5) * bin_width
        else:
            intensities_arr = unique_vals.astype(np.float64)

        # Calculate volume fractions: the values >= each unique value
        counts = N - below_unique
        fractions = counts.astype(np.float64) / float(N)

        # Trapezoidal integration of fraction(I) over I.
        features["area_under_the_ivh_curve_9CMM"] = float(np.trapezoid(fractions, intensities_arr))

    return features


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _value_counts_numba(values: npt.NDArray[Any], low: int, counts: npt.NDArray[np.int64]) -> None:
    """counts[c, v - low] += 1 for each value v of part c of `values` (the parts run in
    parallel, each into its own row)."""
    n = values.size
    parts = counts.shape[0]
    for c in prange(parts):
        row = counts[c]
        for i in range(c * n // parts, (c + 1) * n // parts):
            row[values[i] - low] += 1


@jit(nopython=True, cache=True)  # type: ignore
def _count_values_numba(values: npt.NDArray[Any], low: int, counts: npt.NDArray[np.int64]) -> None:
    """counts[c, v - low] += 1 for each value v, on one core. Four rows take the values
    in turn, so a run of one value waits on no earlier add; one row where the range is
    long (the rows are the memory)."""
    n = values.size
    if counts.shape[0] == 1:
        row = counts[0]
        for i in range(n):
            row[values[i] - low] += 1
        return
    i = 0
    while i + 4 <= n:
        counts[0, values[i] - low] += 1
        counts[1, values[i + 1] - low] += 1
        counts[2, values[i + 2] - low] += 1
        counts[3, values[i + 3] - low] += 1
        i += 4
    while i < n:
        counts[0, values[i] - low] += 1
        i += 1


@jit(nopython=True, cache=True)  # type: ignore
def _whole_as_int32_numba(values: npt.NDArray[Any], out: npt.NDArray[np.int32]) -> bool:
    """out[i] = values[i] as int32 when every value is a whole number; False at the first
    other value (`out` is then partial). The caller checks the int32 range first."""
    for i in range(values.size):
        v = values[i]
        k = np.int32(v)
        if k != v:
            return False
        out[i] = k
    return True


# From this many values on, the IVH counts whole-number values instead of sorting them
# (measured: below, the sort is as fast). From _PARALLEL_COUNT_MIN on, the count runs in
# threads (below, one core with four count rows is as fast; a thread start costs 70 us).
_COUNT_MIN = 1 << 12
_PARALLEL_COUNT_MIN = 1 << 20


def _value_counts(values: npt.NDArray[Any], low: int, size: int) -> npt.NDArray[np.int64]:
    """The number of times each value low + j occurs in the 1-D integer `values`, for j in
    range(size): parts in threads, each part into its own row (a row per thread only
    where `size` is small next to the values), or one core below _PARALLEL_COUNT_MIN."""
    if values.dtype in (np.int32, np.int64):
        data = values
    else:  # int32 copies of the 1- and 2-byte types: numpy casts them 6 times faster
        data = values.astype(np.int32 if values.dtype.itemsize <= 2 else np.int64)
    parts = max(1, min(get_num_threads(), values.size // max(size, 1 << 14)))
    if parts == 1 or values.size < _PARALLEL_COUNT_MIN:
        counts = np.zeros((4 if size <= 1 << 16 else 1, size), dtype=np.int64)
        _count_values_numba(data, low, counts)
    else:
        counts = np.zeros((parts, size), dtype=np.int64)
        _value_counts_numba(data, low, counts)
    return counts.sum(axis=0)


def _counted_view(
    values: npt.NDArray[Any],
) -> Optional[
    tuple[Callable[[Any], Any], Callable[[int], Any], npt.NDArray[Any], npt.NDArray[np.intp]]
]:
    """The `_sorted_view` from one count per value: from _COUNT_MIN values on, integers
    (not uint64) and float32 or float64 values that are all whole numbers of the int32
    range, over a range not much longer than the values. None for all other values."""
    n = values.size
    if n < _COUNT_MIN:
        return None
    if values.dtype.kind in "iu" and values.dtype != np.uint64:
        lo, hi = values.min(), values.max()
    elif values.dtype in (np.float32, np.float64) and values[0] == np.floor(values[0]):
        lo, hi = values.min(), values.max()
        if not (lo >= -2147483648.0 and hi <= 2147483647.0):  # NaN and inf fail too
            return None
    else:
        return None
    low = min(int(lo), 0)  # counts from 0, or from a negative minimum
    size = int(hi) - low + 1
    if size >= 2 * n + 65536:
        return None
    if values.dtype.kind == "f":
        ints = np.empty(n, dtype=np.int32)
        if not _whole_as_int32_numba(values, ints):
            return None
    else:
        ints = values
    counts = _value_counts(ints, low, size)
    cum = np.cumsum(counts)  # the values at or below low + j
    occur = np.flatnonzero(counts)

    def count_below_int(t: Any) -> int:
        j = math.ceil(t) - low  # the integers below t are those below ceil(t)
        return 0 if j <= 0 else int(cum[min(j, cum.size) - 1])

    def kth_int(k: int) -> int:
        return low + int(np.searchsorted(cum, k, side="right"))

    return count_below_int, kth_int, occur + low, cum[occur] - counts[occur]


def _sorted_view(
    values: npt.NDArray[Any],
) -> tuple[Callable[[Any], Any], Callable[[int], Any], npt.NDArray[Any], npt.NDArray[np.intp]]:
    """What the IVH reads from the sorted 1-D `values`: count_below(t), the number of
    values below t; kth(k), the k-th smallest value (from 0); the values that occur,
    ascending; and the number of values below each of them. Many whole-number values over
    a range not much longer than the values take one count per value instead of a sort."""
    counted = _counted_view(values)
    if counted is not None:
        return counted
    sorted_vals = np.sort(values)
    # np.searchsorted converts an integer array to float64 on each call with a float
    # threshold. Convert it one time; the comparisons stay the same.
    sorted_f = sorted_vals.astype(np.float64, copy=False)
    # Unique values from the neighbour differences of the sorted values (no second sort)
    first = np.empty(sorted_vals.shape, dtype=bool)
    first[0] = True
    np.not_equal(sorted_vals[1:], sorted_vals[:-1], out=first[1:])
    starts = np.flatnonzero(first)

    return sorted_f.searchsorted, sorted_vals.__getitem__, sorted_vals[starts], starts


# Peak memory of the FFT sums: 24 to 28 bytes per point of their grid (measured); 32 leaves
# a margin. The slab sums (see _spatial_sums_slabs) hold more spectra at a time: 40 to 55
# bytes per point of the slab grid (measured); 64 leaves a margin.
_FFT_BYTES_PER_POINT = 32
_SLAB_BYTES_PER_POINT = 64
# The time model that chooses the faster method (measured on an M4 Pro, 1 to 10 threads):
# the pair loop takes about 1.05 ns per voxel pair and thread, and its parallel form about
# 45 us + 2.5 us per thread to start. The FFT sums take about 12.5 ns per grid point on one
# worker, of which 15 % does not split into workers (30 % below _SPATIAL_FFT_PARALLEL_MIN
# points, where the numba passes run on one thread), and about 47 us + 38 us per worker to
# start. So a small grid runs faster on one worker: below about 45,000 points on 10
# threads, 19,000 on 4 (measured: 32,768 points 0.45 ms on 1 worker, 0.54 ms on 10;
# 64,000 points 0.81 ms on 1, 0.65 ms on 10). The slab sums with s slabs make 2 s (s + 1)
# FFTs of the slab grid, each 0.27 of the time per point of the FFT sums (measured on 10
# threads: 0.78 to 0.83 ns per point and FFT, against 3.2 to 3.4 ns).
_PAIR_SECONDS = 1.05e-9
_PAIR_START_SECONDS = (45e-6, 2.5e-6)
_FFT_SECONDS_PER_POINT = 12.5e-9
_FFT_SERIAL_PART = (0.3, 0.15)
_FFT_START_SECONDS = (47e-6, 38e-6)
_SLAB_FFT_UNITS = 0.27


def _fft_seconds(points: int, units: float, workers: int) -> float:
    """Model time of FFT work on `workers` workers: `units` times the work per point of the
    FFT sums, on a grid of `points` points (see _PAIR_SECONDS)."""
    serial = _FFT_SERIAL_PART[points >= _SPATIAL_FFT_PARALLEL_MIN]
    share = serial + (1.0 - serial) / workers
    start = _FFT_START_SECONDS[0] + _FFT_START_SECONDS[1] * workers
    return start + units * points * _FFT_SECONDS_PER_POINT * share


def _fft_workers(points: int) -> int:
    """The FFT workers of the spatial sums on a grid of `points` points: numba's thread
    count, or one when the model gives one worker less time (a small grid)."""
    threads = int(get_num_threads())
    if _fft_seconds(points, 1.0, threads) < _fft_seconds(points, 1.0, 1):
        return threads
    return 1


def _pair_seconds(n: int) -> float:
    """Model time of the pair loop of `n` ROI voxels (see _PAIR_SECONDS)."""
    threads = get_num_threads() if n >= _PAIRS_PARALLEL_MIN else 1
    start = _PAIR_START_SECONDS[0] + _PAIR_START_SECONDS[1] * threads if threads > 1 else 0.0
    return start + n * (n - 1) / 2 * _PAIR_SECONDS / threads


def _spatial_plan(n: int, box: tuple[int, ...]) -> tuple[str, int, int, tuple[int, ...]]:
    """The fastest method of the spatial sums of `n` ROI voxels in a box of shape `box`, by
    the time model (see _PAIR_SECONDS), within the memory budget (half of the memory, see
    threads._memory_budget).

    Returns ("fft", 0, 0, grid) for the FFT sums on the whole grid, ("slabs", axis, rows,
    slab grid) for the slab sums with slabs of `rows` planes along `axis` (when the whole
    grid does not fit; the fewest slabs that fit, as the work grows with their number), or
    ("pairs", 0, 0, ()) for the pair loop. When not even slabs of one plane fit, the pair
    loop runs with a warning that gives its expected time.
    """
    budget = _memory_budget()
    pairs = _pair_seconds(n)
    fshape = tuple(scipy.fft.next_fast_len(2 * k - 1, real=True) for k in box)
    grid = math.prod(fshape)
    if _FFT_BYTES_PER_POINT * grid <= budget:
        fft = _fft_seconds(grid, 1.0, _fft_workers(grid))
        return ("fft", 0, 0, fshape) if fft < pairs else ("pairs", 0, 0, ())
    # Cut the longest axis (it allows the thinnest slabs) into the fewest slabs that fit
    axis = max(range(len(box)), key=box.__getitem__)
    count = rows = 2
    while rows > 1:
        rows = -(-box[axis] // count)
        slab = scipy.fft.next_fast_len(2 * rows - 1, real=True)
        sshape = (*fshape[:axis], slab, *fshape[axis + 1 :])
        points = math.prod(sshape)
        if _SLAB_BYTES_PER_POINT * points <= budget:
            count = -(-box[axis] // rows)
            units = _SLAB_FFT_UNITS * 2 * count * (count + 1)
            slabs = _fft_seconds(points, units, _fft_workers(points))
            return ("slabs", axis, rows, sshape) if slabs < pairs else ("pairs", 0, 0, ())
        count += 1
    warnings.warn(
        f"Moran's I and Geary's C: the FFT method needs about "
        f"{_SLAB_BYTES_PER_POINT * points / 2**30:.2f} GiB even in slabs of one plane, more "
        f"than its limit of {budget / 2**30:.2f} GiB (half of the memory). The pair loop runs "
        f"instead. Expected run time: about {pairs / 60:.1f} min.",
        stacklevel=3,
    )
    return ("pairs", 0, 0, ())


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _inverse_distance_numba(
    out: npt.NDArray[np.float64],
    ax: npt.NDArray[np.float64],
    ay: npt.NDArray[np.float64],
    az: npt.NDArray[np.float64],
) -> None:
    """out[i, j, k] = 1 / sqrt(ax[i]^2 + ay[j]^2 + az[k]^2), and 0 at a distance of 0."""
    nx, ny, nz = out.shape
    for i in prange(nx):
        x2 = ax[i] * ax[i]
        for j in range(ny):
            xy = x2 + ay[j] * ay[j]
            for k in range(nz):
                r2 = xy + az[k] * az[k]
                out[i, j, k] = 1.0 / np.sqrt(r2) if r2 > 0.0 else 0.0


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _multiply_spectrum_numba(
    spectrum: npt.NDArray[np.complex128],
    kernel: npt.NDArray[np.float64],
    fold0: npt.NDArray[np.int64],
    fold1: npt.NDArray[np.int64],
) -> None:
    """spectrum[i, j, k] *= kernel[fold0[i], fold1[j], k]: the real kernel spectrum."""
    for i in prange(spectrum.shape[0]):
        a = fold0[i]
        for j in range(spectrum.shape[1]):
            row = kernel[a, fold1[j]]
            for k in range(spectrum.shape[2]):
                spectrum[i, j, k] = spectrum[i, j, k] * row[k]


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _weighted_power_numba(
    spectrum: npt.NDArray[np.complex128],
    kernel: npt.NDArray[np.float64],
    fold0: npt.NDArray[np.int64],
    fold1: npt.NDArray[np.int64],
    n_last: int,
    partial: npt.NDArray[np.float64],
) -> None:
    """partial[i] = sum over j, k of w_k kernel[fold0[i], fold1[j], k] |spectrum[i, j, k]|^2.

    The half spectrum of the last axis (length n_last) holds bins 0 and n_last / 2 once
    (w_k = 1) and every other bin for itself and its conjugate (w_k = 2).
    """
    n0, n1, n2 = spectrum.shape
    for i in prange(n0):
        a = fold0[i]
        s = 0.0
        for j in range(n1):
            row = kernel[a, fold1[j]]
            for k in range(n2):
                c = spectrum[i, j, k]
                p = (c.real * c.real + c.imag * c.imag) * row[k]
                s += p if (k == 0 or 2 * k == n_last) else 2.0 * p
        partial[i] = s


@jit(nopython=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _cross_power_numba(
    a: npt.NDArray[np.complex128],
    kernel: npt.NDArray[np.complex128],
    b: npt.NDArray[np.complex128],
    n_last: int,
    partial: npt.NDArray[np.float64],
) -> None:
    """partial[i] = sum over j, k of w_k Re(conj(a[i, j, k]) kernel[i, j, k] b[i, j, k]),
    with the weights w_k of _weighted_power_numba (half spectra of length n_last)."""
    n0, n1, n2 = a.shape
    for i in prange(n0):
        s = 0.0
        for j in range(n1):
            for k in range(n2):
                x = a[i, j, k]
                y = kernel[i, j, k] * b[i, j, k]
                p = x.real * y.real + x.imag * y.imag
                s += p if (k == 0 or 2 * k == n_last) else 2.0 * p
        partial[i] = s


# Serial twins of the four FFT kernels (see serial_twin). Measured on 10 threads: below
# this many grid points, starting the threads costs more than the work.
_inverse_distance_numba_serial = serial_twin(_inverse_distance_numba)
_multiply_spectrum_numba_serial = serial_twin(_multiply_spectrum_numba)
_weighted_power_numba_serial = serial_twin(_weighted_power_numba)
_cross_power_numba_serial = serial_twin(_cross_power_numba)
_SPATIAL_FFT_PARALLEL_MIN = 1 << 20


def _kernel_spectrum(
    fshape: tuple[int, ...], spacing: tuple[float, float, float], workers: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The real spectrum of the kernel 1 / |r| on the FFT grid, read as kernel[fold0[i], fold1[j], k].

    The kernel is even along each axis, so its spectrum is real. With even grid lengths it
    is the DCT-I of the half grid (n / 2 + 1 points per axis), read with folded indices.
    Else it is the real part of the full rfftn, read with the plain indices.
    """
    grid = math.prod(fshape)
    build = sized(
        _inverse_distance_numba, _inverse_distance_numba_serial, grid, _SPATIAL_FFT_PARALLEL_MIN
    )
    # Distance in mm along each axis, with the offsets wrapped around the grid.
    axes = [
        np.minimum(np.arange(n), n - np.arange(n)) * s for n, s in zip(fshape, spacing, strict=True)
    ]
    if all(n % 2 == 0 for n in fshape):
        half = tuple(n // 2 + 1 for n in fshape)
        kernel = np.empty(half)
        build(kernel, axes[0][: half[0]], axes[1][: half[1]], axes[2][: half[2]])
        kernel = scipy.fft.dctn(kernel, type=1, workers=workers, overwrite_x=True)
        fold0, fold1 = (np.minimum(np.arange(n), n - np.arange(n)) for n in fshape[:2])
    else:
        kernel = np.empty(fshape)
        build(kernel, axes[0], axes[1], axes[2])
        kernel = np.ascontiguousarray(scipy.fft.rfftn(kernel, workers=workers).real)
        fold0, fold1 = (np.arange(n) for n in fshape[:2])
    return kernel, fold0, fold1


def _spatial_sums_fft(
    roi: npt.NDArray[np.bool_],
    diff: npt.NDArray[np.float64],
    spacing: tuple[float, float, float],
    fshape: tuple[int, ...],
) -> tuple[float, float, float]:
    """The three sums of `_calculate_spatial_features_numba`, from FFT convolutions.

    A voxel pair has the weight 1 / distance. The convolution of the ROI with the kernel
    1 / |r| (0 at r = 0) gives, for each voxel, the sum of its weights to all ROI voxels.
    So the sum of weights is roi . (K * roi), the Moran numerator is diff . (K * diff), and
    the Geary numerator is 2 diff^2 . (K * roi) - 2 times the Moran numerator. `diff` holds
    the intensity minus the ROI mean (0 outside the ROI), and `fshape` is the FFT grid.

    The kernel spectrum is real, so the Moran numerator is the kernel-weighted power of the
    spectrum of `diff` (Parseval), with no inverse transform. One padded buffer serves both
    forward transforms.
    """
    grid = math.prod(fshape)
    workers = _fft_workers(grid)
    kernel, fold0, fold1 = _kernel_spectrum(fshape, spacing, workers)
    box = tuple(slice(0, n) for n in roi.shape)
    d = diff[roi]
    buffer = np.zeros(fshape)
    view = buffer[box]
    view[roi] = d
    spectrum = scipy.fft.rfftn(buffer, workers=workers)
    partial = np.empty(fshape[0])
    power = sized(
        _weighted_power_numba, _weighted_power_numba_serial, grid, _SPATIAL_FFT_PARALLEL_MIN
    )
    power(spectrum, kernel, fold0, fold1, fshape[2], partial)
    del spectrum
    numer_moran = float(np.sum(partial)) / grid
    view[...] = roi
    spectrum = scipy.fft.rfftn(buffer, workers=workers)
    del buffer, view
    multiply = sized(
        _multiply_spectrum_numba, _multiply_spectrum_numba_serial, grid, _SPATIAL_FFT_PARALLEL_MIN
    )
    multiply(spectrum, kernel, fold0, fold1)
    del kernel
    full = scipy.fft.irfftn(spectrum, fshape, workers=workers, overwrite_x=True)
    del spectrum
    weights = np.asarray(full[box][roi])
    del full
    numer_geary = 2.0 * float(np.sum(d * d * weights)) - 2.0 * numer_moran
    return numer_moran, numer_geary, float(np.sum(weights))


def _spatial_sums_slabs(
    roi: npt.NDArray[np.bool_],
    diff: npt.NDArray[np.float64],
    spacing: tuple[float, float, float],
    axis: int,
    rows: int,
    fshape: tuple[int, ...],
) -> tuple[float, float, float]:
    """The three sums of `_spatial_sums_fft` for a box whose FFT grid does not fit in the
    memory budget: the box is cut along `axis` into slabs of `rows` planes, and `fshape` is
    the FFT grid of one slab (at least 2 rows - 1 points along `axis`).

    Each sum is a form a . (K * b) of two arrays of the box (diff, the ROI, diff^2) and the
    kernel 1 / |r|: over the slabs P and Q = P + k, the sum of a_P . (K_k * b_Q), where K_k
    holds the kernel at the offsets between the planes of P and those of Q (k * rows planes
    apart, plus -(rows - 1) to rows - 1). On the slab grid these offsets do not wrap
    around, so each term is a weighted product of three spectra (Parseval), with no inverse
    transform. The kernel is symmetric, so the pairs (P, Q) and (Q, P) give the same Moran
    and weight terms, and the loop visits each pair once. The work grows with the square
    of the number of slabs.
    """
    order = (axis, *(a for a in range(3) if a != axis))
    roi, diff = roi.transpose(order), diff.transpose(order)
    sizes = tuple(fshape[a] for a in order)
    n0, n1, n2 = roi.shape
    count = -(-n0 // rows)
    points = math.prod(sizes)
    workers = _fft_workers(points)
    build = sized(
        _inverse_distance_numba, _inverse_distance_numba_serial, points, _SPATIAL_FFT_PARALLEL_MIN
    )
    cross = sized(_cross_power_numba, _cross_power_numba_serial, points, _SPATIAL_FFT_PARALLEL_MIN)
    # Signed plane offsets along the cut axis (offset k * rows apart in mm for slab pairs
    # k apart), and the distances along the other axes, wrapped around the grid
    index = np.arange(sizes[0])
    lag = np.where(index <= sizes[0] // 2, index, index - sizes[0]) * spacing[order[0]]
    across = [
        np.minimum(np.arange(n), n - np.arange(n)) * spacing[a]
        for n, a in zip(sizes[1:], order[1:], strict=True)
    ]
    buffer = np.zeros(sizes)
    view = buffer[:rows, :n1, :n2]
    partial = np.empty(sizes[0])

    def spectrum(p: int, kind: str) -> npt.NDArray[np.complex128]:
        """The spectrum of slab p of diff ("d"), the ROI ("r") or diff^2 ("q")."""
        part = slice(p * rows, min(p * rows + rows, n0))
        filled = view[: part.stop - part.start]
        if kind == "q":
            np.square(diff[part], out=filled)
        else:
            np.copyto(filled, diff[part] if kind == "d" else roi[part])
        view[part.stop - part.start :] = 0.0
        return cast(npt.NDArray[np.complex128], scipy.fft.rfftn(buffer, workers=workers))

    def kernel(k: int) -> npt.NDArray[np.complex128]:
        """The spectrum of the kernel between slabs k apart (the buffer is zero after)."""
        build(buffer, lag - k * rows * spacing[order[0]], *across)
        spec = scipy.fft.rfftn(buffer, workers=workers)
        buffer.fill(0.0)
        return cast(npt.NDArray[np.complex128], spec)

    def form(
        a: npt.NDArray[np.complex128],
        kern: npt.NDArray[np.complex128],
        b: npt.NDArray[np.complex128],
    ) -> float:
        cross(a, kern, b, sizes[2], partial)
        return float(np.sum(partial))

    moran = weights = squares = 0.0  # squares: diff^2 . (K * roi)
    for p in range(count):
        d_p, r_p, q_p = spectrum(p, "d"), spectrum(p, "r"), spectrum(p, "q")
        for k in range(count - p):
            kern = kernel(k)
            if k == 0:
                moran += form(d_p, kern, d_p)
                weights += form(r_p, kern, r_p)
                squares += form(q_p, kern, r_p)
            else:
                # One spectrum of slab p + k at a time
                other = spectrum(p + k, "r")
                weights += 2.0 * form(r_p, kern, other)
                squares += form(q_p, kern, other)
                del other
                other = spectrum(p + k, "q")
                squares += form(r_p, kern, other)
                del other
                other = spectrum(p + k, "d")
                moran += 2.0 * form(d_p, kern, other)
                del other
            del kern
        del d_p, r_p, q_p
    return moran / points, 2.0 * (squares - moran) / points, weights / points


def calculate_spatial_intensity_features(
    image: Image,
    mask: Image,
    *,
    enabled: bool = True,
) -> dict[str, float]:
    """
    Calculate spatial intensity features: Moran's I and Geary's C (IBSI 4.4).

    These features measure spatial autocorrelation of intensity values within the ROI.
    A time model picks the faster of two methods (see _spatial_plan): a loop over all
    voxel pairs, whose time grows with N² (N = number of ROI voxels), or FFT convolutions,
    whose time grows with the ROI box. Both give the same values to about 1e-14
    (relative). The FFT needs about 32 bytes per point of a grid that is twice the ROI box
    along each axis, and it uses at most half of the memory (shared by the workers of
    run_batch). When the grid needs more, the FFT runs on slabs of the box: a few times
    longer, within the memory.

    Args:
        image: Image object containing intensity data.
        mask: Image object containing the ROI mask.
        enabled: If False, returns empty dict immediately (for performance).

    Returns:
        Dictionary with 'morans_i_index_N365' and 'gearys_c_measure_NPT7'.
        Returns NaN values if ROI has fewer than 2 voxels or constant intensity.

    Example:
        ```python
        import numpy as np
        from pictologics.loader import Image
        from pictologics.features.intensity import calculate_spatial_intensity_features

        data = np.random.default_rng(0).normal(size=(6, 6, 6))
        mask_arr = np.zeros((6, 6, 6), dtype=np.uint8)
        mask_arr[2:4, 2:4, 2:4] = 1

        image = Image(array=data, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
        mask = Image(array=mask_arr, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))

        features = calculate_spatial_intensity_features(image, mask)
        print(sorted(features.keys()))
        # ['gearys_c_measure_NPT7', 'morans_i_index_N365']
        ```
    """
    if not enabled:
        return {}

    features: dict[str, float] = {}

    mask_array = mask.array
    data = image.array
    sx, sy, sz = (
        float(image.spacing[0]),
        float(image.spacing[1]),
        float(image.spacing[2]),
    )

    # The ROI (mask != 0) in its bounding box (an empty mask gives an empty box). Its
    # voxels keep their full-image order, so `intensities` is the same as data[mask != 0].
    bbox = compute_nonzero_bbox(mask_array) or (slice(0, 0),) * 3
    roi = mask_array[bbox] != 0
    intensities = np.ascontiguousarray(data[bbox][roi].astype(np.float64, copy=False))

    N = len(intensities)
    if N < 2:
        features["morans_i_index_N365"] = np.nan
        features["gearys_c_measure_NPT7"] = np.nan
        return features

    mean_int = np.mean(intensities)
    denom = _sum_sq_centered(intensities, float(mean_int))

    # The FFT grid is at least 2 n - 1 along each axis of the box (of a slab, along its cut
    # axis), so no pair wraps around.
    method, axis, rows, fshape = _spatial_plan(N, roi.shape)
    if method != "pairs":
        diff = np.zeros(roi.shape)
        diff[roi] = intensities - mean_int
        numer_moran, numer_geary, sum_weights = (
            _spatial_sums_fft(roi, diff, (sx, sy, sz), fshape)
            if method == "fft"
            else _spatial_sums_slabs(roi, diff, (sx, sy, sz), axis, rows, fshape)
        )
    else:
        # Full-image voxel indices, as the pair loop used before the crop.
        x_idx, y_idx, z_idx = (
            np.ascontiguousarray((idx + box.start).astype(np.int32))
            for idx, box in zip(np.nonzero(roi), bbox, strict=True)
        )
        pairs = sized(
            _calculate_spatial_features_numba,
            _calculate_spatial_features_numba_serial,
            N,
            _PAIRS_PARALLEL_MIN,
        )
        numer_moran, numer_geary, sum_weights = pairs(
            x_idx, y_idx, z_idx, intensities, float(mean_int), sx, sy, sz
        )

    # Moran's I - N365

    if denom != 0 and sum_weights != 0:
        moran_i = (N / sum_weights) * (numer_moran / denom)
        features["morans_i_index_N365"] = float(moran_i)
    else:
        features["morans_i_index_N365"] = np.nan

    # Geary's C - NPT7
    if denom != 0 and sum_weights != 0:
        geary_c = ((N - 1) / (2 * sum_weights)) * (numer_geary / denom)
        features["gearys_c_measure_NPT7"] = float(geary_c)
    else:
        features["gearys_c_measure_NPT7"] = np.nan

    return features


# Radius of the 1 cm3 sphere of the local intensity peaks (IBSI 4.5).
_LOCAL_PEAK_RADIUS_MM = 6.2035

# Measured crossover: below 2^15 voxels, the box search of the local intensity crop costs
# more than the crop saves.
_LOCAL_CROP_MIN_SIZE = 1 << 15


# From this many (ROI voxel, sphere offset) pairs on, the local peaks take two stages.
_TWO_STAGE_MIN_WORK = 1 << 16


@lru_cache(maxsize=32)
def _sphere_rows(
    spacing: tuple[float, float, float],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The rows (dx, dy, -dz..dz) of the sphere of the local peaks: row_dx, row_dy, row_dz."""
    rows: dict[tuple[int, int], int] = {}
    for dx, dy, dz in _sphere_offsets_for_radius(spacing, _LOCAL_PEAK_RADIUS_MM).tolist():
        rows[(dx, dy)] = max(rows.get((dx, dy), 0), abs(dz))
    row_dx = np.array([dx for dx, _ in rows], dtype=np.int64)
    row_dy = np.array([dy for _, dy in rows], dtype=np.int64)
    row_dz = np.array(list(rows.values()), dtype=np.int64)
    return row_dx, row_dy, row_dz


# Non-finite voxels of the crop from which the two-stage search gives way to the exact kernel
# (each one costs a test of its sphere offsets)
_NON_FINITE_MAX = 4096


def _without_unreached(
    data: npt.NDArray[Any], roi: npt.NDArray[np.bool_], offsets: npt.NDArray[np.int32]
) -> Optional[npt.NDArray[Any]]:
    """A copy of `data` with its non-finite voxels set to 0 when no sphere of an ROI voxel
    reaches one (the sphere is symmetric, so the test reads `roi` at each voxel plus the
    offsets), else None; None also above _NON_FINITE_MAX such voxels. No exact sphere mean
    of an ROI voxel reads a changed voxel."""
    bad = np.argwhere(~np.isfinite(data))
    if bad.shape[0] > _NON_FINITE_MAX:
        return None
    shape = np.array(data.shape)
    for voxel in bad:
        reached = voxel + offsets
        reached = reached[np.all((reached >= 0) & (reached < shape), axis=1)]
        if roi[reached[:, 0], reached[:, 1], reached[:, 2]].any():
            return None
    out = data.copy()
    out[tuple(bad.T)] = 0.0
    return out


def _local_peaks_two_stage(
    data: npt.NDArray[Any],
    indices: npt.NDArray[np.int32],
    values: npt.NDArray[np.float64],
    runs: npt.NDArray[np.int64],
    offsets: npt.NDArray[np.int32],
    spacing: tuple[float, float, float],
    roi: npt.NDArray[np.bool_],
) -> Optional[tuple[float, float]]:
    """The global and the local intensity peak, the values of `_calculate_local_mean_numba`
    and `_calculate_local_peaks_numba` bit for bit, with the exact kernel on few voxels.

    Stage 1 gives every ROI voxel an approximate sphere mean from row prefix sums (one
    subtraction per sphere row). Both means differ from the true one by at most
    t = 2 u A (rows (2 L^2 + 4) + offsets^2), with u the unit roundoff, A the largest
    |intensity| and L the row length (summation bounds of the prefix sums, the row
    differences and the kernel sum, in any order). So the voxel with the largest exact
    mean has an approximate mean within 2 t of the largest one, and stage 2 runs the
    exact kernel only on those voxels and on the maximum-intensity voxels (the local
    peak). A non-finite intensity that no ROI sphere reaches is set to 0 in a copy (see
    _without_unreached); None (use the kernel on all voxels) when one is in reach.
    """
    row_dx, row_dy, row_dz = _sphere_rows(spacing)
    nx, ny, nz = data.shape
    n = indices.shape[0]
    # Long z-runs: the run kernel, on prefix sums padded by the reach of the sphere rows (a
    # row past the whole z extent reads the same two sums as one clipped to it, so the pad
    # stays within the extent). Short runs (a thin ROI): the voxel kernel, with no pad.
    by_run = n >= _RUN_MEAN_LENGTH_MIN * runs.shape[0]
    pad = min(int(row_dz.max()), nz - 1) if by_run else 0
    if by_run and pad < row_dz.max():
        row_dz = np.minimum(row_dz, pad)
    prefix = np.empty((nx, ny, nz + 1 + 2 * pad), dtype=np.float64)
    lo = np.empty(nx, dtype=np.float64)
    hi = np.empty(nx, dtype=np.float64)
    prefix_sums = sized(_prefix_pad_numba, _prefix_pad_numba_serial, data.size, _CROP_PARALLEL_MIN)
    prefix_sums(data, pad, prefix, lo, hi)
    if not np.isfinite(prefix[:, :, pad + nz]).all():  # a NaN or infinite value: the row totals
        finite = _without_unreached(data, roi, offsets)
        if finite is None:
            return None
        data = finite
        prefix_sums(data, pad, prefix, lo, hi)
    largest = max(abs(float(hi.max())), abs(float(lo.min())))
    work = n * row_dx.shape[0]
    if by_run:
        approx = np.empty(n, dtype=np.float64)
        run_means = sized(
            _approx_runs_numba, _approx_runs_numba_serial, work, _LOCAL_MEAN_PARALLEL_MIN
        )
        run_means(prefix, pad, nz, runs, row_dx, row_dy, row_dz, approx)
    else:
        voxel_means = sized(
            _approximate_local_means_numba,
            _approximate_local_means_numba_serial,
            work,
            _LOCAL_MEAN_PARALLEL_MIN,
        )
        approx = voxel_means(prefix, indices, row_dx, row_dy, row_dz)
    del prefix
    length = nz + 1
    bound = (
        2.0**-52
        * largest
        * (row_dx.shape[0] * (2.0 * length * length + 4.0) + offsets.shape[0] ** 2)
    )
    near_best = np.flatnonzero(approx >= approx.max() - 2.0 * bound)
    brightest = np.flatnonzero(values == values.max())
    selected = np.union1d(near_best, brightest)  # sorted
    exact_mean = sized(
        _calculate_local_mean_numba,
        _calculate_local_mean_numba_serial,
        selected.shape[0] * offsets.shape[0],
        _LOCAL_MEAN_PARALLEL_MIN,
    )
    exact = exact_mean(data, np.ascontiguousarray(indices[selected]), offsets)
    global_peak = exact[np.searchsorted(selected, near_best)].max()
    local_peak = exact[np.searchsorted(selected, brightest)].max()
    return float(global_peak), float(local_peak)


def calculate_local_intensity_features(
    image: Image,
    mask: Image,
    *,
    enabled: bool = True,
) -> dict[str, float]:
    """
    Calculate local intensity features: Local and Global Intensity Peak (IBSI 4.5).

    Computes intensity peaks using a 1 cm³ spherical neighborhood (radius ~6.2mm).
    Global peak is the maximum local mean, local peak is the local mean at max intensity.

    Args:
        image: Image object containing intensity data.
        mask: Image object containing the ROI mask.
        enabled: If False, returns empty dict immediately (for performance).

    Returns:
        Dictionary with 'global_intensity_peak_0F91' and 'local_intensity_peak_VJGA'.

    Example:
        ```python
        import numpy as np
        from pictologics.loader import Image
        from pictologics.features.intensity import calculate_local_intensity_features

        data = np.random.default_rng(0).normal(size=(6, 6, 6))
        mask_arr = np.zeros((6, 6, 6), dtype=np.uint8)
        mask_arr[2:4, 2:4, 2:4] = 1

        image = Image(array=data, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
        mask = Image(array=mask_arr, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))

        features = calculate_local_intensity_features(image, mask)
        print(sorted(features.keys()))
        # ['global_intensity_peak_0F91', 'local_intensity_peak_VJGA']
        ```
    """
    if not enabled:
        return {}

    features: dict[str, float] = {}

    mask_array = mask.array
    data = image.array
    spacing_tuple = (
        float(image.spacing[0]),
        float(image.spacing[1]),
        float(image.spacing[2]),
    )

    # Radius for 1 cm^3 sphere
    offsets = _sphere_offsets_for_radius(spacing_tuple, _LOCAL_PEAK_RADIUS_MM)

    # Crop to the ROI box plus the reach of the sphere. Every sphere neighbour of an ROI
    # voxel that is in the image is also in the crop, so the local means and the peaks
    # stay the same, and the ROI search reads only the crop.
    if mask_array.size >= _LOCAL_CROP_MIN_SIZE:
        bbox = compute_nonzero_bbox(mask_array)
        if bbox is None:
            return features
        reach = np.abs(offsets).max(axis=0)
        crop = tuple(
            slice(max(box.start - int(r), 0), min(box.stop + int(r), size))
            for box, r, size in zip(bbox, reach, data.shape, strict=True)
        )
        data = data[crop]
        mask_array = mask_array[crop]
    # The local kernels read float64 or float32, the types that the import compiles; an image
    # of another type goes as a float64 copy (the same values). A column-order image (a small
    # one, without the crop) goes as a row-order copy, the layout that the import compiles.
    if data.dtype not in (np.float64, np.float32):
        data = data.astype(np.float64, order="C")
    elif data.flags.f_contiguous and not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)

    # The ROI scan reads a row-order bool array (one kernel signature per image type).
    roi = np.ascontiguousarray(mask_array != 0)
    indices, values, runs = _roi_scan(roi, data)
    n = indices.shape[0]
    if n == 0:
        return features

    peaks = None
    if n * offsets.shape[0] >= _TWO_STAGE_MIN_WORK:
        peaks = _local_peaks_two_stage(data, indices, values, runs, offsets, spacing_tuple, roi)
    if peaks is None:
        # The exact sphere mean of every ROI voxel, in threads from _LOCAL_MEAN_PARALLEL_MIN pairs
        exact_mean = sized(
            _calculate_local_mean_numba,
            _calculate_local_mean_numba_serial,
            n * offsets.shape[0],
            _LOCAL_MEAN_PARALLEL_MIN,
        )
        roi_means = exact_mean(data, indices, offsets)
        # Compute both peaks without allocating ROI intensity arrays.
        peaks = _calculate_local_peaks_numba(data, indices, roi_means)
    features["global_intensity_peak_0F91"] = float(peaks[0])
    features["local_intensity_peak_VJGA"] = float(peaks[1])

    return features
