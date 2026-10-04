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
import os
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import numpy as np
import scipy.fft
from numba import get_num_threads, jit, prange
from numpy import typing as npt

from ._utils import PRANGE_ONLY, compute_nonzero_bbox

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


@jit(nopython=True, parallel=True, fastmath=True, cache=True)  # type: ignore
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

    # Full sums: each unordered pair contributes twice to the ordered-pair sums.
    numer_moran = 2.0 * np.sum(local_moran_arr)
    numer_geary = 2.0 * np.sum(local_geary_arr)
    sum_weights = 2.0 * np.sum(local_w_sum_arr)

    return numer_moran, numer_geary, sum_weights


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


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _row_prefix_sums_numba(data: npt.NDArray[Any], out: npt.NDArray[np.float64]) -> None:
    """out[x, y, k] = the sum of data[x, y, :k] (float64), for the sphere sums."""
    for x in prange(data.shape[0]):
        for y in range(data.shape[1]):
            s = 0.0
            out[x, y, 0] = 0.0
            for z in range(data.shape[2]):
                s += float(data[x, y, z])
                out[x, y, z + 1] = s


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


def _percentile_ranks(n: int, dtype: np.dtype[Any]) -> npt.NDArray[np.intp]:
    """0-based ranks of P10, P25, P75 and P90 among n sorted values of type `dtype`.

    This is the index rule of `np.percentile(..., method="inverted_cdf")`, with the same
    float steps, so the ranks are the same as numpy's.
    """
    q = np.true_divide([10, 25, 75, 90], dtype.type(100) if dtype.kind == "f" else 100)
    index = n * q - 1
    below = np.floor(index)
    return np.where(index - below == 0, below, below + 1).astype(np.intp)


# From this many float64 values on, the order statistics come from a radix select: one
# parallel pass for the key range, one parallel count of about 65,536 key buckets over that
# range, then a partition of the few values in the buckets of the ranks. Below it, one
# partition is faster (measured).
_RADIX_SELECT_MIN = 130_000
_SIGN_BIT = np.uint64(1 << 63)
_ALL_BITS = np.uint64(0xFFFFFFFFFFFFFFFF)


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _float_key(bits: np.uint64) -> np.uint64:
    """The float64 order as an unsigned order: flip the sign bit of a positive value and
    every bit of a negative one."""
    return bits ^ _ALL_BITS if bits >> np.uint64(63) else bits ^ _SIGN_BIT


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _key_range_numba(
    values: npt.NDArray[np.float64],
    bits: npt.NDArray[np.uint64],
    lo: npt.NDArray[np.uint64],
    hi: npt.NDArray[np.uint64],
) -> None:
    """lo[t] and hi[t]: the smallest and the largest key of chunk t. A chunk with a NaN
    sets hi[t] = 0 < lo[t] and stops."""
    n = bits.size
    n_chunks = lo.size
    size = (n + n_chunks - 1) // n_chunks
    for t in prange(n_chunks):
        low = _ALL_BITS
        high = np.uint64(0)
        for i in range(t * size, min(n, (t + 1) * size)):
            if values[i] != values[i]:
                low = _ALL_BITS
                high = np.uint64(0)
                break
            k = _float_key(bits[i])
            if k < low:
                low = k
            if k > high:
                high = k
        lo[t] = low
        hi[t] = high


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


def _radix_select(
    values: npt.NDArray[np.float64], ranks: npt.NDArray[np.intp]
) -> Optional[npt.NDArray[np.float64]]:
    """The values at the sorted 0-based `ranks` (np.partition's values), or None when the
    array holds a NaN.

    About 65,536 buckets split the key range of the data evenly (keys order like the
    floats), so close values spread over many buckets. The table holds every bucket from
    the smallest key to the largest: up to 65,537 buckets, because these two keys can be
    65,536 buckets apart. The values of the buckets of the ranks are copied out and
    partitioned at the ranks inside them: a partition, not a sort, so data that crowd into
    few buckets cost about one partition of all values.
    """
    values = np.ascontiguousarray(values)
    bits = values.view(np.uint64)
    threads = get_num_threads()
    lo = np.empty(threads, dtype=np.uint64)
    hi = np.empty(threads, dtype=np.uint64)
    _key_range_numba(values, bits, lo, hi)
    if (hi < lo).any():
        return None
    low, high = int(lo.min()), int(hi.max())
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


def _order_statistics(values: npt.NDArray[Any]) -> tuple[Any, ...]:
    """P10, P25, P75, P90 and the median of a non-empty array.

    `np.percentile(..., method="inverted_cdf")` and `np.median` partition the array one
    time each. One partition at all their ranks gives the same values: the median is
    numpy's mean of the middle value or values. A NaN makes all of them NaN, as in numpy
    (a partition puts NaNs last).
    """
    n = values.size
    ranks = _percentile_ranks(n, values.dtype)
    half = n // 2
    low = half - 1 + n % 2  # lower middle rank (the middle rank when n is odd)
    if values.dtype == np.float64 and n >= _RADIX_SELECT_MIN:
        picked = _radix_select(values, np.concatenate((ranks, [low, half])))
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


def _counts_order_statistics(
    counts: npt.NDArray[Any], origin: int, dtype: np.dtype[Any]
) -> tuple[Any, ...]:
    """P10, P25, P75, P90 and the median of integer values of type `dtype`, from their
    histogram (bin i counts the value origin + i).

    The same values as `_order_statistics` gives, without a partition: the value at a rank
    is the value of the first bin whose running count is above the rank. The median is
    numpy's float64 mean of the middle value or values.
    """
    running = np.cumsum(counts)
    n = int(running[-1])
    half = n // 2
    ranks = np.concatenate((_percentile_ranks(n, dtype), [half - 1 + n % 2, half]))
    p10, p25, p75, p90, low, high = origin + np.searchsorted(running, ranks, side="right")
    return p10, p25, p75, p90, (float(low) + float(high)) / 2.0


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

    # 4.1.2 Intensity variance (ECT3)
    var_val = float(np.var(values, ddof=0))
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
        m2, m3, m4 = _central_moments_2_3_4(values, float(mean_val))
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
    p10, p25, p75, p90, median_val = _order_statistics(values)

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
    energy = float(np.dot(values, values))
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

    features: dict[str, float] = {}

    disc = np.asarray(discretised_values)
    n = disc.size

    min_val_i = int(np.min(disc))
    max_val_i = int(np.max(disc))

    if n_bins is not None:
        # IBSI: histogram over the full discretisation range [1, N_g].
        if min_val_i < 1 or max_val_i > n_bins:
            raise ValueError(
                f"discretised values must lie in [1, n_bins={n_bins}]; "
                f"got range [{min_val_i}, {max_val_i}]"
            )
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

    # 4.2.1 Mean discretised intensity (X6K6)
    mean_disc = float(np.mean(disc))
    features["mean_discretised_intensity_X6K6"] = float(mean_disc)

    # 4.2.2 Discretised intensity variance (CH89)
    var_disc = float(np.var(disc, ddof=0))
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
    # discretised median (WIFQ) is the conventional sample median. Integer values
    # get them from the histogram counts.
    if disc.dtype.kind in "iu":
        p10, p25, p75, p90, median_val = _counts_order_statistics(
            counts_full, hist_origin, disc.dtype
        )
    else:
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

    features["intensity_histogram_mean_absolute_deviation_D2ZX"] = float(
        _mean_abs_dev(disc, float(mean_disc))
    )

    features["intensity_histogram_robust_mean_absolute_deviation_WRZB"] = float(
        _robust_mean_abs_dev(disc, float(p10), float(p90))
    )

    features["intensity_histogram_median_absolute_deviation_4RNL"] = float(
        _mean_abs_dev(disc, float(median_val))
    )

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

    vals = np.asarray(discretised_values)
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
                    idx = np.arange(num_steps, dtype=np.float64)
                    candidates = g_min + (idx + 0.5) * bin_width
                else:
                    idx = np.arange(num_steps + 1, dtype=np.float64)
                    candidates = g_min + idx * bin_width
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


# From this many whole-number values on, the IVH counts the values in threads instead of
# sorting them. Measured: below, the sort is as fast (starting the threads costs about
# 80 us, and numpy's bincount is slow on runs of equal values).
_PARALLEL_COUNT_MIN = 1 << 17


def _value_counts(values: npt.NDArray[Any], low: int, size: int) -> npt.NDArray[np.int64]:
    """The number of times each value low + j occurs in the 1-D integer `values`, for j in
    range(size): parts in threads, each part into its own row (a row per thread only
    where `size` is small next to the values)."""
    data = values if values.dtype in (np.int32, np.int64) else values.astype(np.int64)
    parts = max(1, min(get_num_threads(), values.size // max(size, 1 << 14)))
    counts = np.zeros((parts, size), dtype=np.int64)
    _value_counts_numba(data, low, counts)
    return cast(npt.NDArray[np.int64], counts.sum(axis=0))


def _sorted_view(
    values: npt.NDArray[Any],
) -> tuple[Callable[[Any], Any], Callable[[int], Any], npt.NDArray[Any], npt.NDArray[np.intp]]:
    """What the IVH reads from the sorted 1-D `values`: count_below(t), the number of
    values below t; kth(k), the k-th smallest value (from 0); the values that occur,
    ascending; and the number of values below each of them. Many integer values over a
    range not much longer than the values take one count per value instead of a sort."""
    if (
        values.dtype.kind in "iu"
        and values.dtype != np.uint64
        and values.size >= _PARALLEL_COUNT_MIN
    ):
        low = min(int(values.min()), 0)  # counts from 0, or from a negative minimum
        if int(values.max()) - low < 2 * values.size + 65536:
            counts = _value_counts(values, low, int(values.max()) - low + 1)
            cum = np.cumsum(counts)  # the values at or below low + j
            occur = np.flatnonzero(counts)

            def count_below_int(t: Any) -> int:
                j = math.ceil(t) - low  # the integers below t are those below ceil(t)
                return 0 if j <= 0 else int(cum[min(j, cum.size) - 1])

            def kth_int(k: int) -> int:
                return low + int(np.searchsorted(cum, k, side="right"))

            return count_below_int, kth_int, occur + low, cum[occur] - counts[occur]
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


# The FFT sums use about 27 bytes per point of their grid (measured peak); 32 leaves a
# margin.
_FFT_BYTES_PER_POINT = 32
# Measured crossover: with fewer than 100 voxel pairs per FFT grid point, the pair loop
# is faster.
_FFT_MIN_PAIRS_PER_POINT = 100
# Measured pair loop speed: about 2.5 ns per voxel pair on one thread.
_PAIR_SECONDS_PER_THREAD = 2.5e-9


def _fft_memory_cap() -> int:
    """Memory limit of the FFT sums: 16 GB, but at most half of the physical memory.

    When the system does not report its memory (for example on Windows), the limit
    assumes 8 GB of memory.
    """
    try:
        physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        physical = 8 << 30
    return min(16 << 30, physical // 2)


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
    """
    # Distance in mm along each axis, with the offsets wrapped around the grid.
    axes = [
        np.minimum(np.arange(n), n - np.arange(n)) * s for n, s in zip(fshape, spacing, strict=True)
    ]
    kernel = (axes[0] ** 2)[:, None, None] + (axes[1] ** 2)[None, :, None] + (axes[2] ** 2)
    kernel[0, 0, 0] = 1.0  # a voxel has no weight to itself (set to 0 below)
    np.sqrt(kernel, out=kernel)
    np.divide(1.0, kernel, out=kernel)
    kernel[0, 0, 0] = 0.0
    workers = get_num_threads()
    kernel_f = scipy.fft.rfftn(kernel, workers=workers)
    del kernel
    box = tuple(slice(0, n) for n in roi.shape)

    def convolve(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """K * a at the ROI voxels."""
        spectrum = scipy.fft.rfftn(a, fshape, workers=workers)
        spectrum *= kernel_f
        full = scipy.fft.irfftn(spectrum, fshape, workers=workers, overwrite_x=True)
        return np.asarray(full[box][roi])

    weights = convolve(roi.astype(np.float64))
    d = diff[roi]
    numer_moran = float(np.sum(d * convolve(diff)))
    numer_geary = 2.0 * float(np.sum(d * d * weights)) - 2.0 * numer_moran
    return numer_moran, numer_geary, float(np.sum(weights))


def calculate_spatial_intensity_features(
    image: Image,
    mask: Image,
    *,
    enabled: bool = True,
) -> dict[str, float]:
    """
    Calculate spatial intensity features: Moran's I and Geary's C (IBSI 4.4).

    These features measure spatial autocorrelation of intensity values within the ROI.
    Small ROIs use a loop over all voxel pairs, whose time grows with N² (N = number of
    ROI voxels). Large ROIs use FFT convolutions: the same values to about 1e-14
    (relative) in much less time. The FFT needs about 32 bytes per point of a grid that
    is twice the ROI box along each axis. Above 16 GB, or above half of the memory, the
    pair loop runs instead, with a warning that gives the expected run time.

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
    intensities = np.ascontiguousarray(data[bbox][roi].astype(np.float64))

    N = len(intensities)
    if N < 2:
        features["morans_i_index_N365"] = np.nan
        features["gearys_c_measure_NPT7"] = np.nan
        return features

    mean_int = np.mean(intensities)
    denom = _sum_sq_centered(intensities, float(mean_int))

    # The FFT grid is at least 2 n - 1 along each axis of the box, so no pair wraps around.
    fshape = tuple(scipy.fft.next_fast_len(2 * n - 1, real=True) for n in roi.shape)
    grid = math.prod(fshape)
    use_fft = N * N >= _FFT_MIN_PAIRS_PER_POINT * grid
    if use_fft and _FFT_BYTES_PER_POINT * grid > (cap := _fft_memory_cap()):
        use_fft = False
        minutes = N * (N - 1) / 2 * _PAIR_SECONDS_PER_THREAD / get_num_threads() / 60
        warnings.warn(
            f"Moran's I and Geary's C: the FFT method needs about "
            f"{_FFT_BYTES_PER_POINT * grid / 2**30:.1f} GB, more than its limit of "
            f"{cap / 2**30:.1f} GB (16 GB, or half of the memory). The pair loop runs "
            f"instead. Expected run time: about {minutes:.1f} min.",
            stacklevel=2,
        )

    if use_fft:
        diff = np.zeros(roi.shape)
        diff[roi] = intensities - mean_int
        numer_moran, numer_geary, sum_weights = _spatial_sums_fft(roi, diff, (sx, sy, sz), fshape)
    else:
        # Full-image voxel indices, as the pair loop used before the crop.
        x_idx, y_idx, z_idx = (
            np.ascontiguousarray((idx + box.start).astype(np.int32))
            for idx, box in zip(np.nonzero(roi), bbox, strict=True)
        )
        numer_moran, numer_geary, sum_weights = _calculate_spatial_features_numba(
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
_TWO_STAGE_MIN_WORK = 1 << 24


def _local_peaks_two_stage(
    data: npt.NDArray[Any],
    mask_indices: npt.NDArray[np.int32],
    offsets: npt.NDArray[np.int32],
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
    peak). None (use the kernel on all voxels) when an intensity is not finite.
    """
    largest = max(abs(float(data.max())), abs(float(data.min())))
    if not math.isfinite(largest):
        return None
    rows: dict[tuple[int, int], int] = {}
    for dx, dy, dz in offsets.tolist():
        rows[(dx, dy)] = max(rows.get((dx, dy), 0), abs(dz))
    row_dx = np.array([dx for dx, _ in rows], dtype=np.int64)
    row_dy = np.array([dy for _, dy in rows], dtype=np.int64)
    row_dz = np.array(list(rows.values()), dtype=np.int64)
    prefix = np.empty(data.shape[:2] + (data.shape[2] + 1,), dtype=np.float64)
    _row_prefix_sums_numba(data, prefix)
    approx = _approximate_local_means_numba(prefix, mask_indices, row_dx, row_dy, row_dz)
    del prefix
    length = data.shape[2] + 1
    bound = 2.0**-52 * largest * (len(rows) * (2.0 * length * length + 4.0) + offsets.shape[0] ** 2)
    near_best = np.flatnonzero(approx >= approx.max() - 2.0 * bound)
    values = data[mask_indices[:, 0], mask_indices[:, 1], mask_indices[:, 2]]
    brightest = np.flatnonzero(values == values.max())
    selected = np.union1d(near_best, brightest)  # sorted
    exact = _calculate_local_mean_numba(data, np.ascontiguousarray(mask_indices[selected]), offsets)
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

    # Get ROI indices
    x_idx, y_idx, z_idx = np.where(mask_array != 0)
    if len(x_idx) == 0:
        return features

    mask_indices = np.ascontiguousarray(np.stack([x_idx, y_idx, z_idx], axis=1).astype(np.int32))

    peaks = None
    if mask_indices.shape[0] * offsets.shape[0] >= _TWO_STAGE_MIN_WORK:
        peaks = _local_peaks_two_stage(data, mask_indices, offsets)
    if peaks is None:
        # Calculate local means only for ROI voxels
        roi_means = _calculate_local_mean_numba(data, mask_indices, offsets)
        # Compute both peaks without allocating ROI intensity arrays.
        peaks = _calculate_local_peaks_numba(data, mask_indices, roi_means)
    features["global_intensity_peak_0F91"] = float(peaks[0])
    features["local_intensity_peak_VJGA"] = float(peaks[1])

    return features
