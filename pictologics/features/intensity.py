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

from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import numpy as np
from numba import jit, prange

if TYPE_CHECKING:
    from ..loader import Image


@jit(nopython=True, fastmath=True, cache=True)  # type: ignore
def _sum_sq_centered(values: np.ndarray, mean_val: float) -> float:
    total = 0.0
    for i in range(values.size):
        d = float(values[i]) - mean_val
        total += d * d
    return total


@jit(nopython=True, fastmath=True, cache=True)  # type: ignore
def _central_moments_2_3_4(
    values: np.ndarray, mean_val: float
) -> tuple[float, float, float]:
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
def _mean_abs_dev(values: np.ndarray, center: float) -> float:
    n = values.size
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        total += abs(float(values[i]) - center)
    return float(total / n)


@jit(nopython=True, fastmath=True, cache=True)  # type: ignore
def _robust_mean_abs_dev(values: np.ndarray, lower: float, upper: float) -> float:
    # Two-pass calculation without allocating a filtered array.
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
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    z_idx: np.ndarray,
    intensities: np.ndarray,
    mean_int: float,
    sx: float,
    sy: float,
    sz: float,
) -> tuple[float, float, float, float]:
    """
    Calculate Moran's I and Geary's C components using Numba with Parallelization.

    This feature is O(N^2) complexity where N is the number of ROI voxels.
    Parallel execution significantly speeds up the outer loop.

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
        - numer_moran: Numerator for Moran's I.
        - numer_geary_term1: First term for Geary's C numerator.
        - numer_geary_term2: Second term for Geary's C numerator.
        - sum_weights: Sum of all weights (inverse distances).
    """
    n = intensities.size

    # Reduction arrays to avoid Numba parallel reduction cycle issues
    # Allocate arrays to store partial results for each voxel
    local_moran_arr = np.zeros(n, dtype=np.float64)
    local_geary_1_arr = np.zeros(n, dtype=np.float64)
    local_geary_2_arr = np.zeros(n, dtype=np.float64)
    local_w_sum_arr = np.zeros(n, dtype=np.float64)

    # Parallelize the outer loop
    for i in prange(n):
        local_moran = 0.0
        local_geary_2 = 0.0
        local_w_sum = 0.0

        val_i = float(intensities[i])
        diff_i = val_i - mean_int

        xi = float(x_idx[i])
        yi = float(y_idx[i])
        zi = float(z_idx[i])

        # Inner loop runs sequentially
        for j in range(n):
            if i == j:
                continue

            dx = (xi - float(x_idx[j])) * sx
            dy = (yi - float(y_idx[j])) * sy
            dz = (zi - float(z_idx[j])) * sz
            d_sq = dx * dx + dy * dy + dz * dz

            if d_sq > 0.0:
                w = 1.0 / np.sqrt(d_sq)
                val_j = float(intensities[j])
                diff_j = val_j - mean_int

                local_w_sum += w
                local_moran += w * diff_i * diff_j
                local_geary_2 += w * val_i * val_j

        # Store in arrays
        local_moran_arr[i] = local_moran
        local_geary_1_arr[i] = (val_i * val_i) * local_w_sum
        local_geary_2_arr[i] = local_geary_2
        local_w_sum_arr[i] = local_w_sum

    # Sum up results
    numer_moran = np.sum(local_moran_arr)
    numer_geary_term1 = np.sum(local_geary_1_arr)
    numer_geary_term2 = np.sum(local_geary_2_arr)
    sum_weights = np.sum(local_w_sum_arr)

    return numer_moran, numer_geary_term1, numer_geary_term2, sum_weights


@jit(nopython=True, parallel=True, fastmath=True, cache=True)  # type: ignore
def _calculate_local_mean_numba(
    data: np.ndarray,
    mask_indices: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """
    Calculate local mean intensity for voxels in the mask using parallel execution.
    """
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


@jit(nopython=True, fastmath=True, cache=True)  # type: ignore
def _calculate_local_peaks_numba(
    data: np.ndarray,
    mask_indices: np.ndarray,
    roi_means: np.ndarray,
) -> tuple[float, float]:
    """Compute global/local intensity peaks without allocating ROI intensity arrays."""
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


@jit(nopython=True, fastmath=True, cache=True)  # type: ignore
def _max_mean_at_max_intensity(
    roi_data: np.ndarray, roi_means: np.ndarray, max_val: float
) -> float:
    best = -1.0e308
    for i in range(roi_data.size):
        if float(roi_data[i]) == max_val:
            m = float(roi_means[i])
            if m > best:
                best = m
    return best


@lru_cache(maxsize=32)
def _sphere_offsets_for_radius(
    spacing: tuple[float, float, float], radius_mm: float
) -> np.ndarray:
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


def calculate_intensity_features(values: np.ndarray) -> dict[str, float]:
    """
    Calculate intensity-based features (First Order Statistics) as defined in IBSI.
    """
    if len(values) == 0:
        return {}

    features: dict[str, float] = {}

    # 4.1.1 Mean intensity (Q4LE)
    mean_val = np.mean(values)
    features["mean_intensity_Q4LE"] = float(mean_val)

    # 4.1.2 Intensity variance (ECT3)
    var_val = float(np.var(values, ddof=0))
    features["intensity_variance_ECT3"] = float(var_val)

    # 4.1.3 Intensity skewness (KE2A)
    if var_val == 0.0:
        features["intensity_skewness_KE2A"] = np.nan
        features["intensity_kurtosis_IPH6"] = np.nan
    else:
        m2, m3, m4 = _central_moments_2_3_4(values, float(mean_val))
        denom = m2**1.5
        if denom != 0.0:
            features["intensity_skewness_KE2A"] = float(m3 / denom)
            features["intensity_kurtosis_IPH6"] = float((m4 / (m2 * m2)) - 3.0)

    # 4.1.5 Median intensity (Y12H)
    median_val = np.median(values)
    features["median_intensity_Y12H"] = float(median_val)

    # 4.1.6 Minimum intensity (1GSF)
    min_val = np.min(values)
    features["minimum_intensity_1GSF"] = float(min_val)

    p10, p25, p75, p90 = np.percentile(values, [10, 25, 75, 90])
    features["10th_intensity_percentile_QG58"] = float(p10)
    features["90th_intensity_percentile_8DWT"] = float(p90)

    # 4.1.9 Maximum intensity (84IY)
    max_val = np.max(values)
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
        features["intensity_coefficient_of_variation_7TET"] = float(
            np.sqrt(var_val) / mean_val
        )
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
    discretised_values: np.ndarray,
) -> dict[str, float]:
    """Calculate intensity histogram features as defined in IBSI 4.2."""
    if len(discretised_values) == 0:
        return {}

    features: dict[str, float] = {}

    disc = np.asarray(discretised_values)
    n = disc.size

    # Support negative values by shifting for bincount compatibility
    min_val_i = int(np.min(disc))
    max_val_i = int(np.max(disc))

    shifted = disc.astype(np.int64) - min_val_i
    counts_full = np.bincount(shifted, minlength=(max_val_i - min_val_i + 1))
    total = float(n)
    p = counts_full[counts_full > 0].astype(np.float64) / total

    # 4.2.1 Mean discretised intensity (X6K6)
    mean_disc = float(np.mean(disc))
    features["mean_discretised_intensity_X6K6"] = float(mean_disc)

    # 4.2.2 Discretised intensity variance (CH89)
    var_disc = float(np.var(disc, ddof=0))
    features["discretised_intensity_variance_CH89"] = float(var_disc)

    # 4.2.3 Discretised intensity skewness (88K1)
    if var_disc == 0.0:
        features["discretised_intensity_skewness_88K1"] = np.nan
        features["discretised_intensity_kurtosis_C3I7"] = np.nan
    else:
        m2, m3, m4 = _central_moments_2_3_4(disc, float(mean_disc))
        denom = m2**1.5
        if denom != 0.0:
            features["discretised_intensity_skewness_88K1"] = float(m3 / denom)
            features["discretised_intensity_kurtosis_C3I7"] = float(
                (m4 / (m2 * m2)) - 3.0
            )

    p10, p25, median_val, p75, p90 = np.percentile(disc, [10, 25, 50, 75, 90])

    features["median_discretised_intensity_WIFQ"] = float(median_val)
    features["minimum_discretised_intensity_1PR8"] = float(min_val_i)
    features["10th_discretised_intensity_percentile_1PR"] = float(p10)
    features["90th_discretised_intensity_percentile_GPMT"] = float(p90)
    features["maximum_discretised_intensity_3NCY"] = float(max_val_i)

    mode_index = int(np.argmax(counts_full))
    features["intensity_histogram_mode_AMMC"] = float(min_val_i + mode_index)

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
        features["maximum_histogram_gradient_intensity_8E6O"] = float(
            min_val_i + max_grad_idx
        )
        features["minimum_histogram_gradient_VQB3"] = float(np.min(gradient))
        min_grad_idx = int(np.argmin(gradient))
        features["minimum_histogram_gradient_intensity_RHQZ"] = float(
            min_val_i + min_grad_idx
        )

    return features


def calculate_ivh_features(
    discretised_values: np.ndarray,
    bin_width: Optional[float] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    target_range_min: Optional[float] = None,
    target_range_max: Optional[float] = None,
) -> dict[str, float]:
    """
    Calculate Intensity-Volume Histogram (IVH) features.

    Includes Volume Fraction difference and Area Under the IVH Curve (AUC).
    """
    if len(discretised_values) == 0:
        return {}

    features: dict[str, float] = {}
    N = len(discretised_values)

    vals = np.asarray(discretised_values)
    sorted_vals = np.sort(vals)

    # -------------------------------------------------------------------------
    # 1. Volume Fractions
    # -------------------------------------------------------------------------
    t_min = target_range_min if target_range_min is not None else min_val
    t_max = target_range_max if target_range_max is not None else max_val

    # Fallback to data min/max if still None
    if t_min is None or t_max is None:
        t_min_idx = float(sorted_vals[0])
        t_max_idx = float(sorted_vals[-1])
        val_range_idx = t_max_idx - t_min_idx

        def get_volume_fraction_at_intensity_fraction_indices(frac: float) -> float:
            threshold_idx = t_min_idx + frac * val_range_idx
            idx = int(np.searchsorted(sorted_vals, threshold_idx, side="left"))
            count = N - idx
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

            idx = int(np.searchsorted(sorted_vals, threshold_idx, side="left"))
            count = N - idx
            return float(count / N)

        features["volume_at_intensity_fraction_0.10_BC2M_10"] = (
            get_volume_fraction_at_intensity_fraction_physical(0.10)
        )
        features["volume_at_intensity_fraction_0.90_BC2M_90"] = (
            get_volume_fraction_at_intensity_fraction_physical(0.90)
        )

    features[
        "volume_fraction_difference_between_intensity_0.10_and_0.90_fractions_DDTU"
    ] = float(
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
            and np.issubdtype(sorted_vals.dtype, np.integer)
        ):
            target_count = int(np.floor(vol_frac * N))
            if target_count <= 0:
                return float(sorted_vals[-1])

            # Smallest integer threshold t such that count(vals >= t) <= target_count.
            # Let k = N - target_count. We need searchsorted(t) >= k.
            k = N - target_count
            v = int(sorted_vals[k - 1])
            t = v + 1
            vmax = int(sorted_vals[-1])
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
                num_steps = int(np.round((g_max - g_min) / bin_width))
                if min_val is not None:
                    # Candidates are bin centers
                    idx = np.arange(num_steps, dtype=np.float64)
                    candidates = g_min + (idx + 0.5) * bin_width
                else:
                    idx = np.arange(num_steps + 1, dtype=np.float64)
                    candidates = g_min + idx * bin_width
            else:
                candidates = sorted_vals.astype(np.float64)
        else:
            candidates = sorted_vals

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

            idx = np.searchsorted(sorted_vals, check_val, side="left")
            count = N - idx

            if count <= target_count:
                ans_idx = mid
                high = mid - 1
            else:
                low = mid + 1

        if ans_idx != -1:
            return float(candidates[ans_idx])
        else:
            return float(candidates[-1])

    features["intensity_at_volume_fraction_0.10_GBPN_10"] = (
        get_intensity_at_volume_fraction(0.10)
    )
    features["intensity_at_volume_fraction_0.90_GBPN_90"] = (
        get_intensity_at_volume_fraction(0.90)
    )

    features[
        "intensity_fraction_difference_between_volume_0.10_and_0.90_fractions_CNV2"
    ] = float(
        features["intensity_at_volume_fraction_0.10_GBPN_10"]
        - features["intensity_at_volume_fraction_0.90_GBPN_90"]
    )

    # -------------------------------------------------------------------------
    # 3. Area Under the IVH Curve (AUC)
    # -------------------------------------------------------------------------
    # IVH Curve: Volume Fraction (phi) vs Intensity (I)
    # We construct the curve points from the unique values in the data.
    unique_vals = np.unique(sorted_vals)
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
            intensities_arr = (
                min_val + (unique_vals.astype(np.float64) - 0.5) * bin_width
            )
        else:
            intensities_arr = unique_vals.astype(np.float64)

        # Calculate volume fractions
        # searchsorted returns first index where val fits.
        # Since sorted_vals is sorted, all elements >= val are from searchsorted(val) onwards.
        indices = np.searchsorted(sorted_vals, unique_vals, side="left")
        counts = N - indices
        fractions = counts.astype(np.float64) / float(N)

        # Riemann Sum (Trapezoidal)
        # Integrate fraction(I) over I.
        auc = 0.0
        for k in range(1, len(intensities_arr)):
            i_curr = intensities_arr[k]
            i_prev = intensities_arr[k - 1]
            phi_curr = fractions[k]
            phi_prev = fractions[k - 1]

            # Trapezoid area
            width = i_curr - i_prev
            avg_height = (phi_curr + phi_prev) * 0.5
            auc += width * avg_height

        features["area_under_the_ivh_curve_9CMM"] = float(auc)

    return features


def calculate_spatial_intensity_features(
    image: Image,
    mask: Image,
    *,
    enabled: bool = True,
) -> dict[str, float]:
    """
    Calculate spatial intensity features: Moran's I and Geary's C (IBSI 4.4).
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

    # Get ROI indices (X, Y, Z)
    x_idx, y_idx, z_idx = np.where(mask_array > 0)

    if len(x_idx) < 2:
        features["morans_i_index_N365"] = np.nan
        features["gearys_c_measure_NPT7"] = np.nan
        return features

    xi = np.ascontiguousarray(x_idx.astype(np.int32))
    yi = np.ascontiguousarray(y_idx.astype(np.int32))
    zi = np.ascontiguousarray(z_idx.astype(np.int32))

    intensities = np.ascontiguousarray(data[mask_array > 0].astype(np.float64))

    N = len(intensities)
    mean_int = np.mean(intensities)

    # Calculate terms using Parallelized Numba Function
    numer_moran, numer_geary_1, numer_geary_2, sum_weights = (
        _calculate_spatial_features_numba(
            xi, yi, zi, intensities, float(mean_int), sx, sy, sz
        )
    )

    # Moran's I - N365
    denom = _sum_sq_centered(intensities, float(mean_int))

    if denom != 0 and sum_weights != 0:
        moran_i = (N / sum_weights) * (numer_moran / denom)
        features["morans_i_index_N365"] = float(moran_i)
    else:
        features["morans_i_index_N365"] = np.nan

    # Geary's C - NPT7
    if denom != 0 and sum_weights != 0:
        numer = 2 * numer_geary_1 - 2 * numer_geary_2
        geary_c = ((N - 1) / (2 * sum_weights)) * (numer / denom)
        features["gearys_c_measure_NPT7"] = float(geary_c)
    else:
        features["gearys_c_measure_NPT7"] = np.nan

    return features


def calculate_local_intensity_features(
    image: Image,
    mask: Image,
    *,
    enabled: bool = True,
) -> dict[str, float]:
    """
    Calculate local intensity features: Local and Global Intensity Peak (IBSI 4.5).
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
    radius_mm = 6.2035

    # Get ROI indices
    x_idx, y_idx, z_idx = np.where(mask_array > 0)
    if len(x_idx) == 0:
        return features

    mask_indices = np.ascontiguousarray(
        np.stack([x_idx, y_idx, z_idx], axis=1).astype(np.int32)
    )
    offsets = _sphere_offsets_for_radius(spacing_tuple, radius_mm)

    # Calculate local means only for ROI voxels
    roi_means = _calculate_local_mean_numba(data, mask_indices, offsets)

    # Compute both peaks without allocating ROI intensity arrays.
    global_peak, local_peak = _calculate_local_peaks_numba(
        data, mask_indices, roi_means
    )
    features["global_intensity_peak_0F91"] = float(global_peak)
    features["local_intensity_peak_VJGA"] = float(local_peak)

    return features
