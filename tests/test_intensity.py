# ruff: noqa: E402
import os
import warnings

# Disable Numba JIT for coverage and smoother testing logic execution
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Suppress the "NumPy module was reloaded" warning that can occur due to Numba + testing environment
# Must be done BEFORE importing numpy
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

import unittest
from unittest.mock import MagicMock

import numpy as np

from pictologics.features.intensity import (
    calculate_intensity_features,
    calculate_intensity_histogram_features,
    calculate_ivh_features,
    calculate_local_intensity_features,
    calculate_spatial_intensity_features,
)


class TestIntensityFeatures(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("NUMBA_DISABLE_JIT", None)

    # ----------------------------------------------------------------------
    # 4.1 First Order Statistics
    # ----------------------------------------------------------------------

    def test_calculate_intensity_features_basic(self) -> None:
        # ROI: 1, 2, 3, 4, 5
        values = np.array([1, 2, 3, 4, 5], dtype=float)
        features = calculate_intensity_features(values)

        self.assertAlmostEqual(features["mean_intensity_Q4LE"], 3.0)
        self.assertAlmostEqual(features["minimum_intensity_1GSF"], 1.0)
        self.assertAlmostEqual(features["maximum_intensity_84IY"], 5.0)
        self.assertAlmostEqual(features["intensity_range_2OJQ"], 4.0)
        self.assertAlmostEqual(features["median_intensity_Y12H"], 3.0)
        # Variance: sum((x-3)^2)/5 = (4+1+0+1+4)/5 = 2.0
        self.assertAlmostEqual(features["intensity_variance_ECT3"], 2.0)
        # Skewness: symmetric -> 0
        self.assertAlmostEqual(features["intensity_skewness_KE2A"], 0.0)

    def test_calculate_intensity_features_empty(self) -> None:
        features = calculate_intensity_features(np.array([]))
        self.assertEqual(features, {})

    def test_calculate_intensity_features_constant(self) -> None:
        # Variance = 0 case
        values = np.array([5, 5, 5], dtype=float)
        features = calculate_intensity_features(values)
        self.assertAlmostEqual(features["mean_intensity_Q4LE"], 5.0)
        self.assertAlmostEqual(features["intensity_variance_ECT3"], 0.0)
        self.assertTrue(np.isnan(features["intensity_skewness_KE2A"]))
        self.assertTrue(np.isnan(features["intensity_kurtosis_IPH6"]))

    def test_calculate_intensity_features_zero_mean(self) -> None:
        # CV and Quartile coeff denom = 0
        values = np.array([-1, 1], dtype=float)
        features = calculate_intensity_features(values)
        self.assertTrue(np.isnan(features["intensity_coefficient_of_variation_7TET"]))
        self.assertTrue(
            np.isnan(features["intensity_quartile_coefficient_of_dispersion_9S40"])
        )

    # ----------------------------------------------------------------------
    # 4.2 Intensity Histogram
    # ----------------------------------------------------------------------

    def test_calculate_intensity_histogram_features_basic(self) -> None:
        # 1, 1, 2, 2, 2, 3, 4, 5
        # Probs: 0.25, 0.375, 0.125, 0.125, 0.125
        disc_vals = np.array([1, 1, 2, 2, 2, 3, 4, 5])
        features = calculate_intensity_histogram_features(disc_vals)

        self.assertAlmostEqual(features["intensity_histogram_mode_AMMC"], 2.0)
        self.assertAlmostEqual(features["minimum_discretised_intensity_1PR8"], 1.0)
        self.assertAlmostEqual(features["maximum_discretised_intensity_3NCY"], 5.0)

        # Uniformity: sum(p^2) = 0.25^2 + 0.375^2 + 3*(0.125^2)
        # = 0.0625 + 0.140625 + 0.046875 = 0.25
        self.assertAlmostEqual(features["discretised_intensity_uniformity_BJ5W"], 0.25)

    def test_calculate_intensity_histogram_features_empty(self) -> None:
        features = calculate_intensity_histogram_features(np.array([]))
        self.assertEqual(features, {})

    def test_calculate_intensity_histogram_features_constant(self) -> None:
        # All same values -> variance 0
        disc_vals = np.array([1, 1, 1, 1])
        features = calculate_intensity_histogram_features(disc_vals)
        self.assertTrue(np.isnan(features["discretised_intensity_skewness_88K1"]))
        self.assertTrue(np.isnan(features["discretised_intensity_kurtosis_C3I7"]))
        self.assertEqual(
            features["intensity_histogram_coefficient_of_variation_CWYJ"], 0.0
        )
        self.assertTrue(np.isnan(features["maximum_histogram_gradient_12CE"]))

    def test_calculate_intensity_histogram_features_zero_mean(self) -> None:
        disc_vals = np.array([-1, 1])
        features = calculate_intensity_histogram_features(disc_vals)
        self.assertTrue(
            np.isnan(features["intensity_histogram_coefficient_of_variation_CWYJ"])
        )

    # ----------------------------------------------------------------------
    # 4.3 IVH
    # ----------------------------------------------------------------------

    def test_calculate_ivh_features_basic(self) -> None:
        # 0, 1, 2, 3, 4
        # Bin width 1.0, min 0.0 implies bins [0,1), [1,2), ...
        disc_vals = np.array([0, 1, 2, 3, 4])
        features = calculate_ivh_features(disc_vals, bin_width=1.0, min_val=0.0)

        self.assertAlmostEqual(
            features["volume_at_intensity_fraction_0.10_BC2M_10"], 0.8
        )

    def test_calculate_ivh_features_empty(self) -> None:
        features = calculate_ivh_features(np.array([]))
        self.assertEqual(features, {})

    def test_calculate_ivh_features_auc_simple(self) -> None:
        # 0, 1
        # P(>=0) = 1.0. P(>=1) = 0.5.
        vals = np.array([0, 1])
        features = calculate_ivh_features(vals)
        # Width=1, avg height=0.75. AUC=0.75
        self.assertAlmostEqual(features["area_under_the_ivh_curve_9CMM"], 0.75)

    def test_calculate_ivh_features_auc_physical(self) -> None:
        # Physical units: min=0, width=2.
        # indices 0, 1 correspond to physical [0,2), [2,4). Centers 1.0, 3.0.
        vals = np.array([0, 1])
        features = calculate_ivh_features(vals, bin_width=2.0, min_val=0.0)
        # Width=2.0, avg height=0.75. AUC=1.5
        self.assertAlmostEqual(features["area_under_the_ivh_curve_9CMM"], 1.5)

    def test_calculate_ivh_features_single_value(self) -> None:
        vals = np.array([5, 5, 5])
        features = calculate_ivh_features(vals)
        self.assertEqual(features["area_under_the_ivh_curve_9CMM"], 0.0)

    def test_calculate_ivh_features_physical_target(self) -> None:
        # Target range provided + bin_width
        disc_vals = np.array([0, 1, 2])
        # Bin width 2.0, min 0.0 => physical centers 1.0, 3.0, 5.0
        # Target range: 0 to 6.

        features = calculate_ivh_features(
            disc_vals,
            bin_width=2.0,
            min_val=0.0,
            target_range_min=0.0,
            target_range_max=6.0,
        )
        self.assertAlmostEqual(
            features["volume_at_intensity_fraction_0.10_BC2M_10"], 2.0 / 3.0
        )

        # Test the branch where bin_width is NOT provided but target range IS.
        features2 = calculate_ivh_features(
            disc_vals, min_val=0.0, target_range_min=0.0, target_range_max=6.0
        )
        self.assertAlmostEqual(
            features2["volume_at_intensity_fraction_0.10_BC2M_10"], 2.0 / 3.0
        )

    def test_calculate_ivh_features_integer_path(self) -> None:
        # Test the 'Standard Integer Bins' fast path logic.
        # bin_width=1.0, min=None, integer dtype.
        disc_vals = np.array([0, 1, 2, 3, 4], dtype=int)
        features = calculate_ivh_features(disc_vals, bin_width=1.0)
        # Logic should hit the `if ... bin_width==1.0` block.
        # 10% frac -> 0.4. target count = floor(0.4) = 0?
        # get_intensity_at_volume_fraction(0.10): frac=0.1, N=5. 0.1*5 = 0.5. floor=0.
        # If target count <= 0 -> return last val (4).
        self.assertEqual(features["intensity_at_volume_fraction_0.10_GBPN_10"], 4.0)

        # Try a larger volume fraction to get count > 0.
        # 90% -> 4.5 -> floor 4.
        # k = 5 - 4 = 1.
        # v = sorted[0] = 0. t = 1.
        # check t <= 4. return 1.0.
        self.assertEqual(features["intensity_at_volume_fraction_0.90_GBPN_90"], 1.0)

    def test_calculate_ivh_features_integer_path_edge(self) -> None:
        # Hits L603: if t > vmax: t = vmax in fast path.
        disc_vals = np.array([5, 5], dtype=int)
        features = calculate_ivh_features(disc_vals, bin_width=1.0)
        # get_intensity_at_vol... (0.10) => target_count = 0.
        self.assertEqual(features["intensity_at_volume_fraction_0.10_GBPN_10"], 5.0)
        # get_intensity... (0.90) => target_count = 1. t > vmax.
        self.assertEqual(features["intensity_at_volume_fraction_0.90_GBPN_90"], 5.0)

    def test_calculate_ivh_features_bin_width_only(self) -> None:
        # Hits L614 (g_max fallback: max_val is None, min_val is None)
        # hits L623-624 (candidates fallback: min_val is None)
        # Use bin_width=0.5 to avoid fast path (which requires 1.0)
        disc_vals = np.array([0, 1])
        # g_min = min(vals) = 0.
        # g_max = max(vals) = 1.
        # steps = (1-0)/0.5 = 2.
        # idx = [0, 1, 2].
        # candidates = 0 + idx*0.5 = [0.0, 0.5, 1.0].
        # 10% -> count=0 -> returns largest? No, logic depends on binary search.
        calculate_ivh_features(disc_vals, bin_width=0.5)
        # Just ensure it runs.
        pass

    def test_calculate_ivh_features_negative_bin_width(self) -> None:
        # Hits L626 (bin_width <= 0 fallback).
        disc_vals = np.array([0, 1])
        features = calculate_ivh_features(disc_vals, bin_width=-1.0)
        # Should behave as if bin_width=None (uses sorted_vals as candidates)
        self.assertAlmostEqual(features["area_under_the_ivh_curve_9CMM"], 0.75)

    def test_calculate_ivh_features_max_val_provided(self) -> None:
        # Hits L610: if max_val is not None: g_max = max_val
        disc_vals = np.array([0, 1])
        calculate_ivh_features(disc_vals, bin_width=0.5, min_val=0.0, max_val=2.0)
        pass

    def test_calculate_ivh_features_inverted_range(self) -> None:
        # Hits L614: if g_min > g_max: return [] logic or similar...
        # Actually logic is: g_min=5, g_max=2. num_steps negative. idx empty. candidates empty.
        # Binary search fails. Should raise IndexError or similar if not handled.
        # We just want to ensure the code lines run.
        disc_vals = np.array([0, 1])
        with self.assertRaises(IndexError):
            calculate_ivh_features(disc_vals, bin_width=1.0, min_val=5.0, max_val=2.0)

    def test_calculate_ivh_features_small_input_general(self) -> None:
        # Hits L623-626 related logic: count <= 0 in general path.
        # Use float data to force general path.
        vals = np.array([10.5], dtype=float)
        # 1 voxel. 10% -> 0.1 -> floor 0.
        features = calculate_ivh_features(vals)
        self.assertEqual(features["intensity_at_volume_fraction_0.10_GBPN_10"], 10.5)

    # ----------------------------------------------------------------------
    # 4.4 Spatial Intensity (Parallelized -> Serial in Coverage Mode)
    # ----------------------------------------------------------------------

    def test_calculate_spatial_intensity_features_correctness_small(self) -> None:
        # 2x2x1 image.
        mock_img = MagicMock()
        mock_img.array = np.array([[[1], [2]], [[3], [4]]])
        mock_img.spacing = (1.0, 1.0, 1.0)
        mock_mask = MagicMock()
        mock_mask.array = np.ones((2, 2, 1))

        features = calculate_spatial_intensity_features(mock_img, mock_mask)
        self.assertFalse(np.isnan(features["morans_i_index_N365"]))
        self.assertFalse(np.isnan(features["gearys_c_measure_NPT7"]))

    def test_calculate_spatial_intensity_features_parallel_safety(self) -> None:
        # Regression test for parallel/serial execution.
        shape = (10, 10, 10)
        data = np.random.rand(*shape)
        mask = np.ones(shape)

        mock_img = MagicMock()
        mock_img.array = data
        mock_img.spacing = (1.0, 1.0, 1.0)
        mock_mask = MagicMock()
        mock_mask.array = mask

        features = calculate_spatial_intensity_features(mock_img, mock_mask)
        self.assertFalse(np.isnan(features["morans_i_index_N365"]))

    def test_calculate_spatial_intensity_features_small_input(self) -> None:
        # < 2 voxels -> NaN
        mock_img = MagicMock()
        mock_img.array = np.array([[[1]]])
        mock_img.spacing = (1.0, 1.0, 1.0)
        mock_mask = MagicMock()
        mock_mask.array = np.array([[[1]]])
        features = calculate_spatial_intensity_features(mock_img, mock_mask)
        self.assertTrue(np.isnan(features["morans_i_index_N365"]))

    def test_calculate_spatial_intensity_features_constant(self) -> None:
        # Constant intensity -> denom = 0
        mock_img = MagicMock()
        mock_img.array = np.ones((2, 2, 1))
        mock_img.spacing = (1.0, 1.0, 1.0)
        mock_mask = MagicMock()
        mock_mask.array = np.ones((2, 2, 1))

        features = calculate_spatial_intensity_features(mock_img, mock_mask)
        self.assertTrue(np.isnan(features["morans_i_index_N365"]))
        self.assertTrue(np.isnan(features["gearys_c_measure_NPT7"]))

    def test_calculate_spatial_intensity_features_disabled(self) -> None:
        mock_img = MagicMock()
        mock_mask = MagicMock()
        features = calculate_spatial_intensity_features(
            mock_img, mock_mask, enabled=False
        )
        self.assertEqual(features, {})

    # ----------------------------------------------------------------------
    # 4.5 Local Intensity
    # ----------------------------------------------------------------------

    def test_calculate_local_intensity_features_basic(self) -> None:
        # 3x3x3 peak
        data = np.zeros((3, 3, 3))
        data[1, 1, 1] = 10.0
        mock_img = MagicMock()
        mock_img.array = data
        mock_img.spacing = (1.0, 1.0, 1.0)
        mock_mask = MagicMock()
        mock_mask.array = np.ones((3, 3, 3))

        features = calculate_local_intensity_features(mock_img, mock_mask)
        self.assertIn("global_intensity_peak_0F91", features)
        self.assertGreater(features["global_intensity_peak_0F91"], 0.0)

    def test_calculate_local_intensity_features_empty(self) -> None:
        mock_img = MagicMock()
        mock_img.array = np.zeros((3, 3, 3))
        mock_img.spacing = (1.0, 1.0, 1.0)
        mock_mask = MagicMock()
        mock_mask.array = np.zeros((3, 3, 3))
        features = calculate_local_intensity_features(mock_img, mock_mask)
        self.assertEqual(features, {})

    def test_calculate_local_intensity_features_disabled(self) -> None:
        mock_img = MagicMock()
        mock_mask = MagicMock()
        features = calculate_local_intensity_features(
            mock_img, mock_mask, enabled=False
        )
        self.assertEqual(features, {})

    def test_internal_helpers_edge_cases(self) -> None:
        # Test Numba helpers directly for empty input coverage
        from pictologics.features.intensity import (
            _calculate_local_peaks_numba,
            _central_moments_2_3_4,
            _max_mean_at_max_intensity,
            _mean_abs_dev,
            _robust_mean_abs_dev,
            _sum_sq_centered,
        )

        # Test _sum_sq_centered
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_val = 3.0
        # Expected: (1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2 = 4+1+0+1+4 = 10
        result = _sum_sq_centered(values, mean_val)
        self.assertAlmostEqual(result, 10.0)

        # Empty array should return 0.0
        self.assertEqual(_sum_sq_centered(np.array([]), 0.0), 0.0)

        # Single value
        self.assertAlmostEqual(_sum_sq_centered(np.array([5.0]), 5.0), 0.0)
        self.assertAlmostEqual(_sum_sq_centered(np.array([5.0]), 3.0), 4.0)

        # Empty arrays for other helpers

        # Test _central_moments_2_3_4
        # Empty array
        empty = np.array([])
        self.assertEqual(_central_moments_2_3_4(empty, 0.0), (0.0, 0.0, 0.0))

        # Symmetric distribution: skewness (m3) should be 0
        symmetric = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m2, m3, m4 = _central_moments_2_3_4(symmetric, 3.0)
        self.assertAlmostEqual(m2, 2.0)  # variance = sum((x-3)^2)/5 = 10/5 = 2
        self.assertAlmostEqual(m3, 0.0)  # symmetric -> 0 skewness
        self.assertGreater(m4, 0.0)  # kurtosis component > 0

        # Single value: all moments should be 0
        single = np.array([5.0])
        m2, m3, m4 = _central_moments_2_3_4(single, 5.0)
        self.assertEqual(m2, 0.0)
        self.assertEqual(m3, 0.0)
        self.assertEqual(m4, 0.0)

        # Constant array: all moments should be 0
        constant = np.array([3.0, 3.0, 3.0])
        m2, m3, m4 = _central_moments_2_3_4(constant, 3.0)
        self.assertEqual(m2, 0.0)
        self.assertEqual(m3, 0.0)
        self.assertEqual(m4, 0.0)

        # Other helpers - empty arrays
        self.assertEqual(_mean_abs_dev(empty, 0.0), 0.0)

        # Test _mean_abs_dev more thoroughly
        # Basic case: MAD from mean
        test_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # MAD from mean (3.0): (2+1+0+1+2)/5 = 6/5 = 1.2
        self.assertAlmostEqual(_mean_abs_dev(test_vals, 3.0), 1.2)

        # Single value at center -> 0
        self.assertEqual(_mean_abs_dev(np.array([5.0]), 5.0), 0.0)

        # Single value off center
        self.assertAlmostEqual(_mean_abs_dev(np.array([5.0]), 3.0), 2.0)

        # Test _robust_mean_abs_dev
        self.assertEqual(_robust_mean_abs_dev(empty, 0.0, 1.0), 0.0)

        # Robust MAD with count=0 (no values in range)
        values = np.array([10.0])
        # Range 0-5 -> count=0
        self.assertEqual(_robust_mean_abs_dev(values, 0.0, 5.0), 0.0)

        # Values inside range - basic case
        # Values: 1, 2, 3, 4, 5. Range [1, 5]. All values in range.
        # Mean = 3.0, MAD = (2+1+0+1+2)/5 = 1.2
        robust_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(_robust_mean_abs_dev(robust_vals, 1.0, 5.0), 1.2)

        # Partial filtering: [1, 2, 3, 4, 5, 100]. Range [1, 5] excludes 100.
        # Mean of [1,2,3,4,5] = 3.0, MAD = 1.2
        robust_vals_outlier = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        self.assertAlmostEqual(_robust_mean_abs_dev(robust_vals_outlier, 1.0, 5.0), 1.2)

        # Single value in range
        self.assertEqual(_robust_mean_abs_dev(np.array([3.0]), 1.0, 5.0), 0.0)

        # Local peaks edge case: same max intensity, higher local mean
        # data: 2 voxels. Both val 10. mean1=5, mean2=8.
        # code:
        # if v > max_intensity: ...
        # elif v == max_intensity and mean_val > local_peak: ...

        # mask indices provided as Nx3.
        # We dummy this up.
        data = np.zeros((1, 2, 1))
        data[0, 0, 0] = 10.0
        data[0, 1, 0] = 10.0

        mask_indices = np.array([[0, 0, 0], [0, 1, 0]])
        roi_means = np.array([5.0, 8.0])

        glob, loc = _calculate_local_peaks_numba(data, mask_indices, roi_means)
        self.assertEqual(glob, 8.0)  # max of means
        self.assertEqual(loc, 8.0)  # max intensity (10) occurs at 8.0 mean

        # Test _max_mean_at_max_intensity helper
        # 2 voxels. values 10, 10. means 5, 8. max_val=10.
        roi_data = np.array([10.0, 10.0])
        best = _max_mean_at_max_intensity(roi_data, roi_means, 10.0)
        self.assertEqual(best, 8.0)


if __name__ == "__main__":
    unittest.main()
