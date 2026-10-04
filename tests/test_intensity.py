# ruff: noqa: E402
import os
import warnings

# Disable Numba JIT for coverage and smoother testing logic execution
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Suppress the "NumPy module was reloaded" warning that can occur due to Numba + testing environment
# Must be done BEFORE importing numpy
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from pictologics.features.intensity import (
    calculate_intensity_features,
    calculate_intensity_histogram_features,
    calculate_ivh_features,
    calculate_local_intensity_features,
    calculate_spatial_intensity_features,
)


class TestIntensityFeatures(unittest.TestCase):
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

    def test_calculate_intensity_features_percentiles_ibsi_nearest_rank(self) -> None:
        # IBSI uses the nearest-rank percentile convention (smallest value with at
        # least p% of the data at or below it), not linear interpolation. On 1..10
        # this gives integer percentiles; linear interpolation would give 1.9/9.1.
        values = np.arange(1, 11, dtype=float)
        features = calculate_intensity_features(values)
        self.assertAlmostEqual(features["10th_intensity_percentile_QG58"], 1.0)
        self.assertAlmostEqual(features["90th_intensity_percentile_8DWT"], 9.0)
        # P25 = 3, P75 = 8 -> IQR = 5.
        self.assertAlmostEqual(features["intensity_interquartile_range_SALO"], 5.0)
        # Robust MAD over [P10, P90] = values 1..9: mean 5, MAD = 20/9.
        self.assertAlmostEqual(
            features["intensity_robust_mean_absolute_deviation_1128"], 20.0 / 9.0
        )
        # Median stays the conventional sample median (mean of the two middle values).
        self.assertAlmostEqual(features["median_intensity_Y12H"], 5.5)

    def test_order_statistics_match_numpy(self) -> None:
        # One partition gives numpy's inverted_cdf percentiles and median, bit for bit.
        from pictologics.features.intensity import _order_statistics

        rng = np.random.default_rng(5)
        for x in (
            rng.normal(100.0, 50.0, 101),
            rng.normal(100.0, 50.0, 100),
            rng.integers(-5, 40, 64).astype(np.int32),
            np.round(rng.normal(0.0, 3.0, 57)) * 0.1,
            np.array([7.0]),
        ):
            expected = (
                *np.percentile(x, [10, 25, 75, 90], method="inverted_cdf"),
                np.median(x),
            )
            got = _order_statistics(x)
            np.testing.assert_array_equal(np.array(got, dtype=float), np.array(expected, float))

        # A NaN makes every order statistic NaN, as in numpy.
        with_nan = np.array([1.0, np.nan, 3.0])
        self.assertTrue(np.isnan(_order_statistics(with_nan)).all())
        self.assertTrue(np.isnan(calculate_intensity_features(with_nan)["median_intensity_Y12H"]))

    def test_calculate_intensity_features_empty(self) -> None:
        features = calculate_intensity_features(np.array([]))
        self.assertEqual(features, {})

    def test_calculate_intensity_features_constant(self) -> None:
        # IBSI: skewness and kurtosis are 0 when the variance is 0. Equal values whose
        # mean is not exact (0.1) count too, although np.var gives about 1e-34 for them.
        for value in (5.0, 0.1):
            features = calculate_intensity_features(np.full(3, value))
            self.assertAlmostEqual(features["mean_intensity_Q4LE"], value)
            self.assertAlmostEqual(features["intensity_variance_ECT3"], 0.0)
            self.assertEqual(features["intensity_skewness_KE2A"], 0.0)
            self.assertEqual(features["intensity_kurtosis_IPH6"], 0.0)

    def test_calculate_intensity_features_zero_mean(self) -> None:
        # CV and Quartile coeff denom = 0
        values = np.array([-1, 1], dtype=float)
        features = calculate_intensity_features(values)
        self.assertTrue(np.isnan(features["intensity_coefficient_of_variation_7TET"]))
        self.assertTrue(np.isnan(features["intensity_quartile_coefficient_of_dispersion_9S40"]))

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

    def test_calculate_intensity_histogram_features_percentiles_nearest_rank(self) -> None:
        # Nearest-rank percentiles on the discretised histogram path (1..10).
        disc_vals = np.arange(1, 11, dtype=np.int64)
        features = calculate_intensity_histogram_features(disc_vals, n_bins=10)
        self.assertAlmostEqual(features["10th_discretised_intensity_percentile_1PR"], 1.0)
        self.assertAlmostEqual(features["90th_discretised_intensity_percentile_GPMT"], 9.0)
        self.assertAlmostEqual(features["median_discretised_intensity_WIFQ"], 5.5)

    def test_histogram_order_statistics_match_numpy(self) -> None:
        # Integer values read the percentiles and the median from the histogram counts,
        # float values from one partition; both give numpy's values.
        rng = np.random.default_rng(8)
        for disc_vals, n_bins in (
            (rng.integers(1, 33, 301).astype(np.int32), 32),
            (rng.integers(-4, 9, 300), None),
            (rng.integers(1, 7, 99).astype(np.float64), 6),
        ):
            features = calculate_intensity_histogram_features(disc_vals, n_bins=n_bins)
            p10, p25, p75, p90 = np.percentile(disc_vals, [10, 25, 75, 90], method="inverted_cdf")
            self.assertEqual(features["10th_discretised_intensity_percentile_1PR"], float(p10))
            self.assertEqual(features["90th_discretised_intensity_percentile_GPMT"], float(p90))
            self.assertEqual(
                features["discretised_intensity_interquartile_range_WR0O"], float(p75 - p25)
            )
            self.assertEqual(
                features["median_discretised_intensity_WIFQ"], float(np.median(disc_vals))
            )

    def test_calculate_intensity_histogram_features_multimodal(self) -> None:
        # IBSI AMMC: with multiple modes, select the one closest to the mean.
        # Modes 1 and 5, mean 3.2 -> 5 is closer.
        disc_vals = np.array([1, 1, 4, 5, 5])
        features = calculate_intensity_histogram_features(disc_vals)
        self.assertAlmostEqual(features["intensity_histogram_mode_AMMC"], 5.0)

        # Modes equidistant from the mean -> select the one below it.
        # Modes 2 and 4, mean 3.0 -> 2.
        disc_vals = np.array([2, 2, 4, 4])
        features = calculate_intensity_histogram_features(disc_vals)
        self.assertAlmostEqual(features["intensity_histogram_mode_AMMC"], 2.0)

    def test_calculate_intensity_histogram_features_full_range(self) -> None:
        # IBSI: with n_bins given, the histogram spans [1, N_g] including bins
        # the data does not reach. Values 3, 3, 4 with N_g=6:
        # histogram [0, 0, 2, 1, 0, 0], gradient [0, 1, 0.5, -1, -0.5, 0].
        disc_vals = np.array([3, 3, 4])
        features = calculate_intensity_histogram_features(disc_vals, n_bins=6)
        self.assertAlmostEqual(features["maximum_histogram_gradient_12CE"], 1.0)
        self.assertAlmostEqual(features["maximum_histogram_gradient_intensity_8E6O"], 2.0)
        self.assertAlmostEqual(features["minimum_histogram_gradient_VQB3"], -1.0)
        self.assertAlmostEqual(features["minimum_histogram_gradient_intensity_RHQZ"], 4.0)
        # Value-based features are unaffected by the padded bins.
        self.assertAlmostEqual(features["intensity_histogram_mode_AMMC"], 3.0)
        self.assertAlmostEqual(features["minimum_discretised_intensity_1PR8"], 3.0)
        self.assertAlmostEqual(features["maximum_discretised_intensity_3NCY"], 4.0)

        # Without n_bins the histogram spans only the observed range [3, 4]:
        # histogram [2, 1], gradient [-1, -1].
        features = calculate_intensity_histogram_features(disc_vals)
        self.assertAlmostEqual(features["maximum_histogram_gradient_12CE"], -1.0)
        self.assertAlmostEqual(features["maximum_histogram_gradient_intensity_8E6O"], 3.0)

        # Constant data has a defined gradient over the full histogram
        # (data-range fallback would give NaN): [0, 2, 0, 0] -> [2, 0, -1, 0].
        features = calculate_intensity_histogram_features(np.array([2, 2]), n_bins=4)
        self.assertAlmostEqual(features["maximum_histogram_gradient_12CE"], 2.0)
        self.assertAlmostEqual(features["minimum_histogram_gradient_VQB3"], -1.0)

    def test_calculate_intensity_histogram_features_n_bins_validation(self) -> None:
        # Values outside [1, n_bins] are a configuration error.
        with self.assertRaises(ValueError):
            calculate_intensity_histogram_features(np.array([3, 3, 7]), n_bins=6)
        with self.assertRaises(ValueError):
            calculate_intensity_histogram_features(np.array([0, 1]), n_bins=6)

    def test_calculate_intensity_histogram_features_empty(self) -> None:
        features = calculate_intensity_histogram_features(np.array([]))
        self.assertEqual(features, {})

    def test_calculate_intensity_histogram_features_constant(self) -> None:
        # All same values -> variance 0
        disc_vals = np.array([1, 1, 1, 1])
        features = calculate_intensity_histogram_features(disc_vals)
        self.assertEqual(features["discretised_intensity_skewness_88K1"], 0.0)
        self.assertEqual(features["discretised_intensity_kurtosis_C3I7"], 0.0)
        self.assertEqual(features["intensity_histogram_coefficient_of_variation_CWYJ"], 0.0)
        self.assertTrue(np.isnan(features["maximum_histogram_gradient_12CE"]))

    def test_calculate_intensity_histogram_features_zero_mean(self) -> None:
        disc_vals = np.array([-1, 1])
        features = calculate_intensity_histogram_features(disc_vals)
        self.assertTrue(np.isnan(features["intensity_histogram_coefficient_of_variation_CWYJ"]))

    # ----------------------------------------------------------------------
    # 4.3 IVH
    # ----------------------------------------------------------------------

    def test_calculate_ivh_features_basic(self) -> None:
        # 0, 1, 2, 3, 4
        # Bin width 1.0, min 0.0 implies bins [0,1), [1,2), ...
        disc_vals = np.array([0, 1, 2, 3, 4])
        features = calculate_ivh_features(disc_vals, bin_width=1.0, min_val=0.0)

        self.assertAlmostEqual(features["volume_at_intensity_fraction_0.10_BC2M_10"], 0.8)

    def test_calculate_ivh_features_degenerate_range(self) -> None:
        # max_val == min_val: candidate grid collapses to a single bin center
        # instead of crashing.
        features = calculate_ivh_features(
            np.array([1, 1, 1]), bin_width=1.0, min_val=5.0, max_val=5.0
        )
        self.assertTrue(np.isfinite(features["intensity_at_volume_fraction_0.10_GBPN_10"]))

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

    def test_ivh_counts_give_the_features_of_the_sorted_values(self) -> None:
        # Integers take one count per value, with no sort, also from a negative minimum;
        # integers over a range far longer than the values take the sort, as floats do.
        # Each gives the features of the same values as floats.
        rng = np.random.default_rng(16)
        ints = rng.integers(-40, 60, 500)
        settings = [
            {},
            {"min_val": -50.0, "max_val": 70.0},
            {"target_range_min": 0.0, "target_range_max": 50.0},
        ]
        many = patch("pictologics.features.intensity._PARALLEL_COUNT_MIN", 1)
        for kwargs in settings:
            with many, patch.object(np, "sort", side_effect=AssertionError("sorted")):
                counted = calculate_ivh_features(ints, **kwargs)
            self.assertEqual(counted, calculate_ivh_features(ints.astype(np.float64), **kwargs))
            self.assertEqual(calculate_ivh_features(ints, **kwargs), counted)  # few: the sort
        # Other integers count as int64
        with many:
            short = ints.astype(np.int16)
            self.assertEqual(
                calculate_ivh_features(short), calculate_ivh_features(short.astype(np.float64))
            )
        wide = np.array([0, 10**9, 3, 10**9])
        self.assertEqual(
            calculate_ivh_features(wide), calculate_ivh_features(wide.astype(np.float64))
        )

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
        self.assertAlmostEqual(features["volume_at_intensity_fraction_0.10_BC2M_10"], 2.0 / 3.0)

        # Test the branch where bin_width is NOT provided but target range IS.
        features2 = calculate_ivh_features(
            disc_vals, min_val=0.0, target_range_min=0.0, target_range_max=6.0
        )
        self.assertAlmostEqual(features2["volume_at_intensity_fraction_0.10_BC2M_10"], 2.0 / 3.0)

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
        # max_val < min_val is a configuration error.
        disc_vals = np.array([0, 1])
        with self.assertRaises(ValueError):
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

    def test_calculate_spatial_intensity_features_matches_bruteforce(self) -> None:
        # Verify the optimized (symmetry-exploiting) kernel against a direct
        # implementation of the IBSI definitions over all ordered voxel pairs.
        rng = np.random.default_rng(42)
        shape = (4, 4, 3)
        spacing = (0.7, 0.9, 3.0)
        data = rng.normal(100.0, 25.0, size=shape)
        mask = (rng.random(shape) < 0.6).astype(np.uint8)

        mock_img = MagicMock()
        mock_img.array = data
        mock_img.spacing = spacing
        mock_mask = MagicMock()
        mock_mask.array = mask

        features = calculate_spatial_intensity_features(mock_img, mock_mask)

        xs, ys, zs = np.where(mask > 0)
        vals = data[mask > 0].astype(np.float64)
        n = vals.size
        diff = vals - vals.mean()

        w_sum = 0.0
        moran_num = 0.0
        geary_num = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dx = (float(xs[i]) - float(xs[j])) * spacing[0]
                dy = (float(ys[i]) - float(ys[j])) * spacing[1]
                dz = (float(zs[i]) - float(zs[j])) * spacing[2]
                w = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
                w_sum += w
                moran_num += w * diff[i] * diff[j]
                geary_num += w * (vals[i] - vals[j]) ** 2

        denom = float(np.sum(diff * diff))
        expected_moran = (n / w_sum) * (moran_num / denom)
        expected_geary = ((n - 1) / (2 * w_sum)) * (geary_num / denom)

        self.assertAlmostEqual(features["morans_i_index_N365"], expected_moran)
        self.assertAlmostEqual(features["gearys_c_measure_NPT7"], expected_geary)

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

    @staticmethod
    def _spatial_case() -> tuple[MagicMock, MagicMock]:
        rng = np.random.default_rng(12)
        image, roi = MagicMock(), MagicMock()
        image.array = rng.normal(100.0, 25.0, (9, 8, 7))
        image.spacing = (0.8, 0.8, 2.5)
        roi.array = np.zeros((9, 8, 7))
        roi.array[1:8, 2:8, 1:6] = rng.random((7, 6, 5)) < 0.7
        return image, roi

    def test_spatial_intensity_fft_matches_pair_loop(self) -> None:
        # The FFT sums give the pair loop's Moran's I and Geary's C to about 1e-14.
        image, roi = self._spatial_case()
        module = "pictologics.features.intensity._FFT_MIN_PAIRS_PER_POINT"
        with patch(module, 1 << 62):
            pair = calculate_spatial_intensity_features(image, roi)
        with patch(module, 0):
            fft = calculate_spatial_intensity_features(image, roi)
        for key, value in pair.items():
            self.assertAlmostEqual(fft[key], value, delta=1e-12 * abs(value))

    def test_spatial_intensity_fft_threads_follow_numba(self) -> None:
        # The FFT sums use numba's thread count, with the same values.
        import scipy.fft

        image, roi = self._spatial_case()
        with patch("pictologics.features.intensity._FFT_MIN_PAIRS_PER_POINT", 0):
            expected = calculate_spatial_intensity_features(image, roi)
            with (
                patch("pictologics.features.intensity.get_num_threads", return_value=2),
                patch.object(scipy.fft, "rfftn", wraps=scipy.fft.rfftn) as rfftn,
                patch.object(scipy.fft, "irfftn", wraps=scipy.fft.irfftn) as irfftn,
            ):
                self.assertEqual(calculate_spatial_intensity_features(image, roi), expected)
        calls = rfftn.call_args_list + irfftn.call_args_list
        self.assertTrue(calls)
        self.assertTrue(all(c.kwargs["workers"] == 2 for c in calls))

    def test_spatial_intensity_fft_memory_cap(self) -> None:
        # Above the memory cap, the pair loop runs and a warning gives the expected time.
        from pictologics.features.intensity import _fft_memory_cap

        self.assertLessEqual(_fft_memory_cap(), 16 << 30)
        with patch("pictologics.features.intensity.os.sysconf", side_effect=ValueError):
            self.assertEqual(_fft_memory_cap(), 4 << 30)  # 8 GB assumed

        image, roi = self._spatial_case()
        with patch("pictologics.features.intensity._FFT_MIN_PAIRS_PER_POINT", 1 << 62):
            pair = calculate_spatial_intensity_features(image, roi)
        with (
            patch("pictologics.features.intensity._FFT_MIN_PAIRS_PER_POINT", 0),
            patch("pictologics.features.intensity._fft_memory_cap", return_value=0),
            self.assertWarnsRegex(UserWarning, "Expected run time"),
        ):
            self.assertEqual(calculate_spatial_intensity_features(image, roi), pair)

    def test_calculate_spatial_intensity_features_small_input(self) -> None:
        # < 2 voxels -> NaN
        mock_img = MagicMock()
        mock_img.array = np.array([[[1]]])
        mock_img.spacing = (1.0, 1.0, 1.0)
        mock_mask = MagicMock()
        mock_mask.array = np.array([[[1]]])
        features = calculate_spatial_intensity_features(mock_img, mock_mask)
        self.assertTrue(np.isnan(features["morans_i_index_N365"]))

        # An empty mask has an empty ROI box.
        mock_mask.array = np.array([[[0]]])
        features = calculate_spatial_intensity_features(mock_img, mock_mask)
        self.assertTrue(np.isnan(features["gearys_c_measure_NPT7"]))

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
        features = calculate_spatial_intensity_features(mock_img, mock_mask, enabled=False)
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

    def test_local_intensity_crop_matches_full_image(self) -> None:
        # The crop (ROI box plus sphere reach) gives the full-image result, bit for bit,
        # for an ROI inside the image and for one at its edge.
        from pictologics.features.intensity import (
            _calculate_local_mean_numba,
            _calculate_local_peaks_numba,
            _sphere_offsets_for_radius,
        )

        rng = np.random.default_rng(4)
        spacing = (1.5, 2.0, 3.0)
        data = rng.normal(0.0, 50.0, (20, 18, 12))
        offsets = _sphere_offsets_for_radius(spacing, 6.2035)
        for box in ((slice(8, 11), slice(7, 10), slice(5, 7)), (slice(0, 4), slice(15, 18), 11)):
            mask = np.zeros(data.shape)
            mask[box] = 1.0
            idx = np.ascontiguousarray(np.stack(np.where(mask > 0), axis=1).astype(np.int32))
            means = _calculate_local_mean_numba(data, idx, offsets)
            expected = _calculate_local_peaks_numba(data, idx, means)

            image, roi = MagicMock(), MagicMock()
            image.array, image.spacing, roi.array = data, spacing, mask
            for min_size in (1, 1 << 15):  # with the crop, and without (a small image)
                with patch("pictologics.features.intensity._LOCAL_CROP_MIN_SIZE", min_size):
                    features = calculate_local_intensity_features(image, roi)
                self.assertEqual(features["global_intensity_peak_0F91"], expected[0])
                self.assertEqual(features["local_intensity_peak_VJGA"], expected[1])

        # Every nonzero label is ROI membership, also a negative one (after the crop too);
        # a mask of zeros has no ROI.
        roi.array = -mask
        with patch("pictologics.features.intensity._LOCAL_CROP_MIN_SIZE", 1):
            features = calculate_local_intensity_features(image, roi)
        self.assertEqual(features["global_intensity_peak_0F91"], expected[0])
        roi.array = np.zeros(data.shape)
        with patch("pictologics.features.intensity._LOCAL_CROP_MIN_SIZE", 1):
            self.assertEqual(calculate_local_intensity_features(image, roi), {})

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
        features = calculate_local_intensity_features(mock_img, mock_mask, enabled=False)
        self.assertEqual(features, {})

    def test_internal_helpers_edge_cases(self) -> None:
        # Test Numba helpers directly for empty input coverage
        from pictologics.features.intensity import (
            _calculate_local_peaks_numba,
            _central_moments_2_3_4,
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


class TestFastPaths(unittest.TestCase):
    """The radix select and the two-stage local peaks give the values of the direct
    paths bit for bit; a NaN or an infinite intensity takes the direct path."""

    def test_radix_select_order_statistics(self) -> None:
        from pictologics.features import intensity as intensity_module

        rng = np.random.default_rng(12)
        for values in (
            rng.normal(0.0, 100.0, 301),
            np.round(rng.normal(0.0, 3.0, 300)),
            rng.choice([-0.0, 0.0, 2.5, -np.inf, np.inf, 1e-300], 200),
        ):
            with np.errstate(invalid="ignore"):  # inf - inf in the moments
                expected = calculate_intensity_features(values)
                with patch.object(intensity_module, "_RADIX_SELECT_MIN", 8):
                    # NaN features (inf - inf) count as equal
                    np.testing.assert_equal(calculate_intensity_features(values), expected)
        with_nan = rng.normal(0.0, 1.0, 50)
        with_nan[7] = np.nan
        self.assertIsNone(intensity_module._radix_select(with_nan, np.array([3, 20])))

    def test_radix_select_keeps_the_candidates_of_the_ranks(self) -> None:
        # The search picks the values of np.partition, with the bits of the search of
        # 0.6.0 (numpy copy below: the values of the buckets of the ranks, in their order),
        # also -0.0 or +0.0 at a rank, for 1 and 3 chunks.
        from pictologics.features import intensity as intensity_module

        def search_of_0_6_0(values: np.ndarray, ranks: np.ndarray) -> np.ndarray:
            bits = values.view(np.uint64)
            keys = np.where(bits >> np.uint64(63) != 0, ~bits, bits ^ np.uint64(1 << 63))
            low, high = int(keys.min()), int(keys.max())
            shift = max(0, (high - low).bit_length() - 16)
            bucket = ((keys >> np.uint64(shift)) - np.uint64(low >> shift)).astype(np.int64)
            per_bucket = np.bincount(bucket, minlength=1 << 16)
            below = np.concatenate(([0], np.cumsum(per_bucket)))
            buckets = np.searchsorted(below[1:], ranks, side="right")
            wanted = np.zeros(per_bucket.size, dtype=bool)
            wanted[buckets] = True
            kept = np.where(wanted, per_bucket, 0)
            at = (np.cumsum(kept) - kept)[buckets] + ranks - below[buckets]
            return np.partition(values[wanted[bucket]], np.unique(at))[at]

        rng = np.random.default_rng(14)
        crowded = np.full(2000, 3.0)
        crowded[:40] = rng.normal(0.0, 1e6, 40)
        for values in (
            rng.normal(40.0, 20.0, 2001),
            np.round(rng.normal(0.0, 3.0, 2000)),
            rng.choice([-0.0, 0.0, 1.0, -1.0], 2000, p=[0.3, 0.3, 0.2, 0.2]),
            rng.choice([-np.inf, np.inf, -2.5, 0.0, 7.0], 1999),
            -rng.gamma(2.0, 30.0, 2000),
            crowded,
        ):
            n = values.size
            ranks = intensity_module._percentile_ranks(n, values.dtype)
            ranks = np.concatenate((ranks, [n // 2 - 1 + n % 2, n // 2]))
            expected = search_of_0_6_0(values, ranks)
            np.testing.assert_array_equal(expected, np.partition(values, ranks)[ranks])
            for chunks in (1, 3):
                with patch.object(intensity_module, "get_num_threads", return_value=chunks):
                    got = intensity_module._radix_select(values, ranks)
                self.assertEqual(got.tobytes(), expected.tobytes())

    def test_radix_select_holds_every_bucket_of_the_key_range(self) -> None:
        # The smallest and the largest key can be 65,536 buckets apart (here a key range
        # of 2^17 - 1 from an odd key): the table holds that bucket too, so the values are
        # those of np.partition. 0.6.0 counted it one place past its 65,536 buckets.
        from pictologics.features import intensity as intensity_module

        first = np.array([1.0]).view(np.uint64) + np.uint64(1)  # an odd key
        steps = np.random.default_rng(15).integers(0, 2**17 - 1, 500, dtype=np.uint64)
        steps[[0, 250]] = 0, 2**17 - 1
        values = (first + steps).view(np.float64)
        ranks = np.arange(values.size)
        got = intensity_module._radix_select(values, ranks)
        np.testing.assert_array_equal(got, np.sort(values))

    def test_two_stage_local_peaks(self) -> None:
        from pictologics.features import intensity as intensity_module
        from pictologics.loader import Image

        rng = np.random.default_rng(13)
        data = rng.normal(50.0, 20.0, (12, 13, 14))
        data[6, 6, 6] = data.max() + 1.0  # one brightest voxel
        mask = np.zeros(data.shape, dtype=np.uint8)
        mask[2:10, 3:11, 2:12] = 1
        spacing = (2.0, 2.5, 3.0)
        for array in (data, np.where(mask > 0, np.inf, data)):
            image = Image(array, spacing, (0.0, 0.0, 0.0))
            roi = Image(mask, spacing, (0.0, 0.0, 0.0))
            expected = calculate_local_intensity_features(image, roi)
            with patch.object(intensity_module, "_TWO_STAGE_MIN_WORK", 1):
                self.assertEqual(calculate_local_intensity_features(image, roi), expected)


if __name__ == "__main__":
    unittest.main()
