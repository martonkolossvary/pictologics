# ruff: noqa: E402
import math
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
import pytest

from pictologics.features._utils import compute_nonzero_bbox
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

    def test_the_variance_is_the_second_central_moment_of_numpy(self) -> None:
        # The variance comes from the pass of the central moments, not from np.var: 60
        # random cases stay within 1e-12 of numpy's two-pass variance, and the coefficient
        # of variation with it. The other first-order features use unchanged code.
        rng = np.random.default_rng(11)
        for case in range(60):
            n = int(rng.integers(2, 2000))
            values = (
                rng.normal(40.0, 60.0, n),
                rng.normal(1000.0, 1.0, n),
                rng.integers(-1000, 3000, n).astype(np.float64),
                rng.exponential(5.0, n).astype(np.float32),
            )[case % 4]
            features = calculate_intensity_features(values)
            exact = values.astype(np.float64)
            variance = np.var(exact)
            np.testing.assert_allclose(features["intensity_variance_ECT3"], variance, rtol=1e-12)
            np.testing.assert_allclose(
                features["intensity_coefficient_of_variation_7TET"],
                np.sqrt(variance) / np.mean(exact),
                rtol=1e-12,
            )

    def test_calculate_intensity_features_constant(self) -> None:
        # IBSI: skewness and kurtosis are 0 when the variance is 0. Equal values whose
        # mean is not exact (0.1) count too, although their variance is about 1e-34.
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
        # A zero mean and p75 + p25 = 0 give NaN, on the counted path and on the float path.
        for kind in (np.int64, np.float64):
            features = calculate_intensity_histogram_features(np.array([-1, 1], dtype=kind))
            self.assertTrue(np.isnan(features["intensity_histogram_coefficient_of_variation_CWYJ"]))
            self.assertTrue(
                np.isnan(features["intensity_histogram_quartile_coefficient_of_dispersion_SLWD"])
            )

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
        many = patch("pictologics.features.intensity._COUNT_MIN", 1)
        for kwargs in settings:
            with many, patch.object(np, "sort", side_effect=AssertionError("sorted")):
                counted = calculate_ivh_features(ints, **kwargs)
            self.assertEqual(counted, calculate_ivh_features(ints.astype(np.float64), **kwargs))
            self.assertEqual(calculate_ivh_features(ints, **kwargs), counted)  # few: the sort
        # Other integers count as int32 copies (1- and 2-byte types) or int64 copies
        with many:
            short = ints.astype(np.int16)
            self.assertEqual(
                calculate_ivh_features(short), calculate_ivh_features(short.astype(np.float64))
            )
        wide = np.array([0, 10**9, 3, 10**9])
        self.assertEqual(
            calculate_ivh_features(wide), calculate_ivh_features(wide.astype(np.float64))
        )

    def test_ivh_counts_whole_floats(self) -> None:
        # Whole float64 and float32 values, also negative ones, take one count per value as
        # int32 (no sort), with the features of the sort and of the same values as int64.
        rng = np.random.default_rng(17)
        ints = rng.integers(-40, 60, 500)
        settings = [
            {},
            {"bin_width": 1.0},
            {"bin_width": 2.5, "min_val": -1000.0, "target_range_max": 400.0},
            {"bin_width": 0.3, "min_val": 0.2},
        ]
        for kind in (np.float64, np.float32):
            values = ints.astype(kind)
            for kwargs in settings:
                expected = calculate_ivh_features(values, **kwargs)  # few values: the sort
                with (
                    patch("pictologics.features.intensity._COUNT_MIN", 1),
                    patch.object(np, "sort", side_effect=AssertionError("sorted")),
                ):
                    self.assertEqual(calculate_ivh_features(values, **kwargs), expected)
                self.assertEqual(calculate_ivh_features(ints, **kwargs), expected)

    def test_ivh_sorts_other_floats(self) -> None:
        # Floats that are not all whole numbers of the int32 range, values over a range far
        # longer than the values, bool, uint64 and float16 values take the sort, with its
        # features; the whole-number check runs only when the first value is whole and the
        # range fits int32.
        from pictologics.features import intensity as intensity_module

        base = np.arange(1.0, 21.0)

        def changed(index: int, value: float) -> np.ndarray:
            out = base.copy()
            out[index] = value
            return out

        cases = [
            (changed(0, 2.5), 0),
            (changed(7, 2.5), 1),
            (changed(0, np.nan), 0),
            (changed(7, np.nan), 0),
            (changed(7, np.inf), 0),
            (changed(7, 2147483648.0), 0),
            (changed(7, 1e6), 0),  # a range far longer than the values
            (base > 10, 0),
            (base.astype(np.uint64), 0),
            (base.astype(np.float16), 0),
        ]
        check = intensity_module._whole_as_int32_numba
        for values, checks in cases:
            expected = calculate_ivh_features(values)
            with (
                patch.object(intensity_module, "_COUNT_MIN", 1),
                patch.object(intensity_module, "_whole_as_int32_numba", wraps=check) as spy,
                patch.object(np, "sort", wraps=np.sort) as sort,
            ):
                np.testing.assert_equal(calculate_ivh_features(values), expected)
            self.assertEqual((spy.call_count, sort.call_count > 0), (checks, True))

    def test_value_counts_rows_and_threads(self) -> None:
        # The value counts are those of np.bincount: uint8 and int16 values (int32 copies),
        # int64 values over 70,000 distinct values (one count row), 7 values (the tail of
        # the four rows), and 40,000 int32 values in two parts on threads.
        from pictologics.features import intensity as intensity_module

        rng = np.random.default_rng(18)
        cases = [
            rng.integers(0, 200, 1000).astype(np.uint8),
            rng.integers(-300, 300, 1000).astype(np.int16),
            rng.permutation(70_000).astype(np.int64),
            np.array([3, 1, 4, 1, 5, 9, 2], dtype=np.int32),
        ]
        for values in cases:
            low = min(int(values.min()), 0)
            size = int(values.max()) - low + 1
            got = intensity_module._value_counts(values, low, size)
            np.testing.assert_array_equal(got, np.bincount(values.astype(np.int64) - low))
        values = rng.integers(0, 10, 40_000).astype(np.int32)
        with (
            patch.object(intensity_module, "_PARALLEL_COUNT_MIN", 1),
            patch.object(intensity_module, "get_num_threads", return_value=4),
            patch.object(
                intensity_module, "_value_counts_numba", wraps=intensity_module._value_counts_numba
            ) as threads,
        ):
            got = intensity_module._value_counts(values, 0, 10)
        self.assertEqual(threads.call_args.args[2].shape, (2, 10))
        np.testing.assert_array_equal(got, np.bincount(values))

    def test_ivh_strided_input(self) -> None:
        # A strided view gives the features of its contiguous copy.
        values = np.random.default_rng(19).integers(1, 30, 5000).astype(np.float64)
        view = values[::2]
        self.assertEqual(calculate_ivh_features(view), calculate_ivh_features(view.copy()))

    def test_ivh_grid_candidates(self) -> None:
        # The candidates of a bin width are Python floats one at a time: float32 values (a
        # float32 base) and float32 keywords give the features of float64 ones.
        from pictologics.features.intensity import _Grid

        values = np.array([1, 2, 2, 5, 7], dtype=np.float32)
        self.assertEqual(
            calculate_ivh_features(values, bin_width=0.3),
            calculate_ivh_features(values.astype(np.float64), bin_width=0.3),
        )
        width, low = np.float32(0.3), np.float32(0.2)
        self.assertEqual(
            calculate_ivh_features(values, bin_width=width, min_val=low),
            calculate_ivh_features(values, bin_width=float(width), min_val=float(low)),
        )
        grid = _Grid(3, 1.0, 0.5, 0.5)
        self.assertEqual((len(grid), grid[0], grid[2], grid[-1]), (3, 1.25, 2.25, 2.25))
        self.assertEqual(_Grid(3, 0, 0.5, 0.0)[1], 0.5)

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

    @staticmethod
    def _spatial_even_case() -> tuple[MagicMock, MagicMock]:
        # An ROI box of (4, 6, 10): the FFT grid (8, 12, 20) is even on every axis
        rng = np.random.default_rng(13)
        image, roi = MagicMock(), MagicMock()
        image.array = rng.normal(100.0, 25.0, (7, 9, 13))
        image.spacing = (0.8, 0.9, 2.5)
        roi.array = np.zeros((7, 9, 13))
        box = rng.random((4, 6, 10)) < 0.7
        box[0, 0, 0] = box[-1, -1, -1] = True
        roi.array[2:6, 1:7, 2:12] = box
        return image, roi

    # The time model chooses the method: a pair loop of no cost always wins, and one of
    # 1 s per pair always loses.
    _PAIRS_WIN = patch("pictologics.features.intensity._PAIR_SECONDS", 0.0)
    _FFT_WINS = patch("pictologics.features.intensity._PAIR_SECONDS", 1.0)

    def test_spatial_intensity_fft_matches_pair_loop(self) -> None:
        # The FFT sums give the pair loop's Moran's I and Geary's C to about 1e-14: on an
        # odd grid (the real part of the full kernel spectrum) and on an even grid (the
        # DCT-I of the kernel octant, with the Nyquist bins).
        for image, roi in (self._spatial_case(), self._spatial_even_case()):
            with self._PAIRS_WIN:
                pair = calculate_spatial_intensity_features(image, roi)
            with self._FFT_WINS:
                fft = calculate_spatial_intensity_features(image, roi)
            for key, value in pair.items():
                self.assertAlmostEqual(fft[key], value, delta=1e-12 * abs(value))

    def test_spatial_intensity_slabs_match_pair_loop(self) -> None:
        # When the FFT grid does not fit in the memory budget, the slab sums run on the
        # fewest slabs that fit, with the pair loop's values to about 1e-14. Directly, on
        # every axis and every slab height (one plane up to the whole box, with a shorter
        # last slab), on an odd and an even grid.
        import scipy.fft

        from pictologics.features import intensity as module

        for image, roi in (self._spatial_case(), self._spatial_even_case()):
            with self._PAIRS_WIN:
                pair = calculate_spatial_intensity_features(image, roi)
            box = roi.array[compute_nonzero_bbox(roi.array)] != 0
            fshape = tuple(scipy.fft.next_fast_len(2 * k - 1, real=True) for k in box.shape)
            with (
                self._FFT_WINS,
                patch.object(module, "_memory_budget", return_value=32 * math.prod(fshape) - 1),
                patch.object(
                    module, "_spatial_sums_slabs", wraps=module._spatial_sums_slabs
                ) as slabs,
            ):
                features = calculate_spatial_intensity_features(image, roi)
            self.assertEqual(slabs.call_count, 1)
            for key, value in pair.items():
                self.assertAlmostEqual(features[key], value, delta=1e-12 * abs(value))
            values = image.array[compute_nonzero_bbox(roi.array)][box]
            diff = np.zeros(box.shape)
            diff[box] = values - values.mean()
            spacing = tuple(float(x) for x in image.spacing)
            reference = module._spatial_sums_fft(box, diff, spacing, fshape)
            for axis in range(3):
                for rows in range(1, box.shape[axis] + 1):
                    sshape = list(fshape)
                    sshape[axis] = scipy.fft.next_fast_len(2 * rows - 1, real=True)
                    sums = module._spatial_sums_slabs(box, diff, spacing, axis, rows, tuple(sshape))
                    for got, want in zip(sums, reference, strict=True):
                        self.assertAlmostEqual(got, want, delta=1e-12 * abs(want))

    def test_spatial_intensity_fft_threads_follow_numba(self) -> None:
        # The FFT sums use numba's thread count, or one worker when the model gives one
        # worker less time (here: workers that cost nothing or 1 s each to start), with the
        # same values; the DCT of the kernel runs on an even grid only.
        import scipy.fft

        for image, roi in (self._spatial_case(), self._spatial_even_case()):
            for start, workers in ((0.0, 2), (1.0, 1)):
                with (
                    self._FFT_WINS,
                    patch("pictologics.features.intensity._FFT_START_SECONDS", (0.0, start)),
                ):
                    expected = calculate_spatial_intensity_features(image, roi)
                    with (
                        patch("pictologics.features.intensity.get_num_threads", return_value=2),
                        patch.object(scipy.fft, "rfftn", wraps=scipy.fft.rfftn) as rfftn,
                        patch.object(scipy.fft, "irfftn", wraps=scipy.fft.irfftn) as irfftn,
                        patch.object(scipy.fft, "dctn", wraps=scipy.fft.dctn) as dctn,
                    ):
                        features = calculate_spatial_intensity_features(image, roi)
                for key, value in expected.items():
                    self.assertAlmostEqual(features[key], value, delta=1e-12 * abs(value))
                calls = rfftn.call_args_list + irfftn.call_args_list + dctn.call_args_list
                self.assertTrue(calls)
                self.assertTrue(all(c.kwargs["workers"] == workers for c in calls))
                self.assertEqual(dctn.call_count, 0 if image.array.shape == (9, 8, 7) else 1)

    def test_the_spatial_kernels_take_their_serial_twins_below_their_gates(self) -> None:
        # Below their gates the spatial kernels run as their serial twins; from the gates on,
        # with more than one thread, the parallel kernels run. Both give the same features, on
        # the pair path, the FFT path (an even grid) and the slab path.
        from contextlib import ExitStack

        import scipy.fft

        from pictologics.features import _utils
        from pictologics.features import intensity as module

        names = [
            "_calculate_spatial_features_numba",
            "_inverse_distance_numba",
            "_multiply_spectrum_numba",
            "_weighted_power_numba",
            "_cross_power_numba",
        ]
        image, roi = self._spatial_even_case()
        fshape = tuple(scipy.fft.next_fast_len(2 * k - 1, real=True) for k in (4, 6, 10))
        budgets = {"pairs": 1 << 62, "fft": 1 << 62, "slabs": 32 * math.prod(fshape) - 1}

        def run(gate: int, path: str) -> tuple[dict[str, float], dict[str, int]]:
            with ExitStack() as stack:
                spies = {
                    name: stack.enter_context(
                        patch.object(module, name, wraps=getattr(module, name))
                    )
                    for base in names
                    for name in (base, base + "_serial")
                }
                for name in ("_PAIRS_PARALLEL_MIN", "_SPATIAL_FFT_PARALLEL_MIN"):
                    stack.enter_context(patch.object(module, name, getattr(module, name) * gate))
                stack.enter_context(self._PAIRS_WIN if path == "pairs" else self._FFT_WINS)
                stack.enter_context(
                    patch.object(module, "_memory_budget", return_value=budgets[path])
                )
                stack.enter_context(patch.object(_utils, "get_num_threads", return_value=2))
                features = calculate_spatial_intensity_features(image, roi)
            return features, {name: spy.call_count for name, spy in spies.items()}

        paths = {"pairs": names[:1], "fft": names[1:4], "slabs": [names[1], names[4]]}
        for path, used in paths.items():
            small, calls = run(1, path)
            self.assertTrue(all(calls[n] == 0 and calls[n + "_serial"] > 0 for n in used))
            large, calls = run(0, path)
            self.assertTrue(all(calls[n] > 0 and calls[n + "_serial"] == 0 for n in used))
            for key, value in small.items():
                self.assertAlmostEqual(large[key], value, delta=1e-12 * abs(value))

    def test_spatial_plan_takes_the_fastest_method_that_fits(self) -> None:
        # The FFT sums when the whole grid fits in the memory budget and the model says
        # they are faster; else the fewest slabs that fit; the pair loop when it is
        # faster. When not even slabs of one plane fit, the pair loop runs with a warning
        # that gives its expected time.
        from pictologics.features import intensity as module

        box = (7, 6, 5)  # the FFT grid (15, 12, 9): 1,620 points, 51,840 bytes
        with self._FFT_WINS, patch.object(module, "_memory_budget", return_value=51_840):
            self.assertEqual(module._spatial_plan(100, box), ("fft", 0, 0, (15, 12, 9)))
        with self._FFT_WINS, patch.object(module, "_memory_budget", return_value=51_839):
            # 2 slabs of 4 planes: 8 x 12 x 9 x 64 = 55,296 bytes; 3 of 3: 34,560 bytes
            self.assertEqual(module._spatial_plan(100, box), ("slabs", 0, 3, (5, 12, 9)))
            # the longest axis is cut
            self.assertEqual(module._spatial_plan(100, (5, 7, 6)), ("slabs", 1, 3, (9, 5, 12)))
        with self._PAIRS_WIN:
            for budget in (51_840, 51_839):
                with patch.object(module, "_memory_budget", return_value=budget):
                    self.assertEqual(module._spatial_plan(100, box), ("pairs", 0, 0, ()))
        with (
            patch.object(module, "_memory_budget", return_value=0),
            self.assertWarnsRegex(UserWarning, r"even in slabs of one plane.*Expected run time"),
        ):
            self.assertEqual(module._spatial_plan(100, box), ("pairs", 0, 0, ()))
        image, roi = self._spatial_case()
        with self._PAIRS_WIN:
            pair = calculate_spatial_intensity_features(image, roi)
        with (
            self._FFT_WINS,
            patch.object(module, "_memory_budget", return_value=0),
            self.assertWarnsRegex(UserWarning, "Expected run time"),
        ):
            self.assertEqual(calculate_spatial_intensity_features(image, roi), pair)

    def test_fft_and_pair_time_models(self) -> None:
        # The FFT: a start of 47 us + 38 us per worker, and 12.5 ns per grid point, of which
        # 30 % is serial below _SPATIAL_FFT_PARALLEL_MIN points and 15 % from there on. On
        # 10 threads, a grid of 32,768 points runs on one worker and one of 64,000 on all
        # (as measured). The parallel pair loop: a start of 45 us + 2.5 us per thread, and
        # 1.05 ns per pair and thread; below _PAIRS_PARALLEL_MIN voxels it runs serially.
        from pictologics.features import intensity as module

        self.assertAlmostEqual(module._fft_seconds(1000, 2.0, 1), 85e-6 + 2 * 1000 * 12.5e-9)
        share = 0.3 + 0.7 / 10
        self.assertAlmostEqual(
            module._fft_seconds(1 << 16, 1.0, 10), 427e-6 + (1 << 16) * 12.5e-9 * share
        )
        share = 0.15 + 0.85 / 10
        self.assertAlmostEqual(
            module._fft_seconds(1 << 20, 1.0, 10), 427e-6 + (1 << 20) * 12.5e-9 * share
        )
        with patch.object(module, "get_num_threads", return_value=10):
            self.assertEqual(module._fft_workers(32_768), 1)
            self.assertEqual(module._fft_workers(64_000), 10)
            self.assertAlmostEqual(module._pair_seconds(449), 449 * 448 / 2 * 1.05e-9)
            self.assertAlmostEqual(module._pair_seconds(450), 70e-6 + 450 * 449 / 2 * 0.105e-9)
        with patch.object(module, "get_num_threads", return_value=1):
            self.assertEqual(module._fft_workers(1 << 20), 1)
            self.assertAlmostEqual(module._pair_seconds(450), 450 * 449 / 2 * 1.05e-9)

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
        # The small-n kernel gives way here, so the radix select runs on these sizes.
        from pictologics.features import intensity as intensity_module

        give_way = patch.object(
            intensity_module, "_small_order_statistics_numba", return_value=(False,) + (0.0,) * 5
        )
        rng = np.random.default_rng(12)
        for values in (
            rng.normal(0.0, 100.0, 301),
            np.round(rng.normal(0.0, 3.0, 300)),
            rng.choice([-0.0, 0.0, 2.5, -np.inf, np.inf, 1e-300], 200),
        ):
            with np.errstate(invalid="ignore"):  # inf - inf in the moments
                expected = calculate_intensity_features(values)
                with patch.object(intensity_module, "_RADIX_SELECT_MIN", 8), give_way:
                    # NaN features (inf - inf) count as equal
                    np.testing.assert_equal(calculate_intensity_features(values), expected)
        with_nan = rng.normal(0.0, 1.0, 50)
        with_nan[7] = np.nan
        bounds = (np.min(with_nan), np.max(with_nan))  # NaN, as numpy gives them
        self.assertIsNone(intensity_module._radix_select(with_nan, np.array([3, 20]), bounds))
        # Without the bounds of the caller (the histogram path), the search makes them
        values = rng.normal(0.0, 100.0, 301)
        with patch.object(intensity_module, "_RADIX_SELECT_MIN", 8), give_way:
            np.testing.assert_equal(
                intensity_module._order_statistics(values),
                intensity_module._order_statistics(values, (np.min(values), np.max(values))),
            )

    def test_radix_select_keeps_the_candidates_of_the_ranks(self) -> None:
        # The search picks the values of np.partition, with the bits of the search of
        # 0.6.0 (numpy copy below: the values of the buckets of the ranks, in their order),
        # also -0.0 or +0.0 at a rank or as the only smallest or largest value (a bound of
        # the key range), for 1 and 3 chunks.
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
            np.abs(rng.choice([0.0, 3.0, 7.5], 2000)),  # +0.0 only, the smallest value
            -np.abs(rng.choice([0.0, 3.0, 7.5], 2000)),  # -0.0 only, the largest value
        ):
            n = values.size
            ranks = intensity_module._percentile_ranks(n)
            ranks = np.concatenate((ranks, [n // 2 - 1 + n % 2, n // 2]))
            expected = search_of_0_6_0(values, ranks)
            np.testing.assert_array_equal(expected, np.partition(values, ranks)[ranks])
            for chunks in (1, 3):
                with patch.object(intensity_module, "get_num_threads", return_value=chunks):
                    bounds = (np.min(values), np.max(values))
                    got = intensity_module._radix_select(values, ranks, bounds)
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
        got = intensity_module._radix_select(values, ranks, (values.min(), values.max()))
        np.testing.assert_array_equal(got, np.sort(values))

    def test_linear_select_order_statistics(self) -> None:
        # From 1,000 float64 values on, the linear select gives the values of the other paths
        # bit for bit, serial and in threads, or gives way to them: NaN or infinite bounds,
        # equal values, a span too small for the bucket scale, crowded values (the sample
        # finds them, or the count after it) and a zero rank with both zeros.
        from pictologics.features import intensity as intensity_module

        real = intensity_module._linear_select
        took: list[bool] = []

        def spy(values: np.ndarray, ranks: np.ndarray, bounds: tuple) -> np.ndarray | None:
            self.assertTrue(np.all(np.diff(ranks) >= 0))  # the kernels read them in order
            found = real(values, ranks, bounds)
            took.append(found is not None)
            return found

        def took_it(values: np.ndarray, threads: int = 1) -> bool:
            # The values of the other paths, with the bounds of the caller and without them;
            # whether the linear select gave them
            with patch.object(intensity_module, "_LINEAR_SELECT_MIN", values.size + 1):
                expected = np.array(intensity_module._order_statistics(values), dtype=float)
            took.clear()
            with (
                patch.object(intensity_module, "_linear_select", spy),
                patch.object(intensity_module, "_PARALLEL_SELECT_MIN", 1000),
                patch.object(intensity_module, "get_num_threads", return_value=threads),
            ):
                for bounds in ((np.min(values), np.max(values)), None):
                    got = intensity_module._order_statistics(values, bounds)
                    self.assertEqual(np.array(got, dtype=float).tobytes(), expected.tobytes())
            return all(took)

        rng = np.random.default_rng(16)
        side = np.abs(rng.normal(0.0, 100.0, 999))
        for values in (
            rng.normal(40.0, 20.0, 1001),
            rng.normal(40.0, 20.0, 1000),
            np.round(rng.normal(50.0, 30.0, 1000)),  # ties
            np.concatenate((-side, [0.0, 0.0], side)),  # +0.0 only at the middle ranks
        ):
            for threads in (1, 3):
                self.assertTrue(took_it(values, threads))
        with_nan, with_inf = rng.normal(0.0, 1.0, (2, 1000))
        with_nan[7], with_inf[7] = np.nan, np.inf
        crowded = np.full(2000, 3.0)
        crowded[:40] = rng.normal(0.0, 1e6, 40)
        both_zeros = np.concatenate((-side, [-0.0, 0.0], side))  # np.partition orders them freely
        for values in (
            with_nan,
            with_inf,
            np.full(1000, 0.1),
            rng.normal(0.0, 1.0, 1000) * 1e-310,  # 4,096 buckets over this span: an inf scale
            crowded,
            both_zeros,
        ):
            self.assertFalse(took_it(values))
        self.assertFalse(took_it(both_zeros, 3))
        with patch.object(intensity_module, "_crowded_numba", return_value=False):
            for threads in (1, 3):  # the count finds the crowded values
                self.assertFalse(took_it(crowded, threads))

    def test_two_stage_local_peaks(self) -> None:
        # The two-stage search gives the peaks of the exact kernel on every voxel, bit for
        # bit: the run kernel on long z-runs (also with the pad clipped to a short z
        # extent), the voxel kernel on a one-slice image and on a checkerboard ROI. An
        # infinite value falls back to the exact kernel.
        from contextlib import ExitStack

        from pictologics.features import intensity as intensity_module
        from pictologics.loader import Image

        names = ("_approx_runs_numba", "_approximate_local_means_numba")

        def kernel_calls(
            data: np.ndarray, mask: np.ndarray, spacing: tuple[float, float, float]
        ) -> list[int]:
            image = Image(data, spacing, (0.0, 0.0, 0.0))
            roi = Image(mask, spacing, (0.0, 0.0, 0.0))
            with patch.object(intensity_module, "_TWO_STAGE_MIN_WORK", 1 << 62):
                expected = calculate_local_intensity_features(image, roi)
            with ExitStack() as stack:
                stack.enter_context(patch.object(intensity_module, "_TWO_STAGE_MIN_WORK", 1))
                spies = {
                    name: stack.enter_context(
                        patch.object(intensity_module, name, wraps=getattr(intensity_module, name))
                    )
                    for kernel in names
                    for name in (kernel, kernel + "_serial")
                }
                self.assertEqual(calculate_local_intensity_features(image, roi), expected)
            return [spies[k].call_count + spies[k + "_serial"].call_count for k in names]

        rng = np.random.default_rng(13)
        data = rng.normal(50.0, 20.0, (12, 13, 14))
        data[6, 6, 6] = data.max() + 1.0  # one brightest voxel
        mask = np.zeros(data.shape, dtype=np.uint8)
        mask[2:10, 3:11, 2:12] = 1
        self.assertEqual(kernel_calls(data, mask, (2.0, 2.5, 3.0)), [1, 0])  # z-runs of 10
        thin = np.zeros((12, 13, 10), dtype=np.uint8)
        thin[2:10, 3:11] = 1
        self.assertEqual(kernel_calls(data[:, :, :10], thin, (2.0, 2.5, 0.5)), [1, 0])  # clip
        one = np.zeros((12, 13, 1), dtype=np.uint8)
        one[3:9, 4:10] = 1
        self.assertEqual(kernel_calls(data[:, :, :1], one, (2.0, 2.0, 2.0)), [0, 1])  # no pad
        board = (np.indices(data.shape).sum(axis=0) % 2).astype(np.uint8)
        self.assertEqual(kernel_calls(data, board, (2.0, 2.5, 3.0)), [0, 1])  # z-runs of 1
        infinite = np.where(mask > 0, np.inf, data)
        self.assertEqual(kernel_calls(infinite, mask, (2.0, 2.5, 3.0)), [0, 0])  # fallback

    def test_roi_scan(self) -> None:
        # The ROI scan gives the ROI voxels in row order (the order of np.argwhere), their
        # values as float64 and the z-runs that cover them; an empty mask gives empty arrays.
        from pictologics.features import intensity as intensity_module

        rng = np.random.default_rng(15)
        data = rng.normal(size=(7, 8, 9))
        mask = rng.random(data.shape) > 0.5
        mask[3, 4, :] = True  # one run over the whole z extent
        for roi_data in (data, data.astype(np.float32)):
            indices, values, runs = intensity_module._roi_scan(mask, roi_data)
            self.assertEqual(indices.dtype, np.int32)
            np.testing.assert_array_equal(indices, np.argwhere(mask))
            np.testing.assert_array_equal(values, roi_data[mask].astype(np.float64))
            covered = np.zeros(data.shape, dtype=bool)
            for x, y, z0, z1, i in runs:
                self.assertTrue(mask[x, y, z0:z1].all())
                self.assertFalse(z0 > 0 and mask[x, y, z0 - 1])
                self.assertFalse(z1 < data.shape[2] and mask[x, y, z1])
                np.testing.assert_array_equal(indices[i : i + z1 - z0, 2], np.arange(z0, z1))
                covered[x, y, z0:z1] = True
            np.testing.assert_array_equal(covered, mask)
        empty = intensity_module._roi_scan(np.zeros((3, 4, 5), dtype=bool), data[:3, :4, :5])
        self.assertEqual([part.shape for part in empty], [(0, 3), (0,), (0, 5)])

    def test_histogram_kernels_read_float64_values_only(self) -> None:
        # Integer values take their histogram features from the bin counts, with no kernel
        # call (also uint64, which bincount does not read, goes to the kernels); the
        # kernels read float64 (the type that the import compiles): another float type goes
        # as a float64 copy.
        from pictologics.features import intensity as intensity_module

        values = np.array([1, 2, 2, 3, 5, 5, 5, 4])
        kernel = intensity_module._mean_abs_dev
        with patch.object(intensity_module, "_mean_abs_dev", wraps=kernel) as spy:
            for kind in (np.uint8, np.int16, np.int32, np.int64, np.float32, np.float64):
                calculate_intensity_histogram_features(values.astype(kind))
            self.assertEqual([call.args[0].dtype for call in spy.call_args_list], [np.float64] * 4)
            calculate_intensity_histogram_features(values.astype(np.uint64))
        self.assertEqual(spy.call_count, 6)

    def test_local_kernels_read_float64_or_float32_images(self) -> None:
        # The local kernels read float64 or float32, the types that the import compiles: an
        # image of another type goes as a row-order float64 copy (also from a column-order
        # image), so with the features of float64.
        from pictologics.features import intensity as intensity_module
        from pictologics.loader import Image

        data = np.random.default_rng(14).integers(-100, 100, (12, 13, 14))
        mask = np.zeros(data.shape, dtype=np.uint8)
        mask[2:10, 3:11, 2:12] = 1
        spacing, origin = (2.0, 2.0, 2.0), (0.0, 0.0, 0.0)
        roi = Image(mask, spacing, origin)
        expected = calculate_local_intensity_features(Image(data * 1.0, spacing, origin), roi)
        with patch.object(intensity_module, "_roi_scan", wraps=intensity_module._roi_scan) as spy:
            image = Image(data.astype(np.int16), spacing, origin)
            self.assertEqual(calculate_local_intensity_features(image, roi), expected)
            column = Image(np.asfortranarray(data.astype(np.int16)), spacing, origin)
            self.assertEqual(calculate_local_intensity_features(column, roi), expected)
            calculate_local_intensity_features(Image(data.astype(np.float32), spacing, origin), roi)
        kinds = [call.args[1].dtype for call in spy.call_args_list]
        self.assertEqual(kinds, [np.float64, np.float64, np.float32])
        self.assertTrue(all(call.args[1].flags.c_contiguous for call in spy.call_args_list))


# The histogram features that add the same terms in another order from the bin counts
_NEAR_HISTOGRAM = ("_CH89", "_88K1", "_C3I7", "_CWYJ", "_D2ZX", "_WRZB", "_4RNL")


def test_histogram_features_of_integers_come_from_the_counts() -> None:
    # Integer values take the histogram features from their bin counts, and so do float64
    # whole numbers with n_bins: with n_bins, every type gives the same bits. Without
    # n_bins, float64 values go to the kernels, which give the same features bit for bit,
    # but for the variance, skewness, kurtosis, coefficient of variation and the three
    # absolute deviations: these add the same terms in another order (relative difference
    # 1e-12 or less; skewness and kurtosis near 0 within 1e-12).
    rng = np.random.default_rng(21)
    sizes = [300_000, 1, 2, 9, *np.exp(rng.uniform(0.0, np.log(20_000), 56)).astype(int)]
    for case, n in enumerate(sizes):
        kind = (np.int32, np.uint8, np.int64)[case % 3]
        levels = int(rng.integers(1, 256 if kind == np.uint8 else 401))
        values = rng.integers(1, levels + 1, n)
        if case == 2:  # one level
            values[:] = levels
        if case == 3:  # levels 1, 2 and 3, three times each: the mean is the middle level
            values, levels = np.repeat([1, 2, 3], 3), 3
        for n_bins in (levels, None):
            got = calculate_intensity_histogram_features(values.astype(kind), n_bins=n_bins)
            expected = calculate_intensity_histogram_features(values.astype(np.float64), n_bins)
            assert got.keys() == expected.keys()
            for key, value in expected.items():
                if n_bins is None and key.endswith(_NEAR_HISTOGRAM):
                    assert np.isclose(got[key], value, rtol=1e-12, atol=1e-12, equal_nan=True)
                else:
                    assert np.float64(got[key]).tobytes() == np.float64(value).tobytes(), key


def test_histogram_values_outside_the_levels_raise() -> None:
    # Integer and float64 whole values outside [1, n_bins] raise the error of the float path,
    # with their range. The level count clamps them, so no table of their size is made (no
    # bincount).
    from pictologics.features import intensity as intensity_module

    cases = (
        ([0, 2], r"\[0, 2\]"),
        ([-3, 2], r"\[-3, 2\]"),
        ([1, 9], r"\[1, 9\]"),
        ([1, 50_000_000], r"\[1, 50000000\]"),
    )
    for values, text in cases:
        for kind in (np.int32, np.float64):
            with (
                patch.object(intensity_module.np, "bincount", wraps=np.bincount) as counts,
                pytest.raises(ValueError, match=f"must lie in \\[1, n_bins=8\\]; got range {text}"),
            ):
                calculate_intensity_histogram_features(np.array(values, dtype=kind), n_bins=8)
            assert counts.call_count == 0
    with pytest.raises(ValueError, match="finite whole numbers"):  # not whole: the float path
        calculate_intensity_histogram_features(np.array([0.5, 2.0]), n_bins=8)


def test_histogram_pairwise_sum_gives_the_bits_of_np_sum() -> None:
    # The pairwise sum of the statistics kernel adds in the order of np.sum (a plain loop
    # below 8 terms, 8 partial sums up to 128, then halves), so it gives numpy's bits: any
    # length, mixed magnitudes, positive terms, zeros, and a part of an array.
    from pictologics.features import intensity as intensity_module

    rng = np.random.default_rng(22)
    for n in (*range(300), 1000, 4095, 4096, 100_001):
        mixed = rng.normal(0.0, 1.0, n) * 10.0 ** rng.integers(-8, 9, n)
        zeros = np.where(rng.random(n) < 1 / 3, 0.0, np.abs(mixed))
        for terms in (mixed, np.abs(mixed), zeros):
            got = intensity_module._pairwise_sum_numba(terms, 0, n)
            assert np.float64(got).tobytes() == np.sum(terms).tobytes()
        if n > 5:
            got = intensity_module._pairwise_sum_numba(mixed, 5, n)
            assert np.float64(got).tobytes() == np.sum(mixed[5:n]).tobytes()


def test_histogram_rank_rule_matches_percentile_ranks() -> None:
    # The rank of the statistics kernel is the index rule of _percentile_ranks.
    from pictologics.features import intensity as intensity_module

    quantiles = np.true_divide([10, 25, 75, 90], 100)
    for n in (*range(1, 401), 10**6, 2**31):
        ranks = intensity_module._percentile_ranks(n)
        assert [intensity_module._rank_numba(n, q) for q in quantiles] == list(ranks)


def test_histogram_level_counts() -> None:
    # The level count puts a value below 0 into level 0 and one above n_bins into level
    # n_bins, keeps the true smallest and largest value, and flags whole numbers: a 2.5
    # stops the count. A NaN (the compiled kernels) gives False, with no error. The threads
    # kernel gives, part by part, the counts of the serial kernel.
    import numba

    from pictologics.features import intensity as intensity_module

    for kind in (np.int32, np.int64, np.float64):
        values = np.array([-3, 0, 1, 5, 10], dtype=kind)
        counts, low, high, whole = intensity_module._level_counts_numba(values, 5)
        assert list(counts) == [2, 1, 0, 0, 0, 2] and (low, high, whole) == (-3, 10, True)
    rng = np.random.default_rng(24)
    values = rng.integers(-2, 12, 1001)
    counts, low, high, whole = intensity_module._level_counts_numba(values, 9)
    np.testing.assert_array_equal(counts, np.bincount(np.clip(values, 0, 9), minlength=10))
    assert (low, high, whole) == (values.min(), values.max(), True)
    stop = np.array([1.0, 2.0, 2.5, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    counts, _, _, whole = intensity_module._level_counts_numba(stop, 4)
    assert not whole and counts.sum() < stop.size
    parts = np.empty((3, 10), dtype=np.int64)
    lows, highs, wholes = np.empty(3, np.int64), np.empty(3, np.int64), np.empty(3, np.bool_)
    intensity_module._level_counts_threads_numba(values, 9, parts, lows, highs, wholes)
    np.testing.assert_array_equal(
        parts.sum(axis=0), np.bincount(np.clip(values, 0, 9), minlength=10)
    )
    edges = [0, 333, 667, 1001]
    assert list(lows) == [values[a:b].min() for a, b in zip(edges, edges[1:], strict=False)]
    assert list(highs) == [values[a:b].max() for a, b in zip(edges, edges[1:], strict=False)]
    assert wholes.all()
    with patch.object(numba.config, "DISABLE_JIT", False):  # int(NaN) is never read
        kernels = (
            intensity_module._level_counts_numba,
            intensity_module._level_counts_threads_numba,
        )
        serial, threads = (numba.njit(getattr(kernel, "py_func", kernel)) for kernel in kernels)
        with_nan = np.array([1.0, 2.0, np.nan, 3.0, 1.0, 2.0])
        assert not serial(with_nan, 4)[3]
        flags = np.empty(3, np.bool_)
        threads(with_nan, 4, np.empty((3, 5), np.int64), np.empty(3), np.empty(3), flags)
        assert list(flags) == [True, False, True]


def test_histogram_level_counts_in_threads() -> None:
    # From _PARALLEL_LEVEL_MIN values on, the level count runs in parts on threads, with the
    # result of the serial kernel: the extremes in different parts, a non-whole value in the
    # last part, a uint8 array (as an int64 copy). With one thread, the serial kernel runs.
    from pictologics.features import intensity as intensity_module

    rng = np.random.default_rng(25)
    with (
        patch.object(intensity_module, "_PARALLEL_LEVEL_MIN", 8),
        patch.object(intensity_module, "get_num_threads", return_value=3),
    ):
        for kind in (np.int32, np.int64, np.float64, np.uint8):
            values = rng.integers(1, 9, 30).astype(kind)
            values[2], values[25] = 0, 9  # the extremes in the first and the last part
            got = intensity_module._level_counts(values, 8)
            serial_values = values.astype(np.int64) if kind == np.uint8 else values
            expected = intensity_module._level_counts_numba(serial_values, 8)
            np.testing.assert_array_equal(got[0], expected[0])
            assert got[1:] == expected[1:] == (0, 9, True)
        values = rng.integers(1, 9, 30).astype(np.float64)
        values[28] = 2.5
        assert not intensity_module._level_counts(values, 8)[3]
    with (
        patch.object(intensity_module, "_PARALLEL_LEVEL_MIN", 8),
        patch.object(intensity_module, "get_num_threads", return_value=1),
        patch.object(intensity_module, "_level_counts_threads_numba") as threads,
    ):
        intensity_module._level_counts(values, 8)
    threads.assert_not_called()


def test_histogram_rejects_a_2d_array() -> None:
    # The histogram features read a 1-D array of levels; a 2-D array raises a clear error.
    for kind in (np.int32, np.float64):
        with pytest.raises(ValueError, match="1-D"):
            calculate_intensity_histogram_features(np.ones((3, 4), dtype=kind), n_bins=4)


def test_histogram_whole_float64_values_take_the_counted_path() -> None:
    # float64 whole numbers with n_bins take the counted path: the bits and the key order of
    # the int32 result. Values that are not whole go to the float path, which raises (0.7.1
    # truncated them: the mode and the entropy came from truncated counts).
    from pictologics.features import intensity as intensity_module

    levels = np.random.default_rng(23).integers(1, 17, 5000)
    expected = calculate_intensity_histogram_features(levels.astype(np.int32), n_bins=16)
    stats = intensity_module._counted_statistics_numba
    with patch.object(intensity_module, "_counted_statistics_numba", wraps=stats) as spy:
        got = calculate_intensity_histogram_features(levels.astype(np.float64), n_bins=16)
        assert spy.call_count == 1
        assert list(got) == list(expected)
        for key, value in expected.items():
            assert np.float64(got[key]).tobytes() == np.float64(value).tobytes(), key
        truncated = np.array([1.6, 1.6, 2.4, 2.4, 3.0])
        with pytest.raises(ValueError, match="finite whole numbers"):
            calculate_intensity_histogram_features(truncated, n_bins=4)
        assert spy.call_count == 1


if __name__ == "__main__":
    unittest.main()


def test_local_kernels_read_a_column_order_image_as_a_row_order_copy() -> None:
    # A small column-order image (no crop) goes to the local kernels as a row-order copy:
    # the same peaks, and the layout that the import compiles
    from unittest.mock import patch

    from pictologics.features import intensity as module
    from pictologics.loader import Image

    rng = np.random.default_rng(2)
    values = rng.normal(size=(12, 13, 14))
    mask = np.zeros(values.shape, dtype=np.uint8)
    mask[3:9, 4:10, 5:11] = 1
    row = Image(np.ascontiguousarray(values), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    column = Image(np.asfortranarray(values), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    roi = Image(mask, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    layouts = []
    real = module._calculate_local_mean_numba

    def spy(data, indices, offsets):
        layouts.append(bool(data.flags.c_contiguous))
        return real(data, indices, offsets)

    with (
        patch.object(module, "_calculate_local_mean_numba", spy),
        patch.object(module, "_calculate_local_mean_numba_serial", spy),  # few pairs
    ):
        assert module.calculate_local_intensity_features(column, roi) == (
            module.calculate_local_intensity_features(row, roi)
        )
    assert layouts == [True, True]


def test_percentile_ranks_are_numpys() -> None:
    # The ranks kernel gives the ranks of np.percentile(..., method="inverted_cdf"): the index
    # n * q - 1, one up unless it is whole, with numpy's float64 steps.
    from pictologics.features import intensity as intensity_module

    q = np.true_divide([10, 25, 75, 90], 100)
    for n in (*range(1, 301), 999, 1000, 4095, 65536, 123457, 2000001):
        index = n * q - 1
        below = np.floor(index)
        expected = np.where(index - below == 0, below, below + 1).astype(np.int64)
        ranks = intensity_module._percentile_ranks(n)
        assert ranks.dtype == np.int64 and np.array_equal(ranks, expected)


def _numpy_order_statistics(values: np.ndarray) -> bytes:
    """The bytes of np.percentile (inverted_cdf) at 10, 25, 75, 90 and np.median."""
    percentiles = np.percentile(values, [10, 25, 75, 90], method="inverted_cdf")
    return np.array([*percentiles, np.median(values)], dtype=float).tobytes()


def test_small_order_statistics_match_numpy() -> None:
    # Below 1,000 float64 values one partition in numba gives the bits of np.percentile
    # (inverted_cdf) and np.median: normal values, values with -inf and +inf at the ends, and
    # rounded values with -0.0 (where both zeros sit at a rank, the kernel gives way).
    from pictologics.features import intensity as intensity_module

    rng = np.random.default_rng(31)
    kernel = intensity_module._small_order_statistics_numba
    found: list[bool] = []

    def spy(values: np.ndarray) -> tuple[object, ...]:
        result = kernel(values)
        found.append(bool(result[0]))
        return result

    for n in (1, 2, 3, 4, 5, 7, 8, 9, 10, 16, 99, 100, 101, 500, 999):
        normal = rng.normal(40.0, 20.0, n)
        cases = [normal, np.round(rng.normal(0.0, 2.0, n))]
        if n >= 3:
            ends = normal.copy()
            ends[0], ends[-1] = -np.inf, np.inf
            cases.append(ends)
        for values in cases:
            found.clear()
            with patch.object(intensity_module, "_small_order_statistics_numba", side_effect=spy):
                got = intensity_module._order_statistics(values)
            assert np.array(got, dtype=float).tobytes() == _numpy_order_statistics(values)
            assert len(found) == 1 and (found[0] or values is cases[1])


def test_small_order_statistics_give_way() -> None:
    # With both zeros at a rank the kernel gives way, and the partition of numpy gives the
    # values; with both zeros away from the ranks the kernel answers. All -0.0: -0.0
    # percentiles and a +0.0 median. A NaN: five NaN. A strided view takes one copy; an
    # empty array raises the IndexError of the partition.
    import math

    from pictologics.features import intensity as intensity_module

    kernel = intensity_module._small_order_statistics_numba
    middle = np.array([-1.0, -0.0, 0.0, -0.0, 0.0, 2.0, 3.0])
    assert not kernel(middle)[0]
    got = intensity_module._order_statistics(middle)
    assert np.array(got, dtype=float).tobytes() == _numpy_order_statistics(middle)
    away = np.array([-0.0, 0.0, *range(1, 40)], dtype=float)
    assert kernel(away)[0]
    found, *picks, median = kernel(np.full(7, -0.0))
    assert found and all(math.copysign(1.0, v) < 0.0 for v in picks)
    assert math.copysign(1.0, median) > 0.0
    result = kernel(np.array([1.0, np.nan, 3.0]))
    assert result[0] and np.isnan(result[1:]).all()
    view = np.random.default_rng(32).normal(size=21)[::2]
    got = intensity_module._order_statistics(view)
    assert np.array(got, dtype=float).tobytes() == _numpy_order_statistics(view)
    with pytest.raises(IndexError):
        intensity_module._order_statistics(np.array([]))


def test_the_energy_sums_in_a_fixed_order() -> None:
    # The energy and the RMS come from a fixed-order sum of squares: within 8 ulps of
    # math.fsum for every length (the blocks, the tail, no values), and the same bits for
    # the same values at any buffer offset (np.dot blocks by the buffer address).
    import math

    from pictologics.features import intensity as intensity_module

    rng = np.random.default_rng(51)
    for n in (0, 1, 7, 8, 9, 4096, 4097):
        values = rng.normal(40.0, 20.0, n)
        exact = math.fsum(values * values)
        got = intensity_module._sum_of_squares(values)
        assert abs(got - exact) <= 8 * np.spacing(max(exact, 1.0))

    def at_offset(values: np.ndarray, k: int) -> np.ndarray:
        buffer = np.empty(values.size + 16)
        shift = ((k - buffer.ctypes.data % 64) // 8) % 8
        out = buffer[shift : shift + values.size]
        out[:] = values
        return out

    values = rng.normal(40.0, 20.0, 20_000)
    results = {
        (f["intensity_energy_N8CA"], f["root_mean_square_intensity_5ZWQ"])
        for f in (calculate_intensity_features(at_offset(values, 8 * k)) for k in range(8))
    }
    assert len(results) == 1


def test_histogram_values_must_be_finite_whole_numbers() -> None:
    # Values that are not whole, NaN or infinite raise a clear error (with and without
    # n_bins; the compiled level count finds a NaN with n_bins, its Python form raises
    # before). Whole values of other types (float32, float16, uint64, bool) sum in
    # float64: the features of the same values as float64, bit for bit.
    import numba

    from pictologics.features import intensity as intensity_module

    message = "finite whole numbers"
    for bad in ([1.0, 2.5, 3.0], [1.0, np.nan, 3.0], [1.0, np.inf, 3.0]):
        with pytest.raises(ValueError, match=message):
            calculate_intensity_histogram_features(np.array(bad))
    with pytest.raises(ValueError, match=message):
        calculate_intensity_histogram_features(np.array([1.0, 2.5, 3.0]), n_bins=4)
    with (
        patch.object(numba.config, "DISABLE_JIT", False),  # the kernel compiles at its call
        patch.object(
            intensity_module,
            "_level_counts_numba",
            numba.njit(intensity_module._level_counts_numba),
        ),
    ):
        for bad in ([1.0, np.nan, 3.0], [1.0, np.inf, 3.0]):
            with pytest.raises(ValueError, match=message):
                calculate_intensity_histogram_features(np.array(bad), n_bins=4)
    levels = np.random.default_rng(52).integers(1, 9, 300).astype(np.float64)
    expected = calculate_intensity_histogram_features(levels)
    for kind in (np.float32, np.float16, np.uint64):
        got = calculate_intensity_histogram_features(levels.astype(kind))
        assert list(got) == list(expected)
        for key, value in expected.items():
            assert np.float64(got[key]).tobytes() == np.float64(value).tobytes(), (kind, key)
    flags = calculate_intensity_histogram_features(levels > 4)
    assert flags["mean_discretised_intensity_X6K6"] == float(np.mean(levels > 4))
    # With n_bins, float32 values take the float path over the bins 1..n_bins, as float16
    with_bins = calculate_intensity_histogram_features(levels.astype(np.float32), n_bins=10)
    assert with_bins == calculate_intensity_histogram_features(levels.astype(np.float16), n_bins=10)
    with pytest.raises(ValueError, match=r"got range \[1, 8\]"):
        calculate_intensity_histogram_features(levels.astype(np.float32), n_bins=4)


def test_local_peaks_keep_two_stages_with_an_unreached_nan() -> None:
    # A non-finite voxel that no sphere of an ROI voxel reaches is set to 0 in a copy, and
    # the two stages give the peaks of the exact kernel bit for bit; one in reach, or more
    # than _NON_FINITE_MAX of them, gives way to the exact kernel; finite images run as
    # before.
    from pictologics.features import intensity as intensity_module
    from pictologics.loader import Image

    rng = np.random.default_rng(53)
    data = rng.normal(40.0, 20.0, (6, 6, 6))
    mask = np.zeros(data.shape, dtype=np.uint8)
    mask[0:2, 0:2, 0:2] = 1
    spacing = (4.0, 4.0, 4.0)  # the sphere reaches one voxel on each axis
    roi = Image(mask, spacing, (0.0, 0.0, 0.0))

    def peaks(array: np.ndarray, gate: int) -> tuple[dict[str, float], list[object]]:
        image = Image(array, spacing, (0.0, 0.0, 0.0))
        two_stage = intensity_module._local_peaks_two_stage
        with (
            patch.object(intensity_module, "_TWO_STAGE_MIN_WORK", gate),
            patch.object(intensity_module, "_local_peaks_two_stage", wraps=two_stage) as spy,
        ):
            features = calculate_local_intensity_features(image, roi)
        return features, [call for call in spy.call_args_list]

    far, near = data.copy(), data.copy()
    far[5, 5, 5] = np.nan
    near[2, 1, 1] = np.inf
    for array in (data, far):
        exact, _ = peaks(array, 1 << 62)
        got, calls = peaks(array, 0)
        assert got == exact and len(calls) == 1
    offsets = intensity_module._sphere_offsets_for_radius(spacing, 6.2035)
    assert intensity_module._without_unreached(far, mask != 0, offsets) is not None
    assert intensity_module._without_unreached(near, mask != 0, offsets) is None
    with patch.object(intensity_module, "_NON_FINITE_MAX", 0):
        assert intensity_module._without_unreached(far, mask != 0, offsets) is None
    exact, _ = peaks(near, 1 << 62)
    assert peaks(near, 0)[0] == exact  # the exact kernel on every voxel
