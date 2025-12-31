import os
import unittest
import warnings

# Disable Numba JIT global optimization for coverage analysis
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Suppress the "NumPy module was reloaded" warning that can occur due to Numba + testing environment
# Must be done BEFORE importing numpy
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

from unittest.mock import PropertyMock, patch

import numpy as np

# Import the module under test
import pictologics.features.texture as texture_module


class TestTextureFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (5, 5, 5)
        self.data = np.random.randint(1, 5, self.shape)
        self.mask = np.ones(self.shape, dtype=int)
        self.n_bins = 16
        texture_module._ZoneBufferPool._instance = None

    def test_calculate_all_matrices_basic(self):
        matrices = texture_module.calculate_all_texture_matrices(
            self.data, self.mask, self.n_bins
        )
        self.assertIn("glcm", matrices)

    def test_max_zones_less_than_one(self):
        mask_empty = np.zeros(self.shape, dtype=int)
        res = texture_module.calculate_zone_features(
            self.data, mask_empty, self.data, self.n_bins
        )
        self.assertEqual(np.sum(res[0]), 0)

    def test_glszm_uint8_mask_optimization(self):
        mask_u8 = self.mask.astype(np.uint8)
        f = texture_module.calculate_glszm_features(self.data, mask_u8, self.n_bins)
        self.assertIn("small_zone_emphasis_P001", f)

    def test_empty_matrices_returns(self):
        empty_szm = np.zeros((self.n_bins, 1), dtype=np.uint32)
        f_szm = texture_module.calculate_glszm_features(
            self.data, self.mask, self.n_bins, glszm_matrix=empty_szm
        )
        self.assertEqual(f_szm, {})

        empty_dzm = np.zeros((self.n_bins, 1), dtype=np.uint32)
        f_dzm = texture_module.calculate_gldzm_features(
            self.data, self.mask, self.n_bins, gldzm_matrix=empty_dzm
        )
        self.assertEqual(f_dzm, {})

        empty_ngldm = np.zeros((self.n_bins, 5), dtype=np.uint64)
        f_ngldm = texture_module.calculate_ngldm_features(
            self.data, self.mask, self.n_bins, ngldm_matrix=empty_ngldm
        )
        self.assertEqual(f_ngldm, {})

    def test_ngtdm_zero_denominators(self):
        n_bins = 2
        s = np.zeros(n_bins, dtype=float)
        n = np.array([10.0, 10.0])
        f = texture_module.calculate_ngtdm_features(
            self.data, self.mask, n_bins, ngtdm_matrices=(s, n)
        )
        self.assertEqual(f["coarseness_QCDE"], 1000000.0)

        s2 = np.array([1.0, 0.0])
        n2 = np.array([1.0, 0.0])
        f2 = texture_module.calculate_ngtdm_features(
            self.data, self.mask, n_bins, ngtdm_matrices=(s2, n2)
        )
        self.assertEqual(f2["busyness_NQ30"], 0.0)

    def test_ngtdm_zero_sum_matrices(self):
        # Trigger NGTDM N_vp == 0 exit
        n_bins = 2
        s = np.zeros(n_bins, dtype=float)
        n = np.zeros(n_bins, dtype=float)  # Sum is 0
        f = texture_module.calculate_ngtdm_features(
            self.data, self.mask, n_bins, ngtdm_matrices=(s, n)
        )
        self.assertEqual(f, {})

    def test_ngtdm_ngldm_casting_and_threads(self):
        with patch("pictologics.features.texture.numba.config") as mock_config:
            type(mock_config).NUMBA_NUM_THREADS = PropertyMock(
                side_effect=AttributeError
            )
            texture_module.calculate_ngtdm_features(self.data, self.mask, self.n_bins)
            texture_module.calculate_ngldm_features(self.data, self.mask, self.n_bins)

        n_bins_32 = 1000
        data_32 = np.random.randint(1, n_bins_32, self.shape)
        texture_module.calculate_ngtdm_features(data_32, self.mask, n_bins_32)
        texture_module.calculate_ngldm_features(data_32, self.mask, n_bins_32)

        n_bins_16 = 1000
        data_16 = np.random.randint(1, n_bins_16, self.shape)
        texture_module.calculate_ngtdm_features(data_16, self.mask, n_bins_16)
        texture_module.calculate_ngldm_features(data_16, self.mask, n_bins_16)

    def test_missing_coverage_branches(self):
        with self.assertRaises(ValueError):
            texture_module._compute_nonzero_bbox(np.zeros((5, 5), dtype=int))

        glcm = np.zeros((13, 2, 2), dtype=float)
        glcm[0, 0, 0] = 1.0
        empty_mask = np.zeros((2, 2, 2), dtype=int)
        f = texture_module.calculate_glcm_features(
            np.zeros((2, 2, 2), dtype=int), empty_mask, n_bins=2, glcm_matrix=glcm
        )
        self.assertIn("normalised_inverse_difference_NDRX", f)

        with self.assertRaises(ValueError):
            texture_module._maybe_crop_to_bbox(
                self.data, self.mask, distance_mask=np.zeros((2, 2, 2), dtype=int)
            )

        with self.assertRaises(ValueError):
            texture_module._maybe_crop_to_bbox(np.zeros((3, 3, 3)), np.zeros((2, 2, 2)))

        self.assertIsNone(
            texture_module._compute_nonzero_bbox(np.zeros(self.shape, dtype=int))
        )

        data_orig, mask_orig, dist_orig = texture_module._maybe_crop_to_bbox(
            self.data, np.zeros(self.shape, dtype=int), None
        )
        np.testing.assert_array_equal(data_orig, self.data)
        np.testing.assert_array_equal(mask_orig, np.zeros(self.shape, dtype=int))

    def test_safe_vs_unsafe_offset_logic(self):
        shape_small = (2, 2, 2)
        data_small = np.ones(shape_small, dtype=int)
        mask_small = np.ones(shape_small, dtype=int)
        texture_module.calculate_all_texture_matrices(
            data_small, mask_small, self.n_bins
        )

    def test_zone_features_buffer_pool_resize(self):
        mask_small = np.zeros(self.shape, dtype=int)
        mask_small[2, 2, 2] = 1
        texture_module.calculate_zone_features(
            self.data, mask_small, self.data, self.n_bins
        )

        mask_large = np.ones(self.shape, dtype=int)
        texture_module.calculate_zone_features(
            self.data, mask_large, self.data, self.n_bins
        )

        # Reuse path
        texture_module.calculate_zone_features(
            self.data, mask_small, self.data, self.n_bins
        )

    def test_glrlm_boundary_conditions(self):
        data = np.zeros((3, 3, 5), dtype=int)
        data[1, 1, :] = 1
        mask = np.zeros((3, 3, 5), dtype=int)
        mask[1, 1, :] = 1
        f = texture_module.calculate_glrlm_features(data, mask, n_bins=2)
        self.assertIn("short_runs_emphasis_22OV", f)

    def test_empty_roi_fast_exit(self):
        mask_empty = np.zeros(self.shape, dtype=int)
        matrices = texture_module.calculate_all_texture_matrices(
            self.data, mask_empty, self.n_bins
        )
        self.assertEqual(np.sum(matrices["glcm"]), 0)

    def test_individual_feature_calculators(self):
        texture_module.calculate_glcm_features(self.data, self.mask, self.n_bins)
        texture_module.calculate_glrlm_features(self.data, self.mask, self.n_bins)
        texture_module.calculate_gldzm_features(self.data, self.mask, self.n_bins)

    def test_calculate_all_features_wrapper(self):
        f = texture_module.calculate_all_texture_features(
            self.data, self.mask, self.n_bins
        )
        self.assertIn("joint_maximum_GYBY", f)

    def test_crop_with_disjoint_distance_mask(self):
        mask = np.zeros(self.shape, dtype=int)
        mask[0, 0, 0] = 1
        d_mask = np.zeros(self.shape, dtype=int)
        d_mask[0, 0, 0] = 1
        d_mask[4, 4, 4] = 1

        # Direct call
        d_c, m_c, dist_c = texture_module._maybe_crop_to_bbox(self.data, mask, d_mask)
        self.assertEqual(d_c.shape, (5, 5, 5))
        self.assertIsNotNone(dist_c)

        f = texture_module.calculate_gldzm_features(
            self.data, mask, self.n_bins, distance_mask=d_mask
        )
        self.assertIn("zone_distance_non_uniformity_V294", f)

    def test_invalid_bin_value_in_roi(self):
        data_bad = np.zeros(self.shape, dtype=int)
        mask = np.zeros(self.shape, dtype=int)
        mask[2, 2, 2] = 1
        data_bad[2, 2, 2] = self.n_bins + 10
        m = texture_module.calculate_all_texture_matrices(data_bad, mask, self.n_bins)
        self.assertEqual(np.sum(m["glcm"]), 0)

    def test_glrlm_safe_path_mask_toggle(self):
        mask = np.zeros(self.shape, dtype=int)
        mask[2, 2, 2] = 1
        mask[2, 2, 3] = 1
        f = texture_module.calculate_glrlm_features(self.data, mask, self.n_bins)
        self.assertIn("short_runs_emphasis_22OV", f)

    def test_calculate_all_uint8_mask(self):
        mask_u8 = self.mask.astype(np.uint8)
        m = texture_module.calculate_all_texture_matrices(
            self.data, mask_u8, self.n_bins
        )
        self.assertIn("glcm", m)

    def test_calculate_all_medium_bins(self):
        n_bins = 300
        data = np.random.randint(1, n_bins, self.shape)
        m = texture_module.calculate_all_texture_matrices(data, self.mask, n_bins)
        self.assertIn("glcm", m)

    def test_numba_thread_config_value_error(self):
        with patch("pictologics.features.texture.numba.config") as mock_config:
            # Setting it to a non-integer string causes ValueError in int()
            type(mock_config).NUMBA_NUM_THREADS = PropertyMock(return_value="invalid")

            # 1. Main wrapper (Lines 703-704 check)
            m = texture_module.calculate_all_texture_matrices(
                self.data, self.mask, self.n_bins
            )
            self.assertIn("glcm", m)

            # 2. NGTDM (Lines 1654-1655 check)
            texture_module.calculate_ngtdm_features(self.data, self.mask, self.n_bins)

            # 3. NGLDM (Lines 1795-1796 check)
            texture_module.calculate_ngldm_features(self.data, self.mask, self.n_bins)

    def test_glcm_coverage_combinations(self):
        # Hits lines 820 (uint8 mask), 827 (high bins), 832-833 (thread config error in GLCM)
        # 1. uint8 mask
        mask_u8 = self.mask.astype(np.uint8)

        # 2. n_bins > 256 for int32 cast path (line 827)
        n_bins = 300
        data = np.random.randint(1, n_bins, self.shape)

        # 3. Thread config error (line 832-833)
        with patch("pictologics.features.texture.numba.config") as mock_config:
            type(mock_config).NUMBA_NUM_THREADS = PropertyMock(return_value="invalid")

            # This calls calculate_glcm_features which has the specific try/except block
            f = texture_module.calculate_glcm_features(data, mask_u8, n_bins)
            self.assertIn("contrast_ACUI", f)

    def test_glcm_zero_sum(self):
        # Line 859: if total_sum == 0
        glcm = np.zeros((13, 2, 2), dtype=float)
        f = texture_module.calculate_glcm_features(
            self.data, self.mask, n_bins=2, glcm_matrix=glcm
        )
        self.assertEqual(f, {})

    def test_glrlm_fallback_coverage(self):
        # 1. Thread config fallback in GLRLM (lines 1037-1040)
        with patch("pictologics.features.texture.numba.config") as mock_config:
            type(mock_config).NUMBA_NUM_THREADS = PropertyMock(return_value="invalid")
            f = texture_module.calculate_glrlm_features(
                self.data, self.mask, self.n_bins
            )
            self.assertIn("short_runs_emphasis_22OV", f)

    def test_glrlm_high_bitdepth_coverage(self):
        # Hits lines 1031-1034 (casting logic in calculate_glrlm_features)

        # 1. uint16 path (256 < n_bins <= 65536)
        n_bins_16 = 300
        data_16 = np.random.randint(1, n_bins_16, self.shape)
        f16 = texture_module.calculate_glrlm_features(data_16, self.mask, n_bins_16)
        self.assertIn("short_runs_emphasis_22OV", f16)

        # 2. int32 path (n_bins > 65536)
        n_bins_32 = 1000
        data_32 = np.random.randint(1, n_bins_32, self.shape)
        # GLRLM matrix is (n_bins, max_run_length).
        # For shape (5,5,5), max run is 5.
        # 70000 * 5 * 8 bytes ~ 2.8 MB. Safe.
        f32 = texture_module.calculate_glrlm_features(data_32, self.mask, n_bins_32)
        self.assertIn("short_runs_emphasis_22OV", f32)

    def test_remaining_coverage_lines(self):
        # 1. GLRLM N_runs == 0 (Line 1065)
        # Manually pass zero matrix. code expects 3D (directions, bins, runs) to sum axis 0
        glrlm_zero = np.zeros((1, 16, 5), dtype=int)
        f_glrlm = texture_module.calculate_glrlm_features(
            self.data, self.mask, 16, glrlm_matrix=glrlm_zero
        )
        self.assertEqual(f_glrlm, {})

        # 2. GLDZM max_dist_val == 0 (Line 1300)
        # Force distance map to 0 so all zones have 0 distance
        # Zones with distance 0 are excluded from matrix population (d > 0 check)
        # So matrix remains empty -> N_zones=0 -> returns {}
        dist_map_zero = np.zeros(self.shape, dtype=int)
        f_gldzm = texture_module.calculate_gldzm_features(
            self.data, self.mask, self.n_bins, distance_mask=dist_map_zero
        )
        self.assertEqual(f_gldzm, {})

        # 3. GLSZM mask.dtype != uint8 (Line 1392)
        # Use int64 mask
        mask_int64 = self.mask.astype(int)
        self.assertNotEqual(mask_int64.dtype, np.uint8)
        f_glszm = texture_module.calculate_glszm_features(
            self.data, mask_int64, self.n_bins
        )
        self.assertIn("small_zone_emphasis_P001", f_glszm)

    def test_gldzm_min_dist_update(self):
        # Line 1251: min_dist = d (if d < min_dist)
        # We need a zone where the first voxel visited (linear index) has a
        # higher distance than a later voxel in the same zone.
        # Shape 5x5x5 all ones mask.
        # Center (2,2,2) has taxicab dist 3.
        # Neighbor (2,2,3) has taxicab dist 2.
        # (2,2,2) comes before (2,2,3) in linear order.

        data = np.ones(self.shape, dtype=int)
        data[2, 2, 2] = 2
        data[2, 2, 3] = 2

        # This will compute GLDZM for GL2.
        # Seed (2,2,2) dist=3. Neighbor (2,2,3) dist=2. Update triggers.
        f = texture_module.calculate_gldzm_features(data, self.mask, self.n_bins)
        # Check that we got a result (feature exists)
        self.assertIn("small_distance_emphasis_0GBI", f)


if __name__ == "__main__":
    unittest.main()
