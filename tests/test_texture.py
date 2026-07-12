# ruff: noqa: E402
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
        matrices = texture_module.calculate_all_texture_matrices(self.data, self.mask, self.n_bins)
        self.assertIn("glcm", matrices)

    def test_max_zones_less_than_one(self):
        mask_empty = np.zeros(self.shape, dtype=int)
        res = texture_module.calculate_zone_features(self.data, mask_empty, self.data, self.n_bins)
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
            type(mock_config).NUMBA_NUM_THREADS = PropertyMock(side_effect=AttributeError)
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
            texture_module.compute_nonzero_bbox(np.zeros((5, 5), dtype=int))

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

        self.assertIsNone(texture_module.compute_nonzero_bbox(np.zeros(self.shape, dtype=int)))

        data_orig, mask_orig, dist_orig = texture_module._maybe_crop_to_bbox(
            self.data, np.zeros(self.shape, dtype=int), None
        )
        np.testing.assert_array_equal(data_orig, self.data)
        np.testing.assert_array_equal(mask_orig, np.zeros(self.shape, dtype=int))

    def test_safe_vs_unsafe_offset_logic(self):
        shape_small = (2, 2, 2)
        data_small = np.ones(shape_small, dtype=int)
        mask_small = np.ones(shape_small, dtype=int)
        texture_module.calculate_all_texture_matrices(data_small, mask_small, self.n_bins)

    def test_zone_features_buffer_pool_resize(self):
        mask_small = np.zeros(self.shape, dtype=int)
        mask_small[2, 2, 2] = 1
        texture_module.calculate_zone_features(self.data, mask_small, self.data, self.n_bins)

        mask_large = np.ones(self.shape, dtype=int)
        texture_module.calculate_zone_features(self.data, mask_large, self.data, self.n_bins)

        # Reuse path
        texture_module.calculate_zone_features(self.data, mask_small, self.data, self.n_bins)

    def test_glrlm_boundary_conditions(self):
        data = np.zeros((3, 3, 5), dtype=int)
        data[1, 1, :] = 1
        mask = np.zeros((3, 3, 5), dtype=int)
        mask[1, 1, :] = 1
        f = texture_module.calculate_glrlm_features(data, mask, n_bins=2)
        self.assertIn("short_runs_emphasis_22OV", f)

    def test_empty_roi_fast_exit(self):
        mask_empty = np.zeros(self.shape, dtype=int)
        matrices = texture_module.calculate_all_texture_matrices(self.data, mask_empty, self.n_bins)
        self.assertEqual(np.sum(matrices["glcm"]), 0)

    def test_individual_feature_calculators(self):
        texture_module.calculate_glcm_features(self.data, self.mask, self.n_bins)
        texture_module.calculate_glrlm_features(self.data, self.mask, self.n_bins)
        texture_module.calculate_gldzm_features(self.data, self.mask, self.n_bins)

    def test_calculate_all_features_wrapper(self):
        f = texture_module.calculate_all_texture_features(self.data, self.mask, self.n_bins)
        self.assertIn("joint_maximum_GYBY", f)

    def test_label_mask_values_are_membership_not_weights(self):
        label_mask = (self.mask * 3).astype(np.uint8)
        binary_features = texture_module.calculate_all_texture_features(
            self.data, self.mask, self.n_bins
        )
        label_features = texture_module.calculate_all_texture_features(
            self.data, label_mask, self.n_bins
        )

        for key in [
            "run_percentage_9ZK5",
            "zone_percentage_P30P",
            "zone_percentage_VIWW",
            "dependence_count_percentage_6XV8",
        ]:
            self.assertAlmostEqual(label_features[key], binary_features[key])

        binary_matrices = texture_module.calculate_all_texture_matrices(
            self.data, self.mask, self.n_bins
        )
        label_matrices = texture_module.calculate_all_texture_matrices(
            self.data, label_mask, self.n_bins
        )
        np.testing.assert_array_equal(label_matrices["glcm"], binary_matrices["glcm"])

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
        m = texture_module.calculate_all_texture_matrices(self.data, mask_u8, self.n_bins)
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
            m = texture_module.calculate_all_texture_matrices(self.data, self.mask, self.n_bins)
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
        f = texture_module.calculate_glcm_features(self.data, self.mask, n_bins=2, glcm_matrix=glcm)
        self.assertEqual(f, {})

    def test_glrlm_fallback_coverage(self):
        # 1. Thread config fallback in GLRLM (lines 1037-1040)
        with patch("pictologics.features.texture.numba.config") as mock_config:
            type(mock_config).NUMBA_NUM_THREADS = PropertyMock(return_value="invalid")
            f = texture_module.calculate_glrlm_features(self.data, self.mask, self.n_bins)
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
        f_glszm = texture_module.calculate_glszm_features(self.data, mask_int64, self.n_bins)
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

    # ------------------------------------------------------------------
    # Local-feature kernel branch coverage (interior gaps in the ROI)
    # ------------------------------------------------------------------
    def test_local_kernel_interior_empty_slice(self):
        """An interior all-background z-slice is skipped (it survives the bbox crop)."""
        data = np.ones((3, 5, 5), dtype=int)
        mask = np.zeros((3, 5, 5), dtype=int)
        mask[0] = 1
        mask[2] = 1  # z=1 is empty but z=0/z=2 keep it inside the ROI bbox
        m = texture_module.calculate_all_texture_matrices(data, mask, self.n_bins)
        self.assertIn("glcm", m)

    def test_local_kernel_interior_background_hole(self):
        """A background hole in the safe interior exercises the masked GLRLM branches:
        the skip-background return, the run-start-adjacent-to-background test, and the
        run-walk break when a run hits background before the image edge."""
        data = np.ones((5, 5, 5), dtype=int)
        mask = np.ones((5, 5, 5), dtype=int)
        mask[2, 2, 2] = 0
        m = texture_module.calculate_all_texture_matrices(data, mask, self.n_bins)
        self.assertIn("glrlm", m)

    def test_matrices_zone_only_no_gldzm(self):
        """Disabling every local family and GLDZM exercises the placeholder paths
        (zero local matrices; dummy distance array for the GLSZM-only zone call)."""
        m = texture_module.calculate_all_texture_matrices(
            self.data,
            self.mask,
            self.n_bins,
            calc_glcm=False,
            calc_glrlm=False,
            calc_ngtdm=False,
            calc_ngldm=False,
            calc_gldzm=False,
        )
        self.assertIn("glszm", m)

    # ------------------------------------------------------------------
    # Parallel zone kernel (n_chunks > 1) branch coverage
    # ------------------------------------------------------------------
    def _run_parallel_zone_kernel(
        self, data, mask, dist_map, n_chunks, calc_glszm=True, calc_gldzm=True
    ):
        texture_module._ZoneBufferPool._instance = None
        pool = texture_module._ZoneBufferPool.get_instance()
        max_zones = max(int(np.count_nonzero(mask)), 1)
        res_gl, res_size, res_dist, stack = pool.get_buffers(max_zones)
        return texture_module._calculate_zone_features_numba(
            data,
            mask,
            dist_map,
            self.n_bins,
            res_gl,
            res_size,
            res_dist,
            stack,
            n_chunks,
            calc_glszm,
            calc_gldzm,
        )

    def test_parallel_zone_kernel_merge(self):
        """A single-grey column spanning z is labelled per-chunk, then the boundary
        union-find merges the sub-zones into one zone (with a min-distance update)."""
        depth = 6
        data = np.ones((depth, 3, 3), dtype=int)
        mask = np.zeros((depth, 3, 3), dtype=int)
        mask[:, 1, 1] = 1
        # strictly decreasing distance so the per-chunk DFS updates min_dist
        dist_map = np.zeros((depth, 3, 3), dtype=np.int32)
        dist_map[:, 1, 1] = np.arange(depth, 0, -1)
        glszm, gldzm = self._run_parallel_zone_kernel(data, mask, dist_map, n_chunks=4)
        self.assertEqual(int(glszm.sum()), 1)  # the per-chunk sub-zones merged into one
        self.assertEqual(int(gldzm.sum()), 1)

    def test_uf_find_path_compression(self):
        """_uf_find flattens a multi-hop parent chain onto the root."""
        parent = np.array([0, 0, 1, 2, 3], dtype=np.int32)  # 4 -> 3 -> 2 -> 1 -> 0
        self.assertEqual(texture_module._uf_find(parent, 4), 0)
        self.assertEqual(parent[4], 0)
        self.assertEqual(parent[3], 0)

    def test_parallel_zone_kernel_edge_cases(self):
        depth = 4
        data = np.ones((depth, 3, 3), dtype=int)
        mask = np.zeros((depth, 3, 3), dtype=int)
        mask[:, 1, 1] = 1
        dist = np.zeros((depth, 3, 3), dtype=np.int32)

        # n_chunks < 1 clamps up to 1; n_chunks > depth clamps down to depth
        self._run_parallel_zone_kernel(data, mask, dist, n_chunks=0)
        self._run_parallel_zone_kernel(data, mask, dist, n_chunks=100)
        # calc_gldzm=False uses the dummy distance buffer
        self._run_parallel_zone_kernel(data, mask, dist, n_chunks=2, calc_gldzm=False)
        # all-zero distances -> max_dist_val falls back to 1
        glszm, _ = self._run_parallel_zone_kernel(data, mask, dist, n_chunks=2)
        self.assertEqual(int(glszm.sum()), 1)

        # ROI voxels with an out-of-range grey level are marked invalid and skipped
        data_bad = np.full((depth, 3, 3), self.n_bins + 5, dtype=int)
        glszm_bad, _ = self._run_parallel_zone_kernel(data_bad, mask, dist, n_chunks=2)
        self.assertEqual(int(glszm_bad.sum()), 0)

    def test_parallel_zone_merge_attach_higher_root(self):
        """Two chunk-0 zones both touching one chunk-1 zone force the union to attach a
        higher-id root onto a lower one (the `ra > rb` branch of the boundary merge)."""
        mask = np.zeros((2, 4, 4), dtype=int)
        mask[0, 0, 0] = 1  # chunk-0 zone A (lowest id)
        mask[0, 2, 2] = 1  # chunk-0 zone B (higher id, not adjacent to A)
        mask[1, 1, 1] = 1  # chunk-1 zone C, 26-adjacent to both A and B
        data = np.ones((2, 4, 4), dtype=int)
        dist = np.zeros((2, 4, 4), dtype=np.int32)
        glszm, _ = self._run_parallel_zone_kernel(data, mask, dist, n_chunks=2)
        self.assertEqual(int(glszm.sum()), 1)  # A, B and C all merge into one zone

    def test_parallel_zone_dispatch(self):
        """>= 2^17 voxels with >1 thread routes calculate_zone_features to the parallel kernel."""
        with patch("pictologics.features.texture.numba.config.NUMBA_NUM_THREADS", 4):
            shape = (32, 64, 64)  # 131072 == 2^17
            data = np.ones(shape, dtype=int)
            mask = np.zeros(shape, dtype=int)
            mask[:, 0, 0] = 1
            dist_map = np.zeros(shape, dtype=np.int32)
            glszm, _ = texture_module.calculate_zone_features(data, mask, dist_map, self.n_bins)
            self.assertEqual(int(glszm.sum()), 1)


class TestRoiVoxelCount(unittest.TestCase):
    """_roi_voxel_count returns the same nonzero count across mask dtypes."""

    def test_dtype_gate_matches_count_nonzero(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.integers(0, 2, size=(6, 6, 6))
        # Float path (dtype.kind == "f"): NaN/Inf count as ROI, -0.0 and 0.0 do not.
        f = base.astype(np.float64)
        f.flat[:4] = [np.nan, np.inf, -0.0, 0.0]
        self.assertEqual(texture_module._roi_voxel_count(f), int(np.count_nonzero(f)))
        # Non-float masks keep the direct count path.
        for m in (base.astype(np.int32), base.astype(np.uint8), base.astype(bool)):
            self.assertEqual(texture_module._roi_voxel_count(m), int(np.count_nonzero(m)))


class TestGldzmDistanceMap(unittest.TestCase):
    """`_gldzm_distance_map` (the GLDZM distance-to-border map) matches the reference
    `scipy.ndimage.distance_transform_cdt` computation it replaces for 3D masks, and
    falls back to that same scipy-based computation for non-3D input."""

    @staticmethod
    def _scipy_reference(mask_bool: np.ndarray) -> np.ndarray:
        from scipy.ndimage import distance_transform_cdt

        mask_padded = np.pad(mask_bool, 1, mode="constant", constant_values=0)
        dist_map_padded = distance_transform_cdt(mask_padded, metric="taxicab").astype(np.int32)
        unpad = tuple(slice(1, -1) for _ in range(mask_bool.ndim))
        return dist_map_padded[unpad]

    def test_matches_scipy_reference_3d(self) -> None:
        rng = np.random.default_rng(0)
        for shape in [(1, 1, 1), (5, 5, 5), (1, 6, 7), (9, 3, 4)]:
            mask_bool = rng.random(shape) < 0.5
            expected = self._scipy_reference(mask_bool)
            actual = texture_module._gldzm_distance_map(mask_bool)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, expected.dtype)

    def test_all_background_and_all_foreground(self) -> None:
        for mask_bool in (np.zeros((4, 4, 4), dtype=bool), np.ones((4, 4, 4), dtype=bool)):
            expected = self._scipy_reference(mask_bool)
            actual = texture_module._gldzm_distance_map(mask_bool)
            np.testing.assert_array_equal(actual, expected)

    def test_non_3d_falls_back_to_scipy(self) -> None:
        """Non-3D masks (never produced by the texture pipeline) hit the defensive
        scipy fallback branch."""
        mask_2d = np.array([[True, False, True], [False, True, False]])
        expected = self._scipy_reference(mask_2d)
        actual = texture_module._gldzm_distance_map(mask_2d)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
