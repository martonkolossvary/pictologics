# ruff: noqa: E402
import os
import unittest
import warnings
from typing import Any

# Disable Numba JIT global optimization for coverage analysis
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Suppress the "NumPy module was reloaded" warning that can occur due to Numba + testing environment
# Must be done BEFORE importing numpy
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

from unittest.mock import patch

import numpy as np
import pytest

# Import the module under test
import pictologics.features.texture as texture_module


class TestTextureFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (5, 5, 5)
        self.data = np.random.randint(1, 5, self.shape)
        self.mask = np.ones(self.shape, dtype=int)
        self.n_bins = 16

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

    def test_ngtdm_ngldm_many_levels(self):
        n_bins = 1000
        data = np.random.randint(1, n_bins, self.shape)
        self.assertIn(
            "coarseness_QCDE", texture_module.calculate_ngtdm_features(data, self.mask, n_bins)
        )
        self.assertIn(
            "low_dependence_emphasis_SODN",
            texture_module.calculate_ngldm_features(data, self.mask, n_bins),
        )

    def test_more_levels_than_the_kernels_hold(self):
        with self.assertRaises(ValueError):
            texture_module.calculate_all_texture_matrices(self.data, self.mask, 70_000)

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

    def test_invalid_levels_leave_the_texture_roi(self):
        # An ROI voxel with a level outside [1, n_bins] (0 is the bin of NaN) takes no part:
        # the compact ROI leaves it out, and the matrices are those of the ROI without it.
        data = np.random.default_rng(1).integers(1, 5, self.shape)
        data[2, 2, 2] = 0
        data[1, 1, 1] = self.n_bins + 3
        compact = texture_module._texture_matrices(data, self.mask, self.n_bins, compact=True)
        clean = self.mask.copy()
        clean[2, 2, 2] = clean[1, 1, 1] = 0
        expected = texture_module._texture_matrices(data, clean, self.n_bins, compact=True)
        np.testing.assert_array_equal(compact["roi"], clean != 0)
        for key in ("glcm", "glrlm", "ngtdm_s", "ngtdm_n", "ngldm", "glszm_cells"):
            np.testing.assert_array_equal(compact[key], expected[key])

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

    def test_glcm_uint8_mask_many_levels(self):
        n_bins = 300
        data = np.random.randint(1, n_bins, self.shape)
        f = texture_module.calculate_glcm_features(data, self.mask.astype(np.uint8), n_bins)
        self.assertIn("contrast_ACUI", f)

    def test_glcm_zero_sum(self):
        # Line 859: if total_sum == 0
        glcm = np.zeros((13, 2, 2), dtype=float)
        f = texture_module.calculate_glcm_features(self.data, self.mask, n_bins=2, glcm_matrix=glcm)
        self.assertEqual(f, {})

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
        self,
        data,
        mask,
        dist_map,
        n_chunks,
        calc_glszm=True,
        calc_gldzm=True,
        dense_glszm=True,
    ):
        """The zone matrices with `n_chunks` threads, so at most that many z-chunks (also
        for these small volumes, which else take one serial fill)."""
        vol, counts = texture_module._texture_volume(data, np.asarray(mask) != 0, self.n_bins)
        dist = np.pad(dist_map.astype(np.int32), 1)
        with (
            patch.object(texture_module.numba, "get_num_threads", return_value=n_chunks),
            patch.object(texture_module, "_ZONE_PARALLEL_MIN_SIZE", 0),
        ):
            return texture_module._zone_matrices(
                vol, counts, dist, self.n_bins, calc_glszm, calc_gldzm, dense_glszm
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

    def test_thread_tables_add_up_in_parallel(self):
        # Large thread tables add up in parallel blocks, with the sums of numpy.
        rng = np.random.default_rng(2)
        tables = rng.integers(0, 1000, (3, 2, 5, 4097)).astype(np.uint32)
        expected = tables.sum(axis=0, dtype=np.uint64)
        with patch.object(texture_module, "_PARALLEL_SUM_MIN_CELLS", 1):
            got = texture_module._thread_sum(tables)
        np.testing.assert_array_equal(got, expected)
        self.assertEqual(got.dtype, np.uint64)

    def test_large_thread_tables_hold_only_the_levels_that_occur(self):
        # Above _COMPACT_TABLE_BYTES, the GLCM and GLRLM thread tables hold only the grey
        # levels that occur and go back to all levels: the matrices of the full tables.
        # When every level occurs, or no GLCM or GLRLM is asked for, the rows stay g - 1.
        rng = np.random.default_rng(19)
        mask = np.ones((10, 11, 12), dtype=np.uint8)
        sparse = rng.choice([3, 50, 51, 300], size=mask.shape).astype(np.int32)
        every = (np.arange(mask.size).reshape(mask.shape) % 300 + 1).astype(np.int32)
        for data in (sparse, every):
            for compact in (True, False):
                full = texture_module._texture_matrices(data, mask, 300, compact=compact)
                with patch.object(texture_module, "_COMPACT_TABLE_BYTES", 0):
                    rows = texture_module._texture_matrices(data, mask, 300, compact=compact)
                for key, value in full.items():
                    np.testing.assert_array_equal(rows[key], value)
        vol, _ = texture_module._texture_volume(sparse, mask != 0, 300)
        large = texture_module._COMPACT_TABLE_BYTES + 1
        levels = texture_module._table_levels(vol, 300, large)
        np.testing.assert_array_equal(levels, [2, 49, 50, 299])
        self.assertIsNone(texture_module._table_levels(vol, 300, large - 1))  # small tables
        both = texture_module._texture_matrices(sparse, mask, 300, compact=True)
        with patch.object(texture_module, "_COMPACT_TABLE_BYTES", 0):
            alone = texture_module._texture_matrices(
                sparse, mask, 300, compact=True, calc_glcm=False, calc_glrlm=False
            )
        for key in ("ngtdm_s", "ngtdm_n", "ngldm"):
            np.testing.assert_array_equal(alone[key], both[key])

    def test_large_thread_tables_are_zeroed_in_threads(self):
        # Large GLCM and GLRLM thread tables are zeroed in parallel blocks: the matrices
        # are those of np.zeros tables.
        rng = np.random.default_rng(28)
        data = rng.integers(1, 9, (6, 7, 5)).astype(np.int32)
        mask = np.ones(data.shape, dtype=np.uint8)
        expected = texture_module._texture_matrices(data, mask, 8)
        with patch.object(texture_module, "_PARALLEL_ZERO_MIN", 1):
            got = texture_module._texture_matrices(data, mask, 8)
        for key, value in expected.items():
            np.testing.assert_array_equal(got[key], value)
        flat = np.arange(70_000, dtype=np.uint32)
        texture_module._zero_fill_numba(flat)
        self.assertFalse(flat.any())

    def test_large_volumes_build_the_grey_levels_in_threads(self):
        # The parallel volume kernel of large volumes gives the serial one's volume and
        # counts (an ROI voxel with a level outside [1, n_bins] stays 0).
        rng = np.random.default_rng(25)
        data = rng.integers(0, 10, (6, 5, 4)).astype(np.int32)
        roi = rng.random(data.shape) > 0.3
        serial = texture_module._texture_volume(data, roi, 8)
        with patch.object(texture_module, "_PARALLEL_VOLUME_MIN", 0):
            parallel = texture_module._texture_volume(data, roi, 8)
        for got, expected in zip(parallel, serial, strict=True):
            np.testing.assert_array_equal(got, expected)

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

        # More threads than slices: one chunk per slice
        glszm, _ = self._run_parallel_zone_kernel(data, mask, dist, n_chunks=100)
        self.assertEqual(int(glszm.sum()), 1)
        # calc_gldzm=False uses the dummy distance buffer
        self._run_parallel_zone_kernel(data, mask, dist, n_chunks=2, calc_gldzm=False)
        # all-zero distances -> max_dist_val falls back to 1
        glszm, _ = self._run_parallel_zone_kernel(data, mask, dist, n_chunks=2)
        self.assertEqual(int(glszm.sum()), 1)

        # ROI voxels with an out-of-range grey level take no part (no zone)
        data_bad = np.full((depth, 3, 3), self.n_bins + 5, dtype=int)
        glszm_bad, _ = self._run_parallel_zone_kernel(data_bad, mask, dist, n_chunks=2)
        self.assertEqual(int(glszm_bad.sum()), 0)

    def test_parallel_zone_join_gives_the_zones_of_one_fill(self):
        # The chunk faces find their zone pairs in parallel, each face in its slot, and the
        # unions run in face order: the GLSZM cells and the GLDZM are those of one serial
        # fill, for 2, 8 and 32 grey levels and 2 to 7 chunks.
        rng = np.random.default_rng(41)
        for levels in (2, 8, 32):
            data = rng.integers(1, levels + 1, (9, 7, 8)).astype(np.int32)
            roi = rng.random(data.shape) < 0.7
            vol, counts = texture_module._texture_volume(data, roi, levels)
            dist = texture_module._chamfer_distance_taxicab_numba(roi, False, False, False)
            args = (counts, dist, levels, True, True, False)
            serial = texture_module._zone_matrices(vol.copy(), *args, parallel=False)
            for n_chunks in (2, 3, 7):
                with patch.object(texture_module.numba, "get_num_threads", return_value=n_chunks):
                    joined = texture_module._zone_matrices(vol.copy(), *args, parallel=True)
                for got, expected in zip(joined, serial, strict=True):
                    np.testing.assert_array_equal(got, expected)

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

    def test_compact_matrices_give_the_same_features(self):
        """compact=True keeps one GLCM/GLRLM table (the sum over the 13 directions); the
        other matrices and every feature must stay the same, bit for bit."""
        rng = np.random.default_rng(3)
        data = rng.integers(1, 7, (6, 7, 8))
        mask = (rng.random((6, 7, 8)) > 0.3).astype(float) * 2.0  # label 2, with holes
        full = texture_module.calculate_all_texture_matrices(data, mask, 6)
        compact = texture_module._texture_matrices(data, mask, 6, compact=True)
        self.assertNotIn("roi", full)
        np.testing.assert_array_equal(compact["roi"], mask != 0)
        self.assertEqual(full["glcm"].shape[0], 13)
        for key in ("glcm", "glrlm"):
            self.assertEqual(compact[key].shape[0], 1)
            np.testing.assert_array_equal(compact[key][0], full[key].sum(axis=0))
        for key in ("ngtdm_s", "ngtdm_n", "ngldm", "gldzm"):
            np.testing.assert_array_equal(compact[key], full[key])
        gl_idx, sz_idx = np.nonzero(full["glszm"])
        np.testing.assert_array_equal(
            compact["glszm_cells"], np.stack([gl_idx, sz_idx, full["glszm"][gl_idx, sz_idx]])
        )
        for family, key in (("glcm", "glcm_matrix"), ("glrlm", "glrlm_matrix")):
            calc = getattr(texture_module, f"calculate_{family}_features")
            ref = calc(data, mask, 6, **{key: full[family]})
            self.assertEqual(calc(data, mask, 6, **{key: compact[family]}), ref)
            self.assertEqual(calc(data, mask, 6), ref)  # standalone path: merged kernel
        self.assertEqual(
            texture_module.calculate_all_texture_features(data, mask, 6),
            {
                **texture_module.calculate_glcm_features(data, mask, 6, glcm_matrix=full["glcm"]),
                **texture_module.calculate_glrlm_features(
                    data, mask, 6, glrlm_matrix=full["glrlm"]
                ),
                **texture_module.calculate_glszm_features(
                    data, mask, 6, glszm_matrix=full["glszm"]
                ),
                **texture_module.calculate_gldzm_features(
                    data, mask, 6, gldzm_matrix=full["gldzm"], distance_mask=mask
                ),
                **texture_module.calculate_ngtdm_features(
                    data, mask, 6, ngtdm_matrices=(full["ngtdm_s"], full["ngtdm_n"])
                ),
                **texture_module.calculate_ngldm_features(
                    data, mask, 6, ngldm_matrix=full["ngldm"]
                ),
            },
        )

    @staticmethod
    def _dense_cells(glszm):
        gl_idx, sz_idx = np.nonzero(glszm)
        return np.stack([gl_idx, sz_idx, glszm[gl_idx, sz_idx]]).astype(np.uint32)

    def test_glszm_cells_match_dense_matrix(self):
        """Cell mode lists the non-zero GLSZM cells in np.nonzero order, with one chunk and
        with three, also when the few large zones go through the sorted path."""
        data = np.ones((3, 15, 15), dtype=int)
        mask_u8 = np.zeros((3, 15, 15), dtype=np.uint8)
        # Separate zones (26-connectivity): three equal bars (grey 2, size 3), one of size 4,
        # a size-3 bar in grey 3, and single voxels in grey levels 1 and 3.
        for y, gl, length in ((1, 2, 3), (4, 2, 3), (7, 2, 3), (10, 2, 4), (13, 3, 3)):
            data[1, y, 1 : 1 + length] = gl
            mask_u8[1, y, 1 : 1 + length] = 1
        for y, gl in ((1, 1), (4, 1), (7, 3)):
            data[1, y, 10] = gl
            mask_u8[1, y, 10] = 1
        dist = np.zeros((3, 15, 15), dtype=np.int32)
        dense, _ = texture_module.calculate_zone_features(data, mask_u8, dist, self.n_bins)
        self.assertEqual(int(dense[1, 2]), 3)  # the repeated (grey 2, size 3) cell
        for dense_cells in (1 << 18, 2 * self.n_bins):  # table only; table width 2 + sorting
            with patch("pictologics.features.texture._GLSZM_DENSE_CELLS", dense_cells):
                for n_chunks in (1, 3):
                    par_dense, _ = self._run_parallel_zone_kernel(data, mask_u8, dist, n_chunks)
                    np.testing.assert_array_equal(par_dense, dense)
                    par_cells, _ = self._run_parallel_zone_kernel(
                        data, mask_u8, dist, n_chunks, dense_glszm=False
                    )
                    np.testing.assert_array_equal(par_cells, self._dense_cells(dense))
        # No GLSZM requested: an empty cell array.
        par_cells, _ = self._run_parallel_zone_kernel(
            data, mask_u8, dist, n_chunks=3, calc_glszm=False, dense_glszm=False
        )
        self.assertEqual(par_cells.shape, (3, 0))

    def test_glszm_cells_match_the_dense_matrix_at_many_levels(self):
        """The cells at 2, 1,024, 1,025 and 4,096 grey levels (the small dense table up to
        1,024 levels, the large one above) are those of the dense GLSZM, also for a large
        zone that goes through the sorted path."""
        rng = np.random.default_rng(31)
        mask = rng.random((12, 13, 14)) < 0.8
        dist = np.zeros((14, 15, 16), dtype=np.int32)
        for n_bins in (2, 1024, 1025, 4096):
            data = rng.integers(1, n_bins + 1, mask.shape)
            data[:5] = 1  # one large zone
            vol, counts = texture_module._texture_volume(data, mask, n_bins)
            dense, _ = texture_module._zone_matrices(
                vol.copy(), counts, dist, n_bins, True, False, True
            )
            cells, _ = texture_module._zone_matrices(vol, counts, dist, n_bins, True, False, False)
            np.testing.assert_array_equal(cells, self._dense_cells(dense))

    def test_compact_matrices_empty_and_left_out(self):
        # An empty ROI gives empty matrices; the compact mode leaves out the matrices that
        # are not asked for, and the full mode gives zero placeholders.
        empty = texture_module._texture_matrices(
            self.data, np.zeros(self.shape), self.n_bins, compact=True
        )
        self.assertEqual(empty["glszm_cells"].shape, (3, 0))
        self.assertFalse(empty["roi"].any())
        self.assertEqual(
            texture_module._glszm_features_from_cells(empty["glszm_cells"], self.mask), {}
        )
        no_zones = texture_module._texture_matrices(
            self.data, self.mask, self.n_bins, calc_glszm=False, calc_gldzm=False, compact=True
        )
        self.assertNotIn("glszm_cells", no_zones)
        self.assertIn("glcm", no_zones)
        local = {"calc_glcm": False, "calc_glrlm": False, "calc_ngtdm": False, "calc_ngldm": False}
        zones_only = texture_module._texture_matrices(
            self.data, self.mask, self.n_bins, compact=True, **local
        )
        self.assertEqual(set(zones_only), {"glszm_cells", "gldzm", "roi", "distance_map"})
        full = texture_module._texture_matrices(self.data, self.mask, self.n_bins, **local)
        self.assertFalse(full["glcm"].any())
        no_zones_full = texture_module._texture_matrices(
            self.data, self.mask, self.n_bins, calc_glszm=False, calc_gldzm=False
        )
        self.assertEqual(no_zones_full["glszm"].shape, (self.n_bins, 1))


class TestOneSliceImages(unittest.TestCase):
    """A one-slice image (an axis of size 1) gets the in-plane texture: the GLCM and the
    GLRLM use the 4 in-plane directions, and the GLDZM distance map is the in-plane one.
    A one-slice ROI in a 3D image keeps the 3D rule."""

    def test_in_plane_directions_and_distances(self) -> None:
        data = np.ones((1, 7, 7), dtype=int)
        data[0, 2:5, 2:5] = 2
        mask = np.ones((1, 7, 7), dtype=np.uint8)
        m = texture_module.calculate_all_texture_matrices(data, mask, 2)
        self.assertEqual(int(m["glrlm"].reshape(13, -1).any(axis=1).sum()), 4)
        self.assertEqual(int(m["gldzm"][1, 2]), 1)  # the 3x3 zone of level 2: distance 3
        f = texture_module.calculate_glrlm_features(data, mask, 2)
        self.assertAlmostEqual(f["run_percentage_9ZK5"], float(m["glrlm"].sum()) / (49 * 4))
        self.assertEqual(
            texture_module.calculate_glrlm_features(data, mask, 2, glrlm_matrix=m["glrlm"]), f
        )
        self.assertEqual(
            texture_module.calculate_all_texture_features(data, mask, 2)["run_percentage_9ZK5"],
            f["run_percentage_9ZK5"],
        )
        g = texture_module.calculate_gldzm_features(data, mask, 2)
        self.assertAlmostEqual(g["large_distance_emphasis_MB4I"], (1 + 9) / 2)

        data_3d = np.ones((3, 7, 7), dtype=int)
        data_3d[1] = data[0]
        mask_3d = np.zeros((3, 7, 7), dtype=np.uint8)
        mask_3d[1] = 1
        m3 = texture_module.calculate_all_texture_matrices(data_3d, mask_3d, 2)
        self.assertEqual(int(m3["glrlm"].reshape(13, -1).any(axis=1).sum()), 13)
        self.assertEqual(m3["gldzm"].shape, (2, 1))  # every zone 1 step from the next slice


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

    def test_parallel_distance_map_equals_the_raster_passes(self) -> None:
        # The separable passes (in threads) give the int32 map of the two raster passes:
        # sparse, dense, empty and full masks, one voxel, thin slabs, and every set of
        # planar axes.
        import itertools

        rng = np.random.default_rng(12)
        masks = [rng.random((6, 7, 5)) < p for p in (0.1, 0.5, 0.9)]
        one = np.zeros((5, 6, 7), dtype=bool)
        one[2, 3, 4] = True
        masks += [one, np.zeros((3, 4, 5), dtype=bool), np.ones((3, 4, 5), dtype=bool)]
        masks += [rng.random(shape) < 0.7 for shape in ((1, 9, 8), (7, 1, 6), (6, 8, 1))]
        masks.append(np.ones((1, 1, 1), dtype=bool))
        for mask in masks:
            for planar in itertools.product((False, True), repeat=3):
                expected = texture_module._chamfer_distance_taxicab_numba(mask, *planar)
                got = texture_module._distance_map_parallel_numba(mask, *planar)
                np.testing.assert_array_equal(got, expected)
                self.assertEqual(got.dtype, expected.dtype)

    def test_large_masks_take_the_distance_map_in_threads(self) -> None:
        # From _DISTANCE_PARALLEL_MIN voxels on, a mask of two slices or more takes the
        # distance map in threads, with the same GLDZM features; one slice stays serial.
        rng = np.random.default_rng(13)
        data = rng.integers(1, 5, (4, 6, 7))
        mask = (rng.random(data.shape) < 0.8).astype(np.uint8)
        serial = texture_module.calculate_all_texture_features(data, mask, 4, families=["gldzm"])
        with (
            patch.object(texture_module, "_DISTANCE_PARALLEL_MIN", mask.size),
            patch.object(
                texture_module,
                "_distance_map_parallel_numba",
                wraps=texture_module._distance_map_parallel_numba,
            ) as threads,
        ):
            got = texture_module.calculate_all_texture_features(data, mask, 4, families=["gldzm"])
            self.assertEqual(threads.call_count, 1)
            texture_module._distance_map(mask[:1] != 0, (False, False, False))  # one slice
            texture_module._distance_map(mask[:, :-1] != 0, (False, False, False))  # fewer
            self.assertEqual(threads.call_count, 1)
        self.assertEqual(got, serial)

    def test_non_3d_falls_back_to_scipy(self) -> None:
        """Non-3D masks (never produced by the texture pipeline) hit the defensive
        scipy fallback branch."""
        mask_2d = np.array([[True, False, True], [False, True, False]])
        expected = self._scipy_reference(mask_2d)
        actual = texture_module._gldzm_distance_map(mask_2d)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()


def test_texture_features_skip_the_empty_levels(monkeypatch: Any) -> None:
    """The features add only the matrix rows and columns that hold counts: empty grey
    levels above the ROI levels give the same features (to the last digits). A matrix
    below _OCCUPIED_MIN_CELLS keeps all rows and columns (here 0, to test small ones)."""
    from pictologics.features.texture import _occupied

    small = np.zeros((2, 3))
    assert _occupied(small)[0] is small and _occupied(small)[2].tolist() == [1, 2, 3]
    monkeypatch.setattr(texture_module, "_OCCUPIED_MIN_CELLS", 0)
    full = np.arange(1.0, 13.0).reshape(3, 4)
    same, rows, cols = _occupied(full)
    assert same is full and rows.tolist() == [1, 2, 3] and cols.tolist() == [1, 2, 3, 4]
    sparse = np.zeros((5, 6))
    sparse[1, 2] = 3.0
    sparse[3, 5] = 1.0
    compact, rows, cols = _occupied(sparse)
    assert compact.tolist() == [[3.0, 0.0], [0.0, 1.0]]
    assert rows.tolist() == [2, 4] and cols.tolist() == [3, 6]
    assert _occupied(np.diag([0.0, 2.0, 0.0]), symmetric=True)[1].tolist() == [2]

    rng = np.random.default_rng(12)
    data = rng.integers(1, 7, (10, 10, 10)).astype(np.float64)
    data[data == 3] = 2  # level 3 stays empty
    mask = np.ones((10, 10, 10), dtype=np.uint8)
    for family in ("glcm", "glrlm", "gldzm", "ngldm"):
        calculate = getattr(texture_module, f"calculate_{family}_features")
        six, sixteen = calculate(data, mask, 6), calculate(data, mask, 16)
        assert six.keys() == sixteen.keys()
        for key in six:
            np.testing.assert_allclose(sixteen[key], six[key], rtol=1e-12, err_msg=key)


def _texture_by_loops(
    levels: np.ndarray,
    roi: np.ndarray,
    n_bins: int,
    glcm_distance: int,
    ngtdm_distance: int,
    ngldm_distance: int,
    alpha: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GLCM (the 13 directions summed), NGTDM s and n, and NGLDM by loops over the voxels
    (IBSI 1): GLCM pairs at glcm_distance along each direction, neighbourhoods of the
    voxels up to their Chebyshev distance."""
    import itertools

    shape = levels.shape
    directions = [d for d in itertools.product((-1, 0, 1), repeat=3) if d > (0, 0, 0)]
    glcm = np.zeros((n_bins, n_bins))
    s, n = np.zeros(n_bins), np.zeros(n_bins)
    ngldm = np.zeros((n_bins, (2 * ngldm_distance + 1) ** 3))

    def level(q: tuple[int, ...]) -> int:
        inside = all(0 <= c < m for c, m in zip(q, shape, strict=True))
        return int(levels[q]) if inside and roi[q] else 0

    def neighbours(p: tuple[int, ...], distance: int) -> list[int]:
        steps = itertools.product(range(-distance, distance + 1), repeat=3)
        found = (level(tuple(c + o for c, o in zip(p, d, strict=True))) for d in steps if any(d))
        return [g for g in found if g]

    for p in itertools.product(*(range(m) for m in shape)):
        g = level(p)
        if not g:
            continue
        for d in directions:
            other = level(tuple(c + glcm_distance * o for c, o in zip(p, d, strict=True)))
            if other:
                glcm[g - 1, other - 1] += 1
        near = neighbours(p, ngtdm_distance)
        if near:
            s[g - 1] += abs(g - sum(near) / len(near))
            n[g - 1] += 1
        dependence = 1 + sum(abs(h - g) <= alpha for h in neighbours(p, ngldm_distance))
        ngldm[g - 1, dependence - 1] += 1
    return glcm, s, n, ngldm


def _runs_by_loops(
    levels: np.ndarray, roi: np.ndarray, n_bins: int, directions: np.ndarray
) -> np.ndarray:
    """GLRLM, one table for each direction, by walks along the directions (IBSI 1): a run
    starts where the voxel before has another level."""
    import itertools

    shape = levels.shape
    glrlm = np.zeros((len(directions), n_bins, max(shape) + 1))

    def level(q: tuple[int, ...]) -> int:
        inside = all(0 <= c < m for c, m in zip(q, shape, strict=True))
        return int(levels[q]) if inside and roi[q] else 0

    for k, d in enumerate(directions):
        for p in itertools.product(*(range(m) for m in shape)):
            g = level(p)
            if not g or level(tuple(c - o for c, o in zip(p, d, strict=True))) == g:
                continue
            length = 1
            while level(tuple(c + length * o for c, o in zip(p, d, strict=True))) == g:
                length += 1
            glrlm[k, g - 1, length] += 1
    return glrlm


def test_the_local_kernel_counts_equal_loops_over_the_voxels() -> None:
    # The local kernel (the row passes of the 26 neighbours, the runs of length 1 and 2
    # without a branch) gives the GLCM, GLRLM, NGTDM and NGLDM counts of plain loops over
    # the voxels: one-voxel, one-slice, thin and holed ROIs, 2 to 12 levels, NGLDM alpha 0
    # to 2, one table for all directions or one for each, and the families alone.
    rng = np.random.default_rng(21)
    shapes = [(1, 1, 1), (1, 7, 9), (6, 1, 8), (5, 7, 1), (2, 3, 4), (1, 1, 9), (7, 6, 5), (9, 8, 6)]  # fmt: skip
    for case in range(3 * len(shapes)):
        shape = shapes[case % len(shapes)]
        n_bins, alpha = (2, 5, 12)[case % 3], case % 3
        levels = rng.integers(1, n_bins + 1, shape).astype(np.float64)
        roi = (rng.random(shape) > 0.3).astype(np.uint8)
        roi.flat[0] = 1
        planar = (shape[0] == 1, shape[1] == 1, shape[2] == 1)
        used = texture_module._directions(planar)
        glcm, s, n, ngldm = _texture_by_loops(levels, roi, n_bins, 1, 1, 1, alpha)
        runs = _runs_by_loops(levels, roi, n_bins, texture_module.DIRECTIONS_13[used])
        vol, counts = texture_module._texture_volume(levels, roi, n_bins)
        for merge in (False, True):
            got = texture_module._local_matrices(
                vol, counts, n_bins, True, True, True, True, alpha, merge, planar
            )
            np.testing.assert_array_equal(got[0].sum(axis=0), glcm)
            if merge:
                np.testing.assert_array_equal(got[1][0], runs.sum(axis=0))
            else:
                np.testing.assert_array_equal(got[1][used], runs)
            np.testing.assert_allclose(got[2], s, rtol=1e-12)
            np.testing.assert_array_equal(got[3], n)
            np.testing.assert_array_equal(got[4], ngldm)
        # The families alone: GLCM and GLRLM keep the loop over every voxel of a row
        alone = texture_module._local_matrices(
            vol, counts, n_bins, True, True, False, False, alpha, True, planar
        )
        np.testing.assert_array_equal(alone[0][0], glcm)
        np.testing.assert_array_equal(alone[1][0], runs.sum(axis=0))
        for ngtdm, ngldm_on in ((True, False), (False, True)):
            part = texture_module._local_matrices(
                vol, counts, n_bins, False, False, ngtdm, ngldm_on, alpha, True, planar
            )
            if ngtdm:
                np.testing.assert_allclose(part[2], s, rtol=1e-12)
            else:
                np.testing.assert_array_equal(part[4], ngldm)


def test_the_local_kernel_twin_gives_the_counts_of_the_parallel_kernel() -> None:
    # Below _LOCAL_PARALLEL_MIN_SIZE voxels of the padded volume the serial twin fills one
    # table; the parallel kernel fills a table for each thread: the same counts (whole
    # numbers), with one table for all directions or one for each.
    import numba

    rng = np.random.default_rng(22)
    levels = rng.integers(1, 9, (9, 8, 7)).astype(np.float64)
    roi = (rng.random(levels.shape) > 0.3).astype(np.uint8)
    vol, counts = texture_module._texture_volume(levels, roi, 8)
    for merge in (False, True):
        runs = [
            texture_module._local_matrices(
                vol, counts, 8, True, True, True, True, 1, merge, parallel=parallel
            )  # fmt: skip
            for parallel in (True, False)
        ]
        for a, b in zip(*runs, strict=True):
            assert a.tobytes() == b.tobytes()
    twin = texture_module._local_tables_numba_serial
    for limit, serial in ((vol.size + 1, True), (vol.size, numba.get_num_threads() == 1)):
        with (
            patch.object(texture_module, "_LOCAL_PARALLEL_MIN_SIZE", limit),
            patch.object(texture_module, "_local_tables_numba_serial", wraps=twin) as calls,
        ):
            texture_module._local_matrices(vol, counts, 8, True, True, True, True, 1, True)
        assert calls.call_count == int(serial)


def test_texture_distances_match_loops_over_the_voxels() -> None:
    # GLCM, NGTDM and NGLDM distances above 1 (IBSI 1) give the matrices of plain loops
    # over the voxels: one distance for all, two NGTDM and NGLDM distances (two kernel
    # passes), and the compact tables of the occurring grey levels.
    rng = np.random.default_rng(8)
    levels = rng.integers(1, 6, (7, 6, 5)).astype(np.float64)
    roi = (rng.random(levels.shape) > 0.25).astype(np.uint8)
    for distances, alpha in (((2, 2, 2), 0), ((3, 1, 2), 1), ((1, 2, 1), 0)):
        glcm, s, n, ngldm = _texture_by_loops(levels, roi, 8, *distances, alpha)
        for compact_bytes in (texture_module._COMPACT_TABLE_BYTES, 0):
            with patch.object(texture_module, "_COMPACT_TABLE_BYTES", compact_bytes):
                matrices = texture_module.calculate_all_texture_matrices(
                    levels, roi, 8, ngldm_alpha=alpha, calc_glszm=False, calc_gldzm=False,
                    glcm_distance=distances[0], ngtdm_distance=distances[1], ngldm_distance=distances[2],
                )  # fmt: skip
            np.testing.assert_array_equal(matrices["glcm"].sum(axis=0), glcm)
            np.testing.assert_allclose(matrices["ngtdm_s"], s, rtol=1e-12)
            np.testing.assert_array_equal(matrices["ngtdm_n"], n)
            np.testing.assert_array_equal(matrices["ngldm"], ngldm)
    # The feature functions take the distances too
    features = texture_module.calculate_all_texture_features(levels, roi, 8, glcm_distance=2, ngtdm_distance=2, ngldm_distance=3)  # fmt: skip
    alone = {
        **texture_module.calculate_glcm_features(levels, roi, 8, glcm_distance=2),
        **texture_module.calculate_ngtdm_features(levels, roi, 8, ngtdm_distance=2),
        **texture_module.calculate_ngldm_features(levels, roi, 8, ngldm_distance=3),
    }
    for key, value in alone.items():
        assert np.isclose(features[key], value, rtol=1e-12, equal_nan=True), key
    empty = texture_module.calculate_all_texture_matrices(levels, roi * 0, 8, ngldm_distance=2)
    assert empty["ngldm"].shape == (8, 125)


def test_texture_features_of_chosen_families() -> None:
    # families gives the features of those families alone, the values of the full call,
    # with no matrix of the other families; one name may come as a str.
    rng = np.random.default_rng(10)
    levels = rng.integers(1, 7, (9, 8, 7)).astype(np.float64)
    roi = (rng.random(levels.shape) > 0.2).astype(np.uint8)
    full = texture_module.calculate_all_texture_features(levels, roi, 6)
    for families in (["glcm"], "ngtdm", ("glszm", "gldzm"), ["ngldm", "glrlm"]):
        with patch.object(
            texture_module, "_texture_matrices", wraps=texture_module._texture_matrices
        ) as spy:
            part = texture_module.calculate_all_texture_features(levels, roi, 6, families=families)
        chosen = {families} if isinstance(families, str) else set(families)
        flags = {k[5:] for k, v in spy.call_args.kwargs.items() if k.startswith("calc_") and v}
        assert flags == chosen
        assert part and all(part[key] == full[key] or np.isnan(part[key]) for key in part)
    assert set(full) == set().union(
        *(texture_module.calculate_all_texture_features(levels, roi, 6, families=f) for f in texture_module._TEXTURE_FAMILIES)
    )  # fmt: skip
    with pytest.raises(
        ValueError, match=r"Unknown texture family 'glmc' \(did you mean 'glcm'\?\)"
    ):
        texture_module.calculate_all_texture_features(levels, roi, 6, families=["glmc"])
    with pytest.raises(ValueError, match="Unknown texture family 'shape'. The families"):
        texture_module.calculate_all_texture_features(levels, roi, 6, families="shape")


def test_direct_calls_read_other_level_types_as_int64_or_float64() -> None:
    # A direct call reads grey levels of another integer or bool type as int64 and of
    # another float type as float64, the types that the import compiles; the features stay
    # the same.
    for kind, form in (
        (np.bool_, np.int64),
        (np.int8, np.int64),
        (np.uint8, np.int64),
        (np.int16, np.int64),
        (np.uint16, np.int64),
        (np.uint32, np.int64),
        (np.float16, np.float64),
        (np.float32, np.float64),
    ):
        assert texture_module._level_form(np.ones(3, dtype=kind)).dtype == form
    for kind in (np.int32, np.int64, np.float64):
        levels = np.ones(3, dtype=kind)
        assert texture_module._level_form(levels) is levels
    rng = np.random.default_rng(11)
    levels = rng.integers(1, 7, (9, 8, 7)).astype(np.int32)
    roi = (rng.random(levels.shape) > 0.2).astype(np.uint8)
    expected = texture_module.calculate_all_texture_features(levels, roi, 6)
    for kind in (np.uint8, np.float32):
        found = texture_module.calculate_all_texture_features(levels.astype(kind), roi, 6)
        assert np.array_equal(list(found.values()), list(expected.values()), equal_nan=True)
    empty = texture_module._maybe_crop_to_bbox(levels.astype(np.uint8), roi * 0)
    assert empty[0].dtype == np.int64


def _glcm_reference(glcm: np.ndarray, ng_eff: int) -> dict[str, float]:
    """The GLCM features by the old numpy formulas on the dense matrix: pow for the cluster
    features, and the double sums HXY1 and HXY2 of the information correlations."""
    counts = glcm.sum(axis=0)
    P = (counts + counts.T) / (counts + counts.T).sum()
    I, J = np.meshgrid(*[np.arange(1, P.shape[0] + 1)] * 2, indexing="ij")  # noqa: E741
    mu = np.sum(I * P)
    s = I + J - 2 * mu
    k = np.abs(I - J)
    var = np.sum((I - mu) ** 2 * P)
    nz = P > 0
    p_x = P.sum(axis=1)
    hx = -np.sum(p_x[p_x > 0] * np.log2(p_x[p_x > 0]))
    hxy = -np.sum(P[nz] * np.log2(P[nz]))
    rows, cols = np.nonzero(nz)
    hxy1 = -np.sum(P[nz] * np.log2(p_x[rows] * p_x[cols]))
    prod = np.outer(p_x, p_x)
    hxy2 = -np.sum(prod[prod > 0] * np.log2(prod[prod > 0]))
    return {
        "joint_maximum_GYBY": np.max(P),
        "joint_average_60VM": mu,
        "joint_variance_UR99": var,
        "joint_entropy_TU9B": hxy,
        "difference_average_TF7R": np.sum(k * P),
        "dissimilarity_8S9J": np.sum(k * P),
        "contrast_ACUI": np.sum(k**2 * P),
        "angular_second_moment_8ZQL": np.sum(P**2),
        "inverse_difference_IB1Z": np.sum(P / (1 + k)),
        "normalised_inverse_difference_NDRX": np.sum(P / (1 + k / ng_eff)),
        "inverse_difference_moment_WF0Z": np.sum(P / (1 + k**2)),
        "normalised_inverse_difference_moment_1QCO": np.sum(P / (1 + k**2 / ng_eff**2)),
        "inverse_variance_E8JP": np.sum(P[k > 0] / k[k > 0] ** 2),
        "autocorrelation_QWB0": np.sum(I * J * P),
        "correlation_NI2N": np.sum((I - mu) * (J - mu) * P) / var if var else 1.0,
        "cluster_tendency_DG8W": np.sum(s**2 * P),
        "cluster_shade_7NFM": np.sum(s**3 * P),
        "cluster_prominence_AE86": np.sum(s**4 * P),
        "information_correlation_1_R8DG": (hxy - hxy1) / hx if hx else np.nan,
        "1 - information_correlation_2_JN9H^2": np.exp(-2 * (hxy2 - hxy)),
        "scale of the cluster shade": np.sum(np.abs(s) ** 3 * P),
        "HXY1 - 2 HX": hxy1 - 2 * hx,
        "HXY2 - 2 HX": hxy2 - 2 * hx,
        "2 HX": 2 * hx,
    }


def test_glcm_features_equal_the_old_formulas() -> None:
    # The GLCM features come from two serial passes over the cells that hold counts, the
    # information correlations from closed forms (HXY1 = HXY2 = 2 HX) and the cluster
    # features from products. On dense, sparse, one-level, two-level and 256-level
    # matrices they equal the old numpy formulas within 1e-12 (the cluster shade relative
    # to the size of its terms, as its terms cancel).
    rng = np.random.default_rng(31)
    sizes = [1, 2, 2, 256, 256, *rng.integers(3, 200, 35)]
    for case, n in enumerate(sizes):
        glcm = rng.integers(0, 1000, (2, n, n)).astype(np.uint64)
        if case % 3 == 1:  # sparse
            glcm *= (rng.random((2, n, n)) < 0.03).astype(np.uint64)
        if case == 2:  # one level of the two holds counts
            glcm[:] = 0
            glcm[0, 1, 1] = 5
        glcm[0, 0, 0] += 1  # at least one count
        data = np.array([[[1, n]]], dtype=np.int32)  # Ng_eff = n
        got = texture_module.calculate_glcm_features(data, np.ones(data.shape), n, glcm_matrix=glcm)
        ref = _glcm_reference(glcm, n)
        assert abs(ref["HXY1 - 2 HX"]) <= 1e-12 * max(ref["2 HX"], 1.0)
        assert abs(ref["HXY2 - 2 HX"]) <= 1e-12 * max(ref["2 HX"], 1.0)
        for key, value in ref.items():
            if key == "cluster_shade_7NFM":
                assert abs(got[key] - value) <= 1e-12 * ref["scale of the cluster shade"]
            elif key == "1 - information_correlation_2_JN9H^2":
                assert np.isclose(1 - got["information_correlation_2_JN9H"] ** 2, value, atol=1e-12)
            elif key in got:
                assert np.isclose(got[key], value, rtol=1e-12, atol=1e-12, equal_nan=True), key


def _ngtdm_reference(s: np.ndarray, n: np.ndarray) -> dict[str, float]:
    """The NGTDM features by the old formulas, with grids over the non-zero levels."""
    total = np.sum(n)
    p = n / total
    nz = p > 0
    p_nz, s_nz, levels = p[nz], s[nz], np.arange(1, len(n) + 1)[nz]
    Pi, Pj = np.meshgrid(p_nz, p_nz, indexing="ij")
    Ii, Ij = np.meshgrid(levels, levels, indexing="ij")
    Si, Sj = np.meshgrid(s_nz, s_nz, indexing="ij")
    sum_ps = np.sum(p_nz * s_nz)
    m = len(p_nz)
    ip = levels * p_nz
    busy = np.sum(np.abs(ip[:, None] - ip[None, :]))
    contrast = (
        np.sum(Pi * Pj * (Ii - Ij) ** 2) / (m * (m - 1)) * np.sum(s) / total if m > 1 else 0.0
    )
    return {
        "coarseness_QCDE": 1 / sum_ps if sum_ps > 1e-10 else 1e6,
        "contrast_65HE": contrast,
        "busyness_NQ30": sum_ps / busy if busy > 1e-10 else 0.0,
        "complexity_HDEZ": np.sum(np.abs(Ii - Ij) * (Pi * Si + Pj * Sj) / (Pi + Pj)) / total,
        "strength_1X9X": np.sum((Pi + Pj) * (Ii - Ij) ** 2) / np.sum(s)
        if np.sum(s) > 1e-10
        else 0.0,
    }


def test_ngtdm_features_equal_the_grid_formulas() -> None:
    # The NGTDM sums over the pairs of non-zero levels come from closed forms and one
    # kernel; with empty levels, one level and two levels they equal the grid formulas
    # within 1e-12
    rng = np.random.default_rng(32)
    sizes = [1, 2, 2, 256, *rng.integers(3, 257, 36)]
    for case, n_bins in enumerate(sizes):
        n = rng.integers(0, 50, n_bins).astype(np.float64)
        n[rng.random(n_bins) < 0.3] = 0  # empty levels
        if case == 1:  # one level holds voxels
            n[:] = [0, 7]
        n[0] += 1
        s = np.where(n > 0, rng.random(n_bins) * 20, 0.0)
        got = texture_module.calculate_ngtdm_features(
            np.ones((1, 1, 1)), np.ones((1, 1, 1)), n_bins, ngtdm_matrices=(s, n)
        )
        for key, value in _ngtdm_reference(s, n).items():
            assert np.isclose(got[key], value, rtol=1e-12, atol=0.0), (case, key)
