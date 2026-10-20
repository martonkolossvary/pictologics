# ruff: noqa: E402
import os
import warnings

# Disable Numba JIT for coverage and smoother testing logic execution
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Suppress the "NumPy module was reloaded" warning that can occur due to Numba + testing environment
# Must be done BEFORE importing numpy
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pictologics.features.morphology import (
    _calculate_ellipsoid_surface_area,
    _get_bounding_box_features,
    _get_convex_hull_features,
    _get_intensity_morphology_features,
    _get_mesh_features,
    _get_mvee_features,
    _get_pca_features,
    _max_pairwise_distance_numba,
    _mesh_area_volume_numba,
    _mvee_khachiyan_numba,
    _ombb_extents_numba,
    calculate_morphology_features,
)
from pictologics.loader import Image


class TestMorphologyFeatures(unittest.TestCase):
    def _create_image(self, array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
        return Image(array, spacing, origin)

    # ----------------------------------------------------------------------
    # Basic Feature Tests
    # ----------------------------------------------------------------------

    def test_cube_features(self):
        # 10x10x10 cube
        # Volume = 1000
        # Surface Area = 6 * 100 = 600
        size = 10
        arr = np.zeros((size + 4, size + 4, size + 4), dtype=int)
        arr[2 : 2 + size, 2 : 2 + size, 2 : 2 + size] = 1
        mask = self._create_image(arr)

        features = calculate_morphology_features(mask)

        # Voxel Volume
        self.assertAlmostEqual(features["volume_voxel_counting_YEKZ"], 1000.0)
        # Mesh Volume (approximate)
        self.assertTrue(900 < features["volume_RNU0"] < 1100)
        # Surface Area
        self.assertTrue(500 < features["surface_area_C0JK"] < 700)

    def test_sphere_features(self):
        # Sphere radius 10
        r = 10
        d = 2 * r + 4
        z, y, x = np.ogrid[:d, :d, :d]
        center = d / 2
        dist_sq = (z - center) ** 2 + (y - center) ** 2 + (x - center) ** 2
        arr = (dist_sq <= r**2).astype(int)
        mask = self._create_image(arr)

        features = calculate_morphology_features(mask)

        # Sphericity (QCFX) -> 1 for perfect sphere.
        # Discrete sphere approximation isn't perfect, so check bounds.
        self.assertTrue(0.7 < features["sphericity_QCFX"] <= 1.0)

    def test_elongated_box_pca(self):
        # 20x4x4 box. Elongated along Z (index 0).
        arr = np.zeros((30, 10, 10), dtype=int)
        arr[5:25, 3:7, 3:7] = 1  # 20x4x4
        mask = self._create_image(arr)

        features = calculate_morphology_features(mask)

        # Major axis should be roughly 20
        self.assertTrue(features["major_axis_length_TDIC"] > 15.0)
        self.assertTrue(features["minor_axis_length_P9VJ"] < 10.0)

        self.assertTrue(features["elongation_Q3CK"] < 1.0)
        self.assertTrue(features["flatness_N17B"] < 1.0)

    def test_empty_mask(self):
        arr = np.zeros((10, 10, 10), dtype=int)
        mask = self._create_image(arr)
        features = calculate_morphology_features(mask)

        self.assertEqual(features["volume_voxel_counting_YEKZ"], 0.0)
        self.assertEqual(features.get("volume_RNU0", 0.0), 0.0)

    def test_single_voxel(self):
        arr = np.zeros((5, 5, 5), dtype=int)
        arr[2, 2, 2] = 1
        mask = self._create_image(arr)
        features = calculate_morphology_features(mask)

        self.assertEqual(features["volume_voxel_counting_YEKZ"], 1.0)
        # PCA requires > 3 points. Single voxel has 1 point (or 8 corners depending on implementation).
        # Implementaion uses mask indices for moments. n=1. Code says `if n <= 3: return`.
        self.assertNotIn("major_axis_length_TDIC", features)

    def test_label_mask_values_are_membership_not_weights(self):
        arr = np.zeros((5, 5, 5), dtype=int)
        arr[1:3, 1:3, 1:3] = 2
        mask = self._create_image(arr)
        features = calculate_morphology_features(mask)

        self.assertEqual(features["volume_voxel_counting_YEKZ"], 8.0)

    # ----------------------------------------------------------------------
    # Intensity Weighted Features
    # ----------------------------------------------------------------------

    def test_intensity_features_basic(self):
        # Mask: 3x3x3 cube
        arr = np.zeros((5, 5, 5), dtype=int)
        arr[1:4, 1:4, 1:4] = 1
        mask = self._create_image(arr)

        # Intensity: Constant 10
        img_arr = np.zeros((5, 5, 5), dtype=float)
        img_arr[1:4, 1:4, 1:4] = 10.0
        image = self._create_image(img_arr)

        features = calculate_morphology_features(mask, image=image)

        vol = features["volume_RNU0"]
        # Integrated intensity = Vol * Mean Intensity. Mean = 10.
        self.assertAlmostEqual(features["integrated_intensity_99N0"], vol * 10.0, delta=1e-4)
        # CoM shift should be ~0 as both are symmetric cubes
        self.assertAlmostEqual(features["center_of_mass_shift_KLMA"], 0.0)

    def test_intensity_features_shift(self):
        # Mask: 2 voxels at (0,0,0) and (0,0,1)
        arr = np.zeros((3, 3, 3), dtype=int)
        arr[0, 0, 0] = 1
        arr[0, 0, 1] = 1
        mask = self._create_image(arr)

        # Intensity: (0,0,0)=10, (0,0,1)=100
        # Geom CoM is at z=0.5.
        # Intensity CoM is weighted towards z=1.
        img_arr = np.zeros((3, 3, 3), dtype=float)
        img_arr[0, 0, 0] = 10.0
        img_arr[0, 0, 1] = 100.0
        image = self._create_image(img_arr)

        features = calculate_morphology_features(mask, image=image)
        self.assertGreater(features["center_of_mass_shift_KLMA"], 0.0)

    def test_intensity_zero_sum(self):
        # Mask with valid voxels, but intensity is 0.
        arr = np.zeros((3, 3, 3), dtype=int)
        arr[0, 0, 0] = 1
        mask = self._create_image(arr)
        img_arr = np.zeros((3, 3, 3), dtype=float)
        image = self._create_image(img_arr)

        features = calculate_morphology_features(mask, image=image)
        # sum_w = 0. Should handle gracefully.
        self.assertEqual(features.get("integrated_intensity_99N0", 0.0), 0.0)
        self.assertNotIn("center_of_mass_shift_KLMA", features)

    # ----------------------------------------------------------------------
    # Specific Algorithmic & Corner Case Tests
    # ----------------------------------------------------------------------

    def test_ellipsoid_surface_area_approx(self):
        # Sphere a=b=c=1 -> 4pi
        area = _calculate_ellipsoid_surface_area(1, 1, 1)
        self.assertAlmostEqual(area, 4 * np.pi)

        # Oblate spheroid a=b=2, c=1
        area_oblate = _calculate_ellipsoid_surface_area(2, 2, 1)
        self.assertGreater(area_oblate, 0)

        # Prolate spheroid a=2, b=c=1
        area_prolate = _calculate_ellipsoid_surface_area(2, 1, 1)
        self.assertGreater(area_prolate, 0)

    def test_ellipsoid_degenerate(self):
        self.assertEqual(_calculate_ellipsoid_surface_area(0, 0, 0), 0.0)
        self.assertEqual(_calculate_ellipsoid_surface_area(1, 0, 0), 0.0)

    def test_pca_few_points(self):
        # < 3 points
        arr = np.zeros((5, 5, 5), dtype=int)
        arr[0, 0, 0] = 1
        arr[0, 0, 1] = 1
        mask = self._create_image(arr)

        features, evals, evecs = _get_pca_features(mask, 1.0, 1.0)
        self.assertIsNone(evals)
        self.assertEqual(features, {})

    def test_convex_hull_of_one_voxel(self):
        # One voxel: the hull of its six mesh vertices (an octahedron), both densities and
        # the diameter 1.0. Points in one plane (Qhull raises): no features and no hull
        # points. Any other error of the hull step passes to the caller, also through
        # calculate_morphology_features (no feature is dropped silently).
        from scipy.spatial import QhullError

        padded = np.pad(np.ones((1, 1, 1), dtype=np.uint8), 1)
        args = (padded, np.zeros(3), 1.0, 1.0, (1.0, 1.0, 1.0))
        features, hull = _get_convex_hull_features(*args)
        self.assertEqual(hull.shape, (6, 3))
        self.assertEqual(features["maximum_3d_diameter_L0JK"], 1.0)
        self.assertIn("volume_density_convex_hull_R3ER", features)
        self.assertIn("area_density_convex_hull_7T7F", features)
        no_hull = (False, np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.int64))
        with (
            patch("pictologics.features.morphology._exact_hull_numba", return_value=no_hull),
            patch("pictologics.features.morphology.ConvexHull", side_effect=QhullError("flat")),
        ):
            self.assertEqual(_get_convex_hull_features(*args), ({}, None))
        kernel = "pictologics.features.morphology._max_pairwise_distance_serial_numba"
        with patch(kernel, side_effect=RuntimeError("diameter")):
            with self.assertRaisesRegex(RuntimeError, "diameter"):
                _get_convex_hull_features(*args)
            mask = self._create_image(np.pad(np.ones((3, 3, 3), dtype=np.uint8), 1))
            with self.assertRaisesRegex(RuntimeError, "diameter"):
                calculate_morphology_features(mask)

    def test_bounding_box_empty(self):
        features = _get_bounding_box_features(np.array([]), None, 1.0, 1.0)
        self.assertEqual(features, {})

    def test_mvee_singular(self):
        # Collinear points -> Singular covariance -> MVEE failure
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        # Should ideally handle it gracefully
        A, c = _mvee_khachiyan_numba(points)
        self.assertIsNone(A)
        self.assertIsNone(c)

    def test_mvee_features_none_hull(self):
        features = _get_mvee_features(None, 1.0, 1.0)
        self.assertEqual(features, {})

    def test_marching_cubes_matches_pymcubes(self):
        # The kernel and its parallel form give the PyMCubes 0.1.6 mesh: the same vertices
        # and faces, in the same order. The file holds 32 masks and their PyMCubes meshes,
        # made once with PyMCubes.
        from pictologics.features.morphology import _mesh

        path = os.path.join(os.path.dirname(__file__), "data", "marching_cubes_pymcubes.npz")
        with np.load(path) as ref:
            count = sum(key.startswith("mask_") for key in ref.files)
            self.assertEqual(count, 32)
            for i in range(count):
                for parallel in (False, True):
                    # offset 1 and spacing 1: the vertices in padded voxel units, as PyMCubes
                    padded = np.pad(ref[f"mask_{i}"], 1)
                    verts, faces = _mesh(padded, np.ones(3), np.ones(3), parallel)
                    self.assertEqual(faces.dtype, np.int32)
                    np.testing.assert_array_equal(verts, ref[f"verts_{i}"])
                    np.testing.assert_array_equal(faces, ref[f"faces_{i}"])

    def test_large_volumes_take_the_parallel_marching_cubes(self):
        # From _MESH_PARALLEL_MIN voxels on, with more than one thread, the x planes of
        # cubes run in parallel: the mesh of the serial kernel (random masks).
        from pictologics.features import morphology as morphology_module

        rng = np.random.default_rng(17)
        padded = np.pad((rng.random((7, 8, 6)) < 0.5).astype(np.uint8), 1)
        serial = morphology_module._mesh(padded, np.full(3, 0.5), np.array([0.7, 0.8, 1.5]))
        kernel = morphology_module._marching_cubes_parallel_numba
        for threads, calls in ((2, 1), (1, 0)):
            with (
                patch.object(morphology_module, "_MESH_PARALLEL_MIN", padded.size),
                patch.object(morphology_module, "get_num_threads", return_value=threads),
                patch.object(
                    morphology_module, "_marching_cubes_parallel_numba", wraps=kernel
                ) as parallel,
            ):
                mesh = morphology_module._mesh(padded, np.full(3, 0.5), np.array([0.7, 0.8, 1.5]))
            self.assertEqual(parallel.call_count, calls)
            self.assertEqual(mesh[1].dtype, np.int32)
            for got, expected in zip(mesh, serial, strict=True):
                np.testing.assert_array_equal(got, expected)

    def test_mesh_features_empty_bbox(self):
        # A given bbox with no ROI voxel gives no mesh features.
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[4, 4, 4] = 1
        box = (slice(0, 2), slice(0, 2), slice(0, 2))
        features = calculate_morphology_features(self._create_image(arr), roi_bbox=box)
        self.assertEqual(features, {"volume_voxel_counting_YEKZ": 0.0})

    @patch("pictologics.features.morphology._get_mesh_features")
    def test_shape_features_zero_volume_positive_area(self, mock_get_mesh):
        # Simulate flat mesh: Volume 0, Area > 0
        mock_get_mesh.return_value = (
            {"volume_RNU0": 0.0, "surface_area_C0JK": 10.0},
            np.zeros((3, 3)),
            np.zeros((1, 3)),
        )
        arr = np.zeros((3, 3, 3), dtype=int)
        arr[1, 1, 1] = 1
        mask = self._create_image(arr)
        features = calculate_morphology_features(mask)
        # Code check: if mesh_volume <= 0 or surface_area <= 0: return features
        # So no shape features
        self.assertNotIn("compactness_1_SKGS", features)

    # ----------------------------------------------------------------------
    # Direct Numba Helper Tests (Parallelism Check)
    # ----------------------------------------------------------------------

    def test_max_pairwise_distance_small(self):
        points = np.array([[0, 0, 0], [3, 4, 0]], dtype=float)
        d = _max_pairwise_distance_numba(points)
        self.assertAlmostEqual(d, 5.0)

    def test_max_pairwise_distance_empty(self):
        points = np.array([], dtype=float).reshape(0, 3)
        self.assertEqual(_max_pairwise_distance_numba(points), 0.0)

    def test_max_pairwise_distance_serial_equals_parallel(self):
        # The serial twin (for the morphology worker thread) gives the bits of the
        # parallel kernel, also for one point and for points on a lattice (equal distances)
        from pictologics.features.morphology import _max_pairwise_distance_serial_numba

        rng = np.random.default_rng(5)
        for n in (0, 1, 2, 3, 17, 120):
            for points in (rng.normal(size=(n, 3)) * 40.0, np.round(rng.normal(size=(n, 3)) * 3)):
                a = _max_pairwise_distance_numba(points)
                b = _max_pairwise_distance_serial_numba(points)
                self.assertEqual(np.float64(a).tobytes(), np.float64(b).tobytes())

    def test_morphology_parts_raise_the_first_error_in_feature_order(self):
        # Part 1 (with the bounding box and intensity-weighted features) and part 2 (convex
        # hull and MVEE, here with the serial diameter) give the features of one call, in
        # its order. With two failing parts, the error of the first in the order of the
        # features (hull, bounding box, MVEE, intensity) is raised, as in one call.
        from pictologics.features import morphology as morphology_module

        arr = np.zeros((7, 8, 9), dtype=np.uint8)
        arr[1:5, 2:7, 2:8] = 1
        arr[5, 4, 4] = 1
        mask = self._create_image(arr, spacing=(1.0, 0.8, 1.2))
        image = self._create_image(np.random.default_rng(6).normal(9.0, 2.0, arr.shape))
        whole = calculate_morphology_features(mask, image)
        first, rest = morphology_module._morphology_first(mask, image)
        second = morphology_module._morphology_second(rest, serial=True)
        merged = morphology_module._morphology_merge(first, rest, second)
        self.assertEqual(list(merged.items()), list(whole.items()))
        self.assertIn("maximum_3d_diameter_L0JK", merged)
        pairs = [
            ("_get_convex_hull_features", "_get_bounding_box_features"),
            ("_get_bounding_box_features", "_get_mvee_features"),
            ("_get_mvee_features", "_get_intensity_morphology_features"),
        ]
        for earlier, later in pairs:
            with (
                patch.object(morphology_module, earlier, side_effect=ValueError(earlier)),
                patch.object(morphology_module, later, side_effect=KeyError(later)),
            ):
                with self.assertRaisesRegex(ValueError, earlier):
                    calculate_morphology_features(mask, image)

    def test_mesh_area_volume(self):
        # Simple Tet
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        # Faces: calculate by hand logic or just checking it runs
        # Correct faces for outward normals... just testing it runs without error
        # for regression on non-crash in parallel.
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
        area, vol = _mesh_area_volume_numba(verts, faces)
        self.assertGreater(area, 0.0)
        self.assertGreater(vol, 0.0)

    def test_ombb_extents(self):
        # Points: (1,0,0), (-1,0,0). Center (0,0,0). Evecs Identity.
        verts = np.array([[1, 0, 0], [-1, 0, 0]], dtype=float)
        center = np.array([0, 0, 0], dtype=float)
        evecs = np.eye(3, dtype=float)

        mn, mx = _ombb_extents_numba(verts, center, evecs)
        # x range: -1 to 1. y,z: 0 to 0.
        self.assertAlmostEqual(mn[0], -1.0)
        self.assertAlmostEqual(mx[0], 1.0)
        self.assertAlmostEqual(mn[1], 0.0)
        self.assertAlmostEqual(mx[1], 0.0)

    def test_mesh_area_volume_inverted(self):
        # Inverted normals -> negative volume in calculation -> abs() correction
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        # 0,1,2 is counter-clockwise. Vectorized formula might produce -vol.
        # Swapping nodes 1 and 2 to invert orientation
        faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int32)
        area, vol = _mesh_area_volume_numba(verts, faces)
        self.assertGreater(vol, 0.0)

    def test_mvee_khachiyan_loop_recompute(self):
        # Force > 50 iterations to hit recompute logic
        # Points in a circle/sphere might converge slowly if close to singular or many points?
        # A set of points that converges slowly.
        points = np.random.rand(100, 3)

        # No stop (eps 0) for 60 iterations: the full recompute at iteration 50 runs
        A, c = _mvee_khachiyan_numba(points, eps=0.0, max_iter=60)
        self.assertIsNotNone(A)

    def test_mvee_features_valid(self):
        # Cube vertices: the hull vertices that the MVEE reads
        verts = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=float,
        )

        features = _get_mvee_features(verts, 1.0, 1.0)
        self.assertIn("volume_density_mvee_SWZ1", features)
        self.assertIn("area_density_mvee_BRI8", features)

    @patch("pictologics.features.morphology.ConvexHull")
    def test_convex_hull_valid(self, mock_hull_cls):
        mock_instance = MagicMock()
        # Indices 0..7
        mock_instance.vertices = np.arange(8, dtype=int)
        mock_instance.volume = 123.0
        mock_instance.area = 456.0
        mock_hull_cls.return_value = mock_instance

        mock_instance.points = np.random.rand(8, 3)
        padded = np.pad(np.ones((2, 2, 2), dtype=np.uint8), 1)
        no_hull = (False, np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.int64))
        with patch("pictologics.features.morphology._exact_hull_numba", return_value=no_hull):
            # No exact hull (as for points in one plane): Qhull runs, here a mock
            features, hull = _get_convex_hull_features(
                padded, np.zeros(3), 1.0, 1.0, (1.0, 1.0, 1.0)
            )

        self.assertIsNotNone(hull)
        self.assertEqual(features["volume_density_convex_hull_R3ER"], 1.0 / 123.0)
        self.assertEqual(features["area_density_convex_hull_7T7F"], 1.0 / 456.0)
        self.assertIn("maximum_3d_diameter_L0JK", features)

    def test_bounding_box_features_ombb_coverage(self):
        # Explicit test to cover OMBB branches (lines ~648+)
        verts = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        evecs = np.eye(3, dtype=float)
        features = _get_bounding_box_features(verts, evecs, 1.0, 1.0)
        self.assertIn("volume_density_ombb_ZH1A", features)
        self.assertIn("area_density_ombb_IQYR", features)

    def test_mvee_khachiyan_exceptions(self):
        # Test exceptions in Numba function (Numba disabled allows patching logic inside)
        # 1. Final Inversion Failure (Line 373)
        points = np.random.rand(10, 3)

        # We need to let it run until the end, then fail at final inv(Cov)
        original_inv = np.linalg.inv

        def side_effect_inv(a):
            # Check if this is likely the Cov matrix (3x3) vs X matrix (4x4)
            if a.shape == (3, 3):
                raise np.linalg.LinAlgError("Final inv failed")
            return original_inv(a)

        with patch("numpy.linalg.inv", side_effect=side_effect_inv):
            A, c = _mvee_khachiyan_numba(points, eps=1e-1)
            self.assertIsNone(A)

        # 2. Recompute Inversion Failure (Line 323)
        # Force > 50 iterations by using minimal tolerance 0.0
        # Fail on 2nd call to inv (1st is init, 2nd is recompute at count=50)

        points_sphere = np.random.randn(20, 3)
        points_sphere /= np.linalg.norm(points_sphere, axis=1)[:, np.newaxis]

        call_counter = {"n": 0}

        def side_effect_inv_recompute(a):
            call_counter["n"] += 1
            if call_counter["n"] == 2:
                raise np.linalg.LinAlgError("Recompute failed")
            return original_inv(a)

        with patch("numpy.linalg.inv", side_effect=side_effect_inv_recompute):
            # eps=0.0 forces maximum iterations
            A, c = _mvee_khachiyan_numba(points_sphere, eps=0.0)
            self.assertIsNone(A)
            self.assertIsNone(c)

    def test_intensity_morphology_no_mask_moments(self):
        """Covers _get_intensity_morphology_features fallback when mask_moments=None."""
        arr = np.zeros((5, 5, 5), dtype=int)
        arr[1:4, 1:4, 1:4] = 1
        mask = self._create_image(arr)

        img_arr = np.full((5, 5, 5), 10.0, dtype=float)
        image = self._create_image(img_arr)

        # Call directly without mask_moments to trigger the else branch at line 764
        features = _get_intensity_morphology_features(
            mask, image, mask, mesh_volume=27.0, mask_moments=None
        )
        self.assertIn("integrated_intensity_99N0", features)
        self.assertIn("center_of_mass_shift_KLMA", features)

    def test_get_mesh_features_empty_mask(self):
        """Empty mask -> no nonzero bbox -> mesh features return empty with no mesh data."""
        empty = self._create_image(np.zeros((5, 5, 5), dtype=int))
        features, verts, faces = _get_mesh_features(empty)
        self.assertEqual(features, {})
        self.assertIsNone(verts)
        self.assertIsNone(faces)

    def test_intensity_morphology_empty_intensity_mask(self):
        """Empty intensity mask -> no nonzero bbox -> returns before accumulation."""
        arr = np.zeros((5, 5, 5), dtype=int)
        arr[1:4, 1:4, 1:4] = 1
        mask = self._create_image(arr)
        image = self._create_image(np.full((5, 5, 5), 10.0, dtype=float))
        empty_intensity = self._create_image(np.zeros((5, 5, 5), dtype=int))

        features = _get_intensity_morphology_features(
            mask, image, empty_intensity, mesh_volume=27.0
        )
        self.assertNotIn("integrated_intensity_99N0", features)

    def test_intensity_morphology_reads_float64_or_float32_images(self):
        """An image of another type goes as a float64 copy of the box (the kernel reads the
        types that the import compiles), so with the features of float64."""
        from pictologics.features import morphology as morphology_module

        arr = np.zeros((6, 6, 6), dtype=np.uint8)
        arr[1:5, 1:4, 2:5] = 1
        mask = self._create_image(arr)
        values = np.arange(216).reshape(6, 6, 6) % 17
        float_image = self._create_image(values.astype(np.float64))
        expected = _get_intensity_morphology_features(mask, float_image, mask, mesh_volume=27.0)
        kernel = morphology_module._accumulate_intensity_weighted_moments_numba
        with (
            patch.object(
                morphology_module, "_accumulate_intensity_weighted_moments_numba", wraps=kernel
            ) as spy,
            patch.object(  # a small ROI takes the serial twin: the same spy
                morphology_module, "_accumulate_intensity_weighted_moments_numba_serial", new=spy
            ),
        ):
            for kind in (np.int16, np.float32):
                image = self._create_image(values.astype(kind))
                found = _get_intensity_morphology_features(mask, image, mask, mesh_volume=27.0)
                if kind == np.int16:
                    self.assertEqual(found, expected)
        kinds = [call.args[1].dtype for call in spy.call_args_list]
        self.assertEqual(kinds, [np.float64, np.float32])


def test_the_morphology_kernels_take_their_serial_twins_below_their_gates() -> None:
    # A small ROI takes the serial twin of each parallel kernel (starting the threads costs
    # more than the work); from the gate on, with more than one thread, the kernel runs. Both
    # give the same features.
    from contextlib import ExitStack

    from pictologics.features import morphology as module

    names = [
        "_accumulate_moments_from_mask_numba",
        "_accumulate_intensity_weighted_moments_numba",
        "_mc_counts_numba",
        "_mesh_area_volume_numba",
        "_ombb_extents_numba",
    ]
    gates = ["_MOMENTS_PARALLEL_MIN", "_MC_COUNTS_PARALLEL_MIN", "_MESH_SUM_PARALLEL_MIN"]
    roi = np.zeros((12, 13, 14), dtype=np.uint8)
    roi[2:9, 3:10, 2:11] = 1
    image = Image(
        np.random.default_rng(3).normal(10.0, 2.0, roi.shape), (1.0, 0.9, 1.1), (0.0,) * 3
    )
    mask = Image(roi, image.spacing, image.origin)

    def run(gate: int) -> tuple[dict[str, float], dict[str, int]]:
        with ExitStack() as stack:
            spies = {
                name: stack.enter_context(patch.object(module, name, wraps=getattr(module, name)))
                for base in names
                for name in (base, base + "_serial")
            }
            for name in [*gates, "_OMBB_PARALLEL_MIN"]:
                stack.enter_context(patch.object(module, name, getattr(module, name) * gate))
            features = calculate_morphology_features(mask, image=image, intensity_mask=mask)
        return features, {name: spy.call_count for name, spy in spies.items()}

    small, calls = run(1)
    assert all(calls[name] == 0 and calls[name + "_serial"] > 0 for name in names)
    large, calls = run(0)  # every gate at 0: the parallel kernels
    assert all(calls[name] > 0 and calls[name + "_serial"] == 0 for name in names)
    assert large == small


def _khachiyan_reference(points: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, int]:
    """The steps of the MVEE kernel (Wolfe-Atwood: toward the largest Mahalanobis value or
    away from the smallest weighted one) with the quadratic form q^T invX q of numpy."""
    n, d = points.shape
    d1 = d + 1
    q = np.column_stack([points, np.ones(n)])
    u = np.full(n, 1.0 / n)
    inv = np.linalg.inv(q.T @ q / n)
    count = 0
    while count < 200000:
        g = np.einsum("kr,rc,kc->k", q, inv, q)
        j = int(np.argmax(g))
        weighted = np.flatnonzero(u > 0)
        jm = int(weighted[np.argmin(g[weighted])])
        if g[j] / d1 - 1 <= eps and 1 - g[jm] / d1 <= eps:
            break
        if g[j] / d1 - 1 >= 1 - g[jm] / d1:
            step, value = (g[j] - d1) / (d1 * (g[j] - 1)), g[j]
        else:
            limit = u[jm] / (1 - u[jm])
            step = limit if g[jm] <= 1 else min((d1 - g[jm]) / (d1 * (g[jm] - 1)), limit)
            step, value, j = -step, g[jm], jm
        new = u[j] * (1 - step) + step
        u *= 1 - step
        u[j] = max(new, 0.0)
        if count % 50 == 0 and count > 0:
            inv = np.linalg.inv((q * u[:, None]).T @ q)
        else:
            v = inv @ q[j]
            alpha = step / (1 - step)
            inv = (inv - alpha / (1 + alpha * value) * np.outer(v, v)) / (1 - step)
        count += 1
    c = points.T @ u
    cov = (points * u[:, None]).T @ points - np.outer(c, c)
    return np.linalg.inv(cov) / d, c, count


def test_the_mvee_kernel_takes_the_steps_of_the_quadratic_form() -> None:
    # The kernel reads the quadratic form of each point from its kept products: the steps,
    # and so the ellipsoid, are those of the plain form q^T invX q (40 random point sets).
    rng = np.random.default_rng(21)
    for case in range(40):
        points = rng.normal(0.0, 1.0, (int(rng.integers(8, 40)), 3)) * rng.uniform(0.5, 3.0, 3)
        eps = 1e-3 if case < 4 else 1e-2  # the default, and fewer steps (Python is slow)
        A, c = _mvee_khachiyan_numba(points, eps)
        A_ref, c_ref, _ = _khachiyan_reference(points, eps)
        np.testing.assert_allclose(A, A_ref, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(c, c_ref, rtol=1e-9, atol=1e-12)


def _hull_test_roi(rng: np.random.Generator, case: int) -> np.ndarray:
    """A blob, a box, a one-voxel plate or a line of voxels (by case % 4), never empty."""
    from scipy.ndimage import gaussian_filter

    shape = tuple(int(v) for v in rng.integers(6, 16, 3))
    kind = case % 4
    lo = [int(rng.integers(1, n // 2)) for n in shape]
    hi = [int(rng.integers(n // 2 + 1, n - 1)) for n in shape]
    if kind == 0:
        roi = gaussian_filter(rng.random(shape), 1.5) > 0.5
    else:
        roi = np.zeros(shape, dtype=bool)
        if kind == 2:
            hi[case % 3] = lo[case % 3] + 1
        elif kind == 3:
            for a in range(3):
                if a != case % 3:
                    hi[a] = lo[a] + 1
        roi[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
    roi[lo[0], lo[1], lo[2]] = True
    return roi


def test_the_hull_lattice_keeps_every_hull_vertex() -> None:
    # On 40 masks (blobs, boxes, one-voxel plates and lines, anisotropic spacing) the exact
    # hull on the candidate lattice has the vertex set of Qhull on all mesh vertices, and the
    # candidate points are mesh vertices (the same floats), also with a grid offset. A sphere
    # (more than 64 voxel candidates: the plane filter runs first) and a 3x3x3 cube (fewer)
    # take both paths of the lattice kernel.
    from scipy.spatial import ConvexHull

    from pictologics.features._utils import compute_nonzero_bbox
    from pictologics.features.morphology import (
        _exact_hull_numba,
        _hull_lattice_serial_numba,
        _padded_roi,
        _voxel_hull_candidates_numba,
    )

    def check(roi: np.ndarray, spacing: tuple[float, ...], grid: tuple[int, int, int]) -> int:
        mask = Image(roi.astype(np.uint8), spacing, (0.0, 0.0, 0.0))
        verts = _get_mesh_features(mask, grid_offset=grid)[1]
        bbox = compute_nonzero_bbox(roi)
        offset = np.array([b.start + g for b, g in zip(bbox, grid, strict=True)], dtype=float)
        padded = _padded_roi(roi[bbox].astype(np.uint8))
        lattice, points = _hull_lattice_serial_numba(padded, offset, np.asarray(spacing))
        found, vertices, _ = _exact_hull_numba(lattice)
        full = ConvexHull(verts)
        assert found
        assert {tuple(v) for v in points[vertices]} == {tuple(v) for v in verts[full.vertices]}
        assert {tuple(v) for v in points} <= {tuple(v) for v in verts}
        rows = _row_ends_of(padded)
        return int(_voxel_hull_candidates_numba(*rows).shape[0])

    rng = np.random.default_rng(22)
    for case in range(40):
        spacing = tuple(float(v) for v in rng.uniform(0.5, 2.0, 3))
        grid = (12, 34, 56) if case % 10 == 0 else (0, 0, 0)
        check(_hull_test_roi(rng, case), spacing, grid)
    sphere = (np.indices((15, 15, 15)) - 7.0) ** 2
    assert check(sphere.sum(axis=0) <= 36.0, (0.7, 0.7, 1.25), (0, 0, 0)) > 64
    cube = np.zeros((5, 5, 5), dtype=bool)
    cube[1:4, 1:4, 1:4] = True
    assert check(cube, (1.0, 1.0, 1.0), (0, 0, 0)) <= 64


def _row_ends_of(padded: np.ndarray) -> tuple[np.ndarray, ...]:
    """The six row-end tables of _row_ends_numba, from numpy (size and -1 for an empty row)."""
    inside = padded != 0
    tables = []
    for axis in (2, 1, 0):
        size = padded.shape[axis]
        has = inside.any(axis=axis)
        first = np.where(has, inside.argmax(axis=axis), size)
        last = np.where(has, size - 1 - np.flip(inside, axis=axis).argmax(axis=axis), -1)
        tables += [first.astype(np.int32), last.astype(np.int32)]
    return tuple(tables)


def test_the_row_ends() -> None:
    # The row ends of the kernel are the numpy ends for 1, 2, 3 and 20 chunks of x planes
    # (20 chunks of 11 planes: empty chunks) and for the serial twin.
    from pictologics.features import morphology as module

    padded = np.pad((np.random.default_rng(45).random((9, 10, 8)) < 0.3).astype(np.uint8), 1)
    expected = _row_ends_of(padded)
    runs = [(module._row_ends_numba, n) for n in (1, 2, 3, 20)]
    for kernel, n in [*runs, (module._row_ends_numba_serial, 1)]:
        for got, want in zip(kernel(padded, n), expected, strict=True):
            assert got.dtype == np.int32
            np.testing.assert_array_equal(got, want)


def test_the_voxel_candidates_end_their_rows() -> None:
    # The voxel candidates are the voxels that end each of their three rows, in row order:
    # a random mask, a checkerboard, and a full box (its 8 corners).
    from pictologics.features.morphology import _voxel_hull_candidates_numba

    board = np.pad((np.indices((5, 6, 7)).sum(axis=0) % 2).astype(np.uint8), 1)
    full = np.pad(np.ones((4, 5, 6), dtype=np.uint8), 1)
    blob = np.pad((np.random.default_rng(46).random((6, 7, 8)) < 0.4).astype(np.uint8), 1)
    for padded in (board, full, blob):
        fk, lk, fj, lj, fi, li = rows = _row_ends_of(padded)
        expected = [
            (i, j, k)
            for i, j, k in np.argwhere(padded != 0)
            if k in (fk[i, j], lk[i, j]) and j in (fj[i, k], lj[i, k]) and i in (fi[j, k], li[j, k])
        ]
        got = _voxel_hull_candidates_numba(*rows)
        assert got.dtype == np.int32 and [tuple(v) for v in got] == expected
    assert _voxel_hull_candidates_numba(*_row_ends_of(full)).shape[0] == 8


def test_the_mesh_candidates_are_sorted_in_mesh_order() -> None:
    # With every ROI voxel as a candidate, the mesh candidates are all mesh vertices, in the
    # order of the marching cubes mesh.
    from pictologics.features.morphology import _mesh, _mesh_hull_candidates_numba

    padded = np.pad((np.random.default_rng(47).random((6, 7, 5)) < 0.5).astype(np.uint8), 1)
    lattice = _mesh_hull_candidates_numba(padded, np.argwhere(padded != 0).astype(np.int32))
    verts, _ = _mesh(padded, np.zeros(3), np.ones(3))
    np.testing.assert_array_equal(0.5 * (lattice - 1.0), verts)


def test_the_hull_row_ends_run_in_parallel_above_the_gate() -> None:
    # From _ROW_ENDS_PARALLEL_MIN voxels and _ROW_ENDS_PLANES_PER_CHUNK planes per thread on,
    # the row ends run in parallel (one chunk per thread); with `serial`, one thread or a
    # thin box, the serial lattice kernel runs. Both give the same features.
    from pictologics.features import morphology as module

    cube = np.zeros((10, 9, 8), dtype=np.uint8)
    cube[2:8, 1:7, 2:6] = 1
    padded = np.pad(cube, 1)

    def calls(threads: int, per_chunk: int, serial: bool) -> tuple[tuple[int, int], Any]:
        with (
            patch.object(module, "get_num_threads", return_value=threads),
            patch.object(module, "_ROW_ENDS_PARALLEL_MIN", 1),
            patch.object(module, "_ROW_ENDS_PLANES_PER_CHUNK", per_chunk),
            patch.object(module, "_row_ends_numba", wraps=module._row_ends_numba) as rows,
            patch.object(
                module, "_hull_lattice_serial_numba", wraps=module._hull_lattice_serial_numba
            ) as lattice,
        ):
            features = module._get_convex_hull_features(
                padded, np.zeros(3), 144.0, 120.0, (1.0, 1.0, 1.0), serial
            )
        return (rows.call_count, lattice.call_count), features

    parallel, features = calls(3, 0, False)
    assert parallel == (1, 0)
    for case in ((3, 0, True), (1, 0, False), (3, 100, False)):
        counts, other = calls(*case)
        assert counts == (0, 1) and other[0] == features[0]


def test_the_hull_volume_adds_from_a_hull_vertex() -> None:
    # The volume adds the tetrahedra from a hull vertex: a unit cube far from the origin has
    # the volume 1.0 and the area 6.0 exactly. No triangles: zero area and volume.
    from pictologics.features.morphology import _exact_hull_numba, _hull_area_volume_numba

    corners = np.array([[x, y, z] for x in (0, 2) for y in (0, 2) for z in (0, 2)], dtype=np.int64)
    found, _, triangles = _exact_hull_numba(corners)
    points = 0.5 * corners + np.array([1e5, 2e5, 3e5])
    assert found and _hull_area_volume_numba(points, triangles) == (6.0, 1.0)
    assert _hull_area_volume_numba(points, np.zeros((0, 3), dtype=np.int64)) == (0.0, 0.0)


def test_a_mesh_of_too_many_vertices_raises() -> None:
    # The faces hold vertex numbers as int32: a mesh of more than 2^31 - 1 vertices raises.
    from pictologics.features import morphology as module

    padded = np.pad(np.ones((2, 2, 2), dtype=np.uint8), 1)
    with (
        patch.object(
            module,
            "_mc_counts_numba_serial",
            side_effect=lambda p, e, t, n_v, n_f: n_v.fill(2**31),
        ),
        pytest.raises(ValueError, match="2\\^31 - 1 vertices"),
    ):
        module._mesh(padded, np.zeros(3), np.ones(3))


def test_the_exact_hull_has_the_vertices_of_qhull() -> None:
    # The exact hull of the candidates has the hull vertices of Qhull in the same order, and
    # the volume and area of Qhull to 1e-12, on 30 masks (blobs, boxes and plates with
    # anisotropic spacing). Fewer than four points, points on a line or in a plane have no
    # hull; then Qhull runs, with the features of the exact hull.
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import ConvexHull

    from pictologics.features.morphology import (
        _exact_hull_numba,
        _hull_area_volume_numba,
        _hull_lattice_serial_numba,
    )

    rng = np.random.default_rng(24)
    for case in range(30):
        shape = tuple(int(v) for v in rng.integers(5, 14, 3))
        roi = np.zeros(shape, dtype=bool)
        if case % 3 == 0:
            roi = gaussian_filter(rng.random(shape), 1.2) > 0.5
        else:
            roi[1:-1, 1:-1, 1:-1] = True
            if case % 3 == 2:  # a plate of one voxel
                roi[: shape[0] // 2] = roi[shape[0] // 2 + 1 :] = False
        roi[2, 2, 2] |= case % 3 == 0  # a blob is never empty
        spacing = np.asarray(rng.choice([0.39, 0.7, 1.0, 1.25, 3.0], 3), dtype=np.float64)
        padded = np.pad(roi.astype(np.uint8), 1)
        lattice, points = _hull_lattice_serial_numba(padded, np.zeros(3), spacing)
        found, vertices, triangles = _exact_hull_numba(lattice)
        qhull = ConvexHull(points)
        assert found and np.array_equal(vertices, qhull.vertices)
        area, volume = _hull_area_volume_numba(points, triangles)
        assert abs(volume / qhull.volume - 1) < 1e-12 and abs(area / qhull.area - 1) < 1e-12
    three = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.int64)
    line = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.int64)
    plane = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0], [1, 1, 0]], dtype=np.int64)
    for lattice in (three, line, plane):
        assert not _exact_hull_numba(lattice)[0]
    # A cube with its centre, the lowest x not first: the 8 corners in order, not the centre
    cube = np.array(
        [
            [2, 0, 0],
            [1, 1, 1],
            *([x, y, z] for x in (0, 2) for y in (0, 2) for z in (0, 2) if (x, y, z) != (2, 0, 0)),
        ]
    )
    found, vertices, _ = _exact_hull_numba(cube.astype(np.int64))
    assert found and vertices.tolist() == [0, 2, 3, 4, 5, 6, 7, 8]
    hull_input = (padded, np.zeros(3), 1.0, 1.0, tuple(spacing))
    exact, exact_points = _get_convex_hull_features(*hull_input)
    no_hull = (False, np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.int64))
    with patch("pictologics.features.morphology._exact_hull_numba", return_value=no_hull):
        by_qhull, qhull_points = _get_convex_hull_features(*hull_input)
    np.testing.assert_array_equal(exact_points, qhull_points)
    assert exact.keys() == by_qhull.keys()
    for key, value in exact.items():
        assert abs(value / by_qhull[key] - 1) < 1e-12


if __name__ == "__main__":
    unittest.main()


def test_the_moments_kernel_sums_each_row() -> None:
    # The moments of the row form are the numpy sums over the ROI voxels, exactly: the
    # count and the nine sums of a random uint8 box, for the kernel and its serial twin.
    from pictologics.features import morphology as module

    box = np.random.default_rng(41).integers(0, 4, (9, 10, 8)).astype(np.uint8)
    i, j, k = (axis.astype(np.float64) for axis in np.nonzero(box))
    expected = (
        i.size, i.sum(), j.sum(), k.sum(), (i * i).sum(), (j * j).sum(), (k * k).sum(),
        (i * j).sum(), (i * k).sum(), (j * k).sum(),
    )  # fmt: skip
    for kernel in (
        module._accumulate_moments_from_mask_numba,
        module._accumulate_moments_from_mask_numba_serial,
    ):
        assert tuple(kernel(box)) == expected


def test_the_column_stats_are_those_of_numpy() -> None:
    # The scalar column stats give numpy's minimum, maximum and mean (rows added in order).
    from pictologics.features.morphology import _column_stats_numba

    rng = np.random.default_rng(42)
    for n in (1, 2, 7, 1000):
        points = rng.normal(0.0, 10.0, (n, 3))
        lo, hi, mean = _column_stats_numba(points)
        assert np.array_equal(lo, points.min(axis=0)) and np.array_equal(hi, points.max(axis=0))
        assert np.array_equal(mean, points.mean(axis=0))


def test_the_diameter_takes_the_serial_kernel_below_its_gate() -> None:
    # Below _DIAMETER_PARALLEL_MIN hull vertices, with one thread, or with `serial`, the
    # maximum diameter comes from the serial kernel; else from the parallel one.
    from pictologics.features import _utils
    from pictologics.features import morphology as module

    cube = np.zeros((8, 8, 8), dtype=np.uint8)
    cube[1:7, 1:7, 1:7] = 1
    padded = np.pad(cube, 1)

    def calls(gate: int, threads: int, serial: bool) -> tuple[int, int]:
        with (
            patch.object(module, "_DIAMETER_PARALLEL_MIN", gate),
            patch.object(_utils, "get_num_threads", return_value=threads),
            patch.object(
                module, "_max_pairwise_distance_numba", wraps=module._max_pairwise_distance_numba
            ) as parallel,
            patch.object(
                module,
                "_max_pairwise_distance_serial_numba",
                wraps=module._max_pairwise_distance_serial_numba,
            ) as serial_kernel,
        ):
            module._get_convex_hull_features(padded, np.zeros(3), 216.0, 216.0, (1.0,) * 3, serial)
        return parallel.call_count, serial_kernel.call_count

    assert calls(module._DIAMETER_PARALLEL_MIN, 4, False) == (0, 1)
    assert calls(2, 4, False) == (1, 0)
    assert calls(2, 1, False) == (0, 1)
    assert calls(2, 4, True) == (0, 1)


def test_the_parallel_marching_cubes_in_any_number_of_blocks() -> None:
    # The parallel marching cubes give the serial mesh for any number of blocks: one block,
    # blocks after the first (they count the plane before them again), and more blocks than
    # planes (empty blocks).
    from pictologics.features import morphology as module

    rng = np.random.default_rng(43)
    padded = np.pad((rng.random((7, 8, 6)) < 0.5).astype(np.uint8), 1)
    offset, spacing = np.full(3, 0.5), np.array([0.7, 0.8, 1.5])
    expected = module._mesh(padded, offset, spacing, False)
    for n_chunks in (1, 2, 3, 7, 50):
        with patch.object(module, "get_num_threads", return_value=n_chunks):
            mesh = module._mesh(padded, offset, spacing, True)
        for got, want in zip(mesh, expected, strict=True):
            np.testing.assert_array_equal(got, want)


def test_the_mvee_of_few_points_encloses_them() -> None:
    # The MVEE takes four points at a time and then the rest: for 4 to 9 points (every
    # remainder), the ellipsoid encloses every point (to the eps of the stop rule).
    rng = np.random.default_rng(44)
    for n in (4, 5, 6, 7, 8, 9):
        points = rng.normal(0.0, 1.0, (n, 3))
        A, c = _mvee_khachiyan_numba(points)
        assert A is not None and c is not None
        d = points - c
        assert np.all(np.einsum("ki,ij,kj->k", d, A, d) <= 1.0 + 2e-3)


def test_the_padded_roi_is_one_copy_with_a_zero_border() -> None:
    # _padded_roi keeps the values of a uint8 box and gives another type as box != 0 (a NaN
    # is ROI), with a zero border; the mesh from it equals the mesh of the mask, and an
    # int16 mask gives the features of the same ROI as uint8.
    from pictologics.features.morphology import _padded_roi

    box = np.zeros((3, 4, 5), dtype=np.uint8)
    box[1, 2, 3], box[2, 0, 0] = 7, 255
    padded = _padded_roi(box)
    assert padded.shape == (5, 6, 7) and padded.dtype == np.uint8
    assert np.array_equal(padded[1:-1, 1:-1, 1:-1], box)
    inner = np.zeros(padded.shape, dtype=bool)
    inner[1:-1, 1:-1, 1:-1] = True
    assert not padded[~inner].any()
    values = np.zeros((3, 4, 5))
    values[0, 1, 2], values[1, 1, 1], values[2, 3, 4] = -3.0, np.nan, 0.5
    signed_box = np.zeros((3, 4, 5), dtype=np.int16)
    signed_box[0, 1, 2], signed_box[2, 3, 4] = -3, 5
    for other in (values != 0, signed_box, values):  # bool, int16, float64 with a NaN
        inside = _padded_roi(other)[1:-1, 1:-1, 1:-1]
        assert np.array_equal(inside, (other != 0).astype(np.uint8))
    roi = np.zeros((10, 10, 10), dtype=np.uint8)
    roi[2:7, 3:8, 2:9] = 1
    roi[4, 4, 1] = 1
    mask = Image(roi, (1.0, 0.9, 1.1), (0.0, 0.0, 0.0))
    bbox = (slice(2, 7), slice(3, 8), slice(1, 9))
    plain = _get_mesh_features(mask, roi_bbox=bbox)
    given = _get_mesh_features(mask, roi_bbox=bbox, padded=_padded_roi(roi[bbox]))
    assert plain[0] == given[0]
    for got, want in zip(given[1:], plain[1:], strict=True):
        np.testing.assert_array_equal(got, want)
    signed = Image(roi.astype(np.int16) * -2, mask.spacing, mask.origin)
    assert calculate_morphology_features(signed) == calculate_morphology_features(mask)


def test_the_mvee_stops_on_the_mahalanobis_gap() -> None:
    # The away steps drop a point inside the ellipsoid: the 8 corners of a cube and its
    # centre give the sphere through the corners (A = I / 3, c = 0), and every corner
    # lies within 1 + eps. The volume of a random set is that of a converged fit within
    # about eps. One iteration (max_iter 1) still gives an ellipsoid.
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    points = np.vstack([corners, [[0.0, 0.0, 0.0]]])
    A, c = _mvee_khachiyan_numba(points)
    assert A is not None and c is not None
    np.testing.assert_allclose(A, np.eye(3) / 3.0, rtol=0.0, atol=2e-3)
    np.testing.assert_allclose(c, 0.0, atol=1e-5)
    d = corners - c
    assert np.all(np.einsum("ki,ij,kj->k", d, A, d) <= 1.0 + 2e-3)
    blob = np.random.default_rng(54).normal(size=(20, 3)) * np.array([1.0, 2.0, 0.5])
    A_fit, _ = _mvee_khachiyan_numba(blob)
    A_best, _ = _mvee_khachiyan_numba(blob, eps=1e-8)
    assert A_fit is not None and A_best is not None
    volume_fit, volume_best = (np.linalg.det(m) ** -0.5 for m in (A_fit, A_best))
    assert abs(volume_fit / volume_best - 1.0) < 2e-3
    assert _mvee_khachiyan_numba(blob, max_iter=1)[0] is not None
