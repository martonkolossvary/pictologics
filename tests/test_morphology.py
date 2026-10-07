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

    def test_convex_hull_few_points(self):
        # 3 points -> ConvexHull needs 4 for 3D
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        features, hull = _get_convex_hull_features(verts, 1.0, 1.0, (1.0, 1.0, 1.0))
        self.assertIsNone(hull)
        self.assertEqual(features, {})

    def test_hull_candidates_keep_the_hull(self):
        # Qhull on the candidate vertices finds the same hull vertices, in the same order,
        # and the same volume and area (to rounding), as Qhull on all mesh vertices.
        from scipy.spatial import ConvexHull

        from pictologics.features.morphology import _hull_candidates_numba

        rng = np.random.default_rng(2)
        box = np.zeros((9, 10, 8))
        box[2:7, 3:8, 2:6] = 1  # flat faces: many coplanar vertices
        blob = (rng.random((9, 10, 8)) < 0.5).astype(float)
        blob[[0, -1]] = blob[:, [0, -1]] = blob[:, :, [0, -1]] = 0
        for arr, spacing in ((box, (1.0, 1.0, 1.0)), (blob, (0.8, 0.9, 2.5))):
            _, verts, _ = _get_mesh_features(self._create_image(arr, spacing))
            candidates = _hull_candidates_numba(verts, np.asarray(spacing))
            self.assertLess(len(candidates), len(verts))
            full, reduced = ConvexHull(verts), ConvexHull(verts[candidates])
            np.testing.assert_array_equal(reduced.points[reduced.vertices], verts[full.vertices])
            self.assertAlmostEqual(reduced.volume, full.volume, delta=1e-12 * full.volume)
            self.assertAlmostEqual(reduced.area, full.area, delta=1e-12 * full.area)

    def test_convex_hull_coplanar(self):
        # 4 points on a plane -> Volume 0, scipy might error or return flat hull.
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
        features, hull = _get_convex_hull_features(verts, 1.0, 1.0, (1.0, 1.0, 1.0))
        # Should catch exception or return None
        self.assertIsNone(hull)

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
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=int)
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
        faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=int)
        area, vol = _mesh_area_volume_numba(verts, faces)
        self.assertGreater(vol, 0.0)

    def test_mvee_khachiyan_loop_recompute(self):
        # Force > 50 iterations to hit recompute logic
        # Points in a circle/sphere might converge slowly if close to singular or many points?
        # A set of points that converges slowly.
        points = np.random.rand(100, 3)

        # Test it runs without error with low tolerance
        A, c = _mvee_khachiyan_numba(points, tol=1e-7)
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

        verts = np.random.rand(8, 3)
        mock_instance.points = verts
        no_hull = (False, np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.int64))
        with patch("pictologics.features.morphology._exact_hull_numba", return_value=no_hull):
            # No exact hull (as for points in one plane): Qhull runs, here a mock
            features, hull = _get_convex_hull_features(verts, 1.0, 1.0, (1.0, 1.0, 1.0))

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
            A, c = _mvee_khachiyan_numba(points, tol=1e-1)
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
            # tol=0.0 forces maximum iterations
            A, c = _mvee_khachiyan_numba(points_sphere, tol=0.0)
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


def _khachiyan_reference(points: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray, int]:
    """The steps of the MVEE kernel with the quadratic form q^T invX q of numpy."""
    n, d = points.shape
    q = np.column_stack([points, np.ones(n)])
    u = np.full(n, 1.0 / n)
    inv = np.linalg.inv(q.T @ q / n)
    err, count = 1.0, 0
    while err > tol and count < 1000:
        g = np.einsum("kr,rc,kc->k", q, inv, q)
        j = int(np.argmax(g))
        step = (g[j] - (d + 1)) / ((d + 1) * (g[j] - 1))
        err = step * np.sqrt(np.sum(u**2) - u[j] ** 2 + (1 - u[j]) ** 2)
        u *= 1 - step
        u[j] += step
        if count % 50 == 0 and count > 0:
            inv = np.linalg.inv((q * u[:, None]).T @ q)
        else:
            v = inv @ q[j]
            alpha = step / (1 - step)
            inv = (inv - alpha / (1 + alpha * g[j]) * np.outer(v, v)) / (1 - step)
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
        tol = 0.001 if case < 4 else 0.01  # the default, and fewer steps (Python is slow)
        A, c = _mvee_khachiyan_numba(points, tol)
        A_ref, c_ref, _ = _khachiyan_reference(points, tol)
        np.testing.assert_allclose(A, A_ref, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(c, c_ref, rtol=1e-9, atol=1e-12)


def test_the_hull_candidates_keep_every_hull_vertex() -> None:
    # The candidates are the line ends that are 2-D hull vertices in their three axis planes:
    # on 40 masks (blobs, boxes, one-voxel plates and lines, anisotropic spacing), Qhull on
    # the candidates finds the hull vertices and the volume and area of Qhull on all mesh
    # vertices, from fewer points than the line ends.
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import ConvexHull

    from pictologics.features.morphology import (
        _hull_candidates_numba,
        _line_end_candidates_numba,
    )

    rng = np.random.default_rng(22)
    for case in range(40):
        shape = tuple(int(v) for v in rng.integers(6, 16, 3))
        kind = case % 4
        lo = [int(rng.integers(1, n // 2)) for n in shape]
        hi = [int(rng.integers(n // 2 + 1, n - 1)) for n in shape]
        if kind == 0:  # a blob
            roi = gaussian_filter(rng.random(shape), 1.5) > 0.5
        else:
            roi = np.zeros(shape, dtype=bool)
            if kind == 2:  # a one-voxel plate
                hi[case % 3] = lo[case % 3] + 1
            elif kind == 3:  # a line of voxels
                for a in range(3):
                    if a != case % 3:
                        hi[a] = lo[a] + 1
            roi[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
        roi[lo[0], lo[1], lo[2]] = True  # never empty
        spacing = tuple(float(v) for v in rng.uniform(0.5, 2.0, 3))
        mask = Image(roi.astype(np.uint8), spacing, (0.0, 0.0, 0.0))
        verts = _get_mesh_features(mask)[1]
        grid_spacing = np.asarray(spacing, dtype=np.float64)
        candidates = _hull_candidates_numba(verts, grid_spacing)
        assert len(candidates) <= len(_line_end_candidates_numba(verts, grid_spacing)[0])
        full, kept = ConvexHull(verts), ConvexHull(verts[candidates])
        assert {tuple(v) for v in full.points[full.vertices]} == {
            tuple(v) for v in kept.points[kept.vertices]
        }
        assert abs(kept.volume / full.volume - 1) < 1e-12
        assert abs(kept.area / full.area - 1) < 1e-12


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
        _hull_candidates_numba,
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
        mask = Image(roi.astype(np.uint8), tuple(spacing), (0.0, 0.0, 0.0))
        verts = _get_mesh_features(mask)[1]
        points = verts[_hull_candidates_numba(verts, spacing)]
        found, vertices, triangles = _exact_hull_numba(
            np.rint(2.0 * points / spacing).astype(np.int64)
        )
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
    exact, exact_points = _get_convex_hull_features(verts, 1.0, 1.0, tuple(spacing))
    no_hull = (False, np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.int64))
    with patch("pictologics.features.morphology._exact_hull_numba", return_value=no_hull):
        by_qhull, qhull_points = _get_convex_hull_features(verts, 1.0, 1.0, tuple(spacing))
    np.testing.assert_array_equal(exact_points, qhull_points)
    assert exact.keys() == by_qhull.keys()
    for key, value in exact.items():
        assert abs(value / by_qhull[key] - 1) < 1e-12


if __name__ == "__main__":
    unittest.main()
