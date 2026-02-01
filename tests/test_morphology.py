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
        self.assertAlmostEqual(
            features["integrated_intensity_99N0"], vol * 10.0, delta=1e-4
        )
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
        features, hull = _get_convex_hull_features(verts, 1.0, 1.0)
        self.assertIsNone(hull)
        self.assertEqual(features, {})

    def test_convex_hull_coplanar(self):
        # 4 points on a plane -> Volume 0, scipy might error or return flat hull.
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
        features, hull = _get_convex_hull_features(verts, 1.0, 1.0)
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
        features = _get_mvee_features(None, np.array([]), 1.0, 1.0)
        self.assertEqual(features, {})

    @patch("mcubes.marching_cubes")
    def test_marching_cubes_failure(self, mock_mc):
        mock_mc.side_effect = ValueError("Marching cubes failed")
        arr = np.zeros((5, 5, 5), dtype=int)
        arr[2, 2, 2] = 1
        mask = self._create_image(arr)
        features = calculate_morphology_features(mask)
        # Voxel volume logic is separate, so it should exist
        self.assertIn("volume_voxel_counting_YEKZ", features)
        self.assertNotIn("volume_RNU0", features)

    @patch("mcubes.marching_cubes")
    def test_marching_cubes_empty(self, mock_mc):
        mock_mc.return_value = (
            np.array([]).reshape(0, 3),
            np.array([], dtype=int).reshape(0, 3),
        )
        arr = np.zeros((5, 5, 5), dtype=int)
        arr[2, 2, 2] = 1
        mask = self._create_image(arr)
        features = calculate_morphology_features(mask)
        self.assertNotIn("volume_RNU0", features)

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

    @patch("pictologics.features.morphology.ConvexHull")
    def test_mvee_features_valid(self, mock_hull_cls):
        # Mock ConvexHull to bypass environment issues (Numpy 2.0 vs Scipy)
        mock_instance = MagicMock()
        mock_instance.vertices = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
        mock_instance.volume = 1.0
        mock_instance.area = 6.0
        mock_hull_cls.return_value = mock_instance

        # Cube vertices
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

        features = _get_mvee_features(mock_instance, verts, 1.0, 1.0)
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
        features, hull = _get_convex_hull_features(verts, 1.0, 1.0)

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

        def side_effect_inv_recompute(a):
            # We want to fail when it recomputes.
            # But making it loop 50 times with random data is hard to control.
            # Instead, we just assume line 323 is unreachable with well-behaved data in standard tests
            # unless we force the loop interaction.
            # Given difficulty, we might skip this if 99% is acceptable, OR:
            # We can force the loop counter? No, it's local var.
            return original_inv(a)

        # Since I cannot easily force the loop to 50 and then fail without rewriting logic or deep mocking,
        # I will accept hitting the final exception which is easy (shape check).
        # The recompute exception is very rare (numerical conditioning).
        # I'll add the final exception test which I wrote above.


if __name__ == "__main__":
    unittest.main()
