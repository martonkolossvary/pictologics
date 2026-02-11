import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from pictologics.filters.base import (
    BoundaryCondition,
    _normalized_convolve1d,
    _normalized_gaussian_laplace,
    _normalized_separable_convolve_3d,
    _normalized_uniform_filter,
    _prepare_masked_image,
    ensure_float32,
    get_scipy_mode,
)


class TestBaseFilters:
    def test_ensure_float32(self):
        # Test int input
        arr_int = np.array([1, 2, 3], dtype=int)
        arr_float = ensure_float32(arr_int)
        assert arr_float.dtype == np.float32
        assert_array_equal(arr_float, arr_int.astype(np.float32))

        # Test float16 input
        arr_f16 = np.array([1.5, 2.5], dtype=np.float16)
        arr_f32 = ensure_float32(arr_f16)
        assert arr_f32.dtype == np.float32
        assert_array_almost_equal(arr_f32, arr_f16.astype(np.float32))

        # Test float64 input (should remain float64 or be cast if compliant, implementation says:
        # "return image.astype(np.float32) if image.dtype == np.float16 else image" for floats.
        # Wait, ensure_float32 implementation:
        # if np.issubdtype(image.dtype, np.floating):
        #     return image.astype(np.float32) if image.dtype == np.float16 else image
        # return image.astype(np.float32)
        # So float64 should stay float64.
        arr_f64 = np.array([1.1, 2.2], dtype=np.float64)
        arr_out = ensure_float32(arr_f64)
        assert arr_out.dtype == np.float64
        assert_array_equal(arr_out, arr_f64)

    def test_get_scipy_mode(self):
        assert get_scipy_mode(BoundaryCondition.ZERO) == "constant"
        assert get_scipy_mode(BoundaryCondition.NEAREST) == "nearest"
        assert get_scipy_mode(BoundaryCondition.PERIODIC) == "wrap"
        assert get_scipy_mode(BoundaryCondition.MIRROR) == "reflect"

    def test_prepare_masked_image(self):
        image = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        mask = np.array([True, False, True], dtype=bool)

        # Default fill 0.0
        result = _prepare_masked_image(image, mask)
        expected = np.array([10.0, 0.0, 30.0], dtype=np.float32)
        assert_array_equal(result, expected)

        # Custom fill
        result_fill = _prepare_masked_image(image, mask, fill_value=5.0)
        expected_fill = np.array([10.0, 5.0, 30.0], dtype=np.float32)
        assert_array_equal(result_fill, expected_fill)

    def test_normalized_uniform_filter(self):
        # 1D case for simplicity (function handles n-D via scipy)
        # Image: [100, 100, 100], Mask: [T, F, T]
        # Kernel size 3.
        # Index 0: window [0, 100, 100] (padded?), [0, 0, 100] (0 is padding?) depending on mode.
        # Let's use mode='constant' (0 padding)

        image = np.array([100.0, 200.0, 100.0], dtype=np.float32)
        mask = np.array([True, False, True], dtype=bool)

        # _normalized_uniform_filter expects float image and bol mask
        # It internally zeroes invalid voxels.
        # valid_image = [100, 0, 100]
        # mask_float = [1, 0, 1]

        # At index 1 (center):
        # window (size 3) on valid_image: [100, 0, 100] -> sum = 200
        # window on mask: [1, 0, 1] -> sum = 2
        # result = 200 / 2 = 100.
        # This correctly interpolates the missing center value from neighbors!

        result, valid_out = _normalized_uniform_filter(
            image, mask, size=3, mode="constant"
        )

        # Center pixel check
        assert np.isclose(result[1], 100.0)

        # Output validity mask
        # weight_sum at center is 2/3 = 0.66 > 0.01 -> Valid
        assert valid_out[1]

    def test_normalized_gaussian_laplace(self):
        # Test that it runs and returns reasonable shapes
        shape = (10, 10, 10)
        image = np.zeros(shape, dtype=np.float32)
        image[5, 5, 5] = 100.0
        mask = np.ones(shape, dtype=bool)
        mask[5, 5, 5] = False  # Center invalid (e.g. sentinel)

        # Should not crash
        result, valid_out = _normalized_gaussian_laplace(
            image, mask, sigma=1.0, mode="constant"
        )
        assert result.shape == shape
        assert valid_out.shape == shape

        # Center should be filled/interpolated?
        # Normalized LoG is an approximation.
        assert not np.isnan(result).any()

    def test_normalized_convolve1d(self):
        image = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        mask = np.array([True, False, True], dtype=bool)
        # Kernel: [0.5, 1.0, 0.5] smoothing
        kernel = np.array([0.5, 1.0, 0.5], dtype=np.float32)

        # valid_image: [10, 0, 30]
        # At index 1:
        # data window: [10, 0, 30] * [0.5, 1.0, 0.5] = 5 + 0 + 15 = 20
        # weight window: [1, 0, 1] * [0.5, 1.0, 0.5] = 0.5 + 0 + 0.5 = 1.0
        # result = 20 / 1.0 = 20. Correct interpolation!

        result, valid_out = _normalized_convolve1d(
            image, mask, kernel, axis=0, mode="constant"
        )

        assert np.isclose(result[1], 20.0)

    def test_normalized_separable_convolve_3d(self):
        shape = (5, 5, 5)
        image = np.random.rand(*shape).astype(np.float32)
        mask = np.ones(shape, dtype=bool)
        g = np.array([0.2, 0.6, 0.2], dtype=np.float32)

        result, valid_out = _normalized_separable_convolve_3d(
            image, mask, g, g, g, mode="constant"
        )
        assert result.shape == shape
        assert valid_out.shape == shape
