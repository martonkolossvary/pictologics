"""Tests for pictologics.filters module."""

import os
import signal
import time
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose, assert_array_equal

from pictologics.filters import (
    LAWS_KERNELS,
    BoundaryCondition,
    FilterResult,
    gabor_filter,
    gaussian_filter,
    laplacian_of_gaussian,
    laws_filter,
    mean_filter,
    riesz_log,
    riesz_simoncelli,
    riesz_transform,
    simoncelli_wavelet,
    wavelet_transform,
)
from pictologics.filters.base import (
    _apply_with_boundary_padding,
    _float32_cut,
    _normalized_convolve1d,
    _normalized_gaussian_laplace,
    _normalized_separable_convolve_3d,
    _normalized_uniform_filter,
    _prepare_masked_image,
    ensure_float32,
    get_scipy_mode,
    resolve_boundary,
)
from pictologics.filters.gabor import (
    _apply_gabor_to_plane,
    _create_gabor_kernel_2d,
    _create_gabor_kernel_2d_anisotropic,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def small_3d_image():
    """Small 3D test image (8x8x8)."""
    np.random.seed(42)
    return np.random.rand(8, 8, 8).astype(np.float32)


@pytest.fixture
def impulse_3d():
    """3D impulse response (single non-zero voxel at center)."""
    img = np.zeros((9, 9, 9), dtype=np.float32)
    img[4, 4, 4] = 1.0
    return img


# =============================================================================
# Test base.py
# =============================================================================


class TestBoundaryCondition:
    """Tests for BoundaryCondition enum."""

    def test_zero_boundary(self):
        assert BoundaryCondition.ZERO.value == "constant"

    def test_nearest_boundary(self):
        assert BoundaryCondition.NEAREST.value == "nearest"

    def test_periodic_boundary(self):
        assert BoundaryCondition.PERIODIC.value == "wrap"

    def test_mirror_boundary(self):
        assert BoundaryCondition.MIRROR.value == "reflect"


class TestFilterResult:
    """Tests for FilterResult dataclass."""

    def test_filter_result_creation(self):
        arr = np.ones((5, 5, 5), dtype=np.float32)
        result = FilterResult(response_map=arr, filter_name="test", filter_params={"size": 3})
        assert result.filter_name == "test"
        assert result.filter_params == {"size": 3}

    def test_filter_result_shape(self):
        arr = np.ones((5, 6, 7), dtype=np.float32)
        result = FilterResult(response_map=arr, filter_name="test", filter_params={})
        assert result.shape == (5, 6, 7)

    def test_filter_result_dtype(self):
        arr = np.ones((5, 5, 5), dtype=np.float32)
        result = FilterResult(response_map=arr, filter_name="test", filter_params={})
        assert result.dtype == np.float32


class TestEnsureFloat32:
    """Tests for ensure_float32 function."""

    def test_int_to_float32(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = ensure_float32(arr)
        assert result.dtype == np.float32

    def test_float32_unchanged(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = ensure_float32(arr)
        assert result.dtype == np.float32

    def test_float64_unchanged(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = ensure_float32(arr)
        assert result.dtype == np.float64

    def test_float16_to_float32(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        result = ensure_float32(arr)
        assert result.dtype == np.float32


class TestGetScipyMode:
    """Tests for get_scipy_mode function."""

    def test_all_boundary_conditions(self):
        assert get_scipy_mode(BoundaryCondition.ZERO) == "constant"
        assert get_scipy_mode(BoundaryCondition.NEAREST) == "nearest"
        assert get_scipy_mode(BoundaryCondition.PERIODIC) == "wrap"
        assert get_scipy_mode(BoundaryCondition.MIRROR) == "reflect"


class TestResolveBoundary:
    """Tests for resolve_boundary function."""

    def test_boundary_condition_returned_unchanged(self):
        assert resolve_boundary(BoundaryCondition.MIRROR) is BoundaryCondition.MIRROR

    def test_case_insensitive_string(self):
        assert resolve_boundary("NEAREST") == BoundaryCondition.NEAREST
        assert resolve_boundary("nearest") == BoundaryCondition.NEAREST

    def test_all_member_names_resolve(self):
        for member in BoundaryCondition:
            assert resolve_boundary(member.name.lower()) is member

    def test_invalid_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown boundary condition"):
            resolve_boundary("bogus")


class TestApplyWithBoundaryPadding:
    """Tests for _apply_with_boundary_padding function."""

    def test_periodic_calls_func_directly(self):
        # PERIODIC must be the exact current (no boundary handling) code path:
        # no padding, no copy, `func` called on `image` itself.
        image = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        calls = []

        def func(arr, crop, add=0.0):
            calls.append((arr, crop))
            return arr + add

        result = _apply_with_boundary_padding(func, image, BoundaryCondition.PERIODIC, 5, add=3.0)
        assert calls[0][0] is image and calls[0][1] is None
        assert_array_equal(result, image + 3.0)

    def test_non_periodic_crops_back_to_original_shape(self):
        image = np.ones((4, 4, 4), dtype=np.float32)

        def func(arr, crop):
            assert arr.shape == (8, 8, 8)  # 4 + 2*2 padding
            return _float32_cut(arr, crop)

        result = _apply_with_boundary_padding(func, image, BoundaryCondition.ZERO, 2)
        assert_array_equal(result, image)
        assert result.flags.c_contiguous and result.base is None  # the padding is not kept

    def test_pad_width_as_tuple(self):
        image = np.ones((4, 6, 8), dtype=np.float32)

        def func(arr, crop):
            assert arr.shape == (6, 8, 10)
            return _float32_cut(arr, crop)

        result = _apply_with_boundary_padding(func, image, BoundaryCondition.NEAREST, (1, 1, 1))
        assert result.shape == image.shape

    def test_pad_width_capped_to_axis_length(self):
        image = np.ones((3, 3, 3), dtype=np.float32)

        def func(arr, crop):
            # Requested pad of 100 is capped to the axis length (3), so the
            # padded shape is 3 + 2*3 = 9 per axis, not 3 + 2*100.
            assert arr.shape == (9, 9, 9)
            return _float32_cut(arr, crop)

        result = _apply_with_boundary_padding(func, image, BoundaryCondition.MIRROR, 100)
        assert result.shape == image.shape

    def test_zero_padding_fills_with_zero(self):
        image = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)  # shape (1, 1, 4)
        captured = {}

        def func(arr, crop):
            captured["arr"] = arr.copy()
            return _float32_cut(arr, crop)

        result = _apply_with_boundary_padding(func, image, BoundaryCondition.ZERO, (0, 0, 2))
        assert_array_equal(
            captured["arr"][0, 0], np.array([0, 0, 1, 2, 3, 4, 0, 0], dtype=np.float32)
        )
        assert_array_equal(result, image)

    def test_nearest_padding_replicates_edge(self):
        image = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
        captured = {}

        def func(arr, crop):
            captured["arr"] = arr.copy()
            return _float32_cut(arr, crop)

        _apply_with_boundary_padding(func, image, BoundaryCondition.NEAREST, (0, 0, 2))
        assert_array_equal(
            captured["arr"][0, 0], np.array([1, 1, 1, 2, 3, 4, 4, 4], dtype=np.float32)
        )

    def test_mirror_padding_reflects(self):
        image = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
        captured = {}

        def func(arr, crop):
            captured["arr"] = arr.copy()
            return _float32_cut(arr, crop)

        _apply_with_boundary_padding(func, image, BoundaryCondition.MIRROR, (0, 0, 2))
        assert_array_equal(
            captured["arr"][0, 0], np.array([2, 1, 1, 2, 3, 4, 4, 3], dtype=np.float32)
        )


def test_float32_cut_copies_large_arrays_on_slabs() -> None:
    """A large cut copies on slabs in threads, with the values of one numpy cast."""
    from pictologics.filters import base

    response = np.random.default_rng(30).normal(size=(9, 10, 11))
    crop = (slice(1, 8), slice(2, 9), slice(1, 10))
    with patch.object(base, "_PARALLEL_COPY_MIN", 1), patch.object(base, "_SLAB_MIN_SIZE", 1):
        got = base._float32_cut(response, crop)
    assert_array_equal(got, response[crop].astype(np.float32))
    assert got.flags.c_contiguous and got.base is None


class TestBaseInternalHelpers:
    """Tests for the masked/normalized convolution helpers in base.py."""

    def test_prepare_masked_image(self):
        image = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        mask = np.array([True, False, True], dtype=bool)

        # Default fill 0.0 zeros the masked-out voxel
        assert_array_equal(
            _prepare_masked_image(image, mask),
            np.array([10.0, 0.0, 30.0], dtype=np.float32),
        )
        # Custom fill value
        assert_array_equal(
            _prepare_masked_image(image, mask, fill_value=5.0),
            np.array([10.0, 5.0, 30.0], dtype=np.float32),
        )

    def test_normalized_uniform_filter(self):
        # Normalized convolution interpolates the invalid center from its neighbours:
        # valid_image=[100,0,100] summed over size-3 window = 200, weight sum = 2 -> 100.
        image = np.array([100.0, 200.0, 100.0], dtype=np.float32)
        mask = np.array([True, False, True], dtype=bool)

        result, valid_out = _normalized_uniform_filter(image, mask, size=3, mode="constant")
        assert np.isclose(result[1], 100.0)
        assert valid_out[1]  # weight_sum 2/3 > threshold -> valid

    def test_normalized_gaussian_laplace(self):
        shape = (10, 10, 10)
        image = np.zeros(shape, dtype=np.float32)
        image[5, 5, 5] = 100.0
        mask = np.ones(shape, dtype=bool)
        mask[5, 5, 5] = False  # invalid center (e.g. sentinel)

        result, valid_out = _normalized_gaussian_laplace(image, mask, sigma=1.0, mode="constant")
        assert result.shape == shape
        assert valid_out.shape == shape
        assert not np.isnan(result).any()

    def test_threaded_passes_match_scipy(self):
        # The slab passes in threads give scipy's values bit for bit, and a zero sigma copies
        # the image, as scipy does. The mean sums each window on its own: scipy's values to
        # one float32 unit, also in place.
        from scipy.ndimage import convolve1d, gaussian_laplace, uniform_filter

        from pictologics.filters import base

        rng = np.random.default_rng(8)
        image = rng.normal(0.0, 10.0, (9, 10, 11)).astype(np.float32)
        kernels = (
            np.array([1.0, 2.0, 1.0]),
            np.array([-1.0, 0.0, 1.0]),
            np.array([1.0, -2.0, 1.0]),
        )
        expected_conv = convolve1d(image, kernels[0], axis=0, mode="mirror")
        convolve1d(expected_conv, kernels[1], axis=1, mode="mirror", output=expected_conv)
        convolve1d(expected_conv, kernels[2], axis=2, mode="mirror", output=expected_conv)
        with patch.object(base, "_SLAB_MIN_SIZE", 1):
            pairs = [
                (
                    base._gaussian_laplace(image, (1.5, 1.0, 2.0), "mirror", 4.0),
                    gaussian_laplace(image, sigma=(1.5, 1.0, 2.0), mode="mirror", truncate=4.0),
                ),
                (base._convolve_axes(image, kernels, "mirror"), expected_conv),
                (base._gaussian_filter(image, 0.0, "mirror"), image),
            ]
            in_place = image.copy()
            base._uniform_filter(in_place, 3, "reflect", output=in_place)
            window_sums = base._uniform_filter(image, 5, "nearest")  # one float32 unit
            line = image[:, 0, 0]  # one axis: no other axis to cut into slabs
            pairs.append((base._uniform_filter(line, 3, "reflect"), uniform_filter(line, 3)))
        for got, expected in pairs:
            np.testing.assert_array_equal(got, expected)
            assert got.dtype == expected.dtype
        assert_allclose(in_place, uniform_filter(image, size=3, mode="reflect"), rtol=1e-6)
        assert_allclose(window_sums, uniform_filter(image, size=5, mode="nearest"), rtol=1e-6)

    def test_laplace_terms_share_the_first_pass(self):
        # The terms after the first share the plain pass along axis 0 and one temporary:
        # scipy's values bit for bit, for 3 sizes, every boundary mode, float64 and float32
        # images, and sigmas with a zero on axis 0 or on another axis.
        import tracemalloc

        from scipy.ndimage import gaussian_laplace

        from pictologics.filters import base

        rng = np.random.default_rng(9)
        with patch.object(base, "_SLAB_MIN_SIZE", 1):
            for shape in ((12, 9, 10), (7, 16, 5), (20, 21, 22)):
                for dtype in (np.float64, np.float32):
                    image = rng.normal(0.0, 10.0, shape).astype(dtype)
                    for mode in ("reflect", "mirror", "nearest", "wrap", "constant"):
                        for sigma in (1.2, (1.5, 1.0, 2.0), (0.0, 1.0, 2.0), (1.0, 0.0, 1.5)):
                            got = base._gaussian_laplace(image, sigma, mode)
                            expected = gaussian_laplace(image, sigma=sigma, mode=mode)
                            assert got.dtype == expected.dtype
                            assert got.tobytes() == expected.tobytes()
        # The output, the shared pass and one temporary: three arrays of the image size.
        image = rng.normal(0.0, 10.0, (32, 32, 32))
        tracemalloc.start()
        base._gaussian_laplace(image, 1.2, "nearest")
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert 3 * image.nbytes <= peak < 3.1 * image.nbytes

    def test_a_large_float64_laplace_is_one_fft_convolution(self):
        # From _LOG_FFT_MIN voxels on, a float64 image takes one FFT convolution with the
        # kernels of the passes: within 1e-13 of scipy's largest value (measured: 2.3e-14 for
        # wrap with a kernel longer than an axis, else 2e-15), for 3 shapes, every boundary
        # mode, and scalar, per-axis and long sigmas. float32 images and a zero sigma keep
        # the passes: scipy's bits.
        from scipy.ndimage import gaussian_laplace

        from pictologics.filters import base

        rng = np.random.default_rng(10)
        with patch.object(base, "_LOG_FFT_MIN", 1):
            for shape in ((12, 9, 10), (7, 16, 5), (20, 21, 22)):
                image = rng.normal(0.0, 10.0, shape)
                for mode in ("reflect", "mirror", "nearest", "wrap", "constant"):
                    for sigma in (1.2, (1.5, 1.0, 2.0), 3.0):
                        got = base._gaussian_laplace(image, sigma, mode)
                        expected = gaussian_laplace(image, sigma=sigma, mode=mode)
                        assert got.dtype == np.float64
                        bound = 1e-13 * np.abs(expected).max()
                        np.testing.assert_allclose(got, expected, rtol=0.0, atol=bound)
            image = rng.normal(0.0, 10.0, (12, 9, 10))
            for passes, sigma in ((image.astype(np.float32), 1.2), (image, (0.0, 1.0, 1.5))):
                got = base._gaussian_laplace(passes, sigma, "nearest")
                expected = gaussian_laplace(passes, sigma=sigma, mode="nearest")
                assert got.tobytes() == expected.tobytes()

    def test_ordered_map_keeps_the_item_order(self):
        # Later items end first in the pool, but the results come in item order; one
        # worker maps in this thread.
        from pictologics.filters import base

        def slow_square(k: int) -> int:
            time.sleep(0.005 * (4 - k))
            return k * k

        assert list(base._ordered_map(slow_square, range(5), 3)) == [0, 1, 4, 9, 16]
        assert list(base._ordered_map(slow_square, [2, 3], 1)) == [4, 9]

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
    @pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
    def test_slab_pool_is_new_after_fork(self):
        # A forked process drops the slab pool of its parent, whose threads it does not
        # have, so its threaded passes make a new pool instead of waiting forever.
        from pictologics.filters import base

        image = np.random.default_rng(3).normal(size=(8, 9, 10)).astype(np.float32)
        with patch.object(base, "_SLAB_MIN_SIZE", 1):
            expected = base._uniform_filter(image, 3, "reflect")
            assert base._SLAB_POOL
            pid = os.fork()
            if pid == 0:  # pragma: no cover  (coverage does not follow the child)
                code = 1
                try:
                    if not base._SLAB_POOL:
                        same = np.array_equal(base._uniform_filter(image, 3, "reflect"), expected)
                        code = 0 if same else 2
                finally:
                    os._exit(code)
        deadline = time.monotonic() + 60
        while (done := os.waitpid(pid, os.WNOHANG))[0] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        if done[0] == 0:  # pragma: no cover  (only when the child waits forever)
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            pytest.fail("the forked process waited for the threads of its parent")
        assert os.waitstatus_to_exitcode(done[1]) == 0

    def test_normalized_convolve1d(self):
        # data window [10,0,30]·[0.5,1,0.5]=20; the valid weight [1,0,1]·[0.5,1,0.5]=1 is
        # half of the full weight 2, so the estimate of the plain response is 40.
        image = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        mask = np.array([True, False, True], dtype=bool)
        kernel = np.array([0.5, 1.0, 0.5], dtype=np.float32)

        result, _ = _normalized_convolve1d(image, mask, kernel, axis=0, mode="constant")
        assert np.isclose(result[1], 40.0)

    def test_normalized_separable_convolve_3d(self):
        shape = (5, 5, 5)
        image = np.random.rand(*shape).astype(np.float32)
        mask = np.ones(shape, dtype=bool)
        g = np.array([0.2, 0.6, 0.2], dtype=np.float32)

        result, valid_out = _normalized_separable_convolve_3d(image, mask, g, g, g, mode="constant")
        assert result.shape == shape
        assert valid_out.shape == shape


# =============================================================================
# Test mean.py
# =============================================================================


class TestMeanFilter:
    """Tests for mean_filter function."""

    def test_basic_application(self, small_3d_image):
        result = mean_filter(small_3d_image, support=3)
        assert result.shape == small_3d_image.shape
        assert result.dtype == np.float32

    def test_different_support_sizes(self, small_3d_image):
        for support in [1, 3, 5, 7]:
            result = mean_filter(small_3d_image, support=support)
            assert result.shape == small_3d_image.shape

    def test_all_boundary_conditions(self, small_3d_image):
        for boundary in BoundaryCondition:
            result = mean_filter(small_3d_image, support=3, boundary=boundary)
            assert result.shape == small_3d_image.shape

    def test_string_boundary_condition(self, small_3d_image):
        result = mean_filter(small_3d_image, support=3, boundary="zero")
        assert result.shape == small_3d_image.shape

    def test_invalid_support_even(self, small_3d_image):
        with pytest.raises(ValueError, match="odd positive integer"):
            mean_filter(small_3d_image, support=4)

    def test_invalid_support_zero(self, small_3d_image):
        with pytest.raises(ValueError, match="odd positive integer"):
            mean_filter(small_3d_image, support=0)

    def test_impulse_response(self, impulse_3d):
        """Mean filter on impulse should spread the value."""
        result = mean_filter(impulse_3d, support=3)
        # Center should be 1/27 with zero padding
        assert result[4, 4, 4] == pytest.approx(1.0 / 27, rel=1e-5)


# =============================================================================
# Test log.py
# =============================================================================


class TestLaplacianOfGaussian:
    """Tests for laplacian_of_gaussian function."""

    def test_basic_application(self, small_3d_image):
        result = laplacian_of_gaussian(small_3d_image, sigma_mm=2.0, spacing_mm=1.0)
        assert result.shape == small_3d_image.shape

    def test_with_tuple_spacing(self, small_3d_image):
        result = laplacian_of_gaussian(small_3d_image, sigma_mm=2.0, spacing_mm=(1.0, 1.0, 1.0))
        assert result.shape == small_3d_image.shape

    def test_different_truncation(self, small_3d_image):
        result = laplacian_of_gaussian(small_3d_image, sigma_mm=2.0, spacing_mm=1.0, truncate=3.0)
        assert result.shape == small_3d_image.shape

    def test_all_boundary_conditions(self, small_3d_image):
        for boundary in BoundaryCondition:
            result = laplacian_of_gaussian(
                small_3d_image, sigma_mm=2.0, spacing_mm=1.0, boundary=boundary
            )
            assert result.shape == small_3d_image.shape

    def test_string_boundary_condition(self, small_3d_image):
        result = laplacian_of_gaussian(
            small_3d_image, sigma_mm=2.0, spacing_mm=1.0, boundary="mirror"
        )
        assert result.shape == small_3d_image.shape

    def test_integer_spacing(self, small_3d_image):
        result = laplacian_of_gaussian(small_3d_image, sigma_mm=2.0, spacing_mm=2)
        assert result.shape == small_3d_image.shape


# =============================================================================
# Test laws.py
# =============================================================================


class TestLawsKernels:
    """Tests for LAWS_KERNELS dictionary."""

    def test_all_kernels_exist(self):
        expected_kernels = ["L3", "L5", "E3", "E5", "S3", "S5", "W5", "R5"]
        for name in expected_kernels:
            assert name in LAWS_KERNELS

    def test_kernel_lengths(self):
        assert len(LAWS_KERNELS["L3"]) == 3
        assert len(LAWS_KERNELS["L5"]) == 5


class TestLawsFilter:
    """Tests for laws_filter function."""

    def test_basic_application(self, small_3d_image):
        result = laws_filter(small_3d_image, "E5L5S5")
        assert result.shape == small_3d_image.shape

    def test_different_kernel_combos(self, small_3d_image):
        combos = ["L5E5S5", "E3W5R5", "L3L3L3"]
        for combo in combos:
            result = laws_filter(small_3d_image, combo)
            assert result.shape == small_3d_image.shape

    def test_rotation_invariant(self, small_3d_image):
        result = laws_filter(small_3d_image, "E5L5S5", rotation_invariant=True)
        assert result.shape == small_3d_image.shape

    def test_all_pooling_methods(self, small_3d_image):
        for pooling in ["max", "average", "min"]:
            result = laws_filter(small_3d_image, "E5L5S5", rotation_invariant=True, pooling=pooling)
            assert result.shape == small_3d_image.shape

    def test_compute_energy(self, small_3d_image):
        result = laws_filter(small_3d_image, "E5L5S5", compute_energy=True, energy_distance=3)
        assert result.shape == small_3d_image.shape
        assert np.all(result >= 0)  # Energy is non-negative

    def test_all_boundary_conditions(self, small_3d_image):
        for boundary in BoundaryCondition:
            result = laws_filter(small_3d_image, "L5L5L5", boundary=boundary)
            assert result.shape == small_3d_image.shape

    def test_string_boundary_condition(self, small_3d_image):
        result = laws_filter(small_3d_image, "L5L5L5", boundary="nearest")
        assert result.shape == small_3d_image.shape

    def test_invalid_kernel_string(self, small_3d_image):
        with pytest.raises(ValueError, match="Cannot parse"):
            laws_filter(small_3d_image, "INVALID")

    def test_invalid_kernel_count(self, small_3d_image):
        with pytest.raises(ValueError, match="Expected 3 kernel"):
            laws_filter(small_3d_image, "L5L5")  # Only 2 kernels

    def test_unknown_kernel_name(self, small_3d_image):
        # "X5Y5Z5" parses into 3 well-formed tokens but none are valid Laws kernels.
        with pytest.raises(ValueError, match="Unknown Laws kernel"):
            laws_filter(small_3d_image, "X5Y5Z5")

    def test_invalid_pooling(self, small_3d_image):
        with pytest.raises(ValueError, match="Unknown pooling"):
            laws_filter(small_3d_image, "L5L5L5", rotation_invariant=True, pooling="invalid")

    def test_parallel_execution(self, small_3d_image):
        """Exercise the use_parallel=True branch across pooling and energy modes."""
        for pooling in ["max", "average", "min"]:
            result = laws_filter(
                small_3d_image,
                "E3L3S3",
                rotation_invariant=True,
                pooling=pooling,
                use_parallel=True,
            )
            assert result.shape == small_3d_image.shape

        result = laws_filter(
            small_3d_image,
            "E5L5S5",
            rotation_invariant=True,
            compute_energy=True,
            energy_distance=1,
            use_parallel=True,
        )
        assert result.shape == small_3d_image.shape


# =============================================================================
# Test gabor.py
# =============================================================================


class TestGaborFilter:
    """Tests for gabor_filter function."""

    def test_basic_application(self, small_3d_image):
        result = gabor_filter(
            small_3d_image, sigma_mm=5.0, lambda_mm=2.0, gamma=0.5, spacing_mm=1.0
        )
        assert result.shape == small_3d_image.shape

    def test_rotation_invariant(self, small_3d_image):
        result = gabor_filter(
            small_3d_image,
            sigma_mm=5.0,
            lambda_mm=2.0,
            gamma=0.5,
            rotation_invariant=True,
            delta_theta=np.pi / 4,
        )
        assert result.shape == small_3d_image.shape

    def test_average_over_planes(self, small_3d_image):
        result = gabor_filter(
            small_3d_image,
            sigma_mm=5.0,
            lambda_mm=2.0,
            gamma=0.5,
            average_over_planes=True,
        )
        assert result.shape == small_3d_image.shape

    def test_the_plane_mean_adds_in_float64_in_plane_order(self, small_3d_image):
        # The plane responses add in float64 in plane order, and the mean is cast to
        # float32, on slabs in the slab threads: the values of the numpy steps, bit for bit.
        from pictologics.filters import base, gabor

        with patch.object(base, "_SLAB_MIN_SIZE", 1):
            result = gabor_filter(
                small_3d_image, sigma_mm=5.0, lambda_mm=2.0, gamma=0.5, average_over_planes=True
            )
        planes = [
            gabor._apply_gabor_to_plane(
                small_3d_image, 5.0, 2.0, 0.5, [0.0], axis, (1.0, 1.0, 1.0), "constant", "average"
            )
            for axis in range(3)
        ]
        expected = ((planes[0].astype(np.float64) + planes[1] + planes[2]) / 3.0).astype(np.float32)
        assert result.tobytes() == expected.tobytes()

    def test_all_pooling_methods(self, small_3d_image):
        """Test all pooling methods with multiple orientations to hit all branches."""
        for pooling in ["max", "average", "min"]:
            result = gabor_filter(
                small_3d_image,
                sigma_mm=5.0,
                lambda_mm=2.0,
                gamma=0.5,
                rotation_invariant=True,
                delta_theta=np.pi / 4,  # Ensure 8 orientations to hit pooling branches
                pooling=pooling,
            )
            assert result.shape == small_3d_image.shape

    def test_all_boundary_conditions(self, small_3d_image):
        for boundary in BoundaryCondition:
            result = gabor_filter(
                small_3d_image,
                sigma_mm=5.0,
                lambda_mm=2.0,
                gamma=0.5,
                boundary=boundary,
            )
            assert result.shape == small_3d_image.shape

    def test_string_boundary_condition(self, small_3d_image):
        result = gabor_filter(
            small_3d_image,
            sigma_mm=5.0,
            lambda_mm=2.0,
            gamma=0.5,
            boundary="periodic",
        )
        assert result.shape == small_3d_image.shape

    def test_tuple_spacing(self, small_3d_image):
        result = gabor_filter(
            small_3d_image,
            sigma_mm=5.0,
            lambda_mm=2.0,
            gamma=0.5,
            spacing_mm=(1.0, 1.0, 2.0),
        )
        assert result.shape == small_3d_image.shape

    def test_integer_spacing(self, small_3d_image):
        result = gabor_filter(small_3d_image, sigma_mm=5.0, lambda_mm=2.0, gamma=0.5, spacing_mm=2)
        assert result.shape == small_3d_image.shape

    def test_invalid_pooling(self, small_3d_image):
        """Test that invalid pooling parameter raises ValueError early."""
        with pytest.raises(ValueError, match="Unknown pooling"):
            gabor_filter(
                small_3d_image,
                sigma_mm=5.0,
                lambda_mm=2.0,
                gamma=0.5,
                pooling="invalid",
            )

    def test_rounded_delta_theta_gives_the_whole_number_of_angles(self, small_3d_image):
        # 0.785398 (pi/4 in 6 digits) means 8 orientations, as pi/4 does; ceil gave 9. A
        # step that does not divide 2 pi keeps ceil: 1.0 gives 7 orientations.
        from pictologics.filters import gabor as gabor_module

        def thetas(step: float) -> list[float]:
            with patch.object(
                gabor_module, "_apply_gabor_to_plane", wraps=gabor_module._apply_gabor_to_plane
            ) as plane:
                gabor_filter(
                    small_3d_image, sigma_mm=2.0, lambda_mm=2.0, rotation_invariant=True,
                    delta_theta=step,
                )  # fmt: skip
            return list(plane.call_args.args[4])

        assert thetas(0.785398) == thetas(np.pi / 4) == [k * np.pi / 4 for k in range(4)]
        assert thetas(1.0) == [float(k) for k in range(7)]

    def test_rotation_invariant_requires_delta_theta(self, small_3d_image):
        with pytest.raises(ValueError, match="requires delta_theta"):
            gabor_filter(
                small_3d_image, sigma_mm=5.0, lambda_mm=2.0, gamma=0.5, rotation_invariant=True
            )

    def test_parallel_execution(self, small_3d_image):
        """Exercise the use_parallel=True branch, including average_over_planes."""
        result = gabor_filter(
            small_3d_image,
            sigma_mm=5.0,
            lambda_mm=2.0,
            gamma=0.5,
            rotation_invariant=True,
            delta_theta=np.pi / 4,
            use_parallel=True,
        )
        assert result.shape == small_3d_image.shape

        result = gabor_filter(
            small_3d_image,
            sigma_mm=5.0,
            lambda_mm=2.0,
            gamma=0.5,
            rotation_invariant=True,
            delta_theta=np.pi / 4,
            average_over_planes=True,
            use_parallel=True,
        )
        assert result.shape == small_3d_image.shape

    def test_no_warning_for_default_axial_only_with_anisotropic_z(self, small_3d_image, recwarn):
        """average_over_planes=False (default) only ever uses plane_axis=2, whose
        in-plane axes are 0 and 1. Anisotropy in z (axis 2) is irrelevant to that
        plane, so this must not emit the old (over-eager) anisotropy warning, and
        the result must be identical to genuinely isotropic (1, 1, 1) spacing."""
        result_aniso_z = gabor_filter(
            small_3d_image, sigma_mm=2.0, lambda_mm=1.0, gamma=0.5, spacing_mm=(1.0, 1.0, 3.0)
        )
        assert len(recwarn) == 0

        result_iso = gabor_filter(
            small_3d_image, sigma_mm=2.0, lambda_mm=1.0, gamma=0.5, spacing_mm=(1.0, 1.0, 1.0)
        )
        assert_array_equal(result_aniso_z, result_iso)

    def test_anisotropic_kernel_matches_closed_form_physical_grid(self):
        """`_create_gabor_kernel_2d_anisotropic` must build a rectangular kernel
        with an independent 6*sigma_mm/(gamma*s_i) radius per axis (gamma < 1 makes
        the envelope longer), evaluated on a physical (mm) coordinate grid.
        Recompute the expected kernel from the Gabor formula directly (not by
        calling the function under test) to catch implementation bugs rather than
        just echoing them back."""
        sigma_mm, lambda_mm, gamma, theta = 5.0, 2.0, 0.5, np.pi / 6
        s1, s2 = 1.0, 3.0

        kernel = _create_gabor_kernel_2d_anisotropic(sigma_mm, lambda_mm, gamma, theta, s1, s2)

        radius1 = int(np.ceil(6.0 * sigma_mm / gamma / s1))
        radius2 = int(np.ceil(6.0 * sigma_mm / gamma / s2))
        assert kernel.shape == (2 * radius1 + 1, 2 * radius2 + 1)
        # Per-axis radii differ because s1 != s2, so the kernel is rectangular,
        # unlike the old single-scalar approach, which would reuse s1 for both
        # axes and produce a square kernel (via _create_gabor_kernel_2d, the
        # isotropic-path builder, called with sigma_mm/s1 for both dimensions).
        assert radius1 != radius2
        old_wrong_square_kernel = _create_gabor_kernel_2d(
            sigma_mm / s1, lambda_mm / s1, gamma, theta
        )
        assert old_wrong_square_kernel.shape != kernel.shape
        assert old_wrong_square_kernel.shape[0] == old_wrong_square_kernel.shape[1]

        k1, k2 = np.mgrid[-radius1 : radius1 + 1, -radius2 : radius2 + 1].astype(np.float64)
        p1, p2 = k1 * s1, k2 * s2
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        p1_rot = p1 * cos_t + p2 * sin_t
        p2_rot = -p1 * sin_t + p2 * cos_t
        expected = np.exp(-(p1_rot**2 + gamma**2 * p2_rot**2) / (2 * sigma_mm**2)) * np.exp(
            1j * 2 * np.pi * p1_rot / lambda_mm
        )
        np.testing.assert_allclose(kernel, expected.astype(np.complex64), rtol=1e-6)

    def test_kernel_radius_covers_the_long_envelope_axis(self):
        # The envelope has the scale sigma / gamma for gamma < 1, so the radius is
        # ceil(6 sigma / gamma); gamma >= 1 keeps ceil(6 sigma).
        assert _create_gabor_kernel_2d(5.0, 4.0, 0.5, 0.0).shape == (121, 121)
        assert _create_gabor_kernel_2d(5.0, 4.0, 2.5, 0.0).shape == (61, 61)

    def test_anisotropic_in_plane_scaling_replaces_old_single_scalar_behaviour(
        self, small_3d_image
    ):
        """Before the fix, `sigma_voxels`/`lambda_voxels` were computed once from
        `spacing_mm[0]` and reused for every plane, so a plane's response never
        depended on `spacing_mm[1]`/`spacing_mm[2]`. That is reproducible today by
        calling `_apply_gabor_to_plane` for `plane_axis=0` with `spacing_mm=(1, 1,
        1)`: axis 0's in-plane spacing is still 1.0 either way, so this call is
        exactly what the old code computed for `plane_axis=0` when the real
        spacing was `(1, 1, 3)` (it only ever looked at index 0). The new,
        physically-correct call instead passes the true spacing `(1, 1, 3)`, so
        axis 0's plane now sees its real in-plane spacings (1.0, 3.0). These two
        must now differ -- the fix's whole point is that plane 0 (and 1) can no
        longer ignore z-anisotropy."""
        sigma_mm, lambda_mm, gamma, theta = 2.0, 1.0, 0.5, 0.0
        mode = get_scipy_mode(BoundaryCondition.ZERO)

        old_wrong_response = _apply_gabor_to_plane(
            small_3d_image,
            sigma_mm,
            lambda_mm,
            gamma,
            [theta],
            plane_axis=0,
            spacing_mm=(1.0, 1.0, 1.0),
            mode=mode,
            pooling="average",
            use_parallel=False,
        )
        new_correct_response = _apply_gabor_to_plane(
            small_3d_image,
            sigma_mm,
            lambda_mm,
            gamma,
            [theta],
            plane_axis=0,
            spacing_mm=(1.0, 1.0, 3.0),
            mode=mode,
            pooling="average",
            use_parallel=False,
        )
        assert not np.array_equal(old_wrong_response, new_correct_response)

    def test_average_over_planes_anisotropic_spacing_is_now_handled_correctly(self, small_3d_image):
        """End-to-end: with average_over_planes=True and anisotropic z spacing,
        the overall result must differ from what pure isotropic (1, 1, 1) spacing
        would produce, since planes 0 and 1 (which contain the z axis) now
        correctly incorporate the true z spacing rather than silently treating
        the volume as isotropic."""
        result_aniso = gabor_filter(
            small_3d_image,
            sigma_mm=2.0,
            lambda_mm=1.0,
            gamma=0.5,
            spacing_mm=(1.0, 1.0, 3.0),
            average_over_planes=True,
        )
        result_iso = gabor_filter(
            small_3d_image,
            sigma_mm=2.0,
            lambda_mm=1.0,
            gamma=0.5,
            spacing_mm=(1.0, 1.0, 1.0),
            average_over_planes=True,
        )
        assert not np.array_equal(result_aniso, result_iso)


# =============================================================================
# Test wavelets.py
# =============================================================================


class TestWaveletTransform:
    """Tests for wavelet_transform function."""

    def test_basic_application(self, small_3d_image):
        result = wavelet_transform(small_3d_image, wavelet="db2", level=1)
        assert result.shape == small_3d_image.shape

    def test_different_wavelets(self, small_3d_image):
        wavelets = ["haar", "db2", "db3", "coif1", "coif2"]
        for wavelet in wavelets:
            result = wavelet_transform(small_3d_image, wavelet=wavelet, level=1)
            assert result.shape == small_3d_image.shape

    def test_different_levels(self, small_3d_image):
        for level in [1, 2, 3]:
            result = wavelet_transform(small_3d_image, wavelet="db2", level=level)
            assert result.shape == small_3d_image.shape

    def test_different_decompositions(self, small_3d_image):
        decomps = ["LLL", "HHL", "LHH"]
        for decomp in decomps:
            result = wavelet_transform(small_3d_image, wavelet="db2", level=1, decomposition=decomp)
            assert result.shape == small_3d_image.shape

    def test_rotation_invariant(self, small_3d_image):
        result = wavelet_transform(small_3d_image, wavelet="db2", level=1, rotation_invariant=True)
        assert result.shape == small_3d_image.shape

    def test_all_pooling_methods(self, small_3d_image):
        for pooling in ["max", "average", "min"]:
            result = wavelet_transform(
                small_3d_image,
                wavelet="db2",
                level=1,
                rotation_invariant=True,
                pooling=pooling,
            )
            assert result.shape == small_3d_image.shape

    def test_all_boundary_conditions(self, small_3d_image):
        for boundary in BoundaryCondition:
            result = wavelet_transform(small_3d_image, wavelet="db2", level=1, boundary=boundary)
            assert result.shape == small_3d_image.shape

    def test_string_boundary_condition(self, small_3d_image):
        result = wavelet_transform(small_3d_image, wavelet="db2", level=1, boundary="mirror")
        assert result.shape == small_3d_image.shape

    def test_invalid_pooling(self, small_3d_image):
        with pytest.raises(ValueError, match="Unknown pooling"):
            wavelet_transform(
                small_3d_image,
                wavelet="db2",
                level=1,
                rotation_invariant=True,
                pooling="invalid",
            )

    def test_higher_level_recursion(self, small_3d_image):
        """Test recursive wavelet application at higher levels."""
        result = wavelet_transform(small_3d_image, wavelet="haar", level=3)
        assert result.shape == small_3d_image.shape

    def test_parallel_execution(self, small_3d_image):
        """The use_parallel=True branch pools in rotation order: the sequential result,
        bit for bit, for every pooling mode."""
        for pooling in ["max", "average", "min"]:
            kwargs = dict(
                wavelet="db2",
                level=1,
                decomposition="LHL",
                rotation_invariant=True,
                pooling=pooling,
            )
            result = wavelet_transform(small_3d_image, use_parallel=True, **kwargs)
            assert result.shape == small_3d_image.shape
            sequential = wavelet_transform(small_3d_image, use_parallel=False, **kwargs)
            assert_array_equal(result, sequential)


class TestSimoncelliWavelet:
    """Tests for simoncelli_wavelet function."""

    def test_basic_application(self, small_3d_image):
        result = simoncelli_wavelet(small_3d_image, level=1)
        assert result.shape == small_3d_image.shape

    def test_different_levels(self, small_3d_image):
        for level in [1, 2, 3]:
            result = simoncelli_wavelet(small_3d_image, level=level)
            assert result.shape == small_3d_image.shape

    def test_default_boundary_is_periodic(self, small_3d_image):
        # No boundary argument must be identical to explicit PERIODIC.
        default = simoncelli_wavelet(small_3d_image, level=1)
        explicit = simoncelli_wavelet(small_3d_image, level=1, boundary=BoundaryCondition.PERIODIC)
        assert_array_equal(default, explicit)

    def test_all_boundary_conditions(self, small_3d_image):
        for boundary in BoundaryCondition:
            result = simoncelli_wavelet(small_3d_image, level=1, boundary=boundary)
            assert result.shape == small_3d_image.shape

    def test_string_boundary_condition(self, small_3d_image):
        result = simoncelli_wavelet(small_3d_image, level=1, boundary="nearest")
        assert result.shape == small_3d_image.shape

    def test_invalid_boundary_raises(self, small_3d_image):
        with pytest.raises(ValueError, match="Unknown boundary condition"):
            simoncelli_wavelet(small_3d_image, level=1, boundary="bogus")

    def test_boundary_changes_response(self, small_3d_image):
        # A non-periodic boundary must actually change the (edge-sensitive)
        # response, proving the requested boundary is honoured rather than
        # silently discarded.
        periodic = simoncelli_wavelet(small_3d_image, level=1, boundary="periodic")
        nearest = simoncelli_wavelet(small_3d_image, level=1, boundary="nearest")
        assert not np.array_equal(periodic, nearest)
        assert nearest.shape == small_3d_image.shape


# =============================================================================
# Test riesz.py
# =============================================================================


class TestRieszTransform:
    """Tests for riesz_transform function."""

    def test_basic_application(self, small_3d_image):
        # First order Riesz transform (1, 0, 0)
        result = riesz_transform(small_3d_image, order=(1, 0, 0))
        assert result.shape == small_3d_image.shape

    def test_different_orders(self, small_3d_image):
        orders = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (2, 0, 0)]
        for order in orders:
            result = riesz_transform(small_3d_image, order=order)
            assert result.shape == small_3d_image.shape

    def test_zero_order_raises(self, small_3d_image):
        with pytest.raises(ValueError, match="At least one order"):
            riesz_transform(small_3d_image, order=(0, 0, 0))

    def test_list_order_matches_tuple(self, small_3d_image):
        # A list-typed order (e.g. from a YAML/JSON pipeline config) must work and be
        # byte-identical to the tuple form. Regression guard: the cached transfer keys on
        # `order`, which must be coerced to a hashable tuple.
        for order in [(1, 0, 0), (1, 1, 0), (2, 0, 0)]:
            expected = riesz_transform(small_3d_image, order=order)
            got = riesz_transform(small_3d_image, order=list(order))
            assert_array_equal(got, expected)

    def test_default_boundary_is_periodic(self, small_3d_image):
        default = riesz_transform(small_3d_image, order=(1, 0, 0))
        explicit = riesz_transform(
            small_3d_image, order=(1, 0, 0), boundary=BoundaryCondition.PERIODIC
        )
        assert_array_equal(default, explicit)

    def test_all_boundary_conditions(self, small_3d_image):
        for boundary in BoundaryCondition:
            result = riesz_transform(small_3d_image, order=(1, 0, 0), boundary=boundary)
            assert result.shape == small_3d_image.shape

    def test_string_boundary_condition(self, small_3d_image):
        result = riesz_transform(small_3d_image, order=(1, 0, 0), boundary="mirror")
        assert result.shape == small_3d_image.shape

    def test_invalid_boundary_raises(self, small_3d_image):
        with pytest.raises(ValueError, match="Unknown boundary condition"):
            riesz_transform(small_3d_image, order=(1, 0, 0), boundary="bogus")

    def test_boundary_changes_response(self, small_3d_image):
        periodic = riesz_transform(small_3d_image, order=(1, 0, 0), boundary="periodic")
        nearest = riesz_transform(small_3d_image, order=(1, 0, 0), boundary="nearest")
        assert not np.array_equal(periodic, nearest)


class TestRieszLog:
    """Tests for riesz_log function."""

    def test_basic_application(self, small_3d_image):
        result = riesz_log(small_3d_image, sigma_mm=2.0, order=(1, 0, 0))
        assert result.shape == small_3d_image.shape

    def test_with_spacing(self, small_3d_image):
        result = riesz_log(small_3d_image, sigma_mm=2.0, order=(1, 0, 0), spacing_mm=2.0)
        assert result.shape == small_3d_image.shape

    def test_different_orders(self, small_3d_image):
        orders = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
        for order in orders:
            result = riesz_log(small_3d_image, sigma_mm=2.0, order=order)
            assert result.shape == small_3d_image.shape

    def test_default_boundary_is_periodic(self, small_3d_image):
        default = riesz_log(small_3d_image, sigma_mm=2.0, order=(1, 0, 0))
        explicit = riesz_log(
            small_3d_image, sigma_mm=2.0, order=(1, 0, 0), boundary=BoundaryCondition.PERIODIC
        )
        assert_array_equal(default, explicit)

    def test_all_boundary_conditions(self, small_3d_image):
        for boundary in BoundaryCondition:
            result = riesz_log(small_3d_image, sigma_mm=2.0, order=(1, 0, 0), boundary=boundary)
            assert result.shape == small_3d_image.shape

    def test_string_boundary_condition(self, small_3d_image):
        result = riesz_log(small_3d_image, sigma_mm=2.0, order=(1, 0, 0), boundary="zero")
        assert result.shape == small_3d_image.shape

    def test_invalid_boundary_raises(self, small_3d_image):
        with pytest.raises(ValueError, match="Unknown boundary condition"):
            riesz_log(small_3d_image, sigma_mm=2.0, order=(1, 0, 0), boundary="bogus")

    def test_boundary_changes_response(self, small_3d_image):
        periodic = riesz_log(small_3d_image, sigma_mm=2.0, order=(1, 0, 0), boundary="periodic")
        nearest = riesz_log(small_3d_image, sigma_mm=2.0, order=(1, 0, 0), boundary="nearest")
        assert not np.array_equal(periodic, nearest)

    def test_tuple_spacing_with_boundary(self, small_3d_image):
        # Anisotropic (tuple) spacing exercises the per-axis pad-width branch of
        # _riesz_log_pad_width.
        result = riesz_log(
            small_3d_image,
            sigma_mm=2.0,
            order=(1, 0, 0),
            spacing_mm=(1.0, 1.5, 2.0),
            boundary="nearest",
        )
        assert result.shape == small_3d_image.shape


class TestRieszSimoncelli:
    """Tests for riesz_simoncelli function."""

    def test_basic_application(self, small_3d_image):
        result = riesz_simoncelli(small_3d_image, level=1, order=(1, 0, 0))
        assert result.shape == small_3d_image.shape

    def test_different_levels(self, small_3d_image):
        for level in [1, 2, 3]:
            result = riesz_simoncelli(small_3d_image, level=level, order=(1, 0, 0))
            assert result.shape == small_3d_image.shape

    def test_different_orders(self, small_3d_image):
        orders = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for order in orders:
            result = riesz_simoncelli(small_3d_image, level=1, order=order)
            assert result.shape == small_3d_image.shape

    def test_zero_order_raises(self, small_3d_image):
        """riesz_simoncelli should raise ValueError when all order components are 0."""
        with pytest.raises(ValueError, match="At least one order component must be > 0"):
            riesz_simoncelli(small_3d_image, level=1, order=(0, 0, 0))

    def test_default_boundary_is_periodic(self, small_3d_image):
        default = riesz_simoncelli(small_3d_image, level=1, order=(1, 0, 0))
        explicit = riesz_simoncelli(
            small_3d_image, level=1, order=(1, 0, 0), boundary=BoundaryCondition.PERIODIC
        )
        assert_array_equal(default, explicit)

    def test_all_boundary_conditions(self, small_3d_image):
        for boundary in BoundaryCondition:
            result = riesz_simoncelli(small_3d_image, level=1, order=(1, 0, 0), boundary=boundary)
            assert result.shape == small_3d_image.shape

    def test_string_boundary_condition(self, small_3d_image):
        result = riesz_simoncelli(small_3d_image, level=1, order=(1, 0, 0), boundary="nearest")
        assert result.shape == small_3d_image.shape

    def test_invalid_boundary_raises(self, small_3d_image):
        with pytest.raises(ValueError, match="Unknown boundary condition"):
            riesz_simoncelli(small_3d_image, level=1, order=(1, 0, 0), boundary="bogus")

    def test_boundary_changes_response(self, small_3d_image):
        periodic = riesz_simoncelli(small_3d_image, level=1, order=(0, 2, 0), boundary="periodic")
        nearest = riesz_simoncelli(small_3d_image, level=1, order=(0, 2, 0), boundary="nearest")
        assert not np.array_equal(periodic, nearest)


class TestGetRieszOrders:
    """Tests for get_riesz_orders function."""

    def test_first_order_3d(self):
        from pictologics.filters.riesz import get_riesz_orders

        orders = get_riesz_orders(1, ndim=3)
        assert (1, 0, 0) in orders
        assert (0, 1, 0) in orders
        assert (0, 0, 1) in orders
        assert len(orders) == 3

    def test_second_order_3d(self):
        from pictologics.filters.riesz import get_riesz_orders

        orders = get_riesz_orders(2, ndim=3)
        assert (2, 0, 0) in orders
        assert (1, 1, 0) in orders
        assert (1, 0, 1) in orders
        assert (0, 2, 0) in orders
        assert (0, 1, 1) in orders
        assert (0, 0, 2) in orders
        assert len(orders) == 6


# =============================================================================
# Test __init__.py imports
# =============================================================================


@pytest.fixture
def source_mask_3d():
    """Boolean source mask (True = valid voxel) matching small_3d_image."""
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    return mask


class TestFilterSourceMask:
    """Filters given a source_mask exclude invalid voxels via normalized convolution.

    mean/log/laws (non-rotational) return a (response, output_valid_mask) tuple;
    gabor/wavelet/simoncelli/riesz zero-fill invalid voxels and return a single map.
    """

    def test_mean(self, small_3d_image, source_mask_3d):
        res, valid = mean_filter(small_3d_image, source_mask=source_mask_3d)
        assert res.shape == small_3d_image.shape
        assert valid.shape == small_3d_image.shape

    def test_log(self, small_3d_image, source_mask_3d):
        res, valid = laplacian_of_gaussian(small_3d_image, sigma_mm=1.0, source_mask=source_mask_3d)
        assert res.shape == small_3d_image.shape

    def test_laws_non_rotational(self, small_3d_image, source_mask_3d):
        res, valid = laws_filter(
            small_3d_image, "L5E5S5", source_mask=source_mask_3d, rotation_invariant=False
        )
        assert res.shape == small_3d_image.shape
        assert valid.shape == small_3d_image.shape

    def test_laws_source_mask_keeps_the_plain_response(self):
        # Where every voxel under the kernel is valid (2 voxels from the image edge and
        # from the invalid slice), the source-mask path gives the plain Laws response
        # (before, 0.15 to 0.23 times it).
        rng = np.random.default_rng(3)
        image = rng.normal(100.0, 20.0, (12, 12, 12)).astype(np.float32)
        plain = laws_filter(image, "L5E5S5")
        inner = (slice(3, -2), slice(2, -2), slice(2, -2))
        valid = np.ones(image.shape, dtype=bool)
        for invalid in (None, 0):
            if invalid is not None:
                valid[invalid] = False
            res, _ = laws_filter(image, "L5E5S5", source_mask=valid)
            np.testing.assert_allclose(res[inner], plain[inner], rtol=1e-5, atol=1e-3)

    def test_gabor(self, small_3d_image, source_mask_3d):
        res = gabor_filter(small_3d_image, sigma_mm=1.0, lambda_mm=2.0, source_mask=source_mask_3d)
        assert res.shape == small_3d_image.shape
        assert not np.isnan(res).any()

    def test_wavelet(self, small_3d_image, source_mask_3d):
        res = wavelet_transform(
            small_3d_image,
            wavelet="haar",
            level=1,
            decomposition="LLL",
            source_mask=source_mask_3d,
        )
        assert res.shape == small_3d_image.shape

    def test_simoncelli(self, small_3d_image, source_mask_3d):
        res = simoncelli_wavelet(small_3d_image, level=1, source_mask=source_mask_3d)
        assert res.shape == small_3d_image.shape

    def test_riesz(self, small_3d_image, source_mask_3d):
        res = riesz_log(small_3d_image, sigma_mm=1.0, source_mask=source_mask_3d)
        assert res.shape == small_3d_image.shape
        res2 = riesz_simoncelli(small_3d_image, source_mask=source_mask_3d)
        assert res2.shape == small_3d_image.shape

    def test_simoncelli_with_non_periodic_boundary(self, small_3d_image, source_mask_3d):
        # source_mask has the *original* (unpadded) shape; combined with a
        # non-periodic boundary this exercises the pad-filter-crop path and must
        # not raise a shape-mismatch error.
        res = simoncelli_wavelet(
            small_3d_image, level=1, boundary="nearest", source_mask=source_mask_3d
        )
        assert res.shape == small_3d_image.shape

    def test_riesz_transform_with_non_periodic_boundary(self, small_3d_image, source_mask_3d):
        res = riesz_transform(
            small_3d_image, order=(1, 0, 0), boundary="nearest", source_mask=source_mask_3d
        )
        assert res.shape == small_3d_image.shape

    def test_riesz_log_with_non_periodic_boundary(self, small_3d_image, source_mask_3d):
        # riesz_log re-masks the final, cropped response once (see its
        # docstring), so masked-out voxels are exactly zero even though padding
        # made the mask's shape mismatch the intermediate arrays mid-chain.
        res = riesz_log(
            small_3d_image, sigma_mm=1.0, boundary="nearest", source_mask=source_mask_3d
        )
        assert res.shape == small_3d_image.shape
        assert np.all(res[~source_mask_3d] == 0.0)

    def test_riesz_simoncelli_with_non_periodic_boundary(self, small_3d_image, source_mask_3d):
        res = riesz_simoncelli(small_3d_image, boundary="nearest", source_mask=source_mask_3d)
        assert res.shape == small_3d_image.shape
        assert np.all(res[~source_mask_3d] == 0.0)


class TestModuleImports:
    """Tests for module-level imports in __init__.py."""

    def test_all_exports_available(self):
        from pictologics import filters

        assert hasattr(filters, "mean_filter")
        assert hasattr(filters, "laplacian_of_gaussian")
        assert hasattr(filters, "laws_filter")
        assert hasattr(filters, "gabor_filter")
        assert hasattr(filters, "wavelet_transform")
        assert hasattr(filters, "simoncelli_wavelet")
        assert hasattr(filters, "riesz_transform")
        assert hasattr(filters, "riesz_log")
        assert hasattr(filters, "riesz_simoncelli")
        assert hasattr(filters, "BoundaryCondition")
        assert hasattr(filters, "FilterResult")
        assert hasattr(filters, "LAWS_KERNELS")


def test_cache_by_bytes_keeps_the_newest_results() -> None:
    """The cache drops the least recently used results above its byte limit, and it
    always keeps the newest one."""
    from pictologics.filters.base import cache_by_bytes

    calls: list[int] = []

    @cache_by_bytes(250)
    def make(n: int) -> np.ndarray:
        calls.append(n)
        return np.zeros(n, dtype=np.uint8)

    make(100)
    make(100)  # a hit
    make(120)
    make(100)  # a hit: now the most recently used
    make(60)  # 280 bytes: drops 120
    make(100)
    make(120)
    assert calls == [100, 120, 60, 120]
    make(300)  # alone above the limit: it stays, the rest goes
    make(300)
    assert calls == [100, 120, 60, 120, 300]


def test_transfer_caches_follow_the_memory_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each transfer-function cache keeps at most 2 GiB and at most a quarter of the
    memory budget, read at each new result (a run_batch worker sets its share after the
    import)."""
    from pictologics.filters import base

    monkeypatch.setattr(base, "_memory_budget", lambda: 64 << 30)
    assert base._transfer_cache_bytes() == 2 << 30
    monkeypatch.setattr(base, "_memory_budget", lambda: 2 << 30)
    assert base._transfer_cache_bytes() == 512 << 20

    calls: list[int] = []
    limit = [250]

    @base.cache_by_bytes(lambda: limit[0])
    def make(n: int) -> np.ndarray:
        calls.append(n)
        return np.zeros(n, dtype=np.uint8)

    make(100)
    make(120)
    limit[0] = 150
    make(10)  # 230 bytes, above the new limit: drops 100
    make(120)  # a hit
    assert calls == [100, 120, 10]
    limit[0] = 250
    make(100)  # 230 bytes: all stay
    make(10)  # a hit
    assert calls == [100, 120, 10, 100]


def test_wavelet_passes_in_place_give_the_new_array_result() -> None:
    """Every pass after the first writes into one array; the result equals passes that
    each make a new array, bit for bit, and the input stays unchanged."""
    import pywt
    from scipy.ndimage import convolve1d

    from pictologics.filters.wavelets import _apply_undecimated_wavelet_3d, _atrous_upsample

    image = np.random.default_rng(3).normal(size=(9, 8, 7))
    before = image.copy()
    for wavelet, level, decomposition in (
        ("db2", 1, "LHL"),
        ("coif1", 2, "HHH"),
        ("haar", 3, "LLH"),
    ):
        w = pywt.Wavelet(wavelet)
        lo, hi = np.array(w.dec_lo, dtype=np.float32), np.array(w.dec_hi, dtype=np.float32)
        expected = image
        for j in range(1, level + 1):
            lo_j = _atrous_upsample(lo, j) if j > 1 else lo
            hi_j = _atrous_upsample(hi, j) if j > 1 else hi
            chars = "LLL" if j < level else decomposition
            for axis, char in enumerate(chars):
                expected = convolve1d(
                    expected, {"L": lo_j, "H": hi_j}[char], axis=axis, mode="reflect"
                )
        result = _apply_undecimated_wavelet_3d(image, lo, hi, level, decomposition, "reflect")
        assert_array_equal(result, expected)
    assert_array_equal(image, before)


def _simoncelli_band(shape: tuple[int, ...], level: int) -> np.ndarray:
    """The Simoncelli band on the whole fftn grid, in one volume."""
    max_freq = 1.0 / (2 ** (level - 1))
    center = (np.array(shape) - 1.0) / 2.0
    grids = [np.fft.ifftshift((np.arange(s) - center[i]) / center[i]) for i, s in enumerate(shape)]
    dist = np.sqrt(np.asarray(sum(g**2 for g in np.meshgrid(*grids, indexing="ij", sparse=True))))
    val = 2.0 * dist / max_freq
    with np.errstate(all="ignore"):
        g_sim = np.cos(np.pi / 2.0 * np.log2(np.where(val > 0, val, 1.0)))
    return np.where((dist >= max_freq / 4.0) & (dist <= max_freq), g_sim, 0.0)


def _simoncelli_whole(shape: tuple[int, ...], level: int) -> np.ndarray:
    """The whole Simoncelli table (rfftn layout), in one volume: the even part of the
    band, (g(k) + g(-k)) / 2."""
    band = _simoncelli_band(shape, level)
    negative = band[np.ix_(*[(-np.arange(s)) % s for s in shape])]  # g(-k)
    return (0.5 * (band + negative))[..., : shape[-1] // 2 + 1]


def _riesz_whole(shape: tuple[int, ...], order: tuple[int, ...]) -> np.ndarray:
    """The whole Riesz table (rfftn layout), in one volume."""
    from math import factorial, sqrt

    freqs = [np.fft.fftfreq(s) * 2 * np.pi for s in shape[:-1]] + [
        np.fft.rfftfreq(shape[-1]) * 2 * np.pi
    ]
    nu = np.meshgrid(*freqs, indexing="ij", sparse=True)
    nu_norm = np.sqrt(np.asarray(sum(n**2 for n in nu), dtype=np.float64))
    numerator = np.ones(nu_norm.shape)
    for i, o in enumerate(order):
        if o > 0:
            numerator *= nu[i] ** o
    L = sum(order)
    norm = sqrt(factorial(L) / np.prod([factorial(o) for o in order]))
    t = np.exp(-1j * np.pi * L / 2) * norm * numerator / (np.where(nu_norm > 0, nu_norm, 1.0) ** L)
    return np.where(nu_norm > 0, t, 0)


def test_fft_filters_multiply_in_place_only_when_the_type_holds() -> None:
    """The mirrored product writes into a complex128 spectrum; a complex64 one gets a new
    complex128 array, as the product with a float64 table always was. A row past the
    kept rows reads its mirrored row, with its sign."""
    from pictologics.filters.base import _times_mirrored

    half = np.array([[0.5, 1.0], [2.0, 3.0]])  # rows 0 and 1 of a table of 3 rows
    full = np.array([[0.5, 1.0], [2.0, 3.0], [-2.0, -3.0]])  # row 2 = -row (3 - 2)
    spectrum = np.arange(6).reshape(3, 2) * (1 + 1j)
    expected = spectrum * full
    assert _times_mirrored(spectrum, half, 3, -1) is spectrum
    assert_array_equal(spectrum, expected)
    single = (np.arange(6).reshape(3, 2) * (1 - 1j)).astype(np.complex64)
    product = _times_mirrored(single, half, 3, -1)
    assert product.dtype == np.complex128 and product is not single
    assert_array_equal(product, single.astype(np.complex128) * full)


def test_fft_filters_match_the_out_of_place_product() -> None:
    """Simoncelli and Riesz give the values of the out-of-place product with the whole
    table, bit for bit, for even and odd sizes and float64 and float32 images, also with
    the product in parts on threads."""
    import scipy.fft

    from pictologics.filters import base
    from pictologics.filters.riesz import _riesz_transfer
    from pictologics.filters.wavelets import _simoncelli_transfer

    rng = np.random.default_rng(4)
    for shape in ((12, 11, 10), (9, 8, 7)):
        for dtype in (np.float64, np.float32):
            image = rng.normal(size=shape).astype(dtype)
            spectrum = scipy.fft.rfftn(image, workers=-1) * _simoncelli_whole(shape, 2)
            simoncelli = scipy.fft.irfftn(spectrum, s=shape, workers=-1).astype(np.float32)
            spectrum = scipy.fft.rfftn(image, workers=-1) * _riesz_whole(shape, (1, 1, 0))
            riesz = scipy.fft.irfftn(spectrum, s=shape, workers=-1).astype(np.float32)
            settings = (
                (base._SLAB_MIN_SIZE, base._PRODUCT_PART, base._HALF_TABLE_MIN),
                (1, 1, 0),  # half tables, with the product in parts on threads
                (base._SLAB_MIN_SIZE, base._PRODUCT_PART, 0),  # half tables in one thread
            )
            for slab_min, part, half_min in settings:
                with (
                    patch.object(base, "_SLAB_MIN_SIZE", slab_min),
                    patch.object(base, "_PRODUCT_PART", part),
                    patch.object(base, "_HALF_TABLE_MIN", half_min),
                ):
                    _simoncelli_transfer.cache_clear()  # the tables of this setting
                    _riesz_transfer.cache_clear()
                    assert_array_equal(simoncelli_wavelet(image, level=2), simoncelli)
                    assert_array_equal(riesz_transform(image, order=(1, 1, 0)), riesz)


def test_simoncelli_filters_in_one_real_round_trip() -> None:
    """The even part of the band gives the real part of the complex response, so the
    Simoncelli filter runs in one rfft round trip. Riesz-Simoncelli applies both tables in
    the same round trip: the values of the two filters one after the other, without the
    float32 rounding between them, so closer to a float64 reference."""
    import scipy.fft

    rng = np.random.default_rng(5)
    for shape in ((12, 11, 10), (9, 8, 7)):
        image = rng.normal(size=shape).astype(np.float32)
        for level in (1, 2):
            spectrum = scipy.fft.fftn(image.astype(np.float64)) * _simoncelli_band(shape, level)
            complex_response = np.real(scipy.fft.ifftn(spectrum))
            peak = np.abs(complex_response).max()
            response = simoncelli_wavelet(image, level=level)
            assert np.abs(response - complex_response).max() < 1e-6 * peak
            for order in ((1, 0, 0), (0, 1, 1), (2, 0, 0)):
                table = _simoncelli_whole(shape, level) * _riesz_whole(shape, order)
                spectrum = scipy.fft.rfftn(image.astype(np.float64)) * table
                reference = scipy.fft.irfftn(spectrum, s=shape)
                peak = np.abs(reference).max()
                one_trip = riesz_simoncelli(image, level=level, order=order)
                two_filters = riesz_transform(response, order=order)
                assert np.abs(one_trip - reference).max() < 1e-6 * peak
                assert np.abs(two_filters - reference).max() < 1e-4 * peak


def test_transfer_functions_built_in_slabs_match_one_volume() -> None:
    """The slab-by-slab transfer tables equal the whole-volume tables (the Riesz table:
    its kept rows), bit for bit (a small Riesz table keeps all its rows)."""
    from pictologics.filters import base, riesz, wavelets

    def small_slabs(shape: tuple[int, ...]) -> list[tuple[int, int]]:
        return base._slabs(shape, elements=40, minimum=0)

    assert base._slabs((5, 3, 2), elements=6, minimum=0) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert base._slabs((5, 3, 2)) == [(0, 5)]  # a small table is one slab
    assert wavelets._simoncelli_transfer.__wrapped__((9, 8, 7), 1).shape == (9, 8, 4)
    with (
        patch.object(wavelets, "_slabs", small_slabs),
        patch.object(riesz, "_slabs", small_slabs),
        patch.object(base, "_HALF_TABLE_MIN", 0),
    ):
        for shape in ((9, 8, 7), (6, 10, 5)):
            for level in (1, 2):
                built = wavelets._simoncelli_transfer.__wrapped__(shape, level)
                assert_array_equal(built, _simoncelli_whole(shape, level))
                assert not built.flags.writeable
            kept = slice(shape[0] // 2 + 1)
            for order in ((1, 0, 0), (0, 1, 1), (0, 0, 2)):
                built = riesz._riesz_transfer.__wrapped__(shape, order)
                assert_array_equal(built, _riesz_whole(shape, order)[kept])
                assert not built.flags.writeable


def test_laws_rotations_pool_as_the_rotation_loop() -> None:
    """The rotation-invariant Laws filter pools each base response as soon as it is ready,
    with in-place signs, passes and energy: the values of the loop over all 24 rotations
    with signed copies, bit for bit, for every pooling mode. Only the sign of a zero can
    differ: the filter pools the rotations in another order, and on x86 CPUs the maximum
    or minimum of -0.0 and +0.0 depends on the order (on ARM it does not)."""
    from scipy.ndimage import convolve1d, uniform_filter

    from pictologics.filters.laws import _get_rotation_permutations_3d, _parse_kernel_string

    def rotation_loop(image: np.ndarray, kernels: str, pooling: str, energy: bool) -> np.ndarray:
        g = [LAWS_KERNELS[name].astype(np.float32) for name in _parse_kernel_string(kernels)]
        antisym = [bool(np.allclose(k, -k[::-1])) for k in g]
        rotations = _get_rotation_permutations_3d()
        result = None
        for perm, flips in rotations:
            base = image
            for axis in range(3):
                base = convolve1d(base, g[perm[axis]], axis=axis, mode="constant")
            sign = 1
            for i, do_flip in enumerate(flips):
                if do_flip and antisym[perm[i]]:
                    sign = -sign
            signed = base if sign > 0 else -base
            if result is None:
                result = signed.astype(np.float64) if pooling == "average" else signed.copy()
            elif pooling == "max":
                np.maximum(result, signed, out=result)
            elif pooling == "average":
                result += signed
            else:
                np.minimum(result, signed, out=result)
        if pooling == "average":
            result /= len(rotations)
        if energy:  # in the type of the pooled result: float32 for max and min
            result = uniform_filter(np.abs(result), size=5, mode="constant")
        return result.astype(np.float32)  # every filter gives float32

    image = np.random.default_rng(5).normal(size=(11, 10, 9)).astype(np.float32)
    image[:4] = -0.0
    image[4:6] = 0.0
    # Distinct kernels, a repeated kernel, three antisymmetric kernels (a key comes back
    # after other keys) and three symmetric ones (one key, one sign).
    for kernels in ("E5L5S5", "L5E5E5", "E5W5E5", "L3L3L3"):
        for pooling in ("max", "min", "average"):
            for energy in (False, True):
                expected = rotation_loop(image, kernels, pooling, energy)
                for use_parallel in (False, True):
                    result = laws_filter(
                        image,
                        kernels,
                        rotation_invariant=True,
                        pooling=pooling,
                        compute_energy=energy,
                        energy_distance=2,
                        use_parallel=use_parallel,
                    )
                    assert result.dtype == expected.dtype
                    bits = np.uint32 if result.dtype == np.float32 else np.uint64
                    # + 0.0 makes -0.0 into +0.0 and keeps every other value bit for bit
                    assert_array_equal((result + 0.0).view(bits), (expected + 0.0).view(bits))


def test_filter_threads_follow_the_numba_thread_count() -> None:
    """The FFT filters, the rotation threads of the wavelet and Laws filters (and the pass
    threads of a large wavelet image) and the slice threads of the Gabor filter use numba's
    thread count, with the same values."""
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack
    from unittest.mock import patch

    import scipy.fft

    from pictologics.filters import _wavelet_pass, base, gabor, laws, riesz, wavelets

    image = np.random.default_rng(7).normal(size=(12, 11, 10)).astype(np.float32)
    fft_runs = {
        "simoncelli": lambda: simoncelli_wavelet(image, level=2),
        "riesz": lambda: riesz_transform(image, order=(1, 0, 0)),
    }
    pool_runs = {
        "wavelet": lambda: wavelet_transform(
            image, wavelet="haar", rotation_invariant=True, use_parallel=True
        ),
        "laws": lambda: laws_filter(image, "L3E3S3", rotation_invariant=True, use_parallel=True),
        "gabor": lambda: gabor_filter(image, sigma_mm=2.0, lambda_mm=3.0, use_parallel=True),
    }
    expected = {name: run() for name, run in {**fft_runs, **pool_runs}.items()}
    with ExitStack() as stack:
        for module in (base, gabor, laws, riesz, wavelets):
            stack.enter_context(patch.object(module, "get_num_threads", return_value=2))
        ffts = [
            stack.enter_context(patch.object(scipy.fft, name, wraps=getattr(scipy.fft, name)))
            for name in ("rfftn", "irfftn")
        ]
        for name, run in fft_runs.items():
            assert_array_equal(run(), expected[name])
        assert all(fft.called for fft in ffts)
        assert all(c.kwargs["workers"] == 2 for fft in ffts for c in fft.call_args_list)
        pools = [
            stack.enter_context(
                patch.object(module, "ThreadPoolExecutor", wraps=ThreadPoolExecutor)
            )
            for module in (base, gabor)  # base: the rotation pools of wavelets and Laws
        ]
        for name, run in pool_runs.items():
            assert_array_equal(run(), expected[name])
        assert [[c.kwargs["max_workers"] for c in pool.call_args_list] for pool in pools] == [
            [2, 2],
            [2],
        ]
        # A large wavelet image (the size gates at 1): one rotation at a time with 2 threads,
        # each pass in slabs
        stack.enter_context(patch.object(wavelets, "_PAIRED_PASSES_MIN", 1))
        stack.enter_context(patch.object(_wavelet_pass, "_SLAB_MIN_SIZE", 1))
        slabs = stack.enter_context(
            patch.object(_wavelet_pass, "_slab_pool", wraps=base._slab_pool)
        )
        wavelet_transform(image, wavelet="haar", rotation_invariant=True, use_parallel=True)
        assert slabs.call_count == 24 * 3  # 3 passes of each of the 24 rotations


def test_filter_inputs_are_checked() -> None:
    """Wrong levels, decompositions and Riesz orders raise a clear error; a lowercase
    decomposition and whole numbers as floats work."""
    import re

    image = np.random.default_rng(1).normal(size=(12, 12, 12))
    for call, message in (
        (
            lambda: wavelet_transform(image, level=0),
            "level must be a whole number of 1 or more, not 0",
        ),
        (
            lambda: wavelet_transform(image, level=1.5),
            "level must be a whole number of 1 or more, not 1.5",
        ),
        (
            lambda: wavelet_transform(image, decomposition="LH"),
            "decomposition must be 3 letters L or H",
        ),
        (
            lambda: wavelet_transform(image, decomposition="LXH"),
            "decomposition must be 3 letters L or H",
        ),
        (
            lambda: simoncelli_wavelet(image, level=0),
            "level must be a whole number of 1 or more, not 0",
        ),
        (
            lambda: riesz_transform(image, order=(1, 0)),
            "order must be 3 whole numbers of 0 or more",
        ),
        (
            lambda: riesz_transform(image, order=(2, -1, 0)),
            "order must be 3 whole numbers of 0 or more",
        ),
        (lambda: riesz_transform(image, order=1), "order must be 3 whole numbers of 0 or more"),
        (
            lambda: riesz_log(image, sigma_mm=1.0, order=(0, 0, 0)),
            "At least one order component must be > 0",
        ),
    ):
        with pytest.raises(ValueError, match=re.escape(message)):
            call()
    assert_array_equal(
        wavelet_transform(image, decomposition="lhl"), wavelet_transform(image, decomposition="LHL")
    )
    assert_array_equal(wavelet_transform(image, level=2.0), wavelet_transform(image, level=2))
    assert_array_equal(
        riesz_transform(image, order=[1.0, 0, 0]), riesz_transform(image, order=(1, 0, 0))
    )


def test_transfer_cache_is_safe_for_threads() -> None:
    """Threads that read and evict the transfer cache at the same time get the right
    tables."""
    from concurrent.futures import ThreadPoolExecutor

    from pictologics.filters.base import cache_by_bytes

    @cache_by_bytes(3 * 8 * 100)  # room for about three tables
    def table(n: int) -> np.ndarray:
        return np.full(100, float(n))

    with ThreadPoolExecutor(8) as pool:
        tables = list(pool.map(lambda k: table(k % 5), range(400)))
    for k, values in enumerate(tables):
        assert_array_equal(values, np.full(100, float(k % 5)))


def test_every_filter_returns_float32() -> None:
    """For a float64 image, every filter returns float32."""
    image = np.random.default_rng(2).normal(size=(14, 14, 14))
    mask = image > -1.5
    outputs = {
        "laws": laws_filter(image, "L5E5W5"),
        "laws RI": laws_filter(image, "L5E5W5", rotation_invariant=True, pooling="average"),
        # a symmetric kernel: one pooling step, so the copy at the end gives float32
        "laws RI one step": laws_filter(image, "L5L5L5", rotation_invariant=True, pooling="max"),
        "laws energy": laws_filter(image, "L5E5W5", compute_energy=True),
        "laws masked": laws_filter(image, "L5E5W5", source_mask=mask)[0],
        "wavelet": wavelet_transform(image, level=2),
        "wavelet RI": wavelet_transform(image, rotation_invariant=True),
        "log": laplacian_of_gaussian(image, sigma_mm=1.0, spacing_mm=(1.0, 1.0, 1.0)),
        "mean": mean_filter(image, support=3),
        "gabor": gabor_filter(image, sigma_mm=2.0, lambda_mm=4.0),
        "simoncelli": simoncelli_wavelet(image),
        "riesz": riesz_transform(image, order=(1, 0, 0)),
    }
    assert {name: out.dtype for name, out in outputs.items()} == dict.fromkeys(
        outputs, np.dtype(np.float32)
    )


def test_gabor_short_fft_is_the_linear_convolution() -> None:
    """The Gabor FFT is only as long as the padded slice: the kept part is a kernel
    radius from its ends, so it equals the linear convolution of the padded slice (to
    float32 rounding), also when the FFT length is the padded length itself."""
    from scipy.signal import fftconvolve

    sigma, wavelength, gamma, theta = 2.0, 4.0, 1.0, 0.3
    kernel = _create_gabor_kernel_2d(sigma, wavelength, gamma, theta)
    pad = kernel.shape[0] // 2
    rng = np.random.default_rng(3)
    for size in (64 - 2 * pad, 37):  # an FFT length of 64 = the padded slice; 37: rounded up
        image = rng.normal(size=(3, size, size + 5))
        response = _apply_gabor_to_plane(
            image, sigma, wavelength, gamma, [theta], 0, (1.0, 1.0, 1.0), "mirror", "average",
            use_parallel=False,
        )  # fmt: skip
        for k in range(3):
            padded = np.pad(image[k], pad, mode="reflect")
            full = fftconvolve(padded, kernel)  # float64, the whole linear convolution
            expected = np.abs(full[2 * pad : 2 * pad + size, 2 * pad : 2 * pad + size + 5])
            np.testing.assert_allclose(response[k], expected, rtol=0, atol=2e-6 * expected.max())


def test_slab_ufunc_gives_the_values_of_one_call() -> None:
    """A ufunc on slabs in threads (large arrays) or in one call gives the same values,
    with array and number inputs, a cast output and a unary ufunc."""
    import numba

    from pictologics.filters import base
    from pictologics.filters.base import _SLAB_MIN_SIZE, _slab_ufunc

    rng = np.random.default_rng(4)
    threads = numba.get_num_threads()
    numba.set_num_threads(max(2, min(4, numba.config.NUMBA_NUM_THREADS)))
    try:
        for size, gate in ((1000, 1 << 30), (_SLAB_MIN_SIZE + 17, 1)):
            stack = patch.object(base, "_SLAB_UFUNC_MIN_BYTES", gate)
            stack.start()
            a = rng.normal(size=(size // 10, 10))
            b = rng.normal(size=a.shape)
            out = np.empty(a.shape, dtype=np.float32)
            assert _slab_ufunc(np.minimum, (a, b), out) is out
            assert_array_equal(out, np.minimum(a, b).astype(np.float32))
            in_place = a.copy()
            _slab_ufunc(np.true_divide, (in_place, 24), in_place)
            assert_array_equal(in_place, a / 24)
            _slab_ufunc(np.negative, (in_place,), in_place)
            assert_array_equal(in_place, -(a / 24))
            stack.stop()
    finally:
        numba.set_num_threads(threads)


def test_gabor_region_cuts_each_slice_to_the_region() -> None:
    """With a region, each slice is cut to the region grown by the kernel radius. The
    region keeps the values of the whole image (to float32 rounding), at a region inside
    the image, at an image edge, with anisotropic spacing and over three planes."""
    from unittest.mock import patch

    import pictologics.filters.gabor as gabor

    rng = np.random.default_rng(6)
    image = rng.normal(0.0, 50.0, (70, 64, 20))
    cases = (
        ((slice(30, 40), slice(28, 36), slice(5, 15)), {}),
        ((slice(0, 9), slice(55, 64), slice(0, 20)), {"spacing_mm": (0.8, 1.2, 2.0)}),
        ((slice(30, 40), slice(28, 36), slice(5, 15)), {"average_over_planes": True}),
    )
    for region, extra in cases:
        kwargs = {"sigma_mm": 2.0, "lambda_mm": 3.0, "theta": 0.4, **extra}
        with patch.object(gabor, "_apply_gabor_to_plane", wraps=gabor._apply_gabor_to_plane) as spy:
            part = gabor_filter(image, **kwargs, region=region)
        assert all(call.kwargs["window"] is not None for call in spy.call_args_list)
        whole = gabor_filter(image, **kwargs)[region]
        assert part.shape == whole.shape and part.dtype == np.float32
        np.testing.assert_allclose(part, whole, rtol=0, atol=1e-6 * np.abs(whole).max())


def test_gaussian_filter_is_the_scipy_gaussian() -> None:
    """The Gaussian filter (8BC3) smooths with the sigma of each axis in voxels, as
    scipy.ndimage.gaussian_filter does, for each boundary; with a source mask, it is the
    normalized convolution G * (f m) / G * m where the weight reaches 0.01."""
    from scipy.ndimage import gaussian_filter as scipy_gaussian

    rng = np.random.default_rng(10)
    image = rng.normal(40.0, 20.0, (20, 18, 12))
    spacing = (0.8, 1.0, 2.0)
    sigma = tuple(2.0 / s for s in spacing)
    for boundary, mode in (("zero", "constant"), ("nearest", "nearest"), ("mirror", "reflect"), ("periodic", "wrap")):  # fmt: skip
        response = gaussian_filter(image, 2.0, spacing, truncate=3.0, boundary=boundary)
        assert response.dtype == np.float32
        expected = scipy_gaussian(image, sigma, mode=mode, truncate=3.0)
        np.testing.assert_allclose(response, expected, rtol=1e-6, atol=1e-4)
    mask = rng.random(image.shape) > 0.2
    response, valid = gaussian_filter(image, 2.0, spacing, source_mask=mask)
    weight = scipy_gaussian(mask.astype(np.float64), sigma, mode="constant")
    expected = scipy_gaussian(np.where(mask, image, 0.0), sigma, mode="constant") / weight
    assert_array_equal(valid, weight >= 0.01)
    np.testing.assert_allclose(response[valid], expected[valid], rtol=1e-5)


def test_gabor_response_parts() -> None:
    """The modulus, real, imaginary and angle maps are those parts of the complex
    response. Rotation-invariant pooling pools the part over all orientations: the real
    part keeps the pi symmetry of the modulus, the imaginary part and the angle do not."""
    from scipy.signal import fftconvolve

    sigma, wavelength, gamma, theta = 2.0, 4.0, 1.0, 0.3
    kernel = _create_gabor_kernel_2d(sigma, wavelength, gamma, theta)
    pad = kernel.shape[0] // 2
    image = np.random.default_rng(11).normal(size=(30, 28, 3))
    expected = np.empty(image.shape, dtype=np.complex128)
    for k in range(3):
        full = fftconvolve(np.pad(image[:, :, k], pad), kernel)  # the zero boundary
        expected[:, :, k] = full[2 * pad : 2 * pad + 30, 2 * pad : 2 * pad + 28]
    scale = np.abs(expected).max()
    for part, reference in (("modulus", np.abs), ("real", np.real), ("imaginary", np.imag)):
        got = gabor_filter(image, sigma, wavelength, gamma, theta, boundary="zero", response=part)
        np.testing.assert_allclose(got, reference(expected), rtol=0, atol=2e-6 * scale)
    angle = gabor_filter(image, sigma, wavelength, gamma, theta, boundary="zero", response="angle")
    away = np.abs(expected) > 1e-3 * scale  # the angle of a value near 0 is noise
    np.testing.assert_allclose(angle[away], np.angle(expected)[away], rtol=0, atol=1e-3)
    thetas = [i * np.pi / 4 for i in range(8)]
    for part, pooling in (("real", "max"), ("imaginary", "max"), ("angle", "min")):
        pooled = gabor_filter(
            image, sigma, wavelength, gamma, rotation_invariant=True, delta_theta=np.pi / 4,
            pooling=pooling, response=part,
        )  # fmt: skip
        each = [gabor_filter(image, sigma, wavelength, gamma, t, response=part) for t in thetas]
        reduce = np.max if pooling == "max" else np.min
        np.testing.assert_allclose(pooled, reduce(each, axis=0), rtol=0, atol=1e-5)
    with pytest.raises(ValueError, match="Unknown response: phase"):
        gabor_filter(image, sigma, wavelength, response="phase")


def test_padding_value_is_constant_value_padding() -> None:
    """A padding value C (Z3VE) gives the response of the image padded with C (far enough
    for a spatial filter; by the pad width of an FFT filter), cut back, for every filter,
    also with a source mask. The value 0 is the zero boundary; a value with another
    boundary raises. The Laws energy then pads the response with 0, as for C = 0."""
    from scipy.ndimage import uniform_filter

    from pictologics.filters.base import resolve_boundary

    assert resolve_boundary("constant") is BoundaryCondition.ZERO
    rng = np.random.default_rng(12)
    image = rng.normal(40.0, 20.0, (16, 14, 12)).astype(np.float32)
    value, margin = -50.0, 24  # the margin is more than the reach of each spatial filter
    spatial = {
        "mean": lambda img, **kw: mean_filter(img, support=5, **kw),
        "gaussian": lambda img, **kw: gaussian_filter(img, 1.5, **kw),
        "log": lambda img, **kw: laplacian_of_gaussian(img, 1.5, **kw),
        "laws": lambda img, **kw: laws_filter(img, "E5L5S5", **kw),
        "laws_ri": lambda img, **kw: laws_filter(img, "E3L3S3", rotation_invariant=True, **kw),
        "wavelet": lambda img, **kw: wavelet_transform(
            img, "db2", level=2, decomposition="HLH", **kw
        ),
        "gabor": lambda img, **kw: gabor_filter(img, 1.5, 3.0, **kw),
    }
    fft = {  # (filter, the pad width of its boundary padding, for every axis)
        "simoncelli": (lambda img, **kw: simoncelli_wavelet(img, level=1, **kw), 8),
        "riesz": (lambda img, **kw: riesz_transform(img, order=(1, 0, 0), **kw), 16),
        "riesz_log": (lambda img, **kw: riesz_log(img, 1.0, order=(0, 1, 0), **kw), 20),
        "riesz_simoncelli": (lambda img, **kw: riesz_simoncelli(img, 1, order=(0, 0, 1), **kw), 24),
    }
    cases = [(name, call, (margin,) * 3, "zero") for name, call in spatial.items()]
    # An FFT filter pads each axis by its width, at most the length of the axis
    cases += [
        (name, call, tuple(min(width, n) for n in image.shape), "periodic")
        for name, (call, width) in fft.items()
    ]
    for name, call, widths, outer in cases:
        cut = tuple(slice(w, w + n) for w, n in zip(widths, image.shape, strict=True))
        padded = np.pad(image, [(w, w) for w in widths], constant_values=value)
        expected = call(padded, boundary=outer)[cut]
        got = call(image, boundary="constant", padding_value=value)
        np.testing.assert_allclose(
            got, expected, rtol=0, atol=1e-5 * np.abs(expected).max(), err_msg=name
        )
        assert_array_equal(
            call(image, boundary="zero", padding_value=0.0), call(image, boundary="zero")
        )
        with pytest.raises(ValueError, match="needs the constant"):
            call(image, boundary="mirror", padding_value=value)
    mask = rng.random(image.shape) > 0.1
    response, valid = mean_filter(image, 3, "constant", source_mask=mask, padding_value=value)
    assert response.shape == valid.shape == image.shape
    energy = laws_filter(image, "E5L5S5", "constant", compute_energy=True, energy_distance=2, padding_value=value)  # fmt: skip
    plain = laws_filter(image, "E5L5S5", "constant", padding_value=value)
    np.testing.assert_allclose(energy, uniform_filter(np.abs(plain.astype(np.float64)), 5, mode="constant"), rtol=1e-5)  # fmt: skip
    energy, valid = laws_filter(image, "E5L5S5", "constant", compute_energy=True, source_mask=mask, padding_value=value)  # fmt: skip
    assert energy.shape == valid.shape == image.shape


def test_padding_value_reads_the_constant_with_no_padded_copy() -> None:
    """Without a source mask, mean, Gaussian and Laws read the padding value beyond the image
    (each pass hands the constant to the next): the bits of the padded copy, for float32
    and float64 images, padding values -1000, 0.1 and 1e30, and an image of one plane. With
    a source mask the padded copy stays."""
    from pictologics.filters import base, gaussian, laws, mean
    from pictologics.filters.laws import _laws_energy

    zero = BoundaryCondition.ZERO
    rng = np.random.default_rng(27)
    for shape in ((1, 5, 7), (6, 7, 8)):
        for dtype in (np.float32, np.float64):
            image = rng.normal(40.0, 20.0, shape).astype(dtype)
            for value in (-1000.0, 0.1, 1e30):
                cases = [
                    (mean_filter, {"support": 5}, 2),
                    (mean_filter, {"support": 15}, 7),
                    (gaussian_filter, {"sigma_mm": 1.5, "spacing_mm": (1.0, 0.8, 1.2)}, (6, 8, 5)),
                    (laws_filter, {"kernels": "E5L5S5"}, 2),
                    (laws_filter, {"kernels": "E3L3S3", "rotation_invariant": True}, 1),
                ]
                for func, kw, reach in cases:
                    expected = base._constant_padded(
                        func, image, None, value, reach, boundary=zero, **kw
                    )
                    with patch.object(base, "_constant_padded") as padded:
                        got = func(image, boundary="constant", padding_value=value, **kw)
                    assert padded.call_count == 0
                    assert got.tobytes() == expected.tobytes(), (func.__name__, kw, value)
                plain = base._constant_padded(
                    laws_filter, image, None, value, 2, kernels="E5L5S5", boundary=zero
                )
                energy = laws_filter(
                    image, "E5L5S5", "constant", compute_energy=True, padding_value=value
                )
                assert energy.tobytes() == _laws_energy(plain.copy(), 7, "constant").tobytes()
    # A source mask keeps the padded copy
    image = rng.normal(40.0, 20.0, (6, 7, 8)).astype(np.float32)
    mask = rng.random(image.shape) > 0.2
    for module, call in (
        (mean, lambda: mean_filter(image, 5, "constant", source_mask=mask, padding_value=3.0)),
        (
            gaussian,
            lambda: gaussian_filter(
                image, 1.5, boundary="constant", source_mask=mask, padding_value=3.0
            ),
        ),
        (
            laws,
            lambda: laws_filter(image, "E5L5S5", "constant", source_mask=mask, padding_value=3.0),
        ),
    ):
        with patch.object(module, "_constant_padded", wraps=base._constant_padded) as padded:
            call()
        assert padded.call_count == 1


def test_filters_name_unknown_boundaries() -> None:
    # Every filter resolves its boundary by name, so an unknown name gives one ValueError
    # that lists the names (also "constant")
    from pictologics.filters import gabor_filter, laplacian_of_gaussian, laws_filter, mean_filter

    image = np.zeros((6, 6, 6))
    calls = (
        lambda: mean_filter(image, 3, boundary="reflect"),
        lambda: laplacian_of_gaussian(image, 1.0, boundary="reflect"),
        lambda: laws_filter(image, "L5E5E5", boundary="reflect"),
        lambda: gabor_filter(image, 1.0, 2.0, boundary="reflect"),
    )
    for call in calls:
        with pytest.raises(
            ValueError, match="Valid values: zero, nearest, periodic, mirror, constant"
        ):
            call()


def _old_rotated_wavelet(
    image: np.ndarray, lo: np.ndarray, hi: np.ndarray, level: int, decomposition: str,
    mode: str, rotation: tuple,
) -> np.ndarray:  # fmt: skip
    """One rotation of the wavelet response by the old method: the plain passes on a
    rotated (transposed and flipped) view of the image, and the rotation undone."""
    from pictologics.filters import wavelets

    perm, flips = rotation
    rotated = np.transpose(image, perm)
    for axis, flip in enumerate(flips):
        if flip:
            rotated = np.flip(rotated, axis=axis)
    response = wavelets._apply_undecimated_wavelet_3d(
        rotated, lo, hi, level, decomposition, mode, 1
    )
    for axis, flip in enumerate(flips):
        if flip:
            response = np.flip(response, axis=axis)
    return np.transpose(response, tuple(np.argsort(perm)))


def _old_pooled(responses: list, pooling: str) -> np.ndarray:
    """The responses pooled in rotation order, as the old code pooled them."""
    result = responses[0].astype(np.float64) if pooling == "average" else responses[0].copy()
    for response in responses[1:]:
        if pooling == "average":
            result += response
        else:
            (np.maximum if pooling == "max" else np.minimum)(result, response, out=result)
    return (result / 24 if pooling == "average" else result).astype(np.float32)


def _scipy_wavelet_passes() -> Any:
    """The wavelet passes of scipy (the path before the numba kernels), for a test of the
    rotation steps: with the compiler off, the Python form of the kernels takes hours there.
    test_wavelet_pass_matches_scipy shows that the kernels give the bits of scipy."""
    from scipy.ndimage import convolve1d

    from pictologics.filters import base, wavelets

    def scipy_pass(
        image: np.ndarray,
        axis: int,
        output: Any,
        threads: Any,
        weights: Any,
        mode: str,
        origin: int,
    ) -> np.ndarray:
        kw = {"weights": weights, "mode": mode, "origin": origin}
        return base._slab_pass(convolve1d, image, axis, output, threads, **kw)

    return patch.object(wavelets, "wavelet_pass", scipy_pass)


def test_wavelet_rotations_need_no_rotated_copies() -> None:
    # Each of the 24 rotations of the rotation-invariant wavelet is computed on the image
    # itself: passes along the rotated axes, with reversed kernels on the flipped axes.
    # Every rotation equals the old rotated-copy response within a few float32 units (the
    # identity bit for bit), for 4 wavelets, levels 1 and 2, every boundary and 3
    # decompositions; so do the pooled responses, with and without threads, and with a
    # source mask. The passes are those of scipy here (see _scipy_wavelet_passes).
    with _scipy_wavelet_passes():
        _rotations_need_no_rotated_copies()


def _rotations_need_no_rotated_copies() -> None:
    import pywt

    from pictologics.filters import wavelets
    from pictologics.filters.base import _prepare_masked_image

    image = np.random.default_rng(41).normal(size=(24, 24, 24)).astype(np.float32)
    rotations = wavelets._get_rotation_perms()
    names = {"constant": "zero", "nearest": "nearest", "wrap": "periodic", "reflect": "mirror"}
    for wavelet in ("db2", "db3", "coif1", "haar"):
        lo = np.array(pywt.Wavelet(wavelet).dec_lo, dtype=np.float32)
        hi = np.array(pywt.Wavelet(wavelet).dec_hi, dtype=np.float32)
        for level in (1, 2):
            for mode, name in names.items():
                for decomposition in ("LHL", "HHH", "LLL"):
                    old = [_old_rotated_wavelet(image, lo, hi, level, decomposition, mode, r) for r in rotations]  # fmt: skip
                    for rotation, expected in zip(rotations, old, strict=True):
                        got = wavelets._apply_undecimated_wavelet_3d(
                            image, lo, hi, level, decomposition, mode, 1, rotation=rotation
                        )
                        if rotation == ((0, 1, 2), (False, False, False)):
                            assert got.tobytes() == expected.tobytes()
                        tolerance = 1e-6 * np.max(np.abs(expected))
                        assert_allclose(got, expected, rtol=1e-6, atol=tolerance)
                    for pooling in ("average", "max", "min"):
                        calls = [
                            wavelet_transform(
                                image,
                                wavelet,
                                level,
                                decomposition,
                                name,
                                rotation_invariant=True,
                                pooling=pooling,
                                use_parallel=parallel,
                            )  # fmt: skip
                            for parallel in (False, True)
                        ]
                        assert calls[0].tobytes() == calls[1].tobytes()
                        expected = _old_pooled(old, pooling)
                        assert_allclose(calls[0], expected, rtol=1e-6, atol=1e-6 * np.max(np.abs(expected)))  # fmt: skip
    mask = np.random.default_rng(42).random(image.shape) > 0.2
    lo = np.array(pywt.Wavelet("db2").dec_lo, dtype=np.float32)
    hi = np.array(pywt.Wavelet("db2").dec_hi, dtype=np.float32)
    masked = _prepare_masked_image(image, mask)
    old = [_old_rotated_wavelet(masked, lo, hi, 1, "LHL", "constant", r) for r in rotations]
    got = wavelet_transform(image, "db2", 1, "LHL", rotation_invariant=True, source_mask=mask)
    assert_allclose(got, _old_pooled(old, "average"), rtol=1e-6, atol=1e-6 * np.max(np.abs(got)))


def _gabor_by_slices(image, sigma, thetas, plane_axis, mode, pooling, response, value=0.0):
    """The Gabor response of a plane with numpy, one slice and one orientation at a time
    (np.pad, an FFT, the product, an inverse FFT and the part), pooled in orientation
    order, as the filter did before the blocks."""
    import functools

    import scipy.fft

    kernels = [_create_gabor_kernel_2d(sigma, 2.0 * sigma, 1.0, t) for t in thetas]
    pad = kernels[0].shape[0] // 2
    np_mode = {"constant": "constant", "reflect": "symmetric", "mirror": "reflect"}.get(mode, mode)
    np_mode = {"nearest": "edge"}.get(np_mode, np_mode)
    part = {"modulus": np.abs, "real": np.real, "imaginary": np.imag, "angle": np.angle}[response]
    extra = {"constant_values": value} if np_mode == "constant" else {}
    slices = np.moveaxis(image, plane_axis, 0)
    h, w = slices.shape[1:]
    fshape = (scipy.fft.next_fast_len(h + 2 * pad), scipy.fft.next_fast_len(w + 2 * pad))
    spectra = [scipy.fft.fftn(k, s=fshape) for k in kernels]
    pooled = []
    for slice_2d in slices:
        padded = np.pad(slice_2d, pad, mode=np_mode, **extra).astype(np.complex64)
        spectrum = scipy.fft.fftn(padded, s=fshape)
        crops = [scipy.fft.ifftn(spectrum * k)[2 * pad : 2 * pad + h, 2 * pad : 2 * pad + w] for k in spectra]  # fmt: skip
        parts = [np.ascontiguousarray(part(c)) for c in crops]
        if len(parts) == 1:
            pooled.append(parts[0])
        elif pooling == "average":
            total = functools.reduce(np.add, [p.astype(np.float64) for p in parts])
            pooled.append((total / len(parts)).astype(np.float32))
        else:
            pooled.append(functools.reduce(np.maximum if pooling == "max" else np.minimum, parts))
    return np.moveaxis(np.stack(pooled), 0, plane_axis)


# From SciPy 1.18 an FFT over a batch of slices rounds a slice otherwise than an FFT of the
# slice alone (and by its place in the batch). The Gabor blocks then agree with one FFT for
# each slice within a few float32 units, not bit for bit.
_BATCH_FFT_ROUNDS = tuple(int(p) for p in scipy.__version__.split(".")[:2]) >= (1, 18)


def _assert_gabor_bits(
    got: np.ndarray, expected: np.ndarray, context: object = None, response: str = "modulus"
) -> None:
    """The bits of `expected` before SciPy 1.18; from SciPy 1.18, within 1e-5 of the largest
    value of `expected` (NaN where it has NaN). The angle is not compared from SciPy 1.18: it
    jumps by 2 pi at +-pi, so a change in the last bit of an FFT can move it by 2 pi."""
    got, expected = np.ascontiguousarray(got), np.ascontiguousarray(expected)
    assert _BATCH_FFT_ROUNDS or got.tobytes() == expected.tobytes(), context
    finite = np.abs(expected[np.isfinite(expected)])
    scale = float(finite.max()) if finite.size else 0.0
    near = response != "angle" or not _BATCH_FFT_ROUNDS
    assert not near or np.allclose(got, expected, rtol=1e-5, atol=1e-5 * scale, equal_nan=True), (
        context
    )


def test_gabor_blocks_give_the_bits_of_each_slice() -> None:
    """The slices in blocks (the pad, one FFT call, the products, one inverse FFT call and
    the pooled parts) give the bits of numpy on each slice: every boundary, part and
    pooling, a padding value, pads wider than the slices, float32 and float64 images, the
    three planes, sequential and threaded, and blocks with NaN, huge values or zeros (from
    SciPy 1.18: within a few float32 units, see _assert_gabor_bits)."""
    rng = np.random.default_rng(15)
    image = rng.normal(0.0, 30.0, (6, 7, 5))
    four = [k * np.pi / 4 for k in range(4)]
    cases = [(image, 1.0, [0.3], 2, mode, "average", "modulus") for mode in ("constant", "reflect", "mirror", "nearest", "wrap")]  # fmt: skip
    for response in ("modulus", "real", "imaginary", "angle"):
        for pooling in ("average", "max", "min"):
            cases.append((image.astype(np.float32), 1.0, four, 0, "mirror", pooling, response))
    with_nan = image.copy()
    with_nan[2, 3, 1] = np.nan
    cases += [
        (image, 3.0, [0.2], 1, "mirror", "average", "modulus"),  # pads wider than the slices
        (with_nan, 1.0, four, 2, "constant", "max", "real"),
        (image * 1e15, 1.0, four[:2], 2, "nearest", "average", "modulus"),
        (np.zeros(image.shape), 1.0, four, 2, "constant", "min", "imaginary"),
    ]
    for img, sigma, thetas, axis, mode, pooling, response in cases:
        expected = _gabor_by_slices(img, sigma, thetas, axis, mode, pooling, response)
        for parallel in (False, True):
            got = _apply_gabor_to_plane(
                img, sigma, 2.0 * sigma, 1.0, thetas, axis, (1.0, 1.0, 1.0), mode, pooling,
                use_parallel=parallel, response=response,
            )  # fmt: skip
            _assert_gabor_bits(got, expected, (sigma, axis, mode, pooling, response), response)
    got = _apply_gabor_to_plane(
        image, 1.0, 2.0, 1.0, four, 2, (1.0, 1.0, 1.0), "constant", "average", padding_value=1.5
    )
    expected = _gabor_by_slices(image, 1.0, four, 2, "constant", "average", "modulus", 1.5)
    _assert_gabor_bits(got, expected)


def test_gabor_first_use_checks_take_numpy_where_a_bit_differs() -> None:
    """When the numba product or part differs from numpy in one bit on this machine, the
    first-use check takes numpy for it: the response keeps numpy's bits. The checks also
    pass the numba forms to the slices."""
    from pictologics.filters import gabor

    image = np.random.default_rng(16).normal(size=(5, 6, 4))
    thetas = [0.0, 0.7]
    expected = _gabor_by_slices(image, 1.0, thetas, 2, "mirror", "average", "modulus")
    # The response with the kernels of this machine: the same FFT blocks, so the same bits
    # with every SciPy
    blocks = gabor._apply_gabor_to_plane(image, 1.0, 2.0, 1.0, thetas, 2, (1.0, 1.0, 1.0), "mirror", "average")  # fmt: skip
    _assert_gabor_bits(blocks, expected)
    real_products, real_parts = gabor._products_numba, gabor._pooled_parts_numba

    def wrong_products(spectra, kernels, products):
        real_products(spectra, kernels, products)
        products.real[0, 0, 0] = np.nextafter(products.real[0, 0, 0], np.float32(np.inf))

    def wrong_parts(products, n, r0, c0, part, pooling, first, last, total, acc, out):
        real_parts(products, n, r0, c0, part, pooling, first, last, total, acc, out)
        out[0, 0, 0] = np.nextafter(out[0, 0, 0], np.float32(np.inf))

    for name, wrong, exact in (
        ("_products_numba", wrong_products, gabor._products_exact),
        ("_pooled_parts_numba", wrong_parts, lambda: gabor._part_exact(0)),
    ):
        gabor._products_exact.cache_clear()
        gabor._part_exact.cache_clear()
        with patch.object(gabor, name, wrong):
            assert not exact()
            got = gabor._apply_gabor_to_plane(
                image, 1.0, 2.0, 1.0, thetas, 2, (1.0, 1.0, 1.0), "mirror", "average"
            )
        assert np.ascontiguousarray(got).tobytes() == blocks.tobytes()
    gabor._products_exact.cache_clear()
    gabor._part_exact.cache_clear()
    with (
        patch.object(gabor, "_products_exact", return_value=True),
        patch.object(gabor, "_part_exact", return_value=True),
    ):
        got = gabor._apply_gabor_to_plane(
            image, 1.0, 2.0, 1.0, thetas, 2, (1.0, 1.0, 1.0), "mirror", "average"
        )
    assert np.ascontiguousarray(got).tobytes() == blocks.tobytes()
    # Both check arrays have an odd length: numpy's vector loop and its tail both run
    for values in (gabor._FINITE_VALUES, gabor._SPECIAL_VALUES):
        assert all(z.size % 2 == 1 for z in gabor._check_values(values))


def test_gabor_orientation_groups_give_the_bits_of_one_group() -> None:
    """The orientations in groups (the products of a block within _PRODUCTS_BYTES for each
    worker) give the bits of one group: groups of 1, 2 and 3 of 5 orientations, each pooling
    with the modulus, the real part and the angle, the numba parts (where the first-use check
    of this machine passes) and numpy's, a block with a NaN (an FFT for each slice),
    sequential and threaded (from SciPy 1.18: within a few float32 units, see
    _assert_gabor_bits)."""
    from contextlib import nullcontext

    import scipy.fft

    from pictologics.filters import gabor

    rng = np.random.default_rng(17)
    image = rng.normal(0.0, 30.0, (3, 9, 8)).astype(np.float32)
    with_nan = image.copy()
    with_nan[1, 2, 3] = np.nan
    thetas = [k * np.pi / 5 for k in range(5)]
    pad = _create_gabor_kernel_2d(1.0, 2.0, 1.0, 0.0).shape[0] // 2
    side = scipy.fft.next_fast_len(9 + 2 * pad), scipy.fft.next_fast_len(8 + 2 * pad)
    one_group = gabor._BLOCK_SLICES * 8 * side[0] * side[1]  # the products of one orientation
    cases = [(img, pooling, response) for img in (image, with_nan) for pooling in ("average", "max", "min") for response in ("modulus", "real", "angle")]  # fmt: skip
    for exact in (None, False):
        forced = nullcontext() if exact is None else patch.object(gabor, "_part_exact", return_value=exact)  # fmt: skip
        with forced:
            for img, pooling, response in cases:
                expected = _gabor_by_slices(img, 1.0, thetas, 0, "mirror", pooling, response)
                for size in (5, 3, 2, 1):
                    for parallel in (False, True) if size == 2 else (False,):
                        with patch.object(gabor, "_PRODUCTS_BYTES", size * one_group):
                            got = _apply_gabor_to_plane(
                                img, 1.0, 2.0, 1.0, thetas, 0, (1.0, 1.0, 1.0), "mirror", pooling,
                                use_parallel=parallel, response=response,
                            )  # fmt: skip
                        _assert_gabor_bits(got, expected, (exact, pooling, response, size), response)  # fmt: skip


def test_gabor_pad_kernel_is_np_pad() -> None:
    """The pad of a block of slices is np.pad in each of the five modes (odd shapes, pads
    wider than the slice, a slice of one line), then zeros to the FFT shape. It tells
    whether every value and the constant are finite and within _SAFE_LIMIT."""
    from pictologics.filters import gabor

    rng = np.random.default_rng(17)
    np_modes = ("constant", "symmetric", "reflect", "edge", "wrap")
    for shape, (ph, pw) in (((3, 5, 4), (2, 3)), ((2, 1, 6), (4, 9)), ((1, 7, 1), (5, 2))):
        block = rng.normal(size=shape)
        buf = np.empty((shape[0], shape[1] + 2 * ph + 3, shape[2] + 2 * pw + 1), np.complex64)
        for code, np_mode in enumerate(np_modes):
            extra = {"constant_values": 1.5} if np_mode == "constant" else {}
            assert gabor._pad_slices_numba(block, buf, ph, pw, code, 1.5)
            expected = np.zeros_like(buf)
            for s in range(shape[0]):
                padded = np.pad(block[s], ((ph, ph), (pw, pw)), mode=np_mode, **extra)
                expected[s, : padded.shape[0], : padded.shape[1]] = padded
            assert buf.tobytes() == expected.tobytes()
    for bad in (np.nan, np.inf, -2e12):
        block[0, 3, 0] = bad
        assert not gabor._pad_slices_numba(block, buf, ph, pw, 2, 0.0)
    assert not gabor._pad_slices_numba(block[:, :2], buf, ph, pw, 0, 2e12)


def test_gabor_fft_in_blocks_gives_the_bits_of_single_ffts() -> None:
    """One FFT call over the planes of a block (in place) gives the bits of an FFT for each
    plane, forward and inverse, for blocks of 2, 3 and 4 planes (from SciPy 1.18: within a
    few float32 units)."""
    import scipy.fft

    rng = np.random.default_rng(18)
    for k in (2, 3, 4):
        planes = np.empty((k, 20, 18), dtype=np.complex64)
        planes.real, planes.imag = rng.normal(size=(2, k, 20, 18))
        for transform in (scipy.fft.fftn, scipy.fft.ifftn):
            single = np.stack([transform(p) for p in planes])
            block = transform(planes.copy(), axes=(1, 2), overwrite_x=True)
            assert _BATCH_FFT_ROUNDS or block.tobytes() == single.tobytes()
            assert_allclose(block, single, rtol=1e-5, atol=1e-5 * np.abs(single).max())


def test_gabor_blocks_are_the_same_pairs_for_every_thread_count() -> None:
    """Each worker starts its run of slices on an even slice, so the blocks are the same
    pairs of slices for every number of threads: from SciPy 1.18 a batched FFT rounds a
    slice by the batch, so other pairs would give other bits. 12 slices in 5 threads take
    blocks of 2 only (runs of 3 slices would end in blocks of 1)."""
    from pictologics.filters import gabor

    image = np.random.default_rng(19).normal(size=(12, 9, 8)).astype(np.float32)
    results = []
    for threads in (1, 5, 7):
        with (
            patch.object(gabor, "get_num_threads", return_value=threads),
            patch.object(gabor, "_pad_slices_numba", wraps=gabor._pad_slices_numba) as pads,
        ):
            results.append(
                gabor._apply_gabor_to_plane(
                    image,
                    1.0,
                    2.0,
                    1.0,
                    [0.0, 0.5],
                    0,
                    (1.0, 1.0, 1.0),
                    "mirror",
                    "average",
                    use_parallel=True,
                )  # fmt: skip
            )
        assert sorted(c.args[0].shape[0] for c in pads.call_args_list) == [2] * 6
    assert results[0].tobytes() == results[1].tobytes() == results[2].tobytes()


def test_the_fma_of_the_gabor_kernels_rounds_once() -> None:
    """The fma of the kernels (llvm.fma, compiled here with the compiler on) and its form
    for the compiler off give a * b + c with one rounding, as numpy's vector loops do."""
    import numba

    from pictologics.filters import gabor

    a, b, c = np.float32(1 + 2**-23), np.float32(1 - 2**-23), np.float32(-1.0)
    assert np.float32(a * b) + c == 0  # two roundings lose the low bits
    with patch.object(numba.config, "DISABLE_JIT", False):
        fma = numba.njit(lambda x, y, z: gabor._fma32_intrinsic(x, y, z))
        assert fma(a, b, c) == np.float32(-(2.0**-46))
    assert gabor._fma32_python(a, b, c) == np.float32(-(2.0**-46))


def test_the_fma_of_the_pass_kernels_rounds_once() -> None:
    """The float64 fma of the pass and resample kernels with the compiler off rounds
    a * b + c once, keeps the sign of a zero, and gives inf and NaN as the plain form does
    (every branch). The fma of the compiled kernels (llvm.fma, compiled here with the
    compiler on) gives the same value."""
    import math

    import numba

    from pictologics.filters._passes import _fma_intrinsic, _fma_python

    assert 3.0 * (1 / 3) - 1.0 == 0.0  # the plain form rounds twice
    assert _fma_python(3.0, 1 / 3, -1.0) == -5.551115123125783e-17
    with patch.object(numba.config, "DISABLE_JIT", False):
        fma = numba.njit(lambda x, y, z: _fma_intrinsic(x, y, z))
        assert fma(3.0, 1 / 3, -1.0) == -5.551115123125783e-17
    assert math.copysign(1.0, _fma_python(-0.5, 0.0, -0.0)) == -1.0
    assert _fma_python(1.0, 1e308, 1e308) == math.inf  # the exact sum overflows
    assert math.isnan(_fma_python(0.5, 1.0, math.nan))
    assert _fma_python(2.0, 0.5, -1.0) == 0.0


def test_pass_twins_match_scipy() -> None:
    """The numba twins of the correlation passes give the bits of correlate1d: float32 and
    float64 arrays with an axis of 1 and 3 lines, Gaussian kernels of 1, 9 and 10 pairs (the
    plain pairs of the block of scipy from 10 pairs on), an antisymmetric kernel, general
    kernels of 4, 5 and 1 taps, the 5 modes with kernels longer than the axis, cval 0 and
    -3, each axis."""
    from scipy.ndimage import correlate1d

    from pictologics.filters.base import _gaussian_weights, _numba_pass

    rng = np.random.default_rng(23)
    kernels = [
        _gaussian_weights(0.6, 0, 4.0),  # 1 pair
        _gaussian_weights(2.2, 0, 4.0),  # 9 pairs
        _gaussian_weights(2.4, 1, 4.0),  # 10 pairs, antisymmetric
        _gaussian_weights(2.4, 2, 4.0),  # 10 pairs
        np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / np.sqrt(10.0),  # E5
        np.array([0.1, -0.4, 0.7, 0.2]),
        np.array([0.3, 0.5, -0.1, 0.2, 0.6]),  # odd and neither symmetric nor antisymmetric
        np.array([1.7]),
    ]
    for shape in ((5, 6, 7), (1, 4, 6), (3, 1, 5)):
        for dtype in (np.float32, np.float64):
            image = rng.normal(0.0, 10.0, shape).astype(dtype)
            for axis in range(3):
                for mode in ("constant", "reflect", "mirror", "nearest", "wrap"):
                    for cval in (0.0, -3.0):
                        expected = [
                            correlate1d(image, w, axis, mode=mode, cval=cval) for w in kernels
                        ]
                        got = [
                            _numba_pass(
                                "correlate", image, axis, np.empty_like(image), 1, mode, cval, w
                            )
                            for w in kernels
                        ]
                        for a, b in zip(got, expected, strict=True):
                            assert a.dtype == b.dtype
                            assert a.tobytes() == b.tobytes(), (shape, dtype, axis, mode, cval)


def test_uniform_passes_sum_each_window_on_its_own() -> None:
    """The uniform passes give the exact mean of each window of float32 values, rounded once
    to double and then to float32 (a value does not depend on where the line starts): an
    axis of 1 and 3 lines, the 5 modes with windows longer than the axis, cval 0 and -3, the
    sizes 1, 2, 3, 5, 7 (plain sums), 9 (blocks of 3) and 15 (blocks of 5), each axis, in
    place. float64 values agree with the running sum of scipy to 1e-12. On a line of large
    and tiny values the running sum of scipy misses the exact mean; the window sums do not."""
    from fractions import Fraction

    from scipy.ndimage import uniform_filter1d

    from pictologics.filters.base import _PAD_MODES, _numba_pass

    def exact_means(image: np.ndarray, size: int, axis: int, mode: str, cval: float) -> np.ndarray:
        pads = [(0, 0)] * 3
        pads[axis] = (size // 2, size - size // 2 - 1)
        extra = {"constant_values": cval} if mode == "constant" else {}
        ext = np.moveaxis(np.pad(image, pads, mode=_PAD_MODES[mode], **extra), axis, -1)
        out = np.empty(np.moveaxis(image, axis, -1).shape, dtype=image.dtype)
        for index in np.ndindex(out.shape):
            window = ext[index[:-1]][index[-1] : index[-1] + size].tolist()
            out[index] = float(sum(map(Fraction, window), Fraction(0)) / size)
        return np.moveaxis(out, -1, axis)

    rng = np.random.default_rng(28)
    for shape in ((5, 6, 7), (1, 4, 6), (3, 1, 5)):
        for dtype in (np.float32, np.float64):
            image = rng.normal(0.0, 10.0, shape).astype(dtype)
            for axis in range(3):
                for mode in ("constant", "reflect", "mirror", "nearest", "wrap"):
                    for cval in (0.0, -3.0):
                        for size in (1, 2, 3, 5, 7, 9, 15):
                            got = image.copy()
                            _numba_pass("uniform", got, axis, got, 1, mode, cval, size=size)
                            if dtype == np.float32:
                                expected = exact_means(image, size, axis, mode, cval)
                                assert got.tobytes() == expected.tobytes(), (shape, axis, mode)
                            else:
                                running = uniform_filter1d(image, size, axis, mode=mode, cval=cval)
                                assert_allclose(got, running, rtol=1e-12, atol=1e-12)
    line = np.array(
        [8.574043e-07, 2000.0, 2000.0, 2000.0, 8.631789e-07, 5.4146125e-07, 2.997119e-07, 4.226872e-07, 2.8319672e-08, 1.2428328e-07, 6.706244e-07, 2000.0],
        dtype=np.float32,
    ).reshape(1, 1, 12)  # fmt: skip
    exact = exact_means(line, 3, 2, "constant", 0.0)
    got = _numba_pass("uniform", line, 2, np.empty_like(line), 1, "constant", 0.0, size=3)
    assert got.tobytes() == exact.tobytes()
    assert uniform_filter1d(line, 3, 2, mode="constant").tobytes() != exact.tobytes()


def test_pass_twins_in_threads_and_the_scipy_passes_agree() -> None:
    """A twin pass in threads gives the bits of one call. For an image in column order the
    Gaussian and the convolutions take the scipy passes, with the same values; the uniform
    passes copy it to row order (also in place) for the same window sums. Also with the
    constant of a padded image (cval), which each pass hands to the next."""
    from functools import partial

    from scipy.ndimage import uniform_filter1d
    from scipy.ndimage._filters import _gaussian_kernel1d

    from pictologics.filters import base

    rng = np.random.default_rng(24)
    image = rng.normal(0.0, 10.0, (9, 10, 11))
    kernels = (np.array([1.0, 2.0, 1.0]), np.array([-1.0, 0.0, 1.0]), np.array([1.0, -2.0, 1.0]))
    for cval in (0.0, -3.0):
        calls = [
            lambda a, c=cval: base._gaussian_filter(a, (1.5, 1.0, 2.0), "constant", cval=c),
            lambda a, c=cval: base._uniform_filter(a, 5, "constant", cval=c),
            lambda a, c=cval: base._convolve_axes(a, kernels, "constant", cval=c),
        ]
        for call, twin_passes in zip(calls, (0, 3, 0), strict=True):
            one = call(image)
            with patch.object(base, "_SLAB_MIN_SIZE", 1):
                threaded = call(image)
            with patch.object(base, "_numba_pass", wraps=base._numba_pass) as twin:
                columns = call(np.asfortranarray(image))
            assert twin.call_count == twin_passes
            assert one.tobytes() == threaded.tobytes()
            assert one.tobytes() == np.ascontiguousarray(columns).tobytes()
        in_place = np.asfortranarray(image)
        base._uniform_filter(in_place, 5, "constant", output=in_place, cval=cval)
        assert in_place.flags.f_contiguous
        assert np.ascontiguousarray(in_place).tobytes() == calls[1](image).tobytes()
    # The constant that a pass leaves on a constant line
    mean5 = partial(uniform_filter1d, size=5)
    assert base._pass_constant(mean5, 0.0, np.float32, 5) == 0.0
    assert base._pass_constant(mean5, -3.0, np.float32, 5) == float(np.float32(-3.0))
    # The Gaussian weights are those of scipy
    for sigma, order, truncate in ((0.6, 0, 4.0), (2.4, 1, 4.0), (1.7, 2, 3.0), (3.1, 0, 6.0)):
        radius = int(truncate * sigma + 0.5)
        expected = _gaussian_kernel1d(sigma, order, radius)[::-1]
        assert base._gaussian_weights(sigma, order, truncate).tobytes() == expected.tobytes()


def test_wavelet_pass_matches_scipy() -> None:
    """The numba wavelet pass gives the bits of convolve1d on float32 row-order images:
    db2, haar of level 2 (zero taps), coif3 of level 2 (36 taps: scipy on the short axes),
    bior2.2 (a zero tap at the start), each kernel flipped (origin -1) and not, the five
    modes, axes of 1 and 3 lines, each axis, a new output and in place, in one call and on
    slabs in threads. Other arrays and modes take scipy."""
    import pywt
    from scipy.ndimage import convolve1d

    from pictologics.filters import _wavelet_pass
    from pictologics.filters._wavelet_pass import wavelet_pass
    from pictologics.filters.wavelets import _atrous_upsample

    def kernel(name: str, level: int) -> np.ndarray:
        lo = np.array(pywt.Wavelet(name).dec_lo, dtype=np.float32)
        return _atrous_upsample(lo, level) if level > 1 else lo

    kernels = [kernel("db2", 1), kernel("haar", 2), kernel("coif3", 2), kernel("bior2.2", 1)]
    rng = np.random.default_rng(29)
    for shape in ((5, 6, 7), (3, 1, 5), (1, 4, 6)):
        image = rng.normal(0.0, 10.0, shape).astype(np.float32)
        for weights in kernels:
            for flip, origin in ((False, 0), (True, -1)):
                w = weights[::-1] if flip else weights
                for mode in ("constant", "nearest", "wrap", "reflect", "mirror"):
                    for axis in range(3):
                        expected = convolve1d(image, w, axis=axis, mode=mode, origin=origin)
                        got = wavelet_pass(image, axis, None, 1, w, mode, origin)
                        in_place = image.copy()
                        wavelet_pass(in_place, axis, in_place, 1, w, mode, origin)
                        for result in (got, in_place):
                            assert result.dtype == expected.dtype
                            assert result.tobytes() == expected.tobytes(), (shape, mode, axis)
    image = rng.normal(0.0, 10.0, (9, 10, 11)).astype(np.float32)
    w = kernel("db2", 2)
    for axis in range(3):
        expected = convolve1d(image, w, axis=axis, mode="mirror")
        for threads in (3, 1):
            with patch.object(_wavelet_pass, "_SLAB_MIN_SIZE", 1):
                got = wavelet_pass(image, axis, None, threads, w, "mirror", 0)
            assert got.tobytes() == expected.tobytes()
    # scipy for other arrays, a mode without a code and a kernel longer than the axis
    others = [
        (image.astype(np.float64), 0, "mirror", w),
        (np.asfortranarray(image), 1, "mirror", w),
        (image[0], 1, "mirror", w),
        (image, 2, "grid-wrap", w),
        (image[:2], 0, "mirror", kernel("coif3", 1)),
    ]
    for array, axis, mode, weights in others:
        with (
            patch.object(_wavelet_pass, "_pass_axis0") as k0,
            patch.object(_wavelet_pass, "_pass_axis1") as k1,
            patch.object(_wavelet_pass, "_pass_axis2") as k2,
        ):
            got = wavelet_pass(array, axis, None, 1, weights, mode, 0)
        assert k0.call_count == k1.call_count == k2.call_count == 0
        expected = convolve1d(array, weights, axis=axis, mode=mode)
        assert np.ascontiguousarray(got).tobytes() == np.ascontiguousarray(expected).tobytes()


def test_wavelet_pass_skips_the_zero_taps() -> None:
    """A NaN voxel under a kernel with zero taps (haar of level 2) spreads over the non-zero
    taps only (scipy adds NaN * 0 for a zero tap): the finite voxels are those of scipy, and
    the NaN voxels are a part of the NaN voxels of scipy."""
    import pywt
    from scipy.ndimage import convolve1d

    from pictologics.filters._wavelet_pass import wavelet_pass
    from pictologics.filters.wavelets import _atrous_upsample

    w = _atrous_upsample(np.array(pywt.Wavelet("haar").dec_lo, dtype=np.float32), 2)
    image = np.random.default_rng(30).normal(0.0, 10.0, (6, 7, 8)).astype(np.float32)
    image[3, 3, 3] = np.nan
    for axis in range(3):
        expected = convolve1d(image, w, axis=axis, mode="mirror")
        got = wavelet_pass(image, axis, None, 1, w, "mirror", 0)
        finite = np.isfinite(got)
        assert np.isnan(expected[~finite]).all()
        assert (~np.isfinite(expected)).sum() > (~finite).sum()
        both = finite & np.isfinite(expected)
        assert got[both].tobytes() == expected[both].tobytes()


def test_laws_rotation_invariance_keeps_two_bases_in_flight() -> None:
    """A large image (here the size gate at 1) with 6 distinct bases and 10 threads makes
    two bases at a time."""
    from pictologics.filters import laws

    image = np.random.default_rng(26).normal(size=(6, 7, 8))
    with (
        patch.object(laws, "_SLAB_MIN_SIZE", 1),
        patch.object(laws, "get_num_threads", return_value=10),
        patch.object(laws, "_ordered_map", wraps=laws._ordered_map) as pool,
    ):
        laws_filter(image, "E5L5S5", rotation_invariant=True, use_parallel=True)
    assert pool.call_args.args[2] == 2


def test_laws_pooling_steps_in_threads_give_the_values_of_one_thread() -> None:
    """The pooling steps of the Laws rotation invariance on slabs in threads give the bits
    of one call, for max, min and average pooling."""
    import numba

    from pictologics.filters import base

    image = np.random.default_rng(25).normal(0.0, 10.0, (12, 13, 14))
    threads = numba.get_num_threads()
    try:
        for pooling in ("max", "min", "average"):
            numba.set_num_threads(1)
            one = laws_filter(image, "E5L5S5", rotation_invariant=True, pooling=pooling)
            numba.set_num_threads(max(2, min(3, numba.config.NUMBA_NUM_THREADS)))
            with patch.object(base, "_SLAB_UFUNC_MIN_BYTES", 1):
                threaded = laws_filter(image, "E5L5S5", rotation_invariant=True, pooling=pooling)
            assert one.tobytes() == threaded.tobytes()
    finally:
        numba.set_num_threads(threads)


def test_padded_is_np_pad() -> None:
    """The pad of the padded filters is np.pad, in the same memory order: the five modes,
    odd shapes, widths before and after each axis (also wider than the axis), float32 and
    float64, row, column and strided images, a constant value, in one thread and on slabs
    in threads; another number of axes goes to np.pad."""
    from pictologics.filters import base

    rng = np.random.default_rng(19)
    widths_cases = (
        ((2, 3), (1, 4), (0, 5)),
        ((6, 9), (13, 2), (4, 12)),
        ((10, 1), (1, 10), (7, 7)),
    )
    for shape in ((5, 7, 3), (1, 6, 2), (9, 4, 11)):
        image = rng.normal(size=shape)
        for array in (image, image.astype(np.float32), np.asfortranarray(image), image[:, ::-1]):
            for widths in widths_cases:
                for mode in ("constant", "symmetric", "reflect", "edge", "wrap"):
                    extra = {"constant_values": 1.5} if mode == "constant" else {}
                    expected = np.pad(array, widths, mode=mode, **extra)
                    for threads, limit in ((1, base._SLAB_MIN_SIZE), (3, 1)):
                        with (
                            patch.object(base, "get_num_threads", return_value=threads),
                            patch.object(base, "_SLAB_MIN_SIZE", limit),
                        ):
                            got = base._padded(array, widths, mode, 1.5)
                        assert got.dtype == expected.dtype
                        assert got.flags.f_contiguous == expected.flags.f_contiguous
                        assert_array_equal(got, expected)
    flat = rng.normal(size=(6, 5))
    assert_array_equal(base._padded(flat, ((1, 2), (3, 0)), "reflect"), np.pad(flat, ((1, 2), (3, 0)), "reflect"))  # fmt: skip


def _slabs_from_one_voxel() -> ExitStack:
    """The slab gates of the passes, the adds and the ufuncs at 1, so a small image takes the
    slab paths of a large one."""
    from pictologics.filters import _wavelet_pass, base, wavelets

    stack = ExitStack()
    for module, name in (
        (base, "_SLAB_MIN_SIZE"),
        (base, "_SLAB_UFUNC_MIN_BYTES"),
        (_wavelet_pass, "_SLAB_MIN_SIZE"),
        (wavelets, "_SLAB_MIN_SIZE"),
    ):
        stack.enter_context(patch.object(module, name, 1))
    return stack


def test_wavelet_lll_and_hhh_rotations_share_their_passes() -> None:
    """The rotation-invariant LLL and HHH responses come from a tree of the 8 patterns of
    flipped axes (3 rotations each): 14 chains of level kernels instead of 72 passes. They
    equal the 24 rotated-copy responses pooled, within a few float32 units, for each
    pooling; the bits do not depend on the threads (passes and sums in slabs)."""
    import pywt

    from pictologics.filters import base, wavelets

    lo = np.array(pywt.Wavelet("db2").dec_lo, dtype=np.float32)
    hi = np.array(pywt.Wavelet("db2").dec_hi, dtype=np.float32)
    image = np.random.default_rng(43).normal(size=(20, 18, 16))
    rotations = wavelets._get_rotation_perms()
    large = np.random.default_rng(44).normal(size=(12, 13, 14))  # slabs (the gates at 1)
    for decomposition in ("LLL", "HHH"):
        old = [_old_rotated_wavelet(image, lo, hi, 2, decomposition, "reflect", r) for r in rotations]  # fmt: skip
        for pooling in ("average", "max", "min"):
            with patch.object(wavelets, "_axis_chain", wraps=wavelets._axis_chain) as chains:
                got = wavelet_transform(
                    image, "db2", 2, decomposition, "mirror", rotation_invariant=True,
                    pooling=pooling,
                )  # fmt: skip
            assert chains.call_count == 14
            expected = _old_pooled(old, pooling)
            assert_allclose(got, expected, rtol=1e-6, atol=1e-6 * np.max(np.abs(expected)))
            runs = []
            for threads in (1, 3):
                with (
                    patch.object(wavelets, "get_num_threads", return_value=threads),
                    patch.object(base, "get_num_threads", return_value=threads),
                    _slabs_from_one_voxel(),
                ):
                    runs.append(
                        wavelet_transform(
                            large,
                            "db2",
                            1,
                            decomposition,
                            "mirror",
                            rotation_invariant=True,
                            pooling=pooling,
                            use_parallel=True,
                        )  # fmt: skip
                    )
            assert runs[0].tobytes() == runs[1].tobytes()


def test_separable_filters_run_in_float32() -> None:
    """The mean, Gaussian, Laws (also its energy step) and wavelet filters run their
    passes in float32, and so does the LoG of a float64 image below 2^18 voxels: the
    response of a float64 image is the response of its float32 copy, bit for bit. It
    stays within a few float32 units of the response with float64 passes (the filters
    before): 4e-7 of the largest value, and 2e-6 for plain Laws, as its E5 kernel
    subtracts, so its passes hold values larger than the response."""
    from pictologics.filters import gaussian, laws, log, mean, wavelets

    image = np.cumsum(np.random.default_rng(45).normal(size=(24, 22, 20)), axis=2) * 20.0 + 100.0
    cases = {
        "mean": (lambda x: mean_filter(x, support=5, boundary="mirror"), 4e-7),
        "gaussian": (lambda x: gaussian_filter(x, sigma_mm=1.5, boundary="mirror"), 4e-7),
        "log": (lambda x: laplacian_of_gaussian(x, sigma_mm=1.5, boundary="mirror"), 4e-7),
        "laws": (lambda x: laws_filter(x, "L5E5E5", boundary="mirror"), 2e-6),
        "laws energy": (lambda x: laws_filter(x, "L5E5E5", "mirror", compute_energy=True), 4e-7),
        "laws ri": (lambda x: laws_filter(x, "L5E5E5", "mirror", rotation_invariant=True), 4e-7),
        "laws ri average": (
            lambda x: laws_filter(
                x, "S5S5L5", "mirror", rotation_invariant=True, pooling="average"
            ),
            4e-7,
        ),
        "wavelet": (lambda x: wavelet_transform(x, "db2", 1, "LHL", "mirror"), 4e-7),
        "wavelet lll ri": (
            lambda x: wavelet_transform(x, "db2", 1, "LLL", "mirror", rotation_invariant=True),
            4e-7,
        ),
        "wavelet lhl ri": (
            lambda x: wavelet_transform(x, "db2", 1, "LHL", "mirror", rotation_invariant=True),
            4e-7,
        ),
    }
    float64_passes = [patch.object(m, "as_float32", ensure_float32) for m in (mean, gaussian, laws, log, wavelets)]  # fmt: skip
    for name, (case, tolerance) in cases.items():
        got = case(image)
        assert got.dtype == np.float32, name
        assert_array_equal(got, case(image.astype(np.float32)))
        for change in float64_passes:
            change.start()
        try:
            reference = np.asarray(case(image), dtype=np.float32)
        finally:
            for change in float64_passes:
                change.stop()
        difference = np.abs(got.astype(np.float64) - reference)
        assert 0 < difference.max() <= tolerance * np.abs(reference).max(), name


def test_wavelet_mixed_rotations_share_their_passes_from_a_size() -> None:
    """From _MIXED_TREE_MIN voxels on, a mixed decomposition takes the tree of shared passes
    too, with every pooling: LHL and HLH take 40 chains of level kernels instead of 72
    passes (24 filters of one rotation each). They equal the 24 rotated-copy responses
    pooled, within a few float32 units; the bits do not depend on the threads (max and min
    pool in chunks of planes in threads). Below the limit the 24 rotations run."""
    import pywt

    from pictologics.filters import base, wavelets

    lo = np.array(pywt.Wavelet("db2").dec_lo, dtype=np.float32)
    hi = np.array(pywt.Wavelet("db2").dec_hi, dtype=np.float32)
    image = np.random.default_rng(46).normal(size=(20, 18, 16))
    rotations = wavelets._get_rotation_perms()
    large = np.random.default_rng(47).normal(size=(12, 13, 14))  # slabs (the gates at 1)
    with patch.object(wavelets, "_rotation_tree", wraps=wavelets._rotation_tree) as tree:
        wavelet_transform(image, "db2", 1, "LHL", "mirror", rotation_invariant=True)
    assert tree.call_count == 0
    with patch.object(wavelets, "_MIXED_TREE_MIN", large.size):
        for decomposition in ("LHL", "HLH"):
            old = [_old_rotated_wavelet(image, lo, hi, 2, decomposition, "reflect", r) for r in rotations]  # fmt: skip
            for pooling in ("average", "max", "min"):
                with patch.object(wavelets, "_axis_chain", wraps=wavelets._axis_chain) as chains:
                    got = wavelet_transform(image, "db2", 2, decomposition, "mirror", True, pooling)
                assert chains.call_count == 40
                assert (
                    len({id(c.args[6]) for c in chains.call_args_list}) == 3
                )  # a buffer per depth
                expected = _old_pooled(old, pooling)
                assert_allclose(got, expected, rtol=1e-6, atol=1e-6 * np.max(np.abs(expected)))
                runs = []
                for threads in (1, 3):
                    with (
                        patch.object(wavelets, "get_num_threads", return_value=threads),
                        patch.object(base, "get_num_threads", return_value=threads),
                        _slabs_from_one_voxel(),
                    ):
                        runs.append(
                            wavelet_transform(
                                large,
                                "db2",
                                1,
                                decomposition,
                                "mirror",
                                True,
                                pooling,
                                use_parallel=True,
                            )  # fmt: skip
                        )
                assert runs[0].tobytes() == runs[1].tobytes()


def test_pool_reduce_in_chunks_of_planes() -> None:
    """The max and min pooling of the tree run in fixed chunks of planes, in one thread or in
    threads: the values of one numpy call, also for -0.0 and +0.0 inside a chunk."""
    from pictologics.filters import wavelets

    rng = np.random.default_rng(31)
    total = rng.normal(size=(7, 3, 4)).astype(np.float32)
    response = rng.normal(size=total.shape).astype(np.float32)
    total[0, 0, :2], response[0, 0, :2] = (0.0, -0.0), (-0.0, 0.0)
    for reduce in (np.maximum, np.minimum):
        expected = reduce(total, response)
        for threads in (1, 3):
            got = total.copy()
            with (
                patch.object(wavelets, "_SLAB_MIN_SIZE", 30),  # 2 planes in a chunk: 4 chunks
                patch.object(wavelets, "get_num_threads", return_value=threads),
            ):
                wavelets._pool_reduce(reduce, got, response)
            assert got.tobytes() == expected.tobytes()


def test_the_fft_laplace_on_slabs_gives_the_bits_of_the_rfftn_path() -> None:
    """From _LOG_SLAB_MIN padded voxels on, the FFT LoG runs on slabs (no padded array, the
    inverse of the cropped planes only), with the bits of the rfftn path (from SciPy 1.18
    within a few units, see _assert_gabor_bits): five shapes (one plane on an axis too),
    the five modes, scalar, per-axis and long sigmas, truncate 4 and 6, float32 and float64
    outputs as new row-order arrays; chunks of one plane and one chunk per slab, serial and
    in threads (the threaded transfer too)."""
    from pictologics.filters import base

    rng = np.random.default_rng(23)
    sigmas = (1.2, (1.5, 1.0, 2.0), 3.0)
    case = 0
    for shape in ((12, 9, 10), (1, 9, 10), (9, 1, 10), (9, 10, 1), (7, 16, 5)):
        image = rng.normal(0.0, 10.0, shape)
        for mode in ("reflect", "mirror", "nearest", "wrap", "constant"):
            sigma, truncate = sigmas[case % 3], (4.0, 6.0)[case % 2]
            dtype = (np.float32, np.float64)[(case // 2) % 2]
            case += 1
            for chunk, threads in ((1, 1), (1 << 40, 3)):
                with (
                    patch.object(base, "_LOG_FFT_MIN", 1),
                    patch.object(base, "get_num_threads", return_value=threads),
                    patch.object(base, "_SLAB_MIN_SIZE", 1),
                    patch.object(base, "_FFT_CHUNK_BYTES", chunk),
                ):
                    with patch.object(base, "_LOG_SLAB_MIN", 1 << 60):
                        expected = base._gaussian_laplace(image, sigma, mode, truncate, dtype)
                    with patch.object(base, "_LOG_SLAB_MIN", 1):
                        got = base._gaussian_laplace(image, sigma, mode, truncate, dtype)
                assert got.dtype == expected.dtype == dtype and got.flags.c_contiguous
                _assert_gabor_bits(got, expected, (shape, mode, chunk, threads))


def test_the_fft_laplace_of_a_column_order_image_takes_the_rfftn_path() -> None:
    """A column-order float64 image takes the rfftn path at every size (no copy), with the
    bits of the row-order image; the plain LoG asks for a float32 output, the masked one
    keeps float64."""
    from pictologics.filters import base, laplacian_of_gaussian

    image = np.random.default_rng(24).normal(0.0, 10.0, (12, 9, 10))
    with patch.object(base, "_LOG_FFT_MIN", 1), patch.object(base, "_LOG_SLAB_MIN", 1):
        with patch.object(base, "_scale_cut", wraps=base._scale_cut) as cut:
            column = base._gaussian_laplace(np.asfortranarray(image), 1.5, "mirror")
        assert cut.call_count == 0
        _assert_gabor_bits(column, base._gaussian_laplace(image, 1.5, "mirror"))
        fft = base._fft_gaussian_laplace
        with (
            patch.object(base, "_fft_gaussian_laplace", wraps=fft) as spy,
            patch("pictologics.filters.log._LOG_FFT_MIN", 1),
        ):
            laplacian_of_gaussian(image, 1.5)
            laplacian_of_gaussian(image, 1.5, source_mask=np.ones(image.shape, dtype=bool))
    assert [call.args[4] for call in spy.call_args_list] == [np.float32, np.float64]


def test_the_pad_planes_from_a_first_plane() -> None:
    """The pad kernel writes the planes of np.pad from a given first plane into a smaller
    output (a slab of the padded image)."""
    from pictologics.filters import base

    image = np.random.default_rng(25).normal(size=(5, 7, 3))
    widths = ((2, 3), (1, 4), (0, 5))
    for mode in ("constant", "symmetric", "reflect", "edge", "wrap"):
        extra = {"constant_values": 1.5} if mode == "constant" else {}
        expected = np.pad(image, widths, mode=mode, **extra)[1:4]
        out = np.empty(expected.shape)
        base._pad_planes_numba(image, out, 2, 1, 0, base._PAD_CODES[mode], 1.5, 1)
        assert_array_equal(out, expected)


def test_simoncelli_of_a_single_slice_is_its_2d_response() -> None:
    """A singleton axis has no frequency: the Simoncelli wavelet (and the Riesz-Simoncelli
    filter) of a volume of one slice is the 2-D response of the slice, not zeros."""
    image = np.random.default_rng(56).normal(0.0, 1.0, (24, 24))
    flat = simoncelli_wavelet(image, 1)
    assert np.abs(flat).max() > 0
    for volume, cut in (
        (image[None], (0, slice(None), slice(None))),
        (image[:, None], (slice(None), 0, slice(None))),
        (image[:, :, None], (slice(None), slice(None), 0)),
    ):
        assert_array_equal(simoncelli_wavelet(volume, 1)[cut], flat)
    riesz = riesz_simoncelli(image[None], 1, (0, 1, 0))
    assert np.all(np.isfinite(riesz))
    assert_array_equal(riesz[0], riesz_simoncelli(image, 1, (1, 0)))


def test_filters_reject_bad_parameters() -> None:
    """The Gabor filter and the LoG raise a ValueError that names the bad parameter: a
    sigma, wavelength, aspect ratio, spacing or truncation that is not a finite positive
    number, and a rotation step outside (0, 2 pi]. A good call is unchanged."""
    image = np.random.default_rng(57).normal(0.0, 1.0, (6, 7, 5))
    good = {"sigma_mm": 2.0, "lambda_mm": 4.0}
    cases = [
        ({"sigma_mm": 0.0}, "sigma_mm"), ({"sigma_mm": -2.0}, "sigma_mm"),
        ({"sigma_mm": np.nan}, "sigma_mm"), ({"sigma_mm": np.inf}, "sigma_mm"),
        ({"lambda_mm": 0.0}, "lambda_mm"), ({"lambda_mm": -2.0}, "lambda_mm"),
        ({"gamma": 0.0}, "gamma"), ({"gamma": -1.0}, "gamma"),
        ({"spacing_mm": 0.0}, "spacing_mm"), ({"spacing_mm": -1.0}, "spacing_mm"),
        ({"spacing_mm": (1.0, 1.0, 0.0)}, "spacing_mm"),
        ({"rotation_invariant": True, "delta_theta": 0.0}, "delta_theta"),
        ({"rotation_invariant": True, "delta_theta": -np.pi / 4}, "delta_theta"),
        ({"rotation_invariant": True, "delta_theta": 10.0}, "delta_theta"),
    ]  # fmt: skip
    for change, name in cases:
        with pytest.raises(ValueError, match=name):
            gabor_filter(image, **{**good, **change})
    for change, name in (
        ({"sigma_mm": 0.0}, "sigma_mm"), ({"sigma_mm": np.nan}, "sigma_mm"),
        ({"spacing_mm": -1.0}, "spacing_mm"), ({"spacing_mm": (1.0, 0.0, 1.0)}, "spacing_mm"),
        ({"truncate": 0.0}, "truncate"),
    ):  # fmt: skip
        with pytest.raises(ValueError, match=name):
            laplacian_of_gaussian(image, **{"sigma_mm": 2.0, **change})
    assert gabor_filter(image, **good).shape == image.shape
    assert laplacian_of_gaussian(image, 2.0).shape == image.shape


def test_gabor_returns_a_row_order_array() -> None:
    """The Gabor response is a row-order array on every plane (the default axial plane
    filters along the last axis), also with a region and rotation invariance, with the
    values of the moved view."""
    from pictologics.filters import gabor as gabor_module

    image = np.random.default_rng(58).normal(0.0, 1.0, (6, 7, 5))
    region = (slice(1, 5), slice(2, 6), slice(0, 4))
    moves = []
    move = np.moveaxis

    def spy(*args: Any, **kwargs: Any) -> np.ndarray:
        moves.append(move(*args, **kwargs))
        return moves[-1]

    for kwargs in (
        {},
        {"region": region},
        {"rotation_invariant": True, "delta_theta": np.pi / 2},
    ):
        moves.clear()
        with patch.object(gabor_module.np, "moveaxis", side_effect=spy):
            out = gabor_filter(image, 2.0, 4.0, **kwargs)
        assert out.flags.c_contiguous
        assert out.tobytes() == np.ascontiguousarray(moves[-1]).tobytes()


def test_cache_by_bytes_keeps_the_newest_shapes() -> None:
    """With max_shapes, a new result drops the results of the first arguments (the image
    shapes) that are not among the newest ones by their newest use: a cohort of images
    keeps the tables of two shapes."""
    from pictologics.filters.base import cache_by_bytes

    calls = []

    @cache_by_bytes(1 << 30, 2)
    def table(shape: tuple[int, ...], level: int) -> np.ndarray:
        calls.append((shape, level))
        return np.zeros(shape)

    for request in (((4,), 1), ((4,), 2), ((5,), 1), ((4,), 1), ((6,), 1), ((5,), 1), ((4,), 1)):
        table(*request)
    assert calls == [((4,), 1), ((4,), 2), ((5,), 1), ((6,), 1), ((5,), 1), ((4,), 1)]


def test_the_log_takes_the_fft_from_131072_voxels() -> None:
    """A float64 image of _LOG_FFT_MIN = 131,072 voxels (64 x 64 x 32) takes the FFT
    convolution in float64; one voxel fewer runs the passes on a float32 copy."""
    from pictologics.filters import base
    from pictologics.filters import log as log_module

    assert base._LOG_FFT_MIN == 1 << 17
    at_gate, below = np.zeros((64, 64, 32)), np.zeros((131_071, 1, 1))
    sentinel = np.zeros(1, dtype=np.float32)
    with patch.object(log_module, "_gaussian_laplace", return_value=sentinel) as laplace:
        laplacian_of_gaussian(at_gate, 2.0)
        laplacian_of_gaussian(below, 2.0)
    assert [call.args[0].dtype for call in laplace.call_args_list] == [np.float64, np.float32]
    with patch.object(base, "_fft_gaussian_laplace", return_value=sentinel) as fft:
        assert base._gaussian_laplace(at_gate, 2.0, "mirror") is sentinel
    fft.assert_called_once()
