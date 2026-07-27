"""Tests for pictologics.filters module."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pictologics.filters import (
    LAWS_KERNELS,
    BoundaryCondition,
    FilterResult,
    gabor_filter,
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

        def func(arr, add=0.0):
            calls.append(arr)
            return arr + add

        result = _apply_with_boundary_padding(func, image, BoundaryCondition.PERIODIC, 5, add=3.0)
        assert calls[0] is image
        assert_array_equal(result, image + 3.0)

    def test_non_periodic_crops_back_to_original_shape(self):
        image = np.ones((4, 4, 4), dtype=np.float32)

        def func(arr):
            assert arr.shape == (8, 8, 8)  # 4 + 2*2 padding
            return arr

        result = _apply_with_boundary_padding(func, image, BoundaryCondition.ZERO, 2)
        assert_array_equal(result, image)

    def test_pad_width_as_tuple(self):
        image = np.ones((4, 6, 8), dtype=np.float32)

        def func(arr):
            assert arr.shape == (6, 8, 10)
            return arr

        result = _apply_with_boundary_padding(func, image, BoundaryCondition.NEAREST, (1, 1, 1))
        assert result.shape == image.shape

    def test_pad_width_capped_to_axis_length(self):
        image = np.ones((3, 3, 3), dtype=np.float32)

        def func(arr):
            # Requested pad of 100 is capped to the axis length (3), so the
            # padded shape is 3 + 2*3 = 9 per axis, not 3 + 2*100.
            assert arr.shape == (9, 9, 9)
            return arr

        result = _apply_with_boundary_padding(func, image, BoundaryCondition.MIRROR, 100)
        assert result.shape == image.shape

    def test_zero_padding_fills_with_zero(self):
        image = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)  # shape (1, 1, 4)
        captured = {}

        def func(arr):
            captured["arr"] = arr.copy()
            return arr

        result = _apply_with_boundary_padding(func, image, BoundaryCondition.ZERO, (0, 0, 2))
        assert_array_equal(
            captured["arr"][0, 0], np.array([0, 0, 1, 2, 3, 4, 0, 0], dtype=np.float32)
        )
        assert_array_equal(result, image)

    def test_nearest_padding_replicates_edge(self):
        image = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
        captured = {}

        def func(arr):
            captured["arr"] = arr.copy()
            return arr

        _apply_with_boundary_padding(func, image, BoundaryCondition.NEAREST, (0, 0, 2))
        assert_array_equal(
            captured["arr"][0, 0], np.array([1, 1, 1, 2, 3, 4, 4, 4], dtype=np.float32)
        )

    def test_mirror_padding_reflects(self):
        image = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
        captured = {}

        def func(arr):
            captured["arr"] = arr.copy()
            return arr

        _apply_with_boundary_padding(func, image, BoundaryCondition.MIRROR, (0, 0, 2))
        assert_array_equal(
            captured["arr"][0, 0], np.array([2, 1, 1, 2, 3, 4, 4, 3], dtype=np.float32)
        )


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

    def test_normalized_convolve1d(self):
        # data window [10,0,30]·[0.5,1,0.5]=20, weight window [1,0,1]·[0.5,1,0.5]=1 -> 20.
        image = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        mask = np.array([True, False, True], dtype=bool)
        kernel = np.array([0.5, 1.0, 0.5], dtype=np.float32)

        result, _ = _normalized_convolve1d(image, mask, kernel, axis=0, mode="constant")
        assert np.isclose(result[1], 20.0)

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
                small_3d_image, "E3L3S3", rotation_invariant=True, pooling=pooling, use_parallel=True
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
        with an independent 6*sigma_mm/s_i radius per axis, evaluated on a
        physical (mm) coordinate grid. Recompute the expected kernel from the
        Gabor formula directly (not by calling the function under test) to
        catch implementation bugs rather than just echoing them back."""
        sigma_mm, lambda_mm, gamma, theta = 5.0, 2.0, 0.5, np.pi / 6
        s1, s2 = 1.0, 3.0

        kernel = _create_gabor_kernel_2d_anisotropic(sigma_mm, lambda_mm, gamma, theta, s1, s2)

        radius1 = int(np.ceil(6.0 * sigma_mm / s1))
        radius2 = int(np.ceil(6.0 * sigma_mm / s2))
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

    def test_average_over_planes_anisotropic_spacing_is_now_handled_correctly(
        self, small_3d_image
    ):
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
        """Exercise the use_parallel=True branch across pooling modes."""
        for pooling in ["max", "average", "min"]:
            result = wavelet_transform(
                small_3d_image,
                wavelet="db2",
                level=1,
                decomposition="LHL",
                rotation_invariant=True,
                pooling=pooling,
                use_parallel=True,
            )
            assert result.shape == small_3d_image.shape


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
