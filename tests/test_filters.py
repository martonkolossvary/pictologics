"""Tests for pictologics.filters module."""

import numpy as np
import pytest

from pictologics.filters import (
    BoundaryCondition,
    FilterResult,
    gabor_filter,
    laplacian_of_gaussian,
    laws_filter,
    LAWS_KERNELS,
    mean_filter,
    riesz_log,
    riesz_simoncelli,
    riesz_transform,
    simoncelli_wavelet,
    wavelet_transform,
)
from pictologics.filters.base import ensure_float32, get_scipy_mode


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


@pytest.fixture
def checkerboard_3d():
    """Small 3D checkerboard pattern."""
    img = np.zeros((8, 8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            for k in range(8):
                if (i + j + k) % 2 == 0:
                    img[i, j, k] = 1.0
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
        result = FilterResult(
            response_map=arr, filter_name="test", filter_params={"size": 3}
        )
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
        result = laplacian_of_gaussian(
            small_3d_image, sigma_mm=2.0, spacing_mm=(1.0, 1.0, 1.0)
        )
        assert result.shape == small_3d_image.shape

    def test_different_truncation(self, small_3d_image):
        result = laplacian_of_gaussian(
            small_3d_image, sigma_mm=2.0, spacing_mm=1.0, truncate=3.0
        )
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
            result = laws_filter(
                small_3d_image, "E5L5S5", rotation_invariant=True, pooling=pooling
            )
            assert result.shape == small_3d_image.shape

    def test_compute_energy(self, small_3d_image):
        result = laws_filter(
            small_3d_image, "E5L5S5", compute_energy=True, energy_distance=3
        )
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

    def test_invalid_pooling(self, small_3d_image):
        with pytest.raises(ValueError, match="Unknown pooling"):
            laws_filter(
                small_3d_image, "L5L5L5", rotation_invariant=True, pooling="invalid"
            )


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
        result = gabor_filter(
            small_3d_image, sigma_mm=5.0, lambda_mm=2.0, gamma=0.5, spacing_mm=2
        )
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
            result = wavelet_transform(
                small_3d_image, wavelet="db2", level=1, decomposition=decomp
            )
            assert result.shape == small_3d_image.shape

    def test_rotation_invariant(self, small_3d_image):
        result = wavelet_transform(
            small_3d_image, wavelet="db2", level=1, rotation_invariant=True
        )
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
            result = wavelet_transform(
                small_3d_image, wavelet="db2", level=1, boundary=boundary
            )
            assert result.shape == small_3d_image.shape

    def test_string_boundary_condition(self, small_3d_image):
        result = wavelet_transform(
            small_3d_image, wavelet="db2", level=1, boundary="mirror"
        )
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


class TestSimoncelliWavelet:
    """Tests for simoncelli_wavelet function."""

    def test_basic_application(self, small_3d_image):
        result = simoncelli_wavelet(small_3d_image, level=1)
        assert result.shape == small_3d_image.shape

    def test_different_levels(self, small_3d_image):
        for level in [1, 2, 3]:
            result = simoncelli_wavelet(small_3d_image, level=level)
            assert result.shape == small_3d_image.shape


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


class TestRieszLog:
    """Tests for riesz_log function."""

    def test_basic_application(self, small_3d_image):
        result = riesz_log(small_3d_image, sigma_mm=2.0, order=(1, 0, 0))
        assert result.shape == small_3d_image.shape

    def test_with_spacing(self, small_3d_image):
        result = riesz_log(
            small_3d_image, sigma_mm=2.0, order=(1, 0, 0), spacing_mm=2.0
        )
        assert result.shape == small_3d_image.shape

    def test_different_orders(self, small_3d_image):
        orders = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
        for order in orders:
            result = riesz_log(small_3d_image, sigma_mm=2.0, order=order)
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
