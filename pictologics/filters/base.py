# pictologics/filters/base.py
"""Base classes and utilities for IBSI 2 filter implementations."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple, Union

import numpy as np
from numpy import typing as npt


class BoundaryCondition(Enum):
    """
    IBSI 2 boundary conditions for image padding (GBYQ).

    Maps to scipy.ndimage mode parameter values.
    """

    ZERO = "constant"  # Zero padding (Z3VE)
    NEAREST = "nearest"  # Nearest value padding (SIJG)
    PERIODIC = "wrap"  # Periodic/wrap padding (Z7YO)
    MIRROR = "reflect"  # Mirror/symmetric padding (ZDTV)


@dataclass
class FilterResult:
    """Container for filter response maps and metadata."""

    response_map: npt.NDArray[np.floating[Any]]
    filter_name: str
    filter_params: Dict[str, Any]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the response map."""
        return self.response_map.shape  # type: ignore[no-any-return]

    @property
    def dtype(self) -> np.dtype[Any]:
        """Data type of the response map."""
        return self.response_map.dtype  # type: ignore[no-any-return]


def ensure_float32(
    image: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Ensure image is at least 32-bit floating point precision.

    Per IBSI 2: "The phantom data need to be converted from an integer
    data type to at least 32 bit floating point precision, prior to filtering."

    Args:
        image: Input image array

    Returns:
        Image as float32 (or higher precision if already float64)
    """
    if np.issubdtype(image.dtype, np.floating):
        return image.astype(np.float32) if image.dtype == np.float16 else image
    return image.astype(np.float32)


def get_scipy_mode(boundary: BoundaryCondition) -> str:
    """Convert BoundaryCondition to scipy.ndimage mode string."""
    return boundary.value


# ==============================================================================
# Normalized Convolution Utilities for Source Mask Support
# ==============================================================================


def _normalized_uniform_filter(
    image: npt.NDArray[np.floating[Any]],
    source_mask: npt.NDArray[np.bool_],
    size: int,
    mode: str = "constant",
    weight_threshold: float = 0.01,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]:
    """
    Apply uniform (mean) filter with normalized convolution for source masking.

    Excludes invalid (sentinel) voxels from the mean computation by using
    normalized convolution: output = sum(valid_values) / sum(valid_weights).

    Args:
        image: 3D input image array
        source_mask: Boolean mask where True = valid voxel
        size: Filter support (kernel size)
        mode: Boundary mode for scipy.ndimage
        weight_threshold: Minimum weight to consider output valid.
            Default 0.01 means at least 1% contribution from valid voxels.

    Returns:
        Tuple of (filtered_image, output_valid_mask)
    """
    from scipy.ndimage import uniform_filter

    # Zero out invalid voxels
    valid_image = np.where(source_mask, image, 0.0).astype(np.float64)

    # Compute weighted sum (like regular uniform_filter, but invalid=0)
    weighted_sum = uniform_filter(valid_image, size=size, mode=mode)

    # Compute weight sum (how much "valid" contribution at each point)
    weight_sum = uniform_filter(source_mask.astype(np.float64), size=size, mode=mode)

    # Normalize (avoid division by zero)
    valid_output = weight_sum >= weight_threshold
    result = np.zeros_like(weighted_sum)
    result[valid_output] = weighted_sum[valid_output] / weight_sum[valid_output]

    return result.astype(np.float32), valid_output


def _normalized_gaussian_laplace(
    image: npt.NDArray[np.floating[Any]],
    source_mask: npt.NDArray[np.bool_],
    sigma: Union[float, Tuple[float, ...]],
    mode: str = "constant",
    truncate: float = 4.0,
    weight_threshold: float = 0.01,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]:
    """
    Apply Laplacian of Gaussian with normalized convolution for source masking.

    Note: LoG with normalized convolution is approximated by computing LoG on
    the valid-zeroed image and normalizing by the Gaussian-smoothed mask weights.
    This is an approximation since LoG is a second derivative.

    Args:
        image: 3D input image array
        source_mask: Boolean mask where True = valid voxel
        sigma: Standard deviation in voxels (scalar or per-axis tuple)
        mode: Boundary mode for scipy.ndimage
        truncate: Filter size cutoff in sigma units
        weight_threshold: Minimum weight to consider output valid

    Returns:
        Tuple of (filtered_image, output_valid_mask)
    """
    from scipy.ndimage import gaussian_filter, gaussian_laplace

    # Zero out invalid voxels
    valid_image = np.where(source_mask, image, 0.0).astype(np.float64)

    # Apply LoG to zeroed image
    log_response = gaussian_laplace(
        valid_image, sigma=sigma, mode=mode, truncate=truncate
    )

    # Compute weight sum using Gaussian (not LoG, since LoG sums to 0)
    weight_sum = gaussian_filter(
        source_mask.astype(np.float64), sigma=sigma, mode=mode, truncate=truncate
    )

    # Normalize
    valid_output = weight_sum >= weight_threshold
    result = np.zeros_like(log_response)
    result[valid_output] = log_response[valid_output] / weight_sum[valid_output]

    return result.astype(np.float32), valid_output


def _normalized_convolve1d(
    image: npt.NDArray[np.floating[Any]],
    source_mask: npt.NDArray[np.bool_],
    kernel: npt.NDArray[np.floating[Any]],
    axis: int,
    mode: str = "constant",
    weight_threshold: float = 0.01,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]:
    """
    Apply 1D convolution with normalized convolution for source masking.

    Args:
        image: 3D input image array
        source_mask: Boolean mask where True = valid voxel
        kernel: 1D convolution kernel
        axis: Axis along which to convolve
        mode: Boundary mode for scipy.ndimage
        weight_threshold: Minimum weight to consider output valid

    Returns:
        Tuple of (filtered_image, output_valid_mask)
    """
    from scipy.ndimage import convolve1d

    # Zero out invalid voxels
    valid_image = np.where(source_mask, image, 0.0).astype(np.float64)

    # Apply convolution to zeroed image
    response = convolve1d(valid_image, kernel, axis=axis, mode=mode)

    # Compute weight sum using absolute kernel (for proper normalization)
    abs_kernel = np.abs(kernel)
    weight_sum = convolve1d(
        source_mask.astype(np.float64), abs_kernel, axis=axis, mode=mode
    )

    # Normalize
    valid_output = weight_sum >= weight_threshold
    result = np.zeros_like(response)
    result[valid_output] = response[valid_output] / weight_sum[valid_output]

    return result.astype(np.float32), valid_output


def _normalized_separable_convolve_3d(
    image: npt.NDArray[np.floating[Any]],
    source_mask: npt.NDArray[np.bool_],
    g1: npt.NDArray[np.floating[Any]],
    g2: npt.NDArray[np.floating[Any]],
    g3: npt.NDArray[np.floating[Any]],
    mode: str = "constant",
    weight_threshold: float = 0.01,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]:
    """
    Apply separable 3D convolution with normalized convolution for source masking.

    Uses three 1D kernels applied sequentially along each axis.

    Args:
        image: 3D input image array
        source_mask: Boolean mask where True = valid voxel
        g1, g2, g3: 1D kernels for axes 0, 1, 2
        mode: Boundary mode for scipy.ndimage
        weight_threshold: Minimum weight to consider output valid

    Returns:
        Tuple of (filtered_image, output_valid_mask)
    """
    from scipy.ndimage import convolve1d

    # Zero out invalid voxels
    valid_image = np.where(source_mask, image, 0.0).astype(np.float64)

    # Apply separable convolution to zeroed image
    result = convolve1d(valid_image, g1, axis=0, mode=mode)
    result = convolve1d(result, g2, axis=1, mode=mode)
    result = convolve1d(result, g3, axis=2, mode=mode)

    # Compute weight product using absolute kernels
    abs_g1, abs_g2, abs_g3 = np.abs(g1), np.abs(g2), np.abs(g3)
    weight = convolve1d(source_mask.astype(np.float64), abs_g1, axis=0, mode=mode)
    weight = convolve1d(weight, abs_g2, axis=1, mode=mode)
    weight = convolve1d(weight, abs_g3, axis=2, mode=mode)

    # Normalize
    valid_output = weight >= weight_threshold
    normalized = np.zeros_like(result)
    normalized[valid_output] = result[valid_output] / weight[valid_output]

    return normalized.astype(np.float32), valid_output


def _prepare_masked_image(
    image: npt.NDArray[np.floating[Any]],
    source_mask: npt.NDArray[np.bool_],
    fill_value: float = 0.0,
) -> npt.NDArray[np.floating[Any]]:
    """
    Prepare image for FFT-based filtering by zeroing out invalid voxels.

    For FFT-based filters (Simoncelli, Riesz), we cannot use normalized
    convolution directly. Instead, we zero out invalid voxels which acts
    as a first-order approximation.

    Args:
        image: 3D input image array
        source_mask: Boolean mask where True = valid voxel
        fill_value: Value to use for invalid voxels (default 0.0)

    Returns:
        Image with invalid voxels set to fill_value
    """
    result = image.copy()
    result[~source_mask] = fill_value
    return result
