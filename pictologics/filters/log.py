# pictologics/filters/log.py
"""Laplacian of Gaussian filter implementation (IBSI code: L6PA)."""

from typing import Any, Optional, Tuple, Union, overload

import numpy as np
from numpy import typing as npt

from .base import (
    _LOG_FFT_MIN,
    BoundaryCondition,
    _constant_padded,
    _float32_cut,
    _gaussian_laplace,
    _normalized_gaussian_laplace,
    _padding_value_problem,
    as_float32,
    ensure_float32,
    get_scipy_mode,
    resolve_boundary,
)


@overload
def laplacian_of_gaussian(
    image: npt.NDArray[np.floating[Any]],
    sigma_mm: float,
    spacing_mm: Union[float, Tuple[float, float, float]] = ...,
    truncate: float = ...,
    boundary: Union[BoundaryCondition, str] = ...,
    source_mask: None = ...,
    padding_value: float = ...,
) -> npt.NDArray[np.floating[Any]]: ...


@overload
def laplacian_of_gaussian(
    image: npt.NDArray[np.floating[Any]],
    sigma_mm: float,
    spacing_mm: Union[float, Tuple[float, float, float]] = ...,
    truncate: float = ...,
    boundary: Union[BoundaryCondition, str] = ...,
    source_mask: npt.NDArray[np.bool_] = ...,
    padding_value: float = ...,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]: ...


def laplacian_of_gaussian(
    image: npt.NDArray[np.floating[Any]],
    sigma_mm: float,
    spacing_mm: Union[float, Tuple[float, float, float]] = 1.0,
    truncate: float = 4.0,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.ZERO,
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
    padding_value: float = 0.0,
) -> Union[
    npt.NDArray[np.floating[Any]],
    tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]],
]:
    """
    Apply 3D Laplacian of Gaussian filter (IBSI code: L6PA).

    The LoG is a band-pass, spherically symmetric operator. Per IBSI 2 Eq. 3.

    Args:
        image: 3D input image array
        sigma_mm: Standard deviation in mm (σ*, 41LN)
        spacing_mm: Voxel spacing in mm (scalar for isotropic, or tuple)
        truncate: Filter size cutoff in σ units (default 4.0, WGPM)
        boundary: Boundary condition for padding (GBYQ)
        source_mask: Optional boolean mask where True = valid voxel.
            When provided, uses normalized convolution to exclude invalid
            (sentinel) voxels from computation.
        padding_value: The constant of constant value padding (Z3VE), with the ZERO
            (constant) boundary. Default 0.

    Returns:
        If source_mask is None: Response map with same dimensions as input
        If source_mask provided: Tuple of (response_map, output_valid_mask)

    Example:
        Apply LoG filter with 5.0mm sigma on an image with 2.0mm spacing:

        ```python
        import numpy as np
        from pictologics.filters import laplacian_of_gaussian

        # Create dummy 3D image
        image = np.random.rand(50, 50, 50)

        # Apply filter (original API)
        response = laplacian_of_gaussian(
            image,
            sigma_mm=5.0,
            spacing_mm=(2.0, 2.0, 2.0),
            truncate=4.0
        )

        # With source_mask for sentinel exclusion
        mask = image > -1000
        response, valid_mask = laplacian_of_gaussian(
            image, sigma_mm=5.0, spacing_mm=2.0, source_mask=mask
        )
        ```

    Note:
        - σ is converted from mm to voxels: σ_voxels = σ_mm / spacing_mm
        - Filter size: M = 1 + 2⌊d×σ + 0.5⌋ where d=truncate
        - The kernel should sum to approximately 0 (zero-mean)
        - A float64 image of 64³ voxels or more runs as one FFT convolution with the same
          truncated kernels and boundary: about 2 times faster than the separable passes,
          with values within about 1e-15 of the largest response value.
        - A smaller float64 image runs the passes in float32, as a float32 image does:
          faster, with values within about 1e-7 of the largest response value.
    """
    # Convert to float32 as required by IBSI
    image = ensure_float32(image)

    # Handle scalar spacing
    if isinstance(spacing_mm, (int, float)):
        spacing_mm = (float(spacing_mm),) * 3

    # Convert sigma from mm to voxels for each axis
    sigma_voxels = tuple(sigma_mm / s for s in spacing_mm)

    # Handle string boundary condition
    boundary = resolve_boundary(boundary)

    problem = _padding_value_problem(boundary, padding_value)
    if problem:
        raise ValueError(problem)
    if padding_value:
        reach = tuple(int(truncate * s + 0.5) for s in sigma_voxels)
        return _constant_padded(
            laplacian_of_gaussian, image, source_mask, padding_value, reach,
            sigma_mm=sigma_mm, spacing_mm=spacing_mm, truncate=truncate, boundary=boundary,
        )  # fmt: skip
    mode = get_scipy_mode(boundary)

    if source_mask is not None:
        # Use normalized convolution for source masking
        return _normalized_gaussian_laplace(
            image, source_mask, sigma=sigma_voxels, mode=mode, truncate=truncate
        )
    else:
        # A float64 image below _LOG_FFT_MIN voxels runs its 8 passes in float32 (faster;
        # the FFT path of a larger float64 image keeps float64). The response is float32.
        if image.dtype == np.float64 and image.size < _LOG_FFT_MIN:
            image = as_float32(image)
        result = _gaussian_laplace(image, sigma_voxels, mode, truncate)
        if result.dtype == np.float32:
            return result
        return _float32_cut(result, None)
