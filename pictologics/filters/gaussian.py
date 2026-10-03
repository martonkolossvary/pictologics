# pictologics/filters/gaussian.py
"""Gaussian filter implementation (IBSI code: 8BC3)."""

from typing import Any, Optional, Tuple, Union, overload

import numpy as np
from numpy import typing as npt

from .base import (
    BoundaryCondition,
    _constant_padded,
    _gaussian_filter,
    _normalized_gaussian,
    _padding_value_problem,
    ensure_float32,
    get_scipy_mode,
    resolve_boundary,
)


@overload
def gaussian_filter(
    image: npt.NDArray[np.floating[Any]],
    sigma_mm: float,
    spacing_mm: Union[float, Tuple[float, float, float]] = ...,
    truncate: float = ...,
    boundary: Union[BoundaryCondition, str] = ...,
    source_mask: None = ...,
    padding_value: float = ...,
) -> npt.NDArray[np.floating[Any]]: ...


@overload
def gaussian_filter(
    image: npt.NDArray[np.floating[Any]],
    sigma_mm: float,
    spacing_mm: Union[float, Tuple[float, float, float]] = ...,
    truncate: float = ...,
    boundary: Union[BoundaryCondition, str] = ...,
    source_mask: npt.NDArray[np.bool_] = ...,
    padding_value: float = ...,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]: ...


def gaussian_filter(
    image: npt.NDArray[np.floating[Any]],
    sigma_mm: float,
    spacing_mm: Union[float, Tuple[float, float, float]] = 1.0,
    truncate: float = 4.0,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.ZERO,
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
    padding_value: float = 0.0,
) -> Union[
    npt.NDArray[np.floating[Any]],
    Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]],
]:
    """
    Apply a 3D Gaussian filter (IBSI code: 8BC3): a smoothing low-pass filter.

    Args:
        image: 3D input image array
        sigma_mm: Scale σ* in mm (41LN)
        spacing_mm: Voxel spacing in mm (scalar or a value for each axis)
        truncate: Filter size cutoff in σ units (default 4.0, WGPM)
        boundary: Boundary condition for padding (GBYQ)
        source_mask: Optional boolean mask where True = valid voxel. When provided,
            normalized convolution leaves out the invalid voxels.
        padding_value: The constant of constant value padding (Z3VE), with the ZERO
            (constant) boundary. Default 0.

    Returns:
        If source_mask is None: Response map (float32) with the dimensions of the input.
        If source_mask is given: (response map, output valid mask).

    Raises:
        ValueError: If `padding_value` is not 0 with another boundary than ZERO.

    Example:
        ```python
        import numpy as np
        from pictologics.filters import gaussian_filter

        image = np.random.rand(50, 50, 50)
        smoothed = gaussian_filter(image, sigma_mm=2.0, spacing_mm=(0.8, 0.8, 2.0))
        ```

    Note:
        - σ is converted from mm to voxels: σ_voxels = σ_mm / spacing_mm
        - The kernel of each axis has the radius ⌊truncate × σ_voxels + 0.5⌋ and sums to 1
    """
    image = ensure_float32(image)
    if isinstance(spacing_mm, (int, float)):
        spacing_mm = (float(spacing_mm),) * 3
    sigma_voxels = tuple(sigma_mm / s for s in spacing_mm)
    boundary = resolve_boundary(boundary)
    problem = _padding_value_problem(boundary, padding_value)
    if problem:
        raise ValueError(problem)
    if padding_value:
        reach = tuple(int(truncate * s + 0.5) for s in sigma_voxels)
        return _constant_padded(
            gaussian_filter, image, source_mask, padding_value, reach,
            sigma_mm=sigma_mm, spacing_mm=spacing_mm, truncate=truncate, boundary=boundary,
        )  # fmt: skip
    mode = get_scipy_mode(boundary)
    if source_mask is not None:
        return _normalized_gaussian(image, source_mask, sigma_voxels, mode, truncate)
    # The passes run in the type of the image (a float64 image keeps its precision),
    # then the response is float32 as for the other filters
    return _gaussian_filter(image, sigma_voxels, mode, truncate).astype(np.float32)
