# pictologics/filters/mean.py
"""Mean filter implementation (IBSI code: S60F)."""

from typing import Any, Optional, Union, overload

import numpy as np
from numpy import typing as npt
from scipy.ndimage import uniform_filter

from .base import (
    BoundaryCondition,
    _normalized_uniform_filter,
    ensure_float32,
    get_scipy_mode,
)


@overload
def mean_filter(
    image: npt.NDArray[np.floating[Any]],
    support: int = ...,
    boundary: Union[BoundaryCondition, str] = ...,
    source_mask: None = ...,
) -> npt.NDArray[np.floating[Any]]: ...


@overload
def mean_filter(
    image: npt.NDArray[np.floating[Any]],
    support: int = ...,
    boundary: Union[BoundaryCondition, str] = ...,
    source_mask: npt.NDArray[np.bool_] = ...,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]: ...


def mean_filter(
    image: npt.NDArray[np.floating[Any]],
    support: int = 15,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.ZERO,
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> Union[
    npt.NDArray[np.floating[Any]],
    tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]],
]:
    """
    Apply 3D mean filter (IBSI code: S60F).

    The mean filter computes the average intensity over an M×M×M
    spatial support. Per IBSI 2 Eq. 2.

    Args:
        image: 3D input image array
        support: Filter support M in voxels (must be odd, YNOF)
        boundary: Boundary condition for padding (GBYQ)
        source_mask: Optional boolean mask where True = valid voxel.
            When provided, uses normalized convolution to exclude invalid
            (sentinel) voxels from mean computation.

    Returns:
        If source_mask is None: Response map with same dimensions as input
        If source_mask provided: Tuple of (response_map, output_valid_mask)

    Raises:
        ValueError: If support is not an odd positive integer

    Example:
        Apply Mean filter with 15-voxel support:

        ```python
        import numpy as np
        from pictologics.filters import mean_filter

        # Create dummy 3D image
        image = np.random.rand(50, 50, 50)

        # Apply filter (original API)
        response = mean_filter(image, support=15, boundary="zero")

        # With source_mask for sentinel exclusion
        mask = image > -1000  # Valid voxels
        response, valid_mask = mean_filter(image, support=15, source_mask=mask)
        ```

    Note:
        Support M is defined in voxel units as per IBSI specification.
    """
    # Validate support
    if support < 1 or support % 2 == 0:
        raise ValueError(f"Support must be an odd positive integer, got {support}")

    # Convert to float32 as required by IBSI
    image = ensure_float32(image)

    # Handle string boundary condition
    if isinstance(boundary, str):
        boundary = BoundaryCondition[boundary.upper()]

    mode = get_scipy_mode(boundary)

    if source_mask is not None:
        # Use normalized convolution for source masking
        return _normalized_uniform_filter(image, source_mask, size=support, mode=mode)
    else:
        # Original behavior - return just the result (backward compatible)
        return uniform_filter(image, size=support, mode=mode)  # type: ignore[no-any-return]
