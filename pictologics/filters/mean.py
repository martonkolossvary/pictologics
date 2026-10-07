# pictologics/filters/mean.py
"""Mean filter implementation (IBSI code: S60F)."""

from typing import Any, Optional, Union, overload

import numpy as np
from numpy import typing as npt

from .base import (
    BoundaryCondition,
    _constant_padded,
    _normalized_uniform_filter,
    _padding_value_problem,
    _uniform_filter,
    as_float32,
    ensure_float32,
    get_scipy_mode,
    resolve_boundary,
)


@overload
def mean_filter(
    image: npt.NDArray[np.floating[Any]],
    support: int = ...,
    boundary: Union[BoundaryCondition, str] = ...,
    source_mask: None = ...,
    padding_value: float = ...,
) -> npt.NDArray[np.floating[Any]]: ...


@overload
def mean_filter(
    image: npt.NDArray[np.floating[Any]],
    support: int = ...,
    boundary: Union[BoundaryCondition, str] = ...,
    source_mask: npt.NDArray[np.bool_] = ...,
    padding_value: float = ...,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]: ...


def mean_filter(
    image: npt.NDArray[np.floating[Any]],
    support: int = 15,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.ZERO,
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
    padding_value: float = 0.0,
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
        padding_value: The constant of constant value padding (Z3VE), with the ZERO
            (constant) boundary. Default 0.

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

    # The passes run in float32. The normalized convolution of a source mask keeps the
    # image type: it divides two filtered sums.
    image = ensure_float32(image) if source_mask is not None else as_float32(image)

    # Handle string boundary condition
    boundary = resolve_boundary(boundary)

    problem = _padding_value_problem(boundary, padding_value)
    if problem:
        raise ValueError(problem)
    if padding_value:
        return _constant_padded(
            mean_filter, image, source_mask, padding_value, support // 2, support=support, boundary=boundary
        )  # fmt: skip
    mode = get_scipy_mode(boundary)

    if source_mask is not None:
        # Use normalized convolution for source masking
        return _normalized_uniform_filter(image, source_mask, size=support, mode=mode)
    else:
        # A float32 response: each pass sums its lines in double and writes float32
        return _uniform_filter(image, support, mode)
