# pictologics/filters/mean.py
"""Mean filter implementation (IBSI code: S60F)."""

from typing import Union

import numpy as np
from scipy.ndimage import uniform_filter

from .base import BoundaryCondition, ensure_float32, get_scipy_mode


def mean_filter(
    image: np.ndarray,
    support: int = 15,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.ZERO,
) -> np.ndarray:
    """
    Apply 3D mean filter (IBSI code: S60F).

    The mean filter computes the average intensity over an M×M×M
    spatial support. Per IBSI 2 Eq. 2.

    Args:
        image: 3D input image array
        support: Filter support M in voxels (must be odd, YNOF)
        boundary: Boundary condition for padding (GBYQ)

    Returns:
        Response map with same dimensions as input

    Raises:
        ValueError: If support is not an odd positive integer

    Example:
        >>> response = mean_filter(image, support=15, boundary="zero")

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

    # Apply uniform filter (3D)
    return uniform_filter(image, size=support, mode=mode)  # type: ignore[no-any-return]
