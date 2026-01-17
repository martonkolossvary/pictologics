# pictologics/filters/log.py
"""Laplacian of Gaussian filter implementation (IBSI code: L6PA)."""

from typing import Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_laplace

from .base import BoundaryCondition, ensure_float32, get_scipy_mode


def laplacian_of_gaussian(
    image: np.ndarray,
    sigma_mm: float,
    spacing_mm: Union[float, Tuple[float, float, float]] = 1.0,
    truncate: float = 4.0,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.ZERO,
) -> np.ndarray:
    """
    Apply 3D Laplacian of Gaussian filter (IBSI code: L6PA).

    The LoG is a band-pass, spherically symmetric operator. Per IBSI 2 Eq. 3.

    Args:
        image: 3D input image array
        sigma_mm: Standard deviation in mm (σ*, 41LN)
        spacing_mm: Voxel spacing in mm (scalar for isotropic, or tuple)
        truncate: Filter size cutoff in σ units (default 4.0, WGPM)
        boundary: Boundary condition for padding (GBYQ)

    Returns:
        Response map with same dimensions as input

    Example:
        >>> # Phantoms have 2mm spacing
        >>> response = laplacian_of_gaussian(
        ...     image, sigma_mm=5.0, spacing_mm=2.0, truncate=4.0
        ... )

    Note:
        - σ is converted from mm to voxels: σ_voxels = σ_mm / spacing_mm
        - Filter size: M = 1 + 2⌊d×σ + 0.5⌋ where d=truncate
        - The kernel should sum to approximately 0 (zero-mean)
    """
    # Convert to float32 as required by IBSI
    image = ensure_float32(image)

    # Handle scalar spacing
    if isinstance(spacing_mm, (int, float)):
        spacing_mm = (float(spacing_mm),) * 3

    # Convert sigma from mm to voxels for each axis
    sigma_voxels = tuple(sigma_mm / s for s in spacing_mm)

    # Handle string boundary condition
    if isinstance(boundary, str):
        boundary = BoundaryCondition[boundary.upper()]

    mode = get_scipy_mode(boundary)

    # Apply Laplacian of Gaussian
    # scipy.ndimage.gaussian_laplace already implements LoG correctly
    return gaussian_laplace(image, sigma=sigma_voxels, mode=mode, truncate=truncate)  # type: ignore[no-any-return]
