# pictologics/filters/riesz.py
"""Riesz transform implementation (IBSI code: AYRS)."""

from math import factorial, sqrt
from typing import Tuple, Union

import numpy as np

from .base import ensure_float32


def riesz_transform(
    image: np.ndarray,
    order: Tuple[int, ...],
) -> np.ndarray:
    """
    Apply Riesz transform (IBSI code: AYRS).

    The Riesz transform computes higher-order all-pass image derivatives
    in the Fourier domain. Per IBSI 2 Eq. 34.

    Args:
        image: 3D input image array
        order: Tuple (l1, l2, l3) specifying derivative order per axis
               e.g., (1,0,0) = first-order along k1 (gradient-like)
                     (2,0,0), (1,1,0), (0,2,0) = second-order (Hessian-like)

    Returns:
        Riesz-transformed image (real part)

    Example:
        >>> # First-order Riesz along k1 axis
        >>> response = riesz_transform(image, order=(1, 0, 0))

    Note:
        - First-order Riesz components form the image gradient
        - Second-order Riesz components form the image Hessian
        - All-pass: doesn't amplify high frequencies like regular derivatives
    """
    # Convert to float32
    image = ensure_float32(image)

    L = sum(order)  # Total order

    if L == 0:
        raise ValueError("At least one order component must be > 0")

    # Create frequency grid in [-π, π]
    freqs = [np.fft.fftfreq(s) * 2 * np.pi for s in image.shape]
    nu = np.meshgrid(*freqs, indexing="ij")

    # Compute ||ν||
    nu_norm = np.sqrt(sum(n**2 for n in nu))

    # Avoid division by zero at DC
    nu_norm_safe = np.where(nu_norm > 0, nu_norm, 1.0)

    # Compute normalization factor
    norm_factor = sqrt(factorial(L) / np.prod([factorial(o) for o in order]))

    # Compute numerator
    numerator = np.ones(image.shape, dtype=np.float64)
    for i, ord_val in enumerate(order):
        if ord_val > 0:
            numerator *= nu[i] ** ord_val

    # Riesz transfer function
    phase = np.exp(-1j * np.pi * L / 2)

    transfer = phase * norm_factor * numerator / (nu_norm_safe**L)
    transfer = np.where(nu_norm > 0, transfer, 0)  # Set DC to 0

    # Apply in frequency domain
    F = np.fft.fftn(image)
    response = np.fft.ifftn(F * transfer)

    return np.real(response).astype(np.float32)


def riesz_log(
    image: np.ndarray,
    sigma_mm: float,
    spacing_mm: Union[float, Tuple[float, float, float]] = 1.0,
    order: Tuple[int, ...] = (1, 0, 0),
    truncate: float = 4.0,
) -> np.ndarray:
    """
    Apply Riesz transform to LoG-filtered image.

    Combines multi-scale analysis (LoG) with directional analysis (Riesz).
    First applies LoG filtering, then applies Riesz transform.

    Args:
        image: 3D input image array
        sigma_mm: LoG scale in mm
        spacing_mm: Voxel spacing in mm
        order: Riesz order tuple (l1, l2, l3)
        truncate: LoG truncation parameter

    Returns:
        Riesz-transformed LoG response

    Example:
        >>> # First-order Riesz of LoG at 5mm scale
        >>> response = riesz_log(
        ...     image, sigma_mm=5.0, spacing_mm=2.0, order=(1, 0, 0)
        ... )
    """
    from .log import laplacian_of_gaussian

    # First apply LoG
    log_response = laplacian_of_gaussian(
        image, sigma_mm=sigma_mm, spacing_mm=spacing_mm, truncate=truncate
    )

    # Then apply Riesz transform
    return riesz_transform(log_response, order=order)


def riesz_simoncelli(
    image: np.ndarray,
    level: int = 1,
    order: Tuple[int, ...] = (1, 0, 0),
) -> np.ndarray:
    """
    Apply Riesz transform to Simoncelli wavelet-filtered image.

    Combines isotropic multi-scale analysis (Simoncelli) with
    directional analysis (Riesz) for rotation-invariant directional features.

    Args:
        image: 3D input image array
        level: Simoncelli decomposition level
        order: Riesz order tuple (l1, l2, l3)

    Returns:
        Riesz-transformed Simoncelli response

    Example:
        >>> # Second-order Riesz (Hessian-like) of Simoncelli level 2
        >>> response = riesz_simoncelli(
        ...     image, level=2, order=(2, 0, 0)
        ... )
    """
    from .wavelets import simoncelli_wavelet

    # First apply Simoncelli
    sim_response = simoncelli_wavelet(image, level=level)

    # Then apply Riesz transform
    return riesz_transform(sim_response, order=order)


def get_riesz_orders(max_order: int, ndim: int = 3) -> Tuple[Tuple[int, ...], ...]:
    """
    Generate all Riesz order tuples for a given maximum order.

    Args:
        max_order: Maximum total order L
        ndim: Number of dimensions (default 3)

    Returns:
        Tuple of all valid order tuples

    Example:
        >>> get_riesz_orders(2, ndim=3)
        ((2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2))
    """
    from itertools import combinations_with_replacement

    orders = []
    for combo in combinations_with_replacement(range(ndim), max_order):
        order = [0] * ndim
        for i in combo:
            order[i] += 1
        orders.append(tuple(order))

    return tuple(orders)
