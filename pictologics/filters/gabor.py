# pictologics/filters/gabor.py
"""Gabor filter implementation (IBSI code: Q88H)."""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Union

import numpy as np
from scipy.signal import fftconvolve

from .base import BoundaryCondition, ensure_float32, get_scipy_mode


def gabor_filter(
    image: np.ndarray,
    sigma_mm: float,
    lambda_mm: float,
    gamma: float = 1.0,
    theta: float = 0.0,
    spacing_mm: Union[float, Tuple[float, float, float]] = 1.0,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.ZERO,
    rotation_invariant: bool = False,
    delta_theta: Optional[float] = None,
    pooling: str = "average",
    average_over_planes: bool = False,
) -> np.ndarray:
    """
    Apply 2D Gabor filter to 3D image (IBSI code: Q88H).

    The Gabor filter is applied in the axial plane (k1, k2) and optionally
    averaged over orthogonal planes. Per IBSI 2 Eq. 9.

    Args:
        image: 3D input image array
        sigma_mm: Standard deviation of Gaussian envelope in mm (41LN)
        lambda_mm: Wavelength in mm (S4N6)
        gamma: Spatial aspect ratio (GDR5), typically 0.5 to 2.0
        theta: Orientation angle in radians (FQER), clockwise in (k1,k2)
        spacing_mm: Voxel spacing in mm (scalar or tuple)
        boundary: Boundary condition for padding (GBYQ)
        rotation_invariant: If True, average over orientations
        delta_theta: Orientation step for rotation invariance (XTGK)
        pooling: Pooling method ("average", "max", "min")
        average_over_planes: If True, average 2D responses over 3 orthogonal planes

    Returns:
        Response map (modulus of complex response)

    Example:
        >>> # Gabor with rotation invariance over orthogonal planes
        >>> response = gabor_filter(
        ...     image, sigma_mm=10.0, lambda_mm=4.0, gamma=0.5,
        ...     rotation_invariant=True, delta_theta=np.pi/4,
        ...     average_over_planes=True
        ... )

    Note:
        - Returns modulus |h| = |g ⊗ f| for feature extraction
        - 2D filter applied slice-by-slice, then optionally over planes
    """
    # Convert to float32
    image = ensure_float32(image)

    # Handle spacing
    if isinstance(spacing_mm, (int, float)):
        spacing_mm = (float(spacing_mm),) * 3

    # Convert mm to voxels (use in-plane spacing for 2D filter)
    sigma_voxels = sigma_mm / spacing_mm[0]  # Assume isotropic in-plane
    lambda_voxels = lambda_mm / spacing_mm[0]

    # Handle boundary
    if isinstance(boundary, str):
        boundary = BoundaryCondition[boundary.upper()]
    mode = get_scipy_mode(boundary)

    # Validate pooling parameter early
    valid_poolings = ("max", "average", "min")
    if pooling not in valid_poolings:
        raise ValueError(f"Unknown pooling: {pooling}. Must be one of {valid_poolings}")

    if rotation_invariant and delta_theta is not None:
        # Generate orientations from 0 to 2π
        n_orientations = int(np.ceil(2 * np.pi / delta_theta))
        thetas = [i * delta_theta for i in range(n_orientations)]
    else:
        thetas = [theta]

    if average_over_planes:
        # Apply to all 3 orthogonal planes and average
        responses = []
        for plane_axis in range(3):
            plane_response = _apply_gabor_to_plane(
                image,
                sigma_voxels,
                lambda_voxels,
                gamma,
                thetas,
                plane_axis,
                mode,
                pooling,
            )
            responses.append(plane_response)
        return np.mean(np.stack(responses, axis=0), axis=0)  # type: ignore[no-any-return]
    else:
        # Apply only to axial plane (axis 2 = k3 slices)
        return _apply_gabor_to_plane(
            image,
            sigma_voxels,
            lambda_voxels,
            gamma,
            thetas,
            plane_axis=2,
            mode=mode,
            pooling=pooling,
        )


def _apply_gabor_to_plane(
    image: np.ndarray,
    sigma_voxels: float,
    lambda_voxels: float,
    gamma: float,
    thetas: list[float],
    plane_axis: int,
    mode: str,
    pooling: str,
) -> np.ndarray:
    """Apply Gabor filter to slices along a given axis (parallel processing)."""
    result = np.zeros_like(image)

    # Pre-compute all kernels for efficiency
    kernels = [
        _create_gabor_kernel_2d(sigma_voxels, lambda_voxels, gamma, theta)
        for theta in thetas
    ]

    def process_slice(slice_2d: np.ndarray) -> np.ndarray:
        """Process a single 2D slice with all orientations."""
        orientation_responses = [_fft_convolve_2d(slice_2d, k, mode) for k in kernels]

        if len(orientation_responses) > 1:
            stacked = np.stack(orientation_responses, axis=0)
            if pooling == "average":
                return np.mean(stacked, axis=0)  # type: ignore[no-any-return]
            elif pooling == "max":
                return np.max(stacked, axis=0)  # type: ignore[no-any-return]
            else:  # pooling == "min"
                return np.min(stacked, axis=0)  # type: ignore[no-any-return]
        return orientation_responses[0]

    # Extract all slices
    n_slices = image.shape[plane_axis]
    slices_list: list[np.ndarray] = []
    for i in range(n_slices):
        idx: list[slice | int] = [slice(None)] * 3
        idx[plane_axis] = i
        slices_list.append(image[tuple(idx)])

    # Process slices in parallel
    with ThreadPoolExecutor() as executor:
        processed = list(executor.map(process_slice, slices_list))

    # Reassemble result
    for i, res in enumerate(processed):
        idx2: list[slice | int] = [slice(None)] * 3
        idx2[plane_axis] = i
        result[tuple(idx2)] = res

    return result


def _fft_convolve_2d(image: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
    """
    FFT-based 2D convolution with boundary handling matching scipy.ndimage.convolve.

    Args:
        image: 2D input array
        kernel: 2D complex Gabor kernel
        mode: Boundary mode ('constant', 'reflect', 'mirror', 'nearest', 'wrap')

    Returns:
        Modulus of convolution response
    """
    # Pad image to handle boundaries (matching scipy.ndimage behavior)
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    # Map scipy.ndimage mode names to numpy.pad mode names
    # scipy.ndimage 'reflect' includes edge (d c b a | a b c d)
    # numpy.pad 'reflect' excludes edge (d c b | a b c d)
    # So: scipy 'reflect' -> numpy 'symmetric'
    #     scipy 'mirror' -> numpy 'reflect'
    pad_mode_map = {
        "constant": "constant",
        "reflect": "symmetric",  # scipy includes edge, numpy 'symmetric' includes edge
        "mirror": "reflect",  # scipy excludes edge, numpy 'reflect' excludes edge
        "nearest": "edge",
        "wrap": "wrap",
    }

    # Pad the image
    pad_mode_literal = pad_mode_map.get(mode, "constant")
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=pad_mode_literal)  # type: ignore[call-overload]

    # FFT convolution on padded image
    response_real = fftconvolve(padded, np.real(kernel), mode="same")
    response_imag = fftconvolve(padded, np.imag(kernel), mode="same")

    # Crop back to original size
    h, w = image.shape
    response_real = response_real[pad_h : pad_h + h, pad_w : pad_w + w]
    response_imag = response_imag[pad_h : pad_h + h, pad_w : pad_w + w]

    # Compute modulus
    return np.sqrt(response_real**2 + response_imag**2)  # type: ignore[no-any-return]


def _create_gabor_kernel_2d(
    sigma: float,
    wavelength: float,
    gamma: float,
    theta: float,
) -> np.ndarray:
    """
    Create a 2D Gabor kernel.
    """
    # Determine kernel size (6σ truncation for complete coverage)
    radius = int(np.ceil(6.0 * sigma))

    # Create coordinate grid - row (k1/y) varies along axis 0, col (k2/x) along axis 1
    k1, k2 = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float64)

    # Rotate coordinates per IBSI convention (clockwise)
    # k̃₁ = k1*cos(θ) + k2*sin(θ)
    # k̃₂ = -k1*sin(θ) + k2*cos(θ)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    k1_rot = k1 * cos_t + k2 * sin_t  # k̃₁
    k2_rot = -k1 * sin_t + k2 * cos_t  # k̃₂

    # Gabor formula
    gaussian = np.exp(-(k1_rot**2 + gamma**2 * k2_rot**2) / (2 * sigma**2))
    sinusoid = np.exp(1j * 2 * np.pi * k1_rot / wavelength)

    kernel = gaussian * sinusoid
    return kernel.astype(np.complex64)  # type: ignore[no-any-return]
