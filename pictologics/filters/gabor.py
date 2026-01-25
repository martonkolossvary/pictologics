# pictologics/filters/gabor.py
"""Gabor filter implementation (IBSI code: Q88H)."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Tuple, Union, cast

import numpy as np
from numpy import typing as npt
from scipy.signal import fftconvolve

from .base import BoundaryCondition, ensure_float32, get_scipy_mode

# Threshold for enabling parallel processing (voxels)
# Lower than other filters because Gabor has high per-slice cost
_PARALLEL_THRESHOLD = 100_000  # ~46³


def gabor_filter(
    image: npt.NDArray[np.floating[Any]],
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
    use_parallel: Union[bool, None] = None,
) -> npt.NDArray[np.floating[Any]]:
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
        use_parallel: If True, process slices in parallel. If None (default),
            auto-enables for images > ~80³ voxels.

    Returns:
        Response map (modulus of complex response)

    Example:
        Apply Gabor filter with rotation invariance over orthogonal planes:

        ```python
        import numpy as np
        from pictologics.filters import gabor_filter

        # Create dummy 3D image
        image = np.random.rand(50, 50, 50)

        # Apply filter
        response = gabor_filter(
            image,
            sigma_mm=10.0,
            lambda_mm=4.0,
            gamma=0.5,
            rotation_invariant=True,
            delta_theta=np.pi/4,
            average_over_planes=True
        )
        ```

    Note:
        - Returns modulus |h| = |g ⊗ f| for feature extraction
        - 2D filter applied slice-by-slice, then optionally over planes
        - Uses single complex FFT convolution for ~2x speedup
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

    # Auto-detect parallel mode based on image size
    if use_parallel is None:
        use_parallel = image.size > _PARALLEL_THRESHOLD

    if rotation_invariant and delta_theta is not None:
        # Generate orientations from 0 to 2π
        n_orientations = int(np.ceil(2 * np.pi / delta_theta))
        thetas = [i * delta_theta for i in range(n_orientations)]
    else:
        thetas = [theta]

    if average_over_planes:
        # Apply to all 3 orthogonal planes and average with in-place aggregation
        result: npt.NDArray[np.floating[Any]] | None = None
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
                use_parallel,
            )
            if result is None:
                result = plane_response.astype(np.float64)
            else:
                result += plane_response

        if result is None:  # pragma: no cover
            raise RuntimeError("Result should not be None after plane loop")

        return (result / 3.0).astype(np.float32)  # type: ignore[union-attr]
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
            use_parallel=use_parallel,
        )


def _apply_gabor_to_plane(
    image: npt.NDArray[np.floating[Any]],
    sigma_voxels: float,
    lambda_voxels: float,
    gamma: float,
    thetas: list[float],
    plane_axis: int,
    mode: str,
    pooling: str,
    use_parallel: bool = True,
) -> npt.NDArray[np.floating[Any]]:
    """Apply Gabor filter to slices along a given axis.

    Args:
        use_parallel: If True, process slices in parallel using ThreadPoolExecutor.
            For small images, sequential may be faster due to thread overhead.
    """
    # Pre-compute all kernels for efficiency
    kernels = [
        _create_gabor_kernel_2d(sigma_voxels, lambda_voxels, gamma, theta)
        for theta in thetas
    ]

    def process_slice(
        slice_2d: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        """Process a single 2D slice with all orientations using in-place pooling."""
        # Optimization: Pre-pad and pre-cast the slice once, as all kernels have the same size.
        # This avoids redundant padding and casting inside the loop.

        # Get padding parameters from the first kernel (all have same size)
        kernel_shape = kernels[0].shape
        pad_h = kernel_shape[0] // 2
        pad_w = kernel_shape[1] // 2

        # Map scipy.ndimage mode to numpy.pad mode
        pad_mode_map = {
            "constant": "constant",
            "reflect": "symmetric",
            "mirror": "reflect",
            "nearest": "edge",
            "wrap": "wrap",
        }
        pad_mode_literal = pad_mode_map.get(mode, "constant")

        # Pad and cast to complex64 once
        padded = np.pad(slice_2d, ((pad_h, pad_h), (pad_w, pad_w)), mode=pad_mode_literal)  # type: ignore[call-overload]
        padded_complex = padded.astype(np.complex64)

        # Helper to convolve pre-padded image
        def convolve_prepadded(
            k: npt.NDArray[np.floating[Any]],
        ) -> npt.NDArray[np.floating[Any]]:
            # fftconvolve mode="same" on padded image
            response = fftconvolve(padded_complex, k, mode="same")
            # Crop back to original size
            h, w = slice_2d.shape
            cropped = response[pad_h : pad_h + h, pad_w : pad_w + w]
            # Explicit cast to avoid MyPy 'no-any-return'
            return cast(npt.NDArray[np.floating[Any]], np.abs(cropped))

        if len(kernels) == 1:
            return convolve_prepadded(kernels[0])

        # In-place pooling to avoid allocating n_orientations x slice memory
        result_slice: npt.NDArray[np.floating[Any]] | None = None
        for k in kernels:
            response = convolve_prepadded(k)
            if result_slice is None:
                result_slice = (
                    response.astype(np.float64)
                    if pooling == "average"
                    else response.copy()
                )
            else:
                if pooling == "max":
                    np.maximum(result_slice, response, out=result_slice)
                elif pooling == "average":
                    result_slice += response
                else:  # pooling == "min"
                    np.minimum(result_slice, response, out=result_slice)

        # Mypy check
        if result_slice is None:  # pragma: no cover
            raise RuntimeError("Result slice should not be None")

        if pooling == "average":
            result_slice /= len(kernels)
        return result_slice.astype(np.float32)

    # Use moveaxis for efficient slice access (contiguous memory)
    # This moves plane_axis to position 0 for efficient iteration
    image_reordered = np.moveaxis(image, plane_axis, 0)
    n_slices = image_reordered.shape[0]

    if use_parallel:
        # Parallel processing for large images
        with ThreadPoolExecutor() as executor:
            # image_reordered[i] is a view, no copy needed
            processed = list(
                executor.map(
                    process_slice, [image_reordered[i] for i in range(n_slices)]
                )
            )
    else:
        # Sequential processing for small images
        processed = [process_slice(image_reordered[i]) for i in range(n_slices)]

    # Stack and move axis back to original position
    result_reordered = np.stack(processed, axis=0)
    return np.moveaxis(result_reordered, 0, plane_axis)


def _create_gabor_kernel_2d(
    sigma: float,
    wavelength: float,
    gamma: float,
    theta: float,
) -> npt.NDArray[np.floating[Any]]:
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
