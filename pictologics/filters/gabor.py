# pictologics/filters/gabor.py
"""Gabor filter implementation (IBSI code: Q88H)."""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Tuple, Union, cast

import numpy as np
import scipy.fft
from numba import get_num_threads
from numpy import typing as npt

from .base import (
    BoundaryCondition,
    _padding_value_problem,
    _prepare_masked_image,
    ensure_float32,
    get_scipy_mode,
)

# The parts of the complex Gabor response (IBSI 2 response map, 5P3T)
_RESPONSE_PARTS: dict[str, Callable[[Any], Any]] = {
    "modulus": np.abs,
    "real": np.real,
    "imaginary": np.imag,
    "angle": np.angle,
}

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
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
    region: Optional[Tuple[slice, slice, slice]] = None,
    response: str = "modulus",
    padding_value: float = 0.0,
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
        spacing_mm: Voxel spacing in mm (scalar or per-axis tuple). Each
            plane's kernel is built from that plane's own two in-plane
            axis spacings, so anisotropic spacing (including anisotropic
            z, relevant when `average_over_planes=True`) is handled
            correctly rather than approximated from a single axis.
        boundary: Boundary condition for padding (GBYQ)
        rotation_invariant: If True, average over orientations
        delta_theta: Orientation step for rotation invariance (XTGK)
        pooling: Pooling method ("average", "max", "min")
        average_over_planes: If True, average 2D responses over 3 orthogonal planes
        use_parallel: If True, process slices in parallel. If None (default),
            auto-enables when the slices to filter hold more than ~46³ voxels.
        source_mask: Optional boolean mask where True = valid voxel.
            When provided, zeros out invalid (sentinel) voxels before
            FFT-based convolution to prevent contamination.
        region: Optional box of the image (one slice per axis). Only the response
            in the box is computed, and the result has the box shape. Each plane
            still filters whole slices, so the values are those of the whole image.
        response: The part of the complex response (5P3T): "modulus" (0J1M, default),
            "angle" (V530, in radians), "real" (F7DI) or "imaginary" (LI3Y). The
            orientations pool this part.
        padding_value: The constant of constant value padding (Z3VE), with the ZERO
            (constant) boundary. Default 0.

    Returns:
        Response map (the `response` part of the complex response)

    Raises:
        ValueError: If `pooling` is not "max", "average", or "min", if
            `rotation_invariant=True` is set without providing `delta_theta`, if
            `response` is not one of the four parts, or if `padding_value` is not 0
            with another boundary than ZERO.
        RuntimeError: Defensive check raised if plane averaging fails to
            produce a result; not expected to occur in normal use.

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
            delta_theta=0.7853981633974483,  # pi/4
            average_over_planes=True
        )
        ```

    Note:
        - Returns modulus |h| = |g ⊗ f| for feature extraction
        - 2D filter applied slice-by-slice, then optionally over planes
        - Uses single complex FFT convolution for ~2x speedup
        - Each plane's kernel uses that plane's own two in-plane spacings.
          When they are equal (the isotropic-in-plane case, including the
          default axial-only plane under typical (x, y, z) spacing with
          x == y), the kernel is built on a voxel-unit grid. When they
          differ, the kernel is built on a physical-coordinate (mm) grid
          with a per-axis radius, giving a rectangular kernel that is
          physically correct rather than warning and guessing.
    """
    # Convert to float32
    image = ensure_float32(image)

    # Apply source_mask preprocessing (zero out invalid voxels for FFT-based filter)
    if source_mask is not None:
        image = _prepare_masked_image(image, source_mask)

    # Handle spacing. The mm -> voxel/physical conversion is deferred to
    # _apply_gabor_to_plane, which is per-plane: each plane's in-plane axes
    # (and therefore in-plane spacings) depend on plane_axis.
    if isinstance(spacing_mm, (int, float)):
        spacing_mm = (float(spacing_mm),) * 3

    # Handle boundary
    if isinstance(boundary, str):
        boundary = BoundaryCondition[boundary.upper()]
    mode = get_scipy_mode(boundary)

    # Validate pooling parameter early
    valid_poolings = ("max", "average", "min")
    if pooling not in valid_poolings:
        raise ValueError(f"Unknown pooling: {pooling}. Must be one of {valid_poolings}")
    if response not in _RESPONSE_PARTS:
        raise ValueError(f"Unknown response: {response}. Must be one of {tuple(_RESPONSE_PARTS)}")
    problem = _padding_value_problem(boundary, padding_value)
    if problem:
        raise ValueError(problem)

    if rotation_invariant:
        if delta_theta is None:
            raise ValueError(
                "rotation_invariant=True requires delta_theta (the orientation step in radians)"
            )
        # Generate orientations from 0 to 2π. A step given with few digits (0.785398 for
        # π/4) still means a whole number of orientations, with the exact step 2π / n;
        # ceil would add one orientation.
        steps = 2 * np.pi / delta_theta
        n_orientations = round(steps)
        if n_orientations >= 1 and abs(steps - n_orientations) < 1e-3:
            delta_theta = 2 * np.pi / n_orientations
        else:
            n_orientations = int(np.ceil(steps))
        thetas = [i * delta_theta for i in range(n_orientations)]
        # The Gabor response modulus is π-periodic in theta: kernel(θ+π) = conj(kernel(θ))
        # and the image is real, so |response| (and its real part) is identical for θ
        # and θ+π. When the orientation set is closed under +π (n even and spans exactly
        # 2π), the second half duplicates the first; drop it (max/min/average pooling
        # are unchanged). The imaginary part and the angle change sign at θ+π.
        closed = n_orientations % 2 == 0 and abs(n_orientations * delta_theta - 2 * np.pi) < 1e-9
        if closed and response in ("modulus", "real"):
            thetas = thetas[: n_orientations // 2]
    else:
        thetas = [theta]

    def _plane(plane_axis: int) -> npt.NDArray[np.floating[Any]]:
        """The response of one plane; with a region, of the region only (each slice
        through it is cut to the region, grown by the kernel radius)."""
        part = image
        window = None
        size = image.size
        if region is not None:
            part = image[tuple(region[a] if a == plane_axis else slice(None) for a in range(3))]
            window = (region[0], region[1], region[2])
            size = math.prod(r.stop - r.start for r in window)
        # Auto-detect parallel mode based on the size of the part filtered
        parallel = size > _PARALLEL_THRESHOLD if use_parallel is None else use_parallel
        return _apply_gabor_to_plane(
            part,
            sigma_mm,
            lambda_mm,
            gamma,
            thetas,
            plane_axis=plane_axis,
            spacing_mm=spacing_mm,
            mode=mode,
            pooling=pooling,
            use_parallel=parallel,
            response=response,
            padding_value=padding_value,
            window=None
            if window is None
            else tuple(window[a] for a in range(3) if a != plane_axis),
        )

    if average_over_planes:
        # Apply to all 3 orthogonal planes and average with in-place aggregation
        result: npt.NDArray[np.floating[Any]] | None = None
        for plane_axis in range(3):
            plane_response = _plane(plane_axis)
            if result is None:
                result = plane_response.astype(np.float64)
            else:
                result += plane_response

        if result is None:  # pragma: no cover
            raise RuntimeError("Result should not be None after plane loop")

        return (result / 3.0).astype(np.float32)  # type: ignore[union-attr]
    else:
        # Apply only to axial plane (axis 2 = k3 slices)
        return _plane(2)


def _apply_gabor_to_plane(
    image: npt.NDArray[np.floating[Any]],
    sigma_mm: float,
    lambda_mm: float,
    gamma: float,
    thetas: list[float],
    plane_axis: int,
    spacing_mm: Tuple[float, float, float],
    mode: str,
    pooling: str,
    use_parallel: bool = True,
    window: Optional[Tuple[slice, ...]] = None,
    response: str = "modulus",
    padding_value: float = 0.0,
) -> npt.NDArray[np.floating[Any]]:
    """Apply Gabor filter to slices along a given axis.

    Args:
        use_parallel: If True, process slices in parallel using ThreadPoolExecutor.
            For small images, sequential may be faster due to thread overhead.
        window: The part of each slice to return (two slices of the in-plane axes, in
            their order); None: the whole slices. Each slice is cut to the window grown
            by the kernel radius: the padding of a cut edge inside the image then reaches
            only the grown part, so the window keeps the values of the whole slices (to
            the FFT rounding), and a cut edge at the image edge keeps its boundary.
    """
    # This plane's two in-plane axes (the axes the 2D kernel actually acts on;
    # plane_axis itself is only sliced over, see moveaxis below) and their spacings.
    in_plane_axes = [a for a in range(3) if a != plane_axis]
    s1, s2 = spacing_mm[in_plane_axes[0]], spacing_mm[in_plane_axes[1]]

    if np.isclose(s1, s2):
        # In-plane isotropic: build the kernel on a voxel-unit grid exactly as
        # before (preserved verbatim so isotropic-spacing results stay
        # byte-identical; the physical-grid formulation below is mathematically
        # equivalent here but differs at the ~1e-15 level due to a different
        # floating-point evaluation order).
        sigma_voxels = sigma_mm / s1
        lambda_voxels = lambda_mm / s1
        kernels = [
            _create_gabor_kernel_2d(sigma_voxels, lambda_voxels, gamma, theta) for theta in thetas
        ]
    else:
        # In-plane anisotropic: build the kernel on a physical (mm) grid with a
        # per-axis voxel radius, so each axis is scaled by its own true spacing.
        kernels = [
            _create_gabor_kernel_2d_anisotropic(sigma_mm, lambda_mm, gamma, theta, s1, s2)
            for theta in thetas
        ]

    # Move the plane axis to position 0 so each image_reordered[i] is a 2D slice
    # (a view; the copy happens later in np.pad).
    image_reordered = np.moveaxis(image, plane_axis, 0)
    n_slices = image_reordered.shape[0]
    slice_h, slice_w = int(image_reordered.shape[1]), int(image_reordered.shape[2])

    # All slices share one 2D shape and all kernels share one shape, so the FFT of
    # each padded slice can be computed once and reused across every orientation,
    # and each kernel's FFT can be computed once for the whole plane. This is the
    # equivalent of fftconvolve(padded, k, "same") but without re-transforming the
    # slice per kernel and the kernels per slice.
    kernel_shape = kernels[0].shape
    pad_h = kernel_shape[0] // 2
    pad_w = kernel_shape[1] // 2
    keep_h, keep_w = (0, slice_h), (0, slice_w)  # the rows and columns of the cut kept
    if window is not None:
        h_lo, h_hi = max(window[0].start - pad_h, 0), min(window[0].stop + pad_h, slice_h)
        w_lo, w_hi = max(window[1].start - pad_w, 0), min(window[1].stop + pad_w, slice_w)
        image_reordered = image_reordered[:, h_lo:h_hi, w_lo:w_hi]
        slice_h, slice_w = h_hi - h_lo, w_hi - w_lo
        keep_h = (window[0].start - h_lo, window[0].stop - h_lo)
        keep_w = (window[1].start - w_lo, window[1].stop - w_lo)

    # Map scipy.ndimage mode to numpy.pad mode
    pad_mode_map = {
        "constant": "constant",
        "reflect": "symmetric",
        "mirror": "reflect",
        "nearest": "edge",
        "wrap": "wrap",
    }
    pad_mode_literal = pad_mode_map.get(mode, "constant")
    constant = {"constant_values": padding_value} if pad_mode_literal == "constant" else {}
    part = _RESPONSE_PARTS[response]

    # A circular convolution as long as the padded slice is enough: the kept part of the
    # output (the slice) is at least one kernel radius from the ends of the padded slice,
    # so no wrapped value reaches it
    fshape = (
        scipy.fft.next_fast_len(slice_h + 2 * pad_h),
        scipy.fft.next_fast_len(slice_w + 2 * pad_w),
    )
    kernel_ffts = [scipy.fft.fftn(k, s=fshape) for k in kernels]

    def process_slice(
        slice_2d: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        """Process a single 2D slice with all orientations using in-place pooling."""
        padded = np.pad(
            slice_2d, ((pad_h, pad_h), (pad_w, pad_w)), mode=pad_mode_literal, **constant
        )  # type: ignore[call-overload]
        f_padded = scipy.fft.fftn(padded.astype(np.complex64), s=fshape)

        def convolve_prepadded(
            k_fft: npt.NDArray[np.complexfloating[Any, Any]],
        ) -> npt.NDArray[np.floating[Any]]:
            # Full convolution via FFT; the "same"+unpad crop reduces to a fixed
            # offset of 2*pad because the kernel half-width equals pad.
            full = scipy.fft.ifftn(f_padded * k_fft)
            cropped = full[
                2 * pad_h + keep_h[0] : 2 * pad_h + keep_h[1],
                2 * pad_w + keep_w[0] : 2 * pad_w + keep_w[1],
            ]
            return cast(npt.NDArray[np.floating[Any]], np.ascontiguousarray(part(cropped)))

        if len(kernel_ffts) == 1:
            return convolve_prepadded(kernel_ffts[0])

        # In-place pooling to avoid allocating n_orientations x slice memory
        result_slice: npt.NDArray[np.floating[Any]] | None = None
        for k_fft in kernel_ffts:
            response = convolve_prepadded(k_fft)
            if result_slice is None:
                # The part is a fresh array, so no copy is needed.
                result_slice = response.astype(np.float64) if pooling == "average" else response
            elif pooling == "max":
                np.maximum(result_slice, response, out=result_slice)
            elif pooling == "average":
                result_slice += response
            else:  # pooling == "min"
                np.minimum(result_slice, response, out=result_slice)

        # Mypy check
        if result_slice is None:  # pragma: no cover
            raise RuntimeError("Result slice should not be None")

        if pooling == "average":
            result_slice /= len(kernel_ffts)
        return result_slice.astype(np.float32)

    if use_parallel:
        # Parallel processing for large images
        with ThreadPoolExecutor(max_workers=get_num_threads()) as executor:
            # image_reordered[i] is a view, no copy needed
            processed = list(
                executor.map(process_slice, [image_reordered[i] for i in range(n_slices)])
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
    # Kernel size: 6σ truncation along the long axis of the envelope, which has the
    # scale σ / γ for γ < 1 (IBSI 2 tests 4.a.1 and 4.a.2 then match their references)
    radius = int(np.ceil(6.0 * sigma / min(1.0, gamma)))

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


def _create_gabor_kernel_2d_anisotropic(
    sigma_mm: float,
    wavelength_mm: float,
    gamma: float,
    theta: float,
    s1: float,
    s2: float,
) -> npt.NDArray[np.floating[Any]]:
    """
    Create a 2D Gabor kernel for a plane whose two in-plane axes have different
    physical spacing (s1, s2, in mm). Unlike `_create_gabor_kernel_2d`, the
    kernel is built on a physical-coordinate (mm) grid rather than a voxel grid,
    with a per-axis voxel radius, so the result is a rectangular kernel that
    reflects each axis's true spacing rather than assuming a single scale.
    """
    # Per-axis radius (voxels) for 6σ truncation along each physical axis, with the
    # envelope scale σ / γ for γ < 1 (see _create_gabor_kernel_2d).
    reach_mm = 6.0 * sigma_mm / min(1.0, gamma)
    radius1 = int(np.ceil(reach_mm / s1))
    radius2 = int(np.ceil(reach_mm / s2))

    # Voxel-index grid, then converted to physical (mm) coordinates per axis.
    k1, k2 = np.mgrid[-radius1 : radius1 + 1, -radius2 : radius2 + 1].astype(np.float64)
    p1 = k1 * s1
    p2 = k2 * s2

    # Rotate physical coordinates per IBSI convention (clockwise), same as the
    # isotropic kernel but in mm rather than voxels.
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    p1_rot = p1 * cos_t + p2 * sin_t  # p̃₁
    p2_rot = -p1 * sin_t + p2 * cos_t  # p̃₂

    # Gabor formula, directly in mm (sigma_mm, wavelength_mm used as-is).
    gaussian = np.exp(-(p1_rot**2 + gamma**2 * p2_rot**2) / (2 * sigma_mm**2))
    sinusoid = np.exp(1j * 2 * np.pi * p1_rot / wavelength_mm)

    kernel = gaussian * sinusoid
    return kernel.astype(np.complex64)  # type: ignore[no-any-return]
