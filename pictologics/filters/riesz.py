# pictologics/filters/riesz.py
"""Riesz transform implementation (IBSI code: AYRS)."""

from functools import lru_cache
from math import factorial, sqrt
from typing import Any, Optional, Tuple, Union, cast

import numpy as np
import scipy.fft
from numpy import typing as npt

from .base import (
    BoundaryCondition,
    _apply_with_boundary_padding,
    _prepare_masked_image,
    ensure_float32,
    resolve_boundary,
)

# Default padding (voxels, same for every axis) for boundary-aware Riesz filtering
# via pad-filter-crop. The Riesz transfer function has unit magnitude everywhere
# (an all-pass, phase-only filter, IBSI 2 Eq. 34), so unlike the band-limited
# Simoncelli kernel its real-space kernel is not compactly supported: it decays
# as a power law, similar to a generalised Hilbert transform. Empirically (via
# `np.fft.irfftn` of the transfer function on a 64^3 grid), 16 voxels captures
# ~92-100% of a first/second-order kernel's L2 energy depending on direction; no
# finite pad removes truncation error entirely for this kernel, so 16 is a
# practical balance between accuracy and the cost of enlarging the FFT array.
_RIESZ_BASE_PAD = 16


@lru_cache(maxsize=64)
def _riesz_transfer(
    shape: Tuple[int, ...], order: Tuple[int, ...]
) -> npt.NDArray[np.complexfloating[Any, Any]]:
    """Riesz frequency-domain transfer function (IBSI 2 Eq. 34).

    Depends only on ``shape`` and ``order`` (never on image values or the source
    mask), so it is cached and reused across calls with identical geometry —
    including the many order tuples from ``get_riesz_orders`` that share one image
    shape. The returned array is marked read-only; callers must not mutate it.
    """
    ndim = len(shape)
    L = sum(order)

    # Frequency coordinates for rfftn: the last axis is non-negative freqs only.
    freqs = []
    for i, s in enumerate(shape):
        if i == ndim - 1:
            freqs.append(np.fft.rfftfreq(s) * 2 * np.pi)
        else:
            freqs.append(np.fft.fftfreq(s) * 2 * np.pi)

    # Broadcast (sparse) grid to avoid a full meshgrid the size of the input.
    nu_vectors = np.meshgrid(*freqs, indexing="ij", sparse=True)
    nu_sq_norm = np.asarray(sum(n**2 for n in nu_vectors), dtype=np.float64)
    nu_norm = np.sqrt(nu_sq_norm)
    nu_norm_safe = np.where(nu_norm > 0, nu_norm, 1.0)  # avoid /0 at DC

    norm_factor = sqrt(factorial(L) / np.prod([factorial(o) for o in order]))

    numerator = np.ones(nu_norm.shape, dtype=np.float64)
    for i, ord_val in enumerate(order):
        if ord_val > 0:
            numerator *= nu_vectors[i] ** ord_val

    phase = np.exp(-1j * np.pi * L / 2)
    transfer = phase * norm_factor * numerator / (nu_norm_safe**L)
    transfer = np.where(nu_norm > 0, transfer, 0)  # DC = 0
    transfer.flags.writeable = False  # cached array must not be mutated by callers
    return cast(npt.NDArray[np.complexfloating[Any, Any]], transfer)


def riesz_transform(
    image: npt.NDArray[np.floating[Any]],
    order: Tuple[int, ...],
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.PERIODIC,
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.floating[Any]]:
    """
    Apply Riesz transform (IBSI code: AYRS).

    The Riesz transform computes higher-order all-pass image derivatives
    in the Fourier domain. Per IBSI 2 Eq. 34.

    Args:
        image: 3D input image array
        order: Tuple (l1, l2, l3) specifying derivative order per axis
               e.g., (1,0,0) = first-order along k1 (gradient-like)
                     (2,0,0), (1,1,0), (0,2,0) = second-order (Hessian-like)
        boundary: Boundary condition. The filter is inherently periodic (FFT-based),
            so `BoundaryCondition.PERIODIC` (the default) runs it directly on
            `image`. Any other condition is approximated via pad-filter-crop (see
            `_apply_with_boundary_padding` and `_RIESZ_BASE_PAD`).
        source_mask: Optional boolean mask where True = valid voxel.
            When provided, zeros out invalid (sentinel) voxels before
            FFT-based transform to prevent contamination.

    Returns:
        Riesz-transformed image (real part)

    Raises:
        ValueError: If `order` sums to 0 (i.e. every component is 0), which
            would correspond to a zero-order (identity) transform, or if
            `boundary` is a string that is not a valid `BoundaryCondition`
            member name.

    Example:
        Compute first-order Riesz transform along the k1 axis:

        ```python
        import numpy as np
        from pictologics.filters import riesz_transform

        # Create dummy 3D image
        image = np.random.rand(50, 50, 50)

        # Apply transform (gradient-like along axis 0)
        response = riesz_transform(image, order=(1, 0, 0))
        ```

    Note:
        - First-order Riesz components form the image gradient
        - Second-order Riesz components form the image Hessian
        - All-pass: doesn't amplify high frequencies like regular derivatives
    """
    boundary = resolve_boundary(boundary)

    # Convert to float32
    image = ensure_float32(image)

    # Apply source_mask preprocessing (zero out invalid voxels for FFT-based filter)
    if source_mask is not None:
        image = _prepare_masked_image(image, source_mask)

    L = sum(order)  # Total order

    if L == 0:
        raise ValueError("At least one order component must be > 0")

    # Coerce order to a tuple first so a list-typed order (e.g. from a YAML/JSON
    # pipeline config) stays hashable for the transfer-function cache key.
    order = tuple(order)

    def _core(arr: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        shape = tuple(arr.shape)
        ndim = arr.ndim

        # Transfer function depends only on (shape, order) — never on image values
        # or the source mask — so it is built once and cached (see _riesz_transfer).
        transfer = _riesz_transfer(shape, order)

        # Apply in frequency domain using Real FFT. scipy.fft (multithreaded via
        # workers=-1) is several times faster than the single-threaded np.fft and
        # matches it to float32 precision.
        axes = tuple(range(ndim))
        F = scipy.fft.rfftn(arr, workers=-1)

        # F has shape (N1, N2, N3//2 + 1); transfer is broadcastable to it.
        response = scipy.fft.irfftn(F * transfer, s=shape, axes=axes, workers=-1)

        return cast(npt.NDArray[np.floating[Any]], response.astype(np.float32))

    return _apply_with_boundary_padding(_core, image, boundary, _RIESZ_BASE_PAD)


def _riesz_log_pad_width(
    sigma_mm: float,
    spacing_mm: Union[float, Tuple[float, float, float]],
    truncate: float,
) -> Tuple[int, ...]:
    """
    Default per-axis padding for boundary-aware Riesz-LoG filtering.

    Combines the LoG kernel's own truncation radius (`truncate * sigma` voxels,
    per axis, so anisotropic `spacing_mm` is respected) with `_RIESZ_BASE_PAD`
    (the margin the plain Riesz transform needs, see its module-level comment),
    since the padded array must accommodate both the LoG convolution's edge
    effects and the subsequent global Riesz FFT's boundary sensitivity.
    """
    spacing: Tuple[float, ...]
    if isinstance(spacing_mm, (int, float)):
        spacing = (float(spacing_mm),) * 3
    else:
        spacing = tuple(float(s) for s in spacing_mm)
    return tuple(int(np.ceil(truncate * sigma_mm / s)) + _RIESZ_BASE_PAD for s in spacing)


def riesz_log(
    image: npt.NDArray[np.floating[Any]],
    sigma_mm: float,
    spacing_mm: Union[float, Tuple[float, float, float]] = 1.0,
    order: Tuple[int, ...] = (1, 0, 0),
    truncate: float = 4.0,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.PERIODIC,
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.floating[Any]]:
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
        boundary: Boundary condition for the whole LoG-then-Riesz chain. The
            default `BoundaryCondition.PERIODIC` reproduces today's exact
            behaviour: the internal LoG call keeps its own default (ZERO padding)
            and the Riesz stage stays periodic, with no outer padding at all. Any
            other condition pads `image` once (see `_apply_with_boundary_padding`
            and `_riesz_log_pad_width`), runs the LoG-then-Riesz chain on the
            padded array, and crops back — and is *also* forwarded to the
            internal LoG call so its own edge handling matches the requested
            condition instead of silently staying at ZERO.
        source_mask: Optional boolean mask where True = valid voxel. Because
            `source_mask` shares `image`'s (unpadded) shape, the mid-chain
            re-zeroing this function otherwise performs before the Riesz stage is
            only applied when no padding occurs (the default `PERIODIC` case);
            for any other boundary, the mask is instead applied once to the final,
            already-cropped response.

    Returns:
        Riesz-transformed LoG response

    Raises:
        ValueError: If `boundary` is a string that is not a valid
            `BoundaryCondition` member name.

    Example:
        Compute first-order Riesz transform of LoG-filtered image at 5mm scale:

        ```python
        import numpy as np
        from pictologics.filters import riesz_log

        # Create dummy 3D image
        image = np.random.rand(50, 50, 50)

        # Apply filter
        response = riesz_log(
            image,
            sigma_mm=5.0,
            spacing_mm=(2.0, 2.0, 2.0),
            order=(1, 0, 0)
        )
        ```
    """
    from .log import laplacian_of_gaussian

    boundary = resolve_boundary(boundary)

    def _core(arr: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        # `_core` runs on `image` unchanged when boundary is PERIODIC (the default,
        # no padding), and on a *padded* array otherwise. `source_mask` always has
        # `image`'s original, unpadded shape, so it can only be forwarded to the
        # internal calls below in the PERIODIC case; the non-PERIODIC case masks
        # once, after cropping, below.
        if boundary is BoundaryCondition.PERIODIC:
            log_response = laplacian_of_gaussian(
                arr,
                sigma_mm=sigma_mm,
                spacing_mm=spacing_mm,
                truncate=truncate,
                source_mask=source_mask,
            )
            mask = source_mask
        else:
            # Also forward `boundary` here so LoG's own edge handling (ZERO by
            # default) matches the requested condition instead of silently
            # staying at ZERO.
            log_response = laplacian_of_gaussian(
                arr,
                sigma_mm=sigma_mm,
                spacing_mm=spacing_mm,
                truncate=truncate,
                boundary=boundary,
                source_mask=None,
            )
            mask = None

        # Handle tuple return from LoG if source_mask was used
        if isinstance(log_response, tuple):
            log_response = log_response[0]

        # Then apply Riesz transform. We pass the mask again (PERIODIC case only)
        # to enforce zeroing of invalid regions (though LoG normalized convolution
        # might have filled them, Riesz is global). The Riesz stage keeps its own
        # PERIODIC default: the outer pad-filter-crop below already accounts for
        # the boundary once for the whole chain.
        return riesz_transform(log_response, order=order, source_mask=mask)

    pad_width = _riesz_log_pad_width(sigma_mm, spacing_mm, truncate)
    result = _apply_with_boundary_padding(_core, image, boundary, pad_width)

    if source_mask is not None and boundary is not BoundaryCondition.PERIODIC:
        result = _prepare_masked_image(result, source_mask)

    return result


def riesz_simoncelli(
    image: npt.NDArray[np.floating[Any]],
    level: int = 1,
    order: Tuple[int, ...] = (1, 0, 0),
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.PERIODIC,
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.floating[Any]]:
    """
    Apply Riesz transform to Simoncelli wavelet-filtered image.

    Combines isotropic multi-scale analysis (Simoncelli) with
    directional analysis (Riesz) for rotation-invariant directional features.

    Args:
        image: 3D input image array
        level: Simoncelli decomposition level
        order: Riesz order tuple (l1, l2, l3)
        boundary: Boundary condition for the whole Simoncelli-then-Riesz chain.
            The default `BoundaryCondition.PERIODIC` reproduces today's exact
            behaviour (both stages run periodically, with no outer padding at
            all). Any other condition pads `image` once — covering both FFT
            stages with a single, uniform boundary treatment — runs the chain on
            the padded array (each stage keeping its own PERIODIC default), and
            crops back (see `_apply_with_boundary_padding`).
        source_mask: Optional boolean mask where True = valid voxel. Because
            `source_mask` shares `image`'s (unpadded) shape, the mid-chain
            re-zeroing this function otherwise performs before the Riesz stage is
            only applied when no padding occurs (the default `PERIODIC` case);
            for any other boundary, the mask is instead applied once to the final,
            already-cropped response.

    Returns:
        Riesz-transformed Simoncelli response

    Raises:
        ValueError: If `boundary` is a string that is not a valid
            `BoundaryCondition` member name.

    Example:
        Compute second-order Riesz transform (Hessian-like) of Simoncelli level 2:

        ```python
        import numpy as np
        from pictologics.filters import riesz_simoncelli

        # Create dummy 3D image
        image = np.random.rand(50, 50, 50)

        # Apply filter
        response = riesz_simoncelli(
            image,
            level=2,
            order=(2, 0, 0)
        )
        ```
    """
    from .wavelets import _simoncelli_pad_width, simoncelli_wavelet

    boundary = resolve_boundary(boundary)

    # Preprocess once: float32 conversion + source mask zeroing
    image = ensure_float32(image)
    if source_mask is not None:
        image = _prepare_masked_image(image, source_mask)

    def _core(arr: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        # Apply Simoncelli wavelet (already preprocessed, skip redundant work)
        sim_response = simoncelli_wavelet(arr, level=level)

        # Re-apply source_mask (PERIODIC case only, see docstring): Simoncelli's
        # global FFT spreads energy back into the invalid regions, and the Riesz
        # transform is likewise global, so re-zero before it (mirrors riesz_log).
        mask = source_mask if boundary is BoundaryCondition.PERIODIC else None
        return riesz_transform(sim_response, order=order, source_mask=mask)

    pad_width = _simoncelli_pad_width(level) + _RIESZ_BASE_PAD
    result = _apply_with_boundary_padding(_core, image, boundary, pad_width)

    if source_mask is not None and boundary is not BoundaryCondition.PERIODIC:
        result = _prepare_masked_image(result, source_mask)

    return result


def get_riesz_orders(max_order: int, ndim: int = 3) -> Tuple[Tuple[int, ...], ...]:
    """
    Generate all Riesz order tuples for a given maximum order.

    Args:
        max_order: Maximum total order L
        ndim: Number of dimensions (default 3)

    Returns:
        Tuple of all valid order tuples

    Example:
        Generate all second-order Riesz combinations for 3D:

        ```python
        from pictologics.filters.riesz import get_riesz_orders

        orders = get_riesz_orders(max_order=2, ndim=3)
        # Returns: ((2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), ...)
        ```
    """
    from itertools import combinations_with_replacement

    orders = []
    for combo in combinations_with_replacement(range(ndim), max_order):
        order = [0] * ndim
        for i in combo:
            order[i] += 1
        orders.append(tuple(order))

    return tuple(orders)
