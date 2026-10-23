# pictologics/filters/riesz.py
"""Riesz transform implementation (IBSI code: AYRS)."""

from math import factorial, prod, sqrt
from typing import Any, Optional, Tuple, Union, cast

import numpy as np
import scipy.fft
from numba import get_num_threads, jit, prange
from numpy import typing as npt

from .base import (
    _TABLE_KERNEL_MIN,
    _TRANSFER_CACHE_SHAPES,
    BoundaryCondition,
    _apply_with_boundary_padding,
    _bool_mask,
    _fast_width,
    _fft_gaussian_laplace,
    _float32_cut,
    _irfftn_unscaled,
    _kept_rows,
    _log_parameter_problem,
    _padding_value_problem,
    _prepare_masked_image,
    _slabs,
    _times_mirrored,
    _transfer_cache_bytes,
    _whole_number,
    cache_by_bytes,
    ensure_float32,
    get_scipy_mode,
    resolve_boundary,
)

# Default padding (voxels, same for every axis) for boundary-aware Riesz filtering
# via pad-filter-crop. The Riesz transfer function has unit magnitude everywhere
# (an all-pass, phase-only filter, IBSI 2 Eq. 34), so unlike the band-limited
# Simoncelli kernel its real-space kernel is not compactly supported: it decays
# as a power law, similar to a generalised Hilbert transform. 16 voxels hold 92-100 %
# of the L2 energy of a first- or second-order kernel (97.8 % for the order (1, 0, 0)),
# but the L2 energy is the wrong measure of the error: the tail adds up over a large
# uniform object. Measured against a 64-voxel pad (each pad widened to a fast FFT size,
# see `_fast_width`), the 16-voxel pad of the order (1, 0, 0) differs by up to 8.5e-2 of
# the response range with the mirror boundary and by up to 3.0e-2 with the zero boundary,
# at the border of a CT scan; farther than 16 voxels from the border, by at most 5.1e-2
# with the mirror boundary (1.1e-2 on average). No finite pad removes the truncation
# error, so 16 is a balance between accuracy and the cost of a larger FFT array. The
# periodic boundary (the IBSI default) takes no pad.
_RIESZ_BASE_PAD = 16


@cache_by_bytes(_transfer_cache_bytes, _TRANSFER_CACHE_SHAPES)
def _riesz_transfer(
    shape: Tuple[int, ...], order: Tuple[int, ...]
) -> npt.NDArray[np.complexfloating[Any, Any]]:
    """Riesz frequency-domain transfer function (IBSI 2 Eq. 34).

    Depends only on ``shape`` and ``order`` (never on image values or the source
    mask), so it is cached and reused across calls with identical geometry —
    including the many order tuples from ``get_riesz_orders`` that share one image
    shape. The returned array is marked read-only; callers must not mutate it.

    On axis 0 (when it is not the rfft axis), row k of the table equals row s - k
    times (-1)**order[0] (the frequency changes its sign), so a large table keeps only
    its first s // 2 + 1 rows there, about half of the table (see _kept_rows and
    _times_mirrored).
    """
    ndim = len(shape)
    L = sum(order)

    # Frequency coordinates for rfftn: the last axis is non-negative freqs only.
    freqs = []
    for i, s in enumerate(shape):
        if i == ndim - 1:
            freqs.append(np.fft.rfftfreq(s) * 2 * np.pi)
        elif i == 0:
            cells = prod(shape[:-1]) * (shape[-1] // 2 + 1)
            freqs.append((np.fft.fftfreq(s) * 2 * np.pi)[: _kept_rows(s, s // 2 + 1, cells)])
        else:
            freqs.append(np.fft.fftfreq(s) * 2 * np.pi)

    # Broadcast (sparse) grid to avoid a full meshgrid the size of the input.
    nu_vectors = np.meshgrid(*freqs, indexing="ij", sparse=True)

    # A large table is built slab by slab along the first axis, with the same element-wise
    # operations (so the same values) and temporaries of one slab instead of four full
    # volumes.
    out_shape = np.broadcast_shapes(*(n.shape for n in nu_vectors))
    table = np.empty(out_shape, dtype=np.complex128)
    if ndim == 3 and table.size >= _TABLE_KERNEL_MIN:
        # A large 3-D table: one parallel kernel, the same bits (see _riesz_table)
        scale = _riesz_scale(order, L)
        _riesz_table(*freqs, *order, L, scale.real, scale.imag, table)
    else:
        for start, stop in _slabs(out_shape):
            nu = [n[start:stop] if i == 0 else n for i, n in enumerate(nu_vectors)]
            _riesz_values(nu, order, L, out=table[start:stop])
    table[(0,) * table.ndim] = 0  # the DC value is +0
    table.flags.writeable = False  # cached array must not be mutated by callers
    return cast(npt.NDArray[np.complexfloating[Any, Any]], table)


def _riesz_values(
    nu: list[npt.NDArray[np.float64]],
    order: Tuple[int, ...],
    L: int,
    out: Optional[npt.NDArray[np.complex128]] = None,
) -> npt.NDArray[np.complex128]:
    """The Riesz transfer values (IBSI 2 Eq. 34) at the broadcast frequency vectors `nu`
    of each axis (radians), for the total order L, written into `out` when given. At DC
    the numerator is 0 (L >= 1), so the value is a zero there (of either sign)."""
    nu_sq_norm = np.asarray(sum(n**2 for n in nu), dtype=np.float64)
    nu_norm = np.sqrt(nu_sq_norm)
    nu_norm_safe = np.where(nu_norm > 0, nu_norm, 1.0)  # avoid /0 at DC
    numerator = np.ones(nu_norm.shape, dtype=np.float64)
    for i, ord_val in enumerate(order):
        if ord_val > 0:
            numerator *= nu[i] ** ord_val
    # (phase * norm) * numerator / |nu|^L, in `out` (or one new array): no complex temporary
    values = np.multiply(_riesz_scale(order, L), numerator, out=out)
    return cast(npt.NDArray[np.complex128], np.divide(values, nu_norm_safe**L, out=values))


def _riesz_scale(order: Tuple[int, ...], L: int) -> complex:
    """The constant factor of the Riesz values (IBSI 2 Eq. 34): the phase (-i)**L times the
    norm sqrt(L! / (l1! l2! l3!))."""
    norm_factor = sqrt(factorial(L) / np.prod([factorial(o) for o in order]))
    return complex(np.exp(-1j * np.pi * L / 2) * norm_factor)


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _riesz_power(x: float, n: int) -> float:
    """x ** n as numpy computes it for an array: x for n = 1, x * x for n = 2, else pow."""
    if n == 1:
        return x
    if n == 2:
        return x * x
    return float(np.power(x, float(n)))


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _riesz_table(
    f0: npt.NDArray[np.float64],
    f1: npt.NDArray[np.float64],
    f2: npt.NDArray[np.float64],
    o0: int,
    o1: int,
    o2: int,
    L: int,
    scale_re: float,
    scale_im: float,
    out: npt.NDArray[np.complex128],
) -> None:
    """The Riesz values (see _riesz_values) on the frequency vectors f0, f1, f2 of a 3-D
    table. Each value has the operations of the numpy build in the same order, so the same
    bits: the complex product scale * (numerator + 0j), then numpy's complex division by
    (|nu|**L + 0j), that is a ratio 0 / d and the products with the reciprocal 1 / d."""
    n0, n1, n2 = out.shape
    for i in prange(n0):
        a = f0[i]
        for j in range(n1):
            b = f1[j]
            ab = a * a + b * b
            for k in range(n2):
                c = f2[k]
                norm = np.sqrt(ab + c * c)
                safe = norm if norm > 0 else 1.0
                numerator = 1.0
                if o0 > 0:
                    numerator *= _riesz_power(a, o0)
                if o1 > 0:
                    numerator *= _riesz_power(b, o1)
                if o2 > 0:
                    numerator *= _riesz_power(c, o2)
                d = _riesz_power(safe, L)
                re = scale_re * numerator - scale_im * 0.0
                im = scale_re * 0.0 + scale_im * numerator
                ratio = 0.0 / d
                reciprocal = 1.0 / (d + 0.0 * ratio)
                out[i, j, k] = complex(
                    (re + im * ratio) * reciprocal, (im - re * ratio) * reciprocal
                )


def riesz_transform(
    image: npt.NDArray[np.floating[Any]],
    order: Tuple[int, ...],
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.PERIODIC,
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
    padding_value: float = 0.0,
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
        padding_value: The constant of constant value padding (Z3VE), with the ZERO
            (constant) boundary. Default 0.

    Returns:
        Riesz-transformed image (real part)

    Raises:
        ValueError: If `order` is not one whole number of 0 or more for each axis with a
            sum above 0 (a sum of 0 is the identity), if `boundary` is a string that is not
            a valid `BoundaryCondition` member name, or if `padding_value` is not 0 with a
            boundary other than the constant one.

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
    source_mask = _bool_mask(source_mask)  # a mask of another type: != 0
    boundary = resolve_boundary(boundary)
    problem = _padding_value_problem(boundary, padding_value)
    if problem:
        raise ValueError(problem)

    # Convert to float32
    image = ensure_float32(image)

    # Apply source_mask preprocessing (zero out invalid voxels for FFT-based filter)
    if source_mask is not None:
        image = _prepare_masked_image(image, source_mask)

    order = _riesz_order(order, image.ndim)
    return _apply_with_boundary_padding(
        _riesz_response, image, boundary, _RIESZ_BASE_PAD, padding_value, order=order
    )


def _riesz_order_problem(order: Any, ndim: int) -> Optional[str]:
    """Why `order` is not a Riesz order of an image with `ndim` axes, or None."""
    if not (
        isinstance(order, (list, tuple))
        and len(order) == ndim
        and all(_whole_number(o) and o >= 0 for o in order)
    ):
        return f"order must be {ndim} whole numbers of 0 or more, one for each axis, not {order!r}"
    if sum(order) == 0:  # the total order
        return "At least one order component must be > 0"
    return None


def _riesz_order(order: Any, ndim: int) -> Tuple[int, ...]:
    """`order` as a tuple of ints, so that a list-typed order (e.g. from a YAML/JSON
    pipeline config) stays hashable for the transfer-function cache key."""
    problem = _riesz_order_problem(order, ndim)
    if problem:
        raise ValueError(problem)
    return tuple(int(o) for o in order)


def _riesz_response(
    arr: npt.NDArray[np.floating[Any]],
    crop: Optional[Tuple[slice, ...]],
    order: Tuple[int, ...],
    band: Optional[npt.NDArray[np.floating[Any]]] = None,
) -> npt.NDArray[np.floating[Any]]:
    """The (periodic) Riesz transform of `arr` as float32, cut to `crop` (None: the
    whole array). A `band` table (rfftn layout, every row) filters the spectrum first,
    in the same FFT round trip."""
    shape = tuple(arr.shape)

    # Transfer function depends only on (shape, order) — never on image values
    # or the source mask — so it is built once and cached (see _riesz_transfer).
    transfer = _riesz_transfer(shape, order)

    # Apply in frequency domain using Real FFT. scipy.fft (multithreaded, with
    # numba's thread count) is several times faster than the single-threaded np.fft
    # and matches it to float32 precision.
    workers = get_num_threads()
    # The spectrum has shape (N1, N2, N3//2 + 1), the shape of the transfer.
    spectrum = scipy.fft.rfftn(arr, workers=workers)
    if band is not None:
        spectrum = _times_mirrored(spectrum, band, shape[0], 1)
    sign = (-1) ** order[0]
    spectrum = _times_mirrored(spectrum, transfer, shape[0], sign)
    response, scale = _irfftn_unscaled(spectrum, shape, workers)
    del spectrum  # freed before the cut: the peak holds the response and the cut
    return _float32_cut(response, crop, scale)


def _spacing_tuple(spacing_mm: Union[float, Tuple[float, ...]]) -> Tuple[float, ...]:
    """The spacing as one float for each axis (a single number: the same for the 3 axes)."""
    if isinstance(spacing_mm, (int, float)):
        return (float(spacing_mm),) * 3
    return tuple(float(s) for s in spacing_mm)


def _riesz_log_fft(
    image: npt.NDArray[np.floating[Any]],
    sigma_mm: float,
    spacing_mm: Union[float, Tuple[float, ...]],
    truncate: float,
    order: Tuple[int, ...],
    boundary: BoundaryCondition,
    pad_width: Tuple[int, ...],
    padding_value: float,
) -> npt.NDArray[np.float32]:
    """The LoG then the Riesz transform of a 3-D `image` in one FFT round trip, for a
    boundary other than PERIODIC. The image is padded once with the boundary (the constant
    `padding_value` for ZERO) by at least `pad_width` (at most the axis length, as
    `_apply_with_boundary_padding` does) and at least the LoG radius on both sides of each
    axis, widened to a fast FFT size of the same parity (an even axis has a Nyquist plane,
    where an odd-order Riesz transfer is lost). Its spectrum is multiplied by the LoG
    transfer and the Riesz table, transformed back and cut to the image, as float32.
    A float32 or integer image is filtered as float64."""
    as_float64 = cast(npt.NDArray[np.float64], ensure_float32(image).astype(np.float64, copy=False))
    sigmas = tuple(sigma_mm / s for s in _spacing_tuple(spacing_mm))
    widths = tuple(
        (w, w)
        for w in (
            _fast_width(n, max(min(p, n), int(truncate * s + 0.5)))
            for n, p, s in zip(as_float64.shape, pad_width, sigmas, strict=True)
        )
    )
    sign = (-1) ** order[0]

    def riesz(sizes: Tuple[int, ...]) -> Any:
        table = _riesz_transfer(sizes, order)
        return lambda spectrum: _times_mirrored(spectrum, table, sizes[0], sign)

    return cast(
        npt.NDArray[np.float32],
        _fft_gaussian_laplace(
            as_float64, sigmas, get_scipy_mode(boundary), truncate, np.float32, widths,
            padding_value, riesz,
        ),
    )  # fmt: skip


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
    return tuple(
        int(np.ceil(truncate * sigma_mm / s)) + _RIESZ_BASE_PAD for s in _spacing_tuple(spacing_mm)
    )


def riesz_log(
    image: npt.NDArray[np.floating[Any]],
    sigma_mm: float,
    spacing_mm: Union[float, Tuple[float, float, float]] = 1.0,
    order: Tuple[int, ...] = (1, 0, 0),
    truncate: float = 4.0,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.PERIODIC,
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
    padding_value: float = 0.0,
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
        boundary: Boundary condition for the whole LoG-then-Riesz chain. PERIODIC (the
            default) runs the LoG with its own ZERO boundary, then the Riesz transform on
            the image as it is. Another boundary pads the image once with that boundary
            (see `_riesz_log_pad_width`, widened to a fast FFT size of the same parity).
            It runs the LoG and the Riesz transform in one FFT round trip on the padded
            image, and cuts the result back. A 2-D image keeps the two stages.
        source_mask: Optional boolean mask where True = valid voxel. With PERIODIC, the
            LoG uses normalized convolution, and the mask zeroes the invalid voxels again
            before the Riesz stage. With another boundary, the mask zeroes the invalid
            voxels of the final response.
        padding_value: The constant of constant value padding (Z3VE), with the ZERO
            (constant) boundary. Default 0.

    Returns:
        Riesz-transformed LoG response

    Raises:
        ValueError: If `order` is not valid (see `riesz_transform`), if `boundary` is a
            string that is not a valid `BoundaryCondition` member name, or if
            `padding_value` is not 0 with a boundary other than the constant one. Also
            if `sigma_mm`, an entry of `spacing_mm` or `truncate` is not a finite
            positive number.

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
    source_mask = _bool_mask(source_mask)  # a mask of another type: != 0
    from .log import laplacian_of_gaussian

    boundary = resolve_boundary(boundary)
    problem = _padding_value_problem(boundary, padding_value) or _log_parameter_problem(
        sigma_mm, _spacing_tuple(spacing_mm), truncate
    )
    if problem:
        raise ValueError(problem)
    order = _riesz_order(order, image.ndim)
    pad_width = _riesz_log_pad_width(sigma_mm, spacing_mm, truncate)
    if boundary is not BoundaryCondition.PERIODIC and image.ndim == 3:
        result = _riesz_log_fft(
            image, sigma_mm, spacing_mm, truncate, order, boundary, pad_width, padding_value
        )
        return result if source_mask is None else _prepare_masked_image(result, source_mask)

    def _core(
        arr: npt.NDArray[np.floating[Any]], crop: Optional[Tuple[slice, ...]]
    ) -> npt.NDArray[np.floating[Any]]:
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

        # Then apply Riesz transform. The mask zeroes the invalid regions again
        # (PERIODIC case only; though LoG normalized convolution might have filled
        # them, Riesz is global). The Riesz stage is periodic: the outer
        # pad-filter-crop below already accounts for the boundary once for the chain.
        if mask is not None:
            log_response = _prepare_masked_image(log_response, mask)
        return _riesz_response(log_response, crop, order)

    result = _apply_with_boundary_padding(_core, image, boundary, pad_width, padding_value)

    if source_mask is not None and boundary is not BoundaryCondition.PERIODIC:
        result = _prepare_masked_image(result, source_mask)  # a 2-D image

    return result


def riesz_simoncelli(
    image: npt.NDArray[np.floating[Any]],
    level: int = 1,
    order: Tuple[int, ...] = (1, 0, 0),
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.PERIODIC,
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
    padding_value: float = 0.0,
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
        padding_value: The constant of constant value padding (Z3VE), with the ZERO
            (constant) boundary. Default 0.

    Returns:
        Riesz-transformed Simoncelli response

    Raises:
        ValueError: If `level` or `order` is not valid (see `simoncelli_wavelet` and
            `riesz_transform`), if `boundary` is a string that is not a valid
            `BoundaryCondition` member name, or if `padding_value` is not 0 with a
            boundary other than the constant one.

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
    source_mask = _bool_mask(source_mask)  # a mask of another type: != 0
    from .wavelets import _simoncelli_pad_width, _simoncelli_transfer, simoncelli_wavelet

    boundary = resolve_boundary(boundary)
    problem = _padding_value_problem(boundary, padding_value)
    if problem:
        raise ValueError(problem)
    order = _riesz_order(order, image.ndim)

    # Preprocess once: float32 conversion + source mask zeroing
    image = ensure_float32(image)
    if source_mask is not None:
        image = _prepare_masked_image(image, source_mask)

    def _core(
        arr: npt.NDArray[np.floating[Any]], crop: Optional[Tuple[slice, ...]]
    ) -> npt.NDArray[np.floating[Any]]:
        # Re-apply source_mask (PERIODIC case only, see docstring): Simoncelli's
        # global FFT spreads energy back into the invalid regions, and the Riesz
        # transform is likewise global, so re-zero before it (mirrors riesz_log).
        if source_mask is not None and boundary is BoundaryCondition.PERIODIC:
            sim_response = simoncelli_wavelet(arr, level=level)
            sim_response = _prepare_masked_image(sim_response, source_mask)
            return _riesz_response(sim_response, crop, order)
        # Else both filters in one FFT round trip (see _simoncelli_transfer)
        return _riesz_response(arr, crop, order, _simoncelli_transfer(tuple(arr.shape), level))

    pad_width = _simoncelli_pad_width(level) + _RIESZ_BASE_PAD
    result = _apply_with_boundary_padding(_core, image, boundary, pad_width, padding_value)

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
