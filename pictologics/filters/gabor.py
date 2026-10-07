# pictologics/filters/gabor.py
"""Gabor filter implementation (IBSI code: Q88H)."""

import math
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Callable, Optional, Tuple, Union, cast

import numba
import numpy as np
import scipy.fft
from llvmlite import ir
from numba import get_num_threads, jit, types
from numba.extending import intrinsic
from numpy import typing as npt

from .base import (
    _PAD_CODES,
    _PAD_MODES,
    BoundaryCondition,
    _copy_pass,
    _float32_cut,
    _padding_value_problem,
    _prepare_masked_image,
    _slab_pass,
    _slab_ufunc,
    _source_line,
    ensure_float32,
    get_scipy_mode,
    resolve_boundary,
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

# Slices for each FFT call (measured at 256^3: four are not faster, and take more memory)
_BLOCK_SLICES = 2
# The products of a block take at most this for each worker: the orientations go in groups
# that fit (512^2 slices with 16 orientations would hold 85 MB). Measured against no cap at
# 10 threads: the same time within the noise (x0.83 to x1.05), at 1 thread x0.99 to x1.00,
# and the memory of the code before the blocks; one orientation per group costs 5 % on 64^3
# images, so the budget holds the 16 orientations of a 64^2 slice (1.8 MB) in one group.
_PRODUCTS_BYTES = 4 << 20
# Slices whose values are all finite and at most this large make no NaN in the FFTs and the
# products: the sums stay below 1e34 with 4096 x 4096 slices and kernels of 4801 x 4801. A
# block of such slices takes one FFT call, as it gives the bits of an FFT for each slice.
_SAFE_LIMIT = 1e12


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
        - Returns the part of h = g ⊗ f that `response` names (default the modulus |h|)
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
    boundary = resolve_boundary(boundary)
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
        # Apply to all 3 orthogonal planes and average: the sum runs in float64, in plane
        # order. The copy, the sums, the division and the cast run on slabs in the slab
        # threads (each value as in one numpy call), as the plane responses of axes 1 and 2
        # are strided views.
        first = _plane(0)
        result = np.empty(first.shape, dtype=np.float64)
        _slab_pass(_copy_pass, first, first.ndim - 1, result)
        del first
        for plane_axis in (1, 2):
            _slab_ufunc(np.add, (result, _plane(plane_axis)), result)
        _slab_ufunc(np.true_divide, (result, 3.0), result)
        return _float32_cut(result, None)
    else:
        # Apply only to axial plane (axis 2 = k3 slices)
        return _plane(2)


@intrinsic  # type: ignore
def _fma32_intrinsic(typingctx: Any, a: Any, b: Any, c: Any) -> Any:
    """a * b + c of three float32 values with one rounding (llvm.fma), for the kernels."""
    sig = types.float32(types.float32, types.float32, types.float32)

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        fma = builder.module.declare_intrinsic("llvm.fma", [ir.FloatType()] * 3)
        return builder.call(fma, args)

    return sig, codegen


def _fma32_python(a: Any, b: Any, c: Any) -> np.float32:
    """The fma of the kernels with the compiler off: the exact float64 product plus c,
    then float32 (a second rounding in rare cases; the first-use checks find them)."""
    return np.float32(np.float64(a) * np.float64(b) + np.float64(c))


_fma32 = _fma32_python if numba.config.DISABLE_JIT else _fma32_intrinsic


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _pad_slices_numba(
    block: npt.NDArray[np.floating[Any]],
    buf: npt.NDArray[np.complex64],
    pad_h: int,
    pad_w: int,
    mode: int,
    value: float,
) -> bool:
    """buf[s] = slice s of `block` padded by pad_h lines and pad_w columns (np.pad in mode
    `mode`, see _PAD_CODES; `value` for the constant), then zeros to the FFT shape, as
    complex64. Returns whether every value of the block and `value` is finite and within
    _SAFE_LIMIT."""
    k, h, w = block.shape
    safe = abs(value) <= _SAFE_LIMIT
    for i in range(buf.shape[1]):
        si = _source_line(i - pad_h, h, mode) if i < h + 2 * pad_h else -2
        if si < 0:  # a line of the constant, or of the zeros after the padded slice
            fill = value if si == -1 else 0.0
            for j in range(buf.shape[2]):
                for s in range(k):
                    buf[s, i, j] = fill if j < w + 2 * pad_w else 0.0
            continue
        for j in range(pad_w):  # the left pad
            sj = _source_line(j - pad_w, w, mode)
            for s in range(k):
                buf[s, i, j] = value if sj < 0 else block[s, si, sj]
        for j in range(w):  # every value of the block is read here
            for s in range(k):
                x = block[s, si, j]
                safe &= abs(x) <= _SAFE_LIMIT
                buf[s, i, pad_w + j] = x
        for j in range(pad_w + w, buf.shape[2]):  # the right pad, then zeros
            sj = _source_line(j - pad_w, w, mode) if j < w + 2 * pad_w else -2
            for s in range(k):
                buf[s, i, j] = 0.0 if sj == -2 else (value if sj < 0 else block[s, si, sj])
    return safe


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _products_numba(
    spectra: npt.NDArray[np.complex64],
    kernels: npt.NDArray[np.complex64],
    products: npt.NDArray[np.complex64],
) -> None:
    """products[s * n + o] = spectra[s] * kernels[o] (n kernels), as numpy's complex64
    product: fma(ar, br, -(ai * bi)) + i fma(ar, bi, ai * br)."""
    n = kernels.shape[0]
    for s in range(spectra.shape[0]):
        for o in range(n):
            for i in range(spectra.shape[1]):
                for j in range(spectra.shape[2]):
                    a = spectra[s, i, j]
                    b = kernels[o, i, j]
                    products[s * n + o, i, j] = complex(
                        _fma32(a.real, b.real, -(a.imag * b.imag)),
                        _fma32(a.real, b.imag, a.imag * b.real),
                    )


@jit(nopython=True, inline="always", error_model="numpy", cache=True)  # type: ignore
def _modulus(z: complex) -> Any:
    """np.abs of a complex64 value with numpy's formula: an infinite part gives inf, a NaN
    part NaN, else larger * sqrt(fma(r, r, 1)) with r = smaller / larger. Selects, not
    branches, so that a row of values runs in vector lanes."""
    re = abs(z.real)
    im = abs(z.imag)
    larger = max(re, im)
    smaller = min(re, im)
    ratio = smaller / larger if larger > 0 else np.float32(0.0)
    modulus = np.sqrt(_fma32(ratio, ratio, np.float32(1.0))) * larger
    modulus = np.float32(np.nan) if (re != re or im != im) else modulus
    return np.float32(np.inf) if (re == np.inf or im == np.inf) else modulus


@jit(nopython=True, nogil=True, error_model="numpy", cache=True)  # type: ignore
def _pooled_parts_numba(
    products: npt.NDArray[np.complex64],
    n: int,
    r0: int,
    c0: int,
    part: int,
    pooling: int,
    first: bool,
    last: bool,
    total: int,
    acc: npt.NDArray[np.float64],
    out: npt.NDArray[np.float32],
) -> None:
    """Pool the part `part` (0 modulus, 1 real, 2 imaginary, 3 angle) of the products
    products[s * n + o] from line r0 and column c0 over the n orientations o of this group
    into out[s]: 0 average (a float64 sum in orientation order in `acc`, over `total`
    orientations, as float32 at the `last` group), 1 max, 2 min (a NaN wins; the running
    value in `out`). `first` starts the pooling of the block; one orientation gives the part
    itself. Each row runs in tight loops."""
    k, h, w = out.shape
    values = np.empty(w, dtype=np.float32)  # the part of a row
    for s in range(k):
        for i in range(h):
            for o in range(n):
                line = products[s * n + o, r0 + i, c0 : c0 + w]
                if part == 0:
                    for j in range(w):
                        values[j] = _modulus(line[j])
                elif part == 1:
                    for j in range(w):
                        values[j] = line[j].real
                elif part == 2:
                    for j in range(w):
                        values[j] = line[j].imag
                else:
                    for j in range(w):
                        values[j] = math.atan2(line[j].imag, line[j].real)
                if first and o == 0:
                    if pooling == 0:
                        for j in range(w):
                            acc[s, i, j] = values[j]
                    else:
                        for j in range(w):
                            out[s, i, j] = values[j]
                elif pooling == 0:
                    for j in range(w):
                        acc[s, i, j] += values[j]
                elif pooling == 1:
                    for j in range(w):
                        a = out[s, i, j]
                        b = values[j]
                        out[s, i, j] = a if (a >= b or a != a) else b
                else:
                    for j in range(w):
                        a = out[s, i, j]
                        b = values[j]
                        out[s, i, j] = a if (a <= b or a != a) else b
            if pooling == 0 and last:
                for j in range(w):
                    out[s, i, j] = acc[s, i, j] / total


def _check_values(special: Tuple[float, ...]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Two complex64 arrays of an odd length (numpy's vector loop and its tail both run):
    random values over 30 decades, then every pair of the `special` values."""
    rng = np.random.default_rng(0)
    values = np.array(special, dtype=np.float32)
    size = 4001 + values.size % 2
    arrays = []
    for _ in range(2):
        z = np.empty(size + values.size**2, dtype=np.complex64)
        for part, pairs in ((z.real, np.repeat(values, values.size)), (z.imag, np.tile(values, values.size))):  # fmt: skip
            part[:size] = rng.normal(size=size) * 10.0 ** rng.integers(-15, 15, size)
            part[size:] = pairs
        arrays.append(z)
    return arrays[0], arrays[1][::-1].copy()


# The products run on finite spectra only (see _SAFE_LIMIT); the parts also on NaN and inf
_FINITE_VALUES = (0.0, -0.0, 1.0, -1.5, 1e-40, -3e-39, 1e-45, 1e18, -3e17)
_SPECIAL_VALUES = _FINITE_VALUES + (math.inf, -math.inf, math.nan)


@lru_cache(maxsize=1)
def _products_exact() -> bool:
    """Whether _products_numba gives the bits of np.multiply on this CPU and numpy build."""
    a, b = _check_values(_FINITE_VALUES)
    got = np.empty((1, 1, a.size), dtype=np.complex64)
    _products_numba(a.reshape(1, 1, -1), b.reshape(1, 1, -1), got)
    return bool(np.multiply(a, b).tobytes() == got.tobytes())


@lru_cache(maxsize=4)
def _part_exact(part: int) -> bool:
    """Whether _pooled_parts_numba gives the bits of numpy's part `part` (see
    _RESPONSE_PARTS) on this CPU and numpy build."""
    a = _check_values(_SPECIAL_VALUES)[0]
    got = np.empty((1, 1, a.size), dtype=np.float32)
    with np.errstate(invalid="ignore"):  # inf / inf, when the compiler is off
        _pooled_parts_numba(a.reshape(1, 1, -1), 1, 0, 0, part, 1, True, True, 1, np.empty((0, 0, 0)), got)  # fmt: skip
    expected = np.ascontiguousarray(list(_RESPONSE_PARTS.values())[part](a))
    return expected.tobytes() == got.tobytes()


def _pooled(
    state: Optional[npt.NDArray[Any]], parts: list[npt.NDArray[np.float32]], pooling: str
) -> npt.NDArray[Any]:
    """`parts` (the parts of some orientations of one slice, in order) pooled into `state`
    (None before the first part) with numpy, as the loop of before: a float64 sum for the
    average, else the running max or min."""
    for response in parts:
        if state is None:
            state = response.astype(np.float64) if pooling == "average" else response
        elif pooling == "max":
            np.maximum(state, response, out=state)
        elif pooling == "average":
            state += response
        else:
            np.minimum(state, response, out=state)
    return cast(npt.NDArray[Any], state)


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

    The slices go in blocks of _BLOCK_SLICES: a numba kernel pads them into one complex64
    buffer, one FFT call transforms them, a numba kernel multiplies them with the kernel
    spectra of the orientations (in groups whose products fit _PRODUCTS_BYTES for each
    worker; one group as a rule), one inverse FFT call transforms the products of a group,
    and a numba kernel takes the response part, pools it and writes it into the one output
    array.
    The kernels give the bits of the numpy steps of each slice: the products and the parts
    use numpy's formulas (fma), and a first-use check takes numpy where a bit differs on
    this machine. Max and min pooling of a signed part keep numpy (it orders -0.0 and
    +0.0 in two ways). A block with a NaN, an infinite or a huge value (see _SAFE_LIMIT)
    takes an FFT for each slice and numpy's products, as their NaN bits depend on the
    steps.

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

    # Move the plane axis to position 0 so each image_reordered[i] is a 2D slice (a view)
    image_reordered = np.moveaxis(image, plane_axis, 0)
    n_slices = image_reordered.shape[0]
    slice_h, slice_w = int(image_reordered.shape[1]), int(image_reordered.shape[2])

    # All slices share one 2D shape and all kernels share one shape, so the FFT of each
    # padded slice serves every orientation, and the FFT of each kernel the whole plane.
    # This is fftconvolve(padded, k, "same") without the FFTs again for each kernel.
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

    # A circular convolution as long as the padded slice is enough: the kept part of the
    # output (the slice) is at least one kernel radius from the ends of the padded slice,
    # so no wrapped value reaches it. The crop of "same" + unpad is a fixed offset of
    # 2 * pad, as the kernel half-width equals the pad.
    fshape = (
        scipy.fft.next_fast_len(slice_h + 2 * pad_h),
        scipy.fft.next_fast_len(slice_w + 2 * pad_w),
    )
    kernel_ffts = np.stack([scipy.fft.fftn(k, s=fshape) for k in kernels])
    n = len(kernels)
    out_h, out_w = keep_h[1] - keep_h[0], keep_w[1] - keep_w[0]
    r0, c0 = 2 * pad_h + keep_h[0], 2 * pad_w + keep_w[0]
    out = np.empty((n_slices, out_h, out_w), dtype=np.float32)
    code = _PAD_CODES[_PAD_MODES.get(mode, "constant")]
    part = list(_RESPONSE_PARTS).index(response)
    take = _RESPONSE_PARTS[response]
    numba_products = _products_exact()
    numba_parts = _part_exact(part) and (n == 1 or pooling == "average" or response == "modulus")
    pool = ("average", "max", "min").index(pooling)
    # The orientations in groups whose products fit _PRODUCTS_BYTES for each worker
    group = max(1, min(n, _PRODUCTS_BYTES // (_BLOCK_SLICES * 8 * fshape[0] * fshape[1])))
    groups = [(g0, min(g0 + group, n)) for g0 in range(0, n, group)]

    def run(start: int, stop: int) -> None:
        """The slices start to stop, in blocks, with buffers for this task."""
        k = min(_BLOCK_SLICES, stop - start)
        buf = np.empty((k,) + fshape, dtype=np.complex64)
        products = np.empty((k * group,) + fshape, dtype=np.complex64)
        acc = np.empty((k, out_h, out_w) if numba_parts and pool == 0 else (0, 0, 0))
        for s in range(start, stop, k):
            m = min(k, stop - s)
            block = image_reordered[s : s + m]
            safe = _pad_slices_numba(block, buf[:m], pad_h, pad_w, code, float(padding_value))
            # An unsafe block (NaN, infinite or huge values) takes an FFT for each slice: the
            # NaN bits depend on the steps
            spectra = (
                scipy.fft.fftn(buf[:m], axes=(1, 2), overwrite_x=True)
                if safe
                else np.stack([scipy.fft.fftn(buf[q]) for q in range(m)])
            )
            states: list[Optional[npt.NDArray[Any]]] = [None] * m
            for g0, g1 in groups:
                ng = g1 - g0
                full = products[: m * ng]
                if not safe:
                    for q in range(m):
                        for o in range(ng):
                            full[q * ng + o] = scipy.fft.ifftn(spectra[q] * kernel_ffts[g0 + o])
                else:
                    if numba_products:
                        _products_numba(spectra, kernel_ffts[g0:g1], full)
                    else:
                        for q in range(m):
                            for o in range(ng):
                                np.multiply(spectra[q], kernel_ffts[g0 + o], out=full[q * ng + o])
                    full = scipy.fft.ifftn(full, axes=(1, 2), overwrite_x=True)
                if numba_parts:
                    _pooled_parts_numba(
                        full, ng, r0, c0, part, pool, g0 == 0, g1 == n, n, acc, out[s : s + m]
                    )
                else:
                    for q in range(m):
                        crops = (
                            full[q * ng + o, r0 : r0 + out_h, c0 : c0 + out_w] for o in range(ng)
                        )
                        parts = [np.ascontiguousarray(take(c)) for c in crops]
                        states[q] = _pooled(states[q], parts, pooling)
            if not numba_parts:
                for q in range(m):
                    state = cast(npt.NDArray[Any], states[q])
                    if pooling == "average" and n > 1:
                        state /= n
                    out[s + q] = state

    # One contiguous run of slices for each worker
    workers = get_num_threads() if use_parallel else 1
    step = max(1, math.ceil(n_slices / workers))
    starts = range(0, n_slices, step)
    if len(starts) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(lambda s: run(s, min(s + step, n_slices)), starts))
    else:
        run(0, n_slices)
    return np.moveaxis(out, 0, plane_axis)


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
