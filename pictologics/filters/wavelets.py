# pictologics/filters/wavelets.py
"""Wavelet transform implementations (separable and non-separable)."""

import math
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
import pywt
import scipy.fft
from numba import get_num_threads
from numpy import typing as npt
from scipy.ndimage import convolve1d

from .base import (
    _TRANSFER_CACHE_BYTES,
    BoundaryCondition,
    _apply_with_boundary_padding,
    _float32_cut,
    _kept_rows,
    _ordered_map,
    _prepare_masked_image,
    _slab_pass,
    _slabs,
    _times_mirrored,
    _whole_number,
    cache_by_bytes,
    ensure_float32,
    get_scipy_mode,
    resolve_boundary,
)

# Threads for the 24 rotations from this size (voxels). Measured: they win from about
# 14,000 voxels (db2) or 4,000 voxels (coif3), and run 4-8x faster from 30,000 voxels.
_PARALLEL_THRESHOLD = 15_000


def wavelet_transform(
    image: npt.NDArray[np.floating[Any]],
    wavelet: str = "db2",
    level: int = 1,
    decomposition: str = "LHL",
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.ZERO,
    rotation_invariant: bool = False,
    pooling: str = "average",
    use_parallel: Union[bool, None] = None,
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.floating[Any]]:
    """
    Apply 3D separable wavelet transform (undecimated/stationary).

    Uses the à trous algorithm for undecimated wavelet decomposition.
    The transform is translation-invariant (unlike decimated transform).

    Supported wavelets:
        - "haar" (UOUE): Haar wavelet
        - "db2", "db3": Daubechies wavelets
        - "coif1": Coiflet wavelet

    Args:
        image: 3D input image array
        wavelet: Wavelet name (e.g., "db2", "coif1", "haar")
        level: Decomposition level (GCEK)
        decomposition: Which response map to return, e.g., "LHL", "HHH"
        boundary: Boundary condition for padding
        rotation_invariant: If True, average over 24 rotations
        pooling: Pooling method for rotation invariance
        use_parallel: If True, use parallel processing for rotation_invariant mode.
            If None (default), auto-enables for images > 15,000 voxels.
        source_mask: Optional boolean mask where True = valid voxel.
            When provided, zeros out invalid (sentinel) voxels before
            wavelet decomposition to prevent contamination.

    Returns:
        Response map for the specified decomposition

    Raises:
        ValueError: If `rotation_invariant=True` and `pooling` is not "max",
            "average", or "min".

    Example:
        Apply Daubechies 2 wavelet transform at level 1, returning LHL coefficients:

        ```python
        import numpy as np
        from pictologics.filters import wavelet_transform

        # Create dummy 3D image
        image = np.random.rand(50, 50, 50)

        # Apply transform
        response = wavelet_transform(
            image,
            wavelet="db2",
            level=1,
            decomposition="LHL"
        )
        ```
    """
    problem = _wavelet_problem(level, decomposition)
    if problem:
        raise ValueError(problem)
    level = int(level)
    decomposition = decomposition.upper()

    # Convert to float32
    image = ensure_float32(image)

    # Apply source_mask preprocessing (zero out invalid voxels)
    if source_mask is not None:
        image = _prepare_masked_image(image, source_mask)

    # Handle boundary
    if isinstance(boundary, str):
        boundary = BoundaryCondition[boundary.upper()]
    mode = get_scipy_mode(boundary)

    # Get wavelet filters
    w = pywt.Wavelet(wavelet)
    lo = np.array(w.dec_lo, dtype=np.float32)  # Low-pass decomposition filter
    hi = np.array(w.dec_hi, dtype=np.float32)  # High-pass decomposition filter

    # Auto-detect parallel mode based on image size
    if use_parallel is None:
        use_parallel = image.size > _PARALLEL_THRESHOLD

    if rotation_invariant:
        if pooling not in ("max", "average", "min"):
            raise ValueError(f"Unknown pooling: {pooling}")

        rotations = _get_rotation_perms()

        def apply_rotated_wavelet(
            rotation: Tuple[Tuple[int, int, int], Tuple[bool, bool, bool]],
        ) -> npt.NDArray[np.floating[Any]]:
            """Apply wavelet transform with rotated image."""
            perm, flips = rotation
            # Permute and flip image
            rotated = np.transpose(image, perm)
            for axis, flip in enumerate(flips):
                if flip:
                    rotated = np.flip(rotated, axis=axis)

            # Apply wavelet (one thread per rotation: the rotations run in a pool)
            response = _apply_undecimated_wavelet_3d(
                rotated, lo, hi, level, decomposition, mode, threads=1
            )

            # Undo rotation for response
            for axis, flip in enumerate(flips):
                if flip:
                    response = np.flip(response, axis=axis)
            inv_perm = tuple(np.argsort(perm))
            return np.transpose(response, inv_perm)

        result: npt.NDArray[np.floating[Any]] | None = None

        def _pool(response: npt.NDArray[np.floating[Any]]) -> None:
            nonlocal result
            if result is None:
                result = response.astype(np.float64) if pooling == "average" else response
            elif pooling == "max":
                np.maximum(result, response, out=result)
            elif pooling == "average":
                result += response
            else:  # "min"
                np.minimum(result, response, out=result)

        # Pool the responses in rotation order, so the result does not depend on thread
        # timing. At most `workers` rotations are in flight (and at most about 2 GB of
        # float64 responses), and each response is dropped once pooled. Small images
        # take one rotation at a time.
        workers = 1
        if use_parallel:
            workers = min(len(rotations), get_num_threads(), max(2, (2 << 30) // (8 * image.size)))
        for response in _ordered_map(apply_rotated_wavelet, rotations, workers):
            _pool(response)
            del response  # freed before the next rotation starts

        # Finalize average pooling
        if pooling == "average" and result is not None:
            result /= len(rotations)
        return result.astype(np.float32)  # type: ignore[union-attr]
    else:
        return _apply_undecimated_wavelet_3d(
            image, lo, hi, level, decomposition, mode, last_dtype=np.float32
        )


def _wavelet_problem(level: Any, decomposition: Any = None) -> Optional[str]:
    """Why a wavelet level (or a 3-D decomposition such as "LHL") is not valid, or None."""
    if not _whole_number(level) or level < 1:
        return f"level must be a whole number of 1 or more, not {level!r}"
    if decomposition is not None and not (
        isinstance(decomposition, str)
        and len(decomposition) == 3
        and set(decomposition.upper()) <= {"L", "H"}
    ):
        return (
            "decomposition must be 3 letters L or H, one for each axis (for example 'LHL'), "
            f"not {decomposition!r}"
        )
    return None


def _apply_undecimated_wavelet_3d(
    image: npt.NDArray[np.floating[Any]],
    lo: npt.NDArray[np.floating[Any]],
    hi: npt.NDArray[np.floating[Any]],
    level: int,
    decomposition: str,
    mode: str,
    threads: Optional[int] = None,
    last_dtype: Any = None,
) -> npt.NDArray[np.floating[Any]]:
    """
    Apply undecimated 3D wavelet decomposition using à trous algorithm. The passes run
    in `_slab_pass` with `threads` threads. With `last_dtype`, the last pass writes a
    new array of that type (scipy computes in double: the values of a cast after it).

    For level j, filters are upsampled by inserting 2^(j-1) - 1 zeros.
    """
    # The first pass writes a new array; the later passes write into it. convolve1d
    # copies each line before it writes that line, so the in-place result is the same.
    result: npt.NDArray[np.floating[Any]] | None = None
    for j in range(1, level + 1):
        # À trous: insert zeros into filters for this level
        if j > 1:
            lo_j = _atrous_upsample(lo, j)
            hi_j = _atrous_upsample(hi, j)
        else:
            lo_j = lo
            hi_j = hi

        # The low-pass (LLL) result feeds the next level; the final level applies the
        # requested decomposition.
        filters = {"L": lo_j, "H": hi_j}
        for axis, char in enumerate("LLL" if j < level else decomposition):
            weights = filters[char]
            if result is None:
                result = _slab_pass(
                    convolve1d, image, axis, None, threads, weights=weights, mode=mode
                )
            else:
                last = j == level and axis == 2 and last_dtype is not None
                target = np.empty(result.shape, dtype=last_dtype) if last else result
                result = _slab_pass(
                    convolve1d, result, axis, target, threads, weights=weights, mode=mode
                )
    return cast(npt.NDArray[np.floating[Any]], result)


def _atrous_upsample(
    kernel: npt.NDArray[np.floating[Any]], level: int
) -> npt.NDArray[np.floating[Any]]:
    """
    Upsample filter using à trous algorithm (insert zeros).

    For level j, insert 2^(j-1) - 1 zeros between each coefficient.
    IBSI recommends the second alternative (append zero at end).
    """
    factor = 2 ** (level - 1)
    new_len = len(kernel) + (len(kernel) - 1) * (factor - 1) + (factor - 1)
    upsampled = np.zeros(new_len, dtype=kernel.dtype)
    upsampled[::factor] = kernel

    return upsampled


def _get_rotation_perms() -> List[Tuple[Tuple[int, int, int], Tuple[bool, bool, bool]]]:
    """Get all 24 proper rotations of a cube (octahedral group)."""
    from .laws import _get_rotation_permutations_3d

    return _get_rotation_permutations_3d()


@cache_by_bytes(_TRANSFER_CACHE_BYTES)
def _simoncelli_transfer(shape: Tuple[int, ...], level: int) -> npt.NDArray[np.floating[Any]]:
    """Frequency-domain Simoncelli band-pass transfer function (IBSI 2 Eq. 27).

    Depends only on ``shape`` and ``level`` (never on image values or the source
    mask), so the result is cached and reused across calls with identical geometry.
    The returned array is marked read-only; callers must not mutate it.

    The table mirrors on every axis (row k equals row s - 1 - k for an even size s,
    row s - k for an odd one), so a large table keeps only its first (s + 1) // 2 rows
    on axis 0, half of the table (see _kept_rows and _times_mirrored; a mirror on a
    later axis would make the products slower).
    """
    # IBSI level N corresponds to j = N-1; level 1 = j=0 → max_freq = 1.0 (Nyquist).
    j = level - 1
    max_freq = 1.0 / (2**j)

    # Build frequency grid using centered [-1, 1] coordinates (IBSI 2 convention).
    # NOTE: This grid differs from np.fft.fftfreq by a factor of (N-1)/N.
    # The IBSI 2 reference values were validated with this specific grid, so
    # it must be preserved exactly. The non-symmetric grid for even N also
    # means rfftn/irfftn cannot be used (they assume conjugate symmetry).
    center = (np.array(shape) - 1.0) / 2.0

    grids = []
    for i, s in enumerate(shape):
        dim_grid = np.arange(s)
        # Normalize to [-1, 1] relative to center
        grid_norm = (dim_grid - center[i]) / center[i]
        # Shift to move DC to array start (index 0), matching fftn layout
        grid_shifted = np.fft.ifftshift(grid_norm)
        kept = _kept_rows(s, (s + 1) // 2, math.prod(shape)) if i == 0 else s
        grids.append(grid_shifted[:kept])
    mesh_vectors = np.meshgrid(*grids, indexing="ij", sparse=True)
    half_shape = tuple(len(g) for g in grids)

    # A large table is built slab by slab along the first axis, with the same element-wise
    # operations (so the same values) and temporaries of one slab instead of six full
    # volumes; the values outside the band stay 0.0, as np.where(mask, g, 0.0) gives.
    slabs = _slabs(half_shape)
    table = None if len(slabs) == 1 else np.zeros(half_shape, dtype=np.float64)
    for start, stop in slabs:
        dist_sq = np.asarray(
            sum((g[start:stop] if i == 0 else g) ** 2 for i, g in enumerate(mesh_vectors)),
            dtype=np.float64,
        )
        dist = np.sqrt(dist_sq)

        # Calculate transfer function (Simoncelli band-pass, IBSI 2 Eq. 27)
        val = 2.0 * dist / max_freq
        log_arg = np.where(val > 0, val, 1.0)

        with np.errstate(all="ignore"):
            g_sim = np.cos(np.pi / 2.0 * np.log2(log_arg))

        # Apply band-pass mask
        mask = (dist >= max_freq / 4.0) & (dist <= max_freq)
        if table is None:
            g_sim = np.where(mask, g_sim, 0.0)  # the whole table, as one slab
        else:
            np.copyto(table[start:stop], g_sim, where=mask)
    g_sim = g_sim if table is None else table
    g_sim.flags.writeable = False  # cached array must not be mutated by callers
    return cast(npt.NDArray[np.floating[Any]], g_sim)


def _simoncelli_pad_width(level: int) -> int:
    """
    Default padding (voxels, same for every axis) for boundary-aware Simoncelli
    filtering via pad-filter-crop.

    The Simoncelli transfer function (IBSI 2 Eq. 27) is smooth and band-limited to
    [max_freq/4, max_freq] with max_freq = 1 / 2**(level - 1) (Nyquist = 1), so its
    real-space kernel decays quickly, and — because the frequency band is a fixed
    *fraction* of Nyquist — the kernel's voxel extent depends only on `level`, not
    on the image shape. Empirically (via `np.fft.ifftn` of the transfer function),
    an 8-voxel radius captures ~99.9% of the level-1 kernel's L2 energy; each extra
    level halves the bandwidth (doubling the spatial extent), so the pad doubles
    too.
    """
    return 8 * (1 << (level - 1))


def simoncelli_wavelet(
    image: npt.NDArray[np.floating[Any]],
    level: int = 1,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.PERIODIC,
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.floating[Any]]:
    """
    Apply Simoncelli non-separable wavelet (IBSI code: PRT7).

    The Simoncelli wavelet is isotropic (spherically symmetric) and
    implemented in the Fourier domain. Per IBSI 2 Eq. 27.

    For decomposition level N, the frequency band is scaled by j = N-1:
        - Level 1 (j=0): band [π/4, π] (highest frequencies)
        - Level 2 (j=1): band [π/8, π/2]
        - Level 3 (j=2): band [π/16, π/4]

    Args:
        image: 3D input image array
        level: Decomposition level (1 = highest frequency band)
        boundary: Boundary condition. The filter is inherently periodic (FFT-based),
            so `BoundaryCondition.PERIODIC` (the default) runs it directly on
            `image`. Any other condition is approximated via pad-filter-crop (see
            `_apply_with_boundary_padding` and `_simoncelli_pad_width`).
        source_mask: Optional boolean mask where True = valid voxel

    Returns:
        Band-pass response map (B map) for the specified level

    Raises:
        ValueError: If `boundary` is a string that is not a valid
            `BoundaryCondition` member name.

    Example:
        Apply first-level Simoncelli wavelet (highest frequency band):

        ```python
        import numpy as np
        from pictologics.filters import simoncelli_wavelet

        # Create dummy 3D image
        image = np.random.rand(50, 50, 50)

        # Apply wavelet
        response = simoncelli_wavelet(image, level=1)
        ```
    """
    problem = _wavelet_problem(level)
    if problem:
        raise ValueError(problem)
    level = int(level)
    boundary = resolve_boundary(boundary)

    # Convert to float32
    image = ensure_float32(image)

    # Apply source_mask preprocessing (zero out invalid voxels for FFT-based filter)
    if source_mask is not None:
        image = _prepare_masked_image(image, source_mask)

    def _core(
        arr: npt.NDArray[np.floating[Any]], crop: Optional[Tuple[slice, ...]]
    ) -> npt.NDArray[np.floating[Any]]:
        shape = tuple(arr.shape)
        ndim = arr.ndim

        # Transfer function depends only on (shape, level) — never on image values or
        # the source mask — so it is built once and cached (see _simoncelli_transfer).
        g_sim = _simoncelli_transfer(shape, level)

        # Apply filter in frequency domain using full FFT (full FFT required because
        # the centered grid is non-symmetric for even N). scipy.fft with numba's thread
        # count is multithreaded and matches np.fft to float32 precision.
        axes = tuple(range(ndim))
        workers = get_num_threads()
        # Row k of the table mirrors row s - 1 - k (even size s) or s - k (odd s)
        offset = shape[0] - 1 + shape[0] % 2
        spectrum = _times_mirrored(scipy.fft.fftn(arr, workers=workers), g_sim, offset, 1)
        response = scipy.fft.ifftn(spectrum, s=shape, axes=axes, workers=workers, overwrite_x=True)

        return _float32_cut(np.real(response), crop)

    return _apply_with_boundary_padding(_core, image, boundary, _simoncelli_pad_width(level))
