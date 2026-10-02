# pictologics/filters/laws.py
"""Laws kernels filter implementation (IBSI code: JTXT)."""

import math
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast, overload

import numpy as np
from numba import get_num_threads
from numpy import typing as npt

from .base import (
    _SLAB_MIN_SIZE,
    BoundaryCondition,
    _convolve_axes,
    _float32_cut,
    _normalized_separable_convolve_3d,
    _ordered_map,
    _prepare_masked_image,
    _slab_ufunc,
    _uniform_filter,
    ensure_float32,
    get_scipy_mode,
)

# Normalized Laws kernels (IBSI 2 Table 6)
_LAWS_KERNELS: Dict[str, npt.NDArray[np.floating[Any]]] = {
    # Level (low-pass, averaging)
    "L3": np.array([1, 2, 1]) / math.sqrt(6),  # B5BZ
    "L5": np.array([1, 4, 6, 4, 1]) / math.sqrt(70),  # 6HRH
    # Edge (zero-mean, for detecting edges)
    "E3": np.array([-1, 0, 1]) / math.sqrt(2),  # LJ4T
    "E5": np.array([-1, -2, 0, 2, 1]) / math.sqrt(10),  # 2WPV
    # Spot (zero-mean, for detecting spots)
    "S3": np.array([-1, 2, -1]) / math.sqrt(6),  # MK5Z
    "S5": np.array([-1, 0, 2, 0, -1]) / math.sqrt(6),  # RXA1
    # Wave (zero-mean)
    "W5": np.array([-1, 2, 0, -2, 1]) / math.sqrt(10),  # 4ENO
    # Ripple (zero-mean)
    "R5": np.array([1, -4, 6, -4, 1]) / math.sqrt(70),  # 3A1W
}

LAWS_KERNELS = _LAWS_KERNELS
"""Dictionary of normalized Laws kernels (IBSI 2 Table 6)."""


# Threads for the rotations from this size (voxels). Measured: with 5-tap kernels the
# threads win from about 15,000-20,000 voxels (2x at 0.26-2M voxels); below, their start-up
# costs more than they save.
_PARALLEL_THRESHOLD = 20_000

# Rotation-invariant bases computed at a time, each with its share of the threads.
# Measured at 1.1M-7.1M voxels on 14 threads: 3 at a time (4 threads each) is as fast as
# or faster than 6 at a time, with 40-45 % less memory; 2 at a time is up to 7 % slower.
_BASES_IN_FLIGHT = 3


def _separable_convolve_3d(
    image: npt.NDArray[np.floating[Any]],
    g1: npt.NDArray[np.floating[Any]],
    g2: npt.NDArray[np.floating[Any]],
    g3: npt.NDArray[np.floating[Any]],
    mode: str = "constant",
    threads: Optional[int] = None,
    last_dtype: Any = None,
) -> npt.NDArray[np.floating[Any]]:
    """
    Apply separable 3D convolution using three 1D kernels, each pass in `threads`
    threads (default: numba's thread count). With `last_dtype`, the result has that
    type (see `_convolve_axes`).

    This is ~8x faster than full 3D convolution for 5x5x5 kernels:
    - Full 3D: 125 operations per voxel
    - Separable: 15 operations per voxel (3 × 5)

    Args:
        image: 3D input array
        g1, g2, g3: 1D kernels for axes 0, 1, 2
        mode: Boundary mode for scipy.ndimage.convolve1d

    Returns:
        Convolved 3D array
    """
    # Apply 1D convolutions sequentially along each axis. The first pass makes the
    # output array; the others write into it (convolve1d copies each line before it
    # writes that line, so the values are those of passes that make new arrays).
    result = _convolve_axes(image, (g1, g2, g3), mode, threads, last_dtype=last_dtype)
    return cast(npt.NDArray[np.floating[Any]], result)


def _pool_rotations(
    bases: Iterator[npt.NDArray[np.floating[Any]]],
    steps: List[Tuple[Tuple[str, str, str], int]],
    pooling: str,
) -> Optional[npt.NDArray[np.floating[Any]]]:
    """
    Pool the signed base responses in step order, each base as soon as it is ready.

    `bases` yields one response per key, in the order of the key's first step. For max
    and min pooling, a base changes its sign in place when a step needs the other sign
    (negation is exact); average pooling subtracts it instead. A base is freed after
    the last step of its key. The steps run in the calling thread, while the pool makes
    the next bases (the slab threads are busy with their passes).
    """
    last = {key: i for i, (key, _sign) in enumerate(steps)}
    held: Dict[Tuple[str, str, str], Tuple[npt.NDArray[np.floating[Any]], int]] = {}
    result: Optional[npt.NDArray[np.floating[Any]]] = None
    for i, (key, sign) in enumerate(steps):
        base, base_sign = held.pop(key) if key in held else (next(bases), 1)
        if pooling == "average":
            if result is None:  # the first step, the identity rotation, has sign 1
                kept = base.dtype == np.float64 and last[key] == i
                result = base if kept else base.astype(np.float64)
            elif sign > 0:
                result += base
            else:
                result -= base
        else:
            if base_sign != sign:
                np.negative(base, out=base)
                base_sign = sign
            if result is None:
                result = base if last[key] == i else base.copy()
            elif pooling == "max":
                np.maximum(result, base, out=result)
            else:  # "min"
                np.minimum(result, base, out=result)
        if last[key] > i:
            held[key] = (base, base_sign)
        del base
    return result


def _get_rotation_permutations_3d() -> List[Tuple[Tuple[int, int, int], Tuple[bool, bool, bool]]]:
    """
    Get all 24 right-angle rotation permutations for 3D (the octahedral group).

    Returns list of (axis_permutation, axis_flips) tuples.
    Each rotation is achieved by permuting axes and optionally flipping.
    """
    # All axis permutations
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    # For each permutation, we can flip 0, 1, or 2 axes (8 combinations)
    # But only determinant +1 rotations are valid (24 total, not 48)
    rotations = []
    for perm in perms:
        for f0 in [False, True]:
            for f1 in [False, True]:
                for f2 in [False, True]:
                    # Count flips - need even number for det=+1
                    n_flips = sum([f0, f1, f2])
                    # Perm sign: even perms (identity, 3-cycles) have sign +1
                    # odd perms (transpositions) have sign -1
                    perm_sign = 1 if _perm_parity(perm) == 0 else -1
                    flip_sign = 1 if n_flips % 2 == 0 else -1

                    if perm_sign * flip_sign == 1:  # det = +1
                        rotations.append((perm, (f0, f1, f2)))
    return rotations


def _perm_parity(perm: Tuple[int, int, int]) -> int:
    """Compute parity (0=even, 1=odd) of a permutation."""
    p = list(perm)
    parity = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                parity += 1
    return parity % 2


@overload
def laws_filter(
    image: npt.NDArray[np.floating[Any]],
    kernels: str,
    boundary: Union[BoundaryCondition, str] = ...,
    rotation_invariant: bool = ...,
    pooling: str = ...,
    compute_energy: bool = ...,
    energy_distance: int = ...,
    use_parallel: Union[bool, None] = ...,
    source_mask: None = ...,
) -> npt.NDArray[np.floating[Any]]: ...


@overload
def laws_filter(
    image: npt.NDArray[np.floating[Any]],
    kernels: str,
    boundary: Union[BoundaryCondition, str] = ...,
    rotation_invariant: bool = ...,
    pooling: str = ...,
    compute_energy: bool = ...,
    energy_distance: int = ...,
    use_parallel: Union[bool, None] = ...,
    source_mask: npt.NDArray[np.bool_] = ...,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]: ...


def laws_filter(
    image: npt.NDArray[np.floating[Any]],
    kernels: str,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.ZERO,
    rotation_invariant: bool = False,
    pooling: str = "max",
    compute_energy: bool = False,
    energy_distance: int = 7,
    use_parallel: Union[bool, None] = None,
    source_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> Union[
    npt.NDArray[np.floating[Any]],
    Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]],
]:
    """
    Apply 3D Laws kernel filter (IBSI code: JTXT).

    Laws kernels detect texture patterns via separable 1D filters combined
    into 2D/3D filters via outer products.

    Args:
        image: 3D input image array
        kernels: Kernel specification as string, e.g., "E5L5S5" for 3D
        boundary: Boundary condition for padding (GBYQ)
        rotation_invariant: If True, apply pseudo-rotational invariance (O1AQ)
                            using max pooling over 24 right-angle rotations
        pooling: Pooling method for rotation invariance ("max", "average", "min")
        compute_energy: If True, compute texture energy image (PQSD)
        energy_distance: Chebyshev distance δ for energy computation (I176)
        use_parallel: If True, use parallel processing for rotation_invariant mode.
            If None (default), auto-enables for images > 20,000 voxels.
            Only affects rotation_invariant mode.
        source_mask: Optional boolean mask where True = valid voxel.
            In non-rotation-invariant mode, uses normalized separable convolution
            to exclude invalid (sentinel) voxels. In rotation-invariant mode,
            invalid voxels are zero-filled as a first-order approximation (the
            rotated kernels preclude normalized convolution).

    Returns:
        If source_mask is None: Response map (or energy image if compute_energy=True)
        If source_mask provided: Tuple of (response_map, output_valid_mask)

    Raises:
        ValueError: If `kernels` does not parse into exactly 3 Laws kernel
            codes (wrong count or malformed string), if any parsed kernel
            code is not a recognized name (see `LAWS_KERNELS`), or if
            `rotation_invariant=True` and `pooling` is not "max", "average",
            or "min".
        RuntimeError: Defensive check raised if no response was computed;
            not expected to occur in normal use.

    Example:
        Apply Laws E5L5S5 kernel with rotation invariance and texture energy:

        ```python
        import numpy as np
        from pictologics.filters import laws_filter

        # Create dummy 3D image
        image = np.random.rand(50, 50, 50)

        # Apply filter
        response = laws_filter(
            image,
            "E5L5S5",
            rotation_invariant=True,
            pooling="max",
            compute_energy=True,
            energy_distance=7
        )
        ```

    Note:
        - Kernels are normalized (deviate from Laws' original unnormalized)
        - Energy is computed as: mean(|h|) over δ neighborhood
        - For rotation invariance, energy is computed after pooling
        - Uses separable 1D convolutions for ~8x speedup over full 3D
    """

    # Convert to float32
    image = ensure_float32(image)

    # Parse kernel names (e.g., "E5L5S5" -> ["E5", "L5", "S5"])
    kernel_names = _parse_kernel_string(kernels)
    if len(kernel_names) != 3:
        raise ValueError(f"Expected 3 kernel names for 3D, got {len(kernel_names)}: {kernel_names}")

    # Handle boundary condition
    if isinstance(boundary, str):
        boundary = BoundaryCondition[boundary.upper()]
    mode = get_scipy_mode(boundary)

    # Validate pooling method if used
    if rotation_invariant and pooling not in ("max", "average", "min"):
        raise ValueError(f"Unknown pooling method: {pooling}")

    # Get 1D kernels for separable convolution
    try:
        g1 = LAWS_KERNELS[kernel_names[0]].astype(np.float32)
        g2 = LAWS_KERNELS[kernel_names[1]].astype(np.float32)
        g3 = LAWS_KERNELS[kernel_names[2]].astype(np.float32)
    except KeyError as exc:
        valid = ", ".join(sorted(LAWS_KERNELS))
        raise ValueError(
            f"Unknown Laws kernel {exc.args[0]!r}; valid kernels are: {valid}"
        ) from exc

    # Auto-detect parallel mode based on image size
    if use_parallel is None:
        use_parallel = image.size > _PARALLEL_THRESHOLD

    result: npt.NDArray[np.floating[Any]] | None = None

    if rotation_invariant:
        # Normalized convolution isn't available on the rotated kernels, so zero-fill
        # invalid voxels as a first-order approximation (same approach the FFT-based
        # filters use). The output validity mask is the input mask (zero-fill does
        # not shrink it, unlike normalized convolution).
        if source_mask is not None:
            image = _prepare_masked_image(image, source_mask)
            valid_mask = source_mask
        else:
            valid_mask = None

        rotations = _get_rotation_permutations_3d()

        # Every Laws 1D kernel is odd-length and either symmetric (L, S, R) or
        # antisymmetric (E, W). Reversing a symmetric kernel is a no-op; reversing an
        # antisymmetric one negates it. So each of the 24 rotated separable
        # convolutions equals ±(a convolution with the kernels permuted but not
        # flipped). We therefore compute only the (at most 6) distinct permutations
        # and recover every rotation's response with a sign flip.
        kernel_arrays = [g1, g2, g3]
        antisym = [bool(np.allclose(k, -k[::-1])) for k in kernel_arrays]

        # One step per rotation: the permuted kernels (the key) and the sign.
        steps: List[Tuple[Tuple[str, str, str], int]] = []
        base_perms: Dict[Tuple[str, str, str], Tuple[int, int, int]] = {}
        for perm, flips in rotations:
            key = (kernel_names[perm[0]], kernel_names[perm[1]], kernel_names[perm[2]])
            sign = 1
            for i, do_flip in enumerate(flips):
                if do_flip and antisym[perm[i]]:
                    sign = -sign
            steps.append((key, sign))
            base_perms.setdefault(key, perm)
        if pooling != "average":
            # A (key, sign) step that comes again leaves a max or a min unchanged.
            steps = list(dict.fromkeys(steps))

        # The bases run in a pool (at most _BASES_IN_FLIGHT and about 2 GB of them at a
        # time), and each one has its share of the threads for its passes.
        threads = get_num_threads() if use_parallel else 1
        # A small image has no slab threads in its passes, so all its bases run at once
        in_flight = _BASES_IN_FLIGHT if image.size >= _SLAB_MIN_SIZE else threads
        workers = max(1, min(len(base_perms), in_flight, threads, (2 << 30) // (4 * image.size)))

        def _base(perm: Tuple[int, int, int]) -> npt.NDArray[np.floating[Any]]:
            return _separable_convolve_3d(
                image,
                kernel_arrays[perm[0]],
                kernel_arrays[perm[1]],
                kernel_arrays[perm[2]],
                mode,
                threads=threads // workers,
            )

        bases = _ordered_map(_base, base_perms.values(), workers)
        result = _pool_rotations(bases, steps, pooling)

        # Finalize average pooling (all bases are done: the slab threads are free)
        if pooling == "average" and result is not None:
            _slab_ufunc(np.true_divide, (result, len(rotations)), result)
    else:
        # Non-rotation-invariant: single separable convolution
        if source_mask is not None:
            result, valid_mask = _normalized_separable_convolve_3d(
                image, source_mask, g1, g2, g3, mode
            )
        else:
            # Without the energy step, the last pass gives the float32 response
            last_dtype = None if compute_energy else np.float32
            result = _separable_convolve_3d(image, g1, g2, g3, mode, last_dtype=last_dtype)
            valid_mask = None

    # Compute energy image if requested
    if compute_energy:
        if result is None:  # pragma: no cover
            raise RuntimeError("Result should not be None")

        # Energy = mean of absolute values over δ neighborhood, i.e. uniform_filter
        # on |result|. Accumulate in float64: scipy's running moving-sum otherwise
        # drifts in float32 over long axes. Cast the result back to float32. result is
        # always a new array of this function, so the steps write into it.
        np.abs(result, out=result)
        result = result.astype(np.float64, copy=False)
        energy_support = 2 * energy_distance + 1
        _uniform_filter(result, energy_support, mode, output=result)
        result = result.astype(np.float32)

    if result is None:  # pragma: no cover
        raise RuntimeError("Result should not be None")
    # float32, as the other filters: the rotation-invariant and masked paths pool or divide
    # in float64, and a large result copies in threads
    if result.dtype != np.float32:
        result = _float32_cut(result, None)

    if source_mask is not None and valid_mask is not None:
        return result, valid_mask
    return result  # type: ignore[no-any-return]


def _parse_kernel_string(kernels: str) -> List[str]:
    """
    Parse kernel string like "E5L5S5" into list ["E5", "L5", "S5"].
    """
    result = []
    i = 0
    while i < len(kernels):
        # Each kernel is a letter followed by a digit
        if i + 1 < len(kernels) and kernels[i].isalpha() and kernels[i + 1].isdigit():
            result.append(kernels[i : i + 2])
            i += 2
        else:
            raise ValueError(f"Cannot parse kernel string at position {i}: {kernels}")
    return result
