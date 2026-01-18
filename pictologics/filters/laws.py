# pictologics/filters/laws.py
"""Laws kernels filter implementation (IBSI code: JTXT)."""

import math
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from numpy import typing as npt
from scipy.ndimage import uniform_filter

from .base import BoundaryCondition, ensure_float32, get_scipy_mode

# Normalized Laws kernels (IBSI 2 Table 6.x)
LAWS_KERNELS: Dict[str, npt.NDArray[np.floating[Any]]] = {
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


# Threshold for enabling parallel processing (voxels)
_PARALLEL_THRESHOLD = 2_000_000  # ~128³


def _separable_convolve_3d(
    image: npt.NDArray[np.floating[Any]],
    g1: npt.NDArray[np.floating[Any]],
    g2: npt.NDArray[np.floating[Any]],
    g3: npt.NDArray[np.floating[Any]],
    mode: str = "constant",
) -> npt.NDArray[np.floating[Any]]:
    """
    Apply separable 3D convolution using three 1D kernels.

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
    from scipy.ndimage import convolve1d

    # Apply 1D convolutions sequentially along each axis
    result = convolve1d(image, g1, axis=0, mode=mode)
    result = convolve1d(result, g2, axis=1, mode=mode)
    result = convolve1d(result, g3, axis=2, mode=mode)
    # Explicit cast to fix MyPy 'no-any-return'
    from typing import cast

    return cast(npt.NDArray[np.floating[Any]], result.astype(image.dtype))


def _get_rotation_permutations_3d() -> (
    List[Tuple[Tuple[int, int, int], Tuple[bool, bool, bool]]]
):
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


def laws_filter(
    image: npt.NDArray[np.floating[Any]],
    kernels: str,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.ZERO,
    rotation_invariant: bool = False,
    pooling: str = "max",
    compute_energy: bool = False,
    energy_distance: int = 7,
    use_parallel: Union[bool, None] = None,
) -> npt.NDArray[np.floating[Any]]:
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
            If None (default), auto-enables for images > ~128³ voxels.
            Only affects rotation_invariant mode.

    Returns:
        Response map (or energy image if compute_energy=True)

    Example:
        >>> # E5L5S5 with rotation invariance and energy
        >>> response = laws_filter(
        ...     image, "E5L5S5",
        ...     rotation_invariant=True, pooling="max",
        ...     compute_energy=True, energy_distance=7
        ... )

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
        raise ValueError(
            f"Expected 3 kernel names for 3D, got {len(kernel_names)}: {kernel_names}"
        )

    # Handle boundary condition
    if isinstance(boundary, str):
        boundary = BoundaryCondition[boundary.upper()]
    mode = get_scipy_mode(boundary)

    # Validate pooling method if used
    if rotation_invariant and pooling not in ("max", "average", "min"):
        raise ValueError(f"Unknown pooling method: {pooling}")

    # Get 1D kernels for separable convolution
    g1 = LAWS_KERNELS[kernel_names[0]].astype(np.float32)
    g2 = LAWS_KERNELS[kernel_names[1]].astype(np.float32)
    g3 = LAWS_KERNELS[kernel_names[2]].astype(np.float32)

    # Auto-detect parallel mode based on image size
    if use_parallel is None:
        use_parallel = image.size > _PARALLEL_THRESHOLD

    if rotation_invariant:
        rotations = _get_rotation_permutations_3d()

        def apply_rotated_convolution(
            rotation: Tuple[Tuple[int, int, int], Tuple[bool, bool, bool]],
        ) -> npt.NDArray[np.floating[Any]]:
            """Apply separable convolution with rotated kernels."""
            perm, flips = rotation
            # Permute kernel order to match rotation
            rotated_kernels = [g1, g2, g3]
            rotated_kernels = [rotated_kernels[p] for p in perm]
            # Flip kernels as needed
            for i, do_flip in enumerate(flips):
                if do_flip:
                    rotated_kernels[i] = rotated_kernels[i][::-1].copy()
            # Apply separable convolution
            return _separable_convolve_3d(
                image, rotated_kernels[0], rotated_kernels[1], rotated_kernels[2], mode
            )

        if use_parallel:
            # Parallel processing for large images
            # Use as_completed to process results as they finish, avoiding
            # holding all 24 response maps in memory at once.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor() as executor:
                # Submit all rotation tasks
                future_to_rot = {
                    executor.submit(apply_rotated_convolution, rot): rot
                    for rot in rotations
                }

                # Pool responses incrementally
                result: npt.NDArray[np.floating[Any]] | None = None
                for _, future in enumerate(as_completed(future_to_rot)):
                    response = future.result()

                    if result is None:
                        # Initialize accumulator with first result
                        result = (
                            response.astype(np.float64)
                            if pooling == "average"
                            else response.copy()
                        )
                    else:
                        # At this point, result is guaranteed to be not None because
                        # the loop iterates at least once and the first iteration sets it.
                        if result is None:  # pragma: no cover
                            raise RuntimeError("Result should not be None")

                        # Fix mypy narrowing issue
                        # Assert was enough
                        res = result

                        if pooling == "max":
                            np.maximum(res, response, out=res)
                        elif pooling == "average":
                            res += response
                        elif pooling == "min":
                            np.minimum(res, response, out=res)
                        else:
                            raise ValueError(
                                f"Unknown pooling method: {pooling}"
                            )  # pragma: no cover

                    # Explicitly delete response to free memory
                    del response
        else:
            # Sequential processing for small images (avoid thread overhead)
            result = None
            for i, rotation in enumerate(rotations):
                response = apply_rotated_convolution(rotation)

                if i == 0:
                    result = (
                        response.astype(np.float64)
                        if pooling == "average"
                        else response.copy()
                    )
                else:
                    if result is None:  # pragma: no cover
                        raise RuntimeError("Result should not be None")

                    # Fix mypy narrowing
                    res = result

                    if pooling == "max":
                        np.maximum(res, response, out=res)
                    elif pooling == "average":
                        res += response
                    elif pooling == "min":
                        np.minimum(res, response, out=res)
                    else:
                        raise ValueError(
                            f"Unknown pooling method: {pooling}"
                        )  # pragma: no cover

        # Finalize average pooling
        if pooling == "average" and result is not None:
            result /= len(rotations)
    else:
        # Non-rotation-invariant: single separable convolution
        result = _separable_convolve_3d(image, g1, g2, g3, mode)

    # Compute energy image if requested
    if compute_energy:
        if result is None:  # pragma: no cover
            raise RuntimeError("Result should not be None")

        # Energy = mean of absolute values over δ neighborhood
        # This is equivalent to uniform_filter on |result|
        abs_result = np.abs(result)
        energy_support = 2 * energy_distance + 1
        result = uniform_filter(abs_result, size=energy_support, mode=mode)

    if result is None:  # pragma: no cover
        raise RuntimeError("Result should not be None")

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
