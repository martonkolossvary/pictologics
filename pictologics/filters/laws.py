# pictologics/filters/laws.py
"""Laws kernels filter implementation (IBSI code: JTXT)."""

import math
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.ndimage import convolve, uniform_filter

from .base import BoundaryCondition, ensure_float32, get_scipy_mode

# Normalized Laws kernels (IBSI 2 Table 6.x)
LAWS_KERNELS: Dict[str, np.ndarray] = {
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


def _build_3d_kernel(k1: str, k2: str, k3: str) -> np.ndarray:
    """
    Build a 3D Laws kernel from three 1D kernel names.

    The 3D kernel is the outer product of three 1D kernels.

    Args:
        k1: Kernel name for axis 0 (k1 direction)
        k2: Kernel name for axis 1 (k2 direction)
        k3: Kernel name for axis 2 (k3 direction)

    Returns:
        3D kernel array
    """
    g1 = LAWS_KERNELS[k1]
    g2 = LAWS_KERNELS[k2]
    g3 = LAWS_KERNELS[k3]

    # Outer product: first g1 ⊗ g2, then result ⊗ g3
    kernel_2d = np.outer(g1, g2)
    kernel_3d = np.outer(kernel_2d.flatten(), g3).reshape(len(g1), len(g2), len(g3))
    return kernel_3d


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


def _apply_rotation_to_kernel(
    kernel: np.ndarray, perm: Tuple[int, int, int], flips: Tuple[bool, bool, bool]
) -> np.ndarray:
    """Apply a rotation (permutation + flips) to a 3D kernel."""
    # First permute axes
    rotated = np.transpose(kernel, perm)
    # Then flip axes as needed
    for axis, do_flip in enumerate(flips):
        if do_flip:
            rotated = np.flip(rotated, axis=axis)
    return rotated


def laws_filter(
    image: np.ndarray,
    kernels: str,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.ZERO,
    rotation_invariant: bool = False,
    pooling: str = "max",
    compute_energy: bool = False,
    energy_distance: int = 7,
) -> np.ndarray:
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

    # Build the 3D kernel
    base_kernel = _build_3d_kernel(*kernel_names)

    if rotation_invariant:
        # Apply all 24 rotations and pool
        rotations = _get_rotation_permutations_3d()
        responses = []

        for perm, flips in rotations:
            rotated_kernel = _apply_rotation_to_kernel(base_kernel, perm, flips)
            response = convolve(image, rotated_kernel, mode=mode)
            responses.append(response)

        # Pool responses
        responses_arr = np.stack(responses, axis=0)
        if pooling == "max":
            result = np.max(responses_arr, axis=0)
        elif pooling == "average":
            result = np.mean(responses_arr, axis=0)
        elif pooling == "min":
            result = np.min(responses_arr, axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
    else:
        result = convolve(image, base_kernel, mode=mode)

    # Compute energy image if requested
    if compute_energy:
        # Energy = mean of absolute values over δ neighborhood
        # This is equivalent to uniform_filter on |result|
        abs_result = np.abs(result)
        energy_support = 2 * energy_distance + 1
        result = uniform_filter(abs_result, size=energy_support, mode=mode)

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
