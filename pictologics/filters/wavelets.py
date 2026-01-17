# pictologics/filters/wavelets.py
"""Wavelet transform implementations (separable and non-separable)."""

from typing import List, Tuple, Union

import numpy as np
import pywt
from scipy.ndimage import convolve1d

from .base import BoundaryCondition, ensure_float32, get_scipy_mode


def wavelet_transform(
    image: np.ndarray,
    wavelet: str = "db2",
    level: int = 1,
    decomposition: str = "LHL",
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.ZERO,
    rotation_invariant: bool = False,
    pooling: str = "average",
) -> np.ndarray:
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

    Returns:
        Response map for the specified decomposition

    Example:
        >>> # Daubechies 2, first level, LHL coefficients
        >>> response = wavelet_transform(
        ...     image, wavelet="db2", level=1, decomposition="LHL"
        ... )
    """
    # Convert to float32
    image = ensure_float32(image)

    # Handle boundary
    if isinstance(boundary, str):
        boundary = BoundaryCondition[boundary.upper()]
    mode = get_scipy_mode(boundary)

    # Get wavelet filters
    w = pywt.Wavelet(wavelet)
    lo = np.array(w.dec_lo, dtype=np.float32)  # Low-pass decomposition filter
    hi = np.array(w.dec_hi, dtype=np.float32)  # High-pass decomposition filter

    if rotation_invariant:
        # Apply to all 24 right-angle rotations and pool
        responses = []
        for perm, flips in _get_rotation_perms():
            # Permute and flip image
            rotated = np.transpose(image, perm)
            for axis, flip in enumerate(flips):
                if flip:
                    rotated = np.flip(rotated, axis=axis)

            # Apply wavelet
            response = _apply_undecimated_wavelet_3d(
                rotated, lo, hi, level, decomposition, mode
            )

            # Undo rotation for response
            for axis, flip in enumerate(flips):
                if flip:
                    response = np.flip(response, axis=axis)
            inv_perm = tuple(np.argsort(perm))
            response = np.transpose(response, inv_perm)

            responses.append(response)

        # Pool
        stacked = np.stack(responses, axis=0)
        if pooling == "average":
            return np.mean(stacked, axis=0)  # type: ignore[no-any-return]
        elif pooling == "max":
            return np.max(stacked, axis=0)  # type: ignore[no-any-return]
        elif pooling == "min":
            return np.min(stacked, axis=0)  # type: ignore[no-any-return]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
    else:
        return _apply_undecimated_wavelet_3d(image, lo, hi, level, decomposition, mode)


def _apply_undecimated_wavelet_3d(
    image: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    level: int,
    decomposition: str,
    mode: str,
) -> np.ndarray:
    """
    Apply undecimated 3D wavelet decomposition using à trous algorithm.

    For level j, filters are upsampled by inserting 2^(j-1) - 1 zeros.
    """
    current = image.copy()

    for j in range(1, level + 1):
        # À trous: insert zeros into filters for this level
        if j > 1:
            lo_j = _atrous_upsample(lo, j)
            hi_j = _atrous_upsample(hi, j)
        else:
            lo_j = lo
            hi_j = hi

        # Store the low-pass result for next iteration
        # We only need to track LLL for multi-level decomposition
        if j < level:
            # Apply low-pass along all 3 axes
            current = convolve1d(current, lo_j, axis=0, mode=mode)
            current = convolve1d(current, lo_j, axis=1, mode=mode)
            current = convolve1d(current, lo_j, axis=2, mode=mode)
        else:
            # Final level: compute requested decomposition
            filters = {"L": lo_j, "H": hi_j}
            result = current.copy()
            for axis, char in enumerate(decomposition):
                result = convolve1d(result, filters[char], axis=axis, mode=mode)
            return result

    # This should never be reached (loop always returns on final level)
    # but is needed for type checker
    raise RuntimeError(
        "Unexpected end of wavelet decomposition loop"
    )  # pragma: no cover


def _atrous_upsample(kernel: np.ndarray, level: int) -> np.ndarray:
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


def simoncelli_wavelet(
    image: np.ndarray,
    level: int = 1,
    boundary: Union[BoundaryCondition, str] = BoundaryCondition.PERIODIC,
) -> np.ndarray:
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
        boundary: Boundary condition (FFT is inherently periodic)

    Returns:
        Band-pass response map (B map) for the specified level

    Example:
        >>> # First level Simoncelli wavelet (highest frequency)
        >>> response = simoncelli_wavelet(image, level=1)
    """
    # Convert to float32
    image = ensure_float32(image)

    shape = image.shape

    # IBSI level N corresponds to j = N-1
    # Level 1 = j=0 → max_freq = 1.0 (normalized Nyquist)
    j = level - 1
    # Normalized max frequency for this level (relative to Nyquist=1.0)
    max_freq = 1.0 / (2**j)

    # Use centered grid coordinates [-1, 1] relative to geometric center (N-1)/2
    center = (np.array(shape) - 1.0) / 2.0

    # Generate value grid for each dimension
    grids = []
    for i, s in enumerate(shape):
        dim_grid = np.arange(s)
        # Normalize to [-1, 1] relative to center
        grids.append((dim_grid - center[i]) / center[i])

    # Compute Euclidean distance in the centered frequency domain
    mesh = np.meshgrid(*grids, indexing="ij")
    dist = np.sqrt(sum(g**2 for g in mesh))

    # Avoid log(0) and divide by zero
    val = 2.0 * dist / max_freq
    log_arg = np.where(val > 0, val, 1.0)

    with np.errstate(all="ignore"):
        g_sim = np.cos(np.pi / 2.0 * np.log2(log_arg))

    # Apply band-pass mask
    mask = (dist >= max_freq / 4.0) & (dist <= max_freq)
    g_sim = np.where(mask, g_sim, 0.0)

    # Since the mask is defined on a centered grid, we must shift it
    # to the DFT corner-based layout before multiplying with FFT of image
    g_sim_shifted = np.fft.ifftshift(g_sim)

    # Apply filter in frequency domain
    F = np.fft.fftn(image)
    response = np.real(np.fft.ifftn(F * g_sim_shifted))

    return response.astype(np.float32)
