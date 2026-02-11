# pictologics/filters/wavelets.py
"""Wavelet transform implementations (separable and non-separable)."""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pywt
from numpy import typing as npt
from scipy.ndimage import convolve1d

from .base import (
    BoundaryCondition,
    _prepare_masked_image,
    ensure_float32,
    get_scipy_mode,
)

# Threshold for enabling parallel processing (voxels)
_PARALLEL_THRESHOLD = 2_000_000  # ~128³


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
            If None (default), auto-enables for images > ~128³ voxels.
        source_mask: Optional boolean mask where True = valid voxel.
            When provided, zeros out invalid (sentinel) voxels before
            wavelet decomposition to prevent contamination.

    Returns:
        Response map for the specified decomposition

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
    from concurrent.futures import ThreadPoolExecutor

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

            # Apply wavelet
            response = _apply_undecimated_wavelet_3d(
                rotated, lo, hi, level, decomposition, mode
            )

            # Undo rotation for response
            for axis, flip in enumerate(flips):
                if flip:
                    response = np.flip(response, axis=axis)
            inv_perm = tuple(np.argsort(perm))
            return np.transpose(response, inv_perm)

        if use_parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor() as executor:
                # Submit all rotation tasks
                future_to_rot = {
                    executor.submit(apply_rotated_wavelet, rot): rot
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
                        if result is None:  # pragma: no cover
                            raise RuntimeError("Result should not be None")

                        # Fix mypy narrowing issue
                        # Assert was sufficient
                        res = result

                        if pooling == "max":
                            np.maximum(res, response, out=res)
                        elif pooling == "average":
                            res += response
                        elif pooling == "min":
                            np.minimum(res, response, out=res)
                        else:
                            raise ValueError(
                                f"Unknown pooling: {pooling}"
                            )  # pragma: no cover

                    # Explicitly delete response
                    del response
        else:
            # Sequential processing for small images
            result = None
            for i, rotation in enumerate(rotations):
                response = apply_rotated_wavelet(rotation)

                if i == 0:
                    result = (
                        response.astype(np.float64)
                        if pooling == "average"
                        else response.copy()
                    )
                else:
                    if result is None:  # pragma: no cover
                        raise RuntimeError("Result should not be None")

                    res_seq = result

                    if pooling == "max":
                        np.maximum(res_seq, response, out=res_seq)
                    elif pooling == "average":
                        res_seq += response
                    elif pooling == "min":
                        np.minimum(res_seq, response, out=res_seq)
                    else:
                        raise ValueError(
                            f"Unknown pooling: {pooling}"
                        )  # pragma: no cover

        # Finalize average pooling
        if pooling == "average" and result is not None:
            result /= len(rotations)
        return result.astype(np.float32)  # type: ignore[union-attr]
    else:
        return _apply_undecimated_wavelet_3d(image, lo, hi, level, decomposition, mode)


def _apply_undecimated_wavelet_3d(
    image: npt.NDArray[np.floating[Any]],
    lo: npt.NDArray[np.floating[Any]],
    hi: npt.NDArray[np.floating[Any]],
    level: int,
    decomposition: str,
    mode: str,
) -> npt.NDArray[np.floating[Any]]:
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

    raise RuntimeError(
        "Unexpected end of wavelet decomposition loop"
    )  # pragma: no cover


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
        boundary: Boundary condition (FFT is inherently periodic)
        source_mask: Optional boolean mask where True = valid voxel

    Returns:
        Band-pass response map (B map) for the specified level

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
    # Convert to float32
    image = ensure_float32(image)

    # Apply source_mask preprocessing (zero out invalid voxels for FFT-based filter)
    if source_mask is not None:
        image = _prepare_masked_image(image, source_mask)

    shape = image.shape
    ndim = len(shape)

    # IBSI level N corresponds to j = N-1
    # Level 1 = j=0 → max_freq = 1.0 (normalized Nyquist)
    j = level - 1
    # Normalized max frequency for this level (relative to Nyquist=1.0)
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
        grids.append(grid_shifted)

    # Use broadcasting for full 3D grid
    mesh_vectors = np.meshgrid(*grids, indexing="ij", sparse=True)
    dist_sq = np.asarray(sum(g**2 for g in mesh_vectors), dtype=np.float64)
    dist = np.sqrt(dist_sq)

    # Calculate transfer function (Simoncelli band-pass, IBSI 2 Eq. 27)
    val = 2.0 * dist / max_freq
    log_arg = np.where(val > 0, val, 1.0)

    with np.errstate(all="ignore"):
        g_sim = np.cos(np.pi / 2.0 * np.log2(log_arg))

    # Apply band-pass mask
    mask = (dist >= max_freq / 4.0) & (dist <= max_freq)
    g_sim = np.where(mask, g_sim, 0.0)

    # Apply filter in frequency domain using full FFT
    # (full FFT required because the centered grid is non-symmetric for even N)
    F = np.fft.fftn(image)

    # Explicitly specify axes to avoid NumPy 2.0 DeprecationWarning
    axes = tuple(range(ndim))
    response = np.fft.ifftn(F * g_sim, s=shape, axes=axes)

    return np.real(response).astype(np.float32)
