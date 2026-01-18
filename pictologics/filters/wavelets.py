# pictologics/filters/wavelets.py
"""Wavelet transform implementations (separable and non-separable)."""

from typing import Any, List, Tuple, Union

import numpy as np
import pywt
from numpy import typing as npt
from scipy.ndimage import convolve1d

from .base import BoundaryCondition, ensure_float32, get_scipy_mode

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

    Returns:
        Response map for the specified decomposition

    Example:
        >>> # Daubechies 2, first level, LHL coefficients
        >>> response = wavelet_transform(
        ...     image, wavelet="db2", level=1, decomposition="LHL"
        ... )
    """
    from concurrent.futures import ThreadPoolExecutor

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

                    # Fix mypy narrowing
                    # We can't import cast here easily repeatedly if we want to be clean,
                    # but we can rely on the assert with a variable assignment for mypy
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

    # This should never be reached (loop always returns on final level)
    # but is needed for type checker
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

    Returns:
        Band-pass response map (B map) for the specified level

    Example:
        >>> # First level Simoncelli wavelet (highest frequency)
        >>> response = simoncelli_wavelet(image, level=1)
    """
    # Convert to float32
    image = ensure_float32(image)

    shape = image.shape
    ndim = len(shape)

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

    # Optimize: Use rfftn (Real FFT) logic
    # The output of rfftn has shape (N1, N2, ..., N d//2 + 1)
    # We must construct the frequency grid to match this specific shape

    # Adjust the last dimension grid for rfftn
    # rfftn frequencies correspond to the first N//2 + 1 elements
    grids[-1] = grids[-1][: shape[-1] // 2 + 1]

    # Compute Euclidean distance via broadcasting (lazy evaluation)
    # meshgrid with sparse=True returns coordinate vectors that broadcast
    mesh_vectors = np.meshgrid(*grids, indexing="ij", sparse=True)

    # Compute dist^2 via broadcasting
    dist_sq = np.asarray(sum(g**2 for g in mesh_vectors), dtype=np.float64)
    dist = np.sqrt(dist_sq)

    # Avoid log(0) and divide by zero
    val = 2.0 * dist / max_freq
    log_arg = np.where(val > 0, val, 1.0)

    with np.errstate(all="ignore"):
        g_sim = np.cos(np.pi / 2.0 * np.log2(log_arg))

    # Apply band-pass mask
    mask = (dist >= max_freq / 4.0) & (dist <= max_freq)
    g_sim = np.where(mask, g_sim, 0.0)

    # Note: For rfftn we don't need standard ifftshift because
    # the concept of "centered" frequency domain is different for half-spectrum.
    # However, Simoncelli definition is inherently centered.
    #
    # The previous implementation built a centered spatial grid, calculated distance,
    # applied mask, and then ifftshifted to corner-based layout.
    #
    # With rfftn, the layout is:
    # Axis 0..d-2: Standard corner-based (0..N/2, -N/2..-1)
    # Axis d-1:    0..N/2 (non-negative only)
    #
    # The grid we built above is purely spatial/geometric [-1, 1].
    # But `simoncelli_wavelet` acts in frequency domain. The `grids` variables
    # actually represent normalized frequencies if we interpret `image` essentially
    # as being mapped to this domain.
    #
    # Issue: The `g_sim` we built is centered at (0,0,0) in our [-1,1] logic.
    # For full FFT, `ifftshift` moves (0,0,0) to corner.
    # For rfftn, we need to be careful.
    #
    # Actually, the original code used `ifftshift` on a grid defined from `arange(s)`.
    # Let's look closely at original `grids`:
    # `dim_grid = np.arange(s)` -> `(dim_grid - center)/center` -> [-1, 1]
    # This creates a spatial coordinate system centered at N/2.
    #
    # IBSI says Simoncelli is defined in frequency domain.
    # "frequency vector ν ... where |ν| is the distance from the origin (DC component)"
    #
    # The implementation constructs a grid that looks like spatial coordinates,
    # then shift it. This implies the grid IS the frequency grid.
    #
    # To use `rfftn`, we need the grid to match `rfftfreq` layout.
    # `fftfreq(N)` returns [0, 1, ..., N/2-1, -N/2, ..., -1] / N.
    # This corresponds to interval [0, 0.5] U [-0.5, 0).
    #
    # Our `grids` logic produces [-1, 1]. IBSI uses "normalized frequency" where Nyquist=1.0?
    # Original code: `max_freq = 1.0 / (2**j)`
    #
    # Let's align with `rfftfreq`:
    # Revert to "geometric" frequency grid [-1, 1] to match IBSI reference behavior.
    # The reference implementation used (arange(N) - center) / center, which
    # creates a grid that is slightly asymmetric/offset regarding DC for even N.
    # This asymmetry requires using full FFT (fftn) instead of Real FFT (rfftn),
    # as irfftn enforces conjugate symmetry which this grid violates.

    grids = []
    for i, s in enumerate(shape):
        dim_grid = np.arange(s)
        # Normalize to [-1, 1] relative to center
        grid_norm = (dim_grid - center[i]) / center[i]

        # Shift to move the "center" (DC-like area) to array start/corner
        grid_shifted = np.fft.ifftshift(grid_norm)
        grids.append(grid_shifted)

    # Use broadcasting for full 3D grid
    mesh_vectors = np.meshgrid(*grids, indexing="ij", sparse=True)
    dist_sq = np.asarray(sum(g**2 for g in mesh_vectors), dtype=np.float64)
    dist = np.sqrt(dist_sq)

    # Calculate transfer function (same as before)
    val = 2.0 * dist / max_freq
    log_arg = np.where(val > 0, val, 1.0)

    with np.errstate(all="ignore"):
        g_sim = np.cos(np.pi / 2.0 * np.log2(log_arg))

    # Apply band-pass mask
    mask = (dist >= max_freq / 4.0) & (dist <= max_freq)
    g_sim = np.where(mask, g_sim, 0.0)

    # No ifftshift needed because fftfreq/rfftfreq are already in corner layout!
    # (The previous implementation used spatial arange(N) which is monotonically increasing -1..1,
    # hence needed ifftshift to wrap 0 to corner. fftfreq generates wrapped layout directly).

    # Apply filter in frequency domain using full FFT
    F = np.fft.fftn(image)

    # Explicitly specify axes to avoid NumPy 2.0 DeprecationWarning
    axes = tuple(range(ndim))
    response = np.fft.ifftn(F * g_sim, s=shape, axes=axes)

    # Return magnitude (though theoretically real, minor complex errors exist)
    # IBSI reference suggests modulus? No, "The filter response is... convolution".
    # Wavelet response is usually real.
    return np.real(response).astype(np.float32)
