"""
Image Preprocessing Module
==========================

This module provides a collection of preprocessing functions essential for radiomics
analysis. These functions are designed to be IBSI-compliant where applicable.

Key Features:
-------------
- **Resampling**: Voxel resampling using 'Align grid centers' (IBSI compliant).
- **Discretisation**: Fixed Bin Number (FBN) and Fixed Bin Size (FBS) algorithms.
- **Filtering**: Outlier filtering (mean +/- sigma).
- **Mask Operations**: Resegmentation (thresholding), ROI extraction, Largest Connected Component.
- **Utilities**: Rounding intensities, applying masks.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, cast

import numpy as np
from numba import jit, prange
from numpy import typing as npt
from scipy.ndimage import affine_transform, generate_binary_structure, label

from .features._utils import compute_nonzero_bbox
from .loader import Image, _direction_matrix, _validate_geometry

# Common sentinel values used in medical imaging to denote "no data" or "background"
COMMON_SENTINEL_VALUES: tuple[float, ...] = (
    -2048.0,
    -3024.0,
    -1024.0,
    -1000.0,
    0.0,
    -32768.0,
)

# Below this size the numba parallel-launch overhead exceeds the scan itself;
# the numpy fallback paths are kept for small arrays (same convention as
# features._utils). Discretise/resegment dispatch on it; both paths are
# bit-identical, so the switch is purely a performance decision.
_KERNEL_MIN_SIZE = 1 << 20


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------
# Single-pass parallel replacements for the memory-bound numpy chains below.
# The discretise/resegment kernels apply the exact same per-element operations
# (in the same order) as the numpy code they shadow, so their output is
# bit-identical; uncommon dtypes fall back to numpy. All signatures used by
# the dispatch code are compiled eagerly in warmup.warmup_jit().


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _discretise_fbn_numba(
    flat: npt.NDArray[np.float64],
    n_bins: float,
    current_min: float,
    current_max: float,
    out: npt.NDArray[np.int32],
) -> None:
    """FBN: floor(N_g * (x - min) / (max - min)) + 1, clipped to [1, N_g]; NaN -> 0."""
    denom = current_max - current_min
    for i in prange(flat.size):
        t = n_bins * (flat[i] - current_min)
        t = t / denom
        t = np.floor(t) + 1.0
        if t < 1.0:
            t = 1.0
        elif t > n_bins:
            t = n_bins
        if np.isnan(t):
            out[i] = 0
        else:
            out[i] = np.int32(t)


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _discretise_fbs_numba(
    flat: npt.NDArray[np.float64],
    bin_width: float,
    current_min: float,
    out: npt.NDArray[np.int32],
) -> None:
    """FBS: floor((x - min) / w_b) + 1, clamped to >= 1; NaN -> 0."""
    for i in prange(flat.size):
        t = (flat[i] - current_min) / bin_width
        t = np.floor(t) + 1.0
        if t < 1.0:
            t = 1.0
        if np.isnan(t):
            out[i] = 0
        else:
            out[i] = np.int32(t)


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _resegment_numba(
    img_flat: npt.NDArray[np.float64],
    mask_flat: npt.NDArray[Any],
    range_min: float,
    range_max: float,
    out: npt.NDArray[Any],
) -> None:
    """Copy mask, zeroing voxels whose intensity lies outside [range_min, range_max].

    NaN intensities compare False on both bounds and therefore keep their mask
    value, matching the numpy fallback. Missing bounds are passed as +/-inf.
    """
    for i in prange(img_flat.size):
        v = img_flat[i]
        if v < range_min or v > range_max:
            out[i] = 0
        else:
            out[i] = mask_flat[i]


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _resample_nearest_numba(
    src: npt.NDArray[Any],
    scale: npt.NDArray[np.float64],
    shift: npt.NDArray[np.float64],
    out: npt.NDArray[Any],
) -> None:
    """Nearest-neighbour resampling for a diagonal transform, 'nearest' boundary.

    Replicates scipy's zoom_shift path (used by affine_transform for 1-D
    matrices) exactly: the coordinate is (idx + shift) * scale with
    shift = offset / scale, clipped to the volume, then floor(coord + 0.5)
    selects the voxel. Same expressions and order -> identical output.
    """
    nz, ny, nx = src.shape
    oz, oy, ox = out.shape
    for k in prange(oz):
        zc = (k + shift[0]) * scale[0]
        if zc < 0.0:
            zc = 0.0
        elif zc > nz - 1:
            zc = float(nz - 1)
        iz = int(np.floor(zc + 0.5))
        for j in range(oy):
            yc = (j + shift[1]) * scale[1]
            if yc < 0.0:
                yc = 0.0
            elif yc > ny - 1:
                yc = float(ny - 1)
            iy = int(np.floor(yc + 0.5))
            for i in range(ox):
                xc = (i + shift[2]) * scale[2]
                if xc < 0.0:
                    xc = 0.0
                elif xc > nx - 1:
                    xc = float(nx - 1)
                ix = int(np.floor(xc + 0.5))
                out[k, j, i] = src[iz, iy, ix]


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _resample_trilinear_numba(
    src: npt.NDArray[np.float64],
    scale: npt.NDArray[np.float64],
    shift: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
) -> None:
    """Trilinear resampling for a diagonal transform, 'nearest' boundary.

    Replicates scipy's zoom_shift path: coordinate (idx + shift) * scale
    (shift = offset / scale) clipped to the volume, base index floor(coord),
    upper support index clamped to the edge, and the 8 support terms
    accumulated in scipy's point order with left-to-right weight products.
    """
    nz, ny, nx = src.shape
    oz, oy, ox = out.shape
    for k in prange(oz):
        zc = (k + shift[0]) * scale[0]
        if zc < 0.0:
            zc = 0.0
        elif zc > nz - 1:
            zc = float(nz - 1)
        z0 = int(np.floor(zc))
        tz = zc - z0
        z1 = z0 + 1 if z0 + 1 <= nz - 1 else nz - 1
        wz0 = 1.0 - tz
        for j in range(oy):
            yc = (j + shift[1]) * scale[1]
            if yc < 0.0:
                yc = 0.0
            elif yc > ny - 1:
                yc = float(ny - 1)
            y0 = int(np.floor(yc))
            ty = yc - y0
            y1 = y0 + 1 if y0 + 1 <= ny - 1 else ny - 1
            wy0 = 1.0 - ty
            for i in range(ox):
                xc = (i + shift[2]) * scale[2]
                if xc < 0.0:
                    xc = 0.0
                elif xc > nx - 1:
                    xc = float(nx - 1)
                x0 = int(np.floor(xc))
                tx = xc - x0
                x1 = x0 + 1 if x0 + 1 <= nx - 1 else nx - 1
                wx0 = 1.0 - tx

                # scipy multiplies each coefficient by the per-axis weights
                # sequentially (value * wz * wy * wx, left-associated); keep
                # that exact order so the result is bit-identical.
                acc = src[z0, y0, x0] * wz0 * wy0 * wx0
                acc += src[z0, y0, x1] * wz0 * wy0 * tx
                acc += src[z0, y1, x0] * wz0 * ty * wx0
                acc += src[z0, y1, x1] * wz0 * ty * tx
                acc += src[z1, y0, x0] * tz * wy0 * wx0
                acc += src[z1, y0, x1] * tz * wy0 * tx
                acc += src[z1, y1, x0] * tz * ty * wx0
                acc += src[z1, y1, x1] * tz * ty * tx
                out[k, j, i] = acc


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _resample_trilinear_masked_numba(
    src: npt.NDArray[np.float64],
    valid: npt.NDArray[np.bool_],
    scale: npt.NDArray[np.float64],
    shift: npt.NDArray[np.float64],
    weight_threshold: float,
    out: npt.NDArray[np.float64],
    out_valid: npt.NDArray[np.bool_],
) -> None:
    """Fused normalized (validity-weighted) trilinear resampling.

    Computes in one pass what _resample_with_source_mask obtains from two
    affine_transform calls: per output voxel, the trilinear weights of the 8
    neighbours are accumulated only over valid source voxels; the value is the
    weight-normalized sum, and the voxel is valid iff the accumulated weight
    reaches weight_threshold. Coordinates use scipy's zoom_shift formula (see
    _resample_trilinear_numba).
    """
    nz, ny, nx = src.shape
    oz, oy, ox = out.shape
    for k in prange(oz):
        zc = (k + shift[0]) * scale[0]
        if zc < 0.0:
            zc = 0.0
        elif zc > nz - 1:
            zc = float(nz - 1)
        z0 = int(np.floor(zc))
        tz = zc - z0
        z1 = z0 + 1 if z0 + 1 <= nz - 1 else nz - 1
        wz0 = 1.0 - tz
        for j in range(oy):
            yc = (j + shift[1]) * scale[1]
            if yc < 0.0:
                yc = 0.0
            elif yc > ny - 1:
                yc = float(ny - 1)
            y0 = int(np.floor(yc))
            ty = yc - y0
            y1 = y0 + 1 if y0 + 1 <= ny - 1 else ny - 1
            wy0 = 1.0 - ty
            for i in range(ox):
                xc = (i + shift[2]) * scale[2]
                if xc < 0.0:
                    xc = 0.0
                elif xc > nx - 1:
                    xc = float(nx - 1)
                x0 = int(np.floor(xc))
                tx = xc - x0
                x1 = x0 + 1 if x0 + 1 <= nx - 1 else nx - 1
                wx0 = 1.0 - tx

                # Accumulation mirrors the two affine_transform calls of the
                # scipy path bit-for-bit: values multiply the per-axis weights
                # sequentially (left-associated), the weight sum matches the
                # transform of the 0/1 mask, and invalid voxels contribute
                # exactly 0 to both sums (as the zeroed image does there).
                acc = 0.0
                wacc = 0.0
                if valid[z0, y0, x0]:
                    acc += src[z0, y0, x0] * wz0 * wy0 * wx0
                    wacc += wz0 * wy0 * wx0
                if valid[z0, y0, x1]:
                    acc += src[z0, y0, x1] * wz0 * wy0 * tx
                    wacc += wz0 * wy0 * tx
                if valid[z0, y1, x0]:
                    acc += src[z0, y1, x0] * wz0 * ty * wx0
                    wacc += wz0 * ty * wx0
                if valid[z0, y1, x1]:
                    acc += src[z0, y1, x1] * wz0 * ty * tx
                    wacc += wz0 * ty * tx
                if valid[z1, y0, x0]:
                    acc += src[z1, y0, x0] * tz * wy0 * wx0
                    wacc += tz * wy0 * wx0
                if valid[z1, y0, x1]:
                    acc += src[z1, y0, x1] * tz * wy0 * tx
                    wacc += tz * wy0 * tx
                if valid[z1, y1, x0]:
                    acc += src[z1, y1, x0] * tz * ty * wx0
                    wacc += tz * ty * wx0
                if valid[z1, y1, x1]:
                    acc += src[z1, y1, x1] * tz * ty * tx
                    wacc += tz * ty * tx

                if wacc >= weight_threshold:
                    out[k, j, i] = acc / wacc
                    out_valid[k, j, i] = True
                else:
                    out[k, j, i] = 0.0
                    out_valid[k, j, i] = False


def detect_sentinel_value(
    image: Image,
    candidate_values: tuple[float, ...] = COMMON_SENTINEL_VALUES,
    min_presence_fraction: float = 0.05,
    roi_mask: Optional[Image] = None,
) -> Optional[float]:
    """
    Detect if image contains a common sentinel value outside the ROI.

    A candidate is eligible as a sentinel if:
    1. It occupies a significant fraction of the whole image (>= min_presence_fraction).
    2. If roi_mask is provided, it appears primarily outside the ROI (ratio > 2:1
       outside vs inside). This guard distinguishes a padding/fill value from a
       legitimate low-density tissue value (e.g. real air at ~-1000 HU in a raw,
       un-masked CT) that would otherwise be misdetected on proportion alone.

    When several candidates are eligible, the one occupying the largest fraction
    of the image is returned; exact ties are broken by candidate_values order.

    This is used by the pipeline's AUTO source_mode to automatically detect
    images that have been pre-masked with sentinel values.

    Args:
        image: Input Image object.
        candidate_values: Values to check for sentinel patterns.
            Defaults to common medical imaging sentinels: -2048, -3024, -1024,
            -1000, 0, -32768.
        min_presence_fraction: Minimum fraction of voxels that must equal
            the candidate to consider it a sentinel. Default is 5%.
        roi_mask: Optional ROI mask. If provided, checks that sentinel values
            are primarily outside the mask (ratio > 2:1 outside vs inside).

    Returns:
        The detected sentinel value, or None if not detected.

    Example:
        ```python
        from pictologics.preprocessing import detect_sentinel_value
        from pictologics.loader import load_image

        image = load_image("image_with_sentinel.nii.gz")
        sentinel = detect_sentinel_value(image)

        if sentinel is not None:
            print(f"Detected sentinel value: {sentinel}")
        ```
    """
    array = image.array
    total_voxels = array.size
    roi_arr: Optional[npt.NDArray[np.bool_]] = None
    if roi_mask is not None:
        _validate_geometry(roi_mask, image, "ROI mask", "image")
        roi_arr = roi_mask.array > 0

    best_candidate: Optional[float] = None
    best_fraction = 0.0

    for candidate in candidate_values:
        sentinel_mask = array == candidate
        count = np.count_nonzero(sentinel_mask)
        fraction = count / total_voxels

        if fraction < min_presence_fraction:
            continue

        # If ROI mask provided, verify sentinel is primarily outside (>= 2:1)
        if roi_arr is not None:
            inside_count = np.count_nonzero(sentinel_mask & roi_arr)
            outside_count = count - inside_count
            if outside_count <= inside_count * 2:
                continue

        # Eligible: keep the candidate covering the largest fraction. Strict >
        # means an earlier (higher-priority) candidate wins an exact tie.
        if fraction > best_fraction:
            best_fraction = fraction
            best_candidate = candidate

    return best_candidate


def create_source_mask_from_sentinel(
    image: Image,
    sentinel_value: float,
    tolerance: float = 0.0,
) -> Image:
    """
    Create a source validity mask by marking sentinel voxels as invalid.

    The returned mask has value 1 for valid (non-sentinel) voxels and 0 for
    invalid (sentinel) voxels. This mask can be used with the Image.source_mask
    attribute to exclude sentinel voxels from resampling and filtering.

    Args:
        image: Input Image object.
        sentinel_value: The value considered as sentinel (invalid data).
        tolerance: Values within this tolerance of sentinel_value are also
            considered invalid. Default 0 means exact match only.

    Returns:
        Image object with binary mask (1 = valid, 0 = sentinel).

    Example:
        ```python
        from pictologics.preprocessing import create_source_mask_from_sentinel
        from pictologics.loader import load_image

        image = load_image("ct_with_background.nii.gz")

        # Create mask excluding -2048 sentinel values
        source_mask = create_source_mask_from_sentinel(image, sentinel_value=-2048)

        # Apply to image for sentinel-aware processing
        image_with_mask = image.with_source_mask(source_mask)
        ```
    """
    if tolerance == 0:
        valid = image.array != sentinel_value
    else:
        valid = np.abs(image.array - sentinel_value) > tolerance

    return Image(
        array=valid.astype(np.uint8),
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
        modality="SOURCE_MASK",
    )


def _resample_with_source_mask(
    image_array: npt.NDArray[np.floating[Any]],
    source_mask: npt.NDArray[np.bool_],
    matrix: npt.NDArray[np.floating[Any]],
    offset: npt.NDArray[np.floating[Any]],
    output_shape: tuple[int, ...],
    order: int,
    mode: str,
    weight_threshold: float = 0.5,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.bool_]]:
    """
    Resample image using normalized interpolation that excludes invalid (sentinel) voxels.

    This implements normalized convolution for resampling, where only valid source
    voxels contribute to the interpolated output values.

    Algorithm:
        1. Zero out invalid voxels in image (image * mask)
        2. Resample zeroed image -> weighted_sum
        3. Resample mask (as float) -> weight_sum
        4. Normalize: result = weighted_sum / weight_sum
        5. Mark output voxels with low weight as invalid

    Args:
        image_array: Input image array.
        source_mask: Boolean mask where True = valid voxel.
        matrix: Affine transform matrix (diagonal elements).
        offset: Affine transform offset.
        output_shape: Shape of output array.
        order: Interpolation order.
        mode: Boundary mode.
        weight_threshold: Minimum weight to consider output voxel valid.
            Output voxels with weight < threshold are marked as invalid.
            Default 0.5 means the majority of the interpolation weight must
            come from valid voxels; lower values keep more boundary voxels
            at the cost of heavier extrapolation.

    Returns:
        Tuple of (resampled_image, resampled_source_mask).
    """
    # Step 1: Zero out invalid voxels
    valid_image = np.where(source_mask, image_array, 0.0).astype(np.float64)

    # Step 2: Resample zeroed image - this gives weighted sum where invalid=0
    weighted_sum = affine_transform(
        valid_image,
        matrix=matrix,
        offset=offset,
        output_shape=output_shape,
        order=order,
        mode=mode,
    )

    # Step 3: Resample mask as weights
    weight_sum = affine_transform(
        source_mask.astype(np.float64),
        matrix=matrix,
        offset=offset,
        output_shape=output_shape,
        order=order,
        mode=mode,
    )

    # Step 4: Normalize (avoid division by zero)
    # Voxels with weight_sum < threshold are considered invalid
    valid_output = weight_sum >= weight_threshold

    result = np.zeros_like(weighted_sum)
    result[valid_output] = weighted_sum[valid_output] / weight_sum[valid_output]

    # Output source mask based on interpolation weights
    output_source_mask = valid_output

    return result, output_source_mask


def resample_image(
    image: Image,
    new_spacing: tuple[float, float, float],
    interpolation: str = "linear",
    boundary_mode: str = "nearest",
    round_intensities: bool = False,
    mask_threshold: Optional[float] = None,
    source_mask: Optional[Image | npt.NDArray[np.bool_]] = None,
    weight_threshold: float = 0.5,
) -> Image:
    """
    Resample image to new voxel spacing using IBSI-compliant 'Align grid centers' method.

    The common cases (3D float64 image, 'nearest' boundary, nearest/linear
    interpolation) run on parallel numba kernels; everything else uses
    scipy.ndimage.affine_transform.

    Args:
        image: Input Image object.
        new_spacing: Target spacing (x, y, z). Must be positive.
        interpolation: Interpolation method.
            'nearest': Nearest neighbour (order 0).
            'linear': Trilinear (order 1).
            'cubic': Tricubic spline (order 3).
        boundary_mode: Padding mode for extrapolation.
            'nearest' (default): Replicates edge values (aaaa|abcd|dddd).
            'constant': Pads with constant value (0).
            'reflect': Reflects at boundary.
            'wrap': Wraps around.
        round_intensities: If True, round resulting intensities to nearest integer.
        mask_threshold: If provided, treat output as a binary mask.
                        Values >= threshold become 1, others 0.
                        Commonly 0.5 for partial volume correction.
        source_mask: Optional source validity mask. If provided (or if image.source_mask
                     is set), only valid voxels are used for interpolation. This prevents
                     sentinel values (e.g., -2048 in CT) from contaminating the resampled
                     output. Can be an Image object or a boolean numpy array.
        weight_threshold: Only used with a source mask. Minimum fraction of
                     interpolation weight that must come from valid voxels for an
                     output voxel to be considered valid. Default 0.5 (majority).

    Returns:
        Resampled Image object. If source_mask was used, the output Image will have
        its source_mask attribute set to the resampled validity mask.

    Note:
        When source_mask is active, the function uses normalized interpolation:
        the contribution of each input voxel is weighted by its validity, and the
        result is normalized by the sum of valid weights. This ensures that sentinel
        voxels do not affect the output.

    Raises:
        ValueError: If any element of `new_spacing` is not positive, if
            `interpolation` is not one of 'nearest', 'linear', 'cubic', or if
            `source_mask` does not match the image geometry (when passed as an
            Image) or array shape (when passed as a numpy array).

    Example:
        Resample image to isotropic 1mm spacing using linear interpolation:

        ```python
        from pictologics.preprocessing import resample_image

        # Resample to 1x1x1 mm
        resampled_img = resample_image(
            image,
            new_spacing=(1.0, 1.0, 1.0),
            interpolation="linear"
        )
        ```

        Resample with sentinel-value exclusion:

        ```python
        # Image has -2048 outside ROI
        image_with_sentinel = image.with_source_mask(roi_mask)
        resampled = resample_image(
            image_with_sentinel,
            new_spacing=(1.0, 1.0, 1.0)
        )
        # Sentinel voxels were excluded from interpolation
        ```
    """
    if any(s <= 0 for s in new_spacing):
        raise ValueError(f"New spacing must be positive, got {new_spacing}")

    # Determine effective source mask
    effective_source: Optional[npt.NDArray[np.bool_]] = None

    if source_mask is not None:
        if isinstance(source_mask, Image):
            _validate_geometry(source_mask, image, "source mask", "image")
            effective_source = source_mask.array > 0
        else:
            effective_source = source_mask.astype(bool)
    elif image.has_source_mask:
        effective_source = image.source_mask

    if effective_source is not None and effective_source.shape != image.array.shape:
        raise ValueError(
            f"Source mask shape {effective_source.shape} must match image shape {image.array.shape}"
        )

    # An all-valid mask is equivalent to no mask; skip the normalized
    # interpolation path, which costs a second affine_transform.
    source_all_valid = effective_source is not None and bool(effective_source.all())
    if source_all_valid:
        effective_source = None

    # Map interpolation string to spline order
    interpolation_map = {
        "nearest": 0,
        "linear": 1,
        "cubic": 3,
    }

    if interpolation not in interpolation_map:
        raise ValueError(
            f"Unknown interpolation method: {interpolation}. "
            f"Supported: {list(interpolation_map.keys())}"
        )

    order = interpolation_map[interpolation]

    # Calculate new shape
    # IBSI: nb = ceil(na * sa / sb)
    original_spacing = np.array(image.spacing)
    target_spacing = np.array(new_spacing)

    # Scale factor for dimensions (how many new voxels per old voxel)
    # dim_scale = s_old / s_new
    dim_scale = original_spacing / target_spacing

    # Round to 9 decimals before ceil: floating-point noise can push an exact
    # integer product (e.g. 110.0) to 110.00000000000001, yielding a spurious
    # extra voxel along that axis.
    new_shape = np.ceil(np.round(image.array.shape * dim_scale, 9)).astype(int)

    # Calculate affine transform parameters
    # We map Output Coordinate (x_out) -> Input Coordinate (x_in)
    # x_in = matrix * x_out + offset

    # Scale factor for coordinates (step size in input space per step in output space)
    # step_in = s_new / s_old
    coord_scale = target_spacing / original_spacing
    matrix = coord_scale  # Diagonal matrix elements

    # Calculate offset for 'Align Grid Centers
    center_orig = (np.array(image.array.shape) - 1) / 2.0
    center_new = (new_shape - 1) / 2.0

    offset = center_orig - matrix * center_new

    # Perform resampling
    resampled_array: npt.NDArray[Any]
    new_source_mask: Optional[npt.NDArray[np.bool_]] = None
    out_shape = tuple(int(n) for n in new_shape)

    # The parallel kernels cover the hot path (3D, 'nearest' boundary, common
    # dtypes); anything else falls back to scipy.ndimage. The kernels take the
    # shift form scipy's zoom_shift uses internally (coordinate is
    # (idx + offset/scale) * scale), so their output matches the scipy path.
    kernel_ok = image.array.ndim == 3 and boundary_mode == "nearest"
    kernel_shift = offset / matrix

    # The trilinear kernel matches scipy to ~1 ulp but not bit-for-bit; the
    # discrete post-steps (mask_threshold binarization, round_intensities)
    # are sensitive to that last ulp (flips observed exactly at .5 values),
    # so those combinations keep the scipy path. Nearest (order 0) is pure
    # voxel selection and bit-identical, so it stays on the kernel throughout.
    kernel_exact_out = mask_threshold is None and not round_intensities

    if effective_source is None:
        if kernel_ok and kernel_exact_out and order == 1 and image.array.dtype == np.float64:
            src = np.ascontiguousarray(image.array)
            resampled_array = np.empty(out_shape, dtype=np.float64)
            _resample_trilinear_numba(src, matrix, kernel_shift, resampled_array)
        elif kernel_ok and order == 0 and image.array.dtype in (np.float64, np.uint8, np.bool_):
            src = np.ascontiguousarray(image.array)
            resampled_array = np.empty(out_shape, dtype=image.array.dtype)
            _resample_nearest_numba(src, matrix, kernel_shift, resampled_array)
        else:
            resampled_array = affine_transform(
                image.array,
                matrix=matrix,
                offset=offset,
                output_shape=out_shape,
                order=order,
                mode=boundary_mode,
            )
        if source_all_valid:
            # Preserve the output contract: a resampled image keeps a source mask
            new_source_mask = np.ones(out_shape, dtype=bool)
    elif kernel_ok and kernel_exact_out and order == 1 and image.array.dtype == np.float64:
        # Masked resampling: one fused pass instead of two affine_transforms
        src = np.ascontiguousarray(image.array)
        valid = np.ascontiguousarray(effective_source)
        resampled_array = np.empty(out_shape, dtype=np.float64)
        new_source_mask = np.empty(out_shape, dtype=np.bool_)
        _resample_trilinear_masked_numba(
            src,
            valid,
            matrix,
            kernel_shift,
            float(weight_threshold),
            resampled_array,
            new_source_mask,
        )
    else:
        # Masked resampling using normalized interpolation
        resampled_array, new_source_mask = _resample_with_source_mask(
            image.array,
            effective_source,
            matrix=matrix,
            offset=offset,
            output_shape=out_shape,
            order=order,
            mode=boundary_mode,
            weight_threshold=weight_threshold,
        )

    # Post-processing
    if mask_threshold is not None:
        # Binarize mask
        resampled_array = (resampled_array >= mask_threshold).astype(np.uint8)
    elif round_intensities:
        # Round intensities
        resampled_array = np.round(resampled_array)

    # Update origin to maintain center alignment
    # O_new = O_old + 0.5 * ( (N_old-1)*S_old - (N_new-1)*S_new )
    extent_orig = (np.array(image.array.shape) - 1) * original_spacing
    extent_new = (new_shape - 1) * target_spacing
    origin_shift = 0.5 * (extent_orig - extent_new)
    new_origin = tuple(np.array(image.origin) + _direction_matrix(image.direction) @ origin_shift)

    return Image(
        array=resampled_array,
        spacing=new_spacing,
        origin=new_origin,
        direction=image.direction,
        modality=image.modality,
        source_mask=new_source_mask,
    )


def discretise_image(
    image: Image | npt.NDArray[Any],
    method: str,
    roi_mask: Image | npt.NDArray[Any] | None = None,
    n_bins: Optional[int] = None,
    bin_width: Optional[float] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    cutoffs: Optional[list[float]] = None,
) -> Image | npt.NDArray[np.int32]:
    """
    Discretise image intensities.

    Supports IBSI-compliant Fixed Bin Number (FBN) and Fixed Bin Size (FBS).

    Args:
        image: Input Image object or numpy array.
        method: 'FBN' (Fixed Bin Number), 'FBS' (Fixed Bin Size), or 'FIXED_CUTOFFS'.
        roi_mask: Optional mask to define the ROI for determining min/max values.
        n_bins: Number of bins (required for FBN).
        bin_width: Bin width (required for FBS).
        min_val: Minimum value for discretisation.
                 For FBS, defaults to ROI minimum (or global minimum).
                 For FBN, defaults to ROI minimum.
        max_val: Maximum value for discretisation (FBN only).
                 Defaults to ROI maximum.
        cutoffs: List of cutoffs (required for FIXED_CUTOFFS). Values below the
                 first cutoff map to bin 1; values >= the last cutoff map to
                 bin len(cutoffs) + 1.

    Returns:
        Discretised Image object or numpy array (depending on input).
        Values are 1-based int32 indices; 0 marks NaN (invalid) voxels.

    Raises:
        ValueError: If `roi_mask` shape does not match `image` shape (or, when
            both are Image objects, if their geometry otherwise differs); if
            `method` is 'FBN' and `n_bins` is missing or not positive; if
            `method` is 'FBS' and `bin_width` is missing or not positive; if
            `method` is 'FIXED_CUTOFFS' and `cutoffs` is missing; or if
            `method` is not one of 'FBN', 'FBS', 'FIXED_CUTOFFS'.

    Example:
        Discretise image into 32 fixed bins (FBN):

        ```python
        from pictologics.preprocessing import discretise_image

        # FBN with 32 bins
        disc_image = discretise_image(
            image,
            method="FBN",
            n_bins=32
        )
        ```
    """
    # Handle input type
    array = image.array if isinstance(image, Image) else image

    # Determine ROI values for default min/max (small gather; NaNs excluded)
    roi_values: Optional[npt.NDArray[Any]]
    if roi_mask is not None:
        if isinstance(roi_mask, Image):
            if isinstance(image, Image):
                _validate_geometry(roi_mask, image, "ROI mask", "image")
            mask_arr = roi_mask.array
        else:
            mask_arr = roi_mask

        if mask_arr.shape != array.shape:
            raise ValueError(f"Shape mismatch: Image {array.shape} vs Mask {mask_arr.shape}")

        # Extract ROI values (ignoring NaNs)
        roi_values = array[mask_arr > 0]
        roi_values = roi_values[~np.isnan(roi_values)]
    else:
        roi_values = None

    def _default_bound(roi_reduce: Any, global_reduce: Any) -> Any:
        """Default bin bound: ROI reduction, falling back to the global one."""
        if roi_values is not None and roi_values.size > 0:
            return roi_reduce(roi_values)
        # nanmin/nanmax of an all-NaN image is NaN (warning suppressed); it
        # propagates through the bin math so every voxel ends up invalid (0).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return global_reduce(array)

    discretised: npt.NDArray[Any]

    if array.size == 0:
        # int32 halves memory vs the default int64; bin counts in radiomics
        # never approach the int32 limit
        discretised = np.zeros(array.shape, dtype=np.int32)
        if isinstance(image, Image):
            # Create new Image with discretised array
            return Image(
                array=discretised,
                spacing=image.spacing,
                origin=image.origin,
                direction=image.direction,
                modality=image.modality,
            )
        return discretised

    # The bin maths below run densely on the full array (no boolean
    # gather/scatter): NaN voxels propagate through the float ops and are
    # mapped to bin 0 (invalid) at the end.
    if method == "FBN":
        if n_bins is None:
            raise ValueError("n_bins required for FBN")
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")

        # Determine min/max
        current_min = min_val if min_val is not None else _default_bound(np.min, np.nanmin)
        current_max = max_val if max_val is not None else _default_bound(np.max, np.nanmax)

        if current_max <= current_min:
            # Edge case: flat region or invalid range
            discretised = (~np.isnan(array)).astype(np.int32)
        elif array.dtype == np.float64 and array.size >= _KERNEL_MIN_SIZE:
            # Single-pass kernel; bit-identical to the numpy chain below
            flat = array.ravel()
            binned: npt.NDArray[Any] = np.empty(flat.size, dtype=np.int32)
            _discretise_fbn_numba(
                flat, float(n_bins), float(current_min), float(current_max), binned
            )
            discretised = binned.reshape(array.shape)
        else:
            # IBSI FBN: floor(N_g * (X - X_min) / (X_max - X_min)) + 1
            # (same operation order as before, so bins are bit-identical)
            temp = n_bins * (array - current_min)
            if temp.dtype.kind == "f":
                np.divide(temp, current_max - current_min, out=temp)
            else:
                # integer input: out-of-place division promotes to float
                temp = temp / (current_max - current_min)
            np.floor(temp, out=temp)
            temp += 1

            # Clip to [1, N_g]: the max value falls into N_g + 1 with this
            # formula, and outliers are clipped to the boundary bins
            np.clip(temp, 1, n_bins, out=temp)

            # NaN voxels propagated through the maths; map them to bin 0
            temp[np.isnan(temp)] = 0
            discretised = temp.astype(np.int32)

    elif method == "FBS":
        if bin_width is None:
            raise ValueError("bin_width required for FBS")
        if bin_width <= 0:
            raise ValueError("bin_width must be positive")

        current_min = min_val if min_val is not None else _default_bound(np.min, np.nanmin)

        if array.dtype == np.float64 and array.size >= _KERNEL_MIN_SIZE:
            # Single-pass kernel; bit-identical to the numpy chain below
            flat = array.ravel()
            fbs_binned: npt.NDArray[Any] = np.empty(flat.size, dtype=np.int32)
            _discretise_fbs_numba(flat, float(bin_width), float(current_min), fbs_binned)
            discretised = fbs_binned.reshape(array.shape)
        else:
            # IBSI FBS: floor((X - X_min) / w_b) + 1
            temp = array - current_min
            if temp.dtype.kind == "f":
                np.divide(temp, bin_width, out=temp)
            else:
                # integer input: out-of-place division promotes to float
                temp = temp / bin_width
            np.floor(temp, out=temp)
            temp += 1

            # Ensure minimum bin is 1 (NaN propagates through np.maximum)
            np.maximum(temp, 1, out=temp)

            # NaN voxels propagated through the maths; map them to bin 0
            temp[np.isnan(temp)] = 0
            discretised = temp.astype(np.int32)

    elif method == "FIXED_CUTOFFS":
        if cutoffs is None:
            raise ValueError("cutoffs required for FIXED_CUTOFFS")

        # +1 so bins are 1-based: values below the first cutoff map to bin 1,
        # keeping 0 reserved for invalid (NaN) voxels.
        temp_int = np.digitize(array, bins=np.array(cutoffs)) + 1
        temp_int[np.isnan(array)] = 0
        discretised = temp_int.astype(np.int32)

    else:
        raise ValueError(f"Unknown discretisation method: {method}")

    if isinstance(image, Image):
        return Image(
            array=discretised,
            spacing=image.spacing,
            origin=image.origin,
            direction=image.direction,
            modality=image.modality,
        )
    return discretised


def apply_mask(
    image: Image | npt.NDArray[Any],
    mask: Image | npt.NDArray[Any],
    mask_values: int | list[int] | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """
    Apply mask to image and return flattened array of voxel values.

    Args:
        image: Image object or numpy array.
        mask: Image object (mask) or numpy array.
        mask_values: Optional value(s) in the mask to consider as ROI. When
                     omitted, all nonzero mask values are considered ROI
                     membership. Can be a single integer or a list of integers.

    Returns:
        1D numpy array of values within the mask.

    Raises:
        ValueError: If the image and mask arrays have different shapes.

    Example:
        Extract intensities within a spherical mask:

        ```python
        import numpy as np
        from pictologics.preprocessing import apply_mask

        image = np.arange(27.0).reshape(3, 3, 3)
        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        mask[1, 1, 1] = 1

        values = apply_mask(image, mask)
        # array([13.])
        ```
    """
    # Handle inputs
    img_arr = image.array if isinstance(image, Image) else image
    mask_arr = mask.array if isinstance(mask, Image) else mask

    if isinstance(image, Image) and isinstance(mask, Image):
        _validate_geometry(mask, image, "mask", "image")

    # Ensure shapes match
    if img_arr.shape != mask_arr.shape:
        raise ValueError(
            f"Image shape {img_arr.shape} and mask shape {mask_arr.shape} do not match"
        )

    # Handle mask values
    if mask_values is None:
        roi_mask = mask_arr != 0
    else:
        if isinstance(mask_values, int):
            mask_values = [mask_values]
        roi_mask = np.isin(mask_arr, mask_values)

    # Apply mask (an all-False mask yields an empty array of the image dtype)
    return cast(npt.NDArray[np.floating[Any]], img_arr[roi_mask])


def extract_roi(
    image: Image,
    mask: Image,
    mask_values: int | list[int] | None = None,
) -> Image:
    """
    Extract ROI from image. Voxels outside the mask are set to NaN.
    IBSI 'ROI extraction'.

    Args:
        image: Image object.
        mask: Image object (mask).
        mask_values: Optional value(s) in the mask to consider as ROI. When
            omitted, all nonzero mask values are considered ROI membership.

    Returns:
        New Image object with non-ROI voxels set to NaN.

    Raises:
        ValueError: If `image` and `mask` have different shapes, or otherwise
            occupy different physical geometry (spacing, origin, or direction).

    Example:
        Set voxels outside a mask to NaN:

        ```python
        import numpy as np
        from pictologics.loader import Image
        from pictologics.preprocessing import extract_roi

        array = np.arange(27.0).reshape(3, 3, 3)
        mask_array = np.zeros((3, 3, 3), dtype=np.uint8)
        mask_array[1, 1, 1] = 1

        image = Image(array=array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
        mask = Image(array=mask_array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))

        roi_image = extract_roi(image, mask)
        # roi_image.array is all NaN except at [1, 1, 1], which is 13.0
        ```
    """
    try:
        _validate_geometry(mask, image, "mask", "image")
    except ValueError as exc:
        if "Dimension mismatch" in str(exc):
            raise ValueError("Image and mask must have the same shape.") from exc
        raise

    # Handle mask values
    if mask_values is None:
        roi_mask = mask.array != 0
    else:
        if isinstance(mask_values, int):
            mask_values = [mask_values]
        roi_mask = np.isin(mask.array, mask_values)

    # Preserve floating dtype (NaN-capable); only integer inputs need upcasting
    if np.issubdtype(image.array.dtype, np.floating):
        new_array = image.array.copy()
    else:
        new_array = image.array.astype(float)
    new_array[~roi_mask] = np.nan

    return Image(
        array=new_array,
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
        modality=image.modality,
    )


def resegment_mask(
    image: Image,
    mask: Image,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
) -> Image:
    """
    Update mask to exclude voxels where image intensity is outside the specified range.
    Used for IBSI re-segmentation (e.g. [-1000, 400] HU).

    Args:
        image: Image object.
        mask: Image object (mask).
        range_min: Minimum intensity value (inclusive). If None, no lower bound.
        range_max: Maximum intensity value (inclusive). If None, no upper bound.

    Returns:
        Updated Image object (mask) with re-segmentation applied.

    Example:
        Resegment mask to keep only values between -1000 and 400 (e.g. HU range):

        ```python
        from pictologics.preprocessing import resegment_mask

        # Keep voxels in range [-1000, 400]
        new_mask = resegment_mask(
            image,
            mask,
            range_min=-1000,
            range_max=400
        )
        ```
    """
    try:
        _validate_geometry(mask, image, "mask", "image")
    except ValueError as exc:
        if "Dimension mismatch" in str(exc):
            raise ValueError(
                "Image and mask must have the same shape for re-segmentation."
            ) from exc
        raise

    if (
        image.array.dtype == np.float64
        and image.array.size >= _KERNEL_MIN_SIZE
        and mask.array.dtype in (np.float64, np.uint8, np.bool_)
    ):
        # Single-pass kernel; bit-identical to the numpy chain below
        lo = float("-inf") if range_min is None else float(range_min)
        hi = float("inf") if range_max is None else float(range_max)
        flat_out = np.empty(mask.array.size, dtype=mask.array.dtype)
        _resegment_numba(image.array.ravel(), mask.array.ravel(), lo, hi, flat_out)
        new_mask_array = flat_out.reshape(mask.array.shape)
    else:
        new_mask_array = mask.array.copy()

        # Identify outliers
        outliers = np.zeros(image.array.shape, dtype=bool)

        if range_min is not None:
            outliers |= image.array < range_min

        if range_max is not None:
            outliers |= image.array > range_max

        # Set mask to 0 where outliers exist
        new_mask_array[outliers] = 0

    return Image(
        array=new_mask_array,
        spacing=mask.spacing,
        origin=mask.origin,
        direction=mask.direction,
        modality=mask.modality,
    )


def filter_outliers(image: Image, mask: Image, sigma: float = 3.0) -> Image:
    """
    Exclude outliers from the mask based on mean +/- sigma * std.
    IBSI 3.6.

    Args:
        image: Image object.
        mask: Image object (mask).
        sigma: Number of standard deviations.

    Returns:
        New Image object (mask) with outliers removed.

    Example:
        Remove outliers beyond 3 standard deviations from the mean:

        ```python
        from pictologics.preprocessing import filter_outliers

        # Remove outliers > 3 sigma
        clean_mask = filter_outliers(
            image,
            mask,
            sigma=3.0
        )
        ```
    """
    _validate_geometry(mask, image, "mask", "image")

    # Gather intensities within the mask once; all thresholding below runs on
    # this ROI-sized array instead of the full volume
    roi = mask.array != 0
    roi_values = image.array[roi]

    # Exclude NaNs (e.g. from extract_roi) which would otherwise poison the
    # mean/std and wipe the whole mask
    finite_values = roi_values[~np.isnan(roi_values)]

    if finite_values.size == 0:
        return mask

    mean_val = np.mean(finite_values)
    # IBSI uses population std (no bias correction, ddof=0)
    std_val = np.std(finite_values, ddof=0)

    lower_bound = mean_val - sigma * std_val
    upper_bound = mean_val + sigma * std_val

    # Keep values within [lower, upper]; NaN compares False and is excluded
    keep = (roi_values >= lower_bound) & (roi_values <= upper_bound)

    # Update original mask (zeroing outliers preserves dtype and label values)
    new_mask_array = mask.array.copy()
    roi_mask_values = new_mask_array[roi]
    roi_mask_values[~keep] = 0
    new_mask_array[roi] = roi_mask_values

    return Image(
        array=new_mask_array,
        spacing=mask.spacing,
        origin=mask.origin,
        direction=mask.direction,
        modality=mask.modality,
    )


def round_intensities(image: Image) -> Image:
    """
    Round image intensities to the nearest integer.
    """
    new_array = np.round(image.array)
    return Image(
        array=new_array,
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
        modality=image.modality,
    )


def keep_largest_component(mask: Image) -> Image:
    """
    Keep only the largest connected component in the mask.

    Uses full (26-)connectivity in 3D, i.e. voxels touching at a face, edge, or
    vertex belong to the same component. This matches the connectivity IBSI
    prescribes for 3D zones (GLSZM/GLDZM) and the convention used elsewhere in
    this library, rather than scipy's face-only (6-connectivity) default.
    """
    mask_array = mask.array
    new_array: npt.NDArray[Any] = np.zeros(mask_array.shape, dtype=np.uint8)

    # Label only inside the nonzero bounding box: for a compact ROI in a
    # CT-sized volume this replaces the full-volume connected-component pass
    # with one fast bbox scan plus a small labeling. Cropping removes whole
    # all-zero slabs, so components and their raster-scan label order (which
    # breaks size ties) are unchanged.
    bbox: Optional[tuple[slice, ...]]
    if mask_array.ndim == 3:
        bbox = compute_nonzero_bbox(mask_array)
    else:
        bbox = tuple(slice(None) for _ in mask_array.shape)

    if bbox is not None:
        sub = mask_array[bbox]
        # Full connectivity: connectivity == ndim connects all diagonal neighbours.
        structure = generate_binary_structure(sub.ndim, sub.ndim)
        labeled_sub, num_features = label(sub, structure=structure)
        if num_features <= 1:
            new_array[bbox] = sub != 0
        else:
            counts = np.bincount(labeled_sub.ravel())
            counts[0] = 0  # ignore background
            max_label = int(counts.argmax())
            new_array[bbox] = labeled_sub == max_label

    return Image(
        array=new_array,
        spacing=mask.spacing,
        origin=mask.origin,
        direction=mask.direction,
        modality=mask.modality,
    )
