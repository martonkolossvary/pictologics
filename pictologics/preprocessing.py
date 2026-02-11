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

from typing import Any, Optional

import numpy as np
from numpy import typing as npt
from scipy.ndimage import affine_transform, label

from .loader import Image

# Common sentinel values used in medical imaging to denote "no data" or "background"
COMMON_SENTINEL_VALUES: tuple[float, ...] = (-2048.0, -1024.0, -1000.0, 0.0, -32768.0)


def detect_sentinel_value(
    image: Image,
    candidate_values: tuple[float, ...] = COMMON_SENTINEL_VALUES,
    min_presence_fraction: float = 0.01,
    roi_mask: Optional[Image] = None,
) -> Optional[float]:
    """
    Detect if image contains a common sentinel value outside the ROI.

    A sentinel value is detected if:
    1. It appears in a significant fraction of voxels (>= min_presence_fraction)
    2. If roi_mask is provided, the sentinel primarily appears outside the ROI

    This is used by the pipeline's AUTO source_mode to automatically detect
    images that have been pre-masked with sentinel values.

    Args:
        image: Input Image object.
        candidate_values: Values to check for sentinel patterns.
            Defaults to common medical imaging sentinels: -2048, -1024, -1000, 0, -32768.
        min_presence_fraction: Minimum fraction of voxels that must equal
            the candidate to consider it a sentinel. Default is 1%.
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

    for candidate in candidate_values:
        count = np.sum(array == candidate)
        fraction = count / total_voxels

        if fraction >= min_presence_fraction:
            # If ROI mask provided, verify sentinel is primarily outside
            if roi_mask is not None:
                roi_arr = roi_mask.array > 0
                inside_count = np.sum((array == candidate) & roi_arr)
                outside_count = np.sum((array == candidate) & ~roi_arr)

                # Sentinel should be mostly outside ROI (at least 2:1 ratio)
                if outside_count > inside_count * 2:
                    return candidate
            else:
                return candidate

    return None


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
    weight_threshold: float = 0.01,
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
            Default 0.01 means at least 1% of interpolation weight from valid voxels.

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
) -> Image:
    """
    Resample image to new voxel spacing using IBSI-compliant 'Align grid centers' method.

    Uses scipy.ndimage.affine_transform for memory efficiency.

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

    Returns:
        Resampled Image object. If source_mask was used, the output Image will have
        its source_mask attribute set to the resampled validity mask.

    Note:
        When source_mask is active, the function uses normalized interpolation:
        the contribution of each input voxel is weighted by its validity, and the
        result is normalized by the sum of valid weights. This ensures that sentinel
        voxels do not affect the output.

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
            effective_source = source_mask.array > 0
        else:
            effective_source = source_mask.astype(bool)
    elif image.has_source_mask:
        effective_source = image.source_mask

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

    new_shape = np.ceil(image.array.shape * dim_scale).astype(int)

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
    new_source_mask: Optional[npt.NDArray[np.bool_]] = None

    if effective_source is None:
        # Original behavior - use all voxels
        resampled_array = affine_transform(
            image.array,
            matrix=matrix,
            offset=offset,
            output_shape=tuple(new_shape),
            order=order,
            mode=boundary_mode,
        )
    else:
        # Masked resampling using normalized interpolation
        resampled_array, new_source_mask = _resample_with_source_mask(
            image.array,
            effective_source,
            matrix=matrix,
            offset=offset,
            output_shape=tuple(new_shape),
            order=order,
            mode=boundary_mode,
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
    new_origin = tuple(np.array(image.origin) + origin_shift)

    return Image(
        array=resampled_array,
        spacing=new_spacing,
        origin=new_origin,
        direction=image.direction,
        modality=image.modality,
        source_mask=new_source_mask,
    )


def discretise_image(
    image: Image | npt.NDArray[np.floating[Any]],
    method: str,
    roi_mask: Image | npt.NDArray[np.floating[Any]] | None = None,
    n_bins: Optional[int] = None,
    bin_width: Optional[float] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    cutoffs: Optional[list[float]] = None,
) -> Image | npt.NDArray[np.floating[Any]]:
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
        cutoffs: List of cutoffs (required for FIXED_CUTOFFS).

    Returns:
        Discretised Image object or numpy array (depending on input).
        Values are 1-based indices.

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
    if isinstance(image, Image):
        array = image.array
        is_image_obj = True
    else:
        array = image
        is_image_obj = False

    # Determine ROI values for default min/max
    if roi_mask is not None:
        if isinstance(roi_mask, Image):
            mask_arr = roi_mask.array
        else:
            mask_arr = roi_mask

        if mask_arr.shape != array.shape:
            raise ValueError(
                f"Shape mismatch: Image {array.shape} vs Mask {mask_arr.shape}"
            )

        # Extract ROI values (ignoring NaNs)
        roi_values = array[(mask_arr > 0) & (~np.isnan(array))]
    else:
        roi_values = array[~np.isnan(array)]

    # Initialize result
    discretised = np.zeros(array.shape, dtype=int)

    # We process all non-NaN pixels in the image
    valid_mask = ~np.isnan(array)
    values = array[valid_mask]

    if values.size == 0:
        if is_image_obj:
            # Create new Image with discretised array
            return Image(
                array=discretised,
                spacing=image.spacing,  # type: ignore
                origin=image.origin,  # type: ignore
                direction=image.direction,  # type: ignore
                modality=image.modality,  # type: ignore
            )
        return discretised

    if method == "FBN":
        if n_bins is None:
            raise ValueError("n_bins required for FBN")
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")

        # Determine min/max
        current_min = min_val
        if current_min is None:
            current_min = np.min(roi_values) if roi_values.size > 0 else np.min(values)

        current_max = max_val
        if current_max is None:
            current_max = np.max(roi_values) if roi_values.size > 0 else np.max(values)

        if current_max <= current_min:
            # Edge case: flat region or invalid range
            discretised[valid_mask] = 1
        else:
            # IBSI FBN: floor(N_g * (X - X_min) / (X_max - X_min)) + 1
            temp_discretised = (
                np.floor(n_bins * (values - current_min) / (current_max - current_min))
                + 1
            )

            # Handle max value case (it falls into N_g + 1 with this formula)
            # Also clip outliers
            temp_discretised[values >= current_max] = n_bins
            temp_discretised = np.clip(temp_discretised, 1, n_bins)

            discretised[valid_mask] = temp_discretised.astype(int)

    elif method == "FBS":
        if bin_width is None:
            raise ValueError("bin_width required for FBS")
        if bin_width <= 0:
            raise ValueError("bin_width must be positive")

        current_min = min_val
        if current_min is None:
            current_min = np.min(roi_values) if roi_values.size > 0 else np.min(values)

        # IBSI FBS: floor((X - X_min) / w_b) + 1
        temp_discretised = np.floor((values - current_min) / bin_width) + 1

        # Ensure minimum bin is 1
        temp_discretised[temp_discretised < 1] = 1
        discretised[valid_mask] = temp_discretised.astype(int)

    elif method == "FIXED_CUTOFFS":
        if cutoffs is None:
            raise ValueError("cutoffs required for FIXED_CUTOFFS")

        temp_discretised = np.digitize(values, bins=np.array(cutoffs))
        discretised[valid_mask] = temp_discretised.astype(int)

    else:
        raise ValueError(f"Unknown discretisation method: {method}")

    if is_image_obj:
        return Image(
            array=discretised,
            spacing=image.spacing,  # type: ignore
            origin=image.origin,  # type: ignore
            direction=image.direction,  # type: ignore
            modality=image.modality,  # type: ignore
        )
    return discretised


def apply_mask(
    image: Image | npt.NDArray[np.floating[Any]],
    mask: Image | npt.NDArray[np.floating[Any]],
    mask_values: int | list[int] | None = 1,
) -> npt.NDArray[np.floating[Any]]:
    """
    Apply mask to image and return flattened array of voxel values.

    Args:
        image: Image object or numpy array.
        mask: Image object (mask) or numpy array.
        mask_values: Value(s) in the mask to consider as ROI. Default is 1.
                     Can be a single integer or a list of integers.

    Returns:
        1D numpy array of values within the mask.
    """
    # Handle inputs
    img_arr = image.array if isinstance(image, Image) else image
    mask_arr = mask.array if isinstance(mask, Image) else mask

    # Ensure shapes match
    if img_arr.shape != mask_arr.shape:
        raise ValueError(
            f"Image shape {img_arr.shape} and mask shape {mask_arr.shape} do not match"
        )

    # Handle mask values
    if mask_values is None:
        mask_values = [1]
    elif isinstance(mask_values, int):
        mask_values = [mask_values]

    # Create boolean mask
    roi_mask = np.isin(mask_arr, mask_values)

    if not np.any(roi_mask):
        return np.array([])

    # Apply mask
    return img_arr[roi_mask]


def extract_roi(
    image: Image,
    mask: Image,
    mask_values: int | list[int] | None = 1,
) -> Image:
    """
    Extract ROI from image. Voxels outside the mask are set to NaN.
    IBSI 'ROI extraction'.

    Args:
        image: Image object.
        mask: Image object (mask).
        mask_values: Value(s) in the mask to consider as ROI. Default is 1.

    Returns:
        New Image object with non-ROI voxels set to NaN.
    """
    if image.array.shape != mask.array.shape:
        raise ValueError("Image and mask must have the same shape.")

    # Handle mask values
    if mask_values is None:
        mask_values = [1]
    elif isinstance(mask_values, int):
        mask_values = [mask_values]

    roi_mask = np.isin(mask.array, mask_values)

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
    if image.array.shape != mask.array.shape:
        raise ValueError("Image and mask must have the same shape for re-segmentation.")

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
    # Extract values within the mask
    values = apply_mask(image, mask)

    if values.size == 0:
        return mask

    mean_val = np.mean(values)
    # IBSI uses population std (no bias correction, ddof=0)
    std_val = np.std(values, ddof=0)

    lower_bound = mean_val - sigma * std_val
    upper_bound = mean_val + sigma * std_val

    # Create outlier mask
    # Keep values within [lower, upper]
    valid_mask = (image.array >= lower_bound) & (image.array <= upper_bound)

    # Update original mask
    new_mask_array = mask.array.copy()

    # Ensure boolean or integer type for bitwise operation
    # Assuming mask.array is binary (0/1) or boolean
    if new_mask_array.dtype == bool:
        new_mask_array = new_mask_array & valid_mask
    else:
        new_mask_array = (new_mask_array * valid_mask).astype(np.uint8)

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
    """
    mask_array = mask.array
    labeled_mask, num_features = label(mask_array)
    if num_features <= 1:
        return mask

    max_size = 0
    max_label = 0
    for i in range(1, num_features + 1):
        size = np.sum(labeled_mask == i)
        if size > max_size:
            max_size = size
            max_label = i

    new_array = (labeled_mask == max_label).astype(np.uint8)

    return Image(
        array=new_array,
        spacing=mask.spacing,
        origin=mask.origin,
        direction=mask.direction,
        modality=mask.modality,
    )
