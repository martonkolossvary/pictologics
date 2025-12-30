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

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import affine_transform, label

from .loader import Image


def resample_image(
    image: Image,
    new_spacing: Tuple[float, float, float],
    interpolation: str = "linear",
    boundary_mode: str = "nearest",
    round_intensities: bool = False,
    mask_threshold: Optional[float] = None,
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

    Returns:
        Resampled Image object.
    """
    if any(s <= 0 for s in new_spacing):
        raise ValueError(f"New spacing must be positive, got {new_spacing}")

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

    resampled_array = affine_transform(
        image.array,
        matrix=matrix,
        offset=offset,
        output_shape=new_shape,
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
    )


def discretise_image(
    image: Union[Image, np.ndarray],
    method: str,
    roi_mask: Optional[Union[Image, np.ndarray]] = None,
    n_bins: Optional[int] = None,
    bin_width: Optional[float] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    cutoffs: Optional[List[float]] = None,
) -> Union[Image, np.ndarray]:
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
    image: Union[Image, np.ndarray],
    mask: Union[Image, np.ndarray],
    mask_values: Optional[Union[int, List[int]]] = 1,
) -> np.ndarray:
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
    mask_values: Optional[Union[int, List[int]]] = 1,
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

    new_array = image.array.astype(float).copy()
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
