"""
Image Loading Module
====================

This module handles the loading of medical images from various formats (NIfTI, DICOM)
into a standardized `Image` class. It abstracts away file format differences to provide
a consistent interface for the rest of the library.

Key Features:
-------------
- **Unified Image Class**: Stores 3D data, spacing, origin, direction, and modality.
- **Format Support**:
    - NIfTI (.nii, .nii.gz) via `nibabel`.
    - DICOM Series (directory of DICOM files) via `pydicom`.
    - Single DICOM files.
- **Automatic Detection**: `load_image` automatically detects format and dimensionality.
- **Robust DICOM Sorting**: Sorts slices based on spatial position and orientation.

Axis Conventions:
-----------------
All image arrays are stored in **(X, Y, Z)** order to match ITK/SimpleITK conventions:

- **X (axis 0)**: Left-Right direction (columns in DICOM terminology)
- **Y (axis 1)**: Anterior-Posterior direction (rows in DICOM terminology)
- **Z (axis 2)**: Superior-Inferior direction (slices)

This differs from raw DICOM and matplotlib conventions:

- **DICOM pixel_array**: Returns (Rows, Columns) = (Y, X) for 2D slices
- **Matplotlib imshow**: Expects (height, width) = (Y, X)

The loaders handle the necessary axis transformations automatically. When using
visualization utilities like `visualize_mask_overlay()`, slices are internally
transposed for correct display.

World Coordinate Frames:
------------------------
Origin and direction metadata are reported in the **native world frame of the
source format** and are *not* converted between frames:

- **DICOM** (series, single files, SEG): LPS+ (Left, Posterior, Superior), as
  defined by ``ImagePositionPatient`` / ``ImageOrientationPatient``.
- **NIfTI**: RAS+ (Right, Anterior, Superior), as defined by the NIfTI affine
  read via nibabel. (Note: SimpleITK converts NIfTI to LPS+ on load; this
  library does not.)

The X and Y axes of the two frames point in opposite directions, so origins and
direction matrices from different formats are **not directly comparable**. Do
not mix formats within a single geometric operation (e.g., a DICOM-derived
``reference_image`` with a NIfTI mask): geometry validation will fail or, worse,
repositioning may silently misalign. Keep an image and its masks in the same
format, or convert one externally beforehand. A ``UserWarning`` is emitted when
such mixing is detected.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import nibabel as nib
import numpy as np
import pydicom
from numpy import typing as npt
from numpy.typing import DTypeLike


def _direction_matrix(direction: Any) -> npt.NDArray[np.float64]:
    """Return a 3x3 world direction matrix, defaulting to identity."""
    if direction is None:
        return np.eye(3, dtype=np.float64)
    matrix = np.asarray(direction, dtype=np.float64)
    if matrix.shape == (9,):
        matrix = matrix.reshape((3, 3))
    if matrix.shape != (3, 3):
        raise ValueError(f"Direction must be a 3x3 matrix, got shape {matrix.shape}.")
    return matrix


def _warn_if_mixed_coordinate_frames(image: Image, reference: Image) -> None:
    """Warn when NIfTI- and DICOM-sourced images are combined geometrically.

    NIfTI geometry is in the RAS+ world frame while DICOM geometry is in LPS+
    (see module docstring). Detection is heuristic: images loaded from NIfTI
    carry ``modality == "Nifti"``, everything else is assumed DICOM-sourced.
    """
    if (image.modality == "Nifti") != (reference.modality == "Nifti"):
        warnings.warn(
            "Mixing NIfTI- and DICOM-sourced images: NIfTI geometry is in the "
            "RAS+ world frame while DICOM geometry is in LPS+, and no conversion "
            "is performed. Geometry validation/repositioning may fail or silently "
            "misalign. Use the same source format for an image and its masks, or "
            "convert one externally.",
            UserWarning,
            stacklevel=3,
        )


def _validate_geometry(
    target: Image,
    reference: Image,
    target_name: str = "target image",
    reference_name: str = "reference image",
    *,
    check_shape: bool = True,
    check_spacing: bool = True,
    check_origin: bool = True,
    check_direction: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """Validate that two images occupy the same voxel grid in physical space."""
    if check_shape and target.array.shape != reference.array.shape:
        raise ValueError(
            f"Dimension mismatch between {target_name} {target.array.shape} "
            f"and {reference_name} {reference.array.shape}."
        )
    if check_spacing and not np.allclose(target.spacing, reference.spacing, atol=atol, rtol=rtol):
        raise ValueError(
            f"Spacing mismatch between {target_name} {target.spacing} "
            f"and {reference_name} {reference.spacing}."
        )
    if check_origin and not np.allclose(target.origin, reference.origin, atol=atol, rtol=rtol):
        raise ValueError(
            f"Origin mismatch between {target_name} {target.origin} "
            f"and {reference_name} {reference.origin}."
        )
    if check_direction and not np.allclose(
        _direction_matrix(target.direction),
        _direction_matrix(reference.direction),
        atol=atol,
        rtol=rtol,
    ):
        raise ValueError(f"Direction mismatch between {target_name} and {reference_name}.")


def _normalize_direction_columns(
    matrix: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float64]:
    """Convert affine columns with voxel scaling into unit direction cosines."""
    direction = np.asarray(matrix, dtype=np.float64).copy()
    norms = np.linalg.norm(direction, axis=0)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    direction = direction / safe_norms
    direction[:, norms == 0.0] = np.eye(3)[:, norms == 0.0]
    return cast(npt.NDArray[np.float64], direction)


def _functional_group_item(group: Any, sequence_name: str) -> Any:
    """Return the first item of a DICOM functional-group sequence, or None."""
    sequence = getattr(group, sequence_name, None)
    if not isinstance(sequence, Sequence) or len(sequence) == 0:
        return None
    return sequence[0]


def _shared_functional_group_item(ds: Any, sequence_name: str) -> Any:
    """Look up a functional-group item that applies to all frames of an enhanced
    multiframe object.

    Checks SharedFunctionalGroupsSequence first, then falls back to the first
    item of PerFrameFunctionalGroupsSequence. Returns None if unavailable.
    """
    for parent_name in ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence"):
        parent = getattr(ds, parent_name, None)
        if isinstance(parent, Sequence) and len(parent) > 0:
            item = _functional_group_item(parent[0], sequence_name)
            if item is not None:
                return item
    return None


def _per_frame_positions(ds: Any, n_frames: int) -> list[npt.NDArray[np.float64]] | None:
    """Return ImagePositionPatient for every frame of an enhanced multiframe object.

    Returns None unless PerFrameFunctionalGroupsSequence provides a position for
    each of the ``n_frames`` frames.
    """
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if not isinstance(per_frame, Sequence) or len(per_frame) != n_frames:
        return None
    positions = []
    for group in per_frame:
        item = _functional_group_item(group, "PlanePositionSequence")
        ipp = getattr(item, "ImagePositionPatient", None) if item is not None else None
        if ipp is None or len(ipp) != 3:
            return None
        positions.append(np.array([float(ipp[0]), float(ipp[1]), float(ipp[2])], dtype=np.float64))
    return positions


def _rescale_params_from_functional_group(group: Any) -> tuple[float, float] | None:
    """Return RescaleSlope/Intercept from a DICOM functional group item if present."""
    item = _functional_group_item(group, "PixelValueTransformationSequence")
    if item is None:
        return None
    return (
        float(getattr(item, "RescaleSlope", 1.0)),
        float(getattr(item, "RescaleIntercept", 0.0)),
    )


def _get_dicom_frame_rescale(ds: Any, frame_index: int) -> tuple[float, float]:
    """Get frame-specific DICOM rescale parameters, falling back to shared/top-level tags."""
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    group = (
        per_frame[frame_index]
        if isinstance(per_frame, Sequence) and frame_index < len(per_frame)
        else None
    )
    if group is not None:
        params = _rescale_params_from_functional_group(group)
        if params is not None:
            return params

    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    group = shared[0] if isinstance(shared, Sequence) and len(shared) > 0 else None
    if group is not None:
        params = _rescale_params_from_functional_group(group)
        if params is not None:
            return params

    return (
        float(getattr(ds, "RescaleSlope", 1.0)),
        float(getattr(ds, "RescaleIntercept", 0.0)),
    )


def _apply_dicom_rescale(
    data: npt.NDArray[Any],
    ds: Any,
) -> npt.NDArray[Any]:
    """Apply top-level, shared, or per-frame DICOM rescale to stored pixel data."""
    if data.ndim != 3:
        slope, intercept = _get_dicom_frame_rescale(ds, 0)
        if slope != 1.0 or intercept != 0.0:
            return data.astype(np.float64) * slope + intercept
        return data

    frame_params = [_get_dicom_frame_rescale(ds, frame_idx) for frame_idx in range(data.shape[0])]
    if all(slope == 1.0 and intercept == 0.0 for slope, intercept in frame_params):
        return data

    scaled = data.astype(np.float64)
    for frame_idx, (slope, intercept) in enumerate(frame_params):
        if slope != 1.0 or intercept != 0.0:
            scaled[frame_idx] = scaled[frame_idx] * slope + intercept
    return scaled


@dataclass(eq=False)
class Image:
    """
    A standardized container for 3D medical image data and metadata.

    This class serves as the common interface for all image processing operations
    in the library, abstracting away the differences between file formats like
    DICOM and NIfTI.

    Note:
        ``origin`` and ``direction`` are expressed in the native world frame of
        the source format (LPS+ for DICOM, RAS+ for NIfTI) — see the module
        docstring ("World Coordinate Frames"). Equality (``==``) compares object
        identity: element-wise comparison of the array fields would be ambiguous,
        so dataclass-generated equality is disabled.

    Attributes:
        array (npt.NDArray[np.floating[Any]]): The 3D image data with shape (x, y, z).
        spacing (tuple[float, float, float]): Voxel spacing in millimeters (mm)
            along the (x, y, z) axes.
        origin (tuple[float, float, float]): World coordinates of the image origin
            (center of the first voxel) in millimeters (mm).
        direction (Optional[npt.NDArray[np.floating[Any]]]): 3x3 direction cosine matrix defining the
            orientation of the image axes in world space. Defaults to identity matrix.
        modality (str): The imaging modality (e.g., 'CT', 'MR', 'PT'). Defaults to 'Unknown'.
        source_mask (Optional[npt.NDArray[np.bool_]]): Optional boolean mask indicating
            which voxels contain valid source data (True) vs sentinel/invalid values (False).
            When set, preprocessing operations like resampling and filtering will exclude
            invalid voxels from interpolation/convolution to prevent sentinel value contamination.
            If None, all voxels are assumed to contain valid data (traditional behavior).

    Example:
        ```python
        import numpy as np
        from pictologics.loader import Image

        array = np.zeros((10, 10, 5), dtype=np.float32)
        image = Image(array=array, spacing=(1.0, 1.0, 2.0), origin=(0.0, 0.0, 0.0))
        print(image.array.shape)
        # (10, 10, 5)
        print(image.has_source_mask)
        # False
        ```
    """

    array: npt.NDArray[np.floating[Any]]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: Optional[npt.NDArray[np.floating[Any]]] = None
    modality: str = "Unknown"
    source_mask: Optional[npt.NDArray[np.bool_]] = None

    @property
    def has_source_mask(self) -> bool:
        """Whether this image has a source validity mask (indicating sentinel values were excluded)."""
        return self.source_mask is not None

    def with_source_mask(
        self,
        mask: "npt.NDArray[np.bool_] | npt.NDArray[np.integer[Any]] | Image",
    ) -> "Image":
        """
        Return a copy of this image with a source validity mask applied.

        The source mask indicates which voxels contain valid data (True) vs
        sentinel/invalid values (False). When set, spatial operations like
        resampling and filtering will exclude invalid voxels.

        Args:
            mask: Boolean array, integer array (>0 = valid), or Image object.
                  Must have the same shape as the image array.

        Returns:
            New Image with source_mask set.

        Raises:
            ValueError: If mask shape or physical geometry doesn't match image geometry.

        Example:
            ```python
            from pictologics.loader import load_image

            image = load_image("image_with_sentinel.nii.gz")
            roi_mask = load_image("roi_mask.nii.gz")

            # Use ROI mask as source validity mask
            image_with_source = image.with_source_mask(roi_mask)

            # Now resampling will exclude sentinel voxels
            from pictologics.preprocessing import resample_image
            resampled = resample_image(image_with_source, new_spacing=(1, 1, 1))
            ```
        """
        if isinstance(mask, Image):
            _validate_geometry(mask, self, "source mask", "image")
            # Explicit cast to bool to satisfy type checker
            mask_arr = (mask.array > 0).astype(bool)
        else:
            mask_arr = (mask > 0).astype(bool)

        if mask_arr.shape != self.array.shape:
            raise ValueError(
                f"Source mask shape {mask_arr.shape} must match image shape {self.array.shape}"
            )

        return Image(
            array=self.array.copy(),
            spacing=self.spacing,
            origin=self.origin,
            direction=(self.direction if self.direction is None else self.direction.copy()),
            modality=self.modality,
            source_mask=mask_arr.astype(bool),
        )


def create_full_mask(reference_image: Image, dtype: DTypeLike = np.uint8) -> Image:
    """Create a whole-image ROI mask matching a reference image.

    This utility is primarily used when a user does not provide a segmentation mask.
    The returned mask has the same geometry (shape, spacing, origin, direction) as
    the reference image and contains a value of 1 for every voxel.

    Args:
        reference_image: Image whose geometry should be copied.
        dtype: Numpy dtype to use for the mask array. Defaults to `np.uint8`.

    Returns:
        An `Image` mask with `array == 1` everywhere.

    Raises:
        ValueError: If the reference image does not have a valid 3D array.

    Example:
        ```python
        import numpy as np
        from pictologics.loader import Image, create_full_mask

        image = Image(array=np.zeros((10, 10, 5)), spacing=(1.0, 1.0, 2.0), origin=(0.0, 0.0, 0.0))
        mask = create_full_mask(image)
        print(mask.array.shape, mask.array.dtype)
        # (10, 10, 5) uint8
        print(mask.array.min(), mask.array.max())
        # 1 1
        ```
    """
    if reference_image.array.ndim != 3:
        raise ValueError(
            f"reference_image.array must be 3D, got shape {reference_image.array.shape}"
        )

    mask_array = np.ones(reference_image.array.shape, dtype=dtype)
    return Image(
        array=mask_array,
        spacing=reference_image.spacing,
        origin=reference_image.origin,
        direction=reference_image.direction,
        modality="mask",
    )


def _position_in_reference(
    image: Image,
    reference: Image,
    fill_value: float = 0.0,
    transpose_axes: tuple[int, int, int] | None = None,
    subvoxel_tolerance: float = 0.5,
    subvoxel_warning_threshold: float = 0.01,
    min_overlap_fraction: float = 0.5,
) -> Image:
    """
    Position a smaller (cropped) image within a larger reference volume.

    Uses spatial metadata (origin, spacing, direction) to calculate the correct
    position of the cropped image within the reference coordinate space. This is
    essential for working with cropped segmentation masks that need to be
    repositioned into the original full-sized image space.

    Args:
        image: The smaller/cropped image to position.
        reference: The reference image defining the target coordinate space and shape.
        fill_value: Value to use for voxels outside the cropped region (default: 0.0).
        transpose_axes: Optional tuple to transpose the image axes before positioning.
            Use this if the cropped image has a different axis order than expected.
            E.g., (0, 2, 1) swaps Y and Z axes.
        subvoxel_tolerance: Maximum permitted fractional-voxel offset before raising
            a ``ValueError`` (default: 0.5). At the default of 0.5 — the mathematical
            maximum achievable by rounding — valid masks always succeed without error.
            Lower this (e.g. 0.1) to catch suspiciously imprecise coordinates.
        subvoxel_warning_threshold: Fractional-voxel offset above which a
            ``UserWarning`` is emitted even though processing continues (default:
            0.01, roughly 1 % of a voxel). Offsets below this value are treated as
            floating-point noise and are snapped silently.
        min_overlap_fraction: Minimum fraction of the mask's voxel volume that must
            lie within the reference image space (default: 0.5). If the intersection
            volume is less than this fraction of the mask volume, a ``ValueError`` is
            raised to prevent silently loading a completely wrong mask. Set to 0.0
            to disable this check.

    Returns:
        Image: A new Image with the same shape as reference, containing the
            repositioned data from the input image.

    Raises:
        ValueError: If spacing, direction, sub-voxel tolerance, or minimum overlap
            fraction checks fail.
    """
    _warn_if_mixed_coordinate_frames(image, reference)

    # 0. Validate parameters
    if not 0.0 <= min_overlap_fraction <= 1.0:
        raise ValueError(f"min_overlap_fraction must be in [0.0, 1.0], got {min_overlap_fraction}.")
    if subvoxel_tolerance < 0.0:
        raise ValueError(f"subvoxel_tolerance must be >= 0.0, got {subvoxel_tolerance}.")
    if subvoxel_warning_threshold < 0.0:
        raise ValueError(
            f"subvoxel_warning_threshold must be >= 0.0, got {subvoxel_warning_threshold}."
        )
    if subvoxel_warning_threshold > subvoxel_tolerance:
        raise ValueError(
            f"subvoxel_warning_threshold ({subvoxel_warning_threshold}) must be <= "
            f"subvoxel_tolerance ({subvoxel_tolerance}). "
            "Offsets above the tolerance raise an error before a warning is emitted, "
            "so a threshold above the tolerance would never trigger a warning."
        )

    # 1. Apply optional axis transposition and keep voxel-axis metadata in sync.
    data = image.array
    img_spacing = np.asarray(image.spacing, dtype=np.float64)
    img_direction_arr = _direction_matrix(image.direction)
    if transpose_axes is not None:
        axes = tuple(transpose_axes)
        if sorted(axes) != [0, 1, 2]:
            raise ValueError(
                f"transpose_axes must be a permutation of (0, 1, 2), got {transpose_axes}."
            )
        data = np.transpose(data, axes)
        img_spacing = img_spacing[list(axes)]
        img_direction_arr = img_direction_arr[:, list(axes)]

    # 2. Validate spacing compatibility (allow 1% tolerance)
    ref_spacing = np.asarray(reference.spacing, dtype=np.float64)
    if not np.allclose(img_spacing, ref_spacing, rtol=0.01):
        raise ValueError(
            f"Spacing mismatch: image {tuple(img_spacing)} vs reference {reference.spacing}. "
            "Resampling would be required but is not yet supported."
        )

    # 3. Get spatial parameters
    ref_origin = np.asarray(reference.origin, dtype=np.float64)

    img_origin = np.asarray(image.origin, dtype=np.float64)
    ref_direction_arr = _direction_matrix(reference.direction)

    # 4. Check orientation compatibility
    # Use np.max(np.abs(...)) for robustness
    orientation_diff = np.max(np.abs(img_direction_arr - ref_direction_arr))
    if orientation_diff > 0.01:
        raise ValueError(
            f"Orientation mismatch detected (max diff={orientation_diff:.4f}). "
            "Reorientation or nearest-neighbor resampling is required before "
            "translation-only repositioning."
        )

    # 5. Calculate voxel offset
    # Convert image origin (world coords) to reference local voxel indices.
    world_offset = img_origin - ref_origin
    local_offset = np.linalg.pinv(ref_direction_arr) @ world_offset
    voxel_offset_float = local_offset / ref_spacing
    voxel_offset_rounded = np.round(voxel_offset_float)

    max_offset_diff = float(np.max(np.abs(voxel_offset_float - voxel_offset_rounded)))
    if max_offset_diff > subvoxel_tolerance:
        raise ValueError(
            f"Sub-voxel offset {max_offset_diff:.4f} voxels exceeds the configured "
            f"subvoxel_tolerance={subvoxel_tolerance}. "
            f"Computed per-axis offset: {tuple(np.round(voxel_offset_float, 4))}. "
            "Use a higher subvoxel_tolerance to allow nearest-voxel snapping, or "
            "resample the mask to the reference grid for precise alignment."
        )
    if max_offset_diff > subvoxel_warning_threshold:
        warnings.warn(
            f"Sub-voxel misalignment detected during repositioning "
            f"({max_offset_diff:.4f} voxels max drift). "
            f"Computed per-axis offset: {tuple(np.round(voxel_offset_float, 3))}. "
            "Snapping to nearest voxel. For precise sub-voxel alignment, "
            "resampling should be used.",
            UserWarning,
            stacklevel=2,
        )
    voxel_offset = voxel_offset_rounded.astype(int)

    # 6. Create output array with reference shape, filled with fill_value
    output = np.full(reference.array.shape, fill_value, dtype=data.dtype)

    # 7. Calculate copy ranges with boundary clipping
    # Source (cropped image) ranges
    src_start = np.array([max(0, -voxel_offset[i]) for i in range(3)])
    src_end = np.array(
        [min(data.shape[i], reference.array.shape[i] - voxel_offset[i]) for i in range(3)]
    )

    # Destination (reference) ranges
    dst_start = np.array([max(0, voxel_offset[i]) for i in range(3)])
    dst_end = dst_start + (src_end - src_start)

    # 8. Validate overlap fraction between mask and reference image
    intersection_vol = int(np.prod(np.maximum(0, src_end - src_start)))
    mask_vol = int(np.prod(data.shape))
    overlap_fraction = intersection_vol / mask_vol if mask_vol > 0 else 0.0
    if overlap_fraction < min_overlap_fraction:
        raise ValueError(
            f"Mask overlaps only {overlap_fraction:.1%} of its own volume with the "
            f"reference image (required: {min_overlap_fraction:.1%}). "
            f"Mask origin: {image.origin}, Reference origin: {reference.origin}. "
            "Check that the correct mask is being loaded for this image. "
            "Set min_overlap_fraction=0.0 to disable this check."
        )

    if intersection_vol == 0:
        # Reached only when min_overlap_fraction=0.0 — preserve backward-compatible
        # warning behaviour (return empty volume rather than raising).
        warnings.warn(
            "Cropped image does not overlap with reference volume. "
            f"Image origin: {image.origin}, Reference origin: {reference.origin}",
            UserWarning,
            stacklevel=2,
        )
        return Image(
            array=output,
            spacing=reference.spacing,
            origin=reference.origin,
            direction=reference.direction,
            modality=image.modality,
        )

    # 9. Copy data to correct position
    output[
        dst_start[0] : dst_end[0],
        dst_start[1] : dst_end[1],
        dst_start[2] : dst_end[2],
    ] = data[
        src_start[0] : src_end[0],
        src_start[1] : src_end[1],
        src_start[2] : src_end[2],
    ]

    # 10. Return new Image with reference geometry
    return Image(
        array=output,
        spacing=reference.spacing,
        origin=reference.origin,
        direction=reference.direction,
        modality=image.modality,
    )


def _find_best_dicom_series_dir(root: Path) -> Path:
    """Recursively find the subdirectory with the most DICOM files."""
    if not root.exists():
        raise ValueError(f"Path does not exist: {root}")

    best_dir = None
    best_count = -1

    # Include root itself in the search
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]

    found_any = False

    for d in candidates:
        try:
            # Count DICOMs using pydicom's robust check
            count = sum(1 for f in d.iterdir() if f.is_file() and pydicom.misc.is_dicom(f))
            if count > 0:
                found_any = True

            if count > best_count:
                best_count = count
                best_dir = d
        except OSError:
            continue

    if not found_any or best_dir is None or best_count == 0:
        raise ValueError(f"No DICOM files found in {root} or its subdirectories.")

    return best_dir


def _is_dicom_seg(path: str) -> bool:
    """Check if a DICOM file is a Segmentation object.

    Checks if the SOPClassUID matches the DICOM Segmentation Storage class
    (1.2.840.10008.5.1.4.1.1.66.4).

    Args:
        path: Path to the potential DICOM file.

    Returns:
        True if the file is a DICOM SEG object, False otherwise.
    """
    try:
        dcm = pydicom.dcmread(path, stop_before_pixels=True)
        # DICOM Segmentation Storage SOP Class UID
        return str(getattr(dcm, "SOPClassUID", "")) == "1.2.840.10008.5.1.4.1.1.66.4"
    except Exception:
        return False


def load_image(
    path: str,
    dataset_index: int = 0,
    recursive: bool = False,
    reference_image: Optional[Image] = None,
    transpose_axes: tuple[int, int, int] | None = None,
    fill_value: float = 0.0,
    apply_rescale: bool = True,
    subvoxel_tolerance: float = 0.5,
    subvoxel_warning_threshold: float = 0.01,
    min_overlap_fraction: float = 0.5,
) -> Image:
    """
    Load a medical image from a file path or directory.

    This is the main entry point for loading data. It automatically detects whether
    the input is a NIfTI file, DICOM directory/file (single DICOM or series), or
    a DICOM Segmentation (SEG) object and standardizes it into an `Image` object.

    The resulting image array is always 3D with dimensions (x, y, z).

    Note:
        For DICOM SEG files, this function uses ``pictologics.loaders.load_seg()``
        internally. For more control over segment extraction (e.g., selecting specific
        segments or extracting them separately), use ``load_seg()`` directly.
        ``dataset_index`` and ``fill_value`` do not apply to SEG files and are
        ignored with a ``UserWarning`` if set to non-default values.

    Warning:
        NIfTI and DICOM geometry live in different world coordinate frames
        (RAS+ vs LPS+) and are not converted — do not mix formats between an
        image and its ``reference_image``/masks. See the module docstring
        ("World Coordinate Frames").

    Args:
        path (str): The absolute or relative path to the image file (e.g., .nii.gz,
            .dcm or file with no extension) or the directory containing DICOM files.
        dataset_index (int, optional): For multi-volume datasets, specifies which
            volume to extract (0-indexed). This works for:

            - **4D NIfTI files**: Selects which time point/volume to load.
            - **Multi-phase DICOM series**: Selects which phase to load (e.g., cardiac
              phases, temporal positions, echo numbers). Use
              ``pictologics.utilities.get_dicom_phases()`` to discover available phases.

            Defaults to 0 (the first volume/phase).
        recursive (bool, optional): If True and `path` is a directory, recursively searches
            subdirectories and loads the DICOM series from the folder containing the most
            DICOM files. Defaults to False.
        reference_image (Optional[Image]): If provided and the loaded image has different
            dimensions than the reference, it will be repositioned into the reference
            coordinate space using spatial metadata (origin, spacing). This is useful for
            loading cropped segmentation masks that need to match a full-sized image.
        transpose_axes (tuple[int, int, int] | None): Optional axis transposition to apply
            before repositioning. Use this if the mask's axis order differs from the reference.
            E.g., (0, 2, 1) swaps Y and Z axes. Only used when reference_image is provided;
            when set, repositioning is performed even if the loaded shape already matches
            the reference.
        fill_value (float): Fill value for regions outside the loaded image when
            repositioning (default: 0.0). Only used when reference_image is provided.
        apply_rescale (bool): If True (default), apply RescaleSlope and RescaleIntercept
            transformation for DICOM files to convert stored pixel values to real-world
            values (e.g., Hounsfield Units for CT). NIfTI files always apply their scaling
            factors via nibabel's get_fdata(). Set to False if you need raw stored values.
        subvoxel_tolerance (float): Maximum permitted fractional-voxel offset when
            repositioning (default: 0.5). Only used when reference_image is provided.
            See ``_position_in_reference`` for full description.
        subvoxel_warning_threshold (float): Fractional-voxel drift above which a
            ``UserWarning`` is emitted during repositioning (default: 0.01). Only used
            when reference_image is provided.
        min_overlap_fraction (float): Minimum fraction of the mask volume that must
            intersect with the reference image space (default: 0.5). Only used when
            reference_image is provided.

    Returns:
        Image: An `Image` object containing the 3D numpy array and metadata (spacing, origin, etc.).

    Raises:
        ValueError: If the path does not exist, the file format is not supported,
            or the file is corrupt/unreadable.

    Example:
        **Loading a NIfTI file:**
        ```python
        from pictologics.loader import load_image

        # Load a standard brain scan
        img = load_image("data/brain.nii.gz")
        print(f"Image shape: {img.array.shape}")
        # Output: Image shape: (256, 256, 128)
        ```

        **Loading a DICOM series:**
        ```python
        # Load a CT scan from a folder of DICOM files
        img_ct = load_image("data/patients/001/CT_scan/")
        print(f"Voxel spacing: {img_ct.spacing}")
        # Output: Voxel spacing: (0.97, 0.97, 2.5)
        ```

        **Loading a single DICOM file:**
        ```python
        # Load a single DICOM file (even without .dcm extension)
        img_slice = load_image("data/slice_001")
        print(f"Modality: {img_slice.modality}")
        ```

        **Recursive DICOM loading:**
        ```python
        # Finds the deep subfolder with actual DICOM files
        img = load_image("data/patients/001/", recursive=True)
        ```

        **Loading a specific volume from a 4D file:**
        ```python
        # Load the 5th time point from a 4D fMRI file
        fmri_vol = load_image("data/fmri.nii.gz", dataset_index=4)
        ```

        **Loading a cropped mask and repositioning to match main image:**
        ```python
        main_img = load_image("ct_scan/")
        mask = load_image("cropped_mask.dcm", reference_image=main_img)
        # mask now has same shape as main_img
        ```

        **Loading a DICOM SEG file (auto-detected):**
        ```python
        # DICOM SEG files are automatically detected and loaded
        seg = load_image("segmentation.dcm")
        print(f"Modality: {seg.modality}")  # Output: Modality: SEG
        # Segments are combined into a label image by default
        ```

        **Loading a specific phase from a multi-phase DICOM series:**
        ```python
        from pictologics.utilities import get_dicom_phases

        # Discover available phases
        phases = get_dicom_phases("cardiac_ct/")
        print(f"Found {len(phases)} phases")
        for p in phases:
            print(f"  {p.index}: {p.label} ({p.num_slices} slices)")

        # Load the 5th phase (40%)
        img = load_image("cardiac_ct/", dataset_index=4)
        ```
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"The specified path does not exist: {path}")

    try:
        if path_obj.is_dir():
            target_path = path_obj
            if recursive:
                target_path = _find_best_dicom_series_dir(path_obj)
            loaded_image = _load_dicom_series(target_path, dataset_index, apply_rescale)
        elif path.lower().endswith((".nii", ".nii.gz")):
            loaded_image = _load_nifti(path, dataset_index)
        else:
            # Attempt to load as a single DICOM file if extension is not NIfTI
            # Check if it's a DICOM SEG file first
            if _is_dicom_seg(path):
                from pictologics.loaders.seg_loader import load_seg

                if dataset_index != 0 or fill_value != 0.0:
                    warnings.warn(
                        "dataset_index and fill_value are ignored for DICOM SEG files. "
                        "Use pictologics.loaders.load_seg() directly for segment selection.",
                        UserWarning,
                        stacklevel=2,
                    )
                seg_result = load_seg(
                    path,
                    reference_image=reference_image,
                    transpose_axes=transpose_axes,
                    subvoxel_tolerance=subvoxel_tolerance,
                    subvoxel_warning_threshold=subvoxel_warning_threshold,
                    min_overlap_fraction=min_overlap_fraction,
                )
                # load_seg can return dict when combine_segments=False, but here we use default
                if isinstance(seg_result, dict):
                    # Should not happen with default args, but handle gracefully
                    return next(iter(seg_result.values()))
                # Return early since reference alignment is handled by load_seg
                return seg_result

            try:
                loaded_image = _load_dicom_file(path, apply_rescale)
            except Exception as read_error:
                raise ValueError(
                    f"Unsupported file format or unable to read file: {path}"
                ) from read_error
    except Exception as e:
        # Re-raise ValueErrors directly, wrap others
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Failed to load image from '{path}': {e}") from e

    # Apply repositioning if reference_image is provided and shapes differ,
    # or if an explicit axis transposition was requested (a transposed mask may
    # coincidentally have the same shape as the reference, e.g. cubic volumes).
    if reference_image is not None:
        if loaded_image.array.shape != reference_image.array.shape or transpose_axes is not None:
            loaded_image = _position_in_reference(
                loaded_image,
                reference_image,
                fill_value,
                transpose_axes,
                subvoxel_tolerance,
                subvoxel_warning_threshold,
                min_overlap_fraction,
            )
        else:
            _warn_if_mixed_coordinate_frames(loaded_image, reference_image)
            _validate_geometry(loaded_image, reference_image, "loaded image", "reference image")

    return loaded_image


def load_and_merge_images(
    image_paths: list[str],
    reference_image: Optional[Image] = None,
    conflict_resolution: str = "max",
    dataset_index: int = 0,
    recursive: bool = False,
    binarize: bool | int | list[int] | tuple[int, int] | None = None,
    reposition_to_reference: bool = False,
    transpose_axes: tuple[int, int, int] | None = None,
    fill_value: float = 0.0,
    relabel_masks: bool = False,
    apply_rescale: bool = True,
    subvoxel_tolerance: float = 0.5,
    subvoxel_warning_threshold: float = 0.01,
    min_overlap_fraction: float = 0.5,
) -> Image:
    """
    Load multiple images (e.g., masks or partial scans) and merge them into a single image.

    This function loads images from the provided paths, validates that they all share
    the same geometry (dimensions, spacing, origin, direction), and merges them
    according to the specified conflict resolution strategy.

    **Use Cases:**
    - Merging multiple segmentation masks into a single ROI.
    - Merging split image volumes (though typically less common than mask merging).
    - Merging cropped/bounding-box segmentation masks (with `reposition_to_reference=True`).

    **Format & Path Support:**
    Since this function uses `load_image` internally for each path, it supports:
    - **NIfTI files** (.nii, .nii.gz).
    - **DICOM series** (directories containing DICOM files).
    - **Single DICOM files** (with or without .dcm extension).
    - **Nested directories** (if paths point to folders containing DICOMs).

    Args:
        image_paths (list[str]): List of absolute or relative paths to the images.
            These can be file paths or directory paths.
        reference_image (Optional[Image]): An optional reference image (e.g., the scan
            corresponding to the masks). If provided, the merged image is validated
            against this image's geometry. Required when `reposition_to_reference=True`.
        conflict_resolution (str): Strategy to resolve voxel values when multiple images
            have non-zero values at the same location. Options:
            - 'max': Use the maximum value (default).
            - 'min': Use the minimum value.
            - 'first': Keep the value from the first image encountered (earlier in list).
            - 'last': Overwrite with the value from the last image encountered (later in list).
        dataset_index (int, optional): For multi-volume datasets, specifies which
            volume to extract for all images (0-indexed). This works for:

            - **4D NIfTI files**: Selects which time point/volume to load.
            - **Multi-phase DICOM series**: Selects which phase to load (e.g., cardiac
              phases, temporal positions, echo numbers). Use
              ``pictologics.utilities.get_dicom_phases()`` to discover available phases.

            Defaults to 0 (the first volume/phase).
        recursive (bool, optional): If True, recursively searches subdirectories
            for each path in `image_paths`. Defaults to False.
        binarize (bool | int | list[int] | tuple[int, int] | None, optional):
            Rules for binarizing the merged image.
            - `None` (default): No binarization.
            - `True`: Sets all voxels > 0 to 1, others to 0.
            - `int` (e.g., 2): Sets voxels == value to 1, others to 0.
            - `list[int]` (e.g., [1, 2]): Sets voxels in list to 1, others to 0.
            - `tuple[int, int]` (e.g., (1, 10)): Sets voxels in inclusive range to 1, others to 0.
        reposition_to_reference (bool): If True and reference_image is provided,
            each loaded image will be repositioned into the reference coordinate
            space before merging. This is required when loading cropped segmentation
            masks that have different dimensions than the reference. Geometry validation
            is performed AFTER repositioning. Defaults to False.
        transpose_axes (tuple[int, int, int] | None): Axis transposition to apply
            when repositioning. E.g., (0, 2, 1) swaps Y and Z axes.
            Only used when `reposition_to_reference=True`.
        fill_value (float): Fill value for regions outside cropped masks when
            repositioning (default: 0.0). Only used when `reposition_to_reference=True`.
        relabel_masks (bool): If True, assigns unique label values (1, 2, 3, ...)
            to each mask file based on its order in `image_paths`. This converts
            binary [0,1] masks into multi-label masks where each file gets a
            distinct label, useful for visualization with different colors.
            Label assignment respects the order of `image_paths`. Defaults to False.
        apply_rescale (bool): If True (default), apply RescaleSlope and RescaleIntercept
            transformation for DICOM files to convert stored pixel values to real-world
            values (e.g., Hounsfield Units for CT). Set to False if you need raw stored values.
        subvoxel_tolerance (float): Maximum permitted fractional-voxel offset when
            repositioning (default: 0.5). Only used when ``reposition_to_reference=True``.
            See ``_position_in_reference`` for full description.
        subvoxel_warning_threshold (float): Fractional-voxel drift above which a
            ``UserWarning`` is emitted during repositioning (default: 0.01). Only used
            when ``reposition_to_reference=True``.
        min_overlap_fraction (float): Minimum fraction of each mask volume that must
            intersect with the reference image space (default: 0.5). Only used when
            ``reposition_to_reference=True``.

    Note:
        The `binarize` parameter is intended for **mask filtering** (e.g., selecting specific ROI labels).
        To filter image intensity values (e.g., HU ranges), use the preprocessing steps in the
        radiomics pipeline configuration instead.

    Returns:
        Image: A new `Image` object containing the merged data.

    Raises:
        ValueError: If `image_paths` is empty, if an invalid `conflict_resolution` is provided,
            if `reposition_to_reference=True` but `reference_image` is not provided,
            or if the images (or reference) have mismatched geometries.

    Example:
        **Merging cropped segmentation masks:**
        ```python
        main_img = load_image("ct_scan/", recursive=True)
        seg_paths = [str(f) for f in Path("masks/").glob("*.dcm")]

        merged = load_and_merge_images(
            seg_paths,
            reference_image=main_img,
            reposition_to_reference=True,
            conflict_resolution="max",
        )
        ```
    """
    if not image_paths:
        raise ValueError("image_paths cannot be empty.")

    valid_strategies = {"max", "min", "first", "last"}
    if conflict_resolution not in valid_strategies:
        raise ValueError(
            f"Invalid conflict_resolution '{conflict_resolution}'. "
            f"Must be one of {valid_strategies}."
        )

    if reposition_to_reference and reference_image is None:
        raise ValueError("reference_image must be provided when reposition_to_reference=True.")

    if reposition_to_reference:
        # Mode: Reposition each image to reference space, then merge
        assert reference_image is not None  # Already validated above

        # Initialize merged array with reference geometry
        merged_array = np.full(reference_image.array.shape, fill_value, dtype=np.float64)

        for i, path in enumerate(image_paths):
            try:
                current_image = load_image(
                    path,
                    dataset_index=dataset_index,
                    recursive=recursive,
                    apply_rescale=apply_rescale,
                )
            except Exception as e:
                raise ValueError(f"Failed to load image '{path}': {e}") from e

            # Reposition to reference space
            repositioned = _position_in_reference(
                current_image,
                reference_image,
                fill_value,
                transpose_axes,
                subvoxel_tolerance,
                subvoxel_warning_threshold,
                min_overlap_fraction,
            )

            # Validate geometry after repositioning
            _validate_geometry(
                repositioned,
                reference_image,
                f"repositioned image '{path}'",
                "reference image",
            )

            current_array = repositioned.array

            # Apply relabeling: replace all non-zero values with mask index + 1
            if relabel_masks:
                label_value = i + 1  # 1-indexed labels
                current_array = np.where(current_array != fill_value, label_value, fill_value)

            # Merge with conflict resolution
            if i == 0:
                # First image: just copy non-fill values
                non_fill_mask = current_array != fill_value
                merged_array[non_fill_mask] = current_array[non_fill_mask]
            else:
                # Subsequent images: apply conflict resolution
                # Overlap: non-fill in both
                overlap_mask = (merged_array != fill_value) & (current_array != fill_value)
                # New data: fill in merged, non-fill in current
                new_data_mask = (merged_array == fill_value) & (current_array != fill_value)

                # Apply new data
                merged_array[new_data_mask] = current_array[new_data_mask]

                # Resolve conflicts
                if np.any(overlap_mask):
                    if conflict_resolution == "max":
                        merged_array[overlap_mask] = np.maximum(
                            merged_array[overlap_mask], current_array[overlap_mask]
                        )
                    elif conflict_resolution == "min":
                        merged_array[overlap_mask] = np.minimum(
                            merged_array[overlap_mask], current_array[overlap_mask]
                        )
                    elif conflict_resolution == "last":
                        merged_array[overlap_mask] = current_array[overlap_mask]
                    elif conflict_resolution == "first":
                        pass  # Keep existing values

        # Use reference geometry for output
        consensus_spacing = reference_image.spacing
        consensus_origin = reference_image.origin
        consensus_direction = reference_image.direction

    else:
        # Mode: Standard merging with strict geometry validation
        # Load the first image to serve as the consensus geometry
        try:
            consensus_image = load_image(
                image_paths[0],
                dataset_index=dataset_index,
                recursive=recursive,
                apply_rescale=apply_rescale,
            )
        except Exception as e:
            raise ValueError(f"Failed to load first image '{image_paths[0]}': {e}") from e

        merged_array = consensus_image.array.astype(np.float64)

        # Apply relabeling for the first image
        if relabel_masks:
            merged_array = np.where(merged_array != 0, 1, 0).astype(merged_array.dtype)

        # Iterate through remaining images
        for idx, path in enumerate(image_paths[1:], start=2):
            try:
                current_image = load_image(
                    path,
                    dataset_index=dataset_index,
                    recursive=recursive,
                    apply_rescale=apply_rescale,
                )
            except Exception as e:
                raise ValueError(f"Failed to load image '{path}': {e}") from e

            _validate_geometry(current_image, consensus_image, f"image '{path}'", "consensus image")

            current_array = current_image.array

            # Apply relabeling: replace all non-zero values with mask index
            if relabel_masks:
                label_value = idx  # idx starts at 2 for second file
                current_array = np.where(current_array != 0, label_value, 0).astype(
                    current_array.dtype
                )

            # Identify regions
            overlap_mask = (merged_array != 0) & (current_array != 0)
            new_data_mask = (merged_array == 0) & (current_array != 0)

            # Apply new data
            merged_array[new_data_mask] = current_array[new_data_mask]

            # Resolve conflicts
            if np.any(overlap_mask):
                if conflict_resolution == "max":
                    merged_array[overlap_mask] = np.maximum(
                        merged_array[overlap_mask], current_array[overlap_mask]
                    )
                elif conflict_resolution == "min":
                    merged_array[overlap_mask] = np.minimum(
                        merged_array[overlap_mask], current_array[overlap_mask]
                    )
                elif conflict_resolution == "last":
                    merged_array[overlap_mask] = current_array[overlap_mask]
                elif conflict_resolution == "first":
                    pass  # Already have the 'first' value

        consensus_spacing = consensus_image.spacing
        consensus_origin = consensus_image.origin
        consensus_direction = consensus_image.direction

        # Validate against reference image if provided (for non-reposition mode)
        if reference_image is not None:
            final_merged_image = Image(
                array=merged_array,
                spacing=consensus_spacing,
                origin=consensus_origin,
                direction=consensus_direction,
                modality="Image",
            )
            _validate_geometry(
                final_merged_image, reference_image, "merged image", "reference image"
            )

    # Apply binarization if requested
    if binarize is not None:
        mask_out: npt.NDArray[np.floating[Any]] = np.zeros_like(merged_array, dtype=np.uint8)
        if isinstance(binarize, bool) and binarize is True:
            mask_out[merged_array > 0] = 1
        elif isinstance(binarize, int) and not isinstance(binarize, bool):
            mask_out[merged_array == binarize] = 1
        elif isinstance(binarize, list):
            mask_out[np.isin(merged_array, binarize)] = 1
        elif isinstance(binarize, tuple) and len(binarize) == 2:
            mask_out[(merged_array >= binarize[0]) & (merged_array <= binarize[1])] = 1
        else:
            if binarize is not False:
                raise ValueError(f"Unsupported binarize value: {binarize}")
            mask_out = merged_array

        if binarize is not False:
            merged_array = mask_out.astype(np.float64)

    return Image(
        array=merged_array,
        spacing=consensus_spacing,
        origin=consensus_origin,
        direction=consensus_direction,
        modality="MergedImage",
    )


def _ensure_3d(
    array: npt.NDArray[np.floating[Any]], dataset_index: int = 0
) -> npt.NDArray[np.floating[Any]]:
    """
    Ensure the input array is strictly 3D (x, y, z).

    This helper function handles different input dimensionalities:
    - **2D (x, y)**: Promoted to 3D by adding a singleton dimension (x, y, 1).
    - **3D (x, y, z)**: Returned as is.
    - **4D (x, y, z, t)**: The volume at `dataset_index` is extracted.

    Args:
        array (npt.NDArray[np.floating[Any]]): The input numpy array of arbitrary dimensions.
        dataset_index (int): The index of the volume to extract if the input is 4D.

    Returns:
        npt.NDArray[np.floating[Any]]: A 3D numpy array.

    Raises:
        ValueError: If the array has an unsupported number of dimensions (not 2, 3, or 4)
            or if `dataset_index` is invalid for the 4D array.
    """
    ndim = array.ndim
    if ndim == 2:
        # (x, y) -> (x, y, 1)
        return array[..., np.newaxis]
    elif ndim == 3:
        return array
    elif ndim == 4:
        if dataset_index < 0 or dataset_index >= array.shape[3]:
            raise ValueError(
                f"Dataset index {dataset_index} is out of bounds for 4D image "
                f"with {array.shape[3]} volumes."
            )
        return array[..., dataset_index]
    else:
        raise ValueError(f"Unsupported array dimensionality: {ndim}. Expected 2, 3, or 4.")


def _load_nifti(path: str, dataset_index: int = 0) -> Image:
    """
    Load a NIfTI file (.nii or .nii.gz) using the nibabel library.

    This function extracts the image data, voxel spacing, origin, and direction
    from the NIfTI header.

    Args:
        path (str): Path to the NIfTI file.
        dataset_index (int): The volume index to load if the file is 4D.

    Returns:
        Image: A standardized `Image` object.

    Raises:
        ValueError: If nibabel fails to load the file (e.g., corrupt header).
    """
    try:
        nii_img = nib.load(path)  # type: ignore
    except Exception as e:
        raise ValueError(f"Could not load NIfTI file '{path}': {e}") from e

    # Load image data as float64 to preserve precision
    array = nii_img.get_fdata()  # type: ignore
    array = _ensure_3d(array, dataset_index)

    # Extract metadata
    header = nii_img.header  # type: ignore
    zooms = header.get_zooms()  # type: ignore

    # Ensure spacing has at least 3 dimensions (pad with 1.0 if needed)
    spacing_list = [float(z) for z in zooms]
    while len(spacing_list) < 3:
        spacing_list.append(1.0)
    spacing = (spacing_list[0], spacing_list[1], spacing_list[2])

    # Extract affine for origin and direction
    affine = nii_img.affine  # type: ignore
    origin = (float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3]))
    direction = _normalize_direction_columns(affine[:3, :3])

    return Image(
        array=array,
        spacing=spacing,
        origin=origin,
        direction=direction,
        modality="Nifti",
    )


def _load_dicom_series(
    path: str | Path, dataset_index: int = 0, apply_rescale: bool = True
) -> Image:
    """
    Load a DICOM series (a set of DICOM files) from a directory.

    This function reads all DICOM files in the directory, detects multi-phase
    acquisitions (e.g., cardiac phases, temporal positions), and loads the
    requested phase. Slices are sorted spatially to reconstruct the 3D volume.

    **Multi-Phase Detection:**
    The function automatically detects multi-phase series using the same logic
    as ``DicomDatabase``. Detection is based on:
    - Cardiac phase percentage
    - Temporal position
    - Trigger time
    - Acquisition number
    - Echo number
    - Duplicate spatial positions (fallback)

    **Sorting Logic:**
    Slices are sorted based on the projection of their `ImagePositionPatient`
    onto the slice normal vector (derived from `ImageOrientationPatient`).
    This robustly handles axial, sagittal, coronal, and oblique acquisitions.
    If spatial tags are missing, it falls back to `InstanceNumber`.

    Args:
        path: Directory containing the DICOM files.
        dataset_index: For multi-phase series, which phase to load (0-indexed).
            Default is 0, which loads the first (or only) phase.
        apply_rescale: If True (default), apply RescaleSlope and RescaleIntercept
            to convert stored pixel values to real-world values (e.g., Hounsfield
            Units for CT). Set to False to get raw stored values.

    Returns:
        Image: A standardized `Image` object.

    Raises:
        ValueError: If no DICOM files are found, if they cannot be read/sorted,
            or if dataset_index is out of range for the available phases.

    See Also:
        ``pictologics.utilities.get_dicom_phases()``: Discover available phases.
    """
    from pictologics.utilities.dicom_utils import MULTI_PHASE_TAGS, split_dicom_phases

    # List candidate files; non-DICOM files are skipped during the metadata
    # read below (dcmread rejects them), avoiding a separate is_dicom pass
    # that would open every file twice.
    path_obj = Path(path)
    files = [p for p in path_obj.iterdir() if p.is_file()]
    if not files:
        raise ValueError(f"No DICOM files found in directory: {path}")

    # Extract metadata for phase detection
    file_metadata: list[dict[str, Any]] = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
            meta: dict[str, Any] = {
                "file_path": f,
                "InstanceNumber": getattr(dcm, "InstanceNumber", None),
            }
            # Extract position
            try:
                ipp = dcm.ImagePositionPatient
                meta["ImagePositionPatient"] = (
                    float(ipp[0]),
                    float(ipp[1]),
                    float(ipp[2]),
                )
            except (AttributeError, IndexError, TypeError):
                meta["ImagePositionPatient"] = None

            # Extract multi-phase tags
            for tag in MULTI_PHASE_TAGS:
                val = getattr(dcm, tag, None)
                if val is not None:
                    meta[tag] = val

            file_metadata.append(meta)
        except Exception:
            continue

    if not file_metadata:
        raise ValueError(f"Could not read any DICOM files from: {path}")

    # Detect and split phases
    phases = split_dicom_phases(file_metadata)

    # Validate dataset_index
    if dataset_index >= len(phases):
        raise ValueError(
            f"dataset_index {dataset_index} is out of range. "
            f"Series has {len(phases)} phase(s) (valid indices: 0-{len(phases) - 1}). "
            f"Use pictologics.utilities.get_dicom_phases() to discover available phases."
        )

    # Get files for the requested phase
    selected_files = [m["file_path"] for m in phases[dataset_index]]

    # Read all slices for the selected phase
    try:
        slices = [pydicom.dcmread(f) for f in selected_files]
    except Exception as e:
        raise ValueError(f"Error reading DICOM files in '{path}': {e}") from e

    # Determine sorting direction
    # Calculate the normal vector of the slice plane
    ref = slices[0]
    try:
        orientation = np.array(ref.ImageOrientationPatient, dtype=float)
        row_cosines = orientation[:3]
        col_cosines = orientation[3:]
        slice_normal = np.cross(row_cosines, col_cosines)
    except (AttributeError, ValueError):
        # Fallback to simple Z-sorting if orientation is missing
        slice_normal = np.array([0, 0, 1.0])

    # Sort slices by projection of position onto the normal vector
    try:
        slices.sort(
            key=lambda s: np.dot(np.array(s.ImagePositionPatient, dtype=float), slice_normal)
        )
    except AttributeError:
        # Fallback to InstanceNumber if ImagePositionPatient is missing
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))

    # Stack pixel data
    # pydicom pixel_array is (Rows, Columns) -> (Y, X)
    # We want (X, Y, Z)
    try:
        pixel_data = []
        for s in slices:
            pixels = s.pixel_array
            if apply_rescale:
                slope = float(getattr(s, "RescaleSlope", 1.0))
                intercept = float(getattr(s, "RescaleIntercept", 0.0))
                if slope != 1.0 or intercept != 0.0:
                    pixels = pixels.astype(np.float64) * slope + intercept
            pixel_data.append(pixels)
    except Exception as e:
        raise ValueError("Failed to extract pixel arrays from DICOM slices.") from e

    volume = np.stack(pixel_data, axis=-1)  # Result: (Y, X, Z)
    volume = np.swapaxes(volume, 0, 1)  # Result: (X, Y, Z)
    volume = _ensure_3d(volume)

    # Extract metadata from the first slice (reference)
    ref = slices[0]

    # Spacing
    try:
        pixel_spacing = ref.PixelSpacing
        spacing_x = float(pixel_spacing[1])  # Column spacing (X)
        spacing_y = float(pixel_spacing[0])  # Row spacing (Y)

        # Slice thickness / spacing
        if hasattr(ref, "SpacingBetweenSlices"):
            spacing_z = float(ref.SpacingBetweenSlices)
        elif hasattr(ref, "SliceThickness"):
            spacing_z = float(ref.SliceThickness)
        else:
            # Estimate from position difference if multiple slices
            if len(slices) > 1:
                p1 = np.array(slices[0].ImagePositionPatient)
                p2 = np.array(slices[1].ImagePositionPatient)
                spacing_z = float(np.linalg.norm(p2 - p1))
            else:
                spacing_z = 1.0

        spacing = (spacing_x, spacing_y, spacing_z)
    except (AttributeError, IndexError):
        spacing = (1.0, 1.0, 1.0)

    # Origin
    try:
        origin = (
            float(ref.ImagePositionPatient[0]),
            float(ref.ImagePositionPatient[1]),
            float(ref.ImagePositionPatient[2]),
        )
    except AttributeError:
        origin = (0.0, 0.0, 0.0)

    # Direction
    try:
        orientation = np.array(ref.ImageOrientationPatient, dtype=float)
        row_cosines = orientation[:3]
        col_cosines = orientation[3:]
        slice_cosine = np.cross(row_cosines, col_cosines)
        direction = np.stack([row_cosines, col_cosines, slice_cosine], axis=1)
    except (AttributeError, ValueError):
        direction = np.eye(3)

    return Image(
        array=volume,
        spacing=spacing,
        origin=origin,
        direction=direction,
        modality=getattr(ref, "Modality", "DICOM"),
    )


def _load_dicom_file(path: str, apply_rescale: bool = True) -> Image:
    """
    Load a single DICOM file as a 3D image.

    This handles both 2D DICOM files (X-rays, single slices) and 3D DICOM files
    (segmentation objects, multiframe images). The resulting image will be in
    (X, Y, Z) format with at least 1 slice in the Z dimension.

    Geometry tags (PixelSpacing, SpacingBetweenSlices/SliceThickness,
    ImagePositionPatient, ImageOrientationPatient) are read from the top level,
    falling back to the Shared/PerFrame functional groups used by enhanced
    multiframe objects. When per-frame positions are available, frames are
    sorted spatially (projection onto the slice normal, like
    ``_load_dicom_series``) and the Z spacing is derived from consecutive frame
    positions if no spacing tag is present.

    Args:
        path (str): Path to the DICOM file.
        apply_rescale (bool): If True (default), apply RescaleSlope and RescaleIntercept
            to convert stored pixel values to real-world values (e.g., Hounsfield
            Units for CT). Set to False to get raw stored values.

    Returns:
        Image: A standardized `Image` object.

    Raises:
        ValueError: If the file is not a valid DICOM file.
    """
    try:
        dcm = pydicom.dcmread(path)
        data = dcm.pixel_array
    except Exception as e:
        raise ValueError(f"Corrupt or invalid DICOM file '{path}': {e}") from e

    # Apply rescale while the native DICOM frame axis is still intact.  Enhanced
    # multiframe objects may store distinct transforms per frame.
    if apply_rescale:
        data = _apply_dicom_rescale(data, dcm)

    # Enhanced multiframe objects store geometry in functional groups rather
    # than top-level tags; resolve shared items once (top level takes precedence).
    measures = _shared_functional_group_item(dcm, "PixelMeasuresSequence")
    orientation_item = _shared_functional_group_item(dcm, "PlaneOrientationSequence")

    # Extract direction matrix from ImageOrientationPatient if available
    iop = getattr(dcm, "ImageOrientationPatient", None)
    if iop is None and orientation_item is not None:
        iop = getattr(orientation_item, "ImageOrientationPatient", None)
    slice_cosine = np.array([0.0, 0.0, 1.0])
    direction = np.eye(3)
    if iop is not None:
        try:
            orientation = np.asarray(iop, dtype=float)
            row_cosines = orientation[:3]
            col_cosines = orientation[3:]
            slice_cosine = np.cross(row_cosines, col_cosines)
            direction = np.stack([row_cosines, col_cosines, slice_cosine], axis=1)
        except (TypeError, ValueError):
            slice_cosine = np.array([0.0, 0.0, 1.0])
            direction = np.eye(3)

    # Sort multiframe data spatially when per-frame positions are available
    # (frame storage order is not guaranteed to match spatial order). Rescale
    # has already been applied, so reordering frames here is safe.
    frame_positions = None
    if data.ndim == 3:
        frame_positions = _per_frame_positions(dcm, data.shape[0])
        if frame_positions is not None:
            order = np.argsort([float(np.dot(p, slice_cosine)) for p in frame_positions])
            if not np.array_equal(order, np.arange(len(order))):
                data = data[order]
            frame_positions = [frame_positions[i] for i in order]

    # Handle dimensions
    # DICOM pixel_array format:
    #   - 2D: (Rows, Columns) = (Y, X)
    #   - 3D: (Frames, Rows, Columns) = (Z, Y, X) or (Rows, Columns, Frames) depending on source
    # We standardize to (X, Y, Z) to match _load_dicom_series behavior
    if data.ndim == 2:
        # (Y, X) -> (X, Y)
        data = np.swapaxes(data, 0, 1)
    elif data.ndim == 3:
        # Most DICOM 3D data (including SEG) is (Frames/Z, Rows/Y, Columns/X)
        # We need (X, Y, Z), so swap axes appropriately
        # From (Z, Y, X) to (X, Y, Z): swap 0<->2
        data = np.swapaxes(data, 0, 2)  # (Z, Y, X) -> (X, Y, Z)

    data = _ensure_3d(data)

    # Metadata extraction
    try:
        ps = getattr(dcm, "PixelSpacing", None)
        if ps is None and measures is not None:
            ps = getattr(measures, "PixelSpacing", None)
        if ps is None:
            raise AttributeError("PixelSpacing")

        # Prefer SpacingBetweenSlices over SliceThickness (consistent with _load_dicom_series)
        spacing_z = None
        for tag_source in (dcm, measures):
            if tag_source is None:
                continue
            if hasattr(tag_source, "SpacingBetweenSlices"):
                spacing_z = float(tag_source.SpacingBetweenSlices)
                break
            if hasattr(tag_source, "SliceThickness"):
                spacing_z = float(tag_source.SliceThickness)
                break
        if spacing_z is None:
            # Estimate from consecutive (sorted) frame positions if available
            if frame_positions is not None and len(frame_positions) > 1:
                spacing_z = float(np.linalg.norm(frame_positions[1] - frame_positions[0]))
            else:
                spacing_z = 1.0

        spacing = (
            float(ps[1]),  # Column spacing (X)
            float(ps[0]),  # Row spacing (Y)
            spacing_z,
        )
    except (AttributeError, IndexError):
        spacing = (1.0, 1.0, 1.0)

    # Origin: first (spatially sorted) frame position takes precedence for
    # multiframe objects; otherwise the top-level tag, then functional groups.
    try:
        if frame_positions is not None:
            ipp: Any = frame_positions[0]
        else:
            ipp = getattr(dcm, "ImagePositionPatient", None)
            if ipp is None:
                position_item = _shared_functional_group_item(dcm, "PlanePositionSequence")
                if position_item is not None:
                    ipp = getattr(position_item, "ImagePositionPatient", None)
            if ipp is None:
                raise AttributeError("ImagePositionPatient")
        origin = (float(ipp[0]), float(ipp[1]), float(ipp[2]))
    except (AttributeError, IndexError, TypeError):
        origin = (0.0, 0.0, 0.0)

    return Image(
        array=data,
        spacing=spacing,
        origin=origin,
        direction=direction,
        modality=getattr(dcm, "Modality", "DICOM"),
    )
