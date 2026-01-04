"""
DICOM Segmentation (SEG) Loader
===============================

This module provides functionality for loading DICOM Segmentation objects
as pictologics Image instances. SEG files are specialized DICOM objects
that store segmentation masks with multi-segment support.

Uses highdicom for robust SEG parsing and extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pydicom

if TYPE_CHECKING:
    from pictologics.loader import Image


def load_seg(
    path: str | Path,
    segment_numbers: list[int] | None = None,
    combine_segments: bool = True,
    reference_image: "Image | None" = None,
) -> "Image | dict[int, Image]":
    """Load a DICOM SEG file as a mask Image.

    This function loads a DICOM Segmentation object and converts it to
    the standard pictologics Image format. The resulting Image has the
    same structure as images returned by load_image():

    - array: np.ndarray with shape (X, Y, Z)
    - spacing: tuple[float, float, float] in mm
    - origin: tuple[float, float, float] in mm
    - direction: Optional[np.ndarray] - 3x3 direction cosines
    - modality: str - set to "SEG"

    Args:
        path: Path to the DICOM SEG file.
        segment_numbers: Specific segment numbers to extract. If None, all
            segments are extracted. Segment numbers are 1-indexed as per
            DICOM convention.
        combine_segments: Controls how segments are returned:

            - **True (default)**: Returns a single Image where each segment
              is encoded as its segment number (1, 2, 3...) in the voxel values.
              Background voxels are 0. This is useful when you want a single
              label map for visualization or when segments are mutually exclusive
              (e.g., organ segmentation where each voxel belongs to one structure).

            - **False**: Returns a dict mapping segment numbers to individual
              binary Image masks. Each mask contains only 0s and 1s. This is
              useful when:

              - Segments may overlap (e.g., nested structures like tumor
                within organ)
              - You need to process each segment independently (e.g., extract
                radiomics from each segment separately)
              - You want to select specific segments for different analyses

        reference_image: Optional reference Image for geometry alignment.
            When provided, the output mask will be resampled/repositioned
            to match the reference geometry.

    Returns:
        If combine_segments is True: A single Image with segment labels.
        If combine_segments is False: A dict of {segment_number: Image}.

    Raises:
        ValueError: If the file is not a valid DICOM SEG object.
        FileNotFoundError: If the file does not exist.

    Examples:
        Load a SEG file with all segments combined (label map):

        >>> mask = load_seg("segmentation.dcm")
        >>> print(mask.array.shape)  # (X, Y, Z)
        >>> print(np.unique(mask.array))  # [0, 1, 2, ...]

        Load specific segments as separate binary masks:

        >>> masks = load_seg("segmentation.dcm", segment_numbers=[1, 2],
        ...                  combine_segments=False)
        >>> for seg_num, mask in masks.items():
        ...     print(f"Segment {seg_num}: {mask.array.sum()} voxels")

        Extract radiomics from each coronary territory separately:

        >>> masks = load_seg("cardiac_seg.dcm", combine_segments=False)
        >>> for seg_num, mask in masks.items():
        ...     features = pipeline.run(image=ct, mask=mask)

        Align mask to a reference CT image:

        >>> ct = load_image("ct_scan/")
        >>> mask = load_seg("segmentation.dcm", reference_image=ct)
        >>> assert mask.array.shape == ct.array.shape
    """
    import highdicom as hd

    from pictologics.loader import Image

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"SEG file not found: {path}")

    # Load the DICOM SEG using highdicom
    try:
        seg = hd.seg.segread(str(path_obj))
    except Exception as e:
        raise ValueError(f"Failed to load DICOM SEG file: {e}") from e

    # Verify it's a SEG object
    if not hasattr(seg, "SegmentSequence"):
        raise ValueError(f"File is not a valid DICOM SEG object: {path}")

    # Get available segment numbers
    available_segments = [s.SegmentNumber for s in seg.SegmentSequence]

    # Determine which segments to extract
    if segment_numbers is None:
        target_segments = available_segments
    else:
        # Validate requested segments exist
        for seg_num in segment_numbers:
            if seg_num not in available_segments:
                raise ValueError(
                    f"Segment {seg_num} not found. "
                    f"Available segments: {available_segments}"
                )
        target_segments = segment_numbers

    # Extract geometry information from the SEG
    spacing, origin, direction = _extract_seg_geometry(seg)

    # Extract pixel array - shape is typically (frames, rows, cols)
    pixel_array = seg.pixel_array

    # Get the number of segments and frames
    n_frames = pixel_array.shape[0] if pixel_array.ndim == 3 else 1

    if combine_segments:
        # Create combined label image
        combined_array = _extract_combined_segments(
            seg, pixel_array, target_segments, n_frames
        )

        # Reorder axes from (Z, Y, X) or (frames, rows, cols) to (X, Y, Z)
        combined_array = np.transpose(combined_array, (2, 1, 0))

        result = Image(
            array=combined_array,
            spacing=spacing,
            origin=origin,
            direction=direction,
            modality="SEG",
        )

        # Align to reference if provided
        if reference_image is not None:
            result = _align_to_reference(result, reference_image)

        return result
    else:
        # Return dict of individual segment masks
        result_dict: dict[int, Image] = {}

        for seg_num in target_segments:
            mask_array = _extract_single_segment(seg, pixel_array, seg_num, n_frames)

            # Reorder axes from (Z, Y, X) to (X, Y, Z)
            mask_array = np.transpose(mask_array, (2, 1, 0))

            mask_image = Image(
                array=mask_array.astype(np.uint8),
                spacing=spacing,
                origin=origin,
                direction=direction,
                modality="SEG",
            )

            # Align to reference if provided
            if reference_image is not None:
                mask_image = _align_to_reference(mask_image, reference_image)

            result_dict[seg_num] = mask_image

        return result_dict


def _extract_seg_geometry(
    seg: pydicom.Dataset,
) -> tuple[tuple[float, float, float], tuple[float, float, float], np.ndarray | None]:
    """Extract spatial geometry from a DICOM SEG object.

    Attempts to extract spacing, origin, and direction from the SEG's
    SharedFunctionalGroupsSequence or PerFrameFunctionalGroupsSequence.

    Args:
        seg: The loaded DICOM SEG dataset.

    Returns:
        Tuple of (spacing, origin, direction) where:
        - spacing: (x, y, z) voxel spacing in mm
        - origin: (x, y, z) position of first voxel in mm
        - direction: 3x3 direction cosine matrix or None
    """
    # Default values
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = None

    # Try to get from SharedFunctionalGroupsSequence
    if (
        hasattr(seg, "SharedFunctionalGroupsSequence")
        and seg.SharedFunctionalGroupsSequence
    ):
        shared_fg = seg.SharedFunctionalGroupsSequence[0]

        # Get pixel spacing from PixelMeasuresSequence
        if (
            hasattr(shared_fg, "PixelMeasuresSequence")
            and shared_fg.PixelMeasuresSequence
        ):
            pm = shared_fg.PixelMeasuresSequence[0]
            if hasattr(pm, "PixelSpacing") and pm.PixelSpacing:
                row_spacing = float(pm.PixelSpacing[0])
                col_spacing = float(pm.PixelSpacing[1])
                slice_thickness = float(getattr(pm, "SliceThickness", 1.0) or 1.0)
                # Spacing in (X, Y, Z) = (col, row, slice)
                spacing = (col_spacing, row_spacing, slice_thickness)

        # Get orientation from PlaneOrientationSequence
        if (
            hasattr(shared_fg, "PlaneOrientationSequence")
            and shared_fg.PlaneOrientationSequence
        ):
            po = shared_fg.PlaneOrientationSequence[0]
            if hasattr(po, "ImageOrientationPatient") and po.ImageOrientationPatient:
                iop = [float(x) for x in po.ImageOrientationPatient]
                row_cosines = np.array(iop[:3])
                col_cosines = np.array(iop[3:6])
                slice_cosines = np.cross(row_cosines, col_cosines)
                direction = np.column_stack([row_cosines, col_cosines, slice_cosines])

    # Try to get origin from first frame's PlanePositionSequence
    if (
        hasattr(seg, "PerFrameFunctionalGroupsSequence")
        and seg.PerFrameFunctionalGroupsSequence
    ):
        first_frame = seg.PerFrameFunctionalGroupsSequence[0]
        if (
            hasattr(first_frame, "PlanePositionSequence")
            and first_frame.PlanePositionSequence
        ):
            pp = first_frame.PlanePositionSequence[0]
            if hasattr(pp, "ImagePositionPatient") and pp.ImagePositionPatient:
                ipp = [float(x) for x in pp.ImagePositionPatient]
                origin = (ipp[0], ipp[1], ipp[2])

    # Calculate slice spacing from frame positions if available
    if (
        hasattr(seg, "PerFrameFunctionalGroupsSequence")
        and seg.PerFrameFunctionalGroupsSequence
    ):
        positions = []
        for frame_fg in seg.PerFrameFunctionalGroupsSequence:
            if (
                hasattr(frame_fg, "PlanePositionSequence")
                and frame_fg.PlanePositionSequence
            ):
                pp = frame_fg.PlanePositionSequence[0]
                if hasattr(pp, "ImagePositionPatient") and pp.ImagePositionPatient:
                    positions.append([float(x) for x in pp.ImagePositionPatient])

        if len(positions) >= 2:
            # Calculate slice spacing from consecutive frame positions
            pos_array = np.array(positions)
            if len(pos_array) > 1:
                diffs = np.diff(pos_array, axis=0)
                slice_distances = np.linalg.norm(diffs, axis=1)
                if len(slice_distances) > 0:
                    median_spacing = float(np.median(slice_distances))
                    if median_spacing > 0:
                        spacing = (spacing[0], spacing[1], median_spacing)

    return spacing, origin, direction


def _extract_combined_segments(
    seg: pydicom.Dataset,
    pixel_array: np.ndarray,
    target_segments: list[int],
    n_frames: int,
) -> np.ndarray:
    """Extract and combine multiple segments into a single label array.

    Args:
        seg: The DICOM SEG dataset.
        pixel_array: The raw pixel array from the SEG.
        target_segments: List of segment numbers to include.
        n_frames: Number of frames in the SEG.

    Returns:
        3D numpy array with segment numbers as voxel values.
    """
    # Determine array dimensions
    rows = seg.Rows
    cols = seg.Columns

    # For multi-segment SEGs, we need to figure out the frame organization
    # Each frame belongs to a specific segment and slice position

    # Get segment info for each frame from PerFrameFunctionalGroupsSequence
    frame_to_segment: dict[int, int] = {}
    frame_to_slice: dict[int, int] = {}

    if hasattr(seg, "PerFrameFunctionalGroupsSequence"):
        for frame_idx, frame_fg in enumerate(seg.PerFrameFunctionalGroupsSequence):
            # Get segment number for this frame
            if (
                hasattr(frame_fg, "SegmentIdentificationSequence")
                and frame_fg.SegmentIdentificationSequence
            ):
                seg_id = frame_fg.SegmentIdentificationSequence[0]
                frame_to_segment[frame_idx] = seg_id.ReferencedSegmentNumber

            # Get dimension index for slice position
            if (
                hasattr(frame_fg, "FrameContentSequence")
                and frame_fg.FrameContentSequence
            ):
                fc = frame_fg.FrameContentSequence[0]
                if hasattr(fc, "DimensionIndexValues") and fc.DimensionIndexValues:
                    # Typically [slice_index, segment_number] or similar
                    dim_values = list(fc.DimensionIndexValues)
                    # Use first dimension as slice index (0-indexed)
                    frame_to_slice[frame_idx] = (
                        dim_values[0] - 1 if dim_values else frame_idx
                    )

    # Determine number of slices
    if frame_to_slice:
        n_slices = max(frame_to_slice.values()) + 1
    else:
        # Estimate from number of frames and segments
        n_segments = len(seg.SegmentSequence)
        n_slices = n_frames // n_segments if n_segments > 0 else n_frames

    # Create output array: (Z, Y, X) = (slices, rows, cols)
    combined = np.zeros((n_slices, rows, cols), dtype=np.uint8)

    # Fill in segments
    for frame_idx in range(n_frames):
        seg_num = frame_to_segment.get(frame_idx, 1)
        slice_idx = frame_to_slice.get(frame_idx, frame_idx)

        if seg_num not in target_segments:
            continue

        if slice_idx >= n_slices:
            continue

        # Get frame data
        if pixel_array.ndim == 3:
            frame_data = pixel_array[frame_idx]
        else:
            frame_data = pixel_array

        # Add to combined array (higher segment numbers overwrite lower)
        mask = frame_data > 0
        combined[slice_idx][mask] = seg_num

    return combined


def _extract_single_segment(
    seg: pydicom.Dataset,
    pixel_array: np.ndarray,
    segment_number: int,
    n_frames: int,
) -> np.ndarray:
    """Extract a single segment as a binary mask.

    Args:
        seg: The DICOM SEG dataset.
        pixel_array: The raw pixel array from the SEG.
        segment_number: The segment number to extract.
        n_frames: Number of frames in the SEG.

    Returns:
        3D binary numpy array for the specified segment.
    """
    rows = seg.Rows
    cols = seg.Columns

    # Get frame organization
    frame_to_segment: dict[int, int] = {}
    frame_to_slice: dict[int, int] = {}

    if hasattr(seg, "PerFrameFunctionalGroupsSequence"):
        for frame_idx, frame_fg in enumerate(seg.PerFrameFunctionalGroupsSequence):
            if (
                hasattr(frame_fg, "SegmentIdentificationSequence")
                and frame_fg.SegmentIdentificationSequence
            ):
                seg_id = frame_fg.SegmentIdentificationSequence[0]
                frame_to_segment[frame_idx] = seg_id.ReferencedSegmentNumber

            if (
                hasattr(frame_fg, "FrameContentSequence")
                and frame_fg.FrameContentSequence
            ):
                fc = frame_fg.FrameContentSequence[0]
                if hasattr(fc, "DimensionIndexValues") and fc.DimensionIndexValues:
                    dim_values = list(fc.DimensionIndexValues)
                    frame_to_slice[frame_idx] = (
                        dim_values[0] - 1 if dim_values else frame_idx
                    )

    # Determine number of slices
    if frame_to_slice:
        n_slices = max(frame_to_slice.values()) + 1
    else:
        n_segments = len(seg.SegmentSequence)
        n_slices = n_frames // n_segments if n_segments > 0 else n_frames

    # Create output array
    result = np.zeros((n_slices, rows, cols), dtype=np.uint8)

    # Extract frames for this segment
    for frame_idx in range(n_frames):
        seg_num = frame_to_segment.get(frame_idx, 1)
        slice_idx = frame_to_slice.get(frame_idx, frame_idx)

        if seg_num != segment_number:
            continue

        if slice_idx >= n_slices:
            continue

        if pixel_array.ndim == 3:
            frame_data = pixel_array[frame_idx]
        else:
            frame_data = pixel_array

        result[slice_idx] = (frame_data > 0).astype(np.uint8)

    return result


def _align_to_reference(mask: "Image", reference: "Image") -> "Image":
    """Align a mask Image to a reference Image geometry.

    Uses the same repositioning logic as pictologics.loader._position_in_reference.

    Args:
        mask: The mask Image to align.
        reference: The reference Image with target geometry.

    Returns:
        A new Image aligned to the reference geometry.
    """
    from pictologics.loader import _position_in_reference

    # Use the existing repositioning function
    aligned = _position_in_reference(
        image=mask,
        reference=reference,
        fill_value=0,
        transpose_axes=None,
    )

    return aligned


def get_segment_info(path: str | Path) -> list[dict[str, str | int]]:
    """Get information about segments in a DICOM SEG file.

    Args:
        path: Path to the DICOM SEG file.

    Returns:
        List of dicts with segment information:
        - segment_number: int
        - segment_label: str
        - segment_description: str (if available)
        - algorithm_type: str (if available)

    Raises:
        ValueError: If the file is not a valid DICOM SEG object.
    """
    import highdicom as hd

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"SEG file not found: {path}")

    try:
        seg = hd.seg.segread(str(path_obj))
    except Exception as e:
        raise ValueError(f"Failed to load DICOM SEG file: {e}") from e

    if not hasattr(seg, "SegmentSequence"):
        raise ValueError(f"File is not a valid DICOM SEG object: {path}")

    segments = []
    for segment in seg.SegmentSequence:
        info: dict[str, str | int] = {
            "segment_number": segment.SegmentNumber,
            "segment_label": getattr(segment, "SegmentLabel", ""),
        }

        if hasattr(segment, "SegmentDescription"):
            info["segment_description"] = segment.SegmentDescription

        if hasattr(segment, "SegmentAlgorithmType"):
            info["algorithm_type"] = segment.SegmentAlgorithmType

        segments.append(info)

    return segments
