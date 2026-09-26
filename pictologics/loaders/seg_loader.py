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
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pydicom
from numpy import typing as npt

if TYPE_CHECKING:
    from pictologics.loader import Image


def load_seg(
    path: str | Path,
    segment_numbers: list[int] | None = None,
    combine_segments: bool = True,
    reference_image: "Image | None" = None,
    transpose_axes: tuple[int, int, int] | None = None,
    subvoxel_tolerance: float = 0.5,
    subvoxel_warning_threshold: float = 0.01,
    min_overlap_fraction: float = 0.5,
) -> "Image | dict[int, Image]":
    """Load a DICOM SEG file as a mask Image.

    This function loads a DICOM Segmentation object and converts it to
    the standard pictologics Image format. The resulting Image has the
    same structure as images returned by load_image():

    - array: npt.NDArray[np.floating[Any]] with shape (X, Y, Z)
    - spacing: tuple[float, float, float] in mm
    - origin: tuple[float, float, float] in mm
    - direction: Optional[npt.NDArray[np.floating[Any]]] - 3x3 direction cosines
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
        transpose_axes: Optional axis transposition to apply before reference
            alignment.
        subvoxel_tolerance: Maximum permitted fractional-voxel offset during
            reference alignment.
        subvoxel_warning_threshold: Fractional-voxel drift above which a warning
            is emitted during reference alignment.
        min_overlap_fraction: Minimum fraction of mask volume that must overlap
            the reference image during alignment.

    Returns:
        If combine_segments is True: A single Image with segment labels.
        If combine_segments is False: A dict of {segment_number: Image}.

    Raises:
        ValueError: If the file is not a valid DICOM SEG object.
        FileNotFoundError: If the file does not exist.

    Example:
        Load a SEG file with all segments combined (label map):

        ```python
        from pictologics.loaders import load_seg
        import numpy as np

        mask = load_seg("segmentation.dcm")
        print(mask.array.shape)  # (X, Y, Z)
        print(np.unique(mask.array))  # [0, 1, 2, ...]
        ```

        Load specific segments as separate binary masks:

        ```python
        masks = load_seg("segmentation.dcm", segment_numbers=[1, 2], combine_segments=False)
        for seg_num, mask in masks.items():
            print(f"Segment {seg_num}: {mask.array.sum()} voxels")
        ```

        Align mask to a reference CT image:

        ```python
        from pictologics import load_image

        ct = load_image("ct_scan/")
        mask = load_seg("segmentation.dcm", reference_image=ct)
        assert mask.array.shape == ct.array.shape
        ```
    """
    import highdicom as hd

    from pictologics.loader import Image, _row_order

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
                    f"Segment {seg_num} not found. Available segments: {available_segments}"
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
        combined_array = _extract_combined_segments(seg, pixel_array, target_segments, n_frames)

        # Reorder axes from (Z, Y, X) or (frames, rows, cols) to (X, Y, Z); a large mask
        # goes to row order, like the images.
        combined_array = _row_order(np.transpose(combined_array, (2, 1, 0)))

        result = Image(
            array=combined_array,
            spacing=spacing,
            origin=origin,
            direction=direction,
            modality="SEG",
        )

        # Align to reference if provided
        if reference_image is not None:
            result = _align_to_reference(
                result,
                reference_image,
                transpose_axes=transpose_axes,
                subvoxel_tolerance=subvoxel_tolerance,
                subvoxel_warning_threshold=subvoxel_warning_threshold,
                min_overlap_fraction=min_overlap_fraction,
            )

        return result
    else:
        # Return dict of individual segment masks
        result_dict: dict[int, Image] = {}

        for seg_num in target_segments:
            mask_array = _extract_single_segment(seg, pixel_array, seg_num, n_frames)

            # Reorder axes from (Z, Y, X) to (X, Y, Z)
            mask_array = np.transpose(mask_array, (2, 1, 0))

            mask_image = Image(
                array=_row_order(mask_array.astype(np.uint8)),
                spacing=spacing,
                origin=origin,
                direction=direction,
                modality="SEG",
            )

            # Align to reference if provided
            if reference_image is not None:
                mask_image = _align_to_reference(
                    mask_image,
                    reference_image,
                    transpose_axes=transpose_axes,
                    subvoxel_tolerance=subvoxel_tolerance,
                    subvoxel_warning_threshold=subvoxel_warning_threshold,
                    min_overlap_fraction=min_overlap_fraction,
                )

            result_dict[seg_num] = mask_image

        return result_dict


def _extract_seg_geometry(
    seg: pydicom.Dataset,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    npt.NDArray[np.floating[Any]] | None,
]:
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
    if hasattr(seg, "SharedFunctionalGroupsSequence") and seg.SharedFunctionalGroupsSequence:
        shared_fg = seg.SharedFunctionalGroupsSequence[0]

        # Get pixel spacing from PixelMeasuresSequence
        if hasattr(shared_fg, "PixelMeasuresSequence") and shared_fg.PixelMeasuresSequence:
            pm = shared_fg.PixelMeasuresSequence[0]
            if hasattr(pm, "PixelSpacing") and pm.PixelSpacing:
                row_spacing = float(pm.PixelSpacing[0])
                col_spacing = float(pm.PixelSpacing[1])
                slice_thickness = float(getattr(pm, "SliceThickness", 1.0) or 1.0)
                # Spacing in (X, Y, Z) = (col, row, slice)
                spacing = (col_spacing, row_spacing, slice_thickness)

        # Get orientation from PlaneOrientationSequence
        if hasattr(shared_fg, "PlaneOrientationSequence") and shared_fg.PlaneOrientationSequence:
            po = shared_fg.PlaneOrientationSequence[0]
            if hasattr(po, "ImageOrientationPatient") and po.ImageOrientationPatient:
                iop = [float(x) for x in po.ImageOrientationPatient]
                row_cosines = np.array(iop[:3])
                col_cosines = np.array(iop[3:6])
                slice_cosines = np.cross(row_cosines, col_cosines)
                direction = np.column_stack([row_cosines, col_cosines, slice_cosines])

    # Origin and slice step from the frame positions (see _frame_layout)
    layout = _frame_layout(seg, len(getattr(seg, "PerFrameFunctionalGroupsSequence", None) or []))
    if layout.first_position is not None:
        origin = layout.first_position
    if layout.step is not None:
        spacing = (spacing[0], spacing[1], layout.step)

    return spacing, origin, direction


class _FrameLayout(NamedTuple):
    """Where each frame of a SEG goes."""

    segments: list[int]  # segment number of each frame
    slices: list[int]  # slice index of each frame
    n_slices: int
    step: float | None  # slice step in mm, when the frames have positions
    first_position: tuple[float, float, float] | None  # position of slice 0


def _numbers(value: Any, count: int) -> list[float] | None:
    """`value` as `count` floats, or None if it is not a sequence of `count` numbers."""
    try:
        numbers = [float(x) for x in value]
    except TypeError:
        return None
    return numbers if len(numbers) == count else None


def _frame_layout(seg: pydicom.Dataset, n_frames: int) -> _FrameLayout:
    """Segment number and slice index of each frame.

    The slice index comes from each frame's position along the slice normal: writers
    order the dimension index values differently (highdicom puts the segment number
    first), and they may leave out empty frames. The step is the smallest distance
    between two slice positions, or the SEG's SpacingBetweenSlices when that distance
    is a whole multiple of it (slices without any segment). Frames without a position
    fall back to their first dimension index value, and then to their own order.
    """
    frames = list(getattr(seg, "PerFrameFunctionalGroupsSequence", None) or [])
    segments = [1] * n_frames
    dim_slices: dict[int, int] = {}
    positions: list[list[float] | None] = [None] * n_frames
    for i, fg in enumerate(frames[:n_frames]):
        sid = getattr(fg, "SegmentIdentificationSequence", None)
        if sid:
            segments[i] = int(sid[0].ReferencedSegmentNumber)
        fc = getattr(fg, "FrameContentSequence", None)
        values = list(getattr(fc[0], "DimensionIndexValues", None) or []) if fc else []
        if values:
            dim_slices[i] = int(values[0]) - 1
        pps = getattr(fg, "PlanePositionSequence", None)
        if pps:
            positions[i] = _numbers(getattr(pps[0], "ImagePositionPatient", None), 3)

    if n_frames == 0 or any(pos is None for pos in positions):
        if dim_slices:
            n_slices = max(dim_slices.values()) + 1
        else:
            n_segments = len(seg.SegmentSequence)
            n_slices = n_frames // n_segments if n_segments > 0 else n_frames
        slices = [dim_slices.get(i, i) for i in range(n_frames)]
        return _FrameLayout(segments, slices, n_slices, None, None)

    normal = np.array([0.0, 0.0, 1.0])
    shared = getattr(seg, "SharedFunctionalGroupsSequence", None)
    po = getattr(shared[0], "PlaneOrientationSequence", None) if shared else None
    iop = _numbers(getattr(po[0], "ImageOrientationPatient", None), 6) if po else None
    if iop is not None:
        normal = np.cross(iop[:3], iop[3:])
    pos = np.array(positions)
    proj = pos @ normal
    levels = np.unique(np.round(proj, 3))  # one level per slice, to the micrometre
    step: float | None = None
    slices_arr = np.zeros(n_frames, dtype=np.int64)
    if len(levels) > 1:
        step = float(np.min(np.diff(levels)))
        pm = getattr(shared[0], "PixelMeasuresSequence", None) if shared else None
        declared = getattr(pm[0], "SpacingBetweenSlices", None) if pm else None
        if isinstance(declared, (int, float)) and declared > 0:
            ratio = step / float(declared)
            if abs(ratio - round(ratio)) < 1e-3:
                step = float(declared)
        slices_arr = np.rint((proj - proj.min()) / step).astype(np.int64)
    first = pos[int(np.argmin(proj))]
    return _FrameLayout(
        segments,
        [int(k) for k in slices_arr],
        int(slices_arr.max()) + 1,
        step,
        (float(first[0]), float(first[1]), float(first[2])),
    )


def _extract_combined_segments(
    seg: pydicom.Dataset,
    pixel_array: npt.NDArray[np.floating[Any]],
    target_segments: list[int],
    n_frames: int,
) -> npt.NDArray[np.floating[Any]]:
    """Extract and combine multiple segments into a single label array.

    Args:
        seg: The DICOM SEG dataset.
        pixel_array: The raw pixel array from the SEG.
        target_segments: List of segment numbers to include.
        n_frames: Number of frames in the SEG.

    Returns:
        3D numpy array with segment numbers as voxel values.
    """
    layout = _frame_layout(seg, n_frames)
    # Create output array: (Z, Y, X) = (slices, rows, cols)
    combined = np.zeros((layout.n_slices, seg.Rows, seg.Columns), dtype=np.uint8)
    for frame_idx in range(n_frames):
        seg_num = layout.segments[frame_idx]
        if seg_num not in target_segments or layout.slices[frame_idx] >= layout.n_slices:
            continue
        frame_data = pixel_array[frame_idx] if pixel_array.ndim == 3 else pixel_array
        # Add to combined array (higher segment numbers overwrite lower)
        combined[layout.slices[frame_idx]][frame_data > 0] = seg_num
    return combined


def _extract_single_segment(
    seg: pydicom.Dataset,
    pixel_array: npt.NDArray[np.floating[Any]],
    segment_number: int,
    n_frames: int,
) -> npt.NDArray[np.floating[Any]]:
    """Extract a single segment as a binary mask.

    Args:
        seg: The DICOM SEG dataset.
        pixel_array: The raw pixel array from the SEG.
        segment_number: The segment number to extract.
        n_frames: Number of frames in the SEG.

    Returns:
        3D binary numpy array for the specified segment.
    """
    layout = _frame_layout(seg, n_frames)
    result = np.zeros((layout.n_slices, seg.Rows, seg.Columns), dtype=np.uint8)
    for frame_idx in range(n_frames):
        if (
            layout.segments[frame_idx] != segment_number
            or layout.slices[frame_idx] >= layout.n_slices
        ):
            continue
        frame_data = pixel_array[frame_idx] if pixel_array.ndim == 3 else pixel_array
        result[layout.slices[frame_idx]] = (frame_data > 0).astype(np.uint8)
    return result


def _align_to_reference(
    mask: "Image",
    reference: "Image",
    *,
    transpose_axes: tuple[int, int, int] | None = None,
    subvoxel_tolerance: float = 0.5,
    subvoxel_warning_threshold: float = 0.01,
    min_overlap_fraction: float = 0.5,
) -> "Image":
    """Align a mask Image to a reference Image geometry.

    Uses the same repositioning logic as pictologics.loader._position_in_reference.

    Args:
        mask: The mask Image to align.
        reference: The reference Image with target geometry.
        transpose_axes: Optional axis transposition before positioning.
        subvoxel_tolerance: Maximum permitted fractional-voxel offset.
        subvoxel_warning_threshold: Fractional-voxel drift warning threshold.
        min_overlap_fraction: Minimum required overlap with reference geometry.

    Returns:
        A new Image aligned to the reference geometry.
    """
    from pictologics.loader import _position_in_reference

    # Use the existing repositioning function
    aligned = _position_in_reference(
        image=mask,
        reference=reference,
        fill_value=0,
        transpose_axes=transpose_axes,
        subvoxel_tolerance=subvoxel_tolerance,
        subvoxel_warning_threshold=subvoxel_warning_threshold,
        min_overlap_fraction=min_overlap_fraction,
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

    Example:
        ```python
        from pictologics.loaders import get_segment_info

        segments = get_segment_info("segmentation.dcm")
        for seg in segments:
            print(f"{seg['segment_number']}: {seg['segment_label']}")
        ```
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
