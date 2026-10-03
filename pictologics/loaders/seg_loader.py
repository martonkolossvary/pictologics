"""
DICOM Segmentation (SEG) Loader
===============================

This module provides functionality for loading DICOM Segmentation objects
as pictologics Image instances. SEG files are specialized DICOM objects
that store segmentation masks with multi-segment support.

The files are read with pydicom, and the frames are decoded one at a time, only
for the requested segments.
"""

from __future__ import annotations

from collections.abc import Iterable
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
    fractional_threshold: float = 0.5,
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
        fractional_threshold: For a FRACTIONAL SEG (probability or occupancy), a
            voxel is in a segment when its value is at least this fraction of the
            MaximumFractionalValue (and above 0). Default 0.5. BINARY and LABELMAP
            SEGs do not use it.

    Returns:
        If combine_segments is True: A single Image with segment labels; the labels
        are uint8, or uint16 for segment numbers above 255.
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
    from pydicom.pixels.utils import iter_pixels

    from pictologics.loader import Image, _frame_uid, _row_order

    if not 0.0 <= fractional_threshold <= 1.0:
        raise ValueError(f"fractional_threshold must be in [0, 1], not {fractional_threshold}.")
    seg = _read_seg(path)

    # Get available segment numbers
    available_segments = [s.SegmentNumber for s in seg.SegmentSequence]

    # Determine which segments to extract. A label map may describe its background as
    # segment 0, which is not a segment to extract by default.
    if segment_numbers is None:
        target_segments = [n for n in available_segments if n != 0 or not _is_labelmap(seg)]
    else:
        # Validate requested segments exist
        for seg_num in segment_numbers:
            if seg_num not in available_segments:
                raise ValueError(
                    f"Segment {seg_num} not found. Available segments: {available_segments}"
                )
        target_segments = segment_numbers

    # Where each frame goes, and the geometry
    n_frames = int(getattr(seg, "NumberOfFrames", 1) or 1)
    layout = _frame_layout(seg, n_frames)
    spacing, origin, direction = _extract_seg_geometry(seg, layout)

    # Decode only the frames of the requested segments (a label map: all frames)
    wanted = set(target_segments)
    indices = [
        i
        for i in range(n_frames)
        if layout.slices[i] < layout.n_slices
        and (_is_labelmap(seg) or layout.segments[i] in wanted)
    ]
    frames = zip(indices, iter_pixels(seg, indices=indices), strict=True)
    level = _inside_level(seg, fractional_threshold)

    if combine_segments:
        # Create combined label image
        combined_array = _extract_combined_segments(seg, frames, target_segments, layout, level)

        # Reorder axes from (Z, Y, X) or (frames, rows, cols) to (X, Y, Z); a large mask
        # goes to row order, like the images.
        combined_array = _row_order(np.transpose(combined_array, (2, 1, 0)))

        result = Image(
            array=combined_array,
            spacing=spacing,
            origin=origin,
            direction=direction,
            modality="SEG",
            frame_of_reference_uid=_frame_uid(seg),
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

        masks = _extract_segment_masks(seg, frames, target_segments, layout, level)
        for seg_num in target_segments:
            # Reorder axes from (Z, Y, X) to (X, Y, Z)
            mask_image = Image(
                array=_row_order(np.transpose(masks.pop(seg_num), (2, 1, 0))),
                spacing=spacing,
                origin=origin,
                direction=direction,
                modality="SEG",
                frame_of_reference_uid=_frame_uid(seg),
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


def _read_seg(path: str | Path, stop_before_pixels: bool = False) -> pydicom.Dataset:
    """Read a DICOM SEG file (Segmentation or Label Map Segmentation Storage)."""
    from pictologics.loader import _SEG_SOP_CLASSES

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"SEG file not found: {path}")
    try:
        seg = pydicom.dcmread(str(path_obj), stop_before_pixels=stop_before_pixels)
    except Exception as e:
        raise ValueError(f"Failed to load DICOM SEG file: {e}") from e
    if str(getattr(seg, "SOPClassUID", "")) not in _SEG_SOP_CLASSES or not hasattr(
        seg, "SegmentSequence"
    ):
        raise ValueError(f"File is not a valid DICOM SEG object: {path}")
    return seg


def _inside_level(seg: pydicom.Dataset, fractional_threshold: float) -> float:
    """The smallest stored value inside a segment: 1 (values above 0), or for a
    FRACTIONAL SEG the threshold fraction of its MaximumFractionalValue."""
    if str(getattr(seg, "SegmentationType", "")) != "FRACTIONAL":
        return 1.0
    maximum = float(getattr(seg, "MaximumFractionalValue", 255) or 255)
    return max(fractional_threshold * maximum, 1.0)


def _extract_seg_geometry(
    seg: pydicom.Dataset,
    layout: "_FrameLayout | None" = None,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    npt.NDArray[np.floating[Any]] | None,
]:
    """Extract spatial geometry from a DICOM SEG object.

    The pixel measures and the orientation come from the SharedFunctionalGroupsSequence,
    else from the first PerFrameFunctionalGroupsSequence item (the standard allows
    either place).

    Args:
        seg: The loaded DICOM SEG dataset.
        layout: The frame layout of the SEG (from `_frame_layout`); found when None.

    Returns:
        Tuple of (spacing, origin, direction) where:
        - spacing: (x, y, z) voxel spacing in mm
        - origin: (x, y, z) position of first voxel in mm
        - direction: 3x3 direction cosine matrix or None
    """
    from pictologics.loader import _shared_functional_group_item

    # Default values
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = None

    pm = _shared_functional_group_item(seg, "PixelMeasuresSequence")
    if pm is not None and getattr(pm, "PixelSpacing", None):
        row_spacing = float(pm.PixelSpacing[0])
        col_spacing = float(pm.PixelSpacing[1])
        slice_thickness = float(getattr(pm, "SliceThickness", 1.0) or 1.0)
        # Spacing in (X, Y, Z) = (col, row, slice)
        spacing = (col_spacing, row_spacing, slice_thickness)

    po = _shared_functional_group_item(seg, "PlaneOrientationSequence")
    if po is not None and getattr(po, "ImageOrientationPatient", None):
        iop = [float(x) for x in po.ImageOrientationPatient]
        row_cosines = np.array(iop[:3])
        col_cosines = np.array(iop[3:6])
        slice_cosines = np.cross(row_cosines, col_cosines)
        direction = np.column_stack([row_cosines, col_cosines, slice_cosines])

    # Origin and slice step from the frame positions (see _frame_layout)
    if layout is None:
        layout = _frame_layout(
            seg, len(getattr(seg, "PerFrameFunctionalGroupsSequence", None) or [])
        )
    if layout.first_position is not None:
        origin = layout.first_position
    if layout.step is not None:
        spacing = (spacing[0], spacing[1], layout.step)

    return spacing, origin, direction


def _is_labelmap(seg: pydicom.Dataset) -> bool:
    """A label-map SEG stores segment numbers as pixel values, one frame per position."""
    return str(getattr(seg, "SegmentationType", "")) == "LABELMAP"


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
        values = getattr(fc[0], "DimensionIndexValues", None) if fc else None
        if isinstance(values, int):  # one value, as in a label map (the position)
            values = [values]
        if values:
            dim_slices[i] = int(values[0]) - 1
        pps = getattr(fg, "PlanePositionSequence", None)
        if pps:
            positions[i] = _numbers(getattr(pps[0], "ImagePositionPatient", None), 3)

    if n_frames == 0 or any(pos is None for pos in positions):
        if dim_slices:
            n_slices = max(dim_slices.values()) + 1
        else:
            n_segments = 1 if _is_labelmap(seg) else len(seg.SegmentSequence)
            n_slices = n_frames // n_segments if n_segments > 0 else n_frames
        slices = [dim_slices.get(i, i) for i in range(n_frames)]
        return _FrameLayout(segments, slices, n_slices, None, None)

    from pictologics.loader import _shared_functional_group_item

    normal = np.array([0.0, 0.0, 1.0])
    po = _shared_functional_group_item(seg, "PlaneOrientationSequence")
    iop = _numbers(getattr(po, "ImageOrientationPatient", None), 6)
    if iop is not None:
        normal = np.cross(iop[:3], iop[3:])
    pos = np.array(positions)
    proj = pos @ normal
    levels = np.unique(np.round(proj, 3))  # one level per slice, to the micrometre
    step: float | None = None
    slices_arr = np.zeros(n_frames, dtype=np.int64)
    if len(levels) > 1:
        step = float(np.min(np.diff(levels)))
        pm = _shared_functional_group_item(seg, "PixelMeasuresSequence")
        declared = getattr(pm, "SpacingBetweenSlices", None)
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
    frames: Iterable[tuple[int, npt.NDArray[Any]]],
    target_segments: list[int],
    layout: _FrameLayout,
    level: float = 1.0,
) -> npt.NDArray[Any]:
    """Extract and combine multiple segments into a single label array.

    Args:
        seg: The DICOM SEG dataset.
        frames: (frame index, decoded frame) pairs; frames of other segments may be left out.
        target_segments: List of segment numbers to include.
        layout: The frame layout of the SEG (from `_frame_layout`).
        level: The smallest stored value inside a segment (see `_inside_level`).

    Returns:
        3D (Z, Y, X) array with segment numbers as voxel values: uint8, or uint16 for
        segment numbers above 255.
    """
    labelmap = _is_labelmap(seg)
    wanted = set(target_segments)
    dtype = np.uint8 if max(target_segments, default=0) <= 255 else np.uint16
    # Create output array: (Z, Y, X) = (slices, rows, cols)
    combined = np.zeros((layout.n_slices, seg.Rows, seg.Columns), dtype=dtype)
    for frame_idx, frame_data in frames:
        seg_num = layout.segments[frame_idx]
        if layout.slices[frame_idx] >= layout.n_slices:
            continue
        if labelmap:
            # The pixel values are the segment numbers
            keep = np.isin(frame_data, target_segments)
            combined[layout.slices[frame_idx]][keep] = frame_data[keep]
        elif seg_num in wanted:
            # Add to combined array (higher segment numbers overwrite lower)
            combined[layout.slices[frame_idx]][frame_data >= level] = seg_num
    return combined


def _extract_segment_masks(
    seg: pydicom.Dataset,
    frames: Iterable[tuple[int, npt.NDArray[Any]]],
    target_segments: list[int],
    layout: _FrameLayout,
    level: float = 1.0,
) -> dict[int, npt.NDArray[np.uint8]]:
    """Extract each segment as a binary mask, in one pass over the frames.

    Args:
        seg: The DICOM SEG dataset.
        frames: (frame index, decoded frame) pairs; frames of other segments may be left out.
        target_segments: The segment numbers to extract.
        layout: The frame layout of the SEG (from `_frame_layout`).
        level: The smallest stored value inside a segment (see `_inside_level`).

    Returns:
        {segment number: 3D (Z, Y, X) uint8 mask}.
    """
    labelmap = _is_labelmap(seg)
    masks = {
        n: np.zeros((layout.n_slices, seg.Rows, seg.Columns), dtype=np.uint8)
        for n in target_segments
    }
    for frame_idx, frame_data in frames:
        k = layout.slices[frame_idx]
        if k >= layout.n_slices:
            continue
        if labelmap:
            # A label map holds the segment numbers themselves
            for number, mask in masks.items():
                mask[k] = frame_data == number
        elif layout.segments[frame_idx] in masks:
            masks[layout.segments[frame_idx]][k] = frame_data >= level
    return masks


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
    """Get information about the segments of a DICOM SEG file, the ROIs of a DICOM
    RTSTRUCT file, or the segments of a 3D Slicer .seg.nrrd file.

    Args:
        path: Path to the DICOM SEG file, the RTSTRUCT file, or the .seg.nrrd file.

    Returns:
        List of dicts with segment information. For DICOM SEG:
        - segment_number: int
        - segment_label: str
        - segment_description: str (if available)
        - algorithm_type: str (if available)

        For RTSTRUCT:
        - segment_number: int (the ROI Number, the label of ``load_rtstruct``)
        - segment_label: str (the ROI Name)
        - contour_count: int (closed planar contours)
        - interpreted_type: str (RTROIInterpretedType, if available)

        For .seg.nrrd (the fields ``Segment<N>_...`` of the header):
        - segment_index: int (N)
        - segment_label: str (the segment name)
        - segment_id: str
        - label_value: int (the value of the segment in its layer)
        - layer: int (the volume that ``load_image(path, dataset_index=layer)`` gives,
            when overlapping segments need more than one layer)

    Raises:
        ValueError: If the file is not a valid DICOM SEG or RTSTRUCT object, or a NRRD
            file without segments.

    Example:
        ```python
        from pictologics.loaders import get_segment_info

        segments = get_segment_info("segmentation.dcm")
        for seg in segments:
            print(f"{seg['segment_number']}: {seg['segment_label']}")
        ```
    """
    if str(path).lower().endswith((".nrrd", ".nhdr")):
        from pictologics.loaders.nrrd_loader import _nrrd_segment_info

        return _nrrd_segment_info(path)
    from pictologics.loader import _RTSTRUCT_SOP_CLASS, _dicom_sop_class

    if _dicom_sop_class(str(path)) == _RTSTRUCT_SOP_CLASS:
        from pictologics.loaders.rtstruct_loader import _rtstruct_info

        return _rtstruct_info(path)
    seg = _read_seg(path, stop_before_pixels=True)  # the header only

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
