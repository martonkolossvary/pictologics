"""
DICOM RTSTRUCT Loader
=====================

Reads DICOM RT Structure Sets: the closed planar contours of each ROI, filled onto the
voxel grid of a reference image. A voxel is in an ROI when its center lies inside an
odd number of the contours of the ROI on its slice (the even-odd rule), so a contour
inside another contour cuts a hole.
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from numpy import typing as npt

from pictologics.loader import _RTSTRUCT_SOP_CLASS, Image, _direction_matrix

# The contour types that bound an area (CLOSEDPLANAR_XOR names the even-odd rule)
_CLOSED_TYPES = ("CLOSED_PLANAR", "CLOSEDPLANAR_XOR")


def load_rtstruct(
    path: str | Path,
    reference_image: Image,
    roi_names: list[str] | None = None,
    combine_rois: bool = True,
    subvoxel_tolerance: float = 0.5,
    subvoxel_warning_threshold: float = 0.01,
) -> Image | dict[str, Image]:
    """Load the ROIs of a DICOM RTSTRUCT file as masks on the grid of `reference_image`.

    The closed planar contours of each ROI are filled onto the slices of the reference
    image: a voxel is in the ROI when its center lies inside an odd number of the
    contours of the ROI on its slice, so a contour inside another contour cuts a hole.
    A center on an edge is inside on one side of the edge only, so two ROIs that share
    an edge share no voxel. The plane of each contour must be a slice plane of the
    reference image (axial, sagittal or coronal on its grid).

    Args:
        path: Path to the RTSTRUCT file.
        reference_image: The image whose grid the masks fill (the image that the
            contours were drawn on, in the LPS+ frame as every loaded image).
        roi_names: The names of the ROIs to load. None loads every ROI with closed
            contours (the others are left out with a warning).
        combine_rois: True gives one label image, with the ROI Number of each ROI as
            its label (a later ROI wins where ROIs overlap; uint8, or uint16 for ROI
            numbers above 255). False gives a dict of binary masks by ROI name, which
            keeps overlapping ROIs.
        subvoxel_tolerance: Largest distance (in voxels) of a contour from the nearest
            slice of the reference image; a contour farther away raises an error.
        subvoxel_warning_threshold: Distance (in voxels) above which a warning tells
            that the contours are filled on the nearest slices.

    Returns:
        One label Image (combine_rois=True) or a dict {ROI name: binary mask Image},
        with the geometry of the reference image and modality "RTSTRUCT".

    Raises:
        ValueError: If the file is not an RTSTRUCT, a requested ROI is not in the file
            or has no closed contours, ROI names repeat where names must be unique, or a
            contour does not lie on the slices of the reference image.
        FileNotFoundError: If the file does not exist.

    Example:
        ```python
        from pictologics import load_image
        from pictologics.loaders import get_segment_info, load_rtstruct

        ct = load_image("ct_scan/")
        print(get_segment_info("rtstruct.dcm"))  # ROI numbers and names
        masks = load_rtstruct("rtstruct.dcm", ct, roi_names=["GTV", "PTV"], combine_rois=False)
        ```
    """
    rois = _rois(_read_rtstruct(path))
    counts = Counter(name for name, _ in rois.values())
    if roi_names is None:
        chosen = [number for number, (_, contours) in rois.items() if contours]
        left_out = [name for name, contours in rois.values() if not contours]
        if left_out:
            warnings.warn(
                f"Left out the ROIs without closed contours: {', '.join(map(repr, left_out))}.",
                UserWarning,
                stacklevel=2,
            )
    else:
        numbers = {name: number for number, (name, _) in rois.items()}
        for name in roi_names:
            if name not in numbers:
                raise ValueError(f"ROI {name!r} is not in '{path}'. Its ROIs: {list(counts)}.")
            if counts[name] > 1:
                raise ValueError(f"ROI name {name!r} occurs {counts[name]} times in '{path}'.")
            if not rois[numbers[name]][1]:
                raise ValueError(f"ROI {name!r} has no closed contours in '{path}'.")
        chosen = [numbers[name] for name in roi_names]
    if not combine_rois:
        repeated = sorted({rois[n][0] for n in chosen if counts[rois[n][0]] > 1})
        if repeated:
            raise ValueError(
                f"ROI names {repeated} repeat in '{path}': use combine_rois=True (labels "
                "by ROI Number)."
            )
    masks = {
        number: _filled(rois[number], reference_image, subvoxel_tolerance, subvoxel_warning_threshold)
        for number in chosen
    }  # fmt: skip
    for number, mask in masks.items():
        if not mask.any():
            warnings.warn(
                f"ROI {rois[number][0]!r} covers no voxel center of the reference image.",
                UserWarning,
                stacklevel=2,
            )

    def image(array: npt.NDArray[Any]) -> Image:
        return Image(
            array=array,
            spacing=reference_image.spacing,
            origin=reference_image.origin,
            direction=reference_image.direction,
            modality="RTSTRUCT",
        )

    if not combine_rois:
        return {rois[number][0]: image(mask.view(np.uint8)) for number, mask in masks.items()}
    labels = np.zeros(reference_image.array.shape, np.uint8 if max(masks, default=0) <= 255 else np.uint16)  # fmt: skip
    for number, mask in masks.items():
        labels[mask] = number
    return image(labels)


def _read_rtstruct(path: str | Path) -> pydicom.Dataset:
    """Read a DICOM RTSTRUCT file (RT Structure Set Storage)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"RTSTRUCT file not found: {path}")
    try:
        dataset = pydicom.dcmread(str(path))
    except Exception as e:
        raise ValueError(f"Failed to load DICOM RTSTRUCT file: {e}") from e
    if str(getattr(dataset, "SOPClassUID", "")) != _RTSTRUCT_SOP_CLASS:
        raise ValueError(f"File is not a DICOM RTSTRUCT object: {path}")
    return dataset


def _rois(dataset: pydicom.Dataset) -> dict[int, tuple[str, list[npt.NDArray[np.float64]]]]:
    """The ROIs by ROI Number, in file order: their names and their closed contours, as
    (points, 3) arrays of LPS+ coordinates (mm)."""
    rois: dict[int, tuple[str, list[npt.NDArray[np.float64]]]] = {
        int(item.ROINumber): (str(getattr(item, "ROIName", "")), [])
        for item in getattr(dataset, "StructureSetROISequence", [])
    }
    for item in getattr(dataset, "ROIContourSequence", []):
        number = int(item.ReferencedROINumber)
        for contour in getattr(item, "ContourSequence", []):
            if number in rois and contour.ContourGeometricType in _CLOSED_TYPES:
                # numpy reads the raw text of the numbers ("x\\y\\z..."): the same values,
                # 14 times faster than the decimal strings of pydicom
                text = contour.get_item("ContourData").value
                points = np.array(text.split(b"\\"), dtype=np.float64).reshape(-1, 3)
                rois[number][1].append(points)
    return rois


def _rtstruct_info(path: str | Path) -> list[dict[str, str | int]]:
    """The ROIs of an RTSTRUCT file (see get_segment_info)."""
    dataset = _read_rtstruct(path)
    types = {
        int(item.ReferencedROINumber): str(item.RTROIInterpretedType)
        for item in getattr(dataset, "RTROIObservationsSequence", [])
        if getattr(item, "RTROIInterpretedType", "")
    }
    info: list[dict[str, str | int]] = []
    for number, (name, contours) in _rois(dataset).items():
        entry: dict[str, str | int] = {
            "segment_number": number,
            "segment_label": name,
            "contour_count": len(contours),
        }
        if number in types:
            entry["interpreted_type"] = types[number]
        info.append(entry)
    return info


def _filled(
    roi: tuple[str, list[npt.NDArray[np.float64]]],
    reference: Image,
    tolerance: float,
    warning_threshold: float,
) -> npt.NDArray[np.bool_]:
    """The mask of the closed contours of `roi` on the grid of `reference`."""
    name, contours = roi
    shape = reference.array.shape
    to_index = np.linalg.inv(_direction_matrix(reference.direction) * np.asarray(reference.spacing))
    origin = np.asarray(reference.origin, dtype=np.float64)
    planes: dict[tuple[int, int], list[npt.NDArray[np.float64]]] = {}
    drift = 0.0
    for points in contours:
        index = (points - origin) @ to_index.T  # voxel coordinates: centers at integers
        axis = int(np.argmin(np.ptp(index, axis=0)))  # the normal of the contour plane
        k = int(np.round(np.mean(index[:, axis])))
        offset = float(np.max(np.abs(index[:, axis] - k)))
        if offset > tolerance:
            raise ValueError(
                f"A contour of ROI {name!r} lies {offset:.3f} voxels off the slices of the "
                f"reference image (subvoxel_tolerance={tolerance}): its plane is not a "
                "slice plane of the image."
            )
        drift = max(drift, offset)
        if 0 <= k < shape[axis]:
            planes.setdefault((axis, k), []).append(np.delete(index, axis, axis=1))
    if drift > warning_threshold:
        warnings.warn(
            f"The contours of ROI {name!r} lie up to {drift:.3f} voxels off the slices of "
            "the reference image; they are filled on the nearest slices.",
            UserWarning,
            stacklevel=3,
        )
    mask = np.zeros(shape, dtype=bool)
    for (axis, k), polygons in planes.items():
        plane = mask[tuple(k if a == axis else slice(None) for a in range(3))]
        _fill_even_odd(plane, polygons)
    return mask


def _fill_even_odd(plane: npt.NDArray[np.bool_], polygons: list[npt.NDArray[np.float64]]) -> None:
    """Set the pixels of `plane` whose centers lie inside an odd number of the closed
    `polygons` ((row, column) pixel coordinates, centers at integers).

    Each edge crosses the pixel rows r with min(r0, r1) <= r < max(r0, r1) (a vertex
    counts once); a pixel is inside when an odd number of crossings of its row lie at or
    before its column.
    """
    rows_n, cols_n = plane.shape
    starts = np.concatenate(polygons)
    ends = np.concatenate([np.roll(p, -1, axis=0) for p in polygons])
    r0, c0, r1, c1 = starts[:, 0], starts[:, 1], ends[:, 0], ends[:, 1]
    first = np.clip(np.ceil(np.minimum(r0, r1)), 0, rows_n).astype(np.intp)
    stop = np.clip(np.ceil(np.maximum(r0, r1)), 0, rows_n).astype(np.intp)
    counts = stop - first
    if not counts.any():
        return
    edge = np.repeat(np.arange(len(starts)), counts)
    rows = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts - first, counts)
    cols = c0[edge] + (rows - r0[edge]) * (c1[edge] - c0[edge]) / (r1[edge] - r0[edge])
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    # The crossings of a row pair up: in at crossing 2i, out at crossing 2i + 1
    low = rows[0]
    steps = np.zeros((rows[-1] - low + 1, cols_n + 1), dtype=np.int32)
    begin = np.clip(np.ceil(cols[0::2]), 0, cols_n).astype(np.intp)
    end = np.clip(np.ceil(cols[1::2]), 0, cols_n).astype(np.intp)
    np.add.at(steps, (rows[0::2] - low, begin), 1)
    np.add.at(steps, (rows[0::2] - low, end), -1)
    plane[low : rows[-1] + 1] |= np.cumsum(steps[:, :-1], axis=1) > 0
