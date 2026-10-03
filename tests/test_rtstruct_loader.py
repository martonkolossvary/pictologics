"""Tests of the RTSTRUCT loader, on structure sets that the tests write with pydicom."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from matplotlib.path import Path as PolygonPath
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from pictologics import RadiomicsPipeline, load_image, load_rtstruct
from pictologics.loader import Image
from pictologics.loaders import get_segment_info

# A reference grid with sagittal slices, anisotropic voxels and an origin away from 0
DIRECTION = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
REFERENCE = Image(np.zeros((40, 36, 6)), (0.75, 0.5, 2.5), (-25.0, -30.0, 10.0), DIRECTION)


def _world(slice_index: int, points: list[tuple[float, float]], reference: Image = REFERENCE) -> np.ndarray:  # fmt: skip
    """LPS+ points (mm) of (row, column) voxel coordinates on slice `slice_index`."""
    index = np.column_stack([np.asarray(points, dtype=float), np.full(len(points), slice_index)])
    return np.asarray(reference.origin) + (index * reference.spacing) @ np.asarray(reference.direction).T  # fmt: skip


def _write(path: Path, rois: list[tuple[int, str, list[tuple[str, np.ndarray]]]]) -> Path:
    """An RTSTRUCT file: (ROI number, name, [(contour type, points)]) for each ROI."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = (
        meta.MediaStorageSOPClassUID,
        meta.MediaStorageSOPInstanceUID,
    )
    ds.Modality = "RTSTRUCT"
    ds.StructureSetROISequence, ds.ROIContourSequence = Sequence(), Sequence()
    ds.RTROIObservationsSequence = Sequence()
    for number, name, contours in rois:
        roi = Dataset()
        roi.ROINumber, roi.ROIName = number, name
        ds.StructureSetROISequence.append(roi)
        item = Dataset()
        item.ReferencedROINumber = number
        if contours:
            item.ContourSequence = Sequence()
            for kind, points in contours:
                contour = Dataset()
                contour.ContourGeometricType, contour.NumberOfContourPoints = kind, len(points)
                contour.ContourData = [round(float(v), 6) for v in np.ravel(points)]
                item.ContourSequence.append(contour)
        ds.ROIContourSequence.append(item)
        if number % 2:  # an interpreted type for the odd ROI numbers only
            observation = Dataset()
            observation.ReferencedROINumber, observation.RTROIInterpretedType = number, "ORGAN"
            ds.RTROIObservationsSequence.append(observation)
    ds.save_as(path, enforce_file_format=True)
    return path


SQUARE = [(1.3, 2.2), (30.6, 2.2), (30.6, 25.7), (1.3, 25.7)]
OUTER = [(10.2, 10.4), (35.5, 12.1), (38.3, 33.6), (8.7, 31.2)]
HOLE = [(18.43, 18.31), (28.61, 19.27), (27.12, 27.74), (19.93, 26.48)]
_ANGLES = np.linspace(0, 2 * np.pi, 11)[:-1]
_RADII = np.where(np.arange(10) % 2, 6.3, 15.1)
STAR = list(zip(19.7 + _RADII * np.cos(_ANGLES), 17.4 + _RADII * np.sin(_ANGLES), strict=True))


def _expected(contours: list[tuple[int, list[tuple[float, float]]]]) -> np.ndarray:
    """matplotlib's point-in-polygon test of the voxel centers, the contours of a slice
    combined by XOR."""
    shape = REFERENCE.array.shape
    centers = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij"), -1)
    mask = np.zeros(shape, dtype=bool)
    for k, points in contours:
        mask[:, :, k] ^= (
            PolygonPath(points).contains_points(centers.reshape(-1, 2)).reshape(shape[:2])
        )
    return mask


def test_rtstruct_masks_match_point_in_polygon(tmp_path: Path) -> None:
    # Squares on three slices, a polygon with a hole, a star and a contour partly outside
    # the image fill the voxel centers that matplotlib finds inside (no center lies on an
    # edge here). ROIs without closed contours are left out with a warning.
    off_grid = [(-10.5, -5.5), (20.5, -5.5), (20.5, 15.5), (-10.5, 15.5)]
    rois = {
        "square": [(k, SQUARE) for k in (1, 2, 3)],
        "ring": [(2, OUTER), (2, HOLE)],
        "star": [(4, STAR), (5, off_grid)],
    }
    path = _write(
        tmp_path / "rs.dcm",
        [
            (3, "square", [("CLOSED_PLANAR", _world(k, p)) for k, p in rois["square"]]),
            (7, "ring", [("CLOSED_PLANAR", _world(k, p)) for k, p in rois["ring"]]),
            (8, "star", [("CLOSEDPLANAR_XOR", _world(k, p)) for k, p in rois["star"]]),
            (9, "point", [("POINT", _world(1, [(5.0, 5.0)]))]),
            (10, "empty", []),
        ],
    )
    with pytest.warns(UserWarning, match="Left out the ROIs without closed contours: 'point', 'empty'"):  # fmt: skip
        masks = load_rtstruct(path, REFERENCE, combine_rois=False)
    assert set(masks) == set(rois)
    for name, contours in rois.items():
        assert masks[name].array.dtype == np.uint8 and masks[name].modality == "RTSTRUCT"
        np.testing.assert_array_equal(masks[name].array.astype(bool), _expected(contours))
        assert masks[name].origin == REFERENCE.origin and masks[name].spacing == REFERENCE.spacing
    with pytest.warns(UserWarning, match="Left out"):
        labels = load_image(path, reference_image=REFERENCE)
    assert labels.array.dtype == np.uint8
    expected = np.zeros(REFERENCE.array.shape)
    for number, name in ((3, "square"), (7, "ring"), (8, "star")):  # a later ROI wins
        expected[masks[name].array.astype(bool)] = number
    np.testing.assert_array_equal(labels.array, expected)
    assert get_segment_info(path) == [
        {"segment_number": 3, "segment_label": "square", "contour_count": 3, "interpreted_type": "ORGAN"},
        {"segment_number": 7, "segment_label": "ring", "contour_count": 2, "interpreted_type": "ORGAN"},
        {"segment_number": 8, "segment_label": "star", "contour_count": 2},
        {"segment_number": 9, "segment_label": "point", "contour_count": 0, "interpreted_type": "ORGAN"},
        {"segment_number": 10, "segment_label": "empty", "contour_count": 0},
    ]  # fmt: skip


def test_rtstruct_rois_that_share_an_edge_share_no_voxel(tmp_path: Path) -> None:
    # Two ROIs split a rectangle along edges through voxel centers (a row, a column and
    # a diagonal): no voxel is in both, and together they fill the rectangle.
    rectangle = [(4.0, 3.0), (24.0, 3.0), (24.0, 21.0), (4.0, 21.0)]
    halves = {
        "row": (
            [(4.0, 3.0), (14.0, 3.0), (14.0, 21.0), (4.0, 21.0)],
            [(14.0, 3.0), (24.0, 3.0), (24.0, 21.0), (14.0, 21.0)],
        ),  # fmt: skip
        "column": (
            [(4.0, 3.0), (24.0, 3.0), (24.0, 12.0), (4.0, 12.0)],
            [(4.0, 12.0), (24.0, 12.0), (24.0, 21.0), (4.0, 21.0)],
        ),  # fmt: skip
        "diagonal": (
            [(4.0, 3.0), (24.0, 3.0), (4.0, 21.0)],
            [(24.0, 3.0), (24.0, 21.0), (4.0, 21.0)],
        ),
    }
    whole = load_rtstruct(_write(tmp_path / "whole.dcm", [(1, "r", [("CLOSED_PLANAR", _world(2, rectangle))])]), REFERENCE)  # fmt: skip
    for name, (first, second) in halves.items():
        path = _write(
            tmp_path / f"{name}.dcm",
            [
                (1, "a", [("CLOSED_PLANAR", _world(2, first))]),
                (2, "b", [("CLOSED_PLANAR", _world(2, second))]),
            ],  # fmt: skip
        )
        masks = load_rtstruct(path, REFERENCE, combine_rois=False)
        a, b = (masks[key].array.astype(bool) for key in ("a", "b"))
        assert not np.any(a & b) and a.any() and b.any()
        np.testing.assert_array_equal(a | b, whole.array.astype(bool))


def test_rtstruct_selection_tolerance_and_errors(tmp_path: Path) -> None:
    # roi_names chooses ROIs; ROI numbers above 255 give uint16 labels; contours near a
    # slice snap to it with a warning, far ones raise; clear errors otherwise.
    square = _world(1, SQUARE)
    drifted = square + np.asarray(REFERENCE.direction)[:, 2] * 2.5 * 0.2  # 0.2 voxels off
    path = _write(
        tmp_path / "rs.dcm",
        [
            (300, "GTV", [("CLOSED_PLANAR", square)]),
            (2, "PTV", [("CLOSED_PLANAR", _world(3, OUTER))]),
            (4, "PTV", [("CLOSED_PLANAR", _world(4, OUTER))]),
            (5, "drift", [("CLOSED_PLANAR", drifted)]),
            (6, "away", [("CLOSED_PLANAR", _world(40, SQUARE))]),
            (12, "beside", [("CLOSED_PLANAR", _world(1, [(-9.0, 2.0), (-3.0, 2.0), (-3.0, 9.0)]))]),
            (11, "none", []),
        ],
    )
    labels = load_rtstruct(path, REFERENCE, roi_names=["GTV"])
    assert labels.array.dtype == np.uint16 and set(np.unique(labels.array)) == {0, 300}
    with pytest.warns(UserWarning, match="up to 0.200 voxels off the slices"):
        drift = load_rtstruct(path, REFERENCE, roi_names=["drift"], combine_rois=False)["drift"]
    np.testing.assert_array_equal(drift.array, load_rtstruct(path, REFERENCE, roi_names=["GTV"], combine_rois=False)["GTV"].array)  # fmt: skip
    with pytest.raises(ValueError, match="lies 0.200 voxels off the slices"):
        load_rtstruct(path, REFERENCE, roi_names=["drift"], subvoxel_tolerance=0.1)
    for name in ("away", "beside"):  # a slice outside the image; rows beside it
        with pytest.warns(UserWarning, match=f"ROI '{name}' covers no voxel center"):
            load_rtstruct(path, REFERENCE, roi_names=[name])
    errors = {
        "ROI 'CTV' is not in": dict(roi_names=["CTV"]),
        "ROI name 'PTV' occurs 2 times": dict(roi_names=["PTV"]),
        "ROI 'none' has no closed contours": dict(roi_names=["none"]),
        r"ROI names \['PTV'\] repeat": dict(combine_rois=False),
    }
    for message, options in errors.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=message):
                load_rtstruct(path, REFERENCE, **options)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is an RTSTRUCT file: its contours need reference_image"):
        load_image(path)
    with pytest.raises(FileNotFoundError, match="RTSTRUCT file not found"):
        load_rtstruct(tmp_path / "missing.dcm", REFERENCE)
    (tmp_path / "text.dcm").write_text("not DICOM")
    with pytest.raises(ValueError, match="Failed to load DICOM RTSTRUCT file"):
        load_rtstruct(tmp_path / "text.dcm", REFERENCE)
    other = _write(tmp_path / "other.dcm", [])
    other_ds = __import__("pydicom").dcmread(other)
    other_ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    other_ds.save_as(other)
    with pytest.raises(ValueError, match="File is not a DICOM RTSTRUCT object"):
        load_rtstruct(other, REFERENCE)


def test_rtstruct_mask_in_the_pipeline(tmp_path: Path) -> None:
    # A pipeline run with an RTSTRUCT path as the mask gives the features of its mask
    # in memory.
    values = np.random.default_rng(2).normal(40.0, 10.0, REFERENCE.array.shape)
    image = Image(values, REFERENCE.spacing, REFERENCE.origin, REFERENCE.direction)
    path = _write(tmp_path / "rs.dcm", [(1, "GTV", [("CLOSED_PLANAR", _world(k, STAR)) for k in (1, 2, 3)])])  # fmt: skip
    pipeline = RadiomicsPipeline()
    pipeline.add_config(
        "first_order", [{"step": "extract_features", "params": {"families": ["intensity"]}}]
    )
    mask = load_rtstruct(path, image)
    from_file = pipeline.run(image, path, config_names=["first_order"])
    expected = pipeline.run(image, mask, config_names=["first_order"])
    assert from_file["first_order"].equals(expected["first_order"])
