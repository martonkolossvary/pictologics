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


def _write(
    path: Path,
    rois: list[tuple[int, str, list[tuple[str, np.ndarray]]]],
    frames: dict[int, str] | None = None,
) -> Path:
    """An RTSTRUCT file: (ROI number, name, [(contour type, points)]) for each ROI, and
    the ReferencedFrameOfReferenceUID of the ROI numbers in `frames`."""
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
        if frames and number in frames:
            roi.ReferencedFrameOfReferenceUID = frames[number]
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


def test_rtstruct_frame_of_reference(tmp_path: Path) -> None:
    # ROIs of the frame of the image load quietly, and the masks carry its UID; ROIs of
    # another frame warn (named once for their frame); an image without a UID is not
    # checked.
    from dataclasses import replace

    reference = replace(REFERENCE, frame_of_reference_uid="1.2.3")
    contours = [("CLOSED_PLANAR", _world(2, SQUARE))]
    path = _write(
        tmp_path / "rs.dcm",
        [(1, "GTV", contours), (2, "PTV", contours), (3, "lung", contours)],
        frames={1: "1.2.3", 2: "9.9", 3: "9.9"},
    )
    with pytest.warns(UserWarning, match="ROI 'PTV', 'lung' refers to the frame of reference 9.9"):
        masks = load_rtstruct(path, reference, combine_rois=False)
    assert isinstance(masks, dict)
    assert all(mask.frame_of_reference_uid == "1.2.3" for mask in masks.values())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_rtstruct(path, reference, roi_names=["GTV"])
        load_rtstruct(path, REFERENCE, combine_rois=False)


def test_rtstruct_fill_writes_only_the_columns_of_the_crossings() -> None:
    # A fill writes only the columns from the first to the last crossing of its rows: the
    # pixels of the full-width fill of 0.6.0 (a copy below), also for vertices on pixel
    # centers and polygons partly outside the plane.
    from pictologics.loaders.rtstruct_loader import _fill_even_odd

    def fill_of_0_6_0(plane: np.ndarray, polygons: list[np.ndarray]) -> None:
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
        low = rows[0]
        steps = np.zeros((rows[-1] - low + 1, cols_n + 1), dtype=np.int32)
        begin = np.clip(np.ceil(cols[0::2]), 0, cols_n).astype(np.intp)
        end = np.clip(np.ceil(cols[1::2]), 0, cols_n).astype(np.intp)
        np.add.at(steps, (rows[0::2] - low, begin), 1)
        np.add.at(steps, (rows[0::2] - low, end), -1)
        plane[low : rows[-1] + 1] |= np.cumsum(steps[:, :-1], axis=1) > 0

    rng = np.random.default_rng(5)
    for trial in range(200):
        shape = (int(rng.integers(4, 40)), int(rng.integers(4, 40)))
        polygons = [
            rng.uniform(-4.0, max(shape) + 4.0, (int(rng.integers(3, 10)), 2))
            for _ in range(int(rng.integers(1, 4)))
        ]
        if trial % 2:
            polygons = [np.round(p) for p in polygons]  # vertices on pixel centers
        expected = np.zeros(shape, dtype=bool)
        fill_of_0_6_0(expected, polygons)
        got = np.zeros(shape, dtype=bool)
        _fill_even_odd(got, polygons)
        np.testing.assert_array_equal(got, expected)


def test_rtstruct_labels_on_planes_of_each_axis(tmp_path: Path) -> None:
    # Contours on the planes of each index axis of an axial grid fill the voxel centers
    # that matplotlib finds inside; in the label map a later ROI wins where ROIs overlap,
    # and an ROI that covers no voxel center warns and has no label.
    reference = Image(np.zeros((30, 26, 22)), (0.8, 0.6, 1.5), (-20.0, 15.0, 7.5))
    shape = reference.array.shape

    def contour(axis: int, k: int, points: list[tuple[float, float]]) -> np.ndarray:
        index = np.insert(np.asarray(points, dtype=float), axis, k, axis=1)
        return np.asarray(reference.origin) + index * reference.spacing

    def inside(axis: int, k: int, points: list[tuple[float, float]]) -> np.ndarray:
        plane_shape = tuple(n for a, n in enumerate(shape) if a != axis)
        grid = np.meshgrid(*(np.arange(n) for n in plane_shape), indexing="ij")
        centers = np.stack(grid, -1).reshape(-1, 2)
        mask = np.zeros(shape, dtype=bool)
        plane = PolygonPath(points).contains_points(centers).reshape(plane_shape)
        mask[tuple(k if a == axis else slice(None) for a in range(3))] = plane
        return mask

    shapes = {
        "x": (0, 12, [(3.3, 2.2), (20.6, 4.1), (18.2, 17.7), (2.4, 15.3)]),
        "y": (1, 9, [(5.1, 3.3), (25.7, 2.9), (21.4, 19.6)]),
        "z": (2, 6, [(4.2, 3.1), (26.3, 5.5), (24.1, 21.8), (6.6, 20.2)]),
    }
    rois = [(n, name, [("CLOSED_PLANAR", contour(*shapes[name]))]) for n, name in ((4, "x"), (2, "y"), (9, "z"))]  # fmt: skip
    rois.append((5, "tiny", [("CLOSED_PLANAR", contour(2, 3, [(5.2, 5.2), (5.4, 5.2), (5.3, 5.4)]))]))  # fmt: skip
    path = _write(tmp_path / "rs.dcm", rois)
    with pytest.warns(UserWarning, match="ROI 'tiny' covers no voxel center"):
        masks = load_rtstruct(path, reference, combine_rois=False)
    with pytest.warns(UserWarning, match="ROI 'tiny' covers no voxel center"):
        labels = load_rtstruct(path, reference)
    expected = np.zeros(shape)
    for number, name in ((4, "x"), (2, "y"), (9, "z")):
        mask = inside(*shapes[name])
        np.testing.assert_array_equal(masks[name].array.astype(bool), mask)
        expected[mask] = number
    assert not masks["tiny"].array.any()
    assert np.any(inside(*shapes["x"]) & inside(*shapes["z"]))  # the ROIs overlap
    np.testing.assert_array_equal(labels.array, expected)
