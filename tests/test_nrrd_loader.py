"""Tests of the NRRD loader, on small files that the tests write."""

from __future__ import annotations

import bz2
import gzip
from pathlib import Path

import numpy as np
import pytest

from pictologics import load_image
from pictologics.loaders import get_segment_info

VALUES = np.arange(24, dtype=np.int16).reshape(4, 3, 2) - 5  # (x, y, z)
HEADER = [
    "type: short",
    "dimension: 3",
    "space: left-posterior-superior",
    "sizes: 4 3 2",
    "space directions: (0.5,0,0) (0,0.75,0) (0,0,2)",
    "kinds: domain domain domain",
    "endian: little",
    "encoding: raw",
    "space origin: (1,2,3)",
]


def _write(path: Path, lines: list[str], data: bytes = b"") -> Path:
    """A NRRD file: the magic line, a comment, the header lines, a blank line, the data."""
    path.write_bytes(("NRRD0004\n# a comment\n" + "\n".join(lines) + "\n\n").encode() + data)
    return path


def _raw(values: np.ndarray, dtype: str = "<i2") -> bytes:
    """The bytes of `values` with the first axis fastest, as NRRD stores them."""
    return np.ascontiguousarray(values.transpose()).astype(dtype).tobytes()


def _without(*names: str) -> list[str]:
    return [line for line in HEADER if not line.startswith(names)]


def test_nrrd_encodings_types_and_byte_orders(tmp_path: Path) -> None:
    # Raw data in both byte orders, gzip, bzip2 and text give the stored values as float64,
    # with the geometry of the header; other type names read their own types.
    lines = _without("encoding", "endian")
    cases = {
        "raw": (["encoding: raw", "endian: little"], _raw(VALUES)),
        "big": (["encoding: raw", "endian: big"], _raw(VALUES, ">i2")),
        "gzip": (["encoding: gzip", "endian: little"], gzip.compress(_raw(VALUES))),
        "bzip2": (["encoding: bz2", "endian: little"], bz2.compress(_raw(VALUES))),
        "text": (["encoding: text"], " ".join(map(str, VALUES.transpose().ravel())).encode()),
    }
    for name, (extra, data) in cases.items():
        image = load_image(_write(tmp_path / f"{name}.nrrd", lines + extra, data))
        assert image.array.dtype == np.float64 and image.modality == "Nrrd"
        np.testing.assert_array_equal(image.array, VALUES)
        assert image.spacing == (0.5, 0.75, 2.0) and image.origin == (1.0, 2.0, 3.0)
    for type_name, dtype in (("unsigned char", "u1"), ("float", "<f4"), ("double", "<f8"), ("uint32", "<u4")):  # fmt: skip
        typed = [f"type: {type_name}", *_without("type")]
        image = load_image(_write(tmp_path / f"{dtype}.nrrd", typed, _raw(VALUES + 5, dtype)))
        np.testing.assert_array_equal(image.array, VALUES + 5)


def test_nrrd_large_volume_in_row_order(tmp_path: Path) -> None:
    # A large volume (here: with a small size limit) comes in row order: in one pass for a
    # native type, after a byte swap for a big-endian one.
    from unittest.mock import patch

    with patch("pictologics.loader._ROW_ORDER_MIN_SIZE", 8):
        for endian, dtype in (("little", "<i2"), ("big", ">i2")):
            lines = [f"endian: {endian}", *_without("endian")]
            image = load_image(_write(tmp_path / f"{endian}.nrrd", lines, _raw(VALUES, dtype)))
            np.testing.assert_array_equal(image.array, VALUES)
            assert image.array.flags.c_contiguous


def test_nrrd_spaces_give_lps_geometry(tmp_path: Path) -> None:
    # RAS and LAS change the signs of their X and Y (LAS: Y) directions and origins; a
    # space without a name is taken as it is; other spaces are not supported.
    for space, signs in (
        ("RAS", (-1.0, -1.0, 1.0)),
        ("right-anterior-superior", (-1.0, -1.0, 1.0)),
        ("left-anterior-superior", (1.0, -1.0, 1.0)),
        ("LPS", (1.0, 1.0, 1.0)),
    ):
        path = _write(
            tmp_path / "space.nrrd", [f"space: {space}", *_without("space:")], _raw(VALUES)
        )
        image = load_image(path)
        assert image.origin == (signs[0], 2.0 * signs[1], 3.0)
        assert image.spacing == (0.5, 0.75, 2.0)
        np.testing.assert_array_equal(image.direction, np.diag(signs))
    path = _write(
        tmp_path / "unnamed.nrrd", ["space dimension: 3", *_without("space:")], _raw(VALUES)
    )
    np.testing.assert_array_equal(load_image(path).direction, np.eye(3))
    path = _write(
        tmp_path / "scanner.nrrd", ["space: scanner-xyz", *_without("space:")], _raw(VALUES)
    )
    with pytest.raises(ValueError, match="space 'scanner-xyz' is not supported"):
        load_image(path)


def test_nrrd_detached_data_with_skips(tmp_path: Path) -> None:
    # A .nhdr header reads its data file: after skipped lines and bytes; the last bytes of
    # the file for a byte skip of -1; and for gzip, a skip after the decompression.
    lines = _without("encoding")
    (tmp_path / "a.raw").write_bytes(b"line one\nline two\nXXXX" + _raw(VALUES))
    a = _write(tmp_path / "a.nhdr", [*lines, "encoding: raw", "data file: a.raw", "line skip: 2", "byte skip: 4"])  # fmt: skip
    (tmp_path / "b.raw").write_bytes(b"0123456789" + _raw(VALUES))
    b = _write(tmp_path / "b.nhdr", [*lines, "encoding: raw", "datafile: b.raw", "byteskip: -1"])
    (tmp_path / "c.raw.gz").write_bytes(gzip.compress(b"XX" + _raw(VALUES)))
    c = _write(
        tmp_path / "c.nhdr", [*lines, "encoding: gzip", "data file: c.raw.gz", "byte skip: 2"]
    )
    for header in (a, b, c):
        np.testing.assert_array_equal(load_image(header).array, VALUES)


def test_nrrd_volumes_of_a_list_axis(tmp_path: Path) -> None:
    # The layers of a .seg.nrrd file (a list axis first) and the volumes of a 4D space (as
    # ITK writes) load by dataset_index; two axes that are not spatial do not load.
    layers = np.stack([VALUES, VALUES * 2])  # (layer, x, y, z)
    first = [
        "type: short", "dimension: 4", "space: left-posterior-superior", "sizes: 2 4 3 2",
        "space directions: none (0.5,0,0) (0,0.75,0) (0,0,2)",
        "kinds: list domain domain domain", "endian: little", "encoding: raw", "space origin: (1,2,3)",
    ]  # fmt: skip
    path = _write(tmp_path / "layers.seg.nrrd", first, _raw(layers))
    for k in (0, 1):
        np.testing.assert_array_equal(load_image(path, dataset_index=k).array, layers[k])
    with pytest.raises(ValueError, match="Dataset index 2 is out of bounds for 4D image with 2"):
        load_image(path, dataset_index=2)
    itk = [
        "type: short", "dimension: 4", "space dimension: 4", "sizes: 4 3 2 2",
        "space directions: (0.5,0,0,0) (0,0.75,0,0) (0,0,2,0) (0,0,0,1)",
        "endian: little", "encoding: raw", "space origin: (1,2,3,0)",
    ]  # fmt: skip
    image = load_image(_write(tmp_path / "itk.nrrd", itk, _raw(np.stack([VALUES, -VALUES], -1))), 1)
    np.testing.assert_array_equal(image.array, -VALUES)
    assert image.spacing == (0.5, 0.75, 2.0) and image.origin == (1.0, 2.0, 3.0)
    two = [
        "type: short", "dimension: 5", "sizes: 1 2 4 3 2", "endian: little", "encoding: raw",
        "space directions: none none (0.5,0,0) (0,0.75,0) (0,0,2)",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="2 axes that are not spatial"):
        load_image(_write(tmp_path / "two.nrrd", two, _raw(layers[np.newaxis])))


def test_nrrd_without_space_uses_kinds_and_spacings(tmp_path: Path) -> None:
    # Without space directions: the axes of a spatial kind (all axes without kinds), their
    # spacings (1 for nan or without spacings), the origin 0 and the identity direction.
    base = ["type: short", "endian: little", "encoding: raw"]
    plain = _write(tmp_path / "plain.nrrd", [*base, "dimension: 3", "sizes: 4 3 2", "spacings: 0.5 nan 2"], _raw(VALUES))  # fmt: skip
    image = load_image(plain)
    assert image.spacing == (0.5, 1.0, 2.0) and image.origin == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(image.direction, np.eye(3))
    layers = np.stack([VALUES, VALUES * 3], -1)  # (x, y, z, layer)
    listed = _write(tmp_path / "listed.nrrd", [*base, "dimension: 4", "sizes: 4 3 2 2", "kinds: domain domain space list"], _raw(layers))  # fmt: skip
    image = load_image(listed, dataset_index=1)
    np.testing.assert_array_equal(image.array, VALUES * 3)
    assert image.spacing == (1.0, 1.0, 1.0)


def test_nrrd_errors(tmp_path: Path) -> None:
    # Clear errors for files that are not NRRD, bad headers, and what is not supported.
    (tmp_path / "pgm.nrrd").write_bytes(b"P5 not a NRRD file\n")
    with pytest.raises(ValueError, match="is not a NRRD file"):
        load_image(tmp_path / "pgm.nrrd")
    cases = {
        "is not 'field: value'": ([*HEADER, "a line"], _raw(VALUES)),
        "has no 'encoding' field": (_without("encoding"), _raw(VALUES)),
        "2 sizes for dimension 3": (["sizes: 4 3", *_without("sizes")], _raw(VALUES)),
        "type 'block' is not supported": (["type: block", *_without("type")], _raw(VALUES)),
        "encoding 'hex' is not supported": (["encoding: hex", *_without("encoding")], b"00"),
        "more than one file": (["data file: LIST", *HEADER], b""),
        "data hold 24 of 48 bytes": (HEADER, _raw(VALUES)[:24]),
        "data hold 3 of 24 values": (["encoding: ascii", *_without("encoding")], b"1 2 3"),
        "needs 3 spatial axes in 'space directions'": (
            ["space directions: none (0.5,0,0) (0,0.75,0)", *_without("space directions")],
            _raw(VALUES),
        ),
        "needs 3 spatial axes, it has 1": (
            ["kinds: list list domain", *_without("space", "kinds")],
            _raw(VALUES),
        ),
    }
    for message, (lines, data) in cases.items():
        with pytest.raises(ValueError, match=message):
            load_image(_write(tmp_path / "bad.nrrd", lines, data))
    (tmp_path / "slice001.raw").write_bytes(_raw(VALUES))
    pattern = _write(tmp_path / "pattern.nhdr", ["data file: slice%03d.raw 1 2 1", *HEADER])
    with pytest.raises(ValueError, match="more than one file"):
        load_image(pattern)


def test_seg_nrrd_segment_info(tmp_path: Path) -> None:
    # The Segment fields of a .seg.nrrd header: name, ID, label value and layer (an old
    # file without the last two: value 1, one layer for each segment).
    lines = [
        *HEADER,
        "Segment0_ID:=liver", "Segment0_Name:=liver", "Segment0_LabelValue:=1", "Segment0_Layer:=0",
        "Segment1_ID:=tumor", "Segment1_Name:=tumor: core", "Segment1_LabelValue:=3", "Segment1_Layer:=1",
        "Segment2_Name:=old",
    ]  # fmt: skip
    assert get_segment_info(_write(tmp_path / "s.seg.nrrd", lines, _raw(VALUES))) == [
        {"segment_index": 0, "segment_label": "liver", "segment_id": "liver", "label_value": 1, "layer": 0},
        {"segment_index": 1, "segment_label": "tumor: core", "segment_id": "tumor", "label_value": 3, "layer": 1},
        {"segment_index": 2, "segment_label": "old", "segment_id": "", "label_value": 1, "layer": 2},
    ]  # fmt: skip
    with pytest.raises(ValueError, match="holds no segments"):
        get_segment_info(_write(tmp_path / "plain.nrrd", HEADER, _raw(VALUES)))


def test_nrrd_channel_axes_raise(tmp_path: Path) -> None:
    # An axis of colours (or vectors) holds channels, not volumes: the reader takes one
    # value per voxel, so it raises (a "list" axis of 3D Slicer layers loads as volumes)
    header = [line.replace("dimension: 3", "dimension: 4") for line in HEADER]
    header = [line.replace("sizes: 4 3 2", "sizes: 3 4 3 2") for line in header]
    header = [
        line.replace("space directions: ", "space directions: none ").replace(
            "kinds: domain domain domain", "kinds: RGB-color domain domain domain"
        )
        for line in header
    ]
    path = _write(tmp_path / "rgb.nrrd", header, _raw(np.zeros((2, 3, 4, 3), np.int16)))
    with pytest.raises(ValueError, match="an axis of kind 'rgb-color' holds channels"):
        load_image(path)
