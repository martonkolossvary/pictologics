"""Tests of the MetaImage loader, on small files that the tests write."""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from pictologics import load_image

VALUES = np.arange(24, dtype=np.int16).reshape(4, 3, 2) - 5  # (x, y, z)
HEADER = {
    "ObjectType": "Image",
    "NDims": "3",
    "BinaryData": "True",
    "BinaryDataByteOrderMSB": "False",
    "CompressedData": "False",
    "TransformMatrix": "1 0 0 0 1 0 0 0 1",
    "Offset": "1 2 3",
    "ElementSpacing": "0.5 0.75 2",
    "DimSize": "4 3 2",
    "ElementType": "MET_SHORT",
    "ElementDataFile": "LOCAL",
}


def _header(**fields: Optional[str]) -> dict[str, str]:
    """HEADER with changed fields (None leaves a field out), ElementDataFile last."""
    header = {**HEADER, **fields}
    header["ElementDataFile"] = header.pop("ElementDataFile")
    return {key: value for key, value in header.items() if value is not None}


def _write(path: Path, header: dict[str, str], data: bytes = b"") -> Path:
    path.write_bytes("".join(f"{k} = {v}\n" for k, v in header.items()).encode() + data)
    return path


def _raw(values: np.ndarray, dtype: str = "<i2") -> bytes:
    """The bytes of `values` with the first axis fastest, as MetaImage stores them."""
    return np.ascontiguousarray(values.transpose()).astype(dtype).tobytes()


def test_metaimage_local_data(tmp_path: Path) -> None:
    # Data after the header: raw in both byte orders (two names of the field), compressed,
    # and text (true and false in any case), as float64 with the geometry of the header.
    cases = {
        "raw": (_header(), _raw(VALUES)),
        "msb": (
            _header(BinaryDataByteOrderMSB="True", ElementType="MET_FLOAT"),
            _raw(VALUES, ">f4"),
        ),  # fmt: skip
        "element": (
            _header(BinaryDataByteOrderMSB=None, ElementByteOrderMSB="True"),
            _raw(VALUES, ">i2"),
        ),  # fmt: skip
        "zlib": (_header(CompressedData="true"), zlib.compress(_raw(VALUES))),
        "text": (
            _header(BinaryData="false"),
            " ".join(map(str, VALUES.transpose().ravel())).encode(),
        ),  # fmt: skip
    }
    for name, (header, data) in cases.items():
        image = load_image(_write(tmp_path / f"{name}.mha", header, data))
        assert image.array.dtype == np.float64 and image.modality == "MetaImage"
        np.testing.assert_array_equal(image.array, VALUES)
        assert image.spacing == (0.5, 0.75, 2.0) and image.origin == (1.0, 2.0, 3.0)


def test_metaimage_detached_data(tmp_path: Path) -> None:
    # A .mhd header reads its data file: after HeaderSize bytes, the last bytes of the file
    # for HeaderSize -1, and compressed data (.zraw).
    (tmp_path / "a.raw").write_bytes(b"skip it!" + _raw(VALUES))
    (tmp_path / "b.raw").write_bytes(b"0123456789" + _raw(VALUES))
    (tmp_path / "c.zraw").write_bytes(zlib.compress(_raw(VALUES)))
    headers = (
        _header(HeaderSize="8", ElementDataFile="a.raw"),
        _header(HeaderSize="-1", ElementDataFile="b.raw"),
        _header(CompressedData="True", ElementDataFile="c.zraw"),
    )
    for k, header in enumerate(headers):
        np.testing.assert_array_equal(
            load_image(_write(tmp_path / f"{k}.mhd", header)).array, VALUES
        )


def test_metaimage_geometry_fields(tmp_path: Path) -> None:
    # The first three numbers of TransformMatrix are the direction of axis 0; Origin and
    # Position name the origin as Offset does, ElementSize the spacing; without them the
    # origin is 0, the spacing 1 and the direction the identity.
    turned = _write(tmp_path / "t.mha", _header(TransformMatrix="0 1 0 0 0 1 1 0 0"), _raw(VALUES))
    np.testing.assert_array_equal(load_image(turned).direction, [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    for name in ("Origin", "Position"):
        other = _write(
            tmp_path / f"{name}.mha", _header(Offset=None, **{name: "4 5 6"}), _raw(VALUES)
        )
        assert load_image(other).origin == (4.0, 5.0, 6.0)
    size = _write(
        tmp_path / "size.mha", _header(ElementSpacing=None, ElementSize="2 3 4"), _raw(VALUES)
    )
    assert load_image(size).spacing == (2.0, 3.0, 4.0)
    bare = _write(tmp_path / "bare.mha", _header(TransformMatrix=None, Offset=None, ElementSpacing=None), _raw(VALUES))  # fmt: skip
    image = load_image(bare)
    assert image.spacing == (1.0, 1.0, 1.0) and image.origin == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(image.direction, np.eye(3))


def test_metaimage_4d_volume(tmp_path: Path) -> None:
    # A 4D image gives the volume of dataset_index (the fourth axis), with 3D geometry.
    volumes = np.stack([VALUES, VALUES * 2], -1)
    header = _header(
        NDims="4", DimSize="4 3 2 2", ElementSpacing="0.5 0.75 2 1", Offset="1 2 3 0",
        TransformMatrix="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
    )  # fmt: skip
    image = load_image(_write(tmp_path / "four.mha", header, _raw(volumes)), dataset_index=1)
    np.testing.assert_array_equal(image.array, VALUES * 2)
    assert image.spacing == (0.5, 0.75, 2.0) and image.origin == (1.0, 2.0, 3.0)
    np.testing.assert_array_equal(image.direction, np.eye(3))


def test_metaimage_errors(tmp_path: Path) -> None:
    # Clear errors for files that are not MetaImage and what is not supported.
    (tmp_path / "text.mha").write_bytes(b"just some text\n")
    with pytest.raises(ValueError, match="is not a MetaImage file: its header has no 'NDims'"):
        load_image(tmp_path / "text.mha")
    cases = {
        "2 dimensions; 3 or 4 are supported": (_header(NDims="2", DimSize="4 3"), _raw(VALUES)),
        "2 sizes for 3 dimensions": (_header(DimSize="4 3"), _raw(VALUES)),
        "element type 'MET_FLOAT_ARRAY' is not supported": (
            _header(ElementType="MET_FLOAT_ARRAY"),
            b"",
        ),  # fmt: skip
        "more than one channel": (_header(ElementNumberOfChannels="3"), b""),
        "more than one file": (_header(ElementDataFile="LIST"), b""),
        "data hold 24 of 48 bytes": (_header(), _raw(VALUES)[:24]),
        "data hold 3 of 24 values": (_header(BinaryData="False"), b"1 2 3"),
    }
    for message, (header, data) in cases.items():
        with pytest.raises(ValueError, match=message):
            load_image(_write(tmp_path / "bad.mha", header, data))
