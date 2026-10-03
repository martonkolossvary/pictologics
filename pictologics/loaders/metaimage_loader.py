"""
MetaImage Loader
================

Reads MetaImage files (the ITK format) into pictologics Image instances, without other
packages: ``.mha`` files with the data after the header, and ``.mhd`` headers with a
detached data file (``.raw``, or ``.zraw`` with compressed data).

MetaImage geometry is in the LPS+ world frame, as in ITK and in the package.
"""

from __future__ import annotations

import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy import typing as npt

from pictologics.loader import Image, _float64_volume

# The element types of the MetaImage format and their numpy types (MET_LONG has 4 bytes)
_TYPES = {
    "MET_CHAR": "i1",
    "MET_UCHAR": "u1",
    "MET_SHORT": "i2",
    "MET_USHORT": "u2",
    "MET_INT": "i4",
    "MET_UINT": "u4",
    "MET_LONG": "i4",
    "MET_ULONG": "u4",
    "MET_LONG_LONG": "i8",
    "MET_ULONG_LONG": "u8",
    "MET_FLOAT": "f4",
    "MET_DOUBLE": "f8",
}


def _numbers(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def _true(header: dict[str, str], key: str, default: str = "False") -> bool:
    """Whether a field of the header is true ("True", "true" or "1")."""
    return header.get(key, default).lower() in ("true", "1")


def _load_metaimage(path: str | Path, dataset_index: int = 0) -> Image:
    """Load a MetaImage file (see the module docstring) as an Image.

    A 4D image gives its volume `dataset_index` (the fourth axis).

    Raises:
        ValueError: If the file is not a MetaImage file, misses a required field, or uses
            an element type or a data layout that is not supported.
    """
    path = Path(path)
    header: dict[str, str] = {}
    with open(path, "rb") as file:
        for raw in iter(file.readline, b""):
            key, separator, value = raw.decode("utf-8", errors="replace").partition("=")
            if separator:
                header[key.strip()] = value.strip()
            if key.strip() == "ElementDataFile":
                break  # the last header field
        offset = file.tell()
    missing = [name for name in ("NDims", "DimSize", "ElementType", "ElementDataFile") if name not in header]  # fmt: skip
    if missing:
        raise ValueError(f"'{path}' is not a MetaImage file: its header has no '{missing[0]}'.")
    dimension = int(header["NDims"])
    sizes = [int(s) for s in header["DimSize"].split()]
    if dimension not in (3, 4):
        raise ValueError(f"'{path}': {dimension} dimensions; 3 or 4 are supported.")
    if len(sizes) != dimension:
        raise ValueError(f"'{path}': {len(sizes)} sizes for {dimension} dimensions.")
    if header["ElementType"] not in _TYPES:
        raise ValueError(f"'{path}': the element type '{header['ElementType']}' is not supported.")
    if header.get("ElementNumberOfChannels", "1") != "1":
        raise ValueError(f"'{path}': images with more than one channel are not supported.")
    big = _true(header, "BinaryDataByteOrderMSB", header.get("ElementByteOrderMSB", "False"))
    dtype = np.dtype(_TYPES[header["ElementType"]]).newbyteorder(">" if big else "<")
    values = _read_values(path, header, offset, dtype, math.prod(sizes))
    spacing = _numbers(header.get("ElementSpacing", header.get("ElementSize", "1 1 1")))
    origin = next((header[k] for k in ("Offset", "Origin", "Position") if k in header), "0 0 0")
    matrix = next((header[k] for k in ("TransformMatrix", "Rotation", "Orientation") if k in header), None)  # fmt: skip
    # The first `dimension` numbers are the direction of axis 0 (as ITK reads them)
    direction = (
        np.eye(3) if matrix is None else np.reshape(_numbers(matrix), (dimension, -1)).T[:3, :3]
    )
    o = _numbers(origin)
    return Image(
        array=_float64_volume(values, sizes, (0, 1, 2), dataset_index),
        spacing=(spacing[0], spacing[1], spacing[2]),
        origin=(o[0], o[1], o[2]),
        direction=direction,
        modality="MetaImage",
    )


def _read_values(
    path: Path, header: dict[str, str], offset: int, dtype: np.dtype[Any], count: int
) -> npt.NDArray[Any]:
    """The `count` values of the data, as stored (after the header, or in the data file)."""
    name = header["ElementDataFile"]
    source, start = path, offset
    if name != "LOCAL":
        if name.startswith("LIST") or len(name.split()) > 1:
            raise ValueError(f"'{path}': data in more than one file are not supported.")
        source, start = path.parent / name, max(int(header.get("HeaderSize", "0")), 0)
    with open(source, "rb") as file:
        file.seek(start)
        data = file.read()
    if _true(header, "CompressedData"):
        data = zlib.decompressobj().decompress(data)
    elif header.get("HeaderSize") == "-1":  # the data are the last bytes of the file
        data = data[-count * dtype.itemsize :]
    if not _true(header, "BinaryData", "True"):
        words = data.split()
        if len(words) < count:
            raise ValueError(f"'{path}': the data hold {len(words)} of {count} values.")
        return np.array(words[:count], dtype=np.float64)
    if len(data) < count * dtype.itemsize:
        raise ValueError(f"'{path}': the data hold {len(data)} of {count * dtype.itemsize} bytes.")
    return np.frombuffer(data, dtype=dtype, count=count)
