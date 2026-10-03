"""
NRRD Loader
===========

Reads NRRD files into pictologics Image instances, without other packages: ``.nrrd``
files, ``.nhdr`` headers with a detached data file, and 3D Slicer ``.seg.nrrd``
segmentations (their segments are listed by ``get_segment_info``).

The geometry is converted to the LPS+ world frame of the package from the named NRRD
spaces RAS, LAS and LPS (a space without a name is taken as it is).
"""

from __future__ import annotations

import bz2
import gzip
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy import typing as npt

from pictologics.loader import Image, _float64_volume

# The type names of the NRRD format and their numpy types
_TYPES = {
    name: code
    for code, names in {
        "i1": ("signed char", "int8", "int8_t"),
        "u1": ("uchar", "unsigned char", "uint8", "uint8_t"),
        "i2": ("short", "short int", "signed short", "signed short int", "int16", "int16_t"),
        "u2": ("ushort", "unsigned short", "unsigned short int", "uint16", "uint16_t"),
        "i4": ("int", "signed int", "int32", "int32_t"),
        "u4": ("uint", "unsigned int", "uint32", "uint32_t"),
        "i8": (
            "longlong",
            "long long",
            "long long int",
            "signed long long",
            "signed long long int",
            "int64",
            "int64_t",
        ),  # fmt: skip
        "u8": ("ulonglong", "unsigned long long", "unsigned long long int", "uint64", "uint64_t"),
        "f4": ("float",),
        "f8": ("double",),
    }.items()
    for name in names
}

# The signs that take the world axes of a named NRRD space to LPS+
_SPACES = {
    "left-posterior-superior": (1.0, 1.0, 1.0),
    "lps": (1.0, 1.0, 1.0),
    "right-anterior-superior": (-1.0, -1.0, 1.0),
    "ras": (-1.0, -1.0, 1.0),
    "left-anterior-superior": (1.0, -1.0, 1.0),
    "las": (1.0, -1.0, 1.0),
}

_KEY_VALUE = re.compile(r"([^:]+):=(.*)")
_VECTOR = re.compile(r"none|\(([^)]*)\)")


def _read_header(path: Path) -> tuple[dict[str, str], dict[str, str], int]:
    """The fields (names in lower case, without spaces), the key/value pairs, and the byte
    offset where attached data start."""
    fields: dict[str, str] = {}
    pairs: dict[str, str] = {}
    with open(path, "rb") as file:
        if not file.readline().startswith(b"NRRD000"):
            raise ValueError(f"'{path}' is not a NRRD file: it does not start with 'NRRD000'.")
        for raw in iter(file.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                break  # the blank line before attached data
            if line.startswith("#"):
                continue
            pair = _KEY_VALUE.match(line)
            if pair:
                pairs[pair[1]] = pair[2]
                continue
            field, separator, value = line.partition(":")
            if not separator:
                raise ValueError(f"'{path}': the header line '{line}' is not 'field: value'.")
            fields[field.replace(" ", "").lower()] = value.strip()
        return fields, pairs, file.tell()


def _vector(text: str) -> npt.NDArray[np.float64]:
    """The numbers of a NRRD vector, '(x,y,z)'."""
    return np.array([float(v) for v in text.strip().strip("()").split(",")])


def _read_values(path: Path, fields: dict[str, str], offset: int, count: int) -> npt.NDArray[Any]:
    """The `count` values of the data, as stored (attached at `offset`, or in the data
    file of the header)."""
    try:
        dtype = np.dtype(_TYPES[fields["type"].lower()])
    except KeyError:
        raise ValueError(f"'{path}': the NRRD type '{fields['type']}' is not supported.") from None
    dtype = dtype.newbyteorder(">" if fields.get("endian", "little").lower() == "big" else "<")
    source, start = path, offset
    if "datafile" in fields:
        name = fields["datafile"]
        if name.startswith("LIST") or len(name.split()) > 1:
            raise ValueError(f"'{path}': data in more than one file are not supported.")
        source, start = path.parent / name, 0
    with open(source, "rb") as file:
        file.seek(start)
        for _ in range(int(fields.get("lineskip", 0))):
            file.readline()
        data = file.read()
    encoding = fields["encoding"].lower()
    skip = int(fields.get("byteskip", 0))
    if encoding in ("txt", "text", "ascii"):
        words = data.split()
        if len(words) < count:
            raise ValueError(f"'{path}': the data hold {len(words)} of {count} values.")
        return np.array(words[:count], dtype=np.float64)
    if encoding == "raw":
        data = data[-count * dtype.itemsize :] if skip == -1 else data[skip:]
    elif encoding in ("gzip", "gz"):
        data = gzip.decompress(data)[skip:]
    elif encoding in ("bzip2", "bz2"):
        data = bz2.decompress(data)[skip:]
    else:
        raise ValueError(f"'{path}': the NRRD encoding '{fields['encoding']}' is not supported.")
    if len(data) < count * dtype.itemsize:
        raise ValueError(f"'{path}': the data hold {len(data)} of {count * dtype.itemsize} bytes.")
    return np.frombuffer(data, dtype=dtype, count=count)


def _load_nrrd(path: str | Path, dataset_index: int = 0) -> Image:
    """Load a NRRD file (see the module docstring) as an Image.

    A file with one axis that is not spatial (a list of volumes, as the layers of a
    .seg.nrrd file with overlapping segments) gives its volume `dataset_index`.

    Raises:
        ValueError: If the file is not a NRRD file, misses a required field, or uses a
            type, an encoding, a space or a data layout that is not supported.
    """
    path = Path(path)
    fields, _, offset = _read_header(path)
    missing = [name for name in ("dimension", "sizes", "type", "encoding") if name not in fields]
    if missing:
        raise ValueError(f"'{path}': the NRRD header has no '{missing[0]}' field.")
    dimension = int(fields["dimension"])
    sizes = [int(s) for s in fields["sizes"].split()]
    if len(sizes) != dimension:
        raise ValueError(f"'{path}': {len(sizes)} sizes for dimension {dimension}.")
    space = fields.get("space", "").lower()
    if space and space not in _SPACES:
        raise ValueError(f"'{path}': the NRRD space '{fields['space']}' is not supported.")
    signs = np.array(_SPACES.get(space, (1.0, 1.0, 1.0)))
    if "spacedirections" in fields:
        vectors = [m[1] for m in _VECTOR.finditer(fields["spacedirections"])]
        spatial = [axis for axis, v in enumerate(vectors) if v is not None]
        if len(spatial) == 4 == dimension:  # a 4D space (as ITK writes): axis 3 lists volumes
            spatial = spatial[:3]
        if len(vectors) != dimension or len(spatial) != 3:
            raise ValueError(f"'{path}': the image needs 3 spatial axes in 'space directions'.")
        matrix = np.column_stack([_vector(vectors[a])[:3] for a in spatial]) * signs[:, None]
        spacing = np.linalg.norm(matrix, axis=0)
        direction = matrix / spacing
        origin = signs * _vector(fields.get("spaceorigin", "(0,0,0)"))[:3]
    else:
        # No space: the axes of a spatial kind (all, without kinds), and their "spacings"
        kinds = fields.get("kinds", "domain " * dimension).lower().split()
        spatial = [axis for axis, kind in enumerate(kinds) if kind in ("domain", "space")]
        if len(spatial) != 3:
            raise ValueError(f"'{path}': the image needs 3 spatial axes, it has {len(spatial)}.")
        given = fields.get("spacings", "nan " * dimension).split()
        spacing = np.array([float(given[a]) if given[a] != "nan" else 1.0 for a in spatial])
        direction, origin = np.eye(3), np.zeros(3)
    values = _read_values(path, fields, offset, math.prod(sizes))
    return Image(
        array=_float64_volume(values, sizes, spatial, dataset_index),
        spacing=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        direction=direction,
        modality="Nrrd",
    )


def _nrrd_segment_info(path: str | Path) -> list[dict[str, str | int]]:
    """The segments of a 3D Slicer .seg.nrrd file (see get_segment_info)."""
    _, pairs, _ = _read_header(Path(path))
    indices = sorted({int(m[1]) for key in pairs if (m := re.match(r"Segment(\d+)_", key))})
    if not indices:
        raise ValueError(f"'{path}' holds no segments (no 'Segment0_' fields).")
    return [
        {
            "segment_index": n,
            "segment_label": pairs.get(f"Segment{n}_Name", ""),
            "segment_id": pairs.get(f"Segment{n}_ID", ""),
            "label_value": int(pairs.get(f"Segment{n}_LabelValue", "1")),
            "layer": int(pairs.get(f"Segment{n}_Layer", str(n))),
        }
        for n in indices
    ]
