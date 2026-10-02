"""
DICOM Utility Functions.

This module provides shared utility functions for working with DICOM files,
including multi-phase series detection and splitting logic used by both
the DicomDatabase and the image loader.
"""

from __future__ import annotations

import multiprocessing
import os
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pydicom
from pydicom.multival import MultiValue


@dataclass
class DicomPhaseInfo:
    """Information about a detected phase in a DICOM series.

    Attributes:
        index: Zero-based index of this phase.
        num_slices: Number of slices/instances in this phase.
        file_paths: List of file paths belonging to this phase.
        label: Human-readable label (e.g., "Phase 0%", "Echo 1").
        split_tag: The DICOM tag used to detect this phase, or "spatial"
            if detected via duplicate positions.
        split_value: The value of the split tag for this phase.
    """

    index: int
    num_slices: int
    file_paths: list[Path]
    label: Optional[str] = None
    split_tag: Optional[str] = None
    split_value: Optional[Any] = None


# Priority list of DICOM tags used for multi-phase detection
MULTI_PHASE_TAGS = [
    "NominalPercentageOfCardiacPhase",
    "TemporalPositionIdentifier",
    "TriggerTime",
    "AcquisitionNumber",
    "EchoNumbers",
]


def split_dicom_phases(
    file_metadata: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split DICOM file metadata into multiple phases/groups.

    This function detects multi-phase DICOM series (e.g., cardiac phases,
    multi-echo, dynamic contrast) and splits them into separate groups.

    The detection strategy is:
    1. When every file has a position and no position repeats, the files are one
       volume. Tags such as AcquisitionNumber can change inside one volume (a
       step-and-shoot CT writes one per table step), so they do not split it.
    2. Check for distinctive DICOM tags (CardiacPhase, TemporalPosition, etc.)
       If a tag has >1 unique value, use it to group files. With known positions,
       the groups must also be possible phases: no more groups than the largest
       number of files at one position, and no position twice in one group (a
       trigger time that drifts from slice to slice fails this test).
    3. Fallback: Check for duplicate spatial positions (ImagePositionPatient).
       If duplicates exist, group by order of appearance.

    Args:
        file_metadata: List of dictionaries containing at minimum:
            - 'file_path': Path to the DICOM file
            - 'ImagePositionPatient': Optional tuple of (x, y, z)
            - Any of the MULTI_PHASE_TAGS (optional)

    Returns:
        List of lists, where each inner list contains metadata dicts
        for one phase. Single-phase series return [[all_metadata]].

    Example:
        Split DICOM metadata into separate phases:

        ```python
        from pictologics.utilities.dicom_utils import split_dicom_phases
        from pathlib import Path

        # Assume metadata list already collected
        metadata = [
            {'file_path': Path('slice1.dcm'), 'CardiacPhase': 0},
            {'file_path': Path('slice2.dcm'), 'CardiacPhase': 10},
            # ... more files
        ]
        phases = split_dicom_phases(metadata)
        print(f"Found {len(phases)} phases")
        ```
    """
    return _split_phases(file_metadata)[0]


def _position_key(position: Any) -> tuple[float, ...]:
    """A slice position as a hashable key."""
    return position if isinstance(position, tuple) else tuple(position)


def _split_phases(
    file_metadata: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], Optional[str]]:
    """The groups of `split_dicom_phases`, and the tag that split them ("spatial" for
    the duplicate-position fallback, None for one group)."""
    if len(file_metadata) < 2:
        return [file_metadata], None

    positions = [meta.get("ImagePositionPatient") for meta in file_metadata]
    known = all(position is not None for position in positions)
    repeats = 0
    if known:
        repeats = max(Counter(_position_key(p) for p in positions).values())
        if repeats == 1:
            return [file_metadata], None

    # 1. Try splitting by multi-phase tags
    for tag in MULTI_PHASE_TAGS:
        values: dict[Any, list[dict[str, Any]]] = {}
        for meta in file_metadata:
            val = meta.get(tag)
            if isinstance(val, (list, MultiValue)):  # a multi-valued tag, e.g. EchoNumbers 1\2
                val = tuple(val)
            if val is not None:
                values.setdefault(val, []).append(meta)

        # If we have multiple groups and covered all files
        if len(values) > 1:
            total_grouped = sum(len(g) for g in values.values())
            if total_grouped == len(file_metadata):
                # Sort groups by tag value
                groups = [values[k] for k in sorted(values.keys())]
                if not known or (
                    len(groups) <= repeats
                    and all(
                        len({_position_key(m["ImagePositionPatient"]) for m in g}) == len(g)
                        for g in groups
                    )
                ):
                    return groups, tag

    # 2. Fallback: Spatial duplication check
    pos_map: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for meta in file_metadata:
        pos = meta.get("ImagePositionPatient")
        if pos:
            pos_map.setdefault(_position_key(pos), []).append(meta)

    # Check if we have duplicates (any position has >1 instance)
    if any(len(g) > 1 for g in pos_map.values()):
        num_phases = max(len(g) for g in pos_map.values())
        phase_groups: list[list[dict[str, Any]]] = [[] for _ in range(num_phases)]

        # The k-th instance (by instance number) at each position goes to phase k;
        # files without a position go to the first phase.
        sorted_metadata = sorted(
            file_metadata,
            key=lambda x: x.get("InstanceNumber", 0) or 0,
        )
        seen: Counter[tuple[float, ...]] = Counter()
        for meta in sorted_metadata:
            pos = meta.get("ImagePositionPatient")
            if pos:
                key = _position_key(pos)
                phase_groups[seen[key]].append(meta)
                seen[key] += 1
            else:
                phase_groups[0].append(meta)

        return [g for g in phase_groups if g], "spatial"

    return [file_metadata], None


# Objects with pixel data that are not slices of the image series: a segmentation (load
# it with load_seg) and an RT dose grid.
_NON_IMAGE_SOP_CLASSES = frozenset(
    {
        "1.2.840.10008.5.1.4.1.1.66.4",  # Segmentation Storage
        "1.2.840.10008.5.1.4.1.1.66.7",  # Label Map Segmentation Storage
        "1.2.840.10008.5.1.4.1.1.481.2",  # RT Dose Storage
    }
)
_PIXEL_DATA_KEYWORDS = ("PixelData", "FloatPixelData", "DoubleFloatPixelData")

# Header reads defer values above this size (the pixel data) until they are used.
_DEFER_SIZE = "4 KB"


def _raw_value(dcm: Any, keyword: str) -> Any:
    """The value of an element as the file stores it: bytes while pydicom has not
    converted the element (no conversion cost), else its value; None when absent."""
    return getattr(dcm.get_item(keyword), "value", None)


def _text(value: Any) -> Optional[str]:
    """A stored text value (bytes or str) without its padding; None for other values."""
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace")
    return value.strip("\x00 ") if isinstance(value, str) else None


def _is_image_instance(dcm: Any) -> bool:
    """True for an instance with image pixel data (the `in` test reads no pixel data).

    RTSTRUCT, RTPLAN, SR and presentation states have no pixel data, and a SEG or an RT
    dose grid in the folder of a scan is not one of its slices.
    """
    if _text(_raw_value(dcm, "SOPClassUID")) in _NON_IMAGE_SOP_CLASSES:
        return False
    return any(keyword in dcm for keyword in _PIXEL_DATA_KEYWORDS)


def _series_metadata(dcm: Any, file_path: Any) -> dict[str, Any]:
    """The tags that group the instances of a folder into series and phases.

    The series and layout tags stay as stored (no conversion): they are only compared,
    and decoded for an error message.
    """
    meta: dict[str, Any] = {
        "file_path": file_path,
        "InstanceNumber": getattr(dcm, "InstanceNumber", None),
    }
    try:
        ipp = dcm.ImagePositionPatient
        meta["ImagePositionPatient"] = (float(ipp[0]), float(ipp[1]), float(ipp[2]))
    except (AttributeError, IndexError, TypeError):
        meta["ImagePositionPatient"] = None
    for tag in MULTI_PHASE_TAGS:
        val = getattr(dcm, tag, None)
        if val is not None:
            meta[tag] = val
    meta["SeriesInstanceUID"] = _text(_raw_value(dcm, "SeriesInstanceUID"))
    meta["layout"] = tuple(
        _stored(_raw_value(dcm, tag)) for tag in ("ImageOrientationPatient", "Rows", "Columns")
    )
    return meta


def _stored(value: Any) -> Any:
    """A stored or converted tag value to compare; None for anything else."""
    return value if isinstance(value, (bytes, str, int, float, list, tuple)) else None


def _orientation(value: Any) -> Optional[tuple[float, ...]]:
    """The six direction cosines of a stored or converted ImageOrientationPatient."""
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace").split("\\")
    try:
        numbers = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return numbers if len(numbers) == 6 else None


def _same_layout(a: tuple[Any, Any, Any], b: tuple[Any, Any, Any]) -> bool:
    """Equal image size and (within 1e-3) equal orientation."""
    if a[1:] != b[1:]:
        return False
    if a[0] == b[0]:  # the same stored text
        return True
    x, y = _orientation(a[0]), _orientation(b[0])
    if x is None or y is None:
        return x is None and y is None
    return max(abs(p - q) for p, q in zip(x, y, strict=True)) < 1e-3


def _series_list(by_series: dict[Optional[str], list[dict[str, Any]]]) -> str:
    """One line per series: UID, number, modality, description and file count.

    For an error message only, so the header of the first file is read again here.
    """
    lines = []
    for uid, metas in sorted(by_series.items(), key=lambda item: -len(item[1])):
        try:
            first = pydicom.dcmread(metas[0]["file_path"], stop_before_pixels=True)
        except Exception:
            first = None
        number, modality, description = (
            getattr(first, tag, None) for tag in ("SeriesNumber", "Modality", "SeriesDescription")
        )
        lines.append(
            f"  {uid}: series {number}, {modality}, '{description or ''}', {len(metas)} files"
        )
    return "\n".join(lines)


def _select_image_series(
    file_metadata: list[dict[str, Any]], series_uid: Optional[str], source: Any
) -> list[dict[str, Any]]:
    """The instances of one image series in `file_metadata` (from `_series_metadata`).

    Two series in one folder (two reconstructions of a scan, a scout) must not mix into
    one image, so without `series_uid` that is an error that names them. In the chosen
    series, instances of another orientation or image size are left out, with a warning.
    """
    by_series: dict[Optional[str], list[dict[str, Any]]] = {}
    for meta in file_metadata:
        by_series.setdefault(meta["SeriesInstanceUID"], []).append(meta)
    if series_uid is not None:
        if series_uid not in by_series:
            raise ValueError(
                f"Series {series_uid} is not in {source}. Its image series:\n"
                f"{_series_list(by_series)}"
            )
        chosen = by_series[series_uid]
    elif len(by_series) > 1:
        raise ValueError(
            f"{source} holds {len(by_series)} image series. Load one with series_uid=..., "
            f"or put each series in its own folder:\n{_series_list(by_series)}"
        )
    else:
        (chosen,) = by_series.values()

    groups: list[list[dict[str, Any]]] = []
    for meta in chosen:
        for group in groups:
            if _same_layout(group[0]["layout"], meta["layout"]):
                group.append(meta)
                break
        else:
            groups.append([meta])
    if len(groups) > 1:
        largest = max(groups, key=len)
        warnings.warn(
            f"Left out {len(chosen) - len(largest)} of the {len(chosen)} images of {source}: "
            "their orientation or image size differs from that of the other images "
            "(for example a scout).",
            UserWarning,
            stacklevel=4,
        )
        chosen = largest
    return chosen


def _split_tag(phases: list[list[dict[str, Any]]]) -> Optional[str]:
    """The tag that is constant in each phase and different between the phases."""
    if len(phases) < 2:
        return None
    for tag in MULTI_PHASE_TAGS:
        firsts = []
        for group in phases:
            values = {
                tuple(v) if isinstance(v, (list, MultiValue)) else v
                for v in (meta.get(tag) for meta in group)
            }
            if len(values) != 1 or None in values:
                break
            firsts.append(values.pop())
        else:
            if len(set(firsts)) == len(phases):
                return tag
    return "spatial"


# Labels of the phases that a tag splits (the tag value in braces)
_PHASE_LABELS = {
    "NominalPercentageOfCardiacPhase": "Phase {}%",
    "TemporalPositionIdentifier": "Temporal {}",
    "TriggerTime": "Trigger {}ms",
    "AcquisitionNumber": "Acquisition {}",
    "EchoNumbers": "Echo {}",
}


def get_dicom_phases(
    path: str | Path,
    recursive: bool = False,
    series_uid: Optional[str] = None,
) -> list[DicomPhaseInfo]:
    """Discover phases in a DICOM series directory.

    Scans a directory for DICOM files and detects if the series contains
    multiple phases (e.g., cardiac phases, temporal positions, echo numbers).
    This is useful before calling ``load_image()`` with a specific ``dataset_index``.

    Multi-phase detection uses the same logic as :class:`DicomDatabase` to ensure
    consistent behavior across the library.

    Args:
        path: Path to directory containing DICOM files.
        recursive: If True, searches the subdirectories and uses the folder with the
            most DICOM files, as ``load_image(recursive=True)`` does. Default False.
        series_uid: The SeriesInstanceUID of the series to use when the folder
            holds more than one image series. Default None.

    Returns:
        List of :class:`DicomPhaseInfo` objects describing each detected phase.
        For single-phase series, returns a list with one element.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If no DICOM files are found, or if the folder holds more than one
            image series and ``series_uid`` does not choose one.

    Example:
        Discover phases before loading:

        ```python
        from pictologics.utilities import get_dicom_phases
        from pictologics import load_image

        # Discover phases in a cardiac CT directory
        phases = get_dicom_phases("cardiac_ct/")
        print(f"Found {len(phases)} phases:")
        for phase in phases:
            print(f"  Phase {phase.index}: {phase.num_slices} slices - {phase.label}")

        # Load the 5th phase (40%)
        img = load_image("cardiac_ct/", dataset_index=4)

        # Check if series is multi-phase
        if len(phases) > 1:
            print("Multi-phase series detected!")
        else:
            print("Single-phase series")
        ```

    See Also:
        - :func:`load_image`: Main image loading function with ``dataset_index`` support.
        - :class:`DicomDatabase`: Full DICOM database parsing with automatic phase splitting.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Collect DICOM files
    if path_obj.is_file():
        if pydicom.misc.is_dicom(path_obj):
            dicom_files = [path_obj]
        else:
            raise ValueError(f"File is not a DICOM file: {path}")
    else:
        if recursive:
            from pictologics.loader import _find_best_dicom_series_dir

            path_obj = _find_best_dicom_series_dir(path_obj)
        dicom_files = [f for f in path_obj.iterdir() if f.is_file() and pydicom.misc.is_dicom(f)]

    if not dicom_files:
        raise ValueError(f"No DICOM files found in: {path}")

    # The same image series and phases as load_image finds
    file_metadata: list[dict[str, Any]] = []
    for f in dicom_files:
        try:
            dcm = pydicom.dcmread(f, defer_size=_DEFER_SIZE)
        except Exception:
            continue
        if _is_image_instance(dcm):
            file_metadata.append(_series_metadata(dcm, f))

    if not file_metadata:
        raise ValueError(f"Could not read any DICOM files with image data from: {path}")

    phases = split_dicom_phases(_select_image_series(file_metadata, series_uid, path))
    split_tag = _split_tag(phases)

    # Build DicomPhaseInfo objects
    result: list[DicomPhaseInfo] = []
    for i, phase_meta in enumerate(phases):
        file_paths = [m["file_path"] for m in phase_meta]
        split_value = phase_meta[0].get(split_tag) if split_tag and phase_meta else None

        if split_tag is None:
            label = f"Dataset {i}"
        elif split_tag == "spatial":
            label = f"Volume {i + 1}"
        else:
            label = _PHASE_LABELS[split_tag].format(split_value)

        result.append(
            DicomPhaseInfo(
                index=i,
                num_slices=len(file_paths),
                file_paths=file_paths,
                label=label,
                split_tag=split_tag,
                split_value=split_value,
            )
        )

    return result


@contextmanager
def worker_pool(
    num_workers: int,
    initializer: Optional[Callable[..., None]] = None,
    initargs: tuple[Any, ...] = (),
    spawn: bool = False,
) -> Iterator[ProcessPoolExecutor]:
    """A process pool whose workers skip the JIT warm-up at import.

    Each spawned worker imports pictologics again, which would compile or load every
    numba kernel, although a worker that reads DICOM headers needs none, and a worker of
    `RadiomicsPipeline.run_batch` loads only the kernels it uses from the numba cache. The
    pool starts its workers lazily, so the setting stays in place until the pool closes.
    A forkserver (the Linux default from Python 3.14) keeps the environment of its first
    start for all later pools, so the pool then uses spawn, which reads the setting at
    each start. `spawn` uses spawn on every platform: a forked child of a process that
    ran numba kernels in threads can hang.
    """
    method = "spawn" if spawn else multiprocessing.get_start_method()
    context = multiprocessing.get_context("spawn" if method == "forkserver" else method)
    previous = os.environ.get("PICTOLOGICS_DISABLE_WARMUP")
    os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"
    try:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=context,
            initializer=initializer,
            initargs=initargs,
        ) as executor:
            yield executor
    finally:
        if previous is None:
            del os.environ["PICTOLOGICS_DISABLE_WARMUP"]
        else:
            os.environ["PICTOLOGICS_DISABLE_WARMUP"] = previous
