"""
DICOM Utility Functions.

This module provides shared utility functions for working with DICOM files,
including multi-phase series detection and splitting logic used by both
the DicomDatabase and the image loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pydicom


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
    "EchoNumber",
]


def split_dicom_phases(
    file_metadata: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split DICOM file metadata into multiple phases/groups.

    This function detects multi-phase DICOM series (e.g., cardiac phases,
    multi-echo, dynamic contrast) and splits them into separate groups.

    The detection strategy is:
    1. Check for distinctive DICOM tags (CardiacPhase, TemporalPosition, etc.)
       If a tag has >1 unique value, use it to group files.
    2. Fallback: Check for duplicate spatial positions (ImagePositionPatient).
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
    if len(file_metadata) < 2:
        return [file_metadata]

    # 1. Try splitting by multi-phase tags
    for tag in MULTI_PHASE_TAGS:
        values: dict[Any, list[dict[str, Any]]] = {}
        for meta in file_metadata:
            val = meta.get(tag)
            if val is not None:
                values.setdefault(val, []).append(meta)

        # If we have multiple groups and covered all files
        if len(values) > 1:
            total_grouped = sum(len(g) for g in values.values())
            if total_grouped == len(file_metadata):
                # Sort groups by tag value
                sorted_keys = sorted(values.keys())
                return [values[k] for k in sorted_keys]

    # 2. Fallback: Spatial duplication check
    pos_map: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for meta in file_metadata:
        pos = meta.get("ImagePositionPatient")
        if pos:
            pos_tuple = tuple(pos) if isinstance(pos, (list, tuple)) else pos
            pos_map.setdefault(pos_tuple, []).append(meta)

    # Check if we have duplicates (any position has >1 instance)
    if any(len(g) > 1 for g in pos_map.values()):
        num_phases = max(len(g) for g in pos_map.values())
        phase_groups: list[list[dict[str, Any]]] = [[] for _ in range(num_phases)]

        # Sort by instance number for consistency
        sorted_metadata = sorted(
            file_metadata,
            key=lambda x: x.get("InstanceNumber", 0) or 0,
        )

        # Re-map with sorted metadata
        pos_map_sorted: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
        for meta in sorted_metadata:
            pos = meta.get("ImagePositionPatient")
            if pos:
                pos_tuple = tuple(pos) if isinstance(pos, (list, tuple)) else pos
                pos_map_sorted.setdefault(pos_tuple, []).append(meta)
            else:
                phase_groups[0].append(meta)

        # Distribute duplicates across phases
        for _, metas in pos_map_sorted.items():
            for i, meta in enumerate(metas):
                if i < num_phases:
                    phase_groups[i].append(meta)
                else:
                    phase_groups[-1].append(meta)  # pragma: no cover

        # Filter empty groups
        return [g for g in phase_groups if g]

    return [file_metadata]


def get_dicom_phases(
    path: str,
    recursive: bool = False,
) -> list[DicomPhaseInfo]:
    """Discover phases in a DICOM series directory.

    Scans a directory for DICOM files and detects if the series contains
    multiple phases (e.g., cardiac phases, temporal positions, echo numbers).
    This is useful before calling ``load_image()`` with a specific ``dataset_index``.

    Multi-phase detection uses the same logic as :class:`DicomDatabase` to ensure
    consistent behavior across the library.

    Args:
        path: Path to directory containing DICOM files.
        recursive: If True, recursively searches subdirectories. Default False.

    Returns:
        List of :class:`DicomPhaseInfo` objects describing each detected phase.
        For single-phase series, returns a list with one element.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If no DICOM files are found.

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
            candidates = list(path_obj.rglob("*"))
        else:
            candidates = list(path_obj.iterdir())

        dicom_files = [
            f for f in candidates if f.is_file() and pydicom.misc.is_dicom(f)
        ]

    if not dicom_files:
        raise ValueError(f"No DICOM files found in: {path}")

    # Extract metadata for phase detection
    file_metadata: list[dict[str, Any]] = []
    for f in dicom_files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
            meta: dict[str, Any] = {
                "file_path": f,
                "InstanceNumber": getattr(dcm, "InstanceNumber", None),
            }

            # Extract position
            try:
                ipp = dcm.ImagePositionPatient
                meta["ImagePositionPatient"] = (
                    float(ipp[0]),
                    float(ipp[1]),
                    float(ipp[2]),
                )
            except (AttributeError, IndexError, TypeError):
                meta["ImagePositionPatient"] = None

            # Extract multi-phase tags
            for tag in MULTI_PHASE_TAGS:
                val = getattr(dcm, tag, None)
                if val is not None:
                    meta[tag] = val

            file_metadata.append(meta)
        except Exception:
            continue

    if not file_metadata:
        raise ValueError(f"Could not read any DICOM files from: {path}")

    # Split into phases
    phases = split_dicom_phases(file_metadata)

    # Determine which tag was used for splitting
    split_tag = None
    if len(phases) > 1:
        # Check which tag has different values across phases
        for tag in MULTI_PHASE_TAGS:
            first_val = phases[0][0].get(tag) if phases[0] else None
            if first_val is not None:
                # Check if other phases have different values
                for phase in phases[1:]:
                    if phase and phase[0].get(tag) != first_val:
                        split_tag = tag
                        break
            if split_tag:
                break
        if not split_tag:
            split_tag = "spatial"  # Fallback was used

    # Build DicomPhaseInfo objects
    result: list[DicomPhaseInfo] = []
    for i, phase_meta in enumerate(phases):
        file_paths = [m["file_path"] for m in phase_meta]
        split_value = phase_meta[0].get(split_tag) if split_tag and phase_meta else None

        # Generate label
        if split_tag == "NominalPercentageOfCardiacPhase":
            label = f"Phase {split_value}%" if split_value is not None else f"Phase {i}"
        elif split_tag == "TemporalPositionIdentifier":
            label = (
                f"Temporal {split_value}" if split_value is not None else f"Time {i}"
            )
        elif split_tag == "EchoNumber":
            label = f"Echo {split_value}" if split_value is not None else f"Echo {i}"
        elif split_tag == "AcquisitionNumber":
            label = (
                f"Acquisition {split_value}" if split_value is not None else f"Acq {i}"
            )
        elif split_tag == "TriggerTime":
            label = (
                f"Trigger {split_value}ms"
                if split_value is not None
                else f"Trigger {i}"
            )
        elif split_tag == "spatial":
            label = f"Volume {i + 1}"
        else:
            label = f"Dataset {i}"

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
