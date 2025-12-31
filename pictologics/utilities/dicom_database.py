"""
DICOM Database Module
=====================

This module provides dataclass-based hierarchical organization of DICOM files
with completeness validation and multi-level DataFrame exports.

The implementation supports parallel processing for improved performance on
large datasets, with stateless file processing and immutable intermediate results.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

# ============================================================================
# Dataclass Definitions
# ============================================================================


@dataclass
class DicomInstance:
    """Represents a single DICOM file/instance.

    Attributes:
        sop_instance_uid: Unique identifier for this instance.
        file_path: Absolute path to the DICOM file.
        instance_number: Instance number within the series.
        image_position_patient: (x, y, z) position in patient coordinates.
        image_orientation_patient: Direction cosines for row and column.
        slice_location: Slice location value from DICOM header.
        acquisition_datetime: Combined acquisition date and time.
        projection_score: Calculated projection onto slice normal for sorting.
        metadata: Additional extracted metadata tags.
    """

    sop_instance_uid: str
    file_path: Path
    instance_number: Optional[int] = None
    image_position_patient: Optional[tuple[float, float, float]] = None
    image_orientation_patient: Optional[tuple[float, ...]] = None
    slice_location: Optional[float] = None
    acquisition_datetime: Optional[str] = None
    projection_score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class DicomSeries:
    """Represents a DICOM series containing multiple instances.

    Attributes:
        series_instance_uid: Unique identifier for this series.
        series_number: Series number within the study.
        series_description: Description of the series.
        modality: Imaging modality (CT, MR, etc.).
        frame_of_reference_uid: Frame of reference UID.
        instances: List of DicomInstance objects in this series.
        common_metadata: Metadata tags identical across all instances.
    """

    series_instance_uid: str
    series_number: Optional[int] = None
    series_description: Optional[str] = None
    modality: Optional[str] = None
    frame_of_reference_uid: Optional[str] = None
    instances: list[DicomInstance] = field(default_factory=list)
    common_metadata: dict[str, Any] = field(default_factory=dict)

    def get_sorted_instances(self) -> list[DicomInstance]:
        """Return instances sorted by spatial position (projection score).

        Uses the same methodology as pictologics.loader for spatial sorting.
        Falls back to instance number if projection scores are not available.
        """
        if all(inst.projection_score is not None for inst in self.instances):
            return sorted(self.instances, key=lambda x: x.projection_score or 0)
        return sorted(
            self.instances, key=lambda x: x.instance_number if x.instance_number else 0
        )

    def check_completeness(self, spacing_tolerance: float = 0.1) -> dict[str, Any]:
        """Check if the series has all expected slices.

        Uses geometric validation based on ImagePositionPatient projection
        to detect missing slices and gaps.

        Args:
            spacing_tolerance: Tolerance for gap detection (default 10%).

        Returns:
            Dictionary with completeness information.
        """
        result: dict[str, Any] = {
            "series_uid": self.series_instance_uid,
            "total_instances": len(self.instances),
            "expected_instances": len(self.instances),
            "is_complete": True,
            "has_gaps": False,
            "gap_indices": [],
            "gap_positions": [],
            "spacing_mm": None,
            "spacing_std": None,
            "spacing_uniform": True,
            "first_slice_position": None,
            "last_slice_position": None,
            "frame_of_reference_uid": self.frame_of_reference_uid,
            "warnings": [],
        }

        if len(self.instances) < 2:
            result["warnings"].append("Series has fewer than 2 instances")
            return result

        # Get sorted instances by projection score
        sorted_instances = self.get_sorted_instances()

        # Check if we have projection scores for geometric validation
        projection_scores = [
            inst.projection_score
            for inst in sorted_instances
            if inst.projection_score is not None
        ]

        if len(projection_scores) < 2:
            result["warnings"].append(
                "Insufficient spatial data for geometric completeness check"
            )
            # Fall back to instance number validation
            instance_numbers = [
                inst.instance_number
                for inst in sorted_instances
                if inst.instance_number is not None
            ]
            if len(instance_numbers) >= 2:
                instance_numbers_sorted = sorted(instance_numbers)
                expected_range = set(
                    range(instance_numbers_sorted[0], instance_numbers_sorted[-1] + 1)
                )
                missing = expected_range - set(instance_numbers)
                if missing:
                    result["is_complete"] = False
                    result["has_gaps"] = True
                    result["gap_indices"] = sorted(missing)
                    result["expected_instances"] = len(expected_range)
            return result

        # Calculate spacings between consecutive slices
        spacings = np.diff(projection_scores)
        median_spacing = float(np.median(np.abs(spacings)))
        spacing_std = float(np.std(np.abs(spacings)))

        result["spacing_mm"] = median_spacing
        result["spacing_std"] = spacing_std
        result["first_slice_position"] = projection_scores[0]
        result["last_slice_position"] = projection_scores[-1]

        # Check for uniform spacing
        if median_spacing > 0:
            spacing_cv = spacing_std / median_spacing  # Coefficient of variation
            result["spacing_uniform"] = spacing_cv < spacing_tolerance

        # Detect gaps (spacing significantly larger than expected)
        gap_threshold = median_spacing * (1 + spacing_tolerance)
        gap_indices = []
        gap_positions = []

        for i, spacing in enumerate(np.abs(spacings)):
            if spacing > gap_threshold * 1.5:  # Gap detected
                gap_indices.append(i + 1)
                gap_positions.append(projection_scores[i])

        if gap_indices:
            result["has_gaps"] = True
            result["gap_indices"] = gap_indices
            result["gap_positions"] = gap_positions

            # Estimate expected instances based on position range and median spacing
            position_range = abs(projection_scores[-1] - projection_scores[0])
            if median_spacing > 0:
                expected = int(round(position_range / median_spacing)) + 1
                result["expected_instances"] = expected
                result["is_complete"] = False

        return result

    def get_instance_uids(self) -> list[str]:
        """Get list of all instance SOPInstanceUIDs."""
        return [inst.sop_instance_uid for inst in self.instances]

    def get_file_paths(self) -> list[str]:
        """Get list of all instance file paths as strings."""
        return [str(inst.file_path) for inst in self.instances]


@dataclass
class DicomStudy:
    """Represents a DICOM study containing multiple series.

    Attributes:
        study_instance_uid: Unique identifier for this study.
        study_date: Date of the study.
        study_time: Time of the study.
        study_description: Description of the study.
        series: List of DicomSeries objects in this study.
        common_metadata: Metadata tags identical across all series.
    """

    study_instance_uid: str
    study_date: Optional[str] = None
    study_time: Optional[str] = None
    study_description: Optional[str] = None
    series: list[DicomSeries] = field(default_factory=list)
    common_metadata: dict[str, Any] = field(default_factory=dict)

    def get_instance_uids(self) -> list[str]:
        """Get list of all instance SOPInstanceUIDs in this study."""
        uids = []
        for s in self.series:
            uids.extend(s.get_instance_uids())
        return uids

    def get_file_paths(self) -> list[str]:
        """Get list of all instance file paths in this study."""
        paths = []
        for s in self.series:
            paths.extend(s.get_file_paths())
        return paths


@dataclass
class DicomPatient:
    """Represents a DICOM patient containing multiple studies.

    Attributes:
        patient_id: Patient identifier.
        patients_name: Patient's name.
        patients_birth_date: Patient's birth date.
        patients_sex: Patient's sex.
        studies: List of DicomStudy objects for this patient.
        common_metadata: Metadata tags identical across all studies.
    """

    patient_id: str
    patients_name: Optional[str] = None
    patients_birth_date: Optional[str] = None
    patients_sex: Optional[str] = None
    studies: list[DicomStudy] = field(default_factory=list)
    common_metadata: dict[str, Any] = field(default_factory=dict)

    def get_instance_uids(self) -> list[str]:
        """Get list of all instance SOPInstanceUIDs for this patient."""
        uids = []
        for study in self.studies:
            uids.extend(study.get_instance_uids())
        return uids

    def get_file_paths(self) -> list[str]:
        """Get list of all instance file paths for this patient."""
        paths = []
        for study in self.studies:
            paths.extend(study.get_file_paths())
        return paths


@dataclass
class DicomDatabase:
    """Top-level database containing all patients.

    This class provides the main interface for building a DICOM database
    from folders and exporting to various formats.

    Attributes:
        patients: List of DicomPatient objects.
        spacing_tolerance: Tolerance for gap detection in completeness checks.
    """

    patients: list[DicomPatient] = field(default_factory=list)
    spacing_tolerance: float = 0.1

    @classmethod
    def from_folders(
        cls,
        paths: list[str | Path],
        recursive: bool = True,
        spacing_tolerance: float = 0.1,
        show_progress: bool = True,
        extract_private_tags: bool = True,
        num_workers: Optional[int] = None,
        split_multiseries: bool = True,
    ) -> "DicomDatabase":
        """Build a database from folder paths.

        Args:
            paths: List of folder paths to scan.
            recursive: Whether to scan subdirectories.
            spacing_tolerance: Tolerance for gap detection (default 10%).
            show_progress: Whether to display progress bars.
            extract_private_tags: Whether to extract vendor-specific private tags.
            num_workers: Number of parallel workers. None=auto (cpu_count-1),
                        1=sequential (no multiprocessing).
            split_multiseries: Whether to split multi-phase series (e.g. cardiac)
                              into separate series based on tags or spatial duplicates.

        Returns:
            DicomDatabase instance populated with all discovered DICOM files.
        """
        # Convert paths to Path objects
        path_objs = [Path(p) for p in paths]

        # Determine number of workers
        workers = _get_num_workers(num_workers)

        # Step 1: Discover all DICOM files
        dicom_files = _scan_dicom_files(path_objs, recursive, show_progress, workers)

        if not dicom_files:
            return cls(patients=[], spacing_tolerance=spacing_tolerance)

        # Step 2: Extract metadata from each file (parallel if workers > 1)
        file_metadata = _extract_all_metadata(
            dicom_files, show_progress, extract_private_tags, workers
        )

        # Step 3: Build hierarchy from flat metadata list
        patients = _build_hierarchy(file_metadata, spacing_tolerance, split_multiseries)

        # Step 4: Sort the hierarchy for consistent output
        patients = _sort_hierarchy(patients)

        return cls(patients=patients, spacing_tolerance=spacing_tolerance)

    # ========================================================================
    # DataFrame Export Methods
    # ========================================================================

    def get_patients_df(self) -> pd.DataFrame:
        """Export patient-level summary DataFrame.

        Includes all instance UIDs and file paths for each patient,
        plus aggregated statistics and common child-level metadata.
        """
        rows = []
        for patient in self.patients:
            row = {
                "PatientID": patient.patient_id,
                "PatientsName": patient.patients_name,
                "PatientsBirthDate": patient.patients_birth_date,
                "PatientsSex": patient.patients_sex,
                "NumStudies": len(patient.studies),
                "NumSeries": sum(len(study.series) for study in patient.studies),
                "NumInstances": sum(
                    len(series.instances)
                    for study in patient.studies
                    for series in study.series
                ),
                "InstanceSOPUIDs": patient.get_instance_uids(),
                "InstanceFilePaths": patient.get_file_paths(),
            }

            # Add study date range
            study_dates = [
                study.study_date for study in patient.studies if study.study_date
            ]
            if study_dates:
                row["EarliestStudyDate"] = min(study_dates)
                row["LatestStudyDate"] = max(study_dates)
            else:
                row["EarliestStudyDate"] = None
                row["LatestStudyDate"] = None

            # Add common metadata from patient level
            for key, value in patient.common_metadata.items():
                if key not in row:
                    row[key] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def get_studies_df(self, patient_id: Optional[str] = None) -> pd.DataFrame:
        """Export study-level summary DataFrame.

        Args:
            patient_id: Optional filter by patient ID.

        Returns:
            DataFrame with study information including instance lists.
        """
        rows = []
        for patient in self.patients:
            if patient_id and patient.patient_id != patient_id:
                continue

            for study in patient.studies:
                row = {
                    # Patient info
                    "PatientID": patient.patient_id,
                    "PatientsName": patient.patients_name,
                    "PatientsBirthDate": patient.patients_birth_date,
                    "PatientsSex": patient.patients_sex,
                    # Study info
                    "StudyInstanceUID": study.study_instance_uid,
                    "StudyDate": study.study_date,
                    "StudyTime": study.study_time,
                    "StudyDescription": study.study_description,
                    "NumSeries": len(study.series),
                    "NumInstances": sum(
                        len(series.instances) for series in study.series
                    ),
                    "InstanceSOPUIDs": study.get_instance_uids(),
                    "InstanceFilePaths": study.get_file_paths(),
                }

                # Collect modalities present
                modalities = list(set(s.modality for s in study.series if s.modality))
                row["ModalitiesPresent"] = modalities

                # Add common metadata
                for key, value in patient.common_metadata.items():
                    if key not in row:
                        row[key] = value
                for key, value in study.common_metadata.items():
                    if key not in row:
                        row[key] = value

                rows.append(row)

        return pd.DataFrame(rows)

    def get_series_df(
        self,
        patient_id: Optional[str] = None,
        study_uid: Optional[str] = None,
    ) -> pd.DataFrame:
        """Export series-level summary DataFrame with completeness info.

        Args:
            patient_id: Optional filter by patient ID.
            study_uid: Optional filter by study UID.

        Returns:
            DataFrame with series information including completeness validation.
        """
        rows = []
        for patient in self.patients:
            if patient_id and patient.patient_id != patient_id:
                continue

            for study in patient.studies:
                if study_uid and study.study_instance_uid != study_uid:
                    continue

                for series in study.series:
                    completeness = series.check_completeness(self.spacing_tolerance)

                    row = {
                        # Patient info
                        "PatientID": patient.patient_id,
                        "PatientsName": patient.patients_name,
                        # Study info
                        "StudyInstanceUID": study.study_instance_uid,
                        "StudyDate": study.study_date,
                        "StudyDescription": study.study_description,
                        # Series info
                        "SeriesInstanceUID": series.series_instance_uid,
                        "SeriesNumber": series.series_number,
                        "SeriesDescription": series.series_description,
                        "Modality": series.modality,
                        "FrameOfReferenceUID": series.frame_of_reference_uid,
                        # Completeness
                        "NumInstances": completeness["total_instances"],
                        "ExpectedInstances": completeness["expected_instances"],
                        "IsComplete": completeness["is_complete"],
                        "HasGaps": completeness["has_gaps"],
                        "GapIndices": completeness["gap_indices"],
                        "SpacingMM": completeness["spacing_mm"],
                        "SpacingUniform": completeness["spacing_uniform"],
                        "FirstSlicePosition": completeness["first_slice_position"],
                        "LastSlicePosition": completeness["last_slice_position"],
                        "CompletenessWarnings": completeness["warnings"],
                        # Instance lists
                        "InstanceSOPUIDs": series.get_instance_uids(),
                        "InstanceFilePaths": series.get_file_paths(),
                    }

                    # Add common metadata from all levels
                    for key, value in patient.common_metadata.items():
                        if key not in row:
                            row[key] = value
                    for key, value in study.common_metadata.items():
                        if key not in row:
                            row[key] = value
                    for key, value in series.common_metadata.items():
                        if key not in row:
                            row[key] = value

                    rows.append(row)

        return pd.DataFrame(rows)

    def get_instances_df(
        self,
        patient_id: Optional[str] = None,
        study_uid: Optional[str] = None,
        series_uid: Optional[str] = None,
    ) -> pd.DataFrame:
        """Export instance-level detail DataFrame.

        Args:
            patient_id: Optional filter by patient ID.
            study_uid: Optional filter by study UID.
            series_uid: Optional filter by series UID.

        Returns:
            DataFrame with complete instance information.
        """
        rows = []
        for patient in self.patients:
            if patient_id and patient.patient_id != patient_id:
                continue

            for study in patient.studies:
                if study_uid and study.study_instance_uid != study_uid:
                    continue

                for series in study.series:
                    if series_uid and series.series_instance_uid != series_uid:
                        continue

                    for instance in series.instances:
                        row: dict[str, Any] = {
                            # Hierarchy IDs
                            "PatientID": patient.patient_id,
                            "StudyInstanceUID": study.study_instance_uid,
                            "SeriesInstanceUID": series.series_instance_uid,
                            "SOPInstanceUID": instance.sop_instance_uid,
                            # Instance info
                            "FilePath": str(instance.file_path),
                            "InstanceNumber": instance.instance_number,
                            "SliceLocation": instance.slice_location,
                            "ProjectionScore": instance.projection_score,
                            "AcquisitionDateTime": instance.acquisition_datetime,
                        }

                        # Image position
                        if instance.image_position_patient:
                            row["ImagePositionPatient_X"] = (
                                instance.image_position_patient[0]
                            )
                            row["ImagePositionPatient_Y"] = (
                                instance.image_position_patient[1]
                            )
                            row["ImagePositionPatient_Z"] = (
                                instance.image_position_patient[2]
                            )
                        else:
                            row["ImagePositionPatient_X"] = None
                            row["ImagePositionPatient_Y"] = None
                            row["ImagePositionPatient_Z"] = None

                        # Image orientation
                        row["ImageOrientationPatient"] = (
                            instance.image_orientation_patient
                        )

                        # Add parent-level metadata
                        row["PatientsName"] = patient.patients_name
                        row["StudyDate"] = study.study_date
                        row["StudyDescription"] = study.study_description
                        row["SeriesNumber"] = series.series_number
                        row["SeriesDescription"] = series.series_description
                        row["Modality"] = series.modality

                        # Add instance-specific metadata
                        for key, value in instance.metadata.items():
                            if key not in row:
                                row[key] = value

                        rows.append(row)

        return pd.DataFrame(rows)

    # ========================================================================
    # Export Methods
    # ========================================================================

    def export_csv(
        self,
        base_path: str,
        levels: Optional[list[str]] = None,
    ) -> dict[str, str]:
        """Export DataFrames to separate CSV files.

        Args:
            base_path: Base path for output files (without extension).
            levels: List of levels to export ('patients', 'studies', 'series',
                   'instances'). Defaults to all levels.

        Returns:
            Dictionary mapping level names to created file paths.
        """
        if levels is None:
            levels = ["patients", "studies", "series", "instances"]

        created_files = {}

        level_methods: dict[str, Callable[[], pd.DataFrame]] = {
            "patients": self.get_patients_df,
            "studies": self.get_studies_df,
            "series": self.get_series_df,
            "instances": self.get_instances_df,
        }

        for level in levels:
            if level in level_methods:
                df = level_methods[level]()
                file_path = f"{base_path}_{level}.csv"
                # Convert list columns to JSON strings for CSV compatibility
                for col in df.columns:
                    if df[col].apply(lambda x: isinstance(x, list)).any():
                        df[col] = df[col].apply(
                            lambda x: json.dumps(x) if isinstance(x, list) else x
                        )
                df.to_csv(file_path, index=False)
                created_files[level] = file_path

        return created_files

    def export_json(self, json_path: str) -> str:
        """Export full hierarchy to JSON.

        Args:
            json_path: Path for the output JSON file.

        Returns:
            Path to the created file.
        """
        data: dict[str, list[Any]] = {"patients": []}

        for patient in self.patients:
            patient_dict: dict[str, Any] = {
                "patient_id": patient.patient_id,
                "patients_name": patient.patients_name,
                "patients_birth_date": patient.patients_birth_date,
                "patients_sex": patient.patients_sex,
                "common_metadata": patient.common_metadata,
                "studies": [],
            }

            for study in patient.studies:
                study_dict: dict[str, Any] = {
                    "study_instance_uid": study.study_instance_uid,
                    "study_date": study.study_date,
                    "study_time": study.study_time,
                    "study_description": study.study_description,
                    "common_metadata": study.common_metadata,
                    "series": [],
                }

                for series in study.series:
                    completeness = series.check_completeness(self.spacing_tolerance)
                    series_dict: dict[str, Any] = {
                        "series_instance_uid": series.series_instance_uid,
                        "series_number": series.series_number,
                        "series_description": series.series_description,
                        "modality": series.modality,
                        "frame_of_reference_uid": series.frame_of_reference_uid,
                        "common_metadata": series.common_metadata,
                        "completeness": completeness,
                        "instances": [],
                    }

                    for instance in series.instances:
                        instance_dict = {
                            "sop_instance_uid": instance.sop_instance_uid,
                            "file_path": str(instance.file_path),
                            "instance_number": instance.instance_number,
                            "image_position_patient": instance.image_position_patient,
                            "slice_location": instance.slice_location,
                            "projection_score": instance.projection_score,
                            "metadata": instance.metadata,
                        }
                        series_dict["instances"].append(instance_dict)

                    study_dict["series"].append(series_dict)

                patient_dict["studies"].append(study_dict)

            data["patients"].append(patient_dict)

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return json_path


# ============================================================================
# Private Helper Functions (Splitting & Hierarchy)
# ============================================================================


def _split_series_instances(
    instances: list[DicomInstance],
) -> list[list[DicomInstance]]:
    """Split instances of a series into multiple groups (phases).

    Strategy:
    1. Check for distinctive tags (AcquisitionNumber, TemporalPositionIdentifier, etc.).
       If a tag has > 1 unique value, use it to group.
    2. Fallback: Check for duplicate spatial positions (ImagePositionPatient).
       If duplicates exist, group by order of appearance.
    """
    if len(instances) < 2:
        return [instances]

    # Priority list of tags to check for splitting
    split_tags = [
        "NominalPercentageOfCardiacPhase",
        "TemporalPositionIdentifier",
        "TriggerTime",
        "AcquisitionNumber",
        "EchoNumber",
    ]

    # 1. Try splitting by tags
    for tag in split_tags:
        # Check if all instances have this tag (or at least some do)
        # Actually, if the tag varies, we should split.
        values: dict[Any, list[DicomInstance]] = {}
        for inst in instances:
            val = inst.tags.get(tag)
            # Treat None as a distinct value group if mixed?
            # Ideally we only split if we have clear groups.
            if val is not None:
                values.setdefault(val, []).append(inst)

        # If we have multiple groups and coverage is good
        if len(values) > 1:
            # Check if we covered all instances or if some are None
            total_grouped = sum(len(g) for g in values.values())
            if total_grouped == len(instances):
                # Sort groups by tag value
                sorted_keys = sorted(values.keys())
                return [values[k] for k in sorted_keys]

    # 2. Fallback: Spatial duplication check
    # Group instances by position
    pos_map: dict[tuple[float, float, float], list[DicomInstance]] = {}
    for inst in instances:
        if inst.image_position_patient:
            pos_map.setdefault(inst.image_position_patient, []).append(inst)

    # Check if we have duplicates (any position has > 1 instance)
    if any(len(g) > 1 for g in pos_map.values()):
        # Max number of phases detected
        num_phases = max(len(g) for g in pos_map.values())

        # Initialize groups
        phase_groups: list[list[DicomInstance]] = [[] for _ in range(num_phases)]

        # Sort positions to be deterministic
        # (Though we iterate instances which are not yet sorted by position...
        #  maybe sort instances by instance number first for stability?)
        sorted_instances = sorted(
            instances,
            key=lambda x: x.instance_number if x.instance_number is not None else 0,
        )

        # Re-map with sorted instances
        pos_map_sorted: dict[tuple[float, float, float], list[DicomInstance]] = {}
        for inst in sorted_instances:
            if inst.image_position_patient:
                pos_map_sorted.setdefault(inst.image_position_patient, []).append(inst)
            else:
                # If no position, put in phase 0?
                phase_groups[0].append(inst)  # pragma: no cover

        # Distribute duplicates across phases
        for _, insts in pos_map_sorted.items():
            for i, inst in enumerate(insts):
                if i < num_phases:
                    phase_groups[i].append(inst)
                else:  # pragma: no cover
                    # Should not happen if num_phases is max(len)
                    phase_groups[-1].append(inst)

        # Filter empty groups
        return [g for g in phase_groups if g]

    return [instances]


# ============================================================================
# Private Helper Functions (Parallel Processing)
# ============================================================================


def _get_num_workers(num_workers: Optional[int]) -> int:
    """Determine the number of workers to use.

    Args:
        num_workers: User-specified workers. None=auto, 1=sequential.

    Returns:
        Number of workers to use (minimum 1).
    """
    if num_workers is not None:
        return max(1, num_workers)
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _is_dicom_file(file_path: Path) -> Optional[Path]:
    """Check if a file is a DICOM file (for parallel processing).

    Returns the path if DICOM, None otherwise.
    """
    try:
        if pydicom.misc.is_dicom(file_path):
            return file_path
    except Exception:
        pass
    return None


def _scan_dicom_files(
    paths: list[Path],
    recursive: bool,
    show_progress: bool,
    num_workers: int = 1,
) -> list[Path]:
    """Scan folders for DICOM files with optional parallel processing.

    Args:
        paths: List of folder paths to scan.
        recursive: Whether to scan subdirectories.
        show_progress: Whether to display progress bar.
        num_workers: Number of parallel workers (1=sequential).

    Returns:
        List of paths to DICOM files.
    """
    all_candidates: list[Path] = []

    # Collect all candidate files
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            all_candidates.append(path)
        elif path.is_dir():
            if recursive:
                all_candidates.extend(path.rglob("*"))
            else:
                all_candidates.extend(path.iterdir())

    # Filter to files only
    file_candidates = [p for p in all_candidates if p.is_file()]

    if not file_candidates:
        return []

    # Use parallel processing if num_workers > 1
    if num_workers > 1:
        chunksize = max(1, len(file_candidates) // (num_workers * 4))
        dicom_files: list[Path] = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks and collect results with progress bar
            results = list(
                tqdm(
                    executor.map(_is_dicom_file, file_candidates, chunksize=chunksize),
                    total=len(file_candidates),
                    desc="Scanning for DICOM files",
                    disable=not show_progress,
                )
            )
        # Filter out None results
        dicom_files = [p for p in results if p is not None]
        return dicom_files

    # Sequential processing
    dicom_files = []
    iterator = tqdm(
        file_candidates,
        desc="Scanning for DICOM files",
        disable=not show_progress,
    )

    for file_path in iterator:
        try:
            if pydicom.misc.is_dicom(file_path):
                dicom_files.append(file_path)
        except Exception:
            continue

    return dicom_files


def _extract_metadata_wrapper(args: tuple[Path, bool]) -> Optional[dict[str, Any]]:
    """Wrapper for parallel metadata extraction (must be top-level function)."""
    file_path, extract_private_tags = args
    return _extract_single_file_metadata(file_path, extract_private_tags)


def _extract_all_metadata(
    dicom_files: list[Path],
    show_progress: bool,
    extract_private_tags: bool,
    num_workers: int = 1,
) -> list[dict[str, Any]]:
    """Extract metadata from all DICOM files with optional parallel processing.

    Args:
        dicom_files: List of DICOM file paths.
        show_progress: Whether to display progress bar.
        extract_private_tags: Whether to extract private tags.
        num_workers: Number of parallel workers (1=sequential).

    Returns:
        List of metadata dictionaries.
    """
    if not dicom_files:
        return []

    # Use parallel processing if num_workers > 1
    if num_workers > 1:
        chunksize = max(1, len(dicom_files) // (num_workers * 4))
        # Create argument tuples for parallel execution
        args_list = [(fp, extract_private_tags) for fp in dicom_files]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(
                        _extract_metadata_wrapper, args_list, chunksize=chunksize
                    ),
                    total=len(dicom_files),
                    desc="Extracting metadata",
                    disable=not show_progress,
                )
            )
        # Filter out None results
        return [m for m in results if m is not None]

    # Sequential processing
    metadata_list = []
    iterator = tqdm(
        dicom_files,
        desc="Extracting metadata",
        disable=not show_progress,
    )

    for file_path in iterator:
        try:
            metadata = _extract_single_file_metadata(file_path, extract_private_tags)
            if metadata:
                metadata_list.append(metadata)
        except Exception:
            continue

    return metadata_list


def _sort_hierarchy(patients: list[DicomPatient]) -> list[DicomPatient]:
    """Sort the hierarchy for consistent output ordering.

    Patients sorted by ID, studies by date/UID, series by number/UID,
    instances by projection score or instance number.
    """
    for patient in patients:
        for study in patient.studies:
            for series in study.series:
                # Sort instances by projection score, fallback to instance number
                series.instances.sort(
                    key=lambda x: (
                        (
                            x.projection_score
                            if x.projection_score is not None
                            else float("inf")
                        ),
                        (
                            x.instance_number
                            if x.instance_number is not None
                            else float("inf")
                        ),
                        x.sop_instance_uid,
                    )
                )
            # Sort series by series number, fallback to UID
            study.series.sort(
                key=lambda x: (
                    x.series_number if x.series_number is not None else float("inf"),
                    x.series_instance_uid,
                )
            )
        # Sort studies by date, fallback to UID
        patient.studies.sort(
            key=lambda x: (
                x.study_date if x.study_date is not None else "",
                x.study_instance_uid,
            )
        )
    # Sort patients by ID
    patients.sort(key=lambda x: x.patient_id)
    return patients


def _extract_single_file_metadata(
    file_path: Path,
    extract_private_tags: bool,
) -> Optional[dict[str, Any]]:
    """Extract metadata from a single DICOM file (stateless, parallelizable).

    Uses header-only reading to avoid loading pixel data.

    Returns:
        Dictionary with extracted metadata, or None if extraction fails.
    """
    try:
        # Read header only (no pixel data)
        dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
    except Exception:
        return None

    metadata: dict[str, Any] = {
        "file_path": file_path,
        # Patient level
        "PatientID": _get_tag_value(dcm, "PatientID", "UNKNOWN"),
        "PatientsName": _get_tag_value(dcm, "PatientName"),
        "PatientsBirthDate": _get_tag_value(dcm, "PatientBirthDate"),
        "PatientsSex": _get_tag_value(dcm, "PatientSex"),
        # Study level
        "StudyInstanceUID": _get_tag_value(dcm, "StudyInstanceUID", "UNKNOWN"),
        "StudyDate": _get_tag_value(dcm, "StudyDate"),
        "StudyTime": _get_tag_value(dcm, "StudyTime"),
        "StudyDescription": _get_tag_value(dcm, "StudyDescription"),
        # Series level
        "SeriesInstanceUID": _get_tag_value(dcm, "SeriesInstanceUID", "UNKNOWN"),
        "SeriesNumber": _get_tag_value(dcm, "SeriesNumber"),
        "SeriesDescription": _get_tag_value(dcm, "SeriesDescription"),
        "Modality": _get_tag_value(dcm, "Modality"),
        "FrameOfReferenceUID": _get_tag_value(dcm, "FrameOfReferenceUID"),
        # Instance level
        "SOPInstanceUID": _get_tag_value(dcm, "SOPInstanceUID", "UNKNOWN"),
        "InstanceNumber": _get_tag_value(dcm, "InstanceNumber"),
        "SliceLocation": _get_tag_value(dcm, "SliceLocation"),
        "AcquisitionDate": _get_tag_value(dcm, "AcquisitionDate"),
        "AcquisitionTime": _get_tag_value(dcm, "AcquisitionTime"),
        # Multi-phase splitting tags
        "NominalPercentageOfCardiacPhase": _get_tag_value(
            dcm, "NominalPercentageOfCardiacPhase"
        ),
        "TemporalPositionIdentifier": _get_tag_value(dcm, "TemporalPositionIdentifier"),
        "TriggerTime": _get_tag_value(dcm, "TriggerTime"),
        "AcquisitionNumber": _get_tag_value(dcm, "AcquisitionNumber"),
        "EchoNumber": _get_tag_value(dcm, "EchoNumber"),
    }

    # Extract spatial geometry for completeness validation
    try:
        ipp = dcm.ImagePositionPatient
        metadata["ImagePositionPatient"] = (float(ipp[0]), float(ipp[1]), float(ipp[2]))
    except (AttributeError, IndexError, TypeError):
        metadata["ImagePositionPatient"] = None

    try:
        iop = dcm.ImageOrientationPatient
        metadata["ImageOrientationPatient"] = tuple(float(x) for x in iop)
    except (AttributeError, IndexError, TypeError):
        metadata["ImageOrientationPatient"] = None

    # Calculate projection score for spatial sorting
    if metadata["ImagePositionPatient"] and metadata["ImageOrientationPatient"]:
        try:
            iop = metadata["ImageOrientationPatient"]
            row_cosines = np.array(iop[:3])
            col_cosines = np.array(iop[3:])
            slice_normal = np.cross(row_cosines, col_cosines)
            position = np.array(metadata["ImagePositionPatient"])
            metadata["ProjectionScore"] = float(np.dot(position, slice_normal))
        except Exception:
            metadata["ProjectionScore"] = None
    else:
        metadata["ProjectionScore"] = None

    # Extract additional common tags for series level
    series_tags = [
        "SliceThickness",
        "SpacingBetweenSlices",
        "PixelSpacing",
        "ReconstructionDiameter",
        "ConvolutionKernel",
        "KVP",
        "ExposureTime",
        "XRayTubeCurrent",
        "Manufacturer",
        "ManufacturerModelName",
        "StationName",
        "InstitutionName",
        "BodyPartExamined",
        "ProtocolName",
    ]

    for tag in series_tags:
        value = _get_tag_value(dcm, tag)
        if value is not None:
            metadata[tag] = value

    # Extract private tags if requested
    if extract_private_tags:
        for elem in dcm:
            if elem.tag.is_private:
                try:
                    key = f"Private_{elem.tag.group:04X}_{elem.tag.element:04X}"
                    metadata[key] = str(elem.value)
                except Exception:
                    continue

    return metadata


def _get_tag_value(
    dcm: pydicom.Dataset,
    tag_name: str,
    default: Optional[Any] = None,
) -> Optional[Any]:
    """Safely extract a tag value from a DICOM dataset."""
    try:
        value = getattr(dcm, tag_name, None)
        if value is None:
            return default
        # Convert PersonName to string
        if hasattr(value, "family_name"):
            return str(value)
        # Convert MultiValue to list/tuple
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            return list(value)
        return value
    except Exception:
        return default


def _build_hierarchy(
    file_metadata: list[dict[str, Any]],
    spacing_tolerance: float,
    split_multiseries: bool = True,
) -> list[DicomPatient]:
    """Build hierarchical structure from flat metadata list.

    Groups files by Patient -> Study -> Series -> Instance.
    Extracts common metadata at each level.
    """
    # Group by patient
    patients_dict: dict[str, dict[str, Any]] = {}

    for meta in file_metadata:
        patient_id = meta["PatientID"]
        study_uid = meta["StudyInstanceUID"]
        series_uid = meta["SeriesInstanceUID"]
        sop_uid = meta["SOPInstanceUID"]

        # Initialize patient if needed
        if patient_id not in patients_dict:
            patients_dict[patient_id] = {
                "patient_id": patient_id,
                "patients_name": meta.get("PatientsName"),
                "patients_birth_date": meta.get("PatientsBirthDate"),
                "patients_sex": meta.get("PatientsSex"),
                "studies": {},
                "all_metadata": [],
            }

        patient = patients_dict[patient_id]
        patient["all_metadata"].append(meta)

        # Initialize study if needed
        if study_uid not in patient["studies"]:
            patient["studies"][study_uid] = {
                "study_instance_uid": study_uid,
                "study_date": meta.get("StudyDate"),
                "study_time": meta.get("StudyTime"),
                "study_description": meta.get("StudyDescription"),
                "series": {},
                "all_metadata": [],
            }

        study = patient["studies"][study_uid]
        study["all_metadata"].append(meta)

        # Initialize series if needed
        if series_uid not in study["series"]:
            study["series"][series_uid] = {
                "series_instance_uid": series_uid,
                "series_number": meta.get("SeriesNumber"),
                "series_description": meta.get("SeriesDescription"),
                "modality": meta.get("Modality"),
                "frame_of_reference_uid": meta.get("FrameOfReferenceUID"),
                "instances": {},
                "all_metadata": [],
            }

        series = study["series"][series_uid]
        series["all_metadata"].append(meta)

        # Add instance
        series["instances"][sop_uid] = meta

    # Convert to dataclass objects and extract common metadata
    patients = []

    for patient_id, patient_data in patients_dict.items():
        patient_common = _extract_common_metadata(patient_data["all_metadata"])

        studies = []
        for study_uid, study_data in patient_data["studies"].items():
            study_common = _extract_common_metadata(study_data["all_metadata"])

            series_list = []
            for series_uid, series_data in study_data["series"].items():
                series_common = _extract_common_metadata(series_data["all_metadata"])

                instances = []
                for sop_uid, inst_meta in series_data["instances"].items():
                    instance = DicomInstance(
                        sop_instance_uid=sop_uid,
                        file_path=inst_meta["file_path"],
                        instance_number=inst_meta.get("InstanceNumber"),
                        image_position_patient=inst_meta.get("ImagePositionPatient"),
                        image_orientation_patient=inst_meta.get(
                            "ImageOrientationPatient"
                        ),
                        slice_location=inst_meta.get("SliceLocation"),
                        acquisition_datetime=_combine_datetime(
                            inst_meta.get("AcquisitionDate"),
                            inst_meta.get("AcquisitionTime"),
                        ),
                        projection_score=inst_meta.get("ProjectionScore"),
                        metadata={
                            k: v
                            for k, v in inst_meta.items()
                            if k
                            not in {
                                "file_path",
                                "PatientID",
                                "PatientsName",
                                "PatientsBirthDate",
                                "PatientsSex",
                                "StudyInstanceUID",
                                "StudyDate",
                                "StudyTime",
                                "StudyDescription",
                                "SeriesInstanceUID",
                                "SeriesNumber",
                                "SeriesDescription",
                                "Modality",
                                "FrameOfReferenceUID",
                                "SOPInstanceUID",
                                "InstanceNumber",
                                "SliceLocation",
                                "ImagePositionPatient",
                                "ImageOrientationPatient",
                                "ProjectionScore",
                                "AcquisitionDate",
                                "AcquisitionTime",
                            }
                        },
                        tags={
                            k: inst_meta.get(k)
                            for k in {
                                "NominalPercentageOfCardiacPhase",
                                "TemporalPositionIdentifier",
                                "TriggerTime",
                                "AcquisitionNumber",
                                "EchoNumber",
                            }
                            if inst_meta.get(k) is not None
                        },
                    )
                    instances.append(instance)

                # Split series if requested
                if split_multiseries:
                    grouped_instances = _split_series_instances(instances)
                else:
                    grouped_instances = [instances]

                for i, group in enumerate(grouped_instances):
                    # If split, append suffix to SeriesInstanceUID
                    series_uid_final = series_uid
                    if len(grouped_instances) > 1:
                        series_uid_final = f"{series_uid}.{i + 1}"

                    series_obj = DicomSeries(
                        series_instance_uid=series_uid_final,
                        series_number=series_data["series_number"],
                        series_description=series_data["series_description"],
                        modality=series_data["modality"],
                        frame_of_reference_uid=series_data["frame_of_reference_uid"],
                        instances=group,
                        common_metadata=series_common,
                    )
                    series_list.append(series_obj)

            study_obj = DicomStudy(
                study_instance_uid=study_uid,
                study_date=study_data["study_date"],
                study_time=study_data["study_time"],
                study_description=study_data["study_description"],
                series=series_list,
                common_metadata=study_common,
            )
            studies.append(study_obj)

        patient_obj = DicomPatient(
            patient_id=patient_id,
            patients_name=patient_data["patients_name"],
            patients_birth_date=patient_data["patients_birth_date"],
            patients_sex=patient_data["patients_sex"],
            studies=studies,
            common_metadata=patient_common,
        )
        patients.append(patient_obj)

    return patients


def _extract_common_metadata(
    metadata_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract metadata tags that are identical across all items.

    Used to identify common tags at parent levels (study, series).
    """
    if not metadata_list:
        return {}

    # Skip hierarchical identifier tags
    skip_tags = {
        "file_path",
        "PatientID",
        "PatientsName",
        "PatientsBirthDate",
        "PatientsSex",
        "StudyInstanceUID",
        "StudyDate",
        "StudyTime",
        "StudyDescription",
        "SeriesInstanceUID",
        "SeriesNumber",
        "SeriesDescription",
        "Modality",
        "FrameOfReferenceUID",
        "SOPInstanceUID",
        "InstanceNumber",
        "SliceLocation",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "ProjectionScore",
        "AcquisitionDate",
        "AcquisitionTime",
    }

    common = {}
    first = metadata_list[0]

    for key, value in first.items():
        if key in skip_tags:
            continue
        if value is None:
            continue

        # Check if this value is the same across all items
        is_common = all(_values_equal(meta.get(key), value) for meta in metadata_list)
        if is_common:
            common[key] = value

    return common


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two values for equality, handling special cases."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return list(a) == list(b)
        return bool(a == b)
    except Exception:
        return False


def _combine_datetime(
    date_str: Optional[str], time_str: Optional[str]
) -> Optional[str]:
    """Combine DICOM date and time strings into ISO format."""
    if not date_str:
        return None
    result = date_str
    if time_str:
        result = f"{date_str}T{time_str}"
    return result
