"""
DICOM Structured Report (SR) Parser
====================================

This module provides functionality for parsing DICOM Structured Reports (SR)
and extracting measurement data into structured formats (DataFrames, CSV, JSON).

Supports TID1500 (Measurement Report) and other common SR templates.
Uses highdicom for robust SR parsing and content extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ============================================================================
# Dataclass Definitions (mirroring dicom_database.py pattern)
# ============================================================================


@dataclass
class SRMeasurement:
    """Represents a single measurement from an SR document.

    This dataclass captures individual measurement values extracted from
    DICOM Structured Reports, including the measurement name, value,
    units, and associated context.

    Attributes:
        name: Measurement concept name (e.g., "Agatston Score", "Volume").
        value: Numerical measurement value.
        unit: Unit of measurement (e.g., "mm", "HU", "1" for unitless).
        finding_type: Type of finding this measurement relates to (optional).
        finding_site: Anatomical site of finding (optional).
        derivation: How the measurement was derived (optional).
        tracking_id: Optional tracking identifier for longitudinal studies.
        metadata: Additional extracted attributes not captured above.
    """

    name: str
    value: float
    unit: str
    finding_type: Optional[str] = None
    finding_site: Optional[str] = None
    derivation: Optional[str] = None
    tracking_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SRMeasurementGroup:
    """Represents a group of related measurements.

    SR documents often organize measurements into groups based on
    anatomical site, finding type, or other criteria. This dataclass
    captures such groupings.

    Attributes:
        group_id: Identifier for this measurement group (optional).
        finding_type: Type of finding for this group (optional).
        finding_site: Anatomical site for this group (optional).
        measurements: List of SRMeasurement objects in this group.
        metadata: Additional group-level attributes.
    """

    group_id: Optional[str] = None
    finding_type: Optional[str] = None
    finding_site: Optional[str] = None
    measurements: list[SRMeasurement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SRDocument:
    """Represents a parsed DICOM Structured Report.

    This class provides the main interface for accessing SR content,
    following the same pattern as DicomDatabase. It can be constructed
    from a file using the `from_file()` class method.

    Attributes:
        file_path: Path to the source SR file.
        sop_instance_uid: Unique identifier for this SR instance.
        template_id: SR template identifier (e.g., "1500" for TID1500).
        document_title: Title of the SR document.
        measurement_groups: List of SRMeasurementGroup objects.
        patient_id: Patient identifier.
        study_instance_uid: Study UID.
        series_instance_uid: Series UID.
        content_datetime: When the SR was created.
        metadata: Additional document-level attributes.

    Example:
        Load and parse an SR document:

        ```python
        from pictologics.utilities.sr_parser import SRDocument

        sr = SRDocument.from_file("measurements.dcm")
        print(f"Template: {sr.template_id}")
        print(f"Groups: {len(sr.measurement_groups)}")

        # Export measurements to DataFrame
        df = sr.get_measurements_df()
        print(df[["measurement_name", "value", "unit"]])

        # Export to CSV
        sr.export_csv("measurements.csv")
        ```
    """

    file_path: Path
    sop_instance_uid: str
    template_id: Optional[str] = None
    document_title: Optional[str] = None
    measurement_groups: list[SRMeasurementGroup] = field(default_factory=list)
    patient_id: Optional[str] = None
    study_instance_uid: Optional[str] = None
    series_instance_uid: Optional[str] = None
    content_datetime: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        extract_private_tags: bool = False,
    ) -> "SRDocument":
        """Load and parse an SR document from file.

        This method reads a DICOM Structured Report file and extracts
        all measurement content into the hierarchical dataclass structure.
        Follows the same pattern as DicomDatabase.from_folders().

        Args:
            path: Path to DICOM SR file.
            extract_private_tags: Whether to extract vendor-specific tags
                into the metadata dictionaries. Defaults to False.

        Returns:
            SRDocument instance with parsed content.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid DICOM SR object.
        """
        import pydicom

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"SR file not found: {path}")

        # Load the DICOM file
        try:
            dcm = pydicom.dcmread(str(path_obj))
        except Exception as e:
            raise ValueError(f"Failed to read DICOM file: {e}") from e

        # Check if it's an SR document
        sr_sop_classes = [
            "1.2.840.10008.5.1.4.1.1.88.11",  # Basic Text SR
            "1.2.840.10008.5.1.4.1.1.88.22",  # Enhanced SR
            "1.2.840.10008.5.1.4.1.1.88.33",  # Comprehensive SR
            "1.2.840.10008.5.1.4.1.1.88.34",  # Comprehensive 3D SR
            "1.2.840.10008.5.1.4.1.1.88.35",  # Extensible SR
            "1.2.840.10008.5.1.4.1.1.88.40",  # Procedure Log
        ]
        sop_class = str(getattr(dcm, "SOPClassUID", ""))
        if sop_class not in sr_sop_classes:
            raise ValueError(
                f"File is not a DICOM SR document. SOPClassUID: {sop_class}"
            )

        # Extract basic document info
        sop_instance_uid = str(getattr(dcm, "SOPInstanceUID", ""))
        patient_id = str(getattr(dcm, "PatientID", "")) or None
        study_uid = str(getattr(dcm, "StudyInstanceUID", "")) or None
        series_uid = str(getattr(dcm, "SeriesInstanceUID", "")) or None

        # Content datetime
        content_date = getattr(dcm, "ContentDate", None)
        content_time = getattr(dcm, "ContentTime", None)
        if content_date:
            content_datetime = str(content_date)
            if content_time:
                content_datetime += f"T{content_time}"
        else:
            content_datetime = None

        # Extract document title from ConceptNameCodeSequence
        doc_title = None
        if hasattr(dcm, "ConceptNameCodeSequence") and dcm.ConceptNameCodeSequence:
            concept = dcm.ConceptNameCodeSequence[0]
            doc_title = str(getattr(concept, "CodeMeaning", "")) or None

        # Extract template ID if present
        template_id = None
        if hasattr(dcm, "ContentTemplateSequence") and dcm.ContentTemplateSequence:
            template = dcm.ContentTemplateSequence[0]
            template_id = str(getattr(template, "TemplateIdentifier", "")) or None

        # Parse content sequence for measurements
        measurement_groups = _parse_content_sequence(dcm, extract_private_tags)

        # Build metadata dict
        metadata: dict[str, Any] = {}
        if extract_private_tags:
            # Extract any private tags
            for elem in dcm:
                if elem.tag.is_private:
                    try:
                        metadata[elem.keyword or str(elem.tag)] = str(elem.value)
                    except Exception:
                        pass

        return cls(
            file_path=path_obj,
            sop_instance_uid=sop_instance_uid,
            template_id=template_id,
            document_title=doc_title,
            measurement_groups=measurement_groups,
            patient_id=patient_id,
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            content_datetime=content_datetime,
            metadata=metadata,
        )

    def get_measurements_df(self) -> pd.DataFrame:
        """Export all measurements as a DataFrame.

        Returns a flat DataFrame with all measurements from all groups,
        including group context for each measurement.

        Returns:
            DataFrame with columns:
            - group_id: Identifier of the measurement group
            - finding_type: Type of finding
            - finding_site: Anatomical site
            - measurement_name: Name of the measurement
            - value: Numerical value
            - unit: Unit of measurement
            - derivation: How it was derived
            - tracking_id: Tracking identifier
        """
        rows = []
        for group in self.measurement_groups:
            for meas in group.measurements:
                rows.append(
                    {
                        "group_id": group.group_id,
                        "finding_type": group.finding_type or meas.finding_type,
                        "finding_site": group.finding_site or meas.finding_site,
                        "measurement_name": meas.name,
                        "value": meas.value,
                        "unit": meas.unit,
                        "derivation": meas.derivation,
                        "tracking_id": meas.tracking_id,
                    }
                )

        return pd.DataFrame(rows)

    def get_summary(self) -> dict[str, Any]:
        """Get document summary without full parsing.

        Returns:
            Dictionary with summary information including:
            - sop_instance_uid
            - template_id
            - document_title
            - num_groups
            - num_measurements
            - patient_id
            - study_instance_uid
        """
        total_measurements = sum(len(g.measurements) for g in self.measurement_groups)
        return {
            "sop_instance_uid": self.sop_instance_uid,
            "template_id": self.template_id,
            "document_title": self.document_title,
            "num_groups": len(self.measurement_groups),
            "num_measurements": total_measurements,
            "patient_id": self.patient_id,
            "study_instance_uid": self.study_instance_uid,
            "content_datetime": self.content_datetime,
        }

    def export_csv(self, path: str | Path) -> Path:
        """Export measurements to CSV file.

        Args:
            path: Output path for the CSV file.

        Returns:
            Path to the created CSV file.
        """
        path_obj = Path(path)
        df = self.get_measurements_df()
        df.to_csv(path_obj, index=False)
        return path_obj

    def export_json(self, path: str | Path) -> Path:
        """Export full SR content to JSON.

        Exports the complete document structure including all groups,
        measurements, and metadata.

        Args:
            path: Output path for the JSON file.

        Returns:
            Path to the created JSON file.
        """
        import json

        path_obj = Path(path)

        # Build JSON structure
        data: dict[str, Any] = {
            "sop_instance_uid": self.sop_instance_uid,
            "template_id": self.template_id,
            "document_title": self.document_title,
            "patient_id": self.patient_id,
            "study_instance_uid": self.study_instance_uid,
            "series_instance_uid": self.series_instance_uid,
            "content_datetime": self.content_datetime,
            "metadata": self.metadata,
            "measurement_groups": [],
        }

        for group in self.measurement_groups:
            group_data: dict[str, Any] = {
                "group_id": group.group_id,
                "finding_type": group.finding_type,
                "finding_site": group.finding_site,
                "metadata": group.metadata,
                "measurements": [],
            }
            for meas in group.measurements:
                meas_data = {
                    "name": meas.name,
                    "value": meas.value,
                    "unit": meas.unit,
                    "finding_type": meas.finding_type,
                    "finding_site": meas.finding_site,
                    "derivation": meas.derivation,
                    "tracking_id": meas.tracking_id,
                    "metadata": meas.metadata,
                }
                group_data["measurements"].append(meas_data)
            data["measurement_groups"].append(group_data)

        with open(path_obj, "w") as f:
            json.dump(data, f, indent=2)

        return path_obj

    # ========================================================================
    # Batch Processing (from_folders)
    # ========================================================================

    @classmethod
    def from_folders(
        cls,
        paths: list[str | Path],
        recursive: bool = True,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
        export_csv: bool = True,
        export_json: bool = True,
        extract_private_tags: bool = False,
    ) -> "SRBatch":
        """Batch process SR files from folders.

        Scans directories for DICOM SR files, parses each one, and optionally
        exports individual CSV/JSON files plus a combined output and log.

        This method follows the same pattern as DicomDatabase.from_folders().

        Args:
            paths: List of folder paths to scan for SR files.
            recursive: Whether to scan subdirectories (default: True).
            show_progress: Whether to display progress bars (default: True).
            num_workers: Number of parallel workers. None=auto (cpu_count-1),
                        1=sequential (no multiprocessing).
            output_dir: If specified, exports each SR to this directory.
            export_csv: Export individual CSV files (default: True).
            export_json: Export individual JSON files (default: True).
            extract_private_tags: Whether to extract private tags (default: False).

        Returns:
            SRBatch containing all parsed documents and processing log.

        Example:
            Process all SR files in a folder:

            ```python
            from pictologics.utilities.sr_parser import SRDocument

            # Process folder
            batch = SRDocument.from_folders(["sr_data/"])
            print(f"Found {len(batch.documents)} SR files")
            df = batch.get_combined_measurements_df()

            # Process with exports
            batch = SRDocument.from_folders(
                ["sr_data/"],
                output_dir="sr_exports/",
                export_csv=True,
                export_json=True
            )
            batch.export_log("sr_exports/processing_log.csv")
            ```
        """
        import os
        from concurrent.futures import ProcessPoolExecutor

        from tqdm import tqdm

        # Convert paths to Path objects
        path_objs = [Path(p) for p in paths]

        # Determine number of workers
        if num_workers is None:
            cpu_count = os.cpu_count()
            num_workers = max(1, (cpu_count - 1) if cpu_count else 1)

        # Step 1: Discover all SR files
        sr_files: list[Path] = []
        for path_obj in path_objs:
            if not path_obj.exists():
                continue
            if path_obj.is_file():
                if is_dicom_sr(path_obj):
                    sr_files.append(path_obj)
            else:
                # Directory
                iterator = path_obj.rglob("*") if recursive else path_obj.iterdir()
                for f in iterator:
                    if f.is_file() and is_dicom_sr(f):
                        sr_files.append(f)

        if not sr_files:
            return SRBatch(documents=[], processing_log=[], output_dir=None)

        # Create output directory if specified
        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        # Step 2: Process each SR file
        processing_log: list[dict[str, Any]] = []
        documents: list["SRDocument"] = []

        # Prepare worker arguments
        worker_args = [
            (f, extract_private_tags, out_path, export_csv, export_json)
            for f in sr_files
        ]

        if num_workers == 1:
            # Sequential processing
            iterator = tqdm(
                worker_args, desc="Processing SR files", disable=not show_progress
            )
            for args in iterator:
                result = _process_sr_file_worker(args)
                processing_log.append(result["log"])
                if result["document"] is not None:
                    documents.append(result["document"])
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(
                    tqdm(
                        executor.map(_process_sr_file_worker, worker_args),
                        total=len(worker_args),
                        desc="Processing SR files",
                        disable=not show_progress,
                    )
                )
            for result in results:
                processing_log.append(result["log"])
                if result["document"] is not None:
                    documents.append(result["document"])  # pragma: no cover

        return SRBatch(
            documents=documents,
            processing_log=processing_log,
            output_dir=out_path,
        )


@dataclass
class SRBatch:
    """Collection of parsed SR documents from batch processing.

    This class holds the results of batch SR processing via
    SRDocument.from_folders(). It provides access to all parsed documents
    and methods for combined exports.

    Attributes:
        documents: List of successfully parsed SRDocument objects.
        processing_log: Log entries for each processed file (success/error).
        output_dir: Directory where individual exports were written.

    Example:
        ```python
        from pictologics.utilities.sr_parser import SRDocument

        batch = SRDocument.from_folders(["sr_data/"], output_dir="exports/")
        print(f"Processed {len(batch.documents)} SR files")
        df = batch.get_combined_measurements_df()
        batch.export_log("exports/processing_log.csv")
        ```
    """

    documents: list[SRDocument] = field(default_factory=list)
    processing_log: list[dict[str, Any]] = field(default_factory=list)
    output_dir: Optional[Path] = None

    def get_combined_measurements_df(self) -> pd.DataFrame:
        """Combine measurements from all documents into a single DataFrame.

        Each measurement row includes the source document's SOP Instance UID,
        patient ID, and study UID for traceability.

        Returns:
            DataFrame with all measurements from all documents.
        """
        all_rows: list[dict[str, Any]] = []

        for doc in self.documents:
            for group in doc.measurement_groups:
                for meas in group.measurements:
                    all_rows.append(
                        {
                            "sop_instance_uid": doc.sop_instance_uid,
                            "patient_id": doc.patient_id,
                            "study_instance_uid": doc.study_instance_uid,
                            "group_finding_type": group.finding_type,
                            "group_finding_site": group.finding_site,
                            "measurement_name": meas.name,
                            "value": meas.value,
                            "unit": meas.unit,
                            "finding_type": meas.finding_type,
                            "finding_site": meas.finding_site,
                            "derivation": meas.derivation,
                            "tracking_id": meas.tracking_id,
                        }
                    )

        return pd.DataFrame(all_rows)

    def export_combined_csv(self, path: str | Path) -> Path:
        """Export combined measurements to a single CSV file.

        Args:
            path: Output path for the combined CSV.

        Returns:
            Path to the created CSV file.
        """
        path_obj = Path(path)
        df = self.get_combined_measurements_df()
        df.to_csv(path_obj, index=False)
        return path_obj

    def export_log(self, path: str | Path) -> Path:
        """Export processing log to CSV.

        The log contains one row per processed file with status,
        output paths, and any error messages.

        Args:
            path: Output path for the log CSV.

        Returns:
            Path to the created CSV file.
        """
        path_obj = Path(path)
        df = pd.DataFrame(self.processing_log)
        df.to_csv(path_obj, index=False)
        return path_obj


def _process_sr_file_worker(
    args: tuple[Path, bool, Optional[Path], bool, bool],
) -> dict[str, Any]:
    """Worker function for parallel SR processing.

    Args:
        args: Tuple of (file_path, extract_private_tags, output_dir, export_csv, export_json)

    Returns:
        Dict with 'document' (SRDocument or None) and 'log' (processing log entry).
    """
    import time

    file_path, extract_private_tags, output_dir, export_csv, export_json = args
    start_time = time.time()

    log_entry: dict[str, Any] = {
        "file_path": str(file_path),
        "sop_instance_uid": None,
        "patient_id": None,
        "study_instance_uid": None,
        "status": "error",
        "error_message": None,
        "num_measurements": 0,
        "csv_path": None,
        "json_path": None,
        "processing_time_ms": 0,
    }

    try:
        doc = SRDocument.from_file(file_path, extract_private_tags=extract_private_tags)

        log_entry["sop_instance_uid"] = doc.sop_instance_uid
        log_entry["patient_id"] = doc.patient_id
        log_entry["study_instance_uid"] = doc.study_instance_uid
        log_entry["status"] = "success"
        log_entry["num_measurements"] = sum(
            len(g.measurements) for g in doc.measurement_groups
        )

        # Export if output_dir specified
        if output_dir is not None:
            base_name = doc.sop_instance_uid.replace(".", "_")
            if export_csv:
                csv_path = output_dir / f"{base_name}.csv"
                doc.export_csv(csv_path)
                log_entry["csv_path"] = str(csv_path)
            if export_json:
                json_path = output_dir / f"{base_name}.json"
                doc.export_json(json_path)
                log_entry["json_path"] = str(json_path)

        log_entry["processing_time_ms"] = int((time.time() - start_time) * 1000)
        return {"document": doc, "log": log_entry}

    except Exception as e:
        log_entry["error_message"] = str(e)
        log_entry["processing_time_ms"] = int((time.time() - start_time) * 1000)
        return {"document": None, "log": log_entry}


# ============================================================================
# Helper Functions
# ============================================================================


def _parse_content_sequence(
    dcm: Any,
    extract_private_tags: bool = False,
) -> list[SRMeasurementGroup]:
    """Parse the ContentSequence of an SR document for measurements.

    This function recursively traverses the SR content tree to extract
    measurement groups and individual measurements.

    Args:
        dcm: The DICOM dataset containing ContentSequence.
        extract_private_tags: Whether to extract private tags.

    Returns:
        List of SRMeasurementGroup objects.
    """
    groups: list[SRMeasurementGroup] = []

    if not hasattr(dcm, "ContentSequence"):
        return groups

    # Track current context
    current_group: Optional[SRMeasurementGroup] = None
    measurements: list[SRMeasurement] = []

    for item in dcm.ContentSequence:
        value_type = str(getattr(item, "ValueType", ""))

        # Extract concept name
        concept_name = None
        if hasattr(item, "ConceptNameCodeSequence") and item.ConceptNameCodeSequence:
            concept_name = str(
                getattr(item.ConceptNameCodeSequence[0], "CodeMeaning", "")
            )

        # Handle CONTAINER items (measurement groups)
        if value_type == "CONTAINER":
            # If we have a current group with measurements, save it
            if current_group is not None and measurements:
                current_group.measurements = measurements
                groups.append(current_group)
                measurements = []

            # Start new group
            current_group = SRMeasurementGroup(
                group_id=concept_name,
                finding_type=concept_name,
            )

            # Recursively parse nested content
            nested_groups = _parse_content_sequence(item, extract_private_tags)
            if nested_groups:
                groups.extend(nested_groups)

        # Handle NUM items (numeric measurements)
        elif value_type == "NUM":
            meas = _extract_numeric_measurement(item, concept_name)
            if meas:
                measurements.append(meas)

        # Handle CODE items (may contain site/type info)
        elif value_type == "CODE":
            if current_group is not None:
                code_value = _get_code_value(item)
                if concept_name and "site" in concept_name.lower():
                    current_group.finding_site = code_value
                elif concept_name and "type" in concept_name.lower():
                    current_group.finding_type = code_value

        # Handle TEXT items
        elif value_type == "TEXT":
            # Could contain tracking ID or other text info
            if current_group is not None and concept_name:
                text_value = str(getattr(item, "TextValue", ""))
                if "tracking" in concept_name.lower():
                    # Apply to all measurements in current group
                    current_group.metadata["tracking_id"] = text_value

    # Don't forget the last group
    if current_group is not None and measurements:
        current_group.measurements = measurements
        groups.append(current_group)
    elif measurements:
        # Orphan measurements - create a default group
        groups.append(
            SRMeasurementGroup(
                group_id="default",
                measurements=measurements,
            )
        )

    return groups


def _extract_numeric_measurement(
    item: Any,
    concept_name: Optional[str],
) -> Optional[SRMeasurement]:
    """Extract a numeric measurement from an SR content item.

    Args:
        item: The SR content item with ValueType NUM.
        concept_name: The concept name for this measurement.

    Returns:
        SRMeasurement object if extraction successful, None otherwise.
    """
    if not hasattr(item, "MeasuredValueSequence") or not item.MeasuredValueSequence:
        return None

    mv = item.MeasuredValueSequence[0]

    # Get numeric value
    value = getattr(mv, "NumericValue", None)
    if value is None:
        return None

    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        return None

    # Get unit
    unit = "1"  # Default unitless
    if hasattr(mv, "MeasurementUnitsCodeSequence") and mv.MeasurementUnitsCodeSequence:
        unit_seq = mv.MeasurementUnitsCodeSequence[0]
        unit = str(getattr(unit_seq, "CodeValue", "1"))

    return SRMeasurement(
        name=concept_name or "Unknown",
        value=numeric_value,
        unit=unit,
    )


def _get_code_value(item: Any) -> Optional[str]:
    """Extract the code value from a CODE content item.

    Args:
        item: The SR content item with ValueType CODE.

    Returns:
        The CodeMeaning if available, else CodeValue, else None.
    """
    if hasattr(item, "ConceptCodeSequence") and item.ConceptCodeSequence:
        code = item.ConceptCodeSequence[0]
        return str(getattr(code, "CodeMeaning", "")) or str(
            getattr(code, "CodeValue", "")
        )
    return None


def is_dicom_sr(path: str | Path) -> bool:
    """Check if a DICOM file is a Structured Report.

    Args:
        path: Path to the potential DICOM file.

    Returns:
        True if the file is a DICOM SR object, False otherwise.
    """
    import pydicom

    try:
        dcm = pydicom.dcmread(str(path), stop_before_pixels=True)
        sop_class = str(getattr(dcm, "SOPClassUID", ""))

        sr_sop_classes = [
            "1.2.840.10008.5.1.4.1.1.88.11",  # Basic Text SR
            "1.2.840.10008.5.1.4.1.1.88.22",  # Enhanced SR
            "1.2.840.10008.5.1.4.1.1.88.33",  # Comprehensive SR
            "1.2.840.10008.5.1.4.1.1.88.34",  # Comprehensive 3D SR
            "1.2.840.10008.5.1.4.1.1.88.35",  # Extensible SR
            "1.2.840.10008.5.1.4.1.1.88.40",  # Procedure Log
        ]

        return sop_class in sr_sop_classes
    except Exception:
        return False
