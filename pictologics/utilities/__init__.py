"""
Pictologics Utilities
=====================

This module provides utilities for:
- Organizing DICOM files into hierarchical databases
- Detecting multi-phase DICOM series
- Parsing DICOM Structured Reports (SR) for measurement extraction
- Visualizing medical images and segmentation masks

Key Features:
- Recursive folder scanning with progress indication
- Header-only DICOM reading for performance
- Multi-level DataFrame exports (Patient/Study/Series/Instance)
- Completeness validation using spatial geometry
- CSV and JSON export capabilities
- SR measurement extraction to DataFrame/CSV/JSON
- Interactive slice visualization with optional mask overlay
- Batch export of slice images (PNG, JPEG, TIFF)
- Window/Level normalization for CT/MR viewing
"""

from .dicom_database import (
    DicomDatabase,
    DicomInstance,
    DicomPatient,
    DicomSeries,
    DicomStudy,
)
from .dicom_utils import (
    DicomPhaseInfo,
    get_dicom_phases,
    split_dicom_phases,
)
from .sr_parser import (
    SRBatch,
    SRDocument,
    SRMeasurement,
    SRMeasurementGroup,
    is_dicom_sr,
)
from .visualization import (
    save_slices,
    visualize_slices,
)

__all__ = [
    # DICOM Database
    "DicomDatabase",
    "DicomPatient",
    "DicomStudy",
    "DicomSeries",
    "DicomInstance",
    # DICOM Utilities
    "DicomPhaseInfo",
    "get_dicom_phases",
    "split_dicom_phases",
    # SR Parser
    "SRBatch",
    "SRDocument",
    "SRMeasurement",
    "SRMeasurementGroup",
    "is_dicom_sr",
    # Visualization
    "save_slices",
    "visualize_slices",
]
