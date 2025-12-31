"""
DICOM Database Utilities
========================

This module provides utilities for organizing DICOM files into hierarchical
patient/study/series/instance databases with completeness validation.

Key Features:
- Recursive folder scanning with progress indication
- Header-only DICOM reading for performance
- Multi-level DataFrame exports (Patient/Study/Series/Instance)
- Completeness validation using spatial geometry
- CSV and JSON export capabilities
- Parallelization-ready design
"""

from .dicom_database import (
    DicomDatabase,
    DicomInstance,
    DicomPatient,
    DicomSeries,
    DicomStudy,
)

__all__ = [
    "DicomDatabase",
    "DicomPatient",
    "DicomStudy",
    "DicomSeries",
    "DicomInstance",
]
