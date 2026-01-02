"""
Pictologics Utilities
=====================

This module provides utilities for:
- Organizing DICOM files into hierarchical databases
- Visualizing mask overlays on medical images

Key Features:
- Recursive folder scanning with progress indication
- Header-only DICOM reading for performance
- Multi-level DataFrame exports (Patient/Study/Series/Instance)
- Completeness validation using spatial geometry
- CSV and JSON export capabilities
- Interactive mask overlay visualization
- Batch export of overlay images
"""

from .dicom_database import (
    DicomDatabase,
    DicomInstance,
    DicomPatient,
    DicomSeries,
    DicomStudy,
)
from .mask_visualization import (
    save_mask_overlay_slices,
    visualize_mask_overlay,
)

__all__ = [
    # DICOM Database
    "DicomDatabase",
    "DicomPatient",
    "DicomStudy",
    "DicomSeries",
    "DicomInstance",
    # Mask Visualization
    "save_mask_overlay_slices",
    "visualize_mask_overlay",
]
