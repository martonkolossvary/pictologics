"""
DICOM Loaders Module
====================

This module provides specialized loaders for DICOM objects that require
advanced parsing beyond standard image files.

Currently supported:
- DICOM SEG (Segmentation) objects via load_seg()
"""

from .seg_loader import get_segment_info, load_seg

__all__ = ["load_seg", "get_segment_info"]
