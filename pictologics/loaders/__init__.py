"""
Loaders Module
==============

This module provides the loaders of formats that need more than an image reader.
``load_image`` uses them by the file name.

Currently supported:
- DICOM SEG (Segmentation) objects via load_seg()
- DICOM RTSTRUCT (RT Structure Set) contours via load_rtstruct()
- NRRD (.nrrd, .nhdr, and 3D Slicer .seg.nrrd) and MetaImage (.mha, .mhd) files
- get_segment_info() lists the segments of a DICOM SEG, an RTSTRUCT or a .seg.nrrd file
"""

from .rtstruct_loader import load_rtstruct
from .seg_loader import get_segment_info, load_seg

__all__ = ["load_seg", "load_rtstruct", "get_segment_info"]
