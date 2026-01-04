# Changelog

## [Unreleased]

### Added
- **DICOM Multi-Phase Series Support**: `load_image()` now supports multi-phase DICOM series
  - `dataset_index` parameter works for both 4D NIfTI and multi-phase DICOM
  - Automatic phase detection using cardiac phase, temporal position, trigger time, acquisition number, or echo number
  - New `get_dicom_phases()` utility to discover available phases before loading
  - New `DicomPhaseInfo` dataclass for phase metadata
  - Shared `split_dicom_phases()` logic ensures consistent detection across library
- **DICOM SEG Loader**: New `load_seg()` function for loading DICOM Segmentation objects
  - Returns standard `Image` class compatible with all pictologics functions
  - Multi-segment handling with `combine_segments` option
  - Automatic geometry alignment with `reference_image` parameter
  - `get_segment_info()` for inspecting segment metadata
  - Auto-detection in `load_image()` for seamless SEG file handling
- **DICOM SR Parser**: New `SRDocument` class for parsing Structured Reports
  - Extract measurements to `pandas.DataFrame` with `get_measurements_df()`
  - Export to CSV with `export_csv()` 
  - Export to JSON with `export_json()`
  - Support for TID1500 and other common SR templates
  - **Batch Processing**: `SRDocument.from_folders()` for parallel batch SR processing
    - Recursive folder scanning for SR files
    - Configurable `export_csv` and `export_json` options
    - Combined measurements export via `SRBatch.get_combined_measurements_df()`
    - Processing log with status, errors, and output paths
- Added `highdicom` as core dependency for advanced DICOM handling
- Updated copyright to 2026

### Changed
- `DicomDatabase` now uses shared `split_dicom_phases()` for consistent multi-phase detection
- `load_and_merge_images()` docstring updated to document DICOM multi-phase support alongside NIfTI 4D
- Updated `pandas` minimum version requirement to `>=2.0.0`

### Documentation
- Expanded DICOM SEG section in user guide with comprehensive examples:
  - Merging all segments into one mask
  - Selecting specific segments
  - Loading segments separately for radiomics
  - Alignment with reference image geometry
  - Complete workflow examples
- Added DICOM multi-phase discovery and loading examples


---

## [0.2.0] - 2025-01-31

### Added
- **DICOM Database Utility**: New `DicomDatabase` class for parsing and organizing complex DICOM folder hierarchies
  - Automatic multi-phase/multi-series detection and splitting
  - Patient → Study → Series → Instance hierarchy traversal
  - DataFrame exports at all hierarchy levels
  - Completeness checking for series (gap detection, spacing validation)
  - JSON/CSV export capabilities
- **Visualization Utility**: Flexible functions for viewing images and masks
  - `visualize_slices()` - Interactive slice viewer with optional mask overlay
  - `save_slices()` - Export slice images (PNG, JPEG, TIFF)
  - Multiple display modes: image-only, mask-only, or overlay
  - Window/level normalization support for CT/MR viewing
  - Multiple colormap options (tab10, tab20, Set1, Set2, Paired)
- **Cropped Image Repositioning**: Support for repositioning cropped images/masks into a reference volume's coordinate space
  - `load_image()` now accepts `reference_image` parameter to automatically reposition cropped masks
  - `load_and_merge_images()` now accepts `reposition_to_reference=True` to merge cropped segmentation masks
  - Supports `transpose_axes` parameter for handling different axis orderings
  - Uses spatial metadata (origin, spacing, direction) for accurate positioning
- **Mask relabeling**: `load_and_merge_images(relabel_masks=True)` assigns unique label values (1, 2, 3, ...) to each mask file, enabling visualization with different colors
- `include_instance_lists` parameter for DataFrame and export methods (default `False` to reduce memory)
- API reference documentation for utilities module
- Matplotlib and Pillow added as main dependencies

### Fixed
- Benchmark baseline can now be updated with `--force-continue` flag when regression is detected
- `_load_dicom_file()` now prefers `SpacingBetweenSlices` over `SliceThickness` for Z spacing (consistent with series loading)
- `_load_dicom_file()` now correctly handles 3D DICOM data (e.g., SEG objects) with consistent (X, Y, Z) axis ordering
- `_load_dicom_file()` now extracts direction matrix from `ImageOrientationPatient` when available
- Visualization module now transposes slices for correct display with matplotlib

### Changed
- Updated documentation to reflect new utilities and DICOM parsing capabilities
- Ruff E402 suppressions added to test files

---

## [0.1.0] - 2025-12-28

**Initial commit**

---

[0.2.0]: https://github.com/martonkolossvary/pictologics/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/martonkolossvary/pictologics/releases/tag/v0.1.0
