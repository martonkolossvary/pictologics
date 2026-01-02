# Changelog

## [0.2.0] - 2025-12-31

### Added
- **DICOM Database Utility**: New `DicomDatabase` class for parsing and organizing complex DICOM folder hierarchies
  - Automatic multi-phase/multi-series detection and splitting
  - Patient → Study → Series → Instance hierarchy traversal
  - DataFrame exports at all hierarchy levels
  - Completeness checking for series (gap detection, spacing validation)
  - JSON/CSV export capabilities
- **Mask Visualization Utility**: New functions for visualizing mask overlays
  - `visualize_mask_overlay()` - Interactive slice viewer with scrolling
  - `save_mask_overlay_slices()` - Export overlay images (PNG, JPEG, TIFF)
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
