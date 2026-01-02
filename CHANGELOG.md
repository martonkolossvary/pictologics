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
- `include_instance_lists` parameter for DataFrame and export methods (default `False` to reduce memory)
- API reference documentation for utilities module
- Matplotlib and Pillow added as main dependencies

### Fixed
- Benchmark baseline can now be updated with `--force-continue` flag when regression is detected

### Changed
- Updated documentation to reflect new utilities and DICOM parsing capabilities
- Ruff E402 suppressions added to test files

---

## [0.1.0] - 2025-12-28

**Initial commit**

---

[0.2.0]: https://github.com/martonkolossvary/pictologics/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/martonkolossvary/pictologics/releases/tag/v0.1.0
