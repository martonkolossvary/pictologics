# Changelog

## [0.2.0] - 2025-12-31

### Added
- **DICOM Database Utility**: New `DicomDatabase` class for parsing and organizing complex DICOM folder hierarchies
  - Automatic multi-phase/multi-series detection and splitting
  - Patient → Study → Series → Instance hierarchy traversal
  - DataFrame exports at all hierarchy levels
  - Completeness checking for series (gap detection, spacing validation)
  - JSON/CSV export capabilities
- **Minor bug fixes**

### Changed
- Updated documentation to reflect new utilities and DICOM parsing capabilities

---

## [0.1.0] - 2025-12-28

**Initial commit**

---

[0.2.0]: https://github.com/martonkolossvary/pictologics/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/martonkolossvary/pictologics/releases/tag/v0.1.0
