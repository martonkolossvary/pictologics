# Changelog

## [0.3.0] - 2026-01-25

### Added
- **IBSI 2 Convolutional Filters**: Complete filter module (`pictologics/filters/`) with:
    - Mean filter (3D)
    - Laplacian of Gaussian (LoG)
    - Laws texture energy filters (3D rotation-invariant)
    - Gabor filters (2D per-slice)
    - Wavelet decomposition (Haar, Daubechies, Coiflet, Symlet families)
    - Simoncelli steerable pyramid
- **IBSI 2 Phase 1 Compliance**: Filter response map validation against digital phantoms
- **IBSI 2 Phase 2 Compliance**: Feature extraction from filtered images validated
- **IBSI 2 Phase 3 Compliance**: Multi-modality reproducibility validation across 51 patients × 3 modalities compared to 9 team submissions
- **Filter Pipeline Integration**: New `filter` step in `RadiomicsPipeline` for seamless filtered feature extraction
- **Mask Binarization Pipeline Step**: New `binarize_mask` preprocessing step with configurable `threshold`, `mask_values` (int/list/range tuple), and `apply_to` targeting.

### Changed
- Updated `mkdocs.yml` with IBSI 2 Phase 1, 2, 3 navigation
- Expanded pipeline documentation with filter usage examples and binarization

### Fixed
- **IBSI 1 Compliance (Morphology)**: Achieved passing values for Compactness 2 (`BQWJ`) and Asphericity (`25C7`) in Configs C/D/E and texture matrices in config D by binarizing masks before resampling.

---

## [0.2.0] - 2026-01-06

### Added
- **DICOM Database Utility**: `DicomDatabase` class for parsing complex DICOM folder hierarchies with Patient → Study → Series → Instance traversal, multi-phase detection, and DataFrame/JSON/CSV exports
- **DICOM SEG Loader**: `load_seg()` for loading DICOM Segmentation objects with multi-segment handling, geometry alignment, and seamless auto-detection in `load_image()`
- **DICOM SR Parser**: `SRDocument` class for parsing Structured Reports with measurement extraction, CSV/JSON export, and batch processing via `SRDocument.from_folders()`
- **DICOM Multi-Phase Support**: `load_image()` now supports multi-phase DICOM series with `dataset_index`, plus `get_dicom_phases()` for phase discovery
- **Visualization Utility**: `visualize_slices()` for interactive viewing and `save_slices()` for batch export with window/level normalization and colormap options
- **Cropped Image Repositioning**: `load_image()` and `load_and_merge_images()` support repositioning cropped masks into reference volume coordinate space
- **Intensity Rescaling**: `apply_rescale` parameter in `load_image` and related functions to toggle DICOM rescale slope/intercept application (default: True)
- **Sentinel Value Handling**: Documentation and examples for handling sentinel values (e.g. -2048 in Siemens DICOMs) using the `resegment` preprocessing step
- **Dependencies**: Added `highdicom`, `matplotlib`, `pillow`; updated `pandas>=2.0.0`

### Optimized
- **Morphology Speedup**: Implemented bounding box cropping for morphology features (mesh/moments), significantly accelerating extraction for sparse ROIs in large volumes
- **Texture Speedup**: Added slice-level skipping to texture calculation to ignore empty z-slices, vastly improving performance for disjoint ROIs (e.g. multiple tumors)

### Fixed
- DICOM file loading improvements: proper Z-spacing, 3D SEG handling, direction matrix extraction

### Changed
- `DicomDatabase` uses shared `split_dicom_phases()` for consistent multi-phase detection
- Comprehensive documentation updates for all new utilities

---


## [0.1.0] - 2025-12-28

**Initial commit**

---

[0.3.0]: https://github.com/martonkolossvary/pictologics/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/martonkolossvary/pictologics/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/martonkolossvary/pictologics/releases/tag/v0.1.0
