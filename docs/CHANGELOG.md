# Changelog

## [0.3.4] - 2026-02-14

### Fixed
- **Memory Exhaustion Issue**: Resolved a critical issue where resampled background voxels (value 0) were included in the ROI if they fell within the `resegment` range. The pipeline now explicitly applies the `source_mask` to the `intensity_mask` after resampling when `source_mode="auto"` or an explicit source mask is used. This prevents memory explosions for small ROIs in large volumes with sentinel backgrounds.
- **Pipeline Configuration Serialization**: Fixed a bug where `source_mode` and `sentinel_value` were lost during serialization (`to_dict`/`to_yaml`).
- **Sentinel Detection**: Fixed detection logic to correctly handle auto-generated full masks.

### Changed
- **Documentation**: Updated `Data Loading` and `Pipeline` user guides to clarify the usage of `source_mode="auto"` vs `"full_image"` and its impact on memory and correctness.

---

## [0.3.3] - 2026-02-11

### Changed
- **Sentinel Value Implementation**: Implemented proper handling of sentinel values in the pipeline to assure that they do not influence the feature extraction. 
- **Complete overhaul of User Guide**: Rewrote the user guide to improve clarity and organization. 
- **Benchmark Methodology Updates**: Refined timing methods for benchmarking with optimized measurement techniques, resulting in up to 40% faster execution for PyRadiomics compared to previous implementations. Therefore speed improvements of pictologics are now more modest.

---

## [0.3.2] - 2026-02-01

### Added
- **Feature Deduplication System**: Intelligent optimization for multi-configuration pipelines:
    - Automatically detects when configurations share preprocessing steps but differ only in discretization
    - Computes discretization-independent features (morphology, intensity) once and reuses across configurations
    - `deduplication_stats` property provides reuse/compute statistics after each run
    - Hash-based signature comparison using SHA256 for exact parameter matching
    - Versioned rules system (`DeduplicationRules`) for reproducibility

### Changed
- **Deduplication enabled by default**: `RadiomicsPipeline(deduplicate=True)` is now the default behavior
- **Documentation updated**: 
    - Case examples simplified to reflect default deduplication behavior
    - Benchmark page clarifies methodology (raw timing without caching) and notes additional speedups with deduplication


---

## [0.3.1] - 2026-01-31

### Added
- **Pipeline Configuration Serialization**: Full YAML/JSON export/import for `RadiomicsPipeline` configurations:
    - `save_configs()` / `load_configs()`: File-based configuration persistence
    - `to_yaml()` / `from_yaml()`: String-based YAML serialization
    - `to_json()` / `from_json()`: String-based JSON serialization
    - `to_dict()` / `from_dict()`: Dictionary conversion for programmatic use
- **Configuration Management Methods**:
    - `add_config()`: Register custom configurations
    - `get_config()`: Retrieve configuration by name (deep copy)
    - `remove_config()`: Delete configurations
    - `list_configs()`: List all registered configuration names
    - `merge_configs()`: Combine configurations from multiple pipelines
- **Template System**: YAML-based configuration templates in `pictologics/templates/`:
    - Standard configurations now loaded from `standard_configs.yaml`
    - Template loading API: `list_template_files()`, `load_template_file()`, `get_standard_templates()`, `get_all_templates()`, `get_template_metadata()`
- **Schema Versioning**: Configuration files include `schema_version` for forward compatibility and automatic migration
- **Configuration Validation**: Opt-in validation via `validate=True` parameter logs warnings for unknown steps/parameters
- **Documentation**: New "Predefined Configurations" user guide page with comprehensive examples including end-to-end multi-site study workflow

### Changed
- Standard configurations (`standard_fbn_*`, `standard_fbs_*`) now loaded from YAML templates instead of hardcoded dictionaries
- Updated pipeline.md documentation with condensed configuration section and cross-references

### Dependencies
- Added `pyyaml>=6.0` as core dependency for YAML serialization

---

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

[0.3.3]: https://github.com/martonkolossvary/pictologics/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/martonkolossvary/pictologics/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/martonkolossvary/pictologics/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/martonkolossvary/pictologics/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/martonkolossvary/pictologics/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/martonkolossvary/pictologics/releases/tag/v0.1.0
