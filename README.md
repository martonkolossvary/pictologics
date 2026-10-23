# Pictologics

<p align="center">
    <img src="https://raw.githubusercontent.com/martonkolossvary/pictologics/main/docs/assets/logo.png" width="220" alt="Pictologics logo" />
</p>

[![CI](https://github.com/martonkolossvary/pictologics/actions/workflows/ci.yml/badge.svg)](https://github.com/martonkolossvary/pictologics/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://martonkolossvary.github.io/pictologics/)
[![PyPI](https://img.shields.io/pypi/v/pictologics)](https://pypi.org/project/pictologics/)
[![Python](https://img.shields.io/pypi/pyversions/pictologics)](https://pypi.org/project/pictologics/)
[![Downloads](https://img.shields.io/pypi/dm/pictologics)](https://pypi.org/project/pictologics/)
[![License](https://img.shields.io/github/license/martonkolossvary/pictologics)](https://github.com/martonkolossvary/pictologics/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/martonkolossvary/pictologics/graph/badge.svg)](https://codecov.io/gh/martonkolossvary/pictologics)
[![Ruff](https://img.shields.io/badge/ruff-0%20issues-261230.svg)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/mypy-0%20errors-blue.svg)](https://mypy-lang.org/)

**Pictologics** is a Python library for radiomic feature extraction from medical images. It follows IBSI 1 (the features) and IBSI 2 (the filters), and it checks its results against the IBSI reference values.

Documentation (user guide, tutorials, API, benchmarks): https://martonkolossvary.github.io/pictologics/

## Why Pictologics?

*   **🚀 Fast**: Numba compiles the computations for your computer, and they run on its fast cores. The [Benchmarks](https://martonkolossvary.github.io/pictologics/benchmarks/) page gives the time of each part.
*   **✅ IBSI compliant**: checked against the IBSI phantoms and data sets:
    *   **IBSI 1** (features): 675 feature values pass; 21 more have a reference value without a tolerance ([report](https://martonkolossvary.github.io/pictologics/ibsi1_compliance/)).
    *   **IBSI 2 Phase 1** (filters): 28 of 28 compared tests pass ([report](https://martonkolossvary.github.io/pictologics/ibsi2_compliance/)).
    *   **IBSI 2 Phase 2** (filtered features): 9 of 9 tests pass ([report](https://martonkolossvary.github.io/pictologics/ibsi2_phase2_compliance/)).
    *   **IBSI 2 Phase 3** (reproducibility): 153 scans, compared with 9 teams ([report](https://martonkolossvary.github.io/pictologics/ibsi2_phase3_compliance/)).
*   **🔧 Versatile**: reads NIfTI, NRRD, MetaImage and DICOM images, DICOM SEG, DICOM RTSTRUCT and 3D Slicer segmentations, and DICOM SR reports.
*   **✨ Easy to use**: pip installs it. One pipeline runs the preprocessing and the features, for one image or for a whole study.
*   **🛡️ Predictable results**: each configuration gives all its feature columns, also when a step fails (the values are then `NaN`). A study table never has missing columns.
*   **🛠️ Maintained**: Pictologics is developed to give robust radiomic features that describe the morphology of diseases on radiological images.

## What Is New in 0.9.0

- **Speed**: on a 512×512×200 CT, timed by turns with 0.8.0 on one computer, the configuration `standard_fbn_32` takes 9 % less time, the six standard configurations 14 % less, and a DICOM series loads 13 % faster. Against 0.7.1 that is 19 % and 31 % less for the two configurations.
- **Filters**: `riesz_log` with a zero, nearest or mirror boundary does the LoG and the Riesz transform in one FFT round trip: 0.40 of the time and 0.39 of the peak memory. The Simoncelli and Riesz filters pad to fast FFT sizes (a Riesz-Simoncelli filter with the mirror boundary in 0.23 of the time), make their inverse FFT without SciPy's copy of the spectrum (211 MB instead of 782 MB on that CT), and build their tables in parallel kernels (16 ms instead of 196 ms). The filters with a source mask run in numba kernels: the masked mean filter takes 13 ms instead of 39 ms.
- **Textures**: one compiled kernel for each family, with the pairwise sums of numpy, gives the same bits. An ROI of 1 to 3,000 voxels takes 0.27 to 0.58 of the time; `run_rois` of 200 small ROIs with the texture features takes 35 ms instead of 71 ms. The zone pass keeps 12 bytes for each voxel instead of 20, so the peak memory of the texture features drops by about 30 %, and large zone tables count in threads.
- **Preprocessing**: `resegment_mask` takes 3.1 ms instead of 18.7 ms; `keep_largest_component` of a full mask 7 ms instead of 440 ms; `grow_mask` 3.0 ms instead of 8.9 ms; `normalise_image` half the memory; the sentinel search and the source mask one pass each; `extract_roi` one pass; the resample kernels run at every output size and round in the kernel.
- **Files**: a mask or image that `load_image` turns to another voxel order copies on a tiled kernel (21 ms instead of 140 ms); the NRRD and MetaImage readers read into the array (8 % to 38 % less memory); a DICOM series reads the next slab ahead and reads the SOP class once; `load_rtstruct` fills only the box of each contour (24 ms instead of 73 ms); `load_and_merge_images` merges in one pass (0.54 s instead of 1.41 s for 20 masks); the DICOM database scans a folder with one walk (3.0 s instead of 3.9 s for 6,000 files).
- **Features**: the first-order moments and deviations of large ROIs come from one pass in 64 fixed parts (2.5 ms instead of 3.9 ms, the same values at every number of threads); the energy sums in threads; the ROI range skips the rows without ROI voxels; Moran's I and Geary's C choose their method with a fixed count of 4 threads, so the choice and the values are the same on every computer.
- **Fixes**: a ROI whose values are all equal gives exactly 0 for the variance, the absolute deviations and the coefficient of variation; `from_folders` keeps a DICOM file with one unreadable header element; a strided array gives the values of its copy; a nullable Series gives Python numbers in the wide format; the NaN check of a run gives no warning; the GLDZM raises on a distance map value outside its range.
- **Changes**: the non-periodic boundaries of `riesz_log`, `simoncelli_wavelet`, `riesz_transform` and `riesz_simoncelli` move values toward the boundary condition (the IBSI 2 tests 9.a and 9.b.1 pass with the same errors; the periodic default does not change); the first-order features of ROIs of 524,288 values or more move by less than 1e-10; the spatial method of some ROIs of 2,000 to 5,000 voxels flips, with moves of at most 4e-15. The IBSI compliance is the same. See the [Changelog](https://martonkolossvary.github.io/pictologics/CHANGELOG/).

## Installation

Pictologics needs Python 3.12, 3.13 or 3.14.

```bash
pip install pictologics
```

From the source code:

```bash
git clone https://github.com/martonkolossvary/pictologics.git
cd pictologics
pip install .
```

## Quick Start

```python
from pictologics import RadiomicsPipeline, format_results, save_results

# 1. A pipeline with the six standard configurations
pipeline = RadiomicsPipeline()

# 2. Run the configurations on an image and its mask
results = pipeline.run(
    image="path/to/ct.nii.gz",
    mask="path/to/mask.nii.gz",
    subject_id="Subject_001",
    config_names=["all_standard"],  # the FBS configurations start at -1000 HU, for CT
)

# 3. One table row, with your own columns first
row = format_results(results, fmt="wide", meta={"subject_id": "Subject_001", "group": "control"})

# 4. Save the row to a CSV file
save_results([row], "results.csv")
```

For MR or PET images, use the FBN configurations (`config_names=["standard_fbn_32"]`) or your own. See the [Quick Start](https://martonkolossvary.github.io/pictologics/user_guide/quick_start/) and the [Cookbook](https://martonkolossvary.github.io/pictologics/user_guide/cookbook/).

## Performance Benchmarks

| Task | Size | Time (median) |
|:--|:--|--:|
| run(): one standard configuration (standard_fbn_32) | CT of 512 × 512 × 200 | 8.4 ms |
| run(): the 6 standard configurations | CT of 512 × 512 × 200 | 29.2 ms |
| Texture (all 6 families) | 2,311,384 ROI voxels | 17.7 ms |
| Morphology | 2,311,384 ROI voxels | 4.2 ms |
| LoG (sigma 2 mm) | 256³ voxels | 37.7 ms |
| Gabor (axial, rotation invariant) | 256³ voxels | 52.1 ms |

The median of 5 runs on one computer, after a warm-up run. See the [benchmark page](https://martonkolossvary.github.io/pictologics/benchmarks/) for all results, plots and the computer.


## Quality & Compliance

**IBSI Compliance**: [IBSI 1 Features](https://martonkolossvary.github.io/pictologics/ibsi1_compliance/) | [IBSI 2 Phase 1 Filters](https://martonkolossvary.github.io/pictologics/ibsi2_compliance/) | [Phase 2 Features](https://martonkolossvary.github.io/pictologics/ibsi2_phase2_compliance/) | [Phase 3 Reproducibility](https://martonkolossvary.github.io/pictologics/ibsi2_phase3_compliance/)

### Code Health

- **Test Coverage**: 100.00%
- **Mypy Errors**: 0
- **Ruff Issues**: 0

See [Quality Report](https://martonkolossvary.github.io/pictologics/quality/) for full details.

## Citation

Citation information will be added/updated.

## License

Apache-2.0
