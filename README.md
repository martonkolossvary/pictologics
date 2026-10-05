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

## What Is New in 0.7.0

- **Speed**: on a 512×512×200 CT, timed by turns with 0.6.0 on one computer, the configuration `standard_fbn_32` takes 48 % less time, the six standard configurations 30 % less, and `run_rois` with 20 ROIs 60 % less.
- **Threads**: Pictologics uses the fast cores by default, for example the performance cores of Apple silicon. `set_num_threads()`, `get_num_threads()` and `PICTOLOGICS_NUM_THREADS` set one number for all parallel parts.
- **First run**: the import compiles every numba kernel that a run can use, so no code compiles during a run.
- **Installation**: Intel Macs, and 3D Slicer under Rosetta, can install Pictologics again (with numba 0.62). Matplotlib is now the optional extra `viz`: `pip install "pictologics[viz]"`. pandas 3, Pillow 12, SciPy 1.18 and numba 0.68 work.
- **Fixes**: the DICOM database scan, big-endian and multi-frame DICOM series, the frame positions of DICOM SEG files, NRRD byte skips, and the percentiles of large ROIs (a rare case).
- **Changes**: merged masks keep the type of their inputs (uint8 with `binarize`), and `save_slices` writes one pixel per voxel, with `dpi` as the tag of the file. Some texture and histogram values change in their last digits; the IBSI compliance is the same. See the [Changelog](https://martonkolossvary.github.io/pictologics/CHANGELOG/).

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
| run(): one standard configuration (standard_fbn_32) | CT of 512 × 512 × 200 | 13.8 ms |
| run(): the 6 standard configurations | CT of 512 × 512 × 200 | 54.5 ms |
| Texture (all 6 families) | 2,311,384 ROI voxels | 37.8 ms |
| Morphology | 2,311,384 ROI voxels | 14.0 ms |
| LoG (sigma 2 mm) | 256³ voxels | 144.4 ms |
| Gabor (axial, rotation invariant) | 256³ voxels | 114.9 ms |

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
