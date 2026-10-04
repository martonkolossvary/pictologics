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

## What Is New in 0.6.0

- **Studies**: `run_batch` runs many cases, in more than one process, with one result file for each case. A stopped batch goes on where it stopped. `run_rois` runs each ROI of a label map with one image load.
- **Masks**: `grow_mask` grows or shrinks a mask by a distance in mm, or keeps a ring, for example the fat around a vessel.
- **MR and PET**: the `normalise` step normalises MR intensities, and `load_image(..., suv="bw")` gives PET images in SUV.
- **Files**: new readers for NRRD, MetaImage, 3D Slicer `.seg.nrrd` and DICOM RTSTRUCT, and `save_image` writes NIfTI files.
- **Filters and features**: the Gaussian filter, constant padding for all filters, the parts of the Gabor response, and the IBSI texture distances.
- **Templates**: 60 configurations for cardiac CT (`lv` and `coronary`).
- **Reproducibility**: each log entry records the `config_hash` of its configuration and the versions of the run.
- **Breaking changes**: NIfTI geometry in the LPS+ frame, a fixed FBS start in every image, and the long-format column `feature_key`. See the [Changelog](https://martonkolossvary.github.io/pictologics/CHANGELOG/).

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
| run(): one standard configuration (standard_fbn_32) | CT of 512 × 512 × 200 | 27.7 ms |
| run(): the 6 standard configurations | CT of 512 × 512 × 200 | 80.0 ms |
| Texture (all 6 families) | 2,311,384 ROI voxels | 54.4 ms |
| Morphology | 2,311,384 ROI voxels | 19.5 ms |
| LoG (sigma 2 mm) | 256³ voxels | 179.9 ms |
| Gabor (axial, rotation invariant) | 256³ voxels | 105.8 ms |

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
