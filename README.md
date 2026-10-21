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

## What Is New in 0.8.0

- **Speed**: on a 512×512×200 CT, timed by turns with 0.7.1 on one computer, the configuration `standard_fbn_32` takes 11 % less time, the six standard configurations 20 % less, and `run_rois` with 20 ROIs 12 % less. Against 0.6.0 that is 64 %, 56 % and 89 % less.
- **Resampling**: cubic resampling runs on numba kernels that give the output of scipy bit for bit: the IBSI 2 CT phantom to 1 mm takes about 20 ms instead of 900 ms. A column-order image, for example from nibabel's `get_fdata()`, resamples up to 10 times faster. The pipeline casts an in-memory float32 or integer image where it reads it: `run()` of a float32 CT with the intensity features takes 1 ms instead of 34 ms, with 1.8 MB instead of 400 MB.
- **Filters**: the mean, Gaussian, Laws and wavelet filters run their 1-D passes in numba kernels with the arithmetic of scipy: the same bits, 2 to 5 times faster on a 256³ image. A large LoG runs its FFT in slabs, 30 % faster with 30 % less memory. A filter that computes only the ROI region keeps that region, so a small ROI in a large CT costs much less time and memory.
- **Files**: `save_image` writes a 512×512×200 `.nii.gz` file in 0.8 s instead of 6.8 s, and `load_image` reads it 4 times faster. A mask keeps its stored integer type: 100 MB instead of 450 MB for a uint8 mask of that size. The DICOM database and SR scans of small batches run in the calling process: 1.2 s instead of 2.7 s for 2,000 files.
- **Features**: the histogram, the intensity-volume histogram, the local intensity peaks, the percentiles of small ROIs, Moran's I and Geary's C (a real-spectrum FFT, in slabs when the grid does not fit half of the memory), the GLCM and GLRLM row passes, and the morphology kernels (a third less peak memory, the convex hull from the voxel grid) run faster with the same bits.
- **Memory**: on Windows, the FFT of Moran's I and Geary's C no longer assumes 8 GiB of memory: a body ROI of a 512×512×200 CT takes 0.9 s instead of about 10 hours. Every computer may use half of its memory for it. The Simoncelli and Riesz table caches keep two image shapes: 182 MB instead of 872 MB after 21 scans.
- **Fixes**: the intensity mask stays inside the morphological mask after `keep_largest_component` or `binarize_mask` on the morphological mask only; the GLDZM raises on voxels outside the distance mask instead of dropping zones; a DICOM position written as text no longer fails the loader; a `.nii.gz` file of two gzip members no longer hangs; the convex hull volume no longer loses digits deep in a large grid; the histogram features of float32 values sum in float64; the filters check their parameters; and more (55 entries).
- **Changes**: every mask type takes one threshold rule in a linear mask resample (a few phase 3 values near zero move, with no verdict change); the MVEE stop rule moves `volume_density_mvee` and `area_density_mvee` by 0.2 to 0.5 %; the mean filter and the Laws energy sum each window on its own (within one float32 unit); the LoG of a float64 image is one FFT from 131,072 voxels; the energy sums in a fixed order. The IBSI compliance is the same. See the [Changelog](https://martonkolossvary.github.io/pictologics/CHANGELOG/).

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
| run(): one standard configuration (standard_fbn_32) | CT of 512 × 512 × 200 | 9.8 ms |
| run(): the 6 standard configurations | CT of 512 × 512 × 200 | 34.3 ms |
| Texture (all 6 families) | 2,311,384 ROI voxels | 20.0 ms |
| Morphology | 2,311,384 ROI voxels | 4.5 ms |
| LoG (sigma 2 mm) | 256³ voxels | 37.5 ms |
| Gabor (axial, rotation invariant) | 256³ voxels | 52.4 ms |

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
