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

**Pictologics** is a high-performance, IBSI-compliant Python library for radiomic feature extraction from medical images (NIfTI, DICOM).

Documentation (User Guide, API, Benchmarks): https://martonkolossvary.github.io/pictologics/

## Why Pictologics?

*   **🚀 High Performance**: Uses `numba` for JIT compilation, achieving significant speedups over other libraries (speedups between 15-300x compared to pyradiomics, see [Benchmarks](https://martonkolossvary.github.io/pictologics/benchmarks/) page for details).
*   **✅ IBSI Compliant**: Implements standard algorithms verified against the IBSI digital and CT phantom ([IBSI compliance](https://martonkolossvary.github.io/pictologics/ibsi_compliance/) page for details).
*   **🔧 Flexible**: Configurable pipeline for reproducible research. Provides utilities for DICOM parsing and organization and common image processing tasks.
*   **👁️ Visualization**: Built-in utilities for visual quality control of mask overlays and segmentations.
*   **✨ Easy to Use**: Simple installation and a straightforward pipeline make it easy to get started quickly.
*   **🛠️ Actively Maintained**: Continuously maintained and developed with the intention to provide robust latent radiomic features that can reliably describe morphological characteristics of diseases on radiological images.

## Installation

Pictologics requires Python 3.12+.

```bash
pip install pictologics
```

Or install from source:

```bash
git clone https://github.com/martonkolossvary/pictologics.git
cd pictologics
pip install .
```

## Quick Start

```python
from pictologics import RadiomicsPipeline, format_results, save_results

# 1. Initialize the pipeline
pipeline = RadiomicsPipeline()

# 2. Run the "all_standard" configurations
results = pipeline.run(
    image="path/to/image.nii.gz",
    mask="path/to/mask.nii.gz",
    subject_id="Subject_001",
    config_names=["all_standard"]
)

# 3. Inject subject ID or other metadata directly into the row
row = format_results(
    results, 
    fmt="wide", 
    meta={"subject_id": "Subject_001", "group": "control"}
)

# 4. Save to CSV
save_results([row], "results.csv")
```

Benchmarks skipped.

## Quality & Compliance

**IBSI Compliance**: Full compliance (see [Report](https://martonkolossvary.github.io/pictologics/ibsi_compliance/)).

### Code Health

- **Test Coverage**: 100.00%
- **Mypy Errors**: 0
- **Ruff Issues**: 0

See [Quality Report](https://martonkolossvary.github.io/pictologics/quality/) for full details.

## Citation

Citation information will be added/updated.

## License

Apache-2.0
