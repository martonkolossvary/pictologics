# Welcome to Pictologics

[![CI](https://github.com/martonkolossvary/pictologics/actions/workflows/ci.yml/badge.svg)](https://github.com/martonkolossvary/pictologics/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://martonkolossvary.github.io/pictologics/)
[![PyPI](https://img.shields.io/pypi/v/pictologics)](https://pypi.org/project/pictologics/)
[![Python](https://img.shields.io/pypi/pyversions/pictologics)](https://pypi.org/project/pictologics/)
[![Downloads](https://img.shields.io/pepy/dt/pictologics)](https://pypi.org/project/pictologics/)
[![License](https://img.shields.io/github/license/martonkolossvary/pictologics)](https://github.com/martonkolossvary/pictologics/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/martonkolossvary/pictologics/graph/badge.svg)](https://codecov.io/gh/martonkolossvary/pictologics)
[![Ruff](https://img.shields.io/badge/ruff-0%20issues-261230.svg)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/mypy-0%20errors-blue.svg)](https://mypy-lang.org/)


![Pictologics Icon](assets/logo.png){ align=right width=200 }

**Pictologics** is a pure python, IBSI-compliant library for radiomic feature extraction from medical images.

See also the [NOTICE](NOTICE.md) file for attribution and third-party library information.

## Why Pictologics?

*   **🚀 High Performance**: Uses `numba` for JIT compilation, achieving significant speedups over other libraries (speedups between 15-300x compared to pyradiomics, see [Benchmarks](benchmarks.md) page for details).
*   **✅ IBSI Compliant**: Implements standard algorithms verified against the IBSI digital and CT phantom ([IBSI compliance](ibsi_compliance.md) page for details).
*   **🔧 Flexible**: Configurable pipeline for reproducible research. Provides utilities for DICOM parsing, organization and common image processing tasks.
*   **✨ Easy to Use**: Pure python package, simple installation and a straightforward pipeline make it easy to get started quickly.
*   **🛠️ Actively Maintained**: Continuously maintained and developed with the intention to provide robust latent radiomic features that can reliably describe morphological characteristics of diseases on radiological images.


## Key Features

*   **Loaders**: Support for NIfTI and DICOM image formats.
*   **Preprocessing**: Resampling, resegmentation, outlier filtering, and discretisation.
*   **Features**:
    *   **Morphology**: Volume, Surface Area, Compactness, etc.
    *   **Intensity**: Mean, Median, Skewness, Kurtosis, etc.
    *   **Texture**: GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM.
*   **Utilities**: Built-in DICOM database parsing and organization.

## Getting Started

1.  **Install**: Follow the [Installation](user_guide/installation.md) guide.
2.  **Learn**: Check the [Quick Start](user_guide/quickstart.md) tutorial.
3.  **Reference**: Explore the [API Documentation](api/pipeline.md).
