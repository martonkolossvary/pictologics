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

*   **🚀 High Performance**: Uses `numba` for Just In Time (JIT) compilation, achieving significant speedups over other libraries (speedups between 15-300x compared to pyradiomics, see [Benchmarks](benchmarks.md) page for details).
*   **✅ IBSI Compliant**: Implements standard algorithms verified against the IBSI digital and CT phantom:
    *   **IBSI 1**: Feature extraction ([compliance report](ibsi_compliance.md))
    *   **IBSI 2**: Image filters ([filter compliance](ibsi2_compliance.md))
*   **🔧 Versatile**: Provides utilities for DICOM parsing and common scientific image processing tasks. Natively supports common image formats (NIfTI, DICOM, DICOM-SEG, DICOM-SR).
*   **✨ User-Friendly**: Pure Python implementation with a simple installation process and an intuitive API, ensuring a smooth experience from setup to analysis.
*   **🛠️ Actively Maintained**: Continuously maintained and developed with the intention to provide robust latent radiomic features that can reliably describe morphological characteristics of diseases on radiological images.


## Key Features

*   **Loaders**: Support for NIfTI and DICOM image, segmentation (DICOM-SEG), and report (DICOM-SR) formats.
*   **Preprocessing**: Resampling, resegmentation, outlier filtering, and discretisation.
*   **Features**:
    *   **Morphology**: Volume, Surface Area, Compactness, etc.
    *   **Intensity**: Mean, Median, Skewness, Kurtosis, etc.
    *   **Texture**: GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM.
*   **Filters**: IBSI 2-compliant convolutional filters including Mean, LoG, Laws, Gabor, Wavelets, and Simoncelli.
*   **Utilities**: Built-in DICOM database parsing, organization and viewing tools.

## Getting Started

1.  **Install**: Follow the [Installation](user_guide/installation.md) guide.
2.  **Learn**: Check the [Feature Calculations](user_guide/feature_calculations.md) guide.
3.  **Reference**: Explore the [API Documentation](api/pipeline.md).
