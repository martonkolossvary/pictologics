# Welcome to Pictologics

[![Tests](assets/badges/tests.svg)](quality.md)
[![Coverage](assets/badges/coverage.svg)](quality.md)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)

![Pictologics Icon](assets/logo.png){ align=right width=200 }

**Pictologics** is a pure python, IBSI-compliant library for radiomic feature extraction from medical images.

See also the [NOTICE](NOTICE.md) file for attribution and third-party library information.

## Why Pictologics?

*   **🚀 High Performance**: Uses `numba` for JIT compilation, achieving significant speedups over other libraries (speedups between 15-300x compared to pyradiomics, see [Benchmarks](benchmarks.md) page for details).
*   **✅ IBSI Compliant**: Implements standard algorithms verified against the IBSI digital and CT phantom ([IBSI compliance](ibsi_compliance.md) page for details).
*   **🔧 Flexible**: Configurable pipeline for reproducible research.
*   **✨ Easy to Use**: Simple installation and a straightforward pipeline make it easy to get started quickly.
*   **🛠️ Actively Maintained**: Continuously maintained and developed with the intention to provide robust latent radiomic features that can reliably describe morphological characteristics of diseases on radiological images.


## Key Features

*   **Loaders**: Support for NIfTI and DICOM image formats.
*   **Preprocessing**: Resampling, resegmentation, outlier filtering, and discretisation.
*   **Features**:
    *   **Morphology**: Volume, Surface Area, Compactness, etc.
    *   **Intensity**: Mean, Median, Skewness, Kurtosis, etc.
    *   **Texture**: GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM.

## Getting Started

1.  **Install**: Follow the [Installation](user_guide/installation.md) guide.
2.  **Learn**: Check the [Quick Start](user_guide/quickstart.md) tutorial.
3.  **Reference**: Explore the [API Documentation](api/pipeline.md).
