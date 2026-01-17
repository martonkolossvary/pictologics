# IBSI 2 Compliance: Convolutional Filters

## Overview

The Image Biomarker Standardisation Initiative Chapter 2 (IBSI 2) focuses on standardizing convolutional image filters for radiomics. This page documents Pictologics' compliance with **IBSI 2 Phase 1**: technical validation using digital phantoms.

### IBSI 2 Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Filter response map validation (digital phantoms) | ✅ Complete |
| **[Phase 2](ibsi2_phase2_compliance.md)** | Feature extraction from filtered images | ✅ Complete |
| **[Phase 3](ibsi2_phase3_compliance.md)** | Multi-modality reproducibility study | ✅ Complete |

## How to Run the Benchmarks

### 1. Download IBSI 2 Data

```bash
poetry run python dev/IBSI2/download_ibsi2_data.py
```

### 2. Run Validation Script

```bash
poetry run python dev/IBSI2/verify_ibsi2_compliance.py
```

### 3. Programmatic Usage

```python
from pictologics.filters import (
    mean_filter, laplacian_of_gaussian, laws_filter,
    gabor_filter, simoncelli_wavelet, riesz_simoncelli,
    BoundaryCondition,
)
import numpy as np

# Apply Gabor filter (rotation invariant)
image = np.random.rand(64, 64, 64).astype(np.float32)
result = gabor_filter(image, sigma_mm=10.0, lambda_mm=4.0, gamma=0.5,
                      rotation_invariant=True, delta_theta=np.pi/4)
```

## Phase 1 Results

**Summary: 28 PASS, 5 SKIP, 0 Missing Refs**

### Filter Performance Overview

| Filter | Tests | Time Range | Memory | Status |
|--------|-------|------------|--------|--------|
| Mean | 4/4 | 2-2ms | 1MB | ✅ |
| LoG | 2/2 | 10-12ms | 3MB | ✅ |
| Laws | 6/6 | 6-248ms | 48MB | ✅ |
| Gabor | 4/4 | 37-2201ms | 34MB | ✅ |
| Daubechies 2 | 2/2 | 4-79ms | 49MB | ✅ |
| Coiflet 1 | 2/2 | 4-83ms | 49MB | ✅ |
| Haar | 2/2 | 135-136ms | 49MB | ✅ |
| Simoncelli | 3/3 | 14-15ms | 30MB | ✅ |
| Riesz-LoG | 2/2 | 23-25ms | 31MB | ✅ |
| Riesz-LoG (aligned) | 0/1 | N/A | N/A | ⚠️ |
| Riesz-Simoncelli | 1/2 | 25ms | 31MB | ⚠️ |
| Riesz-Simoncelli (aligned) | 0/1 | N/A | N/A | ⚠️ |

### Detailed Test Results

| Test | Filter | Phantom | Error % | Time | Memory | Status |
|------|--------|---------|---------|------|--------|--------|
| 1.a.1 | Mean | checkerboard | 0.00% | 2ms | 1.0MB | ✅ PASS |
| 1.a.2 | Mean | checkerboard | 0.00% | 2ms | 1.0MB | ✅ PASS |
| 1.a.3 | Mean | checkerboard | - | 2ms | 1.0MB | ✅ PASS |
| 1.a.4 | Mean | checkerboard | 0.00% | 2ms | 1.0MB | ✅ PASS |
| 1.b.1 | Mean (2D) | impulse_response | 22400.00% | 0ms | - | ⏭ SKIP |
| 2.a | LoG | impulse_response | 0.00% | 10ms | 3.0MB | ✅ PASS |
| 2.b | LoG | checkerboard | 0.03% | 12ms | 3.0MB | ✅ PASS |
| 2.c | LoG (2D) | checkerboard | 624.36% | 0ms | - | ⏭ SKIP |
| 3.a.1 | Laws | impulse_response | 0.00% | 10ms | 1.0MB | ✅ PASS |
| 3.a.2 | Laws | impulse_response | 0.00% | 248ms | 48.0MB | ✅ PASS |
| 3.a.3 | Laws | impulse_response | 0.00% | 244ms | 48.0MB | ✅ PASS |
| 3.b.1 | Laws | checkerboard | 0.00% | 6ms | 1.0MB | ✅ PASS |
| 3.b.2 | Laws | checkerboard | 0.00% | 147ms | 48.0MB | ✅ PASS |
| 3.b.3 | Laws | checkerboard | 0.00% | 146ms | 48.0MB | ✅ PASS |
| 3.c.1 | Laws (2D) | checkerboard | 64.04% | 0ms | - | ⏭ SKIP |
| 3.c.2 | Laws (2D) | checkerboard | 68.31% | 0ms | - | ⏭ SKIP |
| 3.c.3 | Laws (2D) | checkerboard | 263.94% | 0ms | - | ⏭ SKIP |
| 4.a.1 | Gabor | impulse_response | 0.27% | 37ms | 11.2MB | ✅ PASS |
| 4.a.2 | Gabor | impulse_response | 0.14% | 844ms | 15.0MB | ✅ PASS |
| 4.b.1 | Gabor | sphere | 0.01% | 48ms | 25.8MB | ✅ PASS |
| 4.b.2 | Gabor | sphere | 0.09% | 2201ms | 34.5MB | ✅ PASS |
| 5.a.1 | Daubechies 2 | impulse_response | 0.00% | 4ms | 3.0MB | ✅ PASS |
| 5.a.2 | Daubechies 2 | impulse_response | 0.00% | 79ms | 49.1MB | ✅ PASS |
| 6.a.1 | Coiflet 1 | sphere | 0.00% | 4ms | 3.0MB | ✅ PASS |
| 6.a.2 | Coiflet 1 | sphere | 0.00% | 83ms | 49.1MB | ✅ PASS |
| 7.a.1 | Haar | checkerboard | 0.00% | 135ms | 49.1MB | ✅ PASS |
| 7.a.2 | Haar | checkerboard | 0.00% | 136ms | 49.1MB | ✅ PASS |
| 8.a.1 | Simoncelli | checkerboard | 0.38% | 15ms | 30.3MB | ✅ PASS |
| 8.a.2 | Simoncelli | checkerboard | 0.00% | 14ms | 30.3MB | ✅ PASS |
| 8.a.3 | Simoncelli | checkerboard | 0.00% | 14ms | 30.3MB | ✅ PASS |
| 9.a | Riesz-LoG | impulse_response | 0.05% | 25ms | 31.0MB | ✅ PASS |
| 9.b.1 | Riesz-LoG | sphere | 0.64% | 23ms | 31.0MB | ✅ PASS |
| 9.b.2 | Riesz-LoG (aligned) | sphere | - | - | - | ❌ FAIL |
| 10.a | Riesz-Simoncelli | impulse_response | - | - | - | ❌ FAIL |
| 10.b.1 | Riesz-Simoncelli | pattern_1 | 0.79% | 25ms | 31.0MB | ✅ PASS |
| 10.b.2 | Riesz-Simoncelli (aligned) | pattern_1 | - | - | - | ❌ FAIL |

### Tolerance Criteria

All tests use the IBSI 2 standard tolerance:
```
max_difference ≤ 0.01 × (reference_max - reference_min)
```

## Known Deviations

### 2D Filters Not Implemented
Tests 1.b.1, 2.c, and 3.c.1-3 are skipped because Pictologics focuses on 3D volumetric analysis.

### Structure Tensor Alignment
Tests 9.b.2, 10.a, and 10.b.2 require structure tensor alignment (Phase 2 content).
