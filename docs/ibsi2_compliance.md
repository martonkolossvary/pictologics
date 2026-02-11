# IBSI 2 Compliance: Convolutional Filters

## Overview

The Image Biomarker Standardisation Initiative Chapter 2 (IBSI 2) focuses on standardizing convolutional image filters for radiomics. This page documents Pictologics' compliance with **IBSI 2 Phase 1**: technical validation using digital phantoms.

!!! important
    **Pictologics implements 3D filters and 3D radiomic features only.**
    
    The library is designed specifically for volumetric medical imaging analysis (CT, MRI, PET). 2D slice-by-slice processing is not supported as it loses critical spatial information needed for accurate radiomic feature extraction.



## How to Run the Benchmarks

### 1. Download the Data

The IBSI 2 reference datasets (digital phantoms) are available on the [IBSI GitHub repository](https://github.com/theibsi/data_sets).

- **Digital Phantoms**: Download the phantom NIfTI files (e.g., `checkerboard.nii.gz`, `impulse_response.nii.gz`) from the `ibsi_2_validation` folder.

Place these files in a local directory (e.g., `data/ibsi2/`) to run the benchmarks.

### 2. Run Configurations Programmatically using `RadiomicsPipeline`

You can run IBSI 2 filter configurations programmatically using the `RadiomicsPipeline` class.

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()

# Define an IBSI 2 Gabor filter configuration
# (Gabor filter, orthogonal rotation invariant, 3D)
gabor_config = [
    # 1. IBSI 2 Preprocessing
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0), "interpolation": "cubic"}},
    {"step": "round_intensities", "params": {}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    
    # 2. Apply Gabor Filter (Phase 1 Validation)
    {"step": "filter", "params": {
        "type": "gabor",
        "sigma_mm": 5.0,
        "lambda_mm": 4.0, 
        "gamma": 0.5,
        "rotation_invariant": True,
        "pooling": "average"
    }},
    
    # 3. Extract Intensity Features from the response map
    {"step": "extract_features", "params": {"families": ["intensity"]}}
]

pipeline.add_config("ibsi2_gabor_demo", gabor_config)

# Run on an image
results = pipeline.run("path/to/phantom.nii.gz", config_names=["ibsi2_gabor_demo"])
print(results["ibsi2_gabor_demo"])
```

!!! note
    The example above shows a **Gabor filter** configuration. This is just one example. You can configure any IBSI 2 compliant filter (Mean, LoG, Laws, Wavelet, etc.) similarly. For full specifications of filter parameters, please refer to the [Image Filtering](user_guide/image_filtering.md) guide and the [IBSI 2 Reference Manual](https://arxiv.org/abs/2006.05470).

## Phase 1 Results

### Filter Performance Overview

| Test | Filter | Phantom | Error % | Time | Memory | Status |
|------|--------|---------|---------|------|--------|--------|
| 1.a.1 | Mean | checkerboard | 0.00% | 2ms | 1.0MB | ✅ PASS |
| 1.a.2 | Mean | checkerboard | 0.00% | 2ms | 1.0MB | ✅ PASS |
| 1.a.3 | Mean | checkerboard | 0.00% | 2ms | 1.0MB | ✅ PASS |
| 1.a.4 | Mean | checkerboard | 0.00% | 2ms | 1.0MB | ✅ PASS |
| 1.b.1 | Mean (2D) | impulse_response | - | - | - | ⏭ SKIP |
| 2.a | LoG | impulse_response | 0.00% | 10ms | 3.0MB | ✅ PASS |
| 2.b | LoG | checkerboard | 0.03% | 12ms | 3.0MB | ✅ PASS |
| 2.c | LoG (2D) | checkerboard | - | - | - | ⏭ SKIP |
| 3.a.1 | Laws | impulse_response | 0.00% | 3ms | 2.0MB | ✅ PASS |
| 3.a.2 | Laws | impulse_response | 0.00% | 52ms | 4.0MB | ✅ PASS |
| 3.a.3 | Laws | impulse_response | 0.00% | 51ms | 4.0MB | ✅ PASS |
| 3.b.1 | Laws | checkerboard | 0.00% | 2ms | 2.0MB | ✅ PASS |
| 3.b.2 | Laws | checkerboard | 0.00% | 48ms | 4.0MB | ✅ PASS |
| 3.b.3 | Laws | checkerboard | 0.00% | 53ms | 4.0MB | ✅ PASS |
| 3.c.1 | Laws (2D) | checkerboard | - | - | - | ⏭ SKIP |
| 3.c.2 | Laws (2D) | checkerboard | - | - | - | ⏭ SKIP |
| 3.c.3 | Laws (2D) | checkerboard | - | - | - | ⏭ SKIP |
| 4.a.1 | Gabor | impulse_response | 0.27% | 27ms | 14.6MB | ✅ PASS |
| 4.a.2 | Gabor | impulse_response | 0.14% | 393ms | 21.5MB | ✅ PASS |
| 4.b.1 | Gabor | sphere | 0.01% | 31ms | 36.5MB | ✅ PASS |
| 4.b.2 | Gabor | sphere | 0.09% | 926ms | 48.2MB | ✅ PASS |
| 5.a.1 | Daubechies 2 | impulse_response | 0.00% | 3ms | 3.0MB | ✅ PASS |
| 5.a.2 | Daubechies 2 | impulse_response | 0.00% | 65ms | 6.0MB | ✅ PASS |
| 6.a.1 | Coiflet 1 | sphere | 0.00% | 3ms | 3.0MB | ✅ PASS |
| 6.a.2 | Coiflet 1 | sphere | 0.00% | 72ms | 6.0MB | ✅ PASS |
| 7.a.1 | Haar | checkerboard | 0.00% | 111ms | 6.0MB | ✅ PASS |
| 7.a.2 | Haar | checkerboard | 0.00% | 109ms | 6.0MB | ✅ PASS |
| 8.a.1 | Simoncelli | checkerboard | 0.38% | 9ms | 24.3MB | ✅ PASS |
| 8.a.2 | Simoncelli | checkerboard | 0.00% | 9ms | 24.3MB | ✅ PASS |
| 8.a.3 | Simoncelli | checkerboard | 0.00% | 9ms | 24.3MB | ✅ PASS |
| 9.a | Riesz-LoG | impulse_response | 0.05% | 13ms | 14.4MB | ✅ PASS |
| 9.b.1 | Riesz-LoG | sphere | 0.64% | 13ms | 14.4MB | ✅ PASS |
| 9.b.2 | Riesz-LoG (aligned) | sphere | - | - | - | ❗ REF. |
| 10.a | Riesz-Simoncelli | impulse_response | - | - | - | ❗ REF. |
| 10.b.1 | Riesz-Simoncelli | pattern_1 | 0.79% | 12ms | 24.3MB | ✅ PASS |
| 10.b.2 | Riesz-Simoncelli (aligned) | pattern_1 | - | - | - | ❗ REF. |

### Tolerance Criteria

All tests use the IBSI 2 standard tolerance:
```
max_difference ≤ 0.01 × (reference_max - reference_min)
```

## Known Deviations

### 2D Filters Not Implemented (5 Tests Skipped)

!!! note
    **Design Decision: 3D Volumetric Processing Only**
    
    Pictologics implements **only 3D convolutional filters** and **only 3D radiomic features only**. This is a deliberate design choice for clinical radiomics workflows with volumetric medical imaging data (CT, MRI, PET scans).

The following tests are intentionally skipped because they require 2D filter implementations: **Test 1.b.1** (Mean Filter 2D), **Test 2.c** (LoG 2D), and **Tests 3.c.1-3.c.3** (Laws 2D).

### Structure Tensor Alignment (Reference Missing)

The following tests are currently not implemented because they require structure tensor alignment **and the IBSI 2 reference dataset does not contain the corresponding validity response maps**:

- **Tests 9.b.2, 10.b.2**: Riesz Filter Alignment (ValidCRM missing)
- **Test 10.a**: Riesz-Simoncelli Alignment (ValidCRM missing)

!!! warning
    **Reference Data Unavailable**
    
    The official IBSI 2 reference dataset **does not contain** the reference validity maps for these tests (`9_b_2-ValidCRM.nii`, `10_a-ValidCRM.nii`, `10_b_2-ValidCRM.nii`). Therefore, these tests cannot be validated and are marked as **❗ REF.** (Reference Missing).

### Summary

In total, **28 tests passed** validating all core 3D filter functionality. **5 tests were skipped** as they relate to 2D filters which are not applicable to this 3D-focused library. **3 tests are marked as missing reference** (structure tensor alignment) because the validation data is not provided by IBSI.
