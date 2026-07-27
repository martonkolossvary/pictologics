# IBSI 2 Compliance: Convolutional Filters

## Overview

The Image Biomarker Standardisation Initiative Chapter 2 (IBSI 2) focuses on standardizing convolutional image filters for radiomics. This page documents Pictologics' compliance with **IBSI 2 Phase 1**: technical validation using digital phantoms.

!!! important
    **Pictologics accepts volumetric images and calculates radiomic features over a 3D ROI.** Its IBSI Gabor implementation uses 2D plane kernels and can average their response maps over three orthogonal planes; this is not a native 3D Gabor convolution, and it does not imply support for a general slice-wise 2D radiomics workflow.
    
    The library is designed specifically for volumetric medical imaging analysis (CT, MRI, PET). A slice-by-slice 2D radiomics workflow is not supported, as it loses spatial information needed for accurate volumetric feature extraction.



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

# Define a Gabor filter configuration (generic example)
# 2D plane kernels, rotation invariant, averaged over the
# three orthogonal planes; features are calculated over the 3D ROI.
gabor_config = [
    # 1. IBSI 2 Preprocessing
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0), "interpolation": "cubic"}},
    {"step": "round_intensities", "params": {}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    
    # 2. Apply Gabor filter
    #    Generic illustrative settings - NOT an official IBSI test
    #    configuration. rotation_invariant=True requires delta_theta.
    {"step": "filter", "params": {
        "type": "gabor",
        "sigma_mm": 5.0,
        "lambda_mm": 4.0,
        "gamma": 0.5,
        "rotation_invariant": True,
        "delta_theta": 0.7853981633974483,  # pi/4
        "average_over_planes": True,
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

### 3. Reproduce an Exact IBSI Phase 1 Test (Test 8.a.1)

The example above is a **generic illustration** of the `RadiomicsPipeline` API, not an official IBSI test. The example below is different: it is one **exact, named IBSI 2 Phase 1 test** — reproduced precisely as the compliance harness (`dev/IBSI2/verify_ibsi2_compliance.py`) runs it — that you can check yourself against the official reference.

- **Official test ID**: `8.a.1` (Simoncelli wavelet, Table 6.1)
- **Phantom file**: `checkerboard.nii.gz` (see [Download the Data](#1-download-the-data) above)
- **Preprocessing**: none. The harness loads the phantom NIfTI directly (`nib.load(...).get_fdata().astype(np.float32)`) and passes it straight to the filter — there is no resampling, intensity rounding, or resegmentation for Phase 1 filter tests.
- **Exact filter call**: `simoncelli_wavelet(phantom, level=1)` — `boundary` defaults to `BoundaryCondition.PERIODIC`, the filter's inherent FFT-based boundary handling.
- **Comparison rule**: `max_diff = abs(response - reference).max()`; the test passes if `max_diff <= 0.01 * (reference.max() - reference.min())` — the same tolerance basis given in [Tolerance Criteria](#tolerance-criteria) below.
- **Reference/response map**: `8_a_1-ValidCRM.nii`, obtained from the `reference_response_maps` subfolder of the [IBSI 2 reference data repository](https://github.com/theibsi/ibsi_2_reference_data); place it under `references/response_maps/` alongside the phantoms (the file validated below has sha256 short-hash `f26254a1dac5`; see [Provenance](#provenance)).

```python
import numpy as np
import nibabel as nib
from pictologics.filters import simoncelli_wavelet

# Official IBSI 2 Phase 1 Test 8.a.1 - exact, not illustrative
phantom = nib.load("data/phantoms/checkerboard.nii.gz").get_fdata().astype(np.float32)
response = simoncelli_wavelet(phantom, level=1)  # boundary defaults to PERIODIC

# Compare against the official IBSI reference response map
reference = nib.load("references/response_maps/8_a_1-ValidCRM.nii").get_fdata().astype(np.float32)
max_diff = np.abs(response - reference).max()
tolerance = 0.01 * (reference.max() - reference.min())
assert max_diff <= tolerance  # IBSI 2 Phase 1 pass criterion
```

## Phase 1 Results

### Filter Performance Overview

| Test | Filter | Phantom | Error % | Time | Memory | Status |
|------|--------|---------|---------|------|--------|--------|
| 1.a.1 | Mean | checkerboard | 0.00% | 2ms | 2.0MB | ✅ PASS |
| 1.a.2 | Mean | checkerboard | 0.00% | 2ms | 2.0MB | ✅ PASS |
| 1.a.3 | Mean | checkerboard | 0.00% | 2ms | 2.0MB | ✅ PASS |
| 1.a.4 | Mean | checkerboard | 0.00% | 2ms | 2.0MB | ✅ PASS |
| 1.b.1 | Mean (2D) | impulse_response | - | - | - | ⏭ SKIP |
| 2.a | LoG | impulse_response | 0.00% | 10ms | 3.0MB | ✅ PASS |
| 2.b | LoG | checkerboard | 0.03% | 12ms | 3.0MB | ✅ PASS |
| 2.c | LoG (2D) | checkerboard | - | - | - | ⏭ SKIP |
| 3.a.1 | Laws | impulse_response | 0.00% | 2ms | 2.0MB | ✅ PASS |
| 3.a.2 | Laws | impulse_response | 0.00% | 16ms | 9.0MB | ✅ PASS |
| 3.a.3 | Laws | impulse_response | 0.00% | 19ms | 13.0MB | ✅ PASS |
| 3.b.1 | Laws | checkerboard | 0.00% | 3ms | 2.0MB | ✅ PASS |
| 3.b.2 | Laws | checkerboard | 0.00% | 16ms | 9.0MB | ✅ PASS |
| 3.b.3 | Laws | checkerboard | 0.00% | 18ms | 12.0MB | ✅ PASS |
| 3.c.1 | Laws (2D) | checkerboard | - | - | - | ⏭ SKIP |
| 3.c.2 | Laws (2D) | checkerboard | - | - | - | ⏭ SKIP |
| 3.c.3 | Laws (2D) | checkerboard | - | - | - | ⏭ SKIP |
| 4.a.1 | Gabor | impulse_response | 0.27% | 19ms | 12.0MB | ✅ PASS |
| 4.a.2 | Gabor | impulse_response | 0.14% | 109ms | 18.5MB | ✅ PASS |
| 4.b.1 | Gabor | sphere | 0.01% | 26ms | 31.5MB | ✅ PASS |
| 4.b.2 | Gabor | sphere | 0.09% | 258ms | 46.7MB | ✅ PASS |
| 5.a.1 | Daubechies 2 | impulse_response | 0.00% | 3ms | 2.0MB | ✅ PASS |
| 5.a.2 | Daubechies 2 | impulse_response | 0.00% | 68ms | 4.0MB | ✅ PASS |
| 6.a.1 | Coiflet 1 | sphere | 0.00% | 3ms | 2.0MB | ✅ PASS |
| 6.a.2 | Coiflet 1 | sphere | 0.00% | 74ms | 4.0MB | ✅ PASS |
| 7.a.1 | Haar | checkerboard | 0.00% | 120ms | 5.0MB | ✅ PASS |
| 7.a.2 | Haar | checkerboard | 0.00% | 124ms | 5.0MB | ✅ PASS |
| 8.a.1 | Simoncelli | checkerboard | 0.38% | 7ms | 12.3MB | ✅ PASS |
| 8.a.2 | Simoncelli | checkerboard | 0.00% | 6ms | 12.3MB | ✅ PASS |
| 8.a.3 | Simoncelli | checkerboard | 0.00% | 6ms | 12.3MB | ✅ PASS |
| 9.a | Riesz-LoG | impulse_response | 0.05% | 43ms | 49.4MB | ✅ PASS |
| 9.b.1 | Riesz-LoG | sphere | 0.32% | 43ms | 49.4MB | ✅ PASS |
| 9.b.2 | Riesz-LoG (aligned) | sphere | - | - | - | ❗ REF. |
| 10.a | Riesz-Simoncelli | impulse_response | - | - | - | ❗ REF. |
| 10.b.1 | Riesz-Simoncelli | pattern_1 | 0.21% | 39ms | 71.1MB | ✅ PASS |
| 10.b.2 | Riesz-Simoncelli (aligned) | pattern_1 | - | - | - | ❗ REF. |

### Tolerance Criteria

All tests use the IBSI 2 standard tolerance:
```
max_difference ≤ 0.01 × (reference_max - reference_min)
```

### Provenance

!!! info "Reproducibility Provenance"
    - **Pictologics version**: `0.5.1`
    - **IBSI 2 reference manual**: version 9 (the revision bundled under `dev/IBSI2/documentation/`)
    - **Reference dataset source**: `reference_response_maps` subfolder of the [IBSI 2 reference data repository](https://github.com/theibsi/ibsi_2_reference_data)
    - **Local reference directory**: `/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/Pictologics/Pictologics/dev/IBSI2/references/response_maps`
    - Per-test reference filenames and short (12-character) SHA-256 content hashes are recorded for all 28 compared tests (passed or failed) in the JSON results this script writes (`--output <file>.json`), keyed by test ID under `ref_file` / `ref_sha256` (e.g. Test `8.a.1` was validated against `8_a_1-ValidCRM.nii`, sha256 short-hash `f26254a1dac5`).

## Known Deviations

### 2D Filters Not Implemented (5 Tests Skipped)

!!! note
    **Design Decision: 3D Volumetric Processing Only**
    
    Pictologics calculates radiomic features over a **3D volumetric ROI** and does not provide a general slice-wise 2D radiomics workflow. This is a deliberate design choice for clinical radiomics workflows with volumetric medical imaging data (CT, MRI, PET scans). Note that the Gabor filter is an explicit exception at the *filter* level: it applies 2D plane kernels and can average the resulting response maps over three orthogonal planes, as IBSI defines. Feature statistics are still calculated over the volumetric ROI.

The following tests are intentionally skipped because they require 2D filter implementations: **Test 1.b.1** (Mean Filter 2D), **Test 2.c** (LoG 2D), and **Tests 3.c.1-3.c.3** (Laws 2D).

### Structure Tensor Alignment (Reference Missing)

The following tests are currently not implemented because they require structure tensor alignment (steering the Riesz filter response towards the local dominant orientation) **and the IBSI 2 reference dataset does not contain the corresponding validity response maps**:

- **Tests 9.b.2, 10.b.2**: Riesz Filter Alignment (ValidCRM missing)

!!! warning
    **Reference Data Unavailable**
    
    The official IBSI 2 reference dataset **does not contain** the reference validity maps for these tests (`9_b_2-ValidCRM.nii`, `10_b_2-ValidCRM.nii`). Therefore, these tests cannot be validated and are marked as **❗ REF.** (Reference Missing).

### Reference Missing (Unsteered Riesz-Simoncelli)

Unlike the two tests above, **Test 10.a is not a steering test**: it is an implemented, unsteered Riesz-Simoncelli response map (`riesz_simoncelli(level=1, order=(1, 0, 0))`, no structure-tensor alignment). It is grouped separately here because the IBSI 2 reference dataset simply does not distribute a validity response map for it (`10_a-ValidCRM.nii` is absent), so it cannot be validated for a reason unrelated to structure-tensor alignment:

- **Test 10.a**: unsteered Riesz-Simoncelli response map (ValidCRM missing)

### Summary

Under **strict 3D validation** (excluding the IBSI-defined 2D Gabor response-map tests), **24 tests passed**. Including the **4 2D Gabor tests** (`4.a.1`, `4.a.2`, `4.b.1`, `4.b.2` — computed via 2D plane kernels, optionally averaged over the three orthogonal planes, as Pictologics implements them), **28 of 28 compared tests passed** in total; these are the same underlying runs, only split by category, not a separate claim. **5 tests were skipped** as they relate to 2D filters (Mean, LoG, Laws) which are not applicable to this 3D-focused library. **3 tests are marked as missing reference** because the IBSI 2 reference dataset does not distribute a validity response map for them — two because they require structure-tensor alignment (`9.b.2`, `10.b.2`), and one unsteered test whose reference map is simply absent (`10.a`).
