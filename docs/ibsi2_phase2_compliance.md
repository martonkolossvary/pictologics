# IBSI 2 Phase 2 Compliance: Radiomic Features

## Overview

IBSI 2 Phase 2 focuses on evaluating the **reproducibility of radiomic features extracted from filtered images**. In this phase, specific filters (Mean, LoG, Laws, Gabor, Wavelets) are applied to a digital phantom (Lung Cancer CT), and intensity features are extracted from the resulting response maps.

This page documents Pictologics' compliance with **IBSI 2 Phase 2 (Configuration B)**, which covers 3D volumetric processing.

## How to Run the Benchmarks

### 1. Download the Data

The IBSI 2 reference datasets are available on the [IBSI GitHub repository](https://github.com/theibsi/data_sets).

-   **CT Phantom**: Download the `PAT1_CT.nii.gz` and `PAT1_GTV.nii.gz` (or similarly named `image.nii.gz` / `mask.nii.gz`) from the `ibsi_2_phase_2` (or `ibsi_2_validation`) folder.

Place these files in a local directory (e.g., `data/ibsi2/data/ct_phantom/`) to run the benchmarks.

### 2. Run Configurations Programmatically using `RadiomicsPipeline`

You can verify any IBSI 2 configuration programmatically using the `RadiomicsPipeline`.

```python
from pictologics import RadiomicsPipeline

# Path to your downloaded phantom
image_path = "data/ibsi2/data/ct_phantom/image.nii.gz"
mask_path = "data/ibsi2/data/ct_phantom/mask.nii.gz"

pipeline = RadiomicsPipeline()

# Define common IBSI 2 Preprocessing (Config B)
preprocess_steps = [
    # 1. Resample to 1x1x1 mm using Tricubic Spline interpolation
    {"step": "resample", "params": {
        "new_spacing": (1.0, 1.0, 1.0), 
        "interpolation": "cubic", 
        "mask_interpolation": "linear", 
        "mask_threshold": 0.5
    }},
    # 2. Round intensities to nearest integer
    {"step": "round_intensities", "params": {}},
    # 3. Resegment range [-1000, 400] HU
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
]

# Define a filter configuration (e.g., Test 3.B: LoG)
log_config = preprocess_steps + [
    {"step": "filter", "params": {
        "type": "log",
        "sigma_mm": 1.5,
        "truncate": 4.0
    }},
    {"step": "extract_features", "params": {"families": ["intensity"]}}
]

pipeline.add_config("ibsi2_test_3b", log_config)

# Run pipeline
results = pipeline.run(image_path, mask_path, config_names=["ibsi2_test_3b"])
print(results["ibsi2_test_3b"])
```

## Phase 2 Results

**Summary**: 9/9 Tests Passed (3D Configuration). Total Features: 161/161 passed.

### Test 1.B: None

**Configuration**: Baseline (no filter)

| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Mean | stat_mean | -46.4 | -46.4 | 5.9 | ✅ PASS |
| Variance | stat_var | 5.26e+04 | 5.26e+04 | 2.8e+03 | ✅ PASS |
| Skewness | stat_skew | -2.18 | -2.18 | 0.09 | ✅ PASS |
| (Excess) kurtosis | stat_kurt | 3.71 | 3.71 | 0.47 | ✅ PASS |
| Median | stat_median | 41 | 41 | 0.7 | ✅ PASS |
| Minimum | stat_min | -997 | -997 | 3 | ✅ PASS |
| 10th percentile | stat_p10 | -427 | -427 | 29 | ✅ PASS |
| 90th percentile | stat_p90 | 92 | 92 | 0.1 | ✅ PASS |
| Maximum | stat_max | 377 | 377 | 15 | ✅ PASS |
| Interquartile range | stat_iqr | 67 | 67 | 9.1 | ✅ PASS |
| Range | stat_range | 1.37e+03 | 1.37e+03 | 20 | ✅ PASS |
| Mean absolute deviation | stat_mad | 159 | 159 | 7 | ✅ PASS |
| Robust mean absolute deviation | stat_rmad | 63.6 | 63.6 | 7.3 | ✅ PASS |
| Median absolute deviation | stat_medad | 121 | 121 | 6 | ✅ PASS |
| Coefficient of variation | stat_cov | -4.94 | -4.94 | 0.64 | ✅ PASS |
| Quartile coefficient of dispersion | stat_qcod | 0.944 | 0.944 | 0.925 | ✅ PASS |
| Energy | stat_energy | 1.96e+10 | 1.96e+10 | 1.9e+09 | ✅ PASS |
| Root mean square | stat_rms | 234 | 234 | 7 | ✅ PASS |

### Test 2.B: Mean

**Configuration**: 3D, support=5

| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Mean | stat_mean | -49.9 | -49.9 | 5.7 | ✅ PASS |
| Variance | stat_var | 4.44e+04 | 4.44e+04 | 2.3e+03 | ✅ PASS |
| Skewness | stat_skew | -2.13 | -2.13 | 0.09 | ✅ PASS |
| (Excess) kurtosis | stat_kurt | 3.59 | 3.59 | 0.46 | ✅ PASS |
| Median | stat_median | 37.3 | 37.3 | 0.6 | ✅ PASS |
| Minimum | stat_min | -906 | -906 | 5 | ✅ PASS |
| 10th percentile | stat_p10 | -389 | -389 | 25 | ✅ PASS |
| 90th percentile | stat_p90 | 77.2 | 77.2 | 0.1 | ✅ PASS |
| Maximum | stat_max | 316 | 316 | 7 | ✅ PASS |
| Interquartile range | stat_iqr | 92.6 | 92.6 | 13.5 | ✅ PASS |
| Range | stat_range | 1.22e+03 | 1.22e+03 | 10 | ✅ PASS |
| Mean absolute deviation | stat_mad | 149 | 149 | 6 | ✅ PASS |
| Robust mean absolute deviation | stat_rmad | 68.1 | 68.1 | 6.9 | ✅ PASS |
| Median absolute deviation | stat_medad | 114 | 114 | 5 | ✅ PASS |
| Coefficient of variation | stat_cov | -4.22 | -4.22 | 0.47 | ✅ PASS |
| Quartile coefficient of dispersion | stat_qcod | 2.97 | 2.97 | 0.58 | ✅ PASS |
| Energy | stat_energy | 1.68e+10 | 1.68e+10 | 1.6e+09 | ✅ PASS |
| Root mean square | stat_rms | 217 | 217 | 7 | ✅ PASS |

### Test 3.B: LoG

**Configuration**: σ=1.5mm, truncate=4σ

| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Mean | stat_mean | -2.92 | -2.94 | 0.2 | ✅ PASS |
| Variance | stat_var | 720 | 720 | 33 | ✅ PASS |
| Skewness | stat_skew | 0.428 | 0.428 | 0.009 | ✅ PASS |
| (Excess) kurtosis | stat_kurt | 6.13 | 6.13 | 0.27 | ✅ PASS |
| Median | stat_median | -0.927 | -0.919 | 0.024 | ✅ PASS |
| Minimum | stat_min | -173 | -173 | 5 | ✅ PASS |
| 10th percentile | stat_p10 | -32.1 | -32.2 | 0.5 | ✅ PASS |
| 90th percentile | stat_p90 | 17.5 | 17.4 | 1.9 | ✅ PASS |
| Maximum | stat_max | 204 | 204 | 1 | ✅ PASS |
| Interquartile range | stat_iqr | 11.4 | 11.4 | 0.3 | ✅ PASS |
| Range | stat_range | 377 | 377 | 5 | ✅ PASS |
| Mean absolute deviation | stat_mad | 15.5 | 15.5 | 0.4 | ✅ PASS |
| Robust mean absolute deviation | stat_rmad | 6.36 | 6.37 | 0.19 | ✅ PASS |
| Median absolute deviation | stat_medad | 15.3 | 15.3 | 0.4 | ✅ PASS |
| Coefficient of variation | stat_cov | -9.18 | -9.12 | 1.63 | ✅ PASS |
| Quartile coefficient of dispersion | stat_qcod | -2.34 | -2.34 | 0.07 | ✅ PASS |
| Energy | stat_energy | 2.61e+08 | 2.61e+08 | 1.9e+07 | ✅ PASS |
| Root mean square | stat_rms | 27 | 27 | 0.6 | ✅ PASS |

### Test 4.B: Laws

**Configuration**: L5E5E5, rot-inv max, energy δ=7

| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Mean | stat_mean | 142 | 142 | 3 | ✅ PASS |
| Variance | stat_var | 1.11e+04 | 1.11e+04 | 300 | ✅ PASS |
| Skewness | stat_skew | 0.645 | 0.645 | 0.028 | ✅ PASS |
| (Excess) kurtosis | stat_kurt | -0.711 | -0.711 | 0.044 | ✅ PASS |
| Median | stat_median | 113 | 113 | 4 | ✅ PASS |
| Minimum | stat_min | 28.5 | 28.5 | 0.1 | ✅ PASS |
| 10th percentile | stat_p10 | 35.6 | 35.6 | 0.1 | ✅ PASS |
| 90th percentile | stat_p90 | 293 | 293 | 4 | ✅ PASS |
| Maximum | stat_max | 525 | 525 | 1 | ✅ PASS |
| Interquartile range | stat_iqr | 188 | 188 | 4 | ✅ PASS |
| Range | stat_range | 496 | 496 | 1 | ✅ PASS |
| Mean absolute deviation | stat_mad | 92.4 | 92.4 | 1.4 | ✅ PASS |
| Robust mean absolute deviation | stat_rmad | 75.9 | 75.9 | 1.4 | ✅ PASS |
| Median absolute deviation | stat_medad | 90.8 | 90.8 | 1.6 | ✅ PASS |
| Coefficient of variation | stat_cov | 0.743 | 0.743 | 0.005 | ✅ PASS |
| Quartile coefficient of dispersion | stat_qcod | 0.699 | 0.699 | 0.003 | ✅ PASS |
| Energy | stat_energy | 1.12e+10 | 1.12e+10 | 7e+08 | ✅ PASS |
| Root mean square | stat_rms | 177 | 177 | 3 | ✅ PASS |

### Test 5.B: Gabor

**Configuration**: σ=5mm, λ=2mm, γ=1.5, rot-inv avg

| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Mean | stat_mean | 40.2 | 40.2 | 0.2 | ✅ PASS |
| Variance | stat_var | 231 | 231 | 2 | ✅ PASS |
| Skewness | stat_skew | 1.57 | 1.57 | 0.03 | ✅ PASS |
| (Excess) kurtosis | stat_kurt | 4.34 | 4.34 | 0.2 | ✅ PASS |
| Median | stat_median | 37.2 | 37.2 | 0.1 | ✅ PASS |
| Minimum | stat_min | 9.53 | 9.53 | 0.11 | ✅ PASS |
| 10th percentile | stat_p10 | 24.6 | 24.6 | 0.1 | ✅ PASS |
| 90th percentile | stat_p90 | 59.3 | 59.3 | 0.3 | ✅ PASS |
| Maximum | stat_max | 175 | 175 | 3 | ✅ PASS |
| Interquartile range | stat_iqr | 17.4 | 17.4 | 0.1 | ✅ PASS |
| Range | stat_range | 165 | 165 | 3 | ✅ PASS |
| Mean absolute deviation | stat_mad | 11.3 | 11.3 | 0.1 | ✅ PASS |
| Robust mean absolute deviation | stat_rmad | 7.31 | 7.31 | 0.06 | ✅ PASS |
| Median absolute deviation | stat_medad | 11 | 11 | 0.1 | ✅ PASS |
| Coefficient of variation | stat_cov | 0.377 | 0.377 | 0.004 | ✅ PASS |
| Quartile coefficient of dispersion | stat_qcod | 0.226 | 0.226 | 0.002 | ✅ PASS |
| Energy | stat_energy | 6.62e+08 | 6.62e+08 | 9e+06 | ✅ PASS |
| Root mean square | stat_rms | 43 | 43 | 0.2 | ✅ PASS |

### Test 6.B: Daubechies 3

**Configuration**: LLH level 1, rot-inv avg

| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Mean | stat_mean | -0.182 | -0.182 | 0.024 | ✅ PASS |
| Variance | stat_var | 250 | 250 | 9 | ✅ PASS |
| Skewness | stat_skew | 0.157 | 0.157 | 0.018 | ✅ PASS |
| (Excess) kurtosis | stat_kurt | 8.98 | 8.98 | 0.35 | ✅ PASS |
| Median | stat_median | 0.0576 | 0.0575 | 0.0046 | ✅ PASS |
| Minimum | stat_min | -148 | -148 | 1 | ✅ PASS |
| 10th percentile | stat_p10 | -13.8 | -13.8 | 0.5 | ✅ PASS |
| 90th percentile | stat_p90 | 12.1 | 12.1 | 0.4 | ✅ PASS |
| Maximum | stat_max | 155 | 155 | 1 | ✅ PASS |
| Interquartile range | stat_iqr | 9.35 | 9.35 | 0.15 | ✅ PASS |
| Range | stat_range | 303 | 303 | 2 | ✅ PASS |
| Mean absolute deviation | stat_mad | 9.26 | 9.26 | 0.22 | ✅ PASS |
| Robust mean absolute deviation | stat_rmad | 4.21 | 4.21 | 0.09 | ✅ PASS |
| Median absolute deviation | stat_medad | 9.25 | 9.25 | 0.22 | ✅ PASS |
| Coefficient of variation | stat_cov | -86.9 | -86.9 | 32.6 | ✅ PASS |
| Quartile coefficient of dispersion | stat_qcod | -162 | -162 | 27 | ✅ PASS |
| Energy | stat_energy | 8.96e+07 | 8.96e+07 | 5.3e+06 | ✅ PASS |
| Root mean square | stat_rms | 15.8 | 15.8 | 0.3 | ✅ PASS |

### Test 7.B: Daubechies 3

**Configuration**: HHH level 2, rot-inv avg

| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Mean | stat_mean | -0.0406 | -0.0406 | 0.0051 | ✅ PASS |
| Variance | stat_var | 422 | 422 | 11 | ✅ PASS |
| Skewness | stat_skew | -0.0112 | -0.0112 | 0.0027 | ✅ PASS |
| (Excess) kurtosis | stat_kurt | 5.45 | 5.45 | 0.09 | ✅ PASS |
| Median | stat_median | -0.0164 | -0.0164 | 0.0013 | ✅ PASS |
| Minimum | stat_min | -203 | -203 | 3 | ✅ PASS |
| 10th percentile | stat_p10 | -20.6 | -20.6 | 0.4 | ✅ PASS |
| 90th percentile | stat_p90 | 20.4 | 20.4 | 0.4 | ✅ PASS |
| Maximum | stat_max | 201 | 201 | 4 | ✅ PASS |
| Interquartile range | stat_iqr | 16.3 | 16.3 | 0.2 | ✅ PASS |
| Range | stat_range | 404 | 404 | 7 | ✅ PASS |
| Mean absolute deviation | stat_mad | 13.4 | 13.4 | 0.2 | ✅ PASS |
| Robust mean absolute deviation | stat_rmad | 7.2 | 7.2 | 0.1 | ✅ PASS |
| Median absolute deviation | stat_medad | 13.4 | 13.4 | 0.2 | ✅ PASS |
| Coefficient of variation | stat_cov | -506 | -506 | 149 | ✅ PASS |
| Quartile coefficient of dispersion | stat_qcod | -684 | -684 | 130 | ✅ PASS |
| Energy | stat_energy | 1.51e+08 | 1.51e+08 | 7e+06 | ✅ PASS |
| Root mean square | stat_rms | 20.6 | 20.6 | 0.3 | ✅ PASS |

### Test 8.B: Simoncelli

**Configuration**: B map level 1

| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Mean | stat_mean | 0.32 | 0.32 | 0.059 | ✅ PASS |
| Variance | stat_var | 1.81e+03 | 1.81e+03 | 70 | ✅ PASS |
| Skewness | stat_skew | -0.0719 | -0.0719 | 0.0163 | ✅ PASS |
| (Excess) kurtosis | stat_kurt | 7.64 | 7.64 | 0.33 | ✅ PASS |
| Median | stat_median | -0.00194 | -0.00947 | 0.0107 | ✅ PASS |
| Minimum | stat_min | -411 | -411 | 5 | ✅ PASS |
| 10th percentile | stat_p10 | -36.6 | -36.5 | 1.3 | ✅ PASS |
| 90th percentile | stat_p90 | 38.1 | 38.1 | 1.3 | ✅ PASS |
| Maximum | stat_max | 374 | 374 | 3 | ✅ PASS |
| Interquartile range | stat_iqr | 25.5 | 25.5 | 0.4 | ✅ PASS |
| Range | stat_range | 785 | 785 | 6 | ✅ PASS |
| Mean absolute deviation | stat_mad | 25.3 | 25.3 | 0.6 | ✅ PASS |
| Robust mean absolute deviation | stat_rmad | 11.7 | 11.7 | 0.3 | ✅ PASS |
| Median absolute deviation | stat_medad | 25.3 | 25.3 | 0.6 | ✅ PASS |
| Coefficient of variation | stat_cov | 133 | 134 | 27 | ✅ PASS |
| Energy | stat_energy | 6.48e+08 | 6.48e+08 | 3.9e+07 | ✅ PASS |
| Root mean square | stat_rms | 42.5 | 42.5 | 0.9 | ✅ PASS |

### Test 9.B: Simoncelli

**Configuration**: B map level 2

| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Mean | stat_mean | 2.68 | 2.68 | 0.22 | ✅ PASS |
| Variance | stat_var | 5.49e+03 | 5.49e+03 | 220 | ✅ PASS |
| Skewness | stat_skew | -0.0858 | -0.0858 | 0.0107 | ✅ PASS |
| (Excess) kurtosis | stat_kurt | 5.58 | 5.58 | 0.18 | ✅ PASS |
| Median | stat_median | 0.233 | 0.233 | 0.046 | ✅ PASS |
| Minimum | stat_min | -605 | -605 | 2 | ✅ PASS |
| 10th percentile | stat_p10 | -65.9 | -65.9 | 2.2 | ✅ PASS |
| 90th percentile | stat_p90 | 82.9 | 82.8 | 1.8 | ✅ PASS |
| Maximum | stat_max | 471 | 471 | 13 | ✅ PASS |
| Interquartile range | stat_iqr | 41.4 | 41 | 1 | ✅ PASS |
| Range | stat_range | 1.08e+03 | 1.08e+03 | 20 | ✅ PASS |
| Mean absolute deviation | stat_mad | 45.1 | 45.1 | 1.1 | ✅ PASS |
| Robust mean absolute deviation | stat_rmad | 21 | 21 | 0.5 | ✅ PASS |
| Median absolute deviation | stat_medad | 45 | 45 | 1.1 | ✅ PASS |
| Coefficient of variation | stat_cov | 27.7 | 27.7 | 20.4 | ✅ PASS |
| Quartile coefficient of dispersion | stat_qcod | 47.4 | 47.4 | 20.7 | ✅ PASS |
| Energy | stat_energy | 1.97e+09 | 1.97e+09 | 1.4e+08 | ✅ PASS |
| Root mean square | stat_rms | 74.1 | 74.1 | 1.6 | ✅ PASS |

## Known Deviations

### 2D Configuration (Config A) Not Implemented

The following tests check filter performance in 2D mode (Config A). These are **skipped** because Pictologics is purely a 3D radiomics library.

-   **Tests 1.A - 9.A**: 2D versions of the above tests.
