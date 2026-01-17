# IBSI 2 Phase 2 Compliance Report

> [!IMPORTANT]
> Only 3D methods (Config B) are implemented. 2D slice-wise filtering (Config A)
> is not supported in the current pictologics implementation.

## Overview

Phase 2 validates feature extraction from filtered response maps using the
lung cancer CT image (PAT1) with GTV segmentation mask.

### Preprocessing (Config B)
- Interpolation: tricubic spline to 1×1×1 mm
- Intensity rounding: nearest integer
- ROI interpolation: trilinear (threshold 0.5)
- Re-segmentation: [-1000, 400] HU

## Results Summary

- **Total Tests**: 9
- **Passed**: 0
- **Failed**: 9
- **Skipped**: 0

## Test Results

| Filter ID | Filter | Status | Features | Time (ms) | Notes |
|-----------|--------|--------|----------|-----------|-------|
| 1.B | None | ❌ FAIL | 15/18 | 22 | Baseline (no filter, just preprocessing) |
| 2.B | Mean | ❌ FAIL | 15/18 | 50 | 3D filter, support M=5 voxels |
| 3.B | LoG | ❌ FAIL | 16/18 | 200 | 3D filter, σ*=1.5mm, truncate=4σ |
| 4.B | Laws | ❌ FAIL | 14/18 | 8558 | L5E5E5, 3D rot-inv max, energy δ=7 |
| 5.B | Gabor | ❌ FAIL | 0/18 | 6969 | σ*=5mm, λ*=2mm, γ=1.5, rot-inv avg, avg over planes |
| 6.B | Daubechies 3 | ❌ FAIL | 12/18 | 1541 | LLH level 1, 3D rot-inv avg |
| 7.B | Daubechies 3 | ❌ FAIL | 7/18 | 3660 | HHH level 2, 3D rot-inv avg |
| 8.B | Simoncelli | ❌ FAIL | 13/17 | 434 | 3D B map level 1 |
| 9.B | Simoncelli | ❌ FAIL | 15/18 | 407 | 3D B map level 2 |

## Feature Details

### 1.B: None

| Feature | Computed | Reference | Tolerance | Status |
|---------|----------|-----------|-----------|--------|
| Mean | -45.94 | -46.4 | ±5.9 | ✅ |
| Variance | 5.21e+04 | 5.26e+04 | ±2800 | ✅ |
| Skewness | -2.186 | -2.18 | ±0.09 | ✅ |
| (Excess) kurtosis | 3.759 | 3.71 | ±0.47 | ✅ |
| Median | 41 | 41 | ±0.7 | ✅ |
| Minimum | -991 | -997 | ±3 | ❌ |
| 10th percentile | -423 | -427 | ±29 | ✅ |
| 90th percentile | 92 | 92 | ±0.1 | ✅ |
| Maximum | 397 | 377 | ±15 | ❌ |
| Interquartile range | 68 | 67 | ±9.1 | ✅ |
| Range | 1388 | 1370 | ±20 | ✅ |
| Mean absolute deviation | 157.9 | 159 | ±7 | ✅ |
| Robust mean absolute deviation | 63.25 | 63.6 | ±7.3 | ✅ |
| Median absolute deviation | 31 | 121 | ±6 | ❌ |
| Coefficient of variation | -4.968 | -4.94 | ±0.64 | ✅ |
| Quartile coefficient of dispersion | 0.9714 | 0.944 | ±0.925 | ✅ |
| Energy | 1.937e+10 | 1.96e+10 | ±1.9e+09 | ✅ |
| Root mean square | 232.8 | 234 | ±7 | ✅ |

### 2.B: Mean

| Feature | Computed | Reference | Tolerance | Status |
|---------|----------|-----------|-----------|--------|
| Mean | -49.43 | -49.9 | ±5.7 | ✅ |
| Variance | 4.384e+04 | 4.44e+04 | ±2300 | ✅ |
| Skewness | -2.133 | -2.13 | ±0.09 | ✅ |
| (Excess) kurtosis | 3.626 | 3.59 | ±0.46 | ✅ |
| Median | 37.14 | 37.3 | ±0.6 | ✅ |
| Minimum | -891.6 | -906 | ±5 | ❌ |
| 10th percentile | -385.3 | -389 | ±25 | ✅ |
| 90th percentile | 77.26 | 77.2 | ±0.1 | ✅ |
| Maximum | 309.1 | 316 | ±7 | ✅ |
| Interquartile range | 93.07 | 92.6 | ±13.5 | ✅ |
| Range | 1201 | 1220 | ±10 | ❌ |
| Mean absolute deviation | 148.4 | 149 | ±6 | ✅ |
| Robust mean absolute deviation | 67.82 | 68.1 | ±6.9 | ✅ |
| Median absolute deviation | 28.46 | 114 | ±5 | ❌ |
| Coefficient of variation | -4.236 | -4.22 | ±0.47 | ✅ |
| Quartile coefficient of dispersion | 3.063 | 2.97 | ±0.58 | ✅ |
| Energy | 1.654e+10 | 1.68e+10 | ±1.6e+09 | ✅ |
| Root mean square | 215.1 | 217 | ±7 | ✅ |

### 3.B: LoG

| Feature | Computed | Reference | Tolerance | Status |
|---------|----------|-----------|-----------|--------|
| Mean | -2.933 | -2.94 | ±0.2 | ✅ |
| Variance | 723.2 | 720 | ±33 | ✅ |
| Skewness | 0.432 | 0.428 | ±0.009 | ✅ |
| (Excess) kurtosis | 6.091 | 6.13 | ±0.27 | ✅ |
| Median | -0.9293 | -0.919 | ±0.024 | ✅ |
| Minimum | -167.3 | -173 | ±5 | ❌ |
| 10th percentile | -32.19 | -32.2 | ±0.5 | ✅ |
| 90th percentile | 17.43 | 17.4 | ±1.9 | ✅ |
| Maximum | 205 | 204 | ±1 | ✅ |
| Interquartile range | 11.45 | 11.4 | ±0.3 | ✅ |
| Range | 372.3 | 377 | ±5 | ✅ |
| Mean absolute deviation | 15.53 | 15.5 | ±0.4 | ✅ |
| Robust mean absolute deviation | 6.374 | 6.37 | ±0.19 | ✅ |
| Median absolute deviation | 5.16 | 15.3 | ±0.4 | ❌ |
| Coefficient of variation | -9.168 | -9.12 | ±1.63 | ✅ |
| Quartile coefficient of dispersion | -2.333 | -2.34 | ±0.07 | ✅ |
| Energy | 2.615e+08 | 2.61e+08 | ±1.9e+07 | ✅ |
| Root mean square | 27.05 | 27 | ±0.6 | ✅ |

### 4.B: Laws

| Feature | Computed | Reference | Tolerance | Status |
|---------|----------|-----------|-----------|--------|
| Mean | 141.7 | 142 | ±3 | ✅ |
| Variance | 1.109e+04 | 1.11e+04 | ±300 | ✅ |
| Skewness | 0.6432 | 0.645 | ±0.028 | ✅ |
| (Excess) kurtosis | -0.7099 | -0.711 | ±0.044 | ✅ |
| Median | 113 | 113 | ±4 | ✅ |
| Minimum | 28.37 | 28.5 | ±0.1 | ❌ |
| 10th percentile | 35.45 | 35.6 | ±0.1 | ❌ |
| 90th percentile | 292.3 | 293 | ±4 | ✅ |
| Maximum | 523.4 | 525 | ±1 | ❌ |
| Interquartile range | 187.6 | 188 | ±4 | ✅ |
| Range | 495 | 496 | ±1 | ✅ |
| Mean absolute deviation | 92.33 | 92.4 | ±1.4 | ✅ |
| Robust mean absolute deviation | 75.85 | 75.9 | ±1.4 | ✅ |
| Median absolute deviation | 75.96 | 90.8 | ±1.6 | ❌ |
| Coefficient of variation | 0.7432 | 0.743 | ±0.005 | ✅ |
| Quartile coefficient of dispersion | 0.6993 | 0.699 | ±0.003 | ✅ |
| Energy | 1.114e+10 | 1.12e+10 | ±7e+08 | ✅ |
| Root mean square | 176.5 | 177 | ±3 | ✅ |

### 5.B: Gabor

| Feature | Computed | Reference | Tolerance | Status |
|---------|----------|-----------|-----------|--------|
| Mean | 40.7 | 40.2 | ±0.2 | ❌ |
| Variance | 165.4 | 231 | ±2 | ❌ |
| Skewness | 1.069 | 1.57 | ±0.03 | ❌ |
| (Excess) kurtosis | 1.979 | 4.34 | ±0.2 | ❌ |
| Median | 38.63 | 37.2 | ±0.1 | ❌ |
| Minimum | 11.14 | 9.53 | ±0.11 | ❌ |
| 10th percentile | 26.37 | 24.6 | ±0.1 | ❌ |
| 90th percentile | 57.61 | 59.3 | ±0.3 | ❌ |
| Maximum | 149.2 | 175 | ±3 | ❌ |
| Interquartile range | 15.99 | 17.4 | ±0.1 | ❌ |
| Range | 138.1 | 165 | ±3 | ❌ |
| Mean absolute deviation | 9.917 | 11.3 | ±0.1 | ❌ |
| Robust mean absolute deviation | 6.685 | 7.31 | ±0.06 | ❌ |
| Median absolute deviation | 7.812 | 11 | ±0.1 | ❌ |
| Coefficient of variation | 0.316 | 0.377 | ±0.004 | ❌ |
| Quartile coefficient of dispersion | 0.202 | 0.226 | ±0.002 | ❌ |
| Energy | 6.511e+08 | 6.62e+08 | ±9e+06 | ❌ |
| Root mean square | 42.69 | 43 | ±0.2 | ❌ |

### 6.B: Daubechies 3

| Feature | Computed | Reference | Tolerance | Status |
|---------|----------|-----------|-----------|--------|
| Mean | -0.1855 | -0.182 | ±0.024 | ✅ |
| Variance | 251.5 | 250 | ±9 | ✅ |
| Skewness | 0.1478 | 0.157 | ±0.018 | ✅ |
| (Excess) kurtosis | 8.732 | 8.98 | ±0.35 | ✅ |
| Median | 0.0423 | 0.0575 | ±0.0046 | ❌ |
| Minimum | -150 | -148 | ±1 | ❌ |
| 10th percentile | -13.84 | -13.8 | ±0.5 | ✅ |
| 90th percentile | 12.2 | 12.1 | ±0.4 | ✅ |
| Maximum | 162 | 155 | ±1 | ❌ |
| Interquartile range | 9.444 | 9.35 | ±0.15 | ✅ |
| Range | 312.1 | 303 | ±2 | ❌ |
| Mean absolute deviation | 9.303 | 9.26 | ±0.22 | ✅ |
| Robust mean absolute deviation | 4.255 | 4.21 | ±0.09 | ✅ |
| Median absolute deviation | 4.721 | 9.25 | ±0.22 | ❌ |
| Coefficient of variation | -85.5 | -86.9 | ±32.6 | ✅ |
| Quartile coefficient of dispersion | -251.5 | -162 | ±27 | ❌ |
| Energy | 8.988e+07 | 8.96e+07 | ±5.3e+06 | ✅ |
| Root mean square | 15.86 | 15.8 | ±0.3 | ✅ |

### 7.B: Daubechies 3

| Feature | Computed | Reference | Tolerance | Status |
|---------|----------|-----------|-----------|--------|
| Mean | -0.033 | -0.0406 | ±0.0051 | ❌ |
| Variance | 409.4 | 422 | ±11 | ❌ |
| Skewness | -0.007073 | -0.0112 | ±0.0027 | ❌ |
| (Excess) kurtosis | 5.539 | 5.45 | ±0.09 | ✅ |
| Median | -0.005667 | -0.0164 | ±0.0013 | ❌ |
| Minimum | -184.4 | -203 | ±3 | ❌ |
| 10th percentile | -20.21 | -20.6 | ±0.4 | ✅ |
| 90th percentile | 20.09 | 20.4 | ±0.4 | ✅ |
| Maximum | 209.9 | 201 | ±4 | ❌ |
| Interquartile range | 15.93 | 16.3 | ±0.2 | ❌ |
| Range | 394.3 | 404 | ±7 | ❌ |
| Mean absolute deviation | 13.21 | 13.4 | ±0.2 | ✅ |
| Robust mean absolute deviation | 7.064 | 7.2 | ±0.1 | ❌ |
| Median absolute deviation | 7.962 | 13.4 | ±0.2 | ❌ |
| Coefficient of variation | -613.1 | -506 | ±149 | ✅ |
| Quartile coefficient of dispersion | -804 | -684 | ±130 | ✅ |
| Energy | 1.463e+08 | 1.51e+08 | ±7e+06 | ✅ |
| Root mean square | 20.23 | 20.6 | ±0.3 | ❌ |

### 8.B: Simoncelli

| Feature | Computed | Reference | Tolerance | Status |
|---------|----------|-----------|-----------|--------|
| Mean | 0.3046 | 0.32 | ±0.059 | ✅ |
| Variance | 1810 | 1810 | ±70 | ✅ |
| Skewness | -0.09348 | -0.0719 | ±0.0163 | ❌ |
| (Excess) kurtosis | 7.588 | 7.64 | ±0.33 | ✅ |
| Median | -0.008755 | -0.00947 | ±0.0107 | ✅ |
| Minimum | -409.3 | -411 | ±5 | ✅ |
| 10th percentile | -36.45 | -36.5 | ±1.3 | ✅ |
| 90th percentile | 38.11 | 38.1 | ±1.3 | ✅ |
| Maximum | 382.6 | 374 | ±3 | ❌ |
| Interquartile range | 25.48 | 25.5 | ±0.4 | ✅ |
| Range | 791.9 | 785 | ±6 | ❌ |
| Mean absolute deviation | 25.36 | 25.3 | ±0.6 | ✅ |
| Robust mean absolute deviation | 11.7 | 11.7 | ±0.3 | ✅ |
| Median absolute deviation | 12.74 | 25.3 | ±0.6 | ❌ |
| Coefficient of variation | 139.7 | 134 | ±27 | ✅ |
| Energy | 6.469e+08 | 6.48e+08 | ±3.9e+07 | ✅ |
| Root mean square | 42.55 | 42.5 | ±0.9 | ✅ |

### 9.B: Simoncelli

| Feature | Computed | Reference | Tolerance | Status |
|---------|----------|-----------|-----------|--------|
| Mean | 2.687 | 2.68 | ±0.22 | ✅ |
| Variance | 5507 | 5490 | ±220 | ✅ |
| Skewness | -0.08408 | -0.0858 | ±0.0107 | ✅ |
| (Excess) kurtosis | 5.54 | 5.58 | ±0.18 | ✅ |
| Median | 0.1963 | 0.233 | ±0.046 | ✅ |
| Minimum | -591.8 | -605 | ±2 | ❌ |
| 10th percentile | -66.09 | -65.9 | ±2.2 | ✅ |
| 90th percentile | 82.9 | 82.8 | ±1.8 | ✅ |
| Maximum | 467.1 | 471 | ±13 | ✅ |
| Interquartile range | 41.53 | 41 | ±1 | ✅ |
| Range | 1059 | 1080 | ±20 | ❌ |
| Mean absolute deviation | 45.24 | 45.1 | ±1.1 | ✅ |
| Robust mean absolute deviation | 21.02 | 21 | ±0.5 | ✅ |
| Median absolute deviation | 20.75 | 45 | ±1.1 | ❌ |
| Coefficient of variation | 27.62 | 27.7 | ±20.4 | ✅ |
| Quartile coefficient of dispersion | 47.86 | 47.4 | ±20.7 | ✅ |
| Energy | 1.97e+09 | 1.97e+09 | ±1.4e+08 | ✅ |
| Root mean square | 74.26 | 74.1 | ±1.6 | ✅ |
