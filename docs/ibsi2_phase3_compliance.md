# IBSI 2 Phase 3 Compliance Report

> [!NOTE]
> Phase 3 validates reproducibility across a 51-patient multi-modality dataset.
> Comparison is made against 9 participating teams' submissions.
> Results are shown for both exact matching (1% of value) and range-based tolerances (1%, 5%, 10% of feature range).

## Summary

- **Patients Processed**: 153
- **Teams Compared**: 9

## Team Comparison Results (Exact Matching)

| Team | Overlap % | Matches | Total | Mismatches |
|:-----|----------:|--------:|------:|-----------:|
| CERR | ❌ 76.5% | 12647 | 16524 | 3877 |
| Cardiff University | ❌ 79.8% | 19782 | 24786 | 5004 |
| King's College London | ❌ 78.0% | 18614 | 23868 | 5254 |
| NCT Dresden | ⚠️ 94.6% | 23446 | 24786 | 1340 |
| Qurit SERA | ❌ 65.4% | 16191 | 24751 | 8560 |
| UCSF | ❌ 22.0% | 5453 | 24786 | 19333 |
| USZ | ✅ 96.9% | 16018 | 16524 | 506 |
| UdeS | ❌ 73.2% | 18154 | 24786 | 6632 |
| Veneto Institute of Oncology | ❌ 72.0% | 17837 | 24786 | 6949 |

## Tolerance Breakdown (% of Feature Range)

Shows how many comparisons match within different tolerances based on the range of each feature across all teams.

| Team | Within 1% | Within 5% | Within 10% | Total |
|:-----|----------:|----------:|-----------:|------:|
| CERR | 16408 (99.3%) | 16492 (99.8%) | 16503 (99.9%) | 16524 |
| Cardiff University | 24532 (99.0%) | 24696 (99.6%) | 24746 (99.8%) | 24786 |
| King's College London | 23744 (99.5%) | 23847 (99.9%) | 23858 (100.0%) | 23868 |
| NCT Dresden | 24674 (99.5%) | 24740 (99.8%) | 24769 (99.9%) | 24786 |
| Qurit SERA | 23073 (93.2%) | 24130 (97.5%) | 24393 (98.6%) | 24751 |
| UCSF | 14858 (59.9%) | 18988 (76.6%) | 20874 (84.2%) | 24786 |
| USZ | 16499 (99.8%) | 16516 (100.0%) | 16520 (100.0%) | 16524 |
| UdeS | 22200 (89.6%) | 23457 (94.6%) | 24180 (97.6%) | 24786 |
| Veneto Institute of Oncology | 22571 (91.1%) | 24016 (96.9%) | 24468 (98.7%) | 24786 |

## Mismatch Details

### CERR

| Feature | Filter ID | Our Value | Team Value | Error |
|:--------|----------:|----------:|-----------:|------:|
| stat_mean | 6 | -0.0001413 | -0.0001545 | 1.32e-05 |
| stat_var | 6 | 0.0003726 | 0.000416 | 4.34e-05 |
| stat_skew | 6 | -0.04434 | -0.06896 | 0.02462 |
| stat_kurt | 6 | 0.2843 | 0.3675 | 0.08322 |
| stat_median | 6 | -0.000158 | -7.339e-05 | 8.463e-05 |
| stat_min | 6 | -0.07198 | -0.08196 | 0.009978 |
| stat_p10 | 6 | -0.02427 | -0.02508 | 0.0008141 |
| stat_p90 | 6 | 0.02398 | 0.02571 | 0.001732 |
| stat_max | 6 | 0.06868 | 0.07037 | 0.001692 |
| stat_iqr | 6 | 0.02486 | 0.02628 | 0.001425 |
| ... | *3867 more* | | | |

### Cardiff University

| Feature | Filter ID | Our Value | Team Value | Error |
|:--------|----------:|----------:|-----------:|------:|
| stat_mean | 8 | 0.1419 | 0.1762 | 0.03421 |
| stat_mean | 9 | 0.8512 | 1.237 | 0.3855 |
| stat_var | 8 | 1.623 | 1.69 | 0.06724 |
| stat_var | 9 | 11.16 | 11.62 | 0.4591 |
| stat_skew | 8 | 0.003916 | -0.01158 | 0.0155 |
| stat_skew | 9 | 0.06082 | 0.3051 | 0.2442 |
| stat_kurt | 8 | 1.461 | 1.679 | 0.2184 |
| stat_kurt | 9 | -0.08945 | -0.3674 | 0.2779 |
| stat_median | 8 | 0.05959 | 0.1135 | 0.05393 |
| stat_median | 9 | 0.6782 | 0.7415 | 0.0633 |
| ... | *4994 more* | | | |

### King's College London

| Feature | Filter ID | Our Value | Team Value | Error |
|:--------|----------:|----------:|-----------:|------:|
| stat_mean | 1 | 7.38 | 7.289 | 0.09059 |
| stat_mean | 2 | 7.083 | 7 | 0.08357 |
| stat_mean | 3 | -0.7383 | -0.7153 | 0.02303 |
| stat_mean | 6 | -0.0001413 | -0.0003606 | 0.0002193 |
| stat_mean | 8 | 0.1419 | 0.1347 | 0.007279 |
| stat_mean | 9 | 0.8512 | 0.7803 | 0.07093 |
| stat_var | 4 | 0.5266 | 0.5393 | 0.01272 |
| stat_var | 6 | 0.0003726 | 0.0003626 | 9.972e-06 |
| stat_skew | 1 | 0.7007 | 0.7259 | 0.02524 |
| stat_skew | 2 | 0.6514 | 0.6781 | 0.02673 |
| ... | *5244 more* | | | |

### NCT Dresden

| Feature | Filter ID | Our Value | Team Value | Error |
|:--------|----------:|----------:|-----------:|------:|
| stat_mean | 6 | 5.055e-06 | 5.121e-06 | 6.565e-08 |
| stat_cov | 6 | 4617 | 4558 | 59.11 |
| stat_median | 6 | -7.188e-05 | -7.593e-05 | 4.049e-06 |
| stat_min | 5 | 0.03145 | 0.03218 | 0.0007243 |
| stat_qcod | 6 | -3851 | -3751 | 99.91 |
| stat_qcod | 6 | 1.748e+04 | 1.63e+04 | 1181 |
| stat_median | 6 | -0.0002505 | -0.0002649 | 1.441e-05 |
| stat_min | 5 | 0.02037 | 0.01903 | 0.001341 |
| stat_mean | 6 | -8.969e-06 | -8.843e-06 | 1.256e-07 |
| stat_cov | 6 | -956.7 | -970.2 | 13.56 |
| ... | *1330 more* | | | |

### Qurit SERA

| Feature | Filter ID | Our Value | Team Value | Error |
|:--------|----------:|----------:|-----------:|------:|
| stat_mean | 7 | 1.011 | 0.3628 | 0.6482 |
| stat_var | 7 | 0.2918 | 0.05545 | 0.2363 |
| stat_skew | 7 | 0.4754 | 1.199 | 0.7233 |
| stat_skew | 8 | 0.003916 | 0.004315 | 0.0003996 |
| stat_skew | 9 | 0.06082 | 0.06011 | 0.0007074 |
| stat_kurt | 7 | -0.1607 | 1.668 | 1.828 |
| stat_median | 7 | 0.9532 | 0.3136 | 0.6396 |
| stat_min | 7 | -0.302 | -0.1523 | 0.1497 |
| stat_p10 | 7 | 0.3523 | 0.1143 | 0.238 |
| stat_p90 | 7 | 1.776 | 0.6689 | 1.107 |
| ... | *8550 more* | | | |

### UCSF

| Feature | Filter ID | Our Value | Team Value | Error |
|:--------|----------:|----------:|-----------:|------:|
| stat_mean | 3 | -0.7383 | 0 | 0.7383 |
| stat_mean | 4 | 2.724 | 0 | 2.724 |
| stat_mean | 5 | 2.921 | 0 | 2.921 |
| stat_mean | 6 | -0.0001413 | 0 | 0.0001413 |
| stat_mean | 7 | 1.011 | 0 | 1.011 |
| stat_mean | 8 | 0.1419 | 0 | 0.1419 |
| stat_mean | 9 | 0.8512 | 0 | 0.8512 |
| stat_var | 3 | 3.251 | 0 | 3.251 |
| stat_var | 4 | 0.5266 | 0 | 0.5266 |
| stat_var | 5 | 6.204 | 0 | 6.204 |
| ... | *19323 more* | | | |

### USZ

| Feature | Filter ID | Our Value | Team Value | Error |
|:--------|----------:|----------:|-----------:|------:|
| stat_qcod | 6 | 1.748e+04 | 2.543e+04 | 7950 |
| stat_mean | 6 | -8.969e-06 | -8.843e-06 | 1.256e-07 |
| stat_cov | 6 | -956.7 | -970.2 | 13.56 |
| stat_qcod | 6 | -343.5 | -349.5 | 5.965 |
| stat_min | 5 | 0.008531 | 0.008372 | 0.0001588 |
| stat_mean | 6 | -5.799e-06 | -7.58e-06 | 1.781e-06 |
| stat_var | 2 | 7.652 | 7.488 | 0.1646 |
| stat_var | 3 | 0.5777 | 0.5702 | 0.007527 |
| stat_var | 4 | 0.4721 | 0.4327 | 0.03938 |
| stat_var | 6 | 0.0002697 | 0.000262 | 7.682e-06 |
| ... | *496 more* | | | |

### UdeS

| Feature | Filter ID | Our Value | Team Value | Error |
|:--------|----------:|----------:|-----------:|------:|
| stat_mean | 8 | 0.1419 | 0 | 0.1419 |
| stat_mean | 9 | 0.8512 | 0 | 0.8512 |
| stat_var | 8 | 1.623 | 0 | 1.623 |
| stat_var | 9 | 11.16 | 0 | 11.16 |
| stat_skew | 8 | 0.003916 | 0 | 0.003916 |
| stat_skew | 9 | 0.06082 | 0 | 0.06082 |
| stat_kurt | 8 | 1.461 | 0 | 1.461 |
| stat_kurt | 9 | -0.08945 | 0 | 0.08945 |
| stat_median | 8 | 0.05959 | 0 | 0.05959 |
| stat_median | 9 | 0.6782 | 0 | 0.6782 |
| ... | *6622 more* | | | |

### Veneto Institute of Oncology

| Feature | Filter ID | Our Value | Team Value | Error |
|:--------|----------:|----------:|-----------:|------:|
| stat_mean | 8 | 0.1419 | 0.7915 | 0.6495 |
| stat_mean | 9 | 0.8512 | 2.809 | 1.958 |
| stat_var | 8 | 1.623 | 10.93 | 9.307 |
| stat_var | 9 | 11.16 | 14.11 | 2.956 |
| stat_skew | 8 | 0.003916 | 0.01743 | 0.01352 |
| stat_skew | 9 | 0.06082 | 0.588 | 0.5272 |
| stat_kurt | 8 | 1.461 | -0.02308 | 1.484 |
| stat_kurt | 9 | -0.08945 | -0.8198 | 0.7304 |
| stat_median | 8 | 0.05959 | 0.6583 | 0.5987 |
| stat_median | 9 | 0.6782 | 1.533 | 0.8548 |
| ... | *6939 more* | | | |
