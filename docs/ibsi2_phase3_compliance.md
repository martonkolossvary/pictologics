# IBSI 2 Phase 3 Compliance Report

> [!NOTE]
> Phase 3 validates reproducibility across a 51-patient multi-modality dataset.
> Comparison is made against 9 participating teams' submissions.

## Summary

- **Patients Processed**: 153
- **Teams Compared**: 9

## Team Comparison Results

| Team | Overlap % | Matches | Total | Mismatches |
|:-----|----------:|--------:|------:|-----------:|
| CERR | ❌ 76.6% | 12652 | 16524 | 3872 |
| Cardiff University | ❌ 79.8% | 19787 | 24786 | 4999 |
| King's College London | ❌ 78.0% | 18618 | 23868 | 5250 |
| NCT Dresden | ❌ 0.0% | 0 | 0 | 0 |
| Qurit SERA | ❌ 65.4% | 16190 | 24751 | 8561 |
| UCSF | ❌ 22.0% | 5453 | 24786 | 19333 |
| USZ | ✅ 97.0% | 16023 | 16524 | 501 |
| UdeS | ❌ 73.3% | 18158 | 24786 | 6628 |
| Veneto Institute of Oncology | ❌ 72.0% | 17842 | 24786 | 6944 |

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
| ... | *3862 more* | | | |

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
| ... | *4989 more* | | | |

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
| ... | *5240 more* | | | |

### NCT Dresden

No mismatches! ✅

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
| ... | *8551 more* | | | |

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
| stat_qcod | 6 | 1.748e+04 | 2.543e+04 | 7949 |
| stat_mean | 6 | -8.969e-06 | -8.843e-06 | 1.256e-07 |
| stat_cov | 6 | -956.7 | -970.2 | 13.56 |
| stat_qcod | 6 | -343.5 | -349.5 | 5.961 |
| stat_mean | 6 | -5.799e-06 | -7.58e-06 | 1.781e-06 |
| stat_var | 2 | 7.652 | 7.488 | 0.1646 |
| stat_var | 3 | 0.5777 | 0.5702 | 0.007527 |
| stat_var | 4 | 0.4721 | 0.4327 | 0.03938 |
| stat_var | 6 | 0.0002697 | 0.000262 | 7.682e-06 |
| stat_skew | 4 | 1.714 | 1.623 | 0.09095 |
| ... | *491 more* | | | |

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
| ... | *6618 more* | | | |

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
| ... | *6934 more* | | | |
