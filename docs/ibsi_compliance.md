# Comprehensive IBSI Benchmark Results

## How to Run the Benchmarks

To reproduce these results or run the IBSI compliance benchmarks on your own system:

### 1. Download the Data
The IBSI reference datasets (Digital Phantom and Lung Cancer CT) are available on the [IBSI GitHub repository](https://github.com/theibsi/data_sets).

- **Digital Phantom**: Download `phantom.nii.gz` and `mask.nii.gz` from the `digital_phantom` folder.
- **Lung Cancer CT**: Download the `PAT1` NIfTI files. Note that IBSI recommends converting these to at least 32-bit floating point and rounding to the nearest integer before processing.

Place these files in `dev/data/` or provide their paths to the script.

### 2. Run Configurations Programmatically using `RadiomicsPipeline`
You can run any of the IBSI configurations programmatically using the `RadiomicsPipeline` class.

```python
from pictologics.pipeline import RadiomicsPipeline
from pictologics.loader import load_image

# 1. Initialize Pipeline
pipeline = RadiomicsPipeline()

# --- DEFINE CONFIGURATIONS ---

# A. Digital Phantom Config (FBS 1.0, no resampling)
config_digital_phantom = [
    {"step": "discretise", "params": {"method": "FBS", "bin_width": 1.0}},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture", "histogram", "ivh"], "include_spatial_intensity": True, "include_local_intensity": True}}
]

# B. Config C (2mm isotropic, resegment [-1000, 400], FBS 25 HU)
config_c = [
    {"step": "resample", "params": {"new_spacing": (2.0, 2.0, 2.0), "interpolation": "linear", "round_intensities": True}},
    {"step": "keep_largest_component", "params": {"apply_to": "morph"}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    {"step": "discretise", "params": {"method": "FBS", "bin_width": 25.0, "min_val": -1000}},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture", "histogram", "ivh"], "include_spatial_intensity": True, "include_local_intensity": True, "ivh_discretisation": {"method": "FBS", "bin_width": 2.5, "min_val": -1000}, "ivh_params": {"target_range_max": 400}}}
]

# C. Config D (2mm isotropic, 3-sigma outlier, FBN 32, Continuous IVH)
config_d = [
    {"step": "resample", "params": {"new_spacing": (2.0, 2.0, 2.0), "interpolation": "linear", "round_intensities": True}},
    {"step": "keep_largest_component", "params": {"apply_to": "morph"}},
    {"step": "filter_outliers", "params": {"sigma": 3.0}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture", "histogram", "ivh"], "include_spatial_intensity": True, "include_local_intensity": True, "ivh_use_continuous": True}}
]

# D. Config E (Cubic resamp, 3-sigma, round last, FBN 32 for tex, FBN 1000 for IVH)
config_e = [
    {"step": "resample", "params": {"new_spacing": (2.0, 2.0, 2.0), "interpolation": "cubic", "round_intensities": False}},
    {"step": "keep_largest_component", "params": {"apply_to": "morph"}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    {"step": "filter_outliers", "params": {"sigma": 3.0}},
    {"step": "round_intensities", "params": {}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture", "histogram", "ivh"], "include_spatial_intensity": True, "include_local_intensity": True, "ivh_discretisation": {"method": "FBN", "n_bins": 1000}, "ivh_params": {"bin_width": 1.0, "min_val": 0.5, "max_val": 1000.5}}}
]

# --- RUN PIPELINE ---
# Choose which config to run:
pipeline.add_config("ibsi_config_c", config_c)

# 3. Load image and mask
# For Config C/D/E (Lung Cancer CT)
image = load_image("path/to/CT_image.nii.gz")
mask = load_image("path/to/CT_mask.nii.gz")

# 4. Run extraction
results = pipeline.run(image, "ibsi_config_c", mask=mask)

# 5. Access results
print(results["ibsi_config_c"])
```

## Known Deviations
For configuration D, Zone size non-uniformity and Zone size entropy are known to fail with minimal differences, but pass for all other conffigurations. The following morphological features consistently fail across all configurations due to differences in surface mesh generation algorithms:

- **Compactness 2 (BQWJ)**: Sensitive to surface area calculation differences.
- **Asphericity (25C7)**: Derived from Compactness 2, inheriting the same deviation.

## Digital Phantom
### Morphology
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Volume | RNU0 | 556.3 | 556 | 4 | ✅ PASS |
| Volume voxel counting | YEKZ | 592 | 592 | 4 | ✅ PASS |
| Surface area | C0JK | 388.1 | 388 | 3 | ✅ PASS |
| Surface to volume ratio | 2PR5 | 0.6976 | 0.698 | 0.004 | ✅ PASS |
| Compactness 1 | SKGS | 0.04106 | 0.0411 | 0.0003 | ✅ PASS |
| Compactness 2 | BQWJ | 0.5989 | 0.599 | 0.004 | ✅ PASS |
| Spherical disproportion | KRCK | 1.186 | 1.19 | 0.01 | ✅ PASS |
| Sphericity | QCFX | 0.8429 | 0.843 | 0.005 | ✅ PASS |
| Asphericity | 25C7 | 0.1863 | 0.186 | 0.001 | ✅ PASS |
| Center of mass shift | KLMA | 0.6715 | 0.672 | 0.004 | ✅ PASS |
| Maximum 3D diameter | L0JK | 13.11 | 13.1 | 0.1 | ✅ PASS |
| Major axis length | TDIC | 11.4 | 11.4 | 0.1 | ✅ PASS |
| Minor axis length | P9VJ | 9.308 | 9.31 | 0.06 | ✅ PASS |
| Least axis length | 7J51 | 8.536 | 8.54 | 0.05 | ✅ PASS |
| Elongation | Q3CK | 0.8163 | 0.816 | 0.005 | ✅ PASS |
| Flatness | N17B | 0.7486 | 0.749 | 0.005 | ✅ PASS |
| Volume density (AABB) | PBX1 | 0.8693 | 0.869 | 0.005 | ✅ PASS |
| Area density (AABB) | R59B | 0.8662 | 0.866 | 0.005 | ✅ PASS |
| Volume density (OMBB) | ZH1A | 0.458 | — | — | ❗ REF. |
| Area density (OMBB) | IQYR | 0.5673 | — | — | ❗ REF. |
| Volume density (AEE) | 6BDE | 1.173 | 1.17 | 0.01 | ✅ PASS |
| Area density (AEE) | RDD2 | 1.355 | 1.36 | 0.01 | ✅ PASS |
| Volume density (MVEE) | SWZ1 | 0.513 | — | — | ❗ REF. |
| Area density (MVEE) | BRI8 | 0.7909 | — | — | ❗ REF. |
| Volume density (convex hull) | R3ER | 0.9609 | 0.961 | 0.006 | ✅ PASS |
| Area density (convex hull) | 7T7F | 1.033 | 1.03 | 0.01 | ✅ PASS |

### Intensity
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Integrated intensity | 99N0 | 1195 | 1.2e3 | 10 | ✅ PASS |
| Moran's I index | N365 | 0.0397 | 0.0397 | 0.0003 | ✅ PASS |
| Geary's C measure | NPT7 | 0.974 | 0.974 | 0.006 | ✅ PASS |
| Local intensity peak  | VJGA | 2.6 | 2.6 | — | ✅ PASS |
| Global intensity peak | 0F91 | 3.103 | 3.1 | — | ✅ PASS |
| Mean intensity | Q4LE | 2.149 | 2.15 | — | ✅ PASS |
| Intensity variance | ECT3 | 3.045 | 3.05 | — | ✅ PASS |
| Intensity skewness | KE2A | 1.084 | 1.08 | — | ✅ PASS |
| Intensity kurtosis | IPH6 | -0.3546 | -0.355 | — | ✅ PASS |
| Median intensity | Y12H | 1 | 1 | — | ✅ PASS |
| Minimum intensity | 1GSF | 1 | 1 | — | ✅ PASS |
| 10th intensity percentile | QG58 | 1 | 1 | — | ✅ PASS |
| 90th intensity percentile | 8DWT | 4 | 4 | — | ✅ PASS |
| Maximum intensity | 84IY | 6 | 6 | — | ✅ PASS |
| Intensity interquartile range | SALO | 3 | 3 | — | ✅ PASS |
| Intensity range | 2OJQ | 5 | 5 | — | ✅ PASS |
| Intensity Mean absolute deviation | 4FUA | 1.552 | 1.55 | — | ✅ PASS |
| Intensity Robust mean absolute deviation | 1128 | 1.114 | 1.11 | — | ✅ PASS |
| Intensity Median absolute deviation | N72L | 1.149 | 1.15 | — | ✅ PASS |
| Intensity Coefficient of variation | 7TET | 0.8122 | 0.812 | — | ✅ PASS |
| Intensity Quartile coefficient of dispersion | 9S40 | 0.6 | 0.6 | — | ✅ PASS |
| Intensity energy | N8CA | 567 | 567 | — | ✅ PASS |
| Root mean square intensity | 5ZWQ | 2.768 | 2.77 | — | ✅ PASS |

### Intensity Histogram
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Mean discretised intensity | X6K6 | 2.149 | 2.15 | — | ✅ PASS |
| Discretised intensity variance | CH89 | 3.045 | 3.05 | — | ✅ PASS |
| Discretised intensity skewness | 88K1 | 1.084 | 1.08 | — | ✅ PASS |
| Discretised intensity kurtosis | C3I7 | -0.3546 | -0.355 | — | ✅ PASS |
| Median discretised intensity | WIFQ | 1 | 1 | — | ✅ PASS |
| Minimum discretised intensity | 1PR8 | 1 | 1 | — | ✅ PASS |
| 10th discretised intensity percentile | 1PR | 1 | 1 | — | ✅ PASS |
| 90th discretised intensity percentile | GPMT | 4 | 4 | — | ✅ PASS |
| Maximum discretised intensity | 3NCY | 6 | 6 | — | ✅ PASS |
| Intensity histogram mode | AMMC | 1 | 1 | — | ✅ PASS |
| Discretised intensity interquartile range | WR0O | 3 | 3 | — | ✅ PASS |
| Discretised intensity range | 5Z3W | 5 | 5 | — | ✅ PASS |
| Intensity histogram mean absolute deviation | D2ZX | 1.552 | 1.55 | — | ✅ PASS |
| Intensity histogram robust mean absolute deviation | WRZB | 1.114 | 1.11 | — | ✅ PASS |
| Intensity histogram median absolute deviation | 4RNL | 1.149 | 1.15 | — | ✅ PASS |
| Intensity histogram coefficient of variation | CWYJ | 0.8122 | 0.812 | — | ✅ PASS |
| Intensity histogram quartile coefficient of dispersion | SLWD | 0.6 | 0.6 | — | ✅ PASS |
| Discretised intensity entropy | TLU2 | 1.266 | 1.27 | — | ✅ PASS |
| Discretised intensity uniformity | BJ5W | 0.5124 | 0.512 | — | ✅ PASS |
| Maximum histogram gradient | 12CE | 8 | 8 | — | ✅ PASS |
| Maximum histogram gradient intensity | 8E6O | 3 | 3 | — | ✅ PASS |
| Minimum histogram gradient | VQB3 | -50 | -50 | — | ✅ PASS |
| Minimum histogram gradient intensity | RHQZ | 1 | 1 | — | ✅ PASS |

### Intensity-Volume Histogram
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Volume at intensity fraction 0.10 | BC2M_10 | 0.3243 | 0.324 | — | ✅ PASS |
| Volume at intensity fraction 0.90 | BC2M_90 | 0.09459 | 0.0946 | — | ✅ PASS |
| Intensity at volume fraction 0.10 | GBPN_10 | 5 | 5 | — | ✅ PASS |
| Intensity at volume fraction 0.90 | GBPN_90 | 2 | 2 | — | ✅ PASS |
| Volume fraction difference between intensity 0.10 and 0.90 fractions | DDTU | 0.2297 | 0.23 | — | ✅ PASS |
| Intensity fraction difference between volume 0.10 and 0.90 fractions | CNV2 | 3 | 3 | — | ✅ PASS |
| Area under the IVH curve | 9CMM | 2.047 | — | — | ❗ REF. |

### GLCM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Joint maximum | GYBY | 0.5085 | 0.509 | — | ✅ PASS |
| Joint average | 60VM | 2.149 | 2.15 | — | ✅ PASS |
| Joint variance | UR99 | 3.132 | 3.13 | — | ✅ PASS |
| Joint entropy | TU9B | 2.574 | 2.57 | — | ✅ PASS |
| Difference average | TF7R | 1.38 | 1.38 | — | ✅ PASS |
| Difference variance | D3YU | 3.215 | 3.21 | — | ✅ PASS |
| Difference entropy | NTRS | 1.641 | 1.64 | — | ✅ PASS |
| Sum average | ZGXS | 4.298 | 4.3 | — | ✅ PASS |
| Sum variance | OEEB | 7.412 | 7.41 | — | ✅ PASS |
| Sum entropy | P6QZ | 2.11 | 2.11 | — | ✅ PASS |
| Angular second moment | 8ZQL | 0.291 | 0.291 | — | ✅ PASS |
| Contrast | ACUI | 5.118 | 5.12 | — | ✅ PASS |
| Dissimilarity | 8S9J | 1.38 | 1.38 | — | ✅ PASS |
| Inverse difference | IB1Z | 0.6877 | 0.688 | — | ✅ PASS |
| Normalised inverse difference | NDRX | 0.8559 | 0.856 | — | ✅ PASS |
| Inverse difference moment | WF0Z | 0.6306 | 0.631 | — | ✅ PASS |
| Normalised inverse difference moment | 1QCO | 0.9022 | 0.902 | — | ✅ PASS |
| Inverse variance | E8JP | 0.05744 | 0.0574 | — | ✅ PASS |
| Correlation | NI2N | 0.1831 | 0.183 | — | ✅ PASS |
| Autocorrelation | QWB0 | 5.192 | 5.19 | — | ✅ PASS |
| Cluster tendency | DG8W | 7.412 | 7.41 | — | ✅ PASS |
| Cluster shade | 7NFM | 17.42 | 17.4 | — | ✅ PASS |
| Cluster prominence | AE86 | 147.5 | 147 | — | ✅ PASS |
| Information correlation 1 | R8DG | -0.0288 | -0.0288 | — | ✅ PASS |
| Information correlation 2 | JN9H | 0.2692 | 0.269 | — | ✅ PASS |

### GLRLM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Short runs emphasis | 22OV | 0.7291 | 0.729 | — | ✅ PASS |
| Long runs emphasis | W4KF | 2.761 | 2.76 | — | ✅ PASS |
| Low grey level run emphasis | V3SW | 0.6067 | 0.607 | — | ✅ PASS |
| High grey level run emphasis | G3QZ | 9.638 | 9.64 | — | ✅ PASS |
| Short run low grey level emphasis | HTZT | 0.3716 | 0.372 | — | ✅ PASS |
| Short run high grey level emphasis | GD3A | 8.672 | 8.67 | — | ✅ PASS |
| Long run low grey level emphasis | IVPO | 2.163 | 2.16 | — | ✅ PASS |
| Long run high grey level emphasis | 3KUM | 15.63 | 15.6 | — | ✅ PASS |
| Grey level non-uniformity | R5YN | 281.3 | 281 | — | ✅ PASS |
| Normalised grey level non-uniformity | OVBL | 0.4301 | 0.43 | — | ✅ PASS |
| Run length non-uniformity | W92Y | 327.7 | 328 | — | ✅ PASS |
| Normalised run length non-uniformity | IC23 | 0.5011 | 0.501 | — | ✅ PASS |
| Run percentage | 9ZK5 | 0.6798 | 0.68 | — | ✅ PASS |
| Grey level variance | 8CE5 | 3.479 | 3.48 | — | ✅ PASS |
| Run length variance | SXLW | 0.5978 | 0.598 | — | ✅ PASS |
| Run entropy | HJ9O | 2.624 | 2.62 | — | ✅ PASS |

### GLSZM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Small zone emphasis | P001 | 0.2552 | 0.255 | — | ✅ PASS |
| Large zone emphasis | 48P8 | 550 | 550 | — | ✅ PASS |
| Low grey level zone emphasis | XMSY | 0.2528 | 0.253 | — | ✅ PASS |
| High grey level zone emphasis | 5GN9 | 15.6 | 15.6 | — | ✅ PASS |
| Small zone low grey level emphasis | 5RAI | 0.0256 | 0.0256 | — | ✅ PASS |
| Small zone high grey level emphasis | HW1V | 2.763 | 2.76 | — | ✅ PASS |
| Large zone low grey level emphasis | YH51 | 502.8 | 503 | — | ✅ PASS |
| Large zone high grey level emphasis | J17V | 1495 | 1.49e3 | — | ✅ PASS |
| Grey level non-uniformity | JNSA | 1.4 | 1.4 | — | ✅ PASS |
| Normalised grey level non-uniformity | Y1RO | 0.28 | 0.28 | — | ✅ PASS |
| Zone size non-uniformity | 4JP3 | 1 | 1 | — | ✅ PASS |
| Normalised zone size non-uniformity | VB3A | 0.2 | 0.2 | — | ✅ PASS |
| Zone percentage | P30P | 0.06757 | 0.0676 | — | ✅ PASS |
| Grey level variance | BYLV | 2.64 | 2.64 | — | ✅ PASS |
| Zone size variance | 3NSA | 331 | 331 | — | ✅ PASS |
| Zone size entropy | GU8N | 2.322 | 2.32 | — | ✅ PASS |

### GLDZM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Small distance emphasis | 0GBI | 1 | 1 | — | ✅ PASS |
| Large distance emphasis | MB4I | 1 | 1 | — | ✅ PASS |
| Low grey level zone emphasis | S1RA | 0.2528 | 0.253 | — | ✅ PASS |
| High grey level zone emphasis | K26C | 15.6 | 15.6 | — | ✅ PASS |
| Small distance low grey level emphasis | RUVG | 0.2528 | 0.253 | — | ✅ PASS |
| Small distance high grey level emphasis | DKNJ | 15.6 | 15.6 | — | ✅ PASS |
| Large distance low grey level emphasis | A7WM | 0.2528 | 0.253 | — | ✅ PASS |
| Large distance high grey level emphasis | KLTH | 15.6 | 15.6 | — | ✅ PASS |
| Grey level non-uniformity | VFT7 | 1.4 | 1.4 | — | ✅ PASS |
| Normalised grey level non-uniformity | 7HP3 | 0.28 | 0.28 | — | ✅ PASS |
| Zone distance non-uniformity | V294 | 5 | 5 | — | ✅ PASS |
| Normalised zone distance non-uniformity | IATH | 1 | 1 | — | ✅ PASS |
| Zone percentage | VIWW | 0.06757 | 0.0676 | — | ✅ PASS |
| Grey level variance | QK93 | 2.64 | 2.64 | — | ✅ PASS |
| Zone distance variance | 7WT1 | 0 | 0 | — | ✅ PASS |
| Zone distance entropy | GBDU | 1.922 | 1.92 | — | ✅ PASS |

### NGTDM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Coarseness | QCDE | 0.0296 | 0.0296 | — | ✅ PASS |
| Contrast | 65HE | 0.5837 | 0.584 | — | ✅ PASS |
| Busyness | NQ30 | 6.544 | 6.54 | — | ✅ PASS |
| Complexity | HDEZ | 13.54 | 13.5 | — | ✅ PASS |
| Strength | 1X9X | 0.7635 | 0.763 | — | ✅ PASS |

### NGLDM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Low dependence emphasis | SODN | 0.045 | 0.045 | — | ✅ PASS |
| High dependence emphasis | IMOQ | 109 | 109 | — | ✅ PASS |
| Low grey level count emphasis | TL9H | 0.6933 | 0.693 | — | ✅ PASS |
| High grey level count emphasis | OAE7 | 7.662 | 7.66 | — | ✅ PASS |
| Low dependence low grey level emphasis | EQ3F | 0.009631 | 0.00963 | — | ✅ PASS |
| Low dependence high grey level emphasis | JA6D | 0.7362 | 0.736 | — | ✅ PASS |
| High dependence low grey level emphasis | NBZI | 102.5 | 102 | — | ✅ PASS |
| High dependence high grey level emphasis | 9QMG | 235 | 235 | — | ✅ PASS |
| Grey level non-uniformity | FP8K | 37.92 | 37.9 | — | ✅ PASS |
| Normalised grey level non-uniformity | 5SPA | 0.5124 | 0.512 | — | ✅ PASS |
| Dependence count non-uniformity | Z87G | 4.865 | 4.86 | — | ✅ PASS |
| Normalised dependence count non-uniformity | OKJI | 0.06574 | 0.0657 | — | ✅ PASS |
| Dependence count percentage | 6XV8 | 1 | 1 | — | ✅ PASS |
| Grey level variance | 1PFV | 3.045 | 3.05 | — | ✅ PASS |
| Dependence count variance | DNX2 | 22.06 | 22.1 | — | ✅ PASS |
| Dependence count entropy | FCBV | 4.404 | 4.4 | — | ✅ PASS |
| Dependence count energy | CAS9 | 0.05332 | 0.0533 | — | ✅ PASS |

## Config C
### Morphology
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Volume | RNU0 | 3.672e+05 | 3.67e5 | 6e3 | ✅ PASS |
| Volume voxel counting | YEKZ | 3.675e+05 | 3.68e5 | 6e3 | ✅ PASS |
| Surface area | C0JK | 3.446e+04 | 3.43e4 | 400 | ✅ PASS |
| Surface to volume ratio | 2PR5 | 0.09384 | 0.0934 | 0.0007 | ✅ PASS |
| Compactness 1 | SKGS | 0.03239 | — | — | ❗ REF. |
| Compactness 2 | BQWJ | 0.3727 | 0.378 | 0.004 | ❌ FAIL |
| Spherical disproportion | KRCK | 1.39 | 1.38 | 0.01 | ✅ PASS |
| Sphericity | QCFX | 0.7196 | 0.723 | 0.003 | ✅ PASS |
| Asphericity | 25C7 | 0.3896 | 0.383 | 0.004 | ❌ FAIL |
| Center of mass shift | KLMA | 45.64 | 45.6 | 2.8 | ✅ PASS |
| Maximum 3D diameter | L0JK | 125.1 | 125 | 1 | ✅ PASS |
| Major axis length | TDIC | 93.26 | 93.3 | 0.5 | ✅ PASS |
| Minor axis length | P9VJ | 81.98 | 82 | 0.5 | ✅ PASS |
| Least axis length | 7J51 | 70.88 | 70.9 | 0.4 | ✅ PASS |
| Elongation | Q3CK | 0.879 | 0.879 | 0.001 | ✅ PASS |
| Flatness | N17B | 0.76 | 0.76 | 0.001 | ✅ PASS |
| Volume density (AABB) | PBX1 | 0.4779 | 0.478 | 0.003 | ✅ PASS |
| Area density (AABB) | R59B | 0.6814 | 0.678 | 0.003 | ✅ PASS |
| Volume density (OMBB) | ZH1A | 0.3405 | — | — | ❗ REF. |
| Area density (OMBB) | IQYR | 0.5412 | — | — | ❗ REF. |
| Volume density (AEE) | 6BDE | 1.294 | 1.29 | 0.01 | ✅ PASS |
| Area density (AEE) | RDD2 | 1.625 | 1.62 | 0.01 | ✅ PASS |
| Volume density (MVEE) | SWZ1 | 0.4956 | — | — | ❗ REF. |
| Area density (MVEE) | BRI8 | 0.8844 | — | — | ❗ REF. |
| Volume density (convex hull) | R3ER | 0.8331 | 0.834 | 0.002 | ✅ PASS |
| Area density (convex hull) | 7T7F | 1.135 | 1.13 | 0.01 | ✅ PASS |

### Intensity
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Integrated intensity | 99N0 | -1.794e+07 | -1.8e7 | 1.4e6 | ✅ PASS |
| Moran's I index | N365 | 0.08246 | 0.0824 | 0.0003 | ✅ PASS |
| Geary's C measure | NPT7 | 0.8461 | 0.846 | 0.001 | ✅ PASS |
| Local intensity peak  | VJGA | 168.6 | 169 | 10 | ✅ PASS |
| Global intensity peak | 0F91 | 179.7 | 180 | 5 | ✅ PASS |
| Mean intensity | Q4LE | -48.85 | -49 | 2.9 | ✅ PASS |
| Intensity variance | ECT3 | 5.056e+04 | 5.06e4 | 1.4e3 | ✅ PASS |
| Intensity skewness | KE2A | -2.142 | -2.14 | 0.05 | ✅ PASS |
| Intensity kurtosis | IPH6 | 3.534 | 3.53 | 0.23 | ✅ PASS |
| Median intensity | Y12H | 40 | 40 | 0.4 | ✅ PASS |
| Minimum intensity | 1GSF | -939 | -939 | 4 | ✅ PASS |
| 10th intensity percentile | QG58 | -424 | -424 | 14 | ✅ PASS |
| 90th intensity percentile | 8DWT | 86 | 86 | 0.1 | ✅ PASS |
| Maximum intensity | 84IY | 393 | 393 | 10 | ✅ PASS |
| Intensity interquartile range | SALO | 67 | 67 | 4.9 | ✅ PASS |
| Intensity range | 2OJQ | 1332 | 1.33e3 | 20 | ✅ PASS |
| Intensity Mean absolute deviation | 4FUA | 157.8 | 158 | 4 | ✅ PASS |
| Intensity Robust mean absolute deviation | 1128 | 66.74 | 66.8 | 3.5 | ✅ PASS |
| Intensity Median absolute deviation | N72L | 119 | 119 | 4 | ✅ PASS |
| Intensity Coefficient of variation | 7TET | -4.603 | -4.59 | 0.29 | ✅ PASS |
| Intensity Quartile coefficient of dispersion | 9S40 | 1.031 | 1.03 | 0.4 | ✅ PASS |
| Intensity energy | N8CA | 2.432e+09 | 2.44e9 | 1.2e8 | ✅ PASS |
| Root mean square intensity | 5ZWQ | 230.1 | 230 | 4 | ✅ PASS |

### Intensity Histogram
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Mean discretised intensity | X6K6 | 38.56 | 38.6 | 0.2 | ✅ PASS |
| Discretised intensity variance | CH89 | 80.99 | 81.1 | 2.1 | ✅ PASS |
| Discretised intensity skewness | 88K1 | -2.139 | -2.14 | 0.05 | ✅ PASS |
| Discretised intensity kurtosis | C3I7 | 3.528 | 3.52 | 0.23 | ✅ PASS |
| Median discretised intensity | WIFQ | 42 | 42 | — | ✅ PASS |
| Minimum discretised intensity | 1PR8 | 3 | 3 | 0.16 | ✅ PASS |
| 10th discretised intensity percentile | 1PR | 24 | 24 | 0.7 | ✅ PASS |
| 90th discretised intensity percentile | GPMT | 44 | 44 | — | ✅ PASS |
| Maximum discretised intensity | 3NCY | 56 | 56 | 0.5 | ✅ PASS |
| Intensity histogram mode | AMMC | 43 | 43 | 0.1 | ✅ PASS |
| Discretised intensity interquartile range | WR0O | 3 | 3 | 0.21 | ✅ PASS |
| Discretised intensity range | 5Z3W | 53 | 53 | 0.6 | ✅ PASS |
| Intensity histogram mean absolute deviation | D2ZX | 6.314 | 6.32 | 0.15 | ✅ PASS |
| Intensity histogram robust mean absolute deviation | WRZB | 2.587 | 2.59 | 0.14 | ✅ PASS |
| Intensity histogram median absolute deviation | 4RNL | 4.745 | 4.75 | 0.12 | ✅ PASS |
| Intensity histogram coefficient of variation | CWYJ | 0.2334 | 0.234 | 0.005 | ✅ PASS |
| Intensity histogram quartile coefficient of dispersion | SLWD | 0.03614 | 0.0361 | 0.0027 | ✅ PASS |
| Discretised intensity entropy | TLU2 | 3.734 | 3.73 | 0.04 | ✅ PASS |
| Discretised intensity uniformity | BJ5W | 0.1396 | 0.14 | 0.003 | ✅ PASS |
| Maximum histogram gradient | 12CE | 4740 | 4.75e3 | 30 | ✅ PASS |
| Maximum histogram gradient intensity | 8E6O | 41 | 41 | — | ✅ PASS |
| Minimum histogram gradient | VQB3 | -4672 | -4.68e3 | 50 | ✅ PASS |
| Minimum histogram gradient intensity | RHQZ | 44 | 44 | — | ✅ PASS |

### Intensity-Volume Histogram
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Volume at intensity fraction 0.10 | BC2M_10 | 0.9976 | 0.998 | 0.001 | ✅ PASS |
| Volume at intensity fraction 0.90 | BC2M_90 | 0.0001524 | 0.000152 | 2e-5 | ✅ PASS |
| Intensity at volume fraction 0.10 | GBPN_10 | 88.75 | 88.8 | 0.2 | ✅ PASS |
| Intensity at volume fraction 0.90 | GBPN_90 | -421.2 | -421 | 14 | ✅ PASS |
| Volume fraction difference between intensity 0.10 and 0.90 fractions | DDTU | 0.9974 | 0.997 | 0.001 | ✅ PASS |
| Intensity fraction difference between volume 0.10 and 0.90 fractions | CNV2 | 510 | 510 | 14 | ✅ PASS |
| Area under the IVH curve | 9CMM | 891.4 | — | — | ❗ REF. |

### GLCM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Joint maximum | GYBY | 0.1109 | 0.111 | 0.002 | ✅ PASS |
| Joint average | 60VM | 38.98 | 39 | 0.2 | ✅ PASS |
| Joint variance | UR99 | 73.68 | 73.8 | 2 | ✅ PASS |
| Joint entropy | TU9B | 6.419 | 6.42 | 0.06 | ✅ PASS |
| Difference average | TF7R | 2.162 | 2.16 | 0.05 | ✅ PASS |
| Difference variance | D3YU | 14.42 | 14.4 | 0.5 | ✅ PASS |
| Difference entropy | NTRS | 2.642 | 2.64 | 0.03 | ✅ PASS |
| Sum average | ZGXS | 77.96 | 78 | 0.3 | ✅ PASS |
| Sum variance | OEEB | 275.6 | 276 | 8 | ✅ PASS |
| Sum entropy | P6QZ | 4.559 | 4.56 | 0.04 | ✅ PASS |
| Angular second moment | 8ZQL | 0.04471 | 0.0447 | 0.001 | ✅ PASS |
| Contrast | ACUI | 19.09 | 19.1 | 0.7 | ✅ PASS |
| Dissimilarity | 8S9J | 2.162 | 2.16 | 0.05 | ✅ PASS |
| Inverse difference | IB1Z | 0.5828 | 0.583 | 0.004 | ✅ PASS |
| Normalised inverse difference | NDRX | 0.9652 | 0.966 | 0.001 | ✅ PASS |
| Inverse difference moment | WF0Z | 0.5479 | 0.548 | 0.004 | ✅ PASS |
| Normalised inverse difference moment | 1QCO | 0.994 | 0.994 | 0.001 | ✅ PASS |
| Inverse variance | E8JP | 0.3906 | 0.39 | 0.003 | ✅ PASS |
| Correlation | NI2N | 0.8704 | 0.871 | 0.001 | ✅ PASS |
| Autocorrelation | QWB0 | 1584 | 1.58e3 | 10 | ✅ PASS |
| Cluster tendency | DG8W | 275.6 | 276 | 8 | ✅ PASS |
| Cluster shade | 7NFM | -1.062e+04 | -1.06e4 | 300 | ✅ PASS |
| Cluster prominence | AE86 | 5.69e+05 | 5.7e5 | 1.1e4 | ✅ PASS |
| Information correlation 1 | R8DG | -0.2283 | -0.228 | 0.001 | ✅ PASS |
| Information correlation 2 | JN9H | 0.8993 | 0.899 | 0.001 | ✅ PASS |

### GLRLM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Short runs emphasis | 22OV | 0.7872 | 0.787 | 0.003 | ✅ PASS |
| Long runs emphasis | W4KF | 3.275 | 3.28 | 0.04 | ✅ PASS |
| Low grey level run emphasis | V3SW | 0.001544 | 0.00155 | 5e-5 | ✅ PASS |
| High grey level run emphasis | G3QZ | 1473 | 1.47e3 | 10 | ✅ PASS |
| Short run low grey level emphasis | HTZT | 0.001358 | 0.00136 | 5e-5 | ✅ PASS |
| Short run high grey level emphasis | GD3A | 1100 | 1.1e3 | 10 | ✅ PASS |
| Long run low grey level emphasis | IVPO | 0.003141 | 0.00314 | 4e-5 | ✅ PASS |
| Long run high grey level emphasis | 3KUM | 5523 | 5.53e3 | 80 | ✅ PASS |
| Grey level non-uniformity | R5YN | 4.129e+04 | 4.13e4 | 100 | ✅ PASS |
| Normalised grey level non-uniformity | OVBL | 0.1018 | 0.102 | 0.003 | ✅ PASS |
| Run length non-uniformity | W92Y | 2.334e+05 | 2.34e5 | 6e3 | ✅ PASS |
| Normalised run length non-uniformity | IC23 | 0.5755 | 0.575 | 0.004 | ✅ PASS |
| Run percentage | 9ZK5 | 0.6792 | 0.679 | 0.003 | ✅ PASS |
| Grey level variance | 8CE5 | 101.2 | 101 | 3 | ✅ PASS |
| Run length variance | SXLW | 1.107 | 1.11 | 0.02 | ✅ PASS |
| Run entropy | HJ9O | 5.349 | 5.35 | 0.03 | ✅ PASS |

### GLSZM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Small zone emphasis | P001 | 0.6951 | 0.695 | 0.001 | ✅ PASS |
| Large zone emphasis | 48P8 | 3.892e+04 | 3.89e4 | 900 | ✅ PASS |
| Low grey level zone emphasis | XMSY | 0.002342 | 0.00235 | 6e-5 | ✅ PASS |
| High grey level zone emphasis | 5GN9 | 971.7 | 971 | 7 | ✅ PASS |
| Small zone low grey level emphasis | 5RAI | 0.001588 | 0.0016 | 4e-5 | ✅ PASS |
| Small zone high grey level emphasis | HW1V | 657.8 | 657 | 4 | ✅ PASS |
| Large zone low grey level emphasis | YH51 | 21.55 | 21.6 | 0.5 | ✅ PASS |
| Large zone high grey level emphasis | J17V | 7.071e+07 | 7.07e7 | 1.5e6 | ✅ PASS |
| Grey level non-uniformity | JNSA | 195 | 195 | 6 | ✅ PASS |
| Normalised grey level non-uniformity | Y1RO | 0.02868 | 0.0286 | 0.0003 | ✅ PASS |
| Zone size non-uniformity | 4JP3 | 3039 | 3.04e3 | 100 | ✅ PASS |
| Normalised zone size non-uniformity | VB3A | 0.447 | 0.447 | 0.001 | ✅ PASS |
| Zone percentage | P30P | 0.148 | 0.148 | 0.003 | ✅ PASS |
| Grey level variance | BYLV | 105.7 | 106 | 1 | ✅ PASS |
| Zone size variance | 3NSA | 3.888e+04 | 3.89e4 | 900 | ✅ PASS |
| Zone size entropy | GU8N | 6.997 | 7 | 0.01 | ✅ PASS |

### GLDZM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Small distance emphasis | 0GBI | 0.5312 | 0.531 | 0.006 | ✅ PASS |
| Large distance emphasis | MB4I | 11.01 | 11 | 0.3 | ✅ PASS |
| Low grey level zone emphasis | S1RA | 0.002342 | 0.00235 | 6e-5 | ✅ PASS |
| High grey level zone emphasis | K26C | 971.7 | 971 | 7 | ✅ PASS |
| Small distance low grey level emphasis | RUVG | 0.001487 | 0.00149 | 4e-5 | ✅ PASS |
| Small distance high grey level emphasis | DKNJ | 476.7 | 476 | 11 | ✅ PASS |
| Large distance low grey level emphasis | A7WM | 0.0154 | 0.0154 | 0.0005 | ✅ PASS |
| Large distance high grey level emphasis | KLTH | 1.334e+04 | 1.34e4 | 200 | ✅ PASS |
| Grey level non-uniformity | VFT7 | 195 | 195 | 6 | ✅ PASS |
| Normalised grey level non-uniformity | 7HP3 | 0.02868 | 0.0286 | 0.0003 | ✅ PASS |
| Zone distance non-uniformity | V294 | 1864 | 1.87e3 | 40 | ✅ PASS |
| Normalised zone distance non-uniformity | IATH | 0.2742 | 0.274 | 0.005 | ✅ PASS |
| Zone percentage | VIWW | 0.148 | 0.148 | 0.003 | ✅ PASS |
| Grey level variance | QK93 | 105.7 | 106 | 1 | ✅ PASS |
| Zone distance variance | 7WT1 | 4.594 | 4.6 | 0.06 | ✅ PASS |
| Zone distance entropy | GBDU | 7.562 | 7.56 | 0.03 | ✅ PASS |

### NGTDM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Coarseness | QCDE | 0.0002163 | 0.000216 | 4e-6 | ✅ PASS |
| Contrast | 65HE | 0.0872 | 0.0873 | 0.0019 | ✅ PASS |
| Busyness | NQ30 | 1.388 | 1.39 | 0.01 | ✅ PASS |
| Complexity | HDEZ | 1808 | 1.81e3 | 60 | ✅ PASS |
| Strength | 1X9X | 0.6517 | 0.651 | 0.015 | ✅ PASS |

### NGLDM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Low dependence emphasis | SODN | 0.1368 | 0.137 | 0.003 | ✅ PASS |
| High dependence emphasis | IMOQ | 126.4 | 126 | 2 | ✅ PASS |
| Low grey level count emphasis | TL9H | 0.001296 | 0.0013 | 4e-5 | ✅ PASS |
| High grey level count emphasis | OAE7 | 1568 | 1.57e3 | 10 | ✅ PASS |
| Low dependence low grey level emphasis | EQ3F | 0.0003046 | 0.000306 | 1.2e-5 | ✅ PASS |
| Low dependence high grey level emphasis | JA6D | 140.7 | 141 | 2 | ✅ PASS |
| High dependence low grey level emphasis | NBZI | 0.08278 | 0.0828 | 0.0003 | ✅ PASS |
| High dependence high grey level emphasis | 9QMG | 2.266e+05 | 2.27e5 | 3e3 | ✅ PASS |
| Grey level non-uniformity | FP8K | 6412 | 6.42e3 | 10 | ✅ PASS |
| Normalised grey level non-uniformity | 5SPA | 0.1396 | 0.14 | 0.003 | ✅ PASS |
| Dependence count non-uniformity | Z87G | 2445 | 2.45e3 | 60 | ✅ PASS |
| Normalised dependence count non-uniformity | OKJI | 0.05322 | 0.0532 | 0.0005 | ✅ PASS |
| Dependence count percentage | 6XV8 | 1 | 1 | — | ✅ PASS |
| Grey level variance | 1PFV | 80.99 | 81.1 | 2.1 | ✅ PASS |
| Dependence count variance | DNX2 | 39.18 | 39.2 | 0.1 | ✅ PASS |
| Dependence count entropy | FCBV | 7.536 | 7.54 | 0.03 | ✅ PASS |
| Dependence count energy | CAS9 | 0.007893 | 0.00789 | 0.00011 | ✅ PASS |

## Config D
### Morphology
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Volume | RNU0 | 3.672e+05 | 3.67e5 | 6e3 | ✅ PASS |
| Volume voxel counting | YEKZ | 3.675e+05 | 3.68e5 | 6e3 | ✅ PASS |
| Surface area | C0JK | 3.446e+04 | 3.43e4 | 400 | ✅ PASS |
| Surface to volume ratio | 2PR5 | 0.09384 | 0.0934 | 0.0007 | ✅ PASS |
| Compactness 1 | SKGS | 0.03239 | 0.0326 | 0.0002 | ✅ PASS |
| Compactness 2 | BQWJ | 0.3727 | 0.378 | 0.004 | ❌ FAIL |
| Spherical disproportion | KRCK | 1.39 | 1.38 | 0.01 | ✅ PASS |
| Sphericity | QCFX | 0.7196 | 0.723 | 0.003 | ✅ PASS |
| Asphericity | 25C7 | 0.3896 | 0.383 | 0.004 | ❌ FAIL |
| Center of mass shift | KLMA | 65.29 | 64.9 | 2.8 | ✅ PASS |
| Maximum 3D diameter | L0JK | 125.1 | 125 | 1 | ✅ PASS |
| Major axis length | TDIC | 93.26 | 93.3 | 0.5 | ✅ PASS |
| Minor axis length | P9VJ | 81.98 | 82 | 0.5 | ✅ PASS |
| Least axis length | 7J51 | 70.88 | 70.9 | 0.4 | ✅ PASS |
| Elongation | Q3CK | 0.879 | 0.879 | 0.001 | ✅ PASS |
| Flatness | N17B | 0.76 | 0.76 | 0.001 | ✅ PASS |
| Volume density (AABB) | PBX1 | 0.4779 | 0.478 | 0.003 | ✅ PASS |
| Area density (AABB) | R59B | 0.6814 | 0.678 | 0.003 | ✅ PASS |
| Volume density (OMBB) | ZH1A | 0.3405 | — | — | ❗ REF. |
| Area density (OMBB) | IQYR | 0.5412 | — | — | ❗ REF. |
| Volume density (AEE) | 6BDE | 1.294 | 1.29 | 0.01 | ✅ PASS |
| Area density (AEE) | RDD2 | 1.625 | 1.62 | 0.01 | ✅ PASS |
| Volume density (MVEE) | SWZ1 | 0.4956 | — | — | ❗ REF. |
| Area density (MVEE) | BRI8 | 0.8844 | — | — | ❗ REF. |
| Volume density (convex hull) | R3ER | 0.8331 | 0.834 | 0.002 | ✅ PASS |
| Area density (convex hull) | 7T7F | 1.135 | 1.13 | 0.01 | ✅ PASS |

### Intensity
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Integrated intensity | 99N0 | -8.551e+06 | -8.64e6 | 1.56e6 | ✅ PASS |
| Moran's I index | N365 | 0.06202 | 0.0622 | 0.0013 | ✅ PASS |
| Geary's C measure | NPT7 | 0.8512 | 0.851 | 0.001 | ✅ PASS |
| Local intensity peak  | VJGA | 200.8 | 201 | 10 | ✅ PASS |
| Global intensity peak | 0F91 | 200.8 | 201 | 5 | ✅ PASS |
| Mean intensity | Q4LE | -23.29 | -23.5 | 3.9 | ✅ PASS |
| Intensity variance | ECT3 | 3.264e+04 | 3.28e4 | 2.1e3 | ✅ PASS |
| Intensity skewness | KE2A | -2.282 | -2.28 | 0.06 | ✅ PASS |
| Intensity kurtosis | IPH6 | 4.361 | 4.35 | 0.32 | ✅ PASS |
| Median intensity | Y12H | 42 | 42 | 0.4 | ✅ PASS |
| Minimum intensity | 1GSF | -723 | -724 | 12 | ✅ PASS |
| 10th intensity percentile | QG58 | -304 | -304 | 20 | ✅ PASS |
| 90th intensity percentile | 8DWT | 86 | 86 | 0.1 | ✅ PASS |
| Maximum intensity | 84IY | 521 | 521 | 22 | ✅ PASS |
| Intensity interquartile range | SALO | 57 | 57 | 4.1 | ✅ PASS |
| Intensity range | 2OJQ | 1244 | 1.24e3 | 40 | ✅ PASS |
| Intensity Mean absolute deviation | 4FUA | 122.2 | 123 | 6 | ✅ PASS |
| Intensity Robust mean absolute deviation | 1128 | 46.82 | 46.8 | 3.6 | ✅ PASS |
| Intensity Median absolute deviation | N72L | 94.51 | 94.7 | 3.8 | ✅ PASS |
| Intensity Coefficient of variation | 7TET | -7.758 | -7.7 | 1.01 | ✅ PASS |
| Intensity Quartile coefficient of dispersion | 9S40 | 0.7403 | 0.74 | 0.011 | ✅ PASS |
| Intensity energy | N8CA | 1.474e+09 | 1.48e9 | 1.4e8 | ✅ PASS |
| Root mean square intensity | 5ZWQ | 182.2 | 183 | 7 | ✅ PASS |

### Intensity Histogram
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Mean discretised intensity | X6K6 | 18.51 | 18.5 | 0.5 | ✅ PASS |
| Discretised intensity variance | CH89 | 21.65 | 21.7 | 0.4 | ✅ PASS |
| Discretised intensity skewness | 88K1 | -2.27 | -2.27 | 0.06 | ✅ PASS |
| Discretised intensity kurtosis | C3I7 | 4.316 | 4.31 | 0.32 | ✅ PASS |
| Median discretised intensity | WIFQ | 20 | 20 | 0.5 | ✅ PASS |
| Minimum discretised intensity | 1PR8 | 1 | 1 | — | ✅ PASS |
| 10th discretised intensity percentile | 1PR | 11 | 11 | 0.7 | ✅ PASS |
| 90th discretised intensity percentile | GPMT | 21 | 21 | 0.5 | ✅ PASS |
| Maximum discretised intensity | 3NCY | 32 | 32 | — | ✅ PASS |
| Intensity histogram mode | AMMC | 20 | 20 | 0.4 | ✅ PASS |
| Discretised intensity interquartile range | WR0O | 2 | 2 | 0.06 | ✅ PASS |
| Discretised intensity range | 5Z3W | 31 | 31 | — | ✅ PASS |
| Intensity histogram mean absolute deviation | D2ZX | 3.148 | 3.15 | 0.05 | ✅ PASS |
| Intensity histogram robust mean absolute deviation | WRZB | 1.33 | 1.33 | 0.06 | ✅ PASS |
| Intensity histogram median absolute deviation | 4RNL | 2.404 | 2.41 | 0.04 | ✅ PASS |
| Intensity histogram coefficient of variation | CWYJ | 0.2515 | 0.252 | 0.006 | ✅ PASS |
| Intensity histogram quartile coefficient of dispersion | SLWD | 0.05 | 0.05 | 0.0021 | ✅ PASS |
| Discretised intensity entropy | TLU2 | 2.937 | 2.94 | 0.01 | ✅ PASS |
| Discretised intensity uniformity | BJ5W | 0.2289 | 0.229 | 0.003 | ✅ PASS |
| Maximum histogram gradient | 12CE | 7256 | 7.26e3 | 200 | ✅ PASS |
| Maximum histogram gradient intensity | 8E6O | 19 | 19 | 0.4 | ✅ PASS |
| Minimum histogram gradient | VQB3 | -6676 | -6.67e3 | 230 | ✅ PASS |
| Minimum histogram gradient intensity | RHQZ | 22 | 22 | 0.4 | ✅ PASS |

### Intensity-Volume Histogram
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Volume at intensity fraction 0.10 | BC2M_10 | 0.9716 | 0.972 | 0.003 | ✅ PASS |
| Volume at intensity fraction 0.90 | BC2M_90 | 9.005e-05 | 9e-5 | 0.000415 | ✅ PASS |
| Intensity at volume fraction 0.10 | GBPN_10 | 87 | 87 | 0.1 | ✅ PASS |
| Intensity at volume fraction 0.90 | GBPN_90 | -303 | -303 | 20 | ✅ PASS |
| Volume fraction difference between intensity 0.10 and 0.90 fractions | DDTU | 0.9715 | 0.971 | 0.001 | ✅ PASS |
| Intensity fraction difference between volume 0.10 and 0.90 fractions | CNV2 | 390 | 390 | 20 | ✅ PASS |
| Area under the IVH curve | 9CMM | 700.2 | — | — | ❗ REF. |

### GLCM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Joint maximum | GYBY | 0.2322 | 0.232 | 0.007 | ✅ PASS |
| Joint average | 60VM | 18.85 | 18.9 | 0.5 | ✅ PASS |
| Joint variance | UR99 | 17.59 | 17.6 | 0.4 | ✅ PASS |
| Joint entropy | TU9B | 4.96 | 4.96 | 0.03 | ✅ PASS |
| Difference average | TF7R | 1.29 | 1.29 | 0.01 | ✅ PASS |
| Difference variance | D3YU | 5.391 | 5.38 | 0.11 | ✅ PASS |
| Difference entropy | NTRS | 2.139 | 2.14 | 0.01 | ✅ PASS |
| Sum average | ZGXS | 37.71 | 37.7 | 0.8 | ✅ PASS |
| Sum variance | OEEB | 63.31 | 63.5 | 1.3 | ✅ PASS |
| Sum entropy | P6QZ | 3.677 | 3.68 | 0.02 | ✅ PASS |
| Angular second moment | 8ZQL | 0.1094 | 0.109 | 0.003 | ✅ PASS |
| Contrast | ACUI | 7.056 | 7.05 | 0.13 | ✅ PASS |
| Dissimilarity | 8S9J | 1.29 | 1.29 | 0.01 | ✅ PASS |
| Inverse difference | IB1Z | 0.6822 | 0.682 | 0.003 | ✅ PASS |
| Normalised inverse difference | NDRX | 0.9651 | 0.965 | 0.001 | ✅ PASS |
| Inverse difference moment | WF0Z | 0.657 | 0.657 | 0.003 | ✅ PASS |
| Normalised inverse difference moment | 1QCO | 0.9937 | 0.994 | 0.001 | ✅ PASS |
| Inverse variance | E8JP | 0.3404 | 0.34 | 0.005 | ✅ PASS |
| Correlation | NI2N | 0.7995 | 0.8 | 0.005 | ✅ PASS |
| Autocorrelation | QWB0 | 369.6 | 370 | 16 | ✅ PASS |
| Cluster tendency | DG8W | 63.31 | 63.5 | 1.3 | ✅ PASS |
| Cluster shade | 7NFM | -1271 | -1.28e3 | 40 | ✅ PASS |
| Cluster prominence | AE86 | 3.558e+04 | 3.57e4 | 1.5e3 | ✅ PASS |
| Information correlation 1 | R8DG | -0.2249 | -0.225 | 0.003 | ✅ PASS |
| Information correlation 2 | JN9H | 0.8459 | 0.846 | 0.003 | ✅ PASS |

### GLRLM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Short runs emphasis | 22OV | 0.7355 | 0.736 | 0.001 | ✅ PASS |
| Long runs emphasis | W4KF | 6.558 | 6.56 | 0.18 | ✅ PASS |
| Low grey level run emphasis | V3SW | 0.02563 | 0.0257 | 0.0012 | ✅ PASS |
| High grey level run emphasis | G3QZ | 326 | 326 | 17 | ✅ PASS |
| Short run low grey level emphasis | HTZT | 0.02319 | 0.0232 | 0.001 | ✅ PASS |
| Short run high grey level emphasis | GD3A | 219.3 | 219 | 13 | ✅ PASS |
| Long run low grey level emphasis | IVPO | 0.04771 | 0.0478 | 0.0031 | ✅ PASS |
| Long run high grey level emphasis | 3KUM | 2627 | 2.63e3 | 30 | ✅ PASS |
| Grey level non-uniformity | R5YN | 4.279e+04 | 4.28e4 | 200 | ✅ PASS |
| Normalised grey level non-uniformity | OVBL | 0.1339 | 0.134 | 0.002 | ✅ PASS |
| Run length non-uniformity | W92Y | 1.601e+05 | 1.6e5 | 3e3 | ✅ PASS |
| Normalised run length non-uniformity | IC23 | 0.5009 | 0.501 | 0.001 | ✅ PASS |
| Run percentage | 9ZK5 | 0.5536 | 0.554 | 0.005 | ✅ PASS |
| Grey level variance | 8CE5 | 31.4 | 31.4 | 0.4 | ✅ PASS |
| Run length variance | SXLW | 3.295 | 3.29 | 0.13 | ✅ PASS |
| Run entropy | HJ9O | 5.08 | 5.08 | 0.02 | ✅ PASS |

### GLSZM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Small zone emphasis | P001 | 0.6395 | 0.637 | 0.005 | ✅ PASS |
| Large zone emphasis | 48P8 | 9.863e+04 | 9.91e4 | 2.8e3 | ✅ PASS |
| Low grey level zone emphasis | XMSY | 0.04051 | 0.0409 | 0.0005 | ✅ PASS |
| High grey level zone emphasis | 5GN9 | 187.2 | 188 | 10 | ✅ PASS |
| Small zone low grey level emphasis | 5RAI | 0.02478 | 0.0248 | 0.0004 | ✅ PASS |
| Small zone high grey level emphasis | HW1V | 116.4 | 117 | 7 | ✅ PASS |
| Large zone low grey level emphasis | YH51 | 239.8 | 241 | 14 | ✅ PASS |
| Large zone high grey level emphasis | J17V | 4.122e+07 | 4.14e7 | 3e5 | ✅ PASS |
| Grey level non-uniformity | JNSA | 212.9 | 212 | 6 | ✅ PASS |
| Normalised grey level non-uniformity | Y1RO | 0.04908 | 0.0491 | 0.0008 | ✅ PASS |
| Zone size non-uniformity | 4JP3 | 1648 | 1.63e3 | 10 | ❌ FAIL |
| Normalised zone size non-uniformity | VB3A | 0.3801 | 0.377 | 0.006 | ✅ PASS |
| Zone percentage | P30P | 0.09764 | 0.0972 | 0.0007 | ✅ PASS |
| Grey level variance | BYLV | 32.64 | 32.7 | 1.6 | ✅ PASS |
| Zone size variance | 3NSA | 9.853e+04 | 9.9e4 | 2.8e3 | ✅ PASS |
| Zone size entropy | GU8N | 6.498 | 6.52 | 0.01 | ❌ FAIL |

### GLDZM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Small distance emphasis | 0GBI | 0.5791 | 0.579 | 0.004 | ✅ PASS |
| Large distance emphasis | MB4I | 10.21 | 10.3 | 0.1 | ✅ PASS |
| Low grey level zone emphasis | S1RA | 0.04051 | 0.0409 | 0.0005 | ✅ PASS |
| High grey level zone emphasis | K26C | 187.2 | 188 | 10 | ✅ PASS |
| Small distance low grey level emphasis | RUVG | 0.02965 | 0.0302 | 0.0006 | ✅ PASS |
| Small distance high grey level emphasis | DKNJ | 99.26 | 99.3 | 5.1 | ✅ PASS |
| Large distance low grey level emphasis | A7WM | 0.1844 | 0.183 | 0.004 | ✅ PASS |
| Large distance high grey level emphasis | KLTH | 2587 | 2.62e3 | 110 | ✅ PASS |
| Grey level non-uniformity | VFT7 | 212.9 | 212 | 6 | ✅ PASS |
| Normalised grey level non-uniformity | 7HP3 | 0.04908 | 0.0491 | 0.0008 | ✅ PASS |
| Zone distance non-uniformity | V294 | 1373 | 1.37e3 | 20 | ✅ PASS |
| Normalised zone distance non-uniformity | IATH | 0.3165 | 0.317 | 0.004 | ✅ PASS |
| Zone percentage | VIWW | 0.09764 | 0.0972 | 0.0007 | ✅ PASS |
| Grey level variance | QK93 | 32.64 | 32.7 | 1.6 | ✅ PASS |
| Zone distance variance | 7WT1 | 4.586 | 4.61 | 0.04 | ✅ PASS |
| Zone distance entropy | GBDU | 6.613 | 6.61 | 0.03 | ✅ PASS |

### NGTDM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Coarseness | QCDE | 0.0002084 | 0.000208 | 4e-6 | ✅ PASS |
| Contrast | 65HE | 0.04598 | 0.046 | 0.0005 | ✅ PASS |
| Busyness | NQ30 | 5.143 | 5.14 | 0.14 | ✅ PASS |
| Complexity | HDEZ | 400.5 | 400 | 5 | ✅ PASS |
| Strength | 1X9X | 0.1617 | 0.162 | 0.008 | ✅ PASS |

### NGLDM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Low dependence emphasis | SODN | 0.09168 | 0.0912 | 0.0007 | ✅ PASS |
| High dependence emphasis | IMOQ | 222.9 | 223 | 5 | ✅ PASS |
| Low grey level count emphasis | TL9H | 0.01673 | 0.0168 | 0.0009 | ✅ PASS |
| High grey level count emphasis | OAE7 | 364.1 | 364 | 16 | ✅ PASS |
| Low dependence low grey level emphasis | EQ3F | 0.003592 | 0.00357 | 4e-5 | ✅ PASS |
| Low dependence high grey level emphasis | JA6D | 18.95 | 18.9 | 1.1 | ✅ PASS |
| High dependence low grey level emphasis | NBZI | 0.7971 | 0.798 | 0.072 | ✅ PASS |
| High dependence high grey level emphasis | 9QMG | 9.282e+04 | 9.28e4 | 1.3e3 | ✅ PASS |
| Grey level non-uniformity | FP8K | 1.017e+04 | 1.02e4 | 300 | ✅ PASS |
| Normalised grey level non-uniformity | 5SPA | 0.2289 | 0.229 | 0.003 | ✅ PASS |
| Dependence count non-uniformity | Z87G | 1834 | 1.84e3 | 30 | ✅ PASS |
| Normalised dependence count non-uniformity | OKJI | 0.0413 | 0.0413 | 0.0003 | ✅ PASS |
| Dependence count percentage | 6XV8 | 1 | 1 | — | ✅ PASS |
| Grey level variance | 1PFV | 21.65 | 21.7 | 0.4 | ✅ PASS |
| Dependence count variance | DNX2 | 63.94 | 63.9 | 1.3 | ✅ PASS |
| Dependence count entropy | FCBV | 6.979 | 6.98 | 0.01 | ✅ PASS |
| Dependence count energy | CAS9 | 0.0113 | 0.0113 | 0.0002 | ✅ PASS |

## Config E
### Morphology
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Volume | RNU0 | 3.672e+05 | 3.67e5 | 6e3 | ✅ PASS |
| Volume voxel counting | YEKZ | 3.675e+05 | 3.68e5 | 6e3 | ✅ PASS |
| Surface area | C0JK | 3.446e+04 | 3.43e4 | 400 | ✅ PASS |
| Surface to volume ratio | 2PR5 | 0.09384 | 0.0934 | 0.0007 | ✅ PASS |
| Compactness 1 | SKGS | 0.03239 | 0.0326 | 0.0002 | ✅ PASS |
| Compactness 2 | BQWJ | 0.3727 | 0.378 | 0.004 | ❌ FAIL |
| Spherical disproportion | KRCK | 1.39 | 1.38 | 0.01 | ✅ PASS |
| Sphericity | QCFX | 0.7196 | 0.723 | 0.003 | ✅ PASS |
| Asphericity | 25C7 | 0.3896 | 0.383 | 0.004 | ❌ FAIL |
| Center of mass shift | KLMA | 68.89 | 68.5 | 2.1 | ✅ PASS |
| Maximum 3D diameter | L0JK | 125.1 | 125 | 1 | ✅ PASS |
| Major axis length | TDIC | 93.26 | 93.3 | 0.5 | ✅ PASS |
| Minor axis length | P9VJ | 81.98 | 82 | 0.5 | ✅ PASS |
| Least axis length | 7J51 | 70.88 | 70.9 | 0.4 | ✅ PASS |
| Elongation | Q3CK | 0.879 | 0.879 | 0.001 | ✅ PASS |
| Flatness | N17B | 0.76 | 0.76 | 0.001 | ✅ PASS |
| Volume density (AABB) | PBX1 | 0.4779 | 0.478 | 0.003 | ✅ PASS |
| Area density (AABB) | R59B | 0.6814 | 0.678 | 0.003 | ✅ PASS |
| Volume density (OMBB) | ZH1A | 0.3405 | — | — | ❗ REF. |
| Area density (OMBB) | IQYR | 0.5412 | — | — | ❗ REF. |
| Volume density (AEE) | 6BDE | 1.294 | 1.29 | 0.01 | ✅ PASS |
| Area density (AEE) | RDD2 | 1.625 | 1.62 | 0.01 | ✅ PASS |
| Volume density (MVEE) | SWZ1 | 0.4956 | — | — | ❗ REF. |
| Area density (MVEE) | BRI8 | 0.8844 | — | — | ❗ REF. |
| Volume density (convex hull) | R3ER | 0.8331 | 0.834 | 0.002 | ✅ PASS |
| Area density (convex hull) | 7T7F | 1.135 | 1.13 | 0.01 | ✅ PASS |

### Intensity
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Integrated intensity | 99N0 | -8.244e+06 | -8.31e6 | 1.6e6 | ✅ PASS |
| Moran's I index | N365 | 0.05951 | 0.0596 | 0.0014 | ✅ PASS |
| Geary's C measure | NPT7 | 0.8531 | 0.853 | 0.001 | ✅ PASS |
| Local intensity peak  | VJGA | 180.8 | 181 | 13 | ✅ PASS |
| Global intensity peak | 0F91 | 180.8 | 181 | 5 | ✅ PASS |
| Mean intensity | Q4LE | -22.45 | -22.6 | 4.1 | ✅ PASS |
| Intensity variance | ECT3 | 3.499e+04 | 3.51e4 | 2.2e3 | ✅ PASS |
| Intensity skewness | KE2A | -2.302 | -2.3 | 0.07 | ✅ PASS |
| Intensity kurtosis | IPH6 | 4.449 | 4.44 | 0.33 | ✅ PASS |
| Median intensity | Y12H | 43 | 43 | 0.5 | ✅ PASS |
| Minimum intensity | 1GSF | -743 | -743 | 13 | ✅ PASS |
| 10th intensity percentile | QG58 | -310 | -310 | 21 | ✅ PASS |
| 90th intensity percentile | 8DWT | 93 | 93 | 0.2 | ✅ PASS |
| Maximum intensity | 84IY | 345 | 345 | 9 | ✅ PASS |
| Intensity interquartile range | SALO | 62 | 62 | 3.5 | ✅ PASS |
| Intensity range | 2OJQ | 1088 | 1.09e3 | 30 | ✅ PASS |
| Intensity Mean absolute deviation | 4FUA | 125.1 | 125 | 6 | ✅ PASS |
| Intensity Robust mean absolute deviation | 1128 | 46.44 | 46.5 | 3.7 | ✅ PASS |
| Intensity Median absolute deviation | N72L | 97.71 | 97.9 | 3.9 | ✅ PASS |
| Intensity Coefficient of variation | 7TET | -8.331 | -8.28 | 0.95 | ✅ PASS |
| Intensity Quartile coefficient of dispersion | 9S40 | 0.7949 | 0.795 | 0.337 | ✅ PASS |
| Intensity energy | N8CA | 1.577e+09 | 1.58e9 | 1.4e8 | ✅ PASS |
| Root mean square intensity | 5ZWQ | 188.4 | 189 | 7 | ✅ PASS |

### Intensity Histogram
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Mean discretised intensity | X6K6 | 21.71 | 21.7 | 0.3 | ✅ PASS |
| Discretised intensity variance | CH89 | 30.33 | 30.4 | 0.8 | ✅ PASS |
| Discretised intensity skewness | 88K1 | -2.291 | -2.29 | 0.07 | ✅ PASS |
| Discretised intensity kurtosis | C3I7 | 4.415 | 4.4 | 0.33 | ✅ PASS |
| Median discretised intensity | WIFQ | 24 | 24 | 0.2 | ✅ PASS |
| Minimum discretised intensity | 1PR8 | 1 | 1 | — | ✅ PASS |
| 10th discretised intensity percentile | 1PR | 13 | 13 | 0.7 | ✅ PASS |
| 90th discretised intensity percentile | GPMT | 25 | 25 | 0.2 | ✅ PASS |
| Maximum discretised intensity | 3NCY | 32 | 32 | — | ✅ PASS |
| Intensity histogram mode | AMMC | 24 | 24 | 0.1 | ✅ PASS |
| Discretised intensity interquartile range | WR0O | 1 | 1 | 0.06 | ✅ PASS |
| Discretised intensity range | 5Z3W | 31 | 31 | — | ✅ PASS |
| Intensity histogram mean absolute deviation | D2ZX | 3.681 | 3.69 | 0.1 | ✅ PASS |
| Intensity histogram robust mean absolute deviation | WRZB | 1.455 | 1.46 | 0.09 | ✅ PASS |
| Intensity histogram median absolute deviation | 4RNL | 2.889 | 2.89 | 0.07 | ✅ PASS |
| Intensity histogram coefficient of variation | CWYJ | 0.2537 | 0.254 | 0.006 | ✅ PASS |
| Intensity histogram quartile coefficient of dispersion | SLWD | 0.02128 | 0.0213 | 0.0015 | ✅ PASS |
| Discretised intensity entropy | TLU2 | 3.221 | 3.22 | 0.02 | ✅ PASS |
| Discretised intensity uniformity | BJ5W | 0.1837 | 0.184 | 0.001 | ✅ PASS |
| Maximum histogram gradient | 12CE | 6002 | 6.01e3 | 130 | ✅ PASS |
| Maximum histogram gradient intensity | 8E6O | 23 | 23 | 0.2 | ✅ PASS |
| Minimum histogram gradient | VQB3 | -6102 | -6.11e3 | 180 | ✅ PASS |
| Minimum histogram gradient intensity | RHQZ | 25 | 25 | 0.2 | ✅ PASS |

### Intensity-Volume Histogram
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Volume at intensity fraction 0.10 | BC2M_10 | 0.9748 | 0.975 | 0.002 | ✅ PASS |
| Volume at intensity fraction 0.90 | BC2M_90 | 0.0001575 | 0.000157 | 0.000248 | ✅ PASS |
| Intensity at volume fraction 0.10 | GBPN_10 | 770 | 770 | 5 | ✅ PASS |
| Intensity at volume fraction 0.90 | GBPN_90 | 399 | 399 | 17 | ✅ PASS |
| Volume fraction difference between intensity 0.10 and 0.90 fractions | DDTU | 0.9746 | 0.974 | 0.001 | ✅ PASS |
| Intensity fraction difference between volume 0.10 and 0.90 fractions | CNV2 | 371 | 371 | 13 | ✅ PASS |
| Area under the IVH curve | 9CMM | 662.3 | — | — | ❗ REF. |

### GLCM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Joint maximum | GYBY | 0.153 | 0.153 | 0.003 | ✅ PASS |
| Joint average | 60VM | 22.14 | 22.1 | 0.3 | ✅ PASS |
| Joint variance | UR99 | 24.35 | 24.4 | 0.9 | ✅ PASS |
| Joint entropy | TU9B | 5.613 | 5.61 | 0.03 | ✅ PASS |
| Difference average | TF7R | 1.695 | 1.7 | 0.01 | ✅ PASS |
| Difference variance | D3YU | 8.228 | 8.23 | 0.06 | ✅ PASS |
| Difference entropy | NTRS | 2.397 | 2.4 | 0.01 | ✅ PASS |
| Sum average | ZGXS | 44.27 | 44.3 | 0.4 | ✅ PASS |
| Sum variance | OEEB | 86.3 | 86.7 | 3.3 | ✅ PASS |
| Sum entropy | P6QZ | 3.966 | 3.97 | 0.02 | ✅ PASS |
| Angular second moment | 8ZQL | 0.06354 | 0.0635 | 0.0009 | ✅ PASS |
| Contrast | ACUI | 11.1 | 11.1 | 0.1 | ✅ PASS |
| Dissimilarity | 8S9J | 1.695 | 1.7 | 0.01 | ✅ PASS |
| Inverse difference | IB1Z | 0.6084 | 0.608 | 0.001 | ✅ PASS |
| Normalised inverse difference | NDRX | 0.9552 | 0.955 | 0.001 | ✅ PASS |
| Inverse difference moment | WF0Z | 0.5769 | 0.577 | 0.001 | ✅ PASS |
| Normalised inverse difference moment | 1QCO | 0.9905 | 0.99 | 0.001 | ✅ PASS |
| Inverse variance | E8JP | 0.4101 | 0.41 | 0.004 | ✅ PASS |
| Correlation | NI2N | 0.7721 | 0.773 | 0.006 | ✅ PASS |
| Autocorrelation | QWB0 | 508.8 | 509 | 8 | ✅ PASS |
| Cluster tendency | DG8W | 86.3 | 86.7 | 3.3 | ✅ PASS |
| Cluster shade | 7NFM | -2065 | -2.08e3 | 70 | ✅ PASS |
| Cluster prominence | AE86 | 6.86e+04 | 6.9e4 | 2.1e3 | ✅ PASS |
| Information correlation 1 | R8DG | -0.1752 | -0.175 | 0.003 | ✅ PASS |
| Information correlation 2 | JN9H | 0.8122 | 0.813 | 0.004 | ✅ PASS |

### GLRLM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Short runs emphasis | 22OV | 0.7771 | 0.777 | 0.001 | ✅ PASS |
| Long runs emphasis | W4KF | 3.521 | 3.52 | 0.07 | ✅ PASS |
| Low grey level run emphasis | V3SW | 0.0201 | 0.0204 | 0.0008 | ✅ PASS |
| High grey level run emphasis | G3QZ | 471.5 | 471 | 9 | ✅ PASS |
| Short run low grey level emphasis | HTZT | 0.01843 | 0.0186 | 0.0007 | ✅ PASS |
| Short run high grey level emphasis | GD3A | 347.3 | 347 | 7 | ✅ PASS |
| Long run low grey level emphasis | IVPO | 0.0307 | 0.0311 | 0.0016 | ✅ PASS |
| Long run high grey level emphasis | 3KUM | 1888 | 1.89e3 | 20 | ✅ PASS |
| Grey level non-uniformity | R5YN | 5.194e+04 | 5.19e4 | 200 | ✅ PASS |
| Normalised grey level non-uniformity | OVBL | 0.1354 | 0.135 | 0.003 | ✅ PASS |
| Run length non-uniformity | W92Y | 2.149e+05 | 2.15e5 | 4e3 | ✅ PASS |
| Normalised run length non-uniformity | IC23 | 0.5602 | 0.56 | 0.001 | ✅ PASS |
| Run percentage | 9ZK5 | 0.6639 | 0.664 | 0.003 | ✅ PASS |
| Grey level variance | 8CE5 | 39.62 | 39.7 | 0.9 | ✅ PASS |
| Run length variance | SXLW | 1.252 | 1.25 | 0.05 | ✅ PASS |
| Run entropy | HJ9O | 4.869 | 4.87 | 0.03 | ✅ PASS |

### GLSZM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Small zone emphasis | P001 | 0.6763 | 0.676 | 0.003 | ✅ PASS |
| Large zone emphasis | 48P8 | 5.849e+04 | 5.86e4 | 800 | ✅ PASS |
| Low grey level zone emphasis | XMSY | 0.03423 | 0.034 | 0.0004 | ✅ PASS |
| High grey level zone emphasis | 5GN9 | 285.6 | 286 | 6 | ✅ PASS |
| Small zone low grey level emphasis | 5RAI | 0.02236 | 0.0224 | 0.0004 | ✅ PASS |
| Small zone high grey level emphasis | HW1V | 185.8 | 186 | 4 | ✅ PASS |
| Large zone low grey level emphasis | YH51 | 104.8 | 105 | 4 | ✅ PASS |
| Large zone high grey level emphasis | J17V | 3.352e+07 | 3.36e7 | 3e5 | ✅ PASS |
| Grey level non-uniformity | JNSA | 231.1 | 231 | 6 | ✅ PASS |
| Normalised grey level non-uniformity | Y1RO | 0.04137 | 0.0414 | 0.0003 | ✅ PASS |
| Zone size non-uniformity | 4JP3 | 2367 | 2.37e3 | 40 | ✅ PASS |
| Normalised zone size non-uniformity | VB3A | 0.4236 | 0.424 | 0.004 | ✅ PASS |
| Zone percentage | P30P | 0.1257 | 0.126 | 0.001 | ✅ PASS |
| Grey level variance | BYLV | 50.84 | 50.8 | 0.9 | ✅ PASS |
| Zone size variance | 3NSA | 5.842e+04 | 5.85e4 | 800 | ✅ PASS |
| Zone size entropy | GU8N | 6.563 | 6.57 | 0.01 | ✅ PASS |

### GLDZM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Small distance emphasis | 0GBI | 0.5276 | 0.527 | 0.004 | ✅ PASS |
| Large distance emphasis | MB4I | 12.54 | 12.6 | 0.1 | ✅ PASS |
| Low grey level zone emphasis | S1RA | 0.03423 | 0.034 | 0.0004 | ✅ PASS |
| High grey level zone emphasis | K26C | 285.6 | 286 | 6 | ✅ PASS |
| Small distance low grey level emphasis | RUVG | 0.02311 | 0.0228 | 0.0003 | ✅ PASS |
| Small distance high grey level emphasis | DKNJ | 136.2 | 136 | 4 | ✅ PASS |
| Large distance low grey level emphasis | A7WM | 0.179 | 0.179 | 0.004 | ✅ PASS |
| Large distance high grey level emphasis | KLTH | 4844 | 4.85e3 | 60 | ✅ PASS |
| Grey level non-uniformity | VFT7 | 231.1 | 231 | 6 | ✅ PASS |
| Normalised grey level non-uniformity | 7HP3 | 0.04137 | 0.0414 | 0.0003 | ✅ PASS |
| Zone distance non-uniformity | V294 | 1506 | 1.5e3 | 30 | ✅ PASS |
| Normalised zone distance non-uniformity | IATH | 0.2695 | 0.269 | 0.003 | ✅ PASS |
| Zone percentage | VIWW | 0.1257 | 0.126 | 0.001 | ✅ PASS |
| Grey level variance | QK93 | 50.84 | 50.8 | 0.9 | ✅ PASS |
| Zone distance variance | 7WT1 | 5.55 | 5.56 | 0.05 | ✅ PASS |
| Zone distance entropy | GBDU | 7.061 | 7.06 | 0.01 | ✅ PASS |

### NGTDM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Coarseness | QCDE | 0.0001886 | 0.000188 | 4e-6 | ✅ PASS |
| Contrast | 65HE | 0.07502 | 0.0752 | 0.0019 | ✅ PASS |
| Busyness | NQ30 | 4.645 | 4.65 | 0.1 | ✅ PASS |
| Complexity | HDEZ | 574.2 | 574 | 1 | ✅ PASS |
| Strength | 1X9X | 0.1676 | 0.167 | 0.006 | ✅ PASS |

### NGLDM
| Feature | Code | Calc | Ref | Tol | Status |
|---|---|---|---|---|---|
| Low dependence emphasis | SODN | 0.1183 | 0.118 | 0.001 | ✅ PASS |
| High dependence emphasis | IMOQ | 134.3 | 134 | 3 | ✅ PASS |
| Low grey level count emphasis | TL9H | 0.01519 | 0.0154 | 0.0007 | ✅ PASS |
| High grey level count emphasis | OAE7 | 501.6 | 502 | 8 | ✅ PASS |
| Low dependence low grey level emphasis | EQ3F | 0.00386 | 0.00388 | 4e-5 | ✅ PASS |
| Low dependence high grey level emphasis | JA6D | 36.65 | 36.7 | 0.5 | ✅ PASS |
| High dependence low grey level emphasis | NBZI | 0.4499 | 0.457 | 0.031 | ✅ PASS |
| High dependence high grey level emphasis | 9QMG | 7.599e+04 | 7.6e4 | 600 | ✅ PASS |
| Grey level non-uniformity | FP8K | 8162 | 8.17e3 | 130 | ✅ PASS |
| Normalised grey level non-uniformity | 5SPA | 0.1837 | 0.184 | 0.001 | ✅ PASS |
| Dependence count non-uniformity | Z87G | 2245 | 2.25e3 | 30 | ✅ PASS |
| Normalised dependence count non-uniformity | OKJI | 0.05051 | 0.0505 | 0.0003 | ✅ PASS |
| Dependence count percentage | 6XV8 | 1 | 1 | — | ✅ PASS |
| Grey level variance | 1PFV | 30.33 | 30.4 | 0.8 | ✅ PASS |
| Dependence count variance | DNX2 | 39.44 | 39.4 | 1 | ✅ PASS |
| Dependence count entropy | FCBV | 7.064 | 7.06 | 0.02 | ✅ PASS |
| Dependence count energy | CAS9 | 0.01062 | 0.0106 | 0.0001 | ✅ PASS |
