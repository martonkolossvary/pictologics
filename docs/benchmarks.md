# Benchmarks


## Performance Benchmarks

### Benchmark Configuration

Comparisons between **Pictologics** and **PyRadiomics** (single-thread parity). 

**Test Data Generation:**

- **Texture**: 3D correlated noise generated using Gaussian smoothing.
- **Mask**: Blob-like structures generated via thresholded smooth noise with random holes.
- **Voxel Distribution**: Mean=486.04, Std=90.24, Min=0.00, Max=1000.00.

### HARDWARE USED FOR CALCULATIONS

- **Hardware**: Apple M4 Pro, 14 cores, 48 GB
- **OS**: macOS 26.2 (arm64)
- **Python**: 3.12.10
- **Core deps**: pictologics 0.2.0, numpy 2.3.5, scipy 1.16.3, numba 0.62.1, pandas 2.3.3, matplotlib 3.10.7
- **PyRadiomics stack (parity runs)**: pyradiomics 3.1.1.dev111+g8ed579383, SimpleITK 2.5.3
- **BLAS/LAPACK**: Apple Accelerate (from `numpy.show_config()`)

Note: the benchmark script explicitly calls `warmup_jit()` before timing to avoid including Numba compilation overhead in the measured runtimes.

### Intensity

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Intensity time](assets/benchmarks/intensity_execution_time_log.png)](assets/benchmarks/intensity_execution_time_log.png) | [![Intensity speedup](assets/benchmarks/intensity_speedup_factor.png)](assets/benchmarks/intensity_speedup_factor.png) |

**Pictologics-only intensity families (IVH + spatial/local intensity):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0327 s                | 0.81 MB                |
|     25 | FBS 25.0         | 0.0322 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0329 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0296 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0279 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0294 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.3945 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.3741 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.3545 s                | 6.33 MB                |
|     50 | FBN 16           | 1.3710 s                | 6.33 MB                |
|     50 | FBN 32           | 1.3921 s                | 6.33 MB                |
|     50 | FBN 64           | 1.3827 s                | 6.33 MB                |
|     75 | FBS 10.0         | Not calculated          | Not calculated         |
|     75 | FBS 25.0         | Not calculated          | Not calculated         |
|     75 | FBS 50.0         | Not calculated          | Not calculated         |
|     75 | FBN 16           | Not calculated          | Not calculated         |
|     75 | FBN 32           | Not calculated          | Not calculated         |
|     75 | FBN 64           | Not calculated          | Not calculated         |
|    100 | FBS 10.0         | Not calculated          | Not calculated         |
|    100 | FBS 25.0         | Not calculated          | Not calculated         |
|    100 | FBS 50.0         | Not calculated          | Not calculated         |
|    100 | FBN 16           | Not calculated          | Not calculated         |
|    100 | FBN 32           | Not calculated          | Not calculated         |
|    100 | FBN 64           | Not calculated          | Not calculated         |


### Morphology

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Morphology time](assets/benchmarks/morphology_execution_time_log.png)](assets/benchmarks/morphology_execution_time_log.png) | [![Morphology speedup](assets/benchmarks/morphology_speedup_factor.png)](assets/benchmarks/morphology_speedup_factor.png) |

**Pictologics-only morphology families (intensity-weighted morphology):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0040 s                | 1.11 MB                |
|     25 | FBS 25.0         | 0.0041 s                | 1.10 MB                |
|     25 | FBS 50.0         | 0.0047 s                | 1.11 MB                |
|     25 | FBN 16           | 0.0041 s                | 1.11 MB                |
|     25 | FBN 32           | 0.0041 s                | 1.11 MB                |
|     25 | FBN 64           | 0.0042 s                | 1.11 MB                |
|     50 | FBS 10.0         | 0.0108 s                | 5.37 MB                |
|     50 | FBS 25.0         | 0.0104 s                | 5.37 MB                |
|     50 | FBS 50.0         | 0.0105 s                | 5.37 MB                |
|     50 | FBN 16           | 0.0102 s                | 5.37 MB                |
|     50 | FBN 32           | 0.0106 s                | 5.37 MB                |
|     50 | FBN 64           | 0.0106 s                | 5.37 MB                |
|     75 | FBS 10.0         | 0.0169 s                | 8.77 MB                |
|     75 | FBS 25.0         | 0.0175 s                | 8.77 MB                |
|     75 | FBS 50.0         | 0.0173 s                | 8.77 MB                |
|     75 | FBN 16           | 0.0167 s                | 8.77 MB                |
|     75 | FBN 32           | 0.0170 s                | 8.77 MB                |
|     75 | FBN 64           | 0.0161 s                | 8.77 MB                |
|    100 | FBS 10.0         | 0.0320 s                | 20.40 MB               |
|    100 | FBS 25.0         | 0.0309 s                | 20.40 MB               |
|    100 | FBS 50.0         | 0.0306 s                | 20.40 MB               |
|    100 | FBN 16           | 0.0314 s                | 20.40 MB               |
|    100 | FBN 32           | 0.0315 s                | 20.40 MB               |
|    100 | FBN 64           | 0.0338 s                | 20.40 MB               |


### Texture

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Texture time](assets/benchmarks/texture_execution_time_log.png)](assets/benchmarks/texture_execution_time_log.png) | [![Texture speedup](assets/benchmarks/texture_speedup_factor.png)](assets/benchmarks/texture_speedup_factor.png) |

**Pictologics-only texture families (GLDZM):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0002 s                | 0.08 MB                |
|     25 | FBS 25.0         | 0.0002 s                | 0.07 MB                |
|     25 | FBS 50.0         | 0.0002 s                | 0.07 MB                |
|     25 | FBN 16           | 0.0002 s                | 0.07 MB                |
|     25 | FBN 32           | 0.0002 s                | 0.07 MB                |
|     25 | FBN 64           | 0.0002 s                | 0.07 MB                |
|     50 | FBS 10.0         | 0.0003 s                | 0.10 MB                |
|     50 | FBS 25.0         | 0.0002 s                | 0.08 MB                |
|     50 | FBS 50.0         | 0.0003 s                | 0.07 MB                |
|     50 | FBN 16           | 0.0002 s                | 0.07 MB                |
|     50 | FBN 32           | 0.0002 s                | 0.07 MB                |
|     50 | FBN 64           | 0.0002 s                | 0.09 MB                |
|     75 | FBS 10.0         | 0.0003 s                | 0.15 MB                |
|     75 | FBS 25.0         | 0.0003 s                | 0.10 MB                |
|     75 | FBS 50.0         | 0.0003 s                | 0.08 MB                |
|     75 | FBN 16           | 0.0003 s                | 0.08 MB                |
|     75 | FBN 32           | 0.0003 s                | 0.09 MB                |
|     75 | FBN 64           | 0.0003 s                | 0.12 MB                |
|    100 | FBS 10.0         | 0.0004 s                | 0.14 MB                |
|    100 | FBS 25.0         | 0.0003 s                | 0.09 MB                |
|    100 | FBS 50.0         | 0.0003 s                | 0.08 MB                |
|    100 | FBN 16           | 0.0003 s                | 0.07 MB                |
|    100 | FBN 32           | 0.0003 s                | 0.09 MB                |
|    100 | FBN 64           | 0.0004 s                | 0.11 MB                |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Intensity  |     25 | FBN 16           | 0.0008 s           | 0.0131 s           | 16.15x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 32           | 0.0008 s           | 0.0127 s           | 15.64x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 64           | 0.0008 s           | 0.0136 s           | 16.50x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 10.0         | 0.0010 s           | 0.0140 s           | 14.48x    | 0.24 MB           | 0.74 MB           |
| Intensity  |     25 | FBS 25.0         | 0.0008 s           | 0.0134 s           | 16.79x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 50.0         | 0.0009 s           | 0.0138 s           | 15.60x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     50 | FBN 16           | 0.0031 s           | 0.0643 s           | 20.76x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 32           | 0.0033 s           | 0.0648 s           | 19.48x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 64           | 0.0031 s           | 0.0676 s           | 21.89x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 10.0         | 0.0033 s           | 0.0665 s           | 20.26x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 25.0         | 0.0029 s           | 0.0670 s           | 23.02x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 50.0         | 0.0029 s           | 0.0644 s           | 22.58x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     75 | FBN 16           | 0.0110 s           | 0.2529 s           | 22.91x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 32           | 0.0119 s           | 0.2516 s           | 21.06x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 64           | 0.0116 s           | 0.2629 s           | 22.64x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 10.0         | 0.0117 s           | 0.2587 s           | 22.02x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 25.0         | 0.0114 s           | 0.2612 s           | 22.89x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 50.0         | 0.0118 s           | 0.2493 s           | 21.14x    | 5.81 MB           | 17.95 MB          |
| Intensity  |    100 | FBN 16           | 0.0207 s           | 0.5560 s           | 26.82x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 32           | 0.0213 s           | 0.5346 s           | 25.13x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 64           | 0.0232 s           | 0.5464 s           | 23.56x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 10.0         | 0.0231 s           | 0.5215 s           | 22.53x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 25.0         | 0.0210 s           | 0.5059 s           | 24.09x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 50.0         | 0.0201 s           | 0.4995 s           | 24.82x    | 12.16 MB          | 39.01 MB          |
| Morphology |     25 | FBN 16           | 0.0040 s           | 0.0513 s           | 12.79x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBN 32           | 0.0037 s           | 0.0508 s           | 13.66x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBN 64           | 0.0039 s           | 0.0551 s           | 14.09x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 10.0         | 0.0038 s           | 0.0572 s           | 15.14x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 25.0         | 0.0041 s           | 0.0553 s           | 13.56x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 50.0         | 0.0041 s           | 0.0548 s           | 13.40x    | 1.11 MB           | 1.18 MB           |
| Morphology |     50 | FBN 16           | 0.0101 s           | 0.9658 s           | 95.30x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBN 32           | 0.0105 s           | 0.9653 s           | 92.19x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBN 64           | 0.0100 s           | 0.9652 s           | 96.30x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 10.0         | 0.0103 s           | 0.9490 s           | 92.07x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 25.0         | 0.0098 s           | 0.9595 s           | 97.41x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 50.0         | 0.0102 s           | 0.9602 s           | 94.18x    | 5.37 MB           | 8.68 MB           |
| Morphology |     75 | FBN 16           | 0.0163 s           | 1.7437 s           | 107.07x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBN 32           | 0.0171 s           | 1.7374 s           | 101.53x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBN 64           | 0.0160 s           | 1.7295 s           | 108.38x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 10.0         | 0.0163 s           | 1.7609 s           | 108.26x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 25.0         | 0.0168 s           | 1.7502 s           | 103.99x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 50.0         | 0.0169 s           | 1.7527 s           | 103.76x   | 8.77 MB           | 36.60 MB          |
| Morphology |    100 | FBN 16           | 0.0313 s           | 8.6947 s           | 277.87x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBN 32           | 0.0315 s           | 8.7095 s           | 276.75x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBN 64           | 0.0329 s           | 8.7769 s           | 266.66x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 10.0         | 0.0311 s           | 8.4436 s           | 271.83x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 25.0         | 0.0304 s           | 8.3123 s           | 273.33x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 50.0         | 0.0303 s           | 8.4807 s           | 280.00x   | 20.40 MB          | 77.49 MB          |
| Texture    |     25 | FBN 16           | 0.0049 s           | 0.0465 s           | 9.58x     | 2.13 MB           | 0.69 MB           |
| Texture    |     25 | FBN 32           | 0.0051 s           | 0.0509 s           | 10.06x    | 2.07 MB           | 0.75 MB           |
| Texture    |     25 | FBN 64           | 0.0054 s           | 0.0659 s           | 12.24x    | 4.53 MB           | 1.94 MB           |
| Texture    |     25 | FBS 10.0         | 0.0077 s           | 0.0937 s           | 12.19x    | 10.10 MB          | 3.77 MB           |
| Texture    |     25 | FBS 25.0         | 0.0053 s           | 0.0552 s           | 10.45x    | 2.16 MB           | 0.91 MB           |
| Texture    |     25 | FBS 50.0         | 0.0053 s           | 0.0487 s           | 9.26x     | 2.13 MB           | 0.71 MB           |
| Texture    |     50 | FBN 16           | 0.0219 s           | 0.2995 s           | 13.67x    | 19.85 MB          | 6.29 MB           |
| Texture    |     50 | FBN 32           | 0.0240 s           | 0.2994 s           | 12.46x    | 19.95 MB          | 6.29 MB           |
| Texture    |     50 | FBN 64           | 0.0185 s           | 0.3181 s           | 17.17x    | 9.83 MB           | 4.82 MB           |
| Texture    |     50 | FBS 10.0         | 0.0182 s           | 0.3440 s           | 18.89x    | 11.96 MB          | 5.41 MB           |
| Texture    |     50 | FBS 25.0         | 0.0224 s           | 0.3089 s           | 13.81x    | 19.30 MB          | 6.09 MB           |
| Texture    |     50 | FBS 50.0         | 0.0242 s           | 0.3021 s           | 12.46x    | 21.08 MB          | 6.60 MB           |
| Texture    |     75 | FBN 16           | 0.0746 s           | 1.1982 s           | 16.06x    | 84.43 MB          | 25.62 MB          |
| Texture    |     75 | FBN 32           | 0.0819 s           | 1.2069 s           | 14.73x    | 86.59 MB          | 26.23 MB          |
| Texture    |     75 | FBN 64           | 0.0697 s           | 1.2055 s           | 17.29x    | 63.72 MB          | 20.14 MB          |
| Texture    |     75 | FBS 10.0         | 0.0478 s           | 1.2687 s           | 26.54x    | 13.36 MB          | 17.08 MB          |
| Texture    |     75 | FBS 25.0         | 0.0825 s           | 1.2166 s           | 14.75x    | 89.44 MB          | 25.80 MB          |
| Texture    |     75 | FBS 50.0         | 0.0828 s           | 1.2043 s           | 14.55x    | 88.88 MB          | 25.65 MB          |
| Texture    |    100 | FBN 16           | 0.1738 s           | 2.5015 s           | 14.40x    | 213.37 MB         | 64.12 MB          |
| Texture    |    100 | FBN 32           | 0.1928 s           | 2.5579 s           | 13.27x    | 224.11 MB         | 66.17 MB          |
| Texture    |    100 | FBN 64           | 0.2062 s           | 2.5424 s           | 12.33x    | 204.71 MB         | 60.72 MB          |
| Texture    |    100 | FBS 10.0         | 0.0999 s           | 2.4952 s           | 24.98x    | 22.17 MB          | 37.15 MB          |
| Texture    |    100 | FBS 25.0         | 0.1833 s           | 2.4170 s           | 13.18x    | 230.74 MB         | 68.10 MB          |
| Texture    |    100 | FBS 50.0         | 0.1727 s           | 2.5056 s           | 14.51x    | 229.77 MB         | 68.48 MB          |

