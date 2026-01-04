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
|     25 | FBS 10.0         | 0.0306 s                | 0.81 MB                |
|     25 | FBS 25.0         | 0.0300 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0295 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0275 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0356 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0339 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.4161 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.3752 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.4271 s                | 6.33 MB                |
|     50 | FBN 16           | 1.3548 s                | 6.33 MB                |
|     50 | FBN 32           | 1.3427 s                | 6.33 MB                |
|     50 | FBN 64           | 1.3978 s                | 6.33 MB                |
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
|     25 | FBS 10.0         | 0.0044 s                | 1.11 MB                |
|     25 | FBS 25.0         | 0.0046 s                | 1.10 MB                |
|     25 | FBS 50.0         | 0.0044 s                | 1.11 MB                |
|     25 | FBN 16           | 0.0043 s                | 1.10 MB                |
|     25 | FBN 32           | 0.0049 s                | 1.10 MB                |
|     25 | FBN 64           | 0.0043 s                | 1.10 MB                |
|     50 | FBS 10.0         | 0.0104 s                | 5.37 MB                |
|     50 | FBS 25.0         | 0.0107 s                | 5.37 MB                |
|     50 | FBS 50.0         | 0.0107 s                | 5.37 MB                |
|     50 | FBN 16           | 0.0110 s                | 5.37 MB                |
|     50 | FBN 32           | 0.0105 s                | 5.37 MB                |
|     50 | FBN 64           | 0.0110 s                | 5.37 MB                |
|     75 | FBS 10.0         | 0.0170 s                | 8.77 MB                |
|     75 | FBS 25.0         | 0.0180 s                | 8.77 MB                |
|     75 | FBS 50.0         | 0.0168 s                | 8.77 MB                |
|     75 | FBN 16           | 0.0170 s                | 8.77 MB                |
|     75 | FBN 32           | 0.0172 s                | 8.77 MB                |
|     75 | FBN 64           | 0.0161 s                | 8.77 MB                |
|    100 | FBS 10.0         | 0.0312 s                | 20.40 MB               |
|    100 | FBS 25.0         | 0.0351 s                | 20.40 MB               |
|    100 | FBS 50.0         | 0.0343 s                | 20.40 MB               |
|    100 | FBN 16           | 0.0351 s                | 20.40 MB               |
|    100 | FBN 32           | 0.0349 s                | 20.40 MB               |
|    100 | FBN 64           | 0.0342 s                | 20.40 MB               |


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
|     50 | FBS 10.0         | 0.0002 s                | 0.10 MB                |
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
|    100 | FBS 25.0         | 0.0004 s                | 0.09 MB                |
|    100 | FBS 50.0         | 0.0003 s                | 0.08 MB                |
|    100 | FBN 16           | 0.0003 s                | 0.07 MB                |
|    100 | FBN 32           | 0.0004 s                | 0.09 MB                |
|    100 | FBN 64           | 0.0004 s                | 0.11 MB                |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Intensity  |     25 | FBN 16           | 0.0009 s           | 0.0147 s           | 17.16x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 32           | 0.0010 s           | 0.0144 s           | 15.01x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 64           | 0.0010 s           | 0.0142 s           | 14.47x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 10.0         | 0.0008 s           | 0.0169 s           | 20.63x    | 0.24 MB           | 0.74 MB           |
| Intensity  |     25 | FBS 25.0         | 0.0008 s           | 0.0137 s           | 17.34x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 50.0         | 0.0008 s           | 0.0138 s           | 16.95x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     50 | FBN 16           | 0.0030 s           | 0.0667 s           | 21.96x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 32           | 0.0032 s           | 0.0648 s           | 20.46x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 64           | 0.0033 s           | 0.0663 s           | 20.22x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 10.0         | 0.0029 s           | 0.0672 s           | 22.91x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 25.0         | 0.0031 s           | 0.0660 s           | 21.03x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 50.0         | 0.0029 s           | 0.0668 s           | 22.83x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     75 | FBN 16           | 0.0111 s           | 0.2501 s           | 22.47x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 32           | 0.0116 s           | 0.2554 s           | 21.95x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 64           | 0.0114 s           | 0.2545 s           | 22.30x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 10.0         | 0.0121 s           | 0.2655 s           | 22.00x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 25.0         | 0.0118 s           | 0.2581 s           | 21.84x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 50.0         | 0.0113 s           | 0.2597 s           | 23.05x    | 5.81 MB           | 17.95 MB          |
| Intensity  |    100 | FBN 16           | 0.0286 s           | 0.5406 s           | 18.91x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 32           | 0.0243 s           | 0.5494 s           | 22.57x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 64           | 0.0256 s           | 0.5530 s           | 21.64x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 10.0         | 0.0225 s           | 0.5270 s           | 23.44x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 25.0         | 0.0242 s           | 0.5531 s           | 22.86x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 50.0         | 0.0243 s           | 0.5535 s           | 22.77x    | 12.16 MB          | 39.01 MB          |
| Morphology |     25 | FBN 16           | 0.0039 s           | 0.0542 s           | 13.89x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBN 32           | 0.0045 s           | 0.0555 s           | 12.31x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBN 64           | 0.0042 s           | 0.0554 s           | 13.30x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 10.0         | 0.0040 s           | 0.0547 s           | 13.61x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 25.0         | 0.0044 s           | 0.0543 s           | 12.40x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 50.0         | 0.0040 s           | 0.0541 s           | 13.54x    | 1.11 MB           | 1.18 MB           |
| Morphology |     50 | FBN 16           | 0.0112 s           | 0.9511 s           | 85.25x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBN 32           | 0.0102 s           | 0.9521 s           | 93.71x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBN 64           | 0.0104 s           | 0.9533 s           | 91.49x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 10.0         | 0.0105 s           | 0.9559 s           | 90.90x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 25.0         | 0.0102 s           | 0.9498 s           | 93.31x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 50.0         | 0.0102 s           | 0.9475 s           | 92.82x    | 5.37 MB           | 8.68 MB           |
| Morphology |     75 | FBN 16           | 0.0159 s           | 1.7204 s           | 108.02x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBN 32           | 0.0161 s           | 1.7285 s           | 107.06x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBN 64           | 0.0159 s           | 1.7150 s           | 107.60x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 10.0         | 0.0162 s           | 1.7469 s           | 107.98x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 25.0         | 0.0171 s           | 1.7401 s           | 101.56x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 50.0         | 0.0165 s           | 1.7340 s           | 105.06x   | 8.77 MB           | 36.60 MB          |
| Morphology |    100 | FBN 16           | 0.0350 s           | 10.6975 s          | 305.45x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBN 32           | 0.0349 s           | 9.0469 s           | 259.51x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBN 64           | 0.0340 s           | 9.0318 s           | 265.49x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 10.0         | 0.0309 s           | 8.6890 s           | 281.09x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 25.0         | 0.0340 s           | 9.4016 s           | 276.65x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 50.0         | 0.0351 s           | 9.0966 s           | 258.90x   | 20.40 MB          | 77.49 MB          |
| Texture    |     25 | FBN 16           | 0.0049 s           | 0.0514 s           | 10.43x    | 2.13 MB           | 0.69 MB           |
| Texture    |     25 | FBN 32           | 0.0058 s           | 0.0542 s           | 9.34x     | 2.07 MB           | 0.75 MB           |
| Texture    |     25 | FBN 64           | 0.0058 s           | 0.0672 s           | 11.55x    | 4.53 MB           | 1.94 MB           |
| Texture    |     25 | FBS 10.0         | 0.0074 s           | 0.0999 s           | 13.47x    | 10.10 MB          | 3.77 MB           |
| Texture    |     25 | FBS 25.0         | 0.0055 s           | 0.0552 s           | 9.94x     | 2.16 MB           | 0.91 MB           |
| Texture    |     25 | FBS 50.0         | 0.0053 s           | 0.0515 s           | 9.78x     | 2.13 MB           | 0.71 MB           |
| Texture    |     50 | FBN 16           | 0.0241 s           | 0.3061 s           | 12.69x    | 19.85 MB          | 6.29 MB           |
| Texture    |     50 | FBN 32           | 0.0232 s           | 0.3062 s           | 13.19x    | 19.95 MB          | 6.29 MB           |
| Texture    |     50 | FBN 64           | 0.0189 s           | 0.3191 s           | 16.90x    | 9.83 MB           | 4.82 MB           |
| Texture    |     50 | FBS 10.0         | 0.0183 s           | 0.3598 s           | 19.66x    | 11.96 MB          | 5.41 MB           |
| Texture    |     50 | FBS 25.0         | 0.0234 s           | 0.3079 s           | 13.17x    | 19.30 MB          | 6.09 MB           |
| Texture    |     50 | FBS 50.0         | 0.0247 s           | 0.3012 s           | 12.22x    | 21.08 MB          | 6.60 MB           |
| Texture    |     75 | FBN 16           | 0.0787 s           | 1.2109 s           | 15.38x    | 84.43 MB          | 25.62 MB          |
| Texture    |     75 | FBN 32           | 0.0795 s           | 1.2032 s           | 15.13x    | 86.59 MB          | 26.23 MB          |
| Texture    |     75 | FBN 64           | 0.0638 s           | 1.2195 s           | 19.12x    | 63.72 MB          | 20.14 MB          |
| Texture    |     75 | FBS 10.0         | 0.0495 s           | 1.2798 s           | 25.83x    | 13.36 MB          | 17.08 MB          |
| Texture    |     75 | FBS 25.0         | 0.0850 s           | 1.2267 s           | 14.43x    | 89.44 MB          | 25.80 MB          |
| Texture    |     75 | FBS 50.0         | 0.0781 s           | 1.1993 s           | 15.35x    | 88.88 MB          | 25.65 MB          |
| Texture    |    100 | FBN 16           | 0.2046 s           | 3.0934 s           | 15.12x    | 213.37 MB         | 64.12 MB          |
| Texture    |    100 | FBN 32           | 0.2234 s           | 2.6080 s           | 11.67x    | 224.11 MB         | 66.17 MB          |
| Texture    |    100 | FBN 64           | 0.2048 s           | 2.6290 s           | 12.84x    | 204.71 MB         | 60.72 MB          |
| Texture    |    100 | FBS 10.0         | 0.0980 s           | 2.7150 s           | 27.71x    | 22.17 MB          | 37.15 MB          |
| Texture    |    100 | FBS 25.0         | 0.2199 s           | 2.6987 s           | 12.27x    | 230.74 MB         | 68.10 MB          |
| Texture    |    100 | FBS 50.0         | 0.2248 s           | 3.1756 s           | 14.13x    | 229.77 MB         | 68.48 MB          |

