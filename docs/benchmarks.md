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
|     25 | FBS 10.0         | 0.0353 s                | 0.81 MB                |
|     25 | FBS 25.0         | 0.0293 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0287 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0316 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0338 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0342 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.4988 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.5404 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.6110 s                | 6.33 MB                |
|     50 | FBN 16           | 1.5486 s                | 6.33 MB                |
|     50 | FBN 32           | 1.5232 s                | 6.33 MB                |
|     50 | FBN 64           | 1.4987 s                | 6.33 MB                |
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
|     25 | FBS 10.0         | 0.0046 s                | 1.11 MB                |
|     25 | FBS 25.0         | 0.0044 s                | 1.11 MB                |
|     25 | FBS 50.0         | 0.0041 s                | 1.11 MB                |
|     25 | FBN 16           | 0.0045 s                | 1.11 MB                |
|     25 | FBN 32           | 0.0045 s                | 1.11 MB                |
|     25 | FBN 64           | 0.0047 s                | 1.11 MB                |
|     50 | FBS 10.0         | 0.0110 s                | 5.37 MB                |
|     50 | FBS 25.0         | 0.0112 s                | 5.37 MB                |
|     50 | FBS 50.0         | 0.0112 s                | 5.37 MB                |
|     50 | FBN 16           | 0.0109 s                | 5.37 MB                |
|     50 | FBN 32           | 0.0108 s                | 5.37 MB                |
|     50 | FBN 64           | 0.0107 s                | 5.37 MB                |
|     75 | FBS 10.0         | 0.0180 s                | 8.77 MB                |
|     75 | FBS 25.0         | 0.0169 s                | 8.77 MB                |
|     75 | FBS 50.0         | 0.0180 s                | 8.77 MB                |
|     75 | FBN 16           | 0.0172 s                | 8.77 MB                |
|     75 | FBN 32           | 0.0183 s                | 8.77 MB                |
|     75 | FBN 64           | 0.0184 s                | 8.77 MB                |
|    100 | FBS 10.0         | 0.0335 s                | 20.40 MB               |
|    100 | FBS 25.0         | 0.0342 s                | 20.40 MB               |
|    100 | FBS 50.0         | 0.0350 s                | 20.40 MB               |
|    100 | FBN 16           | 0.0333 s                | 20.40 MB               |
|    100 | FBN 32           | 0.0362 s                | 20.40 MB               |
|    100 | FBN 64           | 0.0339 s                | 20.40 MB               |


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
|     50 | FBS 50.0         | 0.0002 s                | 0.07 MB                |
|     50 | FBN 16           | 0.0002 s                | 0.07 MB                |
|     50 | FBN 32           | 0.0003 s                | 0.07 MB                |
|     50 | FBN 64           | 0.0002 s                | 0.09 MB                |
|     75 | FBS 10.0         | 0.0003 s                | 0.15 MB                |
|     75 | FBS 25.0         | 0.0003 s                | 0.10 MB                |
|     75 | FBS 50.0         | 0.0003 s                | 0.08 MB                |
|     75 | FBN 16           | 0.0003 s                | 0.08 MB                |
|     75 | FBN 32           | 0.0003 s                | 0.09 MB                |
|     75 | FBN 64           | 0.0003 s                | 0.12 MB                |
|    100 | FBS 10.0         | 0.0004 s                | 0.14 MB                |
|    100 | FBS 25.0         | 0.0004 s                | 0.09 MB                |
|    100 | FBS 50.0         | 0.0004 s                | 0.08 MB                |
|    100 | FBN 16           | 0.0004 s                | 0.07 MB                |
|    100 | FBN 32           | 0.0004 s                | 0.09 MB                |
|    100 | FBN 64           | 0.0004 s                | 0.11 MB                |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Intensity  |     25 | FBN 16           | 0.0008 s           | 0.0140 s           | 18.49x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 32           | 0.0008 s           | 0.0142 s           | 17.32x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 64           | 0.0008 s           | 0.0142 s           | 18.05x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 10.0         | 0.0008 s           | 0.0145 s           | 17.06x    | 0.24 MB           | 0.74 MB           |
| Intensity  |     25 | FBS 25.0         | 0.0008 s           | 0.0135 s           | 16.56x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 50.0         | 0.0008 s           | 0.0133 s           | 15.84x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     50 | FBN 16           | 0.0031 s           | 0.0658 s           | 21.46x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 32           | 0.0033 s           | 0.0655 s           | 19.81x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 64           | 0.0033 s           | 0.0646 s           | 19.43x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 10.0         | 0.0032 s           | 0.0683 s           | 21.15x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 25.0         | 0.0033 s           | 0.0669 s           | 20.23x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 50.0         | 0.0031 s           | 0.0661 s           | 21.32x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     75 | FBN 16           | 0.0120 s           | 0.2643 s           | 22.11x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 32           | 0.0128 s           | 0.2701 s           | 21.13x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 64           | 0.0129 s           | 0.2626 s           | 20.40x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 10.0         | 0.0129 s           | 0.2710 s           | 21.06x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 25.0         | 0.0118 s           | 0.2715 s           | 23.01x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 50.0         | 0.0122 s           | 0.2683 s           | 21.91x    | 5.81 MB           | 17.95 MB          |
| Intensity  |    100 | FBN 16           | 0.0232 s           | 0.5541 s           | 23.88x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 32           | 0.0251 s           | 0.5434 s           | 21.64x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 64           | 0.0248 s           | 0.5643 s           | 22.71x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 10.0         | 0.0253 s           | 0.5585 s           | 22.07x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 25.0         | 0.0236 s           | 0.5546 s           | 23.54x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 50.0         | 0.0245 s           | 0.5402 s           | 22.06x    | 12.16 MB          | 39.01 MB          |
| Morphology |     25 | FBN 16           | 0.0041 s           | 0.0578 s           | 14.28x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBN 32           | 0.0043 s           | 0.0571 s           | 13.43x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBN 64           | 0.0043 s           | 0.0571 s           | 13.17x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 10.0         | 0.0044 s           | 0.0569 s           | 13.01x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 25.0         | 0.0039 s           | 0.0548 s           | 13.89x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 50.0         | 0.0039 s           | 0.0552 s           | 14.00x    | 1.10 MB           | 1.18 MB           |
| Morphology |     50 | FBN 16           | 0.0107 s           | 0.9712 s           | 90.74x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBN 32           | 0.0104 s           | 0.9747 s           | 93.74x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBN 64           | 0.0101 s           | 0.9709 s           | 95.95x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 10.0         | 0.0104 s           | 0.9803 s           | 94.35x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 25.0         | 0.0104 s           | 0.9776 s           | 94.39x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 50.0         | 0.0123 s           | 0.9775 s           | 79.25x    | 5.37 MB           | 8.68 MB           |
| Morphology |     75 | FBN 16           | 0.0173 s           | 1.7789 s           | 102.83x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBN 32           | 0.0177 s           | 1.7736 s           | 100.11x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBN 64           | 0.0173 s           | 1.7802 s           | 102.80x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 10.0         | 0.0179 s           | 1.8085 s           | 100.93x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 25.0         | 0.0167 s           | 1.8023 s           | 108.01x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 50.0         | 0.0177 s           | 1.7925 s           | 101.39x   | 8.77 MB           | 36.60 MB          |
| Morphology |    100 | FBN 16           | 0.0332 s           | 9.0760 s           | 273.11x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBN 32           | 0.0350 s           | 9.0828 s           | 259.39x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBN 64           | 0.0339 s           | 9.1238 s           | 268.79x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 10.0         | 0.0328 s           | 9.1532 s           | 279.01x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 25.0         | 0.0335 s           | 9.0536 s           | 269.93x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 50.0         | 0.0352 s           | 9.0734 s           | 257.81x   | 20.40 MB          | 77.49 MB          |
| Texture    |     25 | FBN 16           | 0.0054 s           | 0.0502 s           | 9.35x     | 2.13 MB           | 0.69 MB           |
| Texture    |     25 | FBN 32           | 0.0054 s           | 0.0537 s           | 9.90x     | 2.07 MB           | 0.75 MB           |
| Texture    |     25 | FBN 64           | 0.0059 s           | 0.0683 s           | 11.62x    | 4.53 MB           | 1.94 MB           |
| Texture    |     25 | FBS 10.0         | 0.0072 s           | 0.0987 s           | 13.67x    | 10.10 MB          | 3.77 MB           |
| Texture    |     25 | FBS 25.0         | 0.0054 s           | 0.0539 s           | 10.07x    | 2.16 MB           | 0.91 MB           |
| Texture    |     25 | FBS 50.0         | 0.0052 s           | 0.0500 s           | 9.63x     | 2.13 MB           | 0.71 MB           |
| Texture    |     50 | FBN 16           | 0.0251 s           | 0.3082 s           | 12.27x    | 19.85 MB          | 6.29 MB           |
| Texture    |     50 | FBN 32           | 0.0247 s           | 0.3099 s           | 12.52x    | 19.95 MB          | 6.29 MB           |
| Texture    |     50 | FBN 64           | 0.0181 s           | 0.3252 s           | 17.97x    | 9.83 MB           | 4.82 MB           |
| Texture    |     50 | FBS 10.0         | 0.0228 s           | 0.3642 s           | 15.95x    | 11.96 MB          | 5.41 MB           |
| Texture    |     50 | FBS 25.0         | 0.0255 s           | 0.3175 s           | 12.47x    | 19.30 MB          | 6.09 MB           |
| Texture    |     50 | FBS 50.0         | 0.0265 s           | 0.3085 s           | 11.65x    | 21.08 MB          | 6.60 MB           |
| Texture    |     75 | FBN 16           | 0.0807 s           | 1.2401 s           | 15.36x    | 84.43 MB          | 25.62 MB          |
| Texture    |     75 | FBN 32           | 0.0922 s           | 1.2519 s           | 13.57x    | 86.59 MB          | 26.23 MB          |
| Texture    |     75 | FBN 64           | 0.0676 s           | 1.2368 s           | 18.31x    | 63.72 MB          | 20.14 MB          |
| Texture    |     75 | FBS 10.0         | 0.0516 s           | 1.3053 s           | 25.29x    | 13.36 MB          | 17.08 MB          |
| Texture    |     75 | FBS 25.0         | 0.0855 s           | 1.2606 s           | 14.75x    | 89.44 MB          | 25.80 MB          |
| Texture    |     75 | FBS 50.0         | 0.0899 s           | 1.2427 s           | 13.83x    | 88.88 MB          | 25.65 MB          |
| Texture    |    100 | FBN 16           | 0.1905 s           | 2.5891 s           | 13.59x    | 213.37 MB         | 64.12 MB          |
| Texture    |    100 | FBN 32           | 0.2279 s           | 2.6340 s           | 11.56x    | 224.11 MB         | 66.17 MB          |
| Texture    |    100 | FBN 64           | 0.2198 s           | 2.6527 s           | 12.07x    | 204.71 MB         | 60.72 MB          |
| Texture    |    100 | FBS 10.0         | 0.1050 s           | 2.6916 s           | 25.63x    | 22.17 MB          | 37.15 MB          |
| Texture    |    100 | FBS 25.0         | 0.2131 s           | 2.5853 s           | 12.13x    | 230.74 MB         | 68.10 MB          |
| Texture    |    100 | FBS 50.0         | 0.2275 s           | 2.5998 s           | 11.43x    | 229.77 MB         | 68.48 MB          |

