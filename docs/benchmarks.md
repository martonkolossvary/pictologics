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
- **OS**: macOS 26.3 (arm64)
- **Python**: 3.12.10
- **Core deps**: pictologics 0.3.4, numpy 2.2.6, scipy 1.17.0, numba 0.62.1, pandas 2.3.3, matplotlib 3.10.7
- **PyRadiomics stack (parity runs)**: pyradiomics 3.1.1.dev111+g8ed579383, SimpleITK 2.5.3
- **BLAS/LAPACK**: Apple Accelerate (from `numpy.show_config()`)

Note: the benchmark script explicitly calls `warmup_jit()` before timing to avoid including Numba compilation overhead in the measured runtimes. Timing and memory measurement are separated — `tracemalloc` is NOT active during timing to avoid biasing the comparison (its per-allocation hooks penalise pure-Python code more than JIT/C code). All calculations are repeated 5 times and the **mean** runtime is reported; peak memory is measured once separately.

### Intensity

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Intensity time](assets/benchmarks/intensity_execution_time_log.png)](assets/benchmarks/intensity_execution_time_log.png) | [![Intensity speedup](assets/benchmarks/intensity_speedup_factor.png)](assets/benchmarks/intensity_speedup_factor.png) |

**Pictologics-only intensity families (IVH + spatial/local intensity):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0307 s                | 0.81 MB                |
|     25 | FBS 25.0         | 0.0290 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0290 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0310 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0296 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0319 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.3952 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.4372 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.4561 s                | 6.33 MB                |
|     50 | FBN 16           | 1.3729 s                | 6.33 MB                |
|     50 | FBN 32           | 1.3993 s                | 6.33 MB                |
|     50 | FBN 64           | 1.3824 s                | 6.33 MB                |
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
|     25 | FBS 10.0         | 0.0029 s                | 1.17 MB                |
|     25 | FBS 25.0         | 0.0030 s                | 1.17 MB                |
|     25 | FBS 50.0         | 0.0031 s                | 1.17 MB                |
|     25 | FBN 16           | 0.0031 s                | 1.17 MB                |
|     25 | FBN 32           | 0.0031 s                | 1.17 MB                |
|     25 | FBN 64           | 0.0029 s                | 1.17 MB                |
|     50 | FBS 10.0         | 0.0095 s                | 5.43 MB                |
|     50 | FBS 25.0         | 0.0097 s                | 5.43 MB                |
|     50 | FBS 50.0         | 0.0095 s                | 5.43 MB                |
|     50 | FBN 16           | 0.0094 s                | 5.43 MB                |
|     50 | FBN 32           | 0.0094 s                | 5.43 MB                |
|     50 | FBN 64           | 0.0094 s                | 5.43 MB                |
|     75 | FBS 10.0         | 0.0171 s                | 8.84 MB                |
|     75 | FBS 25.0         | 0.0160 s                | 8.84 MB                |
|     75 | FBS 50.0         | 0.0161 s                | 8.84 MB                |
|     75 | FBN 16           | 0.0180 s                | 8.84 MB                |
|     75 | FBN 32           | 0.0163 s                | 8.84 MB                |
|     75 | FBN 64           | 0.0166 s                | 8.84 MB                |
|    100 | FBS 10.0         | 0.0343 s                | 20.46 MB               |
|    100 | FBS 25.0         | 0.0357 s                | 20.46 MB               |
|    100 | FBS 50.0         | 0.0359 s                | 20.46 MB               |
|    100 | FBN 16           | 0.0352 s                | 20.46 MB               |
|    100 | FBN 32           | 0.0330 s                | 20.46 MB               |
|    100 | FBN 64           | 0.0353 s                | 20.46 MB               |


### Texture

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Texture time](assets/benchmarks/texture_execution_time_log.png)](assets/benchmarks/texture_execution_time_log.png) | [![Texture speedup](assets/benchmarks/texture_speedup_factor.png)](assets/benchmarks/texture_speedup_factor.png) |

**Pictologics-only texture families (GLDZM):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0001 s                | 0.15 MB                |
|     25 | FBS 25.0         | 0.0001 s                | 0.13 MB                |
|     25 | FBS 50.0         | 0.0001 s                | 0.13 MB                |
|     25 | FBN 16           | 0.0000 s                | 0.13 MB                |
|     25 | FBN 32           | 0.0001 s                | 0.13 MB                |
|     25 | FBN 64           | 0.0001 s                | 0.14 MB                |
|     50 | FBS 10.0         | 0.0001 s                | 0.16 MB                |
|     50 | FBS 25.0         | 0.0001 s                | 0.14 MB                |
|     50 | FBS 50.0         | 0.0001 s                | 0.13 MB                |
|     50 | FBN 16           | 0.0001 s                | 0.13 MB                |
|     50 | FBN 32           | 0.0001 s                | 0.14 MB                |
|     50 | FBN 64           | 0.0001 s                | 0.15 MB                |
|     75 | FBS 10.0         | 0.0001 s                | 0.21 MB                |
|     75 | FBS 25.0         | 0.0001 s                | 0.16 MB                |
|     75 | FBS 50.0         | 0.0001 s                | 0.14 MB                |
|     75 | FBN 16           | 0.0001 s                | 0.14 MB                |
|     75 | FBN 32           | 0.0002 s                | 0.15 MB                |
|     75 | FBN 64           | 0.0002 s                | 0.18 MB                |
|    100 | FBS 10.0         | 0.0002 s                | 0.20 MB                |
|    100 | FBS 25.0         | 0.0002 s                | 0.16 MB                |
|    100 | FBS 50.0         | 0.0002 s                | 0.14 MB                |
|    100 | FBN 16           | 0.0002 s                | 0.14 MB                |
|    100 | FBN 32           | 0.0002 s                | 0.15 MB                |
|    100 | FBN 64           | 0.0002 s                | 0.17 MB                |


### Filters

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Filters time](assets/benchmarks/filters_execution_time_log.png)](assets/benchmarks/filters_execution_time_log.png) | [![Filters speedup](assets/benchmarks/filters_speedup_factor.png)](assets/benchmarks/filters_speedup_factor.png) |

**Pictologics-only filters (Gabor, Laws, Simoncelli, Riesz, Mean):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0043 s                | 1.70 MB                |
|     25 | FBS 25.0         | 0.0040 s                | 1.69 MB                |
|     25 | FBS 50.0         | 0.0044 s                | 1.70 MB                |
|     25 | FBN 16           | 0.0043 s                | 1.70 MB                |
|     25 | FBN 32           | 0.0038 s                | 1.70 MB                |
|     25 | FBN 64           | 0.0042 s                | 1.69 MB                |
|     50 | FBS 10.0         | 0.0167 s                | 13.49 MB               |
|     50 | FBS 25.0         | 0.0167 s                | 13.49 MB               |
|     50 | FBS 50.0         | 0.0162 s                | 13.49 MB               |
|     50 | FBN 16           | 0.0161 s                | 13.49 MB               |
|     50 | FBN 32           | 0.0162 s                | 13.49 MB               |
|     50 | FBN 64           | 0.0167 s                | 13.49 MB               |
|     75 | FBS 10.0         | 0.0412 s                | 45.49 MB               |
|     75 | FBS 25.0         | 0.0414 s                | 45.49 MB               |
|     75 | FBS 50.0         | 0.0415 s                | 45.49 MB               |
|     75 | FBN 16           | 0.0480 s                | 45.49 MB               |
|     75 | FBN 32           | 0.0435 s                | 45.49 MB               |
|     75 | FBN 64           | 0.0415 s                | 45.49 MB               |
|    100 | FBS 10.0         | 0.0869 s                | 107.80 MB              |
|    100 | FBS 25.0         | 0.0899 s                | 107.80 MB              |
|    100 | FBS 50.0         | 0.0950 s                | 107.80 MB              |
|    100 | FBN 16           | 0.1076 s                | 107.80 MB              |
|    100 | FBN 32           | 0.0913 s                | 107.80 MB              |
|    100 | FBN 64           | 0.1120 s                | 107.80 MB              |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Filters    |     25 | FBN 16           | 0.0008 s           | 0.0037 s           | 4.52x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBN 32           | 0.0008 s           | 0.0037 s           | 4.62x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBN 64           | 0.0009 s           | 0.0040 s           | 4.36x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 10.0         | 0.0009 s           | 0.0039 s           | 4.49x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 25.0         | 0.0008 s           | 0.0038 s           | 4.66x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 50.0         | 0.0008 s           | 0.0038 s           | 4.73x     | 0.48 MB           | 0.94 MB           |
| Filters    |     50 | FBN 16           | 0.0056 s           | 0.0094 s           | 1.69x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBN 32           | 0.0055 s           | 0.0093 s           | 1.70x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBN 64           | 0.0054 s           | 0.0093 s           | 1.72x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 10.0         | 0.0056 s           | 0.0091 s           | 1.63x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 25.0         | 0.0057 s           | 0.0094 s           | 1.66x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 50.0         | 0.0055 s           | 0.0091 s           | 1.66x     | 3.82 MB           | 6.68 MB           |
| Filters    |     75 | FBN 16           | 0.0194 s           | 0.0273 s           | 1.41x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBN 32           | 0.0187 s           | 0.0264 s           | 1.41x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBN 64           | 0.0189 s           | 0.0258 s           | 1.37x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 10.0         | 0.0188 s           | 0.0264 s           | 1.40x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 25.0         | 0.0187 s           | 0.0261 s           | 1.40x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 50.0         | 0.0188 s           | 0.0302 s           | 1.61x     | 12.88 MB          | 23.38 MB          |
| Filters    |    100 | FBN 16           | 0.0457 s           | 0.0542 s           | 1.19x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBN 32           | 0.0449 s           | 0.0627 s           | 1.40x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBN 64           | 0.0470 s           | 0.0585 s           | 1.24x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 10.0         | 0.0449 s           | 0.0585 s           | 1.30x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 25.0         | 0.0448 s           | 0.0597 s           | 1.33x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 50.0         | 0.0459 s           | 0.0632 s           | 1.38x     | 30.52 MB          | 53.41 MB          |
| Intensity  |     25 | FBN 16           | 0.0004 s           | 0.0049 s           | 11.80x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 32           | 0.0004 s           | 0.0052 s           | 11.96x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 64           | 0.0004 s           | 0.0051 s           | 12.00x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 10.0         | 0.0012 s           | 0.0053 s           | 4.52x     | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 25.0         | 0.0004 s           | 0.0053 s           | 12.74x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 50.0         | 0.0004 s           | 0.0051 s           | 12.38x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     50 | FBN 16           | 0.0028 s           | 0.0164 s           | 5.95x     | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 32           | 0.0026 s           | 0.0168 s           | 6.35x     | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 64           | 0.0027 s           | 0.0179 s           | 6.53x     | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 10.0         | 0.0026 s           | 0.0181 s           | 7.04x     | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 25.0         | 0.0026 s           | 0.0175 s           | 6.70x     | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 50.0         | 0.0027 s           | 0.0168 s           | 6.21x     | 1.40 MB           | 4.61 MB           |
| Intensity  |     75 | FBN 16           | 0.0119 s           | 0.0623 s           | 5.23x     | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 32           | 0.0120 s           | 0.0623 s           | 5.21x     | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 64           | 0.0117 s           | 0.0671 s           | 5.75x     | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 10.0         | 0.0122 s           | 0.0673 s           | 5.49x     | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 25.0         | 0.0110 s           | 0.0625 s           | 5.70x     | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 50.0         | 0.0111 s           | 0.0589 s           | 5.32x     | 5.81 MB           | 17.95 MB          |
| Intensity  |    100 | FBN 16           | 0.0248 s           | 0.1318 s           | 5.32x     | 12.16 MB          | 39.00 MB          |
| Intensity  |    100 | FBN 32           | 0.0231 s           | 0.1245 s           | 5.38x     | 12.16 MB          | 39.00 MB          |
| Intensity  |    100 | FBN 64           | 0.0250 s           | 0.1457 s           | 5.83x     | 12.16 MB          | 39.00 MB          |
| Intensity  |    100 | FBS 10.0         | 0.0240 s           | 0.1352 s           | 5.64x     | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 25.0         | 0.0240 s           | 0.1238 s           | 5.16x     | 12.16 MB          | 39.00 MB          |
| Intensity  |    100 | FBS 50.0         | 0.0223 s           | 0.1242 s           | 5.56x     | 12.16 MB          | 39.00 MB          |
| Morphology |     25 | FBN 16           | 0.0034 s           | 0.0539 s           | 16.04x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 32           | 0.0032 s           | 0.0543 s           | 17.05x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 64           | 0.0032 s           | 0.0548 s           | 17.11x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 10.0         | 0.0043 s           | 0.0549 s           | 12.79x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 25.0         | 0.0030 s           | 0.0557 s           | 18.75x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 50.0         | 0.0030 s           | 0.0544 s           | 17.94x    | 1.17 MB           | 1.18 MB           |
| Morphology |     50 | FBN 16           | 0.0094 s           | 0.9593 s           | 101.67x   | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 32           | 0.0095 s           | 0.9583 s           | 100.62x   | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 64           | 0.0094 s           | 0.9615 s           | 102.70x   | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 10.0         | 0.0092 s           | 0.9564 s           | 103.40x   | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 25.0         | 0.0093 s           | 0.9649 s           | 103.68x   | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 50.0         | 0.0096 s           | 0.9544 s           | 99.48x    | 5.43 MB           | 8.68 MB           |
| Morphology |     75 | FBN 16           | 0.0178 s           | 1.7601 s           | 99.00x    | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 32           | 0.0168 s           | 1.7566 s           | 104.46x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 64           | 0.0170 s           | 1.7464 s           | 102.89x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 10.0         | 0.0164 s           | 1.7626 s           | 107.69x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 25.0         | 0.0164 s           | 1.7570 s           | 106.97x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 50.0         | 0.0163 s           | 1.7757 s           | 108.73x   | 8.84 MB           | 36.60 MB          |
| Morphology |    100 | FBN 16           | 0.0355 s           | 8.8534 s           | 249.70x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 32           | 0.0328 s           | 8.9564 s           | 272.91x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 64           | 0.0351 s           | 9.1153 s           | 259.46x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 10.0         | 0.0333 s           | 8.8673 s           | 266.37x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 25.0         | 0.0350 s           | 8.9843 s           | 256.97x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 50.0         | 0.0343 s           | 8.8695 s           | 258.32x   | 20.46 MB          | 77.49 MB          |
| Texture    |     25 | FBN 16           | 0.0033 s           | 0.0137 s           | 4.17x     | 2.13 MB           | 0.69 MB           |
| Texture    |     25 | FBN 32           | 0.0036 s           | 0.0166 s           | 4.60x     | 2.07 MB           | 0.75 MB           |
| Texture    |     25 | FBN 64           | 0.0041 s           | 0.0352 s           | 8.68x     | 4.53 MB           | 1.95 MB           |
| Texture    |     25 | FBS 10.0         | 0.0055 s           | 0.0752 s           | 13.71x    | 10.10 MB          | 3.77 MB           |
| Texture    |     25 | FBS 25.0         | 0.0039 s           | 0.0195 s           | 4.95x     | 2.16 MB           | 0.91 MB           |
| Texture    |     25 | FBS 50.0         | 0.0036 s           | 0.0144 s           | 3.96x     | 2.13 MB           | 0.71 MB           |
| Texture    |     50 | FBN 16           | 0.0210 s           | 0.0674 s           | 3.21x     | 19.85 MB          | 6.28 MB           |
| Texture    |     50 | FBN 32           | 0.0215 s           | 0.0716 s           | 3.34x     | 19.95 MB          | 6.29 MB           |
| Texture    |     50 | FBN 64           | 0.0170 s           | 0.0918 s           | 5.40x     | 9.83 MB           | 4.82 MB           |
| Texture    |     50 | FBS 10.0         | 0.0151 s           | 0.1425 s           | 9.44x     | 11.96 MB          | 5.41 MB           |
| Texture    |     50 | FBS 25.0         | 0.0219 s           | 0.0759 s           | 3.47x     | 19.30 MB          | 6.10 MB           |
| Texture    |     50 | FBS 50.0         | 0.0223 s           | 0.0683 s           | 3.07x     | 21.08 MB          | 6.60 MB           |
| Texture    |     75 | FBN 16           | 0.0752 s           | 0.2440 s           | 3.24x     | 84.43 MB          | 25.62 MB          |
| Texture    |     75 | FBN 32           | 0.0701 s           | 0.2494 s           | 3.56x     | 86.59 MB          | 26.22 MB          |
| Texture    |     75 | FBN 64           | 0.0782 s           | 0.2736 s           | 3.50x     | 63.72 MB          | 20.14 MB          |
| Texture    |     75 | FBS 10.0         | 0.0477 s           | 0.3160 s           | 6.63x     | 13.36 MB          | 17.08 MB          |
| Texture    |     75 | FBS 25.0         | 0.0711 s           | 0.2527 s           | 3.55x     | 89.44 MB          | 25.79 MB          |
| Texture    |     75 | FBS 50.0         | 0.0704 s           | 0.2526 s           | 3.59x     | 88.88 MB          | 25.65 MB          |
| Texture    |    100 | FBN 16           | 0.1726 s           | 0.5023 s           | 2.91x     | 213.37 MB         | 64.12 MB          |
| Texture    |    100 | FBN 32           | 0.1752 s           | 0.5413 s           | 3.09x     | 224.11 MB         | 66.17 MB          |
| Texture    |    100 | FBN 64           | 0.1876 s           | 0.5781 s           | 3.08x     | 204.71 MB         | 60.71 MB          |
| Texture    |    100 | FBS 10.0         | 0.1039 s           | 0.6081 s           | 5.85x     | 22.17 MB          | 37.15 MB          |
| Texture    |    100 | FBS 25.0         | 0.2181 s           | 0.5288 s           | 2.42x     | 230.74 MB         | 68.10 MB          |
| Texture    |    100 | FBS 50.0         | 0.1997 s           | 0.5414 s           | 2.71x     | 229.77 MB         | 68.48 MB          |

