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
|     25 | FBS 10.0         | 0.0339 s                | 0.81 MB                |
|     25 | FBS 25.0         | 0.0333 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0329 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0313 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0327 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0326 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.3753 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.4053 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.3617 s                | 6.33 MB                |
|     50 | FBN 16           | 1.3843 s                | 6.33 MB                |
|     50 | FBN 32           | 1.3937 s                | 6.33 MB                |
|     50 | FBN 64           | 1.3865 s                | 6.33 MB                |
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
|     25 | FBS 10.0         | 0.0030 s                | 1.17 MB                |
|     25 | FBS 25.0         | 0.0032 s                | 1.17 MB                |
|     25 | FBS 50.0         | 0.0032 s                | 1.17 MB                |
|     25 | FBN 16           | 0.0031 s                | 1.17 MB                |
|     25 | FBN 32           | 0.0029 s                | 1.17 MB                |
|     25 | FBN 64           | 0.0030 s                | 1.17 MB                |
|     50 | FBS 10.0         | 0.0092 s                | 5.43 MB                |
|     50 | FBS 25.0         | 0.0095 s                | 5.43 MB                |
|     50 | FBS 50.0         | 0.0095 s                | 5.43 MB                |
|     50 | FBN 16           | 0.0093 s                | 5.43 MB                |
|     50 | FBN 32           | 0.0094 s                | 5.43 MB                |
|     50 | FBN 64           | 0.0093 s                | 5.43 MB                |
|     75 | FBS 10.0         | 0.0158 s                | 8.84 MB                |
|     75 | FBS 25.0         | 0.0161 s                | 8.84 MB                |
|     75 | FBS 50.0         | 0.0156 s                | 8.84 MB                |
|     75 | FBN 16           | 0.0156 s                | 8.84 MB                |
|     75 | FBN 32           | 0.0157 s                | 8.84 MB                |
|     75 | FBN 64           | 0.0160 s                | 8.84 MB                |
|    100 | FBS 10.0         | 0.0319 s                | 20.46 MB               |
|    100 | FBS 25.0         | 0.0344 s                | 20.46 MB               |
|    100 | FBS 50.0         | 0.0334 s                | 20.46 MB               |
|    100 | FBN 16           | 0.0340 s                | 20.46 MB               |
|    100 | FBN 32           | 0.0335 s                | 20.46 MB               |
|    100 | FBN 64           | 0.0326 s                | 20.46 MB               |


### Texture

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Texture time](assets/benchmarks/texture_execution_time_log.png)](assets/benchmarks/texture_execution_time_log.png) | [![Texture speedup](assets/benchmarks/texture_speedup_factor.png)](assets/benchmarks/texture_speedup_factor.png) |

**Pictologics-only texture families (GLDZM):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0001 s                | 0.15 MB                |
|     25 | FBS 25.0         | 0.0001 s                | 0.13 MB                |
|     25 | FBS 50.0         | 0.0000 s                | 0.13 MB                |
|     25 | FBN 16           | 0.0000 s                | 0.13 MB                |
|     25 | FBN 32           | 0.0001 s                | 0.13 MB                |
|     25 | FBN 64           | 0.0001 s                | 0.14 MB                |
|     50 | FBS 10.0         | 0.0001 s                | 0.16 MB                |
|     50 | FBS 25.0         | 0.0001 s                | 0.14 MB                |
|     50 | FBS 50.0         | 0.0001 s                | 0.13 MB                |
|     50 | FBN 16           | 0.0001 s                | 0.13 MB                |
|     50 | FBN 32           | 0.0001 s                | 0.14 MB                |
|     50 | FBN 64           | 0.0001 s                | 0.15 MB                |
|     75 | FBS 10.0         | 0.0002 s                | 0.21 MB                |
|     75 | FBS 25.0         | 0.0002 s                | 0.16 MB                |
|     75 | FBS 50.0         | 0.0001 s                | 0.14 MB                |
|     75 | FBN 16           | 0.0001 s                | 0.14 MB                |
|     75 | FBN 32           | 0.0001 s                | 0.15 MB                |
|     75 | FBN 64           | 0.0001 s                | 0.18 MB                |
|    100 | FBS 10.0         | 0.0003 s                | 0.20 MB                |
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
|     25 | FBS 10.0         | 0.0039 s                | 1.70 MB                |
|     25 | FBS 25.0         | 0.0039 s                | 1.70 MB                |
|     25 | FBS 50.0         | 0.0038 s                | 1.70 MB                |
|     25 | FBN 16           | 0.0039 s                | 1.70 MB                |
|     25 | FBN 32           | 0.0053 s                | 1.70 MB                |
|     25 | FBN 64           | 0.0039 s                | 1.69 MB                |
|     50 | FBS 10.0         | 0.0157 s                | 13.49 MB               |
|     50 | FBS 25.0         | 0.0158 s                | 13.50 MB               |
|     50 | FBS 50.0         | 0.0164 s                | 13.49 MB               |
|     50 | FBN 16           | 0.0156 s                | 13.49 MB               |
|     50 | FBN 32           | 0.0155 s                | 13.49 MB               |
|     50 | FBN 64           | 0.0154 s                | 13.51 MB               |
|     75 | FBS 10.0         | 0.0394 s                | 45.50 MB               |
|     75 | FBS 25.0         | 0.0390 s                | 45.50 MB               |
|     75 | FBS 50.0         | 0.0394 s                | 45.49 MB               |
|     75 | FBN 16           | 0.0401 s                | 45.49 MB               |
|     75 | FBN 32           | 0.0410 s                | 45.49 MB               |
|     75 | FBN 64           | 0.0398 s                | 45.49 MB               |
|    100 | FBS 10.0         | 0.0821 s                | 107.80 MB              |
|    100 | FBS 25.0         | 0.0852 s                | 107.80 MB              |
|    100 | FBS 50.0         | 0.0855 s                | 107.79 MB              |
|    100 | FBN 16           | 0.0898 s                | 107.80 MB              |
|    100 | FBN 32           | 0.0874 s                | 107.80 MB              |
|    100 | FBN 64           | 0.0853 s                | 107.80 MB              |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Filters    |     25 | FBN 16           | 0.0009 s           | 0.0038 s           | 4.24x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBN 32           | 0.0009 s           | 0.0037 s           | 4.25x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBN 64           | 0.0009 s           | 0.0036 s           | 4.19x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 10.0         | 0.0008 s           | 0.0037 s           | 4.48x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 25.0         | 0.0008 s           | 0.0038 s           | 5.04x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 50.0         | 0.0008 s           | 0.0036 s           | 4.74x     | 0.48 MB           | 0.94 MB           |
| Filters    |     50 | FBN 16           | 0.0054 s           | 0.0088 s           | 1.63x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBN 32           | 0.0054 s           | 0.0091 s           | 1.68x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBN 64           | 0.0053 s           | 0.0091 s           | 1.74x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 10.0         | 0.0053 s           | 0.0091 s           | 1.70x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 25.0         | 0.0055 s           | 0.0087 s           | 1.60x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 50.0         | 0.0057 s           | 0.0088 s           | 1.55x     | 3.82 MB           | 6.68 MB           |
| Filters    |     75 | FBN 16           | 0.0191 s           | 0.0264 s           | 1.38x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBN 32           | 0.0188 s           | 0.0252 s           | 1.34x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBN 64           | 0.0182 s           | 0.0272 s           | 1.50x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 10.0         | 0.0192 s           | 0.0256 s           | 1.33x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 25.0         | 0.0182 s           | 0.0250 s           | 1.37x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 50.0         | 0.0185 s           | 0.0253 s           | 1.36x     | 12.88 MB          | 23.38 MB          |
| Filters    |    100 | FBN 16           | 0.0438 s           | 0.0545 s           | 1.24x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBN 32           | 0.0444 s           | 0.0531 s           | 1.19x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBN 64           | 0.0441 s           | 0.0561 s           | 1.27x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 10.0         | 0.0442 s           | 0.0548 s           | 1.24x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 25.0         | 0.0445 s           | 0.0567 s           | 1.28x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 50.0         | 0.0446 s           | 0.0527 s           | 1.18x     | 30.52 MB          | 53.41 MB          |
| Intensity  |     25 | FBN 16           | 0.0005 s           | 0.0051 s           | 11.17x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 32           | 0.0004 s           | 0.0049 s           | 11.44x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 64           | 0.0004 s           | 0.0050 s           | 11.11x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 10.0         | 0.0013 s           | 0.0054 s           | 4.34x     | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 25.0         | 0.0004 s           | 0.0050 s           | 12.98x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 50.0         | 0.0004 s           | 0.0050 s           | 12.18x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     50 | FBN 16           | 0.0025 s           | 0.0165 s           | 6.64x     | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 32           | 0.0025 s           | 0.0166 s           | 6.64x     | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 64           | 0.0026 s           | 0.0172 s           | 6.68x     | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 10.0         | 0.0026 s           | 0.0178 s           | 6.76x     | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 25.0         | 0.0025 s           | 0.0167 s           | 6.54x     | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 50.0         | 0.0025 s           | 0.0165 s           | 6.71x     | 1.40 MB           | 4.61 MB           |
| Intensity  |     75 | FBN 16           | 0.0103 s           | 0.0569 s           | 5.54x     | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 32           | 0.0107 s           | 0.0611 s           | 5.73x     | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 64           | 0.0111 s           | 0.0658 s           | 5.91x     | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 10.0         | 0.0114 s           | 0.0646 s           | 5.68x     | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 25.0         | 0.0110 s           | 0.0608 s           | 5.50x     | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 50.0         | 0.0103 s           | 0.0572 s           | 5.54x     | 5.81 MB           | 17.95 MB          |
| Intensity  |    100 | FBN 16           | 0.0218 s           | 0.1185 s           | 5.44x     | 12.16 MB          | 39.00 MB          |
| Intensity  |    100 | FBN 32           | 0.0226 s           | 0.1250 s           | 5.53x     | 12.16 MB          | 39.00 MB          |
| Intensity  |    100 | FBN 64           | 0.0234 s           | 0.1307 s           | 5.60x     | 12.16 MB          | 39.00 MB          |
| Intensity  |    100 | FBS 10.0         | 0.0233 s           | 0.1290 s           | 5.54x     | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 25.0         | 0.0235 s           | 0.1236 s           | 5.27x     | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 50.0         | 0.0223 s           | 0.1220 s           | 5.47x     | 12.16 MB          | 39.01 MB          |
| Morphology |     25 | FBN 16           | 0.0029 s           | 0.0537 s           | 18.58x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 32           | 0.0031 s           | 0.0537 s           | 17.42x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 64           | 0.0031 s           | 0.0533 s           | 17.43x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 10.0         | 0.0044 s           | 0.0526 s           | 12.05x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 25.0         | 0.0029 s           | 0.0533 s           | 18.69x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 50.0         | 0.0029 s           | 0.0532 s           | 18.09x    | 1.17 MB           | 1.18 MB           |
| Morphology |     50 | FBN 16           | 0.0094 s           | 0.9560 s           | 101.95x   | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 32           | 0.0094 s           | 0.9498 s           | 101.07x   | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 64           | 0.0092 s           | 0.9556 s           | 103.70x   | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 10.0         | 0.0094 s           | 0.9490 s           | 100.98x   | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 25.0         | 0.0092 s           | 0.9534 s           | 103.24x   | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 50.0         | 0.0095 s           | 0.9493 s           | 99.79x    | 5.43 MB           | 8.68 MB           |
| Morphology |     75 | FBN 16           | 0.0159 s           | 1.6975 s           | 107.02x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 32           | 0.0158 s           | 1.7155 s           | 108.84x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 64           | 0.0161 s           | 1.7131 s           | 106.08x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 10.0         | 0.0161 s           | 1.7199 s           | 106.95x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 25.0         | 0.0161 s           | 1.7027 s           | 106.04x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 50.0         | 0.0158 s           | 1.7040 s           | 108.18x   | 8.84 MB           | 36.60 MB          |
| Morphology |    100 | FBN 16           | 0.0327 s           | 8.9499 s           | 273.58x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 32           | 0.0336 s           | 8.8913 s           | 264.66x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 64           | 0.0331 s           | 9.0493 s           | 273.55x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 10.0         | 0.0324 s           | 8.7498 s           | 270.28x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 25.0         | 0.0334 s           | 9.0167 s           | 269.65x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 50.0         | 0.0329 s           | 8.9591 s           | 272.59x   | 20.46 MB          | 77.49 MB          |
| Texture    |     25 | FBN 16           | 0.0039 s           | 0.0131 s           | 3.38x     | 2.13 MB           | 0.69 MB           |
| Texture    |     25 | FBN 32           | 0.0033 s           | 0.0153 s           | 4.57x     | 2.07 MB           | 0.75 MB           |
| Texture    |     25 | FBN 64           | 0.0040 s           | 0.0267 s           | 6.66x     | 4.53 MB           | 1.95 MB           |
| Texture    |     25 | FBS 10.0         | 0.0055 s           | 0.0509 s           | 9.27x     | 10.10 MB          | 3.77 MB           |
| Texture    |     25 | FBS 25.0         | 0.0038 s           | 0.0168 s           | 4.47x     | 2.16 MB           | 0.91 MB           |
| Texture    |     25 | FBS 50.0         | 0.0034 s           | 0.0135 s           | 3.94x     | 2.13 MB           | 0.71 MB           |
| Texture    |     50 | FBN 16           | 0.0208 s           | 0.0656 s           | 3.15x     | 19.85 MB          | 6.29 MB           |
| Texture    |     50 | FBN 32           | 0.0222 s           | 0.0685 s           | 3.09x     | 19.95 MB          | 6.29 MB           |
| Texture    |     50 | FBN 64           | 0.0168 s           | 0.0836 s           | 4.99x     | 9.83 MB           | 4.82 MB           |
| Texture    |     50 | FBS 10.0         | 0.0151 s           | 0.1102 s           | 7.28x     | 11.96 MB          | 5.41 MB           |
| Texture    |     50 | FBS 25.0         | 0.0216 s           | 0.0722 s           | 3.34x     | 19.30 MB          | 6.09 MB           |
| Texture    |     50 | FBS 50.0         | 0.0223 s           | 0.0671 s           | 3.01x     | 21.08 MB          | 6.60 MB           |
| Texture    |     75 | FBN 16           | 0.0770 s           | 0.2352 s           | 3.05x     | 84.43 MB          | 25.62 MB          |
| Texture    |     75 | FBN 32           | 0.0707 s           | 0.2428 s           | 3.44x     | 86.59 MB          | 26.22 MB          |
| Texture    |     75 | FBN 64           | 0.0722 s           | 0.2605 s           | 3.61x     | 63.72 MB          | 20.14 MB          |
| Texture    |     75 | FBS 10.0         | 0.0481 s           | 0.2832 s           | 5.89x     | 13.36 MB          | 17.08 MB          |
| Texture    |     75 | FBS 25.0         | 0.0709 s           | 0.2450 s           | 3.45x     | 89.44 MB          | 25.79 MB          |
| Texture    |     75 | FBS 50.0         | 0.0710 s           | 0.2437 s           | 3.43x     | 88.88 MB          | 25.65 MB          |
| Texture    |    100 | FBN 16           | 0.1847 s           | 0.5083 s           | 2.75x     | 213.37 MB         | 64.12 MB          |
| Texture    |    100 | FBN 32           | 0.2045 s           | 0.5217 s           | 2.55x     | 224.11 MB         | 66.17 MB          |
| Texture    |    100 | FBN 64           | 0.1959 s           | 0.5690 s           | 2.90x     | 204.71 MB         | 60.71 MB          |
| Texture    |    100 | FBS 10.0         | 0.0984 s           | 0.5682 s           | 5.78x     | 22.17 MB          | 37.15 MB          |
| Texture    |    100 | FBS 25.0         | 0.2138 s           | 0.5338 s           | 2.50x     | 230.74 MB         | 68.10 MB          |
| Texture    |    100 | FBS 50.0         | 0.1916 s           | 0.5222 s           | 2.73x     | 229.77 MB         | 68.48 MB          |

