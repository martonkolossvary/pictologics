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
- **Core deps**: pictologics 0.3.0, numpy 2.2.6, scipy 1.17.0, numba 0.62.1, pandas 2.3.3, matplotlib 3.10.7
- **PyRadiomics stack (parity runs)**: pyradiomics 3.1.1.dev111+g8ed579383, SimpleITK 2.5.3
- **BLAS/LAPACK**: Apple Accelerate (from `numpy.show_config()`)

Note: the benchmark script explicitly calls `warmup_jit()` before timing to avoid including Numba compilation overhead in the measured runtimes. All calculations are repeated 5 times and the average runtime is reported.

### Intensity

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Intensity time](assets/benchmarks/intensity_execution_time_log.png)](assets/benchmarks/intensity_execution_time_log.png) | [![Intensity speedup](assets/benchmarks/intensity_speedup_factor.png)](assets/benchmarks/intensity_speedup_factor.png) |

**Pictologics-only intensity families (IVH + spatial/local intensity):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0362 s                | 0.84 MB                |
|     25 | FBS 25.0         | 0.0330 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0322 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0291 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0304 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0331 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.5267 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.5144 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.5061 s                | 6.33 MB                |
|     50 | FBN 16           | 1.5137 s                | 6.33 MB                |
|     50 | FBN 32           | 1.5072 s                | 6.33 MB                |
|     50 | FBN 64           | 1.5058 s                | 6.33 MB                |
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
|     25 | FBS 10.0         | 0.0045 s                | 1.17 MB                |
|     25 | FBS 25.0         | 0.0046 s                | 1.17 MB                |
|     25 | FBS 50.0         | 0.0041 s                | 1.17 MB                |
|     25 | FBN 16           | 0.0041 s                | 1.17 MB                |
|     25 | FBN 32           | 0.0044 s                | 1.17 MB                |
|     25 | FBN 64           | 0.0043 s                | 1.17 MB                |
|     50 | FBS 10.0         | 0.0106 s                | 5.43 MB                |
|     50 | FBS 25.0         | 0.0115 s                | 5.43 MB                |
|     50 | FBS 50.0         | 0.0109 s                | 5.43 MB                |
|     50 | FBN 16           | 0.0107 s                | 5.43 MB                |
|     50 | FBN 32           | 0.0111 s                | 5.43 MB                |
|     50 | FBN 64           | 0.0113 s                | 5.43 MB                |
|     75 | FBS 10.0         | 0.0176 s                | 8.84 MB                |
|     75 | FBS 25.0         | 0.0168 s                | 8.84 MB                |
|     75 | FBS 50.0         | 0.0183 s                | 8.84 MB                |
|     75 | FBN 16           | 0.0172 s                | 8.84 MB                |
|     75 | FBN 32           | 0.0188 s                | 8.84 MB                |
|     75 | FBN 64           | 0.0174 s                | 8.84 MB                |
|    100 | FBS 10.0         | 0.0330 s                | 20.46 MB               |
|    100 | FBS 25.0         | 0.0319 s                | 20.46 MB               |
|    100 | FBS 50.0         | 0.0352 s                | 20.46 MB               |
|    100 | FBN 16           | 0.0368 s                | 20.46 MB               |
|    100 | FBN 32           | 0.0345 s                | 20.46 MB               |
|    100 | FBN 64           | 0.0343 s                | 20.46 MB               |


### Texture

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Texture time](assets/benchmarks/texture_execution_time_log.png)](assets/benchmarks/texture_execution_time_log.png) | [![Texture speedup](assets/benchmarks/texture_speedup_factor.png)](assets/benchmarks/texture_speedup_factor.png) |

**Pictologics-only texture families (GLDZM):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0002 s                | 0.15 MB                |
|     25 | FBS 25.0         | 0.0002 s                | 0.13 MB                |
|     25 | FBS 50.0         | 0.0002 s                | 0.13 MB                |
|     25 | FBN 16           | 0.0002 s                | 0.13 MB                |
|     25 | FBN 32           | 0.0002 s                | 0.13 MB                |
|     25 | FBN 64           | 0.0002 s                | 0.14 MB                |
|     50 | FBS 10.0         | 0.0003 s                | 0.16 MB                |
|     50 | FBS 25.0         | 0.0002 s                | 0.14 MB                |
|     50 | FBS 50.0         | 0.0003 s                | 0.13 MB                |
|     50 | FBN 16           | 0.0003 s                | 0.13 MB                |
|     50 | FBN 32           | 0.0003 s                | 0.14 MB                |
|     50 | FBN 64           | 0.0003 s                | 0.15 MB                |
|     75 | FBS 10.0         | 0.0003 s                | 0.21 MB                |
|     75 | FBS 25.0         | 0.0004 s                | 0.16 MB                |
|     75 | FBS 50.0         | 0.0003 s                | 0.14 MB                |
|     75 | FBN 16           | 0.0003 s                | 0.14 MB                |
|     75 | FBN 32           | 0.0004 s                | 0.15 MB                |
|     75 | FBN 64           | 0.0003 s                | 0.18 MB                |
|    100 | FBS 10.0         | 0.0004 s                | 0.20 MB                |
|    100 | FBS 25.0         | 0.0004 s                | 0.16 MB                |
|    100 | FBS 50.0         | 0.0004 s                | 0.14 MB                |
|    100 | FBN 16           | 0.0004 s                | 0.14 MB                |
|    100 | FBN 32           | 0.0004 s                | 0.15 MB                |
|    100 | FBN 64           | 0.0004 s                | 0.17 MB                |


### Filters

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Filters time](assets/benchmarks/filters_execution_time_log.png)](assets/benchmarks/filters_execution_time_log.png) | [![Filters speedup](assets/benchmarks/filters_speedup_factor.png)](assets/benchmarks/filters_speedup_factor.png) |

**Pictologics-only filters (Gabor, Laws, Simoncelli, Riesz, Mean):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0100 s                | 1.70 MB                |
|     25 | FBS 25.0         | 0.0094 s                | 1.70 MB                |
|     25 | FBS 50.0         | 0.0092 s                | 1.70 MB                |
|     25 | FBN 16           | 0.0092 s                | 1.70 MB                |
|     25 | FBN 32           | 0.0091 s                | 1.70 MB                |
|     25 | FBN 64           | 0.0112 s                | 1.70 MB                |
|     50 | FBS 10.0         | 0.0296 s                | 13.50 MB               |
|     50 | FBS 25.0         | 0.0277 s                | 13.50 MB               |
|     50 | FBS 50.0         | 0.0297 s                | 13.50 MB               |
|     50 | FBN 16           | 0.0294 s                | 13.50 MB               |
|     50 | FBN 32           | 0.0293 s                | 13.50 MB               |
|     50 | FBN 64           | 0.0286 s                | 13.50 MB               |
|     75 | FBS 10.0         | 0.0713 s                | 45.53 MB               |
|     75 | FBS 25.0         | 0.0576 s                | 45.50 MB               |
|     75 | FBS 50.0         | 0.0605 s                | 45.50 MB               |
|     75 | FBN 16           | 0.0615 s                | 45.50 MB               |
|     75 | FBN 32           | 0.0651 s                | 45.50 MB               |
|     75 | FBN 64           | 0.0603 s                | 45.50 MB               |
|    100 | FBS 10.0         | 0.1029 s                | 107.80 MB              |
|    100 | FBS 25.0         | 0.1010 s                | 107.80 MB              |
|    100 | FBS 50.0         | 0.1203 s                | 107.80 MB              |
|    100 | FBN 16           | 0.1285 s                | 107.80 MB              |
|    100 | FBN 32           | 0.1172 s                | 107.80 MB              |
|    100 | FBN 64           | 0.1180 s                | 107.80 MB              |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Filters    |     25 | FBN 16           | 0.0010 s           | 0.0047 s           | 4.72x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBN 32           | 0.0011 s           | 0.0048 s           | 4.42x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBN 64           | 0.0014 s           | 0.0048 s           | 3.43x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 10.0         | 0.0011 s           | 0.0047 s           | 4.16x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 25.0         | 0.0011 s           | 0.0046 s           | 4.07x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 50.0         | 0.0011 s           | 0.0046 s           | 4.22x     | 0.48 MB           | 0.94 MB           |
| Filters    |     50 | FBN 16           | 0.0058 s           | 0.0097 s           | 1.68x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBN 32           | 0.0057 s           | 0.0101 s           | 1.76x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBN 64           | 0.0057 s           | 0.0099 s           | 1.74x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 10.0         | 0.0056 s           | 0.0103 s           | 1.84x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 25.0         | 0.0056 s           | 0.0100 s           | 1.78x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 50.0         | 0.0058 s           | 0.0098 s           | 1.68x     | 3.82 MB           | 6.68 MB           |
| Filters    |     75 | FBN 16           | 0.0183 s           | 0.0276 s           | 1.51x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBN 32           | 0.0197 s           | 0.0252 s           | 1.28x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBN 64           | 0.0191 s           | 0.0251 s           | 1.31x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 10.0         | 0.0199 s           | 0.0260 s           | 1.31x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 25.0         | 0.0188 s           | 0.0270 s           | 1.44x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 50.0         | 0.0196 s           | 0.0264 s           | 1.34x     | 12.88 MB          | 23.38 MB          |
| Filters    |    100 | FBN 16           | 0.0492 s           | 0.0544 s           | 1.11x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBN 32           | 0.0466 s           | 0.0559 s           | 1.20x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBN 64           | 0.0462 s           | 0.0561 s           | 1.22x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 10.0         | 0.0424 s           | 0.0509 s           | 1.20x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 25.0         | 0.0431 s           | 0.0564 s           | 1.31x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 50.0         | 0.0460 s           | 0.0648 s           | 1.41x     | 30.52 MB          | 53.41 MB          |
| Intensity  |     25 | FBN 16           | 0.0008 s           | 0.0129 s           | 15.99x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 32           | 0.0008 s           | 0.0138 s           | 16.66x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 64           | 0.0008 s           | 0.0140 s           | 17.40x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 10.0         | 0.0026 s           | 0.0149 s           | 5.73x     | 0.25 MB           | 0.75 MB           |
| Intensity  |     25 | FBS 25.0         | 0.0008 s           | 0.0142 s           | 16.90x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 50.0         | 0.0008 s           | 0.0139 s           | 16.84x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     50 | FBN 16           | 0.0029 s           | 0.0650 s           | 22.23x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 32           | 0.0031 s           | 0.0658 s           | 21.10x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 64           | 0.0031 s           | 0.0675 s           | 21.50x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 10.0         | 0.0032 s           | 0.0675 s           | 20.84x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 25.0         | 0.0031 s           | 0.0671 s           | 21.72x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 50.0         | 0.0030 s           | 0.0662 s           | 22.31x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     75 | FBN 16           | 0.0109 s           | 0.2601 s           | 23.78x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 32           | 0.0119 s           | 0.2645 s           | 22.20x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 64           | 0.0116 s           | 0.2683 s           | 23.04x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 10.0         | 0.0126 s           | 0.2702 s           | 21.47x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 25.0         | 0.0111 s           | 0.2674 s           | 24.01x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 50.0         | 0.0114 s           | 0.2626 s           | 23.01x    | 5.81 MB           | 17.95 MB          |
| Intensity  |    100 | FBN 16           | 0.0242 s           | 0.5461 s           | 22.56x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 32           | 0.0237 s           | 0.5401 s           | 22.79x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 64           | 0.0253 s           | 0.5547 s           | 21.97x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 10.0         | 0.0234 s           | 0.5397 s           | 23.03x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 25.0         | 0.0219 s           | 0.5164 s           | 23.61x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 50.0         | 0.0236 s           | 0.5400 s           | 22.87x    | 12.16 MB          | 39.01 MB          |
| Morphology |     25 | FBN 16           | 0.0039 s           | 0.0562 s           | 14.59x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 32           | 0.0043 s           | 0.0570 s           | 13.17x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 64           | 0.0042 s           | 0.0566 s           | 13.33x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 10.0         | 0.0061 s           | 0.0576 s           | 9.41x     | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 25.0         | 0.0041 s           | 0.0573 s           | 13.95x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 50.0         | 0.0039 s           | 0.0574 s           | 14.83x    | 1.17 MB           | 1.18 MB           |
| Morphology |     50 | FBN 16           | 0.0105 s           | 0.9782 s           | 93.17x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 32           | 0.0109 s           | 0.9755 s           | 89.55x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 64           | 0.0108 s           | 0.9770 s           | 90.44x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 10.0         | 0.0107 s           | 0.9773 s           | 91.56x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 25.0         | 0.0102 s           | 0.9733 s           | 95.06x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 50.0         | 0.0104 s           | 0.9766 s           | 93.97x    | 5.43 MB           | 8.68 MB           |
| Morphology |     75 | FBN 16           | 0.0171 s           | 1.7787 s           | 104.04x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 32           | 0.0177 s           | 1.7799 s           | 100.29x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 64           | 0.0164 s           | 1.7835 s           | 108.45x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 10.0         | 0.0174 s           | 1.7793 s           | 102.43x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 25.0         | 0.0172 s           | 1.7806 s           | 103.30x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 50.0         | 0.0178 s           | 1.7789 s           | 100.16x   | 8.84 MB           | 36.60 MB          |
| Morphology |    100 | FBN 16           | 0.0357 s           | 9.0888 s           | 254.93x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 32           | 0.0338 s           | 9.1026 s           | 269.26x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 64           | 0.0361 s           | 9.0672 s           | 251.10x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 10.0         | 0.0327 s           | 8.6577 s           | 264.65x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 25.0         | 0.0320 s           | 8.9799 s           | 280.58x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 50.0         | 0.0343 s           | 9.0821 s           | 264.57x   | 20.46 MB          | 77.49 MB          |
| Texture    |     25 | FBN 16           | 0.0053 s           | 0.0493 s           | 9.23x     | 2.13 MB           | 0.69 MB           |
| Texture    |     25 | FBN 32           | 0.0056 s           | 0.0539 s           | 9.61x     | 2.07 MB           | 0.75 MB           |
| Texture    |     25 | FBN 64           | 0.0057 s           | 0.0693 s           | 12.16x    | 4.53 MB           | 1.95 MB           |
| Texture    |     25 | FBS 10.0         | 0.0077 s           | 0.0971 s           | 12.62x    | 10.10 MB          | 3.77 MB           |
| Texture    |     25 | FBS 25.0         | 0.0058 s           | 0.0560 s           | 9.72x     | 2.16 MB           | 0.91 MB           |
| Texture    |     25 | FBS 50.0         | 0.0053 s           | 0.0501 s           | 9.51x     | 2.13 MB           | 0.71 MB           |
| Texture    |     50 | FBN 16           | 0.0230 s           | 0.3078 s           | 13.35x    | 19.85 MB          | 6.29 MB           |
| Texture    |     50 | FBN 32           | 0.0236 s           | 0.3145 s           | 13.30x    | 19.95 MB          | 6.29 MB           |
| Texture    |     50 | FBN 64           | 0.0188 s           | 0.3293 s           | 17.53x    | 9.83 MB           | 4.82 MB           |
| Texture    |     50 | FBS 10.0         | 0.0171 s           | 0.3630 s           | 21.23x    | 11.96 MB          | 5.41 MB           |
| Texture    |     50 | FBS 25.0         | 0.0234 s           | 0.3157 s           | 13.47x    | 19.30 MB          | 6.09 MB           |
| Texture    |     50 | FBS 50.0         | 0.0243 s           | 0.3113 s           | 12.83x    | 21.08 MB          | 6.60 MB           |
| Texture    |     75 | FBN 16           | 0.0747 s           | 1.2351 s           | 16.53x    | 84.43 MB          | 25.62 MB          |
| Texture    |     75 | FBN 32           | 0.0757 s           | 1.2523 s           | 16.54x    | 86.59 MB          | 26.23 MB          |
| Texture    |     75 | FBN 64           | 0.0729 s           | 1.2440 s           | 17.06x    | 63.72 MB          | 20.14 MB          |
| Texture    |     75 | FBS 10.0         | 0.0502 s           | 1.2982 s           | 25.88x    | 13.36 MB          | 17.08 MB          |
| Texture    |     75 | FBS 25.0         | 0.0734 s           | 1.2536 s           | 17.07x    | 89.44 MB          | 25.80 MB          |
| Texture    |     75 | FBS 50.0         | 0.0759 s           | 1.2451 s           | 16.40x    | 88.88 MB          | 25.65 MB          |
| Texture    |    100 | FBN 16           | 0.2084 s           | 2.5765 s           | 12.36x    | 213.37 MB         | 64.12 MB          |
| Texture    |    100 | FBN 32           | 0.2115 s           | 2.6318 s           | 12.44x    | 224.11 MB         | 66.17 MB          |
| Texture    |    100 | FBN 64           | 0.2029 s           | 2.6326 s           | 12.97x    | 204.71 MB         | 60.72 MB          |
| Texture    |    100 | FBS 10.0         | 0.0994 s           | 2.6030 s           | 26.19x    | 22.17 MB          | 37.15 MB          |
| Texture    |    100 | FBS 25.0         | 0.1822 s           | 2.6013 s           | 14.27x    | 230.74 MB         | 68.10 MB          |
| Texture    |    100 | FBS 50.0         | 0.2079 s           | 2.6949 s           | 12.96x    | 229.77 MB         | 68.48 MB          |

