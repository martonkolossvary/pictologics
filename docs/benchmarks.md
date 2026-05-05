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
- **OS**: macOS 26.4.1 (arm64)
- **Python**: 3.12.10
- **Core deps**: pictologics 0.4.1, numpy 2.2.6, scipy 1.17.0, numba 0.62.1, pandas 2.3.3, matplotlib 3.10.7
- **BLAS/LAPACK**: Apple Accelerate (from `numpy.show_config()`)

Note: the benchmark script explicitly calls `warmup_jit()` before timing to avoid including Numba compilation overhead in the measured runtimes. Timing and memory measurement are separated — `tracemalloc` is NOT active during timing to avoid biasing the comparison (its per-allocation hooks penalise pure-Python code more than JIT/C code). All calculations are repeated 5 times and the **mean** runtime is reported; peak memory is measured once separately.

### Intensity

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Intensity time](assets/benchmarks/intensity_execution_time_log.png)](assets/benchmarks/intensity_execution_time_log.png) | [![Intensity speedup](assets/benchmarks/intensity_speedup_factor.png)](assets/benchmarks/intensity_speedup_factor.png) |

**Pictologics-only intensity families (IVH + spatial/local intensity):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0325 s                | 0.81 MB                |
|     25 | FBS 25.0         | 0.0304 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0312 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0310 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0319 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0312 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.3359 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.3489 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.3918 s                | 6.33 MB                |
|     50 | FBN 16           | 1.3653 s                | 6.33 MB                |
|     50 | FBN 32           | 1.3865 s                | 6.33 MB                |
|     50 | FBN 64           | 1.3950 s                | 6.33 MB                |
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
|     25 | FBS 10.0         | 0.0028 s                | 1.17 MB                |
|     25 | FBS 25.0         | 0.0032 s                | 1.17 MB                |
|     25 | FBS 50.0         | 0.0027 s                | 1.17 MB                |
|     25 | FBN 16           | 0.0032 s                | 1.17 MB                |
|     25 | FBN 32           | 0.0029 s                | 1.17 MB                |
|     25 | FBN 64           | 0.0029 s                | 1.17 MB                |
|     50 | FBS 10.0         | 0.0097 s                | 5.43 MB                |
|     50 | FBS 25.0         | 0.0102 s                | 5.43 MB                |
|     50 | FBS 50.0         | 0.0105 s                | 5.43 MB                |
|     50 | FBN 16           | 0.0105 s                | 5.43 MB                |
|     50 | FBN 32           | 0.0106 s                | 5.43 MB                |
|     50 | FBN 64           | 0.0105 s                | 5.43 MB                |
|     75 | FBS 10.0         | 0.0175 s                | 8.84 MB                |
|     75 | FBS 25.0         | 0.0164 s                | 8.84 MB                |
|     75 | FBS 50.0         | 0.0164 s                | 8.84 MB                |
|     75 | FBN 16           | 0.0162 s                | 8.84 MB                |
|     75 | FBN 32           | 0.0163 s                | 8.84 MB                |
|     75 | FBN 64           | 0.0166 s                | 8.84 MB                |
|    100 | FBS 10.0         | 0.0333 s                | 20.46 MB               |
|    100 | FBS 25.0         | 0.0334 s                | 20.46 MB               |
|    100 | FBS 50.0         | 0.0335 s                | 20.46 MB               |
|    100 | FBN 16           | 0.0337 s                | 20.46 MB               |
|    100 | FBN 32           | 0.0333 s                | 20.46 MB               |
|    100 | FBN 64           | 0.0338 s                | 20.46 MB               |


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
|     25 | FBN 16           | 0.0001 s                | 0.13 MB                |
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
|     75 | FBN 16           | 0.0002 s                | 0.14 MB                |
|     75 | FBN 32           | 0.0002 s                | 0.15 MB                |
|     75 | FBN 64           | 0.0002 s                | 0.18 MB                |
|    100 | FBS 10.0         | 0.0002 s                | 0.20 MB                |
|    100 | FBS 25.0         | 0.0002 s                | 0.16 MB                |
|    100 | FBS 50.0         | 0.0003 s                | 0.14 MB                |
|    100 | FBN 16           | 0.0002 s                | 0.14 MB                |
|    100 | FBN 32           | 0.0002 s                | 0.15 MB                |
|    100 | FBN 64           | 0.0003 s                | 0.17 MB                |


### Filters

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Filters time](assets/benchmarks/filters_execution_time_log.png)](assets/benchmarks/filters_execution_time_log.png) | [![Filters speedup](assets/benchmarks/filters_speedup_factor.png)](assets/benchmarks/filters_speedup_factor.png) |

**Pictologics-only filters (Gabor, Laws, Simoncelli, Riesz, Mean):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0041 s                | 1.70 MB                |
|     25 | FBS 25.0         | 0.0040 s                | 1.70 MB                |
|     25 | FBS 50.0         | 0.0043 s                | 1.70 MB                |
|     25 | FBN 16           | 0.0042 s                | 1.69 MB                |
|     25 | FBN 32           | 0.0040 s                | 1.69 MB                |
|     25 | FBN 64           | 0.0046 s                | 1.69 MB                |
|     50 | FBS 10.0         | 0.0176 s                | 13.50 MB               |
|     50 | FBS 25.0         | 0.0179 s                | 13.50 MB               |
|     50 | FBS 50.0         | 0.0176 s                | 13.50 MB               |
|     50 | FBN 16           | 0.0178 s                | 13.49 MB               |
|     50 | FBN 32           | 0.0187 s                | 13.49 MB               |
|     50 | FBN 64           | 0.0199 s                | 13.49 MB               |
|     75 | FBS 10.0         | 0.0393 s                | 45.49 MB               |
|     75 | FBS 25.0         | 0.0403 s                | 45.49 MB               |
|     75 | FBS 50.0         | 0.0412 s                | 45.49 MB               |
|     75 | FBN 16           | 0.0414 s                | 45.50 MB               |
|     75 | FBN 32           | 0.0407 s                | 45.49 MB               |
|     75 | FBN 64           | 0.0408 s                | 45.49 MB               |
|    100 | FBS 10.0         | 0.0850 s                | 107.80 MB              |
|    100 | FBS 25.0         | 0.0878 s                | 107.80 MB              |
|    100 | FBS 50.0         | 0.0875 s                | 107.80 MB              |
|    100 | FBN 16           | 0.0874 s                | 107.80 MB              |
|    100 | FBN 32           | 0.0895 s                | 107.80 MB              |
|    100 | FBN 64           | 0.0883 s                | 107.80 MB              |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Filters    |     25 | FBN 16           | 0.0008 s           | N/A                | N/A       | 0.48 MB           | N/A               |
| Filters    |     25 | FBN 32           | 0.0008 s           | N/A                | N/A       | 0.48 MB           | N/A               |
| Filters    |     25 | FBN 64           | 0.0009 s           | N/A                | N/A       | 0.48 MB           | N/A               |
| Filters    |     25 | FBS 10.0         | 0.0008 s           | N/A                | N/A       | 0.48 MB           | N/A               |
| Filters    |     25 | FBS 25.0         | 0.0008 s           | N/A                | N/A       | 0.48 MB           | N/A               |
| Filters    |     25 | FBS 50.0         | 0.0008 s           | N/A                | N/A       | 0.48 MB           | N/A               |
| Filters    |     50 | FBN 16           | 0.0060 s           | N/A                | N/A       | 3.82 MB           | N/A               |
| Filters    |     50 | FBN 32           | 0.0062 s           | N/A                | N/A       | 3.82 MB           | N/A               |
| Filters    |     50 | FBN 64           | 0.0062 s           | N/A                | N/A       | 3.82 MB           | N/A               |
| Filters    |     50 | FBS 10.0         | 0.0056 s           | N/A                | N/A       | 3.82 MB           | N/A               |
| Filters    |     50 | FBS 25.0         | 0.0056 s           | N/A                | N/A       | 3.82 MB           | N/A               |
| Filters    |     50 | FBS 50.0         | 0.0062 s           | N/A                | N/A       | 3.82 MB           | N/A               |
| Filters    |     75 | FBN 16           | 0.0188 s           | N/A                | N/A       | 12.88 MB          | N/A               |
| Filters    |     75 | FBN 32           | 0.0192 s           | N/A                | N/A       | 12.88 MB          | N/A               |
| Filters    |     75 | FBN 64           | 0.0190 s           | N/A                | N/A       | 12.88 MB          | N/A               |
| Filters    |     75 | FBS 10.0         | 0.0181 s           | N/A                | N/A       | 12.88 MB          | N/A               |
| Filters    |     75 | FBS 25.0         | 0.0189 s           | N/A                | N/A       | 12.88 MB          | N/A               |
| Filters    |     75 | FBS 50.0         | 0.0192 s           | N/A                | N/A       | 12.88 MB          | N/A               |
| Filters    |    100 | FBN 16           | 0.0454 s           | N/A                | N/A       | 30.52 MB          | N/A               |
| Filters    |    100 | FBN 32           | 0.0444 s           | N/A                | N/A       | 30.52 MB          | N/A               |
| Filters    |    100 | FBN 64           | 0.0449 s           | N/A                | N/A       | 30.52 MB          | N/A               |
| Filters    |    100 | FBS 10.0         | 0.0451 s           | N/A                | N/A       | 30.52 MB          | N/A               |
| Filters    |    100 | FBS 25.0         | 0.0445 s           | N/A                | N/A       | 30.52 MB          | N/A               |
| Filters    |    100 | FBS 50.0         | 0.0448 s           | N/A                | N/A       | 30.52 MB          | N/A               |
| Intensity  |     25 | FBN 16           | 0.0004 s           | N/A                | N/A       | 0.24 MB           | N/A               |
| Intensity  |     25 | FBN 32           | 0.0005 s           | N/A                | N/A       | 0.24 MB           | N/A               |
| Intensity  |     25 | FBN 64           | 0.0004 s           | N/A                | N/A       | 0.24 MB           | N/A               |
| Intensity  |     25 | FBS 10.0         | 0.0012 s           | N/A                | N/A       | 0.24 MB           | N/A               |
| Intensity  |     25 | FBS 25.0         | 0.0004 s           | N/A                | N/A       | 0.24 MB           | N/A               |
| Intensity  |     25 | FBS 50.0         | 0.0005 s           | N/A                | N/A       | 0.24 MB           | N/A               |
| Intensity  |     50 | FBN 16           | 0.0025 s           | N/A                | N/A       | 1.40 MB           | N/A               |
| Intensity  |     50 | FBN 32           | 0.0029 s           | N/A                | N/A       | 1.40 MB           | N/A               |
| Intensity  |     50 | FBN 64           | 0.0029 s           | N/A                | N/A       | 1.40 MB           | N/A               |
| Intensity  |     50 | FBS 10.0         | 0.0026 s           | N/A                | N/A       | 1.40 MB           | N/A               |
| Intensity  |     50 | FBS 25.0         | 0.0029 s           | N/A                | N/A       | 1.40 MB           | N/A               |
| Intensity  |     50 | FBS 50.0         | 0.0031 s           | N/A                | N/A       | 1.40 MB           | N/A               |
| Intensity  |     75 | FBN 16           | 0.0112 s           | N/A                | N/A       | 5.81 MB           | N/A               |
| Intensity  |     75 | FBN 32           | 0.0114 s           | N/A                | N/A       | 5.81 MB           | N/A               |
| Intensity  |     75 | FBN 64           | 0.0117 s           | N/A                | N/A       | 5.81 MB           | N/A               |
| Intensity  |     75 | FBS 10.0         | 0.0128 s           | N/A                | N/A       | 5.81 MB           | N/A               |
| Intensity  |     75 | FBS 25.0         | 0.0113 s           | N/A                | N/A       | 5.81 MB           | N/A               |
| Intensity  |     75 | FBS 50.0         | 0.0110 s           | N/A                | N/A       | 5.81 MB           | N/A               |
| Intensity  |    100 | FBN 16           | 0.0224 s           | N/A                | N/A       | 12.16 MB          | N/A               |
| Intensity  |    100 | FBN 32           | 0.0231 s           | N/A                | N/A       | 12.16 MB          | N/A               |
| Intensity  |    100 | FBN 64           | 0.0238 s           | N/A                | N/A       | 12.16 MB          | N/A               |
| Intensity  |    100 | FBS 10.0         | 0.0240 s           | N/A                | N/A       | 12.16 MB          | N/A               |
| Intensity  |    100 | FBS 25.0         | 0.0237 s           | N/A                | N/A       | 12.16 MB          | N/A               |
| Intensity  |    100 | FBS 50.0         | 0.0229 s           | N/A                | N/A       | 12.16 MB          | N/A               |
| Morphology |     25 | FBN 16           | 0.0029 s           | N/A                | N/A       | 1.17 MB           | N/A               |
| Morphology |     25 | FBN 32           | 0.0027 s           | N/A                | N/A       | 1.17 MB           | N/A               |
| Morphology |     25 | FBN 64           | 0.0027 s           | N/A                | N/A       | 1.17 MB           | N/A               |
| Morphology |     25 | FBS 10.0         | 0.0039 s           | N/A                | N/A       | 1.17 MB           | N/A               |
| Morphology |     25 | FBS 25.0         | 0.0027 s           | N/A                | N/A       | 1.17 MB           | N/A               |
| Morphology |     25 | FBS 50.0         | 0.0028 s           | N/A                | N/A       | 1.17 MB           | N/A               |
| Morphology |     50 | FBN 16           | 0.0101 s           | N/A                | N/A       | 5.43 MB           | N/A               |
| Morphology |     50 | FBN 32           | 0.0102 s           | N/A                | N/A       | 5.43 MB           | N/A               |
| Morphology |     50 | FBN 64           | 0.0104 s           | N/A                | N/A       | 5.43 MB           | N/A               |
| Morphology |     50 | FBS 10.0         | 0.0097 s           | N/A                | N/A       | 5.43 MB           | N/A               |
| Morphology |     50 | FBS 25.0         | 0.0104 s           | N/A                | N/A       | 5.43 MB           | N/A               |
| Morphology |     50 | FBS 50.0         | 0.0102 s           | N/A                | N/A       | 5.43 MB           | N/A               |
| Morphology |     75 | FBN 16           | 0.0162 s           | N/A                | N/A       | 8.84 MB           | N/A               |
| Morphology |     75 | FBN 32           | 0.0163 s           | N/A                | N/A       | 8.84 MB           | N/A               |
| Morphology |     75 | FBN 64           | 0.0162 s           | N/A                | N/A       | 8.84 MB           | N/A               |
| Morphology |     75 | FBS 10.0         | 0.0173 s           | N/A                | N/A       | 8.84 MB           | N/A               |
| Morphology |     75 | FBS 25.0         | 0.0161 s           | N/A                | N/A       | 8.84 MB           | N/A               |
| Morphology |     75 | FBS 50.0         | 0.0164 s           | N/A                | N/A       | 8.84 MB           | N/A               |
| Morphology |    100 | FBN 16           | 0.0328 s           | N/A                | N/A       | 20.46 MB          | N/A               |
| Morphology |    100 | FBN 32           | 0.0331 s           | N/A                | N/A       | 20.46 MB          | N/A               |
| Morphology |    100 | FBN 64           | 0.0329 s           | N/A                | N/A       | 20.46 MB          | N/A               |
| Morphology |    100 | FBS 10.0         | 0.0331 s           | N/A                | N/A       | 20.46 MB          | N/A               |
| Morphology |    100 | FBS 25.0         | 0.0333 s           | N/A                | N/A       | 20.46 MB          | N/A               |
| Morphology |    100 | FBS 50.0         | 0.0332 s           | N/A                | N/A       | 20.46 MB          | N/A               |
| Texture    |     25 | FBN 16           | 0.0033 s           | N/A                | N/A       | 2.13 MB           | N/A               |
| Texture    |     25 | FBN 32           | 0.0034 s           | N/A                | N/A       | 2.07 MB           | N/A               |
| Texture    |     25 | FBN 64           | 0.0056 s           | N/A                | N/A       | 4.53 MB           | N/A               |
| Texture    |     25 | FBS 10.0         | 0.0057 s           | N/A                | N/A       | 10.10 MB          | N/A               |
| Texture    |     25 | FBS 25.0         | 0.0034 s           | N/A                | N/A       | 2.16 MB           | N/A               |
| Texture    |     25 | FBS 50.0         | 0.0035 s           | N/A                | N/A       | 2.13 MB           | N/A               |
| Texture    |     50 | FBN 16           | 0.0227 s           | N/A                | N/A       | 19.85 MB          | N/A               |
| Texture    |     50 | FBN 32           | 0.0236 s           | N/A                | N/A       | 19.95 MB          | N/A               |
| Texture    |     50 | FBN 64           | 0.0186 s           | N/A                | N/A       | 9.83 MB           | N/A               |
| Texture    |     50 | FBS 10.0         | 0.0154 s           | N/A                | N/A       | 11.96 MB          | N/A               |
| Texture    |     50 | FBS 25.0         | 0.0236 s           | N/A                | N/A       | 19.30 MB          | N/A               |
| Texture    |     50 | FBS 50.0         | 0.0238 s           | N/A                | N/A       | 21.08 MB          | N/A               |
| Texture    |     75 | FBN 16           | 0.0814 s           | N/A                | N/A       | 84.43 MB          | N/A               |
| Texture    |     75 | FBN 32           | 0.0736 s           | N/A                | N/A       | 86.59 MB          | N/A               |
| Texture    |     75 | FBN 64           | 0.0790 s           | N/A                | N/A       | 63.72 MB          | N/A               |
| Texture    |     75 | FBS 10.0         | 0.0493 s           | N/A                | N/A       | 13.36 MB          | N/A               |
| Texture    |     75 | FBS 25.0         | 0.0740 s           | N/A                | N/A       | 89.44 MB          | N/A               |
| Texture    |     75 | FBS 50.0         | 0.0735 s           | N/A                | N/A       | 88.88 MB          | N/A               |
| Texture    |    100 | FBN 16           | 0.1930 s           | N/A                | N/A       | 213.37 MB         | N/A               |
| Texture    |    100 | FBN 32           | 0.2134 s           | N/A                | N/A       | 224.11 MB         | N/A               |
| Texture    |    100 | FBN 64           | 0.2043 s           | N/A                | N/A       | 204.71 MB         | N/A               |
| Texture    |    100 | FBS 10.0         | 0.1033 s           | N/A                | N/A       | 22.17 MB          | N/A               |
| Texture    |    100 | FBS 25.0         | 0.2152 s           | N/A                | N/A       | 230.74 MB         | N/A               |
| Texture    |    100 | FBS 50.0         | 0.2085 s           | N/A                | N/A       | 229.77 MB         | N/A               |

