# Benchmarks

## Speed of Pictologics

This page gives the speed of Pictologics 0.7.0 on one computer. A later version of this page will compare more programs, data sets and computers.

!!! info "How we measured"
    - **Data**: synthetic and seeded. Correlated noise with blob-shaped ROIs (with holes) for the features; normal noise cubes of 1 mm voxels for the filters; a CT-like image of 512 × 512 × 200 voxels (0.7 × 0.7 × 1.25 mm: air, a body of soft tissue with noise, a lesion of 20 mm radius of 54,746 voxels) for the preprocessing, the pipeline and the loaders.
    - **Time**: the wall-clock time of one call, the median of 5 runs after one warm-up run (the warm-up compiles the numba code). The tables give the median; the list tables also give the fastest and the slowest run.
    - **Threads**: numba uses 10 threads (the default of Pictologics on this computer), except in the thread test.
    - **Features**: the feature functions on their own, without the pipeline. The texture rows use FBN 32, except in the grey level table.

### Computer and Software

- **Hardware**: Apple M4 Pro, 14 cores, 48 GB
- **OS**: macOS 27.0.1 (arm64)
- **Python**: 3.12.10
- **Core deps**: pictologics 0.7.0, numpy 2.5.3, scipy 1.17.0, numba 0.67.0, pandas 2.3.3, matplotlib 3.10.7
- **BLAS/LAPACK**: Apple Accelerate (from `numpy.show_config()`)
- **Numba threads**: 10

### Feature Families

ROI cubes: 25³ (9,861 ROI voxels), 50³ (101,638 ROI voxels), 75³ (315,151 ROI voxels), 100³ (830,970 ROI voxels), 150³ (2,311,384 ROI voxels).

[![Feature families](assets/benchmarks/features.png)](assets/benchmarks/features.png)

| Task | 25³ | 50³ | 75³ | 100³ | 150³ |
|:--|--:|--:|--:|--:|--:|
| Intensity (first order) | 0.1 ms | 1.2 ms | 1.4 ms | 2.6 ms | 12.2 ms |
| Intensity histogram | 0.1 ms | 1.2 ms | 3.4 ms | 7.6 ms | 22.7 ms |
| Intensity-volume histogram | 0.1 ms | 0.7 ms | 2.0 ms | 5.1 ms | 14.0 ms |
| Morphology | 1.5 ms | 2.4 ms | 3.9 ms | 5.7 ms | 14.0 ms |
| Local intensity (peaks) | 1.2 ms | 3.3 ms | 9.8 ms | 22.4 ms | 63.5 ms |
| Spatial intensity (Moran's I, Geary's C) | 1.8 ms | 8.1 ms | 21.2 ms | 54.0 ms | 182.5 ms |
| Texture (all 6 families) | 1.0 ms | 3.4 ms | 7.1 ms | 14.6 ms | 37.8 ms |

### Texture Families

Each family on its own (`calculate_all_texture_features(..., families=[...])`), FBN 32.

[![Texture families](assets/benchmarks/texture_families.png)](assets/benchmarks/texture_families.png)

| Task | 25³ | 50³ | 75³ | 100³ | 150³ |
|:--|--:|--:|--:|--:|--:|
| GLCM | 0.2 ms | 0.6 ms | 1.2 ms | 2.0 ms | 5.2 ms |
| GLRLM | 0.4 ms | 1.1 ms | 3.1 ms | 6.9 ms | 19.9 ms |
| GLSZM | 0.3 ms | 1.4 ms | 2.1 ms | 4.0 ms | 8.6 ms |
| GLDZM | 0.3 ms | 1.5 ms | 2.4 ms | 4.6 ms | 10.6 ms |
| NGTDM | 0.3 ms | 0.5 ms | 1.1 ms | 2.4 ms | 5.8 ms |
| NGLDM | 0.3 ms | 0.5 ms | 1.1 ms | 2.4 ms | 5.7 ms |

### Texture and Grey Levels

All six texture families on the 75³ cube (315,151 ROI voxels) with FBN discretisation.

[![Texture and grey levels](assets/benchmarks/texture_grey_levels.png)](assets/benchmarks/texture_grey_levels.png)

| Task | 8 levels | 16 levels | 32 levels | 64 levels | 128 levels | 256 levels |
|:--|--:|--:|--:|--:|--:|--:|
| Texture (all 6 families) | 6.8 ms | 7.5 ms | 7.2 ms | 6.8 ms | 6.7 ms | 7.1 ms |

### Filters

Image cubes of 1 mm voxels. Convolution filters use the mirror boundary; the FFT filters (Simoncelli, Riesz) use their periodic default.

[![Convolution filters](assets/benchmarks/filters_convolution.png)](assets/benchmarks/filters_convolution.png)

[![Wavelet and FFT filters](assets/benchmarks/filters_wavelet.png)](assets/benchmarks/filters_wavelet.png)

| Task | 64³ | 128³ | 192³ | 256³ |
|:--|--:|--:|--:|--:|
| Mean (support 5) | 1.1 ms | 5.0 ms | 14.8 ms | 47.9 ms |
| Gaussian (sigma 2 mm) | 1.6 ms | 6.3 ms | 19.9 ms | 54.4 ms |
| LoG (sigma 2 mm) | 4.4 ms | 16.4 ms | 50.9 ms | 144.4 ms |
| Laws L5E5E5 with energy | 1.9 ms | 9.2 ms | 26.5 ms | 88.9 ms |
| Laws L5E5E5, rotation invariant, energy | 4.2 ms | 21.2 ms | 62.9 ms | 194.6 ms |
| Gabor (axial, rotation invariant) | 11.8 ms | 36.0 ms | 69.7 ms | 114.9 ms |
| Gabor (3 planes, rotation invariant) | 35.3 ms | 119.6 ms | 234.2 ms | 395.7 ms |
| Wavelet db2 LHL, level 1 | 1.1 ms | 4.3 ms | 11.5 ms | 38.6 ms |
| Wavelet db2 LHL, level 1, rotation invariant | 10.8 ms | 94.6 ms | 296.9 ms | 855.2 ms |
| Simoncelli, level 1 | 1.0 ms | 3.3 ms | 11.3 ms | 29.0 ms |
| Riesz, order (1, 0, 0) | 0.8 ms | 3.4 ms | 11.5 ms | 29.6 ms |
| Riesz-LoG (sigma 2 mm) | 5.7 ms | 21.4 ms | 63.2 ms | 170.6 ms |
| Riesz-Simoncelli, level 1 | 1.0 ms | 3.8 ms | 12.4 ms | 31.2 ms |

### Preprocessing

The functions of `pictologics.preprocessing` on the CT-like image.

[![Preprocessing](assets/benchmarks/preprocessing.png)](assets/benchmarks/preprocessing.png)

| Task | Time (median) | Fastest | Slowest |
|:--|--:|--:|--:|
| Resample to 1 mm (linear) | 11.6 ms | 11.3 ms | 11.7 ms |
| Resample the mask to 1 mm (nearest) | 2.8 ms | 2.7 ms | 2.8 ms |
| Discretise, FBN 32 (ROI range) | 9.1 ms | 8.9 ms | 9.2 ms |
| Discretise, FBS 25 HU | 6.3 ms | 6.2 ms | 6.4 ms |
| Resegment, -100 to 400 HU | 3.2 ms | 3.2 ms | 3.3 ms |
| Ring of 3 mm (grow_mask) | 11.4 ms | 10.0 ms | 11.9 ms |
| Keep the largest component | 4.7 ms | 3.2 ms | 4.7 ms |

### Pipeline

Whole pipeline runs on the CT-like image, from the image in memory to the feature Series. The standard configurations resample to 0.5 mm and compute the intensity, morphology, texture, histogram and IVH features; the pipeline resamples only the region around the ROI. The 20 ROIs of `run_rois` are spheres of 8 mm radius.

[![Pipeline runs](assets/benchmarks/pipeline.png)](assets/benchmarks/pipeline.png)

| Task | Time (median) | Fastest | Slowest |
|:--|--:|--:|--:|
| run(): one standard configuration (standard_fbn_32) | 13.8 ms | 13.4 ms | 14.8 ms |
| run(): the 6 standard configurations | 54.5 ms | 53.5 ms | 54.7 ms |
| run(): resample to 1 mm, LoG, intensity | 30.8 ms | 29.7 ms | 31.4 ms |
| run_rois(): 20 ROIs, intensity and morphology at 1 mm | 62.7 ms | 62.3 ms | 63.8 ms |

### Threads

`run()` of `standard_fbn_32` on the CT-like image with 1 to 10 numba threads.

[![Thread scaling](assets/benchmarks/threads.png)](assets/benchmarks/threads.png)

| Task | Time (median) | Fastest | Slowest |
|:--|--:|--:|--:|
| 1 thread | 38.0 ms | 37.8 ms | 39.1 ms |
| 2 threads | 24.4 ms | 23.8 ms | 24.8 ms |
| 4 threads | 17.0 ms | 16.7 ms | 17.3 ms |
| 8 threads | 13.9 ms | 13.7 ms | 14.1 ms |
| 10 threads | 13.2 ms | 13.0 ms | 13.7 ms |

### Loading

`load_image` of the CT-like image (int16 values), from a temporary folder on the internal disk.

[![Loading](assets/benchmarks/loading.png)](assets/benchmarks/loading.png)

| Task | Time (median) | Fastest | Slowest |
|:--|--:|--:|--:|
| NIfTI, gzip (.nii.gz) | 177.8 ms | 176.1 ms | 182.5 ms |
| NIfTI (.nii) | 17.7 ms | 17.3 ms | 18.0 ms |
| DICOM series (200 files) | 75.2 ms | 74.3 ms | 75.8 ms |
