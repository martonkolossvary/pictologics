# Tips for Speed and Memory

Pictologics computes in parallel and only where it must. These tips help it do so.

## Threads

- Pictologics uses one number of threads for all its parallel parts: the numba kernels, the FFT filters and the thread pools of the filters.
- By default, it uses the fast cores of the computer. numba gives each thread an equal part of a loop, so one slow core makes the whole loop wait. On a Mac with Apple silicon, the default is the performance cores. On Windows, it is all cores: the slow cores of an Intel chip are many and fairly fast, so they help. On Linux, it is the CPUs of the process, at most the CPU limit of a container.
- For another number, set the environment variable `PICTOLOGICS_NUM_THREADS` before the import, or call `pictologics.set_num_threads()` in your script. The function sets the number for the thread that calls it, at most the number of the import. Without a number, it sets the default again. `NUMBA_NUM_THREADS` works too.
- The features do not depend on the number of threads, with one exception: from SciPy 1.18, the FFT of SciPy splits its work by thread. So the Riesz and Simoncelli filters, the LoG of float64 images of 131,072 voxels or more, and Moran's I and Geary's C of large ROIs can change in their last digits with the number of threads (Moran's I by about 1e-16, relative, in a test).
- `run_batch(..., workers=4)` gives each worker process a quarter of the threads.
- See the [benchmark page](../benchmarks.md#threads) for the speed-up by threads.

## The First Run

- The first `import pictologics` compiles the numba code and keeps it in a cache. Later imports read the cache, so they take less time.
- Set `PICTOLOGICS_DISABLE_WARMUP=1` to import without the warm-up. The first call of a function then compiles its code.

## Work Only Near the ROI

- When no later step reads the image away from the ROI, the pipeline resamples, filters, binarizes and discretises only the region around the ROI box. A small ROI in a large CT then costs much less time and memory.
- These later steps read the image away from the ROI, so an earlier resample or filter makes the whole grid: a `filter`, a `resample`, a `binarize_mask` that keeps the value 0, and a `normalise` step with region `"image"`. For example, in "resample, then filter", the resample makes the whole new grid, and the filter computes only the region around the ROI.
- An in-memory image of another type than float64 (for example float32 or int16 from SimpleITK) is cast to float64 where the pipeline reads it: only the ROI region when no step reads the whole grid.

## Memory

- Before a resample, the pipeline checks that the new grid fits in the memory of the computer: a configuration needs at least 24 bytes for each voxel of the new grid. When it does not fit, the configuration stops with a `MemoryError` in its log entry.
- A resample of a whole CT to 0.5 mm makes a grid of hundreds of millions of voxels. With an ROI, the pipeline resamples only its region.
- The filters return float32 maps, half the memory of float64.
- A column-order array (nibabel's `dataobj` or `get_fdata()`) is copied into row order once, by the tiled copy of the loaders.
- Each `run_batch` worker holds one case at a time: the memory need grows with the number of workers.
- The FFT of Moran's I and Geary's C uses at most half of the memory (in a Linux container, half of its memory limit). The `run_batch` workers share this limit. A larger ROI box runs in slabs, and each of the Riesz and Simoncelli table caches keeps at most 2 GiB, and at most a quarter of the limit.

## Many Configurations and ROIs

- Configurations that share their first steps share the states after these steps, and families that read the same image and masks share their features (deduplication).
- `run_rois` runs many ROIs of one label map with one image load: much faster than one `run()` for each ROI.
- In direct calls, `calculate_all_texture_features(..., families=["glcm"])` computes only the families that you need.

## Measure

- `pipeline.get_log()` gives the time of each configuration (`elapsed_seconds`).
- The [benchmark page](../benchmarks.md) gives the speed of each part on one computer.
