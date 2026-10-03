# Changelog

<!-- towncrier release notes start -->

## [0.6.0] - 2026-10-03

### Added

- A `normalise` step and the function `normalise_image` map the intensities of MR images and other images without fixed units: a z-score or a percentile range, with the statistics of the ROI or of the whole image (an optional value range leaves out the background). The step cancels the FBS start, as a filter does, and its log entry records the center and the scale. The function checks that the mask has the grid of the image.
- A `run_batch()` case can give a label map (`rois`, and optionally `labels`) in place of `mask`: it runs `run_rois()`, so each ROI gets a row of the result table, with its name in `roi`, and `grow_mask` steps with `nearest_roi` share the rings between the ROIs. The resume compares the label map and the labels too.
- A mask that `load_image` loads with `reference_image` can now hold the grid of the image in another voxel order: axes flipped or swapped, as some converters write NIfTI files. The loader turns the mask to the voxel order of the image. So a DICOM image with a NIfTI mask on the same grid now loads, where it gave "Origin mismatch" before. `load_and_merge_images` and `load_seg` turn their masks in the same way.
- Before a resample, the pipeline now checks that the new grid fits in the memory of the computer. A configuration needs at least 24 bytes for each voxel of the new grid. When the grid does not fit, the configuration stops at once, with a `MemoryError` in its log entry. Before, a PET image at 0.5 mm without a mask filled the memory, and the computer swapped for a long time. Its grid has 4.2 billion voxels and needs at least 94 GB.
- DICOM images, SEG masks and RTSTRUCT masks keep their DICOM FrameOfReferenceUID in the new `Image.frame_of_reference_uid`. A mask with another UID gives a warning when it loads onto the image, because it can belong to another scan. Images from other formats have no UID and are not checked.
- Each log entry now records how its values were made. `config_hash` is the SHA-256 of the configuration, the same in every run and session. `environment` holds the Python version, the platform, the versions of numpy, scipy, numba, PyWavelets, nibabel, pydicom and python-gdcm, and the numba thread count. Before, the log held only the Pictologics version.
- Every filter takes `padding_value`, the constant of constant value padding (IBSI 2 Z3VE), with the boundary `"constant"` (also named `"zero"`). For example, -1000 pads a CT image with air instead of water. A value other than 0 with another boundary is an error, also as a config problem. The default 0 gives the old values.
- The Gabor filter gives the part of its complex response that `response` names (IBSI 2 5P3T): `"modulus"` (the default, as before), `"angle"`, `"real"` or `"imaginary"`. Rotation-invariant pooling pools that part over all orientations. The modulus and the real part keep the shortcut over orientations that repeat at θ + π.
- The Gaussian filter (IBSI 2 8BC3) smooths the image at the scale `sigma_mm`, with the filter size cutoff `truncate`. Use it as `gaussian_filter`, or as the filter type `"gaussian"` in a pipeline. With a source mask, it uses normalized convolution, as the other filters. The capability metadata lists it (schema version 1.1.0).
- The texture features take the IBSI 1 distances: `glcm_distance` (the GLCM pair distance along each of the 13 directions), and `ngtdm_distance` and `ngldm_distance` (the Chebyshev distance of the neighbourhood). All are 1 by default. Give them in `texture_matrix_params`, or to the texture functions. An unknown option or a bad value in `texture_matrix_params` is now a config problem. The matrices equal plain loops over the voxels, and distance 1 gives the old values, bit for bit.
- Two templates hold configurations for cardiac CT: `lv` (left ventricular myocardium: whole, fat, myocardial tissue and calcium) and `coronary` (all, non-calcified, low-attenuation and calcified coronary plaque). Each has 30 configurations with a resegment range for each compartment, FBN and FBS discretisation, and resampling to 0.5 mm. The new `RadiomicsPipeline.from_template()` loads a template by name.
- `RadiomicsPipeline.get_log()` returns a copy of the processing log, with one entry for each configuration run. Each entry now also holds the run time of its configuration (`elapsed_seconds`). Before, the log was only in the private `_log` list, which the documentation and SlicerPictologics read.
- `RadiomicsPipeline.run_batch()` runs the configurations on many cases, also in several processes, each with its share of the threads. Each case gets its own result file. So a later call goes on where a stopped batch stopped, and an error of one case does not stop the batch. It returns a table with the status, error, warnings, run time and features of each case. On 28 CT cases, 4 processes did 6.6 cases per second, and one process 3.1.
- `RadiomicsPipeline.run_rois()` runs the configurations on each ROI of a label map. It gives the results of one `run()` for each label, bit for bit. But it loads the image once, and it makes each mask only inside the box of its label. On a CT with 100 labels and a texture configuration, it took 0.92 s instead of 1.81 s. With intensity features alone, it took 0.29 s instead of 1.21 s.
- `calculate_all_texture_features` takes `families`, for example `["glcm", "ngtdm"]`: it computes the features and the matrices of these families only.
- `grow_mask` grows or shrinks a mask by a distance in mm, or keeps a ring at its edge (for example the fat around a vessel, or the tissue around a lesion), as a function and as a pipeline step. The distances use the spacing of each axis, so the mask changes by the same amount in every direction. In `run_rois`, the step option `nearest_roi` gives each added voxel to its nearest ROI, so the rings of touching ROIs do not overlap.
- `load_image(..., suv="bw")` converts a DICOM PET image to its standardized uptake value by the QIBA vendor-neutral pseudo-code: body weight (`"bw"`), lean body mass by the Janmahasatian formula (`"lbm"`) or by the James formula (`"lbm_james"`), or body surface area by the Du Bois formula (`"bsa"`). On the QIBA FDG-PET/CT digital reference object, it gives the SUV values of the object. A missing or unsupported DICOM attribute raises an error that names it.
- `load_image` now also takes the DICOM files of one series: a list of file paths, or a `DicomPhaseInfo` from `get_dicom_phases`. It then reads only these files. Loading all 20 phases of a 1,000-file cardiac series takes 0.56 s this way, instead of 4.4 s with `dataset_index`, which reads all files for each phase.
- `load_image` now reads NRRD files (`.nrrd`, `.nhdr` with a detached data file, and 3D Slicer `.seg.nrrd`) and MetaImage files (`.mha`, `.mhd`), with its own readers. The geometry is in the LPS+ frame, from the RAS, LAS or LPS space of a NRRD file. A 4D file or the layers of a `.seg.nrrd` file give the volume of `dataset_index`. `get_segment_info` lists the segments of a `.seg.nrrd` file with their label values and layers. An axis of channels (for example colours or vectors) raises a clear error. On 60 files that SimpleITK wrote and on `.seg.nrrd` files of 3D Slicer, the arrays and the geometry are the same as SimpleITK and Slicer give.
- `load_rtstruct` reads the ROIs of a DICOM RTSTRUCT file onto the grid of a reference image: as one label image (the ROI Number is the label) or as binary masks by ROI name. `load_image` and a pipeline run read an RTSTRUCT mask with the image as the reference. A voxel is in an ROI when its center lies inside an odd number of the contours of its slice, so an inner contour cuts a hole. `get_segment_info` lists the ROIs of an RTSTRUCT file.
- `run()` now warns when the mask holds values that are not whole numbers, for example a probability map. Every voxel that is not 0 counts as ROI, so such a mask gives a wrong ROI. The check reads a regular sample of about 65,000 voxels of the ROI box. Its box scan is also the first ROI check of the run. A configuration with a `binarize_mask` step chooses its ROI itself, so it gives no warning.
- `run()`, `run_rois()` and the cases of `run_batch()` take `image_options`, the `load_image` options of an image path: for example a phase of a multi-phase DICOM folder (`{"dataset_index": 4}`) or a PET series as SUV (`{"suv": "bw"}`). Before, a run on a folder path always loaded the first phase. The log and the batch records hold the options, and a batch case runs again when they change.
- `save_image` writes an image, a mask or a response map as a NIfTI file. The geometry goes back to the RAS+ affine of NIfTI, so `load_image` and SimpleITK read the same array and LPS+ geometry. A bool mask is saved as uint8, and a 64-bit integer array as int32 when its values fit. A missing folder is made. So an RTSTRUCT, SEG or `.seg.nrrd` mask can become a NIfTI file.

### Changed

- Breaking change: NIfTI images now give their origin and direction in the LPS+ frame, as DICOM images and SimpleITK do. The loader changes the sign of the X and Y rows of the RAS+ affine. Code that reads `Image.origin` or `Image.direction` of a NIfTI image as RAS+ must change. An in-memory `Image` is in LPS+ too. The warning about mixed NIfTI and DICOM frames is gone, because the frames no longer mix.
- Breaking change: every FBS discretisation of the pipeline now starts at the same value in every image, as IBSI strongly recommends. An FBS step without `min_val` starts at the lower bound of an earlier `resegment` step of the intensity mask. Without both, `add_config` raises an error, and a configuration that skipped the check stops at its discretise step. Before, such bins started at the minimum of each ROI, so one grey level meant another HU range in each image. The standard FBS configurations now start at -1000 HU, the HU of air. This changes the texture, histogram and IVH values of FBS configurations without `min_val`. An FBS `ivh_discretisation` follows the same rule. The log records the start of each FBS step (`min_val_effective`).
- Breaking change: the long output of `format_results` names its feature column `feature_key`, not `feature_name`. The column holds the full feature key with its IBSI code, for example `mean_intensity_Q4LE`. `describe_features()` uses the same name for the same key; its `feature_name` column holds the name without the IBSI code. This applies to the dict, pandas and JSON outputs, and to an empty result. To keep the old name in a script, use `df.rename(columns={"feature_key": "feature_name"})`.
- Deduplication rules 1.1.0 are the new default. A family signature holds only the `extract_features` options that the family reads. For example, configurations that differ only in `ivh_params` now share the texture, intensity, morphology and histogram features. Rules 1.0.0 stay available and unchanged.
- Laws filters without the energy step and wavelet filters without rotation invariance now return float32, as every other filter. Before, they returned float64 for a float64 image, so a full CT map needed twice the memory (629 instead of 315 MB). Their passes still run in float64, so the values move only by the float32 rounding (about 1e-7, relative).
- The FFT filters (Simoncelli and Riesz) and the FFT sums of Moran's I and Geary's C now use numba's thread count. So do the thread pools of the rotation-invariant wavelet, Laws and Gabor filters and of `save_slices`. So one setting, `NUMBA_NUM_THREADS` or `numba.set_num_threads()`, limits all threads of a process, for example when several processes run at once. By default all cores run, as before; the Laws and Gabor filters used the cores plus 4 threads. Values and speed are unchanged (199 bench rows, two runs).
- The centre-of-mass shift (KLMA) is now computed in index units before the spacing, without the image origin, which cancels. The value moves by about 1e-13 (relative), and it no longer depends on where the arrays start in the image.
- With `source_mode="auto"` and no `sentinel_value`, a run now gives one warning about the sentinel search for all its configurations, not one warning for each. The warning names the configurations. The sentinel and NaN warnings also no longer go a second time to the root `logging` logger. So a run of a cardiac template prints one warning, not 60 lines.
- Without `window_center` and `window_width`, `save_slices` and `visualize_slices` now show all slices on one gray scale. The scale is the minimum and maximum of the whole volume, without NaN values. Before, each slice had its own minimum and maximum, so a tissue changed its gray level from slice to slice. A mask shown in gray uses the range of the whole mask. NaN pixels show as black; before, they went through an undefined cast.
- `DicomDatabase` now stores binary private values (for example a 60 KB CSA header) and private sequences as their size, such as `<OB, 60000 bytes>`, not as text. A scan of 10,000 files, half with such a header, needs 0.27 GB instead of 1.05 GB. Text private values stay as they are.
- `save_image`, the `export_csv` and `export_json` methods of `DicomDatabase` and `SRDocument`, and `SRBatch.export_combined_csv` and `export_log` now make a missing folder, as `save_results`, `save_configs` and `save_log` do. Before, they raised an error.
- The documentation is new: the pipeline guide is three pages (the pipeline, its steps, and the results and logs), the cookbook has seven tested recipes, and new tutorials cover masks from other tools, a study from a DICOM archive, rings, batch studies, cardiac phases, PET in SUV, MR normalisation and speed. The benchmark page now gives the speed of Pictologics alone, in tables and plots.
- `run()` without `config_names` still runs every configuration, but it now warns when the standard configurations run too. Pass `config_names`, or create the pipeline with `RadiomicsPipeline(load_standard=False)`, to run only your own.

### Fixed

- A DICOM SEG that stores its plane orientation or pixel spacing in each frame, as the standard allows, now loads with the right geometry. Before, a sagittal SEG of this kind collapsed to one slice.
- A DICOM SEG with more than 255 segments now loads. The combined label image is uint16 when a segment number is above 255. Before, the load failed, or the labels wrapped (300 became 44).
- A DICOM folder with more than one image series, for example two reconstructions of one scan, no longer loads as one mixed image. `load_image` and `get_dicom_phases` now raise an error that lists the series, and their new `series_uid` argument chooses one. Files without image pixel data (RTSTRUCT, RTPLAN, SR), SEG files and RT dose grids are not slices, so the loader skips them. It also leaves out images of another orientation or size than the rest of the series (a scout), with a warning.
- A DICOM pixel decode error now names the file and the cause.
- A NIfTI mask of the shape of its image, but with another origin, now goes to its place in the image by its origin, as a cropped mask does. Before, `load_image(..., reference_image=...)` raised an origin mismatch error.
- A FRACTIONAL DICOM SEG now counts a voxel as inside a segment when its value is at least half of the maximum fractional value. The new `fractional_threshold` argument of `load_seg` sets this fraction. Before, every value above 0 counted as inside.
- A Gabor `delta_theta` with few digits, for example 0.785398 for π/4, now gives the whole number of orientations (8, of which 4 are computed). Before, it gave 9 orientations, and the pooled response moved by up to 12 %.
- A `binarize_mask` step that keeps mask value 0 after a filter now gets filter values for every voxel. Before, the filter computed only the ROI region, so the voxels outside it held 0 (a mean of 7.4 instead of 112.8).
- A bin count of 32.0 from a YAML or JSON file no longer makes all texture features NaN, and an NGLDM alpha of 1.0 no longer compiles a second kernel. Both now work as the whole numbers 32 and 1.
- A configuration with NumPy numbers inside a tuple, for example `new_spacing=(np.float64(0.5), 0.5, 0.5)`, now saves them as plain numbers. Before, the YAML file held Python object tags that `load_configs` cannot read, and the JSON file held NumPy integers as text.
- A constant ROI now gives skewness and kurtosis 0, as IBSI defines, for the intensity and the intensity histogram features. Before, they were NaN. Equal values whose mean is not exact (for example 0.1) also give 0.
- A filter with a periodic boundary read the wrong image end when the pipeline filtered only the ROI region near an image edge. The LoG, wavelet and Laws features were then wrong, up to 7.5 times off. Such an axis is now filtered whole.
- A tag such as AcquisitionNumber no longer splits one DICOM scan into parts. A step-and-shoot CT can write a new number for each table step, so `load_image` loaded only 4 of 40 slices. A tag now splits a series only when slice positions repeat, and each group holds each position once. A trigger time that drifts from slice to slice (cine MR) now splits by the positions. This also fixes the phases of `get_dicom_phases` and the series of `DicomDatabase`.
- An unknown boundary name in a direct call of `mean_filter`, `laplacian_of_gaussian`, `laws_filter` or `gabor_filter` now raises a `ValueError` that lists the valid names, as the other filters do. Before, it raised a bare `KeyError`. `add_config` now checks the boundary of a filter step, and a `BoundaryCondition` member with a `padding_value` passes the check. Before, an unknown boundary failed only at run time, and an enum boundary with a padding value failed the check.
- An ROI voxel with a grey level outside [1, n_bins] (for example bin 0, the bin of a NaN voxel) now takes no part in the texture matrices. Before, NGTDM and NGLDM counted it as a neighbour with grey level 256.
- An enhanced multiframe DICOM file with several volumes no longer loads as one mixed image. Its frames sit at repeated positions, and `dataset_index` now picks one volume by the temporal position index or cardiac phase of the frames. A file with one volume and a `dataset_index` above 0 now raises an error.
- An error in one feature family no longer makes every feature of its configuration NaN. Only the features of that family are NaN, a warning names the family, and the log entry lists it in `family_errors`. `run_batch` marks such a case "incomplete".
- Colour (RGB) DICOM files now raise a clear error. Before, they loaded with wrong axes.
- DICOM SEG files now load each frame into the slice at its position. Before, the loader took the first dimension index as the slice number, but highdicom puts the segment number there. So each segment collapsed into one slice: in a test, a segment on 4 slices came back on 1, with a quarter of its voxels. The SEG origin and slice spacing now also come from the frame positions, also when the writer leaves out empty frames.
- DICOM images without rescale values now load as float64, like the same image from NIfTI. Many MR series have no rescale values, and some CT series have a slope of 1 and an intercept of 0. Before, these images kept their integer type, so resampling rounded the values, cut off negative values, and ran up to 50 times slower. `apply_rescale=False` still gives the stored values in their stored type.
- Deduplication no longer copies wrong values between configurations. Before, its reuse check ignored several settings, for example `source_mode`, `sentinel_value`, some preprocessing steps and `extract_features` options such as `ivh_params`. Now a feature family is reused only when the source mode, every preprocessing step before extraction and the extraction options are the same. Only a final `discretise` step can differ, for families that do not use the binned image. Results with `deduplicate=True` now equal results with `deduplicate=False`, for every rules version, also a pinned `"1.0.0"`.
- Every step now counts a voxel as ROI when its mask value is not 0, as the documentation states. Before, the bin limits, the sentinel search, Moran's I, the local intensity peak and the GLDZM distance map counted only values above 0. So negative labels gave another ROI in some families.
- First-order statistics now sum in float64, also for float32 values such as filter responses. Before, float32 values summed in float32, so the results carried float32 rounding errors and depended on numpy's summation order. numpy 2.3 changed that order; now numpy 2.2 and 2.5 give the same values to within 4e-16 (relative). IBSI 2 phase 2 values move by up to 3e-7, phase 3 values by up to 6e-4 (coefficients of variation with a mean near 0). All IBSI checks pass as before, and the phase 3 team agreement is unchanged.
- Gabor filters with γ below 1 use a kernel radius of ceil(6σ/γ), because the envelope is σ/γ long along one axis. Before, the radius ceil(6σ) cut the kernel short. IBSI 2 tests 4.a.1 and 4.a.2 now match their reference maps to 0.000 % of the range (before, 0.27 % and 0.14 %).
- In `roi_only` and `auto` mode, mask labels of 256 and more now keep their value. Before, the source-mask step cast the mask to uint8, so label 300 became 44, and a later `binarize_mask` for it found nothing.
- JSON output now writes NaN and infinite values as null. Before, it wrote the token NaN, which strict JSON readers reject. `save_results` now raises an error for an unknown file extension (a .parquet file got CSV text before), writes tab-separated files for .tsv, and makes a missing output folder.
- Label-map DICOM SEG files (Label Map Segmentation Storage, SOP class 1.2.840.10008.5.1.4.1.1.66.7) now load as segmentations. Before, `load_image` sent them to the plain DICOM loader, and `load_seg` read them as binary masks, so every label became 1. Now their pixel values are the segment numbers: the combined mask keeps them, and each separate mask holds the voxels of one number. The background, segment 0, is left out unless you ask for it.
- Laws filters without rotation invariance in the `roi_only` and `auto` source modes now give the right response. The normalized convolution divides by the fraction of the kernel weight on valid voxels. Before, it divided by the weight itself, so the responses were 0.15 to 0.23 times the right value.
- Multi-echo series now split by their echo number. The DICOM loader, `get_dicom_phases` and `DicomDatabase` asked for the tag `EchoNumber`, which does not exist; the keyword is `EchoNumbers` (0018,0086). So echoes split only when their slices shared positions. The phase tag, the phase label ("Echo 2"), `DicomPhaseInfo.split_tag` and the database metadata now use `EchoNumbers`. A multi-valued echo number, for example `1\2`, is one group value.
- NGLDM with an `ngldm_alpha` of 1 or more now also counts the neighbours with a lower grey level. Before, an unsigned subtraction counted only the neighbours up to alpha levels above the centre voxel. The default `ngldm_alpha` of 0 (IBSI) is not affected.
- No feature changes with the number of threads now. Surface area, volume and the intensity-weighted shape features add in fixed blocks, and the NGTDM sums are exact integers per grey level and neighbour count. Before, 19 of 174 features changed in the last digits (up to 5e-11, relative) between machines with other core counts. NGTDM values move by at most 1.3e-14.
- One NaN or infinite voxel in the ROI no longer makes every feature of a configuration NaN. Before feature extraction, each configuration leaves such voxels out of the intensity mask and gives a warning with their number. The morphological mask stays as it is. IBSI marks voxels outside the ROI with NaN, so these voxels have no intensity. The default bin limits also skip infinite values, and FBS puts a +inf voxel into bin 0 (before, an integer cast of infinity).
- Rotation-invariant wavelet filters now give the same result on every run. In 0.5.1, their parallel path (images above 2 million voxels) added the 24 rotations in the order in which the threads finished. So average pooling could change in the last bits from run to run. The rotations are now pooled in a fixed order, as in the sequential path.
- Saving and loading a configuration no longer changes its results. A `binarize_mask` range `mask_values: (lo, hi)` was saved as a list, which selects only the listed labels, so `(1, 3)` lost label 2. A range is now saved as `{range: [lo, hi]}`, which YAML and JSON files can also use, and it loads back as the tuple. A `BoundaryCondition` was saved as its scipy mode name, such as `"reflect"`, which loading rejects. It is now saved by its own name, such as `"mirror"`. Deduplication also tells a range from a label list with the same numbers, and these values no longer break `describe_features()`.
- Texture features of a one-slice image use the 4 in-plane directions and the in-plane GLDZM distance map, as for a 2D image. Before, 9 of the 13 directions left the slice: every voxel was a run of length 1 in them, and every zone had the distance 1. A one-slice ROI in a 3D image keeps the 3D rule.
- The DICOM loader now takes the slice spacing from the slice positions when SpacingBetweenSlices or SliceThickness differs from them by more than 1%. It then warns. SliceThickness is the width of a slice, not the step between slices, so overlapping or gapped series got a wrong z spacing. Series whose tags agree with their positions keep the tag value, so their results are unchanged.
- The IBSI 1 page shows the configurations that the check runs and a `run()` call that works. The IBSI 2 page no longer shows a local folder path.
- The IBSI 2 phase 3 page now uses one range per team, modality, filter and feature. The old page used one range per team and feature over all modalities and filters, which made the agreement too high. Within 1% of the range, CERR now agrees in 89.4% of the values (was 99.3%), and NCT Dresden in 98.2% (was 99.5%). The page also explains why the Simoncelli filters use the periodic boundary, not the mirror boundary of the IBSI 2 manual. With it, 58.1% of the Simoncelli values are within 1% of the range, against 56.3% with the mirror boundary. The page lists the 10 largest mismatches per team instead of all of them (14 KB instead of 524 KB).
- The `mask_threshold` of `resample_image`, and of the pipeline's linear or cubic mask resampling, now works for every mask type. Before, a uint8 mask was rounded first, so every threshold acted as 0.5. A bool mask kept only the voxels whose eight neighbours were all in the ROI (in a test, 242 voxels instead of 6,945). A uint8 or bool mask now rounds at the threshold, and a float64 mask compares its value with it, as before. The IBSI configurations binarize the mask first, so their results stay the same, bit for bit. Linear mask resampling of larger images also takes 17-62% less time and a third of the memory.
- The cache of the Simoncelli and Riesz transfer tables now has a lock. So filters that run in several threads at once no longer clash in it.
- The configuration check no longer warns about the `spatial_intensity_params` and `local_intensity_params` of the `extract_features` step. `run()` always accepted them.
- The structured report (SR) parser now reads each measurement group from its own container, as TID 1500 stores it. Each lesion keeps its tracking identifier (now also its `group_id`), finding, finding site and tracking UID. Each measurement keeps its derivation, for example Mean. Before, every group came out as "default" with none of these values, so the lesions could not be told apart.
- The warning about mixed NIfTI and DICOM geometry no longer appears for in-memory images and merged masks, because their source format is unknown.
- The wavelet, Simoncelli and Riesz filters now check their inputs and raise a clear error, also at `add_config`. Before, wavelet level 0 returned no response, a decomposition such as "LH" filtered only 2 of the 3 axes, and a Riesz order with 2 values ran on a 3D image. A lowercase decomposition such as "lhl" now works, and whole numbers such as 2.0 work as levels and orders.
- When the default start method is forkserver (the Linux default from Python 3.14), the header pool of `DicomDatabase` and `SRDocument.from_folders` uses spawn. A forkserver keeps the environment of its first start, so this pool and your own process pools mixed up the warm-up setting.
- `DicomDatabase.from_folders` no longer lists a DICOMDIR file as an image of patient "UNKNOWN". A DICOMDIR only indexes the other files of a DICOM medium, so the scan now skips it.
- `add_config` now checks a configuration and raises one error that lists every problem, with the closest valid name. It checks the step names, the parameters of each step and each filter type, and the feature family names. It also checks the discretise method and bin settings, the new spacing, and that texture features have an earlier discretise step. Before, these mistakes ran silently: a typo dropped a family or a step, and a wrong step gave only NaN values. `validate=False` keeps the old check; `from_dict(validate=True)` warns with the same checks, and an unknown source mode in a file now fails at load time.
- `add_config` now keeps its own copy of the steps, so a later edit of the caller's list no longer changes a stored configuration. It also accepts the `SourceMode` enum. `run()` takes one configuration name as a string, runs a name given twice only once, and suggests the closest name for an unknown one.
- `add_config` now reports a filter parameter that has no default and that the step leaves out: for example a `gaussian` or `log` step without `sigma_mm`, a `gabor` step without `lambda_mm`, or a `riesz` step without `order`. Before, such a configuration failed only when it ran.
- `format_results(fmt="long")` now raises `ValueError` for an unknown `output_type`, as the wide format does. Before, it returned a DataFrame.
- `get_dicom_phases(recursive=True)` now reads the same folder as `load_image(recursive=True)`: the folder with the most DICOM files. Before, it mixed the files of all subfolders. A negative `dataset_index` for DICOM data now raises an error, as it does for NIfTI.
- `load_image` now warns when the slices of a DICOM series are not evenly spaced, for example when a slice is missing. It also warns when the slice positions move sideways from the slice normal (a gantry tilt). The loader does not correct these cases, so the image is wrong after a gap and sheared with a tilt.
- `load_image`, `load_and_merge_images` and `RadiomicsPipeline.run()` now accept `pathlib.Path` objects. Before, `load_image` raised a `ValueError` for a `Path` to a NIfTI file, and `run()` treated a `Path` as an image and failed. The type hints of `save_slices`, `DicomDatabase.export_csv`, `DicomDatabase.export_json` and `get_dicom_phases` now allow a `Path` too, which already worked. `export_json` returns a string also for a `Path`.
- `merge_configs` now marks the deduplication plan out of date, as `add_config` and `remove_config` do. Before, `to_dict()`, `to_json()`, `to_yaml()` and `save_configs()` exported the plan of the configurations before the merge.
- `resegment_mask` now removes the voxels with a NaN intensity, as `filter_outliers` does. A NaN value is in no range.
- `run()` now converts an in-memory image of another type (for example int16 or float32) to float64, as the loaders do. Before, such an image resampled in its own type: rounded, on one core, and with other feature values. A numpy array instead of an `Image` now raises a clear TypeError.
- `save_configs`, `to_dict`, `to_json` and `to_yaml` now raise a `ValueError` for an unknown name in `config_names`, with the closest name, as `run()` does. Before, they left the name out without an error, so a wrong name gave a file without that configuration. They also take `"all_standard"` and a single name.
- `save_slices` raises `ValueError` for a single slice index outside the image, and `visualize_slices` for an `initial_slice` outside the image. Before, -1 saved the last slice as `slice_-001.png`, and an index past the end raised an `IndexError`. A NumPy integer now selects its slice; before, it selected slice 0.
- numpy numbers inside list or tuple parameters, for example a `new_spacing` of numpy integers, no longer stop `run()` when deduplication is on.

### Optimized

- A 4D NIfTI file now loads only the volume that `dataset_index` asks for. A 128 x 128 x 48 x 40 file loads one volume in 42 ms instead of 184 ms, with a peak of 12 MB instead of 480 MB. Before, the returned image also kept the whole 4D array (240 MB) instead of its own 6 MB. The values are the same, bit for bit.
- A Gabor filter in the pipeline now cuts each slice to the ROI region, grown by the kernel radius, before its FFT. Before, it filtered whole slices through the region. On a CT of 512 x 512 voxels with an ROI, the axial Gabor filter is 3.5 to 4.3 times faster. The values in the region move only by the float32 rounding of the smaller FFT (up to 4.4e-7 of the largest response).
- A pipeline run does its start-up work once, not once per configuration. This covers the first empty-ROI check, the auto-mode sentinel search, and the source mask of each source setup, kept until its last configuration. The package version is read once per process, and the deduplication plan is reused while the rules and configurations stay the same. Six light configurations in auto, ROI-only and full-image mode take 15-48% less time (a 512×512×200 CT: from 284 ms to 181 ms). Warnings, logs and features are unchanged.
- Configurations of one pipeline run now share identical preprocessing. When several configurations start with the same steps, source mode and sentinel value, these steps run once, and the later configurations continue from their result. The six standard configurations, which all start with the same 0.5 mm resampling, take 29-40% less time. Results and run logs are unchanged.
- DICOM series load faster and with half the memory. The loader now parses each file once, not twice, and reads the pixel data of the chosen phase only when needed. A rescaled series of 2^20 voxels or more now goes through one kernel into the row-order float64 image; each slice is freed once copied. A 512×512×200 CT series loads in 94 ms instead of 204 ms, with a peak of 528 MB instead of 1,054 MB. The default pipeline from those files takes 38% less time and 38% less memory. Values are unchanged, bit for bit.
- DICOM series, multiframe DICOM files and DICOM SEG masks of 2^20 voxels or more now load in row order (C order), as NIfTI images do. The series loader stacks each slice in one piece, and the fast parallel copy to row order now also takes int16, uint16 and uint8 data. On a 512×512×200 CT series, loading takes 36% less time, and the default pipeline 36-38% less, with the same peak memory. A SEG mask no longer comes in column order next to a row-order image; in a test, that mix made the pipeline 2.7-5.8 times slower. Values are unchanged, bit for bit.
- Discretisation uses its parallel kernels from 80,000 voxels, not from 1,048,576 (2^20) voxels. The kernels read a column-order (Fortran) array in its own order, without a copy; arrays in other layouts keep the old limit. On images of 262,000 to 885,000 voxels, FBS binning takes 0.16-0.24 ms instead of 0.46-1.6 ms, and FBN binning 34-55% of the time. Both use a third of the memory. On these images the pipeline runs up to 21% faster, with up to 58% lower peak memory. The bins are unchanged, bit for bit.
- Enhanced multiframe DICOM files load in one pass from the stored pixels to the float64 output. A 512 x 512 x 200 CT file loads in 72 ms instead of 127 ms, with half the memory (0.53 GB instead of 1.03 GB). The values are the same, bit for bit.
- FBN and FBS discretisation without given bin limits now find the ROI minimum and maximum in one parallel pass over the image and the mask. Before, they gathered the ROI values first. This applies to a row-order float64 image with a float64 or uint8 mask. FBN discretisation runs up to 3.4 times faster (a 512×512×200 CT: from 31 ms to 11 ms). The bins are unchanged, bit for bit.
- FBS discretisation in the pipeline gets the ROI minimum and maximum from one fused pass, with no copy of the ROI values. The FBS configuration with IVH runs 33 % faster on a full CT. The bins are the same.
- FIXED_CUTOFFS discretisation uses one parallel pass. An image of 26 million voxels takes 15 ms instead of 243 ms. The bins are the same.
- Gabor filters now use an FFT as long as the padded slice, not as long as the whole linear convolution. The kept part of the response is the same, so the values move only by the float32 rounding (up to 7e-7 of the largest response). Gabor filters are 1.2 to 1.6 times faster.
- In the `roi_only` and `auto` source modes, the pipeline keeps one boolean source mask and shares it between the steps, with no type copies. A resampled source mask with no invalid voxel no longer changes the masks. The results are the same.
- In the pipeline, an axial Gabor filter (not averaged over planes) now filters only the slices of the region that feature extraction reads. This applies when no later filter or resample step needs the rest of the image. Each slice is still filtered over its whole plane, so the features are unchanged, bit for bit. A Gabor configuration takes 21-45% less time on images with slices outside the ROI region. On a 512×512×200 CT, it goes from 196 ms to 128 ms and from 423 MB to 307 MB.
- Intensity features run faster, and IBSI configuration C runs 2.4 times faster. First-order statistics find their percentiles and median with one partial sort instead of two, and histogram features read them from the bin counts. IVH converts its values to float64 once, and the local intensity peak searches only the ROI box plus the reach of its sphere. Moran's I and Geary's C use FFT convolutions on large ROIs (117,000 voxels: 17 ms instead of 828 ms), and change by at most 1e-14. The FFT needs about 32 bytes per grid point; above 16 GB, or half of the memory, the pair loop runs instead. Then a warning gives the expected run time, and this change leaves all other values the same, bit for bit.
- Linear resampling now runs in parallel kernels that give scipy's output bit for bit, also with rounding, mask thresholds, uint8 masks and source masks. The IBSI configuration C pipeline runs about 2.3 times faster. Plain linear resampling now equals scipy exactly too; before, some edge voxels differed in the last bits (at most 5e-13). Discretisation skips its ROI search when the bin limits are given. `filter_outliers` and sentinel detection run in parallel on large images (up to 3.8 and 6.4 times faster on a 512×512×200 CT). All other outputs are unchanged, bit for bit.
- Morphology features build their surface mesh with a new marching cubes kernel. It gives the PyMCubes 0.1.6 mesh, with the same vertices and faces in the same order, so all features are unchanged, bit for bit. Morphology takes 22-41% less time on ROIs of 36,000 voxels or more (748,000 voxels: 45 to 27 ms), and 6-9% less on smaller ROIs. The default pipeline takes up to 17% less time.
- Morphology features run faster. Qhull now gets only the possible convex hull vertices: the first and the last mesh vertex of each grid line along each axis. It finds the same hull vertices in the same order, so the maximum 3D diameter and the MVEE features stay the same, bit for bit. The convex hull volume and area change by at most 2e-15 (relative). On large ROIs the convex hull step takes 32-65% less time, and morphology features 11% less on average (up to 21%).
- NIfTI images of 2^20 voxels or more load in one pass. The loader scales the stored values as nibabel's `get_fdata()` does and writes them straight into the row-order float64 array. So nibabel's column-order array and its row-order copy are no longer made. Loading takes 8-13% less time and 25-37% less peak memory (a 512×512×200 CT in .nii.gz: 279 to 243 ms, 839 to 524 MB). Values are unchanged, bit for bit.
- NIfTI images of 2^20 voxels or more now load in row order (C order), through a fast parallel copy, instead of nibabel's column order. This removes a full-image copy from each discretise, resegment and resample call. With the default configuration, a 512×512×200 CT runs about 1.7 times faster from load to features, with about 15% lower peak memory. Large IBSI 2 validation CT scans run 4-8% faster. Feature values are unchanged, bit for bit.
- Resampling with `round_intensities` rounds the new array in place. The rounding step uses half the memory. The values are the same.
- Resegmentation uses its parallel kernel from 80,000 voxels and sentinel detection from 200,000 voxels, not from 1,048,576 (2^20) voxels. Both kernels read an image and a mask in column order (Fortran) without a copy, and resegmentation keeps that order. On images of 262,000 to 885,000 voxels, resegmentation takes 0.17-0.33 ms instead of 0.19-2.95 ms. Sentinel detection takes 0.30-0.49 ms instead of 0.38-1.25 ms, and it no longer makes a full-size array for each candidate value. The results are unchanged, bit for bit.
- Rotation-invariant Laws filters compute their base responses 3 at a time, each with its share of the threads. They are 1.4 to 2 times faster, with 40 % to 45 % less memory for kernels with 6 bases. Rotation-invariant wavelets keep at most about 2 GB of responses in flight. The values are the same.
- Rotation-invariant Laws filters now pool each base response as soon as it is ready, and free it after its last rotation. Before, they kept every base response plus a signed copy for each of the 24 rotations. Max and min pooling flip signs in place and skip repeated rotations, and the separable passes and the energy image write into their own arrays. The values are the same, bit for bit. With the IBSI 2 phase 3 settings, a 512×512×200 CT takes 1.27 s instead of 1.82 s, and 2.9 GB instead of 5.0 GB. Smaller CT images take 20-41% less time and 42% less memory, and a Laws energy filter without rotations needs 57% less memory.
- Rotation-invariant wavelet filters use threads from 15,000 voxels and Laws filters from 20,000 voxels, not from 2 million voxels. Below 2 million voxels, the 24 rotations ran one after the other, although threads give the same result, bit for bit. With the IBSI 2 phase 3 settings, a Coiflet 3 wavelet on 262,000 to 1.1 million voxels runs 7-9 times faster (1.58 to 0.17 s). The Laws energy filter runs 1.7-2.2 times faster. Each thread holds one working array, so these images need more memory (for example from 35 MB to 133 MB).
- Rotation-invariant wavelets and the FFT filter caches keep less in memory. The parallel path of rotation-invariant wavelets kept all 24 rotation responses until the end; it now drops each response once pooled. The Simoncelli and Riesz transfer-function caches kept up to 32 and 64 full volumes without a byte limit; now each keeps at most 2 GB. On images of 2 to 2.5 million voxels, rotation-invariant wavelets take 10-14% less time and 29-31% less memory. The IBSI 2 phase 3 run uses 22% less memory. Results are unchanged, bit for bit.
- Simoncelli and Riesz filters with a non-periodic boundary return a compact array, not a view of the padded response. They keep 23 % to 52 % less memory at the same speed. The values are the same.
- Slice overlays color all labels in one step, not in one pass per label. A 512 x 512 slice with 117 labels takes 8 ms instead of 122 ms, and with 1 label 5 ms instead of 7 ms. The pixels are the same.
- Texture families that a configuration lists one by one (for example `glcm` and `glrlm`) share one matrix pass, as `texture` does. The features are the same.
- Texture features now use smaller matrices that give the same features: in the pipeline, in `calculate_all_texture_features` and in the feature functions that build a matrix. GLCM and GLRLM keep one table (the sum over the 13 directions) instead of 13, and GLSZM keeps only its non-zero cells. On the largest IBSI 2 validation CT scan, texture features run 1.5-2.2 times faster with 37-56% lower peak memory. The default configuration takes 21% less time there, with 28% lower peak memory. `calculate_all_texture_matrices` returns the same matrices as before, and feature values are unchanged, bit for bit.
- The GLCM and GLRLM thread tables add up in threads from 32,768 cells, and large tables are zeroed in threads. The texture matrices of 256 levels in 13 directions take 4.3 ms instead of 9.3 ms. The values are the same.
- The GLCM, GLRLM, GLDZM and NGLDM features now add only the rows and columns of their matrix that hold counts. A fixed FBS start, such as the -1000 HU of the standard configurations, leaves many low grey levels empty. Such a configuration is now as fast as one that starts at the ROI minimum; before, it was 4 to 8 % slower. The GLCM information correlations now use exact marginal counts. The features move in the last digits: up to 8e-15 (relative), and the information correlations, which subtract almost equal entropies, up to 4e-12.
- The IVH features of whole-number values come from one count per value instead of a sort. From 131,072 values on, the count runs in threads. It is 1.8 to 10 times faster, with almost no extra memory. Raw values search only the values that occur: 1.3 to 1.4 times faster. The values are the same.
- The JIT warm-up no longer compiles nine strided forms of the ROI box scan and the ROI min/max scans. The package passes row-order arrays to them, so it does not use these forms. A cold first warm-up is 1.7 s shorter.
- The NGTDM and NGLDM neighbour loop of the texture kernel now counts without branches. It sums the neighbours' grey levels as integers, which add exactly in any order. Texture features take 5% less time on average (up to 18% on large ROIs). Feature values are unchanged, bit for bit.
- The Simoncelli and Riesz transfer tables build faster and with less memory. The even Simoncelli table evaluates the band once and copies its mirrored part, and the Riesz table is computed in place. On a 512 x 512 x 200 grid, Simoncelli takes 203 instead of 231 ms and 258 instead of 311 MB, and Riesz takes 282 instead of 345 MB. The table values do not change.
- The Simoncelli and Riesz transfer tables keep only half of their rows, because the other rows mirror them. The products use mirrored views and run in threads, so the values are the same. The cached tables need half the memory and build 1.6 to 2 times faster, and warm calls take 3 % to 20 % less time.
- The Simoncelli filter now uses a real FFT, and the Riesz-Simoncelli filter applies both of its tables in one FFT round trip. The Simoncelli table now holds the even part of the band. For a real image, this part gives the real response that the filter keeps. With a cached table, the Simoncelli filter is 2.2 to 3.2 times faster and the Riesz-Simoncelli filter 3.1 to 3.8 times faster. A first call on a new image shape is 1.2 to 2.5 times faster. The Simoncelli values move by the float32 FFT rounding, up to 1e-6 of the largest response. Riesz-Simoncelli no longer rounds the band to float32 between its two steps. Its values move by up to 1.2e-5 of the largest response and come closer to a float64 reference. With a source mask and the periodic boundary, it still uses two round trips: it sets the band outside the mask to zero between them.
- The `resegment`, `filter_outliers` and `keep_largest_component` steps with `apply_to="both"` (the default) now process a shared mask once and keep it shared. `binarize_mask` did this already. Before, these steps ran twice and split the masks, so later steps also worked twice. Intensity-weighted morphology also reuses the morphology bounding box when the masks are shared. A pipeline with resegmentation and outlier filtering takes 7-31% less time and about 30% less peak memory. Results are unchanged, bit for bit.
- The histogram and the IVH of a configuration share one gather of the ROI values. Configurations with the same masks reuse the GLDZM distance map. The results are the same.
- The local intensity peaks take two stages. Row sums give every ROI voxel a close sphere mean, and the exact sum runs only on the voxels within a strict error bound of the best one and on the brightest voxels. For 3.3 million ROI voxels at 0.6 mm, they take 0.18 s instead of 1.16 s. The values are the same.
- The percentiles and the median of large ROIs (130,000 float values and more) come from a radix select. One parallel pass finds the value range, a parallel count puts the values into 65,536 buckets over that range, and a partition of the few values near the ranks gives the result. For 6.2 million values, the first-order features take 15 to 23 ms instead of 86 to 96 ms, also when the values crowd into a narrow range. The values are the same.
- The pipeline checks a large mask for ROI voxels with the parallel box scan, not in one thread. The result is the same.
- The pipeline filters only the ROI region for more filters. The mean filter and the Laws energy filter the part from the image start to the region end. A Gabor filter over three planes filters only the slices through the region. The source-mask modes also filter the region. The region grows by the local intensity sphere only when the local intensity features are computed. On a 300 × 300 × 160 image with an 80³ ROI, mean takes 35 ms instead of 171 ms. Laws energy takes 42 ms instead of 263 ms. Gabor over three planes takes 100 ms instead of 263 ms, with 93 MB instead of 330 MB. LoG in roi_only mode takes 51 ms instead of 811 ms. The values are the same.
- The pipeline runs faster. Its empty-ROI check reads a shared mask once and no longer runs after discretisation, and `binarize_mask` binarizes a shared mask once. The IVH discretisation bins only the ROI values when the configuration gives the bin limits. On a 512×512×200 CT, the default configuration takes 23% less time. A full-resolution FBS configuration with IVH takes 33% less time and 38% less peak memory, and IBSI configuration C 11-15% less time. Results are unchanged, bit for bit.
- The pipeline's LoG, wavelet and Laws filters now filter only the region that feature extraction reads, if no later step needs the rest. The region is the ROI box, grown by the reach of the local intensity sphere and of the filter; the rest is 0. These filters give each voxel the same value wherever the image ends, so all features are unchanged, bit for bit. A source mask, the mean filter, the Laws energy and the FFT filters still filter the whole image. On images larger than their ROI, a LoG configuration takes 28-90% less time, with up to half the peak memory. A wavelet configuration takes 13-84% less time; on a 512×512×200 CT, LoG goes from 2.5 s to 0.26 s.
- The scipy passes of the LoG, mean, Laws and wavelet filters run on slabs in threads for images of 262,144 voxels and more. The passes share one thread pool. Each line stays in one slab, so the values are the same. On 7.1 million voxels, LoG takes 33 ms instead of 182 ms, and mean 7 ms instead of 30 ms. Laws energy takes 19 ms instead of 69 ms, and a level-2 wavelet 53 ms instead of 390 ms. The source-mask forms are 2 times faster.
- The shape features count the marching cubes cells in parallel, write the vertices in physical units at once, and get the bounding box and the centre of the vertices from one pass. The values are the same.
- The source-mask forms of the mean, LoG and Laws filters make no extra copies. On 7.1 million voxels, the peak memory drops from 294 MB to 142 MB (mean) and from 296 MB to 163 MB (LoG). The values are the same.
- The texture matrices come from one zero-padded volume of grey levels, with no bounds checks and no hidden parallel regions. For an ROI of 764,000 voxels they take 21 ms instead of 43 ms; at 0.5 mm (6.2 million voxels) 140 ms instead of 303 ms, with 22 % less memory. The kernels compile once for every number of grey levels up to 65,535 (the new limit). The zone buffers are made for each call, so no buffer pool stays in memory after `run()`. The matrices are the same.
- The warm-up now compiles every kernel version that a first run needs, also for column-order images, float64 and bool masks, filter responses (float32) and every stored NIfTI type. A first run on a small NIfTI image takes 0.01 s instead of about 12 s of compiles. Kernels that make arrays use prange as their only parallel loop, and the cold warm-up takes 27 s instead of about 48 s.
- Wavelet passes now write into one array: a rotation-invariant wavelet above 2 million voxels needs 46-64% less memory, a plain wavelet 67% less. Simoncelli and Riesz multiply the spectrum in place, and the inverse FFT reuses that array. Simoncelli takes up to 24% less time and 58% less memory (a 512×512×200 CT: 449 to 345 ms, 2,517 to 1,049 MB). Riesz takes up to 18% less time and 17% less memory. Large transfer tables are built in slabs: a first Simoncelli or Riesz filter peaks at 536 or 573 MB, not 2,569 or 1,721 MB. Results are unchanged, bit for bit.
- When no later step reads the image away from the ROI, the discretise step cuts the image and the masks to the ROI box. The cut comes before the binning. Configurations that share an image share its cut. On a 512 × 512 × 200 image with four discretisations, the run takes 80 ms instead of 117 ms. The peak memory is 30 MB instead of 402 MB. The values are the same.
- When no later step reads the new grid away from the ROI, the resample step computes only the region around the ROI box, grown by the local intensity sphere when a later step needs it. The rest of the grid is never made, so a full CT at 0.5 mm needs a fraction of the time and memory. The features are the same, bit for bit. `resample_image` has a new `region` argument for this.
- With many grey levels, the GLCM and GLRLM thread tables can hold only the levels that occur. This happens when the full tables would take more than 64 MB and at most half of the levels occur. At 2,048 levels with 400 in the ROI, the peak memory of the texture pass is 100 MB instead of 276 MB. It is also 16 % faster. The matrices and the features are the same.
- YAML configurations and the bundled templates are parsed with PyYAML's libyaml safe loader when it is present, which gives the objects of `yaml.safe_load`. `from_yaml` on the six standard configurations takes 0.38 ms instead of 2.7 ms, and `RadiomicsPipeline()` 0.4 ms instead of 3.0 ms. `save_log` writes the JSON file while it encodes it: a 300-entry log peaks at 1.5 MB instead of 6.1 MB. The file is the same, byte for byte.
- `DicomDatabase.from_folders` now starts worker processes only for large scans: at most one worker per 1,000 DICOM files. The DICOM check of each file (132 bytes) runs in the calling process. Before, the scan started `cpu_count - 1` workers twice, whatever the number of files, and each worker imported Pictologics. The workers of `DicomDatabase.from_folders` and `SRDocument.from_folders` also skip the JIT warm-up, as they only read headers. A 200-file series takes 0.06 s instead of 3.4 s with the default settings, and 1.9 s instead of 3.7 s with 4 workers. The database is unchanged.
- `SRDocument.from_folders` now finds SR files by reading each file only up to its SOP Class UID. Batches below 2.5 MB (`SR_POOL_BYTES`) are parsed in the calling process, not in `cpu_count - 1` workers that each import Pictologics (about 1 s). From 2.5 MB it starts the requested workers, as before. 200 small reports take 0.53 s instead of 1.67 s, and the same reports among 6,000 CT files 1.12 s instead of 2.42 s. A file that breaks after its SOP Class UID now shows as an error in the processing log, not as a skipped file.
- `format_results(fmt="long")` builds the table as columns, not row by row. A pandas table of 200 cases with 1,020 features each takes 0.05 s instead of 0.08 s. The output is the same.
- `format_results(fmt="wide")` keeps the column names of the last 64 configurations, so the rows of many images share one copy of each name. 1,000 rows of 1,200 features take 55 MB instead of 142 MB, and 110 ms instead of 140 ms to build. The rows are unchanged.
- `import pictologics` takes about 0.3 s less (0.8 s instead of 1.1 s with a warm cache). The JIT warm-up no longer imports `scipy.signal` for an FFT convolution that no filter uses.
- `load_and_merge_images` merges each repositioned mask only in its own box of the reference grid, without a full-size copy per mask. Five cropped masks merge into a 512 x 512 x 200 grid in 41 ms instead of 0.78 s, with 0.40 GB instead of 1.75 GB. Masks on the reference grid merge 20 % faster, with 23 % less memory. The results are the same.
- `load_seg` reads SEG files with pydicom and decodes only the frames of the requested segments, one at a time. A SEG with 2,000 frames of 512 x 512 now needs 0.24 GB instead of 1.10 GB. As separate masks, it loads in 0.5 s instead of 1.0 s. `get_segment_info` reads only the header: 2 ms instead of 0.3 s. The masks are the same.
- `run()` keeps the state after a shared step only when a later configuration starts from it. This uses up to 24 % less memory when a shared step changes the image. The results are the same.
- `save_slices` saves up to 8 slices at a time in threads, as RGB instead of RGBA, with PNG compression level 3 instead of 6. The alpha channel was always 255, because the mask overlay is mixed into the colors. So the pixels are the same, and JPEG files are the same byte for byte. At 300 dpi, 40 slices of a 512×512×200 CT take 0.64 s instead of 17.6 s, and about 120 MB more peak memory. PNG files are 3% smaller at 300 dpi and 30% larger at 72 dpi, where saving takes 0.12 s instead of 1.4 s. TIFF files are 25% smaller.

### Dependencies

- Pictologics now uses numba 0.67 (was 0.62) with llvmlite 0.49, and it supports Python 3.14: CI tests Python 3.12, 3.13 and 3.14. numba 0.67 gives the same IBSI values as 0.62, bit for bit, at the same speed (83 bench rows). On Python 3.14, numpy 2.3.2 or newer is needed.
- PyMCubes is no longer a dependency. Pictologics builds its marching cubes mesh with its own kernel. The tests check that kernel against 32 PyMCubes meshes stored in `tests/data/marching_cubes_pymcubes.npz`.
- Removed four unused development dependencies: rich, psutil, mkdocs-gen-files and genbadge.
- highdicom is no longer a dependency, also not for the tests. Pictologics reads SEG and SR files with pydicom alone, and the tests read SEG and SR files that highdicom made once. pyjpegls goes with it: it has no Python 3.14 build, so pip had to compile it there. pydicom 3.0.1 is now the lowest version, because the SEG reader needs pydicom 3.
- python-gdcm (3.0.10 or newer) is now a dependency. With it, Pictologics loads DICOM data in every common compression: JPEG Lossless, JPEG-LS, JPEG 2000, baseline JPEG and RLE, also in SEG files. 12-bit lossy JPEG (JPEG Extended) needs pylibjpeg-libjpeg (GPL-3.0), which you can install yourself.


## [0.5.1] - 2026-07-26

### Added

- Added -3024 HU (outside the CT reconstruction field of view) to the pool of sentinel values recognised by automatic sentinel detection, alongside the existing -2048, -1024, -1000, 0 and -32768.
- Added versioned, machine-readable filter capability metadata (`FILTER_CAPABILITIES`, `get_filter_capabilities`, `CAPABILITIES_SCHEMA_VERSION`) describing input/kernel dimensionality, plane execution and averaging, rotation pooling, supported and effective boundaries, Riesz orders, structure-tensor steering, and anisotropic-spacing behaviour, so compliance tooling no longer has to infer them from signatures.
- Every executed pipeline `filter` step now records `params_requested` (the parameters as supplied) and `params_effective` (the arguments actually passed to the filter, including pipeline-injected values such as `spacing_mm` and the Riesz `variant` dispatch) in the run log, so a parameter that is defaulted, substituted, or cannot be applied exactly is always visible. Arrays are recorded as compact shape descriptors rather than raw data.
- The FFT-based filters (`simoncelli_wavelet`, `riesz_transform`, `riesz_log`, `riesz_simoncelli`) now honour the `boundary` parameter through a defined pad-filter-crop procedure instead of always being periodic. Periodic remains the default, so existing results are unchanged; `riesz_transform`, `riesz_log` and `riesz_simoncelli` gain a `boundary` argument.

### Changed

- Corrected the IBSI 2 compliance reporting: test 10.a is no longer described as structure-tensor aligned (it is unsteered), a skipped test can no longer be reported as passing, Phase 1 results now separate the strict 3D total from the broader volumetric total with the four 2D Gabor tests named, and the single Phase 2 feature with no published consensus value (8.B `stat_qcod`) is shown explicitly as coverage-only instead of being silently omitted. Both compliance pages now record provenance (package version, IBSI 2 reference manual version, reference dataset source and content hashes) and Phase 1 documents one exact, reproducible named test.

### Fixed

- Gabor filtering now scales its 2D kernel using the true in-plane voxel spacing of each plane instead of deriving every scale from the first spacing component. This corrects results for anisotropic voxels when `average_over_planes=True` (the planes containing the through-plane axis were previously computed with the wrong physical scale). Isotropic in-plane spacing keeps the existing code path and is bit-for-bit unchanged, so IBSI Gabor results are unaffected. The anisotropic-spacing warning has been removed because the case is now handled correctly rather than approximated.
- IBSI 2 Phase 1 verification now requests the padding each test specifies for the Riesz filters (zero padding for 9.a/9.b.1/10.a, nearest for 10.b.1) instead of relying on the periodic default, measurably improving agreement with the published reference response maps.
- The auto-detected sentinel warning no longer rounds a near-total sentinel fraction up to "100.0% of voxels", which wrongly implied that no voxels remained for feature extraction. Fractions above 99.95% are now reported as ">99.9%" (a literal 100.0% is shown only when every voxel is the sentinel), and the message states how many voxels remain valid. The full-precision `sentinel_proportion` recorded in the run log is unchanged.
- The pipeline no longer silently discards a requested `boundary` for the Simoncelli and Riesz filters, and an unsupported boundary value now raises a clear error instead of falling back to mirror padding. Each filter step records both the requested and the effective boundary in the run log.


## [0.5.0] - 2026-07-12

### Optimized

- Cache the Riesz frequency-domain transfer function (depends only on shape and order), making `riesz_transform` (and riesz_log/riesz_simoncelli) ~1.9-2.5x faster on repeat same-shape calls with byte-identical output.
- Cache the Simoncelli wavelet frequency-domain transfer function (depends only on shape and level), making `simoncelli_wavelet` ~1.8-2.9x faster on repeat same-shape calls with byte-identical output.
- Faster Moran's I and Geary's C spatial-intensity features (about 2x on larger ROIs) by exploiting the symmetry of the inverse-distance weights and letting the O(N^2) inner loop vectorise; byte-identical.
- Faster shared nonzero-bounding-box and ROI min/max scans via single-pass numba kernels (with numpy fallbacks for small or non-float inputs), reused across feature families and memoised per extraction pass; byte-identical.
- Much faster preprocessing via numba kernels and ROI-bounded work: image resampling up to ~35-66x, discretise_image ~5-9x, resegment_mask ~4x, and keep_largest_component ~35x on CT-sized volumes; bit-identical for discrete outputs and within 1e-9 for linear resampling.
- Reduce redundant array recomputation in the GLCM/GLRLM/GLDZM/NGLDM/NGTDM feature calculations (~7% faster per family); byte-identical output.
- Replace the GLDZM distance transform with a numba chamfer kernel (~3-4x faster on that step); distances are byte-identical to the previous scipy result.
- Share the resampled mask between the morphological and intensity masks when they are in sync (and drop a redundant mask copy), speeding multi-config pipeline runs with byte-identical output.
- Speed up texture ROI voxel counting on float masks (~5x) with a dtype-gated nonzero count; results are byte-identical.
- Substantially faster texture feature extraction (GLCM/GLRLM/GLSZM/GLDZM/NGTDM/NGLDM): vectorised the GLSZM matrix build, derive GLCM Ng_eff from the ROI bounding box, crop all texture work to the ROI bounding box to eliminate full-volume float64 mask scans, gate matrix computation to the requested families, and parallelise the zone-labelling kernel. Byte-identical results, roughly 1.4-15x faster on CT-sized and sparse volumes.


## [0.4.2] - 2026-05-16

### Changed

- Extended CI to lint tests, added a separate JIT smoke check outside coverage accounting, and made the MkDocs-published changelog the towncrier source with root changelog sync support.

### Fixed

- Fixed IBSI 1 compliance benchmarks that failed due to recent pipeline default changes by explicitly restricting `resegment` and `filter_outliers` steps to the `intensity` mask in standard compliance configurations (Configs C, D, E).
- Fixed compartment-specific pipeline semantics so `resegment` and `filter_outliers` update morphology masks by default, source masks constrain morphology masks after resampling, deduplication treats mask-narrowing steps as morphology dependencies, and `describe_features()` reports mask usage plus effective `apply_to` targets. Nonzero multi-label mask values are now treated as ROI membership rather than numeric weights across preprocessing, morphology, and texture calculations, while `binarize_mask` remains the explicit label-selection step. Configuration exports and processing logs now include mask semantics, package/schema metadata, configuration snapshots, deduplication settings, and effective source/sentinel details for reproducible runs on other machines.
- Fixed deduplication signatures to preserve preprocessing step order, corrected continuous-IVH deduplication dependency detection, applied per-frame DICOM rescale transforms for multiframe files, and propagated DICOM SEG alignment controls through reference-aware loading.

### Optimized

- Removed redundant initializations of `min_rot` and `max_rot` arrays in `_ombb_extents_numba` to slightly improve morphology calculation efficiency.


## [0.4.1] - 2026-05-05

### Changed

- load_image and load_and_merge_images now accept subvoxel_tolerance, subvoxel_warning_threshold, and min_overlap_fraction parameters for configurable sub-voxel alignment handling when repositioning cropped masks. RadiomicsPipeline.run() exposes the same controls via mask_subvoxel_tolerance, mask_subvoxel_warning_threshold, and mask_min_overlap_fraction. Repositioning settings are recorded in the pipeline run log.

### Fixed

- Expanded describe_features() so feature catalog rows record ordered preprocessing metadata, repeated step parameters, source-mode context, and extraction-step parameters in machine-readable columns.
- Fixed multi-configuration deduplication so texture, histogram, and IVH families no longer reuse each other's cached results when they share the same preprocessing signature.
- Fixed physical geometry validation for masks and source masks, including direction-aware cropped repositioning and axis-transpose metadata handling.
- Fixed silent empty-array return when a repositioned mask has no overlap with the reference image. This now raises a ValueError by default (controlled by min_overlap_fraction) to prevent undetected wrong-patient mask loading. Set min_overlap_fraction=0.0 to restore the previous warn-and-continue behaviour.
- Fixed texture subfamily aliases, NIfTI direction normalization, direction-aware geometry updates, and per-slice DICOM rescale handling.
- Updated configuration validation so serialized configs using current resampling, discretisation, and IVH runtime parameters no longer emit false unknown-parameter warnings.


## [0.4.0] - 2026-04-14

### Added

- Added `describe_features()` method to `RadiomicsPipeline` that returns a DataFrame cataloguing every feature the pipeline will produce, with columns for configuration name, feature identity (name, IBSI code, family, broad family group), discretisation/resampling/filter metadata. Useful for exporting data dictionaries and filtering features before extraction.

### Changed

- Pipeline runs now always return a complete, predictable set of feature columns for every configuration. When the ROI is empty after preprocessing (`EmptyROIMaskError`), a full NaN-valued Series is returned instead of raising an error, and processing continues to the next configuration. Partial extraction failures (e.g., PCA with ≤3 voxels, mesh computation errors, empty texture matrices) are backfilled with NaN so that no feature keys are silently dropped. A new `FEATURE_NAMES` registry in `pictologics.features` enumerates all 174 expected feature names by family.
- pipeline.run(subject_id=...) no longer injects subject_id into each configuration's feature Series. The parameter is now used exclusively for the processing log. To include subject identifiers in formatted output, pass them via format_results(meta={"subject_id": ...}).


## [0.3.5] - 2026-02-19

### Changed
- **Configuration Loading Behavior**: `load_configs()`, `from_yaml()`, `from_json()`, and `from_dict()` now default to loading **only** the provided configurations, without including standard predefined configs. Pass `load_standard=True` to include standard configs alongside loaded ones. `RadiomicsPipeline()` default constructor behavior is unchanged.

---

## [0.3.4] - 2026-02-14

### Fixed
- **Memory Exhaustion Issue**: Resolved a critical issue where resampled background voxels (value 0) were included in the ROI if they fell within the `resegment` range. The pipeline now explicitly applies the `source_mask` to the `intensity_mask` after resampling when `source_mode="auto"` or an explicit source mask is used. This prevents memory explosions for small ROIs in large volumes with sentinel backgrounds.
- **Pipeline Configuration Serialization**: Fixed a bug where `source_mode` and `sentinel_value` were lost during serialization (`to_dict`/`to_yaml`).
- **Sentinel Detection**: Fixed detection logic to correctly handle auto-generated full masks.

### Changed
- **Documentation**: Updated `Data Loading` and `Pipeline` user guides to clarify the usage of `source_mode="auto"` vs `"full_image"` and its impact on memory and correctness.

---

## [0.3.3] - 2026-02-11

### Changed
- **Sentinel Value Implementation**: Implemented proper handling of sentinel values in the pipeline to assure that they do not influence the feature extraction. 
- **Complete overhaul of User Guide**: Rewrote the user guide to improve clarity and organization. 
- **Benchmark Methodology Updates**: Refined timing methods for benchmarking with optimized measurement techniques, resulting in up to 40% faster execution for PyRadiomics compared to previous implementations. Therefore speed improvements of pictologics are now more modest.

---

## [0.3.2] - 2026-02-01

### Added
- **Feature Deduplication System**: Intelligent optimization for multi-configuration pipelines:
    - Automatically detects when configurations share preprocessing steps but differ only in discretization
    - Computes discretization-independent features (morphology, intensity) once and reuses across configurations
    - `deduplication_stats` property provides reuse/compute statistics after each run
    - Hash-based signature comparison using SHA256 for exact parameter matching
    - Versioned rules system (`DeduplicationRules`) for reproducibility

### Changed
- **Deduplication enabled by default**: `RadiomicsPipeline(deduplicate=True)` is now the default behavior
- **Documentation updated**: 
    - Case examples simplified to reflect default deduplication behavior
    - Benchmark page clarifies methodology (raw timing without caching) and notes additional speedups with deduplication


---

## [0.3.1] - 2026-01-31

### Added
- **Pipeline Configuration Serialization**: Full YAML/JSON export/import for `RadiomicsPipeline` configurations:
    - `save_configs()` / `load_configs()`: File-based configuration persistence
    - `to_yaml()` / `from_yaml()`: String-based YAML serialization
    - `to_json()` / `from_json()`: String-based JSON serialization
    - `to_dict()` / `from_dict()`: Dictionary conversion for programmatic use
- **Configuration Management Methods**:
    - `add_config()`: Register custom configurations
    - `get_config()`: Retrieve configuration by name (deep copy)
    - `remove_config()`: Delete configurations
    - `list_configs()`: List all registered configuration names
    - `merge_configs()`: Combine configurations from multiple pipelines
- **Template System**: YAML-based configuration templates in `pictologics/templates/`:
    - Standard configurations now loaded from `standard_configs.yaml`
    - Template loading API: `list_template_files()`, `load_template_file()`, `get_standard_templates()`, `get_all_templates()`, `get_template_metadata()`
- **Schema Versioning**: Configuration files include `schema_version` for forward compatibility and automatic migration
- **Configuration Validation**: Opt-in validation via `validate=True` parameter logs warnings for unknown steps/parameters
- **Documentation**: New "Predefined Configurations" user guide page with comprehensive examples including end-to-end multi-site study workflow

### Changed
- Standard configurations (`standard_fbn_*`, `standard_fbs_*`) now loaded from YAML templates instead of hardcoded dictionaries
- Updated pipeline.md documentation with condensed configuration section and cross-references

### Dependencies
- Added `pyyaml>=6.0` as core dependency for YAML serialization

---

## [0.3.0] - 2026-01-25

### Added
- **IBSI 2 Convolutional Filters**: Complete filter module (`pictologics/filters/`) with:
    - Mean filter (3D)
    - Laplacian of Gaussian (LoG)
    - Laws texture energy filters (3D rotation-invariant)
    - Gabor filters (2D per-slice)
    - Wavelet decomposition (Haar, Daubechies, Coiflet, Symlet families)
    - Simoncelli steerable pyramid
- **IBSI 2 Phase 1 Compliance**: Filter response map validation against digital phantoms
- **IBSI 2 Phase 2 Compliance**: Feature extraction from filtered images validated
- **IBSI 2 Phase 3 Compliance**: Multi-modality reproducibility validation across 51 patients × 3 modalities compared to 9 team submissions
- **Filter Pipeline Integration**: New `filter` step in `RadiomicsPipeline` for seamless filtered feature extraction
- **Mask Binarization Pipeline Step**: New `binarize_mask` preprocessing step with configurable `threshold`, `mask_values` (int/list/range tuple), and `apply_to` targeting.

### Changed
- Updated `mkdocs.yml` with IBSI 2 Phase 1, 2, 3 navigation
- Expanded pipeline documentation with filter usage examples and binarization

### Fixed
- **IBSI 1 Compliance (Morphology)**: Achieved passing values for Compactness 2 (`BQWJ`) and Asphericity (`25C7`) in Configs C/D/E and texture matrices in config D by binarizing masks before resampling.

---

## [0.2.0] - 2026-01-06

### Added
- **DICOM Database Utility**: `DicomDatabase` class for parsing complex DICOM folder hierarchies with Patient → Study → Series → Instance traversal, multi-phase detection, and DataFrame/JSON/CSV exports
- **DICOM SEG Loader**: `load_seg()` for loading DICOM Segmentation objects with multi-segment handling, geometry alignment, and seamless auto-detection in `load_image()`
- **DICOM SR Parser**: `SRDocument` class for parsing Structured Reports with measurement extraction, CSV/JSON export, and batch processing via `SRDocument.from_folders()`
- **DICOM Multi-Phase Support**: `load_image()` now supports multi-phase DICOM series with `dataset_index`, plus `get_dicom_phases()` for phase discovery
- **Visualization Utility**: `visualize_slices()` for interactive viewing and `save_slices()` for batch export with window/level normalization and colormap options
- **Cropped Image Repositioning**: `load_image()` and `load_and_merge_images()` support repositioning cropped masks into reference volume coordinate space
- **Intensity Rescaling**: `apply_rescale` parameter in `load_image` and related functions to toggle DICOM rescale slope/intercept application (default: True)
- **Sentinel Value Handling**: Documentation and examples for handling sentinel values (e.g. -2048 in Siemens DICOMs) using the `resegment` preprocessing step
- **Dependencies**: Added `highdicom`, `matplotlib`, `pillow`; updated `pandas>=2.0.0`

### Optimized
- **Morphology Speedup**: Implemented bounding box cropping for morphology features (mesh/moments), significantly accelerating extraction for sparse ROIs in large volumes
- **Texture Speedup**: Added slice-level skipping to texture calculation to ignore empty z-slices, vastly improving performance for disjoint ROIs (e.g. multiple tumors)

### Fixed
- DICOM file loading improvements: proper Z-spacing, 3D SEG handling, direction matrix extraction

### Changed
- `DicomDatabase` uses shared `split_dicom_phases()` for consistent multi-phase detection
- Comprehensive documentation updates for all new utilities

---


## [0.1.0] - 2025-12-28

**Initial commit**

---

[0.6.0]: https://github.com/martonkolossvary/pictologics/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/martonkolossvary/pictologics/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/martonkolossvary/pictologics/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/martonkolossvary/pictologics/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/martonkolossvary/pictologics/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/martonkolossvary/pictologics/compare/v0.3.5...v0.4.0
[0.3.5]: https://github.com/martonkolossvary/pictologics/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/martonkolossvary/pictologics/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/martonkolossvary/pictologics/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/martonkolossvary/pictologics/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/martonkolossvary/pictologics/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/martonkolossvary/pictologics/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/martonkolossvary/pictologics/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/martonkolossvary/pictologics/releases/tag/v0.1.0
