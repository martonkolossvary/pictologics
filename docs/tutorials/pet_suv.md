# PET in SUV

PET images hold the activity concentration (Bq/ml). Radiomics of PET uses the standardized uptake value (SUV) in place of it. This tutorial loads a DICOM PET series as SUV and runs the pipeline on it.

## 1. Load the Series as SUV

```python
from pictologics import load_image

pet = load_image("pet_series/", suv="bw")
print(pet.array.max())  # the SUVmax of the image
```

| `suv` | Normalised by | Unit |
|:--|:--|:--|
| `"bw"` | The body weight | g/ml |
| `"lbm"` | The lean body mass by the Janmahasatian formula | g/ml |
| `"lbm_james"` | The lean body mass by the James formula (PERCIST 1.0, and many older programs) | g/ml |
| `"bsa"` | The body surface area by the Du Bois formula | cm²/ml |

The conversion follows the [QIBA vendor-neutral pseudo-code](https://qibawiki.rsna.org/index.php/Standardized_Uptake_Value_(SUV)):

1. The images must be attenuation and decay corrected: CorrectedImage holds ATTN and DECY, and DecayCorrection is START. (DecayCorrection ADMIN also works: the images are then decay corrected to the injection, so the dose needs no decay. QIBA does not cover this case.)
2. The injected dose decays from the injection to the start of the series.
3. For a post-processed series (a series time after the acquisition), the start of the scan comes from the GE private scan time, else from the frame times, else from the earliest acquisition.
4. Units CNTS (Philips) take the Philips private SUV factor. Units GML are SUVbw already.

The weight, the height, the sex, the dose, its half-life and the times come from the DICOM header. A missing, empty or zero attribute raises an error that names it: the loader never guesses a value. Each slice keeps its own rescale slope.

!!! note "Checked on the QIBA test object"
    On the [QIBA FDG-PET/CT digital reference object](https://depts.washington.edu/petctdro/DROsuv_main.html) (female and male, 2013), `"bw"` gives its SUV values (0.00, 1.00, 4.00, 0.10, 0.90, and the test voxels 4.11 and -0.11) to within 6.4e-5. `"lbm_james"` gives its SUVlbm values; `"lbm"` (Janmahasatian) gives other ones, because the object uses the James formula.

!!! tip "Which lean body mass?"
    The James formula fails for very obese patients: its lean body mass falls with more weight, and can drop below 0. Tahari et al. (J Nucl Med 2014) recommend the Janmahasatian formula for SUL. Use `"lbm_james"` to compare with older results.

## 2. Run the Pipeline in SUV

For an image path, give the conversion as `image_options`:

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline(load_standard=False)
pipeline.add_config("pet_fbs", [
    {"step": "resample", "params": {"new_spacing": (3.0, 3.0, 3.0), "interpolation": "cubic"}},
    {"step": "resegment", "params": {"range_min": 0}},             # the FBS bins start at 0 SUV
    {"step": "discretise", "params": {"method": "FBS", "bin_width": 0.25}},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture"]}},
])
results = pipeline.run("pet_series/", "lesion.nii.gz", image_options={"suv": "bw"}, config_names=["pet_fbs"])
```

- The log entry of each configuration records the `image_options`, so you can see later that the image was in SUV.
- The FBS bins need a fixed start (IBSI). Here the `resegment` step gives it: 0 SUV.
- In `run_batch`, give `"image_options": {"suv": "bw"}` in each case.

## 3. A NIfTI Copy in SUV

Other programs often need NIfTI files. Convert once from DICOM, and save the SUV image:

```python
from pictologics import load_image, save_image

save_image(load_image("pet_series/", suv="bw"), "pet_suv_bw.nii.gz")
```

A NIfTI file has no PET header, so `suv` needs the DICOM series.

See also: [Data Loading](../user_guide/data_loading.md) and [Many ROIs and Batch Studies](batch.md).
