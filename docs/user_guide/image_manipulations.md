# Image Manipulations

This page covers advanced image loading and manipulation features for working with cropped masks, multi-label segmentations, and coordinate system transformations.

---

## Cropped Image Repositioning

Medical imaging software often stores segmentation masks as **cropped volumes** (bounding boxes around the region of interest) to minimize storage. When loading these cropped masks, they need to be repositioned into the original image's coordinate space for proper visualization and analysis.

### Why is this needed?

Cropped masks have smaller dimensions than the full image and are stored with their own spatial origin. The `pictologics` loader uses the spatial metadata (`ImagePositionPatient` for DICOM, affine matrix for NIfTI) to calculate where the cropped mask belongs in the full volume.

### Loading Cropped Masks

#### Single Image with Reference

```python
from pictologics import load_image

# Load the main image first
main_image = load_image("ct_scan/", recursive=True)

# Load a cropped mask and automatically reposition it
mask = load_image(
    "cropped_mask.dcm",
    reference_image=main_image  # Repositions to match main_image's space
)

# mask now has the same shape as main_image
print(f"Main image shape: {main_image.array.shape}")
print(f"Mask shape: {mask.array.shape}")  # Same as main_image
```

#### Merging Multiple Cropped Masks

When you have multiple cropped segmentation files (e.g., one per anatomical region), use `load_and_merge_images` with `reposition_to_reference=True`:

```python
from pictologics import load_image, load_and_merge_images
from pathlib import Path

# Load the main image
main_image = load_image("patient/ct_scan/", recursive=True)

# Find all segmentation files
seg_folder = Path("patient/segmentations/")
seg_files = [str(f) for f in seg_folder.glob("*.dcm")]

# Load and merge all cropped masks
merged_mask = load_and_merge_images(
    image_paths=seg_files,
    reference_image=main_image,
    reposition_to_reference=True,  # Enable cropped mask repositioning
    conflict_resolution="max",      # How to handle overlapping regions
)

# Now visualize
from pictologics.utilities import visualize_slices
visualize_slices(image=main_image, mask=merged_mask)
```

### Handling Axis Transposition

Some imaging software may store masks with different axis orderings. Use the `transpose_axes` parameter to correct this:

```python
# If the mask has Y and Z axes swapped
merged_mask = load_and_merge_images(
    image_paths=seg_files,
    reference_image=main_image,
    reposition_to_reference=True,
    transpose_axes=(0, 2, 1),  # Swap Y and Z before repositioning
)
```

### Error Handling

| Issue | Behavior |
|-------|----------|
| Spacing mismatch | `ValueError` raised (resampling not yet supported) |
| Orientation mismatch | Warning emitted, positioning continues |
| Mask outside reference bounds | Warning emitted, empty volume returned |
| Partial overlap | Valid region is positioned, rest is clipped |

---

## Multi-Label Masks

When loading multiple binary mask files (each containing only 0 and 1), you often want to distinguish them visually with different colors. The `relabel_masks` parameter automatically assigns unique label values (1, 2, 3, ...) to each file:

```python
from pictologics import load_image, load_and_merge_images
from pictologics.utilities import visualize_slices
from pathlib import Path

# Load the main CT scan
main_image = load_image("patient/ct_scan/", recursive=True)

# Find all AHA segment mask files (17 files)
seg_folder = Path("patient/aha_segments/")
seg_files = sorted(seg_folder.glob("*.dcm"))  # Sorted for consistent labeling

# Merge with unique labels per file
merged_mask = load_and_merge_images(
    image_paths=[str(f) for f in seg_files],
    reference_image=main_image,
    reposition_to_reference=True,
    relabel_masks=True,  # Each file gets a unique label: 1, 2, 3, ...
    conflict_resolution="max",
)

# Result: merged_mask.array contains values [0, 1, 2, 3, ..., 17]
# where 0 = background, 1 = first file, 2 = second file, etc.

# Visualize with different colors per segment
visualize_slices(image=main_image, mask=merged_mask, colormap="tab20")
```

!!! tip "Label Order"
    Labels are assigned based on the order of files in `image_paths`. Use `sorted()` for consistent ordering, or specify the exact order you want.

**Without vs With `relabel_masks`:**

| Parameter | Resulting Values | Use Case |
|-----------|-----------------|----------|
| `relabel_masks=False` (default) | `[0, 1]` (binary) | Single combined ROI |
| `relabel_masks=True` | `[0, 1, 2, ..., N]` (multi-label) | Visualizing distinct regions |
