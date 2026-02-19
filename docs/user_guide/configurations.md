# Configuration & Reproducibility

Pictologics provides a comprehensive configuration management system designed for **reproducible radiomics research**. This guide covers the standard configurations, configuration file formats, and tools for sharing and managing pipeline configurations.

## Overview

The configuration system enables you to:

- **Use standard configurations** – Pre-tested, IBSI-compliant setups for common radiomics workflows
- **Export and import configurations** – Save your pipeline settings to YAML/JSON files for version control
- **Share configurations** – Collaborate by exchanging configuration files with colleagues
- **Ensure reproducibility** – Schema versioning ensures configurations remain compatible across versions

## Using Standard Configurations

Pictologics includes **6 standard configurations** optimized for radiomics feature extraction. All standard configurations share these characteristics:

- Isotropic resampling to **0.5mm × 0.5mm × 0.5mm**
- Cubic interpolation for images, nearest-neighbor for masks
- Complete feature extraction: **intensity**, **morphology**, **texture**, **histogram**, and **IVH**
- Performance-optimized: spatial and local intensity features disabled by default

### Running a Single Configuration

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()
results = pipeline.run(image, mask, config_names=["standard_fbn_32"])
```

### Running All Standard Configurations

Process all 6 standard configurations in a single call using the special `"all_standard"` shorthand:

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()
all_results = pipeline.run(image, mask, config_names=["all_standard"], subject_id="patient_001")
```

### Running Multiple Specific Configurations

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()
results = pipeline.run(
    image,
    mask,
    config_names=["standard_fbn_16", "standard_fbn_32", "standard_fbs_16"],
    subject_id="patient_001",
)
```

### Accessing Results

Results are returned as a dictionary mapping configuration names to `pandas.Series` objects:

```python
# Access features for a specific configuration
features = results["standard_fbn_32"]
print(features["mean_intensity_Q4LE"])  # Access a single feature by name

# Iterate over all configurations
for config_name, series in results.items():
    print(f"{config_name}: {len(series)} features")

# Convert to a pandas DataFrame (one row per configuration)
import pandas as pd
df = pd.DataFrame(results).T
```

## Configuration Specifications

### Fixed Bin Number (FBN) Configurations

FBN discretisation divides the intensity range into a **fixed number of bins**, regardless of the actual intensity values. This approach is useful when you want consistent bin counts across different images.

| Configuration | Bins | Use Case |
|---------------|------|----------|
| `standard_fbn_8` | 8 | Low-resolution texture analysis, small ROIs |
| `standard_fbn_16` | 16 | Balanced resolution and noise robustness |
| `standard_fbn_32` | 32 | High-resolution texture analysis (recommended default) |

#### Full Specification: `standard_fbn_32`

```yaml
standard_fbn_32:
  description: "Standard FBN-32: 0.5mm isotropic resampling, 32 fixed bins"
  steps:
    - step: resample
      params:
        new_spacing: [0.5, 0.5, 0.5]
        interpolation: cubic
    - step: discretise
      params:
        method: FBN
        n_bins: 32
    - step: extract_features
      params:
        families:
          - intensity
          - morphology
          - texture
          - histogram
          - ivh
        include_spatial_intensity: false
        include_local_intensity: false
```

### Fixed Bin Size (FBS) Configurations

FBS discretisation uses a **fixed bin width** (in Hounsfield Units for CT), which preserves the physical meaning of intensity values. This is preferred when comparing across studies or when absolute intensity values are clinically meaningful.

| Configuration | Bin Width | Use Case |
|---------------|-----------|----------|
| `standard_fbs_8` | 8.0 HU | High-resolution intensity preservation |
| `standard_fbs_16` | 16.0 HU | Balanced resolution (recommended for CT) |
| `standard_fbs_32` | 32.0 HU | Noise-robust analysis |

#### Full Specification: `standard_fbs_16`

```yaml
standard_fbs_16:
  description: "Standard FBS-16: 0.5mm isotropic resampling, 16.0 HU bin width"
  steps:
    - step: resample
      params:
        new_spacing: [0.5, 0.5, 0.5]
        interpolation: cubic
    - step: discretise
      params:
        method: FBS
        bin_width: 16.0
    - step: extract_features
      params:
        families:
          - intensity
          - morphology
          - texture
          - histogram
          - ivh
        include_spatial_intensity: false
        include_local_intensity: false
```

## Choosing the Right Configuration

### FBN vs FBS: Decision Guide

| Factor | FBN | FBS |
|--------|-----|-----|
| **Intensity range** | Variable (adapts to image) | Fixed (preserves HU meaning) |
| **Cross-study comparison** | Less suitable | Preferred |
| **Small ROIs** | Better (ensures bin coverage) | May have empty bins |
| **CT imaging** | Acceptable | Recommended |
| **MRI imaging** | Recommended | Less suitable (no standard units) |
| **IBSI recommendation** | Supported | Supported |

### Performance Considerations

The standard configurations have `include_spatial_intensity` and `include_local_intensity` set to `false` for performance:

- **Spatial intensity features**: Require distance calculations to ROI boundary (computationally expensive)
- **Local intensity features**: Require local neighborhood analysis

To enable these features, create a custom configuration variant:

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()

# Get the standard config and modify it
config = pipeline.get_config("standard_fbn_32")

# Find the extract_features step and enable spatial intensity
for step in config:
    if step["step"] == "extract_features":
        step["params"]["include_spatial_intensity"] = True
        step["params"]["include_local_intensity"] = True

# Add as a new configuration
pipeline.add_config("fbn_32_with_spatial", config)
```

## Configuration Files

Pictologics supports **YAML** and **JSON** formats for configuration files. YAML is recommended for human readability.

### YAML Format Specification

```yaml
schema_version: "1.0"
exported_at: "2026-01-31T12:00:00.000000"
configs:
  my_custom_config:
    - step: resample
      params:
        new_spacing: [1.0, 1.0, 1.0]
    - step: discretise
      params:
        method: FBN
        n_bins: 32
    - step: extract_features
      params:
        families:
          - intensity
          - morphology
          - texture
```

### JSON Format Specification

```json
{
  "schema_version": "1.0",
  "exported_at": "2026-01-31T12:00:00.000000",
  "configs": {
    "my_custom_config": [
      {
        "step": "resample",
        "params": {"new_spacing": [1.0, 1.0, 1.0]}
      },
      {
        "step": "discretise",
        "params": {"method": "FBN", "n_bins": 32}
      },
      {
        "step": "extract_features",
        "params": {"families": ["intensity", "morphology", "texture"]}
      }
    ]
  }
}
```

### Schema Versioning

Configuration files include a `schema_version` field to ensure forward compatibility:

- **Current version**: `1.0`
- Files without a version are treated as version `1.0`
- Future versions will include automatic migration when loading older configs

## Sharing Configurations

### Exporting Configurations

Save your configurations to share with collaborators or for version control:

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()

# Add custom configurations
pipeline.add_config("my_study_config", [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture"]}},
])

# Export to YAML (recommended for readability)
pipeline.save_configs("my_configs.yaml")

# Export to JSON
pipeline.save_configs("my_configs.json")

# Export specific configs only
pipeline.save_configs("study_config.yaml", config_names=["my_study_config"])
```

### Importing Configurations

Load configurations from files. The resulting pipeline contains **only** the configurations defined in the file:

```python
from pictologics import RadiomicsPipeline

# Load from YAML file (only file configs, no standard configs)
pipeline = RadiomicsPipeline.load_configs("my_configs.yaml")

# Load from JSON file
pipeline = RadiomicsPipeline.load_configs("my_configs.json")

# Load with validation (logs warnings for unknown parameters)
pipeline = RadiomicsPipeline.load_configs("my_configs.yaml", validate=True)

# Load file configs AND include standard configs
pipeline = RadiomicsPipeline.load_configs("my_configs.yaml", load_standard=True)
```

!!! note "Standard Configurations"
    By default, `load_configs()`, `from_yaml()`, `from_json()`, and `from_dict()` create a
    pipeline with **only** the provided configurations. Standard configurations (e.g.,
    `standard_fbn_32`) are not included unless you pass `load_standard=True`.
    
    If you need both your file configs and standard configs, you have two options:
    
    1. Pass `load_standard=True` to any loading method
    2. Load your file, then merge with a standard pipeline:
    
    ```python
    loaded = RadiomicsPipeline.load_configs("my_configs.yaml")
    loaded.merge_configs(RadiomicsPipeline())  # Add standard configs
    ```

### String-based Export/Import

For integration with databases or web services:

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()
pipeline.add_config("my_config", [...])

# Export to string
yaml_string = pipeline.to_yaml()
json_string = pipeline.to_json()

# Import from string
pipeline2 = RadiomicsPipeline.from_yaml(yaml_string)
pipeline3 = RadiomicsPipeline.from_json(json_string)
```

### Merging Configurations

Combine configurations from multiple sources:

```python
from pictologics import RadiomicsPipeline

# Load configs from different sources
pipeline1 = RadiomicsPipeline.load_configs("team_a_configs.yaml")
pipeline2 = RadiomicsPipeline.load_configs("team_b_configs.yaml")

# Merge into pipeline1
pipeline1.merge_configs(pipeline2)

# Handle duplicates explicitly
pipeline1.merge_configs(pipeline2, overwrite=True)  # Overwrite existing
pipeline1.merge_configs(pipeline2, overwrite=False)  # Keep existing (default)
```

### Configuration Validation

When loading configurations, enable validation to catch potential issues:

```python
# Validation logs warnings for:
# - Unknown step types
# - Unknown parameters for known steps
# - Missing required parameters

pipeline = RadiomicsPipeline.load_configs("config.yaml", validate=True)
```

## Template System (Advanced)

For programmatic access to configuration templates, Pictologics provides a template loading API.

### Loading Templates

```python
from pictologics.templates import (
    list_template_files,
    load_template_file,
    get_standard_templates,
    get_all_templates,
    get_template_metadata,
)

# List available template files
files = list_template_files()
# ['standard_configs.yaml']

# Load all standard configurations
standard_configs = get_standard_templates()
# {'standard_fbn_8': [...], 'standard_fbn_16': [...], ...}

# Get metadata from a template file
metadata = get_template_metadata("standard_configs.yaml")
# {'schema_version': '1.0', 'description': '...', 'config_count': 6}

# Load a specific template file
all_configs = load_template_file("standard_configs.yaml")
```

### Creating Custom Template Files

You can create your own template files and load them:

```python
from pictologics import RadiomicsPipeline
import yaml

# Define your organization's standard configs
org_configs = {
    "schema_version": "1.0",
    "description": "Organization standard configurations",
    "configs": {
        "org_standard_ct": [
            {"step": "resample", "params": {"new_spacing": [0.5, 0.5, 0.5]}},
            {"step": "discretise", "params": {"method": "FBS", "bin_width": 25.0}},
            {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture"]}},
        ],
        "org_standard_pet": [
            {"step": "resample", "params": {"new_spacing": [2.0, 2.0, 2.0]}},
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 64}},
            {"step": "extract_features", "params": {"families": ["intensity", "texture"]}},
        ],
    }
}

# Save to file
with open("org_configs.yaml", "w") as f:
    yaml.dump(org_configs, f)

# Load into pipeline
pipeline = RadiomicsPipeline.load_configs("org_configs.yaml")
```

## Best Practices for Reproducibility

### 1. Version Control Your Configurations

Always include configuration files in your version control system:

```bash
# Add to git
git add configs/study_configs.yaml
git commit -m "Add radiomics configuration for study XYZ"
```

### 2. Document Configuration Choices

Include comments in your YAML files explaining parameter choices:

```yaml
schema_version: "1.0"
description: |
  Configurations for lung nodule analysis study.
  FBS-25 chosen based on literature recommendation for CT texture analysis.
  0.5mm resampling matches thin-slice CT acquisition protocol.
configs:
  lung_nodule_primary:
    - step: resample
      params:
        new_spacing: [0.5, 0.5, 0.5]  # Match acquisition protocol
    # ... rest of config
```

### 3. Export Before Major Changes

Before modifying your pipeline, export the current state:

```python
pipeline.save_configs("configs_backup_2026-01-31.yaml")
```

### 4. Use Validation When Loading External Configs

```python
# Always validate configs from external sources
pipeline = RadiomicsPipeline.load_configs("collaborator_config.yaml", validate=True)
```

### 5. Include Schema Version in Publications

When publishing radiomics research, report:

- Pictologics version
- Configuration file (as supplementary material)
- Schema version used

## End-to-End Example: Multi-Site Radiomics Study

This section provides a complete, real-world workflow demonstrating how to create, save, load, and apply configurations across a multi-site radiomics study. The scenario involves:

- **Site A (Primary)**: Creates and validates the study configuration
- **Site B (Collaborator)**: Receives and applies the same configuration
- **Both sites**: Process their local cohorts with identical settings

### Step 1: Design the Study Configuration (Site A)

Site A designs a custom configuration tailored to their lung nodule CT analysis study:

```python
from pictologics import RadiomicsPipeline

# Initialize pipeline
pipeline = RadiomicsPipeline()

# Define a study-specific configuration
# This config is designed for thin-slice chest CT with lung nodule segmentations
lung_nodule_config = [
    # Step 1: Resample to 0.5mm isotropic (matches thin-slice CT)
    {
        "step": "resample",
        "params": {
            "new_spacing": [0.5, 0.5, 0.5],
            "interpolation": "cubic",  # Cubic spline for smooth interpolation
        }
    },
    # Step 2: Resegment to lung window and remove outliers
    {
        "step": "resegment",
        "params": {
            "range_min": -1000,  # Air
            "range_max": 400,    # Soft tissue upper limit
        }
    },
    # Step 3: Keep only the largest connected component (remove satellite lesions)
    {
        "step": "keep_largest_component",
        "params": {}
    },
    # Step 4: Discretise using Fixed Bin Size (preserves HU meaning)
    {
        "step": "discretise",
        "params": {
            "method": "FBS",
            "bin_width": 25.0,  # 25 HU bins (common for CT texture)
        }
    },
    # Step 5: Extract all feature families
    {
        "step": "extract_features",
        "params": {
            "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
            "include_spatial_intensity": False,  # Skip for performance
            "include_local_intensity": False,
        }
    },
]

# Add the configuration to the pipeline
pipeline.add_config("lung_nodule_fbs25", lung_nodule_config)

# Also add a variant with different bin width for sensitivity analysis
lung_nodule_fbs50 = lung_nodule_config.copy()
lung_nodule_fbs50[3] = {
    "step": "discretise",
    "params": {"method": "FBS", "bin_width": 50.0}
}
pipeline.add_config("lung_nodule_fbs50", lung_nodule_fbs50)

# Verify the configurations are registered
print("Available configurations:", pipeline.list_configs())
```

### Step 2: Test the Configuration Locally (Site A)

Before sharing, validate the configuration on a sample case:

```python
from pictologics import load_image

# Load a test case
image = load_image("test_data/ct_scan.nii.gz")
mask = load_image("test_data/nodule_segmentation.nii.gz")

# Run the primary configuration
results = pipeline.run(
    image=image,
    mask=mask,
    subject_id="test_case_001",
    config_names=["lung_nodule_fbs25"],
)

# Inspect results
config_features = results["lung_nodule_fbs25"]
print(f"Extracted {len(config_features)} features")
print(f"Sample features:")
print(f"  - Volume: {config_features.get('volume_mesh_ml_HTUR', 'N/A'):.2f} mm³")
print(f"  - Mean intensity: {config_features.get('mean_intensity_Q4LE', 'N/A'):.2f} HU")

# Save execution log for audit
pipeline.save_log("logs/test_run_001.json")
```

### Step 3: Export Configuration for Sharing (Site A)

Save the configuration to a file that can be shared with collaborators:

```python
# Export to YAML (human-readable, good for version control)
pipeline.save_configs(
    "configs/lung_nodule_study_v1.yaml",
    config_names=["lung_nodule_fbs25", "lung_nodule_fbs50"]
)

print("Configuration exported successfully!")
print("Share 'configs/lung_nodule_study_v1.yaml' with Site B")
```

The exported YAML file will look like this:

```yaml
schema_version: "1.0"
exported_at: "2026-01-31T10:30:00.000000"
configs:
  lung_nodule_fbs25:
    - step: resample
      params:
        new_spacing: [0.5, 0.5, 0.5]
        interpolation: cubic
    - step: resegment
      params:
        range_min: -1000
        range_max: 400
    - step: keep_largest_component
      params: {}
    - step: discretise
      params:
        method: FBS
        bin_width: 25.0
    - step: extract_features
      params:
        families: [intensity, morphology, texture, histogram, ivh]
        include_spatial_intensity: false
        include_local_intensity: false
  lung_nodule_fbs50:
    # ... similar structure with bin_width: 50.0
```

### Step 4: Load and Apply Configuration (Site B)

Site B receives the configuration file and processes their cohort:

```python
from pathlib import Path
from pictologics import RadiomicsPipeline, load_image
from pictologics.results import format_results, save_results

# Load the shared configuration (with validation)
pipeline = RadiomicsPipeline.load_configs(
    "configs/lung_nodule_study_v1.yaml",
    validate=True  # Logs warnings for any issues
)

# Verify loaded configurations
print("Loaded configurations:", pipeline.list_configs())

# Define the local data directory
data_dir = Path("site_b_data/")
output_dir = Path("site_b_results/")
output_dir.mkdir(exist_ok=True)

# Process all cases
all_results = []
for case_folder in sorted(data_dir.glob("patient_*")):
    patient_id = case_folder.name
    
    # Load image and mask
    image = load_image(case_folder / "ct.nii.gz")
    mask = load_image(case_folder / "nodule.nii.gz")
    
    # Run the primary configuration
    results = pipeline.run(
        image=image,
        mask=mask,
        subject_id=patient_id,
        config_names=["lung_nodule_fbs25"],
    )
    
    # Format for CSV export
    row = format_results(
        results,
        fmt="wide",
        meta={"subject_id": patient_id},
    )
    all_results.append(row)
    
    config_features = results["lung_nodule_fbs25"]
    print(f"Processed {patient_id}: {len(config_features)} features")

# Save all results to CSV
save_results(all_results, output_dir / "site_b_features.csv")
print(f"Results saved to {output_dir / 'site_b_features.csv'}")
```

### Step 5: Combine Results from Both Sites

After both sites complete processing, merge the results:

```python
import pandas as pd

# Load results from both sites
site_a_df = pd.read_csv("site_a_results/site_a_features.csv")
site_b_df = pd.read_csv("site_b_results/site_b_features.csv")

# Add site identifier
site_a_df["site"] = "A"
site_b_df["site"] = "B"

# Combine into a single dataset
combined_df = pd.concat([site_a_df, site_b_df], ignore_index=True)

# Verify consistent feature extraction
print(f"Total patients: {len(combined_df)}")
print(f"Features per patient: {len(combined_df.columns) - 2}")  # Exclude id and site
print(f"Site A: {len(site_a_df)} patients")
print(f"Site B: {len(site_b_df)} patients")

# Save combined dataset
combined_df.to_csv("combined_study_features.csv", index=False)
```

### Step 6: Archive Configuration with Study Data

For long-term reproducibility, archive the configuration alongside results:

```python
from datetime import datetime
import shutil

# Create study archive
archive_dir = Path(f"study_archive_{datetime.now().strftime('%Y%m%d')}")
archive_dir.mkdir(exist_ok=True)

# Copy configuration
shutil.copy("configs/lung_nodule_study_v1.yaml", archive_dir / "configuration.yaml")

# Copy results
shutil.copy("combined_study_features.csv", archive_dir / "features.csv")

# Export pipeline logs
pipeline.save_log(archive_dir / "processing_log.json")

# Create a README with study metadata
readme = f"""
# Lung Nodule Radiomics Study
Date: {datetime.now().isoformat()}
Pictologics Version: 1.0.0
Configuration Schema: 1.0

## Configurations Used
- lung_nodule_fbs25: Primary analysis (25 HU bin width)
- lung_nodule_fbs50: Sensitivity analysis (50 HU bin width)

## Sites
- Site A: {len(site_a_df)} patients
- Site B: {len(site_b_df)} patients

## Files
- configuration.yaml: Pipeline configuration (shareable)
- features.csv: Extracted radiomic features
- processing_log.json: Execution audit log
"""

with open(archive_dir / "README.md", "w") as f:
    f.write(readme)

print(f"Study archive created: {archive_dir}/")
```

### Key Takeaways

This workflow demonstrates several important practices:

| Practice | Benefit |
|----------|---------|
| **Custom configurations** | Tailored to specific imaging protocol and clinical question |
| **YAML export** | Human-readable, version-controllable, shareable |
| **Validation on load** | Catches configuration issues early |
| **Consistent processing** | Both sites use identical preprocessing and extraction |
| **Audit logging** | Complete record of processing steps |
| **Archival** | Long-term reproducibility for publications |

!!! success "Reproducibility Achieved"
    By sharing the YAML configuration file, both sites process their data with **identical settings**, 
    ensuring that any differences in extracted features reflect true biological variation rather than 
    methodological inconsistencies.

## Quick Reference

| Task | Method |
|------|--------|
| Run standard config | `pipeline.run(image, mask, config_names=["standard_fbn_32"])` |
| Run all standard configs | `pipeline.run(image, mask, config_names=["all_standard"])` |
| List available configs | `pipeline.list_configs()` |
| Get config details | `pipeline.get_config("config_name")` |
| Add custom config | `pipeline.add_config("name", steps)` |
| Remove config | `pipeline.remove_config("name")` |
| Export to YAML | `pipeline.save_configs("file.yaml")` |
| Export to JSON | `pipeline.save_configs("file.json")` |
| Import from file | `RadiomicsPipeline.load_configs("file.yaml")` |
| Import with standard configs | `RadiomicsPipeline.load_configs("file.yaml", load_standard=True)` |
| Merge configs | `pipeline.merge_configs(other_pipeline)` |
| Export to string | `pipeline.to_yaml()` / `pipeline.to_json()` |
| Import from string | `RadiomicsPipeline.from_yaml(s)` / `RadiomicsPipeline.from_json(s)` |

!!! tip "See Also"
    - [Pipeline & Preprocessing](pipeline.md) – Complete pipeline documentation
    - [Image Filtering](image_filtering.md) – Adding filters to configurations
    - [Cookbook](cookbook.md) – End-to-end batch processing examples
