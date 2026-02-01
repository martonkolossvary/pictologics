# Deduplication API

The deduplication module provides intelligent optimization for multi-configuration radiomic feature extraction. When multiple configurations share preprocessing steps but differ only in discretization, the system avoids redundant computation by identifying which feature families can be computed once and reused.

## Overview

The deduplication system consists of four main components:

1. **`DeduplicationRules`**: Defines which preprocessing steps affect which feature families
2. **`PreprocessingSignature`**: Creates hashable representations of preprocessing states
3. **`ConfigurationAnalyzer`**: Analyzes pipeline configurations to identify optimization opportunities
4. **`DeduplicationPlan`**: Generates optimized execution plans

## Quick Start

!!! info "Enabled by Default"
    Deduplication is **enabled by default** (`deduplicate=True`). You don't need to explicitly enable it—just create a pipeline and run multiple configurations.

```python
from pictologics import RadiomicsPipeline

# Deduplication is enabled by default!
pipeline = RadiomicsPipeline()  # deduplicate=True is the default

# Add multiple configurations with shared preprocessing
# ... (morphology/intensity computed once, reused across configs)

results = pipeline.run(image, mask, config_names=["config1", "config2", "config3"])

# Check performance statistics
print(pipeline.deduplication_stats)
```

For complete usage examples, see [Case 7: Multi-configuration batch with deduplication](../user_guide/case_examples.md#case-7-multi-configuration-batch-with-deduplication).

---

## How Results Are Handled

When deduplication reuses features from a previous configuration, the features are **copied** to the reusing configuration's results—they are never empty or missing.

### Result Behavior

| Scenario | Behavior |
| :--- | :--- |
| **deduplicate=True** (default) | Features computed once, then **copied** to all configs with matching preprocessing. All configs receive complete feature sets. |
| **deduplicate=False** | Features computed independently for each config. Same results, but slower. |

### Example: Results Structure

```python
results = pipeline.run(image, mask, config_names=["fbn_8", "fbn_16", "fbn_32"])

# All configs have IDENTICAL morphology values (computed once, copied to others)
assert results["fbn_8"]["volume_mesh_ml_HTUR"] == results["fbn_16"]["volume_mesh_ml_HTUR"]
assert results["fbn_8"]["volume_mesh_ml_HTUR"] == results["fbn_32"]["volume_mesh_ml_HTUR"]

# Texture features DIFFER (depend on discretization)
assert results["fbn_8"]["glcm_joint_avg_d1_HTUR"] != results["fbn_32"]["glcm_joint_avg_d1_HTUR"]
```

### Data Tables and Concatenation

When you concatenate results into a single DataFrame (e.g., for machine learning), every configuration row is **complete**—no missing values due to deduplication:

```python
import pandas as pd
from pictologics import format_results

# Format results for each config
rows = []
for config_name, features in results.items():
    row = format_results({config_name: features}, fmt="wide", meta={"config": config_name})
    rows.append(row)

# Concatenate into single DataFrame - NO missing values!
df = pd.concat(rows, ignore_index=True)
print(df.shape)  # (3, N) - all rows complete
print(df.isna().sum().sum())  # 0 - no NaN values
```

---

## DeduplicationRules

::: pictologics.deduplication.DeduplicationRules
    options:
      show_source: true
      members:
        - version
        - family_dependencies
        - validate

---

## PreprocessingSignature

::: pictologics.deduplication.PreprocessingSignature
    options:
      show_source: true
      members:
        - from_steps
        - hash_value
        - json_repr

---

## ConfigurationAnalyzer

::: pictologics.deduplication.ConfigurationAnalyzer
    options:
      show_source: true
      members:
        - __init__
        - analyze
        - get_preprocessing_steps
        - get_feature_families

---

## DeduplicationPlan

::: pictologics.deduplication.DeduplicationPlan
    options:
      show_source: true
      members:
        - create
        - get_plan
        - summary

---

## Rules Registry

The `RULES_REGISTRY` provides versioned deduplication rules for reproducibility:

::: pictologics.deduplication.RULES_REGISTRY
    options:
      show_source: false

### Available Versions

| Version | Description |
| :--- | :--- |
| `"1.0.0"` | Initial rules defining feature family dependencies |

### Helper Functions

::: pictologics.deduplication.get_default_rules
    options:
      show_source: true

---

## Feature Family Dependencies

The deduplication system understands which preprocessing steps affect which feature families:

| Feature Family | Relevant Preprocessing Steps |
| :--- | :--- |
| `morphology` | `resample`, `binarize_mask`, `keep_largest_component` |
| `intensity` | `resample`, `resegment`, `filter_outliers`, `filter` |
| `spatial_intensity` | Same as `intensity` |
| `local_intensity` | Same as `intensity` |
| `histogram` | `resample`, `resegment`, `filter_outliers`, `filter`, `binarize_mask`, `keep_largest_component`, `discretise` |
| `ivh` | Same as `histogram` (unless `ivh_use_continuous=True`, which removes `discretise` dependency) |
| `texture` (all subfamilies) | Same as `histogram` |

!!! warning "Filters Affect Intensity Features"
    When using image filters (LoG, Gabor, Wavelets, Laws, etc.), intensity features are computed 
    from the **filtered response map**, not the original image. Therefore, different filter 
    configurations will produce different intensity features and cannot be deduplicated.
    
    **Morphology features are not affected by filters** since they are computed from the mask 
    geometry, not intensity values. This means morphology can be computed once and reused 
    across all filter configurations.

When two configurations share identical values for the relevant preprocessing steps of a feature family, that family is computed once and the result is reused.

---

## Integration with RadiomicsPipeline

The `RadiomicsPipeline` class integrates deduplication through these parameters:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `deduplicate` | `bool` | `True` | Enable/disable deduplication |
| `deduplication_rules` | `str` or `DeduplicationRules` | `"1.0.0"` | Rules version for reproducibility |

These settings are preserved during serialization (`to_dict()`, `save_configs()`, etc.) and restored during deserialization.

```python
# Access pipeline deduplication settings
pipeline = RadiomicsPipeline(deduplicate=True)

print(pipeline.deduplicate)              # True
print(pipeline.deduplication_stats)      # Statistics after run()
```
