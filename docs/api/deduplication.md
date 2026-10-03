# Deduplication API

When configurations share their preprocessing, the deduplication module finds the feature families that get the same input in more than one configuration. The pipeline computes each such family one time and copies the values.

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

For a complete example, see [Many Discretisations](../user_guide/cookbook.md#6-many-discretisations) in the Cookbook, and [Shared Work Between Configurations](../user_guide/pipeline.md#shared-work-between-configurations).

---

## How Results Are Handled

When deduplication reuses features from a previous configuration, the features are **copied** to the reusing configuration's results—they are never empty or missing.

### Result Behavior

| Scenario | Behavior |
| :--- | :--- |
| **deduplicate=True** (default) | Features computed once, then **copied** to all configs with the same feature family and matching preprocessing. All configs receive complete feature sets. |
| **deduplicate=False** | Features computed independently for each config. Same results, but slower. |

### Example: Results Structure

```python
results = pipeline.run(image, mask, config_names=["fbn_8", "fbn_16", "fbn_32"])

# All configs have IDENTICAL morphology values (computed once, copied to others)
assert results["fbn_8"]["volume_RNU0"] == results["fbn_16"]["volume_RNU0"]
assert results["fbn_8"]["volume_RNU0"] == results["fbn_32"]["volume_RNU0"]

# Texture features DIFFER (depend on discretization)
assert results["fbn_8"]["joint_average_60VM"] != results["fbn_32"]["joint_average_60VM"]
```

### Data Tables and Concatenation

Deduplication never leaves a feature out: each configuration gets all features of its families, computed or copied. When the configurations ask for the same families, a table with one row for each configuration has the same columns in every row:

```python
import pandas as pd

rows = [{"config": name, **results[name].to_dict()} for name in results]
table = pd.DataFrame(rows)  # one row for each configuration
```

When the configurations ask for other families (for example an `orig` configuration with the morphology and intensity features, and others with the texture features), each row has the columns of its own families, and the other cells are empty (`NaN`). A feature can also be `NaN` for its own reason, for example a PCA of too few voxels.

---

## DeduplicationRules

::: pictologics.deduplication.DeduplicationRules
    options:
      show_source: true
      members:
        - version
        - family_dependencies
        - family_options
        - get_version
        - to_dict
        - from_dict

---

## PreprocessingSignature

::: pictologics.deduplication.PreprocessingSignature
    options:
      show_source: true
      members:
        - from_steps
        - hash
        - json_repr

---

## ConfigurationAnalyzer

::: pictologics.deduplication.ConfigurationAnalyzer
    options:
      show_source: true
      members:
        - __init__
        - analyze

---

## DeduplicationPlan

::: pictologics.deduplication.DeduplicationPlan
    options:
      show_source: true
      members:
        - should_compute
        - get_source
        - is_stale
        - get_summary
        - to_dict
        - from_dict

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
| `"1.1.0"` | A family signature holds only the `extract_features` options that the family reads (the default) |

### Helper Functions

::: pictologics.deduplication.get_default_rules
    options:
      show_source: true

---

## Feature Family Dependencies

A signature holds the `source_mode` and `sentinel_value`, every preprocessing step before `extract_features` (with its parameters, in order), and the `extract_features` options that the family reads (rules 1.0.0: all options other than `families`). The rules also decide which families can ignore a `discretise` step at the end of the preprocessing: a family whose rules do not list `discretise` leaves that step out.

| Feature Family | Rules list `discretise` | Options in the signature (1.1.0) |
| :--- | :--- | :--- |
| `morphology` | No | None |
| `intensity` | No | `include_spatial_intensity`, `include_local_intensity`, `spatial_intensity_params`, `local_intensity_params` |
| `spatial_intensity` | No | `spatial_intensity_params` |
| `local_intensity` | No | `local_intensity_params` |
| `histogram` | Yes | None |
| `ivh` | Yes, unless `ivh_use_continuous=True` is set at the top level of the `extract_features` params | `ivh_params`, `ivh_use_continuous`, `ivh_discretisation` |
| `texture` (all subfamilies) | Yes | `texture_matrix_params` |

!!! warning "Filters Affect Intensity and Morphology Features"
    When using image filters (LoG, Gabor, Wavelets, Laws, etc.), intensity features are computed
    from the **filtered response map**, not the original image. The intensity-weighted morphology
    features (`integrated_intensity_99N0` and `center_of_mass_shift_KLMA`) use the response map
    too. Configurations with different filters therefore never share these families.

When two configurations have the same signature for a feature family, that family is computed once and the result is reused. Cache reuse is scoped by both feature family and signature, so families such as `texture`, `histogram`, and `ivh` never reuse each other's cached values even when their signatures are identical. A configuration with more than one `extract_features` step is always computed on its own.

---

## Integration with RadiomicsPipeline

The `RadiomicsPipeline` class integrates deduplication through these parameters:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `deduplicate` | `bool` | `True` | Enable/disable deduplication |
| `deduplication_rules` | `str`, `DeduplicationRules`, or `None` | `None` | Rules version for reproducibility (`None` resolves to the current default rules, `"1.1.0"`) |

These settings are preserved during serialization (`to_dict()`, `save_configs()`, etc.) and restored during deserialization.

```python
# Access pipeline deduplication settings
pipeline = RadiomicsPipeline(deduplicate=True)

print(pipeline.deduplication_enabled)    # True
print(pipeline.deduplication_stats)      # Statistics after run()
```
