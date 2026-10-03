# Feature Names

`pictologics.features.FEATURE_NAMES` lists the feature keys of each feature family, in the order of the results. The pipeline uses it to give each configuration all its feature names, also when a step fails (see [Result Guarantees](../../user_guide/pipeline.md#result-guarantees)).

```python
from pictologics.features import FEATURE_NAMES, GLCM_FEATURE_NAMES

print(list(FEATURE_NAMES))        # intensity, histogram, ivh, spatial_intensity, ...
print(len(GLCM_FEATURE_NAMES))    # 25
print(GLCM_FEATURE_NAMES[:2])     # ('joint_maximum_GYBY', 'joint_average_60VM')
```

| Family | Tuple | Features |
|:--|:--|:--|
| `intensity` | `INTENSITY_FEATURE_NAMES` | 18 |
| `histogram` | `HISTOGRAM_FEATURE_NAMES` | 23 |
| `ivh` | `IVH_FEATURE_NAMES` | 7 |
| `spatial_intensity` | `SPATIAL_INTENSITY_FEATURE_NAMES` | 2 |
| `local_intensity` | `LOCAL_INTENSITY_FEATURE_NAMES` | 2 |
| `morphology` | `MORPHOLOGY_FEATURE_NAMES` | 27 |
| `glcm` | `GLCM_FEATURE_NAMES` | 25 |
| `glrlm` | `GLRLM_FEATURE_NAMES` | 16 |
| `glszm` | `GLSZM_FEATURE_NAMES` | 16 |
| `gldzm` | `GLDZM_FEATURE_NAMES` | 16 |
| `ngtdm` | `NGTDM_FEATURE_NAMES` | 5 |
| `ngldm` | `NGLDM_FEATURE_NAMES` | 17 |

A feature key is the feature name and its IBSI code, for example `joint_maximum_GYBY`. A number after the code tells the variant of the feature, for example `_10` in `volume_at_intensity_fraction_0.10_BC2M_10`. `RadiomicsPipeline.describe_features()` gives the keys of a pipeline with their families and preprocessing.
