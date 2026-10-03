# Preprocessing API

The functions of the preprocessing steps, for use outside the pipeline. The pipeline steps call them (see [Pipeline Steps](../user_guide/pipeline_steps.md)).

## Image Resampling

::: pictologics.preprocessing.resample_image

## Discretisation

::: pictologics.preprocessing.discretise_image

## Mask Operations

::: pictologics.preprocessing.apply_mask

::: pictologics.preprocessing.resegment_mask

::: pictologics.preprocessing.keep_largest_component

::: pictologics.preprocessing.grow_mask

::: pictologics.preprocessing.extract_roi

## Intensities

::: pictologics.preprocessing.normalise_image

::: pictologics.preprocessing.round_intensities

## Outlier Filtering

::: pictologics.preprocessing.filter_outliers

## Sentinel Value Handling

Utilities for detecting and masking sentinel values (e.g., -2048 HU for outside-FOV regions in CT).

::: pictologics.preprocessing.detect_sentinel_value

::: pictologics.preprocessing.create_source_mask_from_sentinel
