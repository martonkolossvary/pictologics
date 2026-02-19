"""
Radiomics Pipeline Module
=========================

This module provides a flexible, configurable pipeline for executing radiomic feature
extraction workflows. It allows users to define sequences of preprocessing steps
and feature extraction tasks.

Key Features:
-------------
- **Configurable Workflows**: Define steps like resampling, resegmentation, filtering,
  discretisation, and feature extraction in a declarative manner.
- **State Management**: Tracks the state of the image and masks (morphological and intensity)
  throughout the pipeline.
- **Logging**: Records execution details, parameters, and errors for reproducibility.
- **Batch Processing**: Can process multiple configurations on the same input data.
"""

from __future__ import annotations

import copy
import datetime
import json
import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import yaml
from numpy import typing as npt

from .deduplication import (
    ConfigurationAnalyzer,
    DeduplicationPlan,
    DeduplicationRules,
    get_default_rules,
)
from .features.intensity import (
    calculate_intensity_features,
    calculate_intensity_histogram_features,
    calculate_ivh_features,
    calculate_local_intensity_features,
    calculate_spatial_intensity_features,
)
from .features.morphology import calculate_morphology_features
from .features.texture import (
    calculate_all_texture_matrices,
    calculate_glcm_features,
    calculate_gldzm_features,
    calculate_glrlm_features,
    calculate_glszm_features,
    calculate_ngldm_features,
    calculate_ngtdm_features,
)
from .filters import (
    BoundaryCondition,
    gabor_filter,
    laplacian_of_gaussian,
    laws_filter,
    mean_filter,
    riesz_log,
    riesz_simoncelli,
    riesz_transform,
    simoncelli_wavelet,
    wavelet_transform,
)
from .loader import Image, create_full_mask, load_image
from .preprocessing import (
    apply_mask,
    create_source_mask_from_sentinel,
    detect_sentinel_value,
    discretise_image,
    filter_outliers,
    keep_largest_component,
    resample_image,
    resegment_mask,
    round_intensities,
)
from .templates import get_standard_templates

# Schema version for config serialization - increment when format changes
CONFIG_SCHEMA_VERSION = "1.0"

# Valid schema versions (for backward compatibility)
_VALID_SCHEMA_VERSIONS = {"1.0", "1.1"}


class SourceMode(Enum):
    """
    Determines how voxels outside the ROI mask are treated during spatial operations.

    This setting affects resampling, filtering, and other operations that use
    neighboring voxels for interpolation or convolution.

    Attributes:
        FULL_IMAGE: All voxels contain valid data. Use surrounding voxels for
                    interpolation during resampling and filtering. This is the
                    traditional behavior when a full CT/MR scan is provided.
        ROI_ONLY: Only ROI mask voxels contain valid data. Voxels outside the
                  mask contain sentinel values (-2048, etc.) that must be excluded
                  from all spatial operations.
        AUTO: Automatically detect common sentinel values (-2048, -1024, etc.).
              If detected, behave like ROI_ONLY. Otherwise, behave like FULL_IMAGE.
              Emits a warning when sentinel values are detected.
    """

    FULL_IMAGE = "full_image"
    ROI_ONLY = "roi_only"
    AUTO = "auto"
    """
    Automatically detect sentinel values (e.g., -2048) and exclude them.
    This mode ensures that background voxels are not included in the ROI after
    resampling, even if their intensity (e.g., 0) is within the valid range.
    """


@dataclass
class PipelineState:
    """
    Holds the current state of the image and masks during pipeline execution.

    Attributes:
        image: Current image (may be discretised after discretise step).
        raw_image: Always the non-discretised image (for intensity/morphology).
        morph_mask: Morphological mask for shape-based features.
        intensity_mask: Intensity mask for intensity-based features.
        is_discretised: Whether the image has been discretised.
        n_bins: Number of bins if discretised with FBN.
        bin_width: Bin width if discretised with FBS.
        discretisation_method: Discretisation method used ('FBN' or 'FBS').
        discretisation_min: Minimum value used for discretisation.
        discretisation_max: Maximum value used for discretisation.
        mask_was_generated: Whether the mask was auto-generated (no mask provided).
        is_filtered: Whether a filter has been applied.
        filter_type: Type of filter applied (if any).
        source_mode: How source voxel validity is handled.
        source_mask: Computed validity mask (where real data exists).
        sentinel_detected: True if AUTO mode detected sentinel values.
        sentinel_value: The detected sentinel value (if any).
    """

    image: Image  # May be discretised after discretise step
    raw_image: Image  # Always the non-discretised image (for intensity/morphology)
    morph_mask: Image
    intensity_mask: Image
    is_discretised: bool = False
    n_bins: Optional[int] = None
    bin_width: Optional[float] = None
    discretisation_method: Optional[str] = None
    discretisation_min: Optional[float] = None
    discretisation_max: Optional[float] = None
    mask_was_generated: bool = False
    is_filtered: bool = False
    filter_type: Optional[str] = None
    # Source tracking for sentinel value handling
    source_mode: SourceMode = SourceMode.FULL_IMAGE
    source_mask: Optional[Image] = None
    sentinel_detected: bool = False
    sentinel_value: Optional[float] = None


class EmptyROIMaskError(ValueError):
    """Raised when preprocessing yields an empty ROI mask."""


class RadiomicsPipeline:
    """
    A flexible, configurable pipeline for radiomic feature extraction.
    Allows defining multiple processing configurations (sequences of steps) to be run on data.

    Args:
        deduplicate: Whether to enable feature deduplication across configurations.
            When True (default), features that would be identical due to shared
            preprocessing are computed once and reused.
        deduplication_rules: Specific DeduplicationRules to use, or a version
            string to look up from the registry. If None, uses current default.
    """

    def __init__(
        self,
        deduplicate: bool = True,
        deduplication_rules: DeduplicationRules | str | None = None,
        load_standard: bool = True,
    ) -> None:
        """Initialize pipeline with empty config registry.

        Args:
            deduplicate: Whether to enable feature deduplication across configurations.
            deduplication_rules: Specific DeduplicationRules to use, or a version
                string to look up from the registry. If None, uses current default.
            load_standard: Whether to load standard predefined configurations
                (e.g., ``standard_fbn_32``, ``standard_fbs_16``). Defaults to True
                for direct instantiation. Set to False when loading configurations
                from files or strings to avoid mixing standard configs with
                user-defined ones.
        """
        self._configs: dict[str, list[dict[str, Any]]] = {}
        self._config_metadata: dict[str, dict[str, Any]] = (
            {}
        )  # Stores source_mode, etc.
        self._log: list[dict[str, Any]] = []

        # Deduplication settings
        self._deduplication_enabled = deduplicate

        if deduplication_rules is None:
            self._deduplication_rules = get_default_rules()
        elif isinstance(deduplication_rules, str):
            self._deduplication_rules = DeduplicationRules.get_version(
                deduplication_rules
            )
        else:
            self._deduplication_rules = deduplication_rules

        self._last_deduplication_plan: DeduplicationPlan | None = None
        self._configs_modified_since_plan: bool = False

        # Deduplication statistics (reset on each run)
        self._dedup_reused_count: int = 0
        self._dedup_computed_count: int = 0

        if load_standard:
            self._load_predefined_configs()

    def _load_predefined_configs(self) -> None:
        """
        Load predefined, commonly used pipeline configurations from templates.
        """
        try:
            standard_configs = get_standard_templates()
            for name, steps in standard_configs.items():
                # Convert YAML lists to tuples where needed (e.g., new_spacing)
                converted_steps = self._convert_yaml_steps(steps)
                self._configs[name] = converted_steps
        except Exception as e:
            warnings.warn(
                f"Failed to load standard templates: {e}",
                UserWarning,
                stacklevel=2,
            )
            # Fallback to empty configs - user can add their own

    def _convert_yaml_steps(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert YAML-loaded steps to internal format.

        YAML loads lists, but some parameters expect tuples (e.g., new_spacing).
        """
        converted = []
        for step in steps:
            new_step = {"step": step["step"]}
            if "params" in step:
                params = copy.deepcopy(step["params"])
                # Convert new_spacing list to tuple
                if "new_spacing" in params and isinstance(params["new_spacing"], list):
                    params["new_spacing"] = tuple(params["new_spacing"])
                new_step["params"] = params
            converted.append(new_step)
        return converted

    def get_all_standard_config_names(self) -> list[str]:
        """
        Returns the list of all standard configuration names.

        Returns names from loaded templates that start with 'standard_'.
        """
        return sorted(
            [name for name in self._configs.keys() if name.startswith("standard_")]
        )

    # -------------------------------------------------------------------------
    # Deduplication Properties
    # -------------------------------------------------------------------------

    @property
    def deduplication_enabled(self) -> bool:
        """Whether feature deduplication is enabled."""
        return self._deduplication_enabled

    @deduplication_enabled.setter
    def deduplication_enabled(self, value: bool) -> None:
        """Enable or disable feature deduplication."""
        self._deduplication_enabled = value

    @property
    def deduplication_rules(self) -> DeduplicationRules:
        """Current deduplication rules."""
        return self._deduplication_rules

    @deduplication_rules.setter
    def deduplication_rules(self, value: DeduplicationRules | str) -> None:
        """Set deduplication rules (by version string or DeduplicationRules)."""
        if isinstance(value, str):
            self._deduplication_rules = DeduplicationRules.get_version(value)
        else:
            self._deduplication_rules = value
        # Invalidate existing plan when rules change
        self._configs_modified_since_plan = True

    @property
    def last_deduplication_plan(self) -> DeduplicationPlan | None:
        """The last computed deduplication plan, if any."""
        return self._last_deduplication_plan

    @property
    def deduplication_stats(self) -> dict[str, int | float]:
        """
        Statistics from the last pipeline run with deduplication enabled.

        Returns a dictionary with:
            - 'reused_families': Number of feature families reused from cache
            - 'computed_families': Number of feature families freshly computed
            - 'cache_hit_rate': Fraction of families reused (0.0 to 1.0)

        Returns an empty dict if no features were extracted (with a warning),
        or if deduplication was not enabled during the last run.

        Note:
            Statistics are valid because pipeline configurations run sequentially.
            Parallelization occurs within Numba-accelerated functions, not across configs.
        """
        total = self._dedup_reused_count + self._dedup_computed_count
        if total == 0:
            warnings.warn(
                "No features were extracted with deduplication enabled. "
                "Ensure deduplication is enabled and run() has been called with multiple configs.",
                UserWarning,
                stacklevel=2,
            )
            return {}

        return {
            "reused_families": self._dedup_reused_count,
            "computed_families": self._dedup_computed_count,
            "cache_hit_rate": self._dedup_reused_count / total,
        }

    def add_config(
        self,
        name: str,
        steps: list[dict[str, Any]],
        source_mode: str = "full_image",
        sentinel_value: Optional[float] = None,
    ) -> "RadiomicsPipeline":
        """
        Add a processing configuration.

        Args:
            name: Unique name for this configuration.
            steps: List of steps. Each step is a dict with 'step' (name) and 'params' (dict).
                   Supported steps:
                   - 'resample': params: new_spacing (required), interpolation (optional)
                   - 'resegment': params: range_min, range_max
                   - 'filter_outliers': params: sigma
                   - 'binarize_mask': params: threshold (float, default 0.5),
                       mask_values (int | list[int] | tuple[int, int]), apply_to ('morph'|'intensity'|'both')
                   - 'keep_largest_component': params: None
                   - 'round_intensities': params: None
                   - 'discretise': params: method, n_bins/bin_width, etc.
                   - 'filter': params: type (required), plus filter-specific params
                   - 'extract_features': params: families (list), etc.
            source_mode: How to handle source voxel validity for resampling/filtering:
                - "full_image" (default): All voxels contain real data. Traditional behavior.
                - "roi_only": Only ROI voxels contain real data; others have sentinel values.
                - "auto": Auto-detect sentinel values; emit warning if found.
            sentinel_value: If specified, explicitly set the sentinel value instead
                of auto-detecting. Only used when source_mode is "roi_only" or "auto".

        Note:
            - Texture features require a prior 'discretise' step.
            - IVH features are configured via 'ivh_params' dict.
            - The source_mode setting affects resampling and filtering operations.
              In 'roi_only' mode, boundary regions use normalized interpolation to
              exclude sentinel voxels.

        Example:
            ```python
            pipeline = RadiomicsPipeline()

            # Standard configuration (all voxels valid)
            pipeline.add_config(
                name="standard",
                steps=[...],
            )

            # Configuration for sentinel-masked images
            pipeline.add_config(
                name="sentinel_aware",
                source_mode="roi_only",
                steps=[
                    {"step": "resample", "params": {"new_spacing": (1, 1, 1)}},
                    {"step": "extract_features", "params": {"families": ["all"]}},
                ],
            )
            ```
        """
        if not isinstance(steps, list):
            raise ValueError("Configuration must be a list of steps")

        # Validate source_mode
        valid_modes = {"full_image", "roi_only", "auto"}
        if source_mode not in valid_modes:
            raise ValueError(
                f"Invalid source_mode '{source_mode}'. Must be one of: {valid_modes}"
            )

        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("Each step must be a dictionary")
            if "step" not in step:
                raise ValueError("Each step must have a 'step' key")

        self._configs[name] = steps
        self._config_metadata[name] = {
            "source_mode": source_mode,
            "sentinel_value": sentinel_value,
        }
        self._configs_modified_since_plan = True
        return self

    def run(
        self,
        image: str | Image,
        mask: str | Image | None = None,
        subject_id: Optional[str] = None,
        config_names: Optional[list[str]] = None,
    ) -> dict[str, pd.Series]:
        """
        Run configurations on the provided image and mask.

        Args:
            image: Path to image or Image object.
            mask: Optional path to mask or Image object.
                If omitted (or passed as `None` / empty string), the pipeline will
                treat the **entire image** as the ROI by generating a full (all-ones)
                mask matching the input image geometry.
            subject_id: Optional identifier for the subject.
            config_names: List of specific configuration names to run.
                          If None, runs all registered configurations.
                          Supports "all_standard" to run all 6 standard configs.

        Returns:
            Dictionary mapping config names to pandas Series of features.

        Example:
            Run standard pipeline components:

            ```python
            from pictologics.pipeline import RadiomicsPipeline

            # Initialize
            pipeline = RadiomicsPipeline()

            # Run on image and mask
            results = pipeline.run(
                image="data/image.nii.gz",
                mask="data/mask.nii.gz",
                subject_id="subject_001",
                config_names=["standard_fbn_32"]
            )

            # Access results
            print(results["standard_fbn_32"].head())
            ```
        """
        # 1. Load Data
        if isinstance(image, str):
            orig_img = load_image(image)
            img_source = image
        else:
            orig_img = image
            img_source = "InMemory"

        mask_was_generated = False
        if mask is None or (isinstance(mask, str) and mask.strip() == ""):
            orig_mask = create_full_mask(orig_img)
            mask_source = "GeneratedFullMask"
            mask_was_generated = True
        elif isinstance(mask, str):
            orig_mask = load_image(mask)
            mask_source = mask
        else:
            orig_mask = mask
            mask_source = "InMemory"

        all_results = {}

        # Determine which configs to run
        if config_names is None:
            target_configs = list(self._configs.keys())
        else:
            target_configs = []
            for name in config_names:
                if name == "all_standard":
                    target_configs.extend(self.get_all_standard_config_names())
                elif name in self._configs:
                    target_configs.append(name)
                else:
                    raise ValueError(f"Configuration '{name}' not found.")

        # Create or regenerate deduplication plan if enabled
        dedup_plan: DeduplicationPlan | None = None
        family_cache: dict[str, dict[str, Any]] = {}  # sig_hash -> family_features

        # Reset deduplication statistics for this run
        self._dedup_reused_count = 0
        self._dedup_computed_count = 0

        if self._deduplication_enabled and len(target_configs) > 1:
            # Get configs for analysis
            configs_to_analyze = {name: self._configs[name] for name in target_configs}
            analyzer = ConfigurationAnalyzer(
                configs_to_analyze, self._deduplication_rules
            )
            dedup_plan = analyzer.analyze()
            self._last_deduplication_plan = dedup_plan
            self._configs_modified_since_plan = False

        # Run each configuration
        for config_name in target_configs:
            steps = self._configs[config_name]
            metadata = self._config_metadata.get(config_name, {})

            # Determine source mode for this config
            source_mode_str = metadata.get("source_mode", "full_image")
            source_mode = SourceMode(source_mode_str)
            explicit_sentinel = metadata.get("sentinel_value")

            # Determine source mask based on source_mode
            source_mask: Optional[Image] = None
            sentinel_detected = False
            detected_sentinel_value: Optional[float] = None

            if source_mode == SourceMode.FULL_IMAGE:
                # Default: all voxels valid, no source_mask needed
                pass

            elif source_mode == SourceMode.ROI_ONLY:
                # Use ROI mask as source mask
                source_mask = Image(
                    array=(orig_mask.array > 0).astype(np.uint8),
                    spacing=orig_mask.spacing,
                    origin=orig_mask.origin,
                    direction=orig_mask.direction,
                    modality="SOURCE_MASK",
                )

            elif source_mode == SourceMode.AUTO:
                # Auto-detect sentinel values
                if explicit_sentinel is not None:
                    # User provided explicit sentinel value
                    detected_sentinel_value = explicit_sentinel
                    sentinel_detected = True
                else:
                    # If mask was auto-generated (full mask), do not use it for
                    # "outside-ness" check in detection, as everything is "inside".
                    mask_for_detection = orig_mask if not mask_was_generated else None
                    detected = detect_sentinel_value(
                        orig_img, roi_mask=mask_for_detection
                    )
                    if detected is not None:
                        detected_sentinel_value = detected
                        sentinel_detected = True

                        # Log info instead of warning (user request)
                        # Changed to DEBUG level to avoid console spam in default logging configuration
                        logging.debug(
                            f"Auto-detected sentinel value {detected} in image. "
                            f"Using source validity mask for config '{config_name}'. "
                            f"Voxels with value {detected} will be excluded from "
                            f"resampling/filtering."
                        )

                if sentinel_detected and detected_sentinel_value is not None:
                    source_mask = create_source_mask_from_sentinel(
                        orig_img, detected_sentinel_value
                    )

            # Initialize State with source tracking
            # We start with fresh copies for each config
            state = PipelineState(
                image=orig_img,
                raw_image=orig_img,  # Track non-discretised image
                morph_mask=orig_mask,
                intensity_mask=Image(
                    array=orig_mask.array.copy(),
                    spacing=orig_mask.spacing,
                    origin=orig_mask.origin,
                    direction=orig_mask.direction,
                    modality=orig_mask.modality,
                ),
                mask_was_generated=mask_was_generated,
                source_mode=source_mode,
                source_mask=source_mask,
                sentinel_detected=sentinel_detected,
                sentinel_value=detected_sentinel_value,
            )

            self._ensure_nonempty_roi(state, context="initialization")

            config_log: dict[str, Any] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "subject_id": subject_id,
                "config_name": config_name,
                "image_source": img_source,
                "mask_source": mask_source,
                "source_mode": source_mode.value,
                "sentinel_detected": sentinel_detected,
                "sentinel_value": detected_sentinel_value,
                "steps_executed": [],
            }

            config_features = {}

            try:
                for step_def in steps:
                    step_name = step_def["step"]
                    params = step_def.get("params", {})

                    # Execute Step
                    if step_name == "extract_features":
                        # Use deduplication if plan exists
                        if dedup_plan is not None:
                            features = self._extract_features_with_dedup(
                                state, params, config_name, dedup_plan, family_cache
                            )
                        else:
                            features = self._extract_features(state, params)
                        config_features.update(features)
                    else:
                        self._execute_preprocessing_step(state, step_name, params)

                    # Log
                    config_log["steps_executed"].append(
                        {"step": step_name, "params": params, "status": "completed"}
                    )

            except Exception as e:
                config_log["error"] = str(e)
                config_log["failed_step"] = step_def

                # For empty ROI, fail fast (do not silently return empty/partial features).
                if isinstance(e, EmptyROIMaskError):
                    self._log.append(config_log)
                    raise

            self._log.append(config_log)

            # Create Series
            series = pd.Series(config_features)
            if subject_id:
                series["subject_id"] = subject_id
            all_results[config_name] = series

        return all_results

    def clear_log(self) -> None:
        """Clear the in-memory processing log."""
        self._log.clear()

    def _ensure_nonempty_roi(self, state: PipelineState, context: str) -> None:
        """Raise a clear error if the ROI is empty.

        The pipeline uses `mask_values=1` semantics throughout (see `apply_mask`).
        """
        has_intensity_roi = bool(np.any(state.intensity_mask.array == 1))
        if not has_intensity_roi:
            raise EmptyROIMaskError(
                "ROI is empty after preprocessing "
                f"({context}). Ensure your mask contains at least one voxel with value 1, "
                "or relax resegmentation/outlier filtering thresholds."
            )
        has_morph_roi = bool(np.any(state.morph_mask.array == 1))
        if not has_morph_roi:
            raise EmptyROIMaskError(
                "ROI is empty after preprocessing "
                f"({context}). Ensure your mask contains at least one voxel with value 1, "
                "or relax resegmentation/outlier filtering thresholds."
            )

    def _execute_preprocessing_step(
        self, state: PipelineState, step_name: str, params: dict[str, Any]
    ) -> None:
        """
        Execute a single preprocessing step and update the state in-place.
        """
        if step_name == "resample":
            # Params
            if "new_spacing" not in params:
                raise ValueError("Resample step requires 'new_spacing' parameter.")

            spacing = params["new_spacing"]
            interp_img = params.get("interpolation", "linear")
            interp_mask = params.get("mask_interpolation", "nearest")
            mask_thresh = params.get("mask_threshold", 0.5)
            round_intensities_flag = params.get("round_intensities", False)

            # Determine source_mask for resampling (if not FULL_IMAGE mode)
            source_mask_arg = None
            if (
                state.source_mode != SourceMode.FULL_IMAGE
                and state.source_mask is not None
            ):
                source_mask_arg = state.source_mask

            # Update Image and raw_image
            state.image = resample_image(
                state.image,
                spacing,
                interpolation=interp_img,
                round_intensities=round_intensities_flag,
                source_mask=source_mask_arg,
            )
            state.raw_image = (
                state.image
            )  # Keep raw_image in sync before discretisation

            # Propagate source_mask from resampled image if it was used
            if state.image.has_source_mask and state.image.source_mask is not None:
                state.source_mask = Image(
                    array=state.image.source_mask.astype(np.uint8),
                    spacing=state.image.spacing,
                    origin=state.image.origin,
                    direction=state.image.direction,
                    modality="SOURCE_MASK",
                )

            # Update Masks
            thresh_arg = mask_thresh if interp_mask != "nearest" else None
            state.morph_mask = resample_image(
                state.morph_mask,
                spacing,
                interpolation=interp_mask,
                mask_threshold=thresh_arg,
            )
            state.intensity_mask = resample_image(
                state.intensity_mask,
                spacing,
                interpolation=interp_mask,
                mask_threshold=thresh_arg,
            )

            # CRITICAL: If valid source mask exists, apply it to intensity mask.
            # This prevents background (often 0 after resampling) from being
            # considered part of the ROI if the resegmentation range includes 0.
            if state.source_mask is not None:
                # Ensure binary mask semantics
                valid_mask = state.source_mask.array > 0
                state.intensity_mask = Image(
                    array=(state.intensity_mask.array * valid_mask).astype(np.uint8),
                    spacing=state.intensity_mask.spacing,
                    origin=state.intensity_mask.origin,
                    direction=state.intensity_mask.direction,
                    modality=state.intensity_mask.modality,
                )

            self._ensure_nonempty_roi(state, context="resample")

        elif step_name == "resegment":
            range_min = params.get("range_min")
            range_max = params.get("range_max")
            state.intensity_mask = resegment_mask(
                state.image, state.intensity_mask, range_min, range_max
            )

            # If the mask was auto-generated (mask omitted), treat resegmentation as ROI definition
            # for both intensity and morphology features.
            if state.mask_was_generated:
                state.morph_mask = resegment_mask(
                    state.image, state.morph_mask, range_min, range_max
                )

            self._ensure_nonempty_roi(state, context="resegment")

        elif step_name == "filter_outliers":
            sigma = params.get("sigma", 3.0)
            state.intensity_mask = filter_outliers(
                state.image, state.intensity_mask, sigma
            )

            if state.mask_was_generated:
                state.morph_mask = filter_outliers(state.image, state.morph_mask, sigma)

            self._ensure_nonempty_roi(state, context="filter_outliers")

        elif step_name == "round_intensities":
            state.image = round_intensities(state.image)
            state.raw_image = (
                state.image
            )  # Keep raw_image in sync before discretisation

        elif step_name == "keep_largest_component":
            # apply_to: "morph", "intensity", or "both" (default)
            apply_to = params.get("apply_to", "both")
            if apply_to in ("morph", "both"):
                state.morph_mask = keep_largest_component(state.morph_mask)
            if apply_to in ("intensity", "both"):
                state.intensity_mask = keep_largest_component(state.intensity_mask)

            self._ensure_nonempty_roi(state, context="keep_largest_component")

        elif step_name == "binarize_mask":
            apply_to = params.get("apply_to", "both")
            threshold = params.get("threshold", 0.5)
            mask_values = params.get("mask_values")

            def _binarize(image: Image) -> Image:
                if mask_values is not None:
                    if isinstance(mask_values, tuple) and len(mask_values) == 2:
                        lo, hi = mask_values
                        mask_arr = (image.array >= lo) & (image.array <= hi)
                    else:
                        values = mask_values
                        if isinstance(values, int):
                            values = [values]
                        mask_arr = np.isin(image.array, values)
                else:
                    if threshold is None:
                        raise ValueError(
                            "binarize_mask requires 'threshold' unless mask_values is provided"
                        )
                    mask_arr = image.array >= float(threshold)

                return Image(
                    array=mask_arr.astype(np.uint8),
                    spacing=image.spacing,
                    origin=image.origin,
                    direction=image.direction,
                    modality=image.modality,
                )

            if apply_to in ("morph", "both"):
                state.morph_mask = _binarize(state.morph_mask)
            if apply_to in ("intensity", "both"):
                state.intensity_mask = _binarize(state.intensity_mask)

            self._ensure_nonempty_roi(state, context="binarize_mask")

        elif step_name == "discretise":
            self._ensure_nonempty_roi(state, context="discretise")
            method = params.get("method", "FBN")

            # Avoid passing 'method' twice
            disc_params = params.copy()
            if "method" in disc_params:
                del disc_params["method"]

            state.image = cast(
                Image,
                discretise_image(
                    state.image,
                    method=method,
                    roi_mask=state.intensity_mask,
                    **disc_params,
                ),
            )

            state.is_discretised = True
            state.discretisation_method = method
            state.n_bins = params.get("n_bins")
            state.bin_width = params.get("bin_width")

            # If FBS, n_bins is dynamic. We can estimate it from the result.
            if method == "FBS":
                masked_vals = apply_mask(state.image, state.intensity_mask)
                if len(masked_vals) > 0:
                    state.n_bins = int(np.max(masked_vals))
                else:
                    raise EmptyROIMaskError(
                        "ROI is empty after preprocessing (discretise). "
                        "Cannot infer FBS bin count from an empty ROI."
                    )

        elif step_name == "filter":
            # Apply image filter
            filter_type = params.get("type")
            if not filter_type:
                raise ValueError("Filter step requires 'type' parameter.")

            # Get boundary condition (default: mirror per IBSI 2)
            boundary_str = params.get("boundary", "mirror")
            boundary_map = {
                "mirror": BoundaryCondition.MIRROR,
                "nearest": BoundaryCondition.NEAREST,
                "constant": BoundaryCondition.ZERO,
                "wrap": BoundaryCondition.PERIODIC,
                "zero": BoundaryCondition.ZERO,
                "periodic": BoundaryCondition.PERIODIC,
            }
            boundary = boundary_map.get(boundary_str, BoundaryCondition.MIRROR)

            # Extract filter-specific params (exclude type and boundary)
            filter_params = {
                k: v for k, v in params.items() if k not in ("type", "boundary")
            }

            # Apply filter based on type
            filtered_array: npt.NDArray[np.floating[Any]]

            if filter_type == "mean":
                filter_params["boundary"] = boundary
                # Pass source_mask if not in FULL_IMAGE mode
                if (
                    state.source_mode != SourceMode.FULL_IMAGE
                    and state.source_mask is not None
                ):
                    source_arr = state.source_mask.array > 0
                    result_tuple = mean_filter(
                        state.image.array, source_mask=source_arr, **filter_params
                    )
                    # mean_filter returns tuple[NDArray, NDArray] when source_mask is used
                    if isinstance(result_tuple, tuple):
                        filtered_array = result_tuple[0]
                    else:
                        filtered_array = result_tuple
                else:
                    filtered_array = mean_filter(state.image.array, **filter_params)

            elif filter_type == "log":
                filter_params["boundary"] = boundary
                # Use image spacing if not provided
                if "spacing_mm" not in filter_params:
                    filter_params["spacing_mm"] = state.image.spacing
                # Pass source_mask if not in FULL_IMAGE mode
                if (
                    state.source_mode != SourceMode.FULL_IMAGE
                    and state.source_mask is not None
                ):
                    source_arr = state.source_mask.array > 0
                    result_tuple_log = laplacian_of_gaussian(
                        state.image.array, source_mask=source_arr, **filter_params
                    )
                    if isinstance(result_tuple_log, tuple):
                        filtered_array = result_tuple_log[0]
                    else:
                        filtered_array = result_tuple_log
                else:
                    filtered_array = laplacian_of_gaussian(
                        state.image.array, **filter_params
                    )

            elif filter_type == "laws":
                filter_params["boundary"] = boundary
                # 'kernel' param maps to first positional arg
                kernel = filter_params.pop("kernel", "L5E5E5")
                # Pass source_mask if not in FULL_IMAGE mode
                if (
                    state.source_mode != SourceMode.FULL_IMAGE
                    and state.source_mask is not None
                ):
                    source_arr = state.source_mask.array > 0
                    result_laws = laws_filter(
                        state.image.array,
                        kernel,
                        source_mask=source_arr,
                        **filter_params,
                    )
                    if isinstance(result_laws, tuple):
                        filtered_array = result_laws[0]
                    else:
                        filtered_array = result_laws
                else:
                    filtered_array = laws_filter(
                        state.image.array, kernel, **filter_params
                    )

            elif filter_type == "gabor":
                filter_params["boundary"] = boundary
                if "spacing_mm" not in filter_params:
                    filter_params["spacing_mm"] = state.image.spacing
                # Pass source_mask if not in FULL_IMAGE mode
                if (
                    state.source_mode != SourceMode.FULL_IMAGE
                    and state.source_mask is not None
                ):
                    source_arr = state.source_mask.array > 0
                    filter_params["source_mask"] = source_arr
                filtered_array = gabor_filter(state.image.array, **filter_params)

            elif filter_type == "wavelet":
                filter_params["boundary"] = boundary
                # Pass source_mask if not in FULL_IMAGE mode
                if (
                    state.source_mode != SourceMode.FULL_IMAGE
                    and state.source_mask is not None
                ):
                    source_arr = state.source_mask.array > 0
                    filter_params["source_mask"] = source_arr
                filtered_array = wavelet_transform(state.image.array, **filter_params)

            elif filter_type == "simoncelli":
                # Simoncelli doesn't use boundary param
                # Pass source_mask if not in FULL_IMAGE mode
                if (
                    state.source_mode != SourceMode.FULL_IMAGE
                    and state.source_mask is not None
                ):
                    source_arr = state.source_mask.array > 0
                    filter_params["source_mask"] = source_arr
                filtered_array = simoncelli_wavelet(state.image.array, **filter_params)

            elif filter_type == "riesz":
                # Riesz transform variants
                variant = filter_params.pop("variant", "base")
                # Pass source_mask if not in FULL_IMAGE mode
                if (
                    state.source_mode != SourceMode.FULL_IMAGE
                    and state.source_mask is not None
                ):
                    source_arr = state.source_mask.array > 0
                    filter_params["source_mask"] = source_arr
                if variant == "log":
                    if "spacing_mm" not in filter_params:
                        filter_params["spacing_mm"] = state.image.spacing
                    filtered_array = riesz_log(state.image.array, **filter_params)
                elif variant == "simoncelli":
                    filtered_array = riesz_simoncelli(
                        state.image.array, **filter_params
                    )
                else:
                    filtered_array = riesz_transform(state.image.array, **filter_params)

            else:
                raise ValueError(
                    f"Unknown filter type: {filter_type}. "
                    "Supported: mean, log, laws, gabor, wavelet, simoncelli, riesz"
                )

            # Update state with filtered image
            state.image = Image(
                array=filtered_array,
                spacing=state.image.spacing,
                origin=state.image.origin,
                direction=state.image.direction,
                modality=state.image.modality,
            )
            state.raw_image = state.image  # Update raw_image post-filter
            state.is_filtered = True
            state.filter_type = filter_type

        else:
            raise ValueError(f"Unknown preprocessing step: {step_name}")

    def _extract_features(
        self, state: PipelineState, params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract features based on current state.
        """
        results = {}
        families = params.get(
            "families", ["intensity", "morphology", "texture", "histogram", "ivh"]
        )

        # Optional kwargs pass-through (advanced usage)
        spatial_intensity_params = params.get("spatial_intensity_params", {})
        local_intensity_params = params.get("local_intensity_params", {})
        ivh_params = params.get("ivh_params", {})
        texture_matrix_params = params.get("texture_matrix_params", {})

        if spatial_intensity_params is None:
            spatial_intensity_params = {}
        if local_intensity_params is None:
            local_intensity_params = {}
        if ivh_params is None:
            ivh_params = {}
        if texture_matrix_params is None:
            texture_matrix_params = {}

        if not isinstance(spatial_intensity_params, dict):
            raise ValueError("spatial_intensity_params must be a dict")
        if not isinstance(local_intensity_params, dict):
            raise ValueError("local_intensity_params must be a dict")
        if not isinstance(ivh_params, dict):
            raise ValueError("ivh_params must be a dict")
        if not isinstance(texture_matrix_params, dict):
            raise ValueError("texture_matrix_params must be a dict")

        # Morphology - uses raw_image (non-discretised) for intensity-based features
        if "morphology" in families:
            results.update(
                calculate_morphology_features(
                    state.morph_mask,
                    state.raw_image,
                    intensity_mask=state.intensity_mask,
                )
            )

        # Intensity - uses raw_image (non-discretised)
        if "intensity" in families:
            masked_values = apply_mask(state.raw_image, state.intensity_mask)
            results.update(calculate_intensity_features(masked_values))

            include_spatial = bool(params.get("include_spatial_intensity", False))
            include_local = bool(params.get("include_local_intensity", False))

            if include_spatial:
                results.update(
                    calculate_spatial_intensity_features(
                        state.raw_image,
                        state.intensity_mask,
                        **spatial_intensity_params,
                    )
                )
            if include_local:
                results.update(
                    calculate_local_intensity_features(
                        state.raw_image, state.intensity_mask, **local_intensity_params
                    )
                )

        # Optional explicit families (no-op unless requested)
        if "spatial_intensity" in families and "intensity" not in families:
            results.update(
                calculate_spatial_intensity_features(
                    state.raw_image, state.intensity_mask, **spatial_intensity_params
                )
            )

        if "local_intensity" in families and "intensity" not in families:
            results.update(
                calculate_local_intensity_features(
                    state.raw_image, state.intensity_mask, **local_intensity_params
                )
            )

        # Histogram / IVH
        if "histogram" in families:
            # Usually on discretised image
            if not state.is_discretised:
                warnings.warn(
                    "Histogram features requested but image is not discretised. "
                    "Features may be unreliable or fail if integer bins are expected.",
                    UserWarning,
                    stacklevel=2,
                )

            masked_values = apply_mask(state.image, state.intensity_mask)
            results.update(calculate_intensity_histogram_features(masked_values))

        if "ivh" in families:
            # IVH computation supports three modes:
            # 1. ivh_use_continuous=True: Use raw (pre-discretised) intensity values
            # 2. ivh_discretisation={...}: Apply temporary discretisation just for IVH
            # 3. Default: Use the pipeline's discretised image (if discretised)

            ivh_use_continuous = params.get("ivh_use_continuous", False)
            ivh_discretisation = params.get("ivh_discretisation", None)

            # Track discretisation params for IVH calculation
            ivh_disc_bin_width: Optional[float] = None
            ivh_disc_min_val: Optional[float] = None

            if ivh_use_continuous:
                # Use raw intensity values (non-discretised)
                # This is used for continuous IVH (e.g., IBSI Config D)
                ivh_values = apply_mask(state.raw_image, state.intensity_mask)
            elif ivh_discretisation:
                # Apply temporary discretisation for IVH only
                # This allows different binning for IVH vs texture features
                # Uses raw_image as the base to discretise from raw values
                ivh_disc_params = ivh_discretisation.copy()
                ivh_method = ivh_disc_params.pop("method", "FBS")
                # Save bin_width and min_val for passing to calculate_ivh_features
                ivh_disc_bin_width = ivh_disc_params.get("bin_width")
                ivh_disc_min_val = ivh_disc_params.get("min_val")
                temp_ivh_disc = discretise_image(
                    state.raw_image,
                    method=ivh_method,
                    roi_mask=state.intensity_mask,
                    **ivh_disc_params,
                )
                ivh_values = apply_mask(temp_ivh_disc, state.intensity_mask)
            else:
                # Default: use the current image (which may be discretised)
                ivh_values = apply_mask(state.image, state.intensity_mask)

            # IVH accepts several optional arguments; support both explicit top-level
            # keys and an "ivh_params" dict for full control.
            ivh_kwargs: dict[str, Any] = {}

            # If ivh_discretisation was used, pass its bin_width and min_val
            if ivh_disc_bin_width is not None:
                ivh_kwargs["bin_width"] = ivh_disc_bin_width
            if ivh_disc_min_val is not None:
                ivh_kwargs["min_val"] = ivh_disc_min_val

            # Dict-based params (preferred) - these override discretisation defaults
            if "bin_width" in ivh_params:
                ivh_kwargs["bin_width"] = ivh_params.get("bin_width")
            if "min_val" in ivh_params:
                ivh_kwargs["min_val"] = ivh_params.get("min_val")
            if "max_val" in ivh_params:
                ivh_kwargs["max_val"] = ivh_params.get("max_val")
            if "target_range_min" in ivh_params:
                ivh_kwargs["target_range_min"] = ivh_params.get("target_range_min")
            if "target_range_max" in ivh_params:
                ivh_kwargs["target_range_max"] = ivh_params.get("target_range_max")

            # If not provided, and we are discretised (and not using continuous mode),
            # default bin_width to 1.0 (bin indices)
            if (
                not ivh_use_continuous
                and state.is_discretised
                and ivh_kwargs.get("bin_width") is None
                and not ivh_discretisation
            ):
                ivh_kwargs["bin_width"] = 1.0

            # Only pass non-None arguments
            ivh_kwargs = {k: v for k, v in ivh_kwargs.items() if v is not None}

            results.update(calculate_ivh_features(ivh_values, **ivh_kwargs))

        # Texture
        if "texture" in families:
            if not state.is_discretised:
                raise ValueError(
                    "Texture features requested but image is not discretised. "
                    "You must include a 'discretise' step before extracting texture features."
                )

            disc_image = state.image
            n_bins = state.n_bins if state.n_bins else 32  # Fallback

            # Calculate Matrices
            # Use morphological mask for distance map (GLDZM)
            # Advanced: allow overriding matrix computation parameters via texture_matrix_params.
            matrix_kwargs: dict[str, Any] = {}
            if "ngldm_alpha" in texture_matrix_params:
                matrix_kwargs["ngldm_alpha"] = texture_matrix_params.get("ngldm_alpha")
            matrix_kwargs = {k: v for k, v in matrix_kwargs.items() if v is not None}

            texture_matrices = calculate_all_texture_matrices(
                disc_image.array,
                state.intensity_mask.array,
                n_bins,
                distance_mask=state.morph_mask.array,
                **matrix_kwargs,
            )

            results.update(
                calculate_glcm_features(
                    disc_image.array,
                    state.intensity_mask.array,
                    n_bins,
                    glcm_matrix=texture_matrices["glcm"],
                )
            )
            results.update(
                calculate_glrlm_features(
                    disc_image.array,
                    state.intensity_mask.array,
                    n_bins,
                    glrlm_matrix=texture_matrices["glrlm"],
                )
            )
            results.update(
                calculate_glszm_features(
                    disc_image.array,
                    state.intensity_mask.array,
                    n_bins,
                    glszm_matrix=texture_matrices["glszm"],
                )
            )
            results.update(
                calculate_gldzm_features(
                    disc_image.array,
                    state.intensity_mask.array,
                    n_bins,
                    gldzm_matrix=texture_matrices["gldzm"],
                    distance_mask=state.morph_mask.array,
                )
            )
            results.update(
                calculate_ngtdm_features(
                    disc_image.array,
                    state.intensity_mask.array,
                    n_bins,
                    ngtdm_matrices=(
                        texture_matrices["ngtdm_s"],
                        texture_matrices["ngtdm_n"],
                    ),
                )
            )
            results.update(
                calculate_ngldm_features(
                    disc_image.array,
                    state.intensity_mask.array,
                    n_bins,
                    ngldm_matrix=texture_matrices["ngldm"],
                )
            )

        return results

    def _extract_features_with_dedup(
        self,
        state: PipelineState,
        params: dict[str, Any],
        config_name: str,
        plan: DeduplicationPlan,
        family_cache: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Extract features using deduplication plan to avoid redundant computation.

        For each feature family requested, checks if an identical signature has
        already been computed. If so, reuses cached results. Otherwise computes
        and caches for potential reuse by subsequent configurations.

        Args:
            state: Current pipeline state.
            params: Feature extraction parameters.
            config_name: Name of the current configuration.
            plan: Deduplication plan mapping families to signatures.
            family_cache: Cache of computed family features by signature hash.

        Returns:
            Dictionary of all extracted features.
        """
        results: dict[str, Any] = {}
        families = params.get(
            "families", ["intensity", "morphology", "texture", "histogram", "ivh"]
        )

        for family in families:
            # Normalize family name for signature lookup
            # texture_* families use the base "texture" signature
            sig_family = "texture" if family.startswith("texture_") else family

            # Get signature from plan using (config_name, family) tuple key
            sig = plan.signatures.get((config_name, sig_family))

            if sig and sig.hash in family_cache:
                # Reuse cached results
                cached = family_cache[sig.hash]
                results.update(cached)
                self._dedup_reused_count += 1
            else:
                # Compute this family
                family_results = self._extract_single_family(state, family, params)
                results.update(family_results)

                # Cache if we have a signature
                if sig:
                    family_cache[sig.hash] = family_results
                self._dedup_computed_count += 1

        return results

    def _extract_single_family(
        self,
        state: PipelineState,
        family: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract features for a single family.

        This is a refactored helper to enable per-family deduplication.
        """
        results: dict[str, Any] = {}

        # Optional kwargs pass-through
        spatial_intensity_params = params.get("spatial_intensity_params", {}) or {}
        local_intensity_params = params.get("local_intensity_params", {}) or {}
        ivh_params = params.get("ivh_params", {}) or {}
        texture_matrix_params = params.get("texture_matrix_params", {}) or {}

        if family == "morphology":
            results.update(
                calculate_morphology_features(
                    state.morph_mask,
                    state.raw_image,
                    intensity_mask=state.intensity_mask,
                )
            )

        elif family == "intensity":
            masked_values = apply_mask(state.raw_image, state.intensity_mask)
            results.update(calculate_intensity_features(masked_values))

            include_spatial = bool(params.get("include_spatial_intensity", False))
            include_local = bool(params.get("include_local_intensity", False))

            if include_spatial:
                results.update(
                    calculate_spatial_intensity_features(
                        state.raw_image,
                        state.intensity_mask,
                        **spatial_intensity_params,
                    )
                )
            if include_local:
                results.update(
                    calculate_local_intensity_features(
                        state.raw_image, state.intensity_mask, **local_intensity_params
                    )
                )

        elif family == "spatial_intensity":
            results.update(
                calculate_spatial_intensity_features(
                    state.raw_image, state.intensity_mask, **spatial_intensity_params
                )
            )

        elif family == "local_intensity":
            results.update(
                calculate_local_intensity_features(
                    state.raw_image, state.intensity_mask, **local_intensity_params
                )
            )

        elif family == "histogram":
            if not state.is_discretised:
                warnings.warn(
                    "Histogram features requested but image is not discretised. "
                    "Features may be unreliable.",
                    UserWarning,
                    stacklevel=2,
                )
            masked_values = apply_mask(state.image, state.intensity_mask)
            results.update(calculate_intensity_histogram_features(masked_values))

        elif family == "ivh":
            results.update(self._compute_ivh_features(state, params, ivh_params))

        elif family == "texture" or family.startswith("texture_"):
            results.update(
                self._compute_texture_features(state, family, texture_matrix_params)
            )

        return results

    def _compute_ivh_features(
        self,
        state: PipelineState,
        params: dict[str, Any],
        ivh_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute IVH features (helper for _extract_single_family)."""
        ivh_use_continuous = params.get("ivh_use_continuous", False)
        ivh_discretisation = params.get("ivh_discretisation", None)

        ivh_disc_bin_width: Optional[float] = None
        ivh_disc_min_val: Optional[float] = None

        if ivh_use_continuous:
            ivh_values = apply_mask(state.raw_image, state.intensity_mask)
        elif ivh_discretisation:
            ivh_disc_params = ivh_discretisation.copy()
            ivh_method = ivh_disc_params.pop("method", "FBS")
            ivh_disc_bin_width = ivh_disc_params.get("bin_width")
            ivh_disc_min_val = ivh_disc_params.get("min_val")
            temp_ivh_disc = discretise_image(
                state.raw_image,
                method=ivh_method,
                roi_mask=state.intensity_mask,
                **ivh_disc_params,
            )
            ivh_values = apply_mask(temp_ivh_disc, state.intensity_mask)
        else:
            ivh_values = apply_mask(state.image, state.intensity_mask)

        ivh_kwargs: dict[str, Any] = {}
        if ivh_disc_bin_width is not None:
            ivh_kwargs["bin_width"] = ivh_disc_bin_width
        if ivh_disc_min_val is not None:
            ivh_kwargs["min_val"] = ivh_disc_min_val

        for key in [
            "bin_width",
            "min_val",
            "max_val",
            "target_range_min",
            "target_range_max",
        ]:
            if key in ivh_params:
                ivh_kwargs[key] = ivh_params[key]

        if (
            not ivh_use_continuous
            and state.is_discretised
            and ivh_kwargs.get("bin_width") is None
            and not ivh_discretisation
        ):
            ivh_kwargs["bin_width"] = 1.0

        ivh_kwargs = {k: v for k, v in ivh_kwargs.items() if v is not None}
        return calculate_ivh_features(ivh_values, **ivh_kwargs)

    def _compute_texture_features(
        self,
        state: PipelineState,
        family: str,
        texture_matrix_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute texture features (helper for _extract_single_family)."""
        results: dict[str, Any] = {}

        if not state.is_discretised:
            raise ValueError(
                "Texture features requested but image is not discretised. "
                "You must include a 'discretise' step before extracting texture features."
            )

        disc_image = state.image
        n_bins = state.n_bins if state.n_bins else 32

        matrix_kwargs: dict[str, Any] = {}
        if "ngldm_alpha" in texture_matrix_params:
            matrix_kwargs["ngldm_alpha"] = texture_matrix_params["ngldm_alpha"]

        texture_matrices = calculate_all_texture_matrices(
            disc_image.array,
            state.intensity_mask.array,
            n_bins,
            distance_mask=state.morph_mask.array,
            **matrix_kwargs,
        )

        # If specific texture family requested, only compute that
        if family == "texture" or family == "texture_glcm" or family == "glcm":
            results.update(
                calculate_glcm_features(
                    disc_image.array,
                    state.intensity_mask.array,
                    n_bins,
                    glcm_matrix=texture_matrices["glcm"],
                )
            )
        if family == "texture" or family == "texture_glrlm" or family == "glrlm":
            results.update(
                calculate_glrlm_features(
                    disc_image.array,
                    state.intensity_mask.array,
                    n_bins,
                    glrlm_matrix=texture_matrices["glrlm"],
                )
            )
        if family == "texture" or family == "texture_glszm" or family == "glszm":
            results.update(
                calculate_glszm_features(
                    disc_image.array,
                    state.intensity_mask.array,
                    n_bins,
                    glszm_matrix=texture_matrices["glszm"],
                )
            )
        if family == "texture" or family == "texture_gldzm" or family == "gldzm":
            results.update(
                calculate_gldzm_features(
                    disc_image.array,
                    state.intensity_mask.array,
                    n_bins,
                    gldzm_matrix=texture_matrices["gldzm"],
                    distance_mask=state.morph_mask.array,
                )
            )
        if family == "texture" or family == "texture_ngtdm" or family == "ngtdm":
            results.update(
                calculate_ngtdm_features(
                    disc_image.array,
                    state.intensity_mask.array,
                    n_bins,
                    ngtdm_matrices=(
                        texture_matrices["ngtdm_s"],
                        texture_matrices["ngtdm_n"],
                    ),
                )
            )
        if family == "texture" or family == "texture_ngldm" or family == "ngldm":
            results.update(
                calculate_ngldm_features(
                    disc_image.array,
                    state.intensity_mask.array,
                    n_bins,
                    ngldm_matrix=texture_matrices["ngldm"],
                )
            )

        return results

    def save_log(self, output_path: str) -> None:
        """
        Save the processing log to a JSON file.
        """
        if not output_path.endswith(".json"):
            output_path += ".json"

        with open(output_path, "w") as f:
            json.dump(self._log, f, indent=4, default=str)

    # -------------------------------------------------------------------------
    # Configuration Serialization Methods
    # -------------------------------------------------------------------------

    def list_configs(self) -> list[str]:
        """
        List all registered configuration names.

        Returns:
            List of configuration names.
        """
        return list(self._configs.keys())

    def get_config(self, name: str) -> list[dict[str, Any]]:
        """
        Get a copy of a configuration by name.

        Args:
            name: Configuration name.

        Returns:
            Deep copy of the configuration steps.

        Raises:
            KeyError: If configuration not found.
        """
        if name not in self._configs:
            raise KeyError(f"Configuration '{name}' not found")
        return copy.deepcopy(self._configs[name])

    def remove_config(self, name: str) -> "RadiomicsPipeline":
        """
        Remove a configuration by name.

        Args:
            name: Configuration name to remove.

        Returns:
            Self for method chaining.

        Raises:
            KeyError: If configuration not found.
        """
        if name not in self._configs:
            raise KeyError(f"Configuration '{name}' not found")
        del self._configs[name]
        self._configs_modified_since_plan = True
        return self

    def to_dict(
        self,
        config_names: Optional[list[str]] = None,
        include_metadata: bool = True,
        include_deduplication: bool = True,
    ) -> dict[str, Any]:
        """
        Export configurations to a dictionary.

        Args:
            config_names: Specific configs to export. If None, exports all.
            include_metadata: Whether to include schema version and metadata.
            include_deduplication: Whether to include deduplication settings.

        Returns:
            Dictionary with configs and optional metadata.
        """
        if config_names is None:
            configs_to_export = self._configs
        else:
            configs_to_export = {
                name: self._configs[name]
                for name in config_names
                if name in self._configs
            }

        # Convert tuples to lists for serialization
        serializable_configs: dict[str, Any] = {}
        for name, steps in configs_to_export.items():
            conf_data = {"steps": self._make_serializable(steps)}
            # Include metadata if present
            if name in self._config_metadata:
                meta = self._config_metadata[name]
                if "source_mode" in meta:
                    conf_data["source_mode"] = meta["source_mode"]
                if "sentinel_value" in meta and meta["sentinel_value"] is not None:
                    conf_data["sentinel_value"] = meta["sentinel_value"]
            serializable_configs[name] = conf_data

        result: dict[str, Any] = {}

        if include_metadata:
            result["schema_version"] = CONFIG_SCHEMA_VERSION
            result["exported_at"] = datetime.datetime.now().isoformat()

        result["configs"] = serializable_configs

        if include_deduplication:
            result["deduplication"] = {
                "enabled": self._deduplication_enabled,
                "rules_version": self._deduplication_rules.version,
            }
            # Include last plan if available and not stale
            if self._last_deduplication_plan and not self._configs_modified_since_plan:
                result["deduplication"][
                    "last_plan"
                ] = self._last_deduplication_plan.to_dict()

        return result

    def _make_serializable(self, obj: Any) -> Any:
        """Convert tuples and other non-serializable types to serializable forms."""
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    def to_json(
        self,
        config_names: Optional[list[str]] = None,
        indent: int = 2,
    ) -> str:
        """
        Export configurations to a JSON string.

        Args:
            config_names: Specific configs to export. If None, exports all.
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        data = self.to_dict(config_names=config_names)
        return json.dumps(data, indent=indent, default=str)

    def to_yaml(
        self,
        config_names: Optional[list[str]] = None,
    ) -> str:
        """
        Export configurations to a YAML string.

        Args:
            config_names: Specific configs to export. If None, exports all.

        Returns:
            YAML string representation.
        """
        data = self.to_dict(config_names=config_names)
        result: str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        return result

    def save_configs(
        self,
        output_path: str | Path,
        config_names: Optional[list[str]] = None,
    ) -> None:
        """
        Save configurations to a file (JSON or YAML based on extension).

        Args:
            output_path: Path to output file. Extension determines format.
            config_names: Specific configs to export. If None, exports all.

        Raises:
            ValueError: If file extension is not .json, .yaml, or .yml.
        """
        path = Path(output_path)
        suffix = path.suffix.lower()

        if suffix == ".json":
            content = self.to_json(config_names=config_names)
        elif suffix in (".yaml", ".yml"):
            content = self.to_yaml(config_names=config_names)
        else:
            raise ValueError(
                f"Unsupported file extension: {suffix}. Use .json, .yaml, or .yml"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        validate: bool = False,
        load_standard: bool = False,
    ) -> "RadiomicsPipeline":
        """
        Create a new pipeline instance from a configuration dictionary.

        The resulting pipeline contains only the configurations defined in the
        dictionary by default. Standard configurations are not loaded unless
        explicitly requested.

        Args:
            data: Configuration dictionary with 'configs' key.
            validate: Whether to validate parameters (logs warnings for issues).
            load_standard: Whether to also load standard predefined configurations.
                Defaults to False so that only the provided configs are loaded.

        Returns:
            New RadiomicsPipeline instance with loaded configs.
        """
        # Handle schema version migration if needed
        schema_version = data.get("schema_version", "1.0")
        migrated_data = cls._migrate_config(data, schema_version)

        # Extract deduplication settings if present
        dedup_settings = migrated_data.get("deduplication", {})
        deduplicate = dedup_settings.get("enabled", True)
        dedup_rules_version = dedup_settings.get("rules_version", None)

        # Create pipeline with deduplication settings (no standard configs by default)
        pipeline = cls(
            deduplicate=deduplicate,
            deduplication_rules=dedup_rules_version,
            load_standard=load_standard,
        )

        configs = migrated_data.get("configs", {})
        for name, config_data in configs.items():
            if isinstance(config_data, dict) and "steps" in config_data:
                steps = config_data["steps"]
            elif isinstance(config_data, list):
                steps = config_data
            else:
                warnings.warn(
                    f"Invalid config format for '{name}', skipping",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Extract metadata
            source_mode = "full_image"
            sentinel_value = None

            if isinstance(config_data, dict):
                source_mode = config_data.get("source_mode", "full_image")
                sentinel_value = config_data.get("sentinel_value")

            # Convert YAML lists to tuples where needed
            converted_steps = pipeline._convert_yaml_steps(steps)

            if validate:
                cls._validate_config(name, converted_steps)

            pipeline._configs[name] = converted_steps
            pipeline._config_metadata[name] = {
                "source_mode": source_mode,
                "sentinel_value": sentinel_value,
            }

        # Mark configs as loaded (not modified) so dedup plan from serialized data is valid
        pipeline._configs_modified_since_plan = False

        # Restore last_plan if present and valid
        if "last_plan" in dedup_settings:
            try:
                pipeline._last_deduplication_plan = DeduplicationPlan.from_dict(
                    dedup_settings["last_plan"]
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to restore deduplication plan: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return pipeline

    @classmethod
    def from_json(
        cls,
        json_string: str,
        validate: bool = False,
        load_standard: bool = False,
    ) -> "RadiomicsPipeline":
        """
        Create a new pipeline instance from a JSON string.

        The resulting pipeline contains only the configurations defined in the
        JSON string by default.

        Args:
            json_string: JSON configuration string.
            validate: Whether to validate parameters.
            load_standard: Whether to also load standard predefined configurations.
                Defaults to False so that only the provided configs are loaded.

        Returns:
            New RadiomicsPipeline instance.
        """
        data = json.loads(json_string)
        return cls.from_dict(data, validate=validate, load_standard=load_standard)

    @classmethod
    def from_yaml(
        cls,
        yaml_string: str,
        validate: bool = False,
        load_standard: bool = False,
    ) -> "RadiomicsPipeline":
        """
        Create a new pipeline instance from a YAML string.

        The resulting pipeline contains only the configurations defined in the
        YAML string by default.

        Args:
            yaml_string: YAML configuration string.
            validate: Whether to validate parameters.
            load_standard: Whether to also load standard predefined configurations.
                Defaults to False so that only the provided configs are loaded.

        Returns:
            New RadiomicsPipeline instance.
        """
        data = yaml.safe_load(yaml_string)
        return cls.from_dict(data, validate=validate, load_standard=load_standard)

    @classmethod
    def load_configs(
        cls,
        file_path: str | Path,
        validate: bool = False,
        load_standard: bool = False,
    ) -> "RadiomicsPipeline":
        """
        Load configurations from a file (JSON or YAML).

        The resulting pipeline contains only the configurations defined in the
        file by default. Standard configurations (e.g., ``standard_fbn_32``) are
        not loaded unless ``load_standard=True`` is passed.

        Args:
            file_path: Path to configuration file.
            validate: Whether to validate parameters.
            load_standard: Whether to also load standard predefined configurations.
                Defaults to False so that only the file's configs are loaded.
                Pass True to include standard configs alongside the loaded ones.

        Returns:
            New RadiomicsPipeline instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file extension is unsupported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        suffix = path.suffix.lower()
        content = path.read_text(encoding="utf-8")

        if suffix == ".json":
            return cls.from_json(
                content, validate=validate, load_standard=load_standard
            )
        elif suffix in (".yaml", ".yml"):
            return cls.from_yaml(
                content, validate=validate, load_standard=load_standard
            )
        else:
            raise ValueError(
                f"Unsupported file extension: {suffix}. Use .json, .yaml, or .yml"
            )

    def merge_configs(
        self,
        other: "RadiomicsPipeline",
        overwrite: bool = False,
    ) -> "RadiomicsPipeline":
        """
        Merge configurations from another pipeline instance.

        Args:
            other: Another RadiomicsPipeline to merge from.
            overwrite: Whether to overwrite existing configs with same name.

        Returns:
            Self for method chaining.
        """
        for name, steps in other._configs.items():
            if name in self._configs and not overwrite:
                warnings.warn(
                    f"Config '{name}' already exists, skipping (use overwrite=True)",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            self._configs[name] = copy.deepcopy(steps)
        return self

    # -------------------------------------------------------------------------
    # Schema Migration
    # -------------------------------------------------------------------------

    @staticmethod
    def _migrate_config(data: dict[str, Any], from_version: str) -> dict[str, Any]:
        """
        Migrate configuration from an older schema version to current.

        Args:
            data: Configuration data to migrate.
            from_version: Source schema version.

        Returns:
            Migrated configuration data.
        """
        if from_version == CONFIG_SCHEMA_VERSION:
            return data

        # Validate source version is known
        if from_version not in _VALID_SCHEMA_VERSIONS:
            warnings.warn(
                f"Unknown schema version '{from_version}', proceeding cautiously",
                UserWarning,
                stacklevel=2,
            )

        # Future migrations would go here
        # Example: if from_version == "1.0" and target is "2.0": ...

        return data

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    # Known step types and their valid parameters
    _VALID_STEPS: dict[str, set[str]] = {
        "resample": {"new_spacing", "interpolation"},
        "resegment": {"range_min", "range_max"},
        "filter_outliers": {"sigma"},
        "binarize_mask": {"threshold", "mask_values", "apply_to"},
        "keep_largest_component": set(),
        "round_intensities": set(),
        "discretise": {"method", "n_bins", "bin_width", "min_value", "max_value"},
        "filter": {
            # Shared / dispatch
            "type",
            "boundary",
            # Mean filter
            "support",
            # LoG filter
            "sigma_mm",
            "spacing_mm",
            "truncate",
            # Laws filter
            "kernel",
            "compute_energy",
            "energy_distance",
            # Gabor filter
            "lambda_mm",
            "gamma",
            "theta",
            "delta_theta",
            "average_over_planes",
            # Wavelet filter
            "wavelet",
            "decomposition",
            "level",
            # Riesz transform
            "order",
            "variant",
            # Shared across filters
            "rotation_invariant",
            "pooling",
            "use_parallel",
        },
        "extract_features": {
            "families",
            "include_spatial_intensity",
            "include_local_intensity",
            "texture_matrix_params",
            "ivh_params",
        },
    }

    @classmethod
    def _validate_config(cls, name: str, steps: list[dict[str, Any]]) -> bool:
        """
        Validate a configuration, issuing warnings for issues.

        Args:
            name: Configuration name (for warning messages).
            steps: List of step dictionaries.

        Returns:
            True if valid, False if issues found (warnings are issued).
        """
        is_valid = True

        if not isinstance(steps, list):
            warnings.warn(
                f"Config '{name}': steps must be a list",
                UserWarning,
                stacklevel=2,
            )
            return False

        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                warnings.warn(
                    f"Config '{name}' step {i}: must be a dictionary",
                    UserWarning,
                    stacklevel=2,
                )
                is_valid = False
                continue

            step_type = step.get("step")
            if not step_type:
                warnings.warn(
                    f"Config '{name}' step {i}: missing 'step' key",
                    UserWarning,
                    stacklevel=2,
                )
                is_valid = False
                continue

            if step_type not in cls._VALID_STEPS:
                warnings.warn(
                    f"Config '{name}' step {i}: unknown step type '{step_type}'",
                    UserWarning,
                    stacklevel=2,
                )
                is_valid = False
                continue

            # Check for unknown parameters
            params = step.get("params", {})
            if params:
                valid_params = cls._VALID_STEPS[step_type]
                for param_name in params.keys():
                    if param_name not in valid_params:
                        warnings.warn(
                            f"Config '{name}' step {i} ({step_type}): "
                            f"unknown parameter '{param_name}'",
                            UserWarning,
                            stacklevel=2,
                        )

        return is_valid
