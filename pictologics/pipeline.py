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

import datetime
import json
import warnings
from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from numpy import typing as npt

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
    discretise_image,
    filter_outliers,
    keep_largest_component,
    resample_image,
    resegment_mask,
    round_intensities,
)


@dataclass
class PipelineState:
    """
    Holds the current state of the image and masks during pipeline execution.
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


class EmptyROIMaskError(ValueError):
    """Raised when preprocessing yields an empty ROI mask."""


class RadiomicsPipeline:
    """
    A flexible, configurable pipeline for radiomic feature extraction.
    Allows defining multiple processing configurations (sequences of steps) to be run on data.
    """

    def __init__(self) -> None:
        """Initialize pipeline with empty config registry and load predefined configs."""
        self._configs: dict[str, list[dict[str, Any]]] = {}
        self._log: list[dict[str, Any]] = []
        self._load_predefined_configs()

    def _load_predefined_configs(self) -> None:
        """
        Load predefined, commonly used pipeline configurations.
        """
        # Common resampling parameters
        resample_05mm = {
            "step": "resample",
            "params": {"new_spacing": (0.5, 0.5, 0.5), "interpolation": "linear"},
        }

        # Common feature extraction parameters (all families)
        extract_all = {
            "step": "extract_features",
            "params": {
                "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
                # Performance-friendly defaults: spatial/local intensity extras can be
                # extremely slow on larger ROIs. Users can enable explicitly in custom configs.
                "include_spatial_intensity": False,
                "include_local_intensity": False,
            },
        }

        # --- FBN Configurations ---

        # Steps1: 0.5mm, FBN 8
        self.add_config(
            "standard_fbn_8",
            [
                resample_05mm,
                {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
                extract_all,
            ],
        )

        # Steps2: 0.5mm, FBN 16
        self.add_config(
            "standard_fbn_16",
            [
                resample_05mm,
                {"step": "discretise", "params": {"method": "FBN", "n_bins": 16}},
                extract_all,
            ],
        )

        # Steps3: 0.5mm, FBN 32
        self.add_config(
            "standard_fbn_32",
            [
                resample_05mm,
                {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
                extract_all,
            ],
        )

        # --- FBS Configurations ---

        # Steps4: 0.5mm, FBS 8.0
        self.add_config(
            "standard_fbs_8",
            [
                resample_05mm,
                {"step": "discretise", "params": {"method": "FBS", "bin_width": 8.0}},
                extract_all,
            ],
        )

        # Steps5: 0.5mm, FBS 16.0
        self.add_config(
            "standard_fbs_16",
            [
                resample_05mm,
                {"step": "discretise", "params": {"method": "FBS", "bin_width": 16.0}},
                extract_all,
            ],
        )

        # Steps6: 0.5mm, FBS 32.0
        self.add_config(
            "standard_fbs_32",
            [
                resample_05mm,
                {"step": "discretise", "params": {"method": "FBS", "bin_width": 32.0}},
                extract_all,
            ],
        )

    def get_all_standard_config_names(self) -> list[str]:
        """
        Returns the list of all standard configuration names.
        """
        return [
            "standard_fbn_8",
            "standard_fbn_16",
            "standard_fbn_32",
            "standard_fbs_8",
            "standard_fbs_16",
            "standard_fbs_32",
        ]

    def add_config(self, name: str, steps: list[dict[str, Any]]) -> "RadiomicsPipeline":
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
                   - 'extract_features': params: families (list), etc.

        Note:
            - Texture features require a prior 'discretise' step.
            - IVH features are configured via 'ivh_params' dict.
        """
        if not isinstance(steps, list):
            raise ValueError("Configuration must be a list of steps")

        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("Each step must be a dictionary")
            if "step" not in step:
                raise ValueError("Each step must have a 'step' key")

        self._configs[name] = steps
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

        # Run each configuration
        for config_name in target_configs:
            steps = self._configs[config_name]

            # Initialize State
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
            )

            self._ensure_nonempty_roi(state, context="initialization")

            config_log: dict[str, Any] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "subject_id": subject_id,
                "config_name": config_name,
                "image_source": img_source,
                "mask_source": mask_source,
                "steps_executed": [],
            }

            config_features = {}

            try:
                for step_def in steps:
                    step_name = step_def["step"]
                    params = step_def.get("params", {})

                    # Execute Step
                    if step_name == "extract_features":
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
                print(f"Error in config '{config_name}', step '{step_def}': {e}")

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
        has_morph_roi = bool(np.any(state.morph_mask.array == 1))

        if not has_intensity_roi or not has_morph_roi:
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

            # Update Image and raw_image
            state.image = resample_image(
                state.image,
                spacing,
                interpolation=interp_img,
                round_intensities=round_intensities_flag,
            )
            state.raw_image = (
                state.image
            )  # Keep raw_image in sync before discretisation

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
                filtered_array = mean_filter(state.image.array, **filter_params)

            elif filter_type == "log":
                filter_params["boundary"] = boundary
                # Use image spacing if not provided
                if "spacing_mm" not in filter_params:
                    filter_params["spacing_mm"] = state.image.spacing
                filtered_array = laplacian_of_gaussian(
                    state.image.array, **filter_params
                )

            elif filter_type == "laws":
                filter_params["boundary"] = boundary
                # 'kernel' param maps to first positional arg
                kernel = filter_params.pop("kernel", "L5E5E5")
                filtered_array = laws_filter(state.image.array, kernel, **filter_params)

            elif filter_type == "gabor":
                filter_params["boundary"] = boundary
                if "spacing_mm" not in filter_params:
                    filter_params["spacing_mm"] = state.image.spacing
                filtered_array = gabor_filter(state.image.array, **filter_params)

            elif filter_type == "wavelet":
                filter_params["boundary"] = boundary
                filtered_array = wavelet_transform(state.image.array, **filter_params)

            elif filter_type == "simoncelli":
                # Simoncelli doesn't use boundary param
                filtered_array = simoncelli_wavelet(state.image.array, **filter_params)

            elif filter_type == "riesz":
                # Riesz transform variants
                variant = filter_params.pop("variant", "base")
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

    def save_log(self, output_path: str) -> None:
        """
        Save the processing log to a JSON file.
        """
        if not output_path.endswith(".json"):
            output_path += ".json"

        with open(output_path, "w") as f:
            json.dump(self._log, f, indent=4, default=str)
