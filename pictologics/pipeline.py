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
import difflib
import functools
import hashlib
import inspect
import itertools
import json
import logging
import math
import os
import pickle
import platform
import re
import sys
import time
import warnings
import weakref
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import as_completed
from dataclasses import dataclass, replace
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional, cast

import numba
import numpy as np
import pandas as pd
import pywt
import yaml
from numpy import typing as npt

from .deduplication import (
    ConfigurationAnalyzer,
    DeduplicationPlan,
    DeduplicationRules,
    get_default_rules,
)
from .features import FEATURE_NAMES
from .features._utils import compute_nonzero_bbox, merge_bboxes, roi_min_max
from .features.intensity import (
    _LOCAL_PEAK_RADIUS_MM,
    calculate_intensity_features,
    calculate_intensity_histogram_features,
    calculate_ivh_features,
    calculate_local_intensity_features,
    calculate_spatial_intensity_features,
)
from .features.morphology import calculate_morphology_features
from .features.texture import (
    _TEXTURE_FAMILIES,
    _directions,
    _glszm_features_from_cells,
    _planar_axes,
    _texture_matrices,
    calculate_glcm_features,
    calculate_gldzm_features,
    calculate_glrlm_features,
    calculate_ngldm_features,
    calculate_ngtdm_features,
)
from .filters import (
    BoundaryCondition,
    gabor_filter,
    gaussian_filter,
    laplacian_of_gaussian,
    laws_filter,
    mean_filter,
    riesz_log,
    riesz_simoncelli,
    riesz_transform,
    simoncelli_wavelet,
    wavelet_transform,
)
from .filters.base import _whole_number, resolve_boundary
from .filters.riesz import _riesz_order_problem
from .filters.wavelets import _wavelet_problem
from .loader import Image, _validate_geometry, create_full_mask, load_image
from .preprocessing import (
    _NORMALISE_METHODS,
    _all_finite,
    _grow_problem,
    _nearest_roi_map,
    _nearest_roi_part,
    _normalise_problem,
    _normalised,
    _output_grid,
    _region_origin,
    _roi_region,
    apply_mask,
    detect_sentinel_value,
    discretise_image,
    filter_outliers,
    grow_mask,
    keep_largest_component,
    resample_image,
    resegment_mask,
    round_intensities,
)
from .results import _json_safe, format_results
from .templates import _load_yaml, get_standard_templates, list_template_files, load_template_file

# Schema version for config serialization - increment when format changes
CONFIG_SCHEMA_VERSION = "1.0"

# Valid schema versions (for backward compatibility)
_VALID_SCHEMA_VERSIONS = {"1.0", "1.1"}


@functools.cache
def _get_package_version() -> str | None:
    """Return the installed package version when available (read once per process)."""
    try:
        return version("pictologics")
    except PackageNotFoundError:
        return None


# The packages whose versions can change the feature values or the loading of images
_PROVENANCE_PACKAGES = (
    "numpy",
    "scipy",
    "numba",
    "PyWavelets",
    "nibabel",
    "pydicom",
    "python-gdcm",
)


@functools.cache
def _package_versions() -> dict[str, str | None]:
    """The Python version, the platform and the versions of _PROVENANCE_PACKAGES (read once
    per process; None for a package without metadata)."""
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in _PROVENANCE_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _config_hash(snapshot: dict[str, Any]) -> str:
    """The SHA-256 of a configuration snapshot as canonical JSON: the same configuration
    gives the same hash in every run, process and session."""
    text = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Feature metadata for describe_features()
# ---------------------------------------------------------------------------

_FAMILY_GROUP: dict[str, str] = {
    "intensity": "Intensity",
    "histogram": "Intensity",
    "ivh": "Intensity",
    "morphology": "Morphology",
    "local_intensity": "Morphology",
    "spatial_intensity": "Morphology",
    "glcm": "Texture",
    "glrlm": "Texture",
    "glszm": "Texture",
    "gldzm": "Texture",
    "ngtdm": "Texture",
    "ngldm": "Texture",
}

_REQUIRES_DISCRETISATION: dict[str, bool] = {
    "intensity": False,
    "histogram": True,
    "ivh": True,
    "morphology": False,
    "local_intensity": False,
    "spatial_intensity": False,
    "glcm": True,
    "glrlm": True,
    "glszm": True,
    "gldzm": True,
    "ngtdm": True,
    "ngldm": True,
}

_DEFAULT_FEATURE_FAMILIES = ["intensity", "morphology", "texture", "histogram", "ivh"]
_PREPROCESSING_STEPS = (
    "resample",
    "resegment",
    "filter_outliers",
    "round_intensities",
    "keep_largest_component",
    "grow_mask",
    "binarize_mask",
    "normalise",
    "discretise",
    "filter",
)
_PREPROCESSING_PARAM_COLUMNS = {
    "resample": "resample_params",
    "resegment": "resegment_params",
    "filter_outliers": "filter_outliers_params",
    "round_intensities": "round_intensities_params",
    "keep_largest_component": "keep_largest_component_params",
    "grow_mask": "grow_mask_params",
    "binarize_mask": "binarize_mask_params",
    "normalise": "normalise_params",
    "discretise": "discretise_params",
    "filter": "filter_params",
}
_MASK_APPLY_TARGETS = ("both", "morph", "intensity")
_NORMALISE_REGIONS = ("roi", "image")
# The boundary names of a filter step ("wrap" is the scipy name of "periodic")
_BOUNDARY_NAMES = ("mirror", "nearest", "periodic", "zero", "constant", "wrap")
_INTENSITY_MASK_FAMILIES = {
    "intensity",
    "spatial_intensity",
    "local_intensity",
    "histogram",
    "ivh",
    "glcm",
    "glrlm",
    "glszm",
    "gldzm",
    "ngtdm",
    "ngldm",
}
_INTENSITY_WEIGHTED_MORPHOLOGY_FEATURES = {
    "integrated_intensity_99N0",
    "center_of_mass_shift_KLMA",
}


def _normalize_texture_family(family: str) -> str | None:
    """Map texture aliases to their canonical texture family name."""
    if family == "texture":
        return family
    if family in _TEXTURE_FAMILIES:
        return family
    if family.startswith("texture_"):
        suffix = family.removeprefix("texture_")
        if suffix in _TEXTURE_FAMILIES:
            return suffix
    return None


def _feature_name_families(family: str) -> list[str]:
    """Expand a requested feature family into FEATURE_NAMES registry keys."""
    if family == "texture":
        return list(_TEXTURE_FAMILIES)
    texture_family = _normalize_texture_family(family)
    return [texture_family if texture_family is not None else family]


# Memo of one extraction pass: the nonzero box of each mask (key: the id of the mask
# array) and the ROI values of each state image (key: the ids of the image and mask
# arrays). The state holds these arrays for the whole pass, so an id names one array.
_PassCache = dict[Any, Any]


def _texture_cache(families: list[str]) -> dict[str, Optional[dict[str, Any]]]:
    """The texture families (glcm, glrlm, ...) of `families`, each with no results yet.
    The first texture family computed fills them all from one matrix pass."""
    return dict.fromkeys(
        name
        for family in families
        if _normalize_texture_family(family) is not None
        for name in _feature_name_families(family)
    )


def _mask_values_file_form(value: Any) -> Any:
    """Keep a binarize_mask range apart from a list of label values in files.

    The step reads a ``(lo, hi)`` tuple as an inclusive range but a list as label
    values. Files have no tuples, so a range is written as ``{"range": [lo, hi]}``.
    """
    if isinstance(value, tuple) and len(value) == 2:
        return {"range": list(value)}
    return value


def _has_roi(array: npt.NDArray[Any]) -> bool:
    """Whether a mask has a nonzero voxel. A large mask takes the parallel box scan;
    `ndarray.any` reads it in one thread (22 % of a 1 mm CT run)."""
    if array.size < 1 << 20:
        return bool(array.any())
    return compute_nonzero_bbox(array) is not None


# IBSI: "to maintain consistency between samples, we strongly recommend to always set the
# same minimum value for all samples as defined by the lower bound of the re-segmentation
# range". A start at the minimum of each ROI gives a grey level another meaning in each image.
_FBS_START_PROBLEM = (
    "FBS needs a start that is the same for every image: set min_val, or add a resegment "
    "step with range_min before it (after the last filter or normalise step)"
)


# The fraction check of a mask reads a regular sample of about this many voxels of its ROI box
_FRACTION_SAMPLE = 1 << 16


def _has_fractions(array: npt.NDArray[Any], box: tuple[slice, slice, slice]) -> bool:
    """Whether a float mask holds a value that is not a whole number, such as a probability
    map: checked on a regular sample (every step-th voxel along each axis) of about
    _FRACTION_SAMPLE voxels of `box`, the box of its nonzero voxels."""
    part = array[box]
    step = max(1, round((part.size / _FRACTION_SAMPLE) ** (1 / 3)))
    sample = part[::step, ::step, ::step]
    values = sample[sample != 0]
    return not np.array_equal(values, np.round(values))


def _fbs_start(params: dict[str, Any], state: PipelineState) -> float:
    """The first bin edge of an FBS discretisation: its min_val, else the lower bound of the
    resegment ranges of the intensity mask (see _FBS_START_PROBLEM)."""
    min_val = params.get("min_val")
    start = state.resegment_min if min_val is None else float(min_val)
    if start is None:
        raise ValueError(f"{_FBS_START_PROBLEM}.")
    return start


def _get_apply_to(params: dict[str, Any], step_name: str) -> str:
    """Return a validated mask target for preprocessing steps that support it."""
    apply_to = params.get("apply_to", "both")
    if apply_to not in _MASK_APPLY_TARGETS:
        allowed = "', '".join(_MASK_APPLY_TARGETS)
        raise ValueError(f"{step_name} apply_to must be '{allowed}'")
    return cast(str, apply_to)


def _canonical(value: Any) -> Any:
    """A hashable form of step parameters that keeps tuples (a binarize range) and
    lists (labels) apart."""
    if isinstance(value, dict):
        return tuple(sorted((key, _canonical(item)) for key, item in value.items()))
    if isinstance(value, np.ndarray):
        return ("ndarray", _canonical(value.tolist()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_canonical(item) for item in value))
    return value


def _prefix_keys(steps: list[dict[str, Any]], metadata: dict[str, Any]) -> list[str]:
    """One key per prefix of the leading preprocessing steps (those before the first
    extract_features step). Equal keys give equal states after the prefix: the start
    state depends only on the source mode and the sentinel value."""
    parts = [repr((metadata.get("source_mode", "full_image"), metadata.get("sentinel_value")))]
    keys = []
    for index, step in enumerate(steps):
        if step["step"] == "extract_features":
            break
        part: tuple[Any, ...] = (step["step"], _canonical(step.get("params", {})))
        if step["step"] in ("filter", "resample", "discretise"):  # how much of the image it keeps
            part += (_roi_reach(steps[index + 1 :]),)
        parts.append(repr(part))
        keys.append("|".join(parts))
    return keys


def _needs_full_grid(later_steps: list[dict[str, Any]]) -> bool:
    """Whether a later step reads a filtered image outside the ROI region (another
    filter, a resample, a binarize_mask that selects mask value 0, so voxels outside
    the ROI, or a normalise step over the whole image); otherwise a filter computes only
    the region around the ROI."""
    return any(
        step["step"] in ("filter", "resample")
        or (step["step"] == "binarize_mask" and _selects_background(step.get("params") or {}))
        or (step["step"] == "normalise" and (step.get("params") or {}).get("region") == "image")
        for step in later_steps
    )


# From this many voxels on, a discretise step cuts the arrays to the ROI box (see
# _cut_to_roi). Measured: on smaller images, finding and cutting the box costs more than
# the binning it saves (about 50 us a configuration).
_CUT_MIN_SIZE = 1 << 16
# From this many voxels on, a filter may filter only the ROI region. Measured: below,
# finding the region and filling the image back costs more than it saves (+34 us on an
# 80-voxel phantom); from 32^3 voxels on, an axial Gabor filter is 4 times faster.
_FILTER_REGION_MIN = 1 << 12
# A configuration needs at least this many bytes per voxel of a resampled grid: the image,
# its masks, the ROI values and their working copies. Measured on a whole 0.5 mm grid of
# 32 million voxels: 29 for intensity features alone, 36 with a LoG filter, 50 with texture
# features and 55 with a Simoncelli filter.
_GRID_BYTES_PER_VOXEL = 24


@functools.cache
def _physical_memory() -> int:
    """The bytes of physical memory of this computer."""
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatus(ctypes.Structure):  # MEMORYSTATUSEX: 2 DWORD, 7 DWORDLONG
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64)
                for name in (
                    "total",
                    "free",
                    "page",
                    "free_page",
                    "virtual",
                    "free_virtual",
                    "extended",
                )
            ]

        status = _MemoryStatus()
        status.length = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.total)
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _check_grid_memory(shape: list[int], new_spacing: Any) -> None:
    """Raise a MemoryError before a resample to a grid of `shape` that cannot fit in the
    memory of this computer (see _GRID_BYTES_PER_VOXEL)."""
    need = math.prod(shape) * _GRID_BYTES_PER_VOXEL
    memory = _physical_memory()
    if need > memory:
        raise MemoryError(
            f"Resampling to {tuple(new_spacing)} mm makes a grid of "
            f"{' x '.join(map(str, shape))} voxels. It needs at least {need / 2**30:.1f} GB, "
            f"more than the {memory / 2**30:.1f} GB of memory of this computer. Use a larger "
            "spacing, or a smaller image or ROI."
        )


def _roi_reach(later_steps: list[dict[str, Any]]) -> Optional[float]:
    """How far (mm) outside the ROI box the later steps read: the growth of the later
    grow_mask steps, and the radius of the local intensity sphere around the ROI (as grown
    up to that step) when a later step computes the local peaks. None when a later step
    reads the whole image (see _needs_full_grid); a filter, a resample or a discretise
    step then keeps the whole grid."""
    if _needs_full_grid(later_steps):
        return None
    reach = grown = 0.0
    for step in later_steps:
        grown += _grow_reach([step])
        if step["step"] == "extract_features" and _reads_local_peak(step.get("params") or {}):
            reach = max(reach, grown + _LOCAL_PEAK_RADIUS_MM)
    return max(reach, grown)


def _grow_reach(steps: list[dict[str, Any]]) -> float:
    """How far (mm) the grow_mask steps of `steps` grow the ROI at most. A step with a bad
    to_mm adds nothing: it stops its configuration when it runs."""
    total = 0.0
    for step in steps:
        to_mm = (step.get("params") or {}).get("to_mm") if step["step"] == "grow_mask" else None
        if to_mm is not None and _grow_problem(to_mm) is None:
            total += max(float(to_mm), 0.0)
    return total


def _splits_rois(steps: list[dict[str, Any]]) -> bool:
    """Whether a grow_mask step of `steps` gives each added voxel to its nearest ROI."""
    return any(
        step["step"] == "grow_mask" and (step.get("params") or {}).get("nearest_roi") is True
        for step in steps
    )


def _reads_local_peak(params: dict[str, Any]) -> bool:
    """Whether an extract_features step computes the local intensity peaks."""
    families = params.get("families", _DEFAULT_FEATURE_FAMILIES)
    return "local_intensity" in families or (
        "intensity" in families and bool(params.get("include_local_intensity", False))
    )


def _selects_background(params: dict[str, Any]) -> bool:
    """Whether a binarize_mask step keeps mask voxels of value 0."""
    values = params.get("mask_values")
    if values is None:
        threshold = params.get("threshold", 0.5)
        return threshold is not None and float(threshold) <= 0
    if isinstance(values, tuple) and len(values) == 2:
        return bool(values[0] <= 0 <= values[1])
    return 0 in (values if isinstance(values, (list, tuple)) else [values])


def _filter_reach(
    filter_type: str, params: dict[str, Any], spacing: tuple[float, float, float]
) -> Optional[tuple[int, int, int]]:
    """Voxels that a filter output reads on each side, per axis, for the filters whose
    values do not depend on where the image ends: separable convolutions (LoG, wavelets,
    the Laws response). The running sums of the mean filter and the Laws energy round in
    an order that starts at the line start, so their reach is negative: they read from
    the image start (see _filter_crop). None for the FFT filters and Gabor (which takes
    the region itself)."""
    if filter_type in ("log", "gaussian"):
        spacing_mm = np.broadcast_to(np.asarray(params["spacing_mm"], dtype=float), (3,))
        truncate = params.get("truncate", 4.0)
        r = [int(truncate * params["sigma_mm"] / s + 0.5) + 1 for s in spacing_mm]
        return (r[0], r[1], r[2])
    if filter_type == "wavelet":
        n = pywt.Wavelet(params.get("wavelet", "db2")).dec_len
        reach = sum((n - 1) * 2 ** (j - 1) + 1 for j in range(1, params.get("level", 1) + 1))
        return (reach, reach, reach)
    if filter_type == "laws":
        kernels = params.get("kernel", "L5E5E5")
        half = max(int(kernels[i + 1]) for i in range(0, len(kernels), 2)) // 2 + 1
        if params.get("compute_energy", False):
            half = -(half + params.get("energy_distance", 7) + 1)
        return (half, half, half)
    if filter_type == "mean":
        reach = -(params.get("support", 15) // 2 + 1)
        return (reach, reach, reach)
    return None


def _filter_crop(
    region: tuple[slice, slice, slice],
    reach: tuple[int, int, int],
    shape: tuple[int, ...],
    periodic: bool,
) -> tuple[slice, slice, slice]:
    """The part of the image that a filter reads for its values in `region`: the region
    grown by the reach on both sides, or for a negative reach (running sums) from the
    image start to the region end plus the reach. With a periodic boundary, the whole
    axis where the part meets an image end (the filter then reads the other end)."""
    crop = []
    for r, h, n in zip(region, reach, shape, strict=True):
        lo, hi = (0, r.stop - h) if h < 0 else (r.start - h, r.stop + h)
        if periodic and (lo <= 0 or hi >= n):
            lo, hi = 0, n
        crop.append(slice(max(lo, 0), min(hi, n)))
    return crop[0], crop[1], crop[2]


def _cut_to_roi(state: "PipelineState", cuts: dict[tuple[int, Any], npt.NDArray[Any]]) -> None:
    """Cut the images and masks of `state` to the ROI box, grown by `state.roi_reach` (mm),
    when the box is smaller than the arrays. grid_shape and grid_offset keep the place of
    the box in the whole grid. `cuts` keeps each cut while its array lives, so states that
    share an array (and the memos that compare arrays by identity) share its cut."""
    shape = state.image.array.shape
    margin_mm = cast(float, state.roi_reach)
    bbox = cast(
        tuple[slice, slice, slice],
        merge_bboxes(
            compute_nonzero_bbox(state.morph_mask.array),
            compute_nonzero_bbox(state.intensity_mask.array),
        ),
    )  # the ROI checks keep the masks non-empty
    grow = [math.ceil(margin_mm / s) + 1 if margin_mm else 0 for s in state.image.spacing]
    rs = [
        slice(max(b.start - g, 0), min(b.stop + g, n))
        for b, g, n in zip(bbox, grow, shape, strict=True)
    ]
    region = (rs[0], rs[1], rs[2])
    if all(r.stop - r.start == n for r, n in zip(region, shape, strict=True)):
        return
    bounds = tuple((r.start, r.stop) for r in region)

    def piece(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        key = (id(array), bounds)
        if key not in cuts:  # a copy: a view would keep the whole array alive
            cuts[key] = np.array(array[region], order="C")
            weakref.finalize(array, cuts.pop, key, None)
        return cuts[key]

    def part(image: Image) -> Image:
        return Image(
            array=piece(image.array),
            spacing=image.spacing,
            origin=_region_origin(image.origin, image.direction, image.spacing, region),
            direction=image.direction,
            modality=image.modality,
            source_mask=None if image.source_mask is None else piece(image.source_mask),
        )

    state.image, state.raw_image = part(state.image), part(state.raw_image)
    state.morph_mask, state.intensity_mask = part(state.morph_mask), part(state.intensity_mask)
    if state.source_mask is not None:
        state.source_mask = part(state.source_mask)
    offset = state.grid_offset or (0, 0, 0)
    state.grid_shape = state.grid_shape or (int(shape[0]), int(shape[1]), int(shape[2]))
    state.grid_offset = (
        offset[0] + region[0].start,
        offset[1] + region[1].start,
        offset[2] + region[2].start,
    )


def _source_mask(valid: npt.NDArray[Any], grid: Image) -> Image:
    """A source mask (True = valid voxel) on the grid of `grid`."""
    return Image(
        array=valid,
        spacing=grid.spacing,
        origin=grid.origin,
        direction=grid.direction,
        modality="SOURCE_MASK",
    )


def _finite_intensity_mask(state: PipelineState, config_name: str) -> PipelineState:
    """`state` with the ROI voxels of non-finite intensity (NaN or infinite) left out of
    the intensity mask, with a warning. IBSI marks voxels outside the ROI with NaN, and
    such a voxel has no intensity for any feature. The morphological mask stays as it is.

    Raises:
        EmptyROIMaskError: If no ROI voxel has a finite intensity.
    """
    mask = state.intensity_mask.array
    # The ROI checks of the steps keep the intensity mask non-empty
    box = cast(tuple[slice, slice, slice], compute_nonzero_bbox(mask))
    bad = (mask[box] != 0) & ~np.isfinite(state.raw_image.array[box])
    n_bad = int(np.count_nonzero(bad))
    if n_bad == 0:
        return state
    finite = mask.copy()
    finite[box][bad] = 0
    msg = (
        f"Left out {n_bad:,} ROI voxels with a NaN or infinite intensity from the "
        f"intensity mask of config '{config_name}'."
    )
    logging.warning(msg)
    warnings.warn(msg, UserWarning, stacklevel=3)
    if not _has_roi(finite):
        raise EmptyROIMaskError(f"No ROI voxel of config '{config_name}' has a finite intensity.")
    return replace(state, intensity_mask=replace(state.intensity_mask, array=finite))


def _intersect_mask(mask: Image, valid_mask: npt.NDArray[Any]) -> Image:
    """Return mask intersected with a boolean validity mask, in the type of the mask
    (labels of 256 and more stay as they are)."""
    return Image(
        array=np.where(valid_mask, mask.array, 0).astype(mask.array.dtype, copy=False),
        spacing=mask.spacing,
        origin=mask.origin,
        direction=mask.direction,
        modality=mask.modality,
    )


def _family_uses_morph_mask(family: str) -> bool:
    """Whether features in this family depend on the morphology mask geometry."""
    return family == "morphology" or family == "gldzm"


def _feature_uses_intensity_mask(feature_key: str, family: str) -> bool:
    """Whether this feature row depends on the intensity mask."""
    return (
        family in _INTENSITY_MASK_FAMILIES or feature_key in _INTENSITY_WEIGHTED_MORPHOLOGY_FEATURES
    )


# Regex to extract the IBSI alphanumeric code from a feature key.
# Most codes are 4 characters; a small number (e.g. ``1PR``) are 3.
# An optional trailing ``_\d+`` suffix covers IVH keys like ``_BC2M_10``.
_IBSI_CODE_RE = re.compile(r"_([A-Z0-9]{3,4})(?:_\d+)?$")


_SOURCE_MODES = ("full_image", "roi_only", "auto")
# extract_features options that hold the keyword arguments of one feature function
_OPTION_GROUPS = (
    "spatial_intensity_params",
    "local_intensity_params",
    "texture_matrix_params",
    "ivh_params",
    "ivh_discretisation",
)
_DISCRETISE_METHODS = ("FBN", "FBS", "FIXED_CUTOFFS")
# The options of texture_matrix_params (IBSI 1 texture parameters)
_TEXTURE_DISTANCES = ("glcm_distance", "ngtdm_distance", "ngldm_distance")
_TEXTURE_OPTIONS = ("ngldm_alpha", *_TEXTURE_DISTANCES)
_GABOR_RESPONSES = ("modulus", "angle", "real", "imaginary")
_RIESZ_FUNCTIONS = {"base": riesz_transform, "log": riesz_log, "simoncelli": riesz_simoncelli}
_FILTER_FUNCTIONS: dict[str, Any] = {
    "mean": mean_filter,
    "log": laplacian_of_gaussian,
    "gaussian": gaussian_filter,
    "laws": laws_filter,
    "gabor": gabor_filter,
    "wavelet": wavelet_transform,
    "simoncelli": simoncelli_wavelet,
    "riesz": riesz_transform,
}


def _hint(word: Any, options: Any) -> str:
    """' (did you mean ...?)' for the closest option (any letter case), or ''."""
    by_lower = {str(o).lower(): str(o) for o in options}
    match = difflib.get_close_matches(str(word).lower(), list(by_lower), n=1)
    return f" (did you mean '{by_lower[match[0]]}'?)" if match else ""


def _filter_function(params: dict[str, Any]) -> Any:
    """The filter function of a filter step (for a Riesz step, that of its variant)."""
    filter_type = str(params["type"])
    if filter_type == "riesz":
        return _RIESZ_FUNCTIONS.get(params.get("variant", "base"), riesz_transform)
    return _FILTER_FUNCTIONS[filter_type]


def _missing_filter_parameters(params: dict[str, Any]) -> list[str]:
    """The parameters without a default that the filter of a filter step needs and that the
    step does not give (the image aside; the Laws kernels have the step default)."""
    if params["type"] == "riesz" and params.get("variant", "base") not in _RIESZ_FUNCTIONS:
        return []  # the unknown variant is the problem
    signature = list(inspect.signature(_filter_function(params)).parameters.values())[1:]
    return [
        p.name
        for p in signature
        if p.default is inspect.Parameter.empty and p.name != "kernels" and p.name not in params
    ]


def _filter_parameters(params: dict[str, Any]) -> set[str]:
    """The step parameters that the filter of a filter step takes (its keyword arguments)."""
    filter_type = str(params["type"])
    names = set(list(inspect.signature(_filter_function(params)).parameters)[1:])
    names -= {"source_mask", "region"}
    if filter_type == "laws":
        names = (names - {"kernels"}) | {"kernel"}  # the step's 'kernel' is the first argument
    if filter_type == "riesz":
        names.add("variant")
    return names | {"type", "boundary"}


def _spacing_problem(spacing: Any) -> Optional[str]:
    """Why a new_spacing is not three positive numbers, or None."""
    try:
        values = [float(s) for s in spacing]
    except (TypeError, ValueError):
        return f"new_spacing must be three positive numbers, not {spacing!r}"
    if len(values) != 3 or not all(v > 0 for v in values):
        return f"new_spacing must be three positive numbers, not {spacing!r}"
    return None


def _step_problems(
    name: str, params: dict[str, Any], discretised: bool, fbs_start: bool
) -> list[str]:
    """Problems with the parameter values of one known step. `fbs_start`: an earlier
    resegment step gives FBS its start (see _FBS_START_PROBLEM)."""
    problems = []
    if "apply_to" in params and params["apply_to"] not in _MASK_APPLY_TARGETS:
        problems.append(
            f"apply_to must be one of {_MASK_APPLY_TARGETS}, not {params['apply_to']!r}"
        )
    if name == "resample":
        if "new_spacing" not in params:
            problems.append("missing parameter 'new_spacing'")
        else:
            problem = _spacing_problem(params["new_spacing"])
            problems.extend([problem] if problem else [])
    elif name == "normalise":
        for key, options in (("method", _NORMALISE_METHODS), ("region", _NORMALISE_REGIONS)):
            if key not in params:
                problems.append(f"missing parameter '{key}'")
            elif params[key] not in options:
                problems.append(f"unknown {key} '{params[key]}'{_hint(params[key], options)}")
        if "percentiles" in params and params.get("method") != "percentile":
            problems.append("percentiles need method 'percentile'")
        problem = _normalise_problem(
            params.get("percentiles", (1.0, 99.0)), params.get("range_min"), params.get("range_max")
        )
        problems.extend([problem] if problem else [])
        if discretised:
            problems.append("a normalise step must come before the discretise step")
    elif name == "grow_mask":
        if "to_mm" not in params:
            problems.append("missing parameter 'to_mm'")
        else:
            problem = _grow_problem(params["to_mm"], params.get("from_mm"))
            problems.extend([problem] if problem else [])
        if not isinstance(params.get("nearest_roi", False), bool):
            problems.append(f"nearest_roi must be True or False, not {params['nearest_roi']!r}")
    elif name == "discretise":
        method = params.get("method", "FBN")
        if method not in _DISCRETISE_METHODS:
            problems.append(f"unknown method '{method}'{_hint(method, _DISCRETISE_METHODS)}")
        elif method == "FBN" and not (
            _whole_number(params.get("n_bins")) and params["n_bins"] >= 1
        ):
            problems.append(f"FBN needs a whole n_bins of 1 or more, not {params.get('n_bins')!r}")
        elif method == "FBS" and not (
            isinstance(params.get("bin_width"), (int, float)) and params["bin_width"] > 0
        ):
            problems.append(f"FBS needs a bin_width above 0, not {params.get('bin_width')!r}")
        elif method == "FIXED_CUTOFFS" and not params.get("cutoffs"):
            problems.append("FIXED_CUTOFFS needs cutoffs")
        if method == "FBS" and params.get("min_val") is None and not fbs_start:
            problems.append(_FBS_START_PROBLEM)
    elif name == "filter":
        filter_type = params.get("type")
        problem = None
        if filter_type == "riesz":
            variant = params.get("variant", "base")
            if variant not in _RIESZ_FUNCTIONS:
                problems.append(
                    f"unknown riesz variant '{variant}'{_hint(variant, _RIESZ_FUNCTIONS)}"
                )
            problem = _riesz_order_problem(params.get("order", (1, 0, 0)), 3)
        elif filter_type == "wavelet":
            problem = _wavelet_problem(params.get("level", 1), params.get("decomposition", "LHL"))
        elif filter_type == "simoncelli":
            problem = _wavelet_problem(params.get("level", 1))
        elif filter_type == "gabor" and params.get("response", "modulus") not in _GABOR_RESPONSES:
            response = params["response"]
            problem = f"unknown gabor response '{response}'{_hint(response, _GABOR_RESPONSES)}"
        problems.extend([problem] if problem else [])
        boundary = params.get("boundary")
        boundary_name = (
            boundary.name.lower()
            if isinstance(boundary, BoundaryCondition)
            else str(boundary).lower()
        )
        if boundary is not None and boundary_name not in _BOUNDARY_NAMES:
            problems.append(f"unknown boundary {boundary!r}{_hint(boundary, _BOUNDARY_NAMES)}")
        padding = params.get("padding_value", 0)
        if isinstance(padding, bool) or not isinstance(padding, (int, float)):
            problems.append(f"padding_value must be a number, not {padding!r}")
        elif padding and boundary_name not in ("zero", "constant"):
            problems.append(
                f"padding_value {padding!r} needs boundary 'constant' (or 'zero'), not "
                f"{params.get('boundary', 'the default')!r}"
            )
    elif name == "extract_features":
        for key in _OPTION_GROUPS:
            if params.get(key) is not None and not isinstance(params[key], dict):
                problems.append(f"{key} must be a dict, not {params[key]!r}")
        texture = params.get("texture_matrix_params")
        for key, value in texture.items() if isinstance(texture, dict) else ():
            if key not in _TEXTURE_OPTIONS:
                problems.append(
                    f"texture_matrix_params: unknown option '{key}'{_hint(key, _TEXTURE_OPTIONS)}"
                )
            elif key == "ngldm_alpha" and not (
                isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
            ):
                problems.append(
                    f"texture_matrix_params: ngldm_alpha must be 0 or more, not {value!r}"
                )
            elif key != "ngldm_alpha" and not (_whole_number(value) and value >= 1):
                problems.append(
                    f"texture_matrix_params: {key} must be a whole number of 1 or more, not {value!r}"
                )
        ivh = params.get("ivh_discretisation")
        if (
            isinstance(ivh, dict)
            and ivh.get("method", "FBS") == "FBS"
            and ivh.get("min_val") is None
            and not fbs_start
        ):
            problems.append(f"ivh_discretisation: {_FBS_START_PROBLEM}")
        families = params.get("families", _DEFAULT_FEATURE_FAMILIES)
        if isinstance(families, str) or not isinstance(families, (list, tuple)):
            return problems + [f"families must be a list of names, not {families!r}"]
        known = sorted(
            set(FEATURE_NAMES) | {"texture"} | {f"texture_{f}" for f in _TEXTURE_FAMILIES}
        )
        for family in families:
            if family not in known:
                problems.append(f"unknown feature family '{family}'{_hint(family, known)}")
            elif _normalize_texture_family(family) is not None and not discretised:
                problems.append(f"the '{family}' features need an earlier 'discretise' step")
    return problems


def _config_problems(steps: Any, source_mode: Any = "full_image") -> list[str]:
    """Everything wrong with a configuration, one message per problem ("step 2
    (resample): ..."). An empty list means the configuration is valid."""
    if not isinstance(steps, list):
        return ["steps must be a list"]
    problems = []
    if source_mode not in _SOURCE_MODES:
        problems.append(f"source_mode must be one of {_SOURCE_MODES}, not {source_mode!r}")
    discretised = fbs_start = False
    for index, step in enumerate(steps):
        where = f"step {index}"
        if not isinstance(step, dict):
            problems.append(f"{where}: must be a dictionary")
            continue
        name = step.get("step")
        if not name:
            problems.append(f"{where}: missing 'step' key")
            continue
        if name not in RadiomicsPipeline._VALID_STEPS:
            problems.append(
                f"{where}: unknown step type '{name}'{_hint(name, RadiomicsPipeline._VALID_STEPS)}"
            )
            continue
        where = f"{where} ({name})"
        params = step.get("params") or {}
        if not isinstance(params, dict):
            problems.append(f"{where}: params must be a dictionary")
            continue
        if name == "filter" and params.get("type") not in _FILTER_FUNCTIONS:
            filter_type = params.get("type")
            problems.append(
                f"{where}: unknown filter type {filter_type!r}{_hint(filter_type, _FILTER_FUNCTIONS)}"
            )
            continue
        allowed = (
            _filter_parameters(params) if name == "filter" else RadiomicsPipeline._VALID_STEPS[name]
        )
        for key in params:
            if key not in allowed:
                problems.append(f"{where}: unknown parameter '{key}'{_hint(key, allowed)}")
        if name == "filter":
            for key in _missing_filter_parameters(params):
                problems.append(f"{where}: missing parameter '{key}'")
        problems.extend(
            f"{where}: {problem}"
            for problem in _step_problems(name, params, discretised, fbs_start)
        )
        discretised = discretised or name == "discretise"
        if name == "resegment":  # as PipelineState.resegment_min
            fbs_start = fbs_start or (
                params.get("range_min") is not None
                and params.get("apply_to", "both") in ("both", "intensity")
            )
        elif name in ("filter", "normalise"):  # other units: no FBS start
            fbs_start = False
    return problems


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
        filter_boundary_requested: Boundary condition requested for the most
            recent filter step (or the default that applied, if omitted).
        filter_boundary_effective: Boundary condition actually honoured by the
            most recent filter step (may be "periodic" for an FFT-based filter
            even when a different boundary was requested but not explicitly set).
        filter_params_requested: The most recent filter step's raw ``params``
            dict, exactly as supplied in the step config (JSON-safe).
        filter_params_effective: The keyword arguments actually honoured for
            the most recent filter step after the pipeline's dispatch logic
            (injected defaults such as ``spacing_mm``, the resolved boundary,
            the ``variant``/``kernel`` dispatch, etc.); JSON-safe.
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
    filter_boundary_requested: Optional[str] = None
    filter_boundary_effective: Optional[str] = None
    filter_params_requested: Optional[dict[str, Any]] = None
    filter_params_effective: Optional[dict[str, Any]] = None
    # Source tracking for sentinel value handling
    source_mode: SourceMode = SourceMode.FULL_IMAGE
    source_mask: Optional[Image] = None
    sentinel_detected: bool = False
    sentinel_value: Optional[float] = None
    # How far (mm) outside the ROI box the steps after a filter, a resample or a discretise
    # step read (see _roi_reach), so that the step computes only that region: a filter
    # writes 0 outside it, a resample makes only that region of the new grid, a discretise
    # step cuts the arrays to it. None: the whole image. grid_shape is the whole grid when
    # the arrays hold a region.
    roi_reach: Optional[float] = None
    grid_shape: Optional[tuple[int, int, int]] = None
    grid_offset: Optional[tuple[int, int, int]] = None  # the first voxel of the region
    # The lower bound of the resegment ranges of the intensity mask (None after a filter,
    # whose values have other units): an FBS step without min_val starts its bins there.
    resegment_min: Optional[float] = None
    # In run_rois, for the grow_mask steps with nearest_roi: the nearest ROI of each voxel
    # (see _nearest_roi_map) and the label of this ROI.
    roi_nearest: Optional[tuple[Image, int]] = None
    # The center and the scale of the last normalise step: (x - center) / scale.
    normalisation: Optional[tuple[float, float]] = None


class EmptyROIMaskError(ValueError):
    """Raised internally when preprocessing yields an empty ROI mask.

    When this error occurs during ``run()``, the affected configuration is
    **not** propagated as an exception.  Instead, the pipeline returns a
    ``pandas.Series`` of ``NaN`` values whose index matches the feature names
    that would have been produced by a successful extraction.  Other
    configurations in the same ``run()`` call continue normally.

    This is one of three mechanisms that guarantee a complete feature set for
    every configuration.  The other two are the partial-failure backfill (for
    individual features that cannot be computed) and the general exception
    handler (for unexpected runtime errors).  See the ``run()`` docstring and
    the *Result Guarantees* section of the user guide for details.
    """


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
        self._config_metadata: dict[str, dict[str, Any]] = {}  # Stores source_mode, etc.
        self._log: list[dict[str, Any]] = []

        # Deduplication settings
        self._deduplication_enabled = deduplicate

        if deduplication_rules is None:
            self._deduplication_rules = get_default_rules()
        elif isinstance(deduplication_rules, str):
            self._deduplication_rules = DeduplicationRules.get_version(deduplication_rules)
        else:
            self._deduplication_rules = deduplication_rules

        self._last_deduplication_plan: DeduplicationPlan | None = None
        self._configs_modified_since_plan: bool = False
        # (rules, pickled configs, plan) of the last plan that run() built
        self._plan_cache: tuple[DeduplicationRules, bytes, DeduplicationPlan] | None = None

        # Deduplication statistics (reset on each run)
        self._dedup_reused_count: int = 0
        self._dedup_computed_count: int = 0
        # (morph mask, intensity mask, GLDZM distance map) of the last texture pass in run()
        self._last_distance_map: Optional[tuple[Any, Any, npt.NDArray[Any]]] = None
        # The ROI box cuts of run() (see _cut_to_roi), each kept while its array lives
        self._roi_cuts: dict[tuple[int, Any], npt.NDArray[Any]] = {}
        # (steps, config hash) of each configuration: a configuration changes only with a
        # new steps list (add_config, merge_configs), so its hash is made once
        self._config_hashes: dict[str, tuple[Any, str]] = {}
        # The feature families that failed in the configuration that runs now: {family: error}
        self._family_errors: dict[str, str] = {}

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
                # {"range": [lo, hi]} is the file form of a binarize_mask (lo, hi) range
                mask_values = params.get("mask_values")
                if isinstance(mask_values, dict) and set(mask_values) == {"range"}:
                    params["mask_values"] = tuple(mask_values["range"])
                new_step["params"] = params
            converted.append(new_step)
        return converted

    def get_all_standard_config_names(self) -> list[str]:
        """
        Returns the list of all standard configuration names.

        Returns names from loaded templates that start with 'standard_'.
        """
        return sorted([name for name in self._configs.keys() if name.startswith("standard_")])

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
        source_mode: "str | SourceMode" = "full_image",
        sentinel_value: Optional[float] = None,
        validate: bool = True,
    ) -> "RadiomicsPipeline":
        """
        Add a processing configuration.

        Args:
            name: Unique name for this configuration.
            steps: List of steps. Each step is a dict with 'step' (name) and 'params' (dict).
                   Supported steps (see the step reference of the user guide):
                   - 'resample': params: new_spacing (required), interpolation,
                       mask_interpolation, mask_threshold, round_intensities
                   - 'resegment': params: range_min, range_max, apply_to
                   - 'filter_outliers': params: sigma, apply_to
                   - 'keep_largest_component': params: apply_to
                   - 'grow_mask': params: to_mm (required), from_mm, nearest_roi, apply_to
                   - 'round_intensities': params: None
                   - 'binarize_mask': params: threshold (float, default 0.5),
                       mask_values (int | list[int] | tuple[int, int]), apply_to ('morph'|'intensity'|'both')
                   - 'normalise': params: method and region (required), percentiles,
                       range_min, range_max
                   - 'discretise': params: method (default FBN), n_bins/bin_width, min_val,
                       max_val, cutoffs
                   - 'filter': params: type (required), plus filter-specific params
                   - 'extract_features': params: families (default: intensity, morphology,
                       texture, histogram, ivh), and the option dicts
            source_mode: Which voxels hold image data, for resampling and filtering:
                - "full_image" (default): every voxel.
                - "roi_only": the voxels of the ROI (sentinel_value is not used).
                - "auto": the voxels without the sentinel value: `sentinel_value`, else
                  a value found in the image (with a warning).
            sentinel_value: The padding value of the image, for source_mode "auto".
            validate: If True (default), check the steps now: the step names, the
                parameter names of each step and each filter type (and the missing
                required ones), the parameter values, the feature family names, the
                discretise method and its bin settings, the FBS start, and the step
                order (texture features need an earlier 'discretise' step; normalise
                comes before discretise). A mistake raises one ValueError that lists
                every problem (with the closest valid name), so it cannot give silent
                NaN columns later. False only checks the structure.

        Raises:
            ValueError: If `steps` is not a list, if `source_mode` is not one of
                "full_image", "roi_only", "auto", if any step is not a dict or is
                missing the 'step' key, or (with `validate`) for any problem above.

        Note:
            - Texture features require a prior 'discretise' step.
            - IVH features are configured via 'ivh_params' dict.
            - The source_mode setting affects resampling and filtering operations:
              they leave out the voxels without image data (normalized interpolation
              and convolution).

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
                        {
                            "step": "extract_features",
                            "params": {
                                "families": [
                                    "intensity",
                                    "morphology",
                                    "texture",
                                    "histogram",
                                    "ivh",
                                ]
                            },
                        },
                    ],
                )
            ```
        """
        if not isinstance(steps, list):
            raise ValueError("Configuration must be a list of steps")

        # Validate source_mode
        if isinstance(source_mode, SourceMode):
            source_mode = source_mode.value
        if source_mode not in _SOURCE_MODES:
            raise ValueError(
                f"Invalid source_mode '{source_mode}'. Must be one of: {set(_SOURCE_MODES)}"
            )

        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("Each step must be a dictionary")
            if "step" not in step:
                raise ValueError("Each step must have a 'step' key")
        if validate:
            problems = _config_problems(steps, source_mode)
            if problems:
                raise ValueError(
                    f"Configuration '{name}' has {len(problems)} problem(s):\n  - "
                    + "\n  - ".join(problems)
                )

        # The pipeline keeps its own copy: a later edit of the caller's list (to build
        # the next configuration from it) must not change this one.
        self._configs[name] = copy.deepcopy(steps)
        self._config_metadata[name] = {
            "source_mode": source_mode,
            "sentinel_value": sentinel_value,
        }
        self._configs_modified_since_plan = True
        return self

    def run(
        self,
        image: str | Path | Image,
        mask: str | Path | Image | None = None,
        subject_id: Optional[str] = None,
        config_names: Optional[list[str]] = None,
        mask_subvoxel_tolerance: float = 0.5,
        mask_subvoxel_warning_threshold: float = 0.01,
        mask_min_overlap_fraction: float = 0.5,
        image_options: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, pd.Series]:
        """
        Run configurations on the provided image and mask.

        Args:
            image: Path to image (str or Path) or Image object.
            mask: Optional path to mask (str or Path) or Image object.
                If omitted (or passed as `None` / empty string), the pipeline will
                treat the **entire image** as the ROI by generating a full (all-ones)
                mask matching the input image geometry.
            subject_id: Optional identifier for the subject (used in the
                processing log only; not included in the returned feature Series).
            config_names: List of specific configuration names to run.
                          If None, runs all registered configurations.
                          Supports "all_standard" to run all 6 standard configs.
            mask_subvoxel_tolerance: Maximum permitted fractional-voxel offset when
                repositioning a mask path (default: 0.5). Has no effect when mask is
                a pre-loaded Image object. See ``load_image`` for full description.
            mask_subvoxel_warning_threshold: Fractional-voxel drift above which a
                ``UserWarning`` is emitted during mask repositioning (default: 0.01).
                Has no effect when mask is a pre-loaded Image object.
            mask_min_overlap_fraction: Minimum fraction of the mask volume that must
                intersect with the image space when loading from a path (default: 0.5).
                Has no effect when mask is a pre-loaded Image object.
            image_options: Options of ``load_image`` for an image path, for example
                ``{"dataset_index": 4}`` (the 5th cardiac phase of a DICOM folder),
                ``{"suv": "bw"}`` (a PET series as SUV) or ``{"series_uid": "..."}``.
                The log entry records them. An Image takes no options.

        Returns:
            Dictionary mapping config names to pandas Series of features.
            Every Series contains the **complete set of expected feature names**
            for its configuration, regardless of whether extraction succeeded:

            - If extraction succeeds, values are the computed feature values.
            - If individual features fail (e.g., mesh error, PCA with ≤3 voxels),
                those features are ``NaN``; successfully computed features are preserved.
            - If a feature family fails, its features are ``NaN``, a warning names it,
                and the log entry lists it in ``family_errors``.
            - If the entire configuration fails (empty ROI or unexpected error),
                all values are ``NaN``.

        Raises:
            ValueError: If the loaded mask does not match the image's geometry
                (shape, spacing, origin, or direction), or if `config_names`
                includes a name that is not registered (and is not
                "all_standard"). Errors raised while executing an individual
                configuration's steps are caught internally and reported as
                ``NaN`` features for that configuration instead of propagating.

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
        mask_settings = (
            mask_subvoxel_tolerance,
            mask_subvoxel_warning_threshold,
            mask_min_overlap_fraction,
        )
        orig_img, img_source = self._run_image(image, image_options)
        orig_mask, mask_source, mask_was_generated = self._run_mask(mask, orig_img, mask_settings)
        _validate_geometry(orig_mask, orig_img, "mask", "image")
        if isinstance(config_names, str):
            config_names = [config_names]
        return self._run_loaded(
            orig_img,
            orig_mask,
            img_source,
            mask_source,
            mask_was_generated,
            subject_id,
            config_names,
            self._target_configs(config_names),
            mask_settings,
            image_options=image_options,
        )

    @staticmethod
    def _run_image(
        image: str | Path | Image, image_options: Optional[Mapping[str, Any]] = None
    ) -> tuple[Image, str]:
        """The image of a run (a float64 array) and its source for the log; an image path
        loads with the `load_image` options `image_options`."""
        if isinstance(image, (str, Path)):
            orig_img = load_image(str(image), **(image_options or {}))
            img_source = str(image)
        elif isinstance(image, Image):
            if image_options:
                raise ValueError("image_options apply to an image path, not to an Image.")
            orig_img = image
            img_source = "InMemory"
        else:
            raise TypeError(
                f"image must be a path or an Image, not {type(image).__name__}; wrap an "
                "array as Image(array, spacing, origin)."
            )
        if orig_img.array.dtype != np.float64:
            # The loaders give float64. An integer or float32 array would resample in its
            # own type (rounded, and on one core), so an in-memory image gets float64 too.
            orig_img = replace(orig_img, array=orig_img.array.astype(np.float64))
        return orig_img, img_source

    @staticmethod
    def _run_mask(
        mask: str | Path | Image | None,
        orig_img: Image,
        mask_settings: tuple[float, float, float],
    ) -> tuple[Image, str, bool]:
        """The mask of a run on the grid of `orig_img`, its source for the log, and whether
        it is a generated full mask."""
        if isinstance(mask, Path):
            mask = str(mask)
        if mask is None or (isinstance(mask, str) and mask.strip() == ""):
            return create_full_mask(orig_img), "GeneratedFullMask", True
        if isinstance(mask, str):
            tolerance, warning_threshold, overlap = mask_settings
            loaded = load_image(
                mask,
                reference_image=orig_img,
                subvoxel_tolerance=tolerance,
                subvoxel_warning_threshold=warning_threshold,
                min_overlap_fraction=overlap,
            )
            return loaded, mask, False
        if isinstance(mask, Image):
            return mask, "InMemory", False
        raise TypeError(
            f"mask must be a path, an Image or None, not {type(mask).__name__}; wrap an "
            "array as Image(array, spacing, origin)."
        )

    def _run_loaded(
        self,
        orig_img: Image,
        orig_mask: Image,
        img_source: str,
        mask_source: str,
        mask_was_generated: bool,
        subject_id: Optional[str],
        config_names: Optional[list[str]],
        target_configs: list[str],
        mask_settings: tuple[float, float, float],
        nonfinite: Optional[bool] = None,
        roi_nearest: Optional[tuple[Image, int]] = None,
        image_options: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, pd.Series]:
        """The part of `run()` after the loading: `target_configs` run (see
        `_target_configs`), and the log records the requested `config_names` and the
        `image_options` of the image load. `nonfinite` (whether the image has NaN or
        infinite values) is found when it is None. `roi_nearest`: see
        PipelineState.roi_nearest."""
        mask_subvoxel_tolerance, mask_subvoxel_warning_threshold, mask_min_overlap_fraction = (
            mask_settings
        )
        all_results = {}
        # Every voxel that is not 0 is ROI, also a voxel of 0.05 in a probability map. The
        # box scan of this check also serves as the first ROI check of the run.
        roi_box = None
        fractions = False
        if (
            not mask_was_generated
            and orig_mask.array.dtype.kind == "f"
            and any(
                all(step["step"] != "binarize_mask" for step in self._configs[name])
                for name in target_configs
            )
        ):
            if orig_mask.array.size < _FRACTION_SAMPLE:  # a small mask: read it all
                fractions = not np.array_equal(orig_mask.array, np.round(orig_mask.array))
            else:
                roi_box = compute_nonzero_bbox(orig_mask.array)
                fractions = roi_box is not None and _has_fractions(orig_mask.array, roi_box)
        if fractions:
            warnings.warn(
                "The mask holds values that are not whole numbers (for example a probability "
                "map), and every voxel that is not 0 counts as ROI. To choose the ROI, add a "
                "binarize_mask step with a threshold, for example 0.5.",
                UserWarning,
                stacklevel=3,
            )

        # Create or regenerate deduplication plan if enabled
        dedup_plan: DeduplicationPlan | None = None
        family_cache: dict[tuple[str, str], dict[str, Any]] = {}

        # Reset deduplication statistics for this run
        self._dedup_reused_count = 0
        self._dedup_computed_count = 0
        self._last_distance_map = None
        self._roi_cuts.clear()
        # NaN or infinite intensities leave the intensity mask before each extraction
        if nonfinite is None:
            nonfinite = not _all_finite(orig_img.array)

        if self._deduplication_enabled and len(target_configs) > 1:
            # Get configs for analysis
            configs_to_analyze = {name: self._configs[name] for name in target_configs}
            config_metadata = {name: self._config_metadata.get(name, {}) for name in target_configs}
            # The plan of an earlier run holds while the rules and the pickled configs
            # (names in order, steps and metadata) are the same.
            try:
                plan_key: bytes | None = pickle.dumps((configs_to_analyze, config_metadata))
            except Exception:  # a parameter that pickle cannot write: plan anew
                plan_key = None
            cached = self._plan_cache
            if (
                plan_key is not None
                and cached is not None
                and cached[0] is self._deduplication_rules
                and cached[1] == plan_key
            ):
                dedup_plan = cached[2]
            else:
                analyzer = ConfigurationAnalyzer(
                    configs_to_analyze,
                    self._deduplication_rules,
                    config_metadata=config_metadata,
                )
                dedup_plan = analyzer.analyze()
                if plan_key is not None:
                    self._plan_cache = (self._deduplication_rules, plan_key, dedup_plan)
            self._last_deduplication_plan = dedup_plan
            self._configs_modified_since_plan = False

        # Configurations share identical preprocessing: the state after a step prefix
        # that a later configuration repeats is kept until its last user.
        prefix_keys = {
            name: _prefix_keys(self._configs[name], self._config_metadata.get(name, {}))
            for name in target_configs
        }
        users = Counter(key for name in target_configs for key in prefix_keys[name])
        shared: dict[str, tuple[PipelineState, list[dict[str, Any]]]] = {}

        # Every configuration starts from the same image and mask, so the first ROI check,
        # the sentinel search (value, count) and the source mask of each source setup
        # (source mode, sentinel value) are done once per run. A source mask is kept
        # until the last configuration that uses it.
        roi_checked = False
        detection: Optional[tuple[Optional[float], int]] = None
        source_masks: dict[tuple[Any, Any], Image] = {}
        source_users = Counter(
            (metadata.get("source_mode", "full_image"), metadata.get("sentinel_value"))
            for metadata in (self._config_metadata.get(name, {}) for name in target_configs)
        )

        # Run each configuration
        for config_name in target_configs:
            started = time.perf_counter()
            self._family_errors = {}
            steps = self._configs[config_name]
            metadata = self._config_metadata.get(config_name, {})
            keys = prefix_keys[config_name]

            # Determine source mode for this config
            source_mode_str = metadata.get("source_mode", "full_image")
            source_mode = SourceMode(source_mode_str)
            explicit_sentinel = metadata.get("sentinel_value")
            source_key = (source_mode_str, explicit_sentinel)
            source_users[source_key] -= 1

            # Determine source mask based on source_mode
            source_mask: Optional[Image] = None
            sentinel_detected = False
            detected_sentinel_value: Optional[float] = None
            sentinel_auto_detected = False
            sentinel_proportion: Optional[float] = None

            if source_mode == SourceMode.FULL_IMAGE:
                # Default: all voxels valid, no source_mask needed
                pass

            elif source_mode == SourceMode.ROI_ONLY:
                # Use ROI mask as source mask
                if source_key not in source_masks:
                    source_masks[source_key] = _source_mask(orig_mask.array != 0, orig_mask)
                source_mask = source_masks[source_key]

            elif source_mode == SourceMode.AUTO:
                # Auto-detect sentinel values
                if explicit_sentinel is not None:
                    # User provided explicit sentinel value: accept it regardless
                    # of how much of the image it covers.
                    detected_sentinel_value = explicit_sentinel
                    sentinel_detected = True
                else:
                    if detection is None:
                        # If mask was auto-generated (full mask), do not use it for
                        # "outside-ness" check in detection, as everything is "inside".
                        mask_for_detection = orig_mask if not mask_was_generated else None
                        found = detect_sentinel_value(orig_img, roi_mask=mask_for_detection)
                        count = (
                            0 if found is None else int(np.count_nonzero(orig_img.array == found))
                        )
                        detection = (found, count)
                    detected, n_sentinel = detection
                    if detected is not None:
                        detected_sentinel_value = detected
                        sentinel_detected = True
                        sentinel_auto_detected = True
                        n_total = int(orig_img.array.size)
                        sentinel_proportion = n_sentinel / n_total

                        # Only ever print "100.0%" when literally every voxel is the
                        # sentinel. Plain rounding turns e.g. 99.999% into "100.0%",
                        # which wrongly implies no voxels remain for feature
                        # extraction. Report the surviving voxel count too, so the
                        # amount of usable data is unambiguous.
                        percent = sentinel_proportion * 100.0
                        percent_str = (
                            f"{percent:.1f}%"
                            if n_sentinel == n_total or percent < 99.95
                            else ">99.9%"
                        )

                        # Prominent warning: auto-detection is a heuristic. The user
                        # should confirm the value is a genuine fill/padding value and
                        # not real image data.
                        msg = (
                            f"Auto-detected sentinel value {detected} "
                            f"({percent_str} of voxels; {n_total - n_sentinel:,} of "
                            f"{n_total:,} voxels remain valid) for config "
                            f"'{config_name}'; these voxels will be excluded from "
                            f"resampling/filtering. Verify this is a padding value and "
                            f"not real image data, or set sentinel_value explicitly."
                        )
                        logging.warning(msg)
                        warnings.warn(msg, stacklevel=2)
                    else:
                        # Nothing crossed the threshold. Warn and continue treating
                        # the whole image as valid (no source mask), so batch runs do
                        # not break on images that simply have no sentinel.
                        msg = (
                            f"No sentinel value auto-detected for config "
                            f"'{config_name}' (no candidate reached the presence "
                            f"threshold). Proceeding with the full image; set "
                            f"sentinel_value explicitly if the image is pre-masked "
                            f"with a padding value."
                        )
                        logging.warning(msg)
                        warnings.warn(msg, stacklevel=2)

                if sentinel_detected and detected_sentinel_value is not None:
                    if source_key not in source_masks:
                        # create_source_mask_from_sentinel as bool, with no uint8 copy
                        source_masks[source_key] = _source_mask(
                            orig_img.array != detected_sentinel_value, orig_img
                        )
                    source_mask = source_masks[source_key]
            if source_users[source_key] == 0:
                source_masks.pop(source_key, None)

            # Initialize State with source tracking
            # We start with fresh copies for each config
            state = PipelineState(
                image=orig_img,
                raw_image=orig_img,  # Track non-discretised image
                morph_mask=orig_mask,
                # Share the same array object as morph_mask (rather than a copy) so an
                # `is`-identity check can detect when the masks are still in sync; no
                # code in this package mutates mask/image arrays in place, so aliasing
                # is safe (see _execute_preprocessing_step's resample branch).
                intensity_mask=Image(
                    array=orig_mask.array,
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
                roi_nearest=roi_nearest,
            )

            snapshot = self._config_snapshot(config_name)
            config_log: dict[str, Any] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "schema_version": CONFIG_SCHEMA_VERSION,
                "pictologics_version": _get_package_version(),
                "environment": {**_package_versions(), "threads": numba.get_num_threads()},
                "subject_id": subject_id,
                "config_name": config_name,
                "config_hash": self._hash_of(config_name),
                "image_source": img_source,
                "image_options": self._make_serializable(dict(image_options or {})),
                "mask_source": mask_source,
                "source_mode": source_mode.value,
                "sentinel_detected": sentinel_detected,
                "sentinel_value": detected_sentinel_value,
                "sentinel_auto_detected": sentinel_auto_detected,
                "sentinel_proportion": sentinel_proportion,
                "mask_roi_semantics": "nonzero_values_are_roi_membership",
                "config_snapshot": {
                    "source_mode": snapshot["source_mode"],
                    "sentinel_value": snapshot["sentinel_value"],
                    "effective_sentinel_value": self._make_serializable(detected_sentinel_value),
                    "steps": snapshot["steps"],
                },
                "deduplication": {
                    "enabled": self._deduplication_enabled,
                    "rules_version": self._deduplication_rules.version,
                    "plan_used": dedup_plan is not None,
                },
                "run_parameters": {
                    "requested_config_names": config_names,
                    "target_configs": target_configs,
                },
                "mask_repositioning_settings": {
                    "subvoxel_tolerance": mask_subvoxel_tolerance,
                    "subvoxel_warning_threshold": mask_subvoxel_warning_threshold,
                    "min_overlap_fraction": mask_min_overlap_fraction,
                },
                "status": "started",
                "steps_executed": [],
            }

            config_features: dict[str, Any] = {}
            current_step: dict[str, Any] | None = None

            try:
                if not roi_checked:
                    if not (
                        roi_box is not None
                        and state.intensity_mask.array is orig_mask.array
                        and state.morph_mask.array is orig_mask.array
                    ):
                        self._ensure_nonempty_roi(state, context="initialization")
                    roi_checked = True

                # Continue from the longest prefix that an earlier configuration ran.
                start = 0
                for k in range(len(keys), 0, -1):
                    if keys[k - 1] in shared:
                        shared_state, shared_log = shared[keys[k - 1]]
                        state = replace(shared_state)
                        config_log["steps_executed"].extend(copy.deepcopy(shared_log))
                        start = k
                        break

                for index, step_def in enumerate(steps[start:], start):
                    current_step = step_def
                    step_name = step_def["step"]
                    params = step_def.get("params", {})
                    if step_name in ("filter", "resample"):
                        state.roi_reach = _roi_reach(steps[index + 1 :])
                    elif step_name == "discretise":  # a small image is not cut
                        state.roi_reach = (
                            _roi_reach(steps[index + 1 :])
                            if state.image.array.size >= _CUT_MIN_SIZE
                            else None
                        )

                    # Execute Step
                    if step_name == "extract_features":
                        extract_state = (
                            _finite_intensity_mask(state, config_name) if nonfinite else state
                        )
                        # Use deduplication if plan exists
                        if dedup_plan is not None:
                            features = self._extract_features_with_dedup(
                                extract_state, params, config_name, dedup_plan, family_cache
                            )
                        else:
                            features = self._extract_features(extract_state, params)
                        config_features.update(features)
                    else:
                        self._execute_preprocessing_step(state, step_name, params)

                    # Log
                    step_log_entry: dict[str, Any] = {
                        "step": step_name,
                        "params": self._make_serializable(params),
                        "status": "completed",
                    }
                    if step_name == "discretise" and params.get("method") == "FBS":
                        step_log_entry["min_val_effective"] = state.discretisation_min
                    if step_name == "normalise":
                        center, scale = cast(tuple[float, float], state.normalisation)
                        step_log_entry["center_effective"] = center
                        step_log_entry["scale_effective"] = scale
                    if step_name == "filter":
                        step_log_entry["boundary_requested"] = state.filter_boundary_requested
                        step_log_entry["boundary_effective"] = state.filter_boundary_effective
                        step_log_entry["params_requested"] = state.filter_params_requested
                        step_log_entry["params_effective"] = state.filter_params_effective
                    config_log["steps_executed"].append(step_log_entry)
                    if (
                        index < len(keys)
                        and users[keys[index]] > 1
                        and keys[index] not in shared
                        # a later config starts here, not only past here: else the
                        # state (maybe a full image) would stay unused
                        and users[keys[index]]
                        > (users[keys[index + 1]] if index + 1 < len(keys) else 0)
                    ):
                        shared[keys[index]] = (replace(state), list(config_log["steps_executed"]))
                config_log["status"] = "completed"
                config_log["result_feature_count"] = len(config_features)

            except EmptyROIMaskError as e:
                config_log["status"] = "empty_roi"
                config_log["error"] = str(e)
                config_log["failed_step"] = (
                    current_step if current_step is not None else "initialization"
                )
                config_log["elapsed_seconds"] = time.perf_counter() - started
                self._log.append(config_log)

                # Build a NaN-filled Series with the expected feature names so
                # that downstream formatting/concatenation always sees a
                # complete, predictable set of columns.
                nan_names = self._get_expected_feature_names(steps)
                all_results[config_name] = pd.Series({name: float("nan") for name in nan_names})
                config_log["result_feature_count"] = len(nan_names)
                logging.debug(
                    "Config '%s' produced an empty ROI: %s. Returning NaN for %d features.",
                    config_name,
                    e,
                    len(nan_names),
                )
                continue

            except Exception as e:
                config_log["status"] = "error"
                config_log["error"] = str(e)
                config_log["failed_step"] = current_step
                # Backfill with NaN so the result always has a complete set of
                # feature columns, even when extraction was interrupted.
                nan_names = self._get_expected_feature_names(steps)
                for name in nan_names:
                    config_features.setdefault(name, float("nan"))
                config_log["result_feature_count"] = len(config_features)
            finally:
                for key in keys:
                    users[key] -= 1
                    if users[key] == 0:
                        shared.pop(key, None)
                if self._family_errors:
                    config_log["family_errors"] = dict(self._family_errors)

            config_log["elapsed_seconds"] = time.perf_counter() - started
            self._log.append(config_log)

            # Create Series
            series = pd.Series(config_features)
            all_results[config_name] = series

        self._last_distance_map = None
        self._roi_cuts.clear()
        return all_results

    def run_rois(
        self,
        image: str | Path | Image,
        rois: str | Path | Image,
        labels: Optional[Iterable[float] | Mapping[str, float]] = None,
        subject_id: Optional[str] = None,
        config_names: Optional[list[str]] = None,
        mask_subvoxel_tolerance: float = 0.5,
        mask_subvoxel_warning_threshold: float = 0.01,
        mask_min_overlap_fraction: float = 0.5,
        image_options: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, dict[str, pd.Series]]:
        """
        Run configurations on each ROI of a label map, with one image load.

        The label map `rois` holds a whole number for each voxel: 0 for the background,
        and the label of its ROI elsewhere. Each ROI gets the results of `run()` with a
        mask of that label alone. `run_rois` loads the image once, checks it for NaN
        values once, and makes each mask only inside the box of its label, so many ROIs
        take much less time than one `run()` for each of them. A `grow_mask` step with
        `nearest_roi` gives each added voxel to the ROI of the map with the nearest voxel,
        so the rings of touching ROIs do not overlap.

        Args:
            image: Path to the image, or an Image.
            rois: Path to the label map (for example a NIfTI file, a DICOM SEG, or an
                RTSTRUCT, whose labels are its ROI Numbers), or an Image, on the grid of the
                image.
            labels: The ROIs to run: the labels, or a mapping from ROI names to labels.
                Default: every label in the map.
            subject_id: The subject, for the processing log.
            config_names: The configurations to run, as in `run()`.
            mask_subvoxel_tolerance: As in `run()`, for a label map path.
            mask_subvoxel_warning_threshold: As in `run()`, for a label map path.
            mask_min_overlap_fraction: As in `run()`, for a label map path.
            image_options: As in `run()`: options of `load_image` for an image path.

        Returns:
            For each ROI name (the label as text, such as `"3"`, or the name of the
            mapping), the results of `run()`: a dictionary from configuration names to
            feature Series. The log entries of an ROI have its name in `roi`.

        Raises:
            ValueError: If the label map has values that are not whole numbers or are
                below 0, or if a label is not a whole number of 1 or above.

        Example:
            ```python
            results = pipeline.run_rois("ct.nii.gz", "organs.nii.gz", labels={"liver": 5, "spleen": 1})
            rows = [
                format_results(series, meta={"subject_id": "p001", "roi": roi})
                for roi, series in results.items()
            ]
            save_results(rows, "p001_rois.csv")
            ```
        """
        from scipy import ndimage

        mask_settings = (
            mask_subvoxel_tolerance,
            mask_subvoxel_warning_threshold,
            mask_min_overlap_fraction,
        )
        orig_img, img_source = self._run_image(image, image_options)
        label_map, mask_source, _ = self._run_mask(rois, orig_img, mask_settings)
        _validate_geometry(label_map, orig_img, "mask", "image")
        values = label_map.array
        whole = values if values.dtype.kind in "ui" else values.astype(np.int64)
        if (whole is not values and not np.array_equal(whole, values)) or whole.min() < 0:
            raise ValueError("The labels of an ROI map must be whole numbers of 0 or above.")
        boxes = ndimage.find_objects(whole)  # the box of each label 1, 2, ..., in one pass
        chosen: dict[str, float]
        if labels is None:
            chosen = {str(k + 1): k + 1 for k, box in enumerate(boxes) if box is not None}
        elif isinstance(labels, Mapping):
            chosen = {str(name): label for name, label in labels.items()}
        else:
            chosen = {str(label): label for label in labels}
        for label in chosen.values():
            if label < 1 or label != int(label):
                raise ValueError(f"An ROI label must be a whole number of 1 or above, not {label}.")
        if isinstance(config_names, str):
            config_names = [config_names]
        names = self._target_configs(config_names)
        nonfinite = not _all_finite(orig_img.array)
        # One mask array for all ROIs: each run fills the box of its label and clears it
        # after (a run keeps no array of its masks)
        buffer: npt.NDArray[Any] = np.zeros(values.shape, dtype=np.uint8)
        mask = replace(label_map, array=buffer)
        # A grow_mask step with nearest_roi gives each added voxel to its nearest ROI
        splits = [self._configs[name] for name in names if _splits_rois(self._configs[name])]
        nearest = (
            _nearest_roi_map(replace(label_map, array=whole), max(map(_grow_reach, splits)))
            if splits
            else None
        )
        all_results: dict[str, dict[str, pd.Series]] = {}
        for name, label in chosen.items():
            box = boxes[int(label) - 1] if label <= len(boxes) else None
            if box is not None:
                buffer[box] = whole[box] == label
            start = len(self._log)
            all_results[name] = self._run_loaded(
                orig_img,
                mask,
                img_source,
                mask_source,
                False,
                subject_id,
                config_names,
                names,
                mask_settings,
                nonfinite,
                None if nearest is None else (nearest, int(label)),
                image_options,
            )
            for entry in self._log[start:]:
                entry["roi"] = name
            if box is not None:
                buffer[box] = 0
        return all_results

    def run_batch(
        self,
        cases: Iterable[Mapping[str, Any]] | pd.DataFrame,
        output_dir: str | Path,
        config_names: Optional[list[str]] = None,
        workers: int = 1,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Run configurations on many cases, with one result file for each case.

        Each case is a mapping, or a row of a DataFrame, with the `run()` arguments of one
        image: `subject_id` and `image` (required), `mask`, the mask settings and
        `image_options` (for example a cardiac phase or SUV). A case with a label map
        gives `rois` (and optionally `labels`) in place of `mask`: it runs `run_rois()`,
        so each ROI gets its results (and `grow_mask` steps with `nearest_roi` share the
        rings). When a case ends, `run_batch` writes its result to
        `output_dir/cases/<subject_id>.json`. A later call with the same output folder
        skips each case whose file holds the same image, image options, mask or label map
        and labels, and configurations, so a stopped batch goes on where it stopped.
        A failed case runs again. To run a case again, delete its file.

        With `workers` above 1, the cases run in that many processes, and each process
        uses its share of the numba threads. Each process holds one case at a time, so
        the memory need grows with `workers`.

        Args:
            cases: The cases.
            output_dir: The folder of the result files.
            config_names: The configurations to run, as in `run()`.
            workers: The number of processes.
            show_progress: Whether to show a progress bar.

        Returns:
            A DataFrame with one row for each case, in the order of `cases` (for a case
            with a label map, one row for each ROI, with its name in `roi`):
            `subject_id`; `status` (`"completed"`; `"incomplete"` when a configuration
            ended with an empty ROI or an error; `"failed"` when the case did not run, for
            example because its image did not load); `error`; `warnings`; `seconds`; and
            the features in the wide format of `format_results()`. The file of a case also
            holds the processing log of its configurations.

        Raises:
            ValueError: If a case has no `subject_id` or `image`, has an unknown key, or
                has the file name of another case, or for an unknown configuration name.

        Example:
            ```python
            cases = [
                {"subject_id": "p001", "image": "p001/ct.nii.gz", "mask": "p001/roi.nii.gz"},
                {"subject_id": "p002", "image": "p002/ct.nii.gz", "mask": "p002/roi.nii.gz"},
            ]
            table = pipeline.run_batch(cases, "results", config_names=["study"], workers=4)
            print(table.loc[table["status"] != "completed", ["subject_id", "error"]])
            save_results(table, "results/features.csv")
            ```
        """
        from tqdm import tqdm

        records = (
            cases.to_dict("records")
            if isinstance(cases, pd.DataFrame)
            else [dict(case) for case in cases]
        )
        names = self._target_configs(config_names)
        hashes = {name: self._hash_of(name) for name in names}
        folder = Path(output_dir) / "cases"
        folder.mkdir(parents=True, exist_ok=True)
        keys = (
            set(inspect.signature(self.run).parameters)
            | set(inspect.signature(self.run_rois).parameters)
        ) - {"config_names"}
        paths: list[Path] = []
        stems: dict[str, str] = {}
        for index, case in enumerate(records):
            for key, value in case.items():
                if key not in keys:
                    raise ValueError(f"Case {index}: unknown key '{key}'{_hint(key, keys)}.")
                if isinstance(value, float) and math.isnan(value):  # an empty DataFrame cell
                    case[key] = None
            if case.get("subject_id") is None or case.get("image") is None:
                raise ValueError(f"Case {index} needs a subject_id and an image.")
            if case.get("mask") is not None and case.get("rois") is not None:
                raise ValueError(f"Case {index} gives a mask and rois: give one of them.")
            case["subject_id"] = str(case["subject_id"])
            stem = re.sub(r"[^A-Za-z0-9._-]", "_", case["subject_id"])
            if stem in stems:
                raise ValueError(
                    f"The cases '{stems[stem]}' and '{case['subject_id']}' have the same "
                    f"result file name, {stem}.json."
                )
            stems[stem] = case["subject_id"]
            paths.append(folder / f"{stem}.json")

        outcomes: dict[int, dict[str, Any]] = {}
        for index, (case, path) in enumerate(zip(records, paths, strict=True)):
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if (
                    record["status"] != "failed"
                    and record["config_hashes"] == hashes
                    and all(record.get(k) == v for k, v in _case_identity(case).items())
                ):
                    outcomes[index] = record
        todo = [index for index in range(len(records)) if index not in outcomes]
        setup = (
            {name: self._configs[name] for name in names},
            {name: self._config_metadata[name] for name in names if name in self._config_metadata},
            self._deduplication_enabled,
            self._deduplication_rules,
        )
        with tqdm(
            total=len(records),
            initial=len(outcomes),
            desc="Radiomics",
            unit="case",
            disable=not show_progress,
        ) as bar:
            if workers <= 1 or len(todo) <= 1:
                runner = _batch_pipeline(*setup)
                for index in todo:
                    outcomes[index] = runner._run_case(records[index], names, hashes, paths[index])
                    bar.update()
            else:
                from .utilities.dicom_utils import worker_pool

                threads = max(1, numba.get_num_threads() // workers)
                with worker_pool(
                    min(workers, len(todo)),
                    initializer=_start_batch_worker,
                    initargs=(setup, threads),
                    spawn=True,
                ) as pool:
                    futures = {
                        pool.submit(_batch_case, records[i], names, hashes, paths[i]): i
                        for i in todo
                    }
                    try:
                        for future in as_completed(futures):
                            outcomes[futures[future]] = future.result()
                            bar.update()
                    except BaseException:  # a stop (Ctrl+C) or a lost worker: no new cases
                        for future in futures:
                            future.cancel()
                        raise

        rows = []
        for index in range(len(records)):
            record = outcomes[index]
            meta = {
                "subject_id": record["subject_id"],
                "status": record["status"],
                "error": record["error"],
                "warnings": " | ".join(record["warnings"]) or None,
                "seconds": record["seconds"],
            }
            # A label map case: one row for each ROI (one row without ROIs when it failed)
            by_roi = record["results"] if record.get("rois") else {None: record["results"]}
            for roi, values in (by_roi or {None: {}}).items():
                results = {
                    name: pd.Series(features, dtype=float) for name, features in values.items()
                }
                extra = {"roi": roi} if record.get("rois") else {}
                rows.append(format_results(results, fmt="wide", meta={**meta, **extra}))
        return pd.DataFrame(rows)

    def _run_case(
        self,
        case: dict[str, Any],
        names: list[str],
        hashes: dict[str, str],
        path: Path,
    ) -> dict[str, Any]:
        """Run one case of `run_batch`, write its record to `path` and return the record.
        The file appears as a whole (from a temporary file), so a stop leaves no part."""
        started = time.perf_counter()
        record: dict[str, Any] = {
            "subject_id": case["subject_id"],
            **_case_identity(case),
            "config_hashes": hashes,
        }
        label_map = case.get("rois") is not None
        arguments = {k: v for k, v in case.items() if k not in _OTHER_CASE_KEYS[label_map]}

        def plain(results: dict[str, pd.Series]) -> dict[str, dict[str, float]]:
            return {
                name: {key: float(value) for key, value in series.items()}
                for name, series in results.items()
            }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if label_map:
                    by_roi = self.run_rois(config_names=names, **arguments)
                    values: dict[str, Any] = {roi: plain(res) for roi, res in by_roi.items()}
                else:
                    values = plain(self.run(config_names=names, **arguments))
            except Exception as e:
                record.update(status="failed", error=f"{type(e).__name__}: {e}", results={})
            else:

                def where(entry: dict[str, Any]) -> str:
                    roi = f"ROI {entry['roi']}, " if "roi" in entry else ""
                    return f"{roi}{entry['config_name']}"

                problems = [
                    f"{where(entry)}: {entry['error']}"
                    for entry in self._log
                    if entry["status"] != "completed"
                ] + [
                    f"{where(entry)} ({family}): {error}"
                    for entry in self._log
                    for family, error in entry.get("family_errors", {}).items()
                ]
                record.update(
                    status="incomplete" if problems else "completed",
                    error="; ".join(problems) or None,
                    results=values,
                )
        record["warnings"] = list(dict.fromkeys(str(warning.message) for warning in caught))
        record["seconds"] = time.perf_counter() - started
        record["log"] = self._make_serializable(self._log)
        self.clear_log()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(_json_safe(record), default=str, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
        return record

    def _target_configs(self, config_names: Optional[list[str]]) -> list[str]:
        """The configurations that `run()` runs for `config_names`: each name once, with
        "all_standard" for the standard ones; all configurations for None (with a warning
        when the standard ones are among them)."""
        if config_names is None:
            standard = self.get_all_standard_config_names()
            if standard:
                warnings.warn(
                    f"run() without config_names runs all {len(self._configs)} configurations, "
                    f"also the {len(standard)} standard ones ({', '.join(standard)}). Pass "
                    "config_names to choose, or create the pipeline with "
                    "RadiomicsPipeline(load_standard=False).",
                    UserWarning,
                    stacklevel=3,
                )
            return list(self._configs)
        target_configs = []
        for name in config_names:
            if name == "all_standard":
                target_configs.extend(self.get_all_standard_config_names())
            elif name in self._configs:
                target_configs.append(name)
            else:
                raise ValueError(f"Configuration '{name}' not found.{_hint(name, self._configs)}")
        return list(dict.fromkeys(target_configs))  # each name runs once

    def _config_snapshot(self, name: str) -> dict[str, Any]:
        """The source mode, sentinel value and steps of a configuration (JSON-ready): the
        part of its log snapshot that the config hash covers."""
        metadata = self._config_metadata.get(name, {})
        snapshot: dict[str, Any] = self._make_serializable(
            {
                "source_mode": SourceMode(metadata.get("source_mode", "full_image")).value,
                "sentinel_value": metadata.get("sentinel_value"),
                "steps": self._configs[name],
            }
        )
        return snapshot

    def _hash_of(self, name: str) -> str:
        """The config hash of a configuration (see `_config_hash`), made again only after
        the configuration changed."""
        steps = self._configs[name]
        cached = self._config_hashes.get(name)
        if cached is None or cached[0] is not steps:
            cached = (steps, _config_hash(self._config_snapshot(name)))
            self._config_hashes[name] = cached
        return cached[1]

    def get_log(self) -> list[dict[str, Any]]:
        """
        Return a copy of the processing log.

        The log has one entry for each configuration that `run()` or `run_rois()` ran, in
        run order (`run_batch()` writes the log of each case into its result file). An
        entry holds the configuration name and its `config_hash`, the subject, the image
        and mask sources, the `image_options`, the status (`"completed"`, `"empty_roi"`
        or `"error"`), the error and the failed step when there is one, the errors of
        single feature families (`family_errors`), the executed steps, the feature count,
        the run time (`elapsed_seconds`), the `environment` (the versions of Python and
        the packages, and the thread count) and, for `run_rois()`, the `roi`. The log
        grows with each run until `clear_log()`.

        Example:
            ```python
            results = pipeline.run(image, mask, config_names=["standard_fbn_32"])
            for entry in pipeline.get_log():
                if entry["status"] != "completed":
                    print(entry["config_name"], entry["error"])
            ```
        """
        return copy.deepcopy(self._log)

    def clear_log(self) -> None:
        """Clear the in-memory processing log."""
        self._log.clear()

    def _ensure_nonempty_roi(self, state: PipelineState, context: str) -> None:
        """Raise a clear error if the ROI is empty.

        The pipeline treats any nonzero mask value as ROI membership unless a
        step explicitly binarizes/selects labels first. Each check reads the whole
        mask, so a morph mask that is the intensity mask's array is not read again.
        """
        has_intensity_roi = _has_roi(state.intensity_mask.array)
        if not has_intensity_roi:
            raise EmptyROIMaskError(
                "ROI is empty after preprocessing "
                f"({context}). Ensure your mask contains at least one nonzero voxel, "
                "or relax resegmentation/outlier filtering thresholds."
            )
        if state.morph_mask.array is state.intensity_mask.array:
            return
        has_morph_roi = _has_roi(state.morph_mask.array)
        if not has_morph_roi:
            raise EmptyROIMaskError(
                "ROI is empty after preprocessing "
                f"({context}). Ensure your mask contains at least one nonzero voxel, "
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
            if state.source_mode != SourceMode.FULL_IMAGE and state.source_mask is not None:
                source_mask_arg = state.source_mask

            # When no later step reads the new grid away from the ROI, the resample computes
            # only the region around the ROI box (grown by the reach of the later steps): the
            # arrays then hold that region, with the values of the whole grid.
            region = None
            grid = _output_grid(state.image.array.shape, state.image.spacing, spacing)[0]
            if state.roi_reach is not None:
                margin_mm = state.roi_reach
                region = _roi_region(
                    cast(
                        tuple[slice, slice, slice],
                        merge_bboxes(
                            compute_nonzero_bbox(state.morph_mask.array),
                            compute_nonzero_bbox(state.intensity_mask.array),
                        ),
                    ),  # the ROI checks keep the masks non-empty
                    state.image.array.shape,
                    state.image.spacing,
                    spacing,
                    cast(
                        tuple[int, int, int],
                        tuple(
                            math.ceil(margin_mm / s) + 1 if margin_mm > 0 else 0 for s in spacing
                        ),
                    ),
                )
                if all(r.stop - r.start == n for r, n in zip(region, grid, strict=True)):
                    region = None
                else:
                    state.grid_shape = (int(grid[0]), int(grid[1]), int(grid[2]))
                    state.grid_offset = (region[0].start, region[1].start, region[2].start)
            _check_grid_memory(
                [int(n) for n in grid] if region is None else [r.stop - r.start for r in region],
                spacing,
            )

            # Update Image and raw_image
            state.image = resample_image(
                state.image,
                spacing,
                interpolation=interp_img,
                round_intensities=round_intensities_flag,
                source_mask=source_mask_arg,
                region=region,
            )
            state.raw_image = state.image  # Keep raw_image in sync before discretisation

            # Propagate source_mask from resampled image if it was used (one bool array,
            # shared: nothing changes a source mask in place)
            if state.image.has_source_mask and state.image.source_mask is not None:
                state.source_mask = _source_mask(state.image.source_mask, state.image)

            # Update Masks
            thresh_arg = mask_thresh if interp_mask != "nearest" else None
            # morph_mask and intensity_mask are resampled with identical params
            # (resample has no apply_to), so when they still share the same array
            # object (no preceding step diverged them), resample once and reuse
            # the result for both instead of repeating the work.
            masks_in_sync = state.morph_mask.array is state.intensity_mask.array
            state.morph_mask = resample_image(
                state.morph_mask,
                spacing,
                interpolation=interp_mask,
                mask_threshold=thresh_arg,
                region=region,
            )
            if masks_in_sync:
                state.intensity_mask = state.morph_mask
            else:
                state.intensity_mask = resample_image(
                    state.intensity_mask,
                    spacing,
                    interpolation=interp_mask,
                    mask_threshold=thresh_arg,
                    region=region,
                )

            # CRITICAL: If valid source mask exists, apply it to both masks.
            # This prevents background (often 0 after resampling) from being
            # considered part of the ROI if the resegmentation range includes 0.
            # A source mask with no invalid voxel changes no mask.
            valid_mask = None if state.source_mask is None else state.source_mask.array
            if valid_mask is not None and not valid_mask.all():
                masks_in_sync = state.morph_mask.array is state.intensity_mask.array
                state.morph_mask = _intersect_mask(state.morph_mask, valid_mask)
                if masks_in_sync:
                    state.intensity_mask = state.morph_mask
                else:
                    state.intensity_mask = _intersect_mask(state.intensity_mask, valid_mask)

            self._ensure_nonempty_roi(state, context="resample")

        elif step_name == "resegment":
            range_min = params.get("range_min")
            range_max = params.get("range_max")
            apply_to = _get_apply_to(params, "resegment")

            # Masks that are one array are resegmented once and stay one array.
            masks_in_sync = state.morph_mask.array is state.intensity_mask.array
            if apply_to in ("intensity", "both"):
                state.intensity_mask = resegment_mask(
                    state.image, state.intensity_mask, range_min, range_max
                )
                if range_min is not None:
                    low = -math.inf if state.resegment_min is None else state.resegment_min
                    state.resegment_min = max(float(range_min), low)
            if apply_to == "both" and masks_in_sync:
                state.morph_mask = state.intensity_mask
            elif apply_to in ("morph", "both"):
                state.morph_mask = resegment_mask(
                    state.image, state.morph_mask, range_min, range_max
                )

            self._ensure_nonempty_roi(state, context="resegment")

        elif step_name == "filter_outliers":
            sigma = params.get("sigma", 3.0)
            apply_to = _get_apply_to(params, "filter_outliers")

            # Masks that are one array are filtered once and stay one array.
            masks_in_sync = state.morph_mask.array is state.intensity_mask.array
            if apply_to in ("intensity", "both"):
                state.intensity_mask = filter_outliers(state.image, state.intensity_mask, sigma)
            if apply_to == "both" and masks_in_sync:
                state.morph_mask = state.intensity_mask
            elif apply_to in ("morph", "both"):
                state.morph_mask = filter_outliers(state.image, state.morph_mask, sigma)

            self._ensure_nonempty_roi(state, context="filter_outliers")

        elif step_name == "round_intensities":
            state.image = round_intensities(state.image)
            state.raw_image = state.image  # Keep raw_image in sync before discretisation

        elif step_name == "keep_largest_component":
            # apply_to: "morph", "intensity", or "both" (default)
            apply_to = _get_apply_to(params, "keep_largest_component")
            # Masks that are one array are labelled once and stay one array.
            masks_in_sync = state.morph_mask.array is state.intensity_mask.array
            if apply_to in ("morph", "both"):
                state.morph_mask = keep_largest_component(state.morph_mask)
            if apply_to == "both" and masks_in_sync:
                state.intensity_mask = state.morph_mask
            elif apply_to in ("intensity", "both"):
                state.intensity_mask = keep_largest_component(state.intensity_mask)

            self._ensure_nonempty_roi(state, context="keep_largest_component")

        elif step_name == "grow_mask":
            apply_to = _get_apply_to(params, "grow_mask")
            nearest = state.roi_nearest if params.get("nearest_roi", False) else None
            valid = None if state.source_mask is None else state.source_mask.array

            def _grow(mask: Image) -> Image:
                grown = grow_mask(mask, params["to_mm"], params.get("from_mm"))
                if nearest is not None:
                    grown = _nearest_roi_part(grown, mask, *nearest)
                # A grown voxel must hold image data, as after a resample
                return grown if valid is None or valid.all() else _intersect_mask(grown, valid)

            # Masks that are one array are grown once and stay one array.
            masks_in_sync = state.morph_mask.array is state.intensity_mask.array
            if apply_to in ("morph", "both"):
                state.morph_mask = _grow(state.morph_mask)
            if apply_to == "both" and masks_in_sync:
                state.intensity_mask = state.morph_mask
            elif apply_to in ("intensity", "both"):
                state.intensity_mask = _grow(state.intensity_mask)

            self._ensure_nonempty_roi(state, context="grow_mask")

        elif step_name == "normalise":
            select = state.intensity_mask.array != 0 if params["region"] == "roi" else None
            if state.source_mask is not None:  # only voxels with image data
                valid = state.source_mask.array
                select = valid if select is None else select & valid
            state.image, center, scale = _normalised(
                state.image,
                params["method"],
                select,
                tuple(params.get("percentiles", (1.0, 99.0))),
                params.get("range_min"),
                params.get("range_max"),
            )
            state.normalisation = (center, scale)
            state.raw_image = state.image
            state.resegment_min = None  # other units

        elif step_name == "binarize_mask":
            apply_to = _get_apply_to(params, "binarize_mask")
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
                    # One pass straight into the uint8 mask (same memory order)
                    mask_arr = np.greater_equal(
                        image.array,
                        float(threshold),
                        out=np.empty_like(image.array, dtype=np.uint8),
                    )

                return Image(
                    array=mask_arr.astype(np.uint8, copy=False),
                    spacing=image.spacing,
                    origin=image.origin,
                    direction=image.direction,
                    modality=image.modality,
                )

            # When both masks are one array, binarize once and keep them in sync, so
            # later steps (resample) also work on one mask.
            masks_in_sync = state.morph_mask.array is state.intensity_mask.array
            if apply_to in ("morph", "both"):
                state.morph_mask = _binarize(state.morph_mask)
            if apply_to == "both" and masks_in_sync:
                state.intensity_mask = state.morph_mask
            elif apply_to in ("intensity", "both"):
                state.intensity_mask = _binarize(state.intensity_mask)

            self._ensure_nonempty_roi(state, context="binarize_mask")

        elif step_name == "discretise":
            # No ROI check here: only the steps that change a mask can empty it, and
            # each of them checks the masks itself.
            method = params.get("method", "FBN")

            # Avoid passing 'method' twice
            disc_params = params.copy()
            if "method" in disc_params:
                del disc_params["method"]
            if method == "FBS":
                disc_params["min_val"] = state.discretisation_min = _fbs_start(disc_params, state)

            # When no later step reads the image away from the ROI, the arrays are cut to
            # the ROI box (grown by the local intensity sphere when a later step reads it),
            # so that the binning and the features work on the box only.
            if state.roi_reach is not None:
                _cut_to_roi(state, self._roi_cuts)

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
            n_bins = params.get("n_bins")
            state.n_bins = int(n_bins) if n_bins is not None else None  # 32.0 from a file
            state.bin_width = params.get("bin_width")

            # If FBS, n_bins is dynamic: the largest bin in the ROI (one fused pass over
            # the masks, not a gather of the whole grid)
            if method == "FBS":
                found = roi_min_max(state.image.array, state.intensity_mask.array)
                if found is not None:
                    state.n_bins = int(found[1])
                else:
                    raise EmptyROIMaskError(
                        "ROI is empty after preprocessing (discretise). "
                        "Cannot infer FBS bin count from an empty ROI."
                    )
            elif method == "FIXED_CUTOFFS":
                cutoffs_param = params.get("cutoffs")
                if cutoffs_param is not None:
                    # N_g = len(cutoffs) + 1: values below the first cutoff map to
                    # bin 1, values >= the last cutoff to bin len(cutoffs) + 1.
                    state.n_bins = len(cutoffs_param) + 1

        elif step_name == "filter":
            # Apply image filter
            filter_type = params.get("type")
            if not filter_type:
                raise ValueError("Filter step requires 'type' parameter.")

            # Get boundary condition (default: mirror per IBSI 2). `constant` and
            # `wrap` are accepted aliases for `zero`/`periodic`; anything else
            # unrecognised raises (never a silent fallback to mirror).
            boundary_requested = "boundary" in params
            boundary_str = params.get("boundary", "mirror")
            boundary_aliases = {"constant": "zero", "wrap": "periodic"}
            if isinstance(boundary_str, str):
                boundary = resolve_boundary(
                    boundary_aliases.get(boundary_str.lower(), boundary_str)
                )
            else:
                boundary = resolve_boundary(boundary_str)

            # Extract filter-specific params (exclude type and boundary)
            filter_params = {k: v for k, v in params.items() if k not in ("type", "boundary")}

            # Inject the source mask once for every filter. Outside FULL_IMAGE mode the
            # filters exclude sentinel voxels (normalized convolution for mean/log/laws,
            # zero-fill for the FFT-based ones).
            if state.source_mode != SourceMode.FULL_IMAGE and state.source_mask is not None:
                filter_params["source_mask"] = state.source_mask.array

            img_arr = state.image.array
            if filter_type in ("log", "gaussian"):
                filter_params.setdefault("spacing_mm", state.image.spacing)

            # When no later step reads the image outside the ROI, a filter whose values
            # do not depend on where the image ends filters only the ROI region (grown by
            # the local intensity sphere when a later step reads it) and the part that
            # the region reads; the rest of the image is 0. A source mask is cut the same.
            region = None
            reach = _filter_reach(filter_type, {**params, **filter_params}, state.image.spacing)
            takes_region = filter_type == "gabor"  # it filters whole slices through the region
            if (
                state.roi_reach is not None
                and (reach is not None or takes_region)
                and img_arr.size >= _FILTER_REGION_MIN
            ):
                bbox = merge_bboxes(
                    compute_nonzero_bbox(state.morph_mask.array),
                    compute_nonzero_bbox(state.intensity_mask.array),
                )
                if bbox is not None:
                    shape = img_arr.shape
                    margin_mm = state.roi_reach
                    grow = [
                        math.ceil(margin_mm / s) + 1 if margin_mm else 0
                        for s in state.image.spacing
                    ]
                    rs = [
                        slice(max(b.start - g, 0), min(b.stop + g, n))
                        for b, g, n in zip(bbox, grow, shape, strict=True)
                    ]
                    region = (rs[0], rs[1], rs[2])
                    if reach is None:  # Gabor: the result is the region
                        crop = region
                    else:
                        crop = _filter_crop(
                            region, reach, shape, boundary is BoundaryCondition.PERIODIC
                        )
                        img_arr = img_arr[crop]
                        if "source_mask" in filter_params:
                            filter_params["source_mask"] = filter_params["source_mask"][crop]

            # Each branch calls its filter; mean/log/laws return (result, valid_mask)
            # when a source mask is supplied, the others return a bare array. The
            # tuple is unwrapped uniformly below.
            result: npt.NDArray[np.floating[Any]] | tuple[npt.NDArray[np.floating[Any]], Any]
            # Boundary actually honoured by this step, recorded for run-log metadata.
            boundary_effective = boundary.name.lower()
            if filter_type == "mean":
                filter_params["boundary"] = boundary
                result = mean_filter(img_arr, **filter_params)
            elif filter_type == "log":
                filter_params["boundary"] = boundary
                result = laplacian_of_gaussian(img_arr, **filter_params)
            elif filter_type == "gaussian":
                filter_params["boundary"] = boundary
                result = gaussian_filter(img_arr, **filter_params)
            elif filter_type == "laws":
                filter_params["boundary"] = boundary
                # 'kernel' param maps to first positional arg
                kernel = filter_params.pop("kernel", "L5E5E5")
                result = laws_filter(img_arr, kernel, **filter_params)
            elif filter_type == "gabor":
                filter_params["boundary"] = boundary
                filter_params.setdefault("spacing_mm", state.image.spacing)
                result = gabor_filter(img_arr, **filter_params, region=region)
            elif filter_type == "wavelet":
                filter_params["boundary"] = boundary
                result = wavelet_transform(img_arr, **filter_params)
            elif filter_type == "simoncelli":
                # FFT-based: only forward a boundary the step explicitly requested;
                # otherwise keep simoncelli_wavelet's own PERIODIC default so that
                # IBSI 2 Phase 1 8.a.1-3 / Phase 2 8.B (periodic) stay unaffected.
                if boundary_requested:
                    filter_params["boundary"] = boundary
                else:
                    boundary_effective = "periodic"
                result = simoncelli_wavelet(img_arr, **filter_params)
            elif filter_type == "riesz":
                # Riesz transform variants: same explicit-only forwarding as simoncelli
                # (IBSI 2 Phase 2 9.B).
                variant = filter_params.pop("variant", "base")
                if boundary_requested:
                    filter_params["boundary"] = boundary
                else:
                    boundary_effective = "periodic"
                if variant == "log":
                    filter_params.setdefault("spacing_mm", state.image.spacing)
                    result = riesz_log(img_arr, **filter_params)
                elif variant == "simoncelli":
                    result = riesz_simoncelli(img_arr, **filter_params)
                else:
                    result = riesz_transform(img_arr, **filter_params)
            else:
                raise ValueError(
                    f"Unknown filter type: {filter_type}. "
                    "Supported: mean, gaussian, log, laws, gabor, wavelet, simoncelli, riesz"
                )

            filtered_array: npt.NDArray[np.floating[Any]] = (
                result[0] if isinstance(result, tuple) else result
            )
            if region is not None:
                embedded: npt.NDArray[Any] = np.zeros(state.image.array.shape, filtered_array.dtype)
                embedded[region] = filtered_array[
                    tuple(
                        slice(r.start - c.start, r.stop - c.start)
                        for r, c in zip(region, crop, strict=True)
                    )
                ]
                filtered_array = embedded

            # Snapshot of the parameters actually honoured by this step, for IBSI 2
            # provenance (params_requested / params_effective in the run log).
            # `filter_params` mirrors the kwargs the branch above passed via
            # **filter_params; `kernel` (laws) and `variant` (riesz) are added back
            # even though they were popped for positional use / dispatch selection
            # rather than forwarded as keyword arguments, and `boundary` is
            # normalised to the effective value (covering the FFT-based filters'
            # own "periodic" default when no boundary was explicitly requested).
            effective_params = dict(filter_params)
            if "source_mask" in effective_params:  # the whole mask, not the part filtered
                effective_params["source_mask"] = state.source_mask.array  # type: ignore[union-attr]
            effective_params["boundary"] = boundary_effective
            if filter_type == "laws":
                effective_params["kernel"] = kernel
            elif filter_type == "riesz":
                effective_params["variant"] = variant
            state.filter_params_requested = self._sanitize_filter_params(params)
            state.filter_params_effective = self._sanitize_filter_params(effective_params)

            # Update state with filtered image
            state.image = Image(
                array=filtered_array,
                spacing=state.image.spacing,
                origin=state.image.origin,
                direction=state.image.direction,
                modality=state.image.modality,
            )
            state.raw_image = state.image  # Update raw_image post-filter
            state.resegment_min = None
            state.is_filtered = True
            state.filter_type = filter_type
            state.filter_boundary_requested = boundary.name.lower()
            state.filter_boundary_effective = boundary_effective

        else:
            raise ValueError(f"Unknown preprocessing step: {step_name}")

    @staticmethod
    def _get_expected_feature_names(
        steps: list[dict[str, Any]],
    ) -> list[str]:
        """Return the ordered list of feature names a config would produce.

        Inspects all ``extract_features`` steps in *steps*, expanding family
        names via :data:`FEATURE_NAMES`.  The result is used to build a
        NaN-filled Series when a configuration fails entirely (empty ROI or
        unexpected error), guaranteeing that the returned Series always has the
        same set of feature names as a successful extraction.
        """
        names: list[str] = []
        seen: set[str] = set()

        for step_def in steps:
            if step_def.get("step") != "extract_features":
                continue
            params = step_def.get("params", {})
            families: list[str] = params.get(
                "families",
                _DEFAULT_FEATURE_FAMILIES,
            )

            for family in families:
                for fam in _feature_name_families(family):
                    if fam in FEATURE_NAMES and fam not in seen:
                        names.extend(FEATURE_NAMES[fam])
                        seen.add(fam)

            # intensity family may include spatial/local sub-families
            if "intensity" in families:
                if params.get("include_spatial_intensity", False):
                    if "spatial_intensity" not in seen:
                        names.extend(FEATURE_NAMES["spatial_intensity"])
                        seen.add("spatial_intensity")
                if params.get("include_local_intensity", False):
                    if "local_intensity" not in seen:
                        names.extend(FEATURE_NAMES["local_intensity"])
                        seen.add("local_intensity")

        return names

    @staticmethod
    def _fill_missing_features(
        results: dict[str, Any],
        families: list[str],
        params: dict[str, Any] | None = None,
    ) -> None:
        """Backfill any missing feature keys with ``NaN``.

        After a ``calculate_*`` function returns, some keys may be absent due
        to partial failures (e.g. mesh generation failure in morphology, or an
        empty texture matrix).  This method ensures every expected key is
        present – computed values are preserved and only truly missing keys are
        set to ``NaN``.
        """
        if params is None:
            params = {}
        nan = float("nan")
        for family in families:
            for fam in _feature_name_families(family):
                if fam in FEATURE_NAMES:
                    for key in FEATURE_NAMES[fam]:
                        if key not in results:
                            results[key] = nan

        # spatial/local intensity sub-families that are gated by params
        if "intensity" in families:
            if params.get("include_spatial_intensity", False):
                for key in FEATURE_NAMES.get("spatial_intensity", ()):
                    if key not in results:
                        results[key] = nan
            if params.get("include_local_intensity", False):
                for key in FEATURE_NAMES.get("local_intensity", ()):
                    if key not in results:
                        results[key] = nan

    def _extract_features(self, state: PipelineState, params: dict[str, Any]) -> dict[str, Any]:
        """
        Extract features based on current state.
        """
        results = {}
        families = params.get("families", _DEFAULT_FEATURE_FAMILIES)

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

        # Nonzero mask bboxes are memoised per extraction pass, so morphology and
        # repeated (single-family) texture calls don't rescan the full volume.
        bbox_cache: _PassCache = {}
        texture_cache = _texture_cache(families)
        for family in families:
            results.update(self._guarded_family(state, family, params, bbox_cache, texture_cache))

        # Ensure every expected feature key is present (NaN for partial failures)
        self._fill_missing_features(results, families, params)
        return results

    def _extract_features_with_dedup(
        self,
        state: PipelineState,
        params: dict[str, Any],
        config_name: str,
        plan: DeduplicationPlan,
        family_cache: dict[tuple[str, str], dict[str, Any]],
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
            family_cache: Cache of computed family features by family and signature hash.

        Returns:
            Dictionary of all extracted features.
        """
        results: dict[str, Any] = {}
        families = params.get("families", _DEFAULT_FEATURE_FAMILIES)

        # Nonzero mask bboxes are memoised per extraction pass (see _extract_features).
        bbox_cache: _PassCache = {}
        cache_keys = {}
        for family in families:
            # Normalize texture aliases so raw subfamily and texture_* requests
            # share the same signature/cache behavior.
            sig_family = _normalize_texture_family(family) or family

            # Get signature from plan using (config_name, family) tuple key
            sig = plan.signatures.get((config_name, sig_family))
            cache_keys[family] = (sig_family, sig.hash) if sig else None
        # The texture families still to compute share one matrix pass
        texture_cache = _texture_cache(
            [family for family in families if cache_keys[family] not in family_cache]
        )
        for family in families:
            cache_key = cache_keys[family]
            if cache_key is not None and cache_key in family_cache:
                # Reuse cached results
                cached = family_cache[cache_key]
                results.update(cached)
                self._dedup_reused_count += 1
            else:
                # Compute this family
                family_results = self._guarded_family(
                    state, family, params, bbox_cache, texture_cache
                )
                results.update(family_results)

                # Cache if we have a signature (a failed family is computed again)
                if cache_key is not None and family not in self._family_errors:
                    family_cache[cache_key] = family_results
                self._dedup_computed_count += 1

        # Ensure every expected feature key is present (NaN for partial failures)
        self._fill_missing_features(results, families, params)
        return results

    @staticmethod
    def _cached_nonzero_bbox(
        arr: npt.NDArray[np.floating[Any]],
        cache: _PassCache,
    ) -> Optional[tuple[slice, slice, slice]]:
        """Nonzero bbox of `arr`, memoised by array identity for one extraction pass."""
        key = id(arr)
        if key not in cache:
            cache[key] = compute_nonzero_bbox(arr)
        return cast(Optional[tuple[slice, slice, slice]], cache[key])

    def _masked_values(
        self,
        image: Image | npt.NDArray[Any],
        mask: Image,
        bbox_cache: _PassCache,
    ) -> npt.NDArray[np.floating[Any]]:
        """ROI voxel values, equivalent to ``apply_mask(image, mask)``.

        Crops both arrays to the mask's cached nonzero bbox before gathering, so the
        scan and boolean gather touch only the ROI bounding box rather than the full
        volume. The gathered values (and their order) are bit-identical, since
        cropping only removes voxels that are outside the mask anyway.
        """
        bbox = self._cached_nonzero_bbox(mask.array, bbox_cache)
        if bbox is None:
            return apply_mask(image, mask)
        img_arr = image.array if isinstance(image, Image) else image
        return apply_mask(img_arr[bbox], mask.array[bbox])

    def _roi_values(
        self, image: Image, mask: Image, cache: _PassCache
    ) -> npt.NDArray[np.floating[Any]]:
        """`_masked_values` of an image of the state, gathered once per extraction pass
        (for example, the histogram and the IVH read the same values)."""
        key = (id(image.array), id(mask.array))
        if key not in cache:
            cache[key] = self._masked_values(image, mask, cache)
        return cast(npt.NDArray[np.floating[Any]], cache[key])

    def _guarded_family(
        self,
        state: PipelineState,
        family: str,
        params: dict[str, Any],
        bbox_cache: _PassCache,
        texture_cache: dict[str, Optional[dict[str, Any]]],
    ) -> dict[str, Any]:
        """`_extract_single_family`, but an error leaves only this family without values
        (NaN): the error goes to `_family_errors` and to a warning."""
        try:
            return self._extract_single_family(state, family, params, bbox_cache, texture_cache)
        except EmptyROIMaskError:
            raise
        except Exception as e:
            self._family_errors[family] = f"{type(e).__name__}: {e}"
            warnings.warn(
                f"The {family} features failed ({type(e).__name__}: {e}); they are NaN.",
                UserWarning,
                stacklevel=2,
            )
            return {}

    def _extract_single_family(
        self,
        state: PipelineState,
        family: str,
        params: dict[str, Any],
        bbox_cache: Optional[_PassCache] = None,
        texture_cache: Optional[dict[str, Optional[dict[str, Any]]]] = None,
    ) -> dict[str, Any]:
        """
        Extract features for a single family.

        This is a refactored helper to enable per-family deduplication. The texture
        families of `texture_cache` (see `_texture_cache`) share one matrix pass.
        """
        results: dict[str, Any] = {}
        if bbox_cache is None:
            bbox_cache = {}

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
                    roi_bbox=self._cached_nonzero_bbox(state.morph_mask.array, bbox_cache),
                    grid_offset=state.grid_offset,
                )
            )

        elif family == "intensity":
            masked_values = self._roi_values(state.raw_image, state.intensity_mask, bbox_cache)
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
            masked_values = self._roi_values(state.image, state.intensity_mask, bbox_cache)
            results.update(
                calculate_intensity_histogram_features(
                    masked_values,
                    # IBSI: the histogram spans the full discretisation range
                    # [1, N_g], not just the observed values.
                    n_bins=state.n_bins if state.is_discretised else None,
                )
            )

        elif family == "ivh":
            results.update(self._compute_ivh_features(state, params, ivh_params, bbox_cache))

        elif (texture_family := _normalize_texture_family(family)) is not None:
            results.update(
                self._compute_texture_features(
                    state, texture_family, texture_matrix_params, bbox_cache, texture_cache
                )
            )

        return results

    def _compute_ivh_features(
        self,
        state: PipelineState,
        params: dict[str, Any],
        ivh_params: dict[str, Any],
        bbox_cache: Optional[_PassCache] = None,
    ) -> dict[str, Any]:
        """Compute IVH features (helper for _extract_single_family)."""
        if bbox_cache is None:
            bbox_cache = {}
        ivh_use_continuous = params.get("ivh_use_continuous", False)
        ivh_discretisation = params.get("ivh_discretisation", None)

        ivh_disc_bin_width: Optional[float] = None
        ivh_disc_min_val: Optional[float] = None

        if ivh_use_continuous:
            ivh_values = self._roi_values(state.raw_image, state.intensity_mask, bbox_cache)
        elif ivh_discretisation:
            ivh_disc_params = ivh_discretisation.copy()
            ivh_method = ivh_disc_params.pop("method", "FBS")
            if ivh_method == "FBS":
                ivh_disc_params["min_val"] = _fbs_start(ivh_disc_params, state)
            ivh_disc_bin_width = ivh_disc_params.get("bin_width")
            ivh_disc_min_val = ivh_disc_params.get("min_val")
            # With the bin limits given, the bin rule works voxel by voxel: binning the
            # ROI values alone gives the same bins, without a full-image discretisation.
            limits_given = ivh_method == "FIXED_CUTOFFS" or (
                ivh_disc_min_val is not None
                and (ivh_method != "FBN" or ivh_disc_params.get("max_val") is not None)
            )
            if limits_given:
                raw_values = self._roi_values(state.raw_image, state.intensity_mask, bbox_cache)
                ivh_values = cast(
                    npt.NDArray[Any],
                    discretise_image(raw_values, method=ivh_method, **ivh_disc_params),
                )
            else:
                temp_ivh_disc = discretise_image(
                    state.raw_image,
                    method=ivh_method,
                    roi_mask=state.intensity_mask,
                    **ivh_disc_params,
                )
                ivh_values = self._masked_values(temp_ivh_disc, state.intensity_mask, bbox_cache)
        else:
            ivh_values = self._roi_values(state.image, state.intensity_mask, bbox_cache)

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
        bbox_cache: Optional[_PassCache] = None,
        texture_cache: Optional[dict[str, Optional[dict[str, Any]]]] = None,
    ) -> dict[str, Any]:
        """Compute texture features (helper for _extract_single_family).

        The families of `texture_cache` that have no results yet are computed in the
        same matrix pass as `family`, and their results are kept there.
        """
        if bbox_cache is None:
            bbox_cache = {}
        if texture_cache is None:
            texture_cache = {}
        wanted = _feature_name_families(family)
        if any(texture_cache.get(name) is None for name in wanted):
            todo = {*wanted, *(name for name, done in texture_cache.items() if done is None)}
            texture_cache.update(self._texture_pass(state, todo, texture_matrix_params, bbox_cache))
        results: dict[str, Any] = {}
        for name in wanted:
            results.update(cast(dict[str, Any], texture_cache[name]))
        return results

    def _texture_pass(
        self,
        state: PipelineState,
        families: set[str],
        texture_matrix_params: dict[str, Any],
        bbox_cache: _PassCache,
    ) -> dict[str, dict[str, Any]]:
        """The features of each texture family in `families` (glcm, glrlm, ...), from
        one matrix pass."""
        results: dict[str, dict[str, Any]] = {}

        if not state.is_discretised:
            raise ValueError(
                "Texture features requested but image is not discretised. "
                "You must include a 'discretise' step before extracting texture features."
            )

        disc_image = state.image
        n_bins = state.n_bins if state.n_bins else 32

        matrix_kwargs: dict[str, Any] = {}
        if "ngldm_alpha" in texture_matrix_params:
            # Grey-level differences are whole numbers, so |d| <= alpha is |d| <= floor(alpha)
            # (one compiled kernel for every alpha, also 1.0 from a file)
            matrix_kwargs["ngldm_alpha"] = math.floor(texture_matrix_params["ngldm_alpha"])
        for key in _TEXTURE_DISTANCES:
            if key in texture_matrix_params:
                matrix_kwargs[key] = int(texture_matrix_params[key])

        # Crop once to the ROI bounding box (union of intensity and morph masks, to
        # preserve GLDZM distance-map correctness) and use the cropped arrays for the
        # matrix calculation and every feature family below. The feature functions scan
        # the mask (ROI voxel counts, GLCM Ng_eff), so passing full-volume arrays would
        # repeat full-volume scans per family. Feature values are unchanged: cropping
        # only removes zero-mask voxels. The per-mask bboxes are memoised per
        # extraction pass, so repeated single-family calls scan each mask only once.
        bbox = merge_bboxes(
            self._cached_nonzero_bbox(state.intensity_mask.array, bbox_cache),
            self._cached_nonzero_bbox(state.morph_mask.array, bbox_cache),
        )
        if bbox is None:
            disc_c = disc_image.array
            intensity_mask_c = state.intensity_mask.array
            morph_mask_c = state.morph_mask.array
        else:
            disc_c = disc_image.array[bbox]
            intensity_mask_c = state.intensity_mask.array[bbox]
            morph_mask_c = state.morph_mask.array[bbox]

        # Only the matrices of the requested families are computed.
        want_glcm = "glcm" in families
        want_glrlm = "glrlm" in families
        want_glszm = "glszm" in families
        want_gldzm = "gldzm" in families
        want_ngtdm = "ngtdm" in families
        want_ngldm = "ngldm" in families

        # Compact matrices: the same features from smaller tables. When the two masks are
        # one array, the distance map uses the ROI of the intensity mask, which is the same.
        masks_in_sync = state.morph_mask.array is state.intensity_mask.array
        # A one-slice image has an axis of size 1: the GLCM and the GLRLM use the in-plane
        # directions, and the GLDZM distance map is the in-plane one.
        planar = _planar_axes(state.grid_shape or state.image.array.shape)
        # The GLDZM distance map depends only on the two masks. Configurations that share
        # them (one shared resampled state) reuse the last map; weak references keep no
        # mask alive.
        last = self._last_distance_map
        reuse = (
            want_gldzm
            and last is not None
            and last[0]() is state.morph_mask.array
            and last[1]() is state.intensity_mask.array
        )
        texture_matrices = _texture_matrices(
            disc_c,
            intensity_mask_c,
            n_bins,
            distance_mask=None if masks_in_sync else morph_mask_c,
            calc_glcm=want_glcm,
            calc_glrlm=want_glrlm,
            calc_ngtdm=want_ngtdm,
            calc_ngldm=want_ngldm,
            calc_glszm=want_glszm,
            calc_gldzm=want_gldzm,
            compact=True,
            distance_map=last[2] if reuse and last is not None else None,
            planar=planar,
            **matrix_kwargs,
        )
        if "distance_map" in texture_matrices and not reuse:
            self._last_distance_map = (
                weakref.ref(state.morph_mask.array),
                weakref.ref(state.intensity_mask.array),
                texture_matrices["distance_map"],
            )
        # The bool ROI gives the same ROI voxel counts as intensity_mask_c, from a fast
        # count.
        roi = texture_matrices["roi"]

        if want_glcm:
            results["glcm"] = calculate_glcm_features(
                disc_c,
                roi.view(np.uint8),  # the texture ROI, in one mask type and layout
                n_bins,
                glcm_matrix=texture_matrices["glcm"],
            )
        if want_glrlm:
            results["glrlm"] = calculate_glrlm_features(
                disc_c,
                roi,
                n_bins,
                glrlm_matrix=texture_matrices["glrlm"],
                n_directions=_directions(planar).size,
            )
        if want_glszm:
            results["glszm"] = _glszm_features_from_cells(texture_matrices["glszm_cells"], roi)
        if want_gldzm:
            results["gldzm"] = calculate_gldzm_features(
                disc_c,
                roi,
                n_bins,
                gldzm_matrix=texture_matrices["gldzm"],
                distance_mask=morph_mask_c,
            )
        if want_ngtdm:
            results["ngtdm"] = calculate_ngtdm_features(
                disc_c,
                intensity_mask_c,
                n_bins,
                ngtdm_matrices=(
                    texture_matrices["ngtdm_s"],
                    texture_matrices["ngtdm_n"],
                ),
            )
        if want_ngldm:
            results["ngldm"] = calculate_ngldm_features(
                disc_c,
                roi,
                n_bins,
                ngldm_matrix=texture_matrices["ngldm"],
            )

        return results

    def save_log(self, output_path: str | Path) -> None:
        """
        Save the processing log to a self-describing JSON file.
        """
        path = Path(output_path)
        if not str(path).endswith(".json"):
            path = Path(f"{path}.json")

        payload = {
            "log_schema_version": "1.0",
            "pipeline_schema_version": CONFIG_SCHEMA_VERSION,
            "pictologics_version": _get_package_version(),
            "exported_at": datetime.datetime.now().isoformat(),
            "mask_roi_semantics": "nonzero_values_are_roi_membership",
            "entry_count": len(self._log),
            "entries": self._make_serializable(self._log),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        # Written as it is encoded, 4,096 pieces at a time: the whole text is never held.
        encoder = json.JSONEncoder(indent=4, default=str)
        with path.open("w", encoding="utf-8") as fh:
            for pieces in itertools.batched(encoder.iterencode(payload), 4096):
                fh.write("".join(pieces))

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

    # ------------------------------------------------------------------
    # Feature catalog
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_feature_key(key: str) -> tuple[str, str]:
        """Extract the human-readable name and IBSI code from a feature key.

        Feature keys follow the pattern ``descriptive_name_CODE`` where
        ``CODE`` is a 3–4 character uppercase-alphanumeric IBSI identifier.
        Some IVH features carry an additional numeric suffix (e.g.
        ``volume_at_intensity_fraction_0.10_BC2M_10``).

        Returns:
            ``(stripped_name, ibsi_code)`` — e.g. ``("joint_entropy", "TU9B")``.
        """
        m = _IBSI_CODE_RE.search(key)
        if m is None:
            return key, ""
        code = m.group(1)
        # Strip the code (and optional trailing _digits) from the key
        name = key[: m.start()]
        return name, code

    @staticmethod
    def _extract_config_metadata(
        steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Parse a config’s step list into preprocessing metadata.

        Returns a flat dict with keys used by :meth:`describe_features`.
        """
        records: list[dict[str, Any]] = []
        for i, step_def in enumerate(steps, start=1):
            step_name = step_def.get("step", "")
            if step_name not in _PREPROCESSING_STEPS:
                continue
            step_index = step_def.get("step_index", i)
            records.append(
                {
                    "step_index": step_index,
                    "step": step_name,
                    "params": copy.deepcopy(step_def.get("params", {})),
                }
            )

        meta: dict[str, Any] = {
            "preprocessing_sequence": None,
            "preprocessing_steps": None,
            "is_resampled": False,
            "resampling_spacing": None,
            "interpolation": None,
            "resample_params": None,
            "is_resegmented": False,
            "resegment_apply_to": None,
            "resegment_params": None,
            "is_outlier_filtered": False,
            "filter_outliers_apply_to": None,
            "filter_outliers_params": None,
            "is_intensity_rounded": False,
            "round_intensities_params": None,
            "keeps_largest_component": False,
            "keep_largest_component_apply_to": None,
            "keep_largest_component_params": None,
            "is_mask_grown": False,
            "grow_mask_apply_to": None,
            "grow_mask_params": None,
            "is_mask_binarized": False,
            "binarize_mask_apply_to": None,
            "binarize_mask_params": None,
            "is_normalised": False,
            "normalisation_method": None,
            "normalise_params": None,
            "is_discretised": False,
            "discretisation_method": None,
            "discretisation_param": None,
            "discretise_params": None,
            "is_filtered": False,
            "filter_type": None,
            "filter_params": None,
        }

        if not records:
            return meta

        meta["preprocessing_sequence"] = " > ".join(
            f"{record['step_index']}:{record['step']}" for record in records
        )
        meta["preprocessing_steps"] = RadiomicsPipeline._catalog_json(records)

        records_by_step: dict[str, list[dict[str, Any]]] = {
            step_name: [record for record in records if record.get("step") == step_name]
            for step_name in _PREPROCESSING_STEPS
        }

        for step_name, column in _PREPROCESSING_PARAM_COLUMNS.items():
            step_records = records_by_step[step_name]
            if step_records:
                meta[column] = RadiomicsPipeline._catalog_json(
                    [
                        {
                            "step_index": record["step_index"],
                            "params": record["params"],
                        }
                        for record in step_records
                    ]
                )

        resample_records = records_by_step["resample"]
        if resample_records:
            meta["is_resampled"] = True
            spacings = [record["params"].get("new_spacing") for record in resample_records]
            meta["resampling_spacing"] = RadiomicsPipeline._catalog_value(spacings)
            interpolations = [
                record["params"].get("interpolation", "linear") for record in resample_records
            ]
            meta["interpolation"] = RadiomicsPipeline._catalog_value(interpolations)

        resegment_records = records_by_step["resegment"]
        if resegment_records:
            meta["is_resegmented"] = True
            meta["resegment_apply_to"] = RadiomicsPipeline._catalog_value(
                [record["params"].get("apply_to", "both") for record in resegment_records]
            )

        filter_outlier_records = records_by_step["filter_outliers"]
        if filter_outlier_records:
            meta["is_outlier_filtered"] = True
            meta["filter_outliers_apply_to"] = RadiomicsPipeline._catalog_value(
                [record["params"].get("apply_to", "both") for record in filter_outlier_records]
            )

        round_records = records_by_step["round_intensities"]
        if round_records:
            meta["is_intensity_rounded"] = True

        largest_component_records = records_by_step["keep_largest_component"]
        if largest_component_records:
            meta["keeps_largest_component"] = True
            meta["keep_largest_component_apply_to"] = RadiomicsPipeline._catalog_value(
                [record["params"].get("apply_to", "both") for record in largest_component_records]
            )

        grow_records = records_by_step["grow_mask"]
        if grow_records:
            meta["is_mask_grown"] = True
            meta["grow_mask_apply_to"] = RadiomicsPipeline._catalog_value(
                [record["params"].get("apply_to", "both") for record in grow_records]
            )

        binarize_records = records_by_step["binarize_mask"]
        if binarize_records:
            meta["is_mask_binarized"] = True
            meta["binarize_mask_apply_to"] = RadiomicsPipeline._catalog_value(
                [record["params"].get("apply_to", "both") for record in binarize_records]
            )

        normalise_records = records_by_step["normalise"]
        if normalise_records:
            meta["is_normalised"] = True
            meta["normalisation_method"] = RadiomicsPipeline._catalog_value(
                [record["params"].get("method") for record in normalise_records]
            )

        discretise_records = records_by_step["discretise"]
        if discretise_records:
            meta["is_discretised"] = True
            methods = [record["params"].get("method", "FBN") for record in discretise_records]
            meta["discretisation_method"] = RadiomicsPipeline._catalog_value(methods)
            disc_params: list[Any] = []
            for method, record in zip(methods, discretise_records, strict=True):
                params = record["params"]
                if method == "FBN":
                    disc_params.append(params.get("n_bins"))
                elif method == "FBS":
                    disc_params.append(params.get("bin_width"))
                elif method == "FIXED_CUTOFFS":
                    disc_params.append(params.get("cutoffs"))
                else:
                    disc_params.append(None)
            meta["discretisation_param"] = RadiomicsPipeline._catalog_value(disc_params)

        filter_records = records_by_step["filter"]
        if filter_records:
            meta["is_filtered"] = True
            filter_types = [record["params"].get("type") for record in filter_records]
            meta["filter_type"] = RadiomicsPipeline._catalog_value(filter_types)

        return meta

    @staticmethod
    def _catalog_serializable(obj: Any) -> Any:
        """Convert catalog metadata values to JSON-serializable objects."""
        if isinstance(obj, tuple):
            return [RadiomicsPipeline._catalog_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {
                str(key): RadiomicsPipeline._catalog_serializable(
                    _mask_values_file_form(value) if key == "mask_values" else value
                )
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [RadiomicsPipeline._catalog_serializable(item) for item in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, BoundaryCondition):
            return obj.name.lower()
        return obj

    @staticmethod
    def _catalog_json(obj: Any) -> str:
        """Encode structured catalog metadata as compact JSON."""
        return json.dumps(
            RadiomicsPipeline._catalog_serializable(obj),
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _catalog_value(values: list[Any]) -> Any:
        """Return one scalar value or a JSON list for repeated preprocessing steps."""
        serializable = cast(list[Any], RadiomicsPipeline._catalog_serializable(values))
        if len(serializable) == 1:
            value = serializable[0]
            if isinstance(value, list):
                return str(tuple(value))
            if isinstance(value, dict):
                return RadiomicsPipeline._catalog_json(value)
            return value
        return RadiomicsPipeline._catalog_json(serializable)

    def describe_features(self) -> pd.DataFrame:
        """Return a DataFrame cataloguing every feature the pipeline will produce.

        Each row represents one (configuration, feature) pair. The columns
        describe the feature identity, its family membership, and the
        preprocessing state at the ``extract_features`` step that produces it.
        Repeated preprocessing steps are represented as compact JSON arrays in
        the corresponding ``*_params`` cells.

        This is useful for:

        * Inspecting the full set of features before running the pipeline.
        * Filtering or subsetting features by family, discretisation method,
          filter type, etc.
        * Exporting a data dictionary (``describe_features().to_csv(...)``)
          alongside study results for documentation and reproducibility.

        Returns:
            A `pandas.DataFrame` with columns:

            - **config** – Configuration name.
            - **feature_key** – Full feature key as it appears in the output Series.
            - **feature_name** – Human-readable name (IBSI code stripped).
            - **ibsi_code** – 3–4 character IBSI identifier.
            - **family** – Granular feature family (e.g. ``glcm``, ``ivh``).
            - **family_group** – Broad category: *Intensity*, *Morphology*, or
                *Texture*.
            - **requires_discretisation** – Whether the family needs discretised
                input.
            - **uses_morph_mask** / **uses_intensity_mask** – Which runtime ROI
                mask(s) the feature row depends on.
            - **source_mode** – Source voxel handling mode for the configuration.
            - **sentinel_value** – Explicit sentinel value, if configured.
            - **feature_extraction_step_index** – 1-based position of the
                ``extract_features`` step that produced the row.
            - **feature_extraction_params** – Parameters on that
                ``extract_features`` step as compact JSON.
            - **preprocessing_sequence** – Ordered preprocessing step sequence
                applied before feature extraction.
            - **preprocessing_steps** – Full ordered preprocessing step records
                as compact JSON.
            - **is_discretised** – Whether the configuration includes a
                ``discretise`` step.
            - **discretisation_method** – ``FBN``, ``FBS``, or ``None``.
            - **discretisation_param** – Bin count (FBN) or bin width (FBS).
            - **is_resampled** – Whether the configuration includes a ``resample``
                step.
            - **resampling_spacing** – Target spacing as a string, e.g.
                ``"(0.5, 0.5, 0.5)"``.
            - **interpolation** – Resampling interpolation method.
            - **is_filtered** – Whether a ``filter`` step is present.
            - **filter_type** – Filter type (``log``, ``gabor``, …) or ``None``.
            - **filter_params** – Ordered filter step parameters as compact JSON,
                or ``None``.

        Example:
            ```python
            pipeline = RadiomicsPipeline()
            catalog = pipeline.describe_features()

            # Export as CSV data dictionary
            catalog.to_csv("feature_catalog.csv", index=False)

            # Filter: only texture features from FBN configs
            texture_fbn = catalog[
                (catalog["family_group"] == "Texture")
                & (catalog["discretisation_method"] == "FBN")
            ]
            ```
        """
        # Build reverse lookup: feature_key -> family
        key_to_family: dict[str, str] = {}
        for family, keys in FEATURE_NAMES.items():
            for key in keys:
                key_to_family[key] = family

        rows: list[dict[str, Any]] = []

        for config_name, steps in self._configs.items():
            active_preprocessing: list[dict[str, Any]] = []
            row_by_key: dict[str, dict[str, Any]] = {}
            key_order: list[str] = []
            config_metadata = self._config_metadata.get(config_name, {})
            source_mode = config_metadata.get("source_mode", "full_image")
            sentinel_value = config_metadata.get("sentinel_value")

            for step_index, step_def in enumerate(steps, start=1):
                step_name = step_def.get("step", "")
                params = step_def.get("params", {})

                if step_name == "extract_features":
                    feature_keys = self._get_expected_feature_names([step_def])
                    config_meta = self._extract_config_metadata(active_preprocessing)
                    extraction_params = self._catalog_json(params) if params else None

                    for fkey in feature_keys:
                        fname, code = self._parse_feature_key(fkey)
                        family = key_to_family.get(fkey, "unknown")

                        row: dict[str, Any] = {
                            "config": config_name,
                            "feature_key": fkey,
                            "feature_name": fname,
                            "ibsi_code": code,
                            "family": family,
                            "family_group": _FAMILY_GROUP.get(family, "Unknown"),
                            "requires_discretisation": _REQUIRES_DISCRETISATION.get(family, False),
                            "uses_morph_mask": _family_uses_morph_mask(family),
                            "uses_intensity_mask": _feature_uses_intensity_mask(fkey, family),
                            "source_mode": source_mode,
                            "sentinel_value": sentinel_value,
                            "feature_extraction_step_index": step_index,
                            "feature_extraction_params": extraction_params,
                        }
                        row.update(config_meta)
                        if fkey not in row_by_key:
                            key_order.append(fkey)
                        row_by_key[fkey] = row

                elif step_name in _PREPROCESSING_STEPS:
                    active_preprocessing.append(
                        {
                            "step_index": step_index,
                            "step": step_name,
                            "params": copy.deepcopy(params),
                        }
                    )

            rows.extend(row_by_key[fkey] for fkey in key_order)

        columns = [
            "config",
            "feature_key",
            "feature_name",
            "ibsi_code",
            "family",
            "family_group",
            "requires_discretisation",
            "uses_morph_mask",
            "uses_intensity_mask",
            "source_mode",
            "sentinel_value",
            "feature_extraction_step_index",
            "feature_extraction_params",
            "preprocessing_sequence",
            "preprocessing_steps",
            "is_discretised",
            "discretisation_method",
            "discretisation_param",
            "discretise_params",
            "is_resampled",
            "resampling_spacing",
            "interpolation",
            "resample_params",
            "is_resegmented",
            "resegment_apply_to",
            "resegment_params",
            "is_outlier_filtered",
            "filter_outliers_apply_to",
            "filter_outliers_params",
            "is_intensity_rounded",
            "round_intensities_params",
            "keeps_largest_component",
            "keep_largest_component_apply_to",
            "keep_largest_component_params",
            "is_mask_grown",
            "grow_mask_apply_to",
            "grow_mask_params",
            "is_mask_binarized",
            "binarize_mask_apply_to",
            "binarize_mask_params",
            "is_normalised",
            "normalisation_method",
            "normalise_params",
            "is_filtered",
            "filter_type",
            "filter_params",
        ]
        return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)

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
                name: self._configs[name] for name in config_names if name in self._configs
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
            result["pictologics_version"] = _get_package_version()
            result["exported_at"] = datetime.datetime.now().isoformat()
            result["mask_roi_semantics"] = "nonzero_values_are_roi_membership"

        result["configs"] = serializable_configs

        if include_deduplication:
            result["deduplication"] = {
                "enabled": self._deduplication_enabled,
                "rules_version": self._deduplication_rules.version,
            }
            # Include last plan if available and not stale
            if self._last_deduplication_plan and not self._configs_modified_since_plan:
                result["deduplication"]["last_plan"] = self._last_deduplication_plan.to_dict()

        return result

    def _make_serializable(self, obj: Any) -> Any:
        """Convert tuples and other non-serializable types to serializable forms (also the
        items of a tuple, such as the numpy numbers of a spacing)."""
        if isinstance(obj, dict):
            return {
                k: self._make_serializable(_mask_values_file_form(v) if k == "mask_values" else v)
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, BoundaryCondition):
            # Loading accepts the member name ("mirror"), not scipy's mode ("reflect").
            return obj.name.lower()
        elif isinstance(obj, Enum):
            return obj.value
        return obj

    def _sanitize_filter_param_value(self, value: Any) -> Any:
        """Make a filter-step parameter value JSON-safe for provenance logging.

        A raw array (e.g. the pipeline-injected ``source_mask``) is never logged in
        full, only as a compact shape/voxel-count descriptor. Other values go through
        ``_make_serializable``, which records a ``BoundaryCondition`` by its lowercase
        name (matching ``filter_boundary_effective``).
        """
        if isinstance(value, np.ndarray):
            return {
                "shape": list(value.shape),
                "voxel_count": int(value.size),
                "true_count": int(np.count_nonzero(value)),
            }
        return self._make_serializable(value)

    def _sanitize_filter_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """JSON-safe copy of a filter step's parameter dict, for the run log's
        ``params_requested`` / ``params_effective`` provenance fields."""
        return {k: self._sanitize_filter_param_value(v) for k, v in params.items()}

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
            raise ValueError(f"Unsupported file extension: {suffix}. Use .json, .yaml, or .yml")

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
            if source_mode not in _SOURCE_MODES:
                raise ValueError(
                    f"Config '{name}': source_mode must be one of {_SOURCE_MODES}, "
                    f"not {source_mode!r}"
                )

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
        data = _load_yaml(yaml_string)
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
            return cls.from_json(content, validate=validate, load_standard=load_standard)
        elif suffix in (".yaml", ".yml"):
            return cls.from_yaml(content, validate=validate, load_standard=load_standard)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}. Use .json, .yaml, or .yml")

    @classmethod
    def from_template(cls, name: str, load_standard: bool = False) -> "RadiomicsPipeline":
        """
        Create a pipeline with the configurations of a template of the package.

        Templates:

        - `"standard"`: the six standard configurations (`standard_fbn_8` to
          `standard_fbs_32`).
        - `"lv"`: CT of the left ventricular myocardium, in four compartments (whole,
          fat, myocardial tissue and calcium), 30 configurations.
        - `"coronary"`: coronary plaque in CT angiography, in four plaque types (all,
          non-calcified, low-attenuation and calcified), 30 configurations.

        Args:
            name: The template name.
            load_standard: Whether to also load the standard configurations.

        Returns:
            New RadiomicsPipeline instance.

        Raises:
            ValueError: If no template has this name.

        Example:
            ```python
            pipeline = RadiomicsPipeline.from_template("coronary")
            results = pipeline.run(
                image, mask, config_names=["coronary_cp_orig", "coronary_cp_fbs_16"]
            )

            # add a template to a pipeline
            pipeline.merge_configs(RadiomicsPipeline.from_template("lv"))
            ```
        """
        names = sorted(
            file.removesuffix("_configs.yaml")
            for file in list_template_files()
            if file.endswith("_configs.yaml")
        )
        if name not in names:
            raise ValueError(
                f"Template '{name}' not found.{_hint(name, names)} Templates: {', '.join(names)}."
            )
        data = load_template_file(f"{name}_configs.yaml")
        return cls.from_dict(data, load_standard=load_standard)

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
            if name in other._config_metadata:
                self._config_metadata[name] = copy.deepcopy(other._config_metadata[name])
            else:
                self._config_metadata.pop(name, None)
            self._configs_modified_since_plan = True
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
        "resample": {
            "new_spacing",
            "interpolation",
            "mask_interpolation",
            "mask_threshold",
            "round_intensities",
        },
        "resegment": {"range_min", "range_max", "apply_to"},
        "filter_outliers": {"sigma", "apply_to"},
        "binarize_mask": {"threshold", "mask_values", "apply_to"},
        "keep_largest_component": {"apply_to"},
        "grow_mask": {"to_mm", "from_mm", "nearest_roi", "apply_to"},
        "normalise": {"method", "region", "percentiles", "range_min", "range_max"},
        "round_intensities": set(),
        "discretise": {
            "method",
            "n_bins",
            "bin_width",
            "min_val",
            "max_val",
            "cutoffs",
        },
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
            "spatial_intensity_params",
            "local_intensity_params",
            "texture_matrix_params",
            "ivh_params",
            "ivh_use_continuous",
            "ivh_discretisation",
        },
    }

    @classmethod
    def _validate_config(cls, name: str, steps: list[dict[str, Any]]) -> bool:
        """
        Validate a configuration, issuing warnings for issues.

        The checks are those of `add_config` (see `_config_problems`); here each problem
        gives a warning, so that a configuration file with mistakes still loads.

        Args:
            name: Configuration name (for warning messages).
            steps: List of step dictionaries.

        Returns:
            True if valid, False if issues found (warnings are issued).
        """
        problems = _config_problems(steps)
        for problem in problems:
            warnings.warn(f"Config '{name}' {problem}", UserWarning, stacklevel=2)
        return not problems


# ---------------------------------------------------------------------------
# run_batch: the cases and the workers
# ---------------------------------------------------------------------------


# The case keys that a run_batch case does not pass on: those of run_rois for a case
# with a mask, the mask for a case with a label map (by "label_map")
_OTHER_CASE_KEYS = {False: ("rois", "labels"), True: ("mask",)}


def _case_identity(case: Mapping[str, Any]) -> dict[str, Any]:
    """What the record of a run_batch case holds of its inputs (JSON): its sources and
    options, which a later batch compares to skip the case."""

    def plain(value: Any) -> Any:  # None for no value
        if value is None or len(value) == 0:
            return None
        value = dict(value) if isinstance(value, Mapping) else np.asarray(value).tolist()
        return json.loads(json.dumps(_json_safe(value), default=str))

    return {
        "image": _case_source(case["image"]),
        "image_options": plain(case.get("image_options")),
        "mask": _case_source(case.get("mask")),
        "rois": _case_source(case.get("rois")),
        "labels": plain(case.get("labels")),
    }


def _case_source(value: Any) -> Optional[str]:
    """How the record of a run_batch case names its image or mask: the path,
    "InMemory" for an Image, or None."""
    if value is None:
        return None
    return "InMemory" if isinstance(value, Image) else str(value)


def _batch_pipeline(
    configs: dict[str, list[dict[str, Any]]],
    metadata: dict[str, dict[str, Any]],
    deduplicate: bool,
    rules: DeduplicationRules,
) -> RadiomicsPipeline:
    """A pipeline with these configurations, which runs the cases of run_batch (so the
    log of the calling pipeline stays as it is)."""
    pipeline = RadiomicsPipeline(
        deduplicate=deduplicate, deduplication_rules=rules, load_standard=False
    )
    pipeline._configs = configs
    pipeline._config_metadata = metadata
    return pipeline


# The pipeline of a run_batch worker process
_BATCH_PIPELINE: Optional[RadiomicsPipeline] = None


def _start_batch_worker(setup: tuple[Any, ...], threads: int) -> None:
    """Start a run_batch worker: its share of the numba threads (all thread pools of the
    package follow it) and its pipeline."""
    global _BATCH_PIPELINE
    numba.set_num_threads(threads)
    _BATCH_PIPELINE = _batch_pipeline(*setup)


def _batch_case(
    case: dict[str, Any], names: list[str], hashes: dict[str, str], path: Path
) -> dict[str, Any]:
    """Run one case in a run_batch worker."""
    return cast(RadiomicsPipeline, _BATCH_PIPELINE)._run_case(case, names, hashes, path)
