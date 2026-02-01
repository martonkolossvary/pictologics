"""
Feature Deduplication System
============================

This module provides intelligent deduplication for multi-configuration radiomics pipelines.
When multiple configurations share preprocessing steps (differing only in discretization),
features like morphology and intensity would produce identical results. This system
identifies such redundancies and computes each unique combination once.

Key Classes:
    - DeduplicationRules: Defines which preprocessing steps affect each feature family
    - PreprocessingSignature: Hashable signature for comparing preprocessing configurations
    - ConfigurationAnalyzer: Analyzes configurations to build a deduplication plan
    - DeduplicationPlan: Maps which config/family pairs can reuse cached results

Example:
    >>> from pictologics.deduplication import ConfigurationAnalyzer, DEDUPLICATION_RULES_V1_0_0
    >>> analyzer = ConfigurationAnalyzer(configs, rules=DEDUPLICATION_RULES_V1_0_0)
    >>> plan = analyzer.analyze()
    >>> if plan.should_compute("fbn_64", "morphology"):
    ...     # Compute fresh
    ... else:
    ...     source = plan.get_source("fbn_64", "morphology")
    ...     # Copy from source config
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# Deduplication Rules
# =============================================================================


@dataclass(frozen=True)
class DeduplicationRules:
    """
    Defines which preprocessing steps affect each feature family.

    This is a frozen (immutable) dataclass that specifies the dependencies
    between preprocessing steps and feature families. Rules are versioned
    to ensure reproducibility when sharing configurations.

    Attributes:
        version: Semantic version string for this rules definition.
        family_dependencies: Mapping of feature family names to the set of
            preprocessing step names that affect their output.
        ivh_discretization_dependent_unless: Condition under which IVH becomes
            independent of discretization (e.g., "ivh_use_continuous=True").
        comparison_mode: How to compare preprocessing parameters ("exact_params").
    """

    version: str
    family_dependencies: dict[str, frozenset[str]]
    ivh_discretization_dependent_unless: str
    comparison_mode: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize rules to a dictionary."""
        return {
            "version": self.version,
            "family_dependencies": {
                k: sorted(v) for k, v in self.family_dependencies.items()
            },
            "ivh_discretization_dependent_unless": self.ivh_discretization_dependent_unless,
            "comparison_mode": self.comparison_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeduplicationRules":
        """Deserialize rules from a dictionary."""
        return cls(
            version=data["version"],
            family_dependencies={
                k: frozenset(v) for k, v in data["family_dependencies"].items()
            },
            ivh_discretization_dependent_unless=data["ivh_discretization_dependent_unless"],
            comparison_mode=data["comparison_mode"],
        )

    @classmethod
    def get_version(cls, version: str) -> "DeduplicationRules":
        """
        Get rules for a specific version from the registry.

        Args:
            version: Version string (e.g., "1.0.0").

        Returns:
            The DeduplicationRules for that version.

        Raises:
            ValueError: If the version is not in the registry.
        """
        if version not in RULES_REGISTRY:
            raise ValueError(
                f"Unknown deduplication rules version: {version}. "
                f"Available versions: {list(RULES_REGISTRY.keys())}"
            )
        return RULES_REGISTRY[version]


# =============================================================================
# Rules Definitions - All Historical Versions
# =============================================================================

# Version 1.0.0 - Initial rules definition
DEDUPLICATION_RULES_V1_0_0 = DeduplicationRules(
    version="1.0.0",
    family_dependencies={
        # Morphology depends only on spatial/mask preprocessing
        "morphology": frozenset({
            "resample",
            "binarize_mask",
            "keep_largest_component",
        }),
        # Intensity features depend on intensity preprocessing but NOT discretization
        "intensity": frozenset({
            "resample",
            "resegment",
            "filter_outliers",
            "filter",
        }),
        "spatial_intensity": frozenset({
            "resample",
            "resegment",
            "filter_outliers",
            "filter",
        }),
        "local_intensity": frozenset({
            "resample",
            "resegment",
            "filter_outliers",
            "filter",
        }),
        # Histogram depends on discretization
        "histogram": frozenset({
            "resample",
            "resegment",
            "filter_outliers",
            "filter",
            "binarize_mask",
            "keep_largest_component",
            "discretise",
        }),
        # IVH - by default depends on discretization (unless ivh_use_continuous=True)
        "ivh": frozenset({
            "resample",
            "resegment",
            "filter_outliers",
            "filter",
            "binarize_mask",
            "keep_largest_component",
            "discretise",
        }),
        # All texture features depend on discretization
        "texture": frozenset({
            "resample",
            "resegment",
            "filter_outliers",
            "filter",
            "binarize_mask",
            "keep_largest_component",
            "discretise",
        }),
        "glcm": frozenset({
            "resample",
            "resegment",
            "filter_outliers",
            "filter",
            "binarize_mask",
            "keep_largest_component",
            "discretise",
        }),
        "glrlm": frozenset({
            "resample",
            "resegment",
            "filter_outliers",
            "filter",
            "binarize_mask",
            "keep_largest_component",
            "discretise",
        }),
        "glszm": frozenset({
            "resample",
            "resegment",
            "filter_outliers",
            "filter",
            "binarize_mask",
            "keep_largest_component",
            "discretise",
        }),
        "gldzm": frozenset({
            "resample",
            "resegment",
            "filter_outliers",
            "filter",
            "binarize_mask",
            "keep_largest_component",
            "discretise",
        }),
        "ngtdm": frozenset({
            "resample",
            "resegment",
            "filter_outliers",
            "filter",
            "binarize_mask",
            "keep_largest_component",
            "discretise",
        }),
        "ngldm": frozenset({
            "resample",
            "resegment",
            "filter_outliers",
            "filter",
            "binarize_mask",
            "keep_largest_component",
            "discretise",
        }),
    },
    ivh_discretization_dependent_unless="ivh_use_continuous=True",
    comparison_mode="exact_params",
)

# Registry of all rule versions - NEVER remove old versions
RULES_REGISTRY: dict[str, DeduplicationRules] = {
    "1.0.0": DEDUPLICATION_RULES_V1_0_0,
}

# Current default version
CURRENT_RULES_VERSION = "1.0.0"


def get_default_rules() -> DeduplicationRules:
    """Get the current default deduplication rules."""
    return RULES_REGISTRY[CURRENT_RULES_VERSION]


# =============================================================================
# Preprocessing Signature
# =============================================================================


@dataclass(frozen=True)
class PreprocessingSignature:
    """
    A hashable signature representing a preprocessing configuration.

    Contains both a hash for fast comparison and the full JSON representation
    for human-readable debugging and logging.

    Attributes:
        hash: SHA256 hash of the normalized preprocessing steps.
        json_repr: Full JSON string of the preprocessing steps.
    """

    hash: str
    json_repr: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PreprocessingSignature):
            return NotImplemented
        return self.hash == other.hash

    def __hash__(self) -> int:
        return hash(self.hash)

    @classmethod
    def from_steps(
        cls, steps: list[tuple[str, dict[str, Any]]]
    ) -> "PreprocessingSignature":
        """
        Create a signature from a list of (step_name, params) tuples.

        Args:
            steps: List of (step_name, params_dict) tuples, sorted by step name.

        Returns:
            A PreprocessingSignature with deterministic hash and JSON.
        """
        # Create deterministic JSON representation
        json_repr = json.dumps(
            {step_name: _normalize_params(params) for step_name, params in steps},
            sort_keys=True,
            separators=(",", ":"),
        )

        # Compute SHA256 hash
        hash_value = hashlib.sha256(json_repr.encode("utf-8")).hexdigest()

        return cls(hash=hash_value, json_repr=json_repr)

    def to_dict(self) -> dict[str, str]:
        """Serialize to dictionary."""
        return {"hash": self.hash, "json": self.json_repr}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "PreprocessingSignature":
        """Deserialize from dictionary."""
        return cls(hash=data["hash"], json_repr=data["json"])


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize parameter values for deterministic JSON serialization.

    Converts numpy arrays, tuples, and other non-JSON types to lists/primitives.
    """
    result = {}
    for key, value in sorted(params.items()):
        if hasattr(value, "tolist"):  # numpy array
            result[key] = value.tolist()
        elif isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, dict):
            result[key] = _normalize_params(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [
                _normalize_params(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            result[key] = value
    return result


# =============================================================================
# Configuration Analysis
# =============================================================================


def get_ivh_dependencies(
    config_steps: list[dict[str, Any]],
    rules: DeduplicationRules,
) -> frozenset[str]:
    """
    Determine IVH feature dependencies based on config parameters.

    If ivh_use_continuous=True is set in extract_features params,
    IVH becomes independent of discretization.

    Args:
        config_steps: List of step dictionaries from the config.
        rules: The deduplication rules to use.

    Returns:
        The set of preprocessing steps that affect IVH for this config.
    """
    # Find extract_features step and check ivh_params
    for step in config_steps:
        if step.get("step") == "extract_features":
            params = step.get("params", {})
            ivh_params = params.get("ivh_params", {})
            if ivh_params.get("ivh_use_continuous", False) is True:
                # Remove discretise from dependencies
                deps = rules.family_dependencies.get("ivh", frozenset())
                return deps - {"discretise"}

    # Default: use full dependencies including discretise
    return rules.family_dependencies.get("ivh", frozenset())


def extract_relevant_steps(
    config_steps: list[dict[str, Any]],
    family: str,
    rules: DeduplicationRules,
) -> list[tuple[str, dict[str, Any]]]:
    """
    Extract preprocessing steps relevant to a feature family.

    Args:
        config_steps: Full list of step dictionaries from config.
        family: The feature family name.
        rules: The deduplication rules to use.

    Returns:
        Sorted list of (step_name, params) tuples for relevant steps.
    """
    # Get dependencies for this family
    if family == "ivh":
        dependencies = get_ivh_dependencies(config_steps, rules)
    else:
        dependencies = rules.family_dependencies.get(family, frozenset())

    # Extract matching steps with their params
    relevant = []
    for step in config_steps:
        step_name = step.get("step", "")
        if step_name in dependencies:
            params = step.get("params", {})
            relevant.append((step_name, params))

    # Sort by step name for deterministic ordering
    relevant.sort(key=lambda x: x[0])
    return relevant


# =============================================================================
# Deduplication Plan
# =============================================================================


@dataclass
class DeduplicationPlan:
    """
    A plan describing which config/family pairs should compute vs. reuse.

    Attributes:
        rules: The DeduplicationRules used to create this plan.
        signatures: Mapping of (config_name, family) to PreprocessingSignature.
        sources: Mapping of (config_name, family) to source config name (or None if first).
        configs_hash: Hash of the configs dict to detect modifications.
    """

    rules: DeduplicationRules
    signatures: dict[tuple[str, str], PreprocessingSignature] = field(
        default_factory=dict
    )
    sources: dict[tuple[str, str], str | None] = field(default_factory=dict)
    configs_hash: str = ""

    def should_compute(self, config_name: str, family: str) -> bool:
        """
        Check if this config/family should be computed fresh.

        Returns True if this is the first occurrence of this signature,
        False if it can be copied from another config.
        """
        return self.sources.get((config_name, family)) is None

    def get_source(self, config_name: str, family: str) -> str | None:
        """
        Get the source config to copy from, or None if should compute.
        """
        return self.sources.get((config_name, family))

    def is_stale(self, current_configs: dict[str, list[dict[str, Any]]]) -> bool:
        """
        Check if this plan is stale due to config modifications.
        """
        current_hash = _hash_configs(current_configs)
        return current_hash != self.configs_hash

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan to a dictionary."""
        return {
            "rules": self.rules.to_dict(),
            "configs_hash": self.configs_hash,
            "signatures": [
                {
                    "config": config,
                    "family": family,
                    **sig.to_dict(),
                }
                for (config, family), sig in self.signatures.items()
            ],
            "sources": [
                {
                    "config": config,
                    "family": family,
                    "source": source,
                }
                for (config, family), source in self.sources.items()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeduplicationPlan":
        """Deserialize a plan from a dictionary."""
        rules = DeduplicationRules.from_dict(data["rules"])

        signatures = {}
        for item in data.get("signatures", []):
            key = (item["config"], item["family"])
            signatures[key] = PreprocessingSignature.from_dict(item)

        sources = {}
        for item in data.get("sources", []):
            key = (item["config"], item["family"])
            sources[key] = item["source"]

        return cls(
            rules=rules,
            signatures=signatures,
            sources=sources,
            configs_hash=data.get("configs_hash", ""),
        )

    def get_summary(self) -> dict[str, int]:
        """
        Get a summary of the deduplication plan.

        Returns:
            Dict with counts of computed vs reused families.
        """
        computed = sum(1 for s in self.sources.values() if s is None)
        reused = sum(1 for s in self.sources.values() if s is not None)
        return {"computed": computed, "reused": reused, "total": computed + reused}


def _hash_configs(configs: dict[str, list[dict[str, Any]]]) -> str:
    """Create a hash of the configs dict for staleness detection."""
    json_repr = json.dumps(
        {k: v for k, v in sorted(configs.items())},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(json_repr.encode("utf-8")).hexdigest()


# =============================================================================
# Configuration Analyzer
# =============================================================================


class ConfigurationAnalyzer:
    """
    Analyzes multiple configurations to create a deduplication plan.

    Compares preprocessing steps across configurations for each feature family
    and identifies which config/family pairs produce identical results.

    Args:
        configs: Dict mapping config names to lists of step dicts.
        rules: The DeduplicationRules to use (defaults to current version).
    """

    def __init__(
        self,
        configs: dict[str, list[dict[str, Any]]],
        rules: DeduplicationRules | None = None,
    ):
        self.configs = configs
        self.rules = rules or get_default_rules()

    def analyze(self) -> DeduplicationPlan:
        """
        Analyze configurations and create a deduplication plan.

        Returns:
            A DeduplicationPlan mapping each config/family to its source.
        """
        plan = DeduplicationPlan(
            rules=self.rules,
            configs_hash=_hash_configs(self.configs),
        )

        # Get all feature families from rules
        all_families = set(self.rules.family_dependencies.keys())

        # Track first occurrence of each signature per family
        # signature_hash -> (first_config_name, signature)
        first_occurrence: dict[str, dict[str, tuple[str, PreprocessingSignature]]] = {
            family: {} for family in all_families
        }

        # Process each config
        for config_name, steps in self.configs.items():
            # Determine which families this config extracts
            families_in_config = self._get_families_in_config(steps)

            for family in families_in_config:
                if family not in all_families:
                    # Unknown family, skip
                    continue

                # Extract relevant preprocessing steps
                relevant_steps = extract_relevant_steps(steps, family, self.rules)

                # Create signature
                signature = PreprocessingSignature.from_steps(relevant_steps)
                plan.signatures[(config_name, family)] = signature

                # Check if this signature was seen before
                if signature.hash in first_occurrence[family]:
                    # Reuse from first config with this signature
                    source_config, _ = first_occurrence[family][signature.hash]
                    plan.sources[(config_name, family)] = source_config
                else:
                    # First occurrence - compute fresh
                    first_occurrence[family][signature.hash] = (config_name, signature)
                    plan.sources[(config_name, family)] = None

        return plan

    def _get_families_in_config(self, steps: list[dict[str, Any]]) -> set[str]:
        """
        Determine which feature families a config will extract.
        """
        families: set[str] = set()

        for step in steps:
            if step.get("step") == "extract_features":
                params = step.get("params", {})
                # Get explicitly listed families
                family_list = params.get(
                    "families", ["intensity", "morphology", "texture", "histogram", "ivh"]
                )
                families.update(family_list)

                # Check for texture sub-families
                if "texture" in families:
                    # Texture expands to all texture families
                    families.update(["glcm", "glrlm", "glszm", "gldzm", "ngtdm", "ngldm"])

                # Check optional intensity features
                if params.get("include_spatial_intensity", False):
                    families.add("spatial_intensity")
                if params.get("include_local_intensity", False):
                    families.add("local_intensity")

        return families


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "DeduplicationRules",
    "DeduplicationPlan",
    "PreprocessingSignature",
    "ConfigurationAnalyzer",
    "DEDUPLICATION_RULES_V1_0_0",
    "RULES_REGISTRY",
    "CURRENT_RULES_VERSION",
    "get_default_rules",
    "extract_relevant_steps",
    "get_ivh_dependencies",
]
