"""
Tests for the deduplication module.
"""

import json
from typing import Any

import numpy as np
import pytest

from pictologics.deduplication import (
    CURRENT_RULES_VERSION,
    DEDUPLICATION_RULES_V1_0_0,
    RULES_REGISTRY,
    ConfigurationAnalyzer,
    DeduplicationPlan,
    DeduplicationRules,
    PreprocessingSignature,
    _normalize_params,
    extract_relevant_steps,
    get_default_rules,
    get_ivh_dependencies,
)
from pictologics.loader import Image
from pictologics.pipeline import RadiomicsPipeline


class TestDeduplicationRules:
    """Tests for DeduplicationRules dataclass."""

    def test_rules_are_frozen(self):
        """Rules should be immutable after creation."""
        rules = get_default_rules()
        with pytest.raises(AttributeError):
            rules.version = "99.0.0"  # type: ignore

    def test_default_rules_version(self):
        """Default rules should have correct version."""
        rules = get_default_rules()
        assert rules.version == CURRENT_RULES_VERSION

    def test_get_version_returns_registered_rules(self):
        """get_version should return registered rules."""
        rules = DeduplicationRules.get_version("1.0.0")
        assert rules.version == "1.0.0"
        assert rules == DEDUPLICATION_RULES_V1_0_0

    def test_get_version_raises_for_unknown(self):
        """get_version should raise for unknown versions."""
        with pytest.raises(ValueError, match="Unknown deduplication rules version"):
            DeduplicationRules.get_version("999.0.0")

    def test_rules_registry_contains_v1(self):
        """Registry should contain v1.0.0 rules."""
        assert "1.0.0" in RULES_REGISTRY
        assert RULES_REGISTRY["1.0.0"] == DEDUPLICATION_RULES_V1_0_0

    def test_rules_serialization_roundtrip(self):
        """Rules should serialize and deserialize correctly."""
        rules = get_default_rules()
        data = rules.to_dict()
        restored = DeduplicationRules.from_dict(data)
        assert restored.version == rules.version
        assert restored.family_dependencies == rules.family_dependencies

    def test_rules_json_serialization(self):
        """Rules should be JSON serializable."""
        rules = get_default_rules()
        data = rules.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = DeduplicationRules.from_dict(restored_data)
        assert restored.version == rules.version


class TestPreprocessingSignature:
    """Tests for PreprocessingSignature dataclass."""

    def test_signature_hash_format(self):
        """Signature hash should be a valid hex string."""
        sig = PreprocessingSignature.from_steps([
            ("resample", {"new_spacing": [1.0, 1.0, 1.0]})
        ])
        assert len(sig.hash) == 64  # SHA256 hex digest
        assert all(c in "0123456789abcdef" for c in sig.hash)

    def test_signature_json_is_valid(self):
        """Signature json should be valid JSON."""
        sig = PreprocessingSignature.from_steps([
            ("resample", {"new_spacing": [1.0, 1.0, 1.0]})
        ])
        parsed = json.loads(sig.json_repr)
        assert isinstance(parsed, dict)  # JSON is a dict mapping step names to params

    def test_identical_steps_same_hash(self):
        """Identical preprocessing steps should produce same hash."""
        steps = [
            ("resample", {"new_spacing": [1.0, 1.0, 1.0]}),
            ("filter_outliers", {"sigma": 3.0}),
        ]
        sig1 = PreprocessingSignature.from_steps(steps)
        sig2 = PreprocessingSignature.from_steps(steps)
        assert sig1.hash == sig2.hash

    def test_different_steps_different_hash(self):
        """Different preprocessing steps should produce different hash."""
        sig1 = PreprocessingSignature.from_steps([
            ("resample", {"new_spacing": [1.0, 1.0, 1.0]})
        ])
        sig2 = PreprocessingSignature.from_steps([
            ("resample", {"new_spacing": [2.0, 2.0, 2.0]})
        ])
        assert sig1.hash != sig2.hash

    def test_param_order_does_not_affect_hash(self):
        """Parameter order in dict should not affect hash (sorted keys)."""
        steps1 = [("resample", {"new_spacing": [1.0, 1.0, 1.0], "interpolation": "linear"})]
        steps2 = [("resample", {"interpolation": "linear", "new_spacing": [1.0, 1.0, 1.0]})]
        sig1 = PreprocessingSignature.from_steps(steps1)
        sig2 = PreprocessingSignature.from_steps(steps2)
        assert sig1.hash == sig2.hash

    def test_empty_steps_valid_signature(self):
        """Empty steps list should produce valid signature."""
        sig = PreprocessingSignature.from_steps([])
        assert sig.hash
        assert sig.json_repr == "{}"  # Empty dict as JSON

    def test_serialization_roundtrip(self):
        """Signature should serialize and deserialize correctly."""
        sig = PreprocessingSignature.from_steps([
            ("resample", {"new_spacing": [1.0, 1.0, 1.0]})
        ])
        data = sig.to_dict()
        restored = PreprocessingSignature.from_dict(data)
        assert restored.hash == sig.hash
        assert restored.json_repr == sig.json_repr


class TestExtractRelevantSteps:
    """Tests for extract_relevant_steps helper function."""

    @pytest.fixture
    def sample_config(self) -> list[dict[str, Any]]:
        """Sample configuration with various steps."""
        return [
            {"step": "resample", "params": {"new_spacing": [1.0, 1.0, 1.0]}},
            {"step": "filter_outliers", "params": {"sigma": 3.0}},
            {"step": "resegment", "params": {"range_min": -100, "range_max": 400}},
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["morphology", "intensity", "texture"]}},
        ]

    def test_morphology_excludes_discretisation(self, sample_config: list[dict[str, Any]]):
        """Morphology should not include discretisation step."""
        rules = get_default_rules()
        relevant = extract_relevant_steps(sample_config, "morphology", rules)
        step_names = [step_name for step_name, params in relevant]
        assert "resample" in step_names
        assert "discretise" not in step_names
        assert "extract_features" not in step_names

    def test_intensity_excludes_discretisation(self, sample_config: list[dict[str, Any]]):
        """Intensity should not include discretisation step."""
        rules = get_default_rules()
        relevant = extract_relevant_steps(sample_config, "intensity", rules)
        step_names = [step_name for step_name, params in relevant]
        assert "discretise" not in step_names

    def test_texture_includes_discretisation(self, sample_config: list[dict[str, Any]]):
        """Texture should include discretisation step."""
        rules = get_default_rules()
        relevant = extract_relevant_steps(sample_config, "texture", rules)
        step_names = [step_name for step_name, params in relevant]
        assert "discretise" in step_names

    def test_histogram_includes_discretisation(self, sample_config: list[dict[str, Any]]):
        """Histogram should include discretisation step."""
        rules = get_default_rules()
        relevant = extract_relevant_steps(sample_config, "histogram", rules)
        step_names = [step_name for step_name, params in relevant]
        assert "discretise" in step_names


class TestConfigurationAnalyzer:
    """Tests for ConfigurationAnalyzer class."""

    @pytest.fixture
    def two_configs_same_preprocessing(self) -> dict[str, list[dict[str, Any]]]:
        """Two configs with same preprocessing but different discretisation."""
        base_steps = [
            {"step": "resample", "params": {"new_spacing": [1.0, 1.0, 1.0]}},
            {"step": "filter_outliers", "params": {"sigma": 3.0}},
        ]
        return {
            "config_fbn_32": base_steps + [
                {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
                {"step": "extract_features", "params": {"families": ["morphology", "intensity", "texture"]}},
            ],
            "config_fbn_64": base_steps + [
                {"step": "discretise", "params": {"method": "FBN", "n_bins": 64}},
                {"step": "extract_features", "params": {"families": ["morphology", "intensity", "texture"]}},
            ],
        }

    @pytest.fixture
    def two_configs_different_preprocessing(self) -> dict[str, list[dict[str, Any]]]:
        """Two configs with different preprocessing."""
        return {
            "config_1mm": [
                {"step": "resample", "params": {"new_spacing": [1.0, 1.0, 1.0]}},
                {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
                {"step": "extract_features", "params": {"families": ["morphology", "intensity"]}},
            ],
            "config_2mm": [
                {"step": "resample", "params": {"new_spacing": [2.0, 2.0, 2.0]}},
                {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
                {"step": "extract_features", "params": {"families": ["morphology", "intensity"]}},
            ],
        }

    def test_analyzer_creates_plan(self, two_configs_same_preprocessing: dict[str, list[dict[str, Any]]]):
        """Analyzer should create a valid DeduplicationPlan."""
        analyzer = ConfigurationAnalyzer(two_configs_same_preprocessing)
        plan = analyzer.analyze()
        assert isinstance(plan, DeduplicationPlan)
        assert plan.rules.version == CURRENT_RULES_VERSION

    def test_same_preprocessing_shared_morphology(self, two_configs_same_preprocessing: dict[str, list[dict[str, Any]]]):
        """Configs with same preprocessing should share morphology signature."""
        analyzer = ConfigurationAnalyzer(two_configs_same_preprocessing)
        plan = analyzer.analyze()

        sig1 = plan.signatures.get(("config_fbn_32", "morphology"))
        sig2 = plan.signatures.get(("config_fbn_64", "morphology"))

        assert sig1 is not None
        assert sig2 is not None
        assert sig1.hash == sig2.hash, "Same preprocessing should yield same morphology signature"

    def test_same_preprocessing_shared_intensity(self, two_configs_same_preprocessing: dict[str, list[dict[str, Any]]]):
        """Configs with same preprocessing should share intensity signature."""
        analyzer = ConfigurationAnalyzer(two_configs_same_preprocessing)
        plan = analyzer.analyze()

        sig1 = plan.signatures.get(("config_fbn_32", "intensity"))
        sig2 = plan.signatures.get(("config_fbn_64", "intensity"))

        assert sig1 is not None
        assert sig2 is not None
        assert sig1.hash == sig2.hash, "Same preprocessing should yield same intensity signature"

    def test_different_discretisation_different_texture(self, two_configs_same_preprocessing: dict[str, list[dict[str, Any]]]):
        """Configs with different discretisation should have different texture signatures."""
        analyzer = ConfigurationAnalyzer(two_configs_same_preprocessing)
        plan = analyzer.analyze()

        sig1 = plan.signatures.get(("config_fbn_32", "texture"))
        sig2 = plan.signatures.get(("config_fbn_64", "texture"))

        assert sig1 is not None
        assert sig2 is not None
        assert sig1.hash != sig2.hash, "Different discretisation should yield different texture signature"

    def test_different_preprocessing_different_morphology(self, two_configs_different_preprocessing: dict[str, list[dict[str, Any]]]):
        """Configs with different preprocessing should have different morphology signatures."""
        analyzer = ConfigurationAnalyzer(two_configs_different_preprocessing)
        plan = analyzer.analyze()

        sig1 = plan.signatures.get(("config_1mm", "morphology"))
        sig2 = plan.signatures.get(("config_2mm", "morphology"))

        assert sig1 is not None
        assert sig2 is not None
        assert sig1.hash != sig2.hash, "Different preprocessing should yield different morphology signature"

    def test_sources_identify_first_occurrence(self, two_configs_same_preprocessing: dict[str, list[dict[str, Any]]]):
        """Sources should identify first occurrence as None, subsequent as source config."""
        analyzer = ConfigurationAnalyzer(two_configs_same_preprocessing)
        plan = analyzer.analyze()

        # Get configs in order
        config_names = list(two_configs_same_preprocessing.keys())
        first_config = config_names[0]
        second_config = config_names[1]

        # First config should have None source (compute)
        assert plan.sources.get((first_config, "morphology")) is None

        # Second config should reference first config
        source = plan.sources.get((second_config, "morphology"))
        assert source == first_config, f"Expected source to be {first_config}, got {source}"


class TestDeduplicationPlan:
    """Tests for DeduplicationPlan class."""

    def test_should_compute_for_first_occurrence(self):
        """should_compute should return True for first occurrence."""
        plan = DeduplicationPlan(
            rules=get_default_rules(),
            sources={
                ("config1", "morphology"): None,
                ("config2", "morphology"): "config1",
            },
        )
        assert plan.should_compute("config1", "morphology") is True
        assert plan.should_compute("config2", "morphology") is False

    def test_get_source_returns_source_config(self):
        """get_source should return the source config name."""
        plan = DeduplicationPlan(
            rules=get_default_rules(),
            sources={
                ("config1", "morphology"): None,
                ("config2", "morphology"): "config1",
            },
        )
        assert plan.get_source("config1", "morphology") is None
        assert plan.get_source("config2", "morphology") == "config1"

    def test_is_stale_detects_modification(self):
        """is_stale should detect when configs have changed."""
        configs = {"test": [{"step": "extract_features", "params": {}}]}
        analyzer = ConfigurationAnalyzer(configs)
        plan = analyzer.analyze()

        # Same configs should not be stale
        assert plan.is_stale(configs) is False

        # Modified configs should be stale
        modified_configs = {"test": [{"step": "resample", "params": {"new_spacing": [1.0, 1.0, 1.0]}}]}
        assert plan.is_stale(modified_configs) is True

    def test_serialization_roundtrip(self):
        """Plan should serialize and deserialize correctly."""
        configs = {
            "config1": [
                {"step": "resample", "params": {"new_spacing": [1.0, 1.0, 1.0]}},
                {"step": "extract_features", "params": {"families": ["morphology"]}},
            ],
            "config2": [
                {"step": "resample", "params": {"new_spacing": [1.0, 1.0, 1.0]}},
                {"step": "extract_features", "params": {"families": ["morphology"]}},
            ],
        }
        analyzer = ConfigurationAnalyzer(configs)
        plan = analyzer.analyze()

        data = plan.to_dict()
        restored = DeduplicationPlan.from_dict(data)

        assert restored.rules.version == plan.rules.version
        assert restored.configs_hash == plan.configs_hash
        assert len(restored.signatures) == len(plan.signatures)
        assert len(restored.sources) == len(plan.sources)

    def test_get_summary(self):
        """get_summary should return correct counts of computed vs reused."""
        plan = DeduplicationPlan(
            rules=get_default_rules(),
            sources={
                ("config1", "morphology"): None,  # computed
                ("config1", "intensity"): None,   # computed
                ("config2", "morphology"): "config1",  # reused
                ("config2", "intensity"): "config1",   # reused
                ("config3", "morphology"): "config1",  # reused
                ("config3", "intensity"): "config1",   # reused
            },
        )
        summary = plan.get_summary()

        assert summary["computed"] == 2
        assert summary["reused"] == 4
        assert summary["total"] == 6


class TestPipelineDeduplicationIntegration:
    """Integration tests for deduplication in RadiomicsPipeline."""

    @pytest.fixture
    def simple_image(self) -> Image:
        """Create a simple test image."""
        array = np.random.rand(10, 10, 10).astype(np.float32) * 100
        return Image(
            array=array,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            modality="CT",
        )

    @pytest.fixture
    def simple_mask(self) -> Image:
        """Create a simple test mask."""
        array = np.zeros((10, 10, 10), dtype=np.int16)
        array[3:7, 3:7, 3:7] = 1  # Small central ROI
        return Image(
            array=array,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            modality="SEG",
        )

    def test_pipeline_default_deduplication_enabled(self):
        """Pipeline should have deduplication enabled by default."""
        pipeline = RadiomicsPipeline()
        assert pipeline.deduplication_enabled is True

    def test_pipeline_deduplication_can_be_disabled(self):
        """Pipeline deduplication can be disabled."""
        pipeline = RadiomicsPipeline(deduplicate=False)
        assert pipeline.deduplication_enabled is False

    def test_pipeline_deduplication_rules_by_version(self):
        """Pipeline can load rules by version string."""
        pipeline = RadiomicsPipeline(deduplication_rules="1.0.0")
        assert pipeline.deduplication_rules.version == "1.0.0"

    def test_pipeline_serialization_includes_deduplication(self):
        """Pipeline serialization should include deduplication settings."""
        pipeline = RadiomicsPipeline(deduplicate=True)
        data = pipeline.to_dict()

        assert "deduplication" in data
        assert data["deduplication"]["enabled"] is True
        assert "rules_version" in data["deduplication"]

    def test_pipeline_deserialization_restores_deduplication(self):
        """Pipeline deserialization should restore deduplication settings."""
        original = RadiomicsPipeline(deduplicate=False)
        data = original.to_dict()

        restored = RadiomicsPipeline.from_dict(data)

        assert restored.deduplication_enabled is False

    def test_pipeline_run_with_deduplication_single_config(
        self,
        simple_image: Image,
        simple_mask: Image,
    ):
        """Pipeline run should work with single config (no deduplication needed)."""
        pipeline = RadiomicsPipeline()
        pipeline.add_config("simple", [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ])

        results = pipeline.run(simple_image, simple_mask, config_names=["simple"])
        assert "simple" in results
        assert len(results["simple"]) > 0

    def test_deduplication_stats_empty_before_run(self):
        """deduplication_stats should return empty dict with warning before run()."""
        import warnings

        pipeline = RadiomicsPipeline(deduplicate=True)
        pipeline.add_config("test", [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stats = pipeline.deduplication_stats
            assert stats == {}
            assert len(w) == 1
            assert "No features were extracted" in str(w[0].message)

    def test_deduplication_stats_reset_between_runs(
        self,
        simple_image: Image,
        simple_mask: Image,
    ):
        """deduplication_stats should reset between runs."""
        pipeline = RadiomicsPipeline(deduplicate=True)

        # Two configs with same preprocessing, different discretization
        base_steps = [
            {"step": "resegment", "params": {"range_min": 0, "range_max": 100}},
        ]
        pipeline.add_config("fbn_8", base_steps + [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
            {"step": "extract_features", "params": {"families": ["intensity", "morphology"]}},
        ])
        pipeline.add_config("fbn_16", base_steps + [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 16}},
            {"step": "extract_features", "params": {"families": ["intensity", "morphology"]}},
        ])

        # First run
        pipeline.run(simple_image, simple_mask, config_names=["fbn_8", "fbn_16"])
        stats1 = pipeline.deduplication_stats

        # Second run - stats should reset
        pipeline.run(simple_image, simple_mask, config_names=["fbn_8", "fbn_16"])
        stats2 = pipeline.deduplication_stats

        # Both runs should have same stats (reset worked)
        assert stats1 == stats2
        assert stats1["reused_families"] > 0 or stats1["computed_families"] > 0

    def test_deduplication_stats_correct_counts(
        self,
        simple_image: Image,
        simple_mask: Image,
    ):
        """deduplication_stats should have correct counts after multi-config run."""
        pipeline = RadiomicsPipeline(deduplicate=True)

        # Two configs with same preprocessing -> morphology/intensity reused
        base_steps = [
            {"step": "resegment", "params": {"range_min": 0, "range_max": 100}},
        ]
        pipeline.add_config("fbn_8", base_steps + [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
            {"step": "extract_features", "params": {"families": ["morphology", "intensity"]}},
        ])
        pipeline.add_config("fbn_16", base_steps + [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 16}},
            {"step": "extract_features", "params": {"families": ["morphology", "intensity"]}},
        ])

        pipeline.run(simple_image, simple_mask, config_names=["fbn_8", "fbn_16"])
        stats = pipeline.deduplication_stats

        assert "reused_families" in stats
        assert "computed_families" in stats
        assert "cache_hit_rate" in stats
        assert stats["reused_families"] >= 0
        assert stats["computed_families"] >= 0
        assert 0 <= stats["cache_hit_rate"] <= 1

        # Morphology and intensity should be reused for second config
        total = stats["reused_families"] + stats["computed_families"]
        assert total > 0

    def test_deduplication_stats_keys(
        self,
        simple_image: Image,
        simple_mask: Image,
    ):
        """deduplication_stats should return dict with expected keys when dedup is active."""
        pipeline = RadiomicsPipeline(deduplicate=True)
        # Need at least 2 configs to trigger deduplication
        pipeline.add_config("config1", [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ])
        pipeline.add_config("config2", [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ])

        pipeline.run(simple_image, simple_mask, config_names=["config1", "config2"])
        stats = pipeline.deduplication_stats

        expected_keys = {"reused_families", "computed_families", "cache_hit_rate"}
        assert set(stats.keys()) == expected_keys
        # With identical configs, second should reuse first
        assert stats["reused_families"] >= 1

    def test_deduplication_stats_single_config_returns_empty(
        self,
        simple_image: Image,
        simple_mask: Image,
    ):
        """deduplication_stats returns empty dict with warning when only 1 config."""
        pipeline = RadiomicsPipeline(deduplicate=True)
        pipeline.add_config("single", [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ])

        # Single config doesn't trigger deduplication
        pipeline.run(simple_image, simple_mask, config_names=["single"])

        with pytest.warns(UserWarning, match="No features were extracted with deduplication"):
            stats = pipeline.deduplication_stats
        assert stats == {}

    def test_pipeline_property_setters(self):
        """Pipeline deduplication properties can be set after creation."""
        pipeline = RadiomicsPipeline(deduplicate=True)

        pipeline.deduplication_enabled = False
        assert pipeline.deduplication_enabled is False

        pipeline.deduplication_rules = "1.0.0"
        assert pipeline.deduplication_rules.version == "1.0.0"


class TestSchemaVersionMigration:
    """Tests for schema version migration."""

    def test_v1_0_config_gets_deduplication_defaults(self):
        """v1.0 configs should get deduplication defaults on load."""
        v1_0_data = {
            "schema_version": "1.0",
            "configs": {
                "test": {
                    "steps": [
                        {"step": "extract_features", "params": {"families": ["intensity"]}}
                    ]
                }
            },
        }

        pipeline = RadiomicsPipeline.from_dict(v1_0_data)

        # Should have deduplication enabled (default)
        assert pipeline.deduplication_enabled is True


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestPreprocessingSignatureEdgeCases:
    """Additional edge case tests for PreprocessingSignature."""

    def test_signature_eq_with_non_signature(self):
        """Comparing signature with non-signature should return NotImplemented."""
        sig = PreprocessingSignature.from_steps([("resample", {"new_spacing": [1.0, 1.0, 1.0]})])
        result = sig.__eq__("not a signature")
        assert result is NotImplemented

    def test_signature_eq_with_same_signature(self):
        """Comparing signature with equal signature should return True."""
        sig1 = PreprocessingSignature.from_steps([("resample", {"new_spacing": [1.0, 1.0, 1.0]})])
        sig2 = PreprocessingSignature.from_steps([("resample", {"new_spacing": [1.0, 1.0, 1.0]})])
        assert sig1 == sig2
        assert sig1.__eq__(sig2) is True

    def test_signature_eq_with_different_signature(self):
        """Comparing signature with different signature should return False."""
        sig1 = PreprocessingSignature.from_steps([("resample", {"new_spacing": [1.0, 1.0, 1.0]})])
        sig2 = PreprocessingSignature.from_steps([("resample", {"new_spacing": [2.0, 2.0, 2.0]})])
        assert sig1 != sig2
        assert sig1.__eq__(sig2) is False

    def test_signature_hash_method(self):
        """Signature __hash__ should work correctly."""
        sig = PreprocessingSignature.from_steps([("resample", {"new_spacing": [1.0, 1.0, 1.0]})])
        # Should be hashable and work in sets
        sig_set = {sig}
        assert sig in sig_set

    def test_normalize_params_with_numpy_array(self):
        """_normalize_params should convert numpy arrays to lists."""
        params = {"spacing": np.array([1.0, 2.0, 3.0])}
        result = _normalize_params(params)
        assert result["spacing"] == [1.0, 2.0, 3.0]
        assert isinstance(result["spacing"], list)

    def test_normalize_params_with_tuple(self):
        """_normalize_params should convert tuples to lists."""
        params = {"spacing": (1.0, 2.0, 3.0)}
        result = _normalize_params(params)
        assert result["spacing"] == [1.0, 2.0, 3.0]
        assert isinstance(result["spacing"], list)

    def test_normalize_params_with_nested_dict(self):
        """_normalize_params should recursively normalize nested dicts."""
        params = {
            "outer": {
                "inner": (1, 2, 3),
                "value": 42,
            }
        }
        result = _normalize_params(params)
        assert result["outer"]["inner"] == [1, 2, 3]
        assert result["outer"]["value"] == 42

    def test_normalize_params_with_list_of_dicts(self):
        """_normalize_params should normalize dicts within lists."""
        params = {
            "items": [
                {"a": (1, 2)},
                {"b": 3},
            ]
        }
        result = _normalize_params(params)
        assert result["items"][0]["a"] == [1, 2]
        assert result["items"][1]["b"] == 3


class TestIVHDependencies:
    """Tests for IVH dependency handling."""

    def test_ivh_continuous_mode_removes_discretise_dependency(self):
        """When ivh_use_continuous=True, discretise should not be a dependency."""
        rules = get_default_rules()
        config_steps = [
            {"step": "resample", "params": {"new_spacing": [1.0, 1.0, 1.0]}},
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {
                "families": ["ivh"],
                "ivh_params": {"ivh_use_continuous": True}
            }},
        ]

        deps = get_ivh_dependencies(config_steps, rules)
        assert "discretise" not in deps

    def test_ivh_default_mode_includes_discretise_dependency(self):
        """By default, IVH depends on discretise."""
        rules = get_default_rules()
        config_steps = [
            {"step": "resample", "params": {"new_spacing": [1.0, 1.0, 1.0]}},
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["ivh"]}},
        ]

        deps = get_ivh_dependencies(config_steps, rules)
        assert "discretise" in deps


class TestConfigurationAnalyzerEdgeCases:
    """Additional edge case tests for ConfigurationAnalyzer."""

    def test_analyzer_skips_unknown_families(self):
        """Analyzer should skip unknown families without error."""
        # This tests line 575: unknown family skip
        configs = {
            "test": [
                {"step": "extract_features", "params": {"families": ["unknown_family", "intensity"]}}
            ]
        }
        analyzer = ConfigurationAnalyzer(configs)
        plan = analyzer.analyze()
        # Should have signature for intensity but not unknown_family
        assert ("test", "intensity") in plan.signatures
        assert ("test", "unknown_family") not in plan.signatures

    def test_analyzer_handles_spatial_intensity_flag(self):
        """Analyzer should add spatial_intensity when include_spatial_intensity=True."""
        configs = {
            "test": [
                {"step": "extract_features", "params": {
                    "families": ["intensity"],
                    "include_spatial_intensity": True,
                }}
            ]
        }
        analyzer = ConfigurationAnalyzer(configs)
        plan = analyzer.analyze()
        assert ("test", "spatial_intensity") in plan.signatures

    def test_analyzer_handles_local_intensity_flag(self):
        """Analyzer should add local_intensity when include_local_intensity=True."""
        configs = {
            "test": [
                {"step": "extract_features", "params": {
                    "families": ["intensity"],
                    "include_local_intensity": True,
                }}
            ]
        }
        analyzer = ConfigurationAnalyzer(configs)
        plan = analyzer.analyze()
        assert ("test", "local_intensity") in plan.signatures

    def test_analyzer_expands_texture_to_subfamilies(self):
        """Analyzer should expand 'texture' to individual texture families."""
        configs = {
            "test": [
                {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
                {"step": "extract_features", "params": {"families": ["texture"]}}
            ]
        }
        analyzer = ConfigurationAnalyzer(configs)
        plan = analyzer.analyze()

        # Should have signatures for all texture sub-families
        for subfam in ["glcm", "glrlm", "glszm", "gldzm", "ngtdm", "ngldm"]:
            assert ("test", subfam) in plan.signatures
