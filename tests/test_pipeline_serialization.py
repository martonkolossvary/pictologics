"""
Tests for Pipeline Configuration Serialization
==============================================

Tests for YAML/JSON export/import, templates, validation, and schema migration.
"""

from __future__ import annotations

# ruff: noqa: E402
import warnings

warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

import pytest
import yaml

from pictologics.pipeline import CONFIG_SCHEMA_VERSION, RadiomicsPipeline
from pictologics.templates import (
    get_all_templates,
    get_standard_templates,
    get_template_metadata,
    list_template_files,
    load_template_file,
)

# --- Fixtures ---


@pytest.fixture
def pipeline() -> RadiomicsPipeline:
    """A fresh RadiomicsPipeline instance."""
    return RadiomicsPipeline()


@pytest.fixture
def custom_config() -> list[dict]:
    """A custom configuration for testing."""
    return [
        {
            "step": "resample",
            "params": {"new_spacing": (1.0, 1.0, 1.0), "interpolation": "linear"},
        },
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 16}},
        {
            "step": "extract_features",
            "params": {"families": ["intensity", "morphology"]},
        },
    ]


# --- Template Loading Tests ---


class TestTemplateLoading:
    """Tests for template file loading utilities."""

    def test_list_template_files(self) -> None:
        """Test listing available template files."""
        files = list_template_files()
        assert isinstance(files, list)
        assert "standard_configs.yaml" in files

    def test_load_template_file(self) -> None:
        """Test loading a specific template file."""
        data = load_template_file("standard_configs.yaml")
        assert isinstance(data, dict)
        assert "schema_version" in data
        assert "configs" in data

    def test_load_template_file_not_found(self) -> None:
        """Test error handling for missing template file."""
        with pytest.raises(FileNotFoundError):
            load_template_file("nonexistent.yaml")

    def test_get_standard_templates(self) -> None:
        """Test getting standard configuration templates."""
        templates = get_standard_templates()
        assert isinstance(templates, dict)
        assert "standard_fbn_32" in templates
        assert "standard_fbs_16" in templates
        # Verify structure
        for _name, steps in templates.items():
            assert isinstance(steps, list)
            for step in steps:
                assert "step" in step

    def test_get_all_templates(self) -> None:
        """Test getting all templates from all files."""
        templates = get_all_templates()
        assert isinstance(templates, dict)
        # Should include standard configs
        assert "standard_fbn_32" in templates

    def test_get_template_metadata(self) -> None:
        """Test getting metadata from a template file."""
        metadata = get_template_metadata("standard_configs.yaml")
        assert metadata["schema_version"] == "1.0"
        assert "standard_fbn_32" in metadata["config_names"]


# --- Standard Config Tests ---


class TestStandardConfigs:
    """Tests for standard configuration loading from templates."""

    def test_standard_configs_loaded(self, pipeline: RadiomicsPipeline) -> None:
        """Test that standard configs are loaded from templates."""
        standard_names = pipeline.get_all_standard_config_names()
        assert len(standard_names) == 6
        assert "standard_fbn_8" in standard_names
        assert "standard_fbn_16" in standard_names
        assert "standard_fbn_32" in standard_names
        assert "standard_fbs_8" in standard_names
        assert "standard_fbs_16" in standard_names
        assert "standard_fbs_32" in standard_names

    def test_standard_config_structure(self, pipeline: RadiomicsPipeline) -> None:
        """Test that standard configs have correct structure."""
        config = pipeline.get_config("standard_fbn_32")
        assert len(config) == 3  # resample, discretise, extract_features

        # Check resample step
        assert config[0]["step"] == "resample"
        assert config[0]["params"]["new_spacing"] == (0.5, 0.5, 0.5)

        # Check discretise step
        assert config[1]["step"] == "discretise"
        assert config[1]["params"]["method"] == "FBN"
        assert config[1]["params"]["n_bins"] == 32

        # Check extract_features step
        assert config[2]["step"] == "extract_features"
        assert "families" in config[2]["params"]

    def test_standard_config_new_spacing_is_tuple(
        self, pipeline: RadiomicsPipeline
    ) -> None:
        """Test that new_spacing is converted from YAML list to tuple."""
        config = pipeline.get_config("standard_fbn_32")
        new_spacing = config[0]["params"]["new_spacing"]
        assert isinstance(new_spacing, tuple), "new_spacing should be a tuple"


# --- Config Management Tests ---


class TestConfigManagement:
    """Tests for config listing, getting, and removing."""

    def test_list_configs(self, pipeline: RadiomicsPipeline) -> None:
        """Test listing all configs."""
        configs = pipeline.list_configs()
        assert isinstance(configs, list)
        assert len(configs) >= 6  # At least standard configs

    def test_get_config_returns_copy(self, pipeline: RadiomicsPipeline) -> None:
        """Test that get_config returns a deep copy."""
        config1 = pipeline.get_config("standard_fbn_32")
        config2 = pipeline.get_config("standard_fbn_32")
        assert config1 == config2
        # Modify one and verify the other is unchanged
        config1[0]["params"]["new_spacing"] = (2.0, 2.0, 2.0)
        assert config2[0]["params"]["new_spacing"] == (0.5, 0.5, 0.5)

    def test_get_config_not_found(self, pipeline: RadiomicsPipeline) -> None:
        """Test error handling for missing config."""
        with pytest.raises(KeyError, match="not found"):
            pipeline.get_config("nonexistent")

    def test_remove_config(
        self, pipeline: RadiomicsPipeline, custom_config: list
    ) -> None:
        """Test removing a config."""
        pipeline.add_config("to_remove", custom_config)
        assert "to_remove" in pipeline.list_configs()
        pipeline.remove_config("to_remove")
        assert "to_remove" not in pipeline.list_configs()

    def test_remove_config_not_found(self, pipeline: RadiomicsPipeline) -> None:
        """Test error handling for removing missing config."""
        with pytest.raises(KeyError, match="not found"):
            pipeline.remove_config("nonexistent")

    def test_remove_config_returns_self(
        self, pipeline: RadiomicsPipeline, custom_config: list
    ) -> None:
        """Test that remove_config returns self for chaining."""
        pipeline.add_config("temp", custom_config)
        result = pipeline.remove_config("temp")
        assert result is pipeline


# --- Export Tests ---


class TestExportMethods:
    """Tests for to_dict, to_json, to_yaml, and save_configs."""

    def test_to_dict_all_configs(self, pipeline: RadiomicsPipeline) -> None:
        """Test exporting all configs to dict."""
        data = pipeline.to_dict()
        assert "schema_version" in data
        assert data["schema_version"] == CONFIG_SCHEMA_VERSION
        assert "exported_at" in data
        assert "configs" in data
        assert "standard_fbn_32" in data["configs"]

    def test_to_dict_specific_configs(self, pipeline: RadiomicsPipeline) -> None:
        """Test exporting specific configs to dict."""
        data = pipeline.to_dict(config_names=["standard_fbn_32", "standard_fbs_16"])
        assert len(data["configs"]) == 2
        assert "standard_fbn_32" in data["configs"]
        assert "standard_fbs_16" in data["configs"]
        assert "standard_fbn_8" not in data["configs"]

    def test_to_dict_without_metadata(self, pipeline: RadiomicsPipeline) -> None:
        """Test exporting without metadata."""
        data = pipeline.to_dict(include_metadata=False)
        assert "schema_version" not in data
        assert "exported_at" not in data
        assert "configs" in data

    def test_to_json(self, pipeline: RadiomicsPipeline) -> None:
        """Test exporting to JSON string."""
        json_str = pipeline.to_json()
        data = json.loads(json_str)
        assert "schema_version" in data
        assert "configs" in data

    def test_to_yaml(self, pipeline: RadiomicsPipeline) -> None:
        """Test exporting to YAML string."""
        yaml_str = pipeline.to_yaml()
        data = yaml.safe_load(yaml_str)
        assert "schema_version" in data
        assert "configs" in data

    def test_save_configs_json(self, pipeline: RadiomicsPipeline) -> None:
        """Test saving configs to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "configs.json"
            pipeline.save_configs(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert "configs" in data

    def test_save_configs_yaml(self, pipeline: RadiomicsPipeline) -> None:
        """Test saving configs to YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "configs.yaml"
            pipeline.save_configs(path)
            assert path.exists()
            data = yaml.safe_load(path.read_text())
            assert "configs" in data

    def test_save_configs_creates_directory(self, pipeline: RadiomicsPipeline) -> None:
        """Test that save_configs creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "configs.yaml"
            pipeline.save_configs(path)
            assert path.exists()

    def test_save_configs_invalid_extension(self, pipeline: RadiomicsPipeline) -> None:
        """Test error handling for unsupported file extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "configs.txt"
            with pytest.raises(ValueError, match="Unsupported file extension"):
                pipeline.save_configs(path)


# --- Import Tests ---


class TestImportMethods:
    """Tests for from_dict, from_json, from_yaml, and load_configs."""

    def test_from_dict(self, custom_config: list) -> None:
        """Test creating pipeline from dict (only file configs, no standard)."""
        data = {
            "schema_version": "1.0",
            "configs": {"my_config": {"steps": custom_config}},
        }
        pipeline = RadiomicsPipeline.from_dict(data)
        assert "my_config" in pipeline.list_configs()
        config = pipeline.get_config("my_config")
        assert len(config) == 3
        # Standard configs should NOT be loaded
        assert not any(c.startswith("standard_") for c in pipeline.list_configs())
        assert len(pipeline.list_configs()) == 1

    def test_from_dict_with_validation(self, custom_config: list) -> None:
        """Test creating pipeline with validation enabled."""
        data = {
            "schema_version": "1.0",
            "configs": {"my_config": {"steps": custom_config}},
        }
        pipeline = RadiomicsPipeline.from_dict(data, validate=True)
        assert "my_config" in pipeline.list_configs()

    def test_from_json(self, custom_config: list) -> None:
        """Test creating pipeline from JSON string (only file configs, no standard)."""
        data = {
            "schema_version": "1.0",
            "configs": {"my_config": {"steps": custom_config}},
        }
        json_str = json.dumps(data)
        pipeline = RadiomicsPipeline.from_json(json_str)
        assert "my_config" in pipeline.list_configs()
        # Standard configs should NOT be loaded
        assert not any(c.startswith("standard_") for c in pipeline.list_configs())

    def test_from_yaml(self, custom_config: list) -> None:
        """Test creating pipeline from YAML string (only file configs, no standard)."""
        yaml_str = """
schema_version: "1.0"
configs:
  my_config:
    steps:
      - step: resample
        params:
          new_spacing: [1.0, 1.0, 1.0]
      - step: discretise
        params:
          method: FBN
          n_bins: 16
"""
        pipeline = RadiomicsPipeline.from_yaml(yaml_str)
        assert "my_config" in pipeline.list_configs()
        # Standard configs should NOT be loaded
        assert not any(c.startswith("standard_") for c in pipeline.list_configs())
        # Verify tuple conversion
        config = pipeline.get_config("my_config")
        assert config[0]["params"]["new_spacing"] == (1.0, 1.0, 1.0)

    def test_load_configs_json(self, pipeline: RadiomicsPipeline) -> None:
        """Test loading configs from JSON file (only file configs, no standard)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "configs.json"
            pipeline.save_configs(path, config_names=["standard_fbn_32"])

            loaded = RadiomicsPipeline.load_configs(path)
            assert "standard_fbn_32" in loaded.list_configs()
            # Only the saved config should be present, no extra standard configs
            assert len(loaded.list_configs()) == 1

    def test_load_configs_yaml(self, pipeline: RadiomicsPipeline) -> None:
        """Test loading configs from YAML file (only file configs, no standard)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "configs.yaml"
            pipeline.save_configs(path, config_names=["standard_fbn_32"])

            loaded = RadiomicsPipeline.load_configs(path)
            assert "standard_fbn_32" in loaded.list_configs()
            # Only the saved config should be present
            assert len(loaded.list_configs()) == 1

    def test_load_configs_not_found(self) -> None:
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            RadiomicsPipeline.load_configs("/nonexistent/path.yaml")

    def test_load_configs_invalid_extension(self) -> None:
        """Test error handling for unsupported file extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "configs.txt"
            path.write_text("content")
            with pytest.raises(ValueError, match="Unsupported file extension"):
                RadiomicsPipeline.load_configs(path)


# --- Load Standard Parameter Tests ---


class TestLoadStandardParameter:
    """Tests for the load_standard parameter across all loading methods."""

    def test_init_load_standard_true_by_default(self) -> None:
        """Test that RadiomicsPipeline() loads standard configs by default."""
        pipeline = RadiomicsPipeline()
        standard_names = pipeline.get_all_standard_config_names()
        assert len(standard_names) == 6

    def test_init_load_standard_false(self) -> None:
        """Test that RadiomicsPipeline(load_standard=False) creates empty pipeline."""
        pipeline = RadiomicsPipeline(load_standard=False)
        assert len(pipeline.list_configs()) == 0
        assert len(pipeline.get_all_standard_config_names()) == 0

    def test_from_dict_with_load_standard_true(self, custom_config: list) -> None:
        """Test from_dict with load_standard=True includes standard configs."""
        data = {
            "schema_version": "1.0",
            "configs": {"my_config": {"steps": custom_config}},
        }
        pipeline = RadiomicsPipeline.from_dict(data, load_standard=True)
        assert "my_config" in pipeline.list_configs()
        # Standard configs should also be present
        assert any(c.startswith("standard_") for c in pipeline.list_configs())
        assert len(pipeline.list_configs()) > 1

    def test_from_json_with_load_standard_true(self, custom_config: list) -> None:
        """Test from_json with load_standard=True includes standard configs."""
        data = {
            "schema_version": "1.0",
            "configs": {"my_config": {"steps": custom_config}},
        }
        json_str = json.dumps(data)
        pipeline = RadiomicsPipeline.from_json(json_str, load_standard=True)
        assert "my_config" in pipeline.list_configs()
        assert any(c.startswith("standard_") for c in pipeline.list_configs())

    def test_from_yaml_with_load_standard_true(self) -> None:
        """Test from_yaml with load_standard=True includes standard configs."""
        yaml_str = """
schema_version: "1.0"
configs:
  my_config:
    steps:
      - step: resample
        params:
          new_spacing: [1.0, 1.0, 1.0]
"""
        pipeline = RadiomicsPipeline.from_yaml(yaml_str, load_standard=True)
        assert "my_config" in pipeline.list_configs()
        assert any(c.startswith("standard_") for c in pipeline.list_configs())

    def test_load_configs_with_load_standard_true(
        self, pipeline: RadiomicsPipeline
    ) -> None:
        """Test load_configs with load_standard=True includes standard configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "configs.yaml"
            pipeline.save_configs(path, config_names=["standard_fbn_32"])

            loaded = RadiomicsPipeline.load_configs(path, load_standard=True)
            assert "standard_fbn_32" in loaded.list_configs()
            # All 6 standard configs should be present (the loaded one overlaps)
            assert len(loaded.get_all_standard_config_names()) == 6


# --- Roundtrip Tests ---


class TestRoundtrip:
    """Tests for export-import roundtrips."""

    def test_roundtrip_json(self, pipeline: RadiomicsPipeline) -> None:
        """Test JSON roundtrip preserves config."""
        original = pipeline.get_config("standard_fbn_32")

        # Export and reimport
        json_str = pipeline.to_json(config_names=["standard_fbn_32"])
        loaded = RadiomicsPipeline.from_json(json_str)
        roundtrip = loaded.get_config("standard_fbn_32")

        # Compare (accounting for tuple/list conversion in params)
        assert len(original) == len(roundtrip)
        for orig_step, rt_step in zip(original, roundtrip, strict=True):
            assert orig_step["step"] == rt_step["step"]
            # new_spacing should be tuple in both
            if "new_spacing" in orig_step.get("params", {}):
                assert (
                    orig_step["params"]["new_spacing"]
                    == rt_step["params"]["new_spacing"]
                )

    def test_roundtrip_yaml(self, pipeline: RadiomicsPipeline) -> None:
        """Test YAML roundtrip preserves config."""
        original = pipeline.get_config("standard_fbn_32")

        # Export and reimport
        yaml_str = pipeline.to_yaml(config_names=["standard_fbn_32"])
        loaded = RadiomicsPipeline.from_yaml(yaml_str)
        roundtrip = loaded.get_config("standard_fbn_32")

        # Compare
        assert len(original) == len(roundtrip)
        for orig_step, rt_step in zip(original, roundtrip, strict=True):
            assert orig_step["step"] == rt_step["step"]

    def test_roundtrip_file(self, pipeline: RadiomicsPipeline) -> None:
        """Test file save/load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"
            pipeline.save_configs(
                path, config_names=["standard_fbn_32", "standard_fbs_16"]
            )

            loaded = RadiomicsPipeline.load_configs(path)
            assert "standard_fbn_32" in loaded.list_configs()
            assert "standard_fbs_16" in loaded.list_configs()


# --- Merge Tests ---


class TestMergeConfigs:
    """Tests for merging configurations from another pipeline."""

    def test_merge_configs_basic(
        self, pipeline: RadiomicsPipeline, custom_config: list
    ) -> None:
        """Test basic config merging."""
        import warnings

        other = RadiomicsPipeline()
        other.add_config("custom_merge", custom_config)

        initial_count = len(pipeline.list_configs())
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # Suppress expected "already exists" warnings
            pipeline.merge_configs(other)

        assert len(pipeline.list_configs()) == initial_count + 1
        assert "custom_merge" in pipeline.list_configs()

    def test_merge_configs_no_overwrite(
        self, pipeline: RadiomicsPipeline, custom_config: list
    ) -> None:
        """Test that merge doesn't overwrite by default."""
        import warnings

        # Both pipelines have standard_fbn_32
        other = RadiomicsPipeline()
        original = pipeline.get_config("standard_fbn_32")

        # Modify the other pipeline's version
        other._configs["standard_fbn_32"][0]["params"]["new_spacing"] = (2.0, 2.0, 2.0)

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # Suppress expected "already exists" warnings
            pipeline.merge_configs(other)

        # Original should be unchanged
        assert pipeline.get_config("standard_fbn_32") == original

    def test_merge_configs_with_overwrite(self, custom_config: list) -> None:
        """Test merge with overwrite=True."""
        pipeline = RadiomicsPipeline()
        other = RadiomicsPipeline()

        # Modify other's version
        modified_config = pipeline.get_config("standard_fbn_32")
        modified_config[0]["params"]["new_spacing"] = (2.0, 2.0, 2.0)
        other._configs["standard_fbn_32"] = modified_config

        pipeline.merge_configs(other, overwrite=True)

        # Should have the modified version
        assert pipeline.get_config("standard_fbn_32")[0]["params"]["new_spacing"] == (
            2.0,
            2.0,
            2.0,
        )

    def test_merge_configs_returns_self(
        self, pipeline: RadiomicsPipeline, custom_config: list
    ) -> None:
        """Test that merge_configs returns self for chaining."""
        import warnings

        other = RadiomicsPipeline()
        other.add_config("chain_test", custom_config)

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # Suppress expected "already exists" warnings
            result = pipeline.merge_configs(other)
        assert result is pipeline


# --- Validation Tests ---


class TestValidation:
    """Tests for configuration validation."""

    def test_validate_valid_config(self, custom_config: list) -> None:
        """Test validation passes for valid config."""
        is_valid = RadiomicsPipeline._validate_config("test", custom_config)
        assert is_valid is True

    def test_validate_unknown_step_type(self) -> None:
        """Test validation warns for unknown step type."""
        import warnings

        config = [{"step": "unknown_step", "params": {}}]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RadiomicsPipeline._validate_config("test", config)
            assert len(w) >= 1
            assert any(
                "unknown step type" in str(warning.message).lower() for warning in w
            )

    def test_validate_unknown_parameter(self) -> None:
        """Test validation warns for unknown parameter."""
        import warnings

        config = [
            {
                "step": "resample",
                "params": {"new_spacing": (1.0, 1.0, 1.0), "unknown_param": 123},
            }
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RadiomicsPipeline._validate_config("test", config)
            assert len(w) >= 1
            assert any(
                "unknown parameter" in str(warning.message).lower() for warning in w
            )

    def test_validate_missing_step_key(self) -> None:
        """Test validation warns for missing step key."""
        import warnings

        config = [{"params": {"new_spacing": (1.0, 1.0, 1.0)}}]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            is_valid = RadiomicsPipeline._validate_config("test", config)

            assert is_valid is False
            assert len(w) >= 1
            assert any(
                "missing 'step' key" in str(warning.message).lower() for warning in w
            )

    def test_validate_invalid_structure(self) -> None:
        """Test validation fails for non-list config."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected validation warning
            is_valid = RadiomicsPipeline._validate_config("test", "not a list")  # type: ignore
        assert is_valid is False

    def test_validate_non_dict_step(self) -> None:
        """Test validation warns for non-dict step."""
        import warnings

        config = ["not a dict", {"step": "resample", "params": {}}]  # type: ignore
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            is_valid = RadiomicsPipeline._validate_config("test", config)

            assert is_valid is False
            assert len(w) >= 1
            assert any(
                "must be a dictionary" in str(warning.message).lower() for warning in w
            )


# --- Schema Version Tests ---


class TestSchemaVersion:
    """Tests for schema version handling."""

    def test_schema_version_in_export(self, pipeline: RadiomicsPipeline) -> None:
        """Test that schema version is included in exports."""
        data = pipeline.to_dict()
        assert data["schema_version"] == CONFIG_SCHEMA_VERSION

    def test_schema_version_constant(self) -> None:
        """Test that CONFIG_SCHEMA_VERSION is defined."""
        assert CONFIG_SCHEMA_VERSION == "1.0"

    def test_schema_migration_logging(self) -> None:
        """Test that migration handles different versions gracefully."""
        import warnings

        # Call migration with a different version
        data = {"schema_version": "0.9", "configs": {}}
        # Verify migration doesn't raise (warning is expected but suppressed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected unknown version warning
            RadiomicsPipeline._migrate_config(data, "0.9")


# --- Edge Case Tests ---


class TestEdgeCases:
    """Tests for edge cases and error handling branches."""

    def test_from_dict_invalid_config_format(self) -> None:
        """Test that invalid config format is skipped with warning."""
        import warnings

        data = {
            "schema_version": "1.0",
            "configs": {
                "invalid_config": "not a dict or list",  # Invalid format
                "valid_config": {
                    "steps": [
                        {"step": "resample", "params": {"new_spacing": [1.0, 1.0, 1.0]}}
                    ]
                },
            },
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline = RadiomicsPipeline.from_dict(data)

            # Valid config should be loaded
            assert "valid_config" in pipeline.list_configs()
            # Invalid config should be skipped
            assert "invalid_config" not in pipeline.list_configs()
            assert len(w) >= 1
            assert any(
                "invalid config format" in str(warning.message).lower() for warning in w
            )

    def test_from_dict_direct_list_format(self) -> None:
        """Test that direct list format (without 'steps' key) works."""
        data = {
            "schema_version": "1.0",
            "configs": {
                "direct_list": [
                    {"step": "resample", "params": {"new_spacing": [1.0, 1.0, 1.0]}},
                    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
                ]
            },
        }
        pipeline = RadiomicsPipeline.from_dict(data)

        assert "direct_list" in pipeline.list_configs()
        config = pipeline.get_config("direct_list")
        assert len(config) == 2

    def test_make_serializable_numpy_array(self, pipeline: RadiomicsPipeline) -> None:
        """Test that numpy arrays are converted to lists."""
        import numpy as np

        # Add a config with numpy array
        pipeline.add_config(
            "numpy_test",
            [
                {
                    "step": "resample",
                    "params": {"new_spacing": np.array([1.0, 1.0, 1.0])},
                }
            ],
        )

        # Export should convert array to list
        data = pipeline.to_dict(config_names=["numpy_test"])
        steps = data["configs"]["numpy_test"]["steps"]
        assert isinstance(steps[0]["params"]["new_spacing"], list)

    def test_make_serializable_numpy_scalar(self, pipeline: RadiomicsPipeline) -> None:
        """Test that numpy scalars are converted to Python types."""
        import numpy as np

        # Add a config with numpy scalar
        pipeline.add_config(
            "scalar_test",
            [
                {
                    "step": "discretise",
                    "params": {"method": "FBN", "n_bins": np.int64(32)},
                }
            ],
        )

        # Export should convert scalar to Python int
        json_str = pipeline.to_json(config_names=["scalar_test"])
        # Should not raise - numpy types are serializable now
        import json

        data = json.loads(json_str)
        assert data["configs"]["scalar_test"]["steps"][0]["params"]["n_bins"] == 32


# --- Template Edge Case Tests ---


class TestTemplateEdgeCases:
    """Tests for template loading edge cases."""

    def test_get_all_templates_with_direct_list_format(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_all_templates handles direct list format."""
        from pictologics import templates

        # Create a mock template file with direct list format
        mock_yaml = """
schema_version: "1.0"
configs:
  direct_format:
    - step: resample
      params:
        new_spacing: [1.0, 1.0, 1.0]
"""
        template_file = tmp_path / "test_direct.yaml"
        template_file.write_text(mock_yaml)

        # Mock list_template_files to return our test file
        def mock_list_files() -> list[str]:
            return ["test_direct.yaml"]

        def mock_load_file(filename: str) -> dict:
            return yaml.safe_load(template_file.read_text())

        monkeypatch.setattr(templates, "list_template_files", mock_list_files)
        monkeypatch.setattr(templates, "load_template_file", mock_load_file)

        result = templates.get_all_templates()
        assert "direct_format" in result

    def test_get_all_templates_non_dict_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_all_templates warns for non-dict template file."""
        import warnings

        from pictologics import templates

        # Mock to return non-dict
        def mock_list_files() -> list[str]:
            return ["bad_file.yaml"]

        def mock_load_file(filename: str) -> list:  # type: ignore
            return ["not", "a", "dict"]

        monkeypatch.setattr(templates, "list_template_files", mock_list_files)
        monkeypatch.setattr(templates, "load_template_file", mock_load_file)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = templates.get_all_templates()
            assert result == {}
            assert len(w) >= 1
            assert any(
                "does not contain a dictionary" in str(warning.message) for warning in w
            )

    def test_get_all_templates_exception_handling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_all_templates handles exceptions gracefully."""
        import warnings

        from pictologics import templates

        def mock_list_files() -> list[str]:
            return ["error_file.yaml"]

        def mock_load_file(filename: str) -> dict:
            raise RuntimeError("Simulated error")

        monkeypatch.setattr(templates, "list_template_files", mock_list_files)
        monkeypatch.setattr(templates, "load_template_file", mock_load_file)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = templates.get_all_templates()
            assert result == {}
            assert len(w) >= 1
            assert any(
                "failed to load template file" in str(warning.message).lower()
                for warning in w
            )

    def test_get_standard_templates_file_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_standard_templates handles missing file."""
        import warnings

        from pictologics import templates

        def mock_load_file(filename: str) -> dict:
            raise FileNotFoundError("standard_configs.yaml not found")

        monkeypatch.setattr(templates, "load_template_file", mock_load_file)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = templates.get_standard_templates()
            assert result == {}
            assert len(w) >= 1
            assert any("not found" in str(warning.message).lower() for warning in w)

    def test_get_standard_templates_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_standard_templates handles generic exceptions."""
        import warnings

        from pictologics import templates

        def mock_load_file(filename: str) -> dict:
            raise RuntimeError("Simulated error")

        monkeypatch.setattr(templates, "load_template_file", mock_load_file)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = templates.get_standard_templates()
            assert result == {}
            assert len(w) >= 1
            assert any(
                "failed to load standard templates" in str(warning.message).lower()
                for warning in w
            )

    def test_get_standard_templates_direct_list_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_standard_templates handles direct list format."""
        from pictologics import templates

        def mock_load_file(filename: str) -> dict:
            return {
                "schema_version": "1.0",
                "configs": {
                    "direct_config": [
                        {"step": "resample", "params": {"new_spacing": [1.0, 1.0, 1.0]}}
                    ]
                },
            }

        monkeypatch.setattr(templates, "load_template_file", mock_load_file)

        result = templates.get_standard_templates()
        assert "direct_config" in result


# --- Pipeline Loading Error Tests ---


class TestPipelineLoadingErrors:
    """Tests for pipeline template loading error handling."""

    def test_load_predefined_configs_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _load_predefined_configs handles template loading failure."""
        import warnings

        from pictologics import pipeline as pipeline_module

        def mock_get_standard() -> dict[str, list[dict[str, Any]]]:
            raise RuntimeError("Simulated template loading error")

        # Patch at the pipeline module level where it's imported
        monkeypatch.setattr(
            pipeline_module, "get_standard_templates", mock_get_standard
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Create a new pipeline - should handle the error gracefully
            RadiomicsPipeline()

            # Pipeline should still be usable, just without standard configs
            assert len(w) >= 1
            assert any(
                "failed to load standard templates" in str(warning.message).lower()
                for warning in w
            )
