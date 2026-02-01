"""
Template Loading Module
=======================

Provides utilities for loading pipeline configuration templates from YAML files.
Templates are bundled with the package and provide standardized, reproducible
configurations for radiomics research.
"""

from __future__ import annotations

import warnings
from importlib import resources
from typing import Any

import yaml


def _get_templates_path() -> resources.abc.Traversable:
    """Get the path to the templates directory using importlib.resources."""
    return resources.files("pictologics.templates")


def list_template_files() -> list[str]:
    """
    List all available template YAML files.

    Returns:
        List of template file names (without path).
    """
    templates_dir = _get_templates_path()
    return [
        f.name
        for f in templates_dir.iterdir()
        if f.is_file() and f.name.endswith(".yaml")
    ]


def load_template_file(filename: str) -> dict[str, Any]:
    """
    Load a single template file and return its parsed contents.

    Args:
        filename: Name of the template file (e.g., "standard_configs.yaml").

    Returns:
        Parsed YAML contents as a dictionary.

    Raises:
        FileNotFoundError: If the template file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    templates_dir = _get_templates_path()
    template_file = templates_dir.joinpath(filename)

    if not template_file.is_file():
        raise FileNotFoundError(f"Template file not found: {filename}")

    content = template_file.read_text(encoding="utf-8")
    result: dict[str, Any] = yaml.safe_load(content)
    return result


def get_all_templates() -> dict[str, list[dict[str, Any]]]:
    """
    Load all templates from all YAML files in the templates directory.

    Returns:
        Dictionary mapping config names to their step lists.
    """
    all_configs: dict[str, list[dict[str, Any]]] = {}

    for filename in list_template_files():
        try:
            file_data = load_template_file(filename)
            if not isinstance(file_data, dict):
                warnings.warn(
                    f"Template file {filename} does not contain a dictionary",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            configs = file_data.get("configs", {})
            if isinstance(configs, dict):
                for name, config_data in configs.items():
                    if isinstance(config_data, dict) and "steps" in config_data:
                        all_configs[name] = config_data["steps"]
                    elif isinstance(config_data, list):
                        # Direct list of steps
                        all_configs[name] = config_data

        except Exception as e:
            warnings.warn(
                f"Failed to load template file {filename}: {e}",
                UserWarning,
                stacklevel=2,
            )

    return all_configs


def get_standard_templates() -> dict[str, list[dict[str, Any]]]:
    """
    Load only the standard configuration templates.

    Returns:
        Dictionary mapping standard config names to their step lists.
    """
    try:
        file_data = load_template_file("standard_configs.yaml")
        configs = file_data.get("configs", {})
        result: dict[str, list[dict[str, Any]]] = {}

        for name, config_data in configs.items():
            if isinstance(config_data, dict) and "steps" in config_data:
                result[name] = config_data["steps"]
            elif isinstance(config_data, list):
                result[name] = config_data

        return result

    except FileNotFoundError:
        warnings.warn(
            "standard_configs.yaml not found, returning empty dict",
            UserWarning,
            stacklevel=2,
        )
        return {}
    except Exception as e:
        warnings.warn(
            f"Failed to load standard templates: {e}",
            UserWarning,
            stacklevel=2,
        )
        return {}


def get_template_metadata(filename: str) -> dict[str, Any]:
    """
    Get metadata from a template file (schema_version, description, etc.).

    Args:
        filename: Name of the template file.

    Returns:
        Dictionary containing metadata fields.
    """
    file_data = load_template_file(filename)
    return {
        "schema_version": file_data.get("schema_version", "unknown"),
        "description": file_data.get("description", ""),
        "config_names": list(file_data.get("configs", {}).keys()),
    }
