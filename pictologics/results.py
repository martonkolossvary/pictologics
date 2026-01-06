"""
Results Module
==============

This module provides utilities for formatting and saving radiomic feature
extraction results. It supports multiple output formats (wide, long) and
file formats (CSV, JSON).

Key Functions:
--------------
- **format_results**: Convert pipeline output to various formats (dict, pandas DataFrame, JSON).
- **save_results**: Save results to CSV or JSON files with automatic format detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def format_results(
    results: dict[str, pd.Series],
    fmt: str = "wide",
    meta: dict[str, Any] | None = None,
    output_type: str = "dict",
    config_col: str = "config",
) -> dict[str, Any] | pd.DataFrame | str | list[dict[str, Any]]:
    """
    Format the output of RadiomicsPipeline.run() into a structured format.

    Args:
        results: Dictionary mapping configuration names to pandas Series of features
                 (the standard output of RadiomicsPipeline.run).
        fmt: "wide" or "long".
             - "wide": Flattens keys to '{config}__{feature}'. Returns 1 row (dict/df).
             - "long": Tidy format with columns for config, feature_name, and value.
        meta: Optional dictionary of metadata to prepend to the result (e.g., subject ID).
        output_type: Format of the returned object: "dict", "pandas", or "json".
        config_col: Name of the column holding the configuration name (only used if fmt="long").

    Returns:
        Formatted data in the specified output_type.
    """
    if meta is None:
        meta = {}

    if fmt == "wide":
        # Wide format: { "meta_key": val, "config__feature": val }
        formatted_data = meta.copy()
        for config_name, series in results.items():
            for feature_name, value in series.items():
                col_name = f"{config_name}__{feature_name}"
                formatted_data[col_name] = value

        if output_type == "dict":
            return formatted_data
        elif output_type == "pandas":
            return pd.DataFrame([formatted_data])
        elif output_type == "json":
            return json.dumps(formatted_data)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

    elif fmt == "long":
        # Long format: Rows of [meta_cols..., config, feature_name, value]
        rows = []
        for config_name, series in results.items():
            for feature_name, value in series.items():
                row = meta.copy()
                row[config_col] = config_name
                row["feature_name"] = feature_name
                row["value"] = value
                rows.append(row)

        if not rows:
            # Handle empty results case
            # For pure python output, empty list is fine.
            # For pandas, we need a dataframe with columns.
            if output_type != "pandas":
                if output_type == "dict":
                    return []
                elif output_type == "json":
                    return "[]"

            df = pd.DataFrame(
                columns=list(meta.keys()) + [config_col, "feature_name", "value"]
            )
            return df  # Columns are already in order

        # Reorder keys/columns
        # Determine strict order
        meta_keys = list(meta.keys())
        standard_cols = [config_col, "feature_name", "value"]
        # Ensure we don't duplicate keys
        cols_order = meta_keys + [c for c in standard_cols if c not in meta_keys]

        # If output is dict/json, reorder the dictionaries directly
        if output_type in ("dict", "json"):
            # Ensure each row has keys in the desired order
            ordered_rows = []
            for r in rows:
                new_r = {k: r.get(k) for k in cols_order if k in r}

                ordered_rows.append(new_r)

            if output_type == "dict":
                return ordered_rows
            else:  # json
                return json.dumps(ordered_rows)

        # Output type is 'pandas'
        df = pd.DataFrame(rows)
        # Verify which columns actually exist in the dataframe
        existing_cols = list(df.columns)
        final_order = [c for c in cols_order if c in existing_cols] + [
            c for c in existing_cols if c not in cols_order
        ]
        df = df.reindex(columns=final_order)

        return df

    else:
        raise ValueError(f"Unknown format: {fmt}. Use 'wide' or 'long'.")


def save_results(
    data: (
        dict[str, Any]
        | list[dict[str, Any]]
        | pd.DataFrame
        | list[pd.DataFrame]
        | str
        | list[str]
    ),
    path: str | Path,
    file_format: str | None = None,
) -> None:
    """
    Save results to a file (CSV, JSON, etc.), automatically handling merging of lists.

    Args:
        data: The data to save. Supported types:
              - Dict or List[Dict]
              - DataFrame or List[DataFrame]
              - JSON string or List[JSON strings]
        path: Output file path.
        file_format: "csv" or "json". If None, inferred from file extension.
    """
    path = Path(path)
    if file_format is None:
        if path.suffix.lower() == ".csv":
            file_format = "csv"
        elif path.suffix.lower() == ".json":
            file_format = "json"
        else:
            file_format = "csv"

    # If format is JSON and data is already a dict or list of dicts, bypass pandas
    # to avoid overhead and potential C-extension conflicts in coverage/threading.
    if file_format == "json":
        # Check if data is already in a compatible format
        is_dict = isinstance(data, dict)
        is_list_of_dicts = isinstance(data, list) and (
            not data or isinstance(data[0], dict)
        )

        if is_dict:
            with open(path, "w") as f:
                json.dump([data], f, indent=2)
            return

        if is_list_of_dicts:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return

    # Normalize input to a single DataFrame
    try:
        final_df = _normalize_to_dataframe(data)
    except Exception as e:
        # If normalization fails but we want to debug, re-raise
        raise e

    # Export
    if file_format == "csv":
        final_df.to_csv(path, index=False)
    elif file_format == "json":
        # Use standard json library to avoid potential pandas C-extension issues during coverage
        with open(path, "w") as f:
            json.dump(final_df.to_dict(orient="records"), f, indent=2)
    else:
        raise ValueError(f"Unsupported export format: {file_format}")


def _normalize_to_dataframe(
    data: (
        dict[str, Any]
        | list[dict[str, Any]]
        | pd.DataFrame
        | list[pd.DataFrame]
        | str
        | list[str]
    ),
) -> pd.DataFrame:
    """
    Helper to convert various input types into a single pandas DataFrame.

    Handles:
    - JSON strings (parsed into dicts)
    - Dictionaries (single records)
    - DataFrames
    - Lists of any of the above (merged)
    """
    # 1. Handle JSON strings -> parse them first
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Provided data string is not valid JSON: {e}") from e
    elif isinstance(data, list) and data and isinstance(data[0], str):
        # List of JSON strings
        parsed_list = []
        for i, item in enumerate(data):
            if isinstance(item, str):
                try:
                    parsed_list.append(json.loads(item))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Item at index {i} is not valid JSON: {e}") from e
            else:
                # Should not happen if list type check passed, but for safety in mixed lists cleanup
                raise ValueError(
                    f"Mixed types in list (expected string, got {type(item)})."
                )
        data = parsed_list

    # 2. Now data is Dict, List[Dict], DataFrame, or List[DataFrame]

    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, dict):
        return pd.DataFrame([data])

    if isinstance(data, list):
        if not data:
            return pd.DataFrame()

        first = data[0]
        if isinstance(first, pd.DataFrame):
            return pd.concat(data, ignore_index=True)

        if isinstance(first, dict):
            return pd.DataFrame(data)

        # Fallback for unexpected list contents
        raise ValueError(
            f"List contains unsupported type: {type(first)}. Expected dict or DataFrame."
        )

    # Fallback
    raise ValueError(
        f"Could not normalize data of type {type(data)} to DataFrame. "
        "Expected Dict, DataFrame, JSON str, or lists thereof."
    )
