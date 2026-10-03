"""
Results Module
==============

This module provides utilities for formatting and saving radiomic feature
extraction results. It supports multiple output formats (wide, long) and
file formats (CSV, TSV, JSON).

Key Functions:
--------------
- **format_results**: Convert pipeline output to various formats (dict, pandas DataFrame, JSON).
- **save_results**: Save results to CSV, TSV or JSON files, by the file extension.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

# Wide-format column names of recent configurations, {config: {feature: column}}, so the
# rows of many images share one copy of each name. Only str names are kept (equal numbers
# can print differently), and the cache is emptied when it holds _WIDE_CACHE_CONFIGS
# configurations, so names that change for every image stay bounded.
_WIDE_COLUMNS: dict[str, dict[str, str]] = {}
_WIDE_CACHE_CONFIGS = 64


def format_results(
    results: dict[str, pd.Series],
    fmt: str = "wide",
    meta: dict[str, Any] | None = None,
    output_type: str = "dict",
    config_col: str = "config",
) -> dict[str, Any] | pd.DataFrame | str | list[dict[str, Any]]:
    """
    Format the output of RadiomicsPipeline.run() into a structured format.

    Merging is always **name-based** (by feature name / column key), never
    positional.  Because ``run()`` guarantees that every configuration returns
    the complete set of expected feature names, the formatted output always has
    a predictable, consistent set of columns — even when some features are
    ``NaN`` due to partial or complete extraction failure.

    Args:
        results: Dictionary mapping configuration names to pandas Series of features
                 (the standard output of RadiomicsPipeline.run).
        fmt: "wide" or "long".
             - "wide": Flattens keys to '{config}__{feature}'. Returns 1 row (dict/df).
             - "long": Tidy format with columns for config, feature_key, and value.
               feature_key is the full feature key with its IBSI code (for example
               mean_intensity_Q4LE), as in describe_features().
        meta: Optional dictionary of metadata to prepend to the result (e.g., subject ID).
        output_type: Format of the returned object: "dict", "pandas", or "json". JSON
            writes NaN and infinite values as null, so that every JSON reader accepts it.
        config_col: Name of the column holding the configuration name (only used if fmt="long").

    Returns:
        Formatted data in the specified output_type.

    Raises:
        ValueError: If `fmt` is not "wide" or "long", or if `output_type` is
            not "dict", "pandas", or "json".

    Example:
        Format results as a single pandas DataFrame row (wide format):

        ```python
        from pictologics.results import format_results

        # Assume 'results' is output from pipeline.run()
        df = format_results(
            results,
            fmt="wide",
            meta={"custom_id": 123},
            output_type="pandas"
        )
        ```
    """
    if meta is None:
        meta = {}

    if fmt == "wide":
        # Wide format: { "meta_key": val, "config__feature": val }
        formatted_data = meta.copy()
        for config_name, series in results.items():
            columns = _WIDE_COLUMNS.get(config_name) if type(config_name) is str else {}
            if columns is None:
                if len(_WIDE_COLUMNS) >= _WIDE_CACHE_CONFIGS:
                    _WIDE_COLUMNS.clear()
                columns = _WIDE_COLUMNS[config_name] = {}
            for feature_name, value in series.items():
                col_name = columns.get(feature_name)
                if col_name is None:
                    col_name = f"{config_name}__{feature_name}"
                    if type(feature_name) is str:
                        columns[feature_name] = col_name
                formatted_data[col_name] = value

        if output_type == "dict":
            return formatted_data
        elif output_type == "pandas":
            return pd.DataFrame([formatted_data])
        elif output_type == "json":
            return _json_text(formatted_data)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

    elif fmt == "long":
        if output_type not in ("dict", "pandas", "json"):
            raise ValueError(f"Unknown output_type: {output_type}")

        # Long format: Rows of [meta_cols..., config, feature_key, value], built as
        # columns (a meta key with a standard name holds the standard value)
        meta_keys = list(meta.keys())
        standard_cols = [config_col, "feature_key", "value"]
        cols_order = meta_keys + [c for c in standard_cols if c not in meta_keys]
        names: list[Any] = []
        keys: list[Any] = []
        values: list[Any] = []
        for config_name, series in results.items():
            names.extend([config_name] * len(series))
            keys.extend(series.index)
            values.extend(series.tolist())
        standard = {config_col: names, "feature_key": keys, "value": values}
        table = {
            col: standard[col] if col in standard else [meta[col]] * len(values)
            for col in cols_order
        }

        if not values:
            # Handle empty results case
            # For pure python output, empty list is fine.
            # For pandas, we need a dataframe with columns.
            if output_type == "dict":
                return []
            elif output_type == "json":
                return "[]"
            return pd.DataFrame(columns=cols_order)  # Columns are already in order

        if output_type == "pandas":
            return pd.DataFrame(table)
        rows = [
            dict(zip(cols_order, row, strict=True)) for row in zip(*table.values(), strict=True)
        ]
        return rows if output_type == "dict" else _json_text(rows)

    else:
        raise ValueError(f"Unknown format: {fmt}. Use 'wide' or 'long'.")


def save_results(
    data: (
        dict[str, Any] | list[dict[str, Any]] | pd.DataFrame | list[pd.DataFrame] | str | list[str]
    ),
    path: str | Path,
    file_format: str | None = None,
) -> None:
    """
    Save results to a file (CSV, JSON, etc.), automatically handling merging of lists.

    When saving a list of results (one per subject), rows are merged by
    **column name**.  Because ``RadiomicsPipeline.run()`` guarantees that every
    configuration returns a complete set of feature names, all rows share the
    same columns and the resulting file has no missing columns or ragged rows.

    Args:
        data: The data to save. Supported types:
              - Dict or List[Dict]
              - DataFrame or List[DataFrame]
              - JSON string or List[JSON strings]
        path: Output file path. A missing folder is created.
        file_format: "csv", "tsv" or "json". If None, inferred from the file extension
            (.csv, .tsv, .json; no extension means CSV). JSON writes NaN and infinite
            values as null.

    Raises:
        ValueError: If the file extension is another one (for example .parquet, which
            would otherwise get CSV text), if an explicit `file_format` is not
            "csv", "tsv" or "json", or if `data` cannot be normalized to a DataFrame
            (e.g. an invalid JSON string, mixed types within a list, or an
            unsupported data type).

    Example:
        Save formatted results to JSON:

        ```python
        from pictologics.results import save_results

        save_results(formatted_data, "output/features.json")
        ```
    """
    path = Path(path)
    if file_format is None:
        suffix = path.suffix.lower()
        file_format = {".csv": "csv", ".json": "json", ".tsv": "tsv", "": "csv"}.get(suffix)
        if file_format is None:
            raise ValueError(
                f"Unknown file type '{suffix}': save results as .csv, .tsv or .json, "
                "or pass file_format."
            )
    path.parent.mkdir(parents=True, exist_ok=True)

    # If format is JSON and data is already a dict or list of dicts, bypass pandas
    # to avoid overhead and potential C-extension conflicts in coverage/threading.
    if file_format == "json":
        # Check if data is already in a compatible format
        is_dict = isinstance(data, dict)
        is_list_of_dicts = isinstance(data, list) and (not data or isinstance(data[0], dict))

        if is_dict or is_list_of_dicts:
            with open(path, "w") as f:
                f.write(_json_text([data] if is_dict else data, indent=2))
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
    elif file_format == "tsv":
        final_df.to_csv(path, index=False, sep="\t")
    elif file_format == "json":
        # Use standard json library to avoid potential pandas C-extension issues during coverage
        with open(path, "w") as f:
            f.write(_json_text(final_df.to_dict(orient="records"), indent=2))
    else:
        raise ValueError(f"Unsupported export format: {file_format}")


def _json_safe(value: Any) -> Any:
    """`value` with NaN and infinite floats as None (JSON null), also inside lists and dicts."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _json_text(data: Any, indent: int | None = None) -> str:
    """Strict JSON (no NaN token, which most JSON readers reject): non-finite values
    become null. Data without them takes one pass (the common case)."""
    try:
        return json.dumps(data, indent=indent, allow_nan=False)
    except ValueError:  # a NaN or infinite value
        return json.dumps(_json_safe(data), indent=indent, allow_nan=False)


def _normalize_to_dataframe(
    data: (
        dict[str, Any] | list[dict[str, Any]] | pd.DataFrame | list[pd.DataFrame] | str | list[str]
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
                raise ValueError(f"Mixed types in list (expected string, got {type(item)}).")
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
