from __future__ import annotations

# ruff: noqa: E402
import warnings

# Suppress "NumPy module was reloaded" warning
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

import json
from pathlib import Path

import pandas as pd
import pytest

from pictologics.results import format_results, save_results


@pytest.fixture
def sample_results() -> dict[str, pd.Series]:
    """Provides a standard sample results dictionary."""
    return {
        "config_a": pd.Series({"feature1": 1.0, "feature2": 2.0}),
        "config_b": pd.Series({"feature1": 3.0, "feature2": 4.0}),
    }


class TestFormatResults:
    """Tests for the format_results function."""

    def test_wide_format_defaults(self, sample_results: dict[str, pd.Series]) -> None:
        """Test default wide format (output_type='dict')."""
        result = format_results(sample_results)
        assert isinstance(result, dict)
        assert result["config_a__feature1"] == 1.0
        assert result["config_b__feature2"] == 4.0
        assert "config_a__feature2" in result
        assert "config_b__feature1" in result

    def test_wide_format_pandas(self, sample_results: dict[str, pd.Series]) -> None:
        """Test wide format as pandas DataFrame."""
        result = format_results(sample_results, fmt="wide", output_type="pandas")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["config_a__feature1"] == 1.0

    def test_wide_format_json(self, sample_results: dict[str, pd.Series]) -> None:
        """Test wide format as JSON string."""
        result = format_results(sample_results, fmt="wide", output_type="json")
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["config_a__feature1"] == 1.0

    def test_wide_format_with_metadata(self, sample_results: dict[str, pd.Series]) -> None:
        """Test wide format with metadata."""
        meta = {"subject_id": "sub-001", "age": 25}
        result = format_results(sample_results, fmt="wide", meta=meta)
        assert isinstance(result, dict)
        assert result["subject_id"] == "sub-001"
        assert result["config_a__feature1"] == 1.0

    def test_long_format_defaults(self, sample_results: dict[str, pd.Series]) -> None:
        """Test default long format (output_type='dict')."""
        result = format_results(sample_results, fmt="long")
        assert isinstance(result, list)
        assert len(result) == 4  # 2 configs * 2 features
        # Check structure of one item: the full feature key is named feature_key
        assert list(result[0]) == ["config", "feature_key", "value"]
        assert result[0] == {"config": "config_a", "feature_key": "feature1", "value": 1.0}
        assert all("feature_name" not in row for row in result)

    def test_long_format_pandas(self, sample_results: dict[str, pd.Series]) -> None:
        """Test long format as pandas DataFrame."""
        result = format_results(sample_results, fmt="long", output_type="pandas")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert list(result.columns) == ["config", "feature_key", "value"]
        assert list(result["feature_key"]) == ["feature1", "feature2", "feature1", "feature2"]

    def test_long_format_custom_config_col(self, sample_results: dict[str, pd.Series]) -> None:
        """Test long format with a custom config column name."""
        result = format_results(
            sample_results, fmt="long", output_type="pandas", config_col="configuration"
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["configuration", "feature_key", "value"]

    def test_long_format_with_metadata(self, sample_results: dict[str, pd.Series]) -> None:
        """Test long format with metadata included in every row."""
        meta = {"subject_id": "sub-001"}
        result = format_results(sample_results, fmt="long", output_type="pandas", meta=meta)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["subject_id", "config", "feature_key", "value"]
        assert (result["subject_id"] == "sub-001").all()

    def test_long_format_json(self, sample_results: dict[str, pd.Series]) -> None:
        """Test long format as JSON string."""
        result = format_results(sample_results, fmt="long", output_type="json")
        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 4
        assert data[0] == {"config": "config_a", "feature_key": "feature1", "value": 1.0}

    def test_empty_results(self) -> None:
        """Test handling of empty input results."""
        # pandas
        result_df = format_results({}, fmt="long", output_type="pandas")
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty
        assert list(result_df.columns) == ["config", "feature_key", "value"]

        # dict
        result_dict = format_results({}, fmt="long", output_type="dict")
        assert isinstance(result_dict, list)
        assert len(result_dict) == 0

        # json
        result_json = format_results({}, fmt="long", output_type="json")
        assert isinstance(result_json, str)
        assert result_json == "[]"

    def test_invalid_format(self, sample_results: dict[str, pd.Series]) -> None:
        with pytest.raises(ValueError, match="Unknown format"):
            format_results(sample_results, fmt="invalid")

    def test_invalid_output_type(self, sample_results: dict[str, pd.Series]) -> None:
        with pytest.raises(ValueError, match="Unknown output_type"):
            format_results(sample_results, output_type="invalid")
        for results in (sample_results, {}):
            with pytest.raises(ValueError, match="Unknown output_type"):
                format_results(results, fmt="long", output_type="invalid")

    def test_wide_rows_share_their_keys(self, sample_results: dict[str, pd.Series]) -> None:
        """Rows of many images keep one copy of each column name."""
        first = format_results(sample_results, meta={"subject_id": 1})
        second = format_results(sample_results, meta={"subject_id": 2})
        assert list(first) == list(second)
        for a, b in zip(first, second, strict=True):
            assert a is b

    def test_wide_column_cache_stays_small_and_exact(self) -> None:
        """The name cache holds at most _WIDE_CACHE_CONFIGS configurations and only str
        names: equal numbers keep their own text, and a number as configuration works."""
        from pictologics import results as results_module

        series = pd.Series([1.0, 2.0], index=["a", "b"])
        for k in range(results_module._WIDE_CACHE_CONFIGS + 5):
            assert list(format_results({f"cfg_{k}": series})) == [f"cfg_{k}__a", f"cfg_{k}__b"]
        assert len(results_module._WIDE_COLUMNS) <= results_module._WIDE_CACHE_CONFIGS
        one = pd.Series([1.0], index=pd.Index([1], dtype=object))
        one_float = pd.Series([2.0], index=pd.Index([1.0], dtype=object))
        assert list(format_results({"c": one})) == ["c__1"]
        assert list(format_results({"c": one_float})) == ["c__1.0"]
        assert format_results({7: series}) == {"7__a": 1.0, "7__b": 2.0}
        assert 7 not in results_module._WIDE_COLUMNS

    def test_long_frames_of_many_cases_save_as_one_table(
        self, sample_results: dict[str, pd.Series], tmp_path: Path
    ) -> None:
        """The cookbook's long-format batch: one DataFrame per case, saved together."""
        rows = [
            format_results(sample_results, fmt="long", meta={"subject_id": i}, output_type="pandas")
            for i in range(3)
        ]
        path = tmp_path / "long.csv"
        save_results(rows, path)
        table = pd.read_csv(path)
        assert list(table.columns) == ["subject_id", "config", "feature_key", "value"]
        assert len(table) == 12


class TestSaveResults:
    """Tests for the save_results function."""

    def test_save_dict_to_csv(self, tmp_path: Path) -> None:
        """Test saving a simple dictionary to CSV."""
        data = {"a": 1, "b": 2}
        path = tmp_path / "output.csv"
        save_results(data, path)

        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 1
        assert df.iloc[0]["a"] == 1

    def test_save_dict_to_json(self, tmp_path: Path) -> None:
        """Test saving a simple dictionary to JSON."""
        data = {"a": 1, "b": 2}
        path = tmp_path / "output.json"
        save_results(data, path)

        assert path.exists()
        with open(path) as f:
            content = json.load(f)
        # save_results writes records orientation -> list of dicts
        assert isinstance(content, list)
        assert content[0]["a"] == 1

    def test_save_dataframe_to_csv(self, tmp_path: Path) -> None:
        """Test saving a DataFrame to CSV."""
        df = pd.DataFrame([{"a": 1, "b": 2}])
        path = tmp_path / "output.csv"
        save_results(df, path)

        assert path.exists()
        saved_df = pd.read_csv(path)
        pd.testing.assert_frame_equal(df, saved_df)

    def test_save_list_of_dicts(self, tmp_path: Path) -> None:
        """Test saving a list of dictionaries (should merge into one DF)."""
        data = [{"a": 1}, {"a": 2}]
        path = tmp_path / "output.csv"
        save_results(data, path)

        saved_df = pd.read_csv(path)
        assert len(saved_df) == 2
        assert saved_df.iloc[1]["a"] == 2

    def test_save_list_of_dataframes(self, tmp_path: Path) -> None:
        """Test saving a list of DataFrames (should concat)."""
        df1 = pd.DataFrame([{"a": 1}])
        df2 = pd.DataFrame([{"a": 2}])
        path = tmp_path / "output.csv"
        save_results([df1, df2], path)

        saved_df = pd.read_csv(path)
        assert len(saved_df) == 2
        assert saved_df.iloc[1]["a"] == 2

    def test_save_json_string(self, tmp_path: Path) -> None:
        """Test saving a single JSON string."""
        data_str = '{"a": 1, "b": 2}'
        path = tmp_path / "output.csv"
        save_results(data_str, path)

        saved_df = pd.read_csv(path)
        assert len(saved_df) == 1
        assert saved_df.iloc[0]["a"] == 1

    def test_save_list_of_json_strings(self, tmp_path: Path) -> None:
        """Test saving a list of JSON strings."""
        data_list = ['{"a": 1}', '{"a": 2}']
        path = tmp_path / "output.csv"
        save_results(data_list, path)

        saved_df = pd.read_csv(path)
        assert len(saved_df) == 2
        assert saved_df.iloc[1]["a"] == 2

    def test_explicit_format(self, tmp_path: Path) -> None:
        """Test enforcing a format regardless of extension."""
        data = {"a": 1}
        path = tmp_path / "output.txt"  # weird extension
        save_results(data, path, file_format="json")

        assert path.exists()
        # Verify it is actually JSON
        with open(path) as f:
            content = json.load(f)
        assert content[0]["a"] == 1

    def test_invalid_json_string(self, tmp_path: Path) -> None:
        """Test error handling for bad JSON strings."""
        with pytest.raises(ValueError, match="not valid JSON"):
            save_results("{bad json}", tmp_path / "out.csv")

    def test_invalid_json_in_list(self, tmp_path: Path) -> None:
        """Test error handling for bad JSON strings in a list."""
        with pytest.raises(ValueError, match="not valid JSON"):
            save_results(['{"a": 1}', "{bad}"], tmp_path / "out.csv")

    def test_mixed_types_in_list(self, tmp_path: Path) -> None:
        """Test that list must be uniform (all strings or all dicts/dfs treated as objects)."""
        # The implementation iterates and checks if item is str.
        # If the first item is str, it assumes list of strings.
        # If it encounters a non-string in that loop, it raises.
        with pytest.raises(ValueError, match="Mixed types"):
            save_results(['{"a": 1}', {"b": 2}], tmp_path / "out.csv")

    def test_mixed_dicts_and_frames_raise_value_error(self, tmp_path: Path) -> None:
        # A list of dicts or of DataFrames with an item of another type raises ValueError
        # for every file type (before, JSON raised TypeError and CSV AttributeError)
        frame = pd.DataFrame({"a": [1]})
        for data, kind in (([{"a": 1}, "x"], "str"), ([frame, {"a": 1}], "dict")):
            for name in ("out.json", "out.csv"):
                with pytest.raises(ValueError, match=f"Mixed types in list .*got {kind}"):
                    save_results(data, tmp_path / name)  # type: ignore[arg-type]

    def test_unsupported_list_element(self, tmp_path: Path) -> None:
        """Test a list containing an unsupported type (e.g., list of lists)."""
        # Based on refactored code, list of lists is not supported.
        with pytest.raises(ValueError, match="List contains unsupported type"):
            save_results([[1, 2]], tmp_path / "out.csv")  # type: ignore

    def test_unsupported_root_type(self, tmp_path: Path) -> None:
        """Test passing an unsupported top-level type (e.g., int)."""
        with pytest.raises(ValueError, match="Could not normalize"):
            save_results(123, tmp_path / "out.csv")  # type: ignore

    def test_save_empty_list(self, tmp_path: Path) -> None:
        """Test saving an empty list."""
        path = tmp_path / "output.csv"
        # Should create an empty file (or header only depending on pandas).
        # Our implementation normalizes to an empty DataFrame.
        save_results([], path)
        assert path.exists()
        # Since our implementation might write an empty list as '[]' (json) or empty file (csv),
        # or empty dataframe logic.
        # For CSV, an empty list normalized to empty DataFrame -> to_csv gives empty file (or header).
        # Let's check that the file is indeed empty or effectively empty.

        # If it raised EmptyDataError before, it means the file was likely empty.
        # We can just check file size.
        # If it raised EmptyDataError before, it means the file was likely empty.
        # We can just check file content string to avoid pandas exception quirks in this env.
        # A truly empty dataframe saved to CSV might be an empty file or just newline.
        content = path.read_text().strip()
        assert content == "" or content == '""'  # Quote if pandas puts quotes? usually empty.

    def test_save_list_dicts_to_json(self, tmp_path: Path) -> None:
        """Test saving a list of dictionaries to JSON (hits optimization path)."""
        data = [{"a": 1}, {"a": 2}]
        path = tmp_path / "output.json"
        save_results(data, path)

        assert path.exists()
        with open(path) as f:
            content = json.load(f)
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[1]["a"] == 2

    def test_save_dataframe_to_json(self, tmp_path: Path) -> None:
        """Test saving a DataFrame to JSON (hits non-bypass JSON export)."""
        df = pd.DataFrame([{"a": 1, "b": 2}])
        path = tmp_path / "output.json"
        save_results(df, path)

        assert path.exists()
        with open(path) as f:
            content = json.load(f)
        assert isinstance(content, list)
        assert content[0]["a"] == 1

        """An unknown extension raises (it silently got CSV text before); file_format
        still chooses CSV."""
        data = {"a": 1}
        path = tmp_path / "output.unknown"
        with pytest.raises(ValueError, match="Unknown file type '.unknown'"):
            save_results(data, path)
        save_results(data, path, file_format="csv")

        assert path.exists()
        # Read as CSV to verify
        df = pd.read_csv(path)
        assert len(df) == 1
        assert df.iloc[0]["a"] == 1

    def test_unsupported_file_format(self, tmp_path: Path) -> None:
        """Test passing an invalid file_format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            save_results({"a": 1}, tmp_path / "out.txt", file_format="xml")


def _strict_json(text: str) -> object:
    """Parse JSON like a strict reader: the NaN and Infinity tokens are errors."""

    def reject(token: str) -> None:
        raise ValueError(f"not JSON: {token}")

    return json.loads(text, parse_constant=reject)


def test_json_output_writes_nan_as_null(tmp_path: Path) -> None:
    # NaN and infinite values become null in every JSON output, so strict readers
    # (JavaScript, jq) accept it; CSV keeps them as before.
    with pytest.raises(ValueError, match="not JSON: NaN"):
        _strict_json("[NaN]")  # the old output
    results = {"c": pd.Series({"a": 1.0, "b": float("nan"), "c": float("inf")})}
    wide = _strict_json(format_results(results, output_type="json"))
    assert wide == {"c__a": 1.0, "c__b": None, "c__c": None}
    long = _strict_json(format_results(results, fmt="long", output_type="json"))
    assert [row["value"] for row in long] == [1.0, None, None]
    row = format_results(results, output_type="dict")
    for data in (row, [row], pd.DataFrame([row])):
        path = tmp_path / "out.json"
        save_results(data, path)
        assert _strict_json(path.read_text()) == [{"c__a": 1.0, "c__b": None, "c__c": None}]


def test_json_of_a_frame_with_nan_encodes_once(tmp_path: Path) -> None:
    # A frame with a NaN or infinite float encodes once, with the bytes of the encode
    # after a failed first try (0.6.0); a NaN in an object column (two encodes, as
    # before), a missing value in a nullable float column and a frame without floats
    # give the bytes of 0.6.0 too.
    from unittest.mock import patch

    import numpy as np

    from pictologics import results as results_module

    frames = {
        "finite": pd.DataFrame({"id": ["x", "y"], "a": [1.0, 2.5], "n": [1, 2]}),
        "nan last": pd.DataFrame({"id": ["x", "y"], "a": [1.0, np.nan], "b": [np.inf, 3.0]}),
        "object nan": pd.DataFrame({"id": ["x", "y"], "o": pd.Series([1.0, np.nan], dtype=object)}),
        "nullable": pd.DataFrame({"a": pd.array([1.5, None], dtype="Float64"), "b": [2.0, 3.0]}),
        "no floats": pd.DataFrame({"id": ["x"], "n": [3]}),
    }
    for name, frame in frames.items():
        path = tmp_path / f"{name}.json"
        with patch.object(results_module.json, "dumps", wraps=json.dumps) as dumps:
            save_results(frame, path)
        expected = results_module._json_text(frame.to_dict(orient="records"), indent=2)
        assert path.read_text() == expected
        assert dumps.call_count == (2 if name == "object nan" else 1)


def test_format_results_takes_dictionaries_of_features() -> None:
    # Dictionaries of features give the output of Series, in the wide and the long format
    # and every output type.
    series = {
        "c": pd.Series({"a": 1.0, "b": float("nan"), "c": float("inf")}),
        "d": pd.Series({"a": 2.0}),
    }
    dicts = {name: values.to_dict() for name, values in series.items()}
    for fmt in ("wide", "long"):
        for output_type in ("dict", "json"):
            expected = format_results(series, fmt=fmt, output_type=output_type, meta={"id": 7})
            got = format_results(dicts, fmt=fmt, output_type=output_type, meta={"id": 7})
            assert repr(got) == repr(expected)
        pd.testing.assert_frame_equal(
            format_results(dicts, fmt=fmt, output_type="pandas"),
            format_results(series, fmt=fmt, output_type="pandas"),
        )


def test_save_results_file_types_and_folders(tmp_path: Path) -> None:
    # Known extensions only (a .parquet file got CSV text before); .tsv separates by
    # tabs; no extension means CSV; a missing folder is created.
    row = {"id": "x", "c__a": 1.0}
    with pytest.raises(ValueError, match="Unknown file type '.parquet'"):
        save_results(row, tmp_path / "out.parquet")
    save_results(row, tmp_path / "new" / "deep" / "out.tsv")
    assert (tmp_path / "new" / "deep" / "out.tsv").read_text().splitlines() == [
        "id\tc__a",
        "x\t1.0",
    ]
    save_results(row, tmp_path / "plain")
    assert (tmp_path / "plain").read_text().splitlines() == ["id,c__a", "x,1.0"]
    save_results(row, tmp_path / "forced.txt", file_format="tsv")
    assert (tmp_path / "forced.txt").read_text().startswith("id\tc__a")


def test_long_format_columns_match_rows_of_the_old_build() -> None:
    # The long table, built as columns, holds the rows of the old row-by-row build:
    # meta columns first, then config, feature_key and value; a meta key with a
    # standard name holds the standard value at its meta position.
    results = {
        "a": pd.Series({"f1": 1.0, "f2": float("nan")}),
        "b": pd.Series({"f1": 3.0}),
    }
    meta = {"subject_id": "s1", "value": "meta", "site": 7}
    expected = []
    for config_name, series in results.items():
        for key, value in series.items():
            row = dict(meta)
            row.update({"config": config_name, "feature_key": key, "value": value})
            expected.append(
                {k: row[k] for k in ["subject_id", "value", "site", "config", "feature_key"]}
            )
    rows = format_results(results, fmt="long", meta=meta, output_type="dict")
    assert [list(r) for r in rows] == [list(e) for e in expected]
    assert str(rows) == str(expected)  # NaN compares unequal, its text does not
    frame = format_results(results, fmt="long", meta=meta, output_type="pandas")
    pd.testing.assert_frame_equal(frame, pd.DataFrame(expected))
