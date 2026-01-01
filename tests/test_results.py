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

    def test_wide_format_with_metadata(
        self, sample_results: dict[str, pd.Series]
    ) -> None:
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
        # Check structure of one item
        assert "config" in result[0]
        assert "feature_name" in result[0]
        assert "value" in result[0]

    def test_long_format_pandas(self, sample_results: dict[str, pd.Series]) -> None:
        """Test long format as pandas DataFrame."""
        result = format_results(sample_results, fmt="long", output_type="pandas")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert "config" in result.columns
        assert "feature_name" in result.columns
        assert "value" in result.columns

    def test_long_format_custom_config_col(
        self, sample_results: dict[str, pd.Series]
    ) -> None:
        """Test long format with a custom config column name."""
        result = format_results(
            sample_results, fmt="long", output_type="pandas", config_col="configuration"
        )
        assert isinstance(result, pd.DataFrame)
        assert "configuration" in result.columns
        assert "config" not in result.columns

    def test_long_format_with_metadata(
        self, sample_results: dict[str, pd.Series]
    ) -> None:
        """Test long format with metadata included in every row."""
        meta = {"subject_id": "sub-001"}
        result = format_results(
            sample_results, fmt="long", output_type="pandas", meta=meta
        )
        assert isinstance(result, pd.DataFrame)
        assert "subject_id" in result.columns
        assert (result["subject_id"] == "sub-001").all()

    def test_long_format_json(self, sample_results: dict[str, pd.Series]) -> None:
        """Test long format as JSON string."""
        result = format_results(sample_results, fmt="long", output_type="json")
        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 4
        assert "config" in data[0]

    def test_empty_results(self) -> None:
        """Test handling of empty input results."""
        # pandas
        result_df = format_results({}, fmt="long", output_type="pandas")
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty
        assert "feature_name" in result_df.columns

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
        assert (
            content == "" or content == '""'
        )  # Quote if pandas puts quotes? usually empty.

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

        """Test that unknown extensions default to CSV."""
        data = {"a": 1}
        path = tmp_path / "output.unknown"
        save_results(data, path)

        assert path.exists()
        # Read as CSV to verify
        df = pd.read_csv(path)
        assert len(df) == 1
        assert df.iloc[0]["a"] == 1

    def test_unsupported_file_format(self, tmp_path: Path) -> None:
        """Test passing an invalid file_format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            save_results({"a": 1}, tmp_path / "out.txt", file_format="xml")
