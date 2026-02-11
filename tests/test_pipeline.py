from __future__ import annotations

# ruff: noqa: E402
import warnings

# Suppress "NumPy module was reloaded" warning
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

import os

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

from typing import Any
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

from pictologics.loader import Image
from pictologics.pipeline import EmptyROIMaskError, RadiomicsPipeline

# --- Fixtures ---


@pytest.fixture
def mock_image() -> Image:
    """A simple 10x10x10 dummy image."""
    return Image(
        array=np.zeros((10, 10, 10)),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="CT",
    )


@pytest.fixture
def mock_mask() -> Image:
    """A simple 10x10x10 dummy mask (all ones)."""
    return Image(
        array=np.ones((10, 10, 10), dtype=np.uint8),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="mask",
    )


@pytest.fixture
def pipeline() -> RadiomicsPipeline:
    """A fresh RadiomicsPipeline instance."""
    return RadiomicsPipeline()


# --- Init & Config Tests ---


def test_pipeline_init(pipeline: RadiomicsPipeline) -> None:
    # Predefined configs are loaded
    assert len(pipeline._configs) > 0
    assert "standard_fbn_32" in pipeline._configs
    assert "standard_fbs_16" in pipeline._configs
    assert pipeline._log == []
    assert pipeline.get_all_standard_config_names()


def test_add_config(pipeline: RadiomicsPipeline) -> None:
    config = [{"step": "resample", "params": {"new_spacing": (2.0, 1.0, 1.0)}}]
    pipeline.add_config("custom", config)
    assert "custom" in pipeline._configs
    assert pipeline._configs["custom"] == config


def test_add_config_errors(pipeline: RadiomicsPipeline) -> None:
    with pytest.raises(ValueError, match="must be a list"):
        pipeline.add_config("bad", "notalist")  # type: ignore

    with pytest.raises(ValueError, match="must be a dictionary"):
        pipeline.add_config("bad", ["notadict"])  # type: ignore

    with pytest.raises(ValueError, match="must have a 'step' key"):
        pipeline.add_config("bad", [{"params": {}}])


# --- Run & Loading Tests ---


@patch("pictologics.pipeline.load_image")
def test_run_loading_variations(
    mock_load: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    # 1. Image path, Mask path
    mock_load.side_effect = [mock_image, mock_mask]
    pipeline.add_config("t1", [])
    pipeline.run("img.nii", "mask.nii", config_names=["t1"])
    assert mock_load.call_count == 2

    # 2. Image obj, Mask obj
    mock_load.reset_mock()
    pipeline.run(mock_image, mock_mask, config_names=["t1"])
    assert mock_load.call_count == 0  # No load calls

    # 3. Mask None -> GeneratedFullMask
    mock_load.reset_mock()
    pipeline.run(mock_image, mask=None, config_names=["t1"])
    # We can check the log to confirm mask source
    assert pipeline._log[-1]["mask_source"] == "GeneratedFullMask"

    # 4. Mask Empty String -> GeneratedFullMask
    mock_load.reset_mock()
    pipeline.run(mock_image, mask="", config_names=["t1"])
    assert pipeline._log[-1]["mask_source"] == "GeneratedFullMask"


def test_run_config_selection(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    pipeline.add_config("c1", [])
    pipeline.add_config("c2", [])

    # Run specific
    res = pipeline.run(mock_image, mock_mask, config_names=["c1"])
    assert "c1" in res and "c2" not in res

    # Run all standard
    # Mocking _configs to clear standard ones for cleaner test? No, just use "all_standard" keyword
    with patch.object(pipeline, "get_all_standard_config_names", return_value=["c1"]):
        # Note: c1 must be in _configs.
        res = pipeline.run(mock_image, mock_mask, config_names=["all_standard"])
        assert "c1" in res

    # Run invalid
    with pytest.raises(ValueError, match="Configuration 'invalid' not found"):
        pipeline.run(mock_image, mock_mask, config_names=["invalid"])


def test_run_all_standard_params(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    # Test "all_standard" expansion without patching get_all_standard_config_names
    # This ensures coverage hits the real method call line.
    # We mock execution steps to avoid heavy computation.
    with patch.object(
        pipeline, "_execute_preprocessing_step"
    ) as mock_exec, patch.object(pipeline, "_extract_features") as mock_ext:

        mock_ext.return_value = {}
        pipeline.run(mock_image, mock_mask, config_names=["all_standard"])

        # Should run 6 standard configs
        assert mock_exec.call_count + mock_ext.call_count > 0
        # Check that we got 6 keys in result (if extraction returns valid dicts)
        # Actually run() iterates configs.
        pass


def test_run_defaults_all_configs(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    # Test pipeline.run() with config_names=None (default)
    # This should trigger line 267: if config_names is None: target_configs = list(self._configs.keys())

    with patch.object(
        pipeline, "_execute_preprocessing_step"
    ) as _mock_exec, patch.object(pipeline, "_extract_features") as mock_ext:

        mock_ext.return_value = {}
        # Call without config_names
        res = pipeline.run(mock_image, mock_mask)

        # Should contain all keys present in pipeline._configs
        assert len(res) == len(pipeline._configs)
        assert len(res) >= 6  # At least standard ones


# --- Preprocessing Step Tests ---


@patch("pictologics.pipeline.resample_image")
def test_step_resample_success(
    mock_resample: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config(
        "res",
        [
            {
                "step": "resample",
                "params": {"new_spacing": (2, 2, 2), "interpolation": "bspline"},
            }
        ],
    )
    # Pipeline calls resample 3 times: image, morph_mask, intensity_mask.
    # We must properly return mask objects for the later calls to avoid EmptyROIMaskError
    mock_resample.side_effect = [mock_image, mock_mask, mock_mask]

    pipeline.run(mock_image, mock_mask, config_names=["res"])

    # Called for image, morph, intensity masks
    assert mock_resample.call_count == 3
    # Check image call args using ANY for image object to avoid array comparison ambiguity
    mock_resample.assert_any_call(
        ANY,
        (2, 2, 2),
        interpolation="bspline",
        round_intensities=False,
        source_mask=None,
    )


@patch("pictologics.pipeline.resegment_mask")
def test_step_resegment(
    mock_reseg: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config(
        "reseg", [{"step": "resegment", "params": {"range_min": 0, "range_max": 100}}]
    )
    mock_reseg.return_value = mock_mask

    pipeline.run(mock_image, mock_mask, config_names=["reseg"])
    mock_reseg.assert_called()

    # Test generated mask case: updates morph mask too?
    mock_reseg.reset_mock()
    pipeline.run(mock_image, mask=None, config_names=["reseg"])
    # Should be called twice? Once for intensity, once for morph (since mask_was_generated=True)
    # Actually code: "if state.mask_was_generated: state.morph_mask = resegment..."
    assert mock_reseg.call_count == 2


@patch("pictologics.pipeline.filter_outliers")
def test_step_filter_outliers(
    mock_filt: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config("filt", [{"step": "filter_outliers", "params": {"sigma": 2.0}}])
    mock_filt.return_value = mock_mask

    # 1. Normal case
    pipeline.run(mock_image, mock_mask, config_names=["filt"])
    mock_filt.assert_called_with(ANY, ANY, 2.0)

    # 2. Generated mask case -> covers 'if state.mask_was_generated'
    mock_filt.reset_mock()
    # Need load_image to return something if no mask provided?
    # Or just use mock_image directly if we pass it.
    pipeline.run(mock_image, mask=None, config_names=["filt"])
    # Should be called twice: once for intensity, once for morph
    assert mock_filt.call_count == 2


@patch("pictologics.pipeline.round_intensities")
def test_step_round_intensities(
    mock_round: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config("round", [{"step": "round_intensities"}])
    mock_round.return_value = mock_image

    pipeline.run(mock_image, mock_mask, config_names=["round"])
    mock_round.assert_called_once()


@patch("pictologics.pipeline.keep_largest_component")
def test_step_keep_largest(
    mock_klc: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    # Default apply_to="both"
    pipeline.add_config("klc", [{"step": "keep_largest_component"}])
    mock_klc.return_value = mock_mask

    pipeline.run(mock_image, mock_mask, config_names=["klc"])

    # Called for morph_mask and intensity_mask
    assert mock_klc.call_count == 2


def test_step_binarize_mask(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    """Test binarize_mask preprocessing step with various parameter modes."""
    # Create a mask with varying values
    varied_mask = Image(
        array=np.array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]] * 10).swapaxes(0, 2),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="mask",
    )

    # 1. Test mask_values as tuple (range): keep values 3-7
    pipeline.add_config(
        "bin_range",
        [{"step": "binarize_mask", "params": {"mask_values": (3, 7)}}],
    )
    res = pipeline.run(mock_image, varied_mask, config_names=["bin_range"])
    assert "bin_range" in res

    # 2. Test mask_values as list: keep only values 2, 5
    pipeline.add_config(
        "bin_list",
        [{"step": "binarize_mask", "params": {"mask_values": [2, 5]}}],
    )
    res = pipeline.run(mock_image, varied_mask, config_names=["bin_list"])
    assert "bin_list" in res

    # 3. Test mask_values as int: keep only value 5
    pipeline.add_config(
        "bin_int",
        [{"step": "binarize_mask", "params": {"mask_values": 5}}],
    )
    res = pipeline.run(mock_image, varied_mask, config_names=["bin_int"])
    assert "bin_int" in res

    # 4. Test threshold mode (default behavior)
    pipeline.add_config(
        "bin_thresh",
        [{"step": "binarize_mask", "params": {"threshold": 4.5}}],
    )
    res = pipeline.run(mock_image, varied_mask, config_names=["bin_thresh"])
    assert "bin_thresh" in res

    # 5. Test with apply_to="morph" only
    pipeline.add_config(
        "bin_morph",
        [
            {
                "step": "binarize_mask",
                "params": {"mask_values": (1, 8), "apply_to": "morph"},
            }
        ],
    )
    res = pipeline.run(mock_image, varied_mask, config_names=["bin_morph"])
    assert "bin_morph" in res


@patch("pictologics.pipeline.discretise_image")
def test_step_discretise(
    mock_disc: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config(
        "disc", [{"step": "discretise", "params": {"method": "FBN", "n_bins": 16}}]
    )
    mock_disc.return_value = mock_image

    pipeline.run(mock_image, mock_mask, config_names=["disc"])

    mock_disc.assert_called_with(mock_image, method="FBN", roi_mask=ANY, n_bins=16)


@patch("pictologics.pipeline.apply_mask")
@patch("pictologics.pipeline.discretise_image")
def test_step_discretise_fbs_empty_error(
    mock_disc: MagicMock,
    mock_apply: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    # FBS attempts to calc n_bins from data. If data empty -> Error.
    pipeline.add_config(
        "fbs", [{"step": "discretise", "params": {"method": "FBS", "bin_width": 10}}]
    )
    mock_apply.return_value = np.array([])  # Empty
    mock_disc.return_value = mock_image

    # This specifically raises EmptyROIMaskError which IS re-raised by run()
    with pytest.raises(EmptyROIMaskError, match="ROI is empty"):
        pipeline.run(mock_image, mock_mask, config_names=["fbs"])


def test_step_unknown(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    pipeline.add_config("unknown", [{"step": "fake_step"}])
    pipeline.run(mock_image, mock_mask, config_names=["unknown"])

    log = pipeline._log[-1]
    assert "error" in log
    assert "Unknown preprocessing step" in log["error"]


# --- Feature Extraction Tests ---


@patch("pictologics.pipeline.calculate_morphology_features")
def test_extract_morphology(
    mock_morph: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config(
        "m", [{"step": "extract_features", "params": {"families": ["morphology"]}}]
    )
    mock_morph.return_value = {"vol": 10}

    res = pipeline.run(mock_image, mock_mask, config_names=["m"])
    assert res["m"]["vol"] == 10
    mock_morph.assert_called_once()


@patch("pictologics.pipeline.calculate_intensity_features")
@patch("pictologics.pipeline.calculate_spatial_intensity_features")
@patch("pictologics.pipeline.calculate_local_intensity_features")
@patch("pictologics.pipeline.apply_mask")
def test_extract_intensity_defaults(
    mock_apply: MagicMock,
    mock_local: MagicMock,
    mock_spatial: MagicMock,
    mock_main: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config(
        "i", [{"step": "extract_features", "params": {"families": ["intensity"]}}]
    )
    mock_main.return_value = {"mean": 1}
    mock_spatial.return_value = {}
    mock_local.return_value = {}
    mock_apply.return_value = [1]

    pipeline.run(mock_image, mock_mask, config_names=["i"])

    mock_main.assert_called_once()
    # Defaults are changed to False now!
    mock_spatial.assert_not_called()
    mock_local.assert_not_called()

    # Enable them explicitly
    pipeline.add_config(
        "i_full",
        [
            {
                "step": "extract_features",
                "params": {
                    "families": ["intensity"],
                    "include_spatial_intensity": True,
                    "include_local_intensity": True,
                },
            }
        ],
    )
    pipeline.run(mock_image, mock_mask, config_names=["i_full"])
    mock_spatial.assert_called()
    mock_local.assert_called()


@patch("pictologics.pipeline.calculate_intensity_features")
@patch("pictologics.pipeline.calculate_spatial_intensity_features")
@patch("pictologics.pipeline.calculate_local_intensity_features")
@patch("pictologics.pipeline.apply_mask")
def test_extract_individual_intensity_families(
    mock_apply: MagicMock,
    mock_local: MagicMock,
    mock_spatial: MagicMock,
    mock_main: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    # families=["spatial_intensity"] but NOT "intensity"
    pipeline.add_config(
        "i_parts",
        [
            {
                "step": "extract_features",
                "params": {"families": ["spatial_intensity", "local_intensity"]},
            }
        ],
    )

    mock_main.return_value = {}
    mock_spatial.return_value = {"spatial": 1}
    mock_local.return_value = {"local": 1}

    res = pipeline.run(mock_image, mock_mask, config_names=["i_parts"])

    mock_main.assert_not_called()
    mock_spatial.assert_called_once()
    mock_local.assert_called_once()
    assert res["i_parts"]["spatial"] == 1


@patch("pictologics.pipeline.calculate_intensity_histogram_features")
@patch("pictologics.pipeline.apply_mask")
def test_extract_histogram_warning(
    mock_apply: MagicMock,
    mock_hist: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    mock_hist.return_value = {"hist_mean": 1}
    pipeline.add_config(
        "h", [{"step": "extract_features", "params": {"families": ["histogram"]}}]
    )

    with pytest.warns(
        UserWarning, match="Histogram features requested but image is not discretised"
    ):
        pipeline.run(mock_image, mock_mask, config_names=["h"])

    mock_hist.assert_called_once()


@patch("pictologics.pipeline.calculate_ivh_features")
@patch("pictologics.pipeline.apply_mask")
def test_extract_ivh_params(
    mock_apply: MagicMock,
    mock_ivh: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    mock_ivh.return_value = {"ivh": 1}
    mock_apply.return_value = [1]

    # 1. New style ivh_params
    pipeline.add_config(
        "ivh_new",
        [
            {
                "step": "extract_features",
                "params": {
                    "families": ["ivh"],
                    "ivh_params": {"bin_width": 0.5, "min_val": 0.0},
                },
            }
        ],
    )
    pipeline.run(mock_image, mock_mask, config_names=["ivh_new"])
    mock_ivh.assert_called_with(ANY, bin_width=0.5, min_val=0.0)

    # 2. Legacy params omitted -> Verify they are NOT picked up used
    # But wait, code refactor REMOVED support for them. So if I pass them, they should be ignored/defaults used.
    # Default for non-discretised non-continuous IVH is bin_width=None?
    # ivh logic: if not provided and not continuous and discretised -> default 1.0.
    # If not provided and not discretised -> pass nothing, assume function defaults?

    mock_ivh.reset_mock()


def test_extract_ivh_full_params(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    # Cover all parameter mapping branches
    with patch("pictologics.pipeline.calculate_ivh_features") as mock_ivh, patch(
        "pictologics.pipeline.apply_mask"
    ) as mock_apply:

        mock_apply.return_value = [1]
        mock_ivh.return_value = {}

        params = {
            "bin_width": 0.1,
            "min_val": 0.0,
            "max_val": 100.0,
            "target_range_min": 10.0,
            "target_range_max": 90.0,
        }

        pipeline.add_config(
            "ivh_full",
            [
                {
                    "step": "extract_features",
                    "params": {"families": ["ivh"], "ivh_params": params},
                }
            ],
        )

        pipeline.run(mock_image, mock_mask, config_names=["ivh_full"])

        mock_ivh.assert_called_with(
            ANY,
            bin_width=0.1,
            min_val=0.0,
            max_val=100.0,
            target_range_min=10.0,
            target_range_max=90.0,
        )


@patch("pictologics.pipeline.calculate_glcm_features")
@patch("pictologics.pipeline.calculate_all_texture_matrices")
def test_extract_texture_error_no_discretise(
    mock_matrices: MagicMock,
    mock_glcm: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config(
        "tex_fail", [{"step": "extract_features", "params": {"families": ["texture"]}}]
    )

    # Config fails -> log error.
    pipeline.run(mock_image, mock_mask, config_names=["tex_fail"])
    log = pipeline._log[-1]
    assert "error" in log
    assert "Texture features requested but image is not discretised" in log["error"]
    assert "You must include a 'discretise' step" in log["error"]


@patch("pictologics.pipeline.discretise_image")
@patch("pictologics.pipeline.calculate_all_texture_matrices")
@patch("pictologics.pipeline.calculate_glcm_features")
# ... mock others if needed ...
def test_extract_texture_success(
    mock_glcm: MagicMock,
    mock_matrices: MagicMock,
    mock_disc: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    # Need to setup mocks for texture matrices return
    mock_matrices.return_value = {
        "glcm": 1,
        "glrlm": 1,
        "glszm": 1,
        "gldzm": 1,
        "ngtdm_s": 1,
        "ngtdm_n": 1,
        "ngldm": 1,
    }
    mock_glcm.return_value = {}
    mock_disc.return_value = mock_image

    # With valid discretise step
    pipeline.add_config(
        "tex_ok",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture"]}},
        ],
    )

    # We need to ensure 'pipeline.py' feature extraction calls GLCM, GLRLM etc.
    # I'll rely on patching calculation functions to prevent errors.
    with patch("pictologics.pipeline.calculate_glrlm_features") as mr, patch(
        "pictologics.pipeline.calculate_glszm_features"
    ) as ms, patch("pictologics.pipeline.calculate_gldzm_features") as md, patch(
        "pictologics.pipeline.calculate_ngtdm_features"
    ) as mt, patch(
        "pictologics.pipeline.calculate_ngldm_features"
    ) as mn:

        mr.return_value = {}
        ms.return_value = {}
        md.return_value = {}
        mt.return_value = {}
        mn.return_value = {}

        pipeline.run(mock_image, mock_mask, config_names=["tex_ok"])

        # Verified calls
        mock_matrices.assert_called()
        mock_glcm.assert_called()


def test_save_log(pipeline: RadiomicsPipeline, tmp_path: Any) -> None:
    pipeline._log.append({"test": "entry"})
    p = tmp_path / "log.json"
    pipeline.save_log(str(p))
    assert p.exists()

    p2 = tmp_path / "log_no_ext"
    pipeline.save_log(str(p2))
    assert (tmp_path / "log_no_ext.json").exists()


def test_clear_log(pipeline: RadiomicsPipeline) -> None:
    pipeline._log.append({"a": 1})
    pipeline.clear_log()
    assert len(pipeline._log) == 0


def test_empty_roi_check(pipeline: RadiomicsPipeline, mock_image: Image) -> None:
    # Manual check of helper
    state = MagicMock()
    state.intensity_mask.array = np.zeros((10, 10, 10))  # Empty
    state.morph_mask.array = np.zeros((10, 10, 10))

    with pytest.raises(EmptyROIMaskError):
        pipeline._ensure_nonempty_roi(state, "test")


def test_empty_morph_roi_only(pipeline: RadiomicsPipeline) -> None:
    """Covers the morph-only empty branch in _ensure_nonempty_roi."""
    state = MagicMock()
    state.intensity_mask.array = np.ones((10, 10, 10))   # Non-empty
    state.morph_mask.array = np.zeros((10, 10, 10))       # Empty

    with pytest.raises(EmptyROIMaskError, match="ROI is empty"):
        pipeline._ensure_nonempty_roi(state, "morph_empty")


def test_ivh_discretisation_mode(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    # Test 'ivh_discretisation' param which does temporary discretisation
    pipeline.add_config(
        "ivh_disc",
        [
            {
                "step": "extract_features",
                "params": {
                    "families": ["ivh"],
                    "ivh_discretisation": {"method": "FBN", "n_bins": 10},
                    "ivh_params": {
                        "bin_width": 0.5
                    },  # Should override or exist alongside?
                },
            }
        ],
    )

    with patch("pictologics.pipeline.discretise_image") as mock_disc, patch(
        "pictologics.pipeline.apply_mask"
    ) as mock_apply, patch("pictologics.pipeline.calculate_ivh_features") as mock_calc:

        mock_disc.return_value = mock_image  # Temp disc image
        mock_apply.return_value = [1, 2]
        mock_calc.return_value = {}

        pipeline.run(mock_image, mock_mask, config_names=["ivh_disc"])

        # Should call discretise_image
        mock_disc.assert_called_with(ANY, method="FBN", roi_mask=ANY, n_bins=10)
        # Should call calc with provided bin_width from ivh_params
        mock_calc.assert_called_with(ANY, bin_width=0.5)


def test_ivh_continuous_mode(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    pipeline.add_config(
        "ivh_cont",
        [
            {
                "step": "extract_features",
                "params": {"families": ["ivh"], "ivh_use_continuous": True},
            }
        ],
    )

    with patch("pictologics.pipeline.apply_mask") as mock_apply, patch(
        "pictologics.pipeline.calculate_ivh_features"
    ) as mock_calc:

        mock_apply.return_value = [1.5, 2.5]
        mock_calc.return_value = {}

        pipeline.run(mock_image, mock_mask, config_names=["ivh_cont"])

        # Apply mask called on raw image (we can't check easily, but logic ensures it)
        # Logic: ivh_use_continuous=True -> apply_mask(state.raw_image...)
        mock_calc.assert_called()


def test_params_type_errors(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    # Check that passing non-dict to params raises ValueError
    # e.g. spatial_intensity_params="string"
    pipeline.add_config(
        "type_err",
        [
            {
                "step": "extract_features",
                "params": {
                    "families": ["intensity"],
                    "spatial_intensity_params": "bad",
                },
            }
        ],
    )

    pipeline.run(mock_image, mock_mask, config_names=["type_err"])
    log = pipeline._log[-1]
    assert "error" in log
    assert "spatial_intensity_params must be a dict" in log["error"]


def test_params_type_errors_all(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    # Test other param types
    cases = [
        ("local_intensity_params", "bad"),
        ("ivh_params", "bad"),
        ("texture_matrix_params", "bad"),
    ]

    for param_name, bad_val in cases:
        pipeline.clear_log()
        pipeline.add_config(
            f"bad_{param_name}",
            [
                {
                    "step": "extract_features",
                    "params": {
                        "families": ["intensity", "ivh", "texture"],
                        param_name: bad_val,
                    },
                }
            ],
        )
        # Note: we need "texture" family to hit texture_matrix_params check?
        # Actually code checks params BEFORE family logic?
        # Code: params.get(...) -> Checks isinstance -> Then family logic.
        # So we don't strictly need families set for the check to fire,
        # BUT families default includes all.

        pipeline.run(mock_image, mock_mask, config_names=[f"bad_{param_name}"])
        log = pipeline._log[-1]
        assert "error" in log
        assert f"{param_name} must be a dict" in log["error"]


def test_params_explicit_none(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    # Explicit None should be converted to {}
    with patch(
        "pictologics.pipeline.calculate_spatial_intensity_features"
    ) as mock_calc:
        mock_calc.return_value = {}
        pipeline.add_config(
            "none_params",
            [
                {
                    "step": "extract_features",
                    "params": {
                        "families": ["spatial_intensity"],
                        "spatial_intensity_params": None,
                        "local_intensity_params": None,
                    },
                }
            ],
        )

        # We need mock for calculate_spatial to succeed
        with patch("pictologics.pipeline.apply_mask", return_value=[1]):
            pipeline.run(mock_image, mock_mask, config_names=["none_params"])

        # If it didn't crash and called calc, we good.
        mock_calc.assert_called()


def test_params_explicit_none_all(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    # Test ivh and texture matrix params explicit None
    with patch("pictologics.pipeline.calculate_ivh_features") as mock_ivh, patch(
        "pictologics.pipeline.calculate_all_texture_matrices"
    ) as mock_tex, patch("pictologics.pipeline.calculate_glcm_features"), patch(
        "pictologics.pipeline.calculate_glrlm_features"
    ), patch(
        "pictologics.pipeline.calculate_glszm_features"
    ), patch(
        "pictologics.pipeline.calculate_gldzm_features"
    ), patch(
        "pictologics.pipeline.calculate_ngtdm_features"
    ), patch(
        "pictologics.pipeline.calculate_ngldm_features"
    ), patch(
        "pictologics.pipeline.discretise_image", return_value=mock_image
    ):

        mock_ivh.return_value = {}
        mock_tex.return_value = {
            "glcm": 1,
            "glrlm": 1,
            "glszm": 1,
            "gldzm": 1,
            "ngtdm_s": 1,
            "ngtdm_n": 1,
            "ngldm": 1,
        }

        pipeline.add_config(
            "none_all",
            [
                {
                    "step": "extract_features",
                    "params": {
                        "families": ["ivh", "texture"],
                        "ivh_params": None,
                        "texture_matrix_params": None,
                    },
                }
            ],
        )

        # Need pre-discretisation for texture, or implicit discretise step in config?
        # extract_features complains if not discretised.
        # So we simply cheat pipeline state? Or add discretise step.
        pipeline.add_config(
            "none_all_valid",
            [
                {"step": "discretise", "params": {"n_bins": 10}},
                {
                    "step": "extract_features",
                    "params": {
                        "families": ["ivh", "texture"],
                        "ivh_params": None,
                        "texture_matrix_params": None,
                    },
                },
            ],
        )

        with patch("pictologics.pipeline.apply_mask", return_value=[1]):
            pipeline.run(mock_image, mock_mask, config_names=["none_all_valid"])

        mock_ivh.assert_called()
        mock_tex.assert_called()


def test_run_subject_id(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    pipeline.add_config("subj", [])
    res = pipeline.run(mock_image, mock_mask, subject_id="P001", config_names=["subj"])
    assert res["subj"]["subject_id"] == "P001"


@patch("pictologics.pipeline.apply_mask")
@patch("pictologics.pipeline.discretise_image")
def test_step_discretise_fbs_success(
    mock_disc: MagicMock,
    mock_apply: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config("fbs_ok", [{"step": "discretise", "params": {"method": "FBS"}}])
    mock_disc.return_value = mock_image
    mock_apply.return_value = np.array([10, 20])

    # Run should not fail
    pipeline.run(mock_image, mock_mask, config_names=["fbs_ok"])

    # How to verify n_bins?
    # We can't easily access state. But extraction might use it.
    # Let's add extraction step
    pipeline.add_config(
        "fbs_extract",
        [
            {"step": "discretise", "params": {"method": "FBS"}},
            {"step": "extract_features", "params": {"families": ["texture"]}},
        ],
    )

    with patch(
        "pictologics.pipeline.calculate_all_texture_matrices"
    ) as mock_tex, patch(
        "pictologics.pipeline.calculate_glcm_features", return_value={}
    ), patch(
        "pictologics.pipeline.calculate_glrlm_features", return_value={}
    ), patch(
        "pictologics.pipeline.calculate_glszm_features", return_value={}
    ), patch(
        "pictologics.pipeline.calculate_gldzm_features", return_value={}
    ), patch(
        "pictologics.pipeline.calculate_ngtdm_features", return_value={}
    ), patch(
        "pictologics.pipeline.calculate_ngldm_features", return_value={}
    ):

        mock_tex.return_value = {
            "glcm": 1,
            "glrlm": 1,
            "glszm": 1,
            "gldzm": 1,
            "ngtdm_s": 1,
            "ngtdm_n": 1,
            "ngldm": 1,
        }

        pipeline.run(mock_image, mock_mask, config_names=["fbs_extract"])

        # Check if n_bins=20 (from max(10, 20)) passed to matrices
        args, kwargs = mock_tex.call_args
        assert args[2] == 20


def test_ivh_disc_with_params(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    # Test ivh_discretisation with bin_width (FBS) overwriting ivh params
    pipeline.add_config(
        "ivh_fbs",
        [
            {
                "step": "extract_features",
                "params": {
                    "families": ["ivh"],
                    "ivh_discretisation": {
                        "method": "FBS",
                        "bin_width": 2.5,
                        "min_val": 0.0,  # Cover min_val override
                    },
                },
            }
        ],
    )

    with patch("pictologics.pipeline.calculate_ivh_features") as mock_ivh, patch(
        "pictologics.pipeline.apply_mask", return_value=[1]
    ), patch("pictologics.pipeline.discretise_image", return_value=mock_image):

        mock_ivh.return_value = {}
        pipeline.run(mock_image, mock_mask, config_names=["ivh_fbs"])

        mock_ivh.assert_called_with(ANY, bin_width=2.5, min_val=0.0)


def test_texture_matrix_params_explicit(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    pipeline.add_config(
        "tex_mat",
        [
            {"step": "discretise", "params": {"n_bins": 10}},
            {
                "step": "extract_features",
                "params": {
                    "families": ["texture"],
                    "texture_matrix_params": {"ngldm_alpha": 7},
                },
            },
        ],
    )

    with patch(
        "pictologics.pipeline.calculate_all_texture_matrices"
    ) as mock_tex, patch(
        "pictologics.pipeline.discretise_image", return_value=mock_image
    ), patch(
        "pictologics.pipeline.apply_mask", return_value=[1]
    ), patch(
        "pictologics.pipeline.calculate_glcm_features", return_value={}
    ), patch(
        "pictologics.pipeline.calculate_glrlm_features", return_value={}
    ), patch(
        "pictologics.pipeline.calculate_glszm_features", return_value={}
    ), patch(
        "pictologics.pipeline.calculate_gldzm_features", return_value={}
    ), patch(
        "pictologics.pipeline.calculate_ngtdm_features", return_value={}
    ), patch(
        "pictologics.pipeline.calculate_ngldm_features", return_value={}
    ):

        mock_tex.return_value = {
            "glcm": 1,
            "glrlm": 1,
            "glszm": 1,
            "gldzm": 1,
            "ngtdm_s": 1,
            "ngtdm_n": 1,
            "ngldm": 1,
        }

        pipeline.run(mock_image, mock_mask, config_names=["tex_mat"])

        args, kwargs = mock_tex.call_args
        assert kwargs.get("ngldm_alpha") == 7


# --- Filter Step Tests ---


@patch("pictologics.pipeline.mean_filter")
def test_step_filter_mean(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test mean filter step."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_mean",
        [{"step": "filter", "params": {"type": "mean", "support": 5}}],
    )

    pipeline.run(mock_image, mock_mask, config_names=["filter_mean"])
    mock_filter.assert_called_once()
    call_kwargs = mock_filter.call_args.kwargs
    assert call_kwargs.get("support") == 5
    assert call_kwargs.get("boundary") is not None


@patch("pictologics.pipeline.laplacian_of_gaussian")
def test_step_filter_log(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test LoG filter step with auto-spacing injection."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_log",
        [
            {
                "step": "filter",
                "params": {"type": "log", "sigma_mm": 1.5, "truncate": 4.0},
            }
        ],
    )

    pipeline.run(mock_image, mock_mask, config_names=["filter_log"])
    mock_filter.assert_called_once()
    call_kwargs = mock_filter.call_args.kwargs
    assert call_kwargs.get("sigma_mm") == 1.5
    assert call_kwargs.get("spacing_mm") is not None  # Auto-injected


@patch("pictologics.pipeline.laws_filter")
def test_step_filter_laws(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test Laws filter step."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_laws",
        [
            {
                "step": "filter",
                "params": {
                    "type": "laws",
                    "kernel": "L5E5E5",
                    "rotation_invariant": True,
                    "pooling": "max",
                },
            }
        ],
    )

    pipeline.run(mock_image, mock_mask, config_names=["filter_laws"])
    mock_filter.assert_called_once()
    # Check kernel is passed as first positional arg
    call_args = mock_filter.call_args
    assert call_args.args[1] == "L5E5E5"


@patch("pictologics.pipeline.gabor_filter")
def test_step_filter_gabor(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test Gabor filter step."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_gabor",
        [
            {
                "step": "filter",
                "params": {
                    "type": "gabor",
                    "sigma_mm": 5.0,
                    "lambda_mm": 2.0,
                    "gamma": 1.5,
                },
            }
        ],
    )

    pipeline.run(mock_image, mock_mask, config_names=["filter_gabor"])
    mock_filter.assert_called_once()
    call_kwargs = mock_filter.call_args.kwargs
    assert call_kwargs.get("sigma_mm") == 5.0
    assert call_kwargs.get("spacing_mm") is not None  # Auto-injected


@patch("pictologics.pipeline.wavelet_transform")
def test_step_filter_wavelet(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test wavelet filter step."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_wavelet",
        [
            {
                "step": "filter",
                "params": {
                    "type": "wavelet",
                    "wavelet": "db3",
                    "level": 1,
                    "decomposition": "LLH",
                },
            }
        ],
    )

    pipeline.run(mock_image, mock_mask, config_names=["filter_wavelet"])
    mock_filter.assert_called_once()
    call_kwargs = mock_filter.call_args.kwargs
    assert call_kwargs.get("wavelet") == "db3"


@patch("pictologics.pipeline.simoncelli_wavelet")
def test_step_filter_simoncelli(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test Simoncelli wavelet step (no boundary param)."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_simon",
        [{"step": "filter", "params": {"type": "simoncelli", "level": 2}}],
    )

    pipeline.run(mock_image, mock_mask, config_names=["filter_simon"])
    mock_filter.assert_called_once()
    call_kwargs = mock_filter.call_args.kwargs
    assert call_kwargs.get("level") == 2
    assert "boundary" not in call_kwargs  # Simoncelli doesn't use boundary


@patch("pictologics.pipeline.riesz_transform")
def test_step_filter_riesz_base(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test Riesz transform base variant."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_riesz",
        [{"step": "filter", "params": {"type": "riesz", "order": 1}}],
    )

    pipeline.run(mock_image, mock_mask, config_names=["filter_riesz"])
    mock_filter.assert_called_once()


@patch("pictologics.pipeline.riesz_log")
def test_step_filter_riesz_log(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test Riesz-LoG variant with spacing injection."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_riesz_log",
        [
            {
                "step": "filter",
                "params": {"type": "riesz", "variant": "log", "sigma_mm": 2.0},
            }
        ],
    )

    pipeline.run(mock_image, mock_mask, config_names=["filter_riesz_log"])
    mock_filter.assert_called_once()
    call_kwargs = mock_filter.call_args.kwargs
    assert call_kwargs.get("spacing_mm") is not None


@patch("pictologics.pipeline.riesz_simoncelli")
def test_step_filter_riesz_simoncelli(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test Riesz-Simoncelli variant."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_riesz_simon",
        [{"step": "filter", "params": {"type": "riesz", "variant": "simoncelli"}}],
    )

    pipeline.run(mock_image, mock_mask, config_names=["filter_riesz_simon"])
    mock_filter.assert_called_once()


def test_step_filter_missing_type(
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test filter step error when type is missing."""
    pipeline.add_config(
        "filter_no_type",
        [{"step": "filter", "params": {"support": 5}}],  # Missing 'type'
    )

    pipeline.run(mock_image, mock_mask, config_names=["filter_no_type"])
    log = pipeline._log[-1]
    assert "error" in log
    assert "Filter step requires 'type' parameter" in log["error"]


def test_step_filter_unknown_type(
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test filter step error when type is unknown."""
    pipeline.add_config(
        "filter_bad_type",
        [{"step": "filter", "params": {"type": "invalid_filter"}}],
    )

    pipeline.run(mock_image, mock_mask, config_names=["filter_bad_type"])
    log = pipeline._log[-1]
    assert "error" in log
    assert "Unknown filter type: invalid_filter" in log["error"]


@patch("pictologics.pipeline.mean_filter")
def test_step_filter_boundary_options(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test all boundary condition options."""
    from pictologics.filters import BoundaryCondition

    mock_filter.return_value = mock_image.array

    # Test each boundary option
    for boundary_name, expected in [
        ("mirror", BoundaryCondition.MIRROR),
        ("nearest", BoundaryCondition.NEAREST),
        ("zero", BoundaryCondition.ZERO),
        ("constant", BoundaryCondition.ZERO),
        ("periodic", BoundaryCondition.PERIODIC),
        ("wrap", BoundaryCondition.PERIODIC),
    ]:
        mock_filter.reset_mock()
        config_name = f"filter_boundary_{boundary_name}"
        pipeline.add_config(
            config_name,
            [
                {
                    "step": "filter",
                    "params": {"type": "mean", "support": 3, "boundary": boundary_name},
                }
            ],
        )
        pipeline.run(mock_image, mock_mask, config_names=[config_name])
        call_kwargs = mock_filter.call_args.kwargs
        assert call_kwargs.get("boundary") == expected, f"Failed for {boundary_name}"


def test_step_resample_missing_param(
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test resample step error when new_spacing is missing."""
    pipeline.add_config(
        "resample_no_spacing",
        [
            {"step": "resample", "params": {"interpolation": "linear"}}
        ],  # Missing new_spacing
    )

    pipeline.run(mock_image, mock_mask, config_names=["resample_no_spacing"])
    log = pipeline._log[-1]
    assert "error" in log
    assert "Resample step requires 'new_spacing' parameter" in log["error"]


def test_step_binarize_missing_threshold(
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test binarize_mask step error when threshold is missing and no mask_values provided."""
    pipeline.add_config(
        "binarize_no_thresh",
        [
            {"step": "binarize_mask", "params": {"threshold": None}}
        ],  # Explicit None without mask_values
    )

    pipeline.run(mock_image, mock_mask, config_names=["binarize_no_thresh"])
    log = pipeline._log[-1]
    assert "error" in log
    assert (
        "binarize_mask requires 'threshold' unless mask_values is provided"
        in log["error"]
    )


# --- Deduplication Configuration Tests ---


def test_pipeline_init_with_deduplication_rules_string() -> None:
    """Test pipeline init with deduplication_rules as string version."""
    from pictologics.deduplication import DeduplicationRules

    pipeline = RadiomicsPipeline(deduplication_rules="1.0.0")
    assert pipeline._deduplication_rules.version == "1.0.0"
    assert isinstance(pipeline._deduplication_rules, DeduplicationRules)


def test_pipeline_init_with_deduplication_rules_object() -> None:
    """Test pipeline init with deduplication_rules as DeduplicationRules object."""
    from pictologics.deduplication import DeduplicationRules

    rules = DeduplicationRules.get_version("1.0.0")
    pipeline = RadiomicsPipeline(deduplication_rules=rules)
    assert pipeline._deduplication_rules is rules
    assert pipeline._deduplication_rules.version == "1.0.0"


def test_deduplication_rules_setter_with_string(pipeline: RadiomicsPipeline) -> None:
    """Test deduplication_rules setter with string version."""
    from pictologics.deduplication import DeduplicationRules

    # Set via string
    pipeline.deduplication_rules = "1.0.0"
    assert pipeline._deduplication_rules.version == "1.0.0"
    assert isinstance(pipeline._deduplication_rules, DeduplicationRules)
    # Check that configs are marked as modified
    assert pipeline._configs_modified_since_plan is True


def test_deduplication_rules_setter_with_object(pipeline: RadiomicsPipeline) -> None:
    """Test deduplication_rules setter with DeduplicationRules object."""
    from pictologics.deduplication import DeduplicationRules

    rules = DeduplicationRules.get_version("1.0.0")
    pipeline.deduplication_rules = rules
    assert pipeline._deduplication_rules is rules
    assert pipeline._configs_modified_since_plan is True


def test_last_deduplication_plan_property(pipeline: RadiomicsPipeline) -> None:
    """Test last_deduplication_plan property."""
    # Initially None
    assert pipeline.last_deduplication_plan is None

    # After computing a plan, it should be accessible
    from pictologics.deduplication import ConfigurationAnalyzer

    pipeline.add_config(
        "cfg1",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ],
    )

    analyzer = ConfigurationAnalyzer(pipeline._configs, pipeline._deduplication_rules)
    plan = analyzer.analyze()
    pipeline._last_deduplication_plan = plan
    pipeline._configs_modified_since_plan = False

    assert pipeline.last_deduplication_plan is plan


# --- Individual Texture Family Tests (via Deduplication Path) ---


@patch("pictologics.pipeline.discretise_image")
@patch("pictologics.pipeline.calculate_all_texture_matrices")
@patch("pictologics.pipeline.calculate_glrlm_features")
def test_extract_single_family_texture_glrlm(
    mock_glrlm: MagicMock,
    mock_matrices: MagicMock,
    mock_disc: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test extracting only glrlm texture features via dedup path."""
    # Dedup path requires multiple configs
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_disc.return_value = mock_image
    mock_matrices.return_value = {
        "glcm": np.zeros((32, 32, 13)),
        "glrlm": np.zeros((32, 10, 13)),
        "glszm": np.zeros((32, 10)),
        "gldzm": np.zeros((32, 10)),
        "ngtdm_s": np.zeros(32),
        "ngtdm_n": np.zeros(32),
        "ngldm": np.zeros((32, 10)),
    }
    mock_glrlm.return_value = {"glrlm_sre": 0.5}

    # Add two configs to trigger deduplication path
    pipeline.add_config(
        "cfg1",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture_glrlm"]}},
        ],
    )
    pipeline.add_config(
        "cfg2",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture_glrlm"]}},
        ],
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_glrlm.assert_called()
    assert "glrlm_sre" in result["cfg1"]


@patch("pictologics.pipeline.discretise_image")
@patch("pictologics.pipeline.calculate_all_texture_matrices")
@patch("pictologics.pipeline.calculate_glszm_features")
def test_extract_single_family_texture_glszm(
    mock_glszm: MagicMock,
    mock_matrices: MagicMock,
    mock_disc: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test extracting only glszm texture features via dedup path."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_disc.return_value = mock_image
    mock_matrices.return_value = {
        "glcm": np.zeros((32, 32, 13)),
        "glrlm": np.zeros((32, 10, 13)),
        "glszm": np.zeros((32, 10)),
        "gldzm": np.zeros((32, 10)),
        "ngtdm_s": np.zeros(32),
        "ngtdm_n": np.zeros(32),
        "ngldm": np.zeros((32, 10)),
    }
    mock_glszm.return_value = {"glszm_lze": 0.7}

    pipeline.add_config(
        "cfg1",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture_glszm"]}},
        ],
    )
    pipeline.add_config(
        "cfg2",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture_glszm"]}},
        ],
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_glszm.assert_called()
    assert "glszm_lze" in result["cfg1"]


@patch("pictologics.pipeline.discretise_image")
@patch("pictologics.pipeline.calculate_all_texture_matrices")
@patch("pictologics.pipeline.calculate_gldzm_features")
def test_extract_single_family_texture_gldzm(
    mock_gldzm: MagicMock,
    mock_matrices: MagicMock,
    mock_disc: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test extracting only gldzm texture features via dedup path."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_disc.return_value = mock_image
    mock_matrices.return_value = {
        "glcm": np.zeros((32, 32, 13)),
        "glrlm": np.zeros((32, 10, 13)),
        "glszm": np.zeros((32, 10)),
        "gldzm": np.zeros((32, 10)),
        "ngtdm_s": np.zeros(32),
        "ngtdm_n": np.zeros(32),
        "ngldm": np.zeros((32, 10)),
    }
    mock_gldzm.return_value = {"gldzm_dze": 0.3}

    pipeline.add_config(
        "cfg1",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture_gldzm"]}},
        ],
    )
    pipeline.add_config(
        "cfg2",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture_gldzm"]}},
        ],
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_gldzm.assert_called()
    assert "gldzm_dze" in result["cfg1"]


@patch("pictologics.pipeline.discretise_image")
@patch("pictologics.pipeline.calculate_all_texture_matrices")
@patch("pictologics.pipeline.calculate_ngtdm_features")
def test_extract_single_family_texture_ngtdm(
    mock_ngtdm: MagicMock,
    mock_matrices: MagicMock,
    mock_disc: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test extracting only ngtdm texture features via dedup path."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_disc.return_value = mock_image
    mock_matrices.return_value = {
        "glcm": np.zeros((32, 32, 13)),
        "glrlm": np.zeros((32, 10, 13)),
        "glszm": np.zeros((32, 10)),
        "gldzm": np.zeros((32, 10)),
        "ngtdm_s": np.zeros(32),
        "ngtdm_n": np.zeros(32),
        "ngldm": np.zeros((32, 10)),
    }
    mock_ngtdm.return_value = {"ngtdm_coarseness": 0.8}

    pipeline.add_config(
        "cfg1",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture_ngtdm"]}},
        ],
    )
    pipeline.add_config(
        "cfg2",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture_ngtdm"]}},
        ],
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_ngtdm.assert_called()
    assert "ngtdm_coarseness" in result["cfg1"]


@patch("pictologics.pipeline.discretise_image")
@patch("pictologics.pipeline.calculate_all_texture_matrices")
@patch("pictologics.pipeline.calculate_ngldm_features")
def test_extract_single_family_texture_ngldm(
    mock_ngldm: MagicMock,
    mock_matrices: MagicMock,
    mock_disc: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test extracting only ngldm texture features via dedup path."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_disc.return_value = mock_image
    mock_matrices.return_value = {
        "glcm": np.zeros((32, 32, 13)),
        "glrlm": np.zeros((32, 10, 13)),
        "glszm": np.zeros((32, 10)),
        "gldzm": np.zeros((32, 10)),
        "ngtdm_s": np.zeros(32),
        "ngtdm_n": np.zeros(32),
        "ngldm": np.zeros((32, 10)),
    }
    mock_ngldm.return_value = {"ngldm_lde": 0.6}

    pipeline.add_config(
        "cfg1",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture_ngldm"]}},
        ],
    )
    pipeline.add_config(
        "cfg2",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture_ngldm"]}},
        ],
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_ngldm.assert_called()
    assert "ngldm_lde" in result["cfg1"]


# --- Histogram and IVH via Dedup Path ---


@patch("pictologics.pipeline.discretise_image")
@patch("pictologics.pipeline.calculate_intensity_histogram_features")
def test_extract_histogram_via_dedup_path(
    mock_hist: MagicMock,
    mock_disc: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test histogram features via deduplication path."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_disc.return_value = mock_image
    mock_hist.return_value = {"hist_mean": 0.5}

    pipeline.add_config(
        "cfg1",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["histogram"]}},
        ],
    )
    pipeline.add_config(
        "cfg2",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["histogram"]}},
        ],
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_hist.assert_called()
    assert "hist_mean" in result["cfg1"]


@patch("pictologics.pipeline.calculate_intensity_histogram_features")
def test_extract_histogram_without_discretisation_warns(
    mock_hist: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test histogram without discretisation issues a warning via dedup path."""
    import warnings

    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_hist.return_value = {"hist_mean": 0.5}

    # No discretise step - should trigger warning
    pipeline.add_config(
        "cfg1", [{"step": "extract_features", "params": {"families": ["histogram"]}}]
    )
    pipeline.add_config(
        "cfg2", [{"step": "extract_features", "params": {"families": ["histogram"]}}]
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

        # Check that the expected warning was raised
        warning_messages = [str(warning.message) for warning in w]
        assert any("not discretised" in msg for msg in warning_messages)


@patch("pictologics.pipeline.calculate_ivh_features")
def test_extract_ivh_via_dedup_path(
    mock_ivh: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test IVH features via deduplication path."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_ivh.return_value = {"ivh_v10": 0.5}

    pipeline.add_config(
        "cfg1", [{"step": "extract_features", "params": {"families": ["ivh"]}}]
    )
    pipeline.add_config(
        "cfg2", [{"step": "extract_features", "params": {"families": ["ivh"]}}]
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_ivh.assert_called()
    assert "ivh_v10" in result["cfg1"]


@patch("pictologics.pipeline.calculate_ivh_features")
def test_extract_ivh_via_dedup_path_with_discretisation(
    mock_ivh: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test IVH features via dedup path with ivh_discretisation and verify min_val is passed."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_ivh.return_value = {"ivh_v10": 0.5}

    # Test ivh_discretisation branch via dedup - include min_val to cover line 1155
    with patch("pictologics.pipeline.discretise_image") as mock_disc:
        mock_disc.return_value = mock_image
        pipeline.add_config(
            "cfg1",
            [
                {
                    "step": "extract_features",
                    "params": {
                        "families": ["ivh"],
                        "ivh_discretisation": {
                            "method": "FBS",
                            "bin_width": 2.5,
                            "min_val": -100.0,
                        },
                    },
                }
            ],
        )
        pipeline.add_config(
            "cfg2",
            [
                {
                    "step": "extract_features",
                    "params": {
                        "families": ["ivh"],
                        "ivh_discretisation": {
                            "method": "FBS",
                            "bin_width": 2.5,
                            "min_val": -100.0,
                        },
                    },
                }
            ],
        )
        result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

        mock_disc.assert_called()
        mock_ivh.assert_called()
        # Verify min_val was passed to calculate_ivh_features
        call_kwargs = mock_ivh.call_args.kwargs
        assert call_kwargs.get("min_val") == -100.0
        assert call_kwargs.get("bin_width") == 2.5
        assert "ivh_v10" in result["cfg1"]


@patch("pictologics.pipeline.discretise_image")
@patch("pictologics.pipeline.calculate_ivh_features")
def test_extract_ivh_discretised_auto_bin_width(
    mock_ivh: MagicMock,
    mock_disc: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test IVH features with discretised image auto-sets bin_width=1.0."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_disc.return_value = mock_image
    mock_ivh.return_value = {"ivh_v10": 0.5}

    # Use discretise step but don't set explicit bin_width in ivh_params
    # This should trigger the auto bin_width=1.0 for discretised images (line 1163)
    pipeline.add_config(
        "cfg1",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["ivh"]}},
        ],
    )
    pipeline.add_config(
        "cfg2",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["ivh"]}},
        ],
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_ivh.assert_called()
    # Verify bin_width=1.0 was passed automatically
    call_kwargs = mock_ivh.call_args.kwargs
    assert call_kwargs.get("bin_width") == 1.0
    assert "ivh_v10" in result["cfg1"]


@patch("pictologics.pipeline.calculate_ivh_features")
def test_extract_ivh_with_ivh_params_via_dedup(
    mock_ivh: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test IVH via dedup with ivh_params containing max_val to cover line 1155."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_ivh.return_value = {"ivh_v10": 0.5}

    # Pass ivh_params with max_val to ensure line 1155 is hit
    pipeline.add_config(
        "cfg1",
        [
            {
                "step": "extract_features",
                "params": {"families": ["ivh"], "ivh_params": {"max_val": 500.0}},
            }
        ],
    )
    pipeline.add_config(
        "cfg2",
        [
            {
                "step": "extract_features",
                "params": {"families": ["ivh"], "ivh_params": {"max_val": 500.0}},
            }
        ],
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_ivh.assert_called()
    # Verify max_val was passed from ivh_params
    call_kwargs = mock_ivh.call_args.kwargs
    assert call_kwargs.get("max_val") == 500.0
    assert "ivh_v10" in result["cfg1"]


@patch("pictologics.pipeline.calculate_ivh_features")
def test_extract_ivh_via_dedup_path_continuous(
    mock_ivh: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test IVH features via dedup path with ivh_use_continuous."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_ivh.return_value = {"ivh_v10": 0.5}

    pipeline.add_config(
        "cfg1",
        [
            {
                "step": "extract_features",
                "params": {"families": ["ivh"], "ivh_use_continuous": True},
            }
        ],
    )
    pipeline.add_config(
        "cfg2",
        [
            {
                "step": "extract_features",
                "params": {"families": ["ivh"], "ivh_use_continuous": True},
            }
        ],
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_ivh.assert_called()
    assert "ivh_v10" in result["cfg1"]


@patch("pictologics.pipeline.calculate_morphology_features")
def test_extract_morphology_via_dedup_path(
    mock_morph: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test morphology features via deduplication path."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_morph.return_value = {"morph_volume": 100.0}

    pipeline.add_config(
        "cfg1", [{"step": "extract_features", "params": {"families": ["morphology"]}}]
    )
    pipeline.add_config(
        "cfg2", [{"step": "extract_features", "params": {"families": ["morphology"]}}]
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_morph.assert_called()
    assert "morph_volume" in result["cfg1"]


@patch("pictologics.pipeline.calculate_intensity_features")
@patch("pictologics.pipeline.calculate_spatial_intensity_features")
@patch("pictologics.pipeline.calculate_local_intensity_features")
def test_extract_intensity_with_spatial_local_via_dedup(
    mock_local: MagicMock,
    mock_spatial: MagicMock,
    mock_intensity: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test intensity features with spatial/local via deduplication path."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_intensity.return_value = {"int_mean": 50.0}
    mock_spatial.return_value = {"spatial_peak": 100.0}
    mock_local.return_value = {"local_peak": 75.0}

    pipeline.add_config(
        "cfg1",
        [
            {
                "step": "extract_features",
                "params": {
                    "families": ["intensity"],
                    "include_spatial_intensity": True,
                    "include_local_intensity": True,
                },
            }
        ],
    )
    pipeline.add_config(
        "cfg2",
        [
            {
                "step": "extract_features",
                "params": {
                    "families": ["intensity"],
                    "include_spatial_intensity": True,
                    "include_local_intensity": True,
                },
            }
        ],
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_intensity.assert_called()
    mock_spatial.assert_called()
    mock_local.assert_called()
    assert "int_mean" in result["cfg1"]


@patch("pictologics.pipeline.calculate_spatial_intensity_features")
def test_extract_spatial_intensity_via_dedup(
    mock_spatial: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test spatial_intensity family via deduplication path."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_spatial.return_value = {"spatial_peak": 100.0}

    pipeline.add_config(
        "cfg1",
        [{"step": "extract_features", "params": {"families": ["spatial_intensity"]}}],
    )
    pipeline.add_config(
        "cfg2",
        [{"step": "extract_features", "params": {"families": ["spatial_intensity"]}}],
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_spatial.assert_called()
    assert "spatial_peak" in result["cfg1"]


@patch("pictologics.pipeline.calculate_local_intensity_features")
def test_extract_local_intensity_via_dedup(
    mock_local: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test local_intensity family via deduplication path."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_local.return_value = {"local_peak": 75.0}

    pipeline.add_config(
        "cfg1",
        [{"step": "extract_features", "params": {"families": ["local_intensity"]}}],
    )
    pipeline.add_config(
        "cfg2",
        [{"step": "extract_features", "params": {"families": ["local_intensity"]}}],
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_local.assert_called()
    assert "local_peak" in result["cfg1"]


@patch("pictologics.pipeline.discretise_image")
@patch("pictologics.pipeline.calculate_all_texture_matrices")
@patch("pictologics.pipeline.calculate_glcm_features")
def test_extract_texture_glcm_via_dedup(
    mock_glcm: MagicMock,
    mock_matrices: MagicMock,
    mock_disc: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test texture_glcm via deduplication path."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_disc.return_value = mock_image
    mock_matrices.return_value = {
        "glcm": np.zeros((32, 32, 13)),
        "glrlm": np.zeros((32, 10, 13)),
        "glszm": np.zeros((32, 10)),
        "gldzm": np.zeros((32, 10)),
        "ngtdm_s": np.zeros(32),
        "ngtdm_n": np.zeros(32),
        "ngldm": np.zeros((32, 10)),
    }
    mock_glcm.return_value = {"glcm_energy": 0.5}

    pipeline.add_config(
        "cfg1",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture_glcm"]}},
        ],
    )
    pipeline.add_config(
        "cfg2",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture_glcm"]}},
        ],
    )
    result = pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_glcm.assert_called()
    assert "glcm_energy" in result["cfg1"]


@patch("pictologics.pipeline.discretise_image")
@patch("pictologics.pipeline.calculate_all_texture_matrices")
@patch("pictologics.pipeline.calculate_ngldm_features")
def test_extract_texture_with_ngldm_alpha(
    mock_ngldm: MagicMock,
    mock_matrices: MagicMock,
    mock_disc: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test texture extraction with ngldm_alpha parameter via dedup."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_disc.return_value = mock_image
    mock_matrices.return_value = {
        "glcm": np.zeros((32, 32, 13)),
        "glrlm": np.zeros((32, 10, 13)),
        "glszm": np.zeros((32, 10)),
        "gldzm": np.zeros((32, 10)),
        "ngtdm_s": np.zeros(32),
        "ngtdm_n": np.zeros(32),
        "ngldm": np.zeros((32, 10)),
    }
    mock_ngldm.return_value = {"ngldm_lde": 0.6}

    pipeline.add_config(
        "cfg1",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {
                "step": "extract_features",
                "params": {
                    "families": ["texture_ngldm"],
                    "texture_matrix_params": {"ngldm_alpha": 0.5},
                },
            },
        ],
    )
    pipeline.add_config(
        "cfg2",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {
                "step": "extract_features",
                "params": {
                    "families": ["texture_ngldm"],
                    "texture_matrix_params": {"ngldm_alpha": 0.5},
                },
            },
        ],
    )
    pipeline.run(mock_image, mock_mask, config_names=["cfg1", "cfg2"])

    mock_ngldm.assert_called()
    # Verify ngldm_alpha was passed through
    call_kwargs = mock_matrices.call_args.kwargs
    assert call_kwargs.get("ngldm_alpha") == 0.5


# --- IVH Feature Edge Cases ---


@patch("pictologics.pipeline.calculate_ivh_features")
def test_ivh_with_ivh_params_bin_width(
    mock_ivh: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test IVH features with explicit bin_width in ivh_params."""
    mock_ivh.return_value = {"ivh_v10": 0.5}

    pipeline.add_config(
        "ivh_params_test",
        [
            {
                "step": "extract_features",
                "params": {
                    "families": ["ivh"],
                    "ivh_params": {"bin_width": 5.0, "min_val": 0.0, "max_val": 100.0},
                },
            }
        ],
    )
    pipeline.run(mock_image, mock_mask, config_names=["ivh_params_test"])

    mock_ivh.assert_called()
    call_kwargs = mock_ivh.call_args.kwargs
    assert call_kwargs.get("bin_width") == 5.0
    assert call_kwargs.get("min_val") == 0.0
    assert call_kwargs.get("max_val") == 100.0


@patch("pictologics.pipeline.calculate_ivh_features")
def test_ivh_with_ivh_params_target_range(
    mock_ivh: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test IVH features with target_range parameters."""
    mock_ivh.return_value = {"ivh_v10": 0.5}

    pipeline.add_config(
        "ivh_target_range",
        [
            {
                "step": "extract_features",
                "params": {
                    "families": ["ivh"],
                    "ivh_params": {"target_range_min": 10.0, "target_range_max": 90.0},
                },
            }
        ],
    )
    pipeline.run(mock_image, mock_mask, config_names=["ivh_target_range"])

    mock_ivh.assert_called()
    call_kwargs = mock_ivh.call_args.kwargs
    assert call_kwargs.get("target_range_min") == 10.0
    assert call_kwargs.get("target_range_max") == 90.0


@patch("pictologics.pipeline.calculate_ivh_features")
@patch("pictologics.pipeline.discretise_image")
def test_ivh_discretisation_with_bin_width_in_params(
    mock_discretise: MagicMock,
    mock_ivh: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test IVH features with ivh_discretisation that has bin_width."""
    mock_discretise.return_value = mock_image  # Return same image
    mock_ivh.return_value = {"ivh_v10": 0.5}

    pipeline.add_config(
        "ivh_disc_bw",
        [
            {
                "step": "extract_features",
                "params": {
                    "families": ["ivh"],
                    "ivh_discretisation": {
                        "method": "FBS",
                        "bin_width": 2.5,
                        "min_val": -50.0,
                    },
                },
            }
        ],
    )
    pipeline.run(mock_image, mock_mask, config_names=["ivh_disc_bw"])

    mock_discretise.assert_called()
    mock_ivh.assert_called()
    # bin_width and min_val should be passed to IVH
    call_kwargs = mock_ivh.call_args.kwargs
    assert call_kwargs.get("bin_width") == 2.5
    assert call_kwargs.get("min_val") == -50.0


# --- Serialization with Deduplication Plan Tests ---


def test_to_dict_with_deduplication_plan(pipeline: RadiomicsPipeline) -> None:
    """Test to_dict includes deduplication plan when available."""
    from pictologics.deduplication import ConfigurationAnalyzer

    # Add configs and compute plan
    pipeline.add_config(
        "cfg1",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ],
    )
    pipeline.add_config(
        "cfg2",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 64}},
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ],
    )

    # Compute and store plan
    analyzer = ConfigurationAnalyzer(pipeline._configs, pipeline._deduplication_rules)
    plan = analyzer.analyze()
    pipeline._last_deduplication_plan = plan
    pipeline._configs_modified_since_plan = False

    # Export with deduplication info
    data = pipeline.to_dict(config_names=["cfg1", "cfg2"], include_deduplication=True)

    assert "deduplication" in data
    assert "last_plan" in data["deduplication"]
    assert data["deduplication"]["enabled"] == pipeline._deduplication_enabled


def test_to_dict_deduplication_plan_stale(pipeline: RadiomicsPipeline) -> None:
    """Test to_dict excludes stale deduplication plan."""
    from pictologics.deduplication import ConfigurationAnalyzer

    # Add config and compute plan
    pipeline.add_config(
        "cfg1", [{"step": "extract_features", "params": {"families": ["morphology"]}}]
    )

    analyzer = ConfigurationAnalyzer(pipeline._configs, pipeline._deduplication_rules)
    plan = analyzer.analyze()
    pipeline._last_deduplication_plan = plan
    # Mark as stale
    pipeline._configs_modified_since_plan = True

    data = pipeline.to_dict(config_names=["cfg1"], include_deduplication=True)

    assert "deduplication" in data
    # Should not include stale plan
    assert "last_plan" not in data["deduplication"]


def test_from_dict_restores_deduplication_plan() -> None:
    """Test from_dict restores deduplication plan."""
    from pictologics.deduplication import ConfigurationAnalyzer

    # Create pipeline and add configs
    pipeline = RadiomicsPipeline()
    pipeline.add_config(
        "cfg1",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ],
    )

    # Compute and store plan
    analyzer = ConfigurationAnalyzer(pipeline._configs, pipeline._deduplication_rules)
    plan = analyzer.analyze()
    pipeline._last_deduplication_plan = plan
    pipeline._configs_modified_since_plan = False

    # Export and reimport
    data = pipeline.to_dict(config_names=["cfg1"], include_deduplication=True)
    restored = RadiomicsPipeline.from_dict(data)

    # Plan should be restored
    assert restored._last_deduplication_plan is not None
    assert restored._configs_modified_since_plan is False


def test_from_dict_handles_invalid_deduplication_plan() -> None:
    """Test from_dict handles invalid deduplication plan gracefully."""
    import warnings

    data = {
        "schema_version": "1.0",
        "configs": {
            "test": {
                "steps": [
                    {"step": "extract_features", "params": {"families": ["morphology"]}}
                ]
            }
        },
        "deduplication": {
            "enabled": True,
            "tolerance": 1e-9,
            "rules_version": "1.0.0",
            "last_plan": {"invalid": "plan_data"},  # Invalid format
        },
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipeline = RadiomicsPipeline.from_dict(data)

        # Should still work, just without restored plan
        assert pipeline._last_deduplication_plan is None
        assert len(w) >= 1
        assert any(
            "failed to restore deduplication plan" in str(warning.message).lower()
            for warning in w
        )
