from __future__ import annotations

# ruff: noqa: E402
import warnings

# Suppress "NumPy module was reloaded" warning
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

import copy
import json
import os

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

from typing import Any
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

from pictologics.loader import Image
from pictologics.pipeline import EmptyROIMaskError, PipelineState, RadiomicsPipeline

# --- Fixtures ---


@pytest.fixture(autouse=True)
def _region_paths_for_small_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ROI box cut and the filter region also for the small test images (see
    _CUT_MIN_SIZE and _FILTER_REGION_MIN)."""
    monkeypatch.setattr("pictologics.pipeline._CUT_MIN_SIZE", 0)
    monkeypatch.setattr("pictologics.pipeline._FILTER_REGION_MIN", 0)


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
    image_call, mask_call = mock_load.call_args_list
    assert image_call.args == ("img.nii",)
    assert image_call.kwargs == {}
    assert mask_call.args == ("mask.nii",)
    assert mask_call.kwargs["reference_image"] is mock_image

    # 1b. pathlib.Path image and mask load as their strings
    mock_load.reset_mock()
    mock_load.side_effect = [mock_image, mock_mask]
    pipeline.run(Path("img.nii"), Path("mask.nii"), config_names=["t1"])
    assert [c.args for c in mock_load.call_args_list] == [("img.nii",), ("mask.nii",)]
    assert pipeline._log[-1]["image_source"] == "img.nii"
    assert pipeline._log[-1]["mask_source"] == "mask.nii"

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


def test_run_rejects_mask_origin_mismatch(
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config("t1", [])
    shifted_mask = Image(
        array=mock_mask.array.copy(),
        spacing=mock_mask.spacing,
        origin=(100.0, 0.0, 0.0),
        direction=mock_mask.direction,
        modality=mock_mask.modality,
    )

    with pytest.raises(ValueError, match="Origin mismatch"):
        pipeline.run(mock_image, shifted_mask, config_names=["t1"])


def test_run_rejects_mask_direction_mismatch(
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config("t1", [])
    rotated = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_mask = Image(
        array=mock_mask.array.copy(),
        spacing=mock_mask.spacing,
        origin=mock_mask.origin,
        direction=rotated,
        modality=mock_mask.modality,
    )

    with pytest.raises(ValueError, match="Direction mismatch"):
        pipeline.run(mock_image, rotated_mask, config_names=["t1"])


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


# The mocked preprocessing leaves the image without bins, so the texture family fails
@pytest.mark.filterwarnings("ignore:The texture features failed:UserWarning")
def test_run_all_standard_params(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    # Test "all_standard" expansion without patching get_all_standard_config_names
    # This ensures coverage hits the real method call line.
    # We mock execution steps to avoid heavy computation. Several configurations take the
    # deduplicated path, so it is mocked too (else the families run on an image without bins).
    with (
        patch.object(pipeline, "_execute_preprocessing_step") as mock_exec,
        patch.object(pipeline, "_extract_features") as mock_ext,
        patch.object(pipeline, "_extract_features_with_dedup", return_value={}),
    ):
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

    with (
        patch.object(pipeline, "_execute_preprocessing_step") as _mock_exec,
        patch.object(pipeline, "_extract_features") as mock_ext,
        patch.object(pipeline, "_extract_features_with_dedup", return_value={}),
    ):
        mock_ext.return_value = {}
        # Call without config_names: it runs every config, and it warns because the
        # standard ones run too
        with pytest.warns(UserWarning, match="also the 6 standard ones") as caught:
            res = pipeline.run(mock_image, mock_mask)
        assert caught[0].filename == __file__  # the warning points to the call of run()

        # Should contain all keys present in pipeline._configs
        assert len(res) == len(pipeline._configs)
        assert len(res) >= 6  # At least standard ones

        # Without standard configs there is no warning
        own = RadiomicsPipeline(load_standard=False)
        own.add_config(
            "mine", [{"step": "extract_features", "params": {"families": ["intensity"]}}]
        )
        with patch.object(own, "_extract_features", return_value={}), warnings.catch_warnings():
            warnings.simplefilter("error")
            assert list(own.run(mock_image, mock_mask)) == ["mine"]


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
    # morph_mask and intensity_mask start in sync (they alias the same array), and
    # resample has no apply_to, so the pipeline resamples the mask once and shares the
    # result for both -> resample is called twice (image + one shared mask call).
    # (side_effect keeps a third entry for the diverged-mask fallback path.)
    mock_resample.side_effect = [mock_image, mock_mask, mock_mask]

    pipeline.run(mock_image, mock_mask, config_names=["res"])

    # Called for the image and one shared mask (morph_mask/intensity_mask were in sync)
    assert mock_resample.call_count == 2
    # Check image call args using ANY for image object to avoid array comparison ambiguity
    mock_resample.assert_any_call(
        ANY,
        (2, 2, 2),
        interpolation="bspline",
        round_intensities=False,
        source_mask=None,
        region=None,  # the ROI box covers the whole grid
    )


@patch("pictologics.pipeline.resample_image")
def test_step_resample_diverged_masks_resamples_independently(
    mock_resample: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    # A morph-only binarize step diverges morph_mask from intensity_mask so they no
    # longer share an array; the following resample must therefore resample each mask
    # independently (image + morph + intensity = 3 calls).
    pipeline.add_config(
        "res_div",
        [
            {"step": "binarize_mask", "params": {"apply_to": "morph", "threshold": 0.5}},
            {
                "step": "resample",
                "params": {"new_spacing": (2, 2, 2), "interpolation": "linear"},
            },
        ],
    )
    mock_resample.side_effect = [mock_image, mock_mask, mock_mask]

    pipeline.run(mock_image, mock_mask, config_names=["res_div"])

    # Image, morph_mask, and intensity_mask were each resampled separately.
    assert mock_resample.call_count == 3


@patch("pictologics.pipeline._intersect_mask")
@patch("pictologics.pipeline.resample_image")
def test_step_resample_roi_only_diverged_masks_intersect_independently(
    mock_resample: MagicMock,
    mock_intersect: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    # With a source_mask active (source_mode="roi_only") AND the masks diverged before
    # resample, the source-mask intersection must be applied to morph_mask and
    # intensity_mask independently (two _intersect_mask calls).
    pipeline.add_config(
        "res_roi_div",
        [
            {"step": "binarize_mask", "params": {"apply_to": "morph", "threshold": 0.5}},
            {
                "step": "resample",
                "params": {"new_spacing": (2, 2, 2), "interpolation": "linear"},
            },
        ],
        source_mode="roi_only",
    )

    # Distinct mask objects so morph_mask and intensity_mask stay diverged after resample.
    morph_res = Image(
        array=np.ones((10, 10, 10), dtype=np.uint8),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="mask",
    )
    intensity_res = Image(
        array=np.ones((10, 10, 10), dtype=np.uint8),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="mask",
    )
    mock_resample.side_effect = [mock_image, morph_res, intensity_res] * 2
    mock_intersect.return_value = mock_mask

    # An ROI of all ones gives a source mask with no invalid voxel, which changes no mask.
    pipeline.run(mock_image, mock_mask, config_names=["res_roi_div"])
    assert mock_intersect.call_count == 0

    # With voxels outside the ROI, morph_mask and intensity_mask were each intersected
    # with the source mask separately.
    roi = Image(
        array=np.ones((10, 10, 10), dtype=np.uint8),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="mask",
    )
    roi.array[0] = 0
    pipeline.run(mock_image, roi, config_names=["res_roi_div"])
    assert mock_intersect.call_count == 2


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

    # The two masks start as one array, so one call updates both.
    pipeline.run(mock_image, mock_mask, config_names=["reseg"])
    assert mock_reseg.call_count == 1

    # Test generated mask case: default resegmentation updates both masks too.
    mock_reseg.reset_mock()
    pipeline.run(mock_image, mask=None, config_names=["reseg"])
    assert mock_reseg.call_count == 1


@pytest.mark.parametrize("deduplicate", [False, True])
def test_resegment_updates_morphology_mask_for_compartments(deduplicate: bool) -> None:
    """Morphology features should describe the resegmented compartment mask."""
    array = np.full((8, 8, 8), 100.0)
    array[:2, :, :] = -50.0
    image = Image(
        array=array,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="CT",
    )
    mask = Image(
        array=np.ones_like(array, dtype=np.uint8),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="mask",
    )
    pipeline = RadiomicsPipeline(load_standard=False, deduplicate=deduplicate)
    pipeline.add_config(
        "low",
        [
            {"step": "resegment", "params": {"range_min": -100.0, "range_max": 30.0}},
            {"step": "extract_features", "params": {"families": ["morphology"]}},
        ],
    )
    pipeline.add_config(
        "high",
        [
            {"step": "resegment", "params": {"range_min": 30.0, "range_max": 150.0}},
            {"step": "extract_features", "params": {"families": ["morphology"]}},
        ],
    )

    results = pipeline.run(image, mask, config_names=["low", "high"])

    assert results["low"]["volume_voxel_counting_YEKZ"] == pytest.approx(128.0)
    assert results["high"]["volume_voxel_counting_YEKZ"] == pytest.approx(384.0)
    assert results["low"]["volume_RNU0"] != pytest.approx(results["high"]["volume_RNU0"])


@patch("pictologics.pipeline.filter_outliers")
def test_step_filter_outliers(
    mock_filt: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config("filt", [{"step": "filter_outliers", "params": {"sigma": 2.0}}])
    mock_filt.return_value = mock_mask

    # 1. Normal case: one call, as the two masks start as one array
    pipeline.run(mock_image, mock_mask, config_names=["filt"])
    assert mock_filt.call_count == 1
    mock_filt.assert_called_with(ANY, ANY, 2.0)

    # 2. Generated mask case -> covers 'if state.mask_was_generated'
    mock_filt.reset_mock()
    # Need load_image to return something if no mask provided?
    # Or just use mock_image directly if we pass it.
    pipeline.run(mock_image, mask=None, config_names=["filt"])
    # One call: the generated masks are one array too
    assert mock_filt.call_count == 1


def test_filter_outliers_updates_morphology_mask_by_default() -> None:
    """Outlier filtering should narrow compartment morphology unless targeted."""
    array = np.zeros((8, 8, 8), dtype=float)
    array[:1, :, :] = 1000.0
    image = Image(
        array=array,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="CT",
    )
    mask = Image(
        array=np.ones_like(array, dtype=np.uint8),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="mask",
    )
    pipeline = RadiomicsPipeline(load_standard=False, deduplicate=False)
    pipeline.add_config(
        "filtered",
        [
            {"step": "filter_outliers", "params": {"sigma": 0.5}},
            {"step": "extract_features", "params": {"families": ["morphology"]}},
        ],
    )
    pipeline.add_config(
        "intensity_only",
        [
            {
                "step": "filter_outliers",
                "params": {"sigma": 0.5, "apply_to": "intensity"},
            },
            {"step": "extract_features", "params": {"families": ["morphology"]}},
        ],
    )

    results = pipeline.run(image, mask, config_names=["filtered", "intensity_only"])

    assert results["filtered"]["volume_voxel_counting_YEKZ"] == pytest.approx(448.0)
    assert results["intensity_only"]["volume_voxel_counting_YEKZ"] == pytest.approx(512.0)


def test_source_mask_after_resample_limits_morphology_mask() -> None:
    """AUTO source masks should protect morphology masks from resampled sentinels."""
    array = np.full((8, 8, 8), -2048.0)
    array[:4, :, :] = 50.0
    image = Image(
        array=array,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="CT",
    )
    pipeline = RadiomicsPipeline(load_standard=False, deduplicate=False)
    pipeline.add_config(
        "auto_source",
        [
            {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
            {"step": "extract_features", "params": {"families": ["morphology"]}},
        ],
        source_mode="auto",
    )

    # AUTO detection of the -2048 fill emits a UserWarning; that is expected here.
    with pytest.warns(UserWarning, match="Auto-detected sentinel value"):
        result = pipeline.run(image, mask=None, config_names=["auto_source"])["auto_source"]

    assert result["volume_voxel_counting_YEKZ"] == pytest.approx(256.0)


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

    # One call for morph_mask and intensity_mask, which start as one array
    assert mock_klc.call_count == 1


def test_step_binarize_mask(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    """Test binarize_mask preprocessing step with various parameter modes."""
    # Create a mask with varying values
    varied_array = np.tile(np.arange(10, dtype=np.uint8), (10, 10, 1))
    varied_mask = Image(
        array=varied_array,
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


def test_pipeline_nonzero_mask_labels_are_roi_membership() -> None:
    image_array = np.zeros((5, 5, 5), dtype=float)
    label_array = np.zeros_like(image_array, dtype=np.uint8)
    label_array[1:3, 1:3, 1:3] = 2
    image = Image(
        image_array,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="CT",
    )
    mask = Image(
        label_array,
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
        modality="mask",
    )
    pipeline = RadiomicsPipeline(load_standard=False, deduplicate=False)
    pipeline.add_config(
        "label_mask",
        [{"step": "extract_features", "params": {"families": ["morphology"]}}],
    )

    results = pipeline.run(image, mask, config_names=["label_mask"])

    assert results["label_mask"]["volume_voxel_counting_YEKZ"] == pytest.approx(8.0)
    assert pipeline._log[-1]["mask_roi_semantics"] == "nonzero_values_are_roi_membership"
    assert pipeline._log[-1]["config_snapshot"]["steps"] == pipeline._configs["label_mask"]
    assert pipeline._log[-1]["status"] == "completed"


def test_binarize_mask_selects_labels_before_morphology() -> None:
    image_array = np.zeros((6, 6, 6), dtype=float)
    label_array = np.zeros_like(image_array, dtype=np.uint8)
    label_array[1:3, 1:3, 1:3] = 1
    label_array[3:5, 3:5, 3:5] = 2
    image = Image(
        image_array,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="CT",
    )
    mask = Image(
        label_array,
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
        modality="mask",
    )
    pipeline = RadiomicsPipeline(load_standard=False, deduplicate=False)
    pipeline.add_config(
        "label_2",
        [
            {"step": "binarize_mask", "params": {"mask_values": 2}},
            {"step": "extract_features", "params": {"families": ["morphology"]}},
        ],
    )

    results = pipeline.run(image, mask, config_names=["label_2"])

    assert results["label_2"]["volume_voxel_counting_YEKZ"] == pytest.approx(8.0)


@patch("pictologics.pipeline.discretise_image")
def test_step_discretise(
    mock_disc: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config("disc", [{"step": "discretise", "params": {"method": "FBN", "n_bins": 16}}])
    mock_disc.return_value = mock_image

    pipeline.run(mock_image, mock_mask, config_names=["disc"])

    mock_disc.assert_called_with(mock_image, method="FBN", roi_mask=ANY, n_bins=16)


@patch("pictologics.pipeline.roi_min_max")
@patch("pictologics.pipeline.discretise_image")
def test_step_discretise_fbs_empty_error(
    mock_disc: MagicMock,
    mock_apply: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    # FBS attempts to calc n_bins from data. If data empty -> EmptyROIMaskError
    # is caught per-config and a NaN series is returned instead of raising.
    pipeline.add_config(
        "fbs", [{"step": "discretise", "params": {"method": "FBS", "bin_width": 10, "min_val": 0}}]
    )
    mock_apply.return_value = None  # No ROI voxel
    mock_disc.return_value = mock_image

    results = pipeline.run(mock_image, mock_mask, config_names=["fbs"])
    # Config has no extract_features step, so NaN series is empty
    assert "fbs" in results
    assert len(results["fbs"]) == 0

    log = pipeline._log[-1]
    assert "error" in log
    assert "ROI is empty" in log["error"]


def test_step_unknown(pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image) -> None:
    pipeline.add_config("unknown", [{"step": "fake_step"}], validate=False)
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
    pipeline.add_config("m", [{"step": "extract_features", "params": {"families": ["morphology"]}}])
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
    pipeline.add_config("i", [{"step": "extract_features", "params": {"families": ["intensity"]}}])
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
    pipeline.add_config("h", [{"step": "extract_features", "params": {"families": ["histogram"]}}])

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
    with (
        patch("pictologics.pipeline.calculate_ivh_features") as mock_ivh,
        patch("pictologics.pipeline.apply_mask") as mock_apply,
    ):
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
@patch("pictologics.pipeline._texture_matrices")
def test_extract_texture_error_no_discretise(
    mock_matrices: MagicMock,
    mock_glcm: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config(
        "tex_fail",
        [{"step": "extract_features", "params": {"families": ["texture"]}}],
        validate=False,
    )

    # The texture family fails alone: its features are NaN, the log entry lists it in
    # family_errors, and a warning names it.
    with pytest.warns(UserWarning, match="The texture features failed"):
        pipeline.run(mock_image, mock_mask, config_names=["tex_fail"])
    log = pipeline._log[-1]
    assert log["status"] == "completed"
    error = log["family_errors"]["texture"]
    assert "Texture features requested but image is not discretised" in error
    assert "You must include a 'discretise' step" in error


@patch("pictologics.pipeline.discretise_image")
@patch("pictologics.pipeline._texture_matrices")
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
        "glszm_cells": 1,
        "roi": np.ones((1, 1, 1), dtype=bool),
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
    with (
        patch("pictologics.pipeline.calculate_glrlm_features") as mr,
        patch("pictologics.pipeline._glszm_features_from_cells") as ms,
        patch("pictologics.pipeline.calculate_gldzm_features") as md,
        patch("pictologics.pipeline.calculate_ngtdm_features") as mt,
        patch("pictologics.pipeline.calculate_ngldm_features") as mn,
    ):
        mr.return_value = {}
        ms.return_value = {}
        md.return_value = {}
        mt.return_value = {}
        mn.return_value = {}

        pipeline.run(mock_image, mock_mask, config_names=["tex_ok"])

        # Verified calls
        mock_matrices.assert_called()
        mock_glcm.assert_called()


def test_texture_family_aliases_match_single_and_dedup_paths() -> None:
    """Raw texture subfamilies and texture_* aliases compute the same features."""
    from pictologics.features import FEATURE_NAMES

    image = Image(
        array=np.arange(125, dtype=float).reshape((5, 5, 5)),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="CT",
    )
    mask = Image(
        array=np.ones((5, 5, 5), dtype=np.uint8),
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
        modality="mask",
    )
    config = [
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
        {"step": "extract_features", "params": {"families": ["glcm"]}},
    ]
    alias_config = [
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
        {"step": "extract_features", "params": {"families": ["texture_glcm"]}},
    ]

    for family, steps in {"glcm": config, "texture_glcm": alias_config}.items():
        pipeline = RadiomicsPipeline(load_standard=False, deduplicate=False)
        pipeline.add_config(family, steps)
        series = pipeline.run(image, mask, config_names=[family])[family]

        assert list(series.index) == list(FEATURE_NAMES["glcm"])
        assert not series.isna().any()
        assert series[FEATURE_NAMES["glcm"][0]] > 0

        described = pipeline.describe_features()
        assert described["feature_key"].tolist() == list(FEATURE_NAMES["glcm"])

    dedup_pipeline = RadiomicsPipeline(load_standard=False, deduplicate=True)
    dedup_pipeline.add_config("raw_glcm", config)
    dedup_pipeline.add_config("alias_glcm", alias_config)
    dedup = dedup_pipeline.run(image, mask, config_names=["raw_glcm", "alias_glcm"])

    no_dedup_pipeline = RadiomicsPipeline(load_standard=False, deduplicate=False)
    no_dedup_pipeline.add_config("raw_glcm", config)
    no_dedup_pipeline.add_config("alias_glcm", alias_config)
    no_dedup = no_dedup_pipeline.run(image, mask, config_names=["raw_glcm", "alias_glcm"])

    for config_name in ["raw_glcm", "alias_glcm"]:
        expected = no_dedup[config_name].reindex(FEATURE_NAMES["glcm"])
        observed = dedup[config_name].reindex(FEATURE_NAMES["glcm"])
        assert not observed.isna().any()
        np.testing.assert_allclose(observed.to_numpy(float), expected.to_numpy(float))

    assert dedup_pipeline.deduplication_stats == {
        "computed_families": 1,
        "reused_families": 1,
        "cache_hit_rate": 0.5,
    }


def test_save_log(pipeline: RadiomicsPipeline, tmp_path: Any) -> None:
    pipeline._log.append({"test": "entry"})
    p = tmp_path / "log.json"
    pipeline.save_log(str(p))
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["pipeline_schema_version"] == "1.0"
    assert data["mask_roi_semantics"] == "nonzero_values_are_roi_membership"
    assert data["entry_count"] == 1
    assert data["entries"] == [{"test": "entry"}]

    p2 = tmp_path / "log_no_ext"
    pipeline.save_log(p2)
    assert (tmp_path / "log_no_ext.json").exists()

    # A log of many batches of 4,096 pieces: the text of json.dumps(indent=4).
    pipeline._log.extend({"entry": i, "values": [i, i / 3]} for i in range(2000))
    pipeline.save_log(p)
    text = p.read_text()
    assert text == json.dumps(json.loads(text), indent=4)


def test_clear_log(pipeline: RadiomicsPipeline) -> None:
    pipeline._log.append({"a": 1})
    pipeline.clear_log()
    assert len(pipeline._log) == 0


def test_get_log_copies_the_entries_with_their_run_time() -> None:
    # One entry per configuration run, also for an empty ROI and an error, each with its
    # run time; get_log() returns a copy, so a change of it leaves the log as it is.
    image = Image(np.arange(1000.0).reshape(10, 10, 10), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    roi = np.zeros((10, 10, 10), dtype=np.uint8)
    roi[2:8, 2:8, 2:8] = 1
    mask = Image(roi, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    extract = {"step": "extract_features", "params": {"families": ["intensity"]}}
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("ok", [extract])
    pipeline.add_config(
        "empty", [{"step": "resegment", "params": {"range_min": -5, "range_max": -1}}, extract]
    )
    pipeline.add_config(
        "error",
        [
            {"step": "filter", "params": {"type": "gabor", "sigma_mm": -1.0, "lambda_mm": 2.0}},
            extract,
        ],
    )
    pipeline.run(image, mask, config_names=["ok", "empty", "error"])
    log = pipeline.get_log()
    assert [(e["config_name"], e["status"]) for e in log] == [
        ("ok", "completed"),
        ("empty", "empty_roi"),
        ("error", "error"),
    ]
    assert all(isinstance(e["elapsed_seconds"], float) and e["elapsed_seconds"] >= 0 for e in log)
    log[0]["status"] = "changed"
    log[0]["steps_executed"].clear()
    assert pipeline.get_log()[0]["status"] == "completed"
    assert pipeline.get_log()[0]["steps_executed"]


def test_log_entries_hold_the_environment_and_a_config_hash(tmp_path: Any) -> None:
    import hashlib

    import numba

    image = Image(np.arange(1000.0).reshape(10, 10, 10), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    steps = [
        {"step": "resample", "params": {"new_spacing": (2.0, 2.0, 2.0)}},
        {"step": "extract_features", "params": {"families": ["intensity"]}},
    ]
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("a", steps)
    pipeline.add_config("same", copy.deepcopy(steps))
    pipeline.add_config("auto", steps, source_mode="auto")
    with pytest.warns(UserWarning, match="No sentinel value auto-detected"):
        pipeline.run(image, None, config_names=["a", "same", "auto"])
    pipeline.run(image, None, config_names=["a"])
    log = pipeline.get_log()
    environment = log[0]["environment"]
    assert set(environment) == {
        "python", "platform", "numpy", "scipy", "numba", "PyWavelets", "nibabel",
        "pydicom", "python-gdcm", "threads",
    }  # fmt: skip
    assert environment["numpy"] == np.__version__
    assert environment["threads"] == numba.get_num_threads()
    # the canonical JSON of the snapshot without the sentinel value found in the image
    snapshot = {
        k: v for k, v in log[0]["config_snapshot"].items() if k != "effective_sentinel_value"
    }
    text = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    assert log[0]["config_hash"] == hashlib.sha256(text.encode()).hexdigest()
    hashes = [entry["config_hash"] for entry in log]
    assert hashes[0] == hashes[1] == hashes[3] != hashes[2]
    # a new configuration under the same name gets a new hash
    pipeline.add_config("same", copy.deepcopy(steps)[1:])
    pipeline.run(image, None, config_names=["same"])
    assert pipeline.get_log()[-1]["config_hash"] not in hashes
    # a configuration saved to a file and loaded again keeps its hash
    pipeline.save_configs(tmp_path / "configs.yaml")
    loaded = RadiomicsPipeline.load_configs(tmp_path / "configs.yaml", load_standard=False)
    loaded.run(image, None, config_names=["a"])
    assert loaded.get_log()[0]["config_hash"] == hashes[0]


def test_run_rois_gives_the_results_of_one_run_per_label(tmp_path: Any) -> None:
    # Each ROI of a label map gets the results of run() with a mask of its label alone,
    # also with a NaN voxel in the image and for a label map file (float64 labels).
    import nibabel as nib

    rng = np.random.default_rng(9)
    values = rng.normal(40.0, 20.0, (14, 14, 14))
    values[1, 1, 1] = np.nan
    image = Image(values, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    labels = np.zeros((14, 14, 14), dtype=np.uint8)
    labels[2:6, 2:6, 2:6] = 1
    labels[7:12, 3:9, 4:10] = 3
    labels[1:3, 9:13, 9:13] = 2
    label_map = Image(labels, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    # The NIfTI affine of the LPS+ grid of `image` (RAS+: X and Y change sign)
    nib.save(nib.Nifti1Image(labels, np.diag([-1.0, -1.0, 1.0, 1.0])), tmp_path / "labels.nii.gz")
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config(
        "c",
        [
            {"step": "resample", "params": {"new_spacing": (0.7, 0.7, 0.7)}},
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
            {
                "step": "extract_features",
                "params": {"families": ["intensity", "morphology", "texture"]},
            },
        ],
    )
    with pytest.warns(UserWarning, match="NaN or infinite"):
        one_by_one = {
            str(k): pipeline.run(
                image,
                Image((labels == k).astype(np.uint8), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
                config_names="c",
            )
            for k in (1, 2, 3)
        }
        pipeline.clear_log()
        rois = pipeline.run_rois(image, label_map, subject_id="s", config_names="c")
        from_file = pipeline.run_rois(
            Image(values, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
            str(tmp_path / "labels.nii.gz"),
            labels={"third": 3, "absent": 7},
            config_names=["c"],
        )
    assert list(rois) == ["1", "2", "3"]
    for k in rois:
        assert rois[k]["c"].equals(one_by_one[k]["c"])
    assert from_file["third"]["c"].equals(one_by_one["3"]["c"])
    assert from_file["absent"]["c"].isna().all()
    log = pipeline.get_log()
    assert [(e["roi"], e["subject_id"], e["status"]) for e in log] == [
        ("1", "s", "completed"),
        ("2", "s", "completed"),
        ("3", "s", "completed"),
        ("third", None, "completed"),
        ("absent", None, "empty_roi"),
    ]
    assert log[3]["mask_source"] == str(tmp_path / "labels.nii.gz")
    # label 2 is far from the NaN voxel, so its run gives no warning
    assert pipeline.run_rois(image, label_map, labels=[2], config_names=["c"]).keys() == {"2"}


def test_run_rois_checks_the_labels() -> None:
    image = Image(np.zeros((4, 4, 4)), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    pipeline = RadiomicsPipeline(load_standard=False)
    for values in (np.full((4, 4, 4), 1.5), np.full((4, 4, 4), -1.0)):
        with pytest.raises(ValueError, match="whole numbers of 0 or above"):
            pipeline.run_rois(image, Image(values, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)))
    label_map = Image(np.ones((4, 4, 4), dtype=np.uint8), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    for labels in ([0], [2.5]):
        with pytest.raises(ValueError, match="whole number of 1 or above"):
            pipeline.run_rois(image, label_map, labels=labels)


def _batch_cases(folder: Any) -> list[dict[str, Any]]:
    """Two cases on NIfTI files, and a case whose image file is missing."""
    import nibabel as nib

    rng = np.random.default_rng(8)
    cases = []
    for k in range(2):
        roi = np.zeros((12, 12, 12), dtype=np.uint8)
        roi[3:9, 3:9, 3:9] = 1
        nib.save(
            nib.Nifti1Image(rng.normal(40.0, 20.0, (12, 12, 12)), np.eye(4)),
            folder / f"image{k}.nii.gz",
        )
        nib.save(nib.Nifti1Image(roi, np.eye(4)), folder / f"mask{k}.nii.gz")
        cases.append(
            {
                "subject_id": f"p/{k}",
                "image": str(folder / f"image{k}.nii.gz"),
                "mask": str(folder / f"mask{k}.nii.gz"),
            }
        )
    cases.append({"subject_id": "missing", "image": str(folder / "none.nii.gz")})
    return cases


def _batch_test_pipeline() -> RadiomicsPipeline:
    extract = {"step": "extract_features", "params": {"families": ["intensity"]}}
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("first", [extract])
    pipeline.add_config(
        "empty",
        [{"step": "resegment", "params": {"range_min": 1000, "range_max": 2000}}, extract],
        source_mode="auto",
    )
    return pipeline


def test_run_batch_writes_a_file_per_case_and_resumes(tmp_path: Any) -> None:
    import pandas as pd

    cases = _batch_cases(tmp_path)
    pipeline = _batch_test_pipeline()
    out = tmp_path / "out"
    table = pipeline.run_batch(cases, out, config_names=["first"], show_progress=False)
    assert list(table.columns[:5]) == ["subject_id", "status", "error", "warnings", "seconds"]
    assert table["subject_id"].tolist() == ["p/0", "p/1", "missing"]
    assert table["status"].tolist() == ["completed", "completed", "failed"]
    assert table["error"][2].startswith("ValueError: The specified path does not exist")
    assert pipeline.get_log() == []  # the cases run in another pipeline
    direct = pipeline.run(cases[0]["image"], cases[0]["mask"], config_names=["first"])
    assert table.loc[0, "first__mean_intensity_Q4LE"] == direct["first"]["mean_intensity_Q4LE"]
    assert sorted(p.name for p in (out / "cases").iterdir()) == [
        "missing.json",
        "p_0.json",
        "p_1.json",
    ]
    record = json.loads((out / "cases" / "p_0.json").read_text())
    assert record["config_hashes"] == {"first": pipeline.get_log()[0]["config_hash"]}
    assert record["image"] == cases[0]["image"] and record["log"][0]["status"] == "completed"

    # the next call runs only the failed case again
    run_case = RadiomicsPipeline._run_case
    with patch.object(RadiomicsPipeline, "_run_case", autospec=True, side_effect=run_case) as calls:
        again = pipeline.run_batch(cases, out, config_names=["first"], show_progress=False)
        assert [call.args[1]["subject_id"] for call in calls.call_args_list] == ["missing"]
        pd.testing.assert_frame_equal(again.drop(columns="seconds"), table.drop(columns="seconds"))
        # another mask path or other configurations run a case again
        cases[1]["mask"] = cases[0]["mask"]
        pipeline.run_batch(cases, out, config_names=["first"], show_progress=False)
        assert [call.args[1]["subject_id"] for call in calls.call_args_list[1:]] == [
            "p/1",
            "missing",
        ]
        calls.reset_mock()
        both = pipeline.run_batch(
            pd.DataFrame(cases), out, config_names=["first", "empty"], show_progress=False
        )
        assert len(calls.call_args_list) == 3
    # the two cases from their files: a NaN feature (null in the file) stays a float NaN,
    # also in a column of NaN only
    resumed = pipeline.run_batch(
        pd.DataFrame(cases[:2]), out, config_names=["first", "empty"], show_progress=False
    )
    expected = both.drop(columns="seconds").iloc[:2]
    pd.testing.assert_frame_equal(resumed.drop(columns="seconds"), expected)
    assert resumed["empty__mean_intensity_Q4LE"].dtype == np.float64
    # an empty ROI in one configuration; a warning of a run
    assert both["status"].tolist() == ["incomplete", "incomplete", "failed"]
    assert both["error"][0].startswith("empty: ROI is empty after preprocessing (resegment)")
    assert "No sentinel value auto-detected" in both["warnings"][0]


def test_run_batch_checks_the_cases(tmp_path: Any) -> None:
    pipeline = _batch_test_pipeline()
    for cases, message in (
        (
            [{"subject_id": "a", "image": "x", "msk": "y"}],
            "Case 0: unknown key 'msk' \\(did you mean 'mask'\\?\\)",
        ),
        ([{"subject_id": "a"}], "Case 0 needs a subject_id and an image"),
        (
            [{"subject_id": "a/b", "image": "x"}, {"subject_id": "a_b", "image": "y"}],
            "The cases 'a/b' and 'a_b' have the same result file name, a_b.json",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            pipeline.run_batch(cases, tmp_path, show_progress=False)


def test_run_batch_with_worker_processes(tmp_path: Any) -> None:
    # Spawned workers give the results of the cases in this process.
    import pandas as pd

    cases = _batch_cases(tmp_path)
    pipeline = _batch_test_pipeline()
    serial = pipeline.run_batch(
        cases, tmp_path / "serial", config_names=["first"], show_progress=False
    )
    parallel = pipeline.run_batch(
        cases, tmp_path / "parallel", config_names=["first"], workers=2, show_progress=False
    )
    pd.testing.assert_frame_equal(serial.drop(columns="seconds"), parallel.drop(columns="seconds"))


def test_run_batch_worker_functions(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # The start and the case function of a worker (here in this process).
    import numba

    from pictologics import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "_BATCH_PIPELINE", None)
    pipeline = _batch_test_pipeline()
    setup = ({"first": pipeline.get_config("first")}, {}, True, pipeline.deduplication_rules)
    threads = numba.get_num_threads()
    try:
        pipeline_module._start_batch_worker(setup, 1)
        assert numba.get_num_threads() == 1
    finally:
        numba.set_num_threads(threads)
    case = _batch_cases(tmp_path)[0]
    record = pipeline_module._batch_case(case, ["first"], {"first": "x"}, tmp_path / "case.json")
    assert record["status"] == "completed"
    assert json.loads((tmp_path / "case.json").read_text())["results"] == record["results"]


def test_run_batch_stops_the_workers_on_an_interrupt(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stop (Ctrl+C) cancels the cases that have not started.
    from pictologics import pipeline as pipeline_module

    first = _batch_cases(tmp_path)[0]
    cases = [dict(first, subject_id=f"c{k}") for k in range(6)]

    def interrupted(futures: Any) -> Any:
        raise KeyboardInterrupt
        yield  # a generator, as as_completed

    monkeypatch.setattr(pipeline_module, "as_completed", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _batch_test_pipeline().run_batch(
            cases, tmp_path / "out", config_names=["first"], workers=2, show_progress=False
        )
    assert len(list((tmp_path / "out" / "cases").glob("*.json"))) < 6


def test_fbs_starts_at_min_val_or_the_resegment_lower_bound() -> None:
    # IBSI: the FBS bins of every image start at the same value. An FBS step without
    # min_val starts at the largest lower bound of the resegment steps of the intensity
    # mask; an FBS IVH discretisation does the same.
    rng = np.random.default_rng(3)
    values = rng.normal(40.0, 30.0, (12, 12, 12))
    image = Image(values, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    mask = Image(np.ones((12, 12, 12), dtype=np.uint8), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    roi_min = float(values[(values >= -20) & (values <= 200)].min())
    assert roi_min > -19.9  # so the two starts give other bins

    def resegment(low: float, apply_to: str = "both") -> dict[str, Any]:
        return {
            "step": "resegment",
            "params": {"range_min": low, "range_max": 200, "apply_to": apply_to},
        }

    def fbs(**start: float) -> dict[str, Any]:
        return {"step": "discretise", "params": {"method": "FBS", "bin_width": 10.0, **start}}

    histogram = {"step": "extract_features", "params": {"families": ["histogram"]}}
    ivh = {"bin_width": 2.5, "method": "FBS"}
    configs = {
        "rule": [resegment(-20), fbs(), histogram],
        "given": [resegment(-20), fbs(min_val=-20), histogram],
        "roi_start": [resegment(-20), fbs(min_val=roi_min), histogram],
        "largest": [resegment(-50), resegment(-20), fbs(), histogram],
        "own_start": [resegment(-20), fbs(min_val=-100), histogram],
        "ivh_rule": [
            resegment(-20),
            {
                "step": "extract_features",
                "params": {"families": ["ivh"], "ivh_discretisation": ivh},
            },
        ],
        "ivh_given": [
            resegment(-20),
            {
                "step": "extract_features",
                "params": {"families": ["ivh"], "ivh_discretisation": {**ivh, "min_val": -20}},
            },
        ],
    }
    pipeline = RadiomicsPipeline(load_standard=False, deduplicate=False)
    for name, steps in configs.items():
        pipeline.add_config(name, steps)
    results = pipeline.run(image, mask, config_names=list(configs))
    assert results["rule"].equals(results["given"])
    assert not results["rule"].equals(results["roi_start"])
    assert results["ivh_rule"].equals(results["ivh_given"])
    starts = {
        entry["config_name"]: next(
            step["min_val_effective"]
            for step in entry["steps_executed"]
            if step["step"] == "discretise"
        )
        for entry in pipeline.get_log()
        if not entry["config_name"].startswith("ivh")
    }
    assert starts == {
        "rule": -20.0,
        "given": -20.0,
        "roi_start": roi_min,
        "largest": -20.0,
        "own_start": -100.0,
    }


def test_fbs_without_a_start_stops() -> None:
    # Without min_val and without a resegment lower bound of the intensity mask (none, a
    # resegment of the morph mask alone, or a filter after it), FBS has no start that is
    # the same in every image: add_config raises, and a config that skipped the check
    # stops at its discretise step.
    filtered = [
        {"step": "resegment", "params": {"range_min": -20, "range_max": 200}},
        {"step": "filter", "params": {"type": "mean", "support": 3}},
    ]
    histogram = {"step": "extract_features", "params": {"families": ["histogram"]}}
    fbs = {"step": "discretise", "params": {"method": "FBS", "bin_width": 10.0}}
    cases = {
        "none": [fbs, histogram],
        "morph_only": [
            {
                "step": "resegment",
                "params": {"range_min": -20, "range_max": 200, "apply_to": "morph"},
            },
            fbs,
            histogram,
        ],
        "filtered": [*filtered, fbs, histogram],
        "ivh": [
            {
                "step": "extract_features",
                "params": {"families": ["ivh"], "ivh_discretisation": {"bin_width": 2.5}},
            }
        ],
    }
    pipeline = RadiomicsPipeline(load_standard=False)
    for name, steps in cases.items():
        with pytest.raises(ValueError, match="needs a start that is the same for every image"):
            pipeline.add_config(name, steps)
        pipeline.add_config(name, steps, validate=False)
    image = Image(np.arange(512.0).reshape(8, 8, 8), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    with pytest.warns(UserWarning, match="The ivh features failed"):
        results = pipeline.run(image, None, config_names=list(cases))
    for name in cases:
        assert results[name].isna().all()
    message = "FBS needs a start that is the same for every image"
    for entry in pipeline.get_log():
        if entry["config_name"] == "ivh":  # the IVH discretisation fails in its family
            assert entry["status"] == "completed"
            assert message in entry["family_errors"]["ivh"]
        else:
            assert entry["status"] == "error" and entry["error"].startswith(message)
    # a resegment after the filter gives the start again
    pipeline.add_config(
        "again",
        [
            *filtered,
            {"step": "resegment", "params": {"range_min": 0, "range_max": 900}},
            fbs,
            histogram,
        ],
    )


def test_package_versions_of_a_package_without_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib.metadata import PackageNotFoundError

    from pictologics import pipeline as pipeline_module

    def version(name: str) -> str:
        if name == "python-gdcm":
            raise PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr(pipeline_module, "version", version)
    pipeline_module._package_versions.cache_clear()
    try:
        versions = pipeline_module._package_versions()
        assert versions["python-gdcm"] is None and versions["numpy"] == "1.0"
    finally:
        pipeline_module._package_versions.cache_clear()


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
    state.intensity_mask.array = np.ones((10, 10, 10))  # Non-empty
    state.morph_mask.array = np.zeros((10, 10, 10))  # Empty

    with pytest.raises(EmptyROIMaskError, match="ROI is empty"):
        pipeline._ensure_nonempty_roi(state, "morph_empty")


def test_empty_roi_returns_nan_series(
    pipeline: RadiomicsPipeline,
    mock_image: Image,
) -> None:
    """When a config hits an empty ROI, run() returns a NaN-filled Series
    with the expected feature names instead of raising."""
    # Create a mask that is entirely empty (all zeros)
    empty_mask = Image(
        array=np.zeros_like(mock_image.array, dtype=np.uint8),
        spacing=mock_image.spacing,
        origin=mock_image.origin,
        direction=mock_image.direction,
    )
    pipeline.add_config(
        "will_fail",
        [
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ],
    )
    results = pipeline.run(mock_image, empty_mask, config_names=["will_fail"])

    assert "will_fail" in results
    series = results["will_fail"]
    # Should contain all 18 intensity feature names
    assert len(series) == 18
    assert series.isna().all()
    assert "mean_intensity_Q4LE" in series.index

    # Error is logged
    log = pipeline._log[-1]
    assert "error" in log


def test_empty_roi_continues_other_configs(
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """When one config fails with empty ROI, other configs still execute."""
    # Config that will work (empty steps = no preprocessing = no ROI check issues
    # ... unless the mask starts empty)
    pipeline.add_config("ok_config", [])
    pipeline.add_config(
        "fail_config",
        [
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ],
    )

    # Run with valid mask — ok_config has no steps so succeeds,
    # "fail_config" should also work since mask is valid.
    results = pipeline.run(mock_image, mock_mask, config_names=["ok_config", "fail_config"])
    assert "ok_config" in results
    assert "fail_config" in results


def test_empty_roi_nan_series_with_texture(
    pipeline: RadiomicsPipeline,
    mock_image: Image,
) -> None:
    """NaN series includes all texture sub-family features when 'texture' is requested."""
    empty_mask = Image(
        array=np.zeros_like(mock_image.array, dtype=np.uint8),
        spacing=mock_image.spacing,
        origin=mock_image.origin,
        direction=mock_image.direction,
    )
    pipeline.add_config(
        "tex_fail",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {"step": "extract_features", "params": {"families": ["texture"]}},
        ],
        validate=False,
    )
    results = pipeline.run(mock_image, empty_mask, config_names=["tex_fail"])
    series = results["tex_fail"]

    # texture expands to 6 sub-families: glcm(25)+glrlm(16)+glszm(16)+gldzm(16)+ngtdm(5)+ngldm(17) = 95
    assert len(series) == 25 + 16 + 16 + 16 + 5 + 17
    assert series.isna().all()
    assert "joint_maximum_GYBY" in series.index  # GLCM
    assert "coarseness_QCDE" in series.index  # NGTDM


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
                    "ivh_params": {"bin_width": 0.5},  # Should override or exist alongside?
                },
            }
        ],
    )

    with (
        patch("pictologics.pipeline.discretise_image") as mock_disc,
        patch("pictologics.pipeline.apply_mask") as mock_apply,
        patch("pictologics.pipeline.calculate_ivh_features") as mock_calc,
    ):
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

    with (
        patch("pictologics.pipeline.apply_mask") as mock_apply,
        patch("pictologics.pipeline.calculate_ivh_features") as mock_calc,
    ):
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
        validate=False,
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
            validate=False,
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
    with patch("pictologics.pipeline.calculate_spatial_intensity_features") as mock_calc:
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


# The mocked texture matrices make the texture family fail (a warning)
@pytest.mark.filterwarnings("ignore:The texture features failed:UserWarning")
def test_params_explicit_none_all(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    # Test ivh and texture matrix params explicit None
    with (
        patch("pictologics.pipeline.calculate_ivh_features") as mock_ivh,
        patch("pictologics.pipeline._texture_matrices") as mock_tex,
        patch("pictologics.pipeline.calculate_glcm_features"),
        patch("pictologics.pipeline.calculate_glrlm_features"),
        patch("pictologics.pipeline._glszm_features_from_cells"),
        patch("pictologics.pipeline.calculate_gldzm_features"),
        patch("pictologics.pipeline.calculate_ngtdm_features"),
        patch("pictologics.pipeline.calculate_ngldm_features"),
        patch("pictologics.pipeline.discretise_image", return_value=mock_image),
    ):
        mock_ivh.return_value = {}
        mock_tex.return_value = {
            "glcm": 1,
            "glrlm": 1,
            "glszm": 1,
            "glszm_cells": 1,
            "roi": 1,
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
            validate=False,
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


def test_run_subject_id(pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image) -> None:
    pipeline.add_config("subj", [])
    res = pipeline.run(mock_image, mock_mask, subject_id="P001", config_names=["subj"])
    # subject_id should NOT appear in the feature Series (it's metadata, not a feature)
    assert "subject_id" not in res["subj"].index
    # subject_id should appear in the processing log
    assert any(entry["subject_id"] == "P001" for entry in pipeline._log)


# The mocked texture matrices make the texture family fail (a warning)
@pytest.mark.filterwarnings("ignore:The texture features failed:UserWarning")
@patch("pictologics.pipeline.roi_min_max")
@patch("pictologics.pipeline.discretise_image")
def test_step_discretise_fbs_success(
    mock_disc: MagicMock,
    mock_apply: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    pipeline.add_config(
        "fbs_ok",
        [{"step": "discretise", "params": {"method": "FBS", "min_val": 0.0}}],
        validate=False,
    )
    mock_disc.return_value = mock_image
    mock_apply.return_value = (10.0, 20.0)  # the ROI range of the discretised image

    # Run should not fail
    pipeline.run(mock_image, mock_mask, config_names=["fbs_ok"])

    # How to verify n_bins?
    # We can't easily access state. But extraction might use it.
    # Let's add extraction step
    pipeline.add_config(
        "fbs_extract",
        [
            {"step": "discretise", "params": {"method": "FBS", "min_val": 0.0}},
            {"step": "extract_features", "params": {"families": ["texture"]}},
        ],
        validate=False,
    )

    with (
        patch("pictologics.pipeline._texture_matrices") as mock_tex,
        patch("pictologics.pipeline.calculate_glcm_features", return_value={}),
        patch("pictologics.pipeline.calculate_glrlm_features", return_value={}),
        patch("pictologics.pipeline._glszm_features_from_cells", return_value={}),
        patch("pictologics.pipeline.calculate_gldzm_features", return_value={}),
        patch("pictologics.pipeline.calculate_ngtdm_features", return_value={}),
        patch("pictologics.pipeline.calculate_ngldm_features", return_value={}),
    ):
        mock_tex.return_value = {
            "glcm": 1,
            "glrlm": 1,
            "glszm": 1,
            "glszm_cells": 1,
            "roi": 1,
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

    with (
        patch("pictologics.pipeline.calculate_ivh_features") as mock_ivh,
        patch("pictologics.pipeline.apply_mask", return_value=[1]),
        patch("pictologics.pipeline.discretise_image", return_value=mock_image),
    ):
        mock_ivh.return_value = {}
        pipeline.run(mock_image, mock_mask, config_names=["ivh_fbs"])

        mock_ivh.assert_called_with(ANY, bin_width=2.5, min_val=0.0)


# The mocked texture matrices make the texture family fail (a warning)
@pytest.mark.filterwarnings("ignore:The texture features failed:UserWarning")
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

    with (
        patch("pictologics.pipeline._texture_matrices") as mock_tex,
        patch("pictologics.pipeline.discretise_image", return_value=mock_image),
        patch("pictologics.pipeline.apply_mask", return_value=[1]),
        patch("pictologics.pipeline.calculate_glcm_features", return_value={}),
        patch("pictologics.pipeline.calculate_glrlm_features", return_value={}),
        patch("pictologics.pipeline._glszm_features_from_cells", return_value={}),
        patch("pictologics.pipeline.calculate_gldzm_features", return_value={}),
        patch("pictologics.pipeline.calculate_ngtdm_features", return_value={}),
        patch("pictologics.pipeline.calculate_ngldm_features", return_value={}),
    ):
        mock_tex.return_value = {
            "glcm": 1,
            "glrlm": 1,
            "glszm": 1,
            "glszm_cells": 1,
            "roi": 1,
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
        [{"step": "filter", "params": {"type": "riesz", "order": (1, 0, 0)}}],
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
        validate=False,
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
        validate=False,
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


@patch("pictologics.pipeline.mean_filter")
def test_step_filter_boundary_condition_instance(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """A BoundaryCondition instance (not a string) passed as 'boundary' must resolve
    correctly, exercising the non-string branch of boundary resolution."""
    from pictologics.filters import BoundaryCondition

    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_boundary_instance",
        [
            {
                "step": "filter",
                "params": {"type": "mean", "support": 3, "boundary": BoundaryCondition.NEAREST},
            }
        ],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_boundary_instance"])
    call_kwargs = mock_filter.call_args.kwargs
    assert call_kwargs.get("boundary") == BoundaryCondition.NEAREST


def test_step_filter_unknown_boundary_raises(
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """An unrecognised boundary string must raise (never silently fall back to mirror):
    add_config names it, and without the check the run stops the configuration."""
    steps = [{"step": "filter", "params": {"type": "mean", "support": 3, "boundary": "bogus"}}]
    with pytest.raises(ValueError, match="unknown boundary 'bogus'"):
        pipeline.add_config("filter_bad_boundary", steps)
    pipeline.add_config("filter_bad_boundary", steps, validate=False)
    pipeline.run(mock_image, mock_mask, config_names=["filter_bad_boundary"])
    log = pipeline._log[-1]
    assert "error" in log
    assert "Unknown boundary condition" in log["error"]


@patch("pictologics.pipeline.simoncelli_wavelet")
def test_step_filter_simoncelli_explicit_boundary_forwarded(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """An explicitly requested boundary must be forwarded to simoncelli_wavelet."""
    from pictologics.filters import BoundaryCondition

    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_simon_boundary",
        [{"step": "filter", "params": {"type": "simoncelli", "level": 1, "boundary": "nearest"}}],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_simon_boundary"])
    call_kwargs = mock_filter.call_args.kwargs
    assert call_kwargs.get("boundary") == BoundaryCondition.NEAREST


@patch("pictologics.pipeline.riesz_transform")
def test_step_filter_riesz_explicit_boundary_forwarded(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """An explicitly requested boundary must be forwarded to riesz_transform."""
    from pictologics.filters import BoundaryCondition

    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_riesz_boundary",
        [{"step": "filter", "params": {"type": "riesz", "order": (1, 0, 0), "boundary": "zero"}}],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_riesz_boundary"])
    call_kwargs = mock_filter.call_args.kwargs
    assert call_kwargs.get("boundary") == BoundaryCondition.ZERO


@patch("pictologics.pipeline.mean_filter")
def test_step_filter_log_boundary_metadata_spatial(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Spatial filters (mean/log/laws/gabor/wavelet) log the resolved boundary as both
    'requested' and 'effective', since they always honour it exactly."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_mean_meta",
        [{"step": "filter", "params": {"type": "mean", "support": 3}}],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_mean_meta"])
    step_log = pipeline._log[-1]["steps_executed"][0]
    assert step_log["boundary_requested"] == "mirror"
    assert step_log["boundary_effective"] == "mirror"


@patch("pictologics.pipeline.simoncelli_wavelet")
def test_step_filter_log_boundary_metadata_fft_default(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """When no boundary is requested, an FFT-based filter's logged 'requested' value is
    the pipeline default ('mirror') but 'effective' stays 'periodic' (its own default)."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_simon_meta_default",
        [{"step": "filter", "params": {"type": "simoncelli", "level": 1}}],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_simon_meta_default"])
    step_log = pipeline._log[-1]["steps_executed"][0]
    assert step_log["boundary_requested"] == "mirror"
    assert step_log["boundary_effective"] == "periodic"


@patch("pictologics.pipeline.riesz_simoncelli")
def test_step_filter_log_boundary_metadata_fft_explicit(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """When a boundary is explicitly requested, an FFT-based filter's logged
    'requested' and 'effective' values both match the requested boundary."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_riesz_simon_meta_explicit",
        [
            {
                "step": "filter",
                "params": {"type": "riesz", "variant": "simoncelli", "boundary": "nearest"},
            }
        ],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_riesz_simon_meta_explicit"])
    step_log = pipeline._log[-1]["steps_executed"][0]
    assert step_log["boundary_requested"] == "nearest"
    assert step_log["boundary_effective"] == "nearest"


# --- params_requested / params_effective provenance (IBSI 2) ---


@patch("pictologics.pipeline.mean_filter")
def test_step_filter_params_requested_matches_raw_config(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """params_requested is exactly the raw step config, unmodified (including 'type')."""
    mock_filter.return_value = mock_image.array
    raw_params = {"type": "mean", "support": 5}
    pipeline.add_config("filter_req_raw", [{"step": "filter", "params": raw_params}])
    pipeline.run(mock_image, mock_mask, config_names=["filter_req_raw"])
    step_log = pipeline._log[-1]["steps_executed"][0]
    assert step_log["params_requested"] == raw_params
    # 'boundary' was never supplied, so it must be absent (not silently invented).
    assert "boundary" not in step_log["params_requested"]


@patch("pictologics.pipeline.mean_filter")
def test_step_filter_params_effective_mean(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """params_effective drops 'type' (never a callable kwarg) and resolves 'boundary'
    to its lowercase name, matching filter_boundary_effective."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_eff_mean",
        [{"step": "filter", "params": {"type": "mean", "support": 5}}],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_eff_mean"])
    step_log = pipeline._log[-1]["steps_executed"][0]
    assert step_log["params_effective"] == {"support": 5, "boundary": "mirror"}
    assert "type" not in step_log["params_effective"]


@patch("pictologics.pipeline.laplacian_of_gaussian")
def test_step_filter_params_effective_log_spacing_injected(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """params_effective for 'log' shows the spacing_mm the pipeline injected from the
    image, as a JSON-safe list (not the raw tuple)."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_eff_log",
        [{"step": "filter", "params": {"type": "log", "sigma_mm": 1.5, "truncate": 4.0}}],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_eff_log"])
    step_log = pipeline._log[-1]["steps_executed"][0]
    assert "spacing_mm" not in step_log["params_requested"]
    assert step_log["params_effective"]["spacing_mm"] == list(mock_image.spacing)
    assert step_log["params_effective"]["sigma_mm"] == 1.5
    assert json.dumps(step_log)  # must be JSON-safe (list, not tuple)


@patch("pictologics.pipeline.gabor_filter")
def test_step_filter_params_effective_gabor_delta_theta(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """params_effective for 'gabor' carries through a caller-supplied delta_theta
    alongside the pipeline-injected spacing_mm."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_eff_gabor",
        [
            {
                "step": "filter",
                "params": {"type": "gabor", "sigma_mm": 3.0, "lambda_mm": 2.0, "delta_theta": 45},
            }
        ],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_eff_gabor"])
    step_log = pipeline._log[-1]["steps_executed"][0]
    assert step_log["params_effective"]["delta_theta"] == 45
    assert step_log["params_effective"]["spacing_mm"] == list(mock_image.spacing)


@patch("pictologics.pipeline.laws_filter")
def test_step_filter_params_effective_laws_kernel_positional(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """'kernel' is passed positionally to laws_filter (not via **kwargs), but must
    still be visible in params_effective so it can't silently disappear from the log."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_eff_laws",
        [{"step": "filter", "params": {"type": "laws", "kernel": "E5L5S5"}}],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_eff_laws"])
    call_kwargs = mock_filter.call_args.kwargs
    assert "kernel" not in call_kwargs  # confirms it really is positional, not a kwarg
    step_log = pipeline._log[-1]["steps_executed"][0]
    assert step_log["params_effective"]["kernel"] == "E5L5S5"


@patch("pictologics.pipeline.riesz_transform")
def test_step_filter_params_effective_riesz_variant_default(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """'variant' defaults to 'base' when omitted, and is still recorded in
    params_effective even though it is never forwarded to riesz_transform."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_eff_riesz_base",
        [{"step": "filter", "params": {"type": "riesz", "order": (1, 0, 0)}}],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_eff_riesz_base"])
    call_kwargs = mock_filter.call_args.kwargs
    assert "variant" not in call_kwargs
    step_log = pipeline._log[-1]["steps_executed"][0]
    assert "variant" not in step_log["params_requested"]
    assert step_log["params_effective"]["variant"] == "base"


@patch("pictologics.pipeline.riesz_log")
def test_step_filter_params_effective_riesz_variant_log(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """The riesz 'log' variant's params_effective shows both the dispatched variant
    and the pipeline-injected spacing_mm."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_eff_riesz_log",
        [{"step": "filter", "params": {"type": "riesz", "variant": "log", "sigma_mm": 2.0}}],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_eff_riesz_log"])
    step_log = pipeline._log[-1]["steps_executed"][0]
    assert step_log["params_effective"]["variant"] == "log"
    assert step_log["params_effective"]["spacing_mm"] == list(mock_image.spacing)


@patch("pictologics.pipeline.riesz_simoncelli")
def test_step_filter_params_effective_simoncelli_boundary_periodic(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """When no boundary is requested, an FFT-based filter's params_effective reports
    the true honoured boundary ('periodic'), not the unrequested pipeline default."""
    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_eff_riesz_simon",
        [{"step": "filter", "params": {"type": "riesz", "variant": "simoncelli", "level": 2}}],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_eff_riesz_simon"])
    step_log = pipeline._log[-1]["steps_executed"][0]
    assert step_log["params_effective"]["variant"] == "simoncelli"
    assert step_log["params_effective"]["boundary"] == "periodic"
    assert "boundary" not in step_log["params_requested"]


@patch("pictologics.pipeline.mean_filter")
def test_step_filter_params_effective_boundary_condition_instance(
    mock_filter: MagicMock,
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """A BoundaryCondition instance passed directly as 'boundary' is recorded in
    params_effective by its lowercase name, not the enum object or scipy mode string."""
    from pictologics.filters import BoundaryCondition

    mock_filter.return_value = mock_image.array
    pipeline.add_config(
        "filter_eff_boundary_instance",
        [
            {
                "step": "filter",
                "params": {"type": "mean", "support": 3, "boundary": BoundaryCondition.NEAREST},
            }
        ],
    )
    pipeline.run(mock_image, mock_mask, config_names=["filter_eff_boundary_instance"])
    step_log = pipeline._log[-1]["steps_executed"][0]
    assert step_log["params_effective"]["boundary"] == "nearest"
    assert json.dumps(step_log)  # BoundaryCondition must not leak into the log


def test_step_filter_params_effective_source_mask_descriptor(
    sm_image: Image, sm_mask: Image
) -> None:
    """The pipeline-injected source_mask must never appear as a raw array in the log --
    only a compact, JSON-safe shape/voxel-count descriptor."""
    pipeline = RadiomicsPipeline()
    pipeline.add_config(
        "filter_source_mask_desc",
        [{"step": "filter", "params": {"type": "mean", "support": 3}}],
        source_mode="roi_only",
    )
    with patch("pictologics.pipeline.mean_filter", return_value=sm_image.array) as mock_filter:
        pipeline.run(sm_image, sm_mask, config_names=["filter_source_mask_desc"])
        call_kwargs = mock_filter.call_args.kwargs
        assert isinstance(call_kwargs.get("source_mask"), np.ndarray)  # really was an array

    step_log = pipeline._log[-1]["steps_executed"][0]
    assert "source_mask" not in step_log["params_requested"]  # never user-supplied
    mask_desc = step_log["params_effective"]["source_mask"]
    assert isinstance(mask_desc, dict)
    assert mask_desc["shape"] == [20, 20, 20]
    assert mask_desc["voxel_count"] == 20 * 20 * 20
    assert mask_desc["true_count"] == 10 * 10 * 10
    assert json.dumps(pipeline._log)  # whole run log must round-trip through JSON


def test_step_filter_params_provenance_full_log_json_roundtrip(
    pipeline: RadiomicsPipeline, mock_image: Image, mock_mask: Image
) -> None:
    """A multi-config run mixing several filter types (with a source_mask active)
    must produce a fully JSON-serializable log end to end."""
    from pictologics.filters import BoundaryCondition

    pipeline.add_config(
        "prov_log",
        [{"step": "filter", "params": {"type": "log", "sigma_mm": 1.0}}],
    )
    pipeline.add_config(
        "prov_riesz",
        [
            {
                "step": "filter",
                "params": {
                    "type": "riesz",
                    "variant": "log",
                    "sigma_mm": 1.0,
                    "boundary": BoundaryCondition.ZERO,
                },
            }
        ],
        source_mode="roi_only",
    )
    pipeline.add_config(
        "prov_laws",
        [{"step": "filter", "params": {"type": "laws", "kernel": "L5E5E5"}}],
    )

    with (
        patch("pictologics.pipeline.laplacian_of_gaussian", return_value=mock_image.array),
        patch("pictologics.pipeline.riesz_log", return_value=mock_image.array),
        patch("pictologics.pipeline.laws_filter", return_value=mock_image.array),
    ):
        pipeline.run(mock_image, mock_mask, config_names=["prov_log", "prov_riesz", "prov_laws"])

    serialized = json.dumps(pipeline._log)
    assert serialized  # no exception; every entry is JSON-safe
    reloaded = json.loads(serialized)
    assert len(reloaded) == 3
    for entry in reloaded:
        step_log = entry["steps_executed"][0]
        assert "params_requested" in step_log
        assert "params_effective" in step_log


def test_step_resample_missing_param(
    pipeline: RadiomicsPipeline,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test resample step error when new_spacing is missing."""
    pipeline.add_config(
        "resample_no_spacing",
        [{"step": "resample", "params": {"interpolation": "linear"}}],  # Missing new_spacing
        validate=False,
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
    assert "binarize_mask requires 'threshold' unless mask_values is provided" in log["error"]


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
@patch("pictologics.pipeline._texture_matrices")
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
        "glszm_cells": np.zeros((3, 0), dtype=np.uint32),
        "roi": np.zeros((1, 1, 1), dtype=bool),
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
@patch("pictologics.pipeline._texture_matrices")
@patch("pictologics.pipeline._glszm_features_from_cells")
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
        "glszm_cells": np.zeros((3, 0), dtype=np.uint32),
        "roi": np.zeros((1, 1, 1), dtype=bool),
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
@patch("pictologics.pipeline._texture_matrices")
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
        "glszm_cells": np.zeros((3, 0), dtype=np.uint32),
        "roi": np.zeros((1, 1, 1), dtype=bool),
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
@patch("pictologics.pipeline._texture_matrices")
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
        "glszm_cells": np.zeros((3, 0), dtype=np.uint32),
        "roi": np.zeros((1, 1, 1), dtype=bool),
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
@patch("pictologics.pipeline._texture_matrices")
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
        "glszm_cells": np.zeros((3, 0), dtype=np.uint32),
        "roi": np.zeros((1, 1, 1), dtype=bool),
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


def test_extract_histogram_full_range_fbs(mock_mask: Image) -> None:
    """Gradient features use the full IBSI [1, N_g] range with a custom FBS min_val."""
    arr = np.full((10, 10, 10), 20.0)
    arr[0, 0, 0] = 25.0
    image = Image(
        array=arr,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="CT",
    )

    pipeline = RadiomicsPipeline()
    pipeline.add_config(
        "hist_fbs",
        [
            # min_val far below the data: bins 1..20 exist but stay empty.
            {
                "step": "discretise",
                "params": {"method": "FBS", "bin_width": 1.0, "min_val": 0.0},
            },
            {"step": "extract_features", "params": {"families": ["histogram"]}},
        ],
    )
    feats = pipeline.run(image, mock_mask, config_names=["hist_fbs"])["hist_fbs"]

    # 999 voxels in bin 21, 1 voxel in bin 26, N_g = 26.
    # Central-difference gradient peaks at (999 - 0)/2 around the occupied bin.
    assert feats["maximum_histogram_gradient_12CE"] == pytest.approx(499.5)
    assert feats["maximum_histogram_gradient_intensity_8E6O"] == pytest.approx(20.0)
    assert feats["minimum_histogram_gradient_VQB3"] == pytest.approx(-499.5)
    assert feats["minimum_histogram_gradient_intensity_RHQZ"] == pytest.approx(22.0)


def test_extract_histogram_full_range_fixed_cutoffs(mock_image: Image, mock_mask: Image) -> None:
    """FIXED_CUTOFFS records N_g = len(cutoffs) + 1 for the histogram range."""
    pipeline = RadiomicsPipeline()
    pipeline.add_config(
        "hist_cutoffs",
        [
            {
                "step": "discretise",
                "params": {"method": "FIXED_CUTOFFS", "cutoffs": [0, 10, 20]},
            },
            {"step": "extract_features", "params": {"families": ["histogram"]}},
        ],
    )
    feats = pipeline.run(mock_image, mock_mask, config_names=["hist_cutoffs"])["hist_cutoffs"]

    # All voxels are 0.0 -> bin 2 of N_g=4; histogram [0, 1000, 0, 0].
    assert feats["maximum_histogram_gradient_12CE"] == pytest.approx(1000.0)
    assert feats["maximum_histogram_gradient_intensity_8E6O"] == pytest.approx(1.0)
    assert feats["minimum_histogram_gradient_VQB3"] == pytest.approx(-500.0)
    assert feats["minimum_histogram_gradient_intensity_RHQZ"] == pytest.approx(3.0)


@patch("pictologics.pipeline.calculate_ivh_features")
def test_extract_ivh_via_dedup_path(
    mock_ivh: MagicMock,
    mock_image: Image,
    mock_mask: Image,
) -> None:
    """Test IVH features via deduplication path."""
    pipeline = RadiomicsPipeline(deduplicate=True)
    mock_ivh.return_value = {"ivh_v10": 0.5}

    pipeline.add_config("cfg1", [{"step": "extract_features", "params": {"families": ["ivh"]}}])
    pipeline.add_config("cfg2", [{"step": "extract_features", "params": {"families": ["ivh"]}}])
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
@patch("pictologics.pipeline._texture_matrices")
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
        "glszm_cells": np.zeros((3, 0), dtype=np.uint32),
        "roi": np.zeros((1, 1, 1), dtype=bool),
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
@patch("pictologics.pipeline._texture_matrices")
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
        "glszm_cells": np.zeros((3, 0), dtype=np.uint32),
        "roi": np.zeros((1, 1, 1), dtype=bool),
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
    # ngldm_alpha passes as the whole number of the same integer test: |d| <= 0.5 is d == 0
    call_kwargs = mock_matrices.call_args.kwargs
    assert call_kwargs.get("ngldm_alpha") == 0


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
                "steps": [{"step": "extract_features", "params": {"families": ["morphology"]}}]
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
            "failed to restore deduplication plan" in str(warning.message).lower() for warning in w
        )


def test_feature_name_registry_sync() -> None:
    """Ensure the FEATURE_NAMES registry in features/__init__.py matches
    the actual keys returned by each extraction function."""
    from pictologics.features import (
        FEATURE_NAMES,
        calculate_glcm_features,
        calculate_gldzm_features,
        calculate_glrlm_features,
        calculate_glszm_features,
        calculate_intensity_features,
        calculate_intensity_histogram_features,
        calculate_ivh_features,
        calculate_local_intensity_features,
        calculate_morphology_features,
        calculate_ngldm_features,
        calculate_ngtdm_features,
        calculate_spatial_intensity_features,
    )

    rng = np.random.default_rng(42)
    arr = rng.integers(1, 33, size=(10, 10, 10)).astype(np.float64)
    mask_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    mask_arr[2:8, 2:8, 2:8] = 1
    img = Image(
        array=arr,
        spacing=(1.0, 1.0, 1.0),
        origin=(0, 0, 0),
        direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
    )
    mask = Image(
        array=mask_arr,
        spacing=(1.0, 1.0, 1.0),
        origin=(0, 0, 0),
        direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
    )
    vals = arr[mask_arr == 1]

    actual: dict[str, list[str]] = {
        "intensity": list(calculate_intensity_features(vals).keys()),
        "histogram": list(calculate_intensity_histogram_features(vals).keys()),
        "ivh": list(calculate_ivh_features(vals).keys()),
        "spatial_intensity": list(calculate_spatial_intensity_features(img, mask).keys()),
        "local_intensity": list(calculate_local_intensity_features(img, mask).keys()),
        "morphology": list(calculate_morphology_features(mask, img, intensity_mask=mask).keys()),
        "glcm": list(calculate_glcm_features(arr, mask_arr, 32).keys()),
        "glrlm": list(calculate_glrlm_features(arr, mask_arr, 32).keys()),
        "glszm": list(calculate_glszm_features(arr, mask_arr, 32).keys()),
        "gldzm": list(calculate_gldzm_features(arr, mask_arr, 32, distance_mask=mask_arr).keys()),
        "ngtdm": list(calculate_ngtdm_features(arr, mask_arr, 32).keys()),
        "ngldm": list(calculate_ngldm_features(arr, mask_arr, 32).keys()),
    }

    for family, expected_keys in actual.items():
        registry_keys = list(FEATURE_NAMES[family])
        assert registry_keys == expected_keys, (
            f"FEATURE_NAMES['{family}'] is out of sync with the extraction function. "
            f"Expected {expected_keys}, got {registry_keys}"
        )


# ---------------------------------------------------------------------------
# describe_features() tests
# ---------------------------------------------------------------------------

_DESCRIBE_COLUMNS = [
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


def test_describe_features_default_configs() -> None:
    """Default pipeline (6 standard configs) produces a non-empty catalog."""
    import pandas as pd

    pipeline = RadiomicsPipeline()
    df = pipeline.describe_features()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == _DESCRIBE_COLUMNS

    # 6 standard configs
    assert df["config"].nunique() == 6

    # Each config should have the same number of features (no spatial/local)
    counts = df.groupby("config").size()
    assert counts.nunique() == 1, (
        f"Expected uniform feature count per config, got {counts.to_dict()}"
    )

    # All standard configs are resampled and discretised
    assert df["is_resampled"].all()
    assert df["is_discretised"].all()
    assert not df["is_filtered"].any()

    # Every feature key should have a non-empty IBSI code
    assert (df["ibsi_code"] != "").all()

    # family_group should only contain known values
    assert set(df["family_group"].unique()) <= {"Intensity", "Morphology", "Texture"}


def test_describe_features_custom_config_with_filter() -> None:
    """A config with a filter step should populate filter columns."""
    import pandas as pd

    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config(
        "log_fbn8",
        [
            {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
            {"step": "filter", "params": {"type": "log", "sigma_mm": 3.0}},
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
            {"step": "extract_features", "params": {"families": ["texture"]}},
        ],
    )
    df = pipeline.describe_features()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert (df["config"] == "log_fbn8").all()
    assert df["is_filtered"].all()
    assert (df["filter_type"] == "log").all()
    assert df["filter_params"].notna().all()
    assert df["is_resampled"].all()
    assert df["is_discretised"].all()
    assert (df["discretisation_method"] == "FBN").all()
    # All texture features require discretisation
    assert df["requires_discretisation"].all()
    assert set(df["family_group"].unique()) == {"Texture"}


def test_describe_features_records_all_preprocessing_steps() -> None:
    """Catalog rows include ordered, repeated preprocessing metadata."""
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config(
        "all_preproc",
        [
            {
                "step": "resample",
                "params": {
                    "new_spacing": (1.0, 1.0, 1.0),
                    "interpolation": "linear",
                },
            },
            {"step": "resegment", "params": {"range_min": -100.0}},
            {"step": "filter_outliers", "params": {"sigma": 2.0}},
            {"step": "round_intensities", "params": {}},
            {"step": "keep_largest_component", "params": {"apply_to": "morph"}},
            {
                "step": "binarize_mask",
                "params": {"mask_values": [1, 2], "apply_to": "intensity"},
            },
            {"step": "filter", "params": {"type": "mean", "support": 3}},
            {
                "step": "resample",
                "params": {
                    "new_spacing": (0.5, 0.5, 0.5),
                    "mask_interpolation": "linear",
                    "mask_threshold": 0.25,
                },
            },
            {"step": "resegment", "params": {"range_max": 300.0}},
            {
                "step": "discretise",
                "params": {"method": "FBN", "n_bins": 8, "min_val": -100.0},
            },
            {"step": "extract_features", "params": {"families": ["histogram"]}},
        ],
        source_mode="auto",
        sentinel_value=-2048.0,
    )

    row = pipeline.describe_features().iloc[0]

    assert row["source_mode"] == "auto"
    assert row["sentinel_value"] == -2048.0
    assert not row["uses_morph_mask"]
    assert row["uses_intensity_mask"]
    assert row["feature_extraction_step_index"] == 11
    assert json.loads(row["feature_extraction_params"]) == {"families": ["histogram"]}
    assert row["preprocessing_sequence"] == (
        "1:resample > 2:resegment > 3:filter_outliers > "
        "4:round_intensities > 5:keep_largest_component > 6:binarize_mask > "
        "7:filter > 8:resample > 9:resegment > 10:discretise"
    )

    preprocessing_steps = json.loads(row["preprocessing_steps"])
    assert [record["step"] for record in preprocessing_steps] == [
        "resample",
        "resegment",
        "filter_outliers",
        "round_intensities",
        "keep_largest_component",
        "binarize_mask",
        "filter",
        "resample",
        "resegment",
        "discretise",
    ]

    assert row["is_resampled"]
    assert json.loads(row["resampling_spacing"]) == [
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
    ]
    assert json.loads(row["interpolation"]) == ["linear", "linear"]
    assert len(json.loads(row["resample_params"])) == 2

    assert row["is_resegmented"]
    assert len(json.loads(row["resegment_params"])) == 2
    assert json.loads(row["resegment_apply_to"]) == ["both", "both"]
    assert row["is_outlier_filtered"]
    assert row["filter_outliers_apply_to"] == "both"
    assert json.loads(row["filter_outliers_params"]) == [
        {"params": {"sigma": 2.0}, "step_index": 3}
    ]
    assert row["is_intensity_rounded"]
    assert json.loads(row["round_intensities_params"]) == [{"params": {}, "step_index": 4}]
    assert row["keeps_largest_component"]
    assert row["keep_largest_component_apply_to"] == "morph"
    assert row["is_mask_binarized"]
    assert row["binarize_mask_apply_to"] == "intensity"

    assert row["is_filtered"]
    assert row["filter_type"] == "mean"
    assert json.loads(row["filter_params"]) == [
        {"params": {"support": 3, "type": "mean"}, "step_index": 7}
    ]

    assert row["is_discretised"]
    assert row["discretisation_method"] == "FBN"
    assert row["discretisation_param"] == 8
    assert json.loads(row["discretise_params"]) == [
        {
            "params": {"method": "FBN", "min_val": -100.0, "n_bins": 8},
            "step_index": 10,
        }
    ]


def test_describe_features_uses_preprocessing_at_extraction_point() -> None:
    """Later preprocessing steps are not attached to earlier feature rows."""
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config(
        "two_extracts",
        [
            {
                "step": "extract_features",
                "params": {"families": ["intensity"]},
            },
            {"step": "resegment", "params": {"range_min": 0.0}},
            {"step": "extract_features", "params": {"families": ["morphology"]}},
        ],
    )

    catalog = pipeline.describe_features()
    intensity = catalog[catalog["family"] == "intensity"].iloc[0]
    morphology = catalog[catalog["family"] == "morphology"].iloc[0]

    assert intensity["feature_extraction_step_index"] == 1
    assert intensity["preprocessing_sequence"] is None
    assert not intensity["is_resegmented"]
    assert not intensity["uses_morph_mask"]
    assert intensity["uses_intensity_mask"]
    assert morphology["feature_extraction_step_index"] == 3
    assert morphology["preprocessing_sequence"] == "2:resegment"
    assert morphology["is_resegmented"]
    assert morphology["uses_morph_mask"]


def test_describe_features_catalog_serialization_helpers() -> None:
    """Structured catalog helpers should serialize uncommon supported values."""
    encoded = RadiomicsPipeline._catalog_json(
        {
            "array": np.array([1, 2]),
            "int": np.int64(3),
            "float": np.float64(4.5),
            "bool": np.bool_(True),
        }
    )
    assert json.loads(encoded) == {
        "array": [1, 2],
        "bool": True,
        "float": 4.5,
        "int": 3,
    }
    assert json.loads(RadiomicsPipeline._catalog_value([{"a": 1}])) == {"a": 1}

    metadata = RadiomicsPipeline._extract_config_metadata(
        [
            {"step": "extract_features", "params": {"families": ["intensity"]}},
            {
                "step": "discretise",
                "params": {"method": "FIXED_CUTOFFS", "cutoffs": [0, 10, 20]},
            },
            {"step": "discretise", "params": {"method": "CUSTOM"}},
        ]
    )
    assert json.loads(metadata["discretisation_method"]) == [
        "FIXED_CUTOFFS",
        "CUSTOM",
    ]
    assert json.loads(metadata["discretisation_param"]) == [[0, 10, 20], None]


def test_describe_features_ibsi_code_parsing() -> None:
    """Spot-check that tricky IVH feature keys parse correctly."""
    name, code = RadiomicsPipeline._parse_feature_key("joint_entropy_TU9B")
    assert name == "joint_entropy"
    assert code == "TU9B"

    # IVH feature with numeric suffix
    name, code = RadiomicsPipeline._parse_feature_key("volume_at_intensity_fraction_0.10_BC2M_10")
    assert name == "volume_at_intensity_fraction_0.10"
    assert code == "BC2M"

    # Simple single-word feature
    name, code = RadiomicsPipeline._parse_feature_key("mean_Q4LE")
    assert name == "mean"
    assert code == "Q4LE"

    # No IBSI code at all (graceful fallback)
    name, code = RadiomicsPipeline._parse_feature_key("custom_metric_no_code")
    assert name == "custom_metric_no_code"
    assert code == ""


def test_describe_features_empty_pipeline() -> None:
    """Pipeline with no configs should return an empty DataFrame with correct columns."""
    import pandas as pd

    pipeline = RadiomicsPipeline(load_standard=False)
    df = pipeline.describe_features()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == _DESCRIBE_COLUMNS
    assert len(df) == 0


def test_get_expected_feature_names_includes_spatial_and_local() -> None:
    """Expected names should include spatial/local extras when requested."""
    from pictologics.features import FEATURE_NAMES

    steps = [
        {
            "step": "extract_features",
            "params": {
                "families": ["intensity"],
                "include_spatial_intensity": True,
                "include_local_intensity": True,
            },
        }
    ]

    names = RadiomicsPipeline._get_expected_feature_names(steps)

    assert FEATURE_NAMES["intensity"][0] in names
    assert FEATURE_NAMES["spatial_intensity"][0] in names
    assert FEATURE_NAMES["local_intensity"][0] in names


def test_fill_missing_features_handles_none_params() -> None:
    """_fill_missing_features should accept params=None and backfill NaNs."""
    from pictologics.features import FEATURE_NAMES

    results: dict[str, Any] = {}

    RadiomicsPipeline._fill_missing_features(
        results=results,
        families=["intensity"],
        params=None,
    )

    # Spot-check one known key and ensure it was backfilled as NaN.
    key = FEATURE_NAMES["intensity"][0]
    assert key in results
    assert np.isnan(results[key])


from enum import Enum
from importlib.metadata import PackageNotFoundError
from pathlib import Path


def test_pipeline_serialization_edge_cases():
    pipeline = RadiomicsPipeline()
    # Test Enum and Path serialization

    class DummyEnum(Enum):
        TEST = "test_value"

    res = pipeline._make_serializable(Path("/tmp/test"))
    assert res == "/tmp/test"

    res = pipeline._make_serializable(DummyEnum.TEST)
    assert res == "test_value"


def test_get_apply_to_invalid():
    from pictologics.pipeline import _get_apply_to

    with pytest.raises(ValueError, match="apply_to must be"):
        _get_apply_to({"apply_to": "invalid_target"}, "some_step")


def test_get_package_version_not_found():
    from pictologics.pipeline import _get_package_version

    _get_package_version.cache_clear()
    try:
        with patch("pictologics.pipeline.version") as mock_version:
            mock_version.side_effect = PackageNotFoundError()
            assert _get_package_version() is None
            assert _get_package_version() is None
            assert mock_version.call_count == 1  # read once per process
    finally:
        _get_package_version.cache_clear()


# ===========================================================================
# Source-mode / sentinel handling and family-helper coverage
# (consolidated from the former test_coverage_gap_filler.py)
# ===========================================================================


@pytest.fixture
def sm_image() -> Image:
    """20^3 image used for the source-mode / sentinel tests."""
    arr = np.random.rand(20, 20, 20).astype(np.float32)
    return Image(arr, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))


@pytest.fixture
def sm_mask() -> Image:
    """10^3 ROI centered in the 20^3 volume."""
    m = np.zeros((20, 20, 20), dtype=np.uint8)
    m[5:15, 5:15, 5:15] = 1
    return Image(m, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))


def _sentinel_image() -> Image:
    """20^3 image with a -1000 sentinel background and valid tissue in the ROI."""
    arr = np.full((20, 20, 20), -1000.0, dtype=np.float32)
    arr[5:15, 5:15, 5:15] = 100.0
    return Image(arr, (1, 1, 1), (0, 0, 0))


def _filter_config(include_resample: bool) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if include_resample:
        steps.append({"step": "resample", "params": {"new_spacing": (1, 1, 1)}})
    steps += [
        {"step": "filter", "params": {"type": "mean", "support": 3}},
        {"step": "filter", "params": {"type": "log", "sigma_mm": 1.0}},
        {"step": "filter", "params": {"type": "laws", "kernel": "E5L5S5"}},
        {"step": "filter", "params": {"type": "gabor", "sigma_mm": 1.0, "lambda_mm": 2.0}},
        {
            "step": "filter",
            "params": {"type": "wavelet", "wavelet": "haar", "level": 1, "decomposition": "LLL"},
        },
        {"step": "filter", "params": {"type": "simoncelli", "level": 1}},
        {"step": "filter", "params": {"type": "riesz", "variant": "log", "sigma_mm": 1.0}},
        {
            "step": "extract_features",
            "params": {"families": ["intensity"]},
        },
    ]
    return steps


def test_pipeline_invalid_source_mode() -> None:
    pipeline = RadiomicsPipeline()
    config = [{"step": "resample", "params": {"new_spacing": (1, 1, 1)}}]
    with pytest.raises(ValueError, match="Invalid source_mode"):
        pipeline.add_config("bad", config, source_mode="invalid_mode")


def test_pipeline_roi_only_and_auto_modes(sm_image: Image, sm_mask: Image) -> None:
    pipeline = RadiomicsPipeline()
    config = _filter_config(include_resample=True)

    pipeline.add_config("roi_config", config, source_mode="roi_only")
    assert "roi_config" in pipeline.run(sm_image, sm_mask, config_names=["roi_config"])

    # AUTO mode detects the sentinel background and warns.
    pipeline.add_config("auto_config", config, source_mode="auto")
    with pytest.warns(UserWarning, match="Auto-detected sentinel value"):
        results_auto = pipeline.run(_sentinel_image(), sm_mask, config_names=["auto_config"])
    assert "auto_config" in results_auto

    auto_logs = [e for e in pipeline._log if e["config_name"] == "auto_config"]
    assert auto_logs
    assert auto_logs[-1]["sentinel_detected"] is True
    assert auto_logs[-1]["sentinel_value"] == -1000.0
    assert auto_logs[-1]["sentinel_auto_detected"] is True
    assert auto_logs[-1]["sentinel_proportion"] > 0.05


def test_pipeline_auto_sentinel_percentage_never_rounds_up_to_100() -> None:
    """A near-total sentinel fraction must not be reported as '100.0% of voxels'.

    Plain ``:.1f`` rounding turns 99.999% into "100.0%", which wrongly implies no
    voxels remain for feature extraction -- yet features are still produced from the
    surviving ROI. The warning must say ">99.9%" and state the valid voxel count.
    """
    shape = (40, 40, 40)
    array = np.full(shape, -2048.0)
    mask_array = np.zeros(shape, dtype=np.uint8)
    # Exactly one real voxel: 99.998% sentinel -- rounds up to 100.0% naively.
    array.reshape(-1)[0] = 50.0
    mask_array.reshape(-1)[0] = 1
    image = Image(array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    mask = Image(mask_array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))

    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config(
        "near_total",
        [{"step": "extract_features", "params": {"families": ["intensity"]}}],
        source_mode="auto",
    )
    with pytest.warns(UserWarning, match="Auto-detected sentinel value") as record:
        results = pipeline.run(image, mask, config_names=["near_total"])

    message = str(record[0].message)
    assert "100.0%" not in message
    assert ">99.9%" in message
    assert "1 of 64,000 voxels remain valid" in message
    # The stored proportion keeps full precision even though the display is clamped.
    log_entry = [e for e in pipeline._log if e["config_name"] == "near_total"][-1]
    assert 0.9999 < log_entry["sentinel_proportion"] < 1.0
    # Features are still produced -- the whole point of not claiming 100%.
    assert len(results["near_total"]) > 0


def test_pipeline_auto_sentinel_percentage_reports_exact_value_below_threshold() -> None:
    """Below the rounding-hazard threshold the true percentage is shown verbatim."""
    shape = (40, 40, 40)
    array = np.full(shape, -2048.0)
    mask_array = np.zeros(shape, dtype=np.uint8)
    array.reshape(-1)[:6400] = 50.0  # 90% sentinel
    mask_array.reshape(-1)[:6400] = 1
    image = Image(array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    mask = Image(mask_array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))

    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config(
        "ninety",
        [{"step": "extract_features", "params": {"families": ["intensity"]}}],
        source_mode="auto",
    )
    with pytest.warns(UserWarning, match="Auto-detected sentinel value") as record:
        pipeline.run(image, mask, config_names=["ninety"])

    message = str(record[0].message)
    assert "90.0% of voxels" in message
    assert "6,400 of 64,000 voxels remain valid" in message


def test_sentinel_search_warns_once_per_run(sm_image: Image, sm_mask: Image, caplog: Any) -> None:
    # One warning names the configurations that share the search (the first three and
    # a count), and the logging module gets no copy of it
    import logging

    pipeline = RadiomicsPipeline(load_standard=False)
    steps = [{"step": "extract_features", "params": {"families": ["intensity"]}}]
    for name in ("a", "b", "c", "d"):
        pipeline.add_config(name, copy.deepcopy(steps), source_mode="auto")
    with caplog.at_level(logging.DEBUG), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.run(sm_image, sm_mask, config_names=["a", "b", "c", "d"])
    assert [str(w.message)[:80] for w in caught] == [
        "No sentinel value auto-detected for 4 configs ('a', 'b', 'c' and 1 more) (no can"
    ]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert all(entry["sentinel_detected"] is False for entry in pipeline.get_log())


def test_pipeline_auto_no_sentinel_warns_and_continues(sm_image: Image, sm_mask: Image) -> None:
    pipeline = RadiomicsPipeline()
    config = [{"step": "extract_features", "params": {"families": ["intensity"]}}]
    pipeline.add_config("auto_none", config, source_mode="auto")

    with pytest.warns(UserWarning, match="No sentinel value auto-detected"):
        results = pipeline.run(sm_image, sm_mask, config_names=["auto_none"])
    assert "auto_none" in results

    logs = [e for e in pipeline._log if e["config_name"] == "auto_none"]
    assert logs[-1]["sentinel_detected"] is False
    assert logs[-1]["sentinel_value"] is None
    assert logs[-1]["sentinel_auto_detected"] is False


def test_pipeline_defensive_filter_return_branches(sm_image: Image, sm_mask: Image) -> None:
    # A filter that returns a bare ndarray (not a (result, valid) tuple) even though a
    # source_mask is active exercises the pipeline's defensive non-tuple branches.
    pipeline = RadiomicsPipeline()
    pipeline.add_config(
        "mean_cfg", [{"step": "filter", "params": {"type": "mean"}}], source_mode="roi_only"
    )
    pipeline.add_config(
        "log_cfg",
        [{"step": "filter", "params": {"type": "log", "sigma_mm": 1.0}}],
        source_mode="roi_only",
    )

    with patch(
        "pictologics.pipeline.mean_filter", return_value=np.zeros((20, 20, 20))
    ) as mock_mean:
        pipeline.run(sm_image, sm_mask, config_names=["mean_cfg"])
        mock_mean.assert_called()

    with patch(
        "pictologics.pipeline.laplacian_of_gaussian", return_value=np.zeros((20, 20, 20))
    ) as mock_log:
        pipeline.run(sm_image, sm_mask, config_names=["log_cfg"])
        mock_log.assert_called()


def test_pipeline_explicit_sentinel(sm_mask: Image) -> None:
    pipeline = RadiomicsPipeline()
    config = [{"step": "filter", "params": {"type": "mean"}}]
    pipeline.add_config("explicit_sent", config, source_mode="auto", sentinel_value=-1000)

    results = pipeline.run(_sentinel_image(), sm_mask, config_names=["explicit_sent"])
    assert "explicit_sent" in results

    exported_cfg = pipeline.to_dict(config_names=["explicit_sent"])["configs"]["explicit_sent"]
    assert exported_cfg["source_mode"] == "auto"
    assert exported_cfg["sentinel_value"] == -1000


def test_pipeline_filters_with_source_mask_no_resample(sm_image: Image, sm_mask: Image) -> None:
    pipeline = RadiomicsPipeline()
    pipeline.add_config(
        "nore_config", _filter_config(include_resample=False), source_mode="roi_only"
    )
    results = pipeline.run(sm_image, sm_mask, config_names=["nore_config"])
    assert "nore_config" in results
    assert not any("error" in entry for entry in pipeline._log)


def test_pipeline_laws_rotation_invariant_source_mask(sm_image: Image, sm_mask: Image) -> None:
    # rotation_invariant laws returns a single array even with a source_mask active.
    pipeline = RadiomicsPipeline()
    config = [
        {
            "step": "filter",
            "params": {"type": "laws", "rotation_invariant": True, "kernel": "E5L5S5"},
        },
        {
            "step": "extract_features",
            "params": {"families": ["intensity"]},
        },
    ]
    pipeline.add_config("laws_rot", config, source_mode="roi_only")
    assert "laws_rot" in pipeline.run(sm_image, sm_mask, config_names=["laws_rot"])


def _basic_state(image: Image, mask: Image, **kwargs: Any) -> PipelineState:
    return PipelineState(
        image=image, raw_image=image, morph_mask=mask, intensity_mask=mask, **kwargs
    )


def test_extract_family_helpers_default_bbox_cache() -> None:
    # The family helpers lazily create a bbox cache when called with the default None.
    img = Image(np.random.rand(6, 6, 6).astype(np.float32), (1, 1, 1), (0, 0, 0))
    mask_arr = np.zeros((6, 6, 6), np.uint8)
    mask_arr[1:5, 1:5, 1:5] = 1
    mask = Image(mask_arr, (1, 1, 1), (0, 0, 0))
    pipeline = RadiomicsPipeline()
    state = _basic_state(img, mask)

    assert pipeline._extract_single_family(state, "intensity", {})
    assert isinstance(pipeline._compute_ivh_features(state, {}, {}), dict)


def test_masked_values_empty_mask() -> None:
    # An empty mask has no nonzero bbox, so _masked_values falls back to apply_mask.
    img = Image(np.random.rand(5, 5, 5).astype(np.float32), (1, 1, 1), (0, 0, 0))
    empty = Image(np.zeros((5, 5, 5), np.uint8), (1, 1, 1), (0, 0, 0))
    pipeline = RadiomicsPipeline()
    assert isinstance(pipeline._masked_values(img, empty, {}), np.ndarray)


def test_compute_texture_features_default_cache_empty_mask() -> None:
    # Discretised state + empty masks: default bbox cache and the empty-bbox branch.
    disc = Image(np.ones((5, 5, 5), dtype=np.int32), (1, 1, 1), (0, 0, 0))
    empty = Image(np.zeros((5, 5, 5), np.uint8), (1, 1, 1), (0, 0, 0))
    pipeline = RadiomicsPipeline()
    state = _basic_state(disc, empty, is_discretised=True, n_bins=8)
    assert isinstance(pipeline._compute_texture_features(state, "glcm", {}), dict)


def test_binarize_mask_keeps_one_mask_array_in_sync() -> None:
    # Masks that are one array are binarized once and stay one array; separate masks,
    # or apply_to="intensity", binarize the intensity mask on its own.
    img = Image(np.random.rand(6, 6, 6), (1, 1, 1), (0, 0, 0))
    mask_arr = np.zeros((6, 6, 6))
    mask_arr[1:5, 1:5, 1:5] = 0.7
    mask = Image(mask_arr, (1, 1, 1), (0, 0, 0))
    pipeline = RadiomicsPipeline()
    params = {"threshold": 0.5}

    state = _basic_state(img, mask)
    pipeline._execute_preprocessing_step(state, "binarize_mask", params)
    assert state.intensity_mask is state.morph_mask
    assert state.morph_mask.array.dtype == np.uint8
    np.testing.assert_array_equal(state.morph_mask.array, (mask_arr >= 0.5).astype(np.uint8))

    state = _basic_state(img, mask)
    state.intensity_mask = Image(mask_arr.copy(), (1, 1, 1), (0, 0, 0))
    pipeline._execute_preprocessing_step(state, "binarize_mask", params)
    assert state.intensity_mask is not state.morph_mask
    np.testing.assert_array_equal(state.intensity_mask.array, state.morph_mask.array)

    state = _basic_state(img, mask)
    pipeline._execute_preprocessing_step(
        state, "binarize_mask", {**params, "apply_to": "intensity"}
    )
    assert state.morph_mask is mask
    np.testing.assert_array_equal(state.intensity_mask.array, (mask_arr >= 0.5).astype(np.uint8))


def test_ivh_discretisation_bins_only_the_roi_values() -> None:
    # With the bin limits given, the IVH bins come from the ROI values alone and give
    # the IVH features of a full-image discretisation; missing limits keep that path.
    from pictologics.features.intensity import calculate_ivh_features
    from pictologics.preprocessing import apply_mask, discretise_image

    rng = np.random.default_rng(3)
    raw = rng.normal(-200.0, 300.0, (8, 9, 7))
    raw[2, 2, 2] = np.nan
    mask_arr = np.zeros(raw.shape)
    mask_arr[1:6, 2:8, 1:6] = 1.0
    img, mask = Image(raw, (1, 1, 1), (0, 0, 0)), Image(mask_arr, (1, 1, 1), (0, 0, 0))
    pipeline = RadiomicsPipeline()
    state = _basic_state(img, mask)
    for disc, kwargs in (
        (
            {"method": "FBS", "bin_width": 25.0, "min_val": -1000.0},
            {"bin_width": 25.0, "min_val": -1000.0},
        ),
        ({"method": "FBN", "n_bins": 16, "min_val": -900.0, "max_val": 700.0}, {"min_val": -900.0}),
        ({"method": "FIXED_CUTOFFS", "cutoffs": [-500.0, 0.0, 300.0]}, {}),
        ({"method": "FBN", "n_bins": 16}, {}),  # a missing limit: the full-image path
    ):
        params = {"ivh_discretisation": dict(disc)}
        disc_params = dict(disc)
        method = disc_params.pop("method")
        full = discretise_image(img, method=method, roi_mask=mask, **disc_params)
        expected = calculate_ivh_features(apply_mask(full, mask), **kwargs)
        assert pipeline._compute_ivh_features(state, params, {}) == expected


@pytest.mark.parametrize(
    ("step", "params"),
    [
        ("resegment", {"range_min": 0.2, "range_max": 0.8}),
        ("filter_outliers", {"sigma": 1.0}),
        ("keep_largest_component", {}),
    ],
)
def test_mask_steps_keep_one_mask_array_in_sync(step: str, params: dict[str, Any]) -> None:
    # Masks that are one array are processed once and stay one array; separate masks get
    # their own, equal results.
    rng = np.random.default_rng(41)
    img = Image(rng.random((8, 8, 8)), (1, 1, 1), (0, 0, 0))
    mask_arr = (rng.random((8, 8, 8)) < 0.8).astype(np.uint8)
    mask = Image(mask_arr, (1, 1, 1), (0, 0, 0))
    pipeline = RadiomicsPipeline()

    state = _basic_state(img, mask)
    pipeline._execute_preprocessing_step(state, step, params)
    assert state.intensity_mask is state.morph_mask

    separate = _basic_state(img, mask)
    separate.intensity_mask = Image(mask_arr.copy(), (1, 1, 1), (0, 0, 0))
    pipeline._execute_preprocessing_step(separate, step, params)
    assert separate.intensity_mask is not separate.morph_mask
    np.testing.assert_array_equal(separate.morph_mask.array, state.morph_mask.array)
    np.testing.assert_array_equal(separate.intensity_mask.array, state.morph_mask.array)


def test_configurations_share_identical_preprocessing() -> None:
    # Two configurations with the same resample step run it once and give the results
    # (and step logs) that each gives alone. A binarize range and a label list with the
    # same numbers, or another source mode, share nothing.
    import copy

    from pictologics.pipeline import _prefix_keys
    from pictologics.preprocessing import resample_image

    rng = np.random.default_rng(51)
    img = Image(rng.normal(0.0, 50.0, (12, 11, 10)), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    labels = np.zeros((12, 11, 10))
    labels[2:10, 2:9, 2:8] = 1.0
    labels[4:7, 4:7, 4:7] = 2.0
    mask = Image(labels, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    resample = {"step": "resample", "params": {"new_spacing": (1.5, 1.5, 1.5)}}
    extract = {"step": "extract_features", "params": {"families": ["intensity", "morphology"]}}
    configs = {
        "fbn8": [
            resample,
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
            extract,
        ],
        "fbn16": [
            resample,
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 16}},
            extract,
        ],
        "range": [{"step": "binarize_mask", "params": {"mask_values": (1, 3)}}, extract],
        "labels": [{"step": "binarize_mask", "params": {"mask_values": [1, 3]}}, extract],
    }

    def run(names: list[str], source_mode: str = "full_image") -> tuple[dict[str, Any], int]:
        # No deduplication: only the shared preprocessing can make two results equal.
        pipeline = RadiomicsPipeline(deduplicate=False)
        for name in names:
            pipeline.add_config(name, copy.deepcopy(configs[name]), source_mode=source_mode)
        with patch("pictologics.pipeline.resample_image", wraps=resample_image) as spy:
            out = pipeline.run(img, mask, config_names=names)
        logs = {entry["config_name"]: entry["steps_executed"] for entry in pipeline._log}
        return {name: (out[name], logs[name]) for name in names}, spy.call_count

    together, calls = run(["fbn8", "fbn16"])
    assert calls == 2  # image and mask, once for both configurations
    for name in ("fbn8", "fbn16"):
        alone, _ = run([name])
        assert together[name][0].equals(alone[name][0])
        assert together[name][1] == alone[name][1]
    _, calls = run(["fbn8", "fbn16"], source_mode="roi_only")
    assert calls == 2
    both, _ = run(["range", "labels"])
    assert not both["range"][0].equals(both["labels"][0])  # label 2 only in the range

    # An array parameter counts in full: numpy's repr hides the middle of a large array.
    values = np.zeros(2000)
    changed = values.copy()
    changed[1000] = 1.0
    keys = [
        _prefix_keys([{"step": "binarize_mask", "params": {"mask_values": v}}], {})
        for v in (values, changed)
    ]
    assert keys[0] != keys[1]


def test_filters_of_the_roi_region_keep_every_feature() -> None:
    # A filter that no later step needs outside the ROI filters only the ROI region plus
    # its reach (LoG, wavelets, the Laws response), or for running sums (mean, Laws
    # energy) from the image start to the region end plus the reach. Gabor cuts each
    # slice through the region to the region grown by its kernel radius, in each of its
    # planes (here the grown region is the whole slice). Every feature stays bit for bit.
    from pictologics import pipeline as pipeline_module
    from pictologics.filters import laplacian_of_gaussian
    from pictologics.filters.gabor import _apply_gabor_to_plane

    rng = np.random.default_rng(81)
    img = Image(rng.normal(0.0, 50.0, (40, 36, 28)), (1.0, 1.0, 1.5), (0.0, 0.0, 0.0))
    labels = np.zeros(img.array.shape)
    labels[16:24, 14:22, 10:16] = 1.0
    mask = Image(labels, img.spacing, img.origin)
    filters = {
        "log": {"type": "log", "sigma_mm": 1.5},
        "wavelet": {"type": "wavelet", "wavelet": "db2", "level": 2, "decomposition": "LHL"},
        "wavelet_ri": {
            "type": "wavelet",
            "wavelet": "haar",
            "decomposition": "HHH",
            "rotation_invariant": True,
        },
        "laws": {"type": "laws", "kernel": "L5E5E5"},
        "laws_energy": {"type": "laws", "kernel": "L5E5E5", "compute_energy": True},
        "laws_ri": {"type": "laws", "kernel": "E5W5E5", "rotation_invariant": True},
        "mean": {"type": "mean", "support": 5},
        "gabor": {"type": "gabor", "sigma_mm": 2.0, "lambda_mm": 3.0, "theta": 0.5},
        "gabor_planes": {
            "type": "gabor",
            "sigma_mm": 2.0,
            "lambda_mm": 3.0,
            "average_over_planes": True,
        },
    }
    tail = [
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
        {
            "step": "extract_features",
            "params": {"families": ["intensity", "local_intensity", "glcm"]},
        },
    ]

    def run() -> dict[str, Any]:
        pipeline = RadiomicsPipeline()
        for name, params in filters.items():
            pipeline.add_config(name, [{"step": "filter", "params": dict(params)}, *tail])
        return pipeline.run(img, mask, config_names=list(filters))

    with (
        patch("pictologics.pipeline.laplacian_of_gaussian", wraps=laplacian_of_gaussian) as spy,
        patch(
            "pictologics.filters.gabor._apply_gabor_to_plane", wraps=_apply_gabor_to_plane
        ) as plane_spy,
    ):
        limited = run()
    assert spy.call_args[0][0].size < img.array.size  # the LoG read only the region
    assert len(plane_spy.call_args_list) == 4  # axial: one plane; averaged: three
    for call in plane_spy.call_args_list:
        axis = call.kwargs["plane_axis"]
        assert call.args[0].shape[axis] < img.array.shape[axis]
    with patch.object(pipeline_module, "_needs_full_grid", return_value=True):
        full_grid = run()
    for name in filters:
        assert limited[name].equals(full_grid[name])


def test_roi_region_filters_with_periodic_edges_and_source_masks() -> None:
    # A periodic boundary reads the other image end, so an axis where the filtered part
    # meets an image end is filtered whole. The roi_only source mask is cut with the
    # image, and without local intensity the region is the ROI box. Every feature stays
    # bit for bit.
    from pictologics import pipeline as pipeline_module
    from pictologics.filters import mean_filter

    rng = np.random.default_rng(82)
    img = Image(rng.normal(0.0, 50.0, (40, 36, 28)), (1.0, 1.0, 1.5), (0.0, 0.0, 0.0))
    labels = np.zeros(img.array.shape)
    labels[0:8, 14:22, 10:16] = 1.0  # meets the image start along axis 0
    mask = Image(labels, img.spacing, img.origin)
    filters = {
        "log": {"type": "log", "sigma_mm": 1.5, "boundary": "periodic"},
        "wavelet": {"type": "wavelet", "decomposition": "LHL", "boundary": "periodic"},
        "laws_energy": {
            "type": "laws",
            "kernel": "L5E5E5",
            "compute_energy": True,
            "boundary": "periodic",
        },
        "mean": {"type": "mean", "support": 5, "boundary": "periodic"},
        "mean_mirror": {"type": "mean", "support": 5},
    }
    configs = [
        (f"{name}_{mode}", params, mode)
        for name, params in filters.items()
        for mode in ("full_image", "roi_only")
    ]
    names = [name for name, _, _ in configs]

    def run() -> dict[str, Any]:
        pipeline = RadiomicsPipeline(load_standard=False)
        for name, params, mode in configs:
            pipeline.add_config(name, [
                {"step": "filter", "params": dict(params)},
                {"step": "extract_features", "params": {"families": ["intensity"]}},
            ], source_mode=mode)  # fmt: skip
        return pipeline.run(img, mask, config_names=names)

    with patch("pictologics.pipeline.mean_filter", wraps=mean_filter) as spy:
        limited = run()
    shapes = {call.args[0].shape for call in spy.call_args_list}
    assert shapes == {img.array.shape, (11, 25, 19)}  # periodic: whole; mirror: box end + 3
    for call in spy.call_args_list:
        if "source_mask" in call.kwargs:
            assert call.kwargs["source_mask"].shape == call.args[0].shape
    with patch.object(pipeline_module, "_needs_full_grid", return_value=True):
        full_grid = run()
    for name in names:
        assert limited[name].equals(full_grid[name])


def test_run_does_the_start_up_once_per_run(sm_mask: Image) -> None:
    # The first ROI check, the sentinel search and the source mask of each source setup
    # (source mode, sentinel value) are made once per run; the configurations of one
    # search share one warning, and each keeps its own log values and features.
    from pictologics import pipeline as pipeline_module

    image = _sentinel_image()
    steps = [{"step": "extract_features", "params": {"families": ["intensity"]}}]
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("auto_a", steps, source_mode="auto")
    pipeline.add_config("auto_b", copy.deepcopy(steps), source_mode="auto")
    pipeline.add_config("explicit", copy.deepcopy(steps), source_mode="auto", sentinel_value=-1000)
    pipeline.add_config("roi_a", copy.deepcopy(steps), source_mode="roi_only")
    pipeline.add_config("roi_b", copy.deepcopy(steps), source_mode="roi_only")
    names = ["auto_a", "auto_b", "explicit", "roi_a", "roi_b"]
    check = RadiomicsPipeline._ensure_nonempty_roi
    with (
        patch.object(
            pipeline_module, "detect_sentinel_value", wraps=pipeline_module.detect_sentinel_value
        ) as detect,
        patch.object(
            pipeline_module, "_source_mask", wraps=pipeline_module._source_mask
        ) as source_mask,
        patch.object(
            RadiomicsPipeline, "_ensure_nonempty_roi", autospec=True, side_effect=check
        ) as roi_check,
        pytest.warns(UserWarning, match="Auto-detected sentinel value") as record,
    ):
        together = pipeline.run(image, sm_mask, config_names=names)
    assert detect.call_count == 1
    # The detected value, the explicit value, and the ROI
    assert source_mask.call_count == 3
    contexts = [call.kwargs["context"] for call in roi_check.call_args_list]
    assert contexts.count("initialization") == 1
    messages = [str(w.message) for w in record if "Auto-detected" in str(w.message)]
    assert len(messages) == 1 and "for 2 configs ('auto_a', 'auto_b');" in messages[0]
    logs = {entry["config_name"]: entry for entry in pipeline._log}
    for name in names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alone = pipeline.run(image, sm_mask, config_names=[name])
        assert together[name].equals(alone[name])
        for key in ("sentinel_detected", "sentinel_value", "sentinel_proportion"):
            assert logs[name][key] == pipeline._log[-1][key]


def test_run_frees_each_source_mask_after_its_last_configuration(sm_mask: Image) -> None:
    # The configurations of one source setup share its source mask, and the mask is freed
    # once the last of them has run: auto, auto, ROI-only, ROI-only keep one at a time.
    import weakref

    steps = [{"step": "extract_features", "params": {"families": ["intensity"]}}]
    pipeline = RadiomicsPipeline(load_standard=False, deduplicate=False)
    for name, mode in (("a1", "auto"), ("a2", "auto"), ("r1", "roi_only"), ("r2", "roi_only")):
        pipeline.add_config(name, copy.deepcopy(steps), source_mode=mode)
    refs: list[Any] = []
    shared: list[bool] = []
    alive: list[list[bool]] = []
    extract = RadiomicsPipeline._extract_features

    def spy(self: RadiomicsPipeline, state: PipelineState, params: dict[str, Any]) -> Any:
        refs.append(weakref.ref(state.source_mask))
        shared.append(len(refs) > 1 and refs[-1]() is refs[-2]())
        alive.append([ref() is not None for ref in refs])
        return extract(self, state, params)

    with patch.object(RadiomicsPipeline, "_extract_features", spy), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.run(_sentinel_image(), sm_mask)
    assert shared == [False, True, False, True]
    assert alive[2] == [False, False, True]  # r1 runs: the auto mask is gone


def test_run_reuses_the_plan_while_the_configs_and_rules_stay() -> None:
    # A run builds the deduplication plan again only when the rules object or the pickled
    # configs (names in order, steps and metadata) changed.
    from pictologics import pipeline as pipeline_module
    from pictologics.deduplication import DeduplicationRules

    rng = np.random.default_rng(3)
    image = Image(rng.normal(size=(12, 12, 12)), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    labels = np.zeros(image.array.shape)
    labels[3:9, 3:9, 3:9] = 1.0
    mask = Image(labels, image.spacing, image.origin)

    def config(n_bins: int) -> list[dict[str, Any]]:
        return [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": n_bins}},
            {"step": "extract_features", "params": {"families": ["intensity", "glcm"]}},
        ]

    steps_b = config(16)
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("a", config(8))
    pipeline.add_config("b", steps_b)
    analyzer_class = pipeline_module.ConfigurationAnalyzer
    with patch.object(pipeline_module, "ConfigurationAnalyzer", wraps=analyzer_class) as analyzer:
        first = pipeline.run(image, mask)
        plan = pipeline.last_deduplication_plan
        second = pipeline.run(image, mask)
        assert analyzer.call_count == 1 and pipeline.last_deduplication_plan is plan
        for name in first:
            assert first[name].equals(second[name])
        steps_b[0]["params"]["n_bins"] = 8  # the pipeline keeps its own copy
        pipeline.run(image, mask)
        assert analyzer.call_count == 1
        pipeline.add_config("b", steps_b)  # a changed config: a new plan
        pipeline.run(image, mask)
        assert analyzer.call_count == 2
        pipeline.run(image, mask, config_names=["b", "a"])
        assert analyzer.call_count == 3
        pipeline.deduplication_rules = DeduplicationRules.from_dict(
            pipeline.deduplication_rules.to_dict()
        )
        pipeline.run(image, mask, config_names=["b", "a"])
        assert analyzer.call_count == 4
        # A parameter that pickle cannot write: the plan is built for every run.
        pipeline._configs["b"].append({"step": "note", "params": {"made_by": lambda: None}})
        for _ in range(2):
            pipeline.run(image, mask)
        assert analyzer.call_count == 6


def test_merge_configs_marks_the_plan_out_of_date(pipeline: RadiomicsPipeline) -> None:
    from pictologics.deduplication import ConfigurationAnalyzer

    steps = [{"step": "extract_features", "params": {"families": ["intensity"]}}]
    pipeline.add_config("cfg1", steps)
    pipeline._last_deduplication_plan = ConfigurationAnalyzer(pipeline._configs).analyze()
    pipeline._configs_modified_since_plan = False
    other = RadiomicsPipeline(load_standard=False)
    other.add_config("cfg2", copy.deepcopy(steps))
    pipeline.merge_configs(other)
    data = pipeline.to_dict(config_names=["cfg1", "cfg2"], include_deduplication=True)
    assert "last_plan" not in data["deduplication"]


# --- Config checks, input types and small fixes ---


def test_add_config_finds_missing_filter_parameters() -> None:
    # A filter parameter without a default is a config problem when the step leaves it
    # out (the Laws kernels have the step default; an unknown Riesz variant is the
    # problem itself).
    for params, missing in (
        ({"type": "gaussian"}, ["sigma_mm"]),
        ({"type": "log", "truncate": 3.0}, ["sigma_mm"]),
        ({"type": "gabor", "sigma_mm": 2.0}, ["lambda_mm"]),
        ({"type": "riesz"}, ["order"]),
        ({"type": "riesz", "variant": "log", "order": (1, 0, 0)}, ["sigma_mm"]),
        ({"type": "riesz", "variant": "simoncelli"}, []),
        ({"type": "laws"}, []),
        ({"type": "mean"}, []),
    ):
        problems = _steps_problems([{"step": "filter", "params": params}])
        assert problems == [f"step 0 (filter): missing parameter '{name}'" for name in missing]
    problems = _steps_problems([{"step": "filter", "params": {"type": "riesz", "variant": "logg"}}])
    assert problems == ["step 0 (filter): unknown riesz variant 'logg' (did you mean 'log'?)"]


def test_add_config_checks_the_filter_values() -> None:
    for params, message in (
        ({"type": "wavelet", "level": 0}, "level must be a whole number of 1 or more, not 0"),
        ({"type": "wavelet", "decomposition": "LH"}, "decomposition must be 3 letters L or H"),
        ({"type": "simoncelli", "level": -1}, "level must be a whole number of 1 or more, not -1"),
        ({"type": "riesz", "order": (1, 0)}, "order must be 3 whole numbers of 0 or more"),
        (
            {"type": "riesz", "variant": "log", "sigma_mm": 1.0, "order": (0, 0, 0)},
            "At least one order component",
        ),
    ):
        problems = _steps_problems([{"step": "filter", "params": params}])
        assert len(problems) == 1 and problems[0].startswith(f"step 0 (filter): {message}")
    assert (
        _steps_problems([{"step": "filter", "params": {"type": "wavelet", "decomposition": "hhl"}}])
        == []
    )


def test_masks_with_fractions_give_a_warning() -> None:
    # Every voxel that is not 0 is ROI, also a voxel of 0.05 of a probability map: a
    # warning, unless every configuration chooses the ROI with binarize_mask.
    from pictologics.pipeline import _has_fractions

    spacing, origin = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    image = Image(np.random.default_rng(4).normal(size=(10, 10, 10)), spacing, origin)
    probability = np.zeros((10, 10, 10))
    probability[2:8, 2:8, 2:8] = 0.05
    probability[4:6, 4:6, 4:6] = 0.95
    extract = {"step": "extract_features", "params": {"families": ["intensity"]}}
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("plain", [extract])
    pipeline.add_config(
        "binarized", [{"step": "binarize_mask", "params": {"threshold": 0.5}}, extract]
    )
    mask = Image(probability, spacing, origin)
    with pytest.warns(
        UserWarning, match="The mask holds values that are not whole numbers"
    ) as caught:
        pipeline.run(image, mask, config_names=["plain"])
    assert caught[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipeline.run(image, mask, config_names=["binarized"])
        pipeline.run(image, Image(np.ceil(probability), spacing, origin), config_names=["plain"])
        pipeline.run(
            image,
            Image((probability > 0).astype(np.uint8), spacing, origin),
            config_names=["plain"],
        )
        pipeline.run(image, None, config_names=["plain"])
    # a large mask: its ROI box scan is also the first ROI check, and a sample is read
    big = np.zeros((48, 48, 32))
    big[8:40, 8:40, 4:28] = 0.6
    big_image = Image(np.random.default_rng(5).normal(size=big.shape), spacing, origin)
    with pytest.warns(UserWarning, match="The mask holds values that are not whole numbers"):
        result = pipeline.run(big_image, Image(big, spacing, origin), config_names=["plain"])
    assert result["plain"].notna().all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipeline.run(big_image, Image(np.ceil(big), spacing, origin), config_names=["plain"])
    assert (
        pipeline.run(
            big_image, Image(np.zeros(big.shape), spacing, origin), config_names=["plain"]
        )["plain"]
        .isna()
        .all()
    )  # an empty float mask: an empty ROI, as before
    # a large ROI box: a regular sample of it
    whole = np.ones((80, 80, 80))
    box = (slice(0, 80), slice(0, 80), slice(0, 80))
    assert not _has_fractions(whole, box)
    whole[::4, ::4, ::4] = 0.5  # the voxels of the sample (step 4)
    assert _has_fractions(whole, box)


def test_a_failing_family_leaves_the_other_families(tmp_path: Any) -> None:
    # An error in one family makes only its features NaN; the log entry lists it, a
    # warning names it, and the next configuration computes it again (no reuse of a
    # failure). An empty ROI inside a family stays an empty ROI of the configuration.
    rng = np.random.default_rng(10)
    spacing, origin = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    image = Image(rng.normal(40.0, 20.0, (12, 12, 12)), spacing, origin)
    roi = np.zeros((12, 12, 12), dtype=np.uint8)
    roi[3:9, 3:9, 3:9] = 1
    mask = Image(roi, spacing, origin)
    steps = [
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
        {"step": "extract_features", "params": {"families": ["intensity", "texture", "histogram"]}},
    ]
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("a", steps)
    pipeline.add_config("b", copy.deepcopy(steps))
    boom = RuntimeError("boom")
    with (
        patch("pictologics.pipeline.calculate_glcm_features", side_effect=boom),
        pytest.warns(UserWarning, match=r"The texture features failed \(RuntimeError: boom\)"),
    ):
        results = pipeline.run(image, mask, config_names=["a", "b"])
    for entry in pipeline.get_log():
        assert entry["status"] == "completed"
        assert entry["family_errors"] == {"texture": "RuntimeError: boom"}
    for name in ("a", "b"):
        series = results[name]
        texture = [key for key in series.index if key.startswith(("joint_", "contrast_"))]
        assert texture and series[texture].isna().all()
        assert np.isfinite(series["mean_intensity_Q4LE"])
    # run_batch counts the failed family
    with patch("pictologics.pipeline.calculate_glcm_features", side_effect=boom):
        table = pipeline.run_batch(
            [{"subject_id": "s", "image": image, "mask": mask}],
            tmp_path,
            config_names=["a"],
            show_progress=False,
        )
    assert table["status"][0] == "incomplete"
    assert table["error"][0] == "a (texture): RuntimeError: boom"
    pipeline.clear_log()
    with patch.object(pipeline, "_extract_single_family", side_effect=EmptyROIMaskError("none")):
        pipeline.run(image, mask, config_names=["a"])
    assert pipeline.get_log()[0]["status"] == "empty_roi"


def _steps_problems(steps: list[Any]) -> list[str]:
    from pictologics.pipeline import _config_problems

    return _config_problems(steps)


def test_add_config_lists_every_problem_with_a_hint() -> None:
    # A config mistake raises one error at add_config that lists every problem, with
    # the closest valid name; validate=False keeps the old structure-only check.
    bad = [
        {"step": "resampel", "params": {"new_spacing": (1, 1, 1)}},
        {"step": "resample", "params": {"new_spacing": 1.0}},
        {"step": "resegment", "params": {"min": -100, "max": 400}},
        {"step": "binarize_mask", "params": {"mask_value": 2, "apply_to": "all"}},
        {"step": "discretise", "params": {"method": "fbn", "n_bins": 32}},
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 32.5}},
        {"step": "discretise", "params": {"method": "FBS"}},
        {"step": "discretise", "params": {"method": "FIXED_CUTOFFS"}},
        {"step": "filter", "params": {"type": "gausian"}},
        {"step": "filter", "params": {"type": "log", "sigma": 2.0}},
        {"step": "filter", "params": {"type": "riesz", "variant": "logg"}},
        {"step": "filter", "params": {"type": "laws", "kernel": "L5E5E5", "kernels": "x"}},
        {"step": "extract_features", "params": {"familes": ["intensity"]}},
        {
            "step": "extract_features",
            "params": {"families": ["texure", "shape"], "ivh_params": "x"},
        },
        {"step": "extract_features", "params": {"families": "texture"}},
        {"step": "round_intensities", "params": ["x"]},
        {"step": "resample", "params": {}},
    ]
    pipeline = RadiomicsPipeline(load_standard=False)
    with pytest.raises(ValueError, match="Configuration 'bad' has 23 problem") as error:
        pipeline.add_config("bad", bad)
    message = str(error.value)
    for expected in (
        "step 0: unknown step type 'resampel' (did you mean 'resample'?)",
        "step 1 (resample): new_spacing must be three positive numbers, not 1.0",
        "step 2 (resegment): unknown parameter 'min'",
        "step 3 (binarize_mask): unknown parameter 'mask_value' (did you mean 'mask_values'?)",
        "step 3 (binarize_mask): apply_to must be one of ('both', 'morph', 'intensity'), not 'all'",
        "step 4 (discretise): unknown method 'fbn' (did you mean 'FBN'?)",
        "step 5 (discretise): FBN needs a whole n_bins of 1 or more, not 32.5",
        "step 6 (discretise): FBS needs a bin_width above 0, not None",
        "step 6 (discretise): FBS needs a start that is the same for every image: set min_val",
        "step 7 (discretise): FIXED_CUTOFFS needs cutoffs",
        "step 8 (filter): unknown filter type 'gausian' (did you mean 'gaussian'?)",
        "step 9 (filter): unknown parameter 'sigma' (did you mean 'sigma_mm'?)",
        "step 9 (filter): missing parameter 'sigma_mm'",
        "step 10 (filter): unknown riesz variant 'logg' (did you mean 'log'?)",
        "step 11 (filter): unknown parameter 'kernels' (did you mean 'kernel'?)",
        "step 12 (extract_features): unknown parameter 'familes' (did you mean 'families'?)",
        "step 13 (extract_features): ivh_params must be a dict, not 'x'",
        "step 13 (extract_features): unknown feature family 'texure' (did you mean 'texture'?)",
        "step 13 (extract_features): unknown feature family 'shape'",
        "step 14 (extract_features): families must be a list of names, not 'texture'",
        "step 15 (round_intensities): params must be a dictionary",
        "step 16 (resample): missing parameter 'new_spacing'",
    ):
        assert expected in message, expected
    pipeline.add_config("bad", bad, validate=False)  # the old structure-only check
    assert pipeline.get_config("bad") == bad
    # texture features need an earlier discretise step; histogram and IVH do not
    assert _steps_problems([{"step": "extract_features", "params": {"families": ["glcm"]}}]) == [
        "step 0 (extract_features): the 'glcm' features need an earlier 'discretise' step"
    ]
    assert (
        _steps_problems(
            [{"step": "extract_features", "params": {"families": ["histogram", "ivh"]}}]
        )
        == []
    )
    # structure problems, and the source mode
    from pictologics.pipeline import _config_problems

    assert _config_problems("steps") == ["steps must be a list"]
    assert _config_problems(["x", {"params": {}}], "roi") == [
        "source_mode must be one of ('full_image', 'roi_only', 'auto'), not 'roi'",
        "step 0: must be a dictionary",
        "step 1: missing 'step' key",
    ]


def test_configs_from_files_warn_and_check_the_source_mode() -> None:
    # A configuration file still loads with mistakes (validate=True warns), but an
    # unknown source mode fails at load time, not inside run().
    data = {
        "schema_version": "1.1",
        "configs": {
            "c": {
                "steps": [
                    {
                        "step": "extract_features",
                        "params": {"families": ["intensity"], "familes": []},
                    }
                ]
            }
        },
    }
    with pytest.warns(UserWarning, match="unknown parameter 'familes'"):
        pipeline = RadiomicsPipeline.from_dict(data, validate=True)
    assert pipeline.list_configs() == ["c"]
    data["configs"]["c"]["source_mode"] = "roi"
    with pytest.raises(ValueError, match="source_mode must be one of"):
        RadiomicsPipeline.from_dict(data)


def test_add_config_keeps_its_own_copy_and_takes_the_enum(sm_image: Image, sm_mask: Image) -> None:
    from pictologics.pipeline import SourceMode

    steps = [
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
        {"step": "extract_features", "params": {"families": ["histogram"]}},
    ]
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("fbn8", steps, source_mode=SourceMode.AUTO)
    steps[0]["params"]["n_bins"] = 64  # an edit to build the next config
    pipeline.add_config("fbn64", steps)
    assert pipeline.get_config("fbn8")[0]["params"]["n_bins"] == 8
    assert pipeline._config_metadata["fbn8"]["source_mode"] == "auto"


def test_run_checks_its_inputs_and_config_names(sm_image: Image, sm_mask: Image) -> None:
    # An array (not an Image) is a clear TypeError; one name may be a str; a name given
    # twice runs once; an unknown name gets a hint.
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config(
        "intensity", [{"step": "extract_features", "params": {"families": ["intensity"]}}]
    )
    with pytest.raises(TypeError, match="image must be a path or an Image, not ndarray"):
        pipeline.run(sm_image.array, sm_mask, config_names=["intensity"])
    with pytest.raises(TypeError, match="mask must be a path, an Image or None, not ndarray"):
        pipeline.run(sm_image, sm_mask.array, config_names=["intensity"])
    pipeline.clear_log()
    results = pipeline.run(sm_image, sm_mask, config_names="intensity")
    assert list(results) == ["intensity"]
    pipeline.run(sm_image, sm_mask, config_names=["intensity", "intensity"])
    assert len(pipeline._log) == 2  # one entry per run
    with pytest.raises(ValueError, match="'intensty' not found. \\(did you mean 'intensity'\\?\\)"):
        pipeline.run(sm_image, sm_mask, config_names=["intensty"])


def test_in_memory_images_of_any_type_give_the_float64_features() -> None:
    # An int16 or float32 in-memory image gives the features of the same float64 image
    # (it no longer resamples in its own type: rounded, and on one core).
    from scipy import ndimage

    rng = np.random.default_rng(7)
    hu = np.round(ndimage.gaussian_filter(rng.normal(0, 1, (40, 40, 16)), 1.2) * 300 + 40)
    labels = np.zeros(hu.shape, np.uint8)
    labels[10:30, 8:30, 4:12] = 1
    mask = Image(labels, (0.7, 0.7, 2.3), (0.0, 0.0, 0.0))
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("c", [
        {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 16}},
        {"step": "extract_features", "params": {"families": ["intensity", "glcm"]}},
    ])  # fmt: skip
    expected = pipeline.run(Image(hu, (0.7, 0.7, 2.3), (0.0, 0.0, 0.0)), mask, config_names=["c"])[
        "c"
    ]
    for dtype in (np.int16, np.float32):
        image = Image(hu.astype(dtype), (0.7, 0.7, 2.3), (0.0, 0.0, 0.0))
        result = pipeline.run(image, mask, config_names=["c"])["c"]
        assert result.equals(expected), dtype
        assert image.array.dtype == dtype  # the caller's image is not changed


def test_whole_number_floats_from_files_work(sm_image: Image, sm_mask: Image) -> None:
    # n_bins 32.0 and ngldm_alpha 1.0 (from YAML or JSON) give the features of 32 and 1.
    def config(n_bins: Any, alpha: Any) -> dict[str, Any]:
        return {
            "steps": [
                {"step": "discretise", "params": {"method": "FBN", "n_bins": n_bins}},
                {
                    "step": "extract_features",
                    "params": {
                        "families": ["glcm", "ngldm"],
                        "texture_matrix_params": {"ngldm_alpha": alpha},
                    },
                },
            ]
        }

    data = {
        "schema_version": "1.1",
        "configs": {"floats": config(32.0, 1.0), "ints": config(32, 1)},
    }
    pipeline = RadiomicsPipeline.from_dict(data)
    results = pipeline.run(sm_image, sm_mask, config_names=["floats", "ints"])
    assert not results["floats"].isna().any()
    assert results["floats"].equals(results["ints"])


def test_numpy_numbers_in_params_with_deduplication(sm_image: Image, sm_mask: Image) -> None:
    # A new_spacing of numpy integers no longer stops run() when deduplication is on.
    pipeline = RadiomicsPipeline(load_standard=False)
    for n in (8, 16):
        pipeline.add_config(f"c{n}", [
            {"step": "resample", "params": {"new_spacing": tuple(np.array([1, 1, 1]))}},
            {"step": "discretise", "params": {"method": "FBN", "n_bins": n}},
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ])  # fmt: skip
    pipeline.run(sm_image, sm_mask)
    assert [entry["status"] for entry in pipeline._log] == ["completed", "completed"]


def test_roi_check_and_background_selection_helpers() -> None:
    from pictologics.pipeline import _ArrayMemo, _has_roi, _selects_background, _spacing_problem

    for shape in ((8, 8, 8), (128, 128, 64)):  # small: ndarray.any; large: the box scan
        mask = np.zeros(shape, np.uint8)
        assert not _has_roi(mask, _ArrayMemo())
        mask[3, 4, 5] = 1
        boxes: _ArrayMemo[int, Any] = _ArrayMemo()
        assert _has_roi(mask, boxes)
        # The box scan keeps the box of a large mask for the other checks of the run
        box = (slice(3, 4), slice(4, 5), slice(5, 6))
        assert boxes == ({} if mask.size < 1 << 20 else {id(mask): box})
    cases = [
        ({}, False), ({"threshold": 0.0}, True), ({"threshold": 0.5}, False),
        ({"threshold": None}, False), ({"mask_values": (0, 2)}, True),
        ({"mask_values": (1, 2)}, False), ({"mask_values": [0, 3]}, True),
        ({"mask_values": [1]}, False), ({"mask_values": 0}, True), ({"mask_values": 2}, False),
    ]  # fmt: skip
    for params, expected in cases:
        assert _selects_background(params) is expected, params
    assert (
        _spacing_problem((1.0, 1.0)) == "new_spacing must be three positive numbers, not (1.0, 1.0)"
    )
    assert _spacing_problem((0, 1, 1)) is not None and _spacing_problem([1, 1, 1]) is None


def test_binarize_selecting_background_after_a_filter_uses_the_full_grid() -> None:
    # A binarize_mask that keeps mask value 0 makes the whole image the ROI, so the
    # filter before it must filter the full grid (the ROI region alone left zeros).
    from scipy.ndimage import gaussian_filter

    from pictologics.filters import wavelet_transform

    rng = np.random.default_rng(21)
    arr = gaussian_filter(rng.normal(size=(40, 40, 30)), 1.0) * 100 + 40
    labels = np.zeros(arr.shape)
    labels[15:22, 16:23, 12:17] = 1.0
    image = Image(arr, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    mask = Image(labels, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("all", [
        {"step": "filter", "params": {"type": "wavelet", "wavelet": "db2", "level": 1, "decomposition": "LLL"}},
        {"step": "binarize_mask", "params": {"threshold": 0.0}},
        {"step": "extract_features", "params": {"families": ["intensity"]}},
    ])  # fmt: skip
    mean = pipeline.run(image, mask, config_names=["all"])["all"]["mean_intensity_Q4LE"]
    full = wavelet_transform(arr, "db2", 1, "LLL", boundary="mirror")
    assert mean == pytest.approx(float(np.mean(full.astype(np.float64))), rel=1e-12)


def test_source_masks_keep_labels_above_255(sm_image: Image) -> None:
    # In roi_only mode a uint16 label of 300 stays 300 (uint8 made it 44), so a later
    # binarize_mask for label 300 finds its voxels.
    labels = np.zeros(sm_image.array.shape, np.uint16)
    labels[8:18, 8:18, 8:18] = 300
    mask = Image(labels, sm_image.spacing, sm_image.origin)
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("label300", [
        {"step": "resample", "params": {"new_spacing": sm_image.spacing}},
        {"step": "binarize_mask", "params": {"mask_values": [300]}},
        {"step": "extract_features", "params": {"families": ["morphology"]}},
    ], source_mode="roi_only")  # fmt: skip
    result = pipeline.run(sm_image, mask, config_names=["label300"])["label300"]
    assert pipeline._log[-1]["status"] == "completed"
    assert result["volume_voxel_counting_YEKZ"] > 0


def test_negative_labels_are_roi_in_every_family(sm_image: Image) -> None:
    # Every nonzero label is ROI membership: a label of -1 gives the features of label 1
    # in all families (some steps used `> 0` before).
    labels = np.zeros(sm_image.array.shape)
    labels[6:20, 7:19, 6:18] = 1.0
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("all", [
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 16}},
        {"step": "extract_features", "params": {
            "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
            "include_spatial_intensity": True, "include_local_intensity": True,
        }},
    ])  # fmt: skip
    positive = pipeline.run(sm_image, Image(labels, sm_image.spacing, sm_image.origin))["all"]
    negative = pipeline.run(sm_image, Image(-labels, sm_image.spacing, sm_image.origin))["all"]
    assert not positive.isna().all()
    assert positive.equals(negative)


def test_texture_families_listed_one_by_one_share_one_pass(sm_mask: Image) -> None:
    # Texture families listed one by one take one matrix pass and give the features of
    # 'texture', in the order of the list. With dedup, only the families that are not
    # in the cache join the pass.
    from pictologics import pipeline as pipeline_module
    from pictologics.features import FEATURE_NAMES

    rng = np.random.default_rng(4)
    image = Image(rng.normal(50.0, 20.0, (20, 20, 20)), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    disc = {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}}
    listed = ["ngldm", "intensity", "texture_glcm", "glrlm", "glszm", "gldzm", "ngtdm"]
    pipeline = RadiomicsPipeline(load_standard=False, deduplicate=False)
    pipeline.add_config(
        "all", [disc, {"step": "extract_features", "params": {"families": ["texture"]}}]
    )
    pipeline.add_config(
        "listed", [disc, {"step": "extract_features", "params": {"families": listed}}]
    )
    matrices = pipeline_module._texture_matrices
    with patch.object(pipeline_module, "_texture_matrices", wraps=matrices) as pass_count:
        out = pipeline.run(image, sm_mask, config_names=["all", "listed"])
    assert pass_count.call_count == 2
    texture = out["listed"].drop(list(FEATURE_NAMES["intensity"]))
    order = [
        key
        for family in ("ngldm", "glcm", "glrlm", "glszm", "gldzm", "ngtdm")
        for key in FEATURE_NAMES[family]
    ]
    assert list(texture.index) == order
    assert texture.equals(out["all"][order])

    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config(
        "one", [disc, {"step": "extract_features", "params": {"families": ["glcm"]}}]
    )
    pipeline.add_config(
        "two",
        [disc, {"step": "extract_features", "params": {"families": ["glcm", "glrlm", "ngtdm"]}}],
    )
    with patch.object(pipeline_module, "_texture_matrices", wraps=matrices) as pass_count:
        out = pipeline.run(image, sm_mask, config_names=["one", "two"])
    flags = [
        {name for name in ("glcm", "glrlm", "ngtdm") if call.kwargs[f"calc_{name}"]}
        for call in pass_count.call_args_list
    ]
    assert flags == [{"glcm"}, {"glrlm", "ngtdm"}]
    assert out["two"][list(FEATURE_NAMES["glcm"])].equals(out["one"])


def test_roi_values_and_distance_map_are_made_once(sm_mask: Image) -> None:
    # The histogram and the IVH read one gather of the binned ROI values. Configurations
    # that share their masks reuse the GLDZM distance map of the last texture pass; new
    # masks (here after resegment) get their own map. The values are those of separate
    # runs, and run() keeps no map after it ends.
    from pictologics import pipeline as pipeline_module
    from pictologics.features import texture as texture_module

    rng = np.random.default_rng(8)
    image = Image(rng.normal(50.0, 20.0, (20, 20, 20)), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    resample = {"step": "resample", "params": {"new_spacing": (0.8, 0.8, 0.8)}}
    families = {"families": ["intensity", "histogram", "ivh", "gldzm"]}
    configs = {
        "fbn_8": [resample, {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}}],
        "fbn_16": [resample, {"step": "discretise", "params": {"method": "FBN", "n_bins": 16}}],
        "reseg": [
            resample,
            {"step": "resegment", "params": {"range_min": 20.0, "range_max": 80.0}},
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
        ],
    }
    pipeline = RadiomicsPipeline(load_standard=False, deduplicate=False)
    for name, steps in configs.items():
        pipeline.add_config(name, [*steps, {"step": "extract_features", "params": families}])
    with (
        patch.object(pipeline_module, "apply_mask", wraps=pipeline_module.apply_mask) as gathers,
        patch.object(
            texture_module,
            "_chamfer_distance_taxicab_numba",
            wraps=texture_module._chamfer_distance_taxicab_numba,
        ) as maps,
    ):
        together = pipeline.run(image, sm_mask, config_names=list(configs))
    assert gathers.call_count == 2 * len(configs)  # raw values and binned values
    assert maps.call_count == 2  # the resampled masks, and the resegmented ones
    assert pipeline._last_distance_map is None
    for name in configs:
        alone = pipeline.run(image, sm_mask, config_names=[name])
        assert together[name].equals(alone[name])


def test_nan_and_infinite_voxels_leave_the_intensity_mask(sm_mask: Image) -> None:
    # A NaN or infinite voxel has no intensity: each configuration leaves it out of the
    # intensity mask, with a warning, and keeps the morphological mask. The intensity
    # features are those of the finite ROI values. With no finite ROI voxel, the
    # configuration ends as an empty ROI.
    from pictologics.features.intensity import calculate_intensity_features

    rng = np.random.default_rng(2)
    clean = rng.normal(100.0, 20.0, (20, 20, 20))
    image = clean.copy()
    image[7, 8, 9] = np.nan
    image[10, 10, 10] = np.inf
    families = ["intensity", "morphology", "texture", "histogram", "ivh"]
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config(
        "c",
        [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
            {"step": "extract_features", "params": {"families": families}},
        ],
    )

    def run(array: np.ndarray) -> Any:
        return pipeline.run(Image(array, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)), sm_mask, "c")["c"]

    with pytest.warns(UserWarning, match="Left out 2 ROI voxels with a NaN or infinite"):
        out = run(image)
    assert not out.isna().any()
    assert out["volume_RNU0"] == run(clean)["volume_RNU0"]
    roi = sm_mask.array != 0
    roi[7, 8, 9] = roi[10, 10, 10] = False
    for key, value in calculate_intensity_features(clean[roi]).items():
        assert out[key] == value

    # A NaN outside the ROI leaves the ROI as it is, with no warning
    outside = clean.copy()
    outside[0, 0, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert run(outside).equals(run(clean))

    with pytest.warns(UserWarning, match="Left out 1,000 ROI voxels"):
        out = run(np.full(clean.shape, np.nan))
    assert out.isna().all() and pipeline._log[-1]["status"] == "empty_roi"


def test_discretise_cuts_the_arrays_to_the_roi_box() -> None:
    # Without a resample, the discretise step cuts the images and masks to the ROI box
    # (grown by the local intensity sphere when a later step reads it) before binning, in
    # each source mode; the features are those of the whole grid, bit for bit.
    from pictologics import pipeline as pipeline_module
    from pictologics.preprocessing import discretise_image

    rng = np.random.default_rng(7)
    arr = rng.normal(50.0, 20.0, (40, 36, 30))
    arr[:3] = -1000.0  # padding for the auto source mode
    labels = np.zeros(arr.shape, np.uint8)
    labels[12:20, 10:18, 8:14] = 1
    image = Image(arr, (1.0, 1.0, 1.5), (0.0, 0.0, 0.0))
    mask = Image(labels, image.spacing, image.origin)
    fbn = {"step": "discretise", "params": {"method": "FBN", "n_bins": 16}}
    fbs = {"step": "discretise", "params": {"method": "FBS", "bin_width": 10.0, "min_val": 0.0}}
    cutoffs = {"step": "discretise", "params": {"method": "FIXED_CUTOFFS", "cutoffs": [30.0, 60.0]}}
    configs = {
        "fbn": (fbn, ["intensity", "morphology", "texture", "histogram", "ivh"], "full_image"),
        "fbs_local": (
            fbs,
            ["intensity", "spatial_intensity", "local_intensity", "morphology", "texture"],
            "full_image",
        ),
        "cutoffs": (cutoffs, ["texture", "histogram"], "roi_only"),
        "auto": (fbn, ["intensity", "texture"], "auto"),
    }

    def run() -> dict[str, Any]:
        pipeline = RadiomicsPipeline(load_standard=False)
        for name, (step, families, mode) in configs.items():
            steps = [dict(step), {"step": "extract_features", "params": {"families": families}}]
            pipeline.add_config(name, steps, source_mode=mode)
        with pytest.warns(UserWarning, match="Auto-detected sentinel value -1000"):
            return pipeline.run(image, mask, config_names=list(configs))

    with patch.object(pipeline_module, "discretise_image", wraps=discretise_image) as spy:
        boxed = run()
    shapes = {call.args[0].array.shape for call in spy.call_args_list}
    assert (8, 8, 6) in shapes and arr.shape not in shapes  # the box, or the box grown
    with patch.object(pipeline_module, "_roi_reach", return_value=None):
        whole = run()
    for name in configs:
        assert boxed[name].equals(whole[name])


def test_roi_cuts_are_copies_kept_while_their_array_lives() -> None:
    # A cut is a copy, also when the box spans whole planes (a view would keep the whole
    # array alive); states that share an array share its cut, and the cut goes when the
    # array goes.
    import gc

    from pictologics.pipeline import PipelineState, _ArrayMemo, _cut_to_roi

    labels = np.zeros((8, 4, 5), dtype=np.uint8)
    labels[2:5] = 1  # whole planes
    grid = Image(np.arange(160.0).reshape(8, 4, 5), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    mask = Image(labels, grid.spacing, grid.origin)
    cuts: _ArrayMemo[Any, Any] = _ArrayMemo()
    boxes: _ArrayMemo[int, Any] = _ArrayMemo()
    states = [PipelineState(grid, grid, mask, mask, roi_reach=0.0) for _ in range(2)]
    for state in states:
        _cut_to_roi(state, cuts, boxes)
    first, second = states
    assert first.image.array.base is None and first.image.array.shape == (3, 4, 5)
    assert second.image.array is first.image.array and len(cuts) == 2  # image and mask
    assert first.grid_offset == (2, 0, 0) and first.grid_shape == (8, 4, 5)
    assert len(boxes) == 1  # the box of the mask, found once
    del grid, mask, states, first, second, labels
    gc.collect()
    assert not cuts and not boxes


def test_small_images_are_not_cut_before_binning(monkeypatch: pytest.MonkeyPatch) -> None:
    # Below _CUT_MIN_SIZE voxels, a discretise step bins the whole grid: finding and
    # cutting the box would cost more than it saves. A filter still filters the region,
    # down to _FILTER_REGION_MIN voxels.
    from pictologics import pipeline as pipeline_module
    from pictologics.filters import mean_filter
    from pictologics.preprocessing import discretise_image

    monkeypatch.setattr(pipeline_module, "_CUT_MIN_SIZE", 1 << 16)
    monkeypatch.setattr(pipeline_module, "_FILTER_REGION_MIN", 1 << 12)
    rng = np.random.default_rng(23)
    image = Image(rng.normal(50.0, 20.0, (30, 30, 30)), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    labels = np.zeros(image.array.shape, np.uint8)
    labels[10:15, 10:15, 10:15] = 1
    mask = Image(labels, image.spacing, image.origin)
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("c", [
        {"step": "filter", "params": {"type": "mean", "support": 3}},
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
        {"step": "extract_features", "params": {"families": ["intensity", "texture"]}},
    ])  # fmt: skip
    with (
        patch.object(pipeline_module, "mean_filter", wraps=mean_filter) as filters,
        patch.object(pipeline_module, "discretise_image", wraps=discretise_image) as bins,
    ):
        pipeline.run(image, mask, config_names=["c"])
    assert filters.call_args.args[0].size < image.array.size
    assert bins.call_args.args[0].array.shape == image.array.shape
    # Below _FILTER_REGION_MIN voxels, the filter takes the whole image too
    tiny = Image(image.array[:15, :15, :15], image.spacing, image.origin)
    with patch.object(pipeline_module, "mean_filter", wraps=mean_filter) as filters:
        pipeline.run(tiny, Image(labels[:15, :15, :15], image.spacing, image.origin), "c")
    assert filters.call_args.args[0].shape == tiny.array.shape


def test_resample_computes_only_the_roi_region() -> None:
    # When only ROI-local steps follow, the resample computes the region around the ROI
    # box, grown by the local intensity sphere when a later step needs it; a filter needs
    # the whole grid. The features are those of the whole grid, bit for bit (the mesh is
    # made in the index frame of the whole grid).
    from pictologics import pipeline as pipeline_module
    from pictologics.preprocessing import resample_image

    rng = np.random.default_rng(6)
    image = Image(rng.normal(50.0, 20.0, (24, 24, 24)), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
    roi = np.zeros((24, 24, 24), dtype=np.uint8)
    roi[10:14, 10:14, 10:14] = 1
    mask = Image(roi, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
    resample = {"step": "resample", "params": {"new_spacing": (1.5, 1.5, 1.5)}}
    disc = {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}}
    families = ["intensity", "morphology", "texture", "histogram", "ivh"]
    configs = {
        "plain": [resample, disc, {"step": "extract_features", "params": {"families": families}}],
        "local": [
            resample,
            disc,
            {
                "step": "extract_features",
                "params": {"families": ["intensity"], "include_local_intensity": True},
            },
        ],
        "filter": [
            resample,
            {"step": "filter", "params": {"type": "mean", "support": 3}},
            disc,
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ],
    }
    pipeline = RadiomicsPipeline(load_standard=False, deduplicate=False)
    for name, steps in configs.items():
        pipeline.add_config(name, steps)
    with patch.object(pipeline_module, "resample_image", wraps=resample_image) as calls:
        boxed = pipeline.run(image, mask, config_names=list(configs))
    regions = {}
    for call in calls.call_args_list:
        if call.args[0].array.shape == image.array.shape:  # the image (and masks) of a config
            regions.setdefault(call.kwargs.get("region"), 0)
    sizes = sorted(
        (r[0].stop - r[0].start) if r is not None else 32 for r in regions
    )  # 32 = the whole new grid
    assert len(regions) == 3 and sizes[0] < sizes[1] < sizes[2] == 32
    with patch.object(pipeline_module, "_roi_reach", return_value=None):
        whole = pipeline.run(image, mask, config_names=list(configs))
    for name in configs:
        assert boxed[name].equals(whole[name])


def test_resample_stops_when_its_grid_cannot_fit_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Before a resample, the pipeline checks that the new grid, or the ROI region of it,
    # fits in the memory of the computer at _GRID_BYTES_PER_VOXEL bytes per voxel. A grid
    # that does not fit gives the config a MemoryError in its log entry and NaN features.
    from pictologics import pipeline as pipeline_module

    image = Image(np.ones((10, 10, 10)), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    roi = np.zeros((10, 10, 10), dtype=np.uint8)
    roi[4:6, 4:6, 4:6] = 1
    mask = Image(roi, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config(
        "c",
        [
            {"step": "resample", "params": {"new_spacing": (0.5, 0.5, 0.5)}},
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ],
    )
    # the whole grid: 20 x 20 x 20 voxels x 24 bytes = 192,000 bytes
    monkeypatch.setattr(pipeline_module, "_physical_memory", lambda: 100_000)
    assert pipeline.run(image, None, config_names=["c"])["c"].isna().all()
    entry = pipeline._log[-1]
    assert entry["status"] == "error" and entry["failed_step"]["step"] == "resample"
    assert entry["error"].startswith(
        "Resampling to (0.5, 0.5, 0.5) mm makes a grid of 20 x 20 x 20 voxels."
    )
    # with an ROI, only its region is resampled, and the region fits
    assert pipeline.run(image, mask, config_names=["c"])["c"].notna().all()
    assert pipeline._log[-1]["status"] == "completed"


def test_physical_memory_on_posix_and_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes
    import sys

    from pictologics.pipeline import _physical_memory

    _physical_memory.cache_clear()
    assert _physical_memory() == os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

    def memory_status(status: Any) -> int:
        status._obj.total = 16 << 30
        return 1

    kernel32 = MagicMock()
    kernel32.GlobalMemoryStatusEx.side_effect = memory_status
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", MagicMock(kernel32=kernel32), raising=False)
    _physical_memory.cache_clear()
    try:
        assert _physical_memory() == 16 << 30
        # MEMORYSTATUSEX: 2 numbers of 4 bytes and 7 of 8 bytes
        assert kernel32.GlobalMemoryStatusEx.call_args.args[0]._obj.length == 64
    finally:
        _physical_memory.cache_clear()


def test_texture_distances_in_the_pipeline() -> None:
    # texture_matrix_params gives the GLCM, NGTDM and NGLDM distances to the texture
    # features; bad options are config problems.
    import re

    from pictologics.features.texture import calculate_all_texture_features
    from pictologics.preprocessing import discretise_image

    rng = np.random.default_rng(9)
    image = Image(rng.normal(40.0, 20.0, (14, 12, 10)), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    mask = Image(np.ones((14, 12, 10), dtype=np.uint8), image.spacing, image.origin)
    options = {"glcm_distance": 2, "ngtdm_distance": 3, "ngldm_distance": 2.0, "ngldm_alpha": 1}
    pipeline = RadiomicsPipeline()
    pipeline.add_config("far", [
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
        {"step": "extract_features", "params": {"families": ["texture"], "texture_matrix_params": options}},
    ])  # fmt: skip
    result = pipeline.run(image, mask, config_names=["far"])["far"]
    disc = discretise_image(image, method="FBN", n_bins=8, roi_mask=mask)
    assert isinstance(disc, Image)
    expected = calculate_all_texture_features(
        disc.array,
        mask.array,
        8,
        ngldm_alpha=1,
        glcm_distance=2,
        ngtdm_distance=3,
        ngldm_distance=2,
    )
    for key in ("contrast_ACUI", "coarseness_QCDE", "low_dependence_emphasis_SODN"):
        assert np.isclose(result[key], expected[key], rtol=1e-12), key
    bad = {
        "unknown option 'glcm_distnce' (did you mean 'glcm_distance'?)": {"glcm_distnce": 2},
        "glcm_distance must be a whole number of 1 or more, not 0": {"glcm_distance": 0},
        "ngtdm_distance must be a whole number of 1 or more, not 1.5": {"ngtdm_distance": 1.5},
        "ngldm_alpha must be 0 or more, not -1": {"ngldm_alpha": -1},
    }
    for message, params in bad.items():
        with pytest.raises(ValueError, match=re.escape(message)):
            pipeline.add_config("bad", [
                {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
                {"step": "extract_features", "params": {"families": ["texture"], "texture_matrix_params": params}},
            ])  # fmt: skip


def test_gaussian_filter_and_padding_value_in_the_pipeline() -> None:
    # A gaussian filter step (with the spacing of the image) and a padding value with the
    # constant boundary give the values of the filters; bad options are config problems.
    import re

    from pictologics.filters import gaussian_filter, mean_filter

    rng = np.random.default_rng(13)
    image = Image(rng.normal(40.0, 20.0, (16, 14, 12)), (0.8, 1.0, 2.0), (0.0, 0.0, 0.0))
    mask = Image(np.ones((16, 14, 12), dtype=np.uint8), image.spacing, image.origin)
    pipeline = RadiomicsPipeline()
    steps = {
        "gaussian": {"type": "gaussian", "sigma_mm": 2.0},
        "padded": {"type": "mean", "support": 3, "boundary": "constant", "padding_value": -1000.0},
    }
    for name, params in steps.items():
        pipeline.add_config(name, [
            {"step": "filter", "params": params},
            {"step": "extract_features", "params": {"families": ["intensity"]}},
        ])  # fmt: skip
    results = pipeline.run(image, mask, config_names=list(steps))
    expected = {
        "gaussian": gaussian_filter(image.array, 2.0, image.spacing, boundary="mirror"),
        "padded": mean_filter(image.array, 3, "constant", padding_value=-1000.0),
    }
    for name, response in expected.items():
        mean = response.astype(np.float64).mean()
        assert np.isclose(results[name]["mean_intensity_Q4LE"], mean, rtol=1e-9), name
    bad = {
        "padding_value 5 needs boundary 'constant' (or 'zero'), not 'the default'": {"type": "mean", "padding_value": 5},
        "padding_value must be a number, not 'x'": {"type": "mean", "boundary": "constant", "padding_value": "x"},
        "unknown gabor response 'phase'": {"type": "gabor", "sigma_mm": 1.0, "lambda_mm": 2.0, "response": "phase"},
    }  # fmt: skip
    for message, params in bad.items():
        with pytest.raises(ValueError, match=re.escape(message)):
            pipeline.add_config("bad", [{"step": "filter", "params": params}])


def _grow_case() -> tuple[Image, Image]:
    """A random image and a box mask (0.8 mm voxels)."""
    rng = np.random.default_rng(5)
    image = Image(rng.normal(0.0, 10.0, (20, 20, 20)), (0.8, 0.8, 0.8), (0.0, 0.0, 0.0))
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[8:12, 7:13, 8:11] = 1
    return image, Image(mask, image.spacing, image.origin)


def test_grow_mask_step_gives_the_mask_of_grow_mask() -> None:
    # The step gives the features of the grow_mask mask; with apply_to "intensity" the
    # morphology keeps the mask; after a resample, the region that the later steps read
    # holds the ring, as the whole grid does.
    from pictologics import pipeline as pipeline_module
    from pictologics.preprocessing import grow_mask

    image, mask = _grow_case()
    ring = {"step": "grow_mask", "params": {"from_mm": 0, "to_mm": 2}}
    intensity = {"step": "extract_features", "params": {"families": ["intensity"]}}
    shape = {"step": "extract_features", "params": {"families": ["morphology"]}}
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("ring", [ring, intensity, shape])
    pipeline.add_config("core", [{"step": "grow_mask", "params": {"to_mm": -0.8}}, intensity])
    pipeline.add_config("intensity_ring", [
        {"step": "grow_mask", "params": {"from_mm": 0, "to_mm": 2, "apply_to": "intensity"}},
        intensity,
        shape,
    ])  # fmt: skip
    pipeline.add_config("intensity", [intensity, shape])
    results = pipeline.run(image, mask, config_names=["ring", "core", "intensity_ring"])
    plain = {
        name: pipeline.run(image, grow_mask(mask, *distances), config_names=["intensity"])
        for name, distances in (("ring", (2, 0)), ("core", (-0.8,)), ("mask", (0,)))
    }
    assert results["ring"].equals(plain["ring"]["intensity"])
    core = plain["core"]["intensity"]
    assert results["core"].equals(core[[name for name in core.index if name in results["core"]]])
    volume = "volume_voxel_counting_YEKZ"
    assert results["intensity_ring"][volume] == plain["mask"]["intensity"][volume]
    assert (
        results["intensity_ring"]["mean_intensity_Q4LE"] == results["ring"]["mean_intensity_Q4LE"]
    )
    resampled = RadiomicsPipeline(load_standard=False)
    resampled.add_config("c", [
        {"step": "resample", "params": {"new_spacing": (0.7, 0.7, 0.7)}},
        ring,
        {"step": "extract_features", "params": {"families": ["intensity", "morphology"]}},
    ])  # fmt: skip
    boxed = resampled.run(image, mask, config_names=["c"])["c"]
    with patch.object(pipeline_module, "_roi_reach", return_value=None):
        whole = resampled.run(image, mask, config_names=["c"])["c"]
    assert boxed.equals(whole)


def test_roi_reach_adds_the_growth_of_later_grow_steps() -> None:
    from pictologics.pipeline import _LOCAL_PEAK_RADIUS_MM, _roi_reach

    def grow(to_mm: Any) -> dict[str, Any]:
        return {"step": "grow_mask", "params": {"to_mm": to_mm}}

    local = {"step": "extract_features", "params": {"families": ["local_intensity"]}}
    plain = {"step": "extract_features", "params": {"families": ["intensity"]}}
    assert _roi_reach([grow(2.0), plain, grow(1.5), plain]) == 3.5
    assert _roi_reach([grow(2.0), local, grow(1.0)]) == 2.0 + _LOCAL_PEAK_RADIUS_MM
    assert _roi_reach([grow(-1.0), grow("x"), {"step": "grow_mask"}, plain]) == 0.0
    assert _roi_reach([local]) == _LOCAL_PEAK_RADIUS_MM


def test_grow_mask_step_problems_and_image_data() -> None:
    # add_config checks the distances and nearest_roi; grown voxels without image data
    # (a sentinel value) leave the mask, as after a resample.
    from pictologics.preprocessing import grow_mask

    for params, message in (
        ({}, "missing parameter 'to_mm'"),
        ({"to_mm": "2"}, "to_mm must be a finite number (mm), not '2'"),
        ({"to_mm": 1.0, "from_mm": 1.0}, "from_mm (1.0) must be below to_mm (1.0)"),
        ({"to_mm": 1.0, "nearest_roi": "yes"}, "nearest_roi must be True or False, not 'yes'"),
        ({"to_mm": 1.0, "distance_mm": 1.0}, "unknown parameter 'distance_mm'"),
    ):
        problems = _steps_problems([{"step": "grow_mask", "params": params}])
        assert len(problems) == 1 and problems[0].startswith(f"step 0 (grow_mask): {message}")
    image, mask = _grow_case()
    values = image.array.copy()
    values[:, :, :9] = -2048.0  # no image data next to the mask
    image = Image(values, image.spacing, image.origin)
    pipeline = RadiomicsPipeline(load_standard=False)
    steps = [
        {"step": "grow_mask", "params": {"from_mm": 0, "to_mm": 2}},
        {"step": "extract_features", "params": {"families": ["intensity", "morphology"]}},
    ]
    pipeline.add_config("sentinel", steps, source_mode="auto", sentinel_value=-2048.0)
    pipeline.add_config("all", steps)
    results = pipeline.run(image, mask, config_names=["sentinel", "all"])
    ring = grow_mask(mask, 2, 0).array != 0
    voxel = float(np.prod(image.spacing))
    volume = "volume_voxel_counting_YEKZ"
    assert results["all"][volume] == pytest.approx(ring.sum() * voxel)
    assert results["sentinel"][volume] == pytest.approx((ring & (values != -2048.0)).sum() * voxel)


def test_run_rois_gives_each_grown_voxel_to_its_nearest_roi() -> None:
    # With nearest_roi, the rings of two touching ROIs share out the ring of both: no
    # voxel in two rings, none in an ROI. Without it, the rings overlap. run() has one ROI,
    # so nearest_roi changes nothing there. After a resample, the rings stay apart.
    rng = np.random.default_rng(6)
    image = Image(rng.normal(0.0, 1.0, (30, 30, 30)), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0))
    labels = np.zeros((30, 30, 30), dtype=np.uint8)
    labels[10:20, 10:15, 10:20] = 1
    labels[10:20, 15:20, 10:20] = 2
    label_map = Image(labels, image.spacing, image.origin)
    both = Image((labels != 0).astype(np.uint8), image.spacing, image.origin)
    volume = "volume_voxel_counting_YEKZ"
    extract = {"step": "extract_features", "params": {"families": ["morphology"]}}
    pipeline = RadiomicsPipeline(load_standard=False)
    for name, nearest in (("shared", True), ("own", False)):
        grow = {"step": "grow_mask", "params": {"from_mm": 0, "to_mm": 2, "nearest_roi": nearest}}
        pipeline.add_config(name, [grow, extract])
        pipeline.add_config(f"{name}_resampled", [
            {"step": "resample", "params": {"new_spacing": (0.7, 0.7, 0.7)}},
            grow,
            extract,
        ])  # fmt: skip
    names = ["shared", "own", "shared_resampled", "own_resampled"]
    rois = pipeline.run_rois(image, label_map, config_names=names)
    whole = pipeline.run(image, both, config_names=names)
    for name in names:
        assert whole[name].equals(whole[name.replace("shared", "own")])
    total = {name: rois["1"][name][volume] + rois["2"][name][volume] for name in names}
    assert total["shared"] == whole["shared"][volume]
    assert total["own"] > 1.3 * whole["own"][volume]
    assert 0.9 * whole["shared_resampled"][volume] < total["shared_resampled"]
    assert total["shared_resampled"] <= whole["shared_resampled"][volume]
    assert pipeline.run_rois(image, label_map, config_names=["own"])["1"]["own"].equals(
        rois["1"]["own"]
    )


def test_describe_features_records_grow_steps() -> None:
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("c", [
        {"step": "grow_mask", "params": {"from_mm": 0, "to_mm": 3, "apply_to": "intensity"}},
        {"step": "extract_features", "params": {"families": ["intensity"]}},
    ])  # fmt: skip
    row = pipeline.describe_features().iloc[0]
    assert row["is_mask_grown"]
    assert row["grow_mask_apply_to"] == "intensity"
    assert json.loads(row["grow_mask_params"]) == [
        {"params": {"apply_to": "intensity", "from_mm": 0, "to_mm": 3}, "step_index": 1}
    ]


def test_normalise_step_maps_the_image_by_its_region() -> None:
    # The step gives the image of normalise_image (statistics of the ROI, or of the whole
    # image without the voxels of no data); the log has the center and the scale; the
    # step before a whole-image normalise keeps the whole grid.
    from pictologics.pipeline import _needs_full_grid
    from pictologics.preprocessing import normalise_image

    image, mask = _grow_case()
    values = image.array.copy()
    values[:, :, :3] = -2048.0  # no image data
    image = Image(values + 300.0, image.spacing, image.origin)
    intensity = {"step": "extract_features", "params": {"families": ["intensity"]}}
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("roi", [
        {"step": "normalise", "params": {"method": "zscore", "region": "roi"}},
        intensity,
    ])  # fmt: skip
    pipeline.add_config(
        "image",
        [
            {"step": "normalise", "params": {"method": "percentile", "region": "image", "percentiles": [2, 98], "range_min": 0}},
            intensity,
        ],
        source_mode="auto",
        sentinel_value=-1748.0,
    )  # fmt: skip
    pipeline.add_config("plain", [intensity])
    results = pipeline.run(image, mask, config_names=["roi", "image"])
    assert abs(results["roi"]["mean_intensity_Q4LE"]) < 1e-9
    assert results["roi"]["intensity_variance_ECT3"] == pytest.approx(1.0, rel=1e-9)
    data = Image(values + 300.0, image.spacing, image.origin)
    valid = Image((values != -2048.0).astype(np.uint8), image.spacing, image.origin)
    expected = normalise_image(data, "percentile", valid, (2, 98), range_min=0)
    plain = pipeline.run(expected, mask, config_names=["plain"])["plain"]
    assert results["image"]["mean_intensity_Q4LE"] == pytest.approx(
        plain["mean_intensity_Q4LE"], rel=1e-12
    )
    log = {entry["config_name"]: entry["steps_executed"][0] for entry in pipeline.get_log()}
    kept = (values + 300.0)[(values != -2048.0)]
    assert log["image"]["center_effective"] == pytest.approx(np.percentile(kept, 2))
    assert log["roi"]["scale_effective"] > 0
    assert _needs_full_grid([{"step": "normalise", "params": {"region": "image"}}])
    assert not _needs_full_grid([{"step": "normalise", "params": {"region": "roi"}}])


def test_normalise_step_problems() -> None:
    # add_config checks the options; a normalise step cancels the FBS start, as a filter.
    normalise = {"step": "normalise", "params": {"method": "zscore", "region": "roi"}}
    for params, message in (
        ({"region": "roi"}, "missing parameter 'method'"),
        ({"method": "zcore", "region": "roi"}, "unknown method 'zcore' (did you mean 'zscore'?)"),
        ({"method": "zscore", "region": "body"}, "unknown region 'body'"),
        ({"method": "zscore", "region": "roi", "percentiles": [1, 99]}, "percentiles need method 'percentile'"),
        ({"method": "percentile", "region": "roi", "percentiles": [99, 1]}, "percentiles must be two numbers"),
    ):  # fmt: skip
        problems = _steps_problems([{"step": "normalise", "params": params}])
        assert len(problems) == 1 and problems[0].startswith(f"step 0 (normalise): {message}")
    discretise = {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}}
    assert _steps_problems([discretise, normalise]) == [
        "step 1 (normalise): a normalise step must come before the discretise step"
    ]
    reseg = {"step": "resegment", "params": {"range_min": 0, "range_max": 500}}
    fbs = {"step": "discretise", "params": {"method": "FBS", "bin_width": 0.25}}
    assert _steps_problems([reseg, fbs]) == []
    assert "after the last filter or normalise step" in _steps_problems([reseg, normalise, fbs])[0]
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config(
        "c", [normalise, {"step": "extract_features", "params": {"families": ["intensity"]}}]
    )
    row = pipeline.describe_features().iloc[0]
    assert row["is_normalised"] and row["normalisation_method"] == "zscore"


def test_image_options_load_the_image_path() -> None:
    # run, run_rois and run_batch pass image_options to load_image for an image path
    # (here the second volume of a 4D NIfTI file); the log and the batch record hold
    # them, and a batch case runs again when its options change.
    import tempfile
    from pathlib import Path

    import nibabel as nib

    rng = np.random.default_rng(4)
    volumes = rng.normal(40.0, 20.0, (10, 10, 10, 2))
    volumes[..., 1] = volumes[..., 0] + 100.0
    labels = np.zeros((10, 10, 10), dtype=np.uint8)
    labels[2:6, 2:6, 2:6] = 1
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "image.nii.gz"
        nib.save(nib.Nifti1Image(volumes, np.eye(4)), path)
        nib.save(nib.Nifti1Image(labels, np.eye(4)), Path(folder) / "mask.nii.gz")
        mask = str(Path(folder) / "mask.nii.gz")
        pipeline = RadiomicsPipeline(load_standard=False)
        pipeline.add_config(
            "c", [{"step": "extract_features", "params": {"families": ["intensity"]}}]
        )
        first = pipeline.run(str(path), mask, config_names=["c"])["c"]["mean_intensity_Q4LE"]
        second = pipeline.run(
            str(path), mask, config_names=["c"], image_options={"dataset_index": 1}
        )
        assert second["c"]["mean_intensity_Q4LE"] == pytest.approx(first + 100.0)
        assert pipeline.get_log()[-1]["image_options"] == {"dataset_index": 1}
        rois = pipeline.run_rois(
            str(path), mask, config_names=["c"], image_options={"dataset_index": 1}
        )
        assert rois["1"]["c"].equals(second["c"])
        with pytest.raises(ValueError, match="image_options apply to an image path"):
            pipeline.run(
                Image(volumes[..., 0], (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
                image_options={"dataset_index": 1},
            )
        case = {
            "subject_id": "s",
            "image": str(path),
            "mask": mask,
            "image_options": {"dataset_index": 1},
        }
        out = Path(folder) / "batch"
        table = pipeline.run_batch([case], out, config_names=["c"], show_progress=False)
        record = json.loads((out / "cases" / "s.json").read_text())
        assert record["image_options"] == {"dataset_index": 1}
        assert table.loc[0, "c__mean_intensity_Q4LE"] == pytest.approx(first + 100.0)
        with patch.object(RadiomicsPipeline, "_run_case", side_effect=AssertionError("ran")):
            pipeline.run_batch([case], out, config_names=["c"], show_progress=False)  # resumed
        again = pipeline.run_batch(
            [{**case, "image_options": None}], out, config_names=["c"], show_progress=False
        )
        assert again.loc[0, "c__mean_intensity_Q4LE"] == pytest.approx(first)


def test_run_batch_runs_label_map_cases_by_roi(tmp_path: Any) -> None:
    # A case with rois (and labels) runs run_rois: one row for each ROI, with its name in
    # roi, the results of run_rois; a failed one gives one row; the record holds the
    # label map and the labels, so a resumed batch skips the case and other labels run
    # it again. A case cannot give a mask and rois.
    import nibabel as nib

    cases = _batch_cases(tmp_path)
    labels = np.zeros((12, 12, 12), dtype=np.uint8)
    labels[3:6, 3:9, 3:9] = 1
    labels[6:9, 3:9, 3:9] = 2
    nib.save(nib.Nifti1Image(labels, np.eye(4)), tmp_path / "labels.nii.gz")
    rois = str(tmp_path / "labels.nii.gz")
    batch = [
        {"subject_id": "map", "image": cases[0]["image"], "rois": rois, "labels": {"low": 1, "high": 2}},
        cases[1],
        {"subject_id": "lost", "image": str(tmp_path / "none.nii.gz"), "rois": rois},
    ]  # fmt: skip
    pipeline = _batch_test_pipeline()
    table = pipeline.run_batch(batch, tmp_path / "out", show_progress=False)
    assert table["subject_id"].tolist() == ["map", "map", "p/1", "lost"]
    assert table["roi"].tolist()[:2] == ["low", "high"] and table["roi"].isna().tolist()[2:] == [
        True,
        True,
    ]
    assert table["status"].tolist() == ["incomplete", "incomplete", "incomplete", "failed"]
    assert "ROI low, empty: " in table.loc[0, "error"]
    with pytest.warns(UserWarning, match="No sentinel value auto-detected"):
        direct = pipeline.run_rois(cases[0]["image"], rois, labels={"low": 1, "high": 2})
    for row, roi in ((0, "low"), (1, "high")):
        assert (
            table.loc[row, "first__mean_intensity_Q4LE"]
            == direct[roi]["first"]["mean_intensity_Q4LE"]
        )
    record = json.loads((tmp_path / "out" / "cases" / "map.json").read_text())
    assert (record["rois"], record["labels"], record["mask"]) == (rois, {"low": 1, "high": 2}, None)
    with patch.object(RadiomicsPipeline, "_run_case", wraps=pipeline._run_case) as spy:
        again = pipeline.run_batch(batch[:2], tmp_path / "out", show_progress=False)
        assert spy.call_count == 0
        pipeline.run_batch([{**batch[0], "labels": [1]}], tmp_path / "out", show_progress=False)
        assert spy.call_count == 1
    assert again.equals(table.iloc[:3])
    with pytest.raises(ValueError, match="gives a mask and rois"):
        pipeline.run_batch([{**cases[0], "rois": rois}], tmp_path / "x", show_progress=False)


def test_saved_configs_keep_numpy_numbers_of_tuples(tmp_path: Any) -> None:
    # The numpy numbers inside a tuple (a spacing) become plain numbers, so YAML and JSON
    # files load back with the same configuration; an enum boundary takes a padding value
    from pictologics.filters import BoundaryCondition

    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("c", [
        {"step": "resample", "params": {"new_spacing": (np.float64(1.0), np.int64(1), 1)}},
        {"step": "filter", "params": {"type": "mean", "support": 3, "boundary": BoundaryCondition.ZERO, "padding_value": -1.0}},
        {"step": "extract_features", "params": {"families": ["intensity"]}},
    ])  # fmt: skip
    for suffix in ("yaml", "json"):
        pipeline.save_configs(tmp_path / f"c.{suffix}")
        loaded = RadiomicsPipeline.load_configs(tmp_path / f"c.{suffix}", load_standard=False)
        spacing = loaded.get_config("c")[0]["params"]["new_spacing"]
        assert list(spacing) == [1.0, 1, 1] and all(type(v) in (int, float) for v in spacing)


def _scan_spy(scanned: list[Any]) -> Any:
    """A stand-in for the box scan of the pipeline that keeps each array it scans (alive,
    so that no other array gets its id)."""
    from pictologics import pipeline as pipeline_module

    scan = pipeline_module.compute_nonzero_bbox

    def counted(array: np.ndarray) -> Any:
        scanned.append(array)
        return scan(array)

    return counted


def test_a_run_scans_each_mask_once() -> None:
    # A run scans each mask array once for its nonzero box: the fraction check, the ROI
    # checks, the resample and filter regions, the box cut, the finite intensity mask and
    # the extraction share the box. The results are those of a scan at each use.
    from pictologics import pipeline as pipeline_module

    rng = np.random.default_rng(31)
    values = rng.normal(40.0, 15.0, (48, 48, 40))
    values[20, 20, 20] = np.nan  # an ROI voxel: the finite intensity mask
    image = Image(values, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    roi = np.zeros(values.shape)  # a float mask: the fraction check reads its box
    roi[12:30, 14:34, 10:28] = 1.0
    mask = Image(roi, image.spacing, image.origin)
    discretise = {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}}
    families = ["intensity", "morphology", "texture", "histogram"]
    extract = {"step": "extract_features", "params": {"families": families}}
    configs = {
        "resampled": [{"step": "resample", "params": {"new_spacing": (1.2, 1.2, 1.2)}}],
        "filtered": [{"step": "filter", "params": {"type": "mean", "support": 3}}],
        "resegmented": [
            {"step": "resegment", "params": {"range_min": 20.0, "apply_to": "intensity"}}
        ],
    }
    pipeline = RadiomicsPipeline(load_standard=False)
    for name, steps in configs.items():
        pipeline.add_config(name, [*steps, discretise, extract])
    once: list[Any] = []
    each: list[Any] = []
    with pytest.warns(UserWarning, match="NaN or infinite"):
        with patch.object(pipeline_module, "compute_nonzero_bbox", _scan_spy(once)):
            memo = pipeline.run(image, mask, config_names=list(configs))
        assert not pipeline._mask_boxes  # a run keeps no box after it ends
        scan = _scan_spy(each)
        with patch.object(pipeline_module, "_mask_box", lambda array, boxes: scan(array)):
            scans = pipeline.run(image, mask, config_names=list(configs))
    assert len({id(array) for array in once}) == len(once) < len(each)
    for name in configs:
        assert list(memo[name].index) == list(scans[name].index)
        assert memo[name].to_numpy().tobytes() == scans[name].to_numpy().tobytes()


def test_runs_do_not_collect_finalizers_on_their_arrays() -> None:
    # A run drops the weak finalizers of its mask boxes and ROI cuts when it ends, so the
    # arrays of many runs do not collect one finalizer for each run.
    import weakref

    registry = weakref.finalize._registry

    def finalizers(array: np.ndarray) -> int:
        return sum(1 for f in list(registry) if (info := f.peek()) and info[0] is array)

    rng = np.random.default_rng(35)
    image = Image(rng.normal(40.0, 15.0, (30, 32, 28)), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    roi = np.zeros(image.array.shape)  # a float mask: the fraction check reads its box
    roi[8:20, 10:24, 6:18] = 1.0
    mask = Image(roi, image.spacing, image.origin)
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("c", [
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},  # cuts the arrays
        {"step": "extract_features", "params": {"families": ["intensity", "glcm"]}},
    ])  # fmt: skip
    counts = []
    for _ in range(3):
        pipeline.run(image, mask, config_names=["c"])
        counts.append((finalizers(image.array), finalizers(mask.array)))
    assert counts == [(1, 0)] * 3  # the image keeps the finalizer of its NaN check


def test_run_rois_scans_the_map_once_and_no_label_buffer() -> None:
    # run_rois finds the box of each label with one scan of the map and gives each run the
    # box of its label, so no run scans the label buffer, and each run scans each of its
    # masks once. The results are those of a scan at each use.
    from pictologics import pipeline as pipeline_module

    rng = np.random.default_rng(32)
    image = Image(rng.normal(40.0, 15.0, (30, 32, 28)), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    labels = np.zeros(image.array.shape, dtype=np.uint16)
    labels[3:9, 4:12, 5:10] = 2
    labels[15:24, 18:30, 12:25] = 5
    label_map = Image(labels, image.spacing, image.origin)
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("c", [
        {"step": "resample", "params": {"new_spacing": (0.9, 0.9, 0.9)}},
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
        {"step": "extract_features", "params": {"families": ["intensity", "morphology", "glcm"]}},
    ])  # fmt: skip
    scanned: list[Any] = []
    runs: list[tuple[Any, int]] = []  # the mask buffer of each run, and the scans before it
    run_loaded = RadiomicsPipeline._run_loaded

    def spy(self: RadiomicsPipeline, image: Image, mask: Image, *args: Any) -> Any:
        runs.append((mask.array, len(scanned)))
        return run_loaded(self, image, mask, *args)

    with (
        patch.object(pipeline_module, "compute_nonzero_bbox", _scan_spy(scanned)),
        patch.object(RadiomicsPipeline, "_run_loaded", spy),
    ):
        memo = pipeline.run_rois(image, label_map, labels=[5, 2, 9], config_names=["c"])
    assert scanned[0] is labels and all(array is not labels for array in scanned[1:])
    stops = [start for _, start in runs[1:]] + [len(scanned)]
    for (buffer, start), stop in zip(runs, stops, strict=True):
        assert all(array is not buffer for array in scanned[start:stop])
        assert len({id(array) for array in scanned[start:stop]}) == stop - start
    scan = pipeline_module.compute_nonzero_bbox
    with patch.object(pipeline_module, "_mask_box", lambda array, boxes: scan(array)):
        scans = pipeline.run_rois(image, label_map, labels=[5, 2, 9], config_names=["c"])
    assert list(memo) == ["5", "2", "9"] and memo["9"]["c"].isna().all()
    for name in memo:
        assert list(memo[name]["c"].index) == list(scans[name]["c"].index)
        assert memo[name]["c"].to_numpy().tobytes() == scans[name]["c"].to_numpy().tobytes()


def test_label_boxes_are_those_of_find_objects() -> None:
    # The label boxes come from find_objects on the nonzero box of the map only, shifted
    # back: the same boxes (slices of Python ints) for gaps in the label numbers, an absent
    # label, labels on the edges, every label type, one voxel, an empty map and a 2D map.
    from scipy import ndimage

    from pictologics.pipeline import _label_boxes

    labels = np.zeros((9, 10, 11), dtype=np.int64)
    labels[0, 0, 0] = 1  # a corner; label 2 is absent
    labels[3:5, 2:9, 4:6] = 3
    labels[4:9, 0, 3] = 4  # on a face
    labels[2:4, 5:7, 6:9] = 4
    labels[8, 9, 10] = 7  # the other corner
    inside = np.zeros((9, 10, 11), dtype=np.uint8)
    inside[2:5, 3:7, 4:6] = 3  # a label away from the edges
    one = np.zeros((4, 5, 6), dtype=np.uint8)
    one[2, 3, 4] = 9
    maps = [labels.astype(dtype) for dtype in (np.uint8, np.uint16, np.int32)]
    maps += [labels.astype(np.float64).astype(np.int64), inside, one]
    maps += [np.zeros((4, 5, 6), dtype=np.uint8), labels[:, :, 0]]
    for array in maps:
        boxes = _label_boxes(array)
        assert boxes == ndimage.find_objects(array)
        assert all(type(s.start) is int for box in boxes if box is not None for s in box)
    # An empty map stops at the label check, as before
    pipeline = RadiomicsPipeline(load_standard=False)
    empty = Image(np.zeros((0, 4, 4)), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="zero-size array"):
        pipeline.run_rois(empty, Image(np.zeros((0, 4, 4), np.uint8), empty.spacing, empty.origin))


def test_a_second_run_on_an_image_does_not_check_it_for_nan_again(sm_mask: Image) -> None:
    # The NaN check of an image array runs once while the array lives: a second run on the
    # same image does not read it again, with the same results; another image is read.
    from pictologics import preprocessing

    image = Image(np.random.default_rng(34).normal(5.0, 2.0, (20, 20, 20)), (1.0,) * 3, (0.0,) * 3)
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("c", [{"step": "extract_features", "params": {"families": ["intensity"]}}])
    with (
        patch.object(preprocessing, "_FINITE_PARALLEL_MIN", 0),
        patch.object(
            preprocessing, "_nonfinite_blocks_numba", wraps=preprocessing._nonfinite_blocks_numba
        ) as kernel,
    ):
        first = pipeline.run(image, sm_mask, config_names=["c"])["c"]
        second = pipeline.run(image, sm_mask, config_names=["c"])["c"]
        assert kernel.call_count == 1
        copied = Image(image.array.copy(), image.spacing, image.origin)
        assert pipeline.run(copied, sm_mask, config_names=["c"])["c"].equals(first)
        assert kernel.call_count == 2
    assert second.equals(first)


def _morphology_case() -> tuple[Image, Image]:
    """A 16^3 image and an irregular ROI with many convex hull vertices."""
    rng = np.random.default_rng(33)
    image = Image(rng.normal(50.0, 20.0, (16, 16, 16)), (1.0, 0.9, 1.1), (0.0, 0.0, 0.0))
    index = np.indices(image.array.shape)
    roi = sum(((index[k] - 7.5) / (4.0 + k)) ** 2 for k in range(3)) <= 1.0
    roi[3, 7, 7] = roi[12, 8, 6] = True
    return image, Image(roi.astype(np.uint8), image.spacing, image.origin)


def test_morphology_worker_gives_the_results_of_one_thread() -> None:
    # With other families in the pass, the convex hull and the MVEE run in the worker
    # thread while the other families compute. The features, their values and their order
    # are those of the pass in one thread, also with texture before morphology and with
    # morphology reused by deduplication.
    from pictologics import pipeline as pipeline_module

    image, mask = _morphology_case()
    discretise = {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}}
    orders = {
        "default": {},
        "texture_first": {"families": ["texture", "histogram", "morphology", "intensity"]},
        "two": {"families": ["morphology", "glcm"]},
    }
    for deduplicate in (False, True):
        pipeline = RadiomicsPipeline(load_standard=False, deduplicate=deduplicate)
        for name, params in orders.items():
            pipeline.add_config(name, [discretise, {"step": "extract_features", "params": params}])
        with patch.object(
            pipeline_module, "_morphology_worker", wraps=pipeline_module._morphology_worker
        ) as worker:
            threaded = pipeline.run(image, mask, config_names=list(orders))
        assert worker.call_count == (1 if deduplicate else 3)  # deduplication reuses it
        with patch.object(RadiomicsPipeline, "_morphology_ahead", return_value=None):
            alone = pipeline.run(image, mask, config_names=list(orders))
        for name in orders:
            assert list(threaded[name].index) == list(alone[name].index)
            assert threaded[name].to_numpy().tobytes() == alone[name].to_numpy().tobytes()
            assert threaded[name]["maximum_3d_diameter_L0JK"] > 0


def test_morphology_only_passes_use_no_worker() -> None:
    # A pass with morphology alone, or with morphology reused, computes it in its turn. A
    # part 1 without a rest (no mesh) gives all the features in the turn of morphology.
    from pictologics import pipeline as pipeline_module

    image, mask = _morphology_case()
    pipeline = RadiomicsPipeline(load_standard=False)
    for name, families in (("m", ["morphology"]), ("mi", ["morphology", "intensity"])):
        pipeline.add_config(name, [{"step": "extract_features", "params": {"families": families}}])
    with patch.object(pipeline_module, "_morphology_worker") as worker:
        pipeline.run(image, mask, config_names=["m"])
        pipeline.run(image, mask, config_names=["m", "mi"])  # mi reuses the morphology
        with patch.object(pipeline_module, "_morphology_first", return_value=({"a": 1.0}, None)):
            assert pipeline.run(image, mask, config_names=["mi"])["mi"]["a"] == 1.0
    worker.assert_not_called()


def test_a_worker_that_cannot_start_leaves_the_pass_in_one_thread() -> None:
    # Without a new thread (for example at interpreter exit), part 2 runs in this thread at
    # once, with the results of one thread.
    from pictologics import pipeline as pipeline_module

    image, mask = _morphology_case()
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("c", [
        {"step": "extract_features", "params": {"families": ["morphology", "intensity"]}},
    ])  # fmt: skip
    expected = pipeline.run(image, mask, config_names=["c"])["c"]
    worker = MagicMock()
    worker.submit.side_effect = RuntimeError("cannot schedule new futures after shutdown")
    with patch.object(pipeline_module, "_morphology_worker", return_value=worker):
        out = pipeline.run(image, mask, config_names=["c"])["c"]
    assert worker.submit.call_count == 1
    assert list(out.index) == list(expected.index)
    assert out.to_numpy().tobytes() == expected.to_numpy().tobytes()


def test_a_failing_morphology_part_gives_the_outcome_of_one_thread() -> None:
    # An error in the worker part (convex hull, MVEE) or in part 1 (mesh, bounding box)
    # leaves the morphology features NaN, with the warning and the family errors of the
    # pass in one thread; the family errors keep the order of the families. An empty ROI
    # in either part ends the configuration, as before.
    import contextlib

    from pictologics.features import morphology as morphology_module

    image, mask = _morphology_case()
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("c", [
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
        {"step": "extract_features", "params": {"families": ["intensity", "morphology", "glcm"]}},
    ])  # fmt: skip

    def outcome(target: str, error: Exception, glcm: bool, one_thread: bool) -> tuple[Any, ...]:
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(morphology_module, target, side_effect=error))
            if glcm:  # a family after morphology fails too
                glcm_error = RuntimeError("no glcm")
                stack.enter_context(
                    patch("pictologics.pipeline.calculate_glcm_features", side_effect=glcm_error)
                )
            if one_thread:
                ahead = patch.object(RadiomicsPipeline, "_morphology_ahead", return_value=None)
                stack.enter_context(ahead)
            caught = stack.enter_context(warnings.catch_warnings(record=True))
            warnings.simplefilter("always")
            series = pipeline.run(image, mask, config_names=["c"])["c"]
        entry = pipeline.get_log()[-1]
        return (
            list(series.index),
            series.to_numpy().tobytes(),
            entry["status"],
            list(entry.get("family_errors", {}).items()),
            [str(w.message) for w in caught],
        )

    for target in (
        "_get_convex_hull_features",
        "_get_mvee_features",
        "_get_mesh_features",
        "_get_bounding_box_features",
    ):
        for glcm in (False, True):
            error = RuntimeError(f"no {target}")
            threaded = outcome(target, error, glcm, False)
            assert threaded[:4] == outcome(target, error, glcm, True)[:4]
            assert threaded[2] == "completed" and threaded[3][0] == (
                "morphology",
                f"RuntimeError: no {target}",
            )
            # The warning of an error after part 1 comes when the worker part ends
            assert sorted(threaded[4]) == sorted(outcome(target, error, glcm, True)[4])
            assert len(threaded[4]) == 1 + glcm
        empty = EmptyROIMaskError("none")
        threaded = outcome(target, empty, False, False)
        assert threaded == outcome(target, empty, False, True) and threaded[2] == "empty_roi"


def test_the_histogram_reads_int32_values_with_their_levels() -> None:
    # The pipeline gives the histogram features the int32 values of the discretised image
    # and its number of levels, so they come from the bin counts (one bincount)
    from pictologics import pipeline as pipeline_module

    image, mask = _morphology_case()
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("h", [
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 8}},
        {"step": "extract_features", "params": {"families": ["histogram"]}},
    ])  # fmt: skip
    real = pipeline_module.calculate_intensity_histogram_features
    with patch.object(
        pipeline_module, "calculate_intensity_histogram_features", wraps=real
    ) as histogram:
        pipeline.run(image, mask, config_names=["h"])
    values = histogram.call_args.args[0]
    assert values.dtype == np.int32 and histogram.call_args.kwargs["n_bins"] == 8


def test_other_families_use_one_thread_less_next_to_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Next to part 2 of morphology in the worker thread, the other families of the pass use
    # one numba thread less when the threads fill the fast cores (threads._make_room). The
    # pass sets the number again at its end, also when a family ends the configuration (an
    # empty ROI), with and without deduplication. With a free fast core, nothing changes.
    import numba

    from pictologics import pipeline as pipeline_module
    from pictologics import threads

    state = {"n": 6}
    monkeypatch.setattr(threads, "_cores", 6)
    monkeypatch.setattr(numba, "get_num_threads", lambda: state["n"])
    monkeypatch.setattr(numba, "set_num_threads", lambda n: state.update(n=n))
    seen = []
    intensity = pipeline_module.calculate_intensity_features

    def counted(*args: Any) -> dict[str, float]:
        seen.append(state["n"])
        return intensity(*args)

    monkeypatch.setattr(pipeline_module, "calculate_intensity_features", counted)
    image, mask = _morphology_case()
    for deduplicate in (False, True):
        pipeline = RadiomicsPipeline(load_standard=False, deduplicate=deduplicate)
        pipeline.add_config("c", [
            {"step": "extract_features", "params": {"families": ["morphology", "intensity"]}},
        ])  # fmt: skip
        seen.clear()
        pipeline.run(image, mask, config_names=["c"])
        assert seen == [5] and state["n"] == 6
        empty = EmptyROIMaskError("none")
        with patch.object(pipeline_module, "calculate_intensity_features", side_effect=empty):
            pipeline.run(image, mask, config_names=["c"])
        assert pipeline.get_log()[-1]["status"] == "empty_roi" and state["n"] == 6
    monkeypatch.setattr(threads, "_cores", 8)
    seen.clear()
    pipeline.run(image, mask, config_names=["c"])
    assert seen == [6] and state["n"] == 6


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_morphology_worker_is_new_after_fork() -> None:
    # A forked process drops the morphology worker of its parent, whose thread it does not
    # have, so its passes make a new worker instead of waiting forever.
    import signal
    import time

    from pictologics import pipeline as pipeline_module

    image, mask = _morphology_case()
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("c", [
        {"step": "extract_features", "params": {"families": ["morphology", "intensity"]}},
    ])  # fmt: skip
    expected = pipeline.run(image, mask, config_names=["c"])["c"]
    assert pipeline_module._MORPHOLOGY_WORKER
    pid = os.fork()
    if pid == 0:  # pragma: no cover  (coverage does not follow the child)
        code = 1
        try:
            if not pipeline_module._MORPHOLOGY_WORKER:
                same = pipeline.run(image, mask, config_names=["c"])["c"].equals(expected)
                code = 0 if same else 2
        finally:
            os._exit(code)
    deadline = time.monotonic() + 120
    while (done := os.waitpid(pid, os.WNOHANG))[0] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    if done[0] == 0:  # pragma: no cover  (only when the child waits forever)
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        pytest.fail("the forked process waited for the morphology worker of its parent")
    assert os.waitstatus_to_exitcode(done[1]) == 0
