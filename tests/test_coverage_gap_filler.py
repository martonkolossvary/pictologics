import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pictologics.loader import Image
from pictologics.pipeline import RadiomicsPipeline, SourceMode
from pictologics.filters import (
    gabor_filter,
    laws_filter,
    laplacian_of_gaussian,
    mean_filter,
    riesz_log,
    riesz_simoncelli,
    wavelet_transform,
    simoncelli_wavelet,
    BoundaryCondition,
)
from pictologics.preprocessing import resample_image


class TestCoverageGapFiller:

    def setup_method(self):
        self.shape = (20, 20, 20)
        self.image_arr = np.random.rand(*self.shape).astype(np.float32)
        self.mask_arr = np.zeros(self.shape, dtype=np.uint8)
        self.mask_arr[5:15, 5:15, 5:15] = 1

        self.image = Image(self.image_arr, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        self.mask = Image(self.mask_arr, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

        # Source mask (boolean/uint8)
        self.source_mask_arr = self.mask_arr > 0

    def test_gabor_source_mask(self):
        # Covers gabor.py line 97
        res = gabor_filter(
            self.image_arr,
            source_mask=self.source_mask_arr,
            sigma_mm=1.0,
            lambda_mm=2.0,
        )
        assert res.shape == self.shape
        assert not np.isnan(res).any()

    def test_laws_source_mask_non_rotational(self):
        # Covers laws.py lines 350, 372
        # Laws NON-rotation invariant returns (result, valid_mask) when source_mask is used
        res, valid = laws_filter(
            self.image_arr,
            kernels="L5E5S5",
            source_mask=self.source_mask_arr,
            rotation_invariant=False,
        )
        assert res.shape == self.shape
        assert valid.shape == self.shape

    def test_log_source_mask(self):
        # Covers log.py line 118
        res, valid = laplacian_of_gaussian(
            self.image_arr, sigma_mm=1.0, source_mask=self.source_mask_arr
        )
        assert res.shape == self.shape

    def test_mean_source_mask(self):
        # Covers mean.py line 102
        res, valid = mean_filter(self.image_arr, source_mask=self.source_mask_arr)
        assert res.shape == self.shape

    def test_riesz_source_mask(self):
        # Covers riesz.py line 60 (via variants)
        res = riesz_log(self.image_arr, sigma_mm=1.0, source_mask=self.source_mask_arr)
        assert res.shape == self.shape

        res2 = riesz_simoncelli(self.image_arr, source_mask=self.source_mask_arr)
        assert res2.shape == self.shape

    def test_wavelet_source_mask(self):
        # Covers wavelets.py line 87
        res = wavelet_transform(
            self.image_arr,
            wavelet="haar",
            level=1,
            decomposition="LLL",
            source_mask=self.source_mask_arr,
        )
        assert res.shape == self.shape

    def test_simoncelli_source_mask(self):
        # Covers wavelets.py line 322
        res = simoncelli_wavelet(
            self.image_arr, level=1, source_mask=self.source_mask_arr
        )
        assert res.shape == self.shape

    def test_preprocessing_resample_ndmask(self):
        # Covers preprocessing.py lines 295-298 (else branch)
        # Pass source_mask as numpy array, not Image
        res = resample_image(
            self.image,
            new_spacing=(2.0, 2.0, 2.0),
            source_mask=self.source_mask_arr,  # numpy array
        )
        assert res.has_source_mask

    def test_pipeline_invalid_source_mode(self):
        # Covers pipeline.py line 388
        config = [{"step": "resample", "params": {"new_spacing": (1, 1, 1)}}]
        pipeline = RadiomicsPipeline()
        with pytest.raises(ValueError, match="Invalid source_mode"):
            pipeline.add_config("bad", config, source_mode="invalid_mode")

    def test_pipeline_roi_only_and_auto_modes(self):
        # Covers pipeline.py lines 523-556, 680, 696, 938-973

        pipeline = RadiomicsPipeline()

        # Config with ROI_ONLY
        config_roi = [
            {"step": "resample", "params": {"new_spacing": (1, 1, 1)}},
            {"step": "filter", "params": {"type": "mean", "size": 3}},
            {"step": "filter", "params": {"type": "log", "sigma_mm": 1.0}},
            {
                "step": "filter",
                "params": {"type": "laws", "kernel": "E5L5S5"},
            },  # Default rotation_invariant=False -> returns tuple
            {
                "step": "filter",
                "params": {"type": "gabor", "sigma_mm": 1.0, "lambda_mm": 2.0},
            },
            {
                "step": "filter",
                "params": {
                    "type": "wavelet",
                    "wavelet": "haar",
                    "level": 1,
                    "decomposition": "LLL",
                },
            },
            {"step": "filter", "params": {"type": "simoncelli", "level": 1}},
            {
                "step": "filter",
                "params": {"type": "riesz", "variant": "log", "sigma_mm": 1.0},
            },
            # Use extract_features to ensure pipeline runs completion
            {"step": "extract_features", "params": {"features": ["stat_mean"]}},
        ]
        pipeline.add_config("roi_config", config_roi, source_mode="roi_only")

        results = pipeline.run(self.image, self.mask, config_names=["roi_config"])
        assert "roi_config" in results

        # Test AUTO mode (covers 533-556)
        # Create image with sentinel -1000
        sentinel_val = -1000.0
        img_arr_sent = np.full(self.shape, sentinel_val, dtype=np.float32)
        img_arr_sent[5:15, 5:15, 5:15] = 100.0  # Valid tissue
        img_sent = Image(img_arr_sent, (1, 1, 1), (0, 0, 0))

        pipeline.add_config("auto_config", config_roi, source_mode="auto")

        # AUTO detection now logs at debug level (no UserWarning expected)
        results_auto = pipeline.run(
            img_sent, self.mask, config_names=["auto_config"]
        )
        assert "auto_config" in results_auto

        auto_logs = [entry for entry in pipeline._log if entry["config_name"] == "auto_config"]
        assert auto_logs
        assert auto_logs[-1]["sentinel_detected"] is True
        assert auto_logs[-1]["sentinel_value"] == sentinel_val

    def test_pipeline_defensive_branches(self):
        # Covers pipeline.py defensive else branches for tuple checks
        # Mock filters to return NDArray instead of tuple even when source_mask is passed

        pipeline = RadiomicsPipeline()
        # Minimal config
        mean_step = [{"step": "filter", "params": {"type": "mean"}}]
        log_step = [{"step": "filter", "params": {"type": "log", "sigma_mm": 1.0}}]

        pipeline.add_config("mean_cfg", mean_step, source_mode="roi_only")
        pipeline.add_config("log_cfg", log_step, source_mode="roi_only")

        # Mock mean_filter
        with patch("pictologics.pipeline.mean_filter") as mock_mean:
            # Return just array, no tuple
            mock_mean.return_value = np.zeros(self.shape)
            pipeline.run(self.image, self.mask, config_names=["mean_cfg"])
            # Should have hit 'else' branch in pipeline (line 876)
            mock_mean.assert_called()

        # Mock gaussian_laplace
        with patch("pictologics.pipeline.laplacian_of_gaussian") as mock_log:
            mock_log.return_value = np.zeros(self.shape)
            pipeline.run(self.image, self.mask, config_names=["log_cfg"])
            # Covered line 897
            mock_log.assert_called()

    def test_pipeline_explicit_sentinel(self):
        # Covers pipeline.py lines 537-538
        pipeline = RadiomicsPipeline()
        config = [{"step": "filter", "params": {"type": "mean"}}]

        # Explicit sentinel
        pipeline.add_config(
            "explicit_sent", config, source_mode="auto", sentinel_value=-1000
        )

        # Image with sentinel
        img_arr_sent = np.full(self.shape, -1000.0, dtype=np.float32)
        img_arr_sent[5:15, 5:15, 5:15] = 100.0
        img_sent = Image(img_arr_sent, (1, 1, 1), (0, 0, 0))

        results = pipeline.run(img_sent, self.mask, config_names=["explicit_sent"])
        assert "explicit_sent" in results
        # Verify source_mask was used (check log if possible, or assume based on completion)

        exported = pipeline.to_dict(config_names=["explicit_sent"])
        exported_cfg = exported["configs"]["explicit_sent"]
        assert exported_cfg["source_mode"] == "auto"
        assert exported_cfg["sentinel_value"] == -1000

    def test_pipeline_filters_with_source_mask_no_resample(self):
        # Covers pipeline.py filter blocks with source_mask (lines 875-973)
        # Skip resample step to avoid potential source_mask loss issues in testing

        pipeline = RadiomicsPipeline()
        config = [
            # No resample
            {"step": "filter", "params": {"type": "mean", "support": 3}},
            {"step": "filter", "params": {"type": "log", "sigma_mm": 1.0}},
            {"step": "filter", "params": {"type": "laws", "kernel": "E5L5S5"}},
            {
                "step": "filter",
                "params": {"type": "gabor", "sigma_mm": 1.0, "lambda_mm": 2.0},
            },
            {
                "step": "filter",
                "params": {
                    "type": "wavelet",
                    "wavelet": "haar",
                    "level": 1,
                    "decomposition": "LLL",
                },
            },
            {"step": "filter", "params": {"type": "simoncelli", "level": 1}},
            {
                "step": "filter",
                "params": {"type": "riesz", "variant": "log", "sigma_mm": 1.0},
            },
            {
                "step": "extract_features",
                "params": {"features": ["stat_mean"], "families": ["intensity"]},
            },
        ]

        pipeline.add_config("nore_config", config, source_mode="roi_only")
        results = pipeline.run(self.image, self.mask, config_names=["nore_config"])

        # Check logs for errors
        for entry in pipeline._log:
            if "error" in entry:
                pytest.fail(f"Pipeline error: {entry['error']}")

        assert "nore_config" in results

    def test_pipeline_laws_rotation_invariant_source_mask(self):
        # Covers pipeline.py laws logic with tuple check (lines 922-923)
        # When rotation_invariant=True, laws_filter returns single array even if source_mask passed.

        pipeline = RadiomicsPipeline()
        config = [
            {
                "step": "filter",
                "params": {
                    "type": "laws",
                    "rotation_invariant": True,
                    "kernel": "E5L5S5",
                },
            },
            {
                "step": "extract_features",
                "params": {"features": ["stat_mean"], "families": ["intensity"]},
            },
        ]
        pipeline.add_config("laws_rot", config, source_mode="roi_only")

        results = pipeline.run(self.image, self.mask, config_names=["laws_rot"])
        assert "laws_rot" in results
