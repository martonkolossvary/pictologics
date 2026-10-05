import json
import os
import subprocess
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

# Filter warnings for the test suite itself if needed
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="The NumPy module was reloaded.*", category=UserWarning)


class TestWarmup(unittest.TestCase):
    def setUp(self) -> None:
        # Snapshot os.environ (auto-restored in tearDown) and ensure warmup is
        # enabled for the duration of each test regardless of what other test
        # modules left in the environment.
        self._env_patcher = patch.dict(os.environ, {}, clear=False)
        self._env_patcher.start()
        os.environ.pop("PICTOLOGICS_DISABLE_WARMUP", None)

    def tearDown(self) -> None:
        self._env_patcher.stop()

    def test_warmup_runs(self) -> None:
        """Test that warmup runs successfully and triggers expected internal functions."""
        # We Mock the internal _warmup_* methods to verify flow without paying JIT cost
        with (
            patch("pictologics.warmup._warmup_texture") as mock_tex,
            patch("pictologics.warmup._warmup_intensity") as mock_int,
            patch("pictologics.warmup._warmup_morphology") as mock_morph,
        ):
            from pictologics.warmup import warmup_jit

            warmup_jit()

            mock_tex.assert_called_once()
            mock_int.assert_called_once()
            mock_morph.assert_called_once()

    def test_warmup_disabled(self) -> None:
        """Test that setting PICTOLOGICS_DISABLE_WARMUP stops execution."""
        os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

        # Since INFO logs are removed, just verify no exception is raised
        from pictologics.warmup import warmup_jit

        # Should complete without error when disabled
        warmup_jit()

    @patch("pictologics.warmup._warmup_texture")
    def test_warmup_failure_survived(self, mock_texture: MagicMock) -> None:
        """Test that exception in warmup doesn't crash the program."""
        mock_texture.side_effect = RuntimeError("Something exploded")

        # Warmup failure should emit a RuntimeWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from pictologics.warmup import warmup_jit

            warmup_jit()

            # Verify warning was emitted
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            self.assertTrue(len(runtime_warnings) >= 1)
            self.assertTrue(
                any("warmup failed" in str(x.message).lower() for x in runtime_warnings)
            )

    def test_warmup_integration_data_setup_coverage(self) -> None:
        """Integration test: the data-setup code in each _warmup_* helper runs (and
        compiles/executes the kernels on dummy data) without raising."""
        from pictologics.warmup import (
            _warmup_intensity,
            _warmup_morphology,
            _warmup_texture,
        )

        # A raise here fails the test with the original traceback.
        _warmup_texture()
        _warmup_intensity()
        _warmup_morphology()


# Run with the compiler on (see test_a_pipeline_run_after_the_import_compiles_no_kernel):
# the new signatures of each kernel of the package after a pipeline workload, as JSON.
_NEW_SIGNATURES = """
import importlib, json, os, pkgutil, threading, warnings

import numpy as np
import scipy

with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)  # a warm-up that fails
    import pictologics  # before numba: then each thread starts with the default
import numba
from numba.core.registry import CPUDispatcher
from pictologics import Image, RadiomicsPipeline
from pictologics.features import intensity, morphology, texture
from pictologics.preprocessing import discretise_image


def signatures():
    found = {}
    for info in pkgutil.walk_packages(pictologics.__path__, "pictologics."):
        for value in vars(importlib.import_module(info.name)).values():
            if isinstance(value, CPUDispatcher):
                name = value.py_func.__module__ + "." + value.__name__
                found[name] = [str(s) for s in value.signatures]
    return found


before = signatures()
seen = []
worker = threading.Thread(target=lambda: seen.append(numba.get_num_threads()))
worker.start()
worker.join()
rng = np.random.default_rng(5)
shape = (128, 128, 96)
z, y, x = np.indices(shape)
values = rng.normal(40.0, 20.0, shape)
roi = ((z - 64) / 30.0) ** 2 + ((y - 64) / 30.0) ** 2 + ((x - 48) / 30.0) ** 2 <= 1.0
values[roi] += 30.0
image = Image(values, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
pipeline = RadiomicsPipeline()
pipeline.add_config("filter_fbs", [
    {"step": "filter", "params": {"type": "log", "sigma_mm": 1.5}},
    {"step": "discretise", "params": {"method": "FBS", "bin_width": 2.0, "min_val": -200.0}},
    {"step": "extract_features", "params": {"families": ["texture", "intensity"]}},
])
pipeline.add_config("many_levels", [
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 256}},
    {"step": "extract_features", "params": {"families": ["texture"]}},
])
pipeline.add_config("fbs", [
    {"step": "discretise", "params": {"method": "FBS", "bin_width": 5.0, "min_val": -100.0}},
    {"step": "extract_features", "params": {"families": ["intensity"]}},
])
pipeline.add_config("spatial", [{"step": "extract_features", "params": {
    "families": ["intensity"], "include_spatial_intensity": True, "include_local_intensity": True,
}}])
mask = Image(roi.astype(np.uint8), image.spacing, image.origin)
small = (z - 20) ** 2 + (y - 20) ** 2 + (x - 20) ** 2 <= 16  # the pair loop of Moran's I
small = Image(small.astype(np.uint8), image.spacing, image.origin)
names = ["all_standard", "filter_fbs", "many_levels", "spatial"]
runs = {}
for threads in (None, 1):  # the default, then one thread: the same features
    pictologics.set_num_threads(threads)
    runs[threads] = [pipeline.run(image, mask, config_names=names)]
    runs[threads].append(pipeline.run(image, small, config_names=["spatial"]))
# SciPy 1.18 splits its FFT work by thread, so Moran's I and Geary's C of the large ROI (the
# FFT sums) can change in the last digit with the number of threads; all else is the same bits
fft_by_thread = tuple(int(p) for p in scipy.__version__.split(".")[:2]) >= (1, 18)


def agree(first, again, fft):
    loose = [k for k in ("morans_i_index_N365", "gearys_c_measure_NPT7") if fft and k in first]
    if not np.allclose(first[loose], again[loose], rtol=1e-12, atol=0.0):
        return False
    return first.drop(loose).to_numpy().tobytes() == again.drop(loose).to_numpy().tobytes()


same = all(
    agree(first[name], again[name], fft_by_thread and run == 0)  # run 1: the pair loop
    for run, (first, again) in enumerate(zip(runs[None], runs[1]))
    for name in first
)
pictologics.set_num_threads()
for kind in (np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
             np.uint64, np.float32, np.float64):  # masks of each type and order
    for array in (roi.astype(kind), np.asfortranarray(roi.astype(kind))):
        pipeline.run(image, Image(array, image.spacing, image.origin), config_names=["fbs"])
labels = np.where(roi, 3, 0)
labels[5:15, 5:15, 5:15] = 1
for dtype in (np.uint8, np.uint16, np.int32, np.int64, np.float64):
    label_map = Image(labels.astype(dtype), image.spacing, image.origin)
    pipeline.run_rois(image, label_map, config_names=["standard_fbn_8"])
for kind in (np.float64, np.float32, np.uint8, np.bool_, np.int64):  # column-order copies
    pictologics.save_image(Image(values.astype(kind), image.spacing, image.origin), "x.nii")
levels = np.asarray(discretise_image(values, "FBN", roi_mask=mask.array, n_bins=16))
for kind in (np.uint8, np.int64, np.float32):  # direct calls with other types
    texture.calculate_all_texture_features(levels.astype(kind), mask.array, 16)
    intensity.calculate_intensity_histogram_features(levels[roi].astype(kind))
ct = Image(values.astype(np.int16), image.spacing, image.origin)
intensity.calculate_local_intensity_features(ct, mask)
morphology.calculate_morphology_features(mask, ct)
import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRBigEndian, generate_uid

os.mkdir("big_endian")  # a big-endian series of 2^20 voxels takes the one-pass kernel
for k in range(64):
    ds = pydicom.Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID = CTImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRBigEndian
    ds.SeriesInstanceUID, ds.Modality, ds.InstanceNumber = "1.2.3", "CT", k + 1
    ds.Rows = ds.Columns = 128
    ds.PixelSpacing, ds.ImagePositionPatient = [1, 1], [0, 0, k]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.PixelData = np.full((128, 128), k, dtype=">i2").tobytes()
    ds.save_as(f"big_endian/{k}.dcm", enforce_file_format=True, implicit_vr=False,
               little_endian=False)
pictologics.load_image("big_endian")
texture.calculate_zone_features(levels.astype(np.uint8), np.asfortranarray(mask.array), np.zeros(shape, np.int32), 16)
texture._gldzm_distance_map(roi)
pipeline.run(image, Image(roi.astype(np.float16), image.spacing, image.origin), config_names=["fbs"])
small_shape = (30, 32, 34)  # no ROI cut: the arrays keep the layout of the caller
sz, sy, sx = np.indices(small_shape)
small_values = rng.normal(40.0, 20.0, small_shape)
small_roi = ((sz - 15) / 10.0) ** 2 + ((sy - 16) / 10.0) ** 2 + ((sx - 17) / 10.0) ** 2 <= 1.0
for conv in (np.ascontiguousarray, np.asfortranarray):
    for kind in (np.float64, np.uint8, np.bool_, np.int32, np.uint16, np.float32):
        small_image = Image(conv(small_values), image.spacing, image.origin)
        small_mask = Image(conv(small_roi.astype(kind)), image.spacing, image.origin)
        pipeline.run(small_image, small_mask, config_names=["fbs", "standard_fbn_8", "spatial"])
        pipeline.run(Image(np.ascontiguousarray(small_values), image.spacing, image.origin), small_mask, config_names=["fbs"])
statuses = {entry["status"] for entry in pipeline.get_log()}
after = signatures()
new = {name: [s for s in after[name] if s not in before.get(name, [])] for name in after}
print(json.dumps({
    "statuses": sorted(statuses),
    "new": {k: v for k, v in new.items() if v},
    "same at one thread": same,
    "new thread": seen == [pictologics.get_num_threads()],
}))
"""


def test_a_pipeline_run_after_the_import_compiles_no_kernel(tmp_path: Path) -> None:
    # With the compiler on, the import warms every kernel signature that the package
    # reaches: the standard templates on a CT-like image (large enough for the parallel
    # paths), a filter with many FBS levels, 256 grey levels, masks of each type in row and
    # column order, label maps of five types in run_rois, save_image of five types, direct
    # calls with other input types, and a big-endian DICOM series. One thread gives the
    # same features as the default (also Moran's I and Geary's C; with SciPy 1.18 or later,
    # whose FFT splits its work by thread, these two agree to 1e-12 on the FFT path), and a
    # new thread starts with the default. The check runs in its own numba cache, so it
    # never writes into a cache that another process builds at the same time.
    cache = os.environ.get("NUMBA_CACHE_DIR")
    env = {
        **os.environ,
        "NUMBA_DISABLE_JIT": "0",
        "NUMBA_CACHE_DIR": f"{cache}-warmup-test" if cache else str(tmp_path / "numba"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    env.pop("PICTOLOGICS_DISABLE_WARMUP", None)
    env.pop("NUMBA_NUM_THREADS", None)  # the import chooses the number of threads
    done = subprocess.run(
        [sys.executable, "-c", _NEW_SIGNATURES],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1]) == {
        "statuses": ["completed"],
        "new": {},
        "same at one thread": True,
        "new thread": True,
    }
