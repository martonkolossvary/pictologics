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
# FFT sums) can change in the last digit with the number of threads, and the float32 response
# of the FFT LoG (filter_fbs) by one step in rare voxels; all else is the same bits
fft_by_thread = tuple(int(p) for p in scipy.__version__.split(".")[:2]) >= (1, 18)


def agree(first, again, fft, name):
    if fft and name == "filter_fbs":  # one float32 step of the response moves a feature by 1e-7
        return np.allclose(first, again, rtol=1e-6, atol=0.0, equal_nan=True)
    loose = [k for k in ("morans_i_index_N365", "gearys_c_measure_NPT7") if fft and k in first]
    if not np.allclose(first[loose], again[loose], rtol=1e-12, atol=0.0):
        return False
    return first.drop(loose).to_numpy().tobytes() == again.drop(loose).to_numpy().tobytes()


same = all(
    agree(first[name], again[name], fft_by_thread and run == 0, name)  # run 1: the pair loop
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
for top in (300, 70_000):  # float64 maps: the uint16 and int64 casts, column order, read-only
    float_map = labels.astype(np.float64)
    float_map[0, 0, 0] = top
    pipeline.run_rois(image, Image(float_map, image.spacing, image.origin), labels=[1], config_names=["fbs"])
float_map = np.asfortranarray(labels.astype(np.float64))
pipeline.run_rois(image, Image(float_map, image.spacing, image.origin), labels=[1], config_names=["fbs"])
float_map = labels.astype(np.float64)
float_map.setflags(write=False)
pipeline.run_rois(image, Image(float_map, image.spacing, image.origin), labels=[1], config_names=["fbs"])
for kind in (np.float64, np.float32, np.uint8, np.bool_, np.int64):  # column-order copies
    pictologics.save_image(Image(values.astype(kind), image.spacing, image.origin), "x.nii")
import nibabel
for kind in (np.int32, np.uint8):  # stored-type masks of 2^20 voxels (the tiled row-order copy)
    nibabel.save(nibabel.Nifti1Image(roi.astype(kind)[:, :, :64], np.eye(4), dtype=kind), "m.nii")
    pictologics.load_image("m.nii", stored_type=True)
for kind in (np.float64, np.float32, np.uint8, np.int32):  # the gzip writer; indexed, one pass
    pictologics.save_image(Image(values.astype(kind), image.spacing, image.origin), "x.nii.gz")
    pictologics.load_image("x.nii.gz")
    pictologics.load_image("x.nii.gz", stored_type=True)
nibabel.save(nibabel.Nifti1Image(values.astype(np.int16), np.eye(4)), "s.nii.gz")  # nibabel's file: one pass
pictologics.load_image("s.nii.gz")
levels = np.asarray(discretise_image(values, "FBN", roi_mask=mask.array, n_bins=16))
for kind in (np.uint8, np.int64, np.float32):  # direct calls with other types
    texture.calculate_all_texture_features(levels.astype(kind), mask.array, 16)
    intensity.calculate_intensity_histogram_features(levels[roi].astype(kind))
for kind in (np.uint8, np.int64):  # column-order levels: a row-order copy (see _level_form)
    texture.calculate_all_texture_features(np.asfortranarray(levels.astype(kind)), np.asfortranarray(mask.array), 16)
    texture.calculate_zone_features(np.asfortranarray(levels.astype(kind)), mask.array, np.zeros(shape, np.int32), 16)
intensity.calculate_intensity_features(np.random.default_rng(3).normal(40.0, 20.0, 1_500_000))  # the threaded select: a column-order table
big = np.tile(levels[roi], 5)  # from 2^19 values on, the level count runs in threads
for kind in (np.int32, np.int64, np.float64):
    intensity.calculate_intensity_histogram_features(big.astype(kind), n_bins=16)
    intensity.calculate_intensity_histogram_features(levels[roi].astype(kind), n_bins=16)
for kind in (np.float64, np.float32, np.int32, np.int16):  # the IVH counts: whole floats, int32 copies
    intensity.calculate_ivh_features(levels[roi].astype(kind))
intensity.calculate_ivh_features(np.arange(70_000, dtype=np.int32))  # one count row
intensity.calculate_ivh_features(np.tile(levels[roi].astype(np.int32), 10))  # the threaded count
stored = values[:, :, :64].astype(np.int16).tobytes(order="F")  # 2^20 voxels, x fastest
with open("c.nrrd", "wb") as nrrd_file:  # the writeable values of the NRRD and MHA readers
    nrrd_file.write(b"NRRD0004\\ntype: short\\ndimension: 3\\nspace: left-posterior-superior\\n"
                    b"sizes: 128 128 64\\nspace directions: (1,0,0) (0,1,0) (0,0,1)\\n"
                    b"kinds: domain domain domain\\nendian: little\\nencoding: raw\\n"
                    b"space origin: (0,0,0)\\n\\n" + stored)
pictologics.load_image("c.nrrd")
with open("c.mha", "wb") as mha_file:
    mha_file.write(b"ObjectType = Image\\nNDims = 3\\nBinaryData = True\\nDimSize = 128 128 64\\n"
                   b"ElementType = MET_SHORT\\nElementDataFile = LOCAL\\n" + stored)
pictologics.load_image("c.mha")
from pictologics.filters import gabor_filter
for kind in (np.float32, np.float64):  # the Gabor slices in each order, each part and a region
    cube = values[:20, :18, :16].astype(kind)
    for layout in (cube, np.asfortranarray(cube), cube[:2], np.asfortranarray(cube[:2])):
        gabor_filter(layout, 1.5, 3.0, rotation_invariant=True, delta_theta=np.pi / 4,
                     average_over_planes=True)
    for part in ("real", "imaginary", "angle"):
        gabor_filter(cube, 1.5, 3.0, rotation_invariant=True, delta_theta=np.pi / 4, pooling="max",
                     response=part, region=(slice(2, 9), slice(3, 12), slice(1, 8)))
gabor_filter(values, 1.5, 3.0)  # a large axial result: the tiled row-order copy
from pictologics.filters import gaussian_filter, laws_filter, mean_filter
for kind in (np.float64, np.float32):  # the pass twins of the separable filters (and masked)
    cube = values.astype(kind)
    mean_filter(cube, 5)
    mean_filter(cube, 5, source_mask=roi)
    gaussian_filter(cube, 2.0, (1.0, 1.0, 1.0), source_mask=roi)
    laws_filter(cube, "E5L5S5", rotation_invariant=True, pooling="average", compute_energy=True)
from pictologics.filters import laplacian_of_gaussian
laplacian_of_gaussian(values, 2.0)  # padded 144x144x112: the FFT slabs, a float32 output
laplacian_of_gaussian(values, 2.0, source_mask=roi)  # the masked FFT: a float64 output
from pictologics.filters import wavelet_transform
wavelet_transform(values, "db2", 2, "LHL", rotation_invariant=True)  # the 24 rotations, slabs, in place
wavelet_transform(values.astype(np.float32), "coif1", 1, "HHH", "mirror", rotation_invariant=True, pooling="max")
wavelet_transform(values[:20, :18, :16], "haar", 1, "LHL")  # the serial path
column = Image(np.asfortranarray(values.astype(np.float32)), image.spacing, image.origin)  # one pass
resample_run = RadiomicsPipeline(load_standard=False)
resample_run.add_config("im1", [{"step": "resample", "params": {"new_spacing": (1.5, 1.5, 1.5)}},
                                {"step": "extract_features", "params": {"families": ["intensity"]}}])
resample_run.run(column, mask, config_names=["im1"])
read_only = values.copy()  # read-only in-memory arrays: one copy at the start of a run
read_only.setflags(write=False)
fixed = roi.astype(np.uint8)
fixed.setflags(write=False)
resample_run.add_config("cubic", [{"step": "resample", "params": {"new_spacing": (1.5, 1.5, 1.5), "interpolation": "cubic"}},
                                  {"step": "extract_features", "params": {"families": ["intensity"]}}])
resample_run.run(Image(read_only, image.spacing, image.origin), Image(fixed, image.spacing, image.origin), config_names=["im1", "cubic"])
pipeline.run(Image(read_only, image.spacing, image.origin), Image(fixed, image.spacing, image.origin), config_names=["fbs"])
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
# Each serial twin gives the bits of its parallel kernel (10 sizes, the warmed signatures)
from pictologics import preprocessing
from pictologics.features._mc_tables import EDGE_TABLE, TRIANGLE_COUNT


def outputs(kernel, args, outs):
    result = kernel(*args)
    return [np.asarray(v).tobytes() for v in (*(result if isinstance(result, tuple) else (result,)), *(args[i] for i in outs))]


twins_same = True
twin_rng = np.random.default_rng(12)
for n in (3, 5, 8, 11, 16, 23, 31, 40, 52, 64):
    box = (twin_rng.random((n, n + 1, n + 2)) > 0.4).astype(np.uint8)
    vals = twin_rng.normal(40.0, 10.0, box.shape)
    valid = twin_rng.random(box.shape) > 0.1
    padded = np.pad(box, 1)
    verts, faces = morphology._mesh(padded, np.zeros(3), np.ones(3))
    evecs = np.asfortranarray(np.linalg.eigh(np.cov(verts.T))[1])
    scale, shift, start = np.full(3, 1.25), np.zeros(3), np.zeros(3, dtype=np.int64)
    out = tuple(int((k - 1) / 1.25) + 1 for k in box.shape)
    vpad = np.pad(vals, 12, mode="edge")
    pole = preprocessing._SPLINE_POLE
    gain = (1.0 - pole) * (1.0 - 1.0 / pole)
    coef = preprocessing._spline_coefficients(vals)
    mcoef, wcoef = preprocessing._spline_coefficients_masked(vals, valid)
    roi_box = box != 0
    voxels, nruns = np.empty(n, np.int64), np.empty(n, np.int64)
    intensity._roi_count_numba(roi_box, voxels, nruns)
    vstart, rstart = np.cumsum(voxels) - voxels, np.cumsum(nruns) - nruns
    ind, _, runs = intensity._roi_scan(roi_box, vals)
    rdx, rdy, rdz = intensity._sphere_rows((1.0, 1.0, 3.0))  # dz up to 2: pad 2
    offs = intensity._sphere_offsets_for_radius((1.0, 1.0, 3.0), intensity._LOCAL_PEAK_RADIUS_MM)
    p0, p2 = np.empty((n, n + 1, n + 3)), np.empty((n, n + 1, n + 7))
    sx_idx, sy_idx, sz_idx = (axis.astype(np.int32) for axis in np.nonzero(box))
    spectrum = np.fft.rfftn(vals)
    fold_shape = spectrum.shape
    grid = twin_rng.random(fold_shape)
    folds = [np.arange(k, dtype=np.int64) for k in fold_shape[:2]]
    dist_axes = [np.minimum(np.arange(k), k - np.arange(k)) * 0.7 for k in box.shape]
    intensity._prefix_pad_numba(vals, 0, p0, np.empty(n), np.empty(n))
    intensity._prefix_pad_numba(vals, 2, p2, np.empty(n), np.empty(n))
    cases = [
        (morphology._accumulate_moments_from_mask_numba, lambda: (box,), ()),
        (morphology._accumulate_moments_from_mask_numba, lambda: (box[1:, 1:, 1:],), ()),
        (morphology._accumulate_intensity_weighted_moments_numba, lambda: (box, vals), ()),
        (morphology._accumulate_intensity_weighted_moments_numba, lambda: (box[1:, 1:, 1:], vals[1:, 1:, 1:]), ()),
        (morphology._mc_counts_numba, lambda: (padded, EDGE_TABLE, TRIANGLE_COUNT, np.zeros(n + 2, np.int64), np.zeros(n + 2, np.int64)), (3, 4)),
        (morphology._row_ends_numba, lambda: (padded, 3), ()),
        (morphology._mesh_area_volume_numba, lambda: (verts, faces), ()),
        (morphology._ombb_extents_numba, lambda: (verts, verts.mean(axis=0), evecs), ()),
        (preprocessing._resample_nearest_numba, lambda: (box, scale, shift, start, np.zeros(out, np.uint8)), (4,)),
        (preprocessing._resample_trilinear_numba, lambda: (vals, scale, shift, start, False, float("nan"), np.zeros(out)), (6,)),
        (preprocessing._resample_trilinear_masked_numba, lambda: (vals, valid, scale, shift, start, 0.5, np.zeros(out), np.zeros(out, np.bool_)), (6, 7)),
        (preprocessing._pad_edge_numba, lambda: (vals, 12, np.zeros(vpad.shape)), (2,)),
        (preprocessing._prefilter_axis0_numba, lambda: (vpad.copy(), pole, gain, pole ** vpad.shape[0], 4), (0,)),
        (preprocessing._prefilter_axis1_numba, lambda: (vpad.copy(), pole, gain, pole ** vpad.shape[1]), (0,)),
        (preprocessing._prefilter_axis2_numba, lambda: (vpad.copy(), pole, gain, pole ** vpad.shape[2]), (0,)),
        (preprocessing._resample_tricubic_numba, lambda: (coef, scale, shift, start, 12.0, np.zeros(out)), (5,)),
        (preprocessing._pad_edge_masked_numba, lambda: (vals, valid, 12, np.zeros(vpad.shape), np.zeros(vpad.shape)), (3, 4)),
        (preprocessing._resample_tricubic_masked_numba, lambda: (mcoef, wcoef, scale, shift, start, 12.0, 0.5, np.zeros(out), np.zeros(out, np.bool_)), (7, 8)),
        (intensity._roi_count_numba, lambda: (roi_box, np.zeros(n, np.int64), np.zeros(n, np.int64)), (1, 2)),
        (intensity._roi_fill_numba, lambda: (roi_box, vals, vstart, rstart, np.zeros(ind.shape, np.int32), np.zeros(len(ind)), np.zeros(runs.shape, np.int64)), (4, 5, 6)),
        (intensity._prefix_pad_numba, lambda: (vals, 2, np.zeros(p2.shape), np.zeros(n), np.zeros(n)), (2, 3, 4)),
        (intensity._approximate_local_means_numba, lambda: (p0, ind, rdx, rdy, rdz), ()),
        (intensity._approx_runs_numba, lambda: (p2, 2, n + 2, runs, rdx, rdy, rdz, np.zeros(len(ind))), (7,)),
        (intensity._calculate_local_mean_numba, lambda: (vals, ind, offs), ()),
        (intensity._calculate_spatial_features_numba, lambda: (sx_idx, sy_idx, sz_idx, vals[box != 0], 40.0, 0.8, 0.9, 2.5), ()),
        (intensity._inverse_distance_numba, lambda: (np.empty(box.shape), *dist_axes), (0,)),
        (intensity._multiply_spectrum_numba, lambda: (spectrum.copy(), grid, *folds), (0,)),
        (intensity._weighted_power_numba, lambda: (spectrum, grid, *folds, box.shape[2], np.empty(fold_shape[0])), (5,)),
        (intensity._cross_power_numba, lambda: (spectrum, spectrum * (0.5 - 0.25j), spectrum.conj(), box.shape[2], np.empty(fold_shape[0])), (4,)),
        (intensity._all_whole_numba, lambda: (np.round(vals).ravel(), np.zeros(3, np.bool_)), (1,)),
    ]
    for kernel, args, outs in cases:
        module = next(m for m in (morphology, preprocessing, intensity) if hasattr(m, kernel.__name__))
        twin = getattr(module, kernel.__name__ + "_serial")
        if kernel is intensity._calculate_spatial_features_numba:
            # The pair loop runs with fast math: the vectorized sums of the kernel and of
            # its twin can differ in the last bits (x86-64), so the twin matches to 1e-12
            twins_same = twins_same and bool(np.allclose(kernel(*args()), twin(*args()), rtol=1e-12, atol=0.0))
        else:
            twins_same = twins_same and outputs(kernel, args(), outs) == outputs(twin, args(), outs)
# The compiled cubic kernels give scipy's bits where the scipy build fuses the multiply-add
# as the kernels do (the arm64 wheels); elsewhere (the x86-64 wheels) the values lie within
# 1e-13 of scipy's largest value
from scipy.ndimage import affine_transform, correlate1d

weight = 1.0 - 2.0**-30  # the probe of the scipy_fuses fixture of conftest.py
fuses = correlate1d(np.array([1.0, 1.0, 2.0**-30]), np.array([weight, -1.0, weight]), mode="constant")[1] != 0.0

cubic_shape, cubic_scale, cubic_shift, _ = preprocessing._output_grid(vals.shape, (1.0,) * 3, (1.25,) * 3)
cubic_ref = affine_transform(vals, np.array(cubic_scale), np.array(cubic_shift), output_shape=cubic_shape, order=3, mode="nearest")
cubic_new = preprocessing.resample_image(Image(vals, (1.0,) * 3, (0.0,) * 3), (1.25,) * 3, "cubic").array
cubic_same = cubic_new.tobytes() == cubic_ref.tobytes() if fuses else bool(np.abs(cubic_new - cubic_ref).max() <= 1e-13 * np.abs(cubic_ref).max())
statuses = {entry["status"] for entry in pipeline.get_log()}
after = signatures()
new = {name: [s for s in after[name] if s not in before.get(name, [])] for name in after}
print(json.dumps({
    "statuses": sorted(statuses),
    "new": {k: v for k, v in new.items() if v},
    "same at one thread": same,
    "new thread": seen == [pictologics.get_num_threads()],
    "twins same": twins_same,
    "cubic matches scipy": cubic_same,
}))
"""


def test_a_pipeline_run_after_the_import_compiles_no_kernel(tmp_path: Path) -> None:
    # With the compiler on, the import warms every kernel signature that the package
    # reaches: the standard templates on a CT-like image (large enough for the parallel
    # paths), a filter with many FBS levels, 256 grey levels, masks of each type in row and
    # column order, label maps of five types in run_rois, save_image of five types, direct
    # calls with other input types, and a big-endian DICOM series. One thread gives the
    # same features as the default (also Moran's I and Geary's C; with SciPy 1.18 or later,
    # whose FFT splits its work by thread, these two agree to 1e-12 on the FFT path, and the
    # features of the FFT LoG to 1e-6), and a new thread starts with the default. The check
    # runs in its own numba cache, so it never writes into a cache that another process
    # builds at the same time. Each serial twin of a parallel kernel gives the bits of its
    # kernel on 10 sizes (the fast-math pair loop of Moran's I within 1e-12), and the
    # compiled cubic resample gives scipy's bits where the scipy build fuses the
    # multiply-add as the kernels do, else values within 1e-13 of scipy's largest one.
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
        "twins same": True,
        "cubic matches scipy": True,
    }
