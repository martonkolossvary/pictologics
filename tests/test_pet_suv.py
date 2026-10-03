"""Tests for the PET SUV factor (pictologics.loaders.pet_suv) and load_image(suv=...)."""

from __future__ import annotations

import os

os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from pictologics import load_image
from pictologics.loaders.pet_suv import _suv_factor

DOSE, HALF_LIFE, WEIGHT, HEIGHT = 370e6, 6586.2, 75.0, 1.80


def _pet(k: int = 0, **tags: Any) -> Dataset:
    """The header of slice `k` of a PET series: BQML, decay corrected to the series start
    at 11:00, injected at 10:00, acquired at 11:00:0k (tags override; None deletes)."""
    ds = Dataset()
    ds.Modality, ds.Units = "PT", "BQML"
    ds.CorrectedImage, ds.DecayCorrection = ["ATTN", "DECY"], "START"
    ds.SeriesDate, ds.SeriesTime = "20260101", "110000"
    ds.AcquisitionDate, ds.AcquisitionTime = "20260101", f"1100{k:02d}"
    ds.PatientWeight, ds.PatientSize, ds.PatientSex = WEIGHT, HEIGHT, "M"
    info = Dataset()
    info.RadionuclideTotalDose, info.RadionuclideHalfLife = DOSE, HALF_LIFE
    info.RadiopharmaceuticalStartTime = "100000"
    ds.RadiopharmaceuticalInformationSequence = [info]
    for key, value in tags.items():
        if value is None:
            delattr(ds, key)
        else:
            setattr(ds, key, value)
    return ds


def _bw(decay_seconds: float = 3600.0) -> float:
    return WEIGHT * 1000.0 / (DOSE * 2.0 ** (-decay_seconds / HALF_LIFE))


def test_suv_factors_of_the_three_types() -> None:
    bmi = WEIGHT / HEIGHT**2
    male = 9270.0 * WEIGHT / (6680.0 + 216.0 * bmi)
    female = 9270.0 * WEIGHT / (8780.0 + 244.0 * bmi)
    area = 0.007184 * WEIGHT**0.425 * (100.0 * HEIGHT) ** 0.725 * 1e4  # cm2
    series = [_pet(0), _pet(1)]
    assert _suv_factor(series, "bw") == pytest.approx(_bw(), rel=1e-12)
    assert _suv_factor(series, "lbm") == pytest.approx(_bw() * male / WEIGHT, rel=1e-12)
    assert _suv_factor([_pet(PatientSex="F")], "lbm") == pytest.approx(
        _bw() * female / WEIGHT, rel=1e-12
    )
    assert _suv_factor(series, "bsa") == pytest.approx(_bw() * area / (WEIGHT * 1000.0), rel=1e-12)


def test_suv_injection_and_scan_times() -> None:
    # ADMIN takes the dose without decay; a start date and time (with a UTC offset) or an
    # acquisition date and time count as given; a start time after the scan time is the
    # day before (midnight); without acquisition times, the series time is the scan time.
    assert _suv_factor([_pet(DecayCorrection="ADMIN")], "bw") == pytest.approx(
        WEIGHT * 1000.0 / DOSE
    )
    dated = _pet()
    dated.RadiopharmaceuticalInformationSequence[
        0
    ].RadiopharmaceuticalStartDateTime = "20260101103000+0100"
    assert _suv_factor([dated], "bw") == pytest.approx(_bw(1800.0), rel=1e-12)
    late = _pet(
        SeriesDate="20260102", SeriesTime="003000", AcquisitionDate=None, AcquisitionTime=None
    )
    late.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime = "233000"
    assert _suv_factor([late], "bw") == pytest.approx(_bw(3600.0), rel=1e-12)
    acquired = _pet(AcquisitionDateTime="20260101110010", AcquisitionTime=None)
    assert _suv_factor([acquired], "bw") == pytest.approx(_bw(), rel=1e-12)


def test_suv_scan_time_of_a_post_processed_series() -> None:
    # A series time after the acquisition (QIBA): the GE private scan time, else the time
    # back-computed from the frame times, else the earliest acquisition (with a warning).
    def post(**tags: Any) -> list[Dataset]:
        return [_pet(1, SeriesTime="150000", **tags), _pet(0, SeriesTime="150000")]

    ge = post()
    ge[0].private_block(0x0009, "GEMS_PETD_01", create=True).add_new(0x0D, "DT", "20260101104500")
    assert _suv_factor(ge, "bw") == pytest.approx(_bw(2700.0), rel=1e-12)
    raw = post()
    raw[0].private_block(0x0009, "GEMS_PETD_01", create=True).add_new(
        0x0D, "UN", b"20260101104500 "
    )
    assert _suv_factor(raw, "bw") == pytest.approx(_bw(2700.0), rel=1e-12)
    framed = post(FrameReferenceTime=60000.0, ActualFrameDuration=120000)
    rate = math.log(2.0) / HALF_LIFE
    average = math.log(rate * 120.0 / (1.0 - math.exp(-rate * 120.0))) / rate
    assert _suv_factor(framed, "bw") == pytest.approx(_bw(3601.0 - 60.0 + average), rel=1e-12)
    with pytest.warns(UserWarning, match="post-processed series"):
        assert _suv_factor(post(), "bw") == pytest.approx(_bw(3600.0), rel=1e-12)


def test_suv_of_other_units() -> None:
    # CNTS: the Philips SUV factor, else its activity factor times the BQML factor; GML is
    # SUVbw already.
    philips = _pet(Units="CNTS")
    philips.private_block(0x7053, "Philips PET Private Group", create=True).add_new(
        0x00, "DS", "0.0005"
    )
    assert _suv_factor([philips], "bw") == 0.0005
    activity = _pet(Units="CNTS")
    activity.private_block(0x7053, "Philips PET Private Group", create=True).add_new(
        0x09, "UN", b"2.5 "
    )
    assert _suv_factor([activity], "bw") == pytest.approx(2.5 * _bw(), rel=1e-12)
    gml = _pet(Units="GML", RadiopharmaceuticalInformationSequence=None)
    assert _suv_factor([gml], "bw") == 1.0
    assert _suv_factor([gml], "lbm") == pytest.approx(
        _suv_factor([_pet()], "lbm") / _bw(), rel=1e-12
    )


@pytest.mark.parametrize(
    ("tags", "suv", "message"),
    [
        ({}, "sul", "suv must be one of"),
        ({"Modality": "CT"}, "bw", "needs a PET image"),
        ({"CorrectedImage": "ATTN"}, "bw", "attenuation and decay corrected"),
        ({"DecayCorrection": "NONE"}, "bw", "attenuation and decay corrected"),
        ({"Units": "PROPCPS"}, "bw", "Units 'PROPCPS' are not supported"),
        ({"Units": "CNTS"}, "bw", "Philips SUV factor"),
        (
            {"RadiopharmaceuticalInformationSequence": None},
            "bw",
            "RadiopharmaceuticalInformationSequence",
        ),
        ({"PatientWeight": None}, "bw", "attribute PatientWeight, which is missing"),
        ({"PatientWeight": 0}, "bw", "PatientWeight above 0"),
        ({"PatientSize": 180}, "bsa", "not a height in metres"),
        ({"PatientSex": "O"}, "lbm", "PatientSex M or F"),
        ({"SeriesTime": None}, "bw", "attribute SeriesTime"),
    ],
)
def test_suv_needs_its_attributes(tags: dict[str, Any], suv: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _suv_factor([_pet(**tags)], suv)


def test_suv_needs_a_dose_in_bq() -> None:
    mbq = _pet()
    mbq.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose = 370
    with pytest.raises(ValueError, match="below 1 MBq"):
        _suv_factor([mbq], "bw")


def _write(folder: Path, slopes: tuple[float, ...]) -> np.ndarray:
    """A PET series of one slice per rescale slope; returns its activity volume (X, Y, Z)."""
    rng = np.random.default_rng(0)
    planes = []
    series = generate_uid()
    for k, slope in enumerate(slopes):
        pixels = rng.integers(0, 3000, (4, 3), dtype=np.int16)
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.128"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = _pet(k)
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = (
            meta.MediaStorageSOPClassUID,
            meta.MediaStorageSOPInstanceUID,
        )
        ds.SeriesInstanceUID, ds.InstanceNumber = series, k + 1
        ds.Rows, ds.Columns = pixels.shape
        ds.PixelSpacing, ds.SliceThickness = [4.0, 4.0], 3.0
        ds.ImagePositionPatient = [0.0, 0.0, 3.0 * k]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.RescaleSlope, ds.RescaleIntercept = slope, 0.0
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.PixelData = pixels.tobytes()
        ds.save_as(folder / f"{k}.dcm", enforce_file_format=True)
        planes.append(pixels.T * slope)
    return np.stack(planes, axis=-1)


def test_load_image_as_suv(tmp_path: Path) -> None:
    # A series (each slice with its own rescale slope) and a single file load as SUV; suv
    # needs the rescale and a DICOM PET image (not a NIfTI, SEG or RTSTRUCT file).
    activity = _write(tmp_path, (0.5, 1.0, 2.0))
    for suv in ("bw", "lbm"):
        loaded = load_image(tmp_path, suv=suv)
        np.testing.assert_allclose(loaded.array, activity * _suv_factor([_pet()], suv), rtol=1e-12)
    single = load_image(tmp_path / "2.dcm", suv="bw")
    np.testing.assert_allclose(single.array[..., 0], activity[..., 2] * _bw(), rtol=1e-12)
    with pytest.raises(ValueError, match="apply_rescale=True"):
        load_image(tmp_path, suv="bw", apply_rescale=False)
    nifti = tmp_path / "image.nii.gz"
    nifti.write_bytes(b"")
    with pytest.raises(ValueError, match="needs a DICOM PET image, not the file"):
        load_image(nifti, suv="bw")
    rtstruct = pydicom.dcmread(tmp_path / "0.dcm")
    rtstruct.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
    rtstruct.save_as(tmp_path / "rtstruct.dcm")
    with pytest.raises(ValueError, match="not the mask file"):
        load_image(tmp_path / "rtstruct.dcm", suv="bw")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning for a plain series
        load_image(tmp_path / "0.dcm", suv="bsa")
