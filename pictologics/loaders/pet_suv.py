"""
PET SUV
=======

The factor from the activity concentration of a DICOM PET series to its standardized
uptake value (SUV), by the QIBA vendor-neutral pseudo-code (D. Clunie, 2018-06-26):
https://qibawiki.rsna.org/index.php/Standardized_Uptake_Value_(SUV). The SUV can be
normalised by the body weight (bw), the lean body mass (lbm by the Janmahasatian formula;
lbm_james by the James formula) or the body surface area (bsa, the Du Bois formula).
"""

from __future__ import annotations

import datetime
import math
import warnings
from collections.abc import Sequence
from typing import Any, Optional

from pydicom.valuerep import DA, DT, TM

_SUV_TYPES = ("bw", "lbm", "lbm_james", "bsa")


def _suv_factor(datasets: Sequence[Any], suv: str) -> float:
    """The factor from the rescaled values of the PET slices `datasets` (one series; the
    first gives the patient and dose attributes) to the SUV `suv`.

    SUVbw (g/ml) = activity concentration (Bq/ml) x weight (g) / injected dose (Bq),
    decayed to the scan start. SUVlbm (g/ml) = SUVbw x lean body mass / weight; SUVbsa
    (cm2/ml) = SUVbw x body surface area (cm2) / weight (g). Units CNTS take the Philips
    SUV factor (or its activity factor), and Units GML are SUVbw already, as in QIBA.
    DecayCorrection ADMIN (decay corrected to the injection, not in the QIBA pseudo-code)
    takes the injected dose without decay.

    Raises:
        ValueError: If `suv` is unknown, the image is not PET, or an attribute that the
            SUV needs is missing, empty, zero or not supported (the message names it).
    """
    if suv not in _SUV_TYPES:
        raise ValueError(f"suv must be one of {_SUV_TYPES}, not {suv!r}.")
    ref = datasets[0]
    modality = str(getattr(ref, "Modality", ""))
    if modality != "PT":
        raise ValueError(f"suv needs a PET image (Modality PT), not Modality {modality!r}.")
    corrected = {str(value).upper() for value in _values(ref, "CorrectedImage")}
    decay = str(getattr(ref, "DecayCorrection", "")).upper()
    if not {"ATTN", "DECY"} <= corrected or decay not in ("START", "ADMIN"):
        raise ValueError(
            "SUV needs attenuation and decay corrected images: CorrectedImage with ATTN "
            f"and DECY (here {sorted(corrected)}) and DecayCorrection START or ADMIN (here "
            f"{decay!r})."
        )
    units = str(getattr(ref, "Units", "")).upper()
    if units == "GML":
        factor = 1.0  # SUVbw already
    else:
        if units == "CNTS":
            suv_factor = _philips(ref, 0x00)  # (7053,1000): the SUVbw factor
            if suv_factor is not None:
                factor = suv_factor
            else:
                to_bqml = _philips(ref, 0x09)  # (7053,1009) x RescaleSlope: to Bq/ml
                if to_bqml is None:
                    raise ValueError(
                        "PET Units CNTS need the Philips SUV factor (7053,1000) or the "
                        "activity concentration factor (7053,1009)."
                    )
                factor = to_bqml * _bw_factor(datasets, decay)
        elif units == "BQML":
            factor = _bw_factor(datasets, decay)
        else:
            raise ValueError(f"PET Units {units!r} are not supported for SUV (BQML, CNTS or GML).")
    if suv == "bw":
        return factor
    weight = _positive(ref, "PatientWeight")  # kg
    height = _positive(ref, "PatientSize")  # m
    if height > 3.0:
        raise ValueError(
            f"PatientSize {height:g} is not a height in metres (DICOM uses metres; the file "
            "may hold centimetres)."
        )
    if suv == "bsa":
        area = 0.007184 * math.pow(weight, 0.425) * math.pow(100.0 * height, 0.725) * 1e4  # cm2
        return factor * area / (1000.0 * weight)
    sex = str(getattr(ref, "PatientSex", "")).upper()
    if sex not in ("M", "F"):
        raise ValueError(f"SUV {suv} needs PatientSex M or F, not {sex!r}.")
    if suv == "lbm":  # Janmahasatian
        bmi = weight / height**2
        lean = 9270.0 * weight / ((6680.0 + 216.0 * bmi) if sex == "M" else (8780.0 + 244.0 * bmi))
    else:  # James, with the height in cm
        scale, offset = (1.10, 128.0) if sex == "M" else (1.07, 148.0)
        lean = scale * weight - offset * (weight / (100.0 * height)) ** 2
        if lean <= 0:
            raise ValueError(
                f"The James formula gives no lean body mass ({lean:.1f} kg) for "
                f"{weight:g} kg and {height:g} m: use suv='lbm' (Janmahasatian)."
            )
    return factor * lean / weight


def _bw_factor(datasets: Sequence[Any], decay: str) -> float:
    """The SUVbw factor of activity concentrations (Bq/ml): weight (g) / decayed dose (Bq)."""
    ref = datasets[0]
    items = _values(ref, "RadiopharmaceuticalInformationSequence")
    if not items:
        raise ValueError("SUV needs the RadiopharmaceuticalInformationSequence (0054,0016).")
    info = items[0]
    dose = _positive(info, "RadionuclideTotalDose")  # Bq
    if dose < 1e6:
        raise ValueError(
            f"RadionuclideTotalDose {dose:g} Bq is below 1 MBq: DICOM uses Bq, and the file "
            "may hold MBq."
        )
    if decay == "START":
        half_life = _positive(info, "RadionuclideHalfLife")  # s
        scan = _scan_datetime(datasets, half_life)
        start_value = str(getattr(info, "RadiopharmaceuticalStartDateTime", "") or "")
        if start_value:
            start = _naive(DT(start_value))
        else:  # no date: the day of the scan, or the day before across midnight
            start_time = _required(info, "RadiopharmaceuticalStartTime")
            start = datetime.datetime.combine(scan.date(), TM(start_time))
            if start > scan:
                start -= datetime.timedelta(days=1)
        dose *= 2.0 ** (-(scan - start).total_seconds() / half_life)
    return 1000.0 * _positive(ref, "PatientWeight") / dose


def _scan_datetime(datasets: Sequence[Any], half_life: float) -> datetime.datetime:
    """The start of the scan, by QIBA: the series date and time unless they are after an
    acquisition (a post-processed series); then the GE private scan date and time, else
    the start back-computed from the frame times of the first slice, else the earliest
    acquisition."""
    ref = datasets[0]
    series = datetime.datetime.combine(
        DA(_required(ref, "SeriesDate")), TM(_required(ref, "SeriesTime"))
    )
    acquired = [time for time in map(_acquisition, datasets) if time is not None]
    if not acquired or series <= min(acquired):
        return series
    try:
        ge = ref.get_private_item(0x0009, 0x0D, "GEMS_PETD_01").value
    except KeyError:
        ge = None
    if ge:
        return _naive(DT(ge if isinstance(ge, str) else ge.decode().strip()))
    reference_ms = float(getattr(ref, "FrameReferenceTime", 0) or 0)
    duration_ms = float(getattr(ref, "ActualFrameDuration", 0) or 0)
    first = _acquisition(ref)
    if reference_ms > 0 and duration_ms > 0 and first is not None:
        rate = math.log(2.0) / half_life
        decay_in_frame = rate * duration_ms / 1000.0
        average = math.log(decay_in_frame / (1.0 - math.exp(-decay_in_frame))) / rate
        return first + datetime.timedelta(seconds=average - reference_ms / 1000.0)
    warnings.warn(
        "The PET SeriesDate and SeriesTime are after the acquisition (a post-processed "
        "series): the SUV decay starts at the earliest AcquisitionTime.",
        UserWarning,
        stacklevel=4,
    )
    return min(acquired)


def _acquisition(ds: Any) -> Optional[datetime.datetime]:
    """The acquisition date and time of a slice, or None without an acquisition time."""
    value = str(getattr(ds, "AcquisitionDateTime", "") or "")
    if value:
        return _naive(DT(value))
    time = str(getattr(ds, "AcquisitionTime", "") or "")
    date = str(getattr(ds, "AcquisitionDate", "") or getattr(ds, "SeriesDate", "") or "")
    if not time or not date:
        return None
    return datetime.datetime.combine(DA(date), TM(time))


def _naive(value: datetime.datetime) -> datetime.datetime:
    """A date and time without its UTC offset (the scanner keeps local times)."""
    return datetime.datetime.combine(value.date(), value.time())


def _philips(ds: Any, element: int) -> Optional[float]:
    """A number of the Philips PET private group (7053,10xx), or None."""
    try:
        value = ds.get_private_item(0x7053, element, "Philips PET Private Group").value
    except KeyError:
        return None
    number = float(value.decode().strip() if isinstance(value, bytes) else value)
    return number if number > 0 else None


def _values(ds: Any, keyword: str) -> list[Any]:
    """The values of a multi-valued attribute (or a sequence) as a list ([] if absent)."""
    value = getattr(ds, keyword, None)
    if value is None or value == "":
        return []
    return [value] if isinstance(value, str) else list(value)


def _required(ds: Any, keyword: str) -> str:
    """The text of an attribute that the SUV needs."""
    value = getattr(ds, keyword, None)
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"SUV needs the DICOM attribute {keyword}, which is missing or empty.")
    return text


def _positive(ds: Any, keyword: str) -> float:
    """The value of a number attribute that the SUV needs, above 0."""
    value = float(_required(ds, keyword))
    if value <= 0:
        raise ValueError(f"SUV needs the DICOM attribute {keyword} above 0, not {value:g}.")
    return value
