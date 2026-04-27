"""
FHIR parsing for Synthea-style bundles (Patient, Encounter, Observation, Condition).

Synthea typically exports one JSON file per patient containing a Bundle with
``entry[*].resource`` items. Bulk FHIR uses NDJSON with one resource per line.
This module supports both layouts.

EMS transport label (documented priority):
1) Custom extension on Encounter (used by mock data and any custom exporter):
   URL ``http://csci1470.local/fhir/StructureDefinition/ems-transport``
   - valueBoolean: True = transport (1), False = refusal (0)
   - or valueString / valueCode: ``transport`` / ``refusal`` (case-insensitive)

2) FHIR R4 ``Encounter.hospitalization.dischargeDisposition`` when text/coding
   indicates left AMA / without treatment → refusal; admitted/transferred → transport.

3) If no signal is present, the row is skipped (``drop_unlabeled=True``) so training
   only uses explicit or inferable outcomes. Standard Synthea rarely encodes EMS
   refusal; use mock data or add the extension in a post-processing step.
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

# LOINC codes commonly used in US Core / Synthea-like exports for vitals
LOINC_HEART_RATE = "8867-4"
LOINC_SYSTOLIC = "8480-6"
LOINC_DIASTOLIC = "8462-4"
LOINC_RESP_RATE = "9279-1"
LOINC_SPO2 = "59408-5"  # alternate: 2708-6
LOINC_TEMP_F = "8310-5"  # Body temperature

EMS_TRANSPORT_EXT_URL = "http://csci1470.local/fhir/StructureDefinition/ems-transport"

# Encounter classes / SNOMED hints that we treat as emergency / pre-hospital style
EMS_CLASS_CODES = {"EMER"}
EMS_TYPE_SNOMED = {
    "50849002",  # Emergency room admission (procedure)
    "185347001",  # Encounter for problem
    "32485007",  # Hospital admission
}
FHIR_FILE_SUFFIXES = (".json", ".ndjson")
VITAL_LOINC_TO_COLUMN = {
    LOINC_HEART_RATE: "heart_rate",
    LOINC_SYSTOLIC: "systolic_bp",
    LOINC_DIASTOLIC: "diastolic_bp",
    LOINC_RESP_RATE: "respiratory_rate",
    LOINC_SPO2: "spo2",
    LOINC_TEMP_F: "temperature",
}
TRANSPORT_COMPLAINTS = ("Chest pain", "Shortness of breath", "Syncope")
REFUSAL_COMPLAINTS = ("General weakness", "Fall injury", "Abdominal pain")


def _parse_dt(value: str | None) -> datetime | None:
    """
    Parse an ISO-like FHIR date or datetime string into a ``datetime`` object.

    Parameters
    ----------
    value
        Raw date or datetime string pulled from a FHIR resource field.

    Returns
    -------
    parsed_datetime
        Parsed ``datetime`` value, or ``None`` when the input is missing or invalid.
    """
    if not value or not isinstance(value, str):
        return None
    v = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return None


def _patient_id_from_ref(ref: str | None) -> str | None:
    """
    Extract the patient identifier from a FHIR reference string.

    Parameters
    ----------
    ref
        FHIR reference such as ``Patient/<id>`` or ``urn:uuid:<id>``.

    Returns
    -------
    patient_id
        Patient identifier string parsed from the reference, or ``None`` if unavailable.
    """
    if not ref:
        return None
    # "Patient/abc" or "urn:uuid:abc"
    if "/" in ref:
        return ref.split("/")[-1]
    return ref


def has_fhir_resources(path: str | Path) -> bool:
    """
    Check whether a path already contains FHIR JSON or NDJSON resources.

    Parameters
    ----------
    path
        Candidate file or directory path to inspect for FHIR resources.

    Returns
    -------
    has_resources
        ``True`` when the path points to usable FHIR input data, otherwise ``False``.
    """
    resource_path = Path(path)
    if resource_path.is_file():
        return resource_path.suffix.lower() in FHIR_FILE_SUFFIXES
    if not resource_path.is_dir():
        return False
    return any(resource_path.glob("**/*.json")) or any(resource_path.glob("**/*.ndjson"))


def _iter_resources_from_obj(data: Any) -> Iterator[dict[str, Any]]:
    """
    Iterate over FHIR resources stored inside a parsed JSON-like object.

    Parameters
    ----------
    data
        Parsed JSON value that may represent a single resource, bundle, or list.

    Returns
    -------
    resources
        Iterator yielding each FHIR resource dictionary contained in ``data``.
    """
    if isinstance(data, list):
        for item in data:
            yield from _iter_resources_from_obj(item)
        return
    if not isinstance(data, dict):
        return
    if data.get("resourceType") == "Bundle":
        for entry in data.get("entry", []) or []:
            resource = entry.get("resource")
            if resource:
                yield resource
        return
    if data.get("resourceType"):
        yield data


def _iter_resources_from_path(path: Path) -> Iterator[dict[str, Any]]:
    """
    Iterate over FHIR resource dictionaries stored under a file or directory path.

    Parameters
    ----------
    path
        File or directory containing JSON or NDJSON FHIR resources.

    Returns
    -------
    resources
        Iterator yielding parsed FHIR resource dictionaries discovered at the given path.
    """
    if path.is_file():
        yield from _iter_resources_from_file(path)
        return
    for p in sorted(path.glob("**/*.json")):
        yield from _iter_resources_from_file(p)
    for p in sorted(path.glob("**/*.ndjson")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield from _iter_resources_from_obj(json.loads(line))


def _iter_resources_from_file(path: Path) -> Iterator[dict[str, Any]]:
    """
    Iterate over FHIR resource dictionaries contained in a single JSON file.

    Parameters
    ----------
    path
        JSON file containing a bundle, list of resources, or single resource object.

    Returns
    -------
    resources
        Iterator yielding each parsed FHIR resource dictionary from the file.
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    yield from _iter_resources_from_obj(data)


def _first_coding_list(concept: dict | None) -> list[dict]:
    """
    Extract the list of coding dictionaries from a FHIR CodeableConcept-like object.

    Parameters
    ----------
    concept
        Dictionary that may contain a ``coding`` list.

    Returns
    -------
    codings
        List of coding dictionaries found on the concept, or an empty list if absent.
    """
    if not concept:
        return []
    return [coding for coding in concept.get("coding", []) or [] if isinstance(coding, dict)]


def _concept_text(concept: dict | None) -> str:
    """
    Derive a human-readable text label from a FHIR CodeableConcept-like object.

    Parameters
    ----------
    concept
        Dictionary that may contain ``text`` or ``coding`` fields.

    Returns
    -------
    text
        Preferred text, display, or code string describing the concept.
    """
    if not concept:
        return ""
    t = concept.get("text")
    if t:
        return str(t)
    codings = _first_coding_list(concept)
    if codings:
        d = codings[0].get("display") or codings[0].get("code")
        return str(d or "")
    return ""


def _obs_effective_dt(obs: dict) -> datetime | None:
    """
    Extract the effective datetime associated with an observation.

    Parameters
    ----------
    obs
        Observation resource dictionary that may contain ``effectiveDateTime`` or
        ``effectivePeriod`` information.

    Returns
    -------
    effective_datetime
        Observation timestamp as a ``datetime`` object, or ``None`` if unavailable.
    """
    eff = obs.get("effectiveDateTime")
    if eff:
        return _parse_dt(eff)
    pr = obs.get("effectivePeriod") or {}
    return _parse_dt(pr.get("start"))


def _obs_loinc_code(obs: dict) -> str | None:
    """
    Extract the LOINC code assigned to an observation, when present.

    Parameters
    ----------
    obs
        Observation resource dictionary whose code field may contain LOINC codings.

    Returns
    -------
    loinc_code
        LOINC code string associated with the observation, or ``None`` if absent.
    """
    for coding in obs.get("code", {}).get("coding", []) or []:
        if coding.get("system") in {"http://loinc.org", "http://loinc.org/"}:
            return coding.get("code")
    return None


def _obs_numeric_value(obs: dict) -> float | None:
    """
    Extract a numeric measurement value from an observation resource.

    Parameters
    ----------
    obs
        Observation resource dictionary that may store its value in ``valueQuantity``
        or ``valueString``.

    Returns
    -------
    numeric_value
        Floating-point observation value, or ``None`` when no numeric value can be parsed.
    """
    vq = obs.get("valueQuantity")
    if vq and vq.get("value") is not None:
        try:
            return float(vq["value"])
        except (TypeError, ValueError):
            return None
    # valueString sometimes holds numbers
    vs = obs.get("valueString")
    if vs is not None:
        try:
            return float(vs)
        except (TypeError, ValueError):
            return None
    return None


def _age_at_encounter(birth_date: str | None, enc_start: str | None) -> float | None:
    """
    Compute patient age in years at the start of an encounter.

    Parameters
    ----------
    birth_date
        Patient birth date string from the FHIR patient resource.
    enc_start
        Encounter start date or datetime string.

    Returns
    -------
    age_years
        Patient age in years at encounter time, or ``None`` when the dates are unusable.
    """
    bd = _parse_dt(birth_date)
    es = _parse_dt(enc_start)
    if not bd or not es:
        return None
    days = (es - bd).days
    if days < 0:
        return None
    return days / 365.25


def _is_ems_like_encounter(enc: dict) -> bool:
    """
    Determine whether an encounter should be treated as EMS-like for modeling.

    Parameters
    ----------
    enc
        Encounter resource dictionary to classify with the EMS heuristics.

    Returns
    -------
    is_ems_like
        ``True`` when the encounter matches the configured emergency-class heuristics,
        otherwise ``False``.
    """
    cls = enc.get("class") or {}
    code = (cls.get("code") or "").upper()
    if code in EMS_CLASS_CODES:
        return True
    for t in enc.get("type", []) or []:
        for c in t.get("coding", []) or []:
            if c.get("code") in EMS_TYPE_SNOMED:
                return True
            disp = (c.get("display") or "").lower()
            if "emergency" in disp or "ems" in disp or "ambulance" in disp:
                return True
    return False


def _extract_transport_label(encounter: dict) -> int | None:
    """
    Extract the transport-versus-refusal label from an encounter resource.

    Parameters
    ----------
    encounter
        Encounter resource dictionary that may contain explicit or inferred transport labels.

    Returns
    -------
    label
        ``1`` for transport, ``0`` for refusal, or ``None`` when no label can be inferred.
    """
    for ext in encounter.get("extension", []) or []:
        if ext.get("url") != EMS_TRANSPORT_EXT_URL:
            continue
        if "valueBoolean" in ext:
            return 1 if ext["valueBoolean"] else 0
        for key in ("valueCode", "valueString"):
            if key in ext and ext[key] is not None:
                s = str(ext[key]).strip().lower()
                if s in ("1", "true", "transport", "yes"):
                    return 1
                if s in ("0", "false", "refusal", "no", "refused"):
                    return 0
    hosp = encounter.get("hospitalization") or {}
    dd = hosp.get("dischargeDisposition")
    if not dd:
        return None
    text = _concept_text(dd).lower()
    codings = " ".join(
        (c.get("code") or "") + " " + (c.get("display") or "").lower()
        for c in _first_coding_list(dd)
    )
    blob = text + " " + codings
    refusal_tokens = (
        "against medical advice",
        "left without",
        "ama",
        "refusal",
        "did not wait",
        "lwbs",
    )
    transport_tokens = (
        "admitted",
        "hospitalization",
        "transfer",
        "inpatient",
        "icu",
    )
    if any(t in blob for t in refusal_tokens):
        return 0
    if any(t in blob for t in transport_tokens):
        return 1
    return None


def _chief_complaint(enc: dict, conditions: list[dict], patient_id: str) -> str:
    """
    Derive a chief-complaint text string for an encounter.

    Parameters
    ----------
    enc
        Encounter resource dictionary whose reason codes are checked first.
    conditions
        Condition resources used as a fallback source of complaint text.
    patient_id
        Identifier of the encounter's patient used to match related conditions.

    Returns
    -------
    chief_complaint
        Chief-complaint string associated with the encounter, truncated for storage.
    """
    # Encounter.reasonCode (Synthea often populates)
    for rc in enc.get("reasonCode", []) or []:
        t = _concept_text(rc)
        if t:
            return t[:512]
    # First active condition for this patient with onset near encounter start
    enc_start = _parse_dt((enc.get("period") or {}).get("start"))
    best = ""
    best_delta = None
    for cond in conditions:
        subj = _patient_id_from_ref((cond.get("subject") or {}).get("reference"))
        if subj != patient_id:
            continue
        onset = cond.get("onsetDateTime") or (cond.get("onsetPeriod") or {}).get("start")
        ot = _parse_dt(onset)
        code_text = _concept_text((cond.get("code") or {}))
        if not code_text:
            continue
        if enc_start and ot:
            delta = abs((ot - enc_start).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best = code_text
        elif not best:
            best = code_text
    return (best or "unknown")[:512]


def _latest_numeric_value(candidates: list[tuple[datetime, float]]) -> float | None:
    """
    Select the most recent numeric vital value from timestamped candidates.

    Parameters
    ----------
    candidates
        Timestamp and value pairs collected for one vital-sign channel.

    Returns
    -------
    latest_value
        Most recent numeric value in the candidate list, or ``None`` if empty.
    """
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def parse_fhir_to_dataframe(
    fhir_path: str | Path,
    *,
    drop_unlabeled: bool = True,
    only_ems_like: bool = True,
) -> pd.DataFrame:
    """
    Parse FHIR resources into one modeling row per qualifying encounter.

    Parameters
    ----------
    fhir_path
        File or directory containing JSON or NDJSON FHIR resources to parse.
    drop_unlabeled
        Whether to discard encounters for which no transport label can be extracted.
    only_ems_like
        Whether to keep only encounters that match the EMS-like filtering heuristics.

    Returns
    -------
    encounter_dataframe
        Dataframe with one row per retained encounter containing demographics, vitals,
        chief complaint, and transport label information.
    """
    path = Path(fhir_path)
    patients: dict[str, dict] = {}
    observations: list[dict] = []
    conditions: list[dict] = []
    encounters: list[dict] = []

    for res in _iter_resources_from_path(path):
        rt = res.get("resourceType")
        rid = res.get("id")
        if rt == "Patient" and rid:
            patients[rid] = res
        elif rt == "Observation":
            observations.append(res)
        elif rt == "Condition":
            conditions.append(res)
        elif rt == "Encounter":
            encounters.append(res)

    # Index observations by patient id
    obs_by_patient: dict[str, list[dict]] = {}
    for obs in observations:
        pid = _patient_id_from_ref((obs.get("subject") or {}).get("reference"))
        if not pid:
            continue
        obs_by_patient.setdefault(pid, []).append(obs)

    rows: list[dict[str, Any]] = []
    for enc in encounters:
        if only_ems_like and not _is_ems_like_encounter(enc):
            continue
        label = _extract_transport_label(enc)
        if label is None and drop_unlabeled:
            continue
        pid = _patient_id_from_ref((enc.get("subject") or {}).get("reference"))
        if not pid or pid not in patients:
            continue
        pat = patients[pid]
        period = enc.get("period") or {}
        start_s = period.get("start")
        end_s = period.get("end")
        start = _parse_dt(start_s)
        end = _parse_dt(end_s)
        if start is None:
            continue
        if end is None:
            end = start + timedelta(hours=6)

        window_lo = start - timedelta(hours=24)
        window_hi = end

        vitals_lists: dict[str, list[tuple[datetime, float]]] = {
            column: [] for column in VITAL_LOINC_TO_COLUMN.values()
        }

        for obs in obs_by_patient.get(pid, []):
            ot = _obs_effective_dt(obs)
            if ot is None or ot < window_lo or ot > window_hi:
                continue
            code = _obs_loinc_code(obs)
            if not code:
                continue
            val = _obs_numeric_value(obs)
            if val is None:
                continue
            col = VITAL_LOINC_TO_COLUMN.get(code)
            if col is None:
                continue
            vitals_lists[col].append((ot, val))

        age = _age_at_encounter(pat.get("birthDate"), start_s)
        sex = (pat.get("gender") or "unknown").lower()
        cc = _chief_complaint(enc, conditions, pid)

        row = {
            "encounter_id": enc.get("id"),
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "heart_rate": _latest_numeric_value(vitals_lists["heart_rate"]),
            "systolic_bp": _latest_numeric_value(vitals_lists["systolic_bp"]),
            "diastolic_bp": _latest_numeric_value(vitals_lists["diastolic_bp"]),
            "respiratory_rate": _latest_numeric_value(vitals_lists["respiratory_rate"]),
            "spo2": _latest_numeric_value(vitals_lists["spo2"]),
            "temperature": _latest_numeric_value(vitals_lists["temperature"]),
            "chief_complaint": cc,
            "transport": int(label) if label is not None else None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def write_mock_fhir_directory(output_dir: str | Path, *, seed: int = 42, n_patients: int = 80) -> Path:
    """
    Generate a directory of synthetic FHIR patient bundles for the EMS pipeline.

    Parameters
    ----------
    output_dir
        Directory where the generated patient bundle JSON files should be written.
    seed
        Random seed used to make the synthetic dataset generation reproducible.
    n_patients
        Number of synthetic patient bundle files to create.

    Returns
    -------
    output_path
        Path to the directory containing the generated mock FHIR bundle files.
    """
    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for _ in range(n_patients):
        pid = str(uuid.uuid4())
        birth_year = rng.randint(1940, 2005)
        birth = f"{birth_year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        gender = rng.choice(["male", "female", "other"])

        n_enc = rng.randint(1, 3)
        entries: list[dict[str, Any]] = []

        pat = {
            "resourceType": "Patient",
            "id": pid,
            "gender": gender,
            "birthDate": birth,
        }
        entries.append({"resource": pat})

        for _e in range(n_enc):
            eid = str(uuid.uuid4())
            enc_start = datetime(
                2018 + rng.randint(0, 5),
                rng.randint(1, 12),
                rng.randint(1, 28),
                rng.randint(8, 20),
                0,
                0,
            )
            duration_h = rng.uniform(0.5, 4.0)
            enc_end = enc_start + timedelta(hours=duration_h)
            # Synthetic label (ground truth in extension); vitals loosely correlate for ML demo
            p_transport = 0.55
            if rng.random() < 0.3:
                p_transport += 0.15
            transport = rng.random() < p_transport
            if not transport and rng.random() < 0.2:
                transport = True

            # Chief complaints are biased by transport outcome so the dispatch
            # feature carries real predictive signal.
            if transport:
                cc_pool = list(TRANSPORT_COMPLAINTS) * 7 + list(REFUSAL_COMPLAINTS) * 3
            else:
                cc_pool = list(REFUSAL_COMPLAINTS) * 7 + list(TRANSPORT_COMPLAINTS) * 3
            chief_complaint_text = rng.choice(cc_pool)

            enc = {
                "resourceType": "Encounter",
                "id": eid,
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "EMER",
                },
                "type": [
                    {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "50849002",
                                "display": "Emergency room admission (procedure)",
                            }
                        ]
                    }
                ],
                "subject": {"reference": f"Patient/{pid}"},
                "period": {
                    "start": enc_start.isoformat(),
                    "end": enc_end.isoformat(),
                },
                "reasonCode": [{"text": chief_complaint_text}],
                "extension": [
                    {
                        "url": EMS_TRANSPORT_EXT_URL,
                        "valueBoolean": transport,
                    }
                ],
            }
            entries.append({"resource": enc})

            def add_obs(loinc: str, base: float, noise: float, minutes_after: int):
                """
                Append one synthetic observation resource to the current patient bundle.

                Parameters
                ----------
                loinc
                    LOINC code identifying the vital sign being generated.
                base
                    Baseline numeric value around which the observation is sampled.
                noise
                    Standard deviation of Gaussian noise added to the baseline value.
                minutes_after
                    Minutes after encounter start at which the observation is timestamped.

                Returns
                -------
                None
                    Adds the generated observation resource to the enclosing bundle entry list.
                """
                oid = str(uuid.uuid4())
                eff = enc_start + timedelta(minutes=minutes_after)
                val = base + rng.gauss(0, noise)
                entries.append(
                    {
                        "resource": {
                            "resourceType": "Observation",
                            "id": oid,
                            "status": "final",
                            "subject": {"reference": f"Patient/{pid}"},
                            "effectiveDateTime": eff.isoformat(),
                            "code": {
                                "coding": [{"system": "http://loinc.org", "code": loinc}]
                            },
                            "valueQuantity": {"value": round(val, 2), "unit": "unit"},
                        }
                    }
                )

            hr_base = 90 if transport else 70
            spo2_base = 92 if transport else 98
            # Higher noise makes class distributions overlap (more realistic).
            # ~15% chance each vital is missing (simulates field measurement gaps).
            missing_prob = 0.15
            vital_specs = (
                (LOINC_HEART_RATE, hr_base, 18, 5, 40),
                (LOINC_SYSTOLIC, 130 if transport else 118, 22, 5, 45),
                (LOINC_DIASTOLIC, 82 if transport else 76, 14, 5, 45),
                (LOINC_RESP_RATE, 22 if transport else 16, 5, 5, 50),
                (LOINC_SPO2, spo2_base, 4, 5, 55),
                (LOINC_TEMP_F, 99.2 if transport else 98.4, 1.0, 5, 60),
            )
            for loinc, base, noise, min_minutes, max_minutes in vital_specs:
                if rng.random() > missing_prob:
                    add_obs(loinc, base, noise, rng.randint(min_minutes, max_minutes))

            # Duplicate vital earlier (parser should pick most recent)
            add_obs(LOINC_HEART_RATE, hr_base - 5, 3, 2)

        bundle = {"resourceType": "Bundle", "type": "collection", "entry": entries}
        fname = out / f"Patient_{pid}.json"
        with fname.open("w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)

    return out
