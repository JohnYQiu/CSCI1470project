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
from dataclasses import dataclass
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


def _parse_dt(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    v = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return None


def _patient_id_from_ref(ref: str | None) -> str | None:
    if not ref:
        return None
    # "Patient/abc" or "urn:uuid:abc"
    if "/" in ref:
        return ref.split("/")[-1]
    return ref


def _iter_resources_from_path(path: Path) -> Iterator[dict[str, Any]]:
    """Yield FHIR resource dicts from a file or directory."""
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
                obj = json.loads(line)
                if obj.get("resourceType") == "Bundle":
                    for e in obj.get("entry", []) or []:
                        r = e.get("resource")
                        if r:
                            yield r
                else:
                    yield obj


def _iter_resources_from_file(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("resourceType") == "Bundle":
        for entry in data.get("entry", []) or []:
            res = entry.get("resource")
            if res:
                yield res
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("resourceType"):
                yield item
    elif isinstance(data, dict) and data.get("resourceType"):
        yield data


def _first_coding_list(concept: dict | None) -> list[dict]:
    if not concept:
        return []
    out: list[dict] = []
    for cc in concept.get("coding", []) or []:
        if isinstance(cc, dict):
            out.append(cc)
    return out


def _concept_text(concept: dict | None) -> str:
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
    eff = obs.get("effectiveDateTime")
    if eff:
        return _parse_dt(eff)
    pr = obs.get("effectivePeriod") or {}
    return _parse_dt(pr.get("start"))


def _obs_loinc_code(obs: dict) -> str | None:
    for c in obs.get("code", {}).get("coding", []) or []:
        if c.get("system") in (
            "http://loinc.org",
            "http://loinc.org/",
        ):
            return c.get("code")
    return None


def _obs_numeric_value(obs: dict) -> float | None:
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
    bd = _parse_dt(birth_date)
    es = _parse_dt(enc_start)
    if not bd or not es:
        return None
    days = (es - bd).days
    if days < 0:
        return None
    return days / 365.25


def _is_ems_like_encounter(enc: dict) -> bool:
    """Heuristic: treat emergency encounters as EMS-relevant rows (Synthea proxy)."""
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
    Return 1 transport, 0 refusal, or None if unknown.

    Priority: custom extension → dischargeDisposition semantics.
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


def parse_fhir_to_dataframe(
    fhir_path: str | Path,
    *,
    drop_unlabeled: bool = True,
    only_ems_like: bool = True,
) -> pd.DataFrame:
    """
    Load FHIR JSON from a file or directory and build one row per qualifying Encounter.

    Vitals: most recent Observation per LOINC within encounter period (inclusive of
    observations up to 24h before start to capture vitals taken on scene).
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
            "heart_rate": [],
            "systolic_bp": [],
            "diastolic_bp": [],
            "respiratory_rate": [],
            "spo2": [],
            "temperature": [],
        }
        loinc_map = {
            LOINC_HEART_RATE: "heart_rate",
            LOINC_SYSTOLIC: "systolic_bp",
            LOINC_DIASTOLIC: "diastolic_bp",
            LOINC_RESP_RATE: "respiratory_rate",
            LOINC_SPO2: "spo2",
            LOINC_TEMP_F: "temperature",
        }

        for obs in obs_by_patient.get(pid, []):
            ot = _obs_effective_dt(obs)
            if ot is None or ot < window_lo or ot > window_hi:
                continue
            code = _obs_loinc_code(obs)
            if not code or code not in loinc_map:
                continue
            val = _obs_numeric_value(obs)
            if val is None:
                continue
            col = loinc_map[code]
            vitals_lists[col].append((ot, val))

        def latest(candidates: list[tuple[datetime, float]]) -> float | None:
            if not candidates:
                return None
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]

        age = _age_at_encounter(pat.get("birthDate"), start_s)
        sex = (pat.get("gender") or "unknown").lower()
        cc = _chief_complaint(enc, conditions, pid)

        row = {
            "encounter_id": enc.get("id"),
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "heart_rate": latest(vitals_lists["heart_rate"]),
            "systolic_bp": latest(vitals_lists["systolic_bp"]),
            "diastolic_bp": latest(vitals_lists["diastolic_bp"]),
            "respiratory_rate": latest(vitals_lists["respiratory_rate"]),
            "spo2": latest(vitals_lists["spo2"]),
            "temperature": latest(vitals_lists["temperature"]),
            "chief_complaint": cc,
            "transport": int(label) if label is not None else None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def write_mock_fhir_directory(output_dir: str | Path, *, seed: int = 42, n_patients: int = 80) -> Path:
    """
    Write Synthea-like patient Bundle JSON files with EMS-labeled emergency encounters.

    Labels are set via Encounter.extension (see EMS_TRANSPORT_EXT_URL) so the
    pipeline has ground truth without relying on dischargeDisposition.
    """
    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for _ in range(n_patients):
        pid = str(uuid.uuid4())
        birth_year = rng.randint(1940, 2005)
        birth = f"{birth_year}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
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
            enc_start = datetime(2018 + rng.randint(0, 5), rng.randint(1, 12), rng.randint(1, 28), rng.randint(8, 20), 0, 0)
            duration_h = rng.uniform(0.5, 4.0)
            enc_end = enc_start + timedelta(hours=duration_h)
            # Synthetic label (ground truth in extension); vitals loosely correlate for ML demo
            p_transport = 0.55
            if rng.random() < 0.3:
                p_transport += 0.15
            transport = rng.random() < p_transport
            if not transport and rng.random() < 0.2:
                transport = True

            ext_val = bool(transport)
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
                "reasonCode": [
                    {
                        "text": rng.choice(
                            [
                                "Chest pain",
                                "Shortness of breath",
                                "Fall injury",
                                "Abdominal pain",
                                "Syncope",
                                "General weakness",
                            ]
                        )
                    }
                ],
                "extension": [
                    {
                        "url": EMS_TRANSPORT_EXT_URL,
                        "valueBoolean": ext_val,
                    }
                ],
            }
            entries.append({"resource": enc})

            def add_obs(loinc: str, base: float, noise: float, minutes_after: int):
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

            hr_base = 88 if transport else 72
            spo2_base = 93 if transport else 98
            add_obs(LOINC_HEART_RATE, hr_base, 12, rng.randint(5, 40))
            add_obs(LOINC_SYSTOLIC, 130 if transport else 118, 18, rng.randint(5, 45))
            add_obs(LOINC_DIASTOLIC, 82 if transport else 76, 10, rng.randint(5, 45))
            add_obs(LOINC_RESP_RATE, 22 if transport else 16, 4, rng.randint(5, 50))
            add_obs(LOINC_SPO2, spo2_base, 3, rng.randint(5, 55))
            add_obs(LOINC_TEMP_F, 99.2 if transport else 98.4, 0.8, rng.randint(5, 60))

            # Duplicate vital earlier (parser should pick most recent)
            add_obs(LOINC_HEART_RATE, hr_base - 5, 3, 2)

        bundle = {"resourceType": "Bundle", "type": "collection", "entry": entries}
        fname = out / f"Patient_{pid}.json"
        with fname.open("w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)

    return out
