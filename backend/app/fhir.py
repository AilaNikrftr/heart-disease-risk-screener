"""Export a stored screening as a FHIR R4 Bundle.

The bundle holds one RiskAssessment (the model's result) plus one Observation
per measurement it was based on, coded with LOINC so an EHR can read them
without knowing anything about this app. Sex is a demographic, not an
observation: it goes on an anonymous Patient resource (no name or
identifiers), which FHIR needs as the subject of the RiskAssessment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

LOINC = "http://loinc.org"
SNOMED = "http://snomed.info/sct"
UCUM = "http://unitsofmeasure.org"
RISK_PROBABILITY = "http://terminology.hl7.org/CodeSystem/risk-probability"

# feature -> (LOINC code, display, unit, UCUM code)
MEASUREMENTS: dict[str, tuple[str, str, str, str]] = {
    "age": ("30525-0", "Age", "a", "a"),
    "bmi": ("39156-5", "Body mass index (BMI) [Ratio]", "kg/m2", "kg/m2"),
    "systolic_bp": ("8480-6", "Systolic blood pressure", "mmHg", "mm[Hg]"),
    "diastolic_bp": ("8462-4", "Diastolic blood pressure", "mmHg", "mm[Hg]"),
    "total_cholesterol": ("2093-3", "Cholesterol [Mass/volume] in Serum or Plasma", "mg/dL", "mg/dL"),
    "hdl_cholesterol": ("2085-9", "Cholesterol in HDL [Mass/volume] in Serum or Plasma", "mg/dL", "mg/dL"),
}


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds")


def _observation(obs_id: str, code: dict, when: str, subject: str, **value: Any) -> dict:
    return {
        "fullUrl": f"urn:uuid:{obs_id}",
        "resource": {
            "resourceType": "Observation",
            "id": obs_id,
            "status": "final",
            "code": code,
            "subject": {"reference": subject},
            "effectiveDateTime": when,
            **value,
        },
    }


def screening_to_bundle(screening_id: int, created_at: datetime, answers: dict, score: float, band: str) -> dict:
    """Build a FHIR R4 collection Bundle for one screening."""
    when = _iso(created_at)
    prefix = f"screening-{screening_id}"
    patient_url = f"urn:uuid:{prefix}-patient"
    patient = {
        "fullUrl": patient_url,
        "resource": {"resourceType": "Patient", "id": f"{prefix}-patient", "gender": answers["sex"]},
    }
    entries: list[dict] = []

    for field, (code, display, unit, ucum) in MEASUREMENTS.items():
        entries.append(
            _observation(
                f"{prefix}-{field.replace('_', '-')}",
                {"coding": [{"system": LOINC, "code": code, "display": display}], "text": display},
                when,
                patient_url,
                valueQuantity={"value": answers[field], "unit": unit, "system": UCUM, "code": ucum},
            )
        )

    smoker = bool(answers["current_smoker"])
    entries.append(
        _observation(
            f"{prefix}-smoking",
            {"coding": [{"system": LOINC, "code": "72166-2", "display": "Tobacco smoking status"}]},
            when,
            patient_url,
            valueCodeableConcept=(
                {"coding": [{"system": SNOMED, "code": "77176002", "display": "Smoker"}], "text": "Current smoker"}
                if smoker
                else {"text": "Not a current smoker"}
            ),
        )
    )
    entries.append(
        _observation(
            f"{prefix}-diabetes",
            {"text": "Ever told by a doctor they have diabetes (self-reported)"},
            when,
            patient_url,
            valueBoolean=bool(answers["diabetes"]),
        )
    )

    risk = {
        "fullUrl": f"urn:uuid:{prefix}-risk",
        "resource": {
            "resourceType": "RiskAssessment",
            "id": f"{prefix}-risk",
            "status": "final",
            "subject": {"reference": patient_url},
            "occurrenceDateTime": when,
            "method": {"text": "Logistic regression trained on NHANES 2015-2016 (adults 20+)"},
            "basis": [{"reference": e["fullUrl"]} for e in entries],
            "prediction": [
                {
                    "outcome": {"text": "Coronary heart disease, angina, or heart attack (self-reported)"},
                    "probabilityDecimal": round(float(score), 3),
                    "qualitativeRisk": {
                        "coding": [{"system": RISK_PROBABILITY, "code": band, "display": band.capitalize()}]
                    },
                }
            ],
            "note": [
                {"text": "The score ranks risk; it is not a calibrated probability."},
                {"text": "Educational screening result, not a diagnosis."},
            ],
        },
    }

    return {
        "resourceType": "Bundle",
        "id": prefix,
        "type": "collection",
        "timestamp": when,
        "entry": [risk, patient, *entries],
    }
