"""Formats agent output into minimal FHIR R4 resources.

Not a full FHIR server integration — just enough structured mapping to
demonstrate how a generated clinical note / coding suggestion could be
exported into an interoperable format (e.g. for a downstream EHR write-back
via a real FHIR API).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone


def format_clinical_note_as_fhir(
    note_text: str,
    patient_reference: str = "Patient/example",
    author_reference: str = "Practitioner/example",
) -> dict:
    """Wraps generated note text in a FHIR DocumentReference resource."""
    import base64

    return {
        "resourceType": "DocumentReference",
        "id": str(uuid.uuid4()),
        "status": "current",
        "type": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "11488-4",
                    "display": "Consultation note",
                }
            ]
        },
        "subject": {"reference": patient_reference},
        "author": [{"reference": author_reference}],
        "date": datetime.now(timezone.utc).isoformat(),
        "content": [
            {
                "attachment": {
                    "contentType": "text/plain",
                    "data": base64.b64encode(note_text.encode("utf-8")).decode("ascii"),
                }
            }
        ],
    }


def format_icd10_as_condition(
    code: str,
    description: str,
    patient_reference: str = "Patient/example",
) -> dict:
    """Wraps an ICD-10 suggestion in a FHIR Condition resource."""
    return {
        "resourceType": "Condition",
        "id": str(uuid.uuid4()),
        "clinicalStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                }
            ]
        },
        "code": {
            "coding": [
                {
                    "system": "http://hl7.org/fhir/sid/icd-10-cm",
                    "code": code,
                    "display": description,
                }
            ],
            "text": description,
        },
        "subject": {"reference": patient_reference},
        "recordedDate": datetime.now(timezone.utc).isoformat(),
    }


def format_coding_bundle(codes: list[dict], patient_reference: str = "Patient/example") -> dict:
    """Bundles multiple ICD-10 Condition resources into a FHIR Bundle."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": format_icd10_as_condition(
                    c["code"], c["description"], patient_reference=patient_reference
                )
            }
            for c in codes
        ],
    }
