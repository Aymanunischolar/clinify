from tools.drug_interaction import check_drug_interactions
from tools.fhir_formatter import format_clinical_note_as_fhir, format_icd10_as_condition
from tools.icd10_lookup import search_icd10


def test_icd10_lookup_returns_hypertension_code():
    results = search_icd10("hypertension")
    assert any(r["code"].startswith("I10") for r in results)


def test_drug_interaction_flags_known_pair():
    results = check_drug_interactions(["lisinopril", "spironolactone"])
    assert len(results) > 0


def test_drug_interaction_empty_for_single_drug():
    assert check_drug_interactions(["metformin"]) == []


def test_fhir_note_formatting():
    doc = format_clinical_note_as_fhir("Patient presents with hypertension.")
    assert doc["resourceType"] == "DocumentReference"
    assert doc["content"][0]["attachment"]["contentType"] == "text/plain"


def test_fhir_condition_formatting():
    condition = format_icd10_as_condition("I10", "Essential hypertension")
    assert condition["resourceType"] == "Condition"
    assert condition["code"]["coding"][0]["code"] == "I10"
