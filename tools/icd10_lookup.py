"""ICD-10 lookup tool.

Wraps the NLM Clinical Table Search Service (a free, public, no-auth ICD-10
lookup API used widely for demos/prototyping) with a small local fallback
table covering the conditions in our sample knowledge base, so the tool
still works offline / without network access.
"""
from __future__ import annotations

import os

NLM_ICD10_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"

_FALLBACK_TABLE = {
    "hypertension": [("I10", "Essential (primary) hypertension")],
    "high blood pressure": [("I10", "Essential (primary) hypertension")],
    "type 2 diabetes": [("E11.9", "Type 2 diabetes mellitus without complications")],
    "diabetes mellitus": [("E11.9", "Type 2 diabetes mellitus without complications")],
    "pneumonia": [("J18.9", "Pneumonia, unspecified organism")],
    "community-acquired pneumonia": [("J18.9", "Pneumonia, unspecified organism")],
    "hyperlipidemia": [("E78.5", "Hyperlipidemia, unspecified")],
    "chronic kidney disease": [("N18.9", "Chronic kidney disease, unspecified")],
    "hypoglycemia": [("E16.2", "Hypoglycemia, unspecified")],
    "hypertensive crisis": [("I16.9", "Hypertensive crisis, unspecified")],
}


def _fallback_search(term: str, max_results: int) -> list[dict]:
    term_l = term.lower()
    results: list[dict] = []
    for key, codes in _FALLBACK_TABLE.items():
        if key in term_l or term_l in key:
            for code, desc in codes:
                results.append({"code": code, "description": desc})
    return results[:max_results]


def search_icd10(term: str, max_results: int = 5) -> list[dict]:
    """Search ICD-10-CM codes matching a clinical term.

    Returns a list of {"code": ..., "description": ...} dicts.
    """
    if os.getenv("USE_LOCAL_ICD10") == "1":
        return _fallback_search(term, max_results)

    try:
        import httpx

        resp = httpx.get(
            NLM_ICD10_URL,
            params={"sf": "code,name", "terms": term, "maxList": max_results},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # NLM response shape: [total, [codes], null, [[code, name], ...]]
        codes_and_names = data[3] if len(data) > 3 else []
        return [{"code": c, "description": n} for c, n in codes_and_names]
    except Exception:
        return _fallback_search(term, max_results)
