"""Drug-drug interaction check tool.

Wraps the NLM RxNav interaction API (free, public, no-auth) with a small
local fallback table for common, well-known interactions so the tool keeps
working offline / without network access.
"""
from __future__ import annotations

import os

RXNAV_RXCUI_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
RXNAV_INTERACTION_URL = "https://rxnav.nlm.nih.gov/REST/interaction/list.json"

_FALLBACK_INTERACTIONS = {
    frozenset({"lisinopril", "spironolactone"}): (
        "Increased risk of hyperkalemia when combining an ACE inhibitor with a "
        "potassium-sparing diuretic; monitor serum potassium."
    ),
    frozenset({"metformin", "contrast dye"}): (
        "Iodinated contrast media can precipitate lactic acidosis in patients on "
        "metformin, especially with renal impairment; consider holding metformin "
        "around contrast administration."
    ),
    frozenset({"warfarin", "amiodarone"}): (
        "Amiodarone inhibits warfarin metabolism, significantly increasing INR "
        "and bleeding risk; reduce warfarin dose and monitor INR closely."
    ),
    frozenset({"ace inhibitor", "arb"}): (
        "Combining an ACE inhibitor and an ARB increases risk of hyperkalemia "
        "and renal impairment without added cardiovascular benefit; avoid "
        "combination therapy."
    ),
    frozenset({"simvastatin", "clarithromycin"}): (
        "Clarithromycin (a strong CYP3A4 inhibitor) markedly raises simvastatin "
        "levels, increasing risk of myopathy/rhabdomyolysis; avoid combination "
        "or switch statin."
    ),
}


def _fallback_check(drug_names: list[str]) -> list[dict]:
    normalized = {d.strip().lower() for d in drug_names}
    findings = []
    for pair, description in _FALLBACK_INTERACTIONS.items():
        if pair.issubset(normalized):
            findings.append({"drugs": sorted(pair), "description": description, "source": "local_fallback"})
    return findings


def check_drug_interactions(drug_names: list[str]) -> list[dict]:
    """Check a list of drug names for pairwise interactions.

    Returns a list of {"drugs": [...], "description": ..., "source": ...}.
    """
    if len(drug_names) < 2:
        return []

    if os.getenv("USE_LOCAL_DRUG_CHECK") == "1":
        return _fallback_check(drug_names)

    try:
        import httpx

        rxcuis = []
        with httpx.Client(timeout=5.0) as client:
            for name in drug_names:
                r = client.get(RXNAV_RXCUI_URL, params={"name": name})
                r.raise_for_status()
                ids = r.json().get("idGroup", {}).get("rxnormId", [])
                if ids:
                    rxcuis.append(ids[0])
            if len(rxcuis) < 2:
                return _fallback_check(drug_names)

            r = client.get(RXNAV_INTERACTION_URL, params={"rxcuis": "+".join(rxcuis)})
            r.raise_for_status()
            data = r.json()

        findings = []
        for group in data.get("fullInteractionTypeGroup", []):
            for itype in group.get("fullInteractionType", []):
                for pair in itype.get("interactionPair", []):
                    findings.append(
                        {
                            "drugs": [c["minConceptItem"]["name"] for c in itype.get("comment", "").split()][:2]
                            or drug_names,
                            "description": pair.get("description", ""),
                            "source": "rxnav",
                        }
                    )
        return findings or _fallback_check(drug_names)
    except Exception:
        return _fallback_check(drug_names)
