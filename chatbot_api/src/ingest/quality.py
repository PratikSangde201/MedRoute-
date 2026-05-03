import os
from difflib import SequenceMatcher
from typing import Any, Dict, List

from neo4j import GraphDatabase


def _norm(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(a=_norm(a), b=_norm(b)).ratio()


def validate_entities(structured: Dict[str, Any]) -> Dict[str, Any]:
    diseases = structured.get("diseases", []) if isinstance(structured, dict) else []
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(diseases, list) or not diseases:
        errors.append("No diseases extracted.")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "disease_count": 0,
            "symptom_count": 0,
            "precaution_count": 0,
        }

    disease_names_seen = set()
    symptom_count = 0
    precaution_count = 0

    for idx, disease in enumerate(diseases):
        if not isinstance(disease, dict):
            errors.append(f"Disease item at index {idx} is not an object.")
            continue

        name = (disease.get("name") or "").strip()
        symptoms = disease.get("symptoms") or []
        precautions = disease.get("precautions") or []

        if not name:
            errors.append(f"Disease at index {idx} has empty name.")
        elif _norm(name) in disease_names_seen:
            warnings.append(f"Duplicate disease in extraction: '{name}'.")
        else:
            disease_names_seen.add(_norm(name))

        if not isinstance(symptoms, list):
            errors.append(f"Symptoms must be a list for disease '{name or idx}'.")
            symptoms = []
        if not isinstance(precautions, list):
            errors.append(f"Precautions must be a list for disease '{name or idx}'.")
            precautions = []

        clean_symptoms = [str(s).strip() for s in symptoms if str(s).strip()]
        clean_precautions = [str(p).strip() for p in precautions if str(p).strip()]

        symptom_count += len(set(_norm(s) for s in clean_symptoms))
        precaution_count += len(set(_norm(p) for p in clean_precautions))

        if not clean_symptoms:
            warnings.append(f"No symptoms extracted for disease '{name or idx}'.")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "disease_count": len(diseases),
        "symptom_count": symptom_count,
        "precaution_count": precaution_count,
    }


def get_existing_graph_terms() -> Dict[str, List[str]]:
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not username or not password:
        return {"diseases": [], "symptoms": [], "precautions": []}

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
            disease_rows = session.run("MATCH (d:Disease) RETURN d.name AS name").data()
            symptom_rows = session.run("MATCH (s:Symptom) RETURN s.name AS name").data()
            precaution_rows = session.run("MATCH (p:Precaution) RETURN coalesce(p.name, p.text) AS name").data()
        return {
            "diseases": [row["name"] for row in disease_rows if row.get("name")],
            "symptoms": [row["name"] for row in symptom_rows if row.get("name")],
            "precautions": [row["name"] for row in precaution_rows if row.get("name")],
        }
    except Exception:
        return {"diseases": [], "symptoms": [], "precautions": []}
    finally:
        driver.close()


def deduplicate_entities(structured: Dict[str, Any], existing_terms: Dict[str, List[str]]) -> Dict[str, Any]:
    diseases = structured.get("diseases", []) if isinstance(structured, dict) else []

    suggestions = []
    for disease in diseases:
        if not isinstance(disease, dict):
            continue
        disease_name = str(disease.get("name") or "").strip()
        if disease_name:
            best_match = _best_match(disease_name, existing_terms.get("diseases", []))
            if best_match and best_match["score"] >= 0.85 and _norm(best_match["value"]) != _norm(disease_name):
                suggestions.append(
                    {
                        "type": "disease",
                        "incoming": disease_name,
                        "existing": best_match["value"],
                        "score": round(best_match["score"], 3),
                    }
                )

        for symptom in disease.get("symptoms", []) or []:
            symptom_name = str(symptom).strip()
            if not symptom_name:
                continue
            best_match = _best_match(symptom_name, existing_terms.get("symptoms", []))
            if best_match and best_match["score"] >= 0.9 and _norm(best_match["value"]) != _norm(symptom_name):
                suggestions.append(
                    {
                        "type": "symptom",
                        "incoming": symptom_name,
                        "existing": best_match["value"],
                        "score": round(best_match["score"], 3),
                    }
                )

        for precaution in disease.get("precautions", []) or []:
            precaution_name = str(precaution).strip()
            if not precaution_name:
                continue
            best_match = _best_match(precaution_name, existing_terms.get("precautions", []))
            if best_match and best_match["score"] >= 0.9 and _norm(best_match["value"]) != _norm(precaution_name):
                suggestions.append(
                    {
                        "type": "precaution",
                        "incoming": precaution_name,
                        "existing": best_match["value"],
                        "score": round(best_match["score"], 3),
                    }
                )

    return {
        "duplicate_suggestions": suggestions,
        "duplicate_count": len(suggestions),
    }


def _best_match(value: str, choices: List[str]) -> Dict[str, Any] | None:
    best_value = ""
    best_score = 0.0
    for choice in choices:
        score = _ratio(value, choice)
        if score > best_score:
            best_score = score
            best_value = choice

    if not best_value:
        return None

    return {"value": best_value, "score": best_score}


def score_confidence(
    classification: Dict[str, Any],
    validation_result: Dict[str, Any],
    dedup_result: Dict[str, Any],
) -> float:
    base = float(classification.get("score", 0.0) or 0.0)

    disease_count = validation_result.get("disease_count", 0)
    symptom_count = validation_result.get("symptom_count", 0)
    precaution_count = validation_result.get("precaution_count", 0)

    structure_bonus = min(0.25, (disease_count * 0.03) + (symptom_count * 0.005) + (precaution_count * 0.01))
    errors_penalty = min(0.4, len(validation_result.get("errors", [])) * 0.08)
    warning_penalty = min(0.2, len(validation_result.get("warnings", [])) * 0.03)
    dedup_penalty = min(0.2, dedup_result.get("duplicate_count", 0) * 0.02)

    score = base + structure_bonus - errors_penalty - warning_penalty - dedup_penalty
    return max(0.0, min(1.0, round(score, 3)))
