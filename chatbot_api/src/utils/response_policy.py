from typing import Any
import re


def is_unavailable_answer(text: str) -> bool:
    lowered = (text or "").lower()
    unavailable_markers = [
        "no relevant context found",
        "retrieved context does not contain",
        "provided context does not contain",
        "database does not include",
        "graph retrieval unavailable",
        "rate limit reached",
        "error code: 429",
        "currently unavailable",
        "don't have any documents",
        "donât have any documents",
        "don't have access to the documents",
        "donât have access to the documents",
        "don't have any information",
        "donât have any information",
        "i'm sorry, but i don't have",
        "i’m sorry, but i don’t have",
        "not enough information",
        "no retriever context found",
        "error:",
    ]
    return any(marker in lowered for marker in unavailable_markers)


def build_graph_summary(graph_data: dict[str, Any]) -> str:
    disease_name, symptoms, precautions = _extract_graph_lists(graph_data)
    symptom_preview = ", ".join(symptoms[:6])
    precaution_preview = ", ".join(precautions[:6])

    parts = [f"Summary for {disease_name}:"]
    if symptom_preview:
        parts.append(f"Commonly reported symptoms include {symptom_preview}.")
    if precaution_preview:
        parts.append(f"Helpful precautions include {precaution_preview}.")
    if not symptom_preview and not precaution_preview:
        parts.append("I can see the disease node, but related symptom or precaution links are limited right now.")
    return " ".join(parts)


def _extract_graph_lists(graph_data: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    disease_name = str(graph_data.get("disease_name") or "this condition").strip() or "this condition"
    symptoms = [str(item).strip() for item in list(graph_data.get("symptoms", []) or []) if str(item).strip()]
    precautions = [str(item).strip() for item in list(graph_data.get("precautions", []) or []) if str(item).strip()]

    # Fallback for graph payloads shaped as nodes/edges.
    if (not symptoms and not precautions) and graph_data.get("nodes"):
        for node in graph_data.get("nodes", []):
            label = str(node.get("label") or "")
            name = str(node.get("name") or "").strip()
            if label == "Disease" and name:
                disease_name = name
            elif label == "Symptom" and name:
                symptoms.append(name)
            elif label == "Precaution" and name:
                precautions.append(name)

    clean_symptoms = [item.replace("_", " ") for item in symptoms]
    clean_precautions = [item.replace("_", " ") for item in precautions]
    return disease_name, clean_symptoms, clean_precautions


def _query_topic(query_text: str) -> str | None:
    lowered = (query_text or "").lower()
    if "symptom" in lowered:
        return "symptoms"
    if any(
        token in lowered
        for token in ["precaution", "prevent", "prevention", "treat", "treatment", "manage", "management", "what to do"]
    ):
        return "precautions"
    return None


def _target_from_query(query_text: str) -> str | None:
    text = (query_text or "").strip()
    if not text:
        return None
    patterns = [
        r"\bfor\s+([A-Za-z][A-Za-z\-\s]{2,})",
        r"\bof\s+([A-Za-z][A-Za-z\-\s]{2,})",
        r"\babout\s+([A-Za-z][A-Za-z\-\s]{2,})",
        r"\bhave\s+([A-Za-z][A-Za-z\-\s]{2,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;!?")
            candidate = candidate.split("->", 1)[0].strip(" -.,:;!?")
            candidate = re.split(r"\b(?:what|which|how|when|where|who)\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
            candidate = candidate.strip(" -.,:;!?")
            if candidate:
                return candidate
    return None


def build_graph_answer_for_query(query_text: str, graph_data: dict[str, Any]) -> str:
    disease_name, symptoms, precautions = _extract_graph_lists(graph_data)
    lowered = (query_text or "").lower()
    symptom_intent = "symptom" in lowered
    precaution_intent = any(
        token in lowered
        for token in ["precaution", "prevent", "prevention", "treat", "treatment", "manage", "management", "what to do"]
    )
    topic = _query_topic(query_text)
    target_name = _target_from_query(query_text) or disease_name

    if symptom_intent and precaution_intent:
        lines: list[str] = []
        if symptoms:
            lines.append(f"Symptoms of {target_name}: {', '.join(symptoms)}.")
        else:
            lines.append(f"I do not have enough information here about symptoms for {target_name}.")

        if precautions:
            lines.append(f"Precautions for {target_name}: {', '.join(precautions)}.")
        else:
            lines.append(f"I do not have enough information here about precautions for {target_name}.")

        return "\n".join(lines)

    if topic == "symptoms":
        if symptoms:
            return f"Symptoms of {target_name}: {', '.join(symptoms)}."
        return f"I do not have enough information here about symptoms for {target_name}."

    if topic == "precautions":
        if precautions:
            return f"Precautions for {target_name}: {', '.join(precautions)}."
        return f"I do not have enough information here about precautions for {target_name}."

    return build_graph_summary(graph_data)


def suppress_non_answer_payload(query_response: dict[str, Any]) -> None:
    query_response["sources"] = []
    query_response.pop("debug_context", None)
    query_response.pop("graph_data", None)
    query_response.pop("graph_target", None)
