import io
import json
import os
from typing import Dict, List, Any

from pypdf import PdfReader
from neo4j import GraphDatabase


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts: List[str] = []
    for page in reader.pages:
        try:
            text_parts.append(page.extract_text() or "")
        except Exception:
            # best-effort: skip problematic pages
            continue
    return "\n".join(text_parts)


def classify_medical_text(text: str) -> Dict[str, Any]:
    """Ask the LLM whether the document is medical and relevant.

    Returns a dict with keys: is_medical (bool), explanation (str), score (float 0..1)
    The function expects model names via CHATBOT_QA_MODEL/CHATBOT_CYPHER_MODEL
    and uses the configured provider wrapper. We use a small
    deterministic prompt (temperature=0) and ask for strict JSON output.
    """
    model = (
        os.getenv("CHATBOT_QA_MODEL")
        or os.getenv("CHATBOT_CYPHER_MODEL")
    )
    if not model:
        # fallback model name; the project should set env vars
        model = "llama3.1:8b"

    llm = None

    prompt = (
        "You are a classifier. Given the text of a document, reply with a JSON object "
        "on a single line with the shape: {\"is_medical\": true|false, \"explanation\": \"...\", \"score\": 0.0} . "
        "Return score as a number between 0 and 1 indicating how confident you are that the document "
        "contains medical content about diseases, symptoms, precautions, or clinical guidance.\n\n"
        "Document text:\n" + text[:4000]
    )

    resp = llm.call_as_llm(prompt) if hasattr(llm, "call_as_llm") else llm(prompt)
    # The wrapper can differ; try to get text
    raw = ""
    if isinstance(resp, str):
        raw = resp
    else:
        try:
            # some wrappers return object with .content or .text
            raw = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
        except Exception:
            raw = str(resp)

    # Try to extract JSON from the model response.
    try:
        # Model expected to return pure JSON on a single line
        parsed = json.loads(raw.strip())
        return {
            "is_medical": bool(parsed.get("is_medical")),
            "explanation": parsed.get("explanation", ""),
            "score": float(parsed.get("score", 0.0)),
            "raw": raw,
        }
    except Exception:
        # fallback: simple heuristic
        lower = raw.lower()
        is_med = "disease" in lower or "symptom" in lower or "precaution" in lower or "treatment" in lower
        return {
            "is_medical": is_med,
            "explanation": "Could not parse LLM JSON; used heuristic.",
            "score": 0.5 if is_med else 0.0,
            "raw": raw,
        }


def extract_structured_medical_entities(text: str) -> Dict[str, Any]:
    """Ask the LLM to extract diseases, symptoms and precautions from text.

    Expected return format (JSON): {"diseases": [{"name":"...","symptoms":["..."],"precautions":["..."]}, ...]}
    """
    model = (
        os.getenv("CHATBOT_QA_MODEL")
        or os.getenv("CHATBOT_CYPHER_MODEL")
    )
    if not model:
        model = "llama3.1:8b"

    llm = None

    prompt = (
        "Extract all mentions of diseases, symptoms, and precautions from the document text "
        "and return a single-line JSON object with the following format:\n"
        "{\"diseases\": [{\"name\": <string>, \"symptoms\": [<string>], \"precautions\": [<string>] }, ...]}\n\n"
        "Only include items you are confident are explicitly present in the text. Do not hallucinate. "
        "If a disease is named but has no explicit symptoms/precautions in the text, return empty arrays for them.\n\n"
        "Document text:\n" + text[:6000]
    )

    resp = llm.call_as_llm(prompt) if hasattr(llm, "call_as_llm") else llm(prompt)
    raw = ""
    if isinstance(resp, str):
        raw = resp
    else:
        raw = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)

    try:
        parsed = json.loads(raw.strip())
        # basic normalization
        diseases = parsed.get("diseases", [])
        normalized = []
        for d in diseases:
            name = d.get("name") if isinstance(d, dict) else str(d)
            symptoms = d.get("symptoms", []) if isinstance(d, dict) else []
            precautions = d.get("precautions", []) if isinstance(d, dict) else []
            normalized.append({"name": name, "symptoms": symptoms, "precautions": precautions})
        return {"diseases": normalized, "raw": raw}
    except Exception:
        # If parsing fails, return empty structure with raw for debugging
        return {"diseases": [], "raw": raw}


def insert_into_neo4j(structured: Dict[str, Any]) -> Dict[str, int]:
    """Insert extracted structured data into Neo4j.

    Creates Disease, Symptom, Precaution nodes and relationships between them.
    Uses MERGE so repeated inserts are idempotent.
    Returns counts of created/merged nodes/relationships (best-effort).
    """
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not username or not password:
        raise RuntimeError("NEO4J connection environment variables not set")

    driver = GraphDatabase.driver(uri, auth=(username, password))

    created = {"diseases": 0, "symptoms": 0, "precautions": 0, "relationships": 0}

    def _write(tx, disease_name: str, symptoms: List[str], precautions: List[str]):
        # Merge disease
        tx.run(
            "MERGE (d:Disease {name:$dname}) RETURN id(d)",
            dname=disease_name,
        )
        for s in symptoms:
            tx.run(
                "MERGE (s:Symptom {name:$sname}) RETURN id(s)",
                sname=s,
            )
            tx.run(
                "MATCH (d:Disease {name:$dname}), (s:Symptom {name:$sname}) MERGE (d)-[:HAS_SYMPTOM]->(s)",
                dname=disease_name,
                sname=s,
            )
        for p in precautions:
            tx.run(
                "MERGE (p:Precaution {name:$ptext}) RETURN id(p)",
                ptext=p,
            )
            tx.run(
                "MATCH (d:Disease {name:$dname}), (p:Precaution {name:$ptext}) MERGE (d)-[:HAS_PRECAUTION]->(p)",
                dname=disease_name,
                ptext=p,
            )

    with driver.session() as session:
        diseases = structured.get("diseases", [])
        for d in diseases:
            name = d.get("name")
            if not name:
                continue
            symptoms = d.get("symptoms", []) or []
            precautions = d.get("precautions", []) or []
            session.execute_write(_write, name, symptoms, precautions)
            created["diseases"] += 1
            created["symptoms"] += len(set(symptoms))
            created["precautions"] += len(set(precautions))
            created["relationships"] += len(symptoms) + len(precautions)

    driver.close()
    return created
