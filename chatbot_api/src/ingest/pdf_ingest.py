import io
import json
import os
from typing import Dict, List, Any

import httpx
from pypdf import PdfReader
from neo4j import GraphDatabase


def _ollama_url() -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").strip().rstrip("/")
    return f"{base}/api/generate"


def _ingest_model() -> str:
    return (
        os.getenv("CHATBOT_QA_MODEL")
        or os.getenv("CHATBOT_CYPHER_MODEL")
        or os.getenv("LLM_MODEL")
        or "phi3:mini"
    )


def _call_ollama_sync(prompt: str, num_predict: int = 256) -> str:
    payload = {
        "model": _ingest_model(),
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0},
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(_ollama_url(), json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
    except Exception as exc:
        raise RuntimeError(f"Ollama call failed: {exc}") from exc


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
    prompt = (
        "You are a classifier. Given the text of a document, reply with a JSON object "
        "on a single line with the shape: {\"is_medical\": true|false, \"explanation\": \"...\", \"score\": 0.0} . "
        "Return score as a number between 0 and 1 indicating how confident you are that the document "
        "contains medical content about diseases, symptoms, precautions, or clinical guidance.\n\n"
        "Document text:\n" + text[:4000]
    )
    raw = _call_ollama_sync(prompt, num_predict=200)

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
    prompt = (
        "Extract all mentions of diseases, symptoms, and precautions from the document text "
        "and return a single-line JSON object with the following format:\n"
        "{\"diseases\": [{\"name\": <string>, \"symptoms\": [<string>], \"precautions\": [<string>] }, ...]}\n\n"
        "Only include items you are confident are explicitly present in the text. Do not hallucinate. "
        "If a disease is named but has no explicit symptoms/precautions in the text, return empty arrays for them.\n\n"
        "Document text:\n" + text[:6000]
    )
    raw = _call_ollama_sync(prompt, num_predict=512)

    def _parse_diseases(text_blob: str) -> List[Dict]:
        parsed = json.loads(text_blob.strip())
        diseases = parsed.get("diseases", [])
        normalized = []
        for d in diseases:
            name = d.get("name") if isinstance(d, dict) else str(d)
            symptoms = d.get("symptoms", []) if isinstance(d, dict) else []
            precautions = d.get("precautions", []) if isinstance(d, dict) else []
            if name:
                normalized.append({"name": name, "symptoms": symptoms, "precautions": precautions})
        return normalized

    # 1st attempt: parse the raw response directly
    try:
        return {"diseases": _parse_diseases(raw), "raw": raw}
    except Exception:
        pass

    # 2nd attempt: model wrapped JSON in markdown fences (```json ... ```)
    import re as _re
    fence_match = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, _re.DOTALL)
    if fence_match:
        try:
            return {"diseases": _parse_diseases(fence_match.group(1)), "raw": raw}
        except Exception:
            pass

    # 3rd attempt: find the first { ... } block in the response
    brace_match = _re.search(r"\{.*\}", raw, _re.DOTALL)
    if brace_match:
        try:
            return {"diseases": _parse_diseases(brace_match.group(0)), "raw": raw}
        except Exception:
            pass

    return {"diseases": [], "raw": raw}


def list_ingested_diseases() -> List[Dict[str, Any]]:
    """Return all Disease nodes with their symptom and precaution counts."""
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not username or not password:
        return []
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (d:Disease)
                OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
                OPTIONAL MATCH (d)-[:HAS_PRECAUTION]->(p:Precaution)
                RETURN d.name AS name,
                       count(DISTINCT s) AS symptoms,
                       count(DISTINCT p) AS precautions
                ORDER BY d.name
                """
            ).data()
        return [{"name": r["name"], "symptoms": r["symptoms"], "precautions": r["precautions"]} for r in rows]
    except Exception:
        return []
    finally:
        driver.close()


def delete_disease_from_neo4j(disease_name: str) -> Dict[str, Any]:
    """Delete a disease node, its relationships, and any orphaned symptom/precaution nodes."""
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not username or not password:
        raise RuntimeError("NEO4J connection environment variables not set")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session() as session:
            exists = session.run(
                "MATCH (d:Disease {name: $name}) RETURN count(d) AS cnt", name=disease_name
            ).single()
            if not exists or exists["cnt"] == 0:
                return {"found": False}

            # Delete the disease and all its relationships
            session.run("MATCH (d:Disease {name: $name}) DETACH DELETE d", name=disease_name)

            # Remove symptoms no longer linked to any disease
            s_res = session.run(
                "MATCH (s:Symptom) WHERE NOT ()-[:HAS_SYMPTOM]->(s) "
                "DELETE s RETURN count(s) AS cnt"
            ).single()

            # Remove precautions no longer linked to any disease
            p_res = session.run(
                "MATCH (p:Precaution) WHERE NOT ()-[:HAS_PRECAUTION]->(p) "
                "DELETE p RETURN count(p) AS cnt"
            ).single()

        return {
            "found": True,
            "disease_deleted": True,
            "orphaned_symptoms_removed": s_res["cnt"] if s_res else 0,
            "orphaned_precautions_removed": p_res["cnt"] if p_res else 0,
        }
    finally:
        driver.close()


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
