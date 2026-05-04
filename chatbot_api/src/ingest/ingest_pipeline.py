import asyncio
import os
from typing import Any, Dict
from uuid import uuid4

from fastapi import UploadFile

from ingest.pdf_ingest import (
    extract_text_from_pdf_bytes,
    extract_structured_medical_entities,
    insert_into_neo4j,
)
from ingest.document_chunks import insert_document_chunks
from ingest.quality import (
    validate_entities,
    deduplicate_entities,
    score_confidence,
    get_existing_graph_terms,
)
from models.ingest_job import IngestJob


INGEST_JOBS: Dict[str, IngestJob] = {}
INGEST_LOCK = asyncio.Lock()
INGEST_CONFIDENCE_THRESHOLD = float(os.getenv("INGEST_CONFIDENCE_THRESHOLD", "0.3"))


async def create_job(file: UploadFile) -> IngestJob:
    async with INGEST_LOCK:
        job = IngestJob(
            job_id=str(uuid4()),
            filename=file.filename or "uploaded_file",
            content_type=file.content_type or "application/octet-stream",
        )
        INGEST_JOBS[job.job_id] = job
        return job


async def get_job(job_id: str) -> IngestJob | None:
    async with INGEST_LOCK:
        return INGEST_JOBS.get(job_id)


async def list_jobs() -> list[IngestJob]:
    async with INGEST_LOCK:
        return sorted(INGEST_JOBS.values(), key=lambda job: job.created_at, reverse=True)


async def update_job(job_id: str, **updates: Any) -> IngestJob | None:
    async with INGEST_LOCK:
        job = INGEST_JOBS.get(job_id)
        if job is None:
            return None
        for key, value in updates.items():
            setattr(job, key, value)
        job.touch()
        return job


def _extract_text(file_bytes: bytes, content_type: str, filename: str) -> str:
    lower_name = filename.lower()

    if content_type == "application/pdf" or lower_name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(file_bytes)

    if content_type.startswith("text/") or lower_name.endswith((".txt", ".csv", ".md")):
        return file_bytes.decode("utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Supported: PDF, TXT, CSV, MD")


def _classify_heuristic(text: str) -> Dict[str, Any]:
    """Keyword-based medical classifier — no LLM needed, always reliable."""
    lower = text.lower()
    keywords = [
        "disease", "symptom", "symptoms", "treatment", "precaution", "diagnosis",
        "patient", "medical", "clinical", "infection", "fever", "pain", "health",
        "medicine", "drug", "therapy", "syndrome", "disorder", "condition",
        "doctor", "hospital", "prescription", "dosage", "pathology",
    ]
    hits = sum(1 for kw in keywords if kw in lower)
    score = min(1.0, round(hits / 5.0, 2))
    return {
        "is_medical": hits >= 2,
        "score": score,
        "explanation": f"{hits} medical keywords found in document",
    }


async def run_ingestion(job_id: str, file_bytes: bytes, content_type: str, filename: str) -> None:
    await update_job(job_id, status="processing")

    try:
        text = await asyncio.to_thread(_extract_text, file_bytes, content_type, filename)
        await update_job(job_id, extracted_text=text)

        # Fast keyword classification — no LLM, always reliable
        classification = _classify_heuristic(text)
        await update_job(job_id, classification=classification)

        if not classification.get("is_medical"):
            await update_job(
                job_id,
                status="rejected",
                error_message="Document does not appear to contain medical content.",
            )
            return

        structured = await asyncio.to_thread(extract_structured_medical_entities, text)
        entities_extracted = len(structured.get("diseases", [])) if isinstance(structured, dict) else 0
        validation_result = await asyncio.to_thread(validate_entities, structured)
        existing_terms = await asyncio.to_thread(get_existing_graph_terms)
        dedup_result = await asyncio.to_thread(deduplicate_entities, structured, existing_terms)
        confidence_score = await asyncio.to_thread(
            score_confidence,
            classification,
            validation_result,
            dedup_result,
        )

        # Always go to review_needed — let the user approve or discard.
        # The quality gate was auto-rejecting valid medical docs because
        # phi3:mini returns inconsistent JSON for the extraction prompt.
        await update_job(
            job_id,
            status="review_needed",
            structured=structured,
            entities_extracted=entities_extracted,
            validation_result=validation_result,
            dedup_result=dedup_result,
            confidence_score=confidence_score,
        )
    except Exception as exc:
        await update_job(job_id, status="failed", error_message=str(exc))


def _apply_merge_decisions(
    structured_payload: Dict[str, Any],
    merge_decisions: list[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    if not merge_decisions:
        return structured_payload

    normalized = []
    for disease in structured_payload.get("diseases", []) if isinstance(structured_payload, dict) else []:
        if not isinstance(disease, dict):
            continue
        normalized.append(
            {
                "name": (disease.get("name") or "").strip(),
                "symptoms": [str(item).strip() for item in disease.get("symptoms", []) if str(item).strip()],
                "precautions": [str(item).strip() for item in disease.get("precautions", []) if str(item).strip()],
            }
        )

    for decision in merge_decisions:
        if not isinstance(decision, dict):
            continue
        if str(decision.get("action", "")).lower() != "merge":
            continue

        decision_type = str(decision.get("type", "")).lower()
        incoming = str(decision.get("incoming", "")).strip()
        existing = str(decision.get("existing", "")).strip()
        if not incoming or not existing:
            continue

        if decision_type == "disease":
            for disease in normalized:
                if disease["name"].lower() == incoming.lower():
                    disease["name"] = existing
        elif decision_type == "symptom":
            for disease in normalized:
                disease["symptoms"] = [existing if s.lower() == incoming.lower() else s for s in disease["symptoms"]]
        elif decision_type == "precaution":
            for disease in normalized:
                disease["precautions"] = [existing if p.lower() == incoming.lower() else p for p in disease["precautions"]]

    return {"diseases": normalized}


async def approve_job(
    job_id: str,
    structured_override: Dict[str, Any] | None = None,
    merge_decisions: list[Dict[str, Any]] | None = None,
) -> IngestJob | None:
    job = await get_job(job_id)
    if job is None:
        return None

    structured_payload = structured_override or job.structured

    if not structured_payload:
        await update_job(job_id, status="failed", error_message="No extracted entities available")
        return await get_job(job_id)

    structured_payload = _apply_merge_decisions(structured_payload, merge_decisions)

    await update_job(job_id, status="inserting", merge_decisions=merge_decisions)
    try:
        inserted = await asyncio.to_thread(insert_into_neo4j, structured_payload)
        document_inserted = {"document_chunks": 0, "mentions_links": 0}
        if job.extracted_text:
            document_inserted = await asyncio.to_thread(
                insert_document_chunks,
                job.filename,
                job.extracted_text,
                structured_payload,
            )

        await update_job(
            job_id,
            status="approved",
            structured=structured_payload,
            inserted={**inserted, **document_inserted},
        )
    except Exception as exc:
        await update_job(job_id, status="failed", error_message=str(exc))

    return await get_job(job_id)


async def reject_job(job_id: str, reason: str = "Rejected by user") -> IngestJob | None:
    job = await get_job(job_id)
    if job is None:
        return None

    await update_job(job_id, status="rejected", error_message=reason)
    return await get_job(job_id)
