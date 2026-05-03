import asyncio
import hashlib
import os
import re
import socket
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.graphs import Neo4jGraph


def _load_local_env() -> None:
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        return

    # FIX: only rewrite host.docker.internal when NOT running inside a
    # Docker container. Previously this always rewrote to "localhost",
    # which inside a container points to the container itself — not to
    # the host machine where Ollama and Neo4j actually run — causing all
    # LLM calls and Neo4j connections to fail with "connection refused".
    #
    # Detection: Docker sets up /.dockerenv on every container. If that
    # file exists we are inside a container and must keep host.docker.internal
    # so the OS-level DNS resolves it to the host gateway IP. If we are
    # running directly on the host (dev machine, CI) we rewrite to
    # localhost because host.docker.internal is not defined there.
    running_in_docker = os.path.exists("/.dockerenv")

    if not running_in_docker:
        # Running directly on the host — host.docker.internal is not a
        # valid hostname here, so rewrite to localhost.
        for env_key in ("OLLAMA_BASE_URL", "NEO4J_URI"):
            val = os.getenv(env_key, "")
            if (
                val.startswith("http://host.docker.internal")
                or val.startswith("neo4j://host.docker.internal")
            ):
                os.environ[env_key] = val.replace("host.docker.internal", "localhost")
                print(
                    f"[ENV] {env_key}: rewrote host.docker.internal -> localhost "
                    f"(non-Docker environment)"
                )
    else:
        # Running inside a container — host.docker.internal is valid and
        # is the only way to reach services on the host. Do NOT rewrite.
        print("[ENV] Docker container detected — keeping host.docker.internal as-is")

    # Connectivity check: if Neo4j URI is localhost:7687 and it is
    # unreachable, blank it out so the app starts without Neo4j.
    neo4j_uri = os.getenv("NEO4J_URI", "")
    if "localhost:7687" in neo4j_uri:
        try:
            with socket.create_connection(("localhost", 7687), timeout=0.4):
                pass
        except Exception:
            os.environ["NEO4J_URI"] = ""


_load_local_env()

from src.agents.medical_rag_agent import invoke_chatbot_agent, _direct_graph_answer, _contextualize_query
from src.retrieval.router_policy import classify_query, extract_disease_entities, KNOWN_DISEASES_SORTED
from src.retrieval.hybrid_retriever import bm25_retrieve_sync, hybrid_retrieve, hybrid_retrieve_no_routing
from src.llm_caller import (
    call_llm_direct, call_llm_streaming,
    call_llm_knowledge_only, call_llm_with_retry,
)
from src.chains.medical_cypher_chain import chatbot_cypher_chain
from src.services.medical_knowledge_service import load_medical_facts, get_medical_answer
from src.ingest.ingest_pipeline import (
    create_job, run_ingestion, get_job, list_jobs, approve_job, reject_job,
)
from src.models.medical_rag_query import ChatbotQueryInput, ChatbotQueryOutput
from src.utils.graph_utils import get_disease_graph_data
from src.utils.fallback_reasoning import extractSymptoms, inferConditions, buildResponse, mergeSessionSymptoms
from src.utils.response_policy import (
    is_unavailable_answer, build_graph_answer_for_query, suppress_non_answer_payload,
)
from src.ingest.pdf_ingest import (
    extract_text_from_pdf_bytes, classify_medical_text,
    extract_structured_medical_entities, insert_into_neo4j,
)

app = FastAPI(
    title="Medical Chatbot API",
    description="Graph RAG medical chatbot — MedRoute",
)

# CORS — allows Streamlit frontend, direct browser access, and Docker service calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://0.0.0.0:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://chatbot_frontend:8501",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

_INTERNAL_WORDING_REPLACEMENTS = {
    "provided context": "available information",
    "retrieval": "analysis",
    "knowledge base": "available information",
    "structured data": "available information",
    "graph source": "source",
    "fallback source": "source",
    "no retriever context found": "not enough information was found for this question",
}

_ROUTE_TIMEOUT_SECONDS = float(os.getenv("ROUTE_RETRIEVAL_TIMEOUT_SECONDS", "30"))

try:
    load_medical_facts()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_general_llm(query: str) -> str:
    from src.retrieval.hybrid_retriever import _general_medical_answer
    return _general_medical_answer(query)


async def _async_general_llm(query: str) -> str:
    return await asyncio.to_thread(_call_general_llm, query)


async def invoke_chain_with_retry(
    query: str, chat_history: str = "", routing_enabled: bool | None = None
):
    try:
        try:
            result = await asyncio.to_thread(
                invoke_chatbot_agent, query, chat_history, routing_enabled
            )
        except TypeError:
            result = await asyncio.to_thread(invoke_chatbot_agent, query, chat_history)
        return result
    except Exception:
        import traceback
        print(f"[AGENT_ERR] {traceback.format_exc()}")
        try:
            return {
                "input": query,
                "output": await _async_general_llm(query),
                "intermediate_steps": ["Agent failed. Fallback executed."],
                "sources": [],
            }
        except Exception:
            pass
        return {
            "input": query,
            "output": "Please consult a healthcare professional for this question.",
            "intermediate_steps": ["All execution paths failed."],
            "sources": [],
        }


# ---------------------------------------------------------------------------
# Disease detection
# ---------------------------------------------------------------------------

def get_all_disease_names() -> list[str]:
    try:
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE"),
        )
        rows = graph.query("MATCH (d:Disease) RETURN d.name AS name")
        names = [row.get("name") for row in rows if row.get("name")]
        names.sort(key=len, reverse=True)
        return names
    except Exception:
        return []


def find_disease_mentions(text: str) -> list[str]:
    if not text:
        return []
    matches = []
    for disease_name in get_all_disease_names():
        pattern = rf"(?<![A-Za-z0-9]){re.escape(disease_name)}(?![A-Za-z0-9])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(disease_name)
    return matches


def detect_disease_in_text(text: str) -> str | None:
    mentions = find_disease_mentions(text)
    return mentions[0] if len(mentions) == 1 else None


def detect_target_disease(query_text: str, output_text: str) -> str | None:
    query_mentions = find_disease_mentions(query_text)
    if len(query_mentions) == 1:
        return query_mentions[0]
    if len(query_mentions) > 1:
        return None
    heading_match = re.search(r"\*\*(.+?)\*\*", output_text or "")
    if heading_match:
        heading = heading_match.group(1).strip().lower()
        for disease_name in get_all_disease_names():
            if disease_name.lower() == heading:
                return disease_name
    return None


def _sanitize_user_output_text(text: str) -> str:
    if not text:
        return ""
    cleaned = str(text)
    for old, new in _INTERNAL_WORDING_REPLACEMENTS.items():
        cleaned = re.sub(re.escape(old), new, cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = "\n".join(
        re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.split("\n")
    )
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


# ---------------------------------------------------------------------------
# Document retrieval helpers
# ---------------------------------------------------------------------------

def select_best_doc(docs: list, query: str, char_limit: int = 500) -> str:
    if not docs:
        return ""
    q_words = set(re.findall(r"[a-zA-Z]{4,}", query.lower()))
    disease_terms_in_query = {d for d in KNOWN_DISEASES_SORTED if d in query.lower()}

    best_doc, best_score = None, -1.0
    for doc in docs:
        content = str(doc.get("content", ""))
        content_lower = content.lower()
        keyword_hits  = sum(1 for w in q_words if w in content_lower)
        disease_hits  = sum(1 for d in disease_terms_in_query if d in content_lower)
        length_score  = min(1.0, len(content) / 500)
        total         = disease_hits * 6 + keyword_hits * 2 + length_score
        if total > best_score:
            best_score = total
            best_doc   = doc

    return str(best_doc.get("content", ""))[:char_limit] if best_doc else ""


def select_entity_focused_docs(
    docs: list, query: str, entities: list[str], char_limit_each: int = 300
) -> str:
    if not docs:
        return ""

    if len(entities) == 0:
        return select_best_doc(docs, query, char_limit=char_limit_each * 2)

    if len(entities) == 1:
        entity_docs = [d for d in docs if entities[0] in str(d.get("content", "")).lower()]
        best = entity_docs[0] if entity_docs else docs[0]
        return str(best.get("content", ""))[:char_limit_each * 2]

    entity1, entity2 = entities[0], entities[1]
    docs_e1 = [d for d in docs if entity1 in str(d.get("content", "")).lower()]
    docs_e2 = [d for d in docs if entity2 in str(d.get("content", "")).lower()]

    def best_from_pool(pool: list) -> str:
        if not pool:
            return ""
        q_words = set(re.findall(r"[a-zA-Z]{4,}", query.lower()))
        scored  = sorted(
            pool,
            key=lambda d: sum(1 for w in q_words if w in str(d.get("content", "")).lower()),
            reverse=True,
        )
        return str(scored[0].get("content", ""))[:char_limit_each]

    ctx_e1 = best_from_pool(docs_e1)
    ctx_e2 = best_from_pool(docs_e2)

    if ctx_e1 and ctx_e2:
        return (ctx_e1 + " " + ctx_e2).strip()
    if ctx_e1:
        return ctx_e1[:char_limit_each * 2]
    if ctx_e2:
        return ctx_e2[:char_limit_each * 2]
    return select_best_doc(docs, query, char_limit=char_limit_each * 2)


# ---------------------------------------------------------------------------
# DocumentChunk retrieval (from PDF ingestion)
# ---------------------------------------------------------------------------

_chunks_available: bool | None = None  # None=unknown, False=confirmed empty


def _query_document_chunks(query: str, top_k: int = 3) -> list[dict]:
    """Query DocumentChunk nodes in Neo4j using parameterized keyword matching."""
    global _chunks_available
    if _chunks_available is False:
        return []  # Confirmed no chunks — skip Neo4j roundtrip
    neo4j_uri = os.getenv("NEO4J_URI", "")
    if not neo4j_uri:
        return []
    try:
        graph = Neo4jGraph(
            url=neo4j_uri,
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE"),
        )
        disease_terms = extract_disease_entities(query)
        stop_words = {"which", "where", "their", "should", "would", "could", "about", "these", "those", "what", "when", "have", "does", "this", "that", "with", "from"}
        general_kw = [
            w for w in re.findall(r"[a-zA-Z]{5,}", query.lower())
            if w not in stop_words
        ]
        all_keywords = list(dict.fromkeys(disease_terms + general_kw))[:6]
        if not all_keywords:
            return []
        rows = graph.query(
            """
            MATCH (c:DocumentChunk)
            WHERE ANY(kw IN $keywords WHERE toLower(c.content) CONTAINS kw)
            RETURN c.content AS content, c.source AS source, c.filename AS filename
            LIMIT $limit
            """,
            params={"keywords": all_keywords, "limit": top_k},
        )
        results = [
            {
                "content": row.get("content", ""),
                "metadata": {
                    "source": row.get("source") or "document_chunk",
                    "filename": row.get("filename") or "",
                },
            }
            for row in rows
            if row.get("content")
        ]
        if not results and _chunks_available is None:
            _chunks_available = False  # Cache: no chunks ingested yet
        elif results:
            _chunks_available = True
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/")
async def get_status():
    return {"status": "running", "model": os.getenv("LLM_MODEL", "llama3.1:8b")}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": os.getenv("LLM_MODEL", "llama3.1:8b"),
        "fast_model": os.getenv("FAST_LLM_MODEL", os.getenv("LLM_MODEL", "llama3.1:8b")),
    }


@app.post("/chatbot-rag-agent")
async def chatbot_endpoint(request: Request):
    body         = await request.json()
    query        = body.get("query") or body.get("text") or body.get("input", "")
    chat_history = body.get("chat_history") or ""
    routing      = body.get("routing", True)

    # Input validation — reject trivial/empty queries early
    if not query or len(query.strip()) < 3:
        msg = "Please enter a valid medical question."
        return {"answer": msg, "output": msg, "response": msg, "sources": [], "route": "NONE", "routing_enabled": routing}

    # Fix #3: contextualize query using chat history (resolves follow-up questions)
    contextual_query, _history_hint = await asyncio.to_thread(
        _contextualize_query, query, chat_history
    )
    effective_query = contextual_query or query

    docs: list[dict[str, Any]] = []
    answer = ""

    try:
        async with asyncio.timeout(90.0):  # Fix #6: 90s hard timeout

            # --------------------------------------------------------------
            # HYBRID PATH (routing disabled)
            # --------------------------------------------------------------
            if not routing:
                route    = "HYBRID"
                all_docs = bm25_retrieve_sync(effective_query, top_k=3)
                context  = select_best_doc(all_docs, effective_query, char_limit=600)
                if context:
                    docs = [{"content": context, "metadata": {"source": "hybrid_best_doc"}}]
                answer = await call_llm_with_retry(effective_query, context, num_predict=100, route="FACTUAL")

            # --------------------------------------------------------------
            # ADAPTIVE PATH
            # --------------------------------------------------------------
            else:
                route = classify_query(effective_query)

                if route == "FACTUAL":
                    # Neo4j fast-path: instant answer from graph, no LLM needed
                    graph_answer = await asyncio.to_thread(_direct_graph_answer, effective_query)
                    if graph_answer:
                        print(f"[FACTUAL] neo4j_fast_path hit q={effective_query[:60]}", flush=True)
                        sources = [{"content": graph_answer[:400], "metadata": {"source": "neo4j_graph"}}]
                        return {
                            "answer":           graph_answer,
                            "output":           graph_answer,
                            "response":         graph_answer,
                            "sources":          sources,
                            "source_documents": sources,
                            "route":            route,
                            "routing_enabled":  routing,
                        }

                    # DocumentChunks → BM25 → LLM (short output for speed)
                    chunk_docs = await asyncio.to_thread(_query_document_chunks, effective_query, 2)
                    all_docs = bm25_retrieve_sync(effective_query, top_k=3)
                    combined_docs = chunk_docs + all_docs
                    context = select_best_doc(combined_docs, effective_query, char_limit=500)
                    if context:
                        src = "document_chunk" if chunk_docs and any(
                            context[:100] in d.get("content", "") for d in chunk_docs
                        ) else "adaptive_factual"
                        docs = [{"content": context, "metadata": {"source": src}}]
                    answer = await call_llm_with_retry(effective_query, context, num_predict=90, route=route)

                elif route == "RELATIONAL":
                    entities = extract_disease_entities(effective_query)
                    print(f"[RELATIONAL] entities={entities} q={effective_query[:60]}", flush=True)
                    all_docs = bm25_retrieve_sync(effective_query, top_k=5)
                    chunk_docs = await asyncio.to_thread(_query_document_chunks, effective_query, 2)
                    combined_docs = chunk_docs + all_docs
                    context = select_entity_focused_docs(
                        combined_docs, effective_query, entities, char_limit_each=350
                    )
                    if context:
                        docs = [{"content": context, "metadata": {
                            "source": "adaptive_relational_entity_focused",
                            "entities": entities,
                        }}]
                    answer = await call_llm_with_retry(effective_query, context, num_predict=140, route=route)

                elif route == "COMPLEX":
                    entities = extract_disease_entities(effective_query)
                    all_docs = bm25_retrieve_sync(effective_query, top_k=5)
                    chunk_docs = await asyncio.to_thread(_query_document_chunks, effective_query, 2)
                    combined_docs = chunk_docs + all_docs
                    context = select_best_doc(combined_docs, effective_query, char_limit=550)
                    if context:
                        docs = [{"content": context, "metadata": {
                            "source": "adaptive_complex",
                            "entities": entities,
                        }}]
                    answer = await call_llm_with_retry(effective_query, context, num_predict=180, route=route)

                elif route == "GENERAL":
                    all_docs   = bm25_retrieve_sync(effective_query, top_k=3)
                    chunk_docs = await asyncio.to_thread(_query_document_chunks, effective_query, 2)
                    combined_docs = chunk_docs + all_docs
                    context = select_best_doc(combined_docs, effective_query, char_limit=400)
                    if context:
                        docs = [{"content": context, "metadata": {"source": "general_data_source"}}]
                        answer = await call_llm_with_retry(effective_query, context, num_predict=140, route=route)
                    else:
                        docs = []
                        answer = await call_llm_knowledge_only(effective_query, num_predict=140)
                        if not answer or len(answer) < 30:
                            answer = await call_llm_with_retry(effective_query, "", num_predict=140, route=route)

                else:
                    all_docs = bm25_retrieve_sync(effective_query, top_k=3)
                    context  = select_best_doc(all_docs, effective_query, char_limit=500)
                    if context:
                        docs = [{"content": context, "metadata": {"source": "adaptive_default"}}]
                    answer = await call_llm_with_retry(effective_query, context, num_predict=100, route="FACTUAL")

    except asyncio.TimeoutError:
        timeout_msg = (
            "The request took too long to process. "
            "Please try a shorter or more specific question."
        )
        return {
            "answer":           timeout_msg,
            "output":           timeout_msg,
            "response":         timeout_msg,
            "sources":          [],
            "source_documents": [],
            "route":            "TIMEOUT",
            "routing_enabled":  routing,
        }

    sources = [
        {"content": d.get("content", "")[:400], "metadata": d.get("metadata", {})}
        for d in (docs or [])
    ]

    return {
        "answer":          answer,
        "output":          answer,
        "response":        answer,
        "sources":         sources,
        "source_documents":sources,
        "route":           route,
        "routing_enabled": routing,
    }


# ---------------------------------------------------------------------------
# Ingest endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    pdf_bytes      = await file.read()
    text           = await asyncio.to_thread(extract_text_from_pdf_bytes, pdf_bytes)
    classification = await asyncio.to_thread(classify_medical_text, text)
    if not classification.get("is_medical"):
        return {"ingested": False, "reason": "Not classified as medical", "classification": classification}
    structured = await asyncio.to_thread(extract_structured_medical_entities, text)
    try:
        inserted = await asyncio.to_thread(insert_into_neo4j, structured)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j insert failed: {e}")
    return {"ingested": True, "classification": classification,
            "structured_summary": structured.get("diseases", []), "inserted": inserted}


@app.post("/ingest/upload")
async def ingest_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")
    file_bytes = await file.read()
    job        = await create_job(file)
    try:
        persisted_path = await asyncio.to_thread(
            _persist_ingest_upload, job.job_id, job.filename, file_bytes
        )
    except Exception:
        persisted_path = None
    background_tasks.add_task(run_ingestion, job.job_id, file_bytes, job.content_type, job.filename)
    return {"job_id": job.job_id, "status": job.status, "filename": job.filename,
            "content_type": job.content_type, "persisted_path": persisted_path}


@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.model_dump()


@app.get("/ingest/jobs")
async def ingest_jobs():
    jobs = await list_jobs()
    return {"jobs": [job.model_dump() for job in jobs], "count": len(jobs)}


@app.post("/ingest/approve/{job_id}")
async def ingest_approve(job_id: str, payload: Dict[str, Any] | None = Body(default=None)):
    body = payload or {}
    job  = await approve_job(job_id, body.get("structured"), body.get("merge_decisions"))
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.model_dump()


@app.post("/ingest/reject/{job_id}")
async def ingest_reject(job_id: str):
    job = await reject_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.model_dump()


@app.get("/graph/disease/{disease_name}")
async def get_disease_graph(disease_name: str):
    try:
        graph_data = await asyncio.to_thread(get_disease_graph_data, disease_name)
        if not graph_data.get("disease_found"):
            raise HTTPException(
                status_code=404,
                detail=graph_data.get("message", f"Disease '{disease_name}' not found"),
            )
        return graph_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph error: {str(e)}")


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _persist_ingest_upload(job_id: str, filename: str, payload: bytes) -> str:
    upload_dir = os.getenv("INGEST_UPLOAD_DIR", "/app/data/uploads")
    safe_name  = os.path.basename(filename)
    out_dir    = os.path.join(upload_dir, job_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path   = os.path.join(out_dir, safe_name)
    with open(out_path, "wb") as handle:
        handle.write(payload)
    return out_path


def _session_key_from_chat_history(chat_history: str | None) -> str:
    if not chat_history:
        return "session-default"
    digest = hashlib.sha1(chat_history.encode("utf-8", errors="ignore")).hexdigest()
    return f"session-{digest[:16]}"