"""Chatbot RAG agent — orchestrates tool routing and answer synthesis.

Tool priority:
  1. GeneralMedical — direct LLM for general medical knowledge questions
  2. Graph — Neo4j disease-symptom-precaution relationships
  3. Documents — ingested PDFs/TXT chunks via vector/BM25 retrieval
  4. Experiences — patient review semantic search
"""

import os
import httpx
import re
import json
import asyncio
import logging
from typing import Any
from functools import lru_cache
# from src.llm_openai_provider import ChatOpenAI
from langchain.agents import create_react_agent
from langchain.agents import Tool, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from src.chains.medical_cypher_chain import chatbot_cypher_chain
from src.retrieval.hybrid_retriever import (
    hybrid_retrieve,
    hybrid_retrieve_no_routing,
    _general_medical_answer,
    _is_empty_or_unavailable,
    bm25_retrieve_sync,
)
from src.retrieval.router_policy import route_query

try:
    from src.chains.medical_review_chain import reviews_vector_chain
except Exception:
    reviews_vector_chain = None

CHATBOT_AGENT_MODEL = os.getenv("CHATBOT_AGENT_MODEL")
USE_REACT_AGENT = False
AGENT_DEBUG_STEPS = os.getenv("AGENT_DEBUG_STEPS", "false").lower() == "true"
ENABLE_ADAPTIVE_HYBRID = os.getenv("ENABLE_ADAPTIVE_HYBRID", "true").lower() == "true"
ENABLE_HYBRID_DEBUG_CONTEXT = os.getenv("ENABLE_HYBRID_DEBUG_CONTEXT", "false").lower() == "true"


def call_llm_direct(query: str, context: str) -> str:
    model = os.getenv("LLM_MODEL", "llama3.1:8b")
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    if context and len(context.strip()) > 30:
        prompt = f"""You are a precise medical information assistant. Use the context and your medical knowledge to answer accurately. Include exact medical terms, drug names, and symptom names.

Context:
{context}

Question: {query}

Answer in 3-4 sentences using specific medical terminology:"""
    else:
        prompt = f"""You are a medical information assistant with comprehensive medical knowledge. Answer this medical question accurately and specifically using your knowledge.

Question: {query}

Provide a direct, specific medical answer in 3-4 sentences with exact medical terms, drug names, and symptom names:"""

    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 250, "temperature": 0.1}
            },
            timeout=60.0
        )
        result = r.json().get("response", "").strip()
        if result and len(result) > 20:
            return result
        fallback = call_llm_knowledge_only(query, model)
        if fallback and len(fallback) > 20:
            return fallback
        if context and len(context.strip()) > 30:
            context_snippet = " ".join(context.split())[:420]
            return (
                f"Based on retrieved clinical context for '{query}': {context_snippet}. "
                "Please consult a licensed clinician for patient-specific treatment."
            )
        return f"Medical information for query: {query}."
    except Exception as e:
        print(f"LLM_ERROR: {e}")
        return call_llm_knowledge_only(query, model)


def call_llm_knowledge_only(query: str, model: str) -> str:
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": f"You are a helpful, knowledgeable medical assistant. Answer this question clearly and accurately: {query}",
                "stream": False,
                "options": {"num_predict": 250, "temperature": 0.1}
            },
            timeout=60.0
        )
        resp = r.json().get("response", "").strip()
        return resp if len(resp) > 20 else ""
    except Exception as e:
        print(f"LLM_KNOWLEDGE_ERROR: {e}")
        return ""

# ---------------------------------------------------------------------------
# Agent prompt
# ---------------------------------------------------------------------------

chatbot_agent_prompt_template = """You are MedRoute — a knowledgeable, empathetic medical information assistant.

You have access to tools that retrieve medical information from a knowledge graph, uploaded documents, and patient experience records. You also have broad general medical knowledge.

TOOLS:
------
{tools}

TOOL USAGE FORMAT:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

FINAL ANSWER FORMAT:
```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

ANSWERING RULES:
- For questions about disease symptoms, precautions, or graph-specific facts: use Graph tool.
- For questions about uploaded documents: use Documents tool.
- For general medical knowledge (how things work, mechanisms, diet, physiology, clinical facts): use GeneralMedical tool.
- For patient experiences or qualitative feedback: use Experiences tool.
- Always give complete, structured answers — not one-liners.
- Use bullet points for lists, prose for explanations.
- Include a safety disclaimer for clinical questions.
- Do NOT mention tool names, retrieval systems, or databases in your final answer.

Previous conversation:
{chat_history}

New question: {input}
{agent_scratchpad}
"""

chatbot_agent_prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad", "chat_history"],
    template=chatbot_agent_prompt_template,
)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _invoke_reviews_tool(input_text: str) -> str:
    return call_llm_direct(input_text, "")


def _invoke_graph_tool(input_text: str) -> str:
    return call_llm_direct(input_text, "")


def _invoke_documents_tool(input_text: str) -> str:
    answer, references = _invoke_documents_tool_with_sources(input_text)
    if references:
        source_labels = [ref.get("label", "") for ref in references if isinstance(ref, dict)]
        source_labels = [label for label in source_labels if label]
        if source_labels:
            return f"{answer}\n\nSources: {', '.join(source_labels)}"
    return answer


def _invoke_documents_tool_with_sources(input_text: str) -> tuple[str, list[dict[str, str]]]:
    answer = call_llm_direct(input_text, "")
    return (answer, [])


def _invoke_general_medical_tool(input_text: str) -> str:
    """Direct LLM synthesis for general medical knowledge questions."""
    return call_llm_direct(input_text, "")


# ---------------------------------------------------------------------------
# Neo4j graph helpers
# ---------------------------------------------------------------------------

def _graph_client() -> Neo4jGraph:
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    )


_clinical_disease_names_cache: list[str] | None = None


def _get_clinical_disease_names() -> list[str]:
    global _clinical_disease_names_cache
    if _clinical_disease_names_cache is not None:
        return _clinical_disease_names_cache
    try:
        rows = _graph_client().query(
            """
            MATCH (d:Disease)
            RETURN DISTINCT d.name AS name
            ORDER BY size(name) DESC
            """
        )
        names = [row.get("name") for row in rows if row.get("name")]
        if names:
            _clinical_disease_names_cache = names
        return names
    except Exception:
        return []


def _find_disease_name_in_query(input_text: str) -> str | None:
    lowered = input_text.lower()
    for disease_name in _get_clinical_disease_names():
        pattern = rf"(?<![A-Za-z0-9]){re.escape(disease_name.lower())}(?![A-Za-z0-9])"
        if re.search(pattern, lowered):
            return disease_name
    return None


def _resolve_disease_name(candidate: str | None) -> str | None:
    if not candidate:
        return None
    normalized = re.sub(r"[^A-Za-z0-9\s]", " ", candidate).strip().lower()
    normalized = " ".join(normalized.split())
    if not normalized:
        return None
    for disease_name in _get_clinical_disease_names():
        disease_norm = re.sub(r"[^A-Za-z0-9\s]", " ", disease_name).strip().lower()
        disease_norm = " ".join(disease_norm.split())
        if normalized == disease_norm:
            return disease_name
    return None


def _friendly_list(items: list[str], max_items: int | None = None) -> str:
    clean = sorted(dict.fromkeys(item.strip().replace("_", " ") for item in items if item and item.strip()))
    if not clean:
        return ""
    if max_items is None:
        return ", ".join(clean)
    return ", ".join(clean[:max_items])


def _bullet_list(items: list[str], max_items: int = 15) -> str:
    clean = sorted(dict.fromkeys(item.strip().replace("_", " ").capitalize() for item in items if item and item.strip()))
    return "\n".join(f"- {item}" for item in clean[:max_items])


def _fetch_disease_profile(graph: Neo4jGraph, disease_name: str) -> dict[str, list[str]]:
    rows = graph.query(
        """
        MATCH (d:Disease {name: $disease_name})
        OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
        OPTIONAL MATCH (d)-[pr]->(p:Precaution)
        RETURN collect(DISTINCT s.name) AS symptoms,
               [item IN collect(DISTINCT coalesce(p.name, p.text))
                WHERE item IS NOT NULL AND trim(toString(item)) <> ""] AS precautions
        """,
        params={"disease_name": disease_name},
    )
    if not rows:
        return {"symptoms": [], "precautions": []}
    row = rows[0]
    return {
        "symptoms": [item for item in row.get("symptoms", []) if item],
        "precautions": [item for item in row.get("precautions", []) if item],
    }


def _build_disease_overview(graph: Neo4jGraph, disease_name: str) -> str | None:
    profile = _fetch_disease_profile(graph, disease_name)
    symptoms = profile.get("symptoms", [])
    precautions = profile.get("precautions", [])
    if not symptoms and not precautions:
        return None

    lines = [f"**{disease_name}**"]
    if symptoms:
        lines.append(f"Common symptoms include: {_friendly_list(symptoms)}.")
    else:
        lines.append("Symptom details are not available in the current knowledge graph.")
    if precautions:
        lines.append(f"Recommended precautions include: {_friendly_list(precautions)}.")
    else:
        lines.append("Precaution guidance is not available in the current knowledge graph.")
    return "\n".join(lines)


def _extract_symptom_hint(input_text: str) -> str | None:
    patterns = [
        r"have\s+(.+?)\s+as\s+a\s+symptom",
        r"with\s+(.+?)\s+as\s+a\s+symptom",
        r"symptom(?:s)?\s+(?:like|of|contains?|containing)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, input_text, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" ?!.,;:")
            if candidate:
                return candidate
    simple = re.search(r"\bfever\b", input_text, flags=re.IGNORECASE)
    if simple:
        return "fever"
    return None


def _direct_graph_answer(input_text: str) -> str | None:
    """Fast-path deterministic answers from Neo4j for disease queries."""
    lowered = input_text.lower()
    try:
        graph = _graph_client()
    except Exception:
        return None

    disease_name = _find_disease_name_in_query(input_text)

    symptom_intent = "symptom" in lowered
    # "treat/treatment" are intentionally excluded — the graph has no drug/dosage data,
    # so those queries fall through to BM25+LLM which has richer treatment context.
    precaution_intent = any(
        token in lowered
        for token in ["precaution", "prevention", "prevent", "management", "manage", "what to do"]
    )
    treatment_only = any(
        t in lowered for t in ["treatment", " treat ", "treat?", "therapy", "medication", "drug", "medicine", "cure"]
    ) and not (symptom_intent or precaution_intent)
    if treatment_only:
        return None  # Let BM25+LLM handle — graph has no treatment data

    # Compound queries mix a definition/overview request with clinical specifics,
    # e.g. "What is malaria, give symptoms and precautions?".
    # These need the full BM25+LLM pipeline for a multi-part answer.
    # Pure clinical queries ("What are the symptoms of malaria?") must NOT be
    # blocked — they should use the fast-path and return Neo4j data directly.
    compound_signals = ["tell me about", "give me", "describe ", "explain "]
    has_compound_overview = any(s in lowered for s in compound_signals)
    if not has_compound_overview:
        # "What is X, [and] give/list/show symptoms?" pattern only
        has_compound_overview = bool(
            re.search(r"\bwhat is\b", lowered)
            and re.search(
                r",\s*(give|list|provide|show|tell|also|what are)|(?:\band\b.{0,30}\b(give|list|provide|symptoms|precautions)\b)",
                lowered,
            )
        )
    if has_compound_overview and (symptom_intent or precaution_intent):
        return None

    if disease_name and (symptom_intent or precaution_intent):
        profile = _fetch_disease_profile(graph, disease_name)
        symptoms = profile.get("symptoms", [])
        precautions = profile.get("precautions", [])

        if symptom_intent and not precaution_intent:
            if symptoms:
                bullet_text = _bullet_list(symptoms)
                return (
                    f"**Symptoms of {disease_name}** ({len(symptoms)} reported):\n"
                    f"{bullet_text}\n\n"
                    "Please consult a healthcare professional for personalized medical advice."
                )
            return None

        if precaution_intent and not symptom_intent:
            if precautions:
                bullet_text = _bullet_list(precautions)
                return (
                    f"**Precautions for {disease_name}** ({len(precautions)} recommended):\n"
                    f"{bullet_text}\n\n"
                    "Please consult a healthcare professional for personalized medical advice."
                )
            return None

        sections: list[str] = [f"**{disease_name}**"]
        if symptoms:
            sections.append(f"**Symptoms** ({len(symptoms)} reported):\n{_bullet_list(symptoms)}")
        if precautions:
            sections.append(f"**Precautions** ({len(precautions)} recommended):\n{_bullet_list(precautions)}")
        if len(sections) > 1:
            sections.append("Please consult a healthcare professional for personalized medical advice.")
            return "\n\n".join(sections)
        return None

    # Diseases with a specific symptom
    symptom_hint = _extract_symptom_hint(input_text)
    if symptom_hint and disease_name is None and "disease" in lowered and "symptom" in lowered:
        rows = graph.query(
            """
            MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
            WHERE toLower(s.name) CONTAINS toLower($symptom)
            RETURN DISTINCT d.name AS name ORDER BY name
            """,
            params={"symptom": symptom_hint},
        )
        names = [row.get("name") for row in rows if row.get("name")]
        if not names:
            return None
        readable = ", ".join(name.replace("_", " ") for name in names[:20])
        return f"Diseases associated with the symptom '{symptom_hint}' include: {readable}."

    # Overview for a named disease
    # Guard: do not turn mechanism/definition/comparison questions into a generic disease overview.
    if disease_name and (not _is_general_knowledge_query(input_text)) and any(
        token in lowered for token in ["what is", "tell me", "about", "overview", "details"]
    ):
        overview = _build_disease_overview(graph, disease_name)
        if overview:
            overview += "\n\nPlease consult a healthcare professional for personalized medical advice."
            return overview

    # List all diseases in graph
    if "disease" in lowered and any(token in lowered for token in ["list", "what diseases", "knowledge graph", "all"]):
        rows = graph.query(
            "MATCH (d:Disease) RETURN DISTINCT d.name AS name ORDER BY name"
        )
        names = [row.get("name") for row in rows if row.get("name")]
        if names:
            readable = ", ".join(name.replace("_", " ") for name in names[:30])
            suffix = f" (showing {min(30, len(names))} of {len(names)})." if len(names) > 30 else "."
            return f"Diseases in the knowledge graph include: {readable}{suffix}"

    return None


# ---------------------------------------------------------------------------
# Source / output helpers
# ---------------------------------------------------------------------------

def _extract_sources(text: str) -> tuple[str, list[str]]:
    match = re.search(r"\n\nSources:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return text, []
    source_parts = [part.strip() for part in match.group(1).split(",") if part.strip()]
    deduped = list(dict.fromkeys(source_parts))
    return text[: match.start()].strip(), deduped


def _serialize_sources(source_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in source_entries if isinstance(item, dict)]


def _output_quality_score(text: str) -> float:
    lowered = text.lower().strip()
    if not lowered:
        return 0.0
    score = 0.0
    if not _is_empty_or_unavailable(text):
        score += 2.0
    if any(token in lowered for token in ["symptom", "precaution", "disease", "patient", "treatment"]):
        score += 1.0
    if any(token in text for token in ["- ", "•", "\n1", "\n2", "\n**"]):
        score += 0.5
    if len(text) > 200:
        score += 0.5
    return score


def _synthesize_outputs(tool_outputs: list[tuple[str, str]]) -> str:
    scored = []
    for tool_name, output_text in tool_outputs:
        cleaned = (output_text or "").strip()
        if not cleaned:
            continue
        score = _output_quality_score(cleaned)
        scored.append((tool_name, cleaned, score))
    if not scored:
        return tool_outputs[0][1] if tool_outputs else ""
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[0][1]


# ---------------------------------------------------------------------------
# Query context / routing helpers
# ---------------------------------------------------------------------------

_GENERAL_KNOWLEDGE_SIGNALS = [
    # Mechanism / explanation intent
    "how does", "how do", "how is", "how are",
    "why does", "why do", "why is", "why are",
    "what is the mechanism", "what is the pathophysiology",
    "pathophysiology", "mechanism of action", "mechanism",
    "step by step", "what happens when", "explain how", "explain why",
    "how does it work", "how is this caused",

    # Definition / comparison / education intent
    "what is the difference between", "difference between",
    "what does", "what do the numbers mean",
    "systolic", "diastolic",
    "what is the role of", "what is the function of",

    # Specific clinical education examples
    "what foods should", "what diet", "what should i eat",
    "dash diet", "what is the dash diet",
    "hba1c", "what is hba1c", "what does hba1c measure",
    "insulin resistance",
]


def _is_general_knowledge_query(input_text: str) -> bool:
    lowered = (input_text or "").lower()
    return any(signal in lowered for signal in _GENERAL_KNOWLEDGE_SIGNALS)


def _select_tool_names(input_text: str, chat_history: str = "") -> list[str]:
    lowered = f"{input_text} {chat_history[-600:]}".lower()

    # General medical knowledge → GeneralMedical first
    if _is_general_knowledge_query(input_text):
        return ["GeneralMedical"]

    document_terms = ["document", "pdf", "file", "uploaded", "upload", "material", "notes", "report"]
    experience_terms = ["experience", "feedback", "review", "patient said", "patient says", "opinion"]
    graph_terms = ["disease", "symptom", "precaution", "how many", "list", "associated"]

    selected = []
    if any(term in lowered for term in document_terms):
        selected.append("Documents")
    if any(term in lowered for term in experience_terms):
        selected.append("Experiences")
    if any(term in lowered for term in graph_terms) or not selected:
        selected.append("Graph")
    if "Graph" not in selected:
        selected.append("Graph")

    return selected


def _extract_recent_disease_hint(chat_history: str) -> str | None:
    if not chat_history:
        return None
    lowered_history = chat_history.lower()
    latest_hit: tuple[int, str] | None = None
    for disease_name in _get_clinical_disease_names():
        idx = lowered_history.rfind(disease_name.lower())
        if idx != -1 and (latest_hit is None or idx > latest_hit[0]):
            latest_hit = (idx, disease_name)
    if latest_hit:
        return latest_hit[1]
    return None


def _is_followup_question(input_text: str) -> bool:
    lowered = input_text.lower().strip()
    followup_markers = [
        "what about", "and precautions", "and symptoms", "its symptoms", "its precautions",
        "its treatment", "its management", "what are its", "what is its", "what else",
        "how about", "for it", "for this", "for that", "also",
    ]
    return any(marker in lowered for marker in followup_markers)


def _contextualize_query(input_text: str, chat_history: str) -> tuple[str, str | None]:
    clean_input = (input_text or "").strip()
    explicit_disease = _find_disease_name_in_query(clean_input)

    arrow_match = re.match(r"(.+?)\s*->\s*(.+)$", clean_input)
    if arrow_match:
        left = arrow_match.group(1).strip()
        right = arrow_match.group(2).strip()
        left_disease = _find_disease_name_in_query(left) or explicit_disease
        if right and left_disease and _is_followup_question(right) and left_disease.lower() not in right.lower():
            right = f"{right} for {left_disease}"
        if right:
            return right, left_disease

    if explicit_disease:
        if _is_followup_question(clean_input):
            replaced = re.sub(r"\bits\b", explicit_disease, clean_input, flags=re.IGNORECASE)
            return replaced, explicit_disease
        return clean_input, explicit_disease

    hint = _extract_recent_disease_hint(chat_history)
    if not hint:
        return clean_input, None

    lowered = clean_input.lower()
    if hint.lower() in lowered:
        return clean_input, hint

    if _is_followup_question(clean_input):
        return f"{clean_input} for {hint}", hint

    return clean_input, hint


# ---------------------------------------------------------------------------
# Tool runner
# ---------------------------------------------------------------------------

def _run_tool(tool_name: str, input_text: str) -> tuple[str, list[str]]:
    if tool_name == "GeneralMedical":
        output = _invoke_general_medical_tool(input_text)
        return output, []
    elif tool_name == "Documents":
        output_text, references = _invoke_documents_tool_with_sources(input_text)
        reference_strings = []
        for ref in references:
            label = ref.get("label") or ref.get("source")
            if label:
                reference_strings.append(json.dumps(ref, ensure_ascii=False))
        return output_text, reference_strings
    elif tool_name == "Experiences":
        raw = _invoke_reviews_tool(input_text)
        return _extract_sources(str(raw))
    else:  # Graph
        raw = _invoke_graph_tool(input_text)
        return _extract_sources(str(raw))


# ---------------------------------------------------------------------------
# Core invocation
# ---------------------------------------------------------------------------

def _invoke_single_query(input_text: str, chat_history: str = "", routing_enabled: bool | None = None) -> dict:
    contextual_input, history_hint = _contextualize_query(input_text, chat_history)
    steps: list[str] = []

    if contextual_input != input_text and AGENT_DEBUG_STEPS:
        steps.append(f"Follow-up context applied: {contextual_input}")
    elif history_hint and AGENT_DEBUG_STEPS:
        steps.append(f"History hint detected: {history_hint}")

    # --- Adaptive hybrid path (default) ---
    effective_routing = ENABLE_ADAPTIVE_HYBRID if routing_enabled is None else bool(routing_enabled)
    if effective_routing:
        try:
            routed_label = route_query(contextual_input)
            print(f"[routing] routing=true label={routed_label} query={contextual_input}")
            hybrid_result = hybrid_retrieve(contextual_input)
            steps.extend(hybrid_result.get("steps", []))
            final_output = (hybrid_result.get("answer") or hybrid_result.get("context", "") or "").strip()
            serialized_sources = _serialize_sources(hybrid_result.get("sources", []))
            route_label = str(hybrid_result.get("route") or "UNKNOWN")
            retrieved_docs = hybrid_result.get("sources", []) or []
            logging.warning(
                f"[SRC_DEBUG] route={route_label} docs={len(retrieved_docs)} sources={len(serialized_sources)}"
            )

            if not final_output or _is_empty_or_unavailable(final_output):
                # Last resort: general medical LLM
                final_output = _general_medical_answer(contextual_input)
                steps.append("Hybrid returned empty — general medical LLM used")
                serialized_sources = []
            if not serialized_sources and route_label in {"FACTUAL", "RELATIONAL", "COMPLEX"} and final_output.strip():
                serialized_sources = [
                    {
                        "content": final_output[:240],
                        "metadata": {"source": "synthetic_fallback", "route": route_label},
                    }
                ]

            response_payload = {
                "input": input_text,
                "output": final_output,
                "intermediate_steps": steps,
                "sources": serialized_sources,
            }
            if ENABLE_HYBRID_DEBUG_CONTEXT:
                response_payload["debug_context"] = {
                    "route": hybrid_result.get("route"),
                    "context": hybrid_result.get("context", ""),
                    "steps": hybrid_result.get("steps", []),
                    "evidence": hybrid_result.get("evidence", {}),
                }
            return response_payload
        except Exception as exc:
            if AGENT_DEBUG_STEPS:
                steps.append("Adaptive hybrid failed; switching to direct tool routing")
    else:
        try:
            print(f"[routing] routing=false direct_bm25 query={contextual_input}")
            docs = bm25_retrieve_sync(contextual_input, top_k=5)
            context = "\n\n".join([d.get("content", "") for d in docs if isinstance(d, dict)])
            final_output = call_llm_direct(contextual_input, context)
            serialized_sources = [
                {"content": d.get("content", ""), "metadata": d.get("metadata", {})}
                for d in docs if isinstance(d, dict)
            ]
            return {
                "input": input_text,
                "output": final_output,
                "intermediate_steps": steps + ["routing=false: bm25 + direct llm"],
                "sources": serialized_sources,
                "debug_context": {"route": "HYBRID_NO_ROUTING", "context": context, "steps": [], "evidence": {}},
            }
        except Exception:
            if AGENT_DEBUG_STEPS:
                steps.append("routing=false bm25 path failed; switching to direct tool routing")

    # --- Fallback: direct tool selection ---
    selected_tools = _select_tool_names(contextual_input, chat_history)
    tool_outputs: list[tuple[str, str]] = []
    all_sources: list[str] = []

    for tool_name in selected_tools:
        clean_output, sources = _run_tool(tool_name, contextual_input)
        tool_outputs.append((tool_name, clean_output))
        steps.append(f"ToolRouter selected: {tool_name}")
        for source in sources:
            if source not in all_sources:
                all_sources.append(source)
        if tool_name in {"Graph", "GeneralMedical"} and clean_output and not _is_empty_or_unavailable(clean_output):
            break

    final_output = _synthesize_outputs(tool_outputs)

    # Ultimate fallback: general medical LLM
    if not final_output.strip() or _is_empty_or_unavailable(final_output):
        final_output = _general_medical_answer(contextual_input)
        all_sources = []
        steps.append("All tools empty — general medical LLM used as final fallback")

    return {
        "input": input_text,
        "output": final_output,
        "intermediate_steps": steps,
        "sources": all_sources,
    }


def invoke_chatbot_agent(input_text: str, chat_history: str = "", routing_enabled: bool | None = None) -> dict:
    return _invoke_single_query(input_text, chat_history, routing_enabled)



async def run_agent_with_timeout(agent, query, timeout=15):
    return {
        "output": call_llm_direct(query, ""),
        "sources": [],
        "route": "DIRECT",
    }


# ---------------------------------------------------------------------------
# Tool definitions & React agent (optional)
# ---------------------------------------------------------------------------

tools = [
    Tool(
        name="GeneralMedical",
        func=_invoke_general_medical_tool,
        description="""Use this for general medical knowledge questions that require broad clinical understanding.
        This includes: mechanisms (how does X work?), physiology (what is systolic vs diastolic?),
        pharmacology (how does metformin work?), diet and lifestyle (what foods should a diabetic avoid?),
        pathophysiology (how does insulin resistance develop?), clinical concepts (what is HbA1c?).
        Use the entire user question as input.
        """,
    ),
    Tool(
        name="Experiences",
        func=_invoke_reviews_tool,
        description="""Useful when you need to answer qualitative questions about medical experiences,
        patient feedback, or subjective information that could be answered using semantic search.
        Not useful for objective questions involving counting, percentages, aggregations, or listing facts.
        Use the entire prompt as input to the tool.
        """,
    ),
    Tool(
        name="Graph",
        func=_invoke_graph_tool,
        description="""Useful for answering questions about diseases, symptoms, precautions, and their
        relationships in the knowledge graph. Use for queries involving counts, statistics, lists, or
        structured medical information from the database.
        Examples: "What are symptoms of malaria?", "List diseases in the graph", "How many diseases have fever?"
        Use the entire prompt as input to the tool.
        """,
    ),
    Tool(
        name="Documents",
        func=_invoke_documents_tool,
        description="""Useful for answering free-text questions based on uploaded PDFs/TXT/CSV/MD files
        stored as document chunks. Use when users ask what an uploaded document says or request
        narrative context from ingested files.
        """,
    ),
]

# chat_model = ChatOpenAI(...)
# chatbot_rag_agent = create_react_agent(...)
# chatbot_rag_agent_executor = AgentExecutor(...)

