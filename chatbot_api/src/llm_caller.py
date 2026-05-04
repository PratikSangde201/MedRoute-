import json
import os
import re

import httpx

try:
    from src.retrieval.router_policy import extract_disease_entities, KNOWN_DISEASES_SORTED
except ImportError:
    try:
        from router_policy import extract_disease_entities, KNOWN_DISEASES_SORTED
    except ImportError:
        def extract_disease_entities(query: str) -> list:
            return []
        KNOWN_DISEASES_SORTED = []


def _base_url() -> str:
    raw = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()

    # FIX: only rewrite host.docker.internal when NOT inside a Docker
    # container. Previously this always rewrote to 127.0.0.1, which
    # inside a container points to the container itself — not the host
    # machine where Ollama runs — causing every LLM call to fail with
    # "connection refused".
    #
    # When running inside Docker, host.docker.internal is the correct
    # hostname and must be kept as-is so Docker's internal DNS resolves
    # it to the host gateway. We only rewrite on the host (dev machine)
    # where host.docker.internal is not defined.
    running_in_docker = os.path.exists("/.dockerenv")

    if not running_in_docker:
        raw = raw.replace("host.docker.internal", "127.0.0.1")

    return raw.rstrip("/")


def _primary_model() -> str:
    return os.getenv("LLM_MODEL", "llama3.1:8b")


def _fast_model() -> str:
    """
    Small model for latency-sensitive routes (FACTUAL, GENERAL, HYBRID).
    Falls back to primary model if FAST_LLM_MODEL not set.
    Set FAST_LLM_MODEL=llama3.2:1b or phi3:mini in .env for speed gains.
    """
    return os.getenv("FAST_LLM_MODEL", _primary_model())


def _model_for_route(route: str) -> str:
    """
    Route-aware model selection:
    - RELATIONAL/COMPLEX: primary model (needs reasoning, worth the latency)
    - FACTUAL/GENERAL/HYBRID: fast model if configured
    """
    r = (route or "").upper()
    if r in ("RELATIONAL", "COMPLEX"):
        return _primary_model()
    return _fast_model()


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _detect_relational_intent(query: str) -> str:
    q = query.lower()
    if "what causes" in q or "what cause" in q:
        return "cause"
    if "lead to" in q or "leads to" in q:
        return "causal_chain"
    if "relationship between" in q:
        return "relationship"
    if "mechanism by which" in q or "mechanism of" in q:
        return "mechanism"
    if "affect patients with" in q or ("affect" in q and "patients" in q):
        return "interaction"
    if "complications arise from" in q or ("complications" in q and "patients with" in q):
        return "complications"
    return "relationship"


def _build_factual_instruction(query: str, entities: list[str]) -> str:
    """Build a factual instruction that matches what the user actually asked."""
    q = query.lower()
    disease = entities[0].title() if entities else "this condition"

    if any(w in q for w in ["symptom", "symptoms", "signs and symptoms", "sign of", "signs of"]):
        return f"Describe the main symptoms of {disease} in a clear, helpful way in 3-4 sentences."

    if any(w in q for w in ["treat", "treatment", "therapy", "medication", "drug", "medicine", "manage", "management", "cure"]):
        return f"Explain the treatment and management options for {disease} in 3-4 sentences."

    if any(w in q for w in ["precaution", "precautions", "prevent", "prevention", "avoid", "protection"]):
        return f"Explain the key precautions and prevention steps for {disease} in 3-4 sentences."

    if any(w in q for w in ["diagnos", "test", "detect", "detection", "diagnosis", "how is it detected"]):
        return f"Explain how {disease} is diagnosed in 3-4 sentences, covering key tests and criteria."

    if any(w in q for w in ["compare", "difference", "vs ", "versus", "differ"]):
        return "Compare these conditions clearly in 3-4 sentences, highlighting the key differences."

    if any(w in q for w in ["cause", "causes", "caused by", "etiology", "why"]):
        return f"Explain what causes {disease} in 3-4 sentences, covering the main risk factors."

    if any(w in q for w in ["vaccine", "vaccination", "immuniz"]):
        return f"Explain the vaccination approach for {disease} in 3-4 sentences."

    return f"Answer this question about {disease} clearly and helpfully in 3-4 sentences."


def _build_relational_instruction(query: str, entities: list[str]) -> str:
    intent = _detect_relational_intent(query)
    # Shared suffix: always demand disease-specific clinical vocabulary
    vocab_note = (
        " Use specific clinical terminology: name the relevant organ systems, "
        "biological markers (lab values, pathological terms), and any medication "
        "classes or drug names that are directly relevant to these conditions."
    )

    if intent == "cause":
        if entities:
            d = entities[0].title()
            return (
                f"Explain what causes {d} — include the main risk factors, "
                f"underlying physiological mechanism, and mention specific clinical "
                f"terms such as blood pressure values, hormones, or organ involvement "
                f"in 3-4 sentences.{vocab_note}"
            )
        return (
            f"Explain what causes this condition — include risk factors, mechanism, "
            f"and specific clinical terms in 3-4 sentences.{vocab_note}"
        )

    if intent == "causal_chain" and len(entities) >= 2:
        e1, e2 = entities[0].title(), entities[1].title()
        return (
            f"Explain specifically how {e1} leads to {e2} — describe the "
            f"physiological pathway, name the key organs or lab markers involved, "
            f"and state the clinical outcome in 3-4 sentences.{vocab_note}"
        )

    if intent == "mechanism" and len(entities) >= 2:
        e1, e2 = entities[0].title(), entities[1].title()
        return (
            f"Explain the mechanism by which {e1} causes {e2} — cover the "
            f"biological pathway, name specific cells/organs/markers, "
            f"and state the clinical result in 3-4 sentences.{vocab_note}"
        )

    if intent == "mechanism" and len(entities) == 1:
        d = entities[0].title()
        return (
            f"Explain the mechanism by which {d} causes its main complications — "
            f"name the specific pathological processes, organs, and clinical outcomes "
            f"in 3-4 sentences.{vocab_note}"
        )

    if intent == "relationship" and len(entities) >= 2:
        e1, e2 = entities[0].title(), entities[1].title()
        return (
            f"Explain the clinical relationship between {e1} and {e2} — describe "
            f"how one leads to or worsens the other, include the specific mechanism, "
            f"name the clinical features of both conditions (symptoms, relevant lab "
            f"markers, treatments), and state the clinical outcome in 3-4 sentences.{vocab_note}"
        )

    if intent == "interaction" and len(entities) >= 2:
        e1, e2 = entities[0].title(), entities[1].title()
        return (
            f"Explain how {e1} affects patients who also have {e2} — cover the "
            f"interaction mechanism, added clinical risks, specific complications, "
            f"and relevant management considerations in 3-4 sentences.{vocab_note}"
        )

    if intent == "complications" and len(entities) >= 2:
        e1, e2 = entities[0].title(), entities[1].title()
        return (
            f"Explain what complications arise from {e1} in patients who have {e2} — "
            f"describe the mechanism, name specific clinical complications and "
            f"relevant treatments in 3-4 sentences.{vocab_note}"
        )

    if len(entities) >= 2:
        e1, e2 = entities[0].title(), entities[1].title()
        return (
            f"Explain the clinical relationship or interaction between {e1} and {e2} "
            f"in 3-4 sentences, naming specific mechanisms, organ systems, lab "
            f"markers, and treatments.{vocab_note}"
        )

    if len(entities) == 1:
        d = entities[0].title()
        return (
            f"Answer the question about {d} clearly in 3-4 sentences, naming "
            f"specific pathological processes, clinical features, and relevant "
            f"medical terms.{vocab_note}"
        )

    return (
        f"Explain the relationship or mechanism between the medical concepts in "
        f"this question in 3-4 sentences, using specific clinical terminology, "
        f"organ names, lab markers, and medication names.{vocab_note}"
    )


def _build_complex_instruction(entities: list[str]) -> str:
    vocab_note = (
        " Use the disease's specific clinical vocabulary: name the key hormone, "
        "metabolite, or pathogen involved (e.g. insulin, bilirubin, Mycobacterium), "
        "the organs affected (e.g. pancreas, kidney, lungs), pathological terms for "
        "each complication (e.g. nephropathy, cirrhosis, neuropathy), and relevant "
        "lab or treatment terms (e.g. HbA1c, isoniazid, corticosteroid)."
    )
    if entities:
        d = entities[0].title()
        return (
            f"For {d}: "
            f"(1) Describe the main risk factors — explain them in terms of the "
            f"underlying biological mechanisms (e.g. insulin resistance, impaired "
            f"immune response), not just lifestyle labels; "
            f"(2) Describe the key complications that arise if untreated — name each "
            f"complication by its clinical/pathological term (e.g. nephropathy, not "
            f"just 'kidney damage'). "
            f"Answer in 4-5 sentences — 2 on risk factors, 2-3 on complications.{vocab_note}"
        )
    return (
        "Describe the main risk factors (by biological mechanism name) and the key "
        "clinical complications (by their pathological names) of this condition. "
        f"Answer in 4-5 sentences using specific medical terminology.{vocab_note}"
    )


def _build_general_instruction(query: str) -> str:
    q = query.lower()
    vocab_note = (
        " Include specific clinical terms: name relevant medications or drug classes, "
        "lab values or monitoring parameters (e.g. blood glucose, blood pressure, "
        "platelet count), and the disease's key pathological features where relevant."
    )
    if "lifestyle" in q:
        return (
            "Give practical lifestyle advice for managing this condition — cover diet, "
            "exercise, weight management, and clinical monitoring targets (e.g. blood "
            "glucose levels, blood pressure targets) in 3-4 clear sentences." + vocab_note
        )
    if "dietary" in q or "diet" in q:
        return (
            "Give specific dietary recommendations for this condition — name foods to "
            "favour and avoid, key nutrients to monitor, and relevant clinical targets "
            "in 3-4 sentences." + vocab_note
        )
    if "emergency" in q or "when should" in q:
        return (
            "Explain clearly when a patient should seek emergency care for this "
            "condition — list specific warning signs by their clinical names "
            "(e.g. haemorrhagic shock, consolidation, respiratory failure) requiring "
            "immediate attention in 3-4 sentences." + vocab_note
        )
    return (
        "Answer this medical question clearly and practically in 3-4 sentences, "
        "using specific clinical terminology and naming relevant medications, "
        "lab markers, or pathological features." + vocab_note
    )


def build_prompt(query: str, context: str = "", route: str = "") -> str:
    r = (route or "").strip().upper()
    ctx = (context or "").strip()
    entities = extract_disease_entities(query)

    system = (
        "You are a helpful, knowledgeable medical assistant. "
        "Give clear, accurate, and conversational answers. "
        "Be specific and practical. Do not repeat the question back to the user."
    )

    if r == "RELATIONAL":
        instruction = _build_relational_instruction(query, entities)
        ctx_part = f"\nRelevant medical context:\n{ctx}\n" if ctx else ""
        bullet_hint = (
            "\nAfter your main answer, list 2-3 key clinical points as bullet points using '- '."
            " Each bullet must use the specific medical terms from the context"
            " (e.g. drug names, lab markers, pathological terms such as nephropathy, HbA1c, isoniazid)."
        )
        return f"{system}{ctx_part}\n\nQuestion: {query}\n{instruction}{bullet_hint}\nAnswer:"

    if r == "COMPLEX":
        instruction = _build_complex_instruction(entities)
        ctx_part = f"\nRelevant medical context:\n{ctx}\n" if ctx else ""
        bullet_hint = (
            "\nAfter your main answer, list 2-3 key comparative points as bullet points using '- '."
            " Each bullet must name the specific clinical or pathological term"
            " (e.g. organ system, biomarker, complication name) rather than general language."
        )
        return f"{system}{ctx_part}\n\nQuestion: {query}\n{instruction}{bullet_hint}\nAnswer:"

    if r == "GENERAL":
        instruction = _build_general_instruction(query)
        bullet_hint = (
            "\nThen summarize 2-3 key points as bullet points using '- '."
            " Each bullet must include the specific medical term, drug name, or lab value"
            " most relevant to the question."
        )
        return f"{system}\n\nQuestion: {query}\n{instruction}{bullet_hint}\nAnswer:"

    # FACTUAL / HYBRID — use query-aware instruction
    instruction = _build_factual_instruction(query, entities)
    if ctx:
        return (
            f"{system}\n\n"
            f"Relevant medical context:\n{ctx}\n\n"
            f"Question: {query}\n{instruction}\nAnswer:"
        )
    return f"{system}\n\nQuestion: {query}\n{instruction}\nAnswer:"


# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------

def _generation_options(num_predict: int) -> dict:
    return {
        "num_predict": num_predict,
        "temperature": 0.0,
        "top_k": 10,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }


_HTTP_STREAM_TIMEOUT = httpx.Timeout(90.0, connect=10.0)


# ---------------------------------------------------------------------------
# LLM call functions
# ---------------------------------------------------------------------------

async def call_llm_streaming(
    query: str, context: str = "", num_predict: int = 130, route: str = ""
) -> str:
    model = _model_for_route(route)
    payload = {
        "model": model,
        "prompt": build_prompt(query, context, route),
        "stream": True,
        "options": _generation_options(num_predict),
    }
    collected: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=_HTTP_STREAM_TIMEOUT) as client:
            async with client.stream("POST", f"{_base_url()}/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    collected.append(chunk.get("response", ""))
                    if chunk.get("done", False):
                        break
    except Exception as e:
        print(f"[LLM_STREAM_ERR] model={model} route={route} err={e}", flush=True)
        return ""
    return "".join(collected).strip()


async def call_llm_direct(
    query: str, context: str = "", num_predict: int = 130, route: str = "GENERAL"
) -> str:
    model = _model_for_route(route)
    context_section = context.strip() if context and len(context.strip()) > 20 else ""
    payload = {
        "model": model,
        "prompt": build_prompt(query, context_section, route),
        "stream": False,
        "options": _generation_options(num_predict),
    }
    try:
        async with httpx.AsyncClient(timeout=80.0) as client:
            response = await client.post(f"{_base_url()}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json().get("response", "").strip()
        print(f"[LLM_DIRECT] model={model} len={len(result)} route={route}", flush=True)
        return result if len(result) > 20 else ""
    except Exception as e:
        print(f"[LLM_DIRECT_ERR] model={model} route={route} err={e}", flush=True)
        return ""


async def call_llm_knowledge_only(query: str, num_predict: int = 120) -> str:
    """Pure LLM call — no retrieval. Used for GENERAL route."""
    model = _fast_model()
    payload = {
        "model": model,
        "prompt": build_prompt(query, "", "GENERAL"),
        "stream": False,
        "options": _generation_options(num_predict),
    }
    try:
        async with httpx.AsyncClient(timeout=75.0) as client:
            response = await client.post(f"{_base_url()}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json().get("response", "").strip()
        return result if len(result) > 20 else ""
    except Exception as e:
        print(f"[LLM_KNOWLEDGE_ERR] err={e}", flush=True)
        return ""


# ---------------------------------------------------------------------------
# Quality gate and retry
# ---------------------------------------------------------------------------

def is_weak_answer(answer: str, query: str) -> bool:
    if not answer or len(answer) < 30:
        return True
    lowered = answer.lower()
    bad_phrases = [
        "unable to provide", "cannot provide", "insufficient",
        "no information available", "i don't know",
        "i cannot answer", "not able to", "cannot answer",
        "i'm not sure", "i am not sure", "as an ai",
        "as a language model", "i don't have access",
        "please consult a doctor" if len(answer) < 80 else "__never__",
        "i cannot give medical advice" if len(answer) < 80 else "__never__",
        "medical information for query:",
        "based on retrieved clinical context for",
    ]
    if any(p in lowered for p in bad_phrases):
        return True
    # Check that at least one meaningful query word appears in the answer
    q_words = [w.lower() for w in query.split() if len(w) > 4]
    if len(q_words) >= 3 and sum(1 for w in q_words if w in lowered) == 0:
        return True
    return False


async def call_llm_with_retry(
    query: str, context: str = "", num_predict: int = 120, route: str = "GENERAL"
) -> str:
    """
    Primary streaming call + exactly ONE retry on weak answer.
    Retry uses shorter context and non-streaming call.
    """
    answer = await call_llm_streaming(query, context, num_predict=num_predict, route=route)

    if is_weak_answer(answer, query):
        print(f"[LLM_RETRY] Weak primary answer, retrying. route={route}", flush=True)
        retry_ctx = context[:300] if context else ""
        answer = await call_llm_direct(query, retry_ctx, num_predict=num_predict, route=route)

    if not answer or len(answer) < 10:
        answer = await call_llm_knowledge_only(query, num_predict=num_predict)

    return answer or "Please consult a qualified healthcare professional for guidance on this question."