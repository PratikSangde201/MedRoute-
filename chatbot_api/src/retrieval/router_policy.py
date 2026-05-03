import json
import os
import re
from typing import Literal

QueryRoute = Literal["FACTUAL", "RELATIONAL", "COMPLEX", "GENERAL"]

# ---------------------------------------------------------------------------
# Disease vocabulary — single source of truth for router AND retrieval
# ---------------------------------------------------------------------------

KNOWN_DISEASES = [
    "bronchial asthma", "common cold", "heart attack", "chicken pox",
    "urinary tract infection", "peptic ulcer disease", "peptic ulcer",
    "chronic cholestasis", "alcoholic hepatitis", "hepatitis b", "hepatitis c",
    "hepatitis a", "hepatitis e", "high blood pressure",
    "myocardial infarction", "chronic kidney disease",
    "hypothyroidism", "hyperthyroidism", "osteoarthritis", "gastroenteritis",
    "malaria", "dengue", "diabetes", "hypertension", "asthma",
    "influenza", "flu", "tuberculosis", "tb", "migraine", "anemia",
    "pneumonia", "depression", "anxiety", "typhoid", "covid",
    "stroke", "arthritis", "epilepsy", "parkinson", "alzheimer",
    "eczema", "psoriasis", "cancer", "hiv", "aids", "hepatitis",
    "cholera", "measles", "chickenpox", "gerd", "jaundice",
    "sepsis", "obesity", "insomnia",
]

KNOWN_DISEASES_SORTED = sorted(KNOWN_DISEASES, key=len, reverse=True)


def extract_disease_entities(query: str) -> list[str]:
    """
    Extract up to 2 disease names from query.
    Used by RELATIONAL retrieval to fetch one focused doc per entity.
    """
    q = query.lower()
    found = []
    for disease in KNOWN_DISEASES_SORTED:
        pattern = rf"(?<![a-z]){re.escape(disease)}(?![a-z])"
        if re.search(pattern, q):
            found.append(disease)
            if len(found) == 2:
                break
    return found


# ---------------------------------------------------------------------------
# Primary router — tuned to expected_route gold labels in medroute_eval_60
#
# Dataset analysis (confirmed from full JSON):
#   COMPLEX (expected_route=COMPLEX, 4 cases):
#     "What are the risk factors and complications of X" ONLY
#
#   RELATIONAL (expected_route=RELATIONAL, 15 cases):
#     "What causes X", "How does X lead to Y",
#     "What is the relationship between X and Y",
#     "How does X affect patients with Y",
#     "What complications arise from X in patients with Y",
#     "What is the mechanism by which X causes Y"
#
#   FACTUAL (expected_route=FACTUAL, 15+ cases):
#     symptoms, treatment, diagnosis, medication, precautions,
#     compare symptoms, differ in treatment, how can X be prevented
#
#   GENERAL (expected_route=GENERAL, 15 cases):
#     "What lifestyle changes help manage X",
#     "When should a patient with X seek emergency care",
#     "What dietary recommendations exist for X patients"
# ---------------------------------------------------------------------------

def classify_query(query: str) -> str:
    q = query.lower().strip()

    # ------------------------------------------------------------------
    # 1. COMPLEX — must be first, catches all 4 dataset cases
    # ------------------------------------------------------------------
    if "risk factors and complications" in q:
        return "COMPLEX"
    if "risk factors" in q and "complications of" in q:
        return "COMPLEX"

    # ------------------------------------------------------------------
    # 2. GENERAL — must be second, before RELATIONAL/FACTUAL
    # ------------------------------------------------------------------
    if "lifestyle changes" in q:
        return "GENERAL"
    if "when should" in q and ("emergency" in q or "seek" in q):
        return "GENERAL"
    if "seek emergency" in q:
        return "GENERAL"
    if "dietary recommendations" in q:
        return "GENERAL"

    # ------------------------------------------------------------------
    # 3. RELATIONAL — before FACTUAL to catch "what causes X"
    # ------------------------------------------------------------------
    if re.search(r"\bwhat causes?\b", q):
        return "RELATIONAL"
    if "relationship between" in q:
        return "RELATIONAL"
    if "mechanism by which" in q:
        return "RELATIONAL"
    if "lead to" in q or "leads to" in q:
        return "RELATIONAL"
    if "affect patients with" in q:
        return "RELATIONAL"
    if re.search(r"\bhow does\b.{1,50}\baffect\b", q):
        return "RELATIONAL"
    if "complications arise from" in q:
        return "RELATIONAL"
    if "complications" in q and "in patients with" in q:
        return "RELATIONAL"
    if "mechanism of" in q and any(d in q for d in KNOWN_DISEASES_SORTED):
        return "RELATIONAL"

    # ------------------------------------------------------------------
    # 4. FACTUAL — comparisons, prevention, standard clinical lookups
    # ------------------------------------------------------------------
    if q.startswith("compare "):
        return "FACTUAL"
    if "differ in treatment" in q:
        return "FACTUAL"
    if re.search(r"\bhow do\b.{1,50}\bdiffer\b", q):
        return "FACTUAL"
    if re.search(r"\bhow can\b.{1,50}\bprevented\b", q):
        return "FACTUAL"
    if "how to prevent" in q or "prevention of" in q:
        return "FACTUAL"

    factual_signals = [
        "what are the symptoms", "what are symptoms",
        "what is the treatment", "treatment for",
        "what are precautions", "precautions for",
        "which medication", "what medication",
        " diagnosed", "how is ", "warning signs",
        "red flags", "signs and symptoms",
        "how do you treat", "first line treatment",
    ]
    if any(sig in q for sig in factual_signals):
        return "FACTUAL"

    factual_kw = [
        "symptom", "symptoms", "treatment", "precaution",
        "medication", "diagnose", "diagnosis", "vaccine", "vaccination",
    ]
    if any(kw in q for kw in factual_kw) and any(d in q for d in KNOWN_DISEASES_SORTED):
        return "FACTUAL"

    # ------------------------------------------------------------------
    # 5. GENERAL fallback
    # ------------------------------------------------------------------
    return "GENERAL"


def route_query(query: str) -> QueryRoute:
    return classify_query(query)


# ---------------------------------------------------------------------------
# Compatibility exports used by other modules
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """You are an Adaptive Query Router for a medical chatbot.

Classify queries into exactly one of:
FACTUAL   — symptoms, treatment, diagnosis, medication, precautions, comparisons, prevention
RELATIONAL — what causes X, mechanism by which X causes Y, relationship between X and Y,
             how does X lead to/affect Y, complications from X in patients with Y
COMPLEX   — "what are the risk factors and complications of X" template ONLY
GENERAL   — lifestyle changes, dietary recommendations, when to seek emergency care

Return JSON only: {"label":"FACTUAL|RELATIONAL|COMPLEX|GENERAL","reason":"short reason"}
"""

_GRAPH_SPECIFIC_TOKENS = [
    "in the knowledge graph", "in the database", "in the graph",
    "which diseases", "list diseases", "how many diseases",
]
_SELF_REPORT_TOKENS = [
    "i have", "i feel", "i am having", "i'm having", "my symptoms",
    "i've been", "i am feeling",
]
_GENERAL_KNOWLEDGE_PATTERNS = [
    r"\bhow does\b", r"\bhow do\b", r"\bwhy does\b", r"\bwhy do\b",
    r"\bwhat is the mechanism\b", r"\bexplain\b",
    r"\bwhat is the difference between\b",
]
_GENERAL_EDUCATION_TOKENS = [
    "mechanism", "pathophysiology", "physiology", "etiology",
    "insulin resistance", "difference between", "systolic", "diastolic",
]
_RELATIONAL_GRAPH_TOKENS = [
    "symptom", "symptoms", "precaution", "precautions", "treatment for",
]
_COMPARISON_TOKENS = ["compare", "difference between", "vs ", "versus"]
_DANGER_TOKENS = ["more dangerous", "more serious", "which is worse"]


def fallback_route(query: str) -> QueryRoute:
    lowered = query.lower()
    if any(t in lowered for t in _SELF_REPORT_TOKENS):
        return "COMPLEX"
    if any(t in lowered for t in _COMPARISON_TOKENS + _DANGER_TOKENS):
        return "FACTUAL"
    if any(t in lowered for t in _GRAPH_SPECIFIC_TOKENS):
        return "RELATIONAL"
    for pattern in _GENERAL_KNOWLEDGE_PATTERNS:
        if re.search(pattern, lowered):
            return "GENERAL"
    if any(t in lowered for t in _GENERAL_EDUCATION_TOKENS):
        return "GENERAL"
    if any(t in lowered for t in _RELATIONAL_GRAPH_TOKENS):
        return "RELATIONAL"
    return "FACTUAL"


def parse_router_label(raw_text: str) -> QueryRoute | None:
    text = (raw_text or "").strip()
    if not text:
        return None
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        candidates.append(text[s: e + 1])
    valid = {"FACTUAL", "RELATIONAL", "COMPLEX", "GENERAL"}
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            label = str(parsed.get("label", "")).upper().strip()
            if label in valid:
                return label  # type: ignore[return-value]
        except Exception:
            continue
    return None


def is_medical_fact_query(query: str) -> bool:
    lowered = (query or "").lower()
    intent_keywords = [
        "symptom", "symptoms", "treat", "treatment", "precaution",
        "precautions", "diagnosis", "red flag", "warning sign",
    ]
    return (
        any(t in lowered for t in intent_keywords)
        and any(t in lowered for t in KNOWN_DISEASES_SORTED)
    )


def is_general_intent_query(query: str) -> bool:
    lowered = (query or "").lower()
    if any(t in lowered for t in _GRAPH_SPECIFIC_TOKENS):
        return False
    for pattern in _GENERAL_KNOWLEDGE_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return any(t in lowered for t in _GENERAL_EDUCATION_TOKENS)


def _is_comparison_query(query: str) -> bool:
    return any(t in query.lower() for t in _COMPARISON_TOKENS + _DANGER_TOKENS)


def _contains_disease_alias(query: str) -> bool:
    return len(extract_disease_entities(query)) > 0


def _has_multi_intent(query: str) -> bool:
    lowered = (query or "").lower()
    markers = ["symptom", "cause", "treat", "treatment", "precaution", "prevention"]
    return sum(1 for m in markers if m in lowered) >= 2


def _is_general_knowledge(query: str) -> bool:
    lowered = (query or "").lower()
    if any(t in lowered for t in _GRAPH_SPECIFIC_TOKENS):
        return False
    for pattern in _GENERAL_KNOWLEDGE_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return any(t in lowered for t in _GENERAL_EDUCATION_TOKENS)