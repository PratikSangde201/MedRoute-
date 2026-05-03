"""Structured medical facts service with query-aware response planning.

This module keeps structured medical facts as the first response path,
with deterministic handling for multi-intent and multi-disease queries.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import json
import re
from pathlib import Path
from threading import Lock
from typing import Any

_KNOWLEDGE_LOCK = Lock()
_STYLE_VARIATION_LOCK = Lock()
_MEDICAL_FACTS_CACHE: dict[str, dict[str, Any]] | None = None
_ALIAS_INDEX_CACHE: dict[str, str] | None = None
_PRIORITY_ALIAS_INDEX_CACHE: list[tuple[str, str]] | None = None
_STYLE_VARIATION_COUNTERS: dict[str, int] = {}

_RESPONSE_FORMATS = {
    "SINGLE_DISEASE_SUMMARY",
    "SINGLE_INTENT",
    "MULTI_INTENT_SINGLE_DISEASE",
    "DISEASE_SYMPTOM_LINK",
    "SYMPTOM_RELATION",
    "SYMPTOM_SEVERITY_COMPARISON",
    "CATEGORY_CLARIFICATION",
    "COMPARISON",
    "DANGER_RANKING",
    "DIFFERENTIAL_SYMPTOM_CHECK",
    "AMBIGUOUS",
    "RED_FLAG",
}

_STYLE_MODES = {
    "simple_explanation",
    "relation_answer",
    "severity_comparison_answer",
    "clarification_answer",
    "cautious_uncertainty_answer",
    "comparison_answer",
    "danger_ranking_answer",
    "multi_intent_answer",
    "symptom_check_answer",
    "urgent_safety_answer",
    "clarification_only",
}

_REASONING_RESPONSE_FORMATS = {
    "DISEASE_SYMPTOM_LINK",
    "SYMPTOM_RELATION",
    "SYMPTOM_SEVERITY_COMPARISON",
    "COMPARISON",
    "DANGER_RANKING",
    "DIFFERENTIAL_SYMPTOM_CHECK",
    "RED_FLAG",
    "CATEGORY_CLARIFICATION",
}

_STRUCTURED_SUMMARY_FORMATS = {
    "SINGLE_DISEASE_SUMMARY",
    "SINGLE_INTENT",
    "MULTI_INTENT_SINGLE_DISEASE",
}

_FORBIDDEN_TEMPLATE_MARKERS = {
    "structured data",
    "structured dataset",
    "dataset",
    "knowledge base",
    "documented links",
    "documented symptom links",
    "this text",
    "overlap not strong",
}

_REASONING_EXPLANATION_MARKERS = {
    "because",
    "since",
    "this happens",
    "this usually happens",
    "this can happen",
    "due to",
    "which can",
    "can lead to",
    "can reduce",
    "can cause",
    "can affect",
    "can explain",
    "this means",
    "this indicates",
    "intensity",
}

_EMERGENCY_MARKERS = {
    "chest pain",
    "shortness of breath",
    "left arm pain",
    "loss of consciousness",
    "fainting",
    "seizure",
    "severe bleeding",
    "suicidal",
    "self-harm",
    "persistent vomiting",
    "confusion",
}

_ALWAYS_EMERGENCY_MARKERS = {
    "suicidal",
    "self-harm",
    "loss of consciousness",
    "seizure",
    "severe bleeding",
}

_EMERGENCY_SEVERE_HINTS = {
    "severe",
    "worsening",
    "persistent",
    "cannot",
    "can't",
    "unable",
    "sudden",
    "worst",
    "for weeks",
}

# Priority aliases are matched first, in order.
_PRIORITY_ALIAS_CANDIDATES: list[tuple[str, str]] = [
    ("stomach flu", "Gastroenteritis"),
    ("high blood pressure", "Hypertension"),
    ("blood pressure high", "Hypertension"),
    ("high bp", "Hypertension"),
    ("bp high", "Hypertension"),
    ("high blood sugar", "Diabetes"),
    ("blood sugar high", "Diabetes"),
    ("high sugar level", "Diabetes"),
    ("sugar level high", "Diabetes"),
    ("sugar disease", "Diabetes"),
    ("sugar is high", "Diabetes"),
    ("my sugar is high", "Diabetes"),
    ("high sugar", "Diabetes"),
    ("type 2 diabetes", "Diabetes"),
    ("type ii diabetes", "Diabetes"),
    ("t2dm", "Diabetes"),
    ("tb", "Tuberculosis"),
    ("pulmonary tb", "Tuberculosis"),
    ("covid", "COVID-19"),
    ("corona", "COVID-19"),
    ("clinical depression", "Major Depressive Disorder"),
    ("depression", "Major Depressive Disorder"),
    ("anxiety", "Anxiety Disorder"),
    ("flu", "Influenza"),
]

# Prevent over-matching for generic short aliases.
_GENERIC_ALIAS_RESTRICTIONS: dict[str, set[str]] = {
    "flu": {
        "the",
        "a",
        "an",
        "of",
        "for",
        "with",
        "from",
        "about",
        "is",
        "are",
        "has",
        "have",
        "had",
        "and",
        "or",
        "vs",
        "versus",
        "compare",
        "difference",
        "between",
        "like",
        "after",
        "symptoms",
        "causes",
        "treatment",
        "precautions",
    }
}

_RELAXED_ALIAS_CONTEXT_HINTS = {
    "symptom",
    "symptoms",
    "cause",
    "causes",
    "treat",
    "treatment",
    "precaution",
    "precautions",
    "prevent",
    "prevention",
    "infection",
    "disease",
    "have",
    "having",
    "diagnosed",
}

_GENERIC_ALIAS_ALLOWED_PREFIXES: dict[str, set[str]] = {
    "flu": {"seasonal", "common", "swine", "avian", "bird"},
}

_FUZZY_MATCH_THRESHOLD = 0.74
_MIN_FUZZY_SEGMENT_LEN = 4

_INTENT_ORDER = ["symptoms", "causes", "treatment", "precautions", "red_flags", "summary"]

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "symptoms": ["symptom", "symptoms", "sign", "signs"],
    "causes": ["cause", "causes", "why", "reason", "reasons"],
    "treatment": [
        "treat",
        "treatment",
        "therapy",
        "medicine",
        "medication",
        "cure",
        "management",
        "what should i do",
        "what to do",
    ],
    "precautions": [
        "precaution",
        "precautions",
        "prevent",
        "prevention",
        "avoid",
        "careful",
        "care",
    ],
    "red_flags": [
        "red flag",
        "warning sign",
        "danger sign",
        "dangerous",
        "worse",
        "emergency",
        "urgent",
        "serious",
        "is it serious",
        "should i worry",
        "emergency room",
    ],
    "summary": ["what is", "tell me about", "about", "overview", "explain"],
}

_COMPARISON_MARKERS = ["compare", "difference between", "difference", "vs", "versus"]
_DIFFERENTIAL_MARKERS = ["is it", "could it be", "or", "what could", "what might"]
_DANGER_RANKING_MARKERS = [
    "more dangerous",
    "most dangerous",
    "worse than",
    "which is worse",
    "which is more dangerous",
    "more serious",
    "greater risk",
]
_RELATION_QUERY_MARKERS = [
    "affect",
    "affects",
    "impact",
    "impacts",
    "reduce",
    "reduces",
    "lower",
    "lowers",
    "drop",
    "drops",
    "decrease",
    "decreases",
    "related",
    "relation",
    "linked",
    "link",
    "cause",
    "causes",
    "lead to",
    "leads to",
    "result in",
    "results in",
    "associated with",
]
_SEVERITY_COMPARISON_CUES = [
    "different",
    "difference",
    "differentiate",
    "same",
    "compare",
    "vs",
    "versus",
    "more serious",
    "more severe",
    "worse",
    "higher",
]
_SEVERITY_QUALIFIERS = ["mild", "moderate", "high", "very high", "severe", "low grade", "low-grade"]
_INTERPRETATION_MARKERS = ["what does", "what do", "mean", "means"]
_COEXISTENCE_MARKERS = ["together", "same time", "coexist", "both"]
_CONTEXT_FOLLOWUP_MARKERS = ["what about", "and treatment", "and symptoms", "and causes", "and precautions"]
_SELF_REPORT_MARKERS = ["i have", "i am having", "i'm having", "my symptoms", "i feel", "i am feeling", "i've been"]

_META_LANGUAGE_MARKERS = [
    "structured data",
    "structured dataset",
    "knowledge base",
    "current dataset",
    "documented symptom links",
    "related details are present",
    "not strong from this text alone",
    "retrieval",
    "graph source",
    "fallback source",
    "confidence score",
]

_DANGER_SIGNAL_WEIGHTS = {
    "breathing difficulty": 3.0,
    "shortness of breath": 3.0,
    "severe breathlessness": 3.0,
    "low oxygen": 3.0,
    "blue lips": 3.0,
    "persistent chest pain": 3.0,
    "severe chest pain": 3.0,
    "bleeding": 3.0,
    "coughing blood": 3.0,
    "seizure": 3.0,
    "loss of consciousness": 3.0,
    "confusion": 2.5,
    "fainting": 2.5,
    "persistent vomiting": 2.0,
    "severe dehydration": 2.0,
    "rapid breathing": 2.0,
    "high fever": 1.5,
}

_INTENT_LABELS = {
    "symptoms": "Symptoms",
    "causes": "Causes",
    "treatment": "Treatment",
    "precautions": "Prevention and precautions",
    "red_flags": "Warning signs",
    "summary": "Overview",
}

_SYMPTOM_HINTS = {
    "body pain": "body ache",
    "breathlessness": "shortness of breath",
    "breathless": "shortness of breath",
    "breathing issue": "shortness of breath",
    "breathing issues": "shortness of breath",
    "trouble breathing": "shortness of breath",
    "cough for weeks": "chronic cough",
    "pain behind eyes": "pain behind the eyes",
    "eye pain": "pain behind the eyes",
    "pain in eyes": "pain behind the eyes",
    "weight loss": "weight loss",
    "weakness": "fatigue",
    "tired": "fatigue",
    "blurred eyesight": "blurred vision",
}

_CONCEPT_HINTS = {
    "high sugar": "high blood sugar",
    "high blood sugar": "high blood sugar",
    "oxygen": "low oxygen",
    "reduced oxygen": "low oxygen",
    "reduce oxygen": "low oxygen",
    "low oxygen": "low oxygen",
    "oxygen drop": "low oxygen",
    "oxygen drops": "low oxygen",
    "oxygen level": "low oxygen",
    "oxygen levels": "low oxygen",
    "eyesight": "blurred vision",
    "eye sight": "blurred vision",
    "vision": "blurred vision",
    "vision issue": "blurred vision",
    "vision issues": "blurred vision",
    "eye problem": "blurred vision",
    "eye problems": "blurred vision",
    "blurry vision": "blurred vision",
    "eye pain": "pain behind the eyes",
    "pain in eyes": "pain behind the eyes",
}

_RELATION_EXPANSION_HINTS: dict[str, list[str]] = {
    "vision": ["blurred vision", "light sensitivity", "vision loss"],
    "oxygen": ["low oxygen", "shortness of breath", "breathing difficulty"],
    "eye": ["pain behind the eyes", "blurred vision", "light sensitivity"],
    "breathing": ["shortness of breath", "breathing difficulty", "low oxygen"],
    "fatigue": ["fatigue", "weakness", "dizziness"],
    "weak": ["fatigue", "severe weakness", "dizziness"],
}

_RELATION_MECHANISM_HINTS: dict[str, dict[str, str]] = {
    "diabetes": {
        "blurred vision": "This usually happens because high blood sugar can change fluid balance in the eye lens and temporarily blur vision.",
        "high blood sugar": "This usually happens because diabetes involves insulin deficiency or resistance, so glucose stays higher in the blood.",
    },
    "pneumonia": {
        "low oxygen": "This usually happens because infection in the lungs affects gas exchange, so oxygen transfer into blood can fall.",
        "shortness of breath": "This usually happens because inflamed air sacs make breathing less efficient.",
    },
    "dengue": {
        "pain behind the eyes": "This usually happens because dengue triggers strong inflammatory responses that can cause pain behind the eyes.",
    },
    "anemia": {
        "fatigue": "This usually happens because low hemoglobin reduces oxygen delivery to tissues, leading to fatigue and weakness.",
        "weakness": "This usually happens because lower oxygen delivery from anemia can make muscles feel weak.",
    },
    "asthma": {
        "shortness of breath": "This usually happens because airway inflammation and narrowing make airflow more limited.",
        "breathing difficulty": "This usually happens because inflamed airways can narrow and increase the effort needed to breathe.",
    },
    "migraine": {
        "blurred vision": "This can happen because migraine can involve temporary visual pathway changes and sensory sensitivity.",
        "light sensitivity": "This can happen because migraine often heightens sensory processing, including light sensitivity.",
    },
}


def _data_file_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "medical_facts.json"


def normalize_text(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = re.sub(r"[^a-z0-9\s\-]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _alias_variants(alias_norm: str) -> set[str]:
    """Generate safe lexical variants for alias matching coverage."""
    variants: set[str] = {alias_norm}
    words = alias_norm.split()

    if len(words) == 1 and len(alias_norm) >= 5:
        if alias_norm.endswith("y") and len(alias_norm) > 5:
            variants.add(alias_norm[:-1] + "ies")
        elif alias_norm.endswith(("s", "x", "z", "ch", "sh")):
            variants.add(alias_norm + "es")
        else:
            variants.add(alias_norm + "s")

        # Also allow singular backoff for plural aliases stored in data.
        if alias_norm.endswith("ies") and len(alias_norm) > 6:
            variants.add(alias_norm[:-3] + "y")
        elif alias_norm.endswith("es") and len(alias_norm) > 6:
            variants.add(alias_norm[:-2])
        elif alias_norm.endswith("s") and len(alias_norm) > 5:
            variants.add(alias_norm[:-1])

    # Common diabetes phrasing variants.
    if "type 2" in alias_norm:
        variants.add(alias_norm.replace("type 2", "type ii"))
    if "type ii" in alias_norm:
        variants.add(alias_norm.replace("type ii", "type 2"))

    return {variant for variant in variants if variant}


def _style_cycle(seed: str, size: int) -> int:
    if size <= 0:
        return 0
    with _STYLE_VARIATION_LOCK:
        count = _STYLE_VARIATION_COUNTERS.get(seed, 0)
        _STYLE_VARIATION_COUNTERS[seed] = count + 1
    return (abs(hash(seed)) + count) % size


def _pick_variant(seed: str, options: list[str]) -> str:
    if not options:
        return ""
    index = _style_cycle(seed, len(options))
    return options[index]


def load_medical_facts(force_reload: bool = False) -> dict[str, dict[str, Any]]:
    global _MEDICAL_FACTS_CACHE, _ALIAS_INDEX_CACHE, _PRIORITY_ALIAS_INDEX_CACHE

    with _KNOWLEDGE_LOCK:
        if _MEDICAL_FACTS_CACHE is not None and not force_reload:
            return _MEDICAL_FACTS_CACHE

        path = _data_file_path()
        if not path.exists():
            _MEDICAL_FACTS_CACHE = {}
            _ALIAS_INDEX_CACHE = {}
            _PRIORITY_ALIAS_INDEX_CACHE = []
            return _MEDICAL_FACTS_CACHE

        raw = json.loads(path.read_text(encoding="utf-8"))
        facts = raw if isinstance(raw, dict) else {}

        normalized_facts: dict[str, dict[str, Any]] = {}
        alias_index: dict[str, str] = {}

        for disease_key, record in facts.items():
            if not isinstance(record, dict):
                continue

            canonical_name = str(record.get("canonical_name") or disease_key).strip()
            if not canonical_name:
                continue

            normalized_key = normalize_text(canonical_name)
            aliases = [str(item).strip() for item in (record.get("aliases") or []) if str(item).strip()]

            normalized_record = {
                "canonical_name": canonical_name,
                "aliases": aliases,
                "symptoms": [str(item).strip() for item in (record.get("symptoms") or []) if str(item).strip()],
                "causes": [str(item).strip() for item in (record.get("causes") or []) if str(item).strip()],
                "treatments": [str(item).strip() for item in (record.get("treatments") or []) if str(item).strip()],
                "precautions": [str(item).strip() for item in (record.get("precautions") or []) if str(item).strip()],
                "red_flags": [str(item).strip() for item in (record.get("red_flags") or []) if str(item).strip()],
                "source_name": str(record.get("source_name") or "").strip(),
                "source_url": str(record.get("source_url") or "").strip(),
            }

            normalized_facts[normalized_key] = normalized_record

            def register_alias(raw_alias: str) -> None:
                alias_norm = normalize_text(raw_alias)
                if not alias_norm:
                    return
                for variant in _alias_variants(alias_norm):
                    # Keep first-seen mapping for deterministic behavior.
                    alias_index.setdefault(variant, normalized_key)

            register_alias(canonical_name)

            for alias in aliases:
                register_alias(alias)

        priority_alias_index: list[tuple[str, str]] = []
        for alias, canonical_name in _PRIORITY_ALIAS_CANDIDATES:
            alias_norm = normalize_text(alias)
            canonical_norm = normalize_text(canonical_name)
            if alias_norm and canonical_norm in normalized_facts:
                priority_alias_index.append((alias_norm, canonical_norm))

        _MEDICAL_FACTS_CACHE = normalized_facts
        _ALIAS_INDEX_CACHE = alias_index
        _PRIORITY_ALIAS_INDEX_CACHE = priority_alias_index
        return _MEDICAL_FACTS_CACHE


def _alias_index() -> dict[str, str]:
    if _ALIAS_INDEX_CACHE is None:
        load_medical_facts()
    return _ALIAS_INDEX_CACHE or {}


def _priority_alias_index() -> list[tuple[str, str]]:
    if _PRIORITY_ALIAS_INDEX_CACHE is None:
        load_medical_facts()
    return _PRIORITY_ALIAS_INDEX_CACHE or []


def _facts_records() -> dict[str, dict[str, Any]]:
    return load_medical_facts()


def _all_canonical_names() -> list[str]:
    records = _facts_records()
    return [str(record.get("canonical_name") or "").strip() for record in records.values() if record.get("canonical_name")]


def detect_query_intents(query: str) -> list[str]:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return []

    intents: list[str] = []

    def add_intent(intent: str) -> None:
        if intent not in intents:
            intents.append(intent)

    for intent in _INTENT_ORDER:
        keywords = _INTENT_KEYWORDS[intent]
        if any(keyword in normalized_query for keyword in keywords):
            add_intent(intent)

    # Map prevention phrasing explicitly to precautions.
    if "prevention" in normalized_query or "prevent" in normalized_query:
        add_intent("precautions")

    if not intents:
        if any(marker in normalized_query for marker in _INTENT_KEYWORDS["summary"]):
            intents.append("summary")

    return intents


def detect_query_intent(query: str) -> str:
    intents = detect_query_intents(query)
    if not intents:
        return "unknown"
    return intents[0]


def _query_segments(normalized_query: str) -> list[str]:
    tokens = normalized_query.split()
    if not tokens:
        return []

    segments: set[str] = set(tokens)
    max_window = min(4, len(tokens))
    for size in range(2, max_window + 1):
        for start in range(0, len(tokens) - size + 1):
            segments.add(" ".join(tokens[start : start + size]))

    return sorted(segments, key=len, reverse=True)


def _best_fuzzy_match(normalized_query: str) -> tuple[str | None, float]:
    matches = _best_fuzzy_matches(normalized_query, max_results=1)
    if matches:
        return matches[0]
    return None, 0.0


def _best_fuzzy_matches(
    normalized_query: str,
    exclude_keys: set[str] | None = None,
    max_results: int = 1,
) -> list[tuple[str, float]]:
    index = _alias_index()
    if not index or not normalized_query or max_results <= 0:
        return []

    segments = _query_segments(normalized_query)
    if not segments:
        return []

    excluded = exclude_keys or set()
    best_scores: dict[str, float] = {}
    candidate_floor = max(0.68, _FUZZY_MATCH_THRESHOLD - 0.08)

    for segment in segments:
        if len(segment) < _MIN_FUZZY_SEGMENT_LEN:
            continue
        for alias, disease_key in index.items():
            if disease_key in excluded:
                continue
            if len(alias) < _MIN_FUZZY_SEGMENT_LEN:
                continue
            if abs(len(segment) - len(alias)) > max(3, int(max(len(segment), len(alias)) * 0.5)):
                continue

            score = SequenceMatcher(a=segment, b=alias).ratio()
            if score < candidate_floor:
                continue

            prev = best_scores.get(disease_key, 0.0)
            if score > prev:
                best_scores[disease_key] = score

    ranked = [
        (disease_key, score)
        for disease_key, score in sorted(best_scores.items(), key=lambda item: item[1], reverse=True)
        if score >= _FUZZY_MATCH_THRESHOLD
    ]

    return ranked[:max_results]


def _word_before_index(text: str, index: int) -> str:
    if index <= 0:
        return ""
    left = text[:index].strip()
    if not left:
        return ""
    return left.split()[-1]


def _word_after_index(text: str, index: int) -> str:
    if index >= len(text):
        return ""
    right = text[index:].strip()
    if not right:
        return ""
    return right.split()[0]


def _is_safe_alias_usage(alias_norm: str, normalized_query: str, start: int) -> bool:
    restrictions = _GENERIC_ALIAS_RESTRICTIONS.get(alias_norm)
    if not restrictions:
        return True

    prev_word = _word_before_index(normalized_query, start)
    if prev_word and prev_word not in restrictions:
        allowed_prefixes = _GENERIC_ALIAS_ALLOWED_PREFIXES.get(alias_norm, set())
        if prev_word in allowed_prefixes:
            return True

        # Block adjective-modified generic aliases such as "alien flu".
        return False

    return True


def _span_overlaps(span: tuple[int, int], used_spans: list[tuple[int, int]]) -> bool:
    start, end = span
    for used_start, used_end in used_spans:
        if not (end <= used_start or start >= used_end):
            return True
    return False


def _extract_exact_disease_matches_from_normalized(normalized_query: str) -> list[dict[str, Any]]:
    facts = _facts_records()
    index = _alias_index()

    candidates: list[dict[str, Any]] = []
    used_spans: list[tuple[int, int]] = []

    # 1) Priority aliases first.
    for alias, disease_key in _priority_alias_index():
        pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        for match in re.finditer(pattern, normalized_query):
            span = (match.start(), match.end())
            if _span_overlaps(span, used_spans):
                continue
            if not _is_safe_alias_usage(alias, normalized_query, match.start()):
                continue
            record = facts.get(disease_key)
            if not record:
                continue
            candidates.append(
                {
                    "key": disease_key,
                    "record": record,
                    "match_type": "priority_alias",
                    "confidence": 0.98,
                    "start": match.start(),
                }
            )
            used_spans.append(span)

    # 2) Exact aliases/canonical names.
    for alias in sorted(index.keys(), key=len, reverse=True):
        pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        for match in re.finditer(pattern, normalized_query):
            span = (match.start(), match.end())
            if _span_overlaps(span, used_spans):
                continue
            if not _is_safe_alias_usage(alias, normalized_query, match.start()):
                continue

            disease_key = index[alias]
            record = facts.get(disease_key)
            if not record:
                continue

            candidates.append(
                {
                    "key": disease_key,
                    "record": record,
                    "match_type": "alias_exact",
                    "confidence": 0.95,
                    "start": match.start(),
                }
            )
            used_spans.append(span)

    # Deduplicate by disease, preserving earliest occurrence.
    deduped: dict[str, dict[str, Any]] = {}
    for item in sorted(candidates, key=lambda x: (x["start"], -float(x["confidence"]))):
        key = str(item["key"])
        if key not in deduped:
            deduped[key] = item

    return list(sorted(deduped.values(), key=lambda x: x["start"]))


def _extract_disease_matches(query: str) -> list[dict[str, Any]]:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return []

    matches = _extract_exact_disease_matches_from_normalized(normalized_query)
    matched_keys = {str(item.get("key") or "") for item in matches if item.get("key")}

    comparison_like = any(marker in normalized_query for marker in _COMPARISON_MARKERS)
    desired_match_count = 2 if comparison_like else 1

    remaining_slots = max(0, desired_match_count - len(matches))
    if not matches and remaining_slots == 0:
        remaining_slots = 1

    if remaining_slots > 0:
        fuzzy_candidates = _best_fuzzy_matches(
            normalized_query,
            exclude_keys=matched_keys,
            max_results=remaining_slots,
        )
        for fuzzy_key, fuzzy_score in fuzzy_candidates:
            record = _facts_records().get(fuzzy_key)
            if not record:
                continue
            matches.append(
                {
                    "key": fuzzy_key,
                    "record": record,
                    "match_type": "fuzzy",
                    "confidence": fuzzy_score,
                    # Keep exact spans ahead of fuzzy additions when ordering output matches.
                    "start": len(normalized_query) + len(matches),
                }
            )
            matched_keys.add(fuzzy_key)

    if matches:
        return list(sorted(matches, key=lambda item: int(item.get("start") or 0)))

    return []


def _find_disease_match_with_metadata(query: str) -> tuple[str | None, dict[str, Any] | None, str, float]:
    matches = _extract_disease_matches(query)
    if not matches:
        return None, None, "none", 0.0

    first = matches[0]
    return (
        str(first["key"]),
        first.get("record"),
        str(first.get("match_type") or "alias_exact"),
        float(first.get("confidence") or 0.0),
    )


def find_disease_match(query: str) -> tuple[str | None, dict[str, Any] | None]:
    disease_key, record, _, _ = _find_disease_match_with_metadata(query)
    return disease_key, record


def _format_list(items: list[str], empty_message: str) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return empty_message
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _format_bullets(items: list[str], max_items: int = 5) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()][:max_items]
    if not cleaned:
        return "- Detailed information is not available right now."
    return "\n".join(f"- {item}" for item in cleaned)


def _contains_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
    return bool(re.search(pattern, text))


def _contains_meta_language(answer: str) -> bool:
    normalized = normalize_text(answer)
    if not normalized:
        return False
    return any(marker in normalized for marker in _META_LANGUAGE_MARKERS)


def _sanitize_user_facing_answer(answer: str) -> str:
    if not answer:
        return ""

    replacements = {
        "from the structured data": "from the available clinical information",
        "the structured dataset": "the available information",
        "in this dataset": "in the available information",
        "current dataset": "available information",
        "current knowledge base": "available information",
        "knowledge base": "available information",
        "documented symptom links": "common symptom patterns",
        "related details are present": "some overlap may be present",
        "not strong from this text alone": "not clear from this information alone",
        "structured data": "available information",
        "retrieval": "analysis",
        "graph source": "source",
        "fallback source": "source",
        "confidence score": "confidence",
    }

    sanitized = answer
    for old, new in replacements.items():
        sanitized = re.sub(re.escape(old), new, sanitized, flags=re.IGNORECASE)

    sanitized = sanitized.replace("\r\n", "\n")
    sanitized = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in sanitized.split("\n"))
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()
    return sanitized


def _intent_items(record: dict[str, Any], intent: str) -> list[str]:
    if intent == "symptoms":
        return list(record.get("symptoms") or [])
    if intent == "causes":
        return list(record.get("causes") or [])
    if intent == "treatment":
        return list(record.get("treatments") or [])
    if intent == "precautions":
        return list(record.get("precautions") or [])
    if intent == "red_flags":
        return list(record.get("red_flags") or [])
    return []


def _summary_line(record: dict[str, Any]) -> str:
    pieces: list[str] = []
    symptoms = _intent_items(record, "symptoms")
    causes = _intent_items(record, "causes")
    treatments = _intent_items(record, "treatment")

    if symptoms:
        pieces.append(f"symptoms like {_format_list(symptoms[:3], 'not listed')}")
    if causes:
        pieces.append(f"causes such as {_format_list(causes[:2], 'not listed')}")
    if treatments:
        pieces.append(f"management including {_format_list(treatments[:2], 'not listed')}")

    if not pieces:
        return "Detailed summary information is limited right now."

    return "; ".join(pieces)


def _contains_emergency_marker(query: str, intent: str) -> bool:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return False

    for marker in _ALWAYS_EMERGENCY_MARKERS:
        if marker in normalized_query:
            return True

    has_marker = any(marker in normalized_query for marker in _EMERGENCY_MARKERS)
    if not has_marker:
        return False

    if intent == "red_flags":
        return True

    return any(hint in normalized_query for hint in _EMERGENCY_SEVERE_HINTS)


def _is_self_report_query(query: str) -> bool:
    normalized = normalize_text(query)
    return any(marker in normalized for marker in _SELF_REPORT_MARKERS)


def _is_comparison_query(normalized_query: str) -> bool:
    return any(marker in normalized_query for marker in _COMPARISON_MARKERS)


def _is_danger_ranking_query(normalized_query: str) -> bool:
    return any(marker in normalized_query for marker in _DANGER_RANKING_MARKERS)


def _is_differential_query(normalized_query: str) -> bool:
    return any(marker in normalized_query for marker in _DIFFERENTIAL_MARKERS)


def _is_coexistence_query(normalized_query: str) -> bool:
    return any(marker in normalized_query for marker in _COEXISTENCE_MARKERS)


def _is_contextual_followup(query: str) -> bool:
    normalized = normalize_text(query)
    if not normalized:
        return False

    if any(marker in normalized for marker in _CONTEXT_FOLLOWUP_MARKERS):
        return True

    words = normalized.split()
    if len(words) <= 4 and detect_query_intents(query):
        return True

    return False


def _is_red_flag_query(normalized_query: str, intents: list[str]) -> bool:
    if "red_flags" in intents:
        return True

    if "should i worry" in normalized_query or "is it serious" in normalized_query:
        return True

    if any(hint in normalized_query for hint in _EMERGENCY_SEVERE_HINTS) and any(
        marker in normalized_query for marker in _EMERGENCY_MARKERS
    ):
        return True

    return False


def _is_interpretation_query(normalized_query: str) -> bool:
    if not normalized_query:
        return False
    has_marker = any(marker in normalized_query for marker in _INTERPRETATION_MARKERS)
    return has_marker and ("what" in normalized_query or "mean" in normalized_query)


def _is_basic_summary_query(normalized_query: str) -> bool:
    if not normalized_query:
        return False

    summary_patterns = [
        r"^what is [a-z0-9\-\s]+$",
        r"^give overview of [a-z0-9\-\s]+$",
        r"^overview of [a-z0-9\-\s]+$",
        r"^tell me about [a-z0-9\-\s]+$",
        r"^explain [a-z0-9\-\s]+$",
    ]
    return any(re.match(pattern, normalized_query) for pattern in summary_patterns)


def _is_why_how_reasoning_query(normalized_query: str) -> bool:
    if not normalized_query:
        return False

    if "why" in normalized_query:
        return True

    reasoning_how_patterns = [
        "how does",
        "how do",
        "how can",
        "how is",
        "how are",
        "how could",
        "how might",
    ]

    if any(pattern in normalized_query for pattern in reasoning_how_patterns):
        return True

    non_reasoning_how_patterns = ["how many", "how much", "how long", "how often"]
    if any(pattern in normalized_query for pattern in non_reasoning_how_patterns):
        return False

    return normalized_query.startswith("how ")


def _is_real_life_symptom_query(
    normalized_query: str,
    self_report: bool,
    symptom_terms: list[str],
    differential_query: bool,
) -> bool:
    if not normalized_query:
        return False

    if self_report and symptom_terms:
        return True

    if differential_query and (self_report or len(symptom_terms) >= 2):
        return True

    return False


def _is_category_confusion_query(
    comparison_query: bool,
    disease_count: int,
    symptom_terms: list[str],
    severity_comparison: dict[str, str] | None,
) -> bool:
    if not comparison_query:
        return False
    if disease_count != 1:
        return False
    if severity_comparison:
        return False
    return bool(symptom_terms)


def _style_mode_for_response_format(response_format: str) -> str:
    mapping = {
        "SINGLE_DISEASE_SUMMARY": "simple_explanation",
        "SINGLE_INTENT": "simple_explanation",
        "MULTI_INTENT_SINGLE_DISEASE": "multi_intent_answer",
        "DISEASE_SYMPTOM_LINK": "relation_answer",
        "SYMPTOM_RELATION": "relation_answer",
        "SYMPTOM_SEVERITY_COMPARISON": "severity_comparison_answer",
        "CATEGORY_CLARIFICATION": "clarification_answer",
        "COMPARISON": "comparison_answer",
        "DANGER_RANKING": "danger_ranking_answer",
        "DIFFERENTIAL_SYMPTOM_CHECK": "symptom_check_answer",
        "RED_FLAG": "urgent_safety_answer",
        "AMBIGUOUS": "clarification_only",
    }
    mode = mapping.get(response_format, "clarification_only")
    return mode if mode in _STYLE_MODES else "clarification_only"


def _reasoning_mode_for_response_format(response_format: str, intents: list[str]) -> str:
    if response_format == "SINGLE_DISEASE_SUMMARY":
        return "disease_summary"
    if response_format == "SINGLE_INTENT":
        return "disease_symptom_link" if intents[:1] == ["symptoms"] else "disease_summary"
    if response_format == "MULTI_INTENT_SINGLE_DISEASE":
        return "multi_intent"
    if response_format == "DISEASE_SYMPTOM_LINK":
        return "disease_symptom_link"
    if response_format == "SYMPTOM_RELATION":
        return "symptom_relation"
    if response_format == "SYMPTOM_SEVERITY_COMPARISON":
        return "symptom_severity_comparison"
    if response_format == "CATEGORY_CLARIFICATION":
        return "clarification_needed"
    if response_format in {"COMPARISON", "DANGER_RANKING"}:
        return "disease_vs_disease_comparison"
    if response_format == "RED_FLAG":
        return "emergency"
    if response_format == "DIFFERENTIAL_SYMPTOM_CHECK":
        return "clarification_needed"
    return "ambiguous"


def _is_reasoning_response_format(response_format: str) -> bool:
    return response_format in _REASONING_RESPONSE_FORMATS


def _force_reasoning_response_format(plan: dict[str, Any]) -> str:
    disease_matches = list(plan.get("disease_matches") or [])
    disease_count = len(disease_matches)
    symptom_terms = list(plan.get("symptom_terms") or [])
    intents = {str(intent) for intent in (plan.get("intents") or [])}
    procedural_intent_query = bool(intents.intersection({"treatment", "precautions"}))

    if bool(plan.get("danger_ranking_query")) and disease_count >= 2:
        return "DANGER_RANKING"

    if plan.get("severity_comparison"):
        return "SYMPTOM_SEVERITY_COMPARISON"

    if disease_count >= 2 and (bool(plan.get("comparison_query")) or bool(plan.get("coexistence_query"))):
        return "COMPARISON"

    if disease_count >= 1 and (
        bool(plan.get("disease_symptom_link_query"))
        or bool(plan.get("relation_query"))
        or bool(plan.get("why_how_query"))
        or bool(plan.get("concept_terms"))
        or bool(plan.get("interpretation_query"))
    ) and not procedural_intent_query:
        return "DISEASE_SYMPTOM_LINK"

    if bool(plan.get("self_report")) or bool(plan.get("differential_query")):
        return "DIFFERENTIAL_SYMPTOM_CHECK"

    if bool(plan.get("symptom_relation_query")) or len(symptom_terms) >= 2:
        return "SYMPTOM_RELATION"

    return "DIFFERENTIAL_SYMPTOM_CHECK" if symptom_terms else "AMBIGUOUS"


def _resolve_history_disease_match(chat_history: str | None) -> dict[str, Any] | None:
    if not chat_history:
        return None

    recent_history = normalize_text(chat_history[-1200:])
    if not recent_history:
        return None

    matches = _extract_exact_disease_matches_from_normalized(recent_history)
    if not matches:
        return None

    last = sorted(matches, key=lambda item: int(item["start"]))[-1]
    last_copy = dict(last)
    last_copy["match_type"] = "history_context"
    last_copy["confidence"] = 0.72
    return last_copy


def _symptom_vocabulary() -> list[str]:
    records = _facts_records()
    vocab: set[str] = set()
    for record in records.values():
        for symptom in record.get("symptoms") or []:
            symptom_norm = normalize_text(symptom)
            if symptom_norm:
                vocab.add(symptom_norm)

    for hinted in _SYMPTOM_HINTS.values():
        hinted_norm = normalize_text(hinted)
        if hinted_norm:
            vocab.add(hinted_norm)

    for hinted in _CONCEPT_HINTS.values():
        hinted_norm = normalize_text(hinted)
        if hinted_norm:
            vocab.add(hinted_norm)

    return sorted(vocab, key=len, reverse=True)


def _extract_concept_terms_from_query(query: str) -> list[str]:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return []

    found: list[str] = []

    for hint, mapped in _SYMPTOM_HINTS.items():
        if _contains_phrase(normalized_query, normalize_text(hint)):
            found.append(mapped)

    for hint, mapped in _CONCEPT_HINTS.items():
        if _contains_phrase(normalized_query, normalize_text(hint)):
            found.append(mapped)

    for symptom in _symptom_vocabulary():
        if len(symptom) < 4:
            continue
        if _contains_phrase(normalized_query, symptom):
            found.append(symptom)

    deduped: list[str] = []
    seen: set[str] = set()
    for symptom in found:
        symptom_norm = normalize_text(symptom)
        if symptom_norm and symptom_norm not in seen:
            seen.add(symptom_norm)
            deduped.append(symptom)

    return deduped


def _extract_symptoms_from_query(query: str) -> list[str]:
    return _extract_concept_terms_from_query(query)


def _is_relation_query(normalized_query: str) -> bool:
    return any(marker in normalized_query for marker in _RELATION_QUERY_MARKERS)


def _detect_symptom_severity_comparison(
    normalized_query: str,
    concept_terms: list[str],
    extra_terms: list[str] | None = None,
) -> dict[str, str] | None:
    if not normalized_query:
        return None

    if not any(cue in normalized_query for cue in _SEVERITY_COMPARISON_CUES):
        return None

    candidate_terms = {normalize_text(term) for term in concept_terms if normalize_text(term)}
    for term in extra_terms or []:
        term_norm = normalize_text(term)
        if term_norm:
            candidate_terms.add(term_norm)

    for symptom in _symptom_vocabulary():
        if _contains_phrase(normalized_query, symptom):
            candidate_terms.add(symptom)

    for symptom in sorted(candidate_terms, key=len, reverse=True):
        if len(symptom) < 3:
            continue
        for qualifier in _SEVERITY_QUALIFIERS:
            qualified = f"{qualifier} {symptom}"
            if _contains_phrase(normalized_query, qualified) and _contains_phrase(normalized_query, symptom):
                return {
                    "base_symptom": symptom,
                    "qualified_symptom": qualified,
                    "qualifier": qualifier,
                }

    return None


def _symptom_matches_record(symptom: str, record_symptom: str) -> bool:
    a = normalize_text(symptom)
    b = normalize_text(record_symptom)
    if not a or not b:
        return False

    if a == b or a in b or b in a:
        return True

    return SequenceMatcher(a=a, b=b).ratio() >= 0.78


def _rank_diseases_by_symptoms(symptoms: list[str], disease_keys: list[str] | None = None) -> list[dict[str, Any]]:
    if not symptoms:
        return []

    records = _facts_records()
    keys = disease_keys or list(records.keys())
    ranked: list[dict[str, Any]] = []

    for key in keys:
        record = records.get(key)
        if not record:
            continue

        record_symptoms = [str(item).strip() for item in (record.get("symptoms") or []) if str(item).strip()]
        if not record_symptoms:
            continue

        matched_symptoms: list[str] = []
        for symptom in symptoms:
            if any(_symptom_matches_record(symptom, record_symptom) for record_symptom in record_symptoms):
                matched_symptoms.append(symptom)

        if not matched_symptoms:
            continue

        score = len(matched_symptoms)
        coverage = score / max(1, len(symptoms))
        ranked.append(
            {
                "key": key,
                "record": record,
                "matched_symptoms": matched_symptoms,
                "score": score,
                "coverage": coverage,
            }
        )

    ranked.sort(
        key=lambda item: (
            -int(item["score"]),
            -float(item["coverage"]),
            str((item.get("record") or {}).get("canonical_name") or ""),
        )
    )

    return ranked


def _analyze_query(query: str, chat_history: str | None = None) -> dict[str, Any]:
    normalized_query = normalize_text(query)
    summary_query = _is_basic_summary_query(normalized_query)
    why_how_query = _is_why_how_reasoning_query(normalized_query)
    intents = detect_query_intents(query)
    if not intents:
        intents = ["summary"]
    elif summary_query and "summary" not in intents:
        intents.append("summary")

    disease_matches = _extract_disease_matches(query)
    used_history_context = False

    if not disease_matches and _is_contextual_followup(query):
        history_match = _resolve_history_disease_match(chat_history)
        if history_match:
            disease_matches = [history_match]
            used_history_context = True

    disease_count = len(disease_matches)
    self_report = _is_self_report_query(query)
    comparison_query = _is_comparison_query(normalized_query)
    danger_ranking_query = _is_danger_ranking_query(normalized_query)
    differential_query = _is_differential_query(normalized_query)
    coexistence_query = _is_coexistence_query(normalized_query)
    red_flag_query = _is_red_flag_query(normalized_query, intents)
    interpretation_query = _is_interpretation_query(normalized_query)
    concept_terms = _extract_concept_terms_from_query(query)
    symptom_terms = list(concept_terms)
    relation_query = _is_relation_query(normalized_query)
    disease_terms = [
        str((match.get("record") or {}).get("canonical_name") or "").strip()
        for match in disease_matches
        if (match.get("record") or {}).get("canonical_name")
    ]
    severity_comparison = _detect_symptom_severity_comparison(
        normalized_query,
        concept_terms,
        extra_terms=disease_terms,
    )
    disease_comparison_query = disease_count >= 2 and (comparison_query or danger_ranking_query or coexistence_query)
    real_life_symptom_query = _is_real_life_symptom_query(
        normalized_query=normalized_query,
        self_report=self_report,
        symptom_terms=symptom_terms,
        differential_query=differential_query,
    )
    procedural_intent_query = any(intent in {"treatment", "precautions"} for intent in intents)

    disease_symptom_link_query = disease_count == 1 and bool(concept_terms) and (
        relation_query
        or "symptoms" in intents
        or "causes" in intents
        or interpretation_query
        or why_how_query
    )
    if disease_count == 1 and interpretation_query:
        disease_symptom_link_query = True

    if disease_count == 1 and why_how_query and (relation_query or concept_terms):
        disease_symptom_link_query = True

    symptom_relation_query = disease_count == 0 and (
        (relation_query and len(concept_terms) >= 2)
        or (why_how_query and len(concept_terms) >= 2)
    )
    category_confusion_query = _is_category_confusion_query(
        comparison_query=comparison_query,
        disease_count=disease_count,
        symptom_terms=symptom_terms,
        severity_comparison=severity_comparison,
    )

    response_format = "AMBIGUOUS"

    if danger_ranking_query and disease_count >= 2:
        response_format = "DANGER_RANKING"
    elif severity_comparison:
        response_format = "SYMPTOM_SEVERITY_COMPARISON"
    elif category_confusion_query:
        response_format = "CATEGORY_CLARIFICATION"
    elif red_flag_query:
        response_format = "RED_FLAG"
    elif disease_symptom_link_query:
        response_format = "DISEASE_SYMPTOM_LINK"
    elif (
        disease_count >= 2 and (comparison_query or coexistence_query) and not differential_query
    ):
        response_format = "COMPARISON"
    elif real_life_symptom_query and (differential_query or bool(symptom_terms) or disease_count >= 2):
        response_format = "DIFFERENTIAL_SYMPTOM_CHECK"
    elif symptom_relation_query:
        response_format = "SYMPTOM_RELATION"
    elif disease_count == 1 and len([intent for intent in intents if intent != "summary"]) > 1:
        response_format = "MULTI_INTENT_SINGLE_DISEASE"
    elif disease_count == 1 and intents and intents[0] != "summary":
        response_format = "SINGLE_INTENT"
    elif disease_count == 1 and summary_query:
        response_format = "SINGLE_DISEASE_SUMMARY"
    elif disease_count == 1:
        response_format = "SINGLE_DISEASE_SUMMARY"

    if response_format not in _RESPONSE_FORMATS:
        response_format = "AMBIGUOUS"

    requires_reasoning = any(
        [
            disease_symptom_link_query,
            bool(severity_comparison),
            disease_comparison_query,
            symptom_relation_query,
            real_life_symptom_query,
            bool(category_confusion_query),
            bool(red_flag_query),
            why_how_query
            and not procedural_intent_query
            and (relation_query or bool(concept_terms) or disease_count >= 1),
        ]
    )

    plan: dict[str, Any] = {
        "normalized_query": normalized_query,
        "intents": intents,
        "disease_matches": disease_matches,
        "used_history_context": used_history_context,
        "concept_terms": concept_terms,
        "symptom_terms": symptom_terms,
        "self_report": self_report,
        "summary_query": summary_query,
        "why_how_query": why_how_query,
        "interpretation_query": interpretation_query,
        "relation_query": relation_query,
        "severity_comparison": severity_comparison,
        "disease_symptom_link_query": disease_symptom_link_query,
        "symptom_relation_query": symptom_relation_query,
        "real_life_symptom_query": real_life_symptom_query,
        "disease_comparison_query": disease_comparison_query,
        "category_confusion_query": category_confusion_query,
        "comparison_query": comparison_query,
        "danger_ranking_query": danger_ranking_query,
        "differential_query": differential_query,
        "coexistence_query": coexistence_query,
        "red_flag_query": red_flag_query,
        "requires_reasoning": requires_reasoning,
        "response_format": response_format,
    }

    # Hard rule: if the query class requires reasoning, do not allow summary-style formats.
    if requires_reasoning and not _is_reasoning_response_format(response_format):
        forced_format = _force_reasoning_response_format(plan)
        if forced_format in _RESPONSE_FORMATS:
            response_format = forced_format
            plan["response_format"] = forced_format

    style_mode = _style_mode_for_response_format(response_format)
    reasoning_mode = _reasoning_mode_for_response_format(response_format, intents)
    plan["style_mode"] = style_mode
    plan["reasoning_mode"] = reasoning_mode

    return plan


def _compose_single_disease_summary(record: dict[str, Any], query: str) -> str:
    disease_name = str(record.get("canonical_name") or "this disease").strip()

    opening = _pick_variant(
        f"summary|{query}|{disease_name}",
        [
            f"A quick way to understand {disease_name} is this:",
            f"{disease_name} can be explained in a few practical points.",
            f"Here is a clear overview of {disease_name}.",
        ],
    )

    facts: list[str] = []
    symptoms = _intent_items(record, "symptoms")
    causes = _intent_items(record, "causes")
    treatments = _intent_items(record, "treatment")
    precautions = _intent_items(record, "precautions")

    if symptoms:
        facts.append(f"It often presents with {_format_list(symptoms[:4], 'limited symptom data')}.")
    if causes:
        facts.append(f"Typical causes include {_format_list(causes[:3], 'limited cause data')}.")
    if treatments:
        facts.append(f"Management usually involves {_format_list(treatments[:3], 'limited treatment data')}.")
    if precautions:
        facts.append(f"Helpful prevention steps include {_format_list(precautions[:3], 'limited prevention data')}.")

    if not facts:
        facts.append("Available details for this condition are limited right now.")

    return f"{opening} {' '.join(facts[:4])}".strip()


def _compose_single_intent(record: dict[str, Any], intent: str, query: str) -> str:
    disease_name = str(record.get("canonical_name") or "this disease").strip()
    items = _intent_items(record, intent)

    if intent == "symptoms":
        opening = _pick_variant(
            f"symptoms|{query}|{disease_name}",
            [
                f"For {disease_name}, the most relevant symptoms are {_format_list(items[:5], 'limited symptom data')}.",
                f"{disease_name} commonly shows symptoms such as {_format_list(items[:5], 'limited symptom data')}.",
                f"A typical symptom pattern in {disease_name} includes {_format_list(items[:5], 'limited symptom data')}.",
            ],
        )
        return opening
    elif intent == "causes":
        return _pick_variant(
            f"causes|{query}|{disease_name}",
            [
                f"Common causes associated with {disease_name} include {_format_list(items[:4], 'limited cause data')}.",
                f"{disease_name} is commonly linked to {_format_list(items[:4], 'limited cause data')}.",
                f"Causes of {disease_name} can include {_format_list(items[:4], 'limited cause data')}.",
            ],
        )
    elif intent == "treatment":
        precaution_items = _intent_items(record, "precautions")
        precaution_suffix = ""
        if precaution_items:
            precaution_suffix = (
                f" Helpful daily precautions include "
                f"{_format_list(precaution_items[:2], 'basic healthy habits')}."
            )
        return _pick_variant(
            f"treatment|{query}|{disease_name}",
            [
                f"Treatment for {disease_name} often includes {_format_list(items[:4], 'limited treatment data')}."
                f"{precaution_suffix}",
                f"Management options for {disease_name} include {_format_list(items[:4], 'limited treatment data')}."
                f"{precaution_suffix}",
                f"A typical care plan for {disease_name} may involve {_format_list(items[:4], 'limited treatment data')}."
                f"{precaution_suffix}",
            ],
        )
    elif intent == "precautions":
        return _pick_variant(
            f"precautions|{query}|{disease_name}",
            [
                f"Helpful prevention steps for {disease_name} include {_format_list(items[:4], 'limited prevention data')}.",
                f"To reduce risk around {disease_name}, focus on {_format_list(items[:4], 'limited prevention data')}.",
                f"Practical precautions for {disease_name} are {_format_list(items[:4], 'limited prevention data')}.",
            ],
        )
    elif intent == "red_flags":
        opening = _pick_variant(
            f"red_flags|{query}|{disease_name}",
            [
                f"Warning signs to watch for in {disease_name} are:",
                f"These red flags in {disease_name} need urgent attention:",
                f"Serious signs related to {disease_name} include:",
            ],
        )
        return f"{opening}\n{_format_bullets(items)}"
    else:
        return _compose_single_disease_summary(record, query)


def _compose_multi_intent_single_disease(record: dict[str, Any], intents: list[str], query: str) -> str:
    disease_name = str(record.get("canonical_name") or "this disease").strip()
    opening = _pick_variant(
        f"multi_intent|{query}|{disease_name}",
        [
            f"Here is a focused answer for {disease_name}, covering each part of your question.",
            f"You asked about multiple aspects of {disease_name}; here they are clearly separated.",
            f"For {disease_name}, I will break the answer into the requested sections.",
        ],
    )

    sections: list[str] = []
    for intent in intents:
        if intent == "summary":
            continue
        label = _INTENT_LABELS.get(intent, intent.title())
        items = _intent_items(record, intent)
        sections.append(f"{label}:\n{_format_bullets(items)}")

    if not sections:
        sections.append(_compose_single_disease_summary(record, query))

    return f"{opening}\n\n" + "\n\n".join(sections)


def _normalize_items(items: list[str]) -> set[str]:
    return {normalize_text(item) for item in items if normalize_text(item)}


def _expand_relation_terms(concept_terms: list[str]) -> list[str]:
    expanded: list[str] = []
    for term in concept_terms:
        term_norm = normalize_text(term)
        if term_norm:
            expanded.append(term_norm)
        for key, values in _RELATION_EXPANSION_HINTS.items():
            if key in term_norm:
                expanded.extend(normalize_text(item) for item in values if normalize_text(item))

    deduped: list[str] = []
    seen: set[str] = set()
    for item in expanded:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _best_relation_mechanism(disease_name: str, linked_terms: list[str], causes: list[str]) -> str:
    disease_key = normalize_text(disease_name)
    mechanism_map = _RELATION_MECHANISM_HINTS.get(disease_key, {})

    normalized_terms = [normalize_text(term) for term in linked_terms if normalize_text(term)]
    for term in normalized_terms:
        for hint_key, hint_text in mechanism_map.items():
            hint_norm = normalize_text(hint_key)
            if term == hint_norm or term in hint_norm or hint_norm in term:
                return hint_text

    if causes:
        cause_text = causes[0].rstrip(".")
        focus_text = _format_list(linked_terms[:2], "this symptom pattern")
        return (
            f"This usually happens because {disease_name} can involve {cause_text}, "
            f"which can affect {focus_text}."
        )

    return (
        "This symptom is commonly seen because of how the disease affects the body, "
        "although the exact mechanism can vary."
    )


def _relation_implication_line(
    query: str,
    red_flag_links: list[str],
    symptoms: list[str],
) -> str:
    normalized_query = normalize_text(query)
    if red_flag_links:
        return (
            f"If this occurs with warning signs like {_format_list(red_flag_links[:2], 'severe symptoms')}, "
            "seek urgent in-person care."
        )

    if any(marker in normalized_query for marker in ["persistent", "worsening", "severe", "should i worry"]):
        return "If this is persistent or worsening, a clinical evaluation is the safest next step."

    if symptoms:
        return (
            f"If it keeps recurring with {_format_list(symptoms[:2], 'other symptoms')}, "
            "getting checked early is sensible."
        )

    return "If this symptom persists or interferes with daily activity, a clinical review is recommended."


def _compose_disease_symptom_link(
    disease_match: dict[str, Any],
    concept_terms: list[str],
    query: str,
) -> str:
    record = disease_match.get("record") if isinstance(disease_match, dict) else {}
    record = record if isinstance(record, dict) else {}
    disease_name = str(record.get("canonical_name") or "this condition").strip()
    normalized_query = normalize_text(query)
    interpretation_query = _is_interpretation_query(normalized_query)

    symptoms = _intent_items(record, "symptoms")
    causes = _intent_items(record, "causes")
    red_flags = _intent_items(record, "red_flags")
    expanded_terms = _expand_relation_terms(concept_terms)

    symptom_links: list[str] = []
    cause_links: list[str] = []
    red_flag_links: list[str] = []

    for term in expanded_terms:
        if any(_symptom_matches_record(term, item) for item in symptoms):
            symptom_links.append(term)
        elif any(_symptom_matches_record(term, item) for item in causes):
            cause_links.append(term)
        elif any(_symptom_matches_record(term, item) for item in red_flags):
            red_flag_links.append(term)

    linked_terms = list(dict.fromkeys(symptom_links + cause_links + red_flag_links))
    focus_terms = linked_terms or concept_terms
    focus_text = _format_list(focus_terms[:3], "the symptom you mentioned")
    mechanism_line = _best_relation_mechanism(disease_name, focus_terms, causes)
    implication_line = _relation_implication_line(query, red_flag_links, symptoms)

    if interpretation_query:
        high_phrase_match = re.search(r"(?<![a-z0-9])(high\s+[a-z]+(?:\s+[a-z]+)?)(?![a-z0-9])", normalized_query)
        high_phrase = high_phrase_match.group(1) if high_phrase_match else ""
        high_phrase = re.sub(r"\s+means?$", "", high_phrase).strip()
        if high_phrase:
            direct = _pick_variant(
                f"meaning|{query}|{disease_name}",
                [
                    f"{high_phrase.title()} usually means blood glucose is above the usual range, and {disease_name} is one common explanation.",
                    f"This often points to elevated blood sugar, and {disease_name} is a common cause that should be evaluated.",
                    f"This can mean glucose is running high; {disease_name} is one likely reason depending on the clinical context.",
                ],
            )
            mechanism = _best_relation_mechanism(disease_name, ["high blood sugar"] + focus_terms, causes)
            follow_up = (
                f"If this is persistent or comes with symptoms like {_format_list(symptoms[:3], 'fatigue or other symptoms')}, "
                "it is safest to check with a clinician."
            )
            return f"{direct} {mechanism} {follow_up}".strip()

    if focus_terms:
        direct = _pick_variant(
            f"disease_link|{query}|{disease_name}",
            [
                f"Yes, {disease_name} can affect {focus_text}.",
                f"This is possible: {disease_name} is commonly linked with {focus_text}.",
                f"Yes, this can happen because {disease_name} can involve {focus_text}.",
            ],
        )

        return f"{direct} {mechanism_line} {implication_line}".strip()

    cautious_direct = _pick_variant(
        f"disease_link_uncertain|{query}|{disease_name}",
        [
            f"{disease_name} can still be related to what you described.",
            f"There can be a relationship with {disease_name}, even when symptom detail is limited.",
            f"This can be related to {disease_name}, but more symptom detail helps narrow it.",
        ],
    )
    cautious_reasoning = (
        "This symptom is commonly seen because of how the disease affects the body, "
        "although the exact mechanism can vary."
    )
    return f"{cautious_direct} {cautious_reasoning} {implication_line}".strip()


def _compose_symptom_relation(symptom_terms: list[str], query: str) -> str:
    if len(symptom_terms) < 2:
        return "I can explain symptom relationships, but I need at least two symptom terms to compare or relate."

    ranked = _rank_diseases_by_symptoms(symptom_terms)
    overlap_names = [
        str((item.get("record") or {}).get("canonical_name") or "").strip()
        for item in ranked[:3]
        if (item.get("record") or {}).get("canonical_name")
    ]

    opening = _pick_variant(
        f"symptom_relation|{query}",
        [
            "Yes, these symptoms can be related.",
            "These symptoms can definitely occur together.",
            "There is a real clinical relationship between these symptoms.",
        ],
    )

    mechanism_line = _pick_variant(
        f"symptom_relation_reason|{query}",
        [
            "This usually happens because multiple conditions can trigger overlapping inflammatory or systemic responses.",
            "This can happen because different diseases may share body pathways that produce similar symptom clusters.",
            "This overlap is common because one underlying illness can produce several related symptoms at the same time.",
        ],
    )

    if overlap_names:
        return (
            f"{opening} {mechanism_line} "
            f"They can co-occur in conditions such as {_format_list(overlap_names, 'multiple conditions')}. "
            "So this relation helps pattern recognition, but does not by itself confirm one diagnosis."
        )

    return (
        f"{opening} {mechanism_line} "
        "These symptoms can still happen together, but they are not specific enough on their own to identify one diagnosis."
    )


def _compose_symptom_severity_comparison(severity_comparison: dict[str, str] | None, query: str) -> str:
    if not severity_comparison:
        return (
            "This looks like a severity comparison question. In general, severity words (such as mild, high, or severe) "
            "describe intensity of the same symptom rather than a different disease because they reflect degree, not diagnosis."
        )

    base_symptom = str(severity_comparison.get("base_symptom") or "the symptom").strip()
    qualified_symptom = str(severity_comparison.get("qualified_symptom") or "the higher-intensity symptom").strip()
    normalized_query = normalize_text(query)

    if "same" in normalized_query:
        lines = [
            f"{qualified_symptom.title()} is not a different disease; it is a stronger level of {base_symptom}.",
            f"In plain terms, {qualified_symptom} usually indicates more severity than {base_symptom}.",
            "This happens because words like high or severe describe intensity within the same symptom category.",
        ]
    else:
        lines = [
            f"{qualified_symptom.title()} and {base_symptom} are in the same symptom family, but at different intensity levels.",
            f"In plain terms, {qualified_symptom} usually indicates greater severity than {base_symptom}.",
            "This happens because severity qualifiers describe how strong the symptom is, not a separate disease.",
        ]

    if base_symptom == "fever" and "high fever" in qualified_symptom:
        lines.append(
            "For fever specifically, high fever usually means a higher body temperature and can signal a greater need for timely clinical review."
        )

    lines.append(
        "If symptoms are persistent, worsening, or associated with confusion, breathing difficulty, bleeding, or severe dehydration, seek urgent medical care."
    )

    return " ".join(lines)


def _compose_category_clarification(
    disease_matches: list[dict[str, Any]],
    symptom_terms: list[str],
    query: str,
) -> tuple[str, list[dict[str, Any]]]:
    if not disease_matches:
        return (
            "You are mixing different categories in one comparison. One term appears to be a disease while the others look like symptoms.",
            [],
        )

    primary = disease_matches[0]
    record = primary.get("record") if isinstance(primary, dict) else {}
    record = record if isinstance(record, dict) else {}
    disease_name = str(record.get("canonical_name") or "this condition").strip()
    symptom_text = _format_list(symptom_terms[:4], "those terms")

    lines = [
        f"{disease_name} is a disease, while {symptom_text} are symptoms.",
    ]

    disease_symptoms = _intent_items(record, "symptoms")
    overlap = [term for term in symptom_terms if any(_symptom_matches_record(term, item) for item in disease_symptoms)]

    if overlap:
        lines.append(
            f"If your intent is to check whether {disease_name} can cause these symptoms: yes, symptoms like {_format_list(overlap[:4], 'some of them')} can occur with {disease_name}."
        )
    else:
        lines.append(
            f"These symptoms can occur in many conditions, so they do not by themselves confirm {disease_name}."
        )

    lines.append(f"I can now give a direct explanation of how these symptoms relate to {disease_name}.")
    return " ".join(lines), disease_matches


def _compose_comparison(
    disease_matches: list[dict[str, Any]],
    intents: list[str],
    query: str,
    coexistence_query: bool,
) -> str:
    records = [match.get("record") for match in disease_matches if isinstance(match.get("record"), dict)]
    records = [record for record in records if record]

    disease_names = [str(record.get("canonical_name") or "").strip() for record in records]
    disease_names = [name for name in disease_names if name]

    names_text = _format_list(disease_names, "the listed conditions")

    compare_intents = [intent for intent in intents if intent in {"symptoms", "causes", "treatment", "precautions", "red_flags"}]
    if not compare_intents:
        normalized_query = normalize_text(query)
        if "difference" in normalized_query or "compare" in normalized_query or "versus" in normalized_query or " vs " in normalized_query:
            compare_intents = ["symptoms", "causes"]
        else:
            compare_intents = ["symptoms"]

    if len(compare_intents) > 1:
        order_seed = f"comparison_order|{query}|{names_text}|{'|'.join(compare_intents)}"
        rotate_by = _style_cycle(order_seed, len(compare_intents))
        compare_intents = compare_intents[rotate_by:] + compare_intents[:rotate_by]

    if coexistence_query and compare_intents == ["summary"]:
        lines = [
            f"You asked whether {names_text} can happen together.",
            "I cannot confirm coexistence from this information alone.",
            "Here is what is known for each condition:",
        ]
        for record in records:
            disease_name = str(record.get("canonical_name") or "this condition").strip()
            lines.append(f"- {disease_name}: {_summary_line(record)}")

        lines.append(
            "If symptoms are persistent or worsening, an in-person medical evaluation is the safest next step."
        )
        return "\n".join(lines)

    lines: list[str] = [_pick_variant(
        f"comparison|{query}|{names_text}",
        [
            f"Here is a direct comparison of {names_text}.",
            f"Let us compare {names_text} side by side.",
            f"This is a focused comparison of {names_text}.",
        ],
    )]

    for intent in compare_intents:
        label = _INTENT_LABELS.get(intent, intent.title())
        lines.append(f"\n{label} (Key differences):")

        per_disease_sets: list[set[str]] = []
        for record in records:
            disease_name = str(record.get("canonical_name") or "this disease").strip()
            items = _intent_items(record, intent)
            per_disease_sets.append(_normalize_items(items))
            lines.append(f"- {disease_name}: {_format_list(items[:5], 'details are limited for this part')}")

        if per_disease_sets:
            common = set.intersection(*per_disease_sets) if len(per_disease_sets) > 1 else per_disease_sets[0]
        else:
            common = set()

        if common:
            common_display = [item for item in sorted(common) if item][:5]
            lines.append("\nSimilarities:")
            lines.append(f"- Shared pattern: {_format_list(common_display, 'none')}.")
        else:
            lines.append(
                _pick_variant(
                    f"comparison_no_overlap|{query}|{intent}|{names_text}",
                    [
                        "- For this part, the conditions differ more than they overlap.",
                        "- In this area, overlap is limited and differences are more prominent.",
                        "- For this aspect, differences are clearer than similarities.",
                    ],
                )
            )

    red_flag_union: list[str] = []
    for record in records:
        red_flag_union.extend(_intent_items(record, "red_flags"))

    dedup_red_flags: list[str] = []
    seen_red_flags: set[str] = set()
    for item in red_flag_union:
        key = normalize_text(item)
        if key and key not in seen_red_flags:
            seen_red_flags.add(key)
            dedup_red_flags.append(item)

    lines.append("\nCommon warning signs:")
    if dedup_red_flags:
        lines.append(_format_bullets(dedup_red_flags, max_items=6))
    else:
        lines.append("- Warning-sign details are limited for this comparison.")

    lines.append("\nWhen to seek medical help:")
    lines.append("- Seek urgent care if symptoms become severe, rapidly worsen, or include breathing difficulty, confusion, fainting, or bleeding.")

    return "\n".join(lines)


def _danger_scores(record: dict[str, Any]) -> tuple[float, float]:
    red_flags = _intent_items(record, "red_flags")
    symptoms = _intent_items(record, "symptoms")

    short_term = 0.0
    for item in red_flags:
        item_norm = normalize_text(item)
        for token, weight in _DANGER_SIGNAL_WEIGHTS.items():
            if token in item_norm:
                short_term += weight
                break

    short_term += min(1.5, len(red_flags) * 0.25)

    long_term = 0.0
    chronic_markers = ["chronic", "weight loss", "long", "persistent", "genetic", "resistance"]
    for item in symptoms + _intent_items(record, "causes"):
        item_norm = normalize_text(item)
        if any(marker in item_norm for marker in chronic_markers):
            long_term += 1.0

    long_term += min(1.0, len(_intent_items(record, "treatment")) * 0.15)
    return short_term, long_term


def _compose_danger_ranking(disease_matches: list[dict[str, Any]], query: str) -> str:
    records = [match.get("record") for match in disease_matches if isinstance(match.get("record"), dict)]
    records = [record for record in records if record]
    if len(records) < 2:
        if records:
            disease_name = str(records[0].get("canonical_name") or "this condition").strip()
            return (
                f"I can describe warning signs for {disease_name}, but a direct danger ranking needs at least two clearly identified diseases."
            )
        return "I can give a danger comparison once at least two disease names are clear."

    profiles: list[dict[str, Any]] = []
    for record in records:
        disease_name = str(record.get("canonical_name") or "this condition").strip()
        short_term, long_term = _danger_scores(record)
        profiles.append(
            {
                "name": disease_name,
                "record": record,
                "short_term": short_term,
                "long_term": long_term,
                "red_flags": _intent_items(record, "red_flags"),
            }
        )

    short_ranked = sorted(profiles, key=lambda item: (-float(item["short_term"]), item["name"]))
    long_ranked = sorted(profiles, key=lambda item: (-float(item["long_term"]), item["name"]))

    top_short = short_ranked[0]
    second_short = short_ranked[1] if len(short_ranked) > 1 else short_ranked[0]
    gap = float(top_short["short_term"]) - float(second_short["short_term"])

    names_text = _format_list([str(item["name"]) for item in profiles], "the listed diseases")

    lines: list[str] = []
    if gap >= 1.5:
        lines.append(
            _pick_variant(
                f"danger_opening|{query}|{top_short['name']}|{second_short['name']}",
                [
                    f"Based on typical warning signs, {top_short['name']} is often more concerning for short-term danger than {second_short['name']}.",
                    f"Looking at warning-sign patterns, {top_short['name']} appears more concerning for short-term risk than {second_short['name']}.",
                    f"For immediate-risk concerns, {top_short['name']} is usually more worrisome than {second_short['name']}.",
                ],
            )
        )
    else:
        lines.append(
            _pick_variant(
                f"danger_opening_balanced|{query}|{names_text}",
                [
                    f"Between {names_text}, short-term danger can be high in more than one condition, so context matters clinically.",
                    f"Across {names_text}, short-term risk can be significant in more than one condition.",
                    f"For {names_text}, short-term danger is not one-sided and depends on clinical context.",
                ],
            )
        )

    lines.append("Short-term danger:")
    for item in short_ranked[:3]:
        red_flags = list(item.get("red_flags") or [])
        lines.append(f"- {item['name']}: key warning signs include {_format_list(red_flags[:4], 'limited red-flag data')}.")

    top_long = long_ranked[0]
    long_gap = float(top_long["long_term"]) - float(long_ranked[1]["long_term"]) if len(long_ranked) > 1 else 0.0
    if len(long_ranked) > 1 and top_long["name"] != long_ranked[1]["name"] and long_gap >= 0.75:
        lines.append(
            f"Long-term concern: {top_long['name']} may need more sustained follow-up based on chronic-pattern signals in the available data."
        )
    else:
        lines.append("Long-term concern: both conditions can still be serious depending on individual risk factors and response to treatment.")

    lines.append(
        "Seek urgent medical care if there is breathing difficulty, persistent chest pain, confusion, fainting, active bleeding, or rapid worsening."
    )

    if "which is" in normalize_text(query) or "more dangerous" in normalize_text(query):
        lines.append("This comparison is based on common warning-sign patterns, not a diagnosis for a specific individual.")

    return "\n".join(lines)


def _compose_differential(
    disease_matches: list[dict[str, Any]],
    symptom_terms: list[str],
    query: str,
    red_flag_query: bool,
) -> tuple[str, list[dict[str, Any]]]:
    candidate_matches = list(disease_matches)

    if not candidate_matches and symptom_terms:
        ranked = _rank_diseases_by_symptoms(symptom_terms)
        candidate_matches = [
            {
                "key": item["key"],
                "record": item["record"],
                "match_type": "symptom_differential",
                "confidence": 0.62 + (0.08 * min(2, int(item.get("score") or 0))),
            }
            for item in ranked[:3]
        ]

    candidate_records = [match.get("record") for match in candidate_matches if isinstance(match.get("record"), dict)]
    candidate_records = [record for record in candidate_records if record]

    lines = [
        "These symptoms can happen in more than one condition. Based on the information provided, I cannot confirm a single disease.",
    ]

    if symptom_terms:
        lines.append(f"Symptoms noted from your message: {_format_list(symptom_terms, 'none')}.")

    if candidate_records:
        lines.append("Possible conditions to consider:")
        for record in candidate_records[:3]:
            disease_name = str(record.get("canonical_name") or "this condition").strip()
            symptoms = _intent_items(record, "symptoms")
            overlap = [symptom for symptom in symptom_terms if any(_symptom_matches_record(symptom, item) for item in symptoms)]
            if overlap:
                lines.append(f"- {disease_name}: can overlap with {_format_list(overlap, 'some of your symptoms')}.")
            else:
                lines.append(f"- {disease_name}: some overlap may be present, but it is not clear from this information alone.")

        if len(candidate_records) >= 2:
            lines.append("Key differences to watch for:")
            all_symptom_sets = [_normalize_items(_intent_items(record, "symptoms")) for record in candidate_records]
            shared = set.intersection(*all_symptom_sets) if len(all_symptom_sets) > 1 else set()
            for record in candidate_records[:3]:
                disease_name = str(record.get("canonical_name") or "this condition").strip()
                unique_items = [
                    item
                    for item in _intent_items(record, "symptoms")
                    if normalize_text(item) and normalize_text(item) not in shared
                ]
                lines.append(
                    f"- {disease_name}: {_format_list(unique_items[:3], 'distinctive signs are limited right now')}."
                )

    if red_flag_query or _contains_emergency_marker(query, "red_flags"):
        lines.append(
            "If symptoms are severe, worsening, or include warning signs such as breathing difficulty, confusion, fainting, or bleeding, seek urgent in-person care now."
        )
    else:
        lines.append("If symptoms continue, worsen, or interfere with daily activity, arrange an in-person clinical evaluation.")

    return "\n".join(lines), candidate_matches


def _compose_red_flag(
    disease_matches: list[dict[str, Any]],
    symptom_terms: list[str],
    query: str,
) -> tuple[str, list[dict[str, Any]]]:
    lines = [
        "I cannot confirm a diagnosis from chat alone, but your question points to symptoms that should be taken seriously.",
    ]

    candidate_matches = list(disease_matches)

    if disease_matches:
        for match in disease_matches[:2]:
            record = match.get("record") or {}
            disease_name = str(record.get("canonical_name") or "this condition").strip()
            red_flags = _intent_items(record, "red_flags")
            lines.append(f"For {disease_name}, warning signs include:")
            lines.append(_format_bullets(red_flags, max_items=5))
    elif symptom_terms:
        ranked = _rank_diseases_by_symptoms(symptom_terms)
        if ranked:
            candidate_matches = [
                {
                    "key": item["key"],
                    "record": item["record"],
                    "match_type": "symptom_differential",
                    "confidence": 0.60 + (0.06 * min(2, int(item.get("score") or 0))),
                }
                for item in ranked[:2]
            ]
            candidate_names = [
                str((item.get("record") or {}).get("canonical_name") or "").strip()
                for item in candidate_matches
                if (item.get("record") or {}).get("canonical_name")
            ]
            if candidate_names:
                lines.append(
                    "This pattern can overlap with "
                    f"{_format_list(candidate_names, 'multiple conditions')}.")

    lines.append(
        "Please seek urgent medical care now if symptoms are severe, worsening, or include breathing difficulty, persistent chest pain, confusion, fainting, seizures, or bleeding."
    )

    if "should i worry" in normalize_text(query):
        lines.append("Given what you described, it is reasonable to seek timely in-person assessment.")

    return "\n".join(lines), candidate_matches


def _compose_ambiguous(query: str, intents: list[str]) -> str:
    non_summary_intents = [intent for intent in intents if intent != "summary"]
    if non_summary_intents:
        intent = non_summary_intents[0]
        label = _INTENT_LABELS.get(intent, intent)
        return (
            f"I can help with {label.lower()}, but I need the disease name first. "
            "Which disease are you asking about?"
        )

    return "I could not identify the disease name clearly from your question. Please mention it directly."


def _requires_reasoning(plan: dict[str, Any]) -> bool:
    return bool(plan.get("requires_reasoning"))


def _has_reasoning_explanation(answer: str) -> bool:
    normalized_answer = normalize_text(answer)
    if not normalized_answer:
        return False
    return any(marker in normalized_answer for marker in _REASONING_EXPLANATION_MARKERS)


def _validate_reasoning_requirements(plan: dict[str, Any], answer: str) -> bool:
    if not _requires_reasoning(plan):
        return True

    response_format = str(plan.get("response_format") or "AMBIGUOUS")
    normalized_answer = normalize_text(answer)
    if not normalized_answer:
        return False

    if not _is_reasoning_response_format(response_format):
        return False

    if any(marker in normalized_answer for marker in _FORBIDDEN_TEMPLATE_MARKERS):
        return False

    if _contains_meta_language(answer):
        return False

    if not _has_reasoning_explanation(answer):
        return False

    return True


def _append_cautious_reasoning_line(answer: str) -> str:
    note = (
        "This is commonly seen because of how the disease affects the body, "
        "although the exact mechanism can vary from person to person."
    )
    normalized_answer = normalize_text(answer)
    if normalize_text(note) in normalized_answer:
        return answer

    cleaned = (answer or "").strip()
    if not cleaned:
        return note
    if cleaned.endswith((".", "!", "?")):
        return f"{cleaned} {note}"
    return f"{cleaned}. {note}"


def _regenerate_reasoning_answer(plan: dict[str, Any], query: str) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    regen_plan = dict(plan)
    response_format = str(regen_plan.get("response_format") or "AMBIGUOUS")

    if not _is_reasoning_response_format(response_format):
        forced_format = _force_reasoning_response_format(regen_plan)
        if forced_format in _RESPONSE_FORMATS:
            regen_plan["response_format"] = forced_format
            regen_plan["style_mode"] = _style_mode_for_response_format(forced_format)
            regen_plan["reasoning_mode"] = _reasoning_mode_for_response_format(
                forced_format,
                list(regen_plan.get("intents") or []),
            )

    regenerated, regenerated_matches = _compose_style_safe_fallback(regen_plan, query)

    if _requires_reasoning(regen_plan) and not _has_reasoning_explanation(regenerated):
        regenerated = _append_cautious_reasoning_line(regenerated)

    return regen_plan, regenerated, regenerated_matches


def _validate_response_style(plan: dict[str, Any], answer: str, used_matches: list[dict[str, Any]]) -> bool:
    response_format = str(plan.get("response_format") or "AMBIGUOUS")
    intents = list(plan.get("intents") or [])
    concept_terms = [normalize_text(item) for item in (plan.get("concept_terms") or []) if normalize_text(item)]
    normalized_answer = normalize_text(answer)
    if not normalized_answer:
        return False

    if any(marker in normalized_answer for marker in _FORBIDDEN_TEMPLATE_MARKERS):
        return False

    if _contains_meta_language(answer):
        return False

    used_names = [
        str((match.get("record") or {}).get("canonical_name") or "").strip()
        for match in used_matches
        if (match.get("record") or {}).get("canonical_name")
    ]

    if response_format == "COMPARISON":
        mention_count = sum(
            1 for name in used_names if name and re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", answer, flags=re.IGNORECASE)
        )
        if len(used_names) >= 2 and mention_count < 2:
            return False
        if "difference" not in normalized_answer and "similarit" not in normalized_answer and "comparison" not in normalized_answer:
            return False
        if "similarities:" in normalized_answer and "shared pattern" not in normalized_answer:
            return False

    if response_format == "DANGER_RANKING":
        mention_count = sum(
            1 for name in used_names if name and re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", answer, flags=re.IGNORECASE)
        )
        if len(used_names) >= 2 and mention_count < 2:
            return False
        must_have = ["short-term", "warning sign", "danger"]
        if sum(1 for marker in must_have if marker in normalized_answer) < 2:
            return False

    if response_format == "DISEASE_SYMPTOM_LINK":
        mention_count = sum(
            1 for name in used_names if name and re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", answer, flags=re.IGNORECASE)
        )
        if used_names and mention_count < 1:
            return False
        if concept_terms and not any(term in normalized_answer for term in concept_terms[:4]):
            return False

    if response_format == "SYMPTOM_RELATION":
        if "relation" not in normalized_answer and "overlap" not in normalized_answer and "together" not in normalized_answer:
            return False

    if response_format == "SYMPTOM_SEVERITY_COMPARISON":
        severity_comparison = plan.get("severity_comparison") or {}
        base_symptom = normalize_text(str(severity_comparison.get("base_symptom") or ""))
        qualified_symptom = normalize_text(str(severity_comparison.get("qualified_symptom") or ""))
        if base_symptom and base_symptom not in normalized_answer:
            return False
        if qualified_symptom and qualified_symptom not in normalized_answer:
            return False
        if "severity" not in normalized_answer and "intensity" not in normalized_answer and "more" not in normalized_answer:
            return False

    if response_format == "CATEGORY_CLARIFICATION":
        if "disease" not in normalized_answer or "symptom" not in normalized_answer:
            return False
        if used_names and not any(normalize_text(name) in normalized_answer for name in used_names if name):
            return False

    if response_format == "SINGLE_INTENT" and intents[:1] != ["red_flags"] and "\n-" in answer and len(used_names) <= 1:
        # Keep simple intent responses conversational by default.
        return False

    return True


def _compose_style_safe_fallback(plan: dict[str, Any], query: str) -> tuple[str, list[dict[str, Any]]]:
    response_format = str(plan.get("response_format") or "AMBIGUOUS")
    intents = list(plan.get("intents") or [])
    disease_matches = list(plan.get("disease_matches") or [])
    concept_terms = list(plan.get("concept_terms") or [])
    symptom_terms = list(plan.get("symptom_terms") or [])
    severity_comparison = plan.get("severity_comparison")

    if response_format == "CATEGORY_CLARIFICATION":
        return _compose_category_clarification(disease_matches=disease_matches, symptom_terms=symptom_terms, query=query)

    if response_format == "DISEASE_SYMPTOM_LINK" and disease_matches:
        return _compose_disease_symptom_link(disease_matches[0], concept_terms, query), disease_matches

    if response_format == "SYMPTOM_RELATION":
        return _compose_symptom_relation(symptom_terms=symptom_terms, query=query), []

    if response_format == "SYMPTOM_SEVERITY_COMPARISON":
        return _compose_symptom_severity_comparison(severity_comparison=severity_comparison, query=query), []

    if response_format == "DANGER_RANKING" and disease_matches:
        return _compose_danger_ranking(disease_matches, query), disease_matches

    if response_format == "COMPARISON" and disease_matches:
        return (
            _compose_comparison(
                disease_matches=disease_matches,
                intents=intents,
                query=query,
                coexistence_query=bool(plan.get("coexistence_query")),
            ),
            disease_matches,
        )

    if response_format == "DIFFERENTIAL_SYMPTOM_CHECK":
        return _compose_differential(
            disease_matches=disease_matches,
            symptom_terms=symptom_terms,
            query=query,
            red_flag_query=bool(plan.get("red_flag_query")),
        )

    if response_format == "RED_FLAG":
        return _compose_red_flag(disease_matches=disease_matches, symptom_terms=symptom_terms, query=query)

    if response_format == "MULTI_INTENT_SINGLE_DISEASE" and disease_matches:
        return _compose_multi_intent_single_disease(disease_matches[0]["record"], intents, query), disease_matches

    if response_format == "SINGLE_INTENT" and disease_matches:
        intent = detect_query_intent(query)
        resolved_intent = intent if intent != "unknown" else "summary"
        return _compose_single_intent(disease_matches[0]["record"], resolved_intent, query), disease_matches

    if response_format == "SINGLE_DISEASE_SUMMARY" and disease_matches:
        return _compose_single_disease_summary(disease_matches[0]["record"], query), disease_matches

    return _compose_ambiguous(query, intents), []


def _validate_response_disease_scope(answer: str, allowed_diseases: list[str]) -> tuple[bool, list[str]]:
    if not allowed_diseases:
        return True, []

    allowed = {normalize_text(name) for name in allowed_diseases if normalize_text(name)}
    if not allowed:
        return True, []

    mentioned: set[str] = set()
    for disease_name in _all_canonical_names():
        disease_norm = normalize_text(disease_name)
        if not disease_norm:
            continue
        pattern = rf"(?<![a-z0-9]){re.escape(disease_name)}(?![a-z0-9])"
        if re.search(pattern, answer or "", flags=re.IGNORECASE):
            mentioned.add(disease_norm)

    unexpected = sorted([name for name in mentioned if name not in allowed])
    return len(unexpected) == 0, unexpected


def _compose_answer(plan: dict[str, Any], query: str) -> tuple[str, list[dict[str, Any]]]:
    response_format = str(plan.get("response_format") or "AMBIGUOUS")
    if _requires_reasoning(plan) and not _is_reasoning_response_format(response_format):
        response_format = _force_reasoning_response_format(plan)

    intents = list(plan.get("intents") or [])
    disease_matches = list(plan.get("disease_matches") or [])
    concept_terms = list(plan.get("concept_terms") or [])
    symptom_terms = list(plan.get("symptom_terms") or [])
    severity_comparison = plan.get("severity_comparison")

    if response_format == "CATEGORY_CLARIFICATION":
        return _compose_category_clarification(disease_matches=disease_matches, symptom_terms=symptom_terms, query=query)

    if response_format == "DISEASE_SYMPTOM_LINK" and disease_matches:
        return _compose_disease_symptom_link(disease_matches[0], concept_terms, query), disease_matches

    if response_format == "SYMPTOM_RELATION":
        return _compose_symptom_relation(symptom_terms=symptom_terms, query=query), []

    if response_format == "SYMPTOM_SEVERITY_COMPARISON":
        return _compose_symptom_severity_comparison(severity_comparison=severity_comparison, query=query), []

    if response_format == "SINGLE_DISEASE_SUMMARY" and disease_matches:
        return _compose_single_disease_summary(disease_matches[0]["record"], query), disease_matches

    if response_format == "SINGLE_INTENT" and disease_matches:
        intent = detect_query_intent(query)
        resolved_intent = intent if intent != "unknown" else "summary"
        return _compose_single_intent(disease_matches[0]["record"], resolved_intent, query), disease_matches

    if response_format == "MULTI_INTENT_SINGLE_DISEASE" and disease_matches:
        return _compose_multi_intent_single_disease(disease_matches[0]["record"], intents, query), disease_matches

    if response_format == "COMPARISON" and disease_matches:
        return (
            _compose_comparison(
                disease_matches=disease_matches,
                intents=intents,
                query=query,
                coexistence_query=bool(plan.get("coexistence_query")),
            ),
            disease_matches,
        )

    if response_format == "DANGER_RANKING" and disease_matches:
        return _compose_danger_ranking(disease_matches, query), disease_matches

    if response_format == "DIFFERENTIAL_SYMPTOM_CHECK":
        return _compose_differential(
            disease_matches=disease_matches,
            symptom_terms=symptom_terms,
            query=query,
            red_flag_query=bool(plan.get("red_flag_query")),
        )

    if response_format == "RED_FLAG":
        return _compose_red_flag(
            disease_matches=disease_matches,
            symptom_terms=symptom_terms,
            query=query,
        )

    return _compose_ambiguous(query, intents), []


def _compose_fallback_no_match_answer(query: str, intents: list[str]) -> str:
    non_summary_intents = [intent for intent in intents if intent != "summary"]
    if non_summary_intents:
        intent_label = _INTENT_LABELS.get(non_summary_intents[0], non_summary_intents[0])
        return f"I can help with {intent_label.lower()}, but I could not identify the disease name clearly. Please mention it directly."

    return "I could not identify that disease name clearly. Please mention it directly."


def _build_supporting_contexts(
    used_matches: list[dict[str, Any]],
    intents: list[str],
    response_format: str,
) -> list[str]:
    if not used_matches:
        return []

    intent_set = {str(intent) for intent in intents}
    include_all = response_format in {
        "SINGLE_DISEASE_SUMMARY",
        "MULTI_INTENT_SINGLE_DISEASE",
        "COMPARISON",
        "DANGER_RANKING",
    }

    contexts: list[str] = []
    for match in used_matches:
        record = (match.get("record") if isinstance(match, dict) else None) or {}
        if not isinstance(record, dict):
            continue

        disease_name = str(record.get("canonical_name") or "").strip()
        if disease_name:
            contexts.append(f"Disease: {disease_name}")

        symptoms = [str(item).strip() for item in (record.get("symptoms") or []) if str(item).strip()][:6]
        causes = [str(item).strip() for item in (record.get("causes") or []) if str(item).strip()][:4]
        treatments = [str(item).strip() for item in (record.get("treatments") or []) if str(item).strip()][:5]
        precautions = [str(item).strip() for item in (record.get("precautions") or []) if str(item).strip()][:5]
        red_flags = [str(item).strip() for item in (record.get("red_flags") or []) if str(item).strip()][:4]

        if symptoms and (
            include_all
            or "symptoms" in intent_set
            or response_format in {
                "DISEASE_SYMPTOM_LINK",
                "DIFFERENTIAL_SYMPTOM_CHECK",
                "SYMPTOM_RELATION",
                "SYMPTOM_SEVERITY_COMPARISON",
            }
        ):
            contexts.append("Symptoms: " + ", ".join(symptoms))

        if causes and (
            include_all
            or "causes" in intent_set
            or response_format == "DISEASE_SYMPTOM_LINK"
        ):
            contexts.append("Causes: " + ", ".join(causes))

        if treatments and (
            include_all
            or "treatment" in intent_set
            or response_format in {"SINGLE_INTENT", "MULTI_INTENT_SINGLE_DISEASE"}
        ):
            contexts.append("Treatments: " + ", ".join(treatments))

        if precautions and (
            include_all
            or "precautions" in intent_set
            or "treatment" in intent_set
            or response_format in {"SINGLE_INTENT", "MULTI_INTENT_SINGLE_DISEASE"}
        ):
            contexts.append("Precautions: " + ", ".join(precautions))

        if red_flags and (
            "red_flags" in intent_set or response_format in {"RED_FLAG", "DANGER_RANKING"}
        ):
            contexts.append("Red flags: " + ", ".join(red_flags))

        source_name = str(record.get("source_name") or "").strip()
        source_url = str(record.get("source_url") or "").strip()
        if source_name or source_url:
            source_line = f"Source: {source_name}" if source_name else "Source:"
            if source_url:
                source_line = f"{source_line} ({source_url})"
            contexts.append(source_line.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for context in contexts:
        normalized = normalize_text(context)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(context)
        if len(deduped) >= 16:
            break

    return deduped


def get_medical_answer(query: str, chat_history: str | None = None) -> dict[str, Any]:
    load_medical_facts()

    plan = _analyze_query(query, chat_history=chat_history)
    intents: list[str] = list(plan.get("intents") or [])
    response_format = str(plan.get("response_format") or "AMBIGUOUS")
    reasoning_mode = str(plan.get("reasoning_mode") or "ambiguous")
    style_mode = str(plan.get("style_mode") or "clarification_only")
    requires_reasoning = _requires_reasoning(plan)

    if requires_reasoning and not _is_reasoning_response_format(response_format):
        forced_format = _force_reasoning_response_format(plan)
        if forced_format in _RESPONSE_FORMATS:
            plan["response_format"] = forced_format
            plan["style_mode"] = _style_mode_for_response_format(forced_format)
            plan["reasoning_mode"] = _reasoning_mode_for_response_format(forced_format, intents)
            response_format = forced_format
            style_mode = str(plan.get("style_mode") or style_mode)
            reasoning_mode = str(plan.get("reasoning_mode") or reasoning_mode)

    answer, used_matches = _compose_answer(plan, query)

    if response_format != "AMBIGUOUS" and not _validate_response_style(plan, answer, used_matches):
        answer, used_matches = _compose_style_safe_fallback(plan, query)

    answer = _sanitize_user_facing_answer(answer)

    if requires_reasoning and not _validate_reasoning_requirements(plan, answer):
        regen_plan, regenerated_answer, regenerated_matches = _regenerate_reasoning_answer(plan, query)
        regenerated_answer = _sanitize_user_facing_answer(regenerated_answer)
        if not _validate_reasoning_requirements(regen_plan, regenerated_answer):
            regenerated_answer = _sanitize_user_facing_answer(_append_cautious_reasoning_line(regenerated_answer))

        plan = regen_plan
        answer = regenerated_answer
        used_matches = regenerated_matches
        response_format = str(plan.get("response_format") or response_format)
        reasoning_mode = str(plan.get("reasoning_mode") or reasoning_mode)
        style_mode = str(plan.get("style_mode") or style_mode)

    primary_intent = detect_query_intent(query)
    if primary_intent == "unknown":
        primary_intent = intents[0] if intents else "unknown"

    diseases = [
        str((match.get("record") or {}).get("canonical_name") or "").strip()
        for match in used_matches
        if (match.get("record") or {}).get("canonical_name")
    ]

    needs_clarification = response_format == "AMBIGUOUS"

    if response_format in {
        "SINGLE_DISEASE_SUMMARY",
        "SINGLE_INTENT",
        "MULTI_INTENT_SINGLE_DISEASE",
        "DISEASE_SYMPTOM_LINK",
        "CATEGORY_CLARIFICATION",
        "COMPARISON",
        "DANGER_RANKING",
    } and not diseases:
        needs_clarification = True

    if response_format == "AMBIGUOUS":
        fallback_answer = _sanitize_user_facing_answer(_compose_fallback_no_match_answer(query, intents))
        return {
            "found": False,
            "confidence": 0.0,
            "answer": fallback_answer,
            "intent": primary_intent,
            "intents": intents,
            "disease": None,
            "diseases": [],
            "source": None,
            "match_type": "none",
            "response_format": response_format,
            "reasoning_mode": reasoning_mode,
            "style_mode": style_mode,
            "needs_clarification": True,
            "requires_reasoning": requires_reasoning,
            "supporting_contexts": [],
        }

    match_type = "none"
    confidence = 0.0

    if used_matches:
        match_type_set = {str(match.get("match_type") or "alias_exact") for match in used_matches}
        confidence_values = [float(match.get("confidence") or 0.0) for match in used_matches]
        confidence = min(confidence_values) if confidence_values else 0.0

        if len(match_type_set) == 1:
            match_type = next(iter(match_type_set))
        else:
            match_type = "multi_match"
    elif response_format in {"DIFFERENTIAL_SYMPTOM_CHECK", "RED_FLAG"}:
        confidence = 0.62
        match_type = "symptom_differential"
    elif response_format == "SYMPTOM_SEVERITY_COMPARISON":
        confidence = 0.68
        match_type = "severity_reasoning"
    elif response_format == "SYMPTOM_RELATION":
        symptom_terms = list(plan.get("symptom_terms") or [])
        confidence = 0.58 if len(symptom_terms) >= 2 else 0.52
        match_type = "symptom_relation"
    elif response_format == "CATEGORY_CLARIFICATION":
        confidence = 0.7 if used_matches else 0.55
        match_type = "category_clarification"

    # Keep confidence gate behavior consistent.
    if confidence < 0.5 and response_format not in {"DIFFERENTIAL_SYMPTOM_CHECK", "RED_FLAG", "SYMPTOM_RELATION", "SYMPTOM_SEVERITY_COMPARISON"}:
        return {
            "found": False,
            "confidence": confidence,
            "answer": _compose_fallback_no_match_answer(query, intents),
            "intent": primary_intent,
            "intents": intents,
            "disease": None,
            "diseases": [],
            "source": None,
            "match_type": match_type,
            "response_format": response_format,
            "reasoning_mode": reasoning_mode,
            "style_mode": style_mode,
            "needs_clarification": True,
            "requires_reasoning": requires_reasoning,
            "supporting_contexts": [],
        }

    is_valid_scope, unexpected_mentions = _validate_response_disease_scope(answer, diseases)
    if not is_valid_scope:
        return {
            "found": False,
            "confidence": 0.0,
            "answer": (
                "I could not provide a reliable disease-specific response because the draft answer "
                "contained out-of-scope disease content. Please restate your disease names clearly."
            ),
            "intent": primary_intent,
            "intents": intents,
            "disease": None,
            "diseases": [],
            "source": {
                "unexpected_mentions": unexpected_mentions,
            },
            "match_type": "scope_rejected",
            "response_format": response_format,
            "reasoning_mode": "clarification_needed",
            "style_mode": "clarification_only",
            "needs_clarification": True,
            "requires_reasoning": requires_reasoning,
            "supporting_contexts": [],
        }

    first_record = (used_matches[0].get("record") if used_matches else None) or {}
    source_name = str(first_record.get("source_name") or "").strip()
    source_url = str(first_record.get("source_url") or "").strip()

    if _contains_emergency_marker(query, primary_intent):
        answer = (
            f"{answer}\n\n"
            "If symptoms are severe, worsening, or include emergency warning signs, seek urgent in-person medical care immediately."
        ).strip()
        answer = _sanitize_user_facing_answer(answer)

    supporting_contexts = _build_supporting_contexts(used_matches, intents, response_format)

    return {
        "found": True,
        "confidence": confidence,
        "answer": answer,
        "intent": primary_intent,
        "intents": intents,
        "disease": diseases[0] if diseases else None,
        "diseases": diseases,
        "source": {"name": source_name, "url": source_url} if source_name or source_url else None,
        "match_type": match_type,
        "response_format": response_format,
        "reasoning_mode": reasoning_mode,
        "style_mode": style_mode,
        "needs_clarification": needs_clarification,
        "requires_reasoning": requires_reasoning,
        "used_history_context": bool(plan.get("used_history_context")),
        "supporting_contexts": supporting_contexts,
    }
