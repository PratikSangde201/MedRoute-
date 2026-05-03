import re
from threading import Lock
from typing import Any


# Phrase-to-medical-term normalization applied before all reasoning.
_SYMPTOM_SYNONYMS: dict[str, str] = {
    # Cardiometabolic / endocrine
    "heart racing": "palpitations",
    "rapid heartbeat": "palpitations",
    "fast heartbeat": "palpitations",
    "pounding heart": "palpitations",
    "heart is racing": "palpitations",
    "heart pounding": "palpitations",
    "palpitation": "palpitations",
    "shaky hands": "tremor",
    "tremors": "tremor",
    "feeling hot": "heat intolerance",
    "feeling very hot": "heat intolerance",
    "cannot tolerate heat": "heat intolerance",
    "breathlessness": "shortness of breath",
    "breathing difficulty": "shortness of breath",
    "extremely thirsty": "polydipsia",
    "always thirsty": "polydipsia",
    "thirsty all the time": "polydipsia",
    "very thirsty": "polydipsia",
    "frequent urination": "polyuria",
    "urinating frequently": "polyuria",
    "peeing often": "polyuria",
    "frequent peeing": "polyuria",
    "peeing a lot": "polyuria",
    "blurred eyesight": "blurred vision",
    # Cardiac emergency
    "left arm discomfort": "left arm pain",
    "left arm heaviness": "left arm pain",
    "left shoulder pain": "left arm pain",
    "chest tightness": "chest pain",
    "chest discomfort": "chest pain",
    # Mental health
    "sad": "persistent sadness",
    "feeling sad": "persistent sadness",
    "sad all the time": "persistent sadness",
    "empty": "persistent sadness",
    "no interest": "loss of interest",
    "not interested": "loss of interest",
    "not interested in anything": "loss of interest",
    "low motivation": "loss of interest",
    "low mood": "persistent sadness",
    "feeling down": "persistent sadness",
    "hopeless": "persistent sadness",
    "no energy": "low energy",
    "very tired": "low energy",
    "sleep issues": "sleep disturbance",
    "sleep problems": "sleep disturbance",
    "poor sleep": "sleep disturbance",
    "can't sleep": "sleep disturbance",
    "cant sleep": "sleep disturbance",
    "sleep too much": "sleep disturbance",
    "insomnia": "sleep disturbance",
    "can't concentrate": "poor concentration",
    "cant concentrate": "poor concentration",
    "poor concentration": "poor concentration",
}

# High-confidence clusters.
_STRONG_CLUSTERS: list[dict[str, Any]] = [
    {
        "condition": "Hyperthyroidism",
        "symptoms": {"palpitations", "tremor", "heat intolerance"},
        "min_match": 2,
        "reason": "This cluster is commonly associated with an overactive thyroid state.",
        "tests": ["thyroid profile (TSH, Free T4, Free T3)", "thyroid antibody panel"],
    },
    {
        "condition": "Diabetes",
        "symptoms": {"polydipsia", "polyuria", "fatigue", "blurred vision"},
        "min_match": 2,
        "reason": "This pattern is strongly suggestive of blood sugar dysregulation.",
        "tests": ["fasting glucose", "HbA1c", "urinalysis"],
    },
    {
        "condition": "Myocardial Infarction",
        "symptoms": {"chest pain", "left arm pain", "shortness of breath"},
        "min_match": 2,
        "reason": "This pattern is concerning for acute coronary syndrome and needs urgent assessment.",
        "tests": ["ECG", "troponin", "urgent emergency evaluation"],
    },
    {
        "condition": "Major Depressive Disorder",
        "symptoms": {
            "persistent sadness",
            "low energy",
            "sleep disturbance",
            "loss of interest",
            "poor concentration",
        },
        "min_match": 3,
        "reason": "This cluster strongly matches a major depressive disorder pattern.",
        "tests": ["clinical mental health assessment", "screening questionnaire (PHQ-9)"],
    },
]

# Broader ranking rules for secondary possibilities.
_DISEASE_RULES: list[dict[str, Any]] = [
    {
        "condition": "Anxiety Disorder",
        "symptoms": {"palpitations", "tremor", "sleep disturbance", "shortness of breath"},
        "reason": "Autonomic activation can produce palpitations, tremor, poor sleep, and breathlessness.",
        "tests": ["clinical assessment", "basic metabolic panel"],
    },
    {
        "condition": "Hyperthyroidism",
        "symptoms": {"palpitations", "tremor", "heat intolerance", "low energy"},
        "reason": "Hypermetabolic state can cause tremor, heat intolerance, and cardiovascular symptoms.",
        "tests": ["thyroid profile"],
    },
    {
        "condition": "Diabetes",
        "symptoms": {"polydipsia", "polyuria", "fatigue", "blurred vision"},
        "reason": "Glucose imbalance often presents with thirst, frequent urination, fatigue, and visual blurring.",
        "tests": ["fasting glucose", "HbA1c"],
    },
    {
        "condition": "Major Depressive Disorder",
        "symptoms": {
            "persistent sadness",
            "loss of interest",
            "sleep disturbance",
            "low energy",
            "poor concentration",
        },
        "reason": "Mood, sleep, energy, and concentration changes can indicate a depressive syndrome.",
        "tests": ["clinical interview", "PHQ-9"],
    },
]

_EMERGENCY_CLUSTER = {"chest pain", "left arm pain", "shortness of breath"}

_EMPATHY_LINES = [
    "I understand your concern, and you did the right thing by asking.",
    "That sounds difficult, and it is important to take these symptoms seriously.",
    "I can see why this worries you, and I will walk you through this clearly.",
]

_SUMMARY_LINES = [
    "From what you described, the key symptoms are {symptoms_text}.",
    "Based on your message, the main symptoms are {symptoms_text}.",
    "The symptom pattern I am seeing includes {symptoms_text}.",
]

_REASONING_LINES = [
    "This is based on how strongly your symptoms match known clinical patterns.",
    "I ranked these conditions by the number of matching symptoms.",
    "The ranking reflects symptom-pattern overlap rather than a final diagnosis.",
]

_ADVICE_LINES = [
    "Until you are evaluated, keep good hydration, reduce caffeine, and maintain regular sleep.",
    "For now, focus on fluids, avoid excess caffeine, and keep a stable sleep routine.",
    "As supportive care, prioritize hydration, better sleep hygiene, and lower stimulant intake.",
]

_MENTAL_SUPPORT_LINES = [
    "Try to keep a simple daily routine, maintain sleep timing, and stay connected with someone you trust.",
    "Please consider reaching out to a mental health professional; early support often improves recovery.",
    "If you feel unsafe or have thoughts of self-harm, contact local emergency services or a crisis helpline immediately.",
]

_DISCLAIMER_LINES = [
    "This guidance is informational and does not replace an in-person medical diagnosis.",
    "I cannot confirm a final diagnosis here, so please follow up with a clinician.",
    "Please treat this as decision support and seek direct medical evaluation.",
]

_FOLLOWUP_QUESTIONS_BY_DOMAIN: dict[str, list[str]] = {
    "thyroid": [
        "Have you noticed recent weight loss, increased sweating, or heat intolerance?",
        "Do your palpitations occur at rest or mostly during stress/activity?",
    ],
    "diabetes": [
        "How long have increased thirst and frequent urination been present?",
        "Have you noticed recent weight change or blurry vision at specific times of day?",
    ],
    "cardiac": [
        "Does the chest pain worsen with exertion or radiate to jaw/back?",
        "Do you also have nausea, sweating, or lightheadedness with the chest pain?",
    ],
    "mental": [
        "For how long have low mood and loss of interest been affecting your daily routine?",
        "Has sleep disturbance been early waking, difficulty falling asleep, or both?",
    ],
    "general": [
        "When did these symptoms start, and are they getting worse or stable?",
        "Do you have fever, severe pain, or breathing difficulty with these symptoms?",
    ],
}

_TURN_COUNTER = 0
_TURN_LOCK = Lock()
_MEMORY_LOCK = Lock()
_SESSION_SYMPTOM_MEMORY: dict[str, list[str]] = {}


def _next_turn() -> int:
    global _TURN_COUNTER
    with _TURN_LOCK:
        _TURN_COUNTER += 1
        return _TURN_COUNTER


def _pick(options: list[str], seed: str, turn: int, offset: int = 0) -> str:
    if not options:
        return ""
    index = (sum(ord(c) for c in seed) + turn + offset) % len(options)
    return options[index]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        val = item.strip().lower()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
    return bool(re.search(pattern, text))


def _find_phrase_positions(text: str, phrase: str) -> list[int]:
    pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
    return [match.start() for match in re.finditer(pattern, text)]


def normalizeSymptoms(input_text: str) -> list[str]:
    text = (input_text or "").lower()
    if not text:
        return []

    found_hits: list[tuple[int, str]] = []

    # Phrase normalization first.
    for phrase, canonical in _SYMPTOM_SYNONYMS.items():
        for position in _find_phrase_positions(text, phrase):
            found_hits.append((position, canonical))

    # Also detect canonical terms directly.
    canonical_terms = set(_SYMPTOM_SYNONYMS.values())
    for term in canonical_terms:
        for position in _find_phrase_positions(text, term):
            found_hits.append((position, term))

    found_hits.sort(key=lambda item: item[0])
    ordered = [canonical for _, canonical in found_hits]

    return _dedupe_keep_order(ordered)


def extractSymptoms(inputText: str) -> list[str]:
    return normalizeSymptoms(inputText)


def mergeSessionSymptoms(session_key: str, symptoms: list[str], max_memory: int = 24) -> list[str]:
    key = session_key or "default"
    normalized = _dedupe_keep_order(symptoms)

    with _MEMORY_LOCK:
        existing = list(_SESSION_SYMPTOM_MEMORY.get(key, []))
        merged = _dedupe_keep_order(existing + normalized)
        _SESSION_SYMPTOM_MEMORY[key] = merged[-max_memory:]
        return list(_SESSION_SYMPTOM_MEMORY[key])


def detectSeverity(symptoms: list[str]) -> dict[str, Any]:
    symptom_set = {s.lower() for s in symptoms}
    emergency = _EMERGENCY_CLUSTER.issubset(symptom_set)
    return {
        "is_emergency": emergency,
        "trigger": "Myocardial Infarction" if emergency else "",
        "warning": "This may be a medical emergency consistent with possible heart attack (Myocardial Infarction). Seek immediate emergency care now." if emergency else "",
    }


def _cluster_matches(symptom_set: set[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for cluster in _STRONG_CLUSTERS:
        matched = sorted(symptom_set.intersection(cluster["symptoms"]))
        if len(matched) >= int(cluster["min_match"]):
            hits.append(
                {
                    "condition": cluster["condition"],
                    "match_count": len(matched),
                    "matched_symptoms": matched,
                    "reason": cluster["reason"],
                    "tests": list(cluster["tests"]),
                    "high_confidence": len(matched) >= 3,
                }
            )
    hits.sort(key=lambda x: x["match_count"], reverse=True)
    return hits


def rankDiseases(symptoms: list[str]) -> list[dict[str, Any]]:
    symptom_set = {s.lower() for s in symptoms if s.strip()}
    if not symptom_set:
        return []

    ranked: list[dict[str, Any]] = []

    # High-confidence clusters first.
    ranked.extend(_cluster_matches(symptom_set))

    existing = {r["condition"] for r in ranked}
    for rule in _DISEASE_RULES:
        if rule["condition"] in existing:
            continue
        matched = sorted(symptom_set.intersection(rule["symptoms"]))
        if len(matched) < 2:
            continue
        ranked.append(
            {
                "condition": rule["condition"],
                "match_count": len(matched),
                "matched_symptoms": matched,
                "reason": rule["reason"],
                "tests": list(rule["tests"]),
                "high_confidence": len(matched) >= 3,
            }
        )

    ranked.sort(key=lambda r: r["match_count"], reverse=True)
    return ranked[:2]


def _score_conditions(symptom_set: set[str]) -> dict[str, int]:
    scores: dict[str, int] = {}

    for cluster in _STRONG_CLUSTERS:
        condition = str(cluster["condition"])
        overlap = symptom_set.intersection(cluster["symptoms"])
        if overlap:
            scores[condition] = max(scores.get(condition, 0), len(overlap))

    for rule in _DISEASE_RULES:
        condition = str(rule["condition"])
        overlap = symptom_set.intersection(rule["symptoms"])
        if overlap:
            scores[condition] = max(scores.get(condition, 0), len(overlap))

    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def _confidence_label(best_score: int) -> str:
    if best_score >= 3:
        return "HIGH"
    if best_score == 2:
        return "MODERATE"
    return "LOW"


def inferConditions(symptoms: list[str], inputText: str = "") -> dict[str, Any]:
    symptom_set = {s.lower() for s in symptoms if s.strip()}
    ranked = rankDiseases(symptoms)

    disease_scores = _score_conditions(symptom_set)

    best_match = ""
    best_score = 0
    if disease_scores:
        best_match, best_score = next(iter(disease_scores.items()))
    elif ranked:
        best_match = str(ranked[0].get("condition") or "")
        best_score = int(ranked[0].get("match_count", 0))

    confidence_label = _confidence_label(best_score)

    # Keep uncertain conservative for weak overlap.
    uncertain = len(symptom_set) < 2 or best_score < 2
    confidence_value = 0.9 if confidence_label == "HIGH" else (0.7 if confidence_label == "MODERATE" else 0.45)

    return {
        "ranked": ranked,
        "conditions": [r["condition"] for r in ranked],
        "matched_symptoms": list(symptom_set),
        "tests": ranked[0].get("tests", []) if ranked else ["clinical examination", "basic blood tests"],
        "confidence": confidence_value,
        "confidence_label": confidence_label,
        "best_match": best_match,
        "best_score": best_score,
        "disease_scores": disease_scores,
        "uncertain": uncertain,
    }


def fallbackDiagnosis(symptoms: list[str]) -> dict[str, Any]:
    return inferConditions(symptoms)


def _select_followups(symptoms: list[str], ranked: list[dict[str, Any]]) -> list[str]:
    if ranked:
        top = ranked[0]["condition"].lower()
        if "thyroid" in top:
            return _FOLLOWUP_QUESTIONS_BY_DOMAIN["thyroid"][:2]
        if "diabetes" in top:
            return _FOLLOWUP_QUESTIONS_BY_DOMAIN["diabetes"][:2]
        if "depress" in top:
            return _FOLLOWUP_QUESTIONS_BY_DOMAIN["mental"][:2]
        if "infarction" in top:
            return _FOLLOWUP_QUESTIONS_BY_DOMAIN["cardiac"][:2]

    symptom_set = {s.lower() for s in symptoms}
    if {"chest pain", "shortness of breath"}.intersection(symptom_set):
        return _FOLLOWUP_QUESTIONS_BY_DOMAIN["cardiac"][:2]
    if {"persistent sadness", "loss of interest", "low energy"}.intersection(symptom_set):
        return _FOLLOWUP_QUESTIONS_BY_DOMAIN["mental"][:2]
    if {"polydipsia", "polyuria", "blurred vision"}.intersection(symptom_set):
        return _FOLLOWUP_QUESTIONS_BY_DOMAIN["diabetes"][:2]

    return _FOLLOWUP_QUESTIONS_BY_DOMAIN["general"][:2]


def _format_ranked_conditions(ranked: list[dict[str, Any]]) -> str:
    if not ranked:
        return "I cannot confidently prioritize a single condition yet."

    primary = ranked[0]["condition"]
    if len(ranked) == 1:
        return f"the most likely condition is {primary}"

    secondary = ranked[1]["condition"]
    return f"the most likely condition is {primary}, with {secondary} as another possibility"


def _format_followup_questions(questions: list[str]) -> str:
    if not questions:
        return "Could you share when these symptoms started and whether they are worsening?"
    if len(questions) == 1:
        return questions[0]
    return f"{questions[0]} Also, {questions[1]}"


def _remove_duplicate_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def buildResponse(data: dict[str, Any]) -> str:
    input_text = str(data.get("inputText") or "").strip()
    session_key = str(data.get("sessionKey") or "default")
    style_seed = str(data.get("styleSeed") or input_text or session_key)
    use_memory = bool(data.get("useMemory", True))
    provided_all_symptoms = data.get("allSymptoms")
    new_symptoms = list(data.get("symptoms") or normalizeSymptoms(input_text))
    if isinstance(provided_all_symptoms, list):
        all_symptoms = _dedupe_keep_order([str(item) for item in provided_all_symptoms])
    elif use_memory:
        all_symptoms = mergeSessionSymptoms(session_key, new_symptoms)
    else:
        all_symptoms = _dedupe_keep_order(new_symptoms)
    severity = detectSeverity(all_symptoms)
    diagnosis = dict(data.get("diagnosis") or inferConditions(all_symptoms, input_text))
    ranked = list(diagnosis.get("ranked") or [])

    turn = _next_turn()
    empathy = _pick(_EMPATHY_LINES, style_seed, turn, 0)

    symptoms_text = ", ".join(all_symptoms[:8]) if all_symptoms else "no clearly mapped symptoms yet"
    summary = _pick(_SUMMARY_LINES, style_seed, turn, 1).format(symptoms_text=symptoms_text)
    if use_memory and new_symptoms and len(all_symptoms) > len(new_symptoms):
        summary += " I also included your earlier symptoms to improve pattern matching."

    # Emergency path: no follow-up questions, no lifestyle advice.
    if severity["is_emergency"]:
        concern = "possible Myocardial Infarction"
        reasoning = "The combination of chest pain, left arm pain, and breathlessness is a high-risk emergency pattern."
        next_steps = "Call emergency services immediately or go to the nearest emergency department now. Request urgent ECG and troponin testing."
        disclaimer = "Do not delay care. Online guidance cannot safely rule out a heart attack."

        response_lines = _remove_duplicate_lines(
            [
                empathy,
                summary,
                f"This pattern is concerning for {concern}.",
                f"Why this is urgent: {reasoning}",
                f"What to do right now: {next_steps}",
                f"Safety note: {disclaimer}",
            ]
        )
        return "\n\n".join(response_lines)

    # Non-emergency path.
    disease_scores = dict(diagnosis.get("disease_scores") or {})
    if not disease_scores and ranked:
        for item in ranked:
            disease_scores[str(item["condition"])] = int(item.get("match_count", 0))

    best_match = str(diagnosis.get("best_match") or (ranked[0]["condition"] if ranked else "")).strip()
    best_score = int(diagnosis.get("best_score", disease_scores.get(best_match, 0)))
    confidence_label = str(diagnosis.get("confidence_label") or _confidence_label(best_score)).upper()

    ordered = sorted(disease_scores.items(), key=lambda kv: kv[1], reverse=True)
    alternatives = [name for name, _ in ordered if name != best_match][:2]
    symptom_phrase = ", ".join(all_symptoms[:6]) if all_symptoms else "the symptoms you described"

    if not best_match:
        return (
            f"Based on what you described - {symptom_phrase} - there is not enough symptom overlap "
            f"to prioritize a single condition yet. Confidence: LOW.\n\n"
            "Please consult a healthcare professional for personalized medical advice."
        )

    if confidence_label == "HIGH":
        return (
            f"Based on what you described - {symptom_phrase} - your symptoms closely match {best_match}. "
            "Confidence: HIGH.\n\n"
            "This is pattern-based guidance and not a confirmed diagnosis. "
            "Please consult a healthcare professional for personalized medical advice."
        )

    if confidence_label == "MODERATE":
        alt_text = ", ".join(alternatives) if alternatives else "other related conditions"
        return (
            f"Based on what you described - {symptom_phrase} - your symptoms are moderately consistent with {best_match}. "
            "Confidence: MODERATE.\n\n"
            f"Possible alternatives include {alt_text}. "
            "Please consult a healthcare professional for personalized medical advice."
        )

    possible_text = ", ".join([name for name, _ in ordered[:3]]) if ordered else "several conditions"
    return (
        f"Based on what you described - {symptom_phrase} - the pattern could fit multiple conditions. "
        "Confidence: LOW.\n\n"
        f"Possible matches include {possible_text}. "
        "Please consult a healthcare professional for personalized medical advice."
    )


def generateAdvice(symptoms: list[str]) -> str:
    # Compatibility wrapper (kept for callers that might import this function).
    return _pick(_ADVICE_LINES, "|".join(symptoms), _next_turn(), 0)


def generateFollowUps(symptoms: list[str]) -> list[str]:
    # Compatibility wrapper (kept for callers that might import this function).
    ranked = rankDiseases(symptoms)
    return _select_followups(symptoms, ranked)


def generateResponse(data: dict[str, Any]) -> str:
    return buildResponse(data)
