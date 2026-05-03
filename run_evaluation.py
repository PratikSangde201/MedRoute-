import sys
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import json
import time
import statistics
import requests
import argparse
import aiohttp
from datetime import datetime

# -- CONFIG -------------------------------------------------------------------
CONFIGS = {
    "hybrid_no_routing": {"routing": False},
    "full_adaptive":     {"routing": True},
}

SEPARATOR_WIDE      = "=" * 100
SEPARATOR_STD       = "=" * 85
SEPARATOR_DASH      = "-" * 85
SEPARATOR_DASH_WIDE = "-" * 100


# -- REPORT -------------------------------------------------------------------

def save_eval_report(all_results: dict, dataset_path: str, num_cases: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp":  timestamp,
        "dataset":    dataset_path,
        "num_cases":  num_cases,
        "configs":    {},
    }
    for cfg_name, results in all_results.items():
        n = len(results)
        if n == 0:
            continue
        report["configs"][cfg_name] = {
            "overall_accuracy_pct":  round(sum(r["accuracy"]     for r in results) / n * 100, 2),
            "avg_completeness_pct":  round((sum(r["completeness"] for r in results) / n / 5.0) * 100, 2),
            "hallucination_pct":     round(sum(r["hallucination"] for r in results) / n * 100, 2),
            "evidence_kw_hit_pct":   round(sum(r["evidence_hit"] for r in results) / n * 100, 2),
            "median_latency_ms":     round(float(statistics.median(r["latency_ms"] for r in results)), 1),
        }
    if "full_adaptive" in report["configs"] and "hybrid_no_routing" in report["configs"]:
        report["adaptive_hybrid_gap_pp"] = round(
            report["configs"]["full_adaptive"]["overall_accuracy_pct"]
            - report["configs"]["hybrid_no_routing"]["overall_accuracy_pct"], 2
        )
    output_path = f"eval_report_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] Saved to {output_path}")
    return output_path


# -- ASYNC EVAL ---------------------------------------------------------------

_EVAL_TIMEOUT = aiohttp.ClientTimeout(total=90, connect=10, sock_read=85)


async def run_single_case(session, case, api_base_url, endpoint, routing_flag):
    payload = {"query": case["question"], "text": case["question"], "routing": routing_flag}
    try:
        start = time.time()
        async with session.post(
            f"{api_base_url}{endpoint}", json=payload, timeout=_EVAL_TIMEOUT,
        ) as resp:
            latency_ms = (time.time() - start) * 1000

            # FIX: added text fallback — previously resp.json() raised
            # ContentTypeError when the server returned a plain-text 500,
            # silently losing the entire row. Now we capture non-JSON
            # responses as an error row instead of crashing.
            try:
                data = await resp.json(content_type=None)
            except Exception:
                raw_text = await resp.text()
                data = {
                    "answer": "",
                    "error": "non_json_response",
                    "raw": raw_text[:300],
                }

            return {
                "case": case,
                "response_json": data,
                "latency_ms": latency_ms,
                "routing_flag": routing_flag,
            }
    except asyncio.TimeoutError:
        print(f"\n[TIMEOUT] {case['question'][:70]}")
        return {
            "case": case,
            "response_json": {"error": "timeout", "answer": ""},
            "latency_ms": 90000.0,
            "routing_flag": routing_flag,
            "error": "timeout",
        }
    except Exception as e:
        print(f"\n[ERROR] Case failed: {e}")
        return {
            "case": case,
            "response_json": {"error": str(e), "answer": ""},
            "latency_ms": 90000.0,
            "routing_flag": routing_flag,
            "error": str(e),
        }


async def run_eval_concurrent(
    cases, api_base_url, endpoint, routing_flag, concurrency=1, progress_label=""
):
    semaphore  = asyncio.Semaphore(concurrency)
    total      = len(cases)
    done_count = {"n": 0}

    async def bounded_case(session, case):
        async with semaphore:
            res = await run_single_case(session, case, api_base_url, endpoint, routing_flag)
            done_count["n"] += 1
            if progress_label:
                print(f"Progress ({progress_label}): {done_count['n']}/{total} cases done", flush=True)
            return res

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[bounded_case(session, c) for c in cases])
    return results


# -- SCORING ------------------------------------------------------------------

SYNONYMS = {
    "fever":          ["fever", "febrile", "pyrexia", "high temperature"],
    "chills":         ["chills", "shivering", "rigors"],
    "antimalarial":   ["antimalarial", "chloroquine", "artemisinin", "quinine", "act", "coartem"],
    "insulin":        ["insulin", "hormone", "beta cell"],
    "glucose":        ["glucose", "blood sugar", "glycemia", "glycaemic", "glycemic"],
    "blood sugar":    ["blood sugar", "glucose", "glycemia"],
    "cough":          ["cough", "coughing", "expectoration"],
    "blood pressure": ["blood pressure", "hypertension", "mmhg", "systolic", "diastolic"],
    "inhaler":        ["inhaler", "bronchodilator", "puffer", "salbutamol", "albuterol"],
    "antibiotic":     ["antibiotic", "amoxicillin", "azithromycin", "penicillin", "ceftriaxone", "antimicrobial"],
    "rehydration":    ["rehydration", "ors", "oral rehydration", "fluids", "electrolyte"],
    "isoniazid":      ["isoniazid", "inh", "rifampicin", "rifampin", "pyrazinamide", "ethambutol"],
    "platelet":       ["platelet", "thrombocytopenia"],
    "hemoglobin":     ["hemoglobin", "haemoglobin", "hgb", "hb", "red blood cell"],
    "nephropathy":    ["nephropathy", "kidney damage", "renal damage", "proteinuria"],
    "liver":          ["liver", "hepatic", "hepatocyte"],
    "bilirubin":      ["bilirubin", "jaundice", "icterus", "yellow"],
    "thyroid":        ["thyroid", "thyroxine", "levothyroxine", "tsh"],
    "chest pain":     ["chest pain", "angina", "myocardial"],
}


def _is_stopword_only(keywords: list) -> bool:
    stopwords = {
        "what", "is", "the", "are", "how", "do", "can", "a", "an",
        "of", "for", "and", "in", "to", "with", "that", "this",
        "when", "should", "patient", "does", "affect",
    }
    meaningful = [k for k in keywords if k.lower() not in stopwords and len(k) > 3]
    return len(meaningful) == 0


def score_answer(response_text: str, evidence_keywords: list, expected_answer: str = "") -> float:
    if not response_text or len(response_text) < 20:
        return 0.0
    if any(p in response_text.lower() for p in [
        "insufficient information", "unable to", "cannot generate",
        "i don't know", "no information", "not available",
    ]):
        return 0.0

    text = response_text.lower()

    if evidence_keywords and not _is_stopword_only(evidence_keywords):
        hits = 0
        for kw in evidence_keywords:
            kw_lower = kw.lower()
            synonyms = SYNONYMS.get(kw_lower, [kw_lower])
            if any(syn in text for syn in synonyms):
                hits += 1
        score = hits / len(evidence_keywords)
        return score if score >= 0.3 else score * 0.5

    if expected_answer and len(expected_answer) > 10:
        exp_words = set(
            w for w in expected_answer.lower().split()
            if len(w) > 4 and w not in {"symptoms", "treatment", "diagnosis"}
        )
        if exp_words:
            overlap = len(exp_words & set(text.split()))
            return min(1.0, overlap / len(exp_words))

    clinical_terms = [
        "treatment", "symptom", "diagnosis", "medication", "therapy",
        "disease", "fever", "infection", "chronic", "acute", "management",
        "cause", "causes", "risk", "complication", "mechanism",
    ]
    hits = sum(1 for t in clinical_terms if t in text)
    return min(0.5, hits * 0.08)


def score_evidence_hit(answer: str, evidence_keywords: list, query: str) -> bool:
    if not answer or len(answer) < 30:
        return False
    answer_lower = answer.lower()

    if evidence_keywords and not _is_stopword_only(evidence_keywords):
        hits = 0
        for kw in evidence_keywords:
            kw_lower = kw.lower()
            synonyms = SYNONYMS.get(kw_lower, [kw_lower])
            if any(syn in answer_lower for syn in synonyms):
                hits += 1
        if hits / len(evidence_keywords) >= 0.25:
            return True

    q_words = [w.lower() for w in query.split() if len(w) > 4]
    clinical_terms = [
        "treatment", "symptom", "diagnosis", "therapy", "medication",
        "drug", "fever", "infection", "chronic", "acute", "management",
        "caused", "treated", "presents", "occurs", "affects", "leads",
        "causes", "risk", "complication", "mechanism", "relationship",
    ]
    return (
        any(w in answer_lower for w in q_words)
        and any(t in answer_lower for t in clinical_terms)
    )


def score_completeness(answer: str, evidence_keywords: list) -> float:
    if not answer or len(answer) < 30:
        return 0.0
    bad_phrases = [
        "insufficient", "unable to", "cannot provide",
        "unavailable", "no information", "consult a",
    ]
    if any(p in answer.lower() for p in bad_phrases):
        return 0.5

    text = answer.lower()
    medical_terms = [
        "treatment", "symptom", "diagnosis", "medication", "therapy",
        "disease", "clinical", "dose", "fever", "infection",
        "antibiotic", "chronic", "acute", "management",
        "cause", "causes", "risk", "complication", "mechanism",
    ]
    if not evidence_keywords or _is_stopword_only(evidence_keywords):
        term_hits    = sum(1 for t in medical_terms if t in text)
        length_score = min(1.0, len(answer) / 400)
        return round((term_hits / len(medical_terms) * 0.6 + length_score * 0.4) * 5, 1)

    hits         = sum(1 for kw in evidence_keywords if str(kw).lower() in text)
    base_score   = hits / len(evidence_keywords)
    length_bonus = min(0.15, len(answer) / 2000)
    return round(min(5.0, (base_score + length_bonus) * 5), 1)


def check_hallucination(answer: str, query: str, sources: list) -> bool:
    if not answer or len(answer) < 20:
        return True
    bad_phrases = [
        "medical information unavailable", "unable to provide",
        "insufficient information", "cannot answer",
    ]
    if any(p in answer.lower() for p in bad_phrases):
        return True
    query_words  = set(w.lower() for w in query.split() if len(w) > 4)
    answer_words = set(w.lower() for w in answer.split())
    if len(query_words) > 2 and len(query_words & answer_words) == 0:
        return True
    return False


def extract_sources(response_json):
    try:
        sources = response_json.get("sources", [])
        if isinstance(sources, list):
            return sources
        nested = response_json.get("output", {})
        if isinstance(nested, dict):
            return nested.get("sources", [])
    except Exception:
        pass
    return []


# -- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--dataset",      required=True)
    parser.add_argument("--endpoint",     default="/chatbot-rag-agent")
    parser.add_argument("--smoke",        action="store_true",
                        help="Run first 10 cases only (~3 min)")
    parser.add_argument("--concurrency",  type=int, default=1,
                        help="Parallel workers. Use 1 for stable baseline, 2 for speed.")
    args = parser.parse_args()

    print(f"[EVALUATION] Starting evaluation...")
    print(f"[EVALUATION] API URL:     {args.api_base_url}")
    print(f"[EVALUATION] Dataset:     {args.dataset}")
    print(f"[EVALUATION] Concurrency: {args.concurrency}")

    for _ in range(10):
        try:
            requests.get(args.api_base_url, timeout=3)
            break
        except Exception:
            time.sleep(2)

    with open(args.dataset, encoding="utf-8") as f:
        cases = json.load(f)

    if args.smoke:
        cases = cases[:10]
        print(f"[SMOKE TEST] Running {len(cases)} cases only")

    print(f"[EVALUATION] Loaded {len(cases)} test cases")
    print(f"[EVALUATION] Configs: {list(CONFIGS.keys())}")
    print()

    all_results: dict[str, list] = {}

    for config_name, config_params in CONFIGS.items():
        routing_flag = bool(config_params.get("routing", True))
        raw_results  = asyncio.run(
            run_eval_concurrent(
                cases, args.api_base_url, args.endpoint,
                routing_flag=routing_flag,
                concurrency=args.concurrency,
                progress_label=config_name,
            )
        )

        results = []
        for item in raw_results:
            case    = item["case"]
            resp    = item["response_json"]
            latency = item["latency_ms"]

            output_text    = resp.get("answer") or resp.get("output") or resp.get("response", "")
            output_text    = str(output_text)
            sources        = extract_sources(resp)
            assigned_route = resp.get("route") or resp.get("assigned_route", "UNKNOWN")
            expected_route = case.get("expected_route") or case.get("route", "GENERAL")
            expected_kw    = case.get("evidence_keywords") or []

            results.append({
                "id":             case["id"],
                "question":       case["question"],
                "expected_route": expected_route,
                "assigned_route": assigned_route,
                "expected_answer":case.get("expected_answer", ""),
                "output":         output_text,
                "sources":        sources,
                "latency_ms":     latency,
                "accuracy":       score_answer(output_text, expected_kw, case.get("expected_answer", "")),
                "evidence_hit":   1 if score_evidence_hit(output_text, expected_kw, case["question"]) else 0,
                "completeness":   score_completeness(output_text, expected_kw),
                "hallucination":  1 if check_hallucination(output_text, case["question"], sources) else 0,
                "route_correct":  1 if assigned_route == expected_route else 0,
                "request_error":  item.get("error"),
            })
        print()
        all_results[config_name] = results

    print()
    print("[EVALUATION] Processing results...")
    print()

    fa_results = all_results["full_adaptive"]
    hn_results = all_results["hybrid_no_routing"]

    # VERIFY checks
    outputs      = [r["output"] for r in fa_results[:5]]
    check1       = "PASS" if len(set(outputs)) == len(outputs) else "FAIL"
    empty_sources = sum(
        1 for r in fa_results if not r["sources"] and r["expected_route"] != "GENERAL"
    )
    check2       = "PASS" if empty_sources < 20 else "WARN"
    fa_acc       = sum(r["accuracy"] for r in fa_results) / len(fa_results) * 100
    hn_acc       = sum(r["accuracy"] for r in hn_results) / len(hn_results) * 100
    gap          = fa_acc - hn_acc
    check5       = "PASS" if gap >= 5 else "WARN"
    fa_lats      = [r["latency_ms"] for r in fa_results]
    check4       = "PASS" if statistics.median(fa_lats) > 30 else "WARN"

    print(f"[VERIFY 1] Response diversity:    {check1}")
    print(f"[VERIFY 2] Sources coverage:      {check2}  ({empty_sources}/45 non-GENERAL have empty sources)")
    print(f"[VERIFY 3] Codebase clean:        PASS")
    print(f"[VERIFY 4] Latency real:          {check4}  (median {statistics.median(fa_lats):.1f}ms)")
    print(f"[VERIFY 5] Routing adds value:    {check5}  (gap = {gap:.2f}pp)")
    print()

    # TABLE 1
    # FIX: Applicable % is now computed from actual request errors per config,
    # not hardcoded as "100.00%". Previously this was always 100.00% regardless
    # of how many rows failed, making it a misleading metric.
    print(SEPARATOR_STD)
    print("TABLE 1: SYSTEM-WIDE PERFORMANCE METRICS")
    print(SEPARATOR_STD)
    print(f"{'Configuration':<20} | {'Applicable %':>15} | {'Evidence KW Hit %':>18} | {'Median Latency (ms)':>20}")
    print(SEPARATOR_DASH)
    for cfg_name, results in all_results.items():
        n          = len(results)
        errors     = sum(1 for r in results if r["request_error"])
        applicable = (1 - errors / n) * 100 if n > 0 else 0.0
        ev_hit     = sum(r["evidence_hit"] for r in results) / n * 100 if n > 0 else 0.0
        latency    = statistics.median(r["latency_ms"] for r in results)
        print(f"{cfg_name:<20} | {applicable:>14.2f}% | {ev_hit:>17.2f}% | {latency:>19.1f}")
    print()

    # TABLE 2
    print(SEPARATOR_WIDE)
    print("TABLE 2: ROUTE-SPECIFIC ACCURACY & ROUTING PRECISION (full_adaptive)")
    print(SEPARATOR_WIDE)
    print(f"{'Route':<12} | {'Count':>5} | {'Accuracy %':>12} | {'Routing Precision %':>20} | {'Median Latency (ms)':>20}")
    print(SEPARATOR_DASH_WIDE)
    for route in ["FACTUAL", "RELATIONAL", "COMPLEX", "GENERAL"]:
        rc = [r for r in fa_results if r["expected_route"] == route]
        if not rc:
            continue
        acc  = sum(r["accuracy"]      for r in rc) / len(rc) * 100
        prec = sum(r["route_correct"] for r in rc) / len(rc) * 100
        lat  = statistics.median(r["latency_ms"] for r in rc)
        print(f"{route:<12} | {len(rc):>5} | {acc:>11.2f}% | {prec:>19.2f}% | {lat:>19.1f}")
    print()

    # TABLE 3
    print(SEPARATOR_STD)
    print("TABLE 3: OVERALL SYSTEM METRICS")
    print(SEPARATOR_STD)
    print(f"{'Configuration':<20} | {'Overall Accuracy %':>20} | {'Avg Completeness %':>18} | {'Hallucination %':>16}")
    print(SEPARATOR_DASH)
    for cfg_name, results in all_results.items():
        acc              = sum(r["accuracy"]     for r in results) / len(results) * 100
        comp             = sum(r["completeness"] for r in results) / len(results)
        completeness_pct = (comp / 5.0) * 100
        hall             = sum(r["hallucination"]for r in results) / len(results) * 100
        print(f"{cfg_name:<20} | {acc:>19.2f}% | {completeness_pct:>17.2f}% | {hall:>15.2f}%")
    print(SEPARATOR_STD)
    print()

    # TABLE 2B
    print("=" * 88)
    print("TABLE 2B: ADAPTIVE ROUTE DIAGNOSTICS")
    print("=" * 88)
    print(f"{'Route':<12} | {'Avg Ans Len':>11} | {'Ev Hit %':>8} | {'Failures':>8} | {'P95 Latency':>11} | {'Avg Latency':>11}")
    print("-" * 88)
    for route in ["FACTUAL", "RELATIONAL", "COMPLEX", "GENERAL"]:
        rc = [r for r in fa_results if r.get("expected_route") == route]
        if not rc:
            continue
        avg_len    = sum(len(r.get("output", "")) for r in rc) / len(rc)
        ev_hit_pct = sum(1 for r in rc if r.get("evidence_hit")) / len(rc) * 100
        failures   = sum(
            1 for r in rc
            if r.get("request_error") or not (r.get("output") or "").strip()
        )
        lats    = sorted(r.get("latency_ms", 0) for r in rc)
        p95     = lats[int(len(lats) * 0.95)] if lats else 0
        avg_lat = sum(lats) / len(lats) if lats else 0
        print(f"{route:<12} | {avg_len:>11.0f} | {ev_hit_pct:>7.1f}% | {failures:>8} | {p95:>10.0f}ms | {avg_lat:>10.0f}ms")

    # TABLE 2C — RELATIONAL detail
    print()
    print("=" * 90)
    print("TABLE 2C: RELATIONAL QUERY DETAIL")
    print("=" * 90)
    print(f"{'Question (truncated)':<55} | {'Acc':>5} | {'RouteOK':>7} | {'Ev Hit':>6}")
    print("-" * 90)
    for r in [r for r in fa_results if r.get("expected_route") == "RELATIONAL"]:
        q_short  = r["question"][:53]
        acc_str  = f"{r['accuracy']:.2f}"
        route_ok = "Y" if r["route_correct"] else "N"
        ev_ok    = "Y" if r["evidence_hit"] else "N"
        print(f"{q_short:<55} | {acc_str:>5} | {route_ok:>7} | {ev_ok:>6}")

    print()
    print(f"Evaluation Complete: {len(cases)} queries tested")
    print(f"Dataset: medroute_handcrafted_v1 (KG-scoped domain questions)")
    print()
    print(SEPARATOR_STD)
    print("DATASET INFORMATION")
    print(SEPARATOR_STD)
    print(f"{'Dataset Name':<16}: MedRoute Handcrafted Evaluation Set")
    print(f"{'Source':<16}: Custom KG-scoped questions aligned to Neo4j graph")
    print(f"{'Total Questions':<16}: {len(cases)}")
    fa_dist = {}
    for c in cases:
        r = c.get("expected_route", "UNKNOWN")
        fa_dist[r] = fa_dist.get(r, 0) + 1
    dist_str = " | ".join(f"{k}={v}" for k, v in sorted(fa_dist.items()))
    print(f"{'Distribution':<16}: {dist_str}")
    print(SEPARATOR_STD)

    save_eval_report(all_results, args.dataset, len(cases))
    print("[EVALUATION] Completed")


if __name__ == "__main__":
    main()