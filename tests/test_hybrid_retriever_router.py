import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
API_SRC = os.path.join(ROOT, "chatbot_api", "src")
RETRIEVAL_SRC = os.path.join(API_SRC, "retrieval")
if API_SRC not in sys.path:
    sys.path.insert(0, API_SRC)
if RETRIEVAL_SRC not in sys.path:
    sys.path.insert(0, RETRIEVAL_SRC)

from router_policy import parse_router_label, route_query  # noqa: E402

try:
    import hybrid_retriever as hr  # noqa: E402
    _HYBRID_TESTS_AVAILABLE = True
except Exception:
    hr = None
    _HYBRID_TESTS_AVAILABLE = False


class TestHybridRetrieverRouter(unittest.TestCase):
    def test_parse_router_label_strict_json(self):
        self.assertEqual(parse_router_label('{"label":"FACTUAL","reason":"fact"}'), "FACTUAL")

    def test_parse_router_label_fenced_json(self):
        raw = '```json\n{"label":"RELATIONAL","reason":"graph"}\n```'
        self.assertEqual(parse_router_label(raw), "RELATIONAL")

    def test_parse_router_label_prose_plus_json(self):
        raw = 'I think this is complex. {"label":"COMPLEX","reason":"multi-hop"}'
        self.assertEqual(parse_router_label(raw), "COMPLEX")

    def test_route_query_invalid_json_falls_back(self):
        try:
            from llm_openai_provider import ChatOpenAI  # noqa: F401
        except Exception:
            self.skipTest("LLM provider dependencies unavailable in this environment")

        with patch("llm_openai_provider.ChatOpenAI") as mock_chat_openai:
            mock_model = mock_chat_openai.return_value
            mock_response = type("Resp", (), {"content": "not-json"})()
            mock_model.invoke.return_value = mock_response

            # No deterministic override tokens -> FACTUAL fallback
            self.assertEqual(route_query("Define malaria."), "FACTUAL")

    def test_route_query_medical_facts_fast_path(self):
        self.assertEqual(route_query("What are symptoms of malaria?"), "RELATIONAL")

    def test_route_query_comparison_is_complex(self):
        self.assertEqual(route_query("Compare malaria and dengue symptoms"), "COMPLEX")

    def test_route_query_multi_intent_is_complex(self):
        self.assertEqual(route_query("What are symptoms and treatment of typhoid?"), "COMPLEX")

    def test_route_query_danger_ranking_is_complex(self):
        self.assertEqual(route_query("Which is more dangerous, dengue or malaria?"), "COMPLEX")

    def test_route_query_mechanism_with_disease_name_is_general(self):
        self.assertEqual(
            route_query("Why do diabetic patients develop nerve damage in their feet over time?"),
            "GENERAL",
        )

    def test_route_query_bp_numbers_difference_is_general(self):
        self.assertEqual(
            route_query("What is the difference between systolic and diastolic blood pressure and what do the numbers mean?"),
            "GENERAL",
        )

    def test_route_query_insulin_resistance_mechanism_is_general(self):
        self.assertEqual(
            route_query("How does insulin resistance develop in the body?"),
            "GENERAL",
        )


class TestHybridRetrieverSafety(unittest.TestCase):
    def test_prompts_enforce_exact_question_first(self):
        rule = "First answer the exact question asked by the user"
        if _HYBRID_TESTS_AVAILABLE:
            self.assertIn(rule, hr.SYNTHESIS_SYSTEM_PROMPT)
            self.assertIn(rule, hr.GENERAL_MEDICAL_SYSTEM_PROMPT)
            return

        file_text = Path(RETRIEVAL_SRC, "hybrid_retriever.py").read_text(encoding="utf-8")
        self.assertIn(rule, file_text)

    def test_general_medical_answer_hides_raw_model_errors(self):
        if not _HYBRID_TESTS_AVAILABLE:
            self.skipTest("hybrid retriever dependencies unavailable")
        fake_llm = type("FakeLLM", (), {"invoke": lambda self, _: (_ for _ in ()).throw(RuntimeError("404 model missing"))})()
        with patch.object(hr, "_get_llm", return_value=fake_llm):
            output = hr._general_medical_answer("Explain hypertension")
        self.assertIn("unable to generate", output.lower())
        self.assertNotIn("404", output)
        self.assertNotIn("model missing", output.lower())

    def test_relational_route_uses_graph_formatter_on_synthesis_failure(self):
        if not _HYBRID_TESTS_AVAILABLE:
            self.skipTest("hybrid retriever dependencies unavailable")
        with (
            patch.object(hr, "route_query", return_value="RELATIONAL"),
            patch.object(hr, "_graph_context", return_value="symptoms: chest pain\nprecautions: reduce salt"),
            patch.object(hr, "_synthesize_answer", side_effect=RuntimeError("Ollama call failed with status code 404")),
        ):
            result = hr.hybrid_retrieve("What are symptoms of hypertension?")

        answer = result.get("answer", "")
        self.assertIn("I could not run full synthesis right now", answer)
        self.assertIn("symptoms: chest pain", answer)
        self.assertNotIn("404", answer)
        self.assertNotIn("ollama", answer.lower())
        self.assertTrue(
            any("graph fallback formatter used" in step.lower() for step in result.get("steps", []))
        )

    def test_graph_context_does_not_include_exception_details(self):
        if not _HYBRID_TESTS_AVAILABLE:
            self.skipTest("hybrid retriever dependencies unavailable")
        with patch.object(hr.chatbot_cypher_chain, "invoke", side_effect=RuntimeError("socket timeout details")):
            output = hr._graph_context("What are symptoms of malaria?")
        self.assertEqual(output, "Graph retrieval unavailable.")


if __name__ == "__main__":
    unittest.main()
