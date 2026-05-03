import os
import sys
import unittest
import ast
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
API_SRC = os.path.join(ROOT, "chatbot_api", "src")
if API_SRC not in sys.path:
    sys.path.insert(0, API_SRC)

from utils.response_policy import (  # noqa: E402
    build_graph_summary,
    suppress_non_answer_payload,
    is_unavailable_answer,
)


def _load_check_emergency_from_main():
    main_path = Path(API_SRC) / "main.py"
    source = main_path.read_text(encoding="utf-8")
    module = ast.parse(source)

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check_emergency":
            wrapper = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(wrapper)
            namespace: dict[str, object] = {}
            exec(compile(wrapper, str(main_path), "exec"), namespace, namespace)
            return namespace.get("check_emergency")

    return None


check_emergency = _load_check_emergency_from_main()


class TestNoInfoSuppression(unittest.TestCase):
    def test_unavailable_marker_detected(self):
        self.assertTrue(is_unavailable_answer("I'm sorry, but I don't have that information."))

    def test_graph_summary_is_non_empty(self):
        text = build_graph_summary(
            {
                "disease_name": "Hypertension",
                "symptoms": ["headache", "chest pain"],
                "precautions": ["reduce salt intake", "exercise regularly"],
            }
        )
        self.assertIn("Hypertension", text)
        self.assertIn("symptoms", text.lower())

    def test_suppress_non_answer_payload(self):
        payload = {
            "output": "I don't have any information.",
            "sources": ["x"],
            "debug_context": {"route": "FACTUAL"},
            "graph_data": {"disease_found": True},
            "graph_target": "Hypertension",
        }
        suppress_non_answer_payload(payload)
        self.assertEqual(payload.get("sources"), [])
        self.assertNotIn("debug_context", payload)
        self.assertNotIn("graph_data", payload)
        self.assertNotIn("graph_target", payload)

    def test_check_emergency_detects_cardiac_cluster(self):
        self.assertTrue(callable(check_emergency))
        message = check_emergency("I have chest pain, shortness of breath, and heavy sweating")
        self.assertIsNotNone(message)
        self.assertIn("medical emergency", (message or "").lower())

    def test_check_emergency_non_urgent_returns_none(self):
        self.assertTrue(callable(check_emergency))
        message = check_emergency("I have mild cough and runny nose")
        self.assertIsNone(message)


if __name__ == "__main__":
    unittest.main()
