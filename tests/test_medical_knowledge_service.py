import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
API_SRC = os.path.join(ROOT, "chatbot_api", "src")
if API_SRC not in sys.path:
    sys.path.insert(0, API_SRC)

from services.medical_knowledge_service import (  # noqa: E402
    load_medical_facts,
    detect_query_intent,
    find_disease_match,
    get_medical_answer,
)


class TestMedicalKnowledgeService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_medical_facts(force_reload=True)

    def _assert_reasoning_required_response(self, result: dict):
        self.assertTrue(result["found"])
        self.assertTrue(result.get("requires_reasoning"))
        self.assertNotIn(
            result.get("response_format"),
            {"SINGLE_DISEASE_SUMMARY", "SINGLE_INTENT", "MULTI_INTENT_SINGLE_DISEASE"},
        )
        answer = str(result.get("answer") or "").lower()
        self.assertTrue(
            any(
                marker in answer
                for marker in [
                    "because",
                    "since",
                    "this happens",
                    "this usually happens",
                    "which can",
                    "intensity",
                    "indicates",
                ]
            )
        )

    def test_alias_matching(self):
        key, record = find_disease_match("What causes high BP?")
        self.assertIsNotNone(key)
        self.assertIsNotNone(record)
        self.assertEqual(record["canonical_name"], "Hypertension")

    def test_priority_alias_matching(self):
        result = get_medical_answer("What are symptoms of stomach flu?")
        self.assertTrue(result["found"])
        self.assertEqual(result["disease"], "Gastroenteritis")
        self.assertEqual(result["match_type"], "priority_alias")
        self.assertFalse(result["needs_clarification"])

    def test_intent_detection_symptoms(self):
        intent = detect_query_intent("What are symptoms of malaria?")
        self.assertEqual(intent, "symptoms")

    def test_treatment_query_response(self):
        result = get_medical_answer("Treatment for dengue?")
        self.assertTrue(result["found"])
        self.assertIn("dengue", result["answer"].lower())
        self.assertIn(result["response_format"], {"SINGLE_INTENT", "MULTI_INTENT_SINGLE_DISEASE"})

    def test_fuzzy_match_threshold(self):
        result = get_medical_answer("What are symptoms of malria?")
        self.assertTrue(result["found"])
        self.assertEqual(result["disease"], "Malaria")
        self.assertEqual(result["match_type"], "fuzzy")
        self.assertGreaterEqual(float(result["confidence"]), 0.8)

    def test_conservative_emergency_note(self):
        result = get_medical_answer("What are symptoms of asthma with shortness of breath?")
        self.assertTrue(result["found"])
        self.assertNotIn("seek urgent in-person medical care immediately", result["answer"].lower())

    def test_emergency_note_for_red_flags(self):
        result = get_medical_answer("What are red flag signs of asthma with severe shortness of breath?")
        self.assertTrue(result["found"])
        self.assertIn("seek urgent in-person medical care immediately", result["answer"].lower())

    def test_unknown_disease_fallback(self):
        result = get_medical_answer("What causes madeupdiseasexyz?")
        self.assertFalse(result["found"])
        self.assertIn("could not identify the disease name clearly", result["answer"].lower())
        self.assertTrue(result["needs_clarification"])

    def test_multi_intent_symptoms_and_treatment_malaria(self):
        result = get_medical_answer("What are symptoms and treatment of malaria?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "MULTI_INTENT_SINGLE_DISEASE")
        answer = result["answer"].lower()
        self.assertIn("symptoms", answer)
        self.assertIn("treatment", answer)
        self.assertIn("malaria", answer)

    def test_multi_intent_causes_and_prevention_dengue(self):
        result = get_medical_answer("Causes and prevention of dengue")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "MULTI_INTENT_SINGLE_DISEASE")
        answer = result["answer"].lower()
        self.assertIn("causes", answer)
        self.assertIn("prevention", answer)
        self.assertIn("dengue", answer)

    def test_comparison_malaria_dengue_typhoid_symptoms(self):
        result = get_medical_answer("Compare malaria, dengue, and typhoid symptoms")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "COMPARISON")
        answer = result["answer"].lower()
        self.assertIn("malaria", answer)
        self.assertIn("dengue", answer)
        self.assertIn("typhoid", answer)
        self.assertIn("similarities", answer)
        self.assertIn("key differences", answer)

    def test_compare_malaria_and_dengue_symptoms(self):
        result = get_medical_answer("Compare malaria and dengue symptoms")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "COMPARISON")
        self.assertEqual(result["style_mode"], "comparison_answer")
        answer = result["answer"].lower()
        self.assertIn("malaria", answer)
        self.assertIn("dengue", answer)
        self.assertIn("difference", answer)

    def test_danger_ranking_dengue_vs_malaria(self):
        result = get_medical_answer("Is dengue more dangerous than malaria?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "DANGER_RANKING")
        self.assertEqual(result["style_mode"], "danger_ranking_answer")
        answer = result["answer"].lower()
        self.assertIn("short-term", answer)
        self.assertIn("dengue", answer)
        self.assertIn("malaria", answer)

    def test_danger_ranking_tb_and_pneumonia(self):
        result = get_medical_answer("Which is more dangerous among TB and pneumonia?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "DANGER_RANKING")
        answer = result["answer"].lower()
        self.assertIn("tuberculosis", answer)
        self.assertIn("pneumonia", answer)
        self.assertNotIn("not specified", answer)

    def test_disease_symptom_link_diabetes_eyesight(self):
        result = get_medical_answer("Does diabetes affect eyesight?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "DISEASE_SYMPTOM_LINK")
        self.assertEqual(result["reasoning_mode"], "disease_symptom_link")
        answer = result["answer"].lower()
        self.assertIn("diabetes", answer)
        self.assertTrue("blurred vision" in answer or "eyesight" in answer)

    def test_disease_symptom_link_diabetes_blurred_vision(self):
        result = get_medical_answer("Can diabetes cause blurred vision?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "DISEASE_SYMPTOM_LINK")
        answer = result["answer"].lower()
        self.assertIn("diabetes", answer)
        self.assertIn("blurred vision", answer)

    def test_why_diabetes_cause_blurred_vision(self):
        result = get_medical_answer("Why does diabetes cause blurred vision?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "DISEASE_SYMPTOM_LINK")
        answer = result["answer"].lower()
        self.assertIn("diabetes", answer)
        self.assertIn("blurred vision", answer)

    def test_disease_symptom_link_hypertension_headache(self):
        result = get_medical_answer("Can hypertension cause headache?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "DISEASE_SYMPTOM_LINK")
        answer = result["answer"].lower()
        self.assertIn("hypertension", answer)
        self.assertIn("headache", answer)

    def test_migraine_affect_vision_relation(self):
        result = get_medical_answer("Does migraine affect vision?")
        self._assert_reasoning_required_response(result)
        self.assertEqual(result["response_format"], "DISEASE_SYMPTOM_LINK")
        answer = result["answer"].lower()
        self.assertIn("migraine", answer)
        self.assertTrue("vision" in answer or "light sensitivity" in answer)

    def test_symptom_severity_comparison_fever_vs_high_fever(self):
        result = get_medical_answer("Does fever and high fever are different?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "SYMPTOM_SEVERITY_COMPARISON")
        self.assertEqual(result["reasoning_mode"], "symptom_severity_comparison")
        answer = result["answer"].lower()
        self.assertIn("fever", answer)
        self.assertIn("high fever", answer)
        self.assertTrue("severity" in answer or "intensity" in answer)

    def test_symptom_severity_comparison_differentiate_fever(self):
        result = get_medical_answer("How can we differentiate fever and high fever?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "SYMPTOM_SEVERITY_COMPARISON")
        answer = result["answer"].lower()
        self.assertIn("high fever", answer)
        self.assertIn("fever", answer)

    def test_symptom_severity_comparison_more_serious(self):
        result = get_medical_answer("Is high fever more serious than fever?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "SYMPTOM_SEVERITY_COMPARISON")
        answer = result["answer"].lower()
        self.assertIn("high fever", answer)
        self.assertIn("severity", answer)

    def test_symptom_severity_comparison_same(self):
        result = get_medical_answer("Is fever and high fever the same?")
        self._assert_reasoning_required_response(result)
        self.assertEqual(result["response_format"], "SYMPTOM_SEVERITY_COMPARISON")
        answer = result["answer"].lower()
        self.assertIn("not a different disease", answer)

    def test_symptom_severity_comparison_difference(self):
        result = get_medical_answer("What is the difference between fever and high fever?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "SYMPTOM_SEVERITY_COMPARISON")
        answer = result["answer"].lower()
        self.assertIn("intensity", answer)

    def test_high_sugar_meaning_query(self):
        result = get_medical_answer("What does high sugar mean?")
        self._assert_reasoning_required_response(result)
        self.assertEqual(result["response_format"], "DISEASE_SYMPTOM_LINK")
        answer = result["answer"].lower()
        self.assertTrue(
            "blood glucose" in answer
            or "blood sugar" in answer
            or "common explanation" in answer
            or "likely reason" in answer
        )
        self.assertIn("diabetes", answer)

    def test_disease_vs_symptom_category_clarification(self):
        result = get_medical_answer("Difference between malaria and fever, chills, headache")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "CATEGORY_CLARIFICATION")
        self.assertEqual(result["reasoning_mode"], "clarification_needed")
        answer = result["answer"].lower()
        self.assertIn("malaria is a disease", answer)
        self.assertIn("symptom", answer)

    def test_why_dengue_pain_behind_eyes(self):
        result = get_medical_answer("Why do dengue patients get pain behind eyes?")
        self._assert_reasoning_required_response(result)
        self.assertEqual(result["response_format"], "DISEASE_SYMPTOM_LINK")
        answer = result["answer"].lower()
        self.assertIn("dengue", answer)
        self.assertIn("pain behind", answer)

    def test_why_dengue_cause_eye_pain_exact_prompt(self):
        result = get_medical_answer("Why does dengue cause eye pain?")
        self._assert_reasoning_required_response(result)
        self.assertEqual(result["response_format"], "DISEASE_SYMPTOM_LINK")
        answer = result["answer"].lower()
        self.assertIn("dengue", answer)
        self.assertTrue("eye" in answer or "pain behind" in answer)

    def test_pneumonia_affect_oxygen_levels(self):
        result = get_medical_answer("Does pneumonia affect oxygen levels?")
        self._assert_reasoning_required_response(result)
        self.assertEqual(result["response_format"], "DISEASE_SYMPTOM_LINK")
        answer = result["answer"].lower()
        self.assertIn("pneumonia", answer)
        self.assertTrue("oxygen" in answer or "shortness of breath" in answer)

    def test_pneumonia_reduce_oxygen_exact_prompt(self):
        result = get_medical_answer("Does pneumonia reduce oxygen?")
        self._assert_reasoning_required_response(result)
        self.assertEqual(result["response_format"], "DISEASE_SYMPTOM_LINK")
        self.assertIn("pneumonia", result["answer"].lower())

    def test_anemia_cause_weakness(self):
        result = get_medical_answer("Can anemia cause weakness?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "DISEASE_SYMPTOM_LINK")
        answer = result["answer"].lower()
        self.assertIn("anemia", answer)

    def test_asthma_worse_than_bronchitis(self):
        result = get_medical_answer("Is asthma worse than bronchitis?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "DANGER_RANKING")
        answer = result["answer"].lower()
        self.assertIn("asthma", answer)
        self.assertIn("bronchitis", answer)

    def test_covid_more_serious_than_influenza(self):
        result = get_medical_answer("Is COVID more serious than influenza?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "DANGER_RANKING")
        answer = result["answer"].lower()
        self.assertIn("covid-19", answer)
        self.assertIn("influenza", answer)

    def test_repeated_question_has_style_variation(self):
        first = get_medical_answer("Does diabetes affect eyesight?")
        second = get_medical_answer("Does diabetes affect eyesight?")
        self._assert_reasoning_required_response(first)
        self._assert_reasoning_required_response(second)
        self.assertEqual(first["response_format"], "DISEASE_SYMPTOM_LINK")
        self.assertEqual(second["response_format"], "DISEASE_SYMPTOM_LINK")
        self.assertNotEqual(first["answer"], second["answer"])

    def test_repeated_high_sugar_keeps_reasoning_mode(self):
        first = get_medical_answer("What does high sugar mean?")
        second = get_medical_answer("What does high sugar mean?")
        self._assert_reasoning_required_response(first)
        self._assert_reasoning_required_response(second)
        self.assertEqual(first["response_format"], "DISEASE_SYMPTOM_LINK")
        self.assertEqual(second["response_format"], "DISEASE_SYMPTOM_LINK")
        self.assertNotEqual(first["answer"], second["answer"])

    def test_repeated_danger_question_has_style_variation(self):
        first = get_medical_answer("Is COVID more serious than influenza?")
        second = get_medical_answer("Is COVID more serious than influenza?")
        self.assertTrue(first["found"])
        self.assertTrue(second["found"])
        self.assertEqual(first["response_format"], "DANGER_RANKING")
        self.assertEqual(second["response_format"], "DANGER_RANKING")
        self.assertNotEqual(first["answer"], second["answer"])

    def test_repeated_comparison_question_has_style_variation(self):
        first = get_medical_answer("Compare malaria and dengue symptoms")
        second = get_medical_answer("Compare malaria and dengue symptoms")
        self.assertTrue(first["found"])
        self.assertTrue(second["found"])
        self.assertEqual(first["response_format"], "COMPARISON")
        self.assertEqual(second["response_format"], "COMPARISON")
        self.assertNotEqual(first["answer"], second["answer"])

    def test_comparison_without_forced_similarities_heading(self):
        result = get_medical_answer("Compare depression and typhoid causes")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "COMPARISON")
        answer = result["answer"]
        self.assertNotIn("No single shared pattern appears", answer)

    def test_no_internal_meta_wording_in_relation_answer(self):
        result = get_medical_answer("Does diabetes affect eyesight?")
        self.assertTrue(result["found"])
        answer = result["answer"].lower()
        forbidden = [
            "structured data",
            "knowledge base",
            "current dataset",
            "documented symptom links",
            "retrieval",
        ]
        for marker in forbidden:
            self.assertNotIn(marker, answer)

    def test_differential_symptom_check_malaria_or_dengue(self):
        result = get_medical_answer("I have fever, chills, and body pain - is it malaria or dengue?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "DIFFERENTIAL_SYMPTOM_CHECK")
        answer = result["answer"].lower()
        self.assertIn("cannot confirm a single disease", answer)
        self.assertIn("malaria", answer)
        self.assertIn("dengue", answer)

    def test_difference_between_tb_and_pneumonia_symptoms(self):
        result = get_medical_answer("Difference between TB and pneumonia symptoms")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "COMPARISON")
        answer = result["answer"].lower()
        self.assertIn("tuberculosis", answer)
        self.assertIn("pneumonia", answer)

    def test_difference_between_asthma_and_bronchitis(self):
        result = get_medical_answer("Difference between asthma and bronchitis")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "COMPARISON")
        answer = result["answer"].lower()
        self.assertIn("asthma", answer)
        self.assertIn("bronchitis", answer)

    def test_uncertain_red_flag_symptom_check(self):
        result = get_medical_answer("I have cough for weeks and weight loss - should I worry?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "RED_FLAG")
        answer = result["answer"].lower()
        self.assertIn("cannot confirm a diagnosis", answer)
        self.assertIn("urgent medical care", answer)

    def test_followup_treatment_with_history_context(self):
        history = "User: Tell me about malaria\nAssistant: Malaria symptoms include fever and chills."
        result = get_medical_answer("What about treatment?", chat_history=history)
        self.assertTrue(result["found"])
        self.assertEqual(result["disease"], "Malaria")
        self.assertTrue(result.get("used_history_context"))

    def test_followup_treatment_without_history_needs_clarification(self):
        result = get_medical_answer("What about treatment?")
        self.assertFalse(result["found"])
        self.assertTrue(result["needs_clarification"])

    def test_symptoms_of_alien_flu_no_wrong_mapping(self):
        result = get_medical_answer("Symptoms of alien flu")
        self.assertFalse(result["found"])
        self.assertTrue(result["needs_clarification"])
        self.assertNotIn("influenza", result["answer"].lower())

    def test_my_sugar_is_high_safe_diabetes_response(self):
        result = get_medical_answer("My sugar is high, what should I do?")
        self.assertTrue(result["found"])
        self.assertEqual(result["disease"], "Diabetes")
        self.assertIn("diabetes", result["answer"].lower())

    def test_my_sugar_high_and_tired(self):
        result = get_medical_answer("My sugar is high and I feel tired - what does it mean?")
        self.assertTrue(result["found"])
        self.assertEqual(result["disease"], "Diabetes")
        self.assertIn(
            result["response_format"],
            {
                "SINGLE_INTENT",
                "MULTI_INTENT_SINGLE_DISEASE",
                "DIFFERENTIAL_SYMPTOM_CHECK",
                "SINGLE_DISEASE_SUMMARY",
                "DISEASE_SYMPTOM_LINK",
            },
        )
        self.assertIn("diabetes", result["answer"].lower())

    def test_asthma_and_tb_together_multi_topic(self):
        result = get_medical_answer("Can asthma and TB happen together?")
        self.assertTrue(result["found"])
        self.assertEqual(result["response_format"], "COMPARISON")
        answer = result["answer"].lower()
        self.assertIn("asthma", answer)
        self.assertIn("tuberculosis", answer)


if __name__ == "__main__":
    unittest.main()
