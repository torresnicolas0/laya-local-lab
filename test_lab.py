import copy
import math
import unittest

from lab import cases, load_agent, metrics, predict, read_json, validate_input


class LabChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = load_agent("cpu")
        cls.questions = read_json("questions.json")

    def test_dataset_and_every_input_fits(self):
        rows = cases()
        self.assertEqual(len(rows), 30)
        self.assertEqual(len({r["id"] for r in rows}), 30)
        self.assertEqual(len({r["scenario"] for r in rows}), 10)
        for row in rows:
            validate_input(self.agent, row["state"], self.questions)
            self.assertIn(row["expected"]["department"], self.questions["department"]["criteria"])
            self.assertIn(row["expected"]["urgency"], (0, 1, 2))
            self.assertIsInstance(row["expected"]["refund_requested"], bool)

    def test_invalid_and_oversize_requests_are_rejected(self):
        for state in ("", "   ", {}, [], None, "palabra " * 2000):
            with self.subTest(state_type=type(state).__name__), self.assertRaises(ValueError):
                validate_input(self.agent, state, self.questions)
        for question in ({}, {"x": {"type": "invalid", "instructions": "?"}}, {"x": {"type": "choice", "instructions": "?", "criteria": {}}}):
            with self.assertRaises(ValueError):
                validate_input(self.agent, "Hello", question)
        long_option = copy.deepcopy(self.questions)
        long_option["department"]["criteria"]["billing"] = "word " * 60
        with self.assertRaises(ValueError):
            validate_input(self.agent, "Hello", long_option)

    def test_real_model_contract_not_expected_accuracy(self):
        result, latency = predict(self.agent, "Please send the invoice.", self.questions)
        answers = result["answers"]
        self.assertEqual(set(answers), set(self.questions))
        self.assertIn(answers["department"]["choice"], self.questions["department"]["criteria"])
        self.assertAlmostEqual(sum(answers["department"]["probabilities"].values()), 1, places=3)
        self.assertTrue(0 <= answers["urgency"]["score"] <= 2)
        self.assertTrue(0 <= answers["refund_requested"]["noul"] <= 1)
        self.assertTrue(math.isfinite(latency) and latency > 0)

    def test_metrics_against_hand_calculated_values(self):
        rows = []
        for department, urgency, p, truth, latency in [("billing", 0.5, 0.8, True, 10), ("technical", 1.5, 0.3, False, 30)]:
            rows.append({"expected": {"department": "billing", "urgency": 0, "refund_requested": truth},
                         "result": {"answers": {"department": {"choice": department}, "urgency": {"score": urgency}, "refund_requested": {"noul": p}}}, "latency_ms": latency})
        result = metrics(rows)
        self.assertEqual(result["department_accuracy"], 0.5)
        self.assertEqual(result["urgency_mae_0_to_2"], 1)
        self.assertAlmostEqual(result["refund_brier"], 0.065)
        self.assertEqual(result["refund_accuracy_threshold_0_5"], 1)
        self.assertEqual(result["latency_p50_ms"], 20)
        self.assertEqual(result["latency_p95_ms"], 29)


if __name__ == "__main__":
    unittest.main()
