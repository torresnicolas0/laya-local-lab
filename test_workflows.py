import unittest

from lab import read_json
from workflows import FLOW_NAMES, binary_metrics, decide, validate_state, verify_protocol


class WorkflowChecks(unittest.TestCase):
    def test_frozen_cases_and_expected_policy(self):
        verify_protocol()
        cases = read_json("workflow_cases.json")
        self.assertEqual(len({c["id"] for c in cases}), 36)
        for flow in FLOW_NAMES:
            selected = [c for c in cases if c["workflow"] == flow]
            self.assertEqual(len(selected), 12)
            for c in selected:
                validate_state(flow, c["state"])
                self.assertEqual(c["language"], "en")
                answers = {q: {"noul" if isinstance(v, bool) else "score": v}
                           for q, v in c["expected"].items() if q != "action"}
                self.assertEqual(decide(flow, answers), c["expected"]["action"], c["id"])

    def test_action_precedence_and_boundaries(self):
        self.assertEqual(decide("guardrails", {"attack": {"noul": .6}, "sensitive_data": {"noul": 1}}), "BLOCK")
        self.assertEqual(decide("rag", {"injection": {"noul": .5}, "relevant": {"noul": 0}, "contradicts": {"noul": 1}}), "DROP_INJECTION")
        self.assertEqual(decide("rag", {"injection": {"noul": 0}, "relevant": {"noul": .5}, "contradicts": {"noul": .5}}), "REVIEW_CONTRADICTION")
        self.assertEqual(decide("routing", {"needs_tools": {"noul": .5}, "difficulty": {"score": 2}}), "TOOLS")
        self.assertEqual(decide("routing", {"needs_tools": {"noul": 0}, "difficulty": {"score": 1.5}}), "LARGE")

    def test_binary_metrics_known_values(self):
        m = binary_metrics([(.9, True), (.8, False), (.2, True), (.1, False)])
        self.assertEqual([m[k] for k in ["tp", "fp", "fn", "tn"]], [1, 1, 1, 1])
        self.assertEqual(m["precision"], .5)
        self.assertEqual(m["recall"], .5)
        self.assertAlmostEqual(m["brier"], .325)
        self.assertIsNone(binary_metrics([(.1, False)])["recall"])

    def test_invalid_inputs(self):
        for state in [{}, {"prompt": " "}, {"prompt": 123}, {"prompt": "hello", "unexpected": "x"}]:
            with self.assertRaises(ValueError):
                validate_state("guardrails", state)


if __name__ == "__main__":
    unittest.main()
