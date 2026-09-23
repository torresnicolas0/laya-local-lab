"""A real local-model smoke test for the new interactive workflows."""
import unittest

from streamlit.testing.v1 import AppTest

from lab import ROOT


class WorkflowUIChecks(unittest.TestCase):
    def test_real_inference_navigation_and_empty_input(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        app.selectbox[0].set_value("typed-decisions").run()
        for flow in ("guardrails", "rag", "routing"):
            app.selectbox[1].set_value(flow).run()
            self.assertEqual(len(app.metric), 0)
            self.assertEqual(len(app.selectbox[2].options), 12)
            app.button(key="workflow_analyze").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.metric), 1)
            self.assertEqual(app.session_state["workflow_result"]["workflow"], flow)
            self.assertEqual(app.session_state["workflow_result"]["model"], "typed-decisions")
        app.text_area[0].set_value(" ").run()
        app.button(key="workflow_analyze").click().run()
        self.assertEqual(len(app.error), 1)
        self.assertEqual(len(app.metric), 0)
        app.selectbox[1].set_value("guardrails").run()
        app.button(key="workflow_analyze").click().run()
        app.selectbox[0].set_value("english").run()
        self.assertEqual(len(app.metric), 0)
        app.selectbox[1].set_value("support").run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 3)


if __name__ == "__main__":
    unittest.main()
