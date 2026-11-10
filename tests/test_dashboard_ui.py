"""Regression checks for metric-card contrast and prediction rendering."""

import re
import unittest
from pathlib import Path

import joblib
from streamlit.testing.v1 import AppTest

from src.models.inference import infer_student

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
BUNDLE_PATH = ROOT / "reports" / "model_artifacts" / "at_risk_pipeline.joblib"
INPUTS = {
    "attendance_pct": 45.0,
    "assignment_completion_pct": 35.0,
    "previous_assessment_1_pct": 38.0,
    "previous_assessment_2_pct": 32.0,
}


class DashboardUIRegressionTests(unittest.TestCase):
    def test_metric_card_css_pins_high_contrast_foreground(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('[data-testid="stMetric"]', source)
        self.assertIn('[data-testid="stMetricLabel"]', source)
        self.assertIn('[data-testid="stMetricValue"]', source)
        self.assertRegex(source, re.compile(r"color:\s*#18324b\s*!important", re.IGNORECASE))
        self.assertRegex(source, re.compile(r"background:\s*#f7fafc\s*!important", re.IGNORECASE))
        self.assertRegex(source, re.compile(r"-webkit-text-fill-color:\s*#18324b\s*!important", re.IGNORECASE))

    def test_requested_input_renders_saved_model_score_threshold_and_band(self):
        artifact = joblib.load(BUNDLE_PATH)
        expected = infer_student(INPUTS, artifact)

        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        self.assertFalse(app.exception, [exc.message for exc in app.exception])
        self.assertEqual(len(app.number_input), 4)
        for widget, value in zip(app.number_input, INPUTS.values()):
            widget.set_value(value)
        app.button[0].click().run()
        self.assertFalse(app.exception, [exc.message for exc in app.exception])

        cards = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(cards["Predicted at-risk probability*"], f"{expected['probability']:.1%}")
        self.assertEqual(cards["Selected decision threshold"], f"{expected['threshold']:.1%}")
        self.assertEqual(cards["Advisory risk band"], expected["risk_band"])
        self.assertEqual(expected["risk_band"], "High")


if __name__ == "__main__":
    unittest.main()
