"""Input validation and inference checks for the saved model dashboard."""

import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.inference import global_feature_importance, infer_student, local_median_sensitivity, prepare_student_features

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "reports" / "model_artifacts" / "at_risk_pipeline.joblib"
SCHEMA = [
    "attendance_pct",
    "assignment_completion_pct",
    "previous_assessment_1_pct",
    "previous_assessment_2_pct",
    "previous_assessment_trend_pct",
]
VALID = {
    "attendance_pct": 74.5,
    "assignment_completion_pct": 63.0,
    "previous_assessment_1_pct": 51.2,
    "previous_assessment_2_pct": 59.8,
}


class InferenceInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = joblib.load(ARTIFACT_PATH)

    def test_engineers_trend_in_training_schema_order(self):
        features = prepare_student_features(VALID, SCHEMA)
        self.assertEqual(features.columns.tolist(), SCHEMA)
        self.assertEqual(features.iloc[0]["previous_assessment_trend_pct"], 8.6)
        self.assertNotIn("student_id", features.columns)
        self.assertNotIn("at_risk", features.columns)

    def test_saved_model_inference_and_sensitivity(self):
        result = infer_student(VALID, self.artifact)
        self.assertGreaterEqual(result["probability"], 0)
        self.assertLessEqual(result["probability"], 1)
        self.assertEqual(result["threshold"], self.artifact["threshold"])
        self.assertEqual(result["predicted_class"], int(result["probability"] >= result["threshold"]))
        self.assertIn(result["risk_band"], {"Low", "Moderate", "High"})
        sensitivity = local_median_sensitivity(result["features"], self.artifact)
        self.assertEqual(set(sensitivity["feature"]), set(SCHEMA))
        self.assertTrue(np.isfinite(sensitivity["change_from_current"]).all())
        importance = global_feature_importance(self.artifact)
        self.assertEqual(set(importance["feature"]), set(SCHEMA))
        self.assertTrue((importance["importance"] >= 0).all())

    def test_rejects_missing_unexpected_and_invalid_inputs(self):
        missing = dict(VALID)
        missing.pop("attendance_pct")
        with self.assertRaisesRegex(ValueError, "Missing required"):
            infer_student(missing, self.artifact)

        unexpected = {**VALID, "at_risk": 1}
        with self.assertRaisesRegex(ValueError, "Unexpected inputs"):
            infer_student(unexpected, self.artifact)

        for feature, value in [
            ("attendance_pct", -0.1),
            ("assignment_completion_pct", 100.1),
            ("previous_assessment_1_pct", float("nan")),
            ("previous_assessment_2_pct", "not-a-number"),
            ("attendance_pct", True),
        ]:
            with self.subTest(feature=feature, value=value):
                invalid = dict(VALID)
                invalid[feature] = value
                with self.assertRaises(ValueError):
                    infer_student(invalid, self.artifact)

    def test_rejects_incompatible_saved_schema(self):
        bad_schema = [*SCHEMA[:-1], "next_assessment_score_pct"]
        with self.assertRaisesRegex(ValueError, "unsupported"):
            prepare_student_features(VALID, bad_schema)


if __name__ == "__main__":
    unittest.main()
