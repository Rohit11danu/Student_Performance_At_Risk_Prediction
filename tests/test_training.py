"""Tests for leakage boundaries and reproducible model artifacts."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models.train import (
    DEFAULT_DATA,
    FEATURES,
    TARGET,
    audit_dataset,
    load_artifact,
    predict_at_risk,
    run_training,
    stratified_split_indices,
)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(DEFAULT_DATA)

    def test_feature_schema_excludes_identifier_target_and_future_columns(self):
        audit = audit_dataset(self.data)
        self.assertEqual(audit["features"], FEATURES)
        self.assertNotIn("student_id", audit["features"])
        self.assertNotIn(TARGET, audit["features"])
        self.assertNotIn("next_assessment_score_pct", audit["features"])
        self.assertTrue(set(FEATURES).issubset(self.data.columns))

    def test_split_is_reproducible_stratified_and_disjoint(self):
        y = self.data[TARGET]
        first = stratified_split_indices(y, seed=17)
        second = stratified_split_indices(y, seed=17)
        for part in first:
            self.assertEqual(first[part].tolist(), second[part].tolist())
        sets = [set(first[name]) for name in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertEqual(sum(map(len, sets)), len(y))
        for indices in first.values():
            self.assertGreater(self.data.iloc[indices][TARGET].sum(), 0)

    def test_future_named_column_fails_audit(self):
        altered = self.data.assign(next_exam_score=60)
        with self.assertRaisesRegex(ValueError, "Potential future"):
            audit_dataset(altered)

    def test_training_reproducibility_and_saved_model_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            first = run_training(DEFAULT_DATA, output / "one", seed=23)
            second = run_training(DEFAULT_DATA, output / "two", seed=23)
            self.assertEqual(first["selected_model"], second["selected_model"])
            self.assertEqual(first["selected_threshold"], second["selected_threshold"])
            self.assertEqual(first["validation_metrics"], second["validation_metrics"])
            self.assertEqual(first["test_metrics"], second["test_metrics"])
            artifact = load_artifact(Path(first["artifact_path"]))
            self.assertEqual(artifact["input_features"], FEATURES)
            self.assertNotIn("student_id", artifact["input_features"])
            self.assertNotIn(TARGET, artifact["input_features"])
            predictions = predict_at_risk(self.data.head(8), artifact)
            self.assertEqual(list(predictions.columns), ["at_risk_probability", "at_risk_prediction"])
            self.assertEqual(len(predictions), 8)
            self.assertTrue(predictions["at_risk_probability"].between(0, 1).all())
            self.assertTrue(set(predictions["at_risk_prediction"].unique()).issubset({0, 1}))
            with self.assertRaisesRegex(ValueError, "missing required features"):
                predict_at_risk(self.data[["student_id"]].head(1), artifact)
            saved = json.loads((output / "one" / "evaluation_results.json").read_text())
            self.assertEqual(saved["selected_model"], first["selected_model"])


if __name__ == "__main__":
    unittest.main()
