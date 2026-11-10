"""Checks dashboard formatting and interpretation against saved evaluation data."""

import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.dashboard_metrics import build_test_interpretation, format_metric_table

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "reports" / "evaluation_results.json"


class DashboardMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    def test_tables_format_metrics_and_flatten_confusion_matrix(self):
        for metrics_key in ("validation_metrics", "test_metrics"):
            source = self.results[metrics_key]
            table = format_metric_table(source)
            self.assertEqual(
                table.columns.tolist(),
                ["Model", "Threshold", "Precision", "Recall", "F1", "F2", "PR-AUC", "TN", "FP", "FN", "TP"],
            )
            self.assertNotIn("confusion_matrix", table.columns)
            for model, saved in source.items():
                row = table.loc[table["Model"] == model].iloc[0]
                for label, key in [
                    ("Precision", "precision"),
                    ("Recall", "recall"),
                    ("F1", "f1"),
                    ("F2", "f2"),
                    ("PR-AUC", "pr_auc_average_precision"),
                ]:
                    self.assertRegex(row[label], r"^\d+\.\d{3}$")
                    self.assertEqual(row[label], f"{saved[key]:.3f}")
                for label in ("TN", "FP", "FN", "TP"):
                    self.assertEqual(int(row[label]), saved["confusion_matrix"][label.lower()])
                self.assertEqual(row["Threshold"], f"{saved['threshold']:.4f}")

    def test_performance_page_shows_formatted_tables_and_keeps_saved_charts(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        app.radio[0].set_value("EDA & Model Performance").run()
        self.assertFalse(app.exception, [exc.message for exc in app.exception])
        metric_tables = [element.value for element in app.dataframe if "Model" in element.value.columns]
        self.assertEqual(len(metric_tables), 2)
        for table, key in zip(metric_tables, ("validation_metrics", "test_metrics")):
            expected = format_metric_table(self.results[key])
            self.assertEqual(table.columns.tolist(), expected.columns.tolist())
            self.assertEqual(table.to_dict(orient="records"), expected.to_dict(orient="records"))
        self.assertTrue(any("Selection remains Random Forest" in element.value for element in app.markdown))
        for filename in (
            "test_confusion_matrices.png",
            "feature_distributions.png",
            "model_comparison.png",
        ):
            self.assertTrue((ROOT / "reports" / "figures" / filename).is_file())

    def test_interpretation_uses_saved_metrics_and_validation_selection(self):
        text = build_test_interpretation(self.results)
        logistic = self.results["test_metrics"]["Logistic Regression"]
        forest = self.results["test_metrics"]["Random Forest"]
        for name, metrics in [("Logistic Regression", logistic), ("Random Forest", forest)]:
            cm = metrics["confusion_matrix"]
            self.assertIn(f"**{name}:**", text)
            self.assertIn(f"precision {metrics['precision']:.4f}", text)
            self.assertIn(f"recall {metrics['recall']:.4f}", text)
            self.assertIn(f"F1 {metrics['f1']:.4f}", text)
            self.assertIn(f"PR-AUC {metrics['pr_auc_average_precision']:.4f}", text)
            self.assertIn(f"FN={cm['fn']}, FP={cm['fp']}", text)
        selected = self.results["selected_model"]
        self.assertEqual(selected, "Random Forest")
        self.assertIn("selected using validation F2", text)
        self.assertIn(f"({self.results['validation_metrics'][selected]['f2']:.4f})", text)
        self.assertIn(f"threshold {self.results['selected_threshold']:.4f}", text)
        self.assertIn("before test evaluation", text)
        self.assertIn("only 14 at-risk students", text)
        self.assertIn("uncalibrated", text)
        self.assertIn("synthetic", text)
        self.assertIn("False positives", text)
        self.assertIn("false negative", text)


if __name__ == "__main__":
    unittest.main()
