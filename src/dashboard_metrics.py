"""Presentation helpers for the saved evaluation report used by Streamlit."""

from __future__ import annotations

from typing import Any

import pandas as pd

METRIC_COLUMNS = ["Precision", "Recall", "F1", "F2", "PR-AUC"]
CONFUSION_COLUMNS = ["TN", "FP", "FN", "TP"]
TABLE_COLUMNS = ["Model", "Threshold", *METRIC_COLUMNS, *CONFUSION_COLUMNS]


def format_metric_table(metrics_by_model: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Create a readable table from saved metrics, formatting scores to 3 decimals."""
    rows = []
    field_map = [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("F2", "f2"),
        ("PR-AUC", "pr_auc_average_precision"),
    ]
    for model_name, metrics in metrics_by_model.items():
        matrix = metrics.get("confusion_matrix")
        if not isinstance(matrix, dict) or not all(key.lower() in matrix for key in CONFUSION_COLUMNS):
            raise ValueError(f"Saved confusion matrix is incomplete for {model_name}")
        row: dict[str, Any] = {
            "Model": model_name,
            "Threshold": f"{float(metrics['threshold']):.4f}",
        }
        for display_name, source_name in field_map:
            row[display_name] = f"{float(metrics[source_name]):.3f}"
        for key in CONFUSION_COLUMNS:
            value = matrix[key.lower()]
            if int(value) != value or value < 0:
                raise ValueError(f"Saved confusion matrix count {key} is invalid for {model_name}")
            row[key] = int(value)
        rows.append(row)
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def build_test_interpretation(results: dict[str, Any]) -> str:
    """Summarize actual saved test metrics and preserve validation-only selection."""
    test = results.get("test_metrics", {})
    validation = results.get("validation_metrics", {})
    selected = results.get("selected_model")
    selected_threshold = float(results.get("selected_threshold"))
    if "Logistic Regression" not in test or "Random Forest" not in test:
        raise ValueError("Saved results must contain Logistic Regression and Random Forest test metrics")
    if selected not in validation:
        raise ValueError("Selected model is missing its saved validation metrics")

    split_counts = results.get("split_class_counts", {}).get("test", {})
    positives = int(split_counts.get("1", 0))
    if positives == 0:
        rf_cm = test["Random Forest"].get("confusion_matrix", {})
        positives = int(rf_cm.get("fn", 0)) + int(rf_cm.get("tp", 0))
    if positives <= 0:
        raise ValueError("Could not determine the held-out positive count from saved results")

    model_lines = []
    for name in ("Logistic Regression", "Random Forest"):
        metrics = test[name]
        cm = metrics["confusion_matrix"]
        if int(cm["fn"]) + int(cm["tp"]) != positives:
            raise ValueError(f"Saved confusion matrix positive count is inconsistent for {name}")
        model_lines.append(
            f"**{name}:** precision {float(metrics['precision']):.4f}, "
            f"recall {float(metrics['recall']):.4f}, F1 {float(metrics['f1']):.4f}, "
            f"PR-AUC {float(metrics['pr_auc_average_precision']):.4f}, "
            f"FN={int(cm['fn'])}, FP={int(cm['fp'])}."
        )

    selected_validation_f2 = float(validation[selected]["f2"])
    return "\n\n".join([
        *model_lines,
        f"**Selection remains {selected}.** It was selected using validation F2 "
        f"({selected_validation_f2:.4f}) at threshold {selected_threshold:.4f}, before test evaluation. "
        "The held-out test results are reported for comparison only and do not change the selected model or threshold.",
        f"Recall matters for supportive early warning because a false negative can leave a student who may need help without a timely check-in. "
        f"False positives also matter: each of the {int(test['Random Forest']['confusion_matrix']['fp'])} Random Forest false alerts "
        "uses staff time for review and may prompt unnecessary concern. A support team must consider both missed cases and review capacity.",
        f"The test set contains only {positives} at-risk students, so recall and precision estimates can move substantially with a few records. "
        "The dataset and labels are synthetic, so results do not establish real-student performance. Classifier scores are uncalibrated: "
        "no calibration study was performed, and the scores should not be read as calibrated probabilities.",
    ])
