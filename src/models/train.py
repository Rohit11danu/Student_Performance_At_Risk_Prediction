"""Reproducible training and held-out evaluation for at-risk prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "student_performance.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_ARTIFACT_DIR = DEFAULT_REPORT_DIR / "model_artifacts"
DEFAULT_FIGURE_DIR = DEFAULT_REPORT_DIR / "figures"
TARGET = "at_risk"
IDENTIFIER = "student_id"
# This allowlist intentionally excludes identifiers, the target, and any field
# not documented as available before the next assessment.
FEATURES = [
    "attendance_pct",
    "assignment_completion_pct",
    "previous_assessment_1_pct",
    "previous_assessment_2_pct",
    "previous_assessment_trend_pct",
]
SEED = 42


def audit_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate label and schema, then describe the feature allowlist boundary."""
    missing = sorted(set([TARGET, IDENTIFIER, *FEATURES]) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values")
    labels = set(pd.Series(df[TARGET]).unique().tolist())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError(f"Target must contain both binary classes 0 and 1; found {sorted(labels)}")
    if df[IDENTIFIER].isna().any() or df[IDENTIFIER].duplicated().any():
        raise ValueError("student_id must be present and unique for independent student records")
    if df[FEATURES].isna().all(axis=None):
        raise ValueError("All predictor values are missing")
    suspicious = [
        c for c in df.columns
        if c not in {TARGET, IDENTIFIER, *FEATURES}
        and any(term in c.lower() for term in ("next", "future", "post", "final", "outcome", "risk_score", "label"))
    ]
    if suspicious:
        raise ValueError(f"Potential future/target-derived columns found; review before training: {suspicious}")
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "target": TARGET,
        "target_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "id_unique": bool(df[IDENTIFIER].is_unique),
        "features": list(FEATURES),
        "excluded_columns": [c for c in df.columns if c not in FEATURES],
        "suspicious_columns": suspicious,
        "target_definition": "1 when the simulated next assessment score is below 40%; that score is not stored in the CSV.",
        "temporal_assumption": "One independent synthetic row per unique student; no repeated student sequence is generated.",
    }


def stratified_split_indices(y: pd.Series, seed: int = SEED) -> dict[str, np.ndarray]:
    """Create deterministic, disjoint 70/15/15 stratified row-index splits."""
    indices = np.arange(len(y))
    train_val, test = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=y.to_numpy()
    )
    relative_val_size = 0.15 / 0.85
    train, validation = train_test_split(
        train_val, test_size=relative_val_size, random_state=seed + 1,
        stratify=y.iloc[train_val].to_numpy(),
    )
    return {"train": train, "validation": validation, "test": test}


def _build_preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(steps=numeric_steps)
    return ColumnTransformer(
        transformers=[("numeric", numeric, FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_estimators(seed: int = SEED) -> dict[str, Pipeline]:
    """Create all candidate pipelines; every transformer is fitted within fit()."""
    return {
        "Dummy baseline": Pipeline([
            ("preprocess", _build_preprocessor(scale=False)),
            ("classifier", DummyClassifier(strategy="prior")),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocess", _build_preprocessor(scale=True)),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", _build_preprocessor(scale=False)),
            ("classifier", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                random_state=seed, n_jobs=1,
            )),
        ]),
    }


def _positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["classifier"].classes_)
    return model.predict_proba(X[FEATURES])[:, classes.index(1)]


def choose_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Maximize validation F2; ties favor recall, then precision, then higher threshold."""
    y = np.asarray(y_true, dtype=int)
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    rows = []
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(int)
        rows.append((
            fbeta_score(y, pred, beta=2, zero_division=0),
            recall_score(y, pred, zero_division=0),
            precision_score(y, pred, zero_division=0),
            float(threshold),
        ))
    best = max(rows)
    return {"threshold": best[3], "validation_f2": best[0], "validation_recall": best[1], "validation_precision": best[2]}


def calculate_metrics(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _plot_comparison(test_metrics: dict[str, dict[str, Any]], path: Path) -> None:
    labels = list(test_metrics)
    names = ["precision", "recall", "f1", "pr_auc_average_precision"]
    pretty = ["Precision", "Recall", "F1", "PR-AUC"]
    x = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for i, (metric, label) in enumerate(zip(names, pretty)):
        vals = [test_metrics[name][metric] for name in labels]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label)
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test metrics at validation-selected thresholds")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrices(test_metrics: dict[str, dict[str, Any]], path: Path) -> None:
    names = list(test_metrics)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        c = test_metrics[name]["confusion_matrix"]
        matrix = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), val in np.ndenumerate(matrix):
            ax.text(j, i, str(val), ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
        ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
        ax.set_title(f"{name}\nthreshold={test_metrics[name]['threshold']:.3f}")
    fig.suptitle("Held-out test confusion matrices")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_training(
    data_path: Path = DEFAULT_DATA,
    report_dir: Path = DEFAULT_REPORT_DIR,
    seed: int = SEED,
) -> dict[str, Any]:
    """Train on train split, select only with validation, then report test once."""
    data_path = Path(data_path)
    report_dir = Path(report_dir)
    figure_dir = report_dir / "figures"
    artifact_dir = report_dir / "model_artifacts"
    figure_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    audit = audit_dataset(df)
    splits = stratified_split_indices(df[TARGET], seed)
    X = df[FEATURES].apply(pd.to_numeric, errors="raise")
    y = df[TARGET].astype(int)
    X_train, y_train = X.iloc[splits["train"]], y.iloc[splits["train"]]
    X_val, y_val = X.iloc[splits["validation"]], y.iloc[splits["validation"]]
    X_test, y_test = X.iloc[splits["test"]], y.iloc[splits["test"]]

    models = build_estimators(seed)
    val_summary: dict[str, dict[str, Any]] = {}
    thresholds: dict[str, float] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_prob = _positive_probability(model, X_val)
        # The Dummy baseline retains its conventional 0.5 decision rule.
        if name == "Dummy baseline":
            threshold_info = {"threshold": 0.5}
        else:
            threshold_info = choose_threshold(y_val, val_prob)
        thresholds[name] = float(threshold_info["threshold"])
        val_metrics = calculate_metrics(y_val, val_prob, thresholds[name])
        val_summary[name] = {**threshold_info, **val_metrics}

    # Select from validation only: highest F2, then PR-AUC, then recall.
    candidate_names = ["Logistic Regression", "Random Forest"]
    selected_name = max(
        candidate_names,
        key=lambda name: (
            val_summary[name]["f2"],
            val_summary[name]["pr_auc_average_precision"],
            val_summary[name]["recall"],
        ),
    )
    selected_threshold = thresholds[selected_name]

    # Selection is now frozen. Evaluate each fitted pipeline on test once.
    test_summary: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        test_prob = _positive_probability(model, X_test)
        test_summary[name] = calculate_metrics(y_test, test_prob, thresholds[name])

    _plot_comparison(test_summary, figure_dir / "model_comparison.png")
    _plot_confusion_matrices(test_summary, figure_dir / "test_confusion_matrices.png")

    versions = {
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__, "matplotlib": matplotlib.__version__,
    }
    artifact = {
        "pipeline": models[selected_name],
        "model_name": selected_name,
        "threshold": selected_threshold,
        "input_features": list(FEATURES),
        "target": TARGET,
        "seed": int(seed),
        "dependency_versions": versions,
        "threshold_objective": "Validation F2 (beta=2), prioritizing recall for support triage",
    }
    artifact_path = artifact_dir / "at_risk_pipeline.joblib"
    joblib.dump(artifact, artifact_path)

    results = {
        "dataset_path": str(data_path),
        "dataset_rows": len(df),
        "seed": int(seed),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "split_class_counts": {
            part: {str(k): int(v) for k, v in y.iloc[idx].value_counts().sort_index().items()}
            for part, idx in splits.items()
        },
        "audit": audit,
        "selection_rule": "Among Logistic Regression and Random Forest, maximize validation F2 at each model's validation-tuned threshold; ties use validation PR-AUC then recall. Test metrics did not affect selection.",
        "validation_metrics": val_summary,
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "test_metrics": test_summary,
        "test_interpretation": {
            "false_positive": "An alert for a student whose simulated next score was not below 40%; this may consume staff attention and can create avoidable concern.",
            "false_negative": "A missed alert for a student whose simulated next score was below 40%; the student may not receive timely support. Recall is therefore emphasized during validation threshold selection.",
        },
        "limitations": [
            "All records and labels are synthetic and generated from a known scoring formula; measured performance does not establish real-world effectiveness.",
            "The sample has only 91 positive labels overall, so split metrics have substantial sampling uncertainty.",
            "The binary at-risk threshold is based on one simulated next assessment and omits context, timing, and intervention capacity.",
            "The final artifact is fit on the training partition only; validation is reserved for selection and thresholding, and the test split remains held out.",
            "No calibration study, subgroup fairness analysis, or external validation is performed.",
        ],
        "artifact_path": str(artifact_path),
        "figure_paths": [
            str(figure_dir / "model_comparison.png"),
            str(figure_dir / "test_confusion_matrices.png"),
        ],
        "dependency_versions": versions,
    }
    results_path = report_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([
        {"model": name, **metrics} for name, metrics in test_summary.items()
    ]).to_csv(report_dir / "test_metrics.csv", index=False)
    report_lines = [
        "# Model Training and Evaluation Results", "",
        f"Dataset: `{data_path.name}` ({len(df)} synthetic student records).", "",
        "## Method", "",
        "A fixed-seed 70/15/15 stratified split was used because the generator creates one independent record per unique synthetic student. `student_id` and `at_risk` were excluded from predictors; only the five documented pre-assessment features were used. Imputation and Logistic Regression scaling were fitted inside training pipelines on the training partition. Logistic Regression and Random Forest used balanced class weights.", "",
        "Thresholds were tuned on validation by maximizing positive-class F2, with ties favoring recall and then precision. Candidate selection was based only on validation F2, then PR-AUC, then recall. After selection was frozen, every baseline/candidate pipeline was evaluated on the held-out test partition once. No test result was used to change the model or thresholds.", "",
        f"Selected final pipeline: **{selected_name}**, threshold **{selected_threshold:.6f}** (validation).", "",
        "## Validation results", "",
        "| Model | Threshold | Precision | Recall | F1 | F2 | PR-AUC (average precision) |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in val_summary.items():
        report_lines.append(f"| {name} | {m['threshold']:.4f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['f2']:.3f} | {m['pr_auc_average_precision']:.3f} |")
    report_lines += ["", "## Held-out test results", "", "| Model | Threshold (validation) | Precision | Recall | F1 | F2 | PR-AUC (average precision) | TN | FP | FN | TP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in test_summary.items():
        c = m['confusion_matrix']
        report_lines.append(f"| {name} | {m['threshold']:.4f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['f2']:.3f} | {m['pr_auc_average_precision']:.3f} | {c['tn']} | {c['fp']} | {c['fn']} | {c['tp']} |")
    selected_test = test_summary[selected_name]
    selected_cm = selected_test['confusion_matrix']
    report_lines += [
        "", "## Error interpretation", "",
        f"For the selected model, test results included **{selected_cm['fn']} false negatives**: students labeled at risk by the synthetic outcome who would not be flagged. Missing these students could delay support, so recall was emphasized for threshold selection. It also produced **{selected_cm['fp']} false positives**, which could consume staff attention or cause avoidable concern. The threshold is a validation-selected operating point; educators should set it based on available support capacity and the cost of missed cases.", "",
        "## Limitations", "",
        "The data and labels are synthetic and generated from a known formula. There are only 91 positive records overall, making split metrics uncertain. These scores do not establish real-world effectiveness, calibration, subgroup fairness, or the benefit of interventions. The final artifact remains fitted on training data only; it was not refitted on validation or test data.", "",
        "Figures: `figures/model_comparison.png`, `figures/test_confusion_matrices.png`.", "",
    ]
    (report_dir / "model_evaluation.md").write_text("\n".join(report_lines), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = run_training(args.data, args.report_dir, args.seed)
    print(f"Selected model: {result['selected_model']} at threshold {result['selected_threshold']:.6f}")
    print("Held-out test metrics:")
    for name, metrics in result["test_metrics"].items():
        print(
            f"{name}: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, "
            f"F1={metrics['f1']:.3f}, PR-AUC={metrics['pr_auc_average_precision']:.3f}, "
            f"confusion={metrics['confusion_matrix']}"
        )
    print(f"Saved results: {args.report_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()


def load_artifact(path: Path = DEFAULT_ARTIFACT_DIR / "at_risk_pipeline.joblib") -> dict[str, Any]:
    """Load the complete pipeline bundle saved by run_training."""
    artifact = joblib.load(path)
    required = {"pipeline", "model_name", "threshold", "input_features", "target", "seed", "dependency_versions"}
    if not required.issubset(artifact):
        raise ValueError(f"Model artifact is missing fields: {sorted(required - set(artifact))}")
    return artifact


def predict_at_risk(records: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Return risk probabilities and decisions after enforcing the saved schema."""
    schema = artifact["input_features"]
    missing = [column for column in schema if column not in records.columns]
    if missing:
        raise ValueError(f"Inference input is missing required features: {missing}")
    X = records.loc[:, schema].apply(pd.to_numeric, errors="raise")
    probabilities = _positive_probability(artifact["pipeline"], X)
    return pd.DataFrame({
        "at_risk_probability": probabilities,
        "at_risk_prediction": (probabilities >= float(artifact["threshold"])).astype(int),
    }, index=records.index)
