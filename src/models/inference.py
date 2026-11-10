"""Strict input preparation and saved-pipeline inference helpers for the UI."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

ENGINEERED_FEATURES = {"previous_assessment_trend_pct"}
RAW_FEATURES = [
    "attendance_pct",
    "assignment_completion_pct",
    "previous_assessment_1_pct",
    "previous_assessment_2_pct",
]
RANGED_FEATURES = {
    "attendance_pct": (0.0, 100.0),
    "assignment_completion_pct": (0.0, 100.0),
    "previous_assessment_1_pct": (0.0, 100.0),
    "previous_assessment_2_pct": (0.0, 100.0),
}


def prepare_student_features(values: Mapping[str, Any], feature_schema: list[str]) -> pd.DataFrame:
    """Validate the exact raw UI schema and derive trained features in saved order."""
    if not isinstance(values, Mapping):
        raise ValueError("Inputs must be provided as a mapping of feature names to values")
    if set(feature_schema) != set(RAW_FEATURES) | ENGINEERED_FEATURES or len(feature_schema) != 5:
        raise ValueError("Saved model feature schema is unsupported; expected the trained student schema")
    missing = sorted(set(RAW_FEATURES) - set(values))
    unexpected = sorted(set(values) - set(RAW_FEATURES))
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected inputs: {', '.join(unexpected)}")

    clean: dict[str, float] = {}
    for name in RAW_FEATURES:
        value = values[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a numeric value")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        low, high = RANGED_FEATURES[name]
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low:g} and {high:g}")
        clean[name] = value

    clean["previous_assessment_trend_pct"] = round(
        clean["previous_assessment_2_pct"] - clean["previous_assessment_1_pct"], 1
    )
    return pd.DataFrame([{name: clean[name] for name in feature_schema}], columns=feature_schema)


def infer_student(values: Mapping[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    """Run the saved preprocessing/model pipeline without fitting or mutating it."""
    threshold = artifact.get("threshold")
    if not isinstance(threshold, Real) or isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Saved decision threshold must be a finite value between 0 and 1")
    schema = artifact.get("input_features")
    if not isinstance(schema, list) or not all(isinstance(col, str) for col in schema):
        raise ValueError("Saved model bundle does not contain a valid feature schema")
    X = prepare_student_features(values, schema)
    pipeline = artifact.get("pipeline")
    if pipeline is None or not hasattr(pipeline, "predict_proba"):
        raise ValueError("Saved model bundle does not contain a probability-capable pipeline")
    classes = list(pipeline.named_steps["classifier"].classes_)
    probability = float(pipeline.predict_proba(X)[:, classes.index(1)][0])
    high_cutoff = float(threshold)
    moderate_cutoff = high_cutoff / 2
    if probability >= high_cutoff:
        band = "High"
    elif probability >= moderate_cutoff:
        band = "Moderate"
    else:
        band = "Low"
    return {
        "probability": probability,
        "threshold": high_cutoff,
        "predicted_class": int(probability >= high_cutoff),
        "risk_band": band,
        "features": X,
    }


def local_median_sensitivity(features: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Measure one-feature probability shifts when a feature is reset to its training median."""
    pipeline = artifact["pipeline"]
    schema = artifact["input_features"]
    preprocessor = pipeline.named_steps["preprocess"]
    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    medians = dict(zip(schema, imputer.statistics_))
    base_probability = float(pipeline.predict_proba(features[schema])[0, list(pipeline.named_steps["classifier"].classes_).index(1)])
    rows = []
    for feature in schema:
        changed = features[schema].copy()
        changed.loc[:, feature] = medians[feature]
        changed_probability = float(pipeline.predict_proba(changed)[0, list(pipeline.named_steps["classifier"].classes_).index(1)])
        rows.append({
            "feature": feature,
            "training_median": float(medians[feature]),
            "probability_if_reset_to_median": changed_probability,
            "change_from_current": changed_probability - base_probability,
        })
    return pd.DataFrame(rows).sort_values("change_from_current", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def global_feature_importance(artifact: dict[str, Any]) -> pd.DataFrame:
    """Expose the fitted classifier's native importance signal when supported."""
    classifier = artifact["pipeline"].named_steps["classifier"]
    schema = artifact["input_features"]
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
        method = "Random Forest impurity importance"
    elif hasattr(classifier, "coef_"):
        values = np.mean(np.abs(np.asarray(classifier.coef_, dtype=float)), axis=0)
        method = "Mean absolute linear coefficient"
    else:
        raise ValueError("Saved classifier does not expose a supported native feature importance")
    if len(values) != len(schema) or not np.isfinite(values).all():
        raise ValueError("Saved classifier importance does not match its feature schema")
    return pd.DataFrame({"feature": schema, "importance": values, "method": method}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
