"""Streamlit dashboard for saved student at-risk model inference."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.dashboard_metrics import build_test_interpretation, format_metric_table
from src.models.inference import global_feature_importance, infer_student, local_median_sensitivity

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "reports" / "model_artifacts" / "at_risk_pipeline.joblib"
RESULTS_PATH = ROOT / "reports" / "evaluation_results.json"
FIGURES = ROOT / "reports" / "figures"
FEATURE_SCHEMA_EXPECTED = [
    "attendance_pct",
    "assignment_completion_pct",
    "previous_assessment_1_pct",
    "previous_assessment_2_pct",
    "previous_assessment_trend_pct",
]

st.set_page_config(page_title="Student Risk Insights", page_icon="📘", layout="wide")

st.markdown("""
<style>
:root { --ink: #18324b; --muted: #587084; --line: #dce6ee; --blue: #176b87; }
.block-container { max-width: 1180px; padding-top: 1.8rem; padding-bottom: 3rem; }
.hero { padding: 1.35rem 1.55rem; border-radius: 16px; background: linear-gradient(115deg,#eaf5f8,#f4f7fb); border: 1px solid #d9e8ee; margin-bottom: 1.1rem; }
.hero h1 { color: var(--ink); font-size: 2rem; margin: 0 0 .35rem 0; }
.hero p { color: var(--muted); margin: 0; font-size: 1.02rem; }
.review-note { background:#fff4df; border-left:5px solid #d88a16; border-radius:8px; padding:.9rem 1rem; color:#583d12; font-weight:650; margin:.75rem 0 1.3rem; }
.result-card { border:1px solid var(--line); border-radius:14px; padding:1rem 1.15rem; background:#fff; }
.small-muted { color:var(--muted); font-size:.91rem; }
/* Metric cards keep a light surface in every Streamlit theme, so pin all
   metric text to a high-contrast dark foreground instead of inheriting the
   theme's possibly white text color. */
[data-testid="stMetric"] {
  background:#f7fafc !important;
  color:#18324b !important;
  border:1px solid #dce6ee;
  padding:.8rem 1rem;
  border-radius:12px;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricDelta"],
[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
[data-testid="stMetric"] [data-testid="stMetricValue"] *,
[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
  color:#18324b !important;
  -webkit-text-fill-color:#18324b !important;
  fill:#18324b !important;
  opacity:1 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_saved_artifacts():
    if not BUNDLE_PATH.is_file():
        raise FileNotFoundError(f"Saved model bundle not found: {BUNDLE_PATH}")
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Saved evaluation results not found: {RESULTS_PATH}")
    bundle = joblib.load(BUNDLE_PATH)
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if bundle.get("input_features") != FEATURE_SCHEMA_EXPECTED:
        raise ValueError("Saved model feature schema does not match this dashboard version")
    if bundle.get("model_name") != results.get("selected_model"):
        raise ValueError("Saved model bundle and evaluation report refer to different models")
    if abs(float(bundle.get("threshold", -1)) - float(results.get("selected_threshold", -2))) > 1e-12:
        raise ValueError("Saved model threshold does not match the recorded validation threshold")
    return bundle, results


def show_image(filename: str, caption: str):
    path = FIGURES / filename
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Saved figure is unavailable: {path.name}")


def render_header():
    st.markdown('<div class="hero"><h1>Student Risk Insights</h1><p>Advisory decision support from the saved synthetic-data model.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="review-note">Human review required: Predictions are advisory only and must not determine punishment, grades or admissions.</div>', unsafe_allow_html=True)


try:
    artifact, results = load_saved_artifacts()
except Exception as exc:
    st.error(f"Unable to load the saved model and evaluation report: {exc}")
    st.stop()

render_header()
page = st.radio("Choose a page", ["Student Risk Prediction", "EDA & Model Performance"], horizontal=True, label_visibility="collapsed")

if page == "Student Risk Prediction":
    st.subheader("Student Risk Prediction")
    st.write("Enter the four available pre-assessment measures. The assessment trend is derived automatically as assessment 2 minus assessment 1, matching the training data definition.")

    with st.form("student_inputs"):
        left, right = st.columns(2, gap="large")
        with left:
            attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1, help="Observed attendance before the next assessment.")
            completion = st.number_input("Assignment completion (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1, help="Completed assignments before the next assessment.")
        with right:
            previous_1 = st.number_input("Previous assessment 1 (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
            previous_2 = st.number_input("Previous assessment 2 (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
        trend_preview = round(previous_2 - previous_1, 1)
        st.caption(f"Derived assessment trend: {trend_preview:+.1f} percentage points")
        submitted = st.form_submit_button("Estimate risk", type="primary", width="stretch")

    if submitted:
        raw_values = {
            "attendance_pct": attendance,
            "assignment_completion_pct": completion,
            "previous_assessment_1_pct": previous_1,
            "previous_assessment_2_pct": previous_2,
        }
        try:
            prediction = infer_student(raw_values, artifact)
            sensitivity = local_median_sensitivity(prediction["features"], artifact)
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Input could not be evaluated: {exc}")
        else:
            band_color = {"Low": "#24734d", "Moderate": "#a76b00", "High": "#a63838"}[prediction["risk_band"]]
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted at-risk probability*", f"{prediction['probability']:.1%}")
            c2.metric("Advisory risk band", prediction["risk_band"])
            c3.metric("Selected decision threshold", f"{prediction['threshold']:.1%}")
            st.markdown(f'<div class="result-card"><strong style="color:{band_color}">{prediction["risk_band"]} risk band</strong><br><span class="small-muted">{("At or above" if prediction["predicted_class"] else "Below")} the validation-selected threshold. Treat this as a prompt for a supportive human check-in, not a decision about the student.</span></div>', unsafe_allow_html=True)

            st.markdown("#### Signals affecting this estimate")
            st.caption("One-feature-at-a-time sensitivity: each value is reset to that feature’s training-set median while other inputs stay fixed. Positive change means the model score rises after the reset; negative change means it falls. This is a model response comparison, not a causal explanation or guarantee of how an intervention would change an outcome.")
            display = sensitivity.rename(columns={
                "feature": "Feature", "training_median": "Training median",
                "probability_if_reset_to_median": "Score if reset to median",
                "change_from_current": "Score change from current",
            }).copy()
            display["Score if reset to median"] = display["Score if reset to median"].map(lambda value: f"{value:.1%}")
            display["Score change from current"] = display["Score change from current"].map(lambda value: f"{value:+.1%}")
            display["Training median"] = display["Training median"].map(lambda value: f"{value:.1f}")
            st.dataframe(display, hide_index=True, width="stretch")
            st.markdown("#### Overall model signals")
            importance = global_feature_importance(artifact).set_index("feature")["importance"]
            st.bar_chart(importance, horizontal=True)
            st.caption("The saved Random Forest's impurity importance summarizes how features were used across its trees. It can be biased by feature structure and does not show causation. The local median-reset comparison is also a model sensitivity check, not an intervention effect.")
            st.caption("The score is not calibrated: no probability calibration study was performed. Risk bands are UI categories, not validated student classifications.")

    with st.expander("How the advisory bands are mapped"):
        threshold = float(artifact["threshold"])
        st.write(f"The model’s saved validation threshold is **{threshold:.4f}**. High is a score at or above that threshold; Moderate is from half the threshold up to (but not including) the threshold; Low is below half the threshold. These descriptive bands use the saved classifier score directly. They are not calibrated probability ranges or validated intervention tiers.")
        st.write(f"The trained schema is: `{', '.join(artifact['input_features'])}`. `student_id` is not requested or used.")

    with st.expander("Uncertainty, privacy and responsible use"):
        st.markdown("""
- The model was trained and evaluated on synthetic records. Performance on real students is unknown, and its score is not a calibrated probability.
- False positives can cause unnecessary concern and consume support capacity. False negatives can miss a student who may need help. Review context with a qualified educator before any supportive action.
- Use results to offer voluntary, proportionate support. **Never use them to determine punishment, grades or admissions.**
- This page does not ask for names or IDs and does not save individual entries. When hosted remotely, follow the host’s logging and retention controls and your institution’s privacy rules; avoid entering directly identifying details.
""")

else:
    st.subheader("EDA & Model Performance")
    st.caption("Metrics and charts below are loaded from the saved training evaluation report. The dashboard does not fit or update any model.")
    audit = results.get("audit", {})
    counts = audit.get("target_counts", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset records", f"{results.get('dataset_rows', 0):,}")
    col2.metric("Selected model", results.get("selected_model", "Unavailable"))
    col3.metric("Validation threshold", f"{float(results.get('selected_threshold', 0)):.4f}")

    st.markdown("#### At-risk class distribution")
    if counts:
        balance = pd.DataFrame({"Class": ["Not at risk (0)", "At risk (1)"], "Records": [int(counts.get("0", 0)), int(counts.get("1", 0))]})
        a, b = st.columns([1, 1.35])
        with a:
            st.dataframe(balance, hide_index=True, width="stretch")
            total = max(balance["Records"].sum(), 1)
            st.caption(f"Positive class prevalence: {balance.loc[1, 'Records'] / total:.1%}")
        with b:
            show_image("class_balance.png", "Saved class balance from the existing dataset")

    st.markdown("#### Validation metrics used for selection")
    val_metrics = results.get("validation_metrics", {})
    if val_metrics:
        st.dataframe(format_metric_table(val_metrics), hide_index=True, width="stretch")

    st.markdown("#### Held-out test metrics")
    test_metrics = results.get("test_metrics", {})
    if test_metrics:
        st.dataframe(format_metric_table(test_metrics), hide_index=True, width="stretch")
        st.markdown("#### Test comparison and interpretation")
        st.markdown(build_test_interpretation(results))
    p1, p2 = st.columns(2)
    with p1:
        show_image("model_comparison.png", "Saved test comparison at validation-selected thresholds")
    with p2:
        show_image("test_confusion_matrices.png", "Saved confusion matrices on the held-out test split")

    st.markdown("#### EDA snapshots")
    e1, e2 = st.columns(2)
    with e1:
        show_image("feature_distributions.png", "Numeric distributions from the saved EDA")
    with e2:
        show_image("correlation_heatmap.png", "Descriptive Pearson correlations; association is not causation")

    st.markdown("#### How to read errors")
    error_notes = results.get("test_interpretation", {})
    st.write(f"**False positives:** {error_notes.get('false_positive', 'Not documented in saved results.')}")
    st.write(f"**False negatives:** {error_notes.get('false_negative', 'Not documented in saved results.')}")
    st.info("These are synthetic-data results with a small positive class. Test metrics are estimates for this generated sample, not guarantees for real students. No calibration, subgroup fairness or external validation was performed.")
