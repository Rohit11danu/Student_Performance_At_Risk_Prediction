# Model Training and Evaluation Results

Dataset: `student_performance.csv` (1000 synthetic student records).

## Method

A fixed-seed 70/15/15 stratified split was used because the generator creates one independent record per unique synthetic student. `student_id` and `at_risk` were excluded from predictors; only the five documented pre-assessment features were used. Imputation and Logistic Regression scaling were fitted inside training pipelines on the training partition. Logistic Regression and Random Forest used balanced class weights.

Thresholds were tuned on validation by maximizing positive-class F2, with ties favoring recall and then precision. Candidate selection was based only on validation F2, then PR-AUC, then recall. After selection was frozen, every baseline/candidate pipeline was evaluated on the held-out test partition once. No test result was used to change the model or thresholds.

Selected final pipeline: **Random Forest**, threshold **0.254228** (validation).

## Validation results

| Model | Threshold | Precision | Recall | F1 | F2 | PR-AUC (average precision) |
|---|---:|---:|---:|---:|---:|---:|
| Dummy baseline | 0.5000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.093 |
| Logistic Regression | 0.4588 | 0.343 | 0.857 | 0.490 | 0.659 | 0.543 |
| Random Forest | 0.2542 | 0.407 | 0.786 | 0.537 | 0.663 | 0.527 |

## Held-out test results

| Model | Threshold (validation) | Precision | Recall | F1 | F2 | PR-AUC (average precision) | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy baseline | 0.5000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.093 | 136 | 0 | 14 | 0 |
| Logistic Regression | 0.4588 | 0.333 | 0.857 | 0.480 | 0.652 | 0.566 | 112 | 24 | 2 | 12 |
| Random Forest | 0.2542 | 0.304 | 0.500 | 0.378 | 0.443 | 0.507 | 120 | 16 | 7 | 7 |

## Error interpretation

For the selected model, test results included **7 false negatives**: students labeled at risk by the synthetic outcome who would not be flagged. Missing these students could delay support, so recall was emphasized for threshold selection. It also produced **16 false positives**, which could consume staff attention or cause avoidable concern. The threshold is a validation-selected operating point; educators should set it based on available support capacity and the cost of missed cases.

## Limitations

The data and labels are synthetic and generated from a known formula. There are only 91 positive records overall, making split metrics uncertain. These scores do not establish real-world effectiveness, calibration, subgroup fairness, or the benefit of interventions. The final artifact remains fitted on training data only; it was not refitted on validation or test data.

Figures: `figures/model_comparison.png`, `figures/test_confusion_matrices.png`.
