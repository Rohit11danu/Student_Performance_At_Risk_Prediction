# System Architecture

The project has a training/evaluation path and a separate inference path. The test set enters only after model and validation-threshold selection are frozen; test results flow to reporting and have no path back to selection.

![System architecture](figures/system_architecture.svg)

## Mermaid source

```mermaid
flowchart TD
    G["Synthetic data generator<br/>Python random seed 42"] --> D["Synthetic student CSV<br/>1,000 rows; next score omitted"]
    D --> E["Data validation and EDA<br/>quality, ranges, outliers, balance, leakage"]
    E --> S{"Stratified split<br/>70% train / 15% validation / 15% test"}

    S --> T["Train partition<br/>700 rows / 63 positives"]
    S --> V["Validation partition<br/>150 rows / 14 positives"]
    S --> H["Held-out test partition<br/>150 rows / 14 positives"]

    T --> P["Train-only preprocessing pipelines<br/>median imputer; scale Logistic Regression"]
    P --> C["Fit Dummy baseline, Logistic Regression,<br/>and Random Forest candidates"]
    C --> VS["Validation comparison and threshold tuning<br/>F2 threshold; candidate selection"]
    V --> VS
    VS --> F["Freeze selected model and thresholds<br/>Random Forest selected by validation F2"]
    F --> A["Saved joblib bundle<br/>pipeline, threshold, schema, seed, versions"]
    A --> APP["Streamlit prediction interface"]
    APP --> R["Advisory risk score and band"]
    R --> HR["Human review and supportive context"]

    C -. "fitted candidates" .-> TE["One-time held-out test evaluation<br/>using frozen validation thresholds"]
    F -. "selection already frozen" .-> TE
    H --> TE
    TE --> REP["Saved metrics and comparison plots<br/>reports/evaluation_results.json"]
    REP -. "reporting only; no feedback to selection" .-> N["No test-driven reselection"]

    classDef source fill:#eaf2ff,stroke:#3973b9,color:#18324b;
    classDef process fill:#eef6fa,stroke:#4c8299,color:#18324b;
    classDef validation fill:#fff4df,stroke:#bf861e,color:#493714;
    classDef selected fill:#e6f4ec,stroke:#38805a,color:#163d2a;
    classDef test fill:#fff0e8,stroke:#c56a3a,color:#542b18;
    classDef human fill:#e9e7f8,stroke:#6655a3,color:#27204d;
    class G,D source;
    class E,S,T,V,P,C,APP process;
    class VS,F validation;
    class A,R selected;
    class H,TE,REP,N test;
    class HR human;
```

## Flow notes

- The data generator writes a future-score-derived binary label but omits that future score from the predictor columns.
- The five-feature allowlist excludes `student_id` and `at_risk`. The current generator creates one unique record per synthetic student, supporting stratified independent-row splitting.
- Pipelines fit imputation and (for Logistic Regression) scaling on training rows. Validation is used to tune candidate thresholds and choose the model using F2.
- The final Random Forest and its validation threshold are persisted before test evaluation. The frozen model candidates and their validation thresholds are evaluated on the held-out test partition once for the report. The test branch has no selection feedback.
- Inference loads the persisted artifact; it does not call the training path. The final output is advisory and requires human review.
