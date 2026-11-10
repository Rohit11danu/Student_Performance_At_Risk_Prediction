# GitHub Readiness Audit

**Project:** Student Performance & At-Risk Prediction  
**Organization:** ALGIONS AI LABS PRIVATE LIMITED  
**Audit date:** 2026-11-08  
**Scope:** Repository and publication readiness only. No model training command was run, no committed dataset was replaced, and no saved model, threshold, or evaluation metric was changed.

## Findings

| Status | Area | Finding |
|---|---|---|
| **PASS** | Source and artifacts | Generator, application, training/inference modules, EDA notebook, tests, technical report, architecture source and SVG, evaluation files, figures, and the saved joblib pipeline are present. |
| **PASS** | Dataset provenance | The CSV has 1,000 unique generated IDs and only synthetic assessment/attendance/assignment fields plus the binary label. No name, contact, or birth-date fields were found. Regenerating seed 42 to `/private/tmp/student-performance-audit.csv` produced the same SHA-256 as the checked-in CSV. |
| **PASS** | Runtime imports | Direct runtime imports are covered by `requirements.txt`; notebook execution dependencies (`ipykernel`, `nbclient`, `nbconvert`, `nbformat`) were added. The environment used for this audit had versions matching `constraints-py39.txt`. |
| **PASS** | Ignore rules | Added `.gitignore` entries for virtual environments, bytecode, caches, notebook checkpoints, OS/editor files, temporary/log files, `.env` and common credential files, and raw/local data. The reviewed synthetic CSV is explicitly allow-listed. This allow-list is not a substitute for checking that its contents remain synthetic before committing. |
| **PASS** | Documentation links | README links to the technical report, model evaluation, saved evaluation data, EDA notebook, architecture, and SVG. All relative Markdown links in README and the two reports resolve; architecture SVG parses as valid XML. |
| **PASS** | Size suitability | Largest individual project artifact is the saved model at about 2.1 MB; the notebook is about 384 KB and the largest figure about 196 KB. No individual file is near GitHub's 100 MB hard limit. The `.venv` directory is excluded and must not be committed. |
| **PASS** | Tests | `MPLCONFIGDIR=/private/tmp/mplconfig python3 -m unittest discover -s tests -v` — **16 tests passed**, 0 failures, 5.432 seconds. The training reproducibility test fits models under a temporary directory; it did not alter the saved model bundle. Streamlit emitted expected bare-mode `ScriptRunContext` warnings in UI tests. |
| **PASS** | Protected artifacts | SHA-256 values for the dataset, saved pipeline, evaluation JSON, test metrics, model report, and two saved model comparison figures were identical before and after the test/reproducibility checks. |
| **WARN** | Python support | The artifact records Python 3.9.6, which was the tested interpreter, and `.python-version` records it. Python 3.9 reached end of life on 2025-10-31 according to the [Python Developer's Guide](https://devguide.python.org/versions/). This is reproducible for the recorded environment but is not a supported runtime for a new public deployment. Select a maintained Python release and verify joblib compatibility before relying on this artifact there; this audit did not upgrade Python or dependencies. |
| **WARN** | Dependency lock | `constraints-py39.txt` pins the direct runtime and notebook packages verified in this environment. It is deliberately not a full transitive lock and contains no hashes; resolver-selected indirect dependencies can change. A full lock should be generated and tested in an isolated environment when a supported Python runtime is selected. |
| **WARN** | Git status | `git status --short --ignored` failed with `fatal: not a git repository`. There is no `.git` directory, so tracked/untracked/staged status and the exact Git commit set cannot be established. No Git repository was initialized, as requested. The file inventory below is the prospective project content, not a verified Git index. |
| **WARN** | Local paths | Secret/path scan found no credential values or private keys in project text, but `reports/evaluation_results.json` contains four absolute local paths: `dataset_path`, `artifact_path`, and both entries in `figure_paths`. They reveal a machine username and checkout path. The file was left unchanged because this audit was instructed not to modify evaluation results. Remove or make these metadata paths portable before publication while preserving all metric values. |
| **WARN** | Environment health | `python3 -m pip check` reported conflicts in the pre-existing global environment: TensorFlow/protobuf version mismatch, torchaudio/torch mismatch, and platform support notices for grpcio and mediapipe. These packages are not direct project dependencies. Tests passed in this environment; the conflicts should be rechecked in a clean virtual environment created from the project requirements. |
| **WARN** | License | No project-level `LICENSE` exists. Project redistribution terms are unspecified. No license was added. Obtain authorization before choosing one. |
| **WARN** | Repository automation | No CI workflow or full transitive dependency lock is present. These are not required to run this assessment, but automated clean-environment checks would improve future maintenance. |

## Commands and actual results

### Test suite

```text
MPLCONFIGDIR=/private/tmp/mplconfig python3 -m unittest discover -s tests -v
Ran 16 tests in 5.432s
OK
```

The tests cover dashboard metric formatting and interpretation, UI contrast and displayed model outputs, seeded generation, input validation and feature engineering, saved-model inference, schema/target exclusion, split reproducibility, and model training reproducibility. The training test creates its output in a temporary directory. This test suite's temporary fits are part of the requested verification; no production model was retrained or saved.

### Seeded data reproducibility

```text
python3 -m src.data.generate_data --records 1000 --seed 42 --output /private/tmp/student-performance-audit.csv
generated_sha256:   7e0070b249bdfecb785ad136c53b2c4ea94fa5d27e65560ad4792b3738934ce1
checked_in_sha256:  7e0070b249bdfecb785ad136c53b2c4ea94fa5d27e65560ad4792b3738934ce1
Result: exact byte-for-byte match; checked-in CSV was read only.
```

### Other checks

- Local Markdown links: no missing links in `README.md`, `reports/technical_report.md`, or `reports/architecture.md`.
- `reports/figures/system_architecture.svg`: XML parse passed.
- Synthetic CSV schema: 1,000 rows; 1,000 unique IDs; no name, email, phone, address, or birth-date fields.
- Secret/path text scan: no credential or private-key patterns found. Absolute path matches are the four evaluation JSON metadata fields described above; `.gitignore` mentions credential filenames by design.
- File sizes: largest included file is `reports/model_artifacts/at_risk_pipeline.joblib` at about 2.1 MB; none approaches 100 MB.
- `python3 -m pip check`: **not clean** due to unrelated packages already installed globally (details above). No dependency installation was attempted during this audit.

### Protected artifact hashes

These values were identical before and after checks:

| File | SHA-256 |
|---|---|
| `data/processed/student_performance.csv` | `7e0070b249bdfecb785ad136c53b2c4ea94fa5d27e65560ad4792b3738934ce1` |
| `reports/model_artifacts/at_risk_pipeline.joblib` | `793b5dbf6a04688f441d9cd8578e29a4a5ae248d0acff8384f6b4349ab55d8c3` |
| `reports/evaluation_results.json` | `cc06dd57e091465d563c3be537d8cec09fbb5e99f07d1afb28a5abb3180c284c` |
| `reports/test_metrics.csv` | `c7833d5657c6e467f5b6ad9c59acf8e16e17abc41691ad17108685c9d2c9d4fb` |
| `reports/model_evaluation.md` | `5803f6e45cccfbdc02964b90cd4abfc300f4f0ab74d892a69765ea1458eb885b` |
| `reports/figures/model_comparison.png` | `f3cb94f36b5f62265a70557b3ab1d576c15bc7473608477760486419e0210c48` |
| `reports/figures/test_confusion_matrices.png` | `a6530d48e3e929e19931a1642d212f9727b9c7bd86616997100258a9be196a50` |

## Prospective project inventory

Because there is no Git metadata, every file below is a prospective source-tree inventory, not a statement that it is staged or tracked. If a repository is initialized later, review `git status --short --ignored` and stage only the intended files.

**Project/configuration:** `.gitignore`, `.python-version`, `README.md`, `requirements.txt`, `constraints-py39.txt`.

**Application and source:** `app.py`; `src/__init__.py`, `src/dashboard_metrics.py`, `src/data/__init__.py`, `src/data/generate_data.py`, `src/models/__init__.py`, `src/models/inference.py`, `src/models/train.py`.

**Tests:** `tests/test_dashboard_metrics.py`, `tests/test_dashboard_ui.py`, `tests/test_generate_data.py`, `tests/test_inference.py`, `tests/test_training.py`.

**Data and analysis:** `data/processed/student_performance.csv` (synthetic), `data/raw/.gitkeep`, `notebooks/.gitkeep`, `notebooks/eda.ipynb`.

**Reports and artifacts:** `reports/.gitkeep`, `reports/architecture.md`, `reports/evaluation_results.json`, `reports/model_evaluation.md`, `reports/technical_report.md`, `reports/test_metrics.csv`, `reports/github_readiness_audit.md`, `reports/model_artifacts/at_risk_pipeline.joblib`, and figures `reports/figures/class_balance.png`, `correlation_heatmap.png`, `feature_boxplots.png`, `feature_distributions.png`, `model_comparison.png`, `test_confusion_matrices.png`, and `system_architecture.svg` (all figure paths are under `reports/figures/`).

**Ignored local content:** `.venv/` exists locally and is ignored; Python bytecode/cache directories and notebook checkpoints are also ignored if created. No Git status was available to distinguish already tracked from untracked files.

## Remaining issues before publication

1. Create/use a Git repository only when authorized; inspect the real index and ignored files, then stage a reviewed allow-list.
2. Remove machine-specific absolute paths from evaluation metadata without changing the metric values; rerun the path scan and applicable artifact consistency checks.
3. Decide on a maintained Python version and validate the saved joblib artifact and application under it. Python 3.9 is end-of-life.
4. Build and test the dependency environment in a clean virtual environment; consider a complete, reproducible lock file for the selected supported Python version.
5. Confirm no real or confidential student data has replaced the allow-listed synthetic CSV. Do not commit locally collected data.
6. Obtain the rightsholder's decision on project licensing; no license is implied by this audit.

## Safe commit checklist

- [ ] Confirm repository root and review `git status --short --ignored` after a Git repository exists.
- [ ] Exclude `.venv/`, secrets, caches, raw/local data, and any personal or confidential records.
- [ ] Resolve the four absolute paths in `reports/evaluation_results.json` without changing saved evaluation metrics.
- [ ] Confirm the saved model bundle and synthetic dataset are intended assessment deliverables; verify their hashes if they are copied or regenerated.
- [ ] Run the documented tests in a clean environment and resolve environment-specific dependency conflicts.
- [ ] Choose and document a supported Python version after checking model deserialization compatibility.
- [ ] Confirm the project's licensing terms with the rightsholder; do not add a license by assumption.
- [ ] Review the final staged diff and file list. This audit did not initialize Git, create a remote, commit, push, or publish anything.
