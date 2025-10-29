## Why

Email spam remains a common nuisance and a useful supervised learning exercise for the course. Building a clear, reproducible spam email classification baseline will:

- Teach end-to-end ML workflow for the class (data ingestion, preprocessing, model training, evaluation)
- Provide a reference implementation students can extend in later phases
- Produce an artifact (notebook + model) suitable for grading and experimentation

## What Changes

- Add a new capability: spam email classification baseline using classical ML.
- Phase 1 (baseline): Train and evaluate a logistic regression spam classifier using the provided public SMS spam dataset.
- Phase 2+: Placeholder phases for iterative improvements (feature engineering, model comparison, deployment, etc.).

**BREAKING:** none — non-production, educational code and docs only.

## Impact

- Affected specs: `specs/ml/spec.md` (new capability)
- Affected code: new notebooks and scripts under `notebooks/` or `src/ml/` and test/data fixtures
- External dependency: public dataset URL listed below; standard Python ML libs (scikit-learn, pandas, etc.)

## Phase 1 — Baseline (plan)

- Goal: Produce a reproducible baseline classifier using logistic regression and report evaluation metrics (precision, recall, F1, ROC-AUC).
- Dataset: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- Notes: The dataset is CSV with two columns (label, text) — authors should validate column parsing and encoding.

## Owners

- Author: (please add your name/email)
- Reviewers: course staff / maintainers

## Acceptance Criteria

- [ ] A notebook or script that downloads/reads the dataset and documents preprocessing steps
- [ ] A trained logistic regression baseline with saved model artifact or notebook output
- [ ] Evaluation metrics reported (precision, recall, F1, accuracy, ROC-AUC) on a held-out test set
- [ ] Reproducible instructions in `README` or the notebook to run training locally

---

If you want the baseline to use SVM instead of logistic regression, or to use additional preprocessing (TF-IDF, n-grams), tell me and I will update the tasks/spec accordingly. I assumed logistic regression as the primary baseline because it is simple and aligned with your stated goal.

## CI Validation (added)

This change also includes CI validation for OpenSpec proposals. The repository SHALL include a CI workflow that validates any changed `openspec/changes/<change-id>/` directories by running `openspec validate <change-id> --strict` and failing the PR if validation reports errors. See `openspec/changes/README.md` for local validation instructions.
