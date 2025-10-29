## Overview

This change introduces a multi-phase spam classification project. Phase 1 contains concrete tasks for a baseline logistic regression classifier. Phase 2 and later are placeholders for iterative improvements.

## Phase 1 — Baseline (logistic regression)
- [ ] 1.1 Create `notebooks/spam-classification-baseline.ipynb` (or `src/ml/train_baseline.py`) that:
  - Downloads/reads the dataset from the provided URL
  - Cleans and tokenizes text (basic normalization)
  - Splits data into train/validation/test sets
  - Trains a logistic regression classifier using TF-IDF features
  - Evaluates metrics (precision, recall, F1, accuracy, ROC-AUC)
  - Saves a minimal model artifact and metrics output
- [ ] 1.2 Add a small unit test or script that verifies dataset download and parsing on CI (use a cached small sample)
- [ ] 1.3 Document reproducible run instructions in the notebook README or `notebooks/README.md`

## Phase 2 — (placeholders)
- [ ] 2.1 Feature engineering: n-grams, stop-word optimization
- [ ] 2.2 Model comparison: SVM, RandomForest, and hyperparameter tuning
- [ ] 2.3 Data augmentation and class-balance handling

## Testing
- [ ] Add CI job (optional) to run the notebook conversion or a short script that trains on a very small sample to smoke-test the pipeline

## CI tasks (added)
- [ ] CI-1 Add `.github/workflows/openspec-validate.yml` that:
  - Detects modified `openspec/changes/*` directories in a PR
  - For each top-level change-id modified, runs `openspec validate <change-id> --strict`
  - Fails the job on validation errors and prints validator output
- [ ] CI-2 Add documentation in `openspec/changes/README.md` showing how to run validation locally and common fixes
