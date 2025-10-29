## ADDED Requirements

### Requirement: Spam Email Classification — baseline
The project SHALL provide a reproducible baseline spam classification capability that trains and evaluates a logistic regression classifier on a public SMS spam dataset.

#### Scenario: Dataset ingestion
- **WHEN** the baseline job starts
- **THEN** it downloads or reads the CSV dataset from `https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv` and parses label and text columns successfully

#### Scenario: Model training and evaluation
- **WHEN** the data is preprocessed and split into train/validation/test
- **THEN** the training script SHALL train a logistic regression classifier using TF-IDF features and report precision, recall, F1, accuracy, and ROC-AUC on the test set

#### Scenario: Artifact and reproducibility
- **WHEN** training completes
- **THEN** the pipeline SHALL save a small model artifact (pickle or joblib), a metrics JSON file, and a README explaining how to reproduce the results

### Requirement: Experimentation placeholders
The repository SHALL include placeholders for further phases (feature engineering and model comparisons) under `openspec/changes/` and in the tasks for this change.

#### Scenario: Phase placeholders exist
- **WHEN** a developer inspects the change tasks
- **THEN** they SHALL see entries for Phase 2+ describing planned follow-ups (feature engineering, model comparison, augmentation)
