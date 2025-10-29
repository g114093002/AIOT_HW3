#!/usr/bin/env python3
"""Quick diagnostics: load saved model bundle and run test predictions.

Run from project root using the project's venv:
  .venv\Scripts\python diagnostics\run_model_check.py
"""
from pathlib import Path
import json
import sys

try:
    import joblib
except Exception as e:
    print('ERROR: joblib not available:', e)
    sys.exit(2)

MODEL_PATH = Path('out') / 'model.joblib'
METRICS_PATH = Path('out') / 'metrics.json'

if not MODEL_PATH.exists():
    print('MODEL_MISSING')
    sys.exit(1)

bundle = joblib.load(MODEL_PATH)
print('BUNDLE_TYPE:', type(bundle))
if hasattr(bundle, 'keys'):
    print('BUNDLE_KEYS:', list(bundle.keys()))
else:
    print('BUNDLE not a mapping; repr:', repr(bundle)[:200])

if METRICS_PATH.exists():
    print('\nMETRICS:')
    print(METRICS_PATH.read_text(encoding='utf-8'))

msgs = [
    'Win a free iPhone by clicking here',
    'Lowest price on meds, order now',
    'Meeting at 3pm today, please confirm',
    'Your package has shipped. Track it here',
    'Congratulations! You have been selected for a gift card',
    'Free entry in 2 a weekly competition to win',
    'Hi mom, can you pick me up at 5?'
]

vec = bundle.get('vectorizer') if hasattr(bundle, 'get') else bundle['vectorizer']
clf = bundle.get('classifier') if hasattr(bundle, 'get') else bundle['classifier']

print('\nPREDICTIONS:')
for m in msgs:
    try:
        X = vec.transform([m])
        pred = clf.predict(X)[0]
        prob = clf.predict_proba(X)[0, 1] if hasattr(clf, 'predict_proba') else None
        lab = 'spam' if int(pred) == 1 else 'ham'
        print(f"{m!r} -> {lab} (prob_spam={prob:.4f} if prob is not None else None)")
    except Exception as e:
        print('Error predicting for:', m, '->', e)

print('\nDONE')
