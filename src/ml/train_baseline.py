#!/usr/bin/env python3
"""Train a baseline spam classifier using TF-IDF + Logistic Regression.

Usage:
  python src/ml/train_baseline.py --data-url <CSV_URL> --output-dir out

By default the script reads the SMS spam dataset (label, text) from the
provided URL, trains a TF-IDF + LogisticRegression classifier, and writes:
- model.joblib (sklearn pipeline/vectorizer + model)
- metrics.json (evaluation metrics)

The script is intentionally small and dependency-light for homework use.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def load_dataset(url: str) -> pd.DataFrame:
    """Load dataset from a CSV URL and normalize to columns ['label','text'].

    The dataset at the provided URL is expected to be two columns: label,text
    with no header. This function attempts a few sensible parsing strategies.
    """
    # Try reading without header and name columns
    try:
        df = pd.read_csv(url, header=None, encoding='utf-8')
    except Exception:
        # Fallback: let pandas infer
        df = pd.read_csv(url)

    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ['label', 'text']
    else:
        # If it's a single-column CSV, attempt to split on first comma
        s = df.iloc[:, 0].astype(str)
        parts = s.str.split(',', n=1, expand=True)
        parts.columns = ['label', 'text']
        df = parts

    # normalize labels (lower-case)
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['text'] = df['text'].astype(str)
    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df['text'].values
    y = df['label'].apply(lambda s: 1 if s in ('spam', '1', 'true', 't', 'yes') else 0).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Simple logistic regression baseline
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    y_prob = clf.predict_proba(X_test_tfidf)[:, 1] if hasattr(clf, 'predict_proba') else None

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_test)) > 1:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None

    # Package model components for easy loading in apps
    model_bundle = {
        'vectorizer': vectorizer,
        'classifier': clf,
    }

    return model_bundle, metrics


def save_outputs(bundle, metrics: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / 'model.joblib'
    metrics_path = out_dir / 'metrics.json'
    joblib.dump(bundle, model_path)
    with open(metrics_path, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    return model_path, metrics_path


def main():
    parser = argparse.ArgumentParser(description='Train spam baseline (TF-IDF + Logistic Regression)')
    parser.add_argument('--data-url', default=(
        'https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv'
    ), help='CSV dataset URL (label,text)')
    parser.add_argument('--output-dir', default='out', help='Directory to write model and metrics')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    print('Loading dataset from', args.data_url)
    df = load_dataset(args.data_url)
    print(f'Loaded {len(df)} rows')

    X_train, X_test, y_train, y_test = prepare_data(df, test_size=args.test_size, random_state=args.random_state)
    print(f'Train size: {len(X_train)}, Test size: {len(X_test)}')

    bundle, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)

    out_dir = Path(args.output_dir)
    model_path, metrics_path = save_outputs(bundle, metrics, out_dir)

    print('Saved model to', model_path)
    print('Saved metrics to', metrics_path)
    print('Metrics:')
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
