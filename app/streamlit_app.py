"""Enhanced Streamlit demo for the spam classifier.

Features added compared to the minimal demo:
- Show model metrics (from out/metrics.json) if available
- Dataset preview (download live from canonical URL)
- Display top features (words) that indicate spam/ham from model coefficients
- Batch CSV upload and bulk classification
- Probability bar and nicer layout

Run locally after training and producing `out/model.joblib`:

  pip install -r requirements.txt
  streamlit run app/streamlit_app.py
"""
from pathlib import Path
import io
import json
from typing import Optional, Tuple, List

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.calibration import calibration_curve


MODEL_PATH = Path('out/model.joblib')
METRICS_PATH = Path('out/metrics.json')
DEFAULT_DATA_URL = (
    'https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv'
)


@st.cache_data(show_spinner=False)
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)
@st.cache_data(show_spinner=False)
def load_metrics(path: Path):
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_dataset(url: str, nrows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(url, header=None, encoding='utf-8', nrows=nrows)
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ['label', 'text']
    else:
        # fallback split
        s = df.iloc[:, 0].astype(str)
        parts = s.str.split(',', n=1, expand=True)
        parts.columns = ['label', 'text']
        df = parts
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['text'] = df['text'].astype(str)
    return df


def predict_text(bundle, text: str) -> Tuple[str, Optional[float], int, Optional[float]]:
    vec = bundle['vectorizer']
    clf = bundle['classifier']
    X = vec.transform([text])
    prob = clf.predict_proba(X)[0, 1] if hasattr(clf, 'predict_proba') else None
    pred = int(clf.predict(X)[0])
    score = clf.decision_function(X)[0] if hasattr(clf, 'decision_function') else None
    label = 'spam' if pred == 1 else 'ham'
    return label, prob, pred, score


def top_features(bundle, k: int = 20) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    vec = bundle['vectorizer']
    clf = bundle['classifier']
    try:
        feature_names = vec.get_feature_names_out()
    except Exception:
        feature_names = vec.get_feature_names()
    coefs = clf.coef_[0]
    top_spam_idx = coefs.argsort()[::-1][:k]
    top_ham_idx = coefs.argsort()[:k]
    top_spam = [(feature_names[i], float(coefs[i])) for i in top_spam_idx]
    top_ham = [(feature_names[i], float(coefs[i])) for i in top_ham_idx]
    return top_spam, top_ham


def bulk_classify(bundle, texts: List[str]) -> pd.DataFrame:
    vec = bundle['vectorizer']
    clf = bundle['classifier']
    X = vec.transform(texts)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1] if hasattr(clf, 'predict_proba') else [None] * len(texts)
    df = pd.DataFrame({'text': texts, 'prediction': preds, 'prob_spam': probs})
    df['label'] = df['prediction'].apply(lambda v: 'spam' if int(v) == 1 else 'ham')
    return df
def main():
    st.set_page_config(page_title='Spam classifier demo', layout='wide')
    st.title('Spam classifier demo')
    st.markdown(
        """
        This demo shows a simple TF-IDF + Logistic Regression spam classifier.
        Use the sample messages, type your own text, or upload a CSV to batch classify.
        """
    )

    left, right = st.columns([2, 1])

    # Load model and metrics
    bundle = load_model(MODEL_PATH)
    metrics = load_metrics(METRICS_PATH)

    # Sidebar: controls + quick stats
    with st.sidebar:
        st.title('Demo controls')
        st.markdown('Use these controls to inspect model and dataset behaviour.')

        st.subheader('Model')
        if bundle is None:
            st.error('No model found')
        else:
            st.success('Model loaded')
            if metrics:
                st.metric('Accuracy', f"{metrics.get('accuracy', 0):.3f}")
                st.metric('F1', f"{metrics.get('f1', 0):.3f}")
                st.metric('Recall', f"{metrics.get('recall', 0):.3f}")

        st.markdown('---')
        # Threshold in sidebar so it's always visible
        threshold = st.slider('Classification threshold (spam prob)', 0.0, 1.0, 0.5, step=0.01)

        st.markdown('---')
        st.subheader('Dataset preview')
        try:
            df_small = load_dataset(DEFAULT_DATA_URL, nrows=2000)
            counts = df_small['label'].value_counts()
            st.write(f"Sample rows: {len(df_small)}")
            st.write('Class distribution (sample)')
            st.bar_chart(counts)
            spam_ratio = counts.get('spam', 0) / counts.sum() if counts.sum() > 0 else 0
            st.write(f"Sample spam ratio: {spam_ratio:.2%}")
        except Exception:
            st.write('Dataset preview not available')

        st.markdown('---')
        st.subheader('Quick links')
        st.write('- Use the example messages or type your own')
        st.write('- Adjust threshold to trade off precision/recall')

    with right:
        st.header('Model')
        if bundle is None:
            st.error('No model found at `out/model.joblib`. Run training: `python src/ml/train_baseline.py`')
        else:
            st.success('Model loaded')
            if metrics:
                st.subheader('Metrics')
                # show key metrics in a compact form
                cols = st.columns(4)
                cols[0].metric('Accuracy', f"{metrics.get('accuracy', 0):.3f}")
                cols[1].metric('Precision', f"{metrics.get('precision', 0):.3f}")
                cols[2].metric('Recall', f"{metrics.get('recall', 0):.3f}")
                cols[3].metric('F1', f"{metrics.get('f1', 0):.3f}")
                if metrics.get('roc_auc') is not None:
                    st.write(f"ROC AUC: {metrics.get('roc_auc'):.3f}")

            # Top features (tokens) by class — always shown
            try:
                top_spam, top_ham = top_features(bundle, k=30)
                spam_df = pd.DataFrame(top_spam, columns=['token', 'weight']).head(30)
                ham_df = pd.DataFrame(top_ham, columns=['token', 'weight']).head(30)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('**Top spam tokens**')
                    st.dataframe(spam_df)
                    try:
                        st.bar_chart(spam_df.set_index('token')['weight'])
                    except Exception:
                        pass
                with c2:
                    st.markdown('**Top ham tokens**')
                    st.dataframe(ham_df)
                    try:
                        st.bar_chart(ham_df.set_index('token')['weight'])
                    except Exception:
                        pass
            except Exception:
                st.write('Top token extraction failed')

            # Model evaluation: compute metrics on a held-out sample and show plots
            try:
                st.markdown('---')
                st.subheader('Model evaluation (sample)')
                # load a sample of the dataset to compute evaluation
                df_eval = load_dataset(DEFAULT_DATA_URL, nrows=3000)
                X_all = df_eval['text'].values
                y_all = df_eval['label'].apply(lambda s: 1 if s in ('spam', '1', 'true', 't', 'yes') else 0).values
                from sklearn.model_selection import train_test_split

                X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all if len(set(y_all)) > 1 else None
                )
                vec = bundle['vectorizer']
                clf = bundle['classifier']
                X_test_tfidf = vec.transform(X_test_s)

                if hasattr(clf, 'predict_proba'):
                    y_prob_test = clf.predict_proba(X_test_tfidf)[:, 1]
                else:
                    y_prob_test = None

                # default predictions using current threshold
                if y_prob_test is not None:
                    y_pred_test = (y_prob_test >= threshold).astype(int)
                else:
                    y_pred_test = clf.predict(X_test_tfidf)

                # show numeric metrics
                acc = np.mean(y_pred_test == y_test_s)
                prec = precision_score(y_test_s, y_pred_test, zero_division=0)
                rec = recall_score(y_test_s, y_pred_test, zero_division=0)
                f1 = f1_score(y_test_s, y_pred_test, zero_division=0)
                mcols = st.columns(4)
                mcols[0].metric('Accuracy', f"{acc:.3f}")
                mcols[1].metric('Precision', f"{prec:.3f}")
                mcols[2].metric('Recall', f"{rec:.3f}")
                mcols[3].metric('F1', f"{f1:.3f}")

                # confusion matrix
                cm = confusion_matrix(y_test_s, y_pred_test)
                st.markdown('Confusion matrix (test sample)')
                cm_df = pd.DataFrame(cm, index=['actual_ham', 'actual_spam'], columns=['pred_ham', 'pred_spam'])
                st.dataframe(cm_df)

                # ROC curve
                if y_prob_test is not None:
                    fpr, tpr, _ = roc_curve(y_test_s, y_prob_test)
                    roc_auc = auc(fpr, tpr)
                    st.markdown(f'ROC AUC: {roc_auc:.3f}')
                    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr}).set_index('fpr')
                    st.line_chart(roc_df)

                # Precision-Recall curve
                if y_prob_test is not None:
                    precision_curve_vals, recall_curve_vals, _ = precision_recall_curve(y_test_s, y_prob_test)
                    pr_df = pd.DataFrame({'precision': precision_curve_vals, 'recall': recall_curve_vals})
                    st.markdown('Precision-Recall curve')
                    st.line_chart(pr_df)

                # probability histogram
                if y_prob_test is not None:
                    st.markdown('Probability distribution (spam probability)')
                    prob_df = pd.DataFrame({'prob_spam': y_prob_test})
                    st.bar_chart(pd.cut(y_prob_test, bins=10).value_counts().sort_index())

                # Calibration plot
                if y_prob_test is not None:
                    try:
                        prob_true, prob_pred = calibration_curve(y_test_s, y_prob_test, n_bins=10)
                        calib_df = pd.DataFrame({'prob_pred': prob_pred, 'prob_true': prob_true}).set_index('prob_pred')
                        st.markdown('Calibration curve (sample)')
                        st.line_chart(calib_df)
                    except Exception:
                        pass

                # Threshold sweep: precision/recall/f1
                if y_prob_test is not None:
                    thresholds = np.linspace(0.0, 1.0, 101)
                    pr_vals, rc_vals, f1_vals = [], [], []
                    for t in thresholds:
                        y_pred_t = (y_prob_test >= t).astype(int)
                        pr_vals.append(precision_score(y_test_s, y_pred_t, zero_division=0))
                        rc_vals.append(recall_score(y_test_s, y_pred_t, zero_division=0))
                        f1_vals.append(f1_score(y_test_s, y_pred_t, zero_division=0))
                    sweep_df = pd.DataFrame({'threshold': thresholds, 'precision': pr_vals, 'recall': rc_vals, 'f1': f1_vals}).set_index('threshold')
                    st.markdown('Threshold sweep (precision / recall / f1)')
                    st.line_chart(sweep_df)

                # Precision@k (top-k by prob)
                if y_prob_test is not None:
                    k_list = [10, 20, 50, 100]
                    pk = {}
                    sorted_idx = np.argsort(-y_prob_test)
                    for k in k_list:
                        topk = sorted_idx[:k]
                        pk[k] = float(np.mean(y_test_s[topk]))
                    pk_df = pd.DataFrame.from_dict(pk, orient='index', columns=['precision_at_k'])
                    st.markdown('Precision@k (test sample)')
                    st.table(pk_df)
            except Exception as e:
                st.write('Model evaluation failed:', e)

    with left:
        st.header('Try a single message')
        sample = st.text_area('Message', value='Win a free iPhone by clicking here', height=120)
        # example/sample messages picker
        example_messages = [
            'Win a free iPhone by clicking here',
            'Your package has shipped. Track it here',
            'Lowest price on meds, order now',
            'Meeting at 3pm today, please confirm',
            'Congratulations! You have been selected for a gift card',
        ]
        ex = st.selectbox('Or pick an example', ['(none)'] + example_messages)
        if ex and ex != '(none)':
            sample = ex
            st.text_area('Message (example chosen)', value=sample, height=120)
    # Threshold caption moved to sidebar; keep caption here for context
    with left:
        st.caption('Messages with probability >= threshold (sidebar) will be labeled as SPAM. Adjust to trade off precision vs recall.')

        if st.button('Predict'):
            if bundle is None:
                st.error('Model not available')
            else:
                label, prob, raw_pred, score = predict_text(bundle, sample)
                # apply threshold when probability is available; otherwise fall back to raw_pred
                if prob is not None:
                    applied_label = 'spam' if prob >= threshold else 'ham'
                else:
                    applied_label = 'spam' if int(raw_pred) == 1 else 'ham'

                st.markdown('### Prediction')
                pred_col1, pred_col2 = st.columns([1, 3])
                with pred_col1:
                    if applied_label == 'spam':
                        st.metric('Label', 'SPAM', delta=None)
                    else:
                        st.metric('Label', 'HAM', delta=None)
                with pred_col2:
                    if prob is not None:
                        st.write(f'Probability spam: **{prob:.3f}** (threshold = {threshold:.2f})')
                        # progress-like bar
                        st.progress(int(prob * 100))
                    else:
                        st.write('Probability not available; using raw prediction')

                # Show raw numeric outputs to aid debugging
                with st.expander('Raw model outputs'):
                    st.write({'raw_pred_class': int(raw_pred)})
                    if score is not None:
                        st.write({'decision_score': float(score)})
                    else:
                        st.write('decision_score: not available')

                # If we have probabilities and metrics, show ROC curve preview
                try:
                    if metrics and bundle is not None:
                        # compute ROC on sample of dataset (reuse dataset loader)
                        df = load_dataset(DEFAULT_DATA_URL, nrows=1000)
                        X = df['text'].values
                        y = df['label'].apply(lambda s: 1 if s in ('spam', '1', 'true', 't', 'yes') else 0).values
                        vec = bundle['vectorizer']
                        clf = bundle['classifier']
                        X_t = vec.transform(X)
                        y_prob = clf.predict_proba(X_t)[:, 1] if hasattr(clf, 'predict_proba') else None
                        if y_prob is not None:
                            from sklearn.metrics import roc_curve, auc

                            fpr, tpr, _ = roc_curve(y, y_prob)
                            roc_auc = auc(fpr, tpr)
                            st.write('ROC curve (sample)')
                            import pandas as _pd

                            roc_df = _pd.DataFrame({'fpr': fpr, 'tpr': tpr})
                            roc_df = roc_df.set_index('fpr')
                            st.line_chart(roc_df)
                            st.write(f'Area under curve: {roc_auc:.3f}')
                except Exception:
                    # fail silently for ROC in demo
                    pass

        st.markdown('---')

        st.header('Batch classify (CSV)')
        uploaded = st.file_uploader('Upload CSV with a single text column or header (label optional)', type=['csv'])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                # If single column without header
                if df.shape[1] == 1:
                    texts = df.iloc[:, 0].astype(str).tolist()
                elif 'text' in df.columns:
                    texts = df['text'].astype(str).tolist()
                else:
                    # take the last column as text
                    texts = df.iloc[:, -1].astype(str).tolist()
                res = bulk_classify(bundle, texts)
                # apply threshold-based labeling if probabilities are available
                if 'prob_spam' in res.columns:
                    res['pred_threshold'] = res['prob_spam'].apply(lambda p: 'spam' if (p is not None and p >= threshold) else 'ham')
                else:
                    res['pred_threshold'] = res['prediction'].apply(lambda v: 'spam' if int(v) == 1 else 'ham')

                st.write(res.head(200))
                csv = res.to_csv(index=False).encode('utf-8')
                st.download_button('Download results CSV', data=csv, file_name='predictions.csv')
            except Exception as e:
                st.error(f'Failed to parse CSV: {e}')

        st.markdown('---')

        if st.checkbox('Show dataset preview and compute confusion matrix'):
            try:
                df = load_dataset(DEFAULT_DATA_URL)
                st.write('Dataset sample:')
                st.dataframe(df.head(10))
                if bundle is not None:
                    # Recompute predictions on a held-out split to show confusion matrix
                    # Use same split strategy as training (test_size=0.2, random_state=42)
                    from sklearn.model_selection import train_test_split

                    X = df['text'].values
                    y = df['label'].apply(lambda s: 1 if s in ('spam', '1', 'true', 't', 'yes') else 0).values
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None)
                    vec = bundle['vectorizer']
                    clf = bundle['classifier']
                    X_test_tfidf = vec.transform(X_test)
                    # Use probability + threshold for prediction display when available
                    if hasattr(clf, 'predict_proba'):
                        y_prob_test = clf.predict_proba(X_test_tfidf)[:, 1]
                        y_pred = (y_prob_test >= threshold).astype(int)
                    else:
                        y_pred = clf.predict(X_test_tfidf)
                    cm = confusion_matrix(y_test, y_pred)
                    st.write('Confusion matrix (test split):')
                    cm_df = pd.DataFrame(cm, index=['actual_ham', 'actual_spam'], columns=['pred_ham', 'pred_spam'])
                    st.dataframe(cm_df)
            except Exception as e:
                st.error(f'Error loading dataset or computing matrix: {e}')


if __name__ == '__main__':
    main()
