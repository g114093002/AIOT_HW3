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
import streamlit as st
from sklearn.metrics import confusion_matrix


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


def predict_text(bundle, text: str) -> Tuple[str, Optional[float]]:
    vec = bundle['vectorizer']
    clf = bundle['classifier']
    X = vec.transform([text])
    prob = clf.predict_proba(X)[0, 1] if hasattr(clf, 'predict_proba') else None
    pred = clf.predict(X)[0]
    label = 'spam' if int(pred) == 1 else 'ham'
    return label, prob


def top_features(bundle, k: int = 20) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Return top-k features for spam (positive) and ham (negative).

    Requires a linear classifier with `coef_` and a vectorizer with `get_feature_names_out()`.
    """
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

            # Top features
            if st.checkbox('Show top indicative features'):
                top_spam, top_ham = top_features(bundle, k=20)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('**Top spam features**')
                    for f, w in top_spam[:20]:
                        st.write(f'{f} — {w:.3f}')
                with c2:
                    st.markdown('**Top ham features**')
                    for f, w in top_ham[:20]:
                        st.write(f'{f} — {w:.3f}')

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
        if st.button('Predict'):
            if bundle is None:
                st.error('Model not available')
            else:
                label, prob = predict_text(bundle, sample)
                st.markdown('### Prediction')
                pred_col1, pred_col2 = st.columns([1, 3])
                with pred_col1:
                    if label == 'spam':
                        st.metric('Label', 'SPAM', delta=None)
                    else:
                        st.metric('Label', 'HAM', delta=None)
                with pred_col2:
                    if prob is not None:
                        st.write(f'Probability spam: **{prob:.3f}**')
                        # progress-like bar
                        st.progress(int(prob * 100))
                    else:
                        st.write('Probability not available')

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
                    y_pred = clf.predict(X_test_tfidf)
                    cm = confusion_matrix(y_test, y_pred)
                    st.write('Confusion matrix (test split):')
                    cm_df = pd.DataFrame(cm, index=['actual_ham', 'actual_spam'], columns=['pred_ham', 'pred_spam'])
                    st.dataframe(cm_df)
            except Exception as e:
                st.error(f'Error loading dataset or computing matrix: {e}')


if __name__ == '__main__':
    main()
