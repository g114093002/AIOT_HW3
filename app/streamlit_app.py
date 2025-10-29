"""Simple Streamlit app to demo the spam classifier.

Run locally after training and producing `out/model.joblib`:

  pip install -r requirements.txt
  streamlit run app/streamlit_app.py

The app loads the joblib bundle (vectorizer + classifier) and exposes a text input
for live predictions.
"""
from pathlib import Path

import joblib
import streamlit as st


MODEL_PATH = Path('out/model.joblib')


@st.cache_data
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def predict_text(bundle, text: str):
    vec = bundle['vectorizer']
    clf = bundle['classifier']
    X = vec.transform([text])
    prob = clf.predict_proba(X)[0, 1] if hasattr(clf, 'predict_proba') else None
    pred = clf.predict(X)[0]
    label = 'spam' if int(pred) == 1 else 'ham'
    return label, prob


def main():
    st.title('Spam classifier demo')
    st.write('Enter a short SMS/email text to predict whether it is spam.')

    bundle = load_model(MODEL_PATH)
    if bundle is None:
        st.error(f'No model found at {MODEL_PATH}. Train model with `python src/ml/train_baseline.py`')
        return

    text = st.text_area('Message', value='Win a free iPhone by clicking here')
    if st.button('Predict'):
        label, prob = predict_text(bundle, text)
        st.write('Prediction:', label)
        if prob is not None:
            st.write(f'Probability spam: {prob:.3f}')


if __name__ == '__main__':
    main()
