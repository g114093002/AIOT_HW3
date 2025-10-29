# Notebooks

Place exploratory notebooks in this directory. For the spam baseline project we recommend:

- `notebooks/spam-classification-baseline.ipynb` — end-to-end notebook that:
  - downloads and inspects the dataset
  - performs preprocessing and feature extraction
  - trains the baseline model (TF-IDF + Logistic Regression)
  - evaluates and visualizes metrics

Run the training script instead of the notebook for CI-friendly reproducibility:

```bash
python src/ml/train_baseline.py --output-dir out
```

After training, run the Streamlit demo:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
