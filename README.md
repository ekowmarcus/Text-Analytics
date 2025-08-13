# Toxic Comment Classification — FastText-Only Streamlit App

A page-to-page Streamlit app that detects toxic, offensive, and hateful language using **FastText** embeddings and classic ML models (**Logistic Regression** and **Random Forest**). It includes robust guards for common pitfalls (single-class data, empty vocabulary) and a polished Overview page.

---

## Features
- **FastText only** (no TF-IDF) with explicit `build_vocab` and safe retry if the vocab is empty.
- **Binary target**: if any of `toxic, severe_toxic, obscene, threat, insult, identity_hate` is 1 → label = 1; else 0.
- **Models**: Logistic Regression (balanced) and Random Forest (200 trees).
- **Metrics**: Precision, Recall, F1, ROC-AUC; Confusion matrices; ROC curves.
- **Pages**: Overview · Dataset · Modeling & Evaluation · Predict · Model Comparison · Conclusion.

---

## Requirements
- Python 3.9+ (3.8–3.12 works)
- Install dependencies:
  ```bash
  pip install streamlit gensim scikit-learn matplotlib nltk pandas numpy
  ```
  *(NLTK stopwords are optional; the app uses them if available.)*

---

## Quickstart
1. Save the app as **`group12_fasttext_only_app_fixed.py`**.
2. Run:
   ```bash
   streamlit run group12_fasttext_only_app_fixed.py
   ```
3. In the app:
   - Go to **Dataset** → click **Use Sample Data** to test instantly, or
   - Upload a CSV with at least a `comment_text` column (see format below).

---

## Data Format
- **Minimum column**: `comment_text`
- **Optional Jigsaw labels**: `toxic, severe_toxic, obscene, threat, insult, identity_hate`
- The app creates a binary label:
  ```
  is_toxic = (toxic + severe_toxic + obscene + threat + insult + identity_hate > 0)
  ```

**Example CSV**
```csv
comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate
"This is a great video!",0,0,0,0,0,0
"You are so stupid and worthless",1,0,1,0,1,0
```

---

## Workflow (Pages)

### 1) Overview
Project summary, quick metrics, and navigation.

### 2) Dataset
- Upload CSV or use sample data.
- Text cleaning (lowercase, strip URLs/HTML, keep letters).
- Builds `is_toxic`; shows rows, toxic %, class balance, preview, per-label counts.

### 3) Modeling & Evaluation
- Stratified train/test split (configurable test size).
- **FastText hyperparameters**:
  - Dimension (vector size): 50–300 (default 100)
  - Window: 2–10 (default 5)
  - Min word count: 1–10 (default 2; app retries with 1 if vocab is empty)
  - Algorithm: Skip-gram (`sg=1`) or CBOW (`sg=0`)
  - Epochs: 5–30 (default 10)
- Trains **Logistic Regression** and **Random Forest** on FastText document vectors.
- Displays metrics table, confusion matrices, and ROC curves.

### 4) Predict
- **Single comment**: probability + class (with adjustable threshold).
- **Batch CSV**: upload a file with a text column (`comment_text`, `comment`, `text`, or `content`) and download predictions.

### 5) Model Comparison
- Re-evaluate trained models on a fresh split; side-by-side metric table and bar chart.

### 6) Conclusion
- Short recap and suggested next steps.

---

## How It Works
- **Cleaning**: URL/HTML removal, non-letters dropped, whitespace normalized; optional stopword filtering.
- **Embedding**: FastText trained on your training texts; document vectors = mean of in-vocab token vectors (zeros if none).
- **Guards**:
  - **Single-class data** → clear error (you need both 0 and 1 classes).
  - **Empty vocabulary** → explicit `build_vocab`; if empty, auto-retry with `min_count=1` and explain the issue.

---

## Troubleshooting
**“Dataset has a single class only.”**  
Your data contains only toxic (1) or only non-toxic (0). Upload a CSV with both classes; optionally reduce test size.

**“FastText vocabulary is empty…” or “you must first build vocabulary…”**  
The app now builds vocab explicitly and retries with `min_count=1`. If it still fails, your cleaned texts may be empty (e.g., only URLs/emoji). Provide actual words or reduce cleaning strictness.

**Weak metrics / overfitting**  
Increase dataset size, adjust epochs, try CBOW vs Skip-gram, or tune `min_count` and the decision threshold.

---

## Customize
- **Branding/colors**: edit the CSS block in the script.
- **Models**: add SVM or XGBoost on top of FastText vectors.
- **Persistence**: save `fasttext_model` and the best classifier with `pickle`/`joblib` for reuse.
