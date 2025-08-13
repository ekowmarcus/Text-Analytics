# group12_fasttext_only_app_fixed.py — Toxic Comment Classification (FastText ONLY, vocab-safe)
# --------------------------------------------------------------------------------------------
# Pages: Overview • Dataset • Modeling & Evaluation • Predict • Model Comparison • Conclusion
# Embedding: FastText (gensim) only — no TF-IDF fallback
# Models: Logistic Regression & Random Forest
# Metrics: Precision, Recall, F1, ROC-AUC + Confusion Matrices + ROC Curves
# Robust session_state init + class-balance guard + explicit FastText vocab building

import warnings
warnings.filterwarnings("ignore")

import re, numpy as np, pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# ---- REQUIRED: FastText / gensim ----
try:
    from gensim.models import FastText
    from gensim.utils import simple_preprocess
except Exception:
    st.set_page_config(page_title="Group 12 • Toxic Comment Classifier", page_icon="🛡️", layout="wide")
    st.error(
        "This app requires **gensim** for FastText.\n\n"
        "Install with:\n\n"
        "`pip install gensim`"
    )
    st.stop()

# Optional NLTK stopwords (best effort)
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOP = set(stopwords.words('english'))
except Exception:
    STOP = set()

# --------------------------------- Aesthetics ---------------------------------
st.set_page_config(page_title="Group 12 • Toxic Comment Classifier", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
:root{ --brand:#0f4c81; --bg:#f7fafc; --card:#ffffff; --muted:#64748b; --ink:#0f172a; }
.main { background: var(--bg); }
section[data-testid="stSidebar"] { background: #0f4c8112; }
h1,h2,h3 { font-weight: 800; letter-spacing:.2px; }
.hero { padding:1rem 1.25rem; background:linear-gradient(90deg,#0f4c81 0%,#1d75b3 100%);
        color:#fff; border-radius:16px; box-shadow:0 10px 30px rgba(15,76,129,.25); }
.hero h1 { margin:0 0 .25rem 0; font-size:2rem; }
.hero p { margin:.25rem 0 0 0; opacity:.95; }
.metric { background:var(--card); border:1px solid #e5e7eb; border-radius:14px; padding:14px; text-align:center; }
.metric h4{ margin:.1rem 0; color:var(--muted); font-weight:700;}
.metric div{ font-size:1.2rem; font-weight:800; color:var(--ink);}
.stButton>button{ background:var(--brand); color:#fff; border:0; border-radius:12px; padding:.6rem 1rem; font-weight:700; }
.stButton>button:hover{ filter:brightness(1.07); }
</style>
""", unsafe_allow_html=True)

# --------------------------- Session-state defaults ---------------------------
DEFAULTS = {
    "df": None,
    "fasttext_model": None,     # gensim FastText model
    "fasttext_dim": 100,
    "trained_models": {},       # {"Logistic Regression": est, "Random Forest": est}
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------------- Helpers ---------------------------
TOX_COLS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

def clean_text(x: str) -> str:
    if pd.isna(x): return ''
    t = str(x).lower()
    t = re.sub(r'http\S+|www\S+|https\S+', ' ', t)   # URLs
    t = re.sub(r'<.*?>', ' ', t)                     # HTML
    t = re.sub(r'[^a-zA-Z\s]', ' ', t)               # keep letters/spaces only
    t = re.sub(r'\s+', ' ', t).strip()
    if STOP:
        toks = [w for w in t.split() if w not in STOP and len(w) > 2]
        return ' '.join(toks)
    return t

def build_binary_target(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in TOX_COLS if c in df.columns]
    if cols:
        return (df[cols].sum(axis=1) > 0).astype(int)
    if 'label' in df.columns: return df['label'].astype(int)
    if 'is_toxic' in df.columns: return df['is_toxic'].astype(int)
    # last resort: all zero (forces user to provide labels)
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

def ensure_two_classes(X: pd.Series, y: pd.Series):
     # TODO: implement
    pass
    counts = y.value_counts()
    if y.nunique() < 2:
        raise ValueError("Dataset has a single class only. Include both toxic (1) and non-toxic (0) rows.")
    # light upsample if minority is tiny to stabilize stratified split
    if counts.min() < 5:
        maj = counts.idxmax(); minc = counts.idxmin()
        df = pd.DataFrame({'text': X, 'label': y})
        df_min = df[df.label == minc].sample(n=max(10, counts[minc]*3), replace=True, random_state=42)
        df_maj = df[df.label == maj]
        df_bal = pd.concat([df_maj, df_min], ignore_index=True).sample(frac=1.0, random_state=42)
        return df_bal['text'], df_bal['label']
    return X, y

def fasttext_doc_vec(ft: FastText, text: str, dim: int) -> np.ndarray:
    if pd.isna(text) or not str(text).strip(): return np.zeros(dim)
    toks = simple_preprocess(str(text), deacc=True)
    vecs = [ft.wv[w] for w in toks if w in ft.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

def make_fasttext(train_texts, test_texts, dim=100, window=5, min_count=2, sg=1, epochs=10):
    """Train FastText robustly: explicit build_vocab, retry with min_count=1 if vocab is empty."""
    # Normalize inputs
    train_texts = pd.Series(train_texts).fillna("").astype(str)
    test_texts  = pd.Series(test_texts).fillna("").astype(str)

    # Tokenize training texts into sentences (lists of tokens)
    sents = [simple_preprocess(t, deacc=True) for t in train_texts]
    # Drop completely empty sentences
    sents = [s for s in sents if len(s) > 0]

    if len(sents) == 0:
        raise ValueError("No valid tokens in training data after cleaning. Check your input or lower min_count.")

    # Initialize FastText WITHOUT sentences (so we control build_vocab explicitly)
    ft = FastText(vector_size=dim, window=window, min_count=min_count, workers=4, sg=sg)
    # Build vocab from sentences
    ft.build_vocab(corpus_iterable=sents, update=False)

    # If vocab is empty due to a too-high min_count or tiny dataset, retry with min_count=1
    if len(ft.wv) == 0:
        if min_count > 1:
            st.info(f"FastText: vocabulary empty with min_count={min_count}; retrying with min_count=1.")
            ft = FastText(vector_size=dim, window=window, min_count=1, workers=4, sg=sg)
            ft.build_vocab(corpus_iterable=sents, update=False)
        if len(ft.wv) == 0:
            raise ValueError("FastText vocabulary is empty even with min_count=1. Text may be too short after cleaning.")

    # Train model
    ft.train(corpus_iterable=sents, total_examples=len(sents), epochs=epochs)

    # Build document vectors
    Xtr = np.vstack([fasttext_doc_vec(ft, t, ft.vector_size) for t in train_texts])
    Xte = np.vstack([fasttext_doc_vec(ft, t, ft.vector_size) for t in test_texts])

    return Xtr, Xte, ft

# --- Models & evaluation (FastText vectors -> classic ML) ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def evaluate(est, Xte, yte, thresh=0.5):
    prob = est.predict_proba(Xte)[:,1]
    pred = (prob >= thresh).astype(int)
    metrics = {
        'precision': precision_score(yte, pred, zero_division=0),
        'recall':    recall_score(yte, pred, zero_division=0),
        'f1':        f1_score(yte, pred, zero_division=0),
        'roc_auc':   roc_auc_score(yte, prob)
    }
    cm = confusion_matrix(yte, pred)
    fpr, tpr, _ = roc_curve(yte, prob)
    return metrics, cm, (fpr, tpr)

def demo_dataframe():
    data = {
        'comment_text': [
            "This is a great video, thanks for sharing!",
            "You are so stupid and worthless",
            "I disagree with your opinion but respect your view",
            "Go kill yourself, nobody likes you",
            "Amazing content, keep up the good work!",
            "What an idiot, this is garbage",
            "Thanks for the tutorial, very helpful",
            "You're a pathetic loser",
            "Nice explanation, learned something new",
            "Hate this channel, waste of time"
        ],
        'toxic': [0,1,0,1,0,1,0,1,0,1],
        'severe_toxic':[0,0,0,1,0,0,0,0,0,0],
        'obscene':[0,1,0,1,0,1,0,1,0,0],
        'threat':[0,0,0,1,0,0,0,0,0,0],
        'insult':[0,1,0,1,0,1,0,1,0,0],
        'identity_hate':[0]*10
    }
    return pd.DataFrame(data)

# --------------------------------- PAGES ---------------------------------
def page_overview():
    st.markdown("""
<div class="hero">
  <h1>Group 12 • Toxic Comment Classification</h1>
  <p>Detect toxic, offensive, and hateful language using <b>FastText</b> embeddings and classic ML models.</p>
</div>
""", unsafe_allow_html=True)
    st.write("")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown('<div class="metric"><h4>Embedding</h4><div>FastText</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric"><h4>Models</h4><div>LogReg & RF</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric"><h4>Metrics</h4><div>Precision • Recall • F1 • AUC</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric"><h4>Dataset</h4><div>Jigsaw (Kaggle)</div></div>', unsafe_allow_html=True)

    st.markdown("### Workflow")
    st.markdown("**Dataset → Modeling & Evaluation → Predict → Model Comparison → Conclusion**")

def page_dataset():
    st.subheader("Dataset")
    c1, c2 = st.columns([2,1])
    with c1:
        up = st.file_uploader("Upload Jigsaw-style CSV (must include `comment_text`)", type=['csv'])
    with c2:
        if st.button("Use Sample Data", use_container_width=True):
            st.session_state["df"] = demo_dataframe()

    if up is not None:
        try:
            st.session_state["df"] = pd.read_csv(up)
            st.success("File loaded!")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if st.session_state["df"] is None:
        st.info("Upload a CSV or click **Use Sample Data**.")
        return

    df = st.session_state["df"].copy()
    if "comment_text" not in df.columns:
        st.error("CSV must include a text column named `comment_text`.")
        return

    df["comment_text_clean"] = df["comment_text"].apply(clean_text)
    df["is_toxic"] = build_binary_target(df)

    # Quick stats
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric"><h4>Rows</h4><div>{len(df):,}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric"><h4>Toxic %</h4><div>{(df["is_toxic"].mean()*100):.1f}%</div></div>', unsafe_allow_html=True)
    with c3:
        classes = df["is_toxic"].value_counts().to_dict()
        st.markdown(f'<div class="metric"><h4>Class balance</h4><div>{classes}</div></div>', unsafe_allow_html=True)

    st.markdown("#### Preview")
    st.dataframe(df.head(12), use_container_width=True)

    # Label bars if original columns exist
    present = [c for c in TOX_COLS if c in df.columns]
    if present:
        st.markdown("#### Label coverage")
        st.bar_chart(df[present].sum())

    st.session_state["df"] = df

def page_modeling():
    st.subheader("Modeling & Evaluation")

    df = st.session_state.get("df")
    if df is None or "comment_text_clean" not in df.columns:
        st.warning("Go to **Dataset** and upload/prepare your data first.")
        return

    X_raw, y_raw = df['comment_text_clean'], df['is_toxic']
    try:
        X_bal, y_bal = ensure_two_classes(X_raw, y_raw)
    except ValueError as e:
        st.error(str(e)); return

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    from sklearn.model_selection import train_test_split
    Xtr_txt, Xte_txt, ytr, yte = train_test_split(X_bal, y_bal, test_size=test_size,
                                                 random_state=42, stratify=y_bal)

    # FastText hyperparameters
    colA, colB, colC, colD = st.columns(4)
    dim   = colA.slider("FT dimension", 50, 300, 100, 50)
    win   = colB.slider("FT window", 2, 10, 5)
    minc  = colC.slider("Min word count", 1, 10, 2)
    algo  = colD.selectbox("Algorithm", ["Skip-gram (sg=1)", "CBOW (sg=0)"], index=0)
    sg    = 1 if algo.startswith("Skip") else 0
    epochs= st.slider("Epochs", 5, 30, 10)

    with st.spinner("Training FastText & building document vectors..."):
        try:
            Xtr, Xte, ft = make_fasttext(Xtr_txt, Xte_txt, dim=dim, window=win, min_count=minc, sg=sg, epochs=epochs)
        except Exception as e:
            st.error(f"FastText failed: {e}")
            return
        st.session_state["fasttext_model"] = ft
        st.session_state["fasttext_dim"] = dim
        st.success(f"FastText ready: train {Xtr.shape}, test {Xte.shape}")

    # Train models on FastText vectors
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    with st.spinner("Training models..."):
        lr = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced').fit(Xtr, ytr)
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        st.session_state["trained_models"] = {"Logistic Regression": lr, "Random Forest": rf}

    # Evaluate
    rows, rocs = [], {}
    for name, est in st.session_state["trained_models"].items():
        m, cm, roc = evaluate(est, Xte, yte)
        rows.append({"model": name, **m})
        rocs[name] = roc
    dfm = pd.DataFrame(rows).set_index('model').round(4)

    st.markdown("#### Test metrics")
    st.dataframe(dfm, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Confusion matrices")
        for name, est in st.session_state["trained_models"].items():
            _, cm, _ = evaluate(est, Xte, yte)
            st.markdown(f"**{name}**")
            st.write(pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1']))
    with col2:
        st.markdown("#### ROC curves")
        fig, ax = plt.subplots(figsize=(5.5,4.5))
        for name, (fpr, tpr) in rocs.items():
            ax.plot(fpr, tpr, label=f"{name} (AUC={dfm.loc[name,'roc_auc']:.3f})")
        ax.plot([0,1],[0,1],'k--', alpha=0.6)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)

def page_predict():
    st.subheader("Predict")
    if not st.session_state["trained_models"]:
        st.warning("Train models in **Modeling & Evaluation** first.")
        return

    model_names = list(st.session_state["trained_models"].keys())
    tab1, tab2 = st.tabs(["Single comment", "Batch CSV"])

    with tab1:
        text = st.text_area("Enter a comment", height=120, placeholder="Type/paste a comment…")
        model_sel = st.selectbox("Model", model_names, index=0)
        thr = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05)
        if st.button("Predict"):
            ft = st.session_state["fasttext_model"]; dim = st.session_state["fasttext_dim"]
            Xone = np.vstack([fasttext_doc_vec(ft, clean_text(text), dim)])
            p = st.session_state["trained_models"][model_sel].predict_proba(Xone)[:,1][0]
            pred = int(p >= thr)
            st.markdown(f"**Probability (toxic):** {p:.3f} → **Pred:** {'🚨 TOXIC' if pred==1 else '✅ SAFE'}")

    with tab2:
        up2 = st.file_uploader("Upload CSV for batch predictions (must contain a text column)", type=['csv'])
        if up2 is not None:
            dfb = pd.read_csv(up2)
            # pick a likely text column
            text_col = None
            for c in ['comment_text','comment','text','content']:
                if c in dfb.columns: text_col = c; break
            if text_col is None:
                st.error("No valid text column found.")
            else:
                model_sel_b = st.selectbox("Model", model_names, index=0, key="ms2")
                thr_b = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05, key="thr_b")
                ft = st.session_state["fasttext_model"]; dim = st.session_state["fasttext_dim"]
                Xb = np.vstack([fasttext_doc_vec(ft, clean_text(t), dim) for t in dfb[text_col].fillna("").astype(str)])
                prob = st.session_state["trained_models"][model_sel_b].predict_proba(Xb)[:,1]
                pred = (prob >= thr_b).astype(int)
                out = dfb.copy()
                out['toxicity_probability'] = prob
                out['predicted_toxic'] = pred
                st.dataframe(out.head(20), use_container_width=True)
                csv = out.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV",
                                   data=csv,
                                   file_name=f"toxicity_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv")

def page_compare():
    st.subheader("Model Comparison")
    if not st.session_state["trained_models"]:
        st.info("Train models on **Modeling & Evaluation** first.")
        return

    # Quick side-by-side metric table using a fresh test split
    df = st.session_state.get("df")
    if df is None or "comment_text_clean" not in df.columns:
        st.warning("Go back to **Dataset** and prepare data.")
        return

    X_raw, y_raw = df['comment_text_clean'], df['is_toxic']
    try:
        X_bal, y_bal = ensure_two_classes(X_raw, y_raw)
    except ValueError as e:
        st.error(str(e)); return

    from sklearn.model_selection import train_test_split
    Xtr_txt, Xte_txt, ytr, yte = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)

    ft = st.session_state["fasttext_model"]; dim = st.session_state["fasttext_dim"]
    Xte = np.vstack([fasttext_doc_vec(ft, t, dim) for t in Xte_txt])

    rows = []
    for name, est in st.session_state["trained_models"].items():
        m, _, _ = evaluate(est, Xte, yte)
        rows.append({"model": name, **m})
    dfm = pd.DataFrame(rows).set_index("model").round(4)
    st.dataframe(dfm, use_container_width=True)

    fig, ax = plt.subplots(figsize=(6,4))
    dfm[['precision','recall','f1','roc_auc']].plot(kind='bar', ax=ax)
    ax.set_ylim(0,1); ax.set_title("Metric comparison (FastText vectors)")
    st.pyplot(fig)

def page_conclusion():
    st.subheader("Conclusion & Next Steps")
    st.markdown(
        "- **What worked:** FastText + Logistic Regression is strong and fast; RF is robust.\n"
        "- **What to try next:** class-weight/threshold tuning, ElasticNet LR, pre-trained sentence transformers.\n"
        "- **Deployment:** freeze the FastText model and the best classifier; serve with Streamlit."
    )
    st.markdown("**Thanks!** 🎯")

# ----------------------------- Navigation -----------------------------
PAGES = {
    "Overview": page_overview,
    "Dataset": page_dataset,
    "Modeling & Evaluation": page_modeling,
    "Predict": page_predict,
    "Model Comparison": page_compare,
    "Conclusion": page_conclusion,
}
choice = st.sidebar.selectbox("📚 Navigate", list(PAGES.keys()))
PAGES[choice]()


