import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from metrics_utils import compute_metrics

st.set_page_config(page_title="Classifier Evaluation Lab", layout="wide")
st.title("Classifier Evaluation Lab")
st.caption("Upload a test CSV, pick a pre-trained model, and inspect its performance. No retraining happens here.")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

@st.cache_resource
def load_assets():
    meta = json.load(open("model/meta.json"))
    le = joblib.load("model/label_encoder.joblib")
    models = {k: joblib.load(v) for k, v in MODEL_FILES.items()}
    return meta, le, models

meta, le, models = load_assets()
target, features = meta["target"], meta["features"]

with st.sidebar:
    st.header("Controls")
    choice = st.selectbox("Select model", list(MODEL_FILES))
    up = st.file_uploader("Upload test CSV", type="csv")
    st.markdown(f"**Required columns:** {len(features)} features + `{target}`")
    with st.expander("Feature list"):
        st.write(features)

if up is None:
    st.info("Upload your test CSV (e.g. test_data.csv from the repo) in the sidebar to begin.")
    st.stop()

df = pd.read_csv(up)
missing = [c for c in features + [target] if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(), use_container_width=True)

known = set(le.classes_.astype(str))
if not set(df[target].astype(str)).issubset(known):
    st.error(f"Target contains labels not seen in training. Expected: {sorted(known)}")
    st.stop()
y_true = le.transform(df[target].astype(le.classes_.dtype.type) if le.classes_.dtype.kind in "iuf" else df[target].astype(str))
X = df[features]

def evaluate(name):
    pipe = models[name]
    return y_true, pipe.predict(X), pipe.predict_proba(X)

yt, yp, ypr = evaluate(choice)
m = compute_metrics(yt, yp, ypr)

st.subheader(f"Results: {choice}")
cols = st.columns(6)
for c, (k, v) in zip(cols, m.items()):
    c.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(confusion_matrix(yt, yp), annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = classification_report(yt, yp, target_names=[str(c) for c in le.classes_], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

st.subheader("All models on this test file")
rows = [{"Model": n, **compute_metrics(*(lambda r: (r[0], r[1], r[2]))(evaluate(n)))} for n in models]
comp = pd.DataFrame(rows).set_index("Model").round(4)
st.dataframe(comp.style.highlight_max(axis=0, color="#cfe8cf"), use_container_width=True)
st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
