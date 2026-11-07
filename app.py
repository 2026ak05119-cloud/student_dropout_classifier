import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics_utils import compute_metrics

# ---------- page setup ----------
st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="wide")

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
MODEL_PATHS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "KNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Support Vector Machine": MODEL_DIR / "support_vector_machine.joblib",
}
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


@st.cache_resource
def load_everything():
    with open(MODEL_DIR / "meta.json") as fh:
        meta_info = json.load(fh)
    label_enc = joblib.load(MODEL_DIR / "label_encoder.joblib")
    fitted = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
    return meta_info, label_enc, fitted


meta_info, label_enc, fitted = load_everything()
target_col = meta_info["target"]
feature_cols = meta_info["features"]
class_names = [str(c) for c in label_enc.classes_]


def score_model(model_name, frame, y_encoded):
    est = fitted[model_name]
    preds = est.predict(frame[feature_cols])
    probs = est.predict_proba(frame[feature_cols])
    return preds, compute_metrics(y_encoded, preds, probs)


# ---------- header ----------
st.title("🎓 Student Success Predictor")
st.markdown(
    "Evaluate five pre-trained classifiers on student records and see how well each one "
    "separates **Dropout**, **Enrolled** and **Graduate** outcomes. "
    "Models are loaded from disk, so nothing is retrained on your upload."
)

# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Setup")
    st.write("**Step 1.** Upload the held-out test CSV.")
    uploaded = st.file_uploader("Test data (CSV)", type="csv")
    st.write("**Step 2.** Choose a model to inspect.")
    chosen = st.selectbox("Classifier", list(MODEL_PATHS))
    st.divider()
    st.caption("Dataset: UCI - Predict Students' Dropout and Academic Success")
    st.caption(f"Needs {len(feature_cols)} feature columns plus `{target_col}`.")
    with st.expander("Show expected columns"):
        st.write(feature_cols + [target_col])

if uploaded is None:
    st.info("👈 Upload `test_data.csv` from the project repository to get started.")
    st.stop()

# ---------- validate the upload ----------
data = pd.read_csv(uploaded)
data.columns = data.columns.str.strip()
absent = [c for c in feature_cols + [target_col] if c not in data.columns]
if absent:
    st.error(f"The uploaded file is missing these columns: {absent}")
    st.stop()

label_lookup = {name: idx for idx, name in enumerate(class_names)}
raw_labels = data[target_col].astype(str).str.strip()
unknown = sorted(set(raw_labels) - set(label_lookup))
if unknown:
    st.error(f"Unrecognised labels in `{target_col}`: {unknown}. Expected: {class_names}")
    st.stop()
y_true = raw_labels.map(label_lookup).values

# ---------- tabs ----------
tab_perf, tab_cm, tab_all, tab_data = st.tabs(
    ["📈 Performance", "🧩 Confusion matrix & report", "🏁 Compare all models", "🗂️ Uploaded data"]
)

y_pred, scores = score_model(chosen, data, y_true)

with tab_perf:
    st.subheader(f"{chosen}: results on {len(data)} test students")
    for box, metric in zip(st.columns(6), METRIC_ORDER):
        box.metric(metric, f"{scores[metric]:.4f}")
    st.caption("Precision, Recall and F1 are weighted averages over the three classes; "
               "AUC is one-vs-rest weighted.")

with tab_cm:
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(4.8, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="YlGnBu",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(y_true, y_pred, target_names=class_names,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3))

with tab_all:
    st.subheader("All five models on this file")
    table = pd.DataFrame(
        {name: score_model(name, data, y_true)[1] for name in MODEL_PATHS}
    ).T[METRIC_ORDER].round(4)
    st.dataframe(table.style.highlight_max(axis=0, color="#c8e6c9"))
    st.bar_chart(table[["Accuracy", "F1", "MCC"]])
    best_name = table["MCC"].idxmax()
    st.success(f"Highest MCC on this file: **{best_name}** ({table.loc[best_name, 'MCC']:.4f})")

with tab_data:
    st.subheader("Preview of the uploaded file")
    st.write(f"{data.shape[0]} rows x {data.shape[1]} columns")
    st.dataframe(data.head(20))
    st.write("Class counts in the uploaded data:")
    st.bar_chart(raw_labels.value_counts())
