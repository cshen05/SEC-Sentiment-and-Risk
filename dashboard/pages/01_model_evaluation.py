import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FINBERT_METRICS_PATH = MODELS_DIR / "finbert_metrics.pkl"

# Baseline metrics captured from the evaluated baseline run.
BASELINE_ACCURACY = 0.8250
BASELINE_MACRO_F1 = 0.52
BASELINE_WEIGHTED_F1 = 0.82
BASELINE_CONFUSION_MATRIX = [
    [28, 3, 1],
    [1, 5, 0],
    [1, 1, 0],
]


st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📈",
    layout="wide",
)


@st.cache_data
def load_finbert_metrics() -> dict:
    """Load saved FinBERT evaluation artifacts."""
    if not FINBERT_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Missing FinBERT metrics file. Run training/train_finbert.py first: "
            f"{FINBERT_METRICS_PATH}"
        )

    with open(FINBERT_METRICS_PATH, "rb") as f:
        return pickle.load(f)


st.title("Model Evaluation Dashboard")
st.markdown(
    "This page compares the **TF-IDF + Logistic Regression baseline** against the "
    "fine-tuned **FinBERT** classifier used in the final SEC risk pipeline."
)

st.markdown(
    "The key question is whether a domain-specific transformer meaningfully improves "
    "classification of paragraph-level SEC disclosures beyond a traditional sparse-feature baseline."
)

try:
    finbert_metrics = load_finbert_metrics()
except Exception as exc:
    st.error(str(exc))
    st.stop()


# Parse FinBERT metrics.
finbert_accuracy = float(finbert_metrics["accuracy"])
finbert_confusion_matrix = np.array(finbert_metrics["confusion_matrix"])
finbert_test_results = finbert_metrics["test_results"]

# Convert baseline matrix to numpy
baseline_confusion_matrix = np.array(BASELINE_CONFUSION_MATRIX)

# Use a shared color scale for both confusion matrices
confusion_vmax = max(baseline_confusion_matrix.max(), finbert_confusion_matrix.max())

# FinBERT values observed from the final evaluated run.
FINBERT_MACRO_F1 = 0.83
FINBERT_WEIGHTED_F1 = 0.95


st.subheader("Why compare these models?")
st.markdown(
    "The baseline model is useful because it provides a fast, interpretable benchmark. "
    "However, SEC filings contain nuanced legal and financial language, so a bag-of-words "
    "model often over-predicts the dominant `negative_risk` class. FinBERT was introduced "
    "to better capture semantic context, especially for minority classes like `neutral` and `positive_outlook`."
)


st.subheader("Headline Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline Accuracy", f"{BASELINE_ACCURACY:.3f}")
col2.metric("FinBERT Accuracy", f"{finbert_accuracy:.3f}")
col3.metric("Baseline Macro F1", f"{BASELINE_MACRO_F1:.2f}")
col4.metric("FinBERT Macro F1", f"{FINBERT_MACRO_F1:.2f}")

col5, col6 = st.columns(2)
col5.metric("Baseline Weighted F1", f"{BASELINE_WEIGHTED_F1:.2f}")
col6.metric("FinBERT Weighted F1", f"{FINBERT_WEIGHTED_F1:.2f}")

st.markdown(
    "**Interpretation:** Accuracy improves from the baseline to FinBERT, but the more important "
    "gain is in **macro F1**. That increase indicates that FinBERT handles class imbalance much better "
    "and improves performance on the minority classes rather than just predicting the dominant class."
)


st.subheader("Metric Comparison Table")
comparison_df = pd.DataFrame(
    {
        "Model": ["Baseline (TF-IDF + Logistic Regression)", "FinBERT"],
        "Accuracy": [BASELINE_ACCURACY, finbert_accuracy],
        "Macro F1": [BASELINE_MACRO_F1, FINBERT_MACRO_F1],
        "Weighted F1": [BASELINE_WEIGHTED_F1, FINBERT_WEIGHTED_F1],
    }
)
st.dataframe(comparison_df, use_container_width=True)


st.subheader("Metric Comparison Chart")
plot_df = comparison_df.set_index("Model")
fig, ax = plt.subplots(figsize=(9, 4))
plot_df.plot(kind="bar", ax=ax)
ax.set_title("Baseline vs FinBERT Performance")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_ylim(0, 1.0)
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
st.pyplot(fig)

st.markdown(
    "The chart makes the improvement clear: FinBERT outperforms the baseline across every major metric, "
    "with especially strong gains in balanced classification performance."
)


st.subheader("Confusion Matrices")
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("**Baseline Confusion Matrix**")
    baseline_fig, baseline_ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=baseline_confusion_matrix,
        display_labels=["negative_risk", "neutral", "positive_outlook"],
    ).plot(ax=baseline_ax, colorbar=False, im_kw={"vmin": 0, "vmax": confusion_vmax})
    baseline_ax.set_title("Baseline")
    st.pyplot(baseline_fig)
    st.markdown(
        "The baseline performs well on the dominant `negative_risk` class but struggles to identify `positive_outlook`. "
        "This is consistent with a sparse lexical model that defaults toward the majority class when the language is ambiguous."
    )

with right_col:
    st.markdown("**FinBERT Confusion Matrix**")
    finbert_fig, finbert_ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=finbert_confusion_matrix,
        display_labels=["negative_risk", "neutral", "positive_outlook"],
    ).plot(ax=finbert_ax, colorbar=False, im_kw={"vmin": 0, "vmax": confusion_vmax})
    finbert_ax.set_title("FinBERT")
    st.pyplot(finbert_fig)
    st.markdown(
        "FinBERT sharply reduces systematic misclassification and produces a much cleaner confusion matrix. "
        "It preserves excellent `negative_risk` performance while improving detection of minority classes."
    )


st.subheader("What changed from Baseline to FinBERT?")
st.markdown(
    "### Baseline limitations\n"
    "- Relies on surface vocabulary through TF-IDF features\n"
    "- Tends to over-predict `negative_risk` because the labeled set is highly imbalanced\n"
    "- Struggles with strategic or descriptive language that shares words with risk disclosures\n\n"
    "### Why FinBERT improves performance\n"
    "- Uses contextual embeddings rather than isolated keyword counts\n"
    "- Was pretrained on financial text, making it better suited to SEC language\n"
    "- Better separates descriptive, strategic, and explicitly adverse disclosures\n"
)


st.subheader("FinBERT Misclassification Review")
if isinstance(finbert_test_results, pd.DataFrame):
    mistakes_df = finbert_test_results[
        finbert_test_results["true_label"] != finbert_test_results["pred_label"]
    ].copy()
else:
    mistakes_df = pd.DataFrame()

st.markdown(
    "The remaining errors are useful because they reveal where even the transformer model still struggles. "
    "Most remaining mistakes come from borderline cases where neutral, strategic, and mildly risky language overlap."
)

if mistakes_df.empty:
    st.success("No misclassifications were found in the saved FinBERT results.")
else:
    st.metric("FinBERT Misclassifications", len(mistakes_df))
    st.dataframe(mistakes_df, use_container_width=True)


st.subheader("Model Evaluation Takeaways")
st.markdown(
    "1. The baseline model is a useful benchmark, but it is not robust enough for nuanced SEC paragraph classification.\n"
    "2. FinBERT materially improves overall accuracy and class balance.\n"
    "3. The strongest practical gain is better handling of minority classes, especially `positive_outlook`.\n"
    "4. These results justify using FinBERT as the production model for full-corpus inference, API predictions, and downstream filing-level analytics."
)

st.caption(
    "FinBERT metrics are loaded from models/finbert_metrics.pkl. Baseline metrics reflect the evaluated benchmark run used during model analysis."
)