

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
PARAGRAPHS_PATH = REPORTS_DIR / "paragraph_predictions_full.csv"
FILING_SUMMARY_PATH = REPORTS_DIR / "filing_prediction_summary.csv"
TOP_RISK_PATH = REPORTS_DIR / "top_high_confidence_risk_paragraphs.csv"


st.set_page_config(
    page_title="SEC Sentiment & Risk Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load precomputed report CSVs for the dashboard."""
    missing = [
        path for path in [PARAGRAPHS_PATH, FILING_SUMMARY_PATH, TOP_RISK_PATH]
        if not path.exists()
    ]
    if missing:
        missing_str = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing dashboard input files. Run the batch pipeline and summarization first:\n"
            f"{missing_str}"
        )

    paragraphs_df = pd.read_csv(PARAGRAPHS_PATH)
    filing_summary_df = pd.read_csv(FILING_SUMMARY_PATH)
    top_risk_df = pd.read_csv(TOP_RISK_PATH)

    for df in [paragraphs_df, filing_summary_df, top_risk_df]:
        if "filing_date" in df.columns:
            df["filing_date"] = df["filing_date"].fillna("unknown_date").astype(str)

    if "risk_score" not in filing_summary_df.columns and {
        "high_conf_risk_count",
        "total_paragraphs",
    }.issubset(filing_summary_df.columns):
        filing_summary_df["risk_score"] = (
            filing_summary_df["high_conf_risk_count"]
            / filing_summary_df["total_paragraphs"]
        ).round(4)

    return paragraphs_df, filing_summary_df, top_risk_df


def filter_filing_summary(df: pd.DataFrame, tickers: list[str], forms: list[str]) -> pd.DataFrame:
    filtered = df.copy()
    if tickers:
        filtered = filtered[filtered["ticker"].isin(tickers)]
    if forms:
        filtered = filtered[filtered["form_type"].isin(forms)]
    return filtered


st.title("SEC Sentiment & Risk Dashboard")
st.markdown(
    "Interactive dashboard for exploring paragraph-level SEC risk predictions, "
    "filing-level risk summaries, and top high-confidence disclosures."
)

try:
    paragraphs_df, filing_summary_df, top_risk_df = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()


st.sidebar.header("Filters")
available_tickers = sorted(filing_summary_df["ticker"].dropna().unique().tolist())
available_forms = sorted(filing_summary_df["form_type"].dropna().unique().tolist())

selected_tickers = st.sidebar.multiselect(
    "Ticker",
    options=available_tickers,
    default=available_tickers,
)
selected_forms = st.sidebar.multiselect(
    "Form Type",
    options=available_forms,
    default=available_forms,
)
min_confidence = st.sidebar.slider(
    "Minimum confidence for paragraph tables",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.01,
)
paragraph_search = st.sidebar.text_input("Search paragraph text")

filtered_filing_summary = filter_filing_summary(
    filing_summary_df,
    selected_tickers,
    selected_forms,
)

filtered_paragraphs = paragraphs_df.copy()
if selected_tickers:
    filtered_paragraphs = filtered_paragraphs[filtered_paragraphs["ticker"].isin(selected_tickers)]
if selected_forms:
    filtered_paragraphs = filtered_paragraphs[filtered_paragraphs["form_type"].isin(selected_forms)]
if "confidence" in filtered_paragraphs.columns:
    filtered_paragraphs = filtered_paragraphs[filtered_paragraphs["confidence"] >= min_confidence]
if paragraph_search:
    filtered_paragraphs = filtered_paragraphs[
        filtered_paragraphs["text"].astype(str).str.contains(paragraph_search, case=False, na=False)
    ]

filtered_top_risk = top_risk_df.copy()
if selected_tickers:
    filtered_top_risk = filtered_top_risk[filtered_top_risk["ticker"].isin(selected_tickers)]
if selected_forms:
    filtered_top_risk = filtered_top_risk[filtered_top_risk["form_type"].isin(selected_forms)]
if "confidence" in filtered_top_risk.columns:
    filtered_top_risk = filtered_top_risk[filtered_top_risk["confidence"] >= min_confidence]
if paragraph_search:
    filtered_top_risk = filtered_top_risk[
        filtered_top_risk["text"].astype(str).str.contains(paragraph_search, case=False, na=False)
    ]


st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Filings", len(filtered_filing_summary))
col2.metric("Filtered Paragraphs", len(filtered_paragraphs))
col3.metric(
    "Avg Filing Risk Score",
    f"{filtered_filing_summary['risk_score'].mean():.3f}" if not filtered_filing_summary.empty else "N/A",
)
col4.metric(
    "Avg Paragraph Confidence",
    f"{filtered_paragraphs['confidence'].mean():.3f}" if not filtered_paragraphs.empty else "N/A",
)


st.subheader("Prediction Distribution")
pred_counts = filtered_paragraphs["prediction"].value_counts()
if pred_counts.empty:
    st.info("No paragraph predictions match the current filters.")
else:
    fig, ax = plt.subplots(figsize=(7, 4))
    pred_counts.plot(kind="bar", ax=ax)
    ax.set_title("Predicted Class Distribution")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    st.pyplot(fig)


st.subheader("Filing-Level Risk Summary")
summary_columns = [
    col
    for col in [
        "ticker",
        "company_name",
        "form_type",
        "filing_date",
        "total_paragraphs",
        "negative_risk",
        "neutral",
        "positive_outlook",
        "high_conf_risk_count",
        "pct_negative_risk",
        "risk_score",
        "avg_confidence",
    ]
    if col in filtered_filing_summary.columns
]

if filtered_filing_summary.empty:
    st.info("No filings match the current filters.")
else:
    display_summary = filtered_filing_summary[summary_columns].sort_values(
        ["risk_score", "pct_negative_risk"],
        ascending=[False, False],
    )
    st.dataframe(display_summary, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    rank_df = display_summary.copy()
    labels = [f"{row.ticker} {row.form_type}" for row in rank_df.itertuples()]
    ax.bar(range(len(rank_df)), rank_df["risk_score"])
    ax.set_title("Risk Score by Filing")
    ax.set_ylabel("Risk Score")
    ax.set_xlabel("Filing")
    ax.set_xticks(range(len(rank_df)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    st.pyplot(fig)


st.subheader("Company Comparison")
if filtered_filing_summary.empty:
    st.info("No company comparison available for the current filters.")
else:
    company_summary = (
        filtered_filing_summary.groupby("ticker")
        .agg(
            filings=("filing_id", "count"),
            total_paragraphs=("total_paragraphs", "sum"),
            avg_pct_negative_risk=("pct_negative_risk", "mean"),
            avg_risk_score=("risk_score", "mean"),
            avg_confidence=("avg_confidence", "mean"),
        )
        .round(4)
        .reset_index()
    )
    st.dataframe(company_summary, use_container_width=True)


st.subheader("Top High-Confidence Risk Paragraphs")
if filtered_top_risk.empty:
    st.info("No high-confidence risk paragraphs match the current filters.")
else:
    top_risk_display_cols = [
        col
        for col in [
            "ticker",
            "company_name",
            "form_type",
            "filing_date",
            "confidence",
            "risk_rank_within_filing",
            "text",
        ]
        if col in filtered_top_risk.columns
    ]
    st.dataframe(
        filtered_top_risk[top_risk_display_cols].sort_values(
            ["confidence", "risk_rank_within_filing"],
            ascending=[False, True],
        ),
        use_container_width=True,
    )


st.subheader("Paragraph Explorer")
paragraph_display_cols = [
    col
    for col in [
        "ticker",
        "company_name",
        "form_type",
        "filing_date",
        "prediction",
        "confidence",
        "text",
    ]
    if col in filtered_paragraphs.columns
]

if filtered_paragraphs.empty:
    st.info("No paragraphs match the current filters.")
else:
    st.dataframe(
        filtered_paragraphs[paragraph_display_cols].sort_values(
            ["confidence"],
            ascending=[False],
        ),
        use_container_width=True,
    )


st.caption(
    "Dashboard inputs are generated by training.summarize_predictions after running the full SEC risk pipeline."
)