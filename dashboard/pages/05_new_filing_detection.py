

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
STATE_DIR = PROJECT_ROOT / "data" / "state"

NEW_FILINGS_REPORT_PATH = REPORTS_DIR / "new_filings_detected.csv"
STATE_PATH = STATE_DIR / "filing_detection_state.json"


st.set_page_config(
    page_title="New Filing Detection",
    page_icon="🆕",
    layout="wide",
)


@st.cache_data
def load_new_filings_report() -> pd.DataFrame:
    """Load the latest new-filings detection report."""
    if not NEW_FILINGS_REPORT_PATH.exists():
        raise FileNotFoundError(
            "Missing new filings report. Run pipeline/check_new_filings.py first: "
            f"{NEW_FILINGS_REPORT_PATH}"
        )

    df = pd.read_csv(NEW_FILINGS_REPORT_PATH)
    if df.empty:
        return df

    for col in ["detection_timestamp_utc", "filing_date"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown_date").astype(str)

    return df


@st.cache_data
def load_detection_state() -> pd.DataFrame:
    """Load the persisted filing detection state and flatten it into a DataFrame."""
    if not STATE_PATH.exists():
        raise FileNotFoundError(
            "Missing filing detection state file. Run pipeline/check_new_filings.py first: "
            f"{STATE_PATH}"
        )

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()
        if not raw_text:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "company_name",
                    "cik",
                    "sector",
                    "latest_seen_accession",
                    "latest_seen_filing_date",
                    "last_checked_timestamp_utc",
                    "seen_accessions_count",
                ]
            )
        state: dict[str, Any] = json.loads(raw_text)

    rows = []
    for ticker, payload in state.items():
        seen_accessions = payload.get("seen_accessions", []) or []
        rows.append(
            {
                "ticker": ticker,
                "company_name": payload.get("company_name"),
                "cik": payload.get("cik"),
                "sector": payload.get("sector"),
                "latest_seen_accession": payload.get("latest_seen_accession"),
                "latest_seen_filing_date": payload.get("latest_seen_filing_date"),
                "last_checked_timestamp_utc": payload.get("last_checked_timestamp_utc"),
                "seen_accessions_count": len(seen_accessions),
            }
        )

    state_df = pd.DataFrame(rows)
    if state_df.empty:
        return state_df

    for col in ["latest_seen_filing_date", "last_checked_timestamp_utc"]:
        if col in state_df.columns:
            state_df[col] = state_df[col].fillna("unknown_date").astype(str)

    return state_df.sort_values("ticker").reset_index(drop=True)


st.title("New Filing Detection")
st.markdown(
    "This page shows the output of the **new filing detection prototype**, which monitors a watchlist "
    "for newly observed SEC filings and records the latest filing state for each tracked company."
)

st.markdown(
    "The goal of this feature is to detect unseen `10-K` and `10-Q` filings using the SEC submissions API "
    "before the downstream risk-analysis pipeline is run."
)

try:
    new_filings_df = load_new_filings_report()
    state_df = load_detection_state()
except Exception as exc:
    st.error(str(exc))
    st.stop()


st.sidebar.header("Filters")
available_tickers = (
    sorted(state_df["ticker"].dropna().astype(str).unique().tolist())
    if not state_df.empty
    else []
)
available_sectors = (
    sorted(state_df["sector"].dropna().astype(str).unique().tolist())
    if not state_df.empty
    else []
)
available_forms = (
    sorted(new_filings_df["form_type"].dropna().astype(str).unique().tolist())
    if not new_filings_df.empty and "form_type" in new_filings_df.columns
    else []
)

selected_tickers = st.sidebar.multiselect(
    "Ticker",
    options=available_tickers,
    default=available_tickers,
)
selected_sectors = st.sidebar.multiselect(
    "Sector",
    options=available_sectors,
    default=available_sectors,
)
selected_forms = st.sidebar.multiselect(
    "Form Type",
    options=available_forms,
    default=available_forms,
)

filtered_state_df = state_df.copy()
if selected_tickers:
    filtered_state_df = filtered_state_df[filtered_state_df["ticker"].isin(selected_tickers)]
if selected_sectors and "sector" in filtered_state_df.columns:
    filtered_state_df = filtered_state_df[filtered_state_df["sector"].isin(selected_sectors)]

filtered_new_filings_df = new_filings_df.copy()
if selected_tickers and not filtered_new_filings_df.empty:
    filtered_new_filings_df = filtered_new_filings_df[
        filtered_new_filings_df["ticker"].isin(selected_tickers)
    ]
if selected_sectors and not filtered_new_filings_df.empty and "sector" in filtered_new_filings_df.columns:
    filtered_new_filings_df = filtered_new_filings_df[
        filtered_new_filings_df["sector"].isin(selected_sectors)
    ]
if selected_forms and not filtered_new_filings_df.empty and "form_type" in filtered_new_filings_df.columns:
    filtered_new_filings_df = filtered_new_filings_df[
        filtered_new_filings_df["form_type"].isin(selected_forms)
    ]


st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tracked Companies", len(filtered_state_df))
col2.metric("New Filings Detected", len(filtered_new_filings_df))
col3.metric(
    "Unique Forms Detected",
    filtered_new_filings_df["form_type"].nunique()
    if not filtered_new_filings_df.empty and "form_type" in filtered_new_filings_df.columns
    else 0,
)
col4.metric(
    "Latest Detection Timestamp",
    filtered_new_filings_df["detection_timestamp_utc"].max()
    if not filtered_new_filings_df.empty and "detection_timestamp_utc" in filtered_new_filings_df.columns
    else "N/A",
)

st.markdown(
    "**Interpretation:** A non-empty detection report means the system has identified filings that have not yet "
    "been seen in prior detection runs. These can then be routed into the rest of the risk-analysis pipeline."
)


st.subheader("Newly Detected Filings")
if filtered_new_filings_df.empty:
    st.success("No new tracked filings match the current filters.")
else:
    display_cols = [
        col
        for col in [
            "detection_timestamp_utc",
            "ticker",
            "company_name",
            "sector",
            "form_type",
            "filing_date",
            "accession_number",
            "primary_document",
            "filing_url",
        ]
        if col in filtered_new_filings_df.columns
    ]
    rank_df = filtered_new_filings_df.sort_values(
        ["filing_date", "ticker", "accession_number"],
        ascending=[False, True, True],
    )
    st.dataframe(rank_df[display_cols], use_container_width=True)

st.markdown(
    "This table is the core output of the prototype. It shows which filings were newly identified and provides "
    "direct links to the SEC filing documents."
)


st.subheader("Detection State Summary")
if filtered_state_df.empty:
    st.info("No detection state rows match the current filters.")
else:
    state_display_cols = [
        col
        for col in [
            "ticker",
            "company_name",
            "sector",
            "cik",
            "latest_seen_accession",
            "latest_seen_filing_date",
            "last_checked_timestamp_utc",
            "seen_accessions_count",
        ]
        if col in filtered_state_df.columns
    ]
    st.dataframe(filtered_state_df[state_display_cols], use_container_width=True)

st.markdown(
    "The detection state is what makes the prototype incremental. It stores the set of accession numbers already "
    "seen for each company, so later runs can identify only newly observed filings instead of re-reporting old ones."
)


st.subheader("How this fits into the pipeline")
st.markdown(
    "- **Watchlist metadata** supplies ticker, company name, CIK, and sector.\n"
    "- **SEC submissions JSON** is queried for each CIK.\n"
    "- **Recent filings** are filtered to tracked forms such as `10-K` and `10-Q`.\n"
    "- **Accession numbers** are compared against persisted state.\n"
    "- **Unseen filings** are written to the detection report for downstream processing."
)

st.caption(
    "Detection outputs are generated by pipeline/check_new_filings.py using the SEC submissions API and the persisted filing-detection state."
)