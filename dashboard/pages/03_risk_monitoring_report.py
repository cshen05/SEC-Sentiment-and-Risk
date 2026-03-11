

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

MONITORING_REPORT_PATH = REPORTS_DIR / "risk_monitoring_report.csv"
RISK_ALERTS_PATH = REPORTS_DIR / "risk_alerts.csv"


st.set_page_config(
    page_title="Risk Monitoring Report",
    page_icon="🚨",
    layout="wide",
)


@st.cache_data
def load_monitoring_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load monitoring outputs generated from portfolio risk signals."""
    required_paths = [MONITORING_REPORT_PATH, RISK_ALERTS_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing monitoring files. Run training/build_risk_monitoring_report.py first:\n"
            + "\n".join(missing)
        )

    monitoring_df = pd.read_csv(MONITORING_REPORT_PATH)
    alerts_df = pd.read_csv(RISK_ALERTS_PATH)

    if "run_timestamp_utc" in monitoring_df.columns:
        monitoring_df["run_timestamp_utc"] = pd.to_datetime(
            monitoring_df["run_timestamp_utc"], errors="coerce"
        )
    if "run_timestamp_utc" in alerts_df.columns:
        alerts_df["run_timestamp_utc"] = pd.to_datetime(
            alerts_df["run_timestamp_utc"], errors="coerce"
        )

    if "filing_date" in monitoring_df.columns:
        monitoring_df["filing_date"] = monitoring_df["filing_date"].fillna("unknown_date").astype(str)

    return monitoring_df, alerts_df


st.title("Risk Monitoring Report")
st.markdown(
    "This page surfaces **active risk alerts**, **highest-risk entities**, and **largest risk-score movers** "
    "from the monitoring layer built on top of the SEC portfolio risk signals."
)

st.markdown(
    "The purpose of this monitoring view is to move beyond static model outputs and support an ongoing workflow for "
    "tracking disclosure risk, identifying emerging stress, and highlighting names or sectors that warrant closer review."
)

try:
    monitoring_df, alerts_df = load_monitoring_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()


company_monitor_df = monitoring_df[monitoring_df.get("monitor_type", "") == "company"].copy()
sector_monitor_df = monitoring_df[monitoring_df.get("monitor_type", "") == "sector"].copy()
top_movers_df = monitoring_df[monitoring_df.get("monitor_type", "") == "top_movers"].copy()


st.sidebar.header("Filters")
alert_types = sorted(alerts_df["alert_type"].dropna().astype(str).unique().tolist()) if not alerts_df.empty else []
entity_types = sorted(alerts_df["entity_type"].dropna().astype(str).unique().tolist()) if not alerts_df.empty else []
company_sectors = sorted(company_monitor_df["sector"].dropna().astype(str).unique().tolist()) if not company_monitor_df.empty else []
company_tickers = sorted(company_monitor_df["ticker"].dropna().astype(str).unique().tolist()) if not company_monitor_df.empty else []

selected_alert_types = st.sidebar.multiselect(
    "Alert Type",
    options=alert_types,
    default=alert_types,
)
selected_entity_types = st.sidebar.multiselect(
    "Entity Type",
    options=entity_types,
    default=entity_types,
)
selected_sectors = st.sidebar.multiselect(
    "Sector",
    options=company_sectors,
    default=company_sectors,
)
selected_tickers = st.sidebar.multiselect(
    "Ticker",
    options=company_tickers,
    default=company_tickers,
)

filtered_alerts_df = alerts_df.copy()
if selected_alert_types:
    filtered_alerts_df = filtered_alerts_df[
        filtered_alerts_df["alert_type"].isin(selected_alert_types)
    ]
if selected_entity_types:
    filtered_alerts_df = filtered_alerts_df[
        filtered_alerts_df["entity_type"].isin(selected_entity_types)
    ]
if selected_sectors and "sector" in filtered_alerts_df.columns:
    filtered_alerts_df = filtered_alerts_df[
        filtered_alerts_df["sector"].fillna("Unknown").isin(selected_sectors)
    ]
if selected_tickers and "entity_id" in filtered_alerts_df.columns:
    filtered_alerts_df = filtered_alerts_df[
        (~filtered_alerts_df["entity_type"].eq("company"))
        | (filtered_alerts_df["entity_id"].isin(selected_tickers))
    ]

filtered_company_monitor_df = company_monitor_df.copy()
if selected_sectors:
    filtered_company_monitor_df = filtered_company_monitor_df[
        filtered_company_monitor_df["sector"].isin(selected_sectors)
    ]
if selected_tickers:
    filtered_company_monitor_df = filtered_company_monitor_df[
        filtered_company_monitor_df["ticker"].isin(selected_tickers)
    ]

filtered_sector_monitor_df = sector_monitor_df.copy()
if selected_sectors:
    filtered_sector_monitor_df = filtered_sector_monitor_df[
        filtered_sector_monitor_df["sector"].isin(selected_sectors)
    ]

filtered_movers_df = top_movers_df.copy()
if selected_tickers and "ticker" in filtered_movers_df.columns:
    filtered_movers_df = filtered_movers_df[
        filtered_movers_df["ticker"].isin(selected_tickers)
    ]
if selected_sectors and not filtered_company_monitor_df.empty:
    valid_tickers = set(filtered_company_monitor_df["ticker"].tolist())
    filtered_movers_df = filtered_movers_df[
        filtered_movers_df["ticker"].isin(valid_tickers)
    ]


st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Monitoring Rows", len(monitoring_df))
col2.metric("Active Alerts", len(filtered_alerts_df))
col3.metric("Companies Monitored", len(filtered_company_monitor_df))
col4.metric("Sectors Monitored", len(filtered_sector_monitor_df))

st.markdown(
    "**Interpretation:** The monitoring system highlights three kinds of conditions: "
    "(1) companies with elevated risk scores, (2) sectors with elevated average risk, and "
    "(3) firms whose risk score has risen materially between filings."
)


st.subheader("Active Alerts")
if filtered_alerts_df.empty:
    st.success("No active alerts match the current filters.")
else:
    alert_display_cols = [
        col
        for col in [
            "run_timestamp_utc",
            "alert_type",
            "entity_type",
            "entity_id",
            "entity_name",
            "sector",
            "metric_name",
            "metric_value",
            "threshold",
            "alert_reason",
        ]
        if col in filtered_alerts_df.columns
    ]
    st.dataframe(
        filtered_alerts_df[alert_display_cols].sort_values(
            ["alert_type", "metric_value"],
            ascending=[True, False],
        ),
        use_container_width=True,
    )

    alert_counts = filtered_alerts_df["alert_type"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    alert_counts.plot(kind="bar", ax=ax)
    ax.set_title("Alert Counts by Type")
    ax.set_xlabel("Alert Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)

st.markdown(
    "Alerts are simple threshold-based rules designed to surface risk concentrations and notable changes quickly. "
    "In a fuller production system, these could be pushed to email, Slack, or a downstream monitoring service."
)


st.subheader("Highest Company Risk Scores")
if filtered_company_monitor_df.empty:
    st.info("No company monitoring rows match the current filters.")
else:
    company_display_cols = [
        col
        for col in [
            "ticker",
            "company_name",
            "sector",
            "company_risk_score",
            "avg_pct_negative_risk",
            "avg_high_conf_risk_ratio",
            "high_company_risk_flag",
            "company_risk_bucket",
            "filing_count",
            "avg_confidence",
        ]
        if col in filtered_company_monitor_df.columns
    ]
    company_rank_df = filtered_company_monitor_df.sort_values(
        ["company_risk_score", "avg_pct_negative_risk"],
        ascending=[False, False],
    )
    st.dataframe(company_rank_df[company_display_cols], use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(company_rank_df["ticker"], company_rank_df["company_risk_score"])
    ax.set_title("Highest Company Risk Scores")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Company Risk Score")
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.markdown(
    "This table shows which names are currently the most risk-intensive based on filing language. "
    "These are the companies most likely to trigger additional qualitative review in a monitoring workflow."
)


st.subheader("Sector Monitoring")
if filtered_sector_monitor_df.empty:
    st.info("No sector monitoring rows match the current filters.")
else:
    sector_display_cols = [
        col
        for col in [
            "sector",
            "company_count",
            "filing_count",
            "avg_company_risk_score",
            "avg_pct_negative_risk",
            "avg_high_conf_risk_ratio",
            "high_sector_risk_flag",
            "sector_risk_bucket",
            "avg_confidence",
        ]
        if col in filtered_sector_monitor_df.columns
    ]
    sector_rank_df = filtered_sector_monitor_df.sort_values(
        ["avg_company_risk_score", "avg_pct_negative_risk"],
        ascending=[False, False],
    )
    st.dataframe(sector_rank_df[sector_display_cols], use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(sector_rank_df["sector"], sector_rank_df["avg_company_risk_score"])
    ax.set_title("Average Company Risk Score by Sector")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Avg Company Risk Score")
    plt.xticks(rotation=25, ha="right")
    st.pyplot(fig)

st.markdown(
    "Sector monitoring is useful for identifying whether disclosure stress is broad-based or concentrated in a specific part of the market."
)


st.subheader("Top Movers")
if filtered_movers_df.empty:
    st.info("No top-mover rows match the current filters.")
else:
    mover_display_cols = [
        col
        for col in [
            "ticker",
            "company_name",
            "form_type",
            "filing_date",
            "risk_score",
            "previous_risk_score",
            "risk_score_change",
            "mover_direction",
        ]
        if col in filtered_movers_df.columns
    ]
    st.dataframe(
        filtered_movers_df[mover_display_cols].sort_values(
            ["risk_score_change"],
            ascending=[False],
        ),
        use_container_width=True,
    )

    mover_plot_df = filtered_movers_df.sort_values("risk_score_change", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [f"{row.ticker} {row.filing_date}" for row in mover_plot_df.itertuples()]
    ax.bar(range(len(mover_plot_df)), mover_plot_df["risk_score_change"])
    ax.set_title("Risk Score Change by Filing")
    ax.set_xlabel("Filing")
    ax.set_ylabel("Risk Score Change")
    ax.set_xticks(range(len(mover_plot_df)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    st.pyplot(fig)

st.markdown(
    "Top movers identify which filings show the largest increase or decrease in risk intensity relative to the previous filing. "
    "This is often the most actionable monitoring signal because it highlights **change**, not just level."
)


st.subheader("Monitoring Takeaways")
st.markdown(
    "- **High company risk alerts** identify names whose disclosure language is unusually risk-heavy.\n"
    "- **High sector risk alerts** identify concentrations of disclosure stress at the sector level.\n"
    "- **Rising company risk alerts** identify companies whose filings have become materially more adverse over time.\n"
    "- Together, these signals turn the project from a static classifier into a lightweight disclosure-monitoring system."
)

st.caption(
    "Monitoring outputs are generated by training/build_risk_monitoring_report.py using company, sector, and trend signals derived from the SEC filing risk pipeline."
)