from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"

PREDICTION_HISTORY_PATH = MONITORING_DIR / "prediction_distribution_history.csv"
CONFIDENCE_HISTORY_PATH = MONITORING_DIR / "confidence_history.csv"
COMPANY_HISTORY_PATH = MONITORING_DIR / "company_risk_history.csv"
SECTOR_HISTORY_PATH = MONITORING_DIR / "sector_risk_history.csv"
DRIFT_REPORT_PATH = MONITORING_DIR / "monitoring_drift_report.csv"
DRIFT_ALERTS_PATH = MONITORING_DIR / "monitoring_drift_alerts.csv"


st.set_page_config(
    page_title="Model Monitoring",
    page_icon="📡",
    layout="wide",
)


@st.cache_data
def load_monitoring_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load monitoring history, drift report, and drift alerts."""
    required_paths = [
        PREDICTION_HISTORY_PATH,
        CONFIDENCE_HISTORY_PATH,
        COMPANY_HISTORY_PATH,
        SECTOR_HISTORY_PATH,
        DRIFT_REPORT_PATH,
        DRIFT_ALERTS_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing monitoring history files. Run training/update_monitoring_history.py first:\n"
            + "\n".join(missing)
        )

    prediction_history = pd.read_csv(PREDICTION_HISTORY_PATH)
    confidence_history = pd.read_csv(CONFIDENCE_HISTORY_PATH)
    company_history = pd.read_csv(COMPANY_HISTORY_PATH)
    sector_history = pd.read_csv(SECTOR_HISTORY_PATH)
    drift_report = pd.read_csv(DRIFT_REPORT_PATH)
    drift_alerts = pd.read_csv(DRIFT_ALERTS_PATH)

    for df in [prediction_history, confidence_history, company_history, sector_history, drift_report, drift_alerts]:
        if "run_timestamp_utc" in df.columns:
            df["run_timestamp_utc"] = pd.to_datetime(df["run_timestamp_utc"], errors="coerce")
        if "current_run_timestamp_utc" in df.columns:
            df["current_run_timestamp_utc"] = pd.to_datetime(df["current_run_timestamp_utc"], errors="coerce")
        if "previous_run_timestamp_utc" in df.columns:
            df["previous_run_timestamp_utc"] = pd.to_datetime(df["previous_run_timestamp_utc"], errors="coerce")

    return (
        prediction_history,
        confidence_history,
        company_history,
        sector_history,
        drift_report,
        drift_alerts,
    )


st.title("Model Monitoring")
st.markdown(
    "This page tracks **run-to-run drift** in model outputs and downstream portfolio signals. "
    "It is designed to answer three questions:"
)

st.markdown(
    "1. Is the **prediction distribution** changing over time?\n"
    "2. Is model **confidence** becoming less stable?\n"
    "3. Are **company or sector risk scores** moving materially between runs?"
)

try:
    (
        prediction_history,
        confidence_history,
        company_history,
        sector_history,
        drift_report,
        drift_alerts,
    ) = load_monitoring_history()
except Exception as exc:
    st.error(str(exc))
    st.stop()


st.sidebar.header("Filters")
drift_types = sorted(drift_report["drift_type"].dropna().astype(str).unique().tolist()) if not drift_report.empty else []
alert_types = sorted(drift_alerts["alert_type"].dropna().astype(str).unique().tolist()) if not drift_alerts.empty else []
tickers = sorted(company_history["ticker"].dropna().astype(str).unique().tolist()) if not company_history.empty else []
sectors = sorted(sector_history["sector"].dropna().astype(str).unique().tolist()) if not sector_history.empty else []

selected_drift_types = st.sidebar.multiselect(
    "Drift Type",
    options=drift_types,
    default=drift_types,
)
selected_alert_types = st.sidebar.multiselect(
    "Alert Type",
    options=alert_types,
    default=alert_types,
)
selected_tickers = st.sidebar.multiselect(
    "Ticker",
    options=tickers,
    default=tickers,
)
selected_sectors = st.sidebar.multiselect(
    "Sector",
    options=sectors,
    default=sectors,
)

filtered_drift_report = drift_report.copy()
if selected_drift_types:
    filtered_drift_report = filtered_drift_report[
        filtered_drift_report["drift_type"].isin(selected_drift_types)
    ]
if selected_tickers and "entity_type" in filtered_drift_report.columns and "entity_id" in filtered_drift_report.columns:
    filtered_drift_report = filtered_drift_report[
        (~filtered_drift_report["entity_type"].eq("company"))
        | (filtered_drift_report["entity_id"].isin(selected_tickers))
    ]
if selected_sectors and "entity_type" in filtered_drift_report.columns and "entity_id" in filtered_drift_report.columns:
    filtered_drift_report = filtered_drift_report[
        (~filtered_drift_report["entity_type"].eq("sector"))
        | (filtered_drift_report["entity_id"].isin(selected_sectors))
    ]

filtered_drift_alerts = drift_alerts.copy()
if selected_alert_types:
    filtered_drift_alerts = filtered_drift_alerts[
        filtered_drift_alerts["alert_type"].isin(selected_alert_types)
    ]
if selected_tickers and "entity_type" in filtered_drift_alerts.columns and "entity_id" in filtered_drift_alerts.columns:
    filtered_drift_alerts = filtered_drift_alerts[
        (~filtered_drift_alerts["entity_type"].eq("company"))
        | (filtered_drift_alerts["entity_id"].isin(selected_tickers))
    ]
if selected_sectors and "entity_type" in filtered_drift_alerts.columns and "entity_id" in filtered_drift_alerts.columns:
    filtered_drift_alerts = filtered_drift_alerts[
        (~filtered_drift_alerts["entity_type"].eq("sector"))
        | (filtered_drift_alerts["entity_id"].isin(selected_sectors))
    ]

filtered_company_history = company_history.copy()
if selected_tickers:
    filtered_company_history = filtered_company_history[
        filtered_company_history["ticker"].isin(selected_tickers)
    ]
if selected_sectors and "sector" in filtered_company_history.columns:
    filtered_company_history = filtered_company_history[
        filtered_company_history["sector"].isin(selected_sectors)
    ]

filtered_sector_history = sector_history.copy()
if selected_sectors:
    filtered_sector_history = filtered_sector_history[
        filtered_sector_history["sector"].isin(selected_sectors)
    ]


st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prediction Runs", prediction_history["run_timestamp_utc"].nunique() if not prediction_history.empty else 0)
col2.metric("Confidence Runs", confidence_history["run_timestamp_utc"].nunique() if not confidence_history.empty else 0)
col3.metric("Drift Rows", len(filtered_drift_report))
col4.metric("Drift Alerts", len(filtered_drift_alerts))

st.markdown(
    "**Interpretation:** Monitoring becomes meaningful once there are at least **two runs** in history. "
    "The first run initializes the baseline snapshot, and subsequent runs produce drift deltas and alert conditions."
)


st.subheader("Prediction Distribution History")
if prediction_history.empty:
    st.info("Prediction distribution history is empty.")
else:
    pred_display_cols = [
        "run_timestamp_utc",
        "total_paragraphs",
        "pct_negative_risk",
        "pct_neutral",
        "pct_positive_outlook",
    ]
    st.dataframe(prediction_history[pred_display_cols].sort_values("run_timestamp_utc"), use_container_width=True)

    history_plot_df = prediction_history.sort_values("run_timestamp_utc")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history_plot_df["run_timestamp_utc"], history_plot_df["pct_negative_risk"], marker="o", label="pct_negative_risk")
    ax.plot(history_plot_df["run_timestamp_utc"], history_plot_df["pct_neutral"], marker="o", label="pct_neutral")
    ax.plot(history_plot_df["run_timestamp_utc"], history_plot_df["pct_positive_outlook"], marker="o", label="pct_positive_outlook")
    ax.set_title("Prediction Distribution Over Runs")
    ax.set_xlabel("Run Timestamp")
    ax.set_ylabel("Share of Predictions")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

st.markdown(
    "This chart tracks whether the model is producing materially different class mixtures across runs. "
    "A large swing in `pct_negative_risk` or `pct_positive_outlook` can indicate a change in the incoming filing mix or a shift in the model's behavior on new data."
)


st.subheader("Confidence History")
if confidence_history.empty:
    st.info("Confidence history is empty.")
else:
    conf_display_cols = [
        "run_timestamp_utc",
        "avg_confidence_overall",
        "avg_conf_negative_risk",
        "avg_conf_neutral",
        "avg_conf_positive_outlook",
    ]
    st.dataframe(confidence_history[conf_display_cols].sort_values("run_timestamp_utc"), use_container_width=True)

    conf_plot_df = confidence_history.sort_values("run_timestamp_utc")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(conf_plot_df["run_timestamp_utc"], conf_plot_df["avg_confidence_overall"], marker="o", label="overall")
    ax.plot(conf_plot_df["run_timestamp_utc"], conf_plot_df["avg_conf_negative_risk"], marker="o", label="negative_risk")
    ax.plot(conf_plot_df["run_timestamp_utc"], conf_plot_df["avg_conf_neutral"], marker="o", label="neutral")
    ax.plot(conf_plot_df["run_timestamp_utc"], conf_plot_df["avg_conf_positive_outlook"], marker="o", label="positive_outlook")
    ax.set_title("Confidence Over Runs")
    ax.set_xlabel("Run Timestamp")
    ax.set_ylabel("Average Confidence")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

st.markdown(
    "Confidence history helps identify whether the model is becoming less certain over time. "
    "A sustained drop in confidence can be an early sign of data drift or reduced fit between the model and newly ingested filings."
)


st.subheader("Company Risk Score History")
if filtered_company_history.empty:
    st.info("No company history rows match the current filters.")
else:
    company_display_cols = [
        "run_timestamp_utc",
        "ticker",
        "company_name",
        "sector",
        "company_risk_score",
        "avg_pct_negative_risk",
        "avg_high_conf_risk_ratio",
        "avg_confidence",
    ]
    st.dataframe(filtered_company_history[company_display_cols].sort_values(["ticker", "run_timestamp_utc"]), use_container_width=True)

    st.markdown("**Per-company risk score trajectories**")
    for ticker, group in filtered_company_history.groupby("ticker"):
        plot_group = group.sort_values("run_timestamp_utc")
        if plot_group["run_timestamp_utc"].notna().sum() < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(plot_group["run_timestamp_utc"], plot_group["company_risk_score"], marker="o")
        ax.set_title(f"{ticker} Company Risk Score Over Runs")
        ax.set_xlabel("Run Timestamp")
        ax.set_ylabel("Company Risk Score")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

st.markdown(
    "Company risk history shows whether the portfolio-level signal for a given ticker is stable or shifting between runs. "
    "This is especially useful when the filing universe is refreshed frequently."
)


st.subheader("Sector Risk Score History")
if filtered_sector_history.empty:
    st.info("No sector history rows match the current filters.")
else:
    sector_display_cols = [
        "run_timestamp_utc",
        "sector",
        "avg_company_risk_score",
        "avg_pct_negative_risk",
        "avg_high_conf_risk_ratio",
        "avg_confidence",
    ]
    st.dataframe(filtered_sector_history[sector_display_cols].sort_values(["sector", "run_timestamp_utc"]), use_container_width=True)

    st.markdown("**Per-sector risk score trajectories**")
    for sector, group in filtered_sector_history.groupby("sector"):
        plot_group = group.sort_values("run_timestamp_utc")
        if plot_group["run_timestamp_utc"].notna().sum() < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(plot_group["run_timestamp_utc"], plot_group["avg_company_risk_score"], marker="o")
        ax.set_title(f"{sector} Sector Risk Score Over Runs")
        ax.set_xlabel("Run Timestamp")
        ax.set_ylabel("Avg Company Risk Score")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

st.markdown(
    "Sector history helps identify whether disclosure-risk conditions are changing in specific parts of the market rather than at the individual company level alone."
)


st.subheader("Latest Drift Report")
if filtered_drift_report.empty:
    st.info("No drift report rows match the current filters. Drift requires at least two monitoring runs.")
else:
    drift_display_cols = [
        "drift_type",
        "entity_type",
        "entity_id",
        "metric_name",
        "previous_value",
        "current_value",
        "delta",
        "previous_run_timestamp_utc",
        "current_run_timestamp_utc",
    ]
    st.dataframe(filtered_drift_report[drift_display_cols].sort_values(["drift_type", "entity_type", "entity_id", "metric_name"]), use_container_width=True)

st.markdown(
    "The drift report compares the **latest run** against the **previous run** and computes deltas for the monitored metrics."
)


st.subheader("Drift Alerts")
if filtered_drift_alerts.empty:
    st.success("No drift alerts match the current filters.")
else:
    alert_display_cols = [
        "alert_type",
        "drift_type",
        "entity_type",
        "entity_id",
        "metric_name",
        "previous_value",
        "current_value",
        "delta",
        "threshold",
        "previous_run_timestamp_utc",
        "current_run_timestamp_utc",
        "alert_reason",
    ]
    st.dataframe(filtered_drift_alerts[alert_display_cols].sort_values(["alert_type", "metric_name", "entity_id"]), use_container_width=True)

    alert_counts = filtered_drift_alerts["alert_type"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    alert_counts.plot(kind="bar", ax=ax)
    ax.set_title("Drift Alerts by Type")
    ax.set_xlabel("Alert Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)

st.markdown(
    "These alerts are designed to surface meaningful changes in prediction mix, confidence, or portfolio-level risk signals. "
    "They are intentionally simple threshold rules that can be extended later into more advanced monitoring logic."
)


st.subheader("Model Monitoring Takeaways")
st.markdown(
    "- **Prediction distribution drift** highlights when the class mix changes materially across runs.\n"
    "- **Confidence drift** highlights when the model becomes less certain on new data.\n"
    "- **Company and sector risk score drift** highlights whether downstream portfolio signals are moving materially between runs.\n"
    "- Together, these components turn the project into a lightweight monitoring system rather than a one-time analysis pipeline."
)

st.caption(
    "Monitoring history and drift outputs are generated by training/update_monitoring_history.py from the latest paragraph predictions and portfolio risk signals."
)
