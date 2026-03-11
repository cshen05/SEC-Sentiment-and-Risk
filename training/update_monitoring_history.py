

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"

PARAGRAPHS_PATH = REPORTS_DIR / "paragraph_predictions_full.csv"
COMPANY_SIGNALS_PATH = REPORTS_DIR / "company_risk_signals.csv"
SECTOR_SIGNALS_PATH = REPORTS_DIR / "sector_risk_signals.csv"

PREDICTION_HISTORY_PATH = MONITORING_DIR / "prediction_distribution_history.csv"
CONFIDENCE_HISTORY_PATH = MONITORING_DIR / "confidence_history.csv"
COMPANY_HISTORY_PATH = MONITORING_DIR / "company_risk_history.csv"
SECTOR_HISTORY_PATH = MONITORING_DIR / "sector_risk_history.csv"

DRIFT_REPORT_PATH = MONITORING_DIR / "monitoring_drift_report.csv"
DRIFT_ALERTS_PATH = MONITORING_DIR / "monitoring_drift_alerts.csv"

PREDICTION_DRIFT_THRESHOLD = 0.05
POSITIVE_OUTLOOK_DRIFT_THRESHOLD = 0.03
CONFIDENCE_DROP_THRESHOLD = 0.05
RISK_SCORE_DRIFT_THRESHOLD = 0.05

REQUIRED_PARAGRAPH_COLUMNS = {
    "ticker",
    "prediction",
    "confidence",
}

REQUIRED_COMPANY_COLUMNS = {
    "ticker",
    "company_name",
    "sector",
    "company_risk_score",
    "avg_pct_negative_risk",
    "avg_high_conf_risk_ratio",
    "filing_count",
    "avg_confidence",
}

REQUIRED_SECTOR_COLUMNS = {
    "sector",
    "company_count",
    "filing_count",
    "avg_company_risk_score",
    "avg_pct_negative_risk",
    "avg_high_conf_risk_ratio",
    "avg_confidence",
}


def load_csv(path: Path, required_columns: set[str], label: str) -> pd.DataFrame:
    """Load a report CSV and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} file: {path}")

    df = pd.read_csv(path)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
    return df



def build_prediction_snapshot(paragraphs_df: pd.DataFrame, run_timestamp: str) -> pd.DataFrame:
    """Create a one-row snapshot of prediction distribution for the current run."""
    total = len(paragraphs_df)
    counts = paragraphs_df["prediction"].value_counts()

    snapshot = pd.DataFrame(
        [
            {
                "run_timestamp_utc": run_timestamp,
                "total_paragraphs": total,
                "negative_risk_count": int(counts.get("negative_risk", 0)),
                "neutral_count": int(counts.get("neutral", 0)),
                "positive_outlook_count": int(counts.get("positive_outlook", 0)),
                "pct_negative_risk": round(counts.get("negative_risk", 0) / total, 4) if total else 0.0,
                "pct_neutral": round(counts.get("neutral", 0) / total, 4) if total else 0.0,
                "pct_positive_outlook": round(counts.get("positive_outlook", 0) / total, 4) if total else 0.0,
            }
        ]
    )
    return snapshot



def build_confidence_snapshot(paragraphs_df: pd.DataFrame, run_timestamp: str) -> pd.DataFrame:
    """Create a one-row snapshot of confidence summary statistics for the current run."""
    class_means = paragraphs_df.groupby("prediction")["confidence"].mean().to_dict()

    snapshot = pd.DataFrame(
        [
            {
                "run_timestamp_utc": run_timestamp,
                "avg_confidence_overall": round(float(paragraphs_df["confidence"].mean()), 4),
                "avg_conf_negative_risk": round(float(class_means.get("negative_risk", float("nan"))), 4),
                "avg_conf_neutral": round(float(class_means.get("neutral", float("nan"))), 4),
                "avg_conf_positive_outlook": round(float(class_means.get("positive_outlook", float("nan"))), 4),
            }
        ]
    )
    return snapshot



def build_company_history_snapshot(company_df: pd.DataFrame, run_timestamp: str) -> pd.DataFrame:
    """Create a company-level snapshot for the current run."""
    snapshot = company_df.copy()
    snapshot["run_timestamp_utc"] = run_timestamp
    ordered_cols = [
        "run_timestamp_utc",
        "ticker",
        "company_name",
        "sector",
        "company_risk_score",
        "avg_pct_negative_risk",
        "avg_high_conf_risk_ratio",
        "filing_count",
        "avg_confidence",
    ]
    return snapshot[ordered_cols]



def build_sector_history_snapshot(sector_df: pd.DataFrame, run_timestamp: str) -> pd.DataFrame:
    """Create a sector-level snapshot for the current run."""
    snapshot = sector_df.copy()
    snapshot["run_timestamp_utc"] = run_timestamp
    ordered_cols = [
        "run_timestamp_utc",
        "sector",
        "company_count",
        "filing_count",
        "avg_company_risk_score",
        "avg_pct_negative_risk",
        "avg_high_conf_risk_ratio",
        "avg_confidence",
    ]
    return snapshot[ordered_cols]



def append_history(history_path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append new snapshot rows to a history CSV and return the updated history."""
    if history_path.exists():
        existing = pd.read_csv(history_path)
        updated = pd.concat([existing, new_rows], ignore_index=True)
    else:
        updated = new_rows.copy()

    updated.to_csv(history_path, index=False)
    return updated



def compute_prediction_drift(pred_history: pd.DataFrame) -> pd.DataFrame:
    """Compute latest-vs-previous drift for prediction distribution snapshots."""
    if len(pred_history) < 2:
        return pd.DataFrame()

    ordered = pred_history.sort_values("run_timestamp_utc").reset_index(drop=True)
    previous = ordered.iloc[-2]
    current = ordered.iloc[-1]

    metrics = [
        "pct_negative_risk",
        "pct_neutral",
        "pct_positive_outlook",
        "total_paragraphs",
    ]

    rows = []
    for metric in metrics:
        rows.append(
            {
                "drift_type": "prediction_distribution",
                "entity_type": "global",
                "entity_id": "all_predictions",
                "metric_name": metric,
                "previous_value": previous[metric],
                "current_value": current[metric],
                "delta": round(float(current[metric] - previous[metric]), 4),
                "current_run_timestamp_utc": current["run_timestamp_utc"],
                "previous_run_timestamp_utc": previous["run_timestamp_utc"],
            }
        )

    return pd.DataFrame(rows)



def compute_confidence_drift(conf_history: pd.DataFrame) -> pd.DataFrame:
    """Compute latest-vs-previous drift for confidence snapshots."""
    if len(conf_history) < 2:
        return pd.DataFrame()

    ordered = conf_history.sort_values("run_timestamp_utc").reset_index(drop=True)
    previous = ordered.iloc[-2]
    current = ordered.iloc[-1]

    metrics = [
        "avg_confidence_overall",
        "avg_conf_negative_risk",
        "avg_conf_neutral",
        "avg_conf_positive_outlook",
    ]

    rows = []
    for metric in metrics:
        prev_val = previous[metric]
        curr_val = current[metric]
        rows.append(
            {
                "drift_type": "confidence",
                "entity_type": "global",
                "entity_id": "all_predictions",
                "metric_name": metric,
                "previous_value": prev_val,
                "current_value": curr_val,
                "delta": round(float(curr_val - prev_val), 4) if pd.notna(prev_val) and pd.notna(curr_val) else None,
                "current_run_timestamp_utc": current["run_timestamp_utc"],
                "previous_run_timestamp_utc": previous["run_timestamp_utc"],
            }
        )

    return pd.DataFrame(rows)



def compute_entity_drift(
    history_df: pd.DataFrame,
    entity_col: str,
    metric_cols: list[str],
    drift_type: str,
) -> pd.DataFrame:
    """Compute latest-vs-previous drift for company or sector history."""
    if history_df.empty or history_df["run_timestamp_utc"].nunique() < 2:
        return pd.DataFrame()

    ordered = history_df.sort_values("run_timestamp_utc")
    run_times = ordered["run_timestamp_utc"].dropna().drop_duplicates().tolist()
    if len(run_times) < 2:
        return pd.DataFrame()

    previous_run = run_times[-2]
    current_run = run_times[-1]

    previous_df = ordered[ordered["run_timestamp_utc"] == previous_run].copy()
    current_df = ordered[ordered["run_timestamp_utc"] == current_run].copy()

    merged = current_df.merge(
        previous_df,
        on=entity_col,
        how="left",
        suffixes=("_current", "_previous"),
    )

    rows = []
    entity_type = "company" if entity_col == "ticker" else "sector"
    for row in merged.itertuples(index=False):
        for metric in metric_cols:
            current_value = getattr(row, f"{metric}_current", None)
            previous_value = getattr(row, f"{metric}_previous", None)
            delta = (
                round(float(current_value - previous_value), 4)
                if pd.notna(current_value) and pd.notna(previous_value)
                else None
            )
            rows.append(
                {
                    "drift_type": drift_type,
                    "entity_type": entity_type,
                    "entity_id": getattr(row, entity_col),
                    "metric_name": metric,
                    "previous_value": previous_value,
                    "current_value": current_value,
                    "delta": delta,
                    "current_run_timestamp_utc": current_run,
                    "previous_run_timestamp_utc": previous_run,
                }
            )

    return pd.DataFrame(rows)



def build_drift_alerts(drift_df: pd.DataFrame) -> pd.DataFrame:
    """Generate alerts when drift exceeds configured thresholds."""
    if drift_df.empty:
        return pd.DataFrame(
            columns=[
                "alert_type",
                "drift_type",
                "entity_type",
                "entity_id",
                "metric_name",
                "previous_value",
                "current_value",
                "delta",
                "threshold",
                "current_run_timestamp_utc",
                "previous_run_timestamp_utc",
                "alert_reason",
            ]
        )

    alerts = []
    for row in drift_df.itertuples(index=False):
        if row.metric_name == "pct_negative_risk" and row.delta is not None and abs(row.delta) >= PREDICTION_DRIFT_THRESHOLD:
            alerts.append(
                {
                    "alert_type": "prediction_distribution_drift",
                    "drift_type": row.drift_type,
                    "entity_type": row.entity_type,
                    "entity_id": row.entity_id,
                    "metric_name": row.metric_name,
                    "previous_value": row.previous_value,
                    "current_value": row.current_value,
                    "delta": row.delta,
                    "threshold": PREDICTION_DRIFT_THRESHOLD,
                    "current_run_timestamp_utc": row.current_run_timestamp_utc,
                    "previous_run_timestamp_utc": row.previous_run_timestamp_utc,
                    "alert_reason": (
                        f"{row.metric_name} changed by {row.delta:.4f}, exceeding threshold {PREDICTION_DRIFT_THRESHOLD:.2f}"
                    ),
                }
            )
        elif row.metric_name == "pct_positive_outlook" and row.delta is not None and abs(row.delta) >= POSITIVE_OUTLOOK_DRIFT_THRESHOLD:
            alerts.append(
                {
                    "alert_type": "positive_outlook_drift",
                    "drift_type": row.drift_type,
                    "entity_type": row.entity_type,
                    "entity_id": row.entity_id,
                    "metric_name": row.metric_name,
                    "previous_value": row.previous_value,
                    "current_value": row.current_value,
                    "delta": row.delta,
                    "threshold": POSITIVE_OUTLOOK_DRIFT_THRESHOLD,
                    "current_run_timestamp_utc": row.current_run_timestamp_utc,
                    "previous_run_timestamp_utc": row.previous_run_timestamp_utc,
                    "alert_reason": (
                        f"{row.metric_name} changed by {row.delta:.4f}, exceeding threshold {POSITIVE_OUTLOOK_DRIFT_THRESHOLD:.2f}"
                    ),
                }
            )
        elif row.metric_name.startswith("avg_conf") and row.delta is not None and row.delta <= -CONFIDENCE_DROP_THRESHOLD:
            alerts.append(
                {
                    "alert_type": "confidence_drop",
                    "drift_type": row.drift_type,
                    "entity_type": row.entity_type,
                    "entity_id": row.entity_id,
                    "metric_name": row.metric_name,
                    "previous_value": row.previous_value,
                    "current_value": row.current_value,
                    "delta": row.delta,
                    "threshold": CONFIDENCE_DROP_THRESHOLD,
                    "current_run_timestamp_utc": row.current_run_timestamp_utc,
                    "previous_run_timestamp_utc": row.previous_run_timestamp_utc,
                    "alert_reason": (
                        f"{row.metric_name} fell by {abs(row.delta):.4f}, exceeding threshold {CONFIDENCE_DROP_THRESHOLD:.2f}"
                    ),
                }
            )
        elif row.metric_name in {"company_risk_score", "avg_company_risk_score"} and row.delta is not None and abs(row.delta) >= RISK_SCORE_DRIFT_THRESHOLD:
            alerts.append(
                {
                    "alert_type": "risk_score_drift",
                    "drift_type": row.drift_type,
                    "entity_type": row.entity_type,
                    "entity_id": row.entity_id,
                    "metric_name": row.metric_name,
                    "previous_value": row.previous_value,
                    "current_value": row.current_value,
                    "delta": row.delta,
                    "threshold": RISK_SCORE_DRIFT_THRESHOLD,
                    "current_run_timestamp_utc": row.current_run_timestamp_utc,
                    "previous_run_timestamp_utc": row.previous_run_timestamp_utc,
                    "alert_reason": (
                        f"{row.metric_name} changed by {row.delta:.4f}, exceeding threshold {RISK_SCORE_DRIFT_THRESHOLD:.2f}"
                    ),
                }
            )

    if not alerts:
        return pd.DataFrame(
            columns=[
                "alert_type",
                "drift_type",
                "entity_type",
                "entity_id",
                "metric_name",
                "previous_value",
                "current_value",
                "delta",
                "threshold",
                "current_run_timestamp_utc",
                "previous_run_timestamp_utc",
                "alert_reason",
            ]
        )

    return pd.DataFrame(alerts).sort_values(
        ["alert_type", "metric_name", "entity_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)



def print_console_summary(
    prediction_history: pd.DataFrame,
    confidence_history: pd.DataFrame,
    company_history: pd.DataFrame,
    sector_history: pd.DataFrame,
    drift_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
) -> None:
    """Print a concise summary of monitoring history and drift results."""
    print(f"[INFO] Prediction history rows: {len(prediction_history)}")
    print(f"[INFO] Confidence history rows: {len(confidence_history)}")
    print(f"[INFO] Company history rows: {len(company_history)}")
    print(f"[INFO] Sector history rows: {len(sector_history)}")
    print(f"[INFO] Drift report rows: {len(drift_df)}")
    print(f"[INFO] Drift alert rows: {len(alerts_df)}")

    if not drift_df.empty:
        print("\n[INFO] Latest drift metrics:")
        print(drift_df.head(20).to_string(index=False))

    if not alerts_df.empty:
        print("\n[INFO] Active drift alerts:")
        print(alerts_df.to_string(index=False))



def main() -> None:
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).isoformat()

    paragraphs_df = load_csv(PARAGRAPHS_PATH, REQUIRED_PARAGRAPH_COLUMNS, "paragraph predictions")
    company_df = load_csv(COMPANY_SIGNALS_PATH, REQUIRED_COMPANY_COLUMNS, "company signals")
    sector_df = load_csv(SECTOR_SIGNALS_PATH, REQUIRED_SECTOR_COLUMNS, "sector signals")

    pred_snapshot = build_prediction_snapshot(paragraphs_df, run_timestamp)
    conf_snapshot = build_confidence_snapshot(paragraphs_df, run_timestamp)
    company_snapshot = build_company_history_snapshot(company_df, run_timestamp)
    sector_snapshot = build_sector_history_snapshot(sector_df, run_timestamp)

    prediction_history = append_history(PREDICTION_HISTORY_PATH, pred_snapshot)
    confidence_history = append_history(CONFIDENCE_HISTORY_PATH, conf_snapshot)
    company_history = append_history(COMPANY_HISTORY_PATH, company_snapshot)
    sector_history = append_history(SECTOR_HISTORY_PATH, sector_snapshot)

    drift_frames = [
        compute_prediction_drift(prediction_history),
        compute_confidence_drift(confidence_history),
        compute_entity_drift(
            company_history,
            entity_col="ticker",
            metric_cols=["company_risk_score", "avg_pct_negative_risk", "avg_high_conf_risk_ratio", "avg_confidence"],
            drift_type="company_risk",
        ),
        compute_entity_drift(
            sector_history,
            entity_col="sector",
            metric_cols=["avg_company_risk_score", "avg_pct_negative_risk", "avg_high_conf_risk_ratio", "avg_confidence"],
            drift_type="sector_risk",
        ),
    ]
    drift_df = pd.concat([df for df in drift_frames if not df.empty], ignore_index=True, sort=False) if any(not df.empty for df in drift_frames) else pd.DataFrame()
    alerts_df = build_drift_alerts(drift_df)

    drift_df.to_csv(DRIFT_REPORT_PATH, index=False)
    alerts_df.to_csv(DRIFT_ALERTS_PATH, index=False)

    print_console_summary(
        prediction_history,
        confidence_history,
        company_history,
        sector_history,
        drift_df,
        alerts_df,
    )

    print(f"\n[INFO] Wrote prediction history to {PREDICTION_HISTORY_PATH}")
    print(f"[INFO] Wrote confidence history to {CONFIDENCE_HISTORY_PATH}")
    print(f"[INFO] Wrote company risk history to {COMPANY_HISTORY_PATH}")
    print(f"[INFO] Wrote sector risk history to {SECTOR_HISTORY_PATH}")
    print(f"[INFO] Wrote drift report to {DRIFT_REPORT_PATH}")
    print(f"[INFO] Wrote drift alerts to {DRIFT_ALERTS_PATH}")


if __name__ == "__main__":
    main()