

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

COMPANY_SIGNALS_PATH = REPORTS_DIR / "company_risk_signals.csv"
SECTOR_SIGNALS_PATH = REPORTS_DIR / "sector_risk_signals.csv"
TREND_SIGNALS_PATH = REPORTS_DIR / "risk_trend_signals.csv"

MONITORING_REPORT_PATH = REPORTS_DIR / "risk_monitoring_report.csv"
RISK_ALERTS_PATH = REPORTS_DIR / "risk_alerts.csv"

HIGH_COMPANY_RISK_THRESHOLD = 0.80
RISK_SCORE_CHANGE_THRESHOLD = 0.05
HIGH_SECTOR_RISK_THRESHOLD = 0.75
TOP_MOVER_LIMIT = 10


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

REQUIRED_TREND_COLUMNS = {
    "ticker",
    "company_name",
    "form_type",
    "filing_date",
    "risk_score",
    "previous_risk_score",
    "risk_score_change",
    "pct_negative_risk",
    "pct_negative_risk_change",
}



def load_csv(path: Path, required_columns: set[str], label: str) -> pd.DataFrame:
    """Load a CSV report and validate that required columns are present."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label} file. Generate it first: {path}"
        )

    df = pd.read_csv(path)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {sorted(missing)}"
        )
    return df



def build_company_monitoring(company_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare company-level monitoring output."""
    report = company_df.copy()
    report["monitor_type"] = "company"
    report["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    report["high_company_risk_flag"] = (
        report["company_risk_score"] >= HIGH_COMPANY_RISK_THRESHOLD
    )
    report["company_risk_bucket"] = pd.cut(
        report["company_risk_score"],
        bins=[-0.01, 0.4, 0.7, 1.0],
        labels=["low", "medium", "high"],
    ).astype(str)
    return report



def build_sector_monitoring(sector_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare sector-level monitoring output."""
    report = sector_df.copy()
    report["monitor_type"] = "sector"
    report["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    report["high_sector_risk_flag"] = (
        report["avg_company_risk_score"] >= HIGH_SECTOR_RISK_THRESHOLD
    )
    report["sector_risk_bucket"] = pd.cut(
        report["avg_company_risk_score"],
        bins=[-0.01, 0.4, 0.7, 1.0],
        labels=["low", "medium", "high"],
    ).astype(str)
    return report



def build_top_movers(trend_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the largest positive and negative company risk score changes."""
    movers = trend_df.dropna(subset=["risk_score_change"]).copy()
    if movers.empty:
        return movers

    movers["filing_date"] = movers["filing_date"].fillna("unknown_date").astype(str)

    positive = movers.sort_values("risk_score_change", ascending=False).head(TOP_MOVER_LIMIT)
    negative = movers.sort_values("risk_score_change", ascending=True).head(TOP_MOVER_LIMIT)

    positive = positive.copy()
    positive["mover_direction"] = "increase"

    negative = negative.copy()
    negative["mover_direction"] = "decrease"

    return pd.concat([positive, negative], ignore_index=True)



def build_alerts(
    company_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    trend_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate simple threshold-based alerts for monitoring."""
    alerts: list[dict] = []
    run_timestamp = datetime.now(timezone.utc).isoformat()

    for row in company_df.itertuples(index=False):
        if row.company_risk_score >= HIGH_COMPANY_RISK_THRESHOLD:
            alerts.append(
                {
                    "run_timestamp_utc": run_timestamp,
                    "alert_type": "high_company_risk",
                    "entity_type": "company",
                    "entity_id": row.ticker,
                    "entity_name": row.company_name,
                    "sector": row.sector,
                    "metric_name": "company_risk_score",
                    "metric_value": row.company_risk_score,
                    "threshold": HIGH_COMPANY_RISK_THRESHOLD,
                    "alert_reason": (
                        f"{row.ticker} company_risk_score {row.company_risk_score:.4f} "
                        f"exceeds threshold {HIGH_COMPANY_RISK_THRESHOLD:.2f}"
                    ),
                }
            )

    for row in sector_df.itertuples(index=False):
        if row.avg_company_risk_score >= HIGH_SECTOR_RISK_THRESHOLD:
            alerts.append(
                {
                    "run_timestamp_utc": run_timestamp,
                    "alert_type": "high_sector_risk",
                    "entity_type": "sector",
                    "entity_id": row.sector,
                    "entity_name": row.sector,
                    "sector": row.sector,
                    "metric_name": "avg_company_risk_score",
                    "metric_value": row.avg_company_risk_score,
                    "threshold": HIGH_SECTOR_RISK_THRESHOLD,
                    "alert_reason": (
                        f"Sector {row.sector} avg_company_risk_score {row.avg_company_risk_score:.4f} "
                        f"exceeds threshold {HIGH_SECTOR_RISK_THRESHOLD:.2f}"
                    ),
                }
            )

    trend_changes = trend_df.dropna(subset=["risk_score_change"]).copy()
    for row in trend_changes.itertuples(index=False):
        if row.risk_score_change >= RISK_SCORE_CHANGE_THRESHOLD:
            alerts.append(
                {
                    "run_timestamp_utc": run_timestamp,
                    "alert_type": "rising_company_risk",
                    "entity_type": "company",
                    "entity_id": row.ticker,
                    "entity_name": row.company_name,
                    "sector": None,
                    "metric_name": "risk_score_change",
                    "metric_value": row.risk_score_change,
                    "threshold": RISK_SCORE_CHANGE_THRESHOLD,
                    "alert_reason": (
                        f"{row.ticker} risk_score_change {row.risk_score_change:.4f} "
                        f"on filing date {row.filing_date} exceeds threshold {RISK_SCORE_CHANGE_THRESHOLD:.2f}"
                    ),
                }
            )

    if not alerts:
        return pd.DataFrame(
            columns=[
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
        )

    return pd.DataFrame(alerts).sort_values(
        ["alert_type", "metric_value"],
        ascending=[True, False],
    ).reset_index(drop=True)



def build_monitoring_report(
    company_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    top_movers_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine company, sector, and mover views into one long monitoring report."""
    company_report = build_company_monitoring(company_df)
    sector_report = build_sector_monitoring(sector_df)

    movers_report = top_movers_df.copy()
    if not movers_report.empty:
        movers_report["monitor_type"] = "top_movers"
        movers_report["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    report_frames = [company_report, sector_report]
    if not movers_report.empty:
        report_frames.append(movers_report)

    return pd.concat(report_frames, ignore_index=True, sort=False)



def print_console_summary(
    monitoring_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    company_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    top_movers_df: pd.DataFrame,
) -> None:
    """Print a concise overview of the monitoring outputs."""
    print(f"[INFO] Monitoring report rows: {len(monitoring_df)}")
    print(f"[INFO] Alert rows: {len(alerts_df)}")
    print(f"[INFO] Companies monitored: {len(company_df)}")
    print(f"[INFO] Sectors monitored: {len(sector_df)}")

    if not company_df.empty:
        print("\n[INFO] Highest company risk scores:")
        print(
            company_df[
                [
                    "ticker",
                    "company_name",
                    "sector",
                    "company_risk_score",
                    "avg_pct_negative_risk",
                    "avg_high_conf_risk_ratio",
                ]
            ]
            .sort_values("company_risk_score", ascending=False)
            .head(10)
            .to_string(index=False)
        )

    if not sector_df.empty:
        print("\n[INFO] Sector risk summary:")
        print(sector_df.to_string(index=False))

    if not top_movers_df.empty:
        print("\n[INFO] Largest risk score movers:")
        print(
            top_movers_df[
                [
                    "ticker",
                    "filing_date",
                    "risk_score",
                    "previous_risk_score",
                    "risk_score_change",
                    "mover_direction",
                ]
            ].to_string(index=False)
        )

    if not alerts_df.empty:
        print("\n[INFO] Active alerts:")
        print(alerts_df.to_string(index=False))



def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    company_df = load_csv(COMPANY_SIGNALS_PATH, REQUIRED_COMPANY_COLUMNS, "company signals")
    sector_df = load_csv(SECTOR_SIGNALS_PATH, REQUIRED_SECTOR_COLUMNS, "sector signals")
    trend_df = load_csv(TREND_SIGNALS_PATH, REQUIRED_TREND_COLUMNS, "trend signals")

    top_movers_df = build_top_movers(trend_df)
    alerts_df = build_alerts(company_df, sector_df, trend_df)
    monitoring_df = build_monitoring_report(company_df, sector_df, top_movers_df)

    monitoring_df.to_csv(MONITORING_REPORT_PATH, index=False)
    alerts_df.to_csv(RISK_ALERTS_PATH, index=False)

    print_console_summary(
        monitoring_df,
        alerts_df,
        company_df,
        sector_df,
        top_movers_df,
    )

    print(f"\n[INFO] Wrote monitoring report to {MONITORING_REPORT_PATH}")
    print(f"[INFO] Wrote risk alerts to {RISK_ALERTS_PATH}")


if __name__ == "__main__":
    main()