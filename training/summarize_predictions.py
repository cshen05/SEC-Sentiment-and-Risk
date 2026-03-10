

import sys
from pathlib import Path

import pandas as pd

# Allow running as: python3 -m training.summarize_predictions
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.db import get_connection

OUTPUT_DIR = PROJECT_ROOT / "data" / "reports"
FULL_OUTPUT_PATH = OUTPUT_DIR / "paragraph_predictions_full.csv"
FILING_SUMMARY_PATH = OUTPUT_DIR / "filing_prediction_summary.csv"
TOP_RISK_PATH = OUTPUT_DIR / "top_high_confidence_risk_paragraphs.csv"
HIGH_CONFIDENCE_THRESHOLD = 0.90
TOP_RISK_PER_FILING = 5


def fetch_prediction_dataframe() -> pd.DataFrame:
    """Load paragraph-level predictions joined with filing metadata."""
    conn = get_connection()
    query = """
        SELECT
            p.id AS paragraph_id,
            p.filing_id,
            p.section,
            p.text,
            p.weak_label,
            p.manual_label,
            p.prediction,
            p.confidence,
            f.ticker,
            f.company_name,
            f.form_type,
            f.filing_date,
            f.accession_number
        FROM paragraphs p
        JOIN filings f
            ON p.filing_id = f.id
        WHERE p.prediction IS NOT NULL
        ORDER BY f.ticker, f.filing_date, p.id
    """

    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    finally:
        cur.close()
        conn.close()

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        raise ValueError("No predicted paragraphs found in the database. Run full-corpus prediction first.")

    return df


def build_filing_summary(df: pd.DataFrame) -> pd.DataFrame:
    # pivot_table drops rows when index columns contain NaN (e.g., filing_date).
    # Replace NaN filing_date values so all rows participate in aggregation.
    df = df.copy()
    if df["filing_date"].isna().any():
        df["filing_date"] = df["filing_date"].fillna("unknown_date")
    """Aggregate prediction counts and confidence metrics at the filing level."""
    base = (
        df.groupby([
            "filing_id",
            "ticker",
            "company_name",
            "form_type",
            "filing_date",
            "accession_number",
        ], dropna=False)
        .agg(
            total_paragraphs=("paragraph_id", "count"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )

    pred_counts = df.pivot_table(
        index=[
            "filing_id",
            "ticker",
            "company_name",
            "form_type",
            "filing_date",
            "accession_number",
        ],
        columns="prediction",
        values="paragraph_id",
        aggfunc="count",
        fill_value=0,
    )
    pred_counts.columns.name = None
    pred_counts = pred_counts.reset_index()

    summary = base.merge(
        pred_counts,
        on=[
            "filing_id",
            "ticker",
            "company_name",
            "form_type",
            "filing_date",
            "accession_number",
        ],
        how="left",
    )

    for col in ["negative_risk", "neutral", "positive_outlook"]:
        if col not in summary.columns:
            summary[col] = 0

    high_conf_risk = df[
        (df["prediction"] == "negative_risk")
        & (df["confidence"] >= HIGH_CONFIDENCE_THRESHOLD)
    ]
    high_conf_summary = (
        high_conf_risk.groupby("filing_id")
        .size()
        .rename("high_conf_risk_count")
        .reset_index()
    )

    summary = summary.merge(high_conf_summary, on="filing_id", how="left")
    summary["high_conf_risk_count"] = (
        summary["high_conf_risk_count"].fillna(0).astype(int)
    )

    summary["pct_negative_risk"] = (summary["negative_risk"] / summary["total_paragraphs"]).round(4)
    summary["pct_neutral"] = (summary["neutral"] / summary["total_paragraphs"]).round(4)
    summary["pct_positive_outlook"] = (summary["positive_outlook"] / summary["total_paragraphs"]).round(4)
    summary["avg_confidence"] = summary["avg_confidence"].round(4)

    summary = summary.sort_values(["ticker", "filing_date", "form_type"]).reset_index(drop=True)
    return summary


def build_top_risk_paragraphs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest-confidence negative-risk paragraphs for each filing."""
    risk_df = df[df["prediction"] == "negative_risk"].copy()
    if risk_df.empty:
        return risk_df

    risk_df = risk_df.sort_values(
        ["filing_id", "confidence", "paragraph_id"],
        ascending=[True, False, True],
    )
    risk_df["risk_rank_within_filing"] = risk_df.groupby("filing_id").cumcount() + 1
    risk_df = risk_df[risk_df["risk_rank_within_filing"] <= TOP_RISK_PER_FILING].copy()
    return risk_df.reset_index(drop=True)


def print_console_summary(df: pd.DataFrame, filing_summary: pd.DataFrame, top_risk_df: pd.DataFrame) -> None:
    """Print a concise summary of prediction outputs."""
    print(f"[INFO] Total predicted paragraphs: {len(df)}")
    print(f"[INFO] Prediction counts: {df['prediction'].value_counts().to_dict()}")
    print(
        "[INFO] Average confidence by class: "
        f"{df.groupby('prediction')['confidence'].mean().round(4).to_dict()}"
    )
    print(f"[INFO] Total filings summarized: {len(filing_summary)}")

    high_conf = df[df["confidence"] >= HIGH_CONFIDENCE_THRESHOLD]
    print(
        f"[INFO] High-confidence paragraphs (confidence >= {HIGH_CONFIDENCE_THRESHOLD:.2f}): "
        f"{len(high_conf)}"
    )

    print("\n[INFO] Filing-level negative risk summary:")
    cols = [
        "ticker",
        "form_type",
        "filing_date",
        "total_paragraphs",
        "negative_risk",
        "neutral",
        "positive_outlook",
        "high_conf_risk_count",
        "pct_negative_risk",
        "avg_confidence",
    ]
    print(filing_summary[cols].to_string(index=False, max_colwidth=40))

    if not top_risk_df.empty:
        preview_cols = [
            "ticker",
            "form_type",
            "filing_date",
            "confidence",
            "risk_rank_within_filing",
            "text",
        ]
        print("\n[INFO] Top high-confidence risk paragraphs per filing:")
        print(top_risk_df[preview_cols].head(15).to_string(index=False, max_colwidth=100))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = fetch_prediction_dataframe()
    filing_summary = build_filing_summary(df)
    top_risk_df = build_top_risk_paragraphs(df)

    df.to_csv(FULL_OUTPUT_PATH, index=False)
    filing_summary.to_csv(FILING_SUMMARY_PATH, index=False)
    top_risk_df.to_csv(TOP_RISK_PATH, index=False)

    print_console_summary(df, filing_summary, top_risk_df)

    print(f"\n[INFO] Wrote full paragraph predictions to {FULL_OUTPUT_PATH}")
    print(f"[INFO] Wrote filing summary to {FILING_SUMMARY_PATH}")
    print(f"[INFO] Wrote top risk paragraphs to {TOP_RISK_PATH}")


if __name__ == "__main__":
    main()