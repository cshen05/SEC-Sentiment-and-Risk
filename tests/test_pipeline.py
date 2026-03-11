

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.summarize_predictions import (
    HIGH_CONFIDENCE_THRESHOLD,
    TOP_RISK_PER_FILING,
    build_filing_summary,
    build_top_risk_paragraphs,
)


def make_sample_predictions_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paragraph_id": 1,
                "filing_id": 100,
                "section": None,
                "text": "Risk paragraph one.",
                "weak_label": "negative_risk",
                "manual_label": None,
                "prediction": "negative_risk",
                "confidence": 0.95,
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "form_type": "10-K",
                "filing_date": None,
                "accession_number": "0001",
            },
            {
                "paragraph_id": 2,
                "filing_id": 100,
                "section": None,
                "text": "Neutral paragraph.",
                "weak_label": "neutral",
                "manual_label": None,
                "prediction": "neutral",
                "confidence": 0.91,
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "form_type": "10-K",
                "filing_date": None,
                "accession_number": "0001",
            },
            {
                "paragraph_id": 3,
                "filing_id": 100,
                "section": None,
                "text": "Risk paragraph two.",
                "weak_label": "negative_risk",
                "manual_label": None,
                "prediction": "negative_risk",
                "confidence": 0.89,
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "form_type": "10-K",
                "filing_date": None,
                "accession_number": "0001",
            },
            {
                "paragraph_id": 4,
                "filing_id": 200,
                "section": None,
                "text": "Positive paragraph.",
                "weak_label": "positive_outlook",
                "manual_label": None,
                "prediction": "positive_outlook",
                "confidence": 0.72,
                "ticker": "MSFT",
                "company_name": "Microsoft Corp.",
                "form_type": "10-Q",
                "filing_date": None,
                "accession_number": "0002",
            },
            {
                "paragraph_id": 5,
                "filing_id": 200,
                "section": None,
                "text": "High confidence risk paragraph.",
                "weak_label": "negative_risk",
                "manual_label": None,
                "prediction": "negative_risk",
                "confidence": 0.97,
                "ticker": "MSFT",
                "company_name": "Microsoft Corp.",
                "form_type": "10-Q",
                "filing_date": None,
                "accession_number": "0002",
            },
            {
                "paragraph_id": 6,
                "filing_id": 200,
                "section": None,
                "text": "Another risk paragraph.",
                "weak_label": "negative_risk",
                "manual_label": None,
                "prediction": "negative_risk",
                "confidence": 0.93,
                "ticker": "MSFT",
                "company_name": "Microsoft Corp.",
                "form_type": "10-Q",
                "filing_date": None,
                "accession_number": "0002",
            },
        ]
    )



def test_build_filing_summary_aggregates_prediction_counts() -> None:
    df = make_sample_predictions_df()
    summary = build_filing_summary(df)

    assert len(summary) == 2

    aapl_row = summary[summary["filing_id"] == 100].iloc[0]
    assert aapl_row["total_paragraphs"] == 3
    assert aapl_row["negative_risk"] == 2
    assert aapl_row["neutral"] == 1
    assert aapl_row["positive_outlook"] == 0
    assert aapl_row["high_conf_risk_count"] == 1
    assert aapl_row["pct_negative_risk"] == round(2 / 3, 4)

    msft_row = summary[summary["filing_id"] == 200].iloc[0]
    assert msft_row["total_paragraphs"] == 3
    assert msft_row["negative_risk"] == 2
    assert msft_row["neutral"] == 0
    assert msft_row["positive_outlook"] == 1
    assert msft_row["high_conf_risk_count"] == 2
    assert msft_row["pct_negative_risk"] == round(2 / 3, 4)



def test_build_filing_summary_handles_missing_filing_dates() -> None:
    df = make_sample_predictions_df()
    assert df["filing_date"].isna().all()

    summary = build_filing_summary(df)

    assert "filing_date" in summary.columns
    assert set(summary["filing_date"].unique()) == {"unknown_date"}



def test_build_top_risk_paragraphs_filters_and_ranks_risk_rows() -> None:
    df = make_sample_predictions_df()
    top_risk = build_top_risk_paragraphs(df)

    assert not top_risk.empty
    assert set(top_risk["prediction"].unique()) == {"negative_risk"}
    assert (top_risk["risk_rank_within_filing"] >= 1).all()

    msft_rows = top_risk[top_risk["filing_id"] == 200].sort_values("risk_rank_within_filing")
    assert msft_rows.iloc[0]["confidence"] >= msft_rows.iloc[1]["confidence"]
    assert len(msft_rows) == 2



def test_build_top_risk_paragraphs_respects_top_k_limit() -> None:
    base = make_sample_predictions_df()

    extra_rows = []
    for idx in range(10):
        extra_rows.append(
            {
                "paragraph_id": 1000 + idx,
                "filing_id": 300,
                "section": None,
                "text": f"Extra risk paragraph {idx}",
                "weak_label": "negative_risk",
                "manual_label": None,
                "prediction": "negative_risk",
                "confidence": 0.99 - idx * 0.01,
                "ticker": "NVDA",
                "company_name": "NVIDIA Corp.",
                "form_type": "10-K",
                "filing_date": None,
                "accession_number": "0003",
            }
        )

    df = pd.concat([base, pd.DataFrame(extra_rows)], ignore_index=True)
    top_risk = build_top_risk_paragraphs(df)

    nvda_rows = top_risk[top_risk["filing_id"] == 300]
    assert len(nvda_rows) == TOP_RISK_PER_FILING
    assert nvda_rows["risk_rank_within_filing"].max() == TOP_RISK_PER_FILING



def test_high_confidence_threshold_used_in_summary() -> None:
    df = make_sample_predictions_df()
    summary = build_filing_summary(df)

    expected_high_conf = df[
        (df["prediction"] == "negative_risk")
        & (df["confidence"] >= HIGH_CONFIDENCE_THRESHOLD)
    ]

    expected_counts = expected_high_conf.groupby("filing_id").size().to_dict()
    actual_counts = dict(zip(summary["filing_id"], summary["high_conf_risk_count"]))

    for filing_id, expected_count in expected_counts.items():
        assert actual_counts[filing_id] == expected_count