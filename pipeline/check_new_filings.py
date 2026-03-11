

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
STATE_DIR = PROJECT_ROOT / "data" / "state"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

TICKER_METADATA_PATH = REFERENCE_DIR / "ticker_metadata.csv"
STATE_PATH = STATE_DIR / "filing_detection_state.json"
NEW_FILINGS_REPORT_PATH = REPORTS_DIR / "new_filings_detected.csv"

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
USER_AGENT = os.getenv("EDGAR_IDENTITY", "SEC Sentiment Project contact: edgar-access@example.com")
REQUEST_TIMEOUT = 30
REQUEST_SLEEP_SECONDS = 0.2
DEFAULT_FORMS = {"10-K", "10-Q"}

REQUIRED_METADATA_COLUMNS = {"ticker", "company_name", "cik", "sector"}


def load_ticker_metadata() -> pd.DataFrame:
    """Load the watchlist of companies to monitor for new filings."""
    if not TICKER_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing ticker metadata file: {TICKER_METADATA_PATH}"
        )

    metadata = pd.read_csv(TICKER_METADATA_PATH)
    missing = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(
            f"Ticker metadata is missing required columns: {sorted(missing)}"
        )

    metadata = metadata.copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.strip().str.upper()
    metadata["company_name"] = metadata["company_name"].astype(str).str.strip()
    metadata["cik"] = metadata["cik"].astype(str).str.strip().str.zfill(10)
    metadata["sector"] = metadata["sector"].fillna("Unknown").astype(str).str.strip()

    return metadata.drop_duplicates(subset=["ticker"]).reset_index(drop=True)



def load_detection_state() -> dict[str, Any]:
    """Load the persisted filing detection state."""
    if not STATE_PATH.exists():
        return {}

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()
            if not raw_text:
                print("[WARN] Filing detection state file is empty. Initializing fresh state.")
                return {}
            return json.loads(raw_text)
    except json.JSONDecodeError:
        print("[WARN] Filing detection state file is invalid JSON. Initializing fresh state.")
        return {}



def save_detection_state(state: dict[str, Any]) -> None:
    """Persist the filing detection state to disk."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)



def fetch_submissions_json(cik: str) -> dict[str, Any]:
    """Fetch SEC submissions metadata for a single company by CIK."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    time.sleep(REQUEST_SLEEP_SECONDS)
    return response.json()



def build_recent_filings_df(submissions_json: dict[str, Any]) -> pd.DataFrame:
    """Convert the SEC recent filings arrays into a normalized DataFrame."""
    recent = submissions_json.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()

    form_values = recent.get("form", [])
    length = len(form_values)
    if length == 0:
        return pd.DataFrame()

    def safe_get(field: str, index: int) -> Any:
        values = recent.get(field, [])
        return values[index] if index < len(values) else None

    rows = []
    for idx in range(length):
        accession_number = safe_get("accessionNumber", idx)
        primary_document = safe_get("primaryDocument", idx)
        filing_date = safe_get("filingDate", idx)
        form_type = safe_get("form", idx)
        is_xbrl = safe_get("isXBRL", idx)
        is_inline_xbrl = safe_get("isInlineXBRL", idx)

        if accession_number:
            accession_compact = str(accession_number).replace("-", "")
        else:
            accession_compact = None

        cik = str(submissions_json.get("cik", "")).zfill(10)
        cik_no_leading_zeros = str(int(cik)) if cik.isdigit() else cik

        filing_url = None
        if accession_number and primary_document:
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_no_leading_zeros}/{accession_compact}/{primary_document}"
            )

        rows.append(
            {
                "accession_number": accession_number,
                "form_type": form_type,
                "filing_date": filing_date,
                "primary_document": primary_document,
                "is_xbrl": is_xbrl,
                "is_inline_xbrl": is_inline_xbrl,
                "filing_url": filing_url,
            }
        )

    return pd.DataFrame(rows)



def detect_new_filings(
    metadata_df: pd.DataFrame,
    state: dict[str, Any],
    forms_to_track: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect new tracked filings that have not been seen in prior runs."""
    detection_timestamp = datetime.now(timezone.utc).isoformat()
    detected_rows: list[dict[str, Any]] = []
    updated_state = dict(state)

    for row in metadata_df.itertuples(index=False):
        print(f"[INFO] Checking {row.ticker} ({row.cik}) for new filings")
        submissions_json = fetch_submissions_json(row.cik)
        recent_filings_df = build_recent_filings_df(submissions_json)
        if recent_filings_df.empty:
            print(f"[WARN] No recent filings returned for {row.ticker}")
            continue

        tracked_df = recent_filings_df[
            recent_filings_df["form_type"].isin(forms_to_track)
        ].copy()
        if tracked_df.empty:
            print(f"[INFO] No tracked forms found for {row.ticker}")
            continue

        tracked_df = tracked_df.sort_values(
            ["filing_date", "accession_number"],
            ascending=[False, False],
        ).reset_index(drop=True)

        seen_accessions = set(
            updated_state.get(row.ticker, {}).get("seen_accessions", [])
        )

        newly_detected = tracked_df[
            ~tracked_df["accession_number"].isin(seen_accessions)
        ].copy()

        if newly_detected.empty:
            print(f"[INFO] No new tracked filings found for {row.ticker}")
        else:
            print(
                f"[INFO] Detected {len(newly_detected)} new tracked filings for {row.ticker}"
            )
            for filing in newly_detected.itertuples(index=False):
                detected_rows.append(
                    {
                        "detection_timestamp_utc": detection_timestamp,
                        "ticker": row.ticker,
                        "company_name": row.company_name,
                        "cik": row.cik,
                        "sector": row.sector,
                        "accession_number": filing.accession_number,
                        "form_type": filing.form_type,
                        "filing_date": filing.filing_date,
                        "primary_document": filing.primary_document,
                        "filing_url": filing.filing_url,
                    }
                )

        latest_tracked = tracked_df.iloc[0]
        updated_seen_accessions = set(seen_accessions)
        updated_seen_accessions.update(
            tracked_df["accession_number"].dropna().astype(str).tolist()
        )

        updated_state[row.ticker] = {
            "ticker": row.ticker,
            "company_name": row.company_name,
            "cik": row.cik,
            "sector": row.sector,
            "latest_seen_accession": latest_tracked["accession_number"],
            "latest_seen_filing_date": latest_tracked["filing_date"],
            "seen_accessions": sorted(updated_seen_accessions),
            "last_checked_timestamp_utc": detection_timestamp,
        }

    detected_df = pd.DataFrame(detected_rows)
    return detected_df, updated_state



def write_detected_filings_report(detected_df: pd.DataFrame) -> None:
    """Write the new-filings detection report to disk."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if detected_df.empty:
        empty_df = pd.DataFrame(
            columns=[
                "detection_timestamp_utc",
                "ticker",
                "company_name",
                "cik",
                "sector",
                "accession_number",
                "form_type",
                "filing_date",
                "primary_document",
                "filing_url",
            ]
        )
        empty_df.to_csv(NEW_FILINGS_REPORT_PATH, index=False)
    else:
        detected_df.sort_values(
            ["filing_date", "ticker", "accession_number"],
            ascending=[False, True, True],
        ).to_csv(NEW_FILINGS_REPORT_PATH, index=False)



def print_console_summary(detected_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    """Print a concise summary of the filing detection run."""
    print(f"[INFO] Companies checked: {len(metadata_df)}")
    print(f"[INFO] New filings detected: {len(detected_df)}")

    if detected_df.empty:
        print("[INFO] No new tracked 10-K/10-Q filings detected.")
        return

    print("\n[INFO] Newly detected filings:")
    display_cols = [
        "ticker",
        "company_name",
        "form_type",
        "filing_date",
        "accession_number",
        "filing_url",
    ]
    print(detected_df[display_cols].to_string(index=False))



def main() -> None:
    metadata_df = load_ticker_metadata()
    state = load_detection_state()
    detected_df, updated_state = detect_new_filings(
        metadata_df=metadata_df,
        state=state,
        forms_to_track=DEFAULT_FORMS,
    )

    write_detected_filings_report(detected_df)
    save_detection_state(updated_state)
    print_console_summary(detected_df, metadata_df)

    print(f"\n[INFO] Wrote new filing report to {NEW_FILINGS_REPORT_PATH}")
    print(f"[INFO] Updated filing detection state at {STATE_PATH}")


if __name__ == "__main__":
    main()