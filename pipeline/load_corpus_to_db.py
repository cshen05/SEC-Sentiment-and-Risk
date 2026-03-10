import json
import os
import sys
from pathlib import Path
from typing import Any

# Allow this script to be run directly from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.db import get_connection

INPUT_FILE = os.environ.get("SEC_CORPUS_PATH", "data/sec_risk_corpus.json")


def get_value(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first non-null value from a list of possible keys."""
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default



def normalize_text(text: Any) -> str | None:
    """Convert text to a clean string for database insertion."""
    if text is None:
        return None

    text = str(text).strip()
    return text or None


# --- Added helpers for more flexible corpus schemas ---
def get_nested_value(record: dict[str, Any], *paths: tuple[str, ...], default: Any = None) -> Any:
    """Return the first non-null value found across possible nested key paths."""
    for path in paths:
        current: Any = record
        found = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                found = False
                break
            current = current[key]
        if found and current is not None:
            return current
    return default


def extract_record_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Map different corpus JSON schemas into the DB loader's expected fields."""
    raw_company = get_value(record, "company", "company_name", "company_name_normalized")
    ticker = get_value(record, "ticker", "symbol", "stock", "company_ticker")
    form_type = get_value(record, "form_type", "form", "filing_type")
    filing_date = get_value(record, "filing_date", "date", "filed_at", "filed_on")
    accession_number = get_value(record, "accession_number", "accession", "accession_no")
    section = get_value(record, "section", "section_name", "item")
    text = normalize_text(get_value(record, "normalized_text", "text", "paragraph", "paragraph_text", "content"))
    company_name = str(raw_company).strip() if raw_company is not None else None
    cik = get_value(record, "cik")

    if isinstance(cik, list):
        cik = cik[0] if cik else None

    # Support nested metadata/file info shapes produced by some corpus builders.
    ticker = ticker or get_nested_value(
        record,
        ("metadata", "ticker"),
        ("filing", "ticker"),
        ("filing_meta", "ticker"),
        ("company", "ticker"),
        default=None,
    )
    form_type = form_type or get_nested_value(
        record,
        ("metadata", "form_type"),
        ("metadata", "form"),
        ("filing", "form_type"),
        ("filing", "form"),
        ("filing_meta", "form_type"),
        default=None,
    )
    filing_date = filing_date or get_nested_value(
        record,
        ("metadata", "filing_date"),
        ("metadata", "date"),
        ("filing", "filing_date"),
        ("filing", "date"),
        ("filing_meta", "filing_date"),
        default=None,
    )
    accession_number = accession_number or get_nested_value(
        record,
        ("metadata", "accession_number"),
        ("metadata", "accession"),
        ("filing", "accession_number"),
        ("filing", "accession"),
        ("filing_meta", "accession_number"),
        default=None,
    )
    section = section or get_nested_value(
        record,
        ("metadata", "section"),
        ("filing", "section"),
        default=None,
    )
    text = text or normalize_text(
        get_nested_value(
            record,
            ("metadata", "text"),
            ("paragraph_data", "text"),
            ("paragraph_data", "paragraph_text"),
            default=None,
        )
    )
    company_name = company_name or get_nested_value(
        record,
        ("metadata", "company_name"),
        ("filing", "company_name"),
        ("company", "name"),
        default=None,
    )
    cik = cik or get_nested_value(
        record,
        ("metadata", "cik"),
        ("filing", "cik"),
        ("filing_meta", "cik"),
        default=None,
    )

    if isinstance(cik, list):
        cik = cik[0] if cik else None

    # This corpus stores the ticker in `company` (e.g. "AAPL").
    if ticker is None and company_name:
        ticker = company_name

    if ticker is None and cik is not None:
        ticker = str(cik).strip()

    return {
        "ticker": ticker,
        "company_name": company_name,
        "form_type": form_type,
        "filing_date": filing_date,
        "accession_number": accession_number,
        "section": section,
        "text": text,
    }




def main() -> None:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input corpus file not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("Corpus JSON must be a non-empty list of paragraph records.")

    print(f"[INFO] Loaded {len(data)} paragraph records from {INPUT_FILE}")

    sample_keys = list(data[0].keys()) if data and isinstance(data[0], dict) else []
    print(f"[DEBUG] First record keys: {sample_keys}")

    conn = get_connection()
    cur = conn.cursor()

    accession_to_filing_id: dict[str, int] = {}
    inserted_filings = 0
    inserted_paragraphs = 0
    skipped_records = 0

    try:
        for idx, record in enumerate(data):
            if not isinstance(record, dict):
                skipped_records += 1
                print(f"[WARN] Skipping record {idx}: expected object, got {type(record).__name__}")
                continue

            fields = extract_record_fields(record)
            ticker = fields["ticker"]
            company_name = fields["company_name"]
            form_type = fields["form_type"]
            filing_date = fields["filing_date"]
            accession_number = fields["accession_number"]
            section = fields["section"]
            text = fields["text"]

            if not ticker or not form_type or not accession_number or not text:
                skipped_records += 1
                print(
                    f"[WARN] Skipping record {idx}: "
                    f"ticker={ticker!r}, form_type={form_type!r}, accession_number={accession_number!r}, "
                    f"text_present={bool(text)}"
                )
                continue

            if accession_number not in accession_to_filing_id:
                cur.execute(
                    """
                    INSERT INTO filings (ticker, company_name, form_type, filing_date, accession_number)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (accession_number) DO NOTHING
                    RETURNING id
                    """,
                    (ticker, company_name, form_type, filing_date, accession_number),
                )

                result = cur.fetchone()
                if result is not None:
                    filing_id = result[0]
                    inserted_filings += 1
                else:
                    cur.execute(
                        "SELECT id FROM filings WHERE accession_number = %s",
                        (accession_number,),
                    )
                    existing = cur.fetchone()
                    if existing is None:
                        raise RuntimeError(
                            f"Failed to retrieve filing id for accession_number={accession_number}"
                        )
                    filing_id = existing[0]

                accession_to_filing_id[accession_number] = filing_id

            filing_id = accession_to_filing_id[accession_number]

            cur.execute(
                """
                INSERT INTO paragraphs (filing_id, section, text)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (filing_id, section, text),
            )

            if cur.rowcount > 0:
                inserted_paragraphs += 1

        conn.commit()

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

    print(f"[INFO] Inserted filings: {inserted_filings}")
    print(f"[INFO] Inserted paragraphs: {inserted_paragraphs}")
    print(f"[INFO] Skipped records: {skipped_records}")
    print("[INFO] Corpus successfully loaded into database")


if __name__ == "__main__":
    main()
