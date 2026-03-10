import json
import os
from typing import Any

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


def get_connection() -> PGConnection:
    """Create a PostgreSQL connection using environment variables or defaults."""
    return psycopg2.connect(**DB_CONFIG)


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input corpus file not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("Corpus JSON must be a non-empty list of paragraph records.")

    print(f"[INFO] Loaded {len(data)} paragraph records from {INPUT_FILE}")

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

            ticker = get_value(record, "ticker", default=None)
            company_name = get_value(record, "company_name", "company", default=None)
            form_type = get_value(record, "form_type", "form", default=None)
            filing_date = get_value(record, "filing_date", "date", default=None)
            accession_number = get_value(record, "accession_number", "accession", default=None)
            section = get_value(record, "section", default=None)
            text = normalize_text(get_value(record, "text", default=None))

            if not ticker or not form_type or not accession_number or not text:
                skipped_records += 1
                print(
                    "[WARN] Skipping record "
                    f"{idx}: missing one of required fields "
                    "(ticker, form_type/form, accession_number/accession, text)"
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
