

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from sec_client import get_all_filings_for_company
from fetch_filing_text import get_filing_text_from_hit
from section_extractor import extract_risk_section
from paragraph_extractor import extract_paragraphs, build_paragraph_records


def deduplicate_filing_hits(filing_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate SEC search hits by accession number."""
    seen_accessions = set()
    unique_hits: List[Dict[str, Any]] = []

    for hit in filing_hits:
        source = hit.get("_source", {})
        accession_number = (
            source.get("adsh")
            or source.get("accessionNo")
            or source.get("accessionNumber")
        )

        if accession_number:
            if accession_number in seen_accessions:
                continue
            seen_accessions.add(accession_number)

        unique_hits.append(hit)

    return unique_hits


def build_corpus_for_company(
    company: str,
    form_types: List[str],
    start_date: str,
    end_date: str,
    filings_per_page: int = 50,
    max_filings: int | None = None,
    dedupe: bool = True,
) -> List[Dict[str, Any]]:
    """Build paragraph-level corpus records for a single company."""
    all_records: List[Dict[str, Any]] = []

    for form_type in form_types:
        try:
            filing_hits = get_all_filings_for_company(
                company_identifier=company,
                form_type=form_type,
                start_date=start_date,
                end_date=end_date,
                page_size=filings_per_page,
            )
        except Exception as exc:
            print(f"[WARN] Failed to search {form_type} filings for {company}: {exc}")
            continue

        raw_hit_count = len(filing_hits)
        filing_hits = deduplicate_filing_hits(filing_hits)

        if max_filings is not None:
            filing_hits = filing_hits[:max_filings]

        print(
            f"[INFO] {company} | {form_type} | found {len(filing_hits)} unique filings "
            f"({raw_hit_count} raw hits)"
        )

        for filing_index, hit in enumerate(filing_hits):
            source = hit.get("_source", {})
            accession_number = (
                source.get("adsh")
                or source.get("accessionNo")
                or source.get("accessionNumber")
            )
            filing_date = source.get("filedAt")
            cik = source.get("ciks")

            try:
                filing_text = get_filing_text_from_hit(hit)
            except Exception as exc:
                print(
                    f"[WARN] Failed to fetch filing text for {company} | {form_type} | "
                    f"index={filing_index} | accession={accession_number}: {exc}"
                )
                continue

            risk_text = extract_risk_section(filing_text, form_type=form_type)
            if not risk_text:
                print(
                    f"[WARN] No risk section found for {company} | {form_type} | "
                    f"index={filing_index}"
                )
                continue

            paragraphs = extract_paragraphs(risk_text)
            records = build_paragraph_records(
                paragraphs,
                company=company,
                form_type=form_type,
                filing_index=filing_index,
                dedupe=dedupe,
            )

            for record in records:
                record["filing_date"] = filing_date
                record["accession_number"] = accession_number
                record["cik"] = cik

            all_records.extend(records)
            print(
                f"[INFO] {company} | {form_type} | index={filing_index} | "
                f"generated {len(records)} paragraph records"
            )

    return all_records



def save_records_to_json(records: List[Dict[str, Any]], output_path: str) -> None:
    """Save corpus records to JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)



def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paragraph-level SEC risk corpus")
    parser.add_argument(
        "companies",
        nargs="+",
        help="Ticker(s), CIK(s), or company identifiers",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        default=["10-K", "10-Q"],
        help="SEC form types to ingest (default: 10-K 10-Q)",
    )
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        default="2024-12-31",
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="Number of filings to request per SEC search page",
    )
    parser.add_argument(
        "--max-filings",
        type=int,
        default=None,
        help="Optional cap on filings per company per form type",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable paragraph-level deduplication",
    )
    parser.add_argument(
        "--output",
        default="data/sec_risk_corpus.json",
        help="Output JSON path",
    )

    args = parser.parse_args()

    all_records: List[Dict[str, Any]] = []

    for company in args.companies:
        company_records = build_corpus_for_company(
            company=company,
            form_types=args.forms,
            start_date=args.start_date,
            end_date=args.end_date,
            filings_per_page=args.page_size,
            max_filings=args.max_filings,
            dedupe=not args.no_dedupe,
        )
        all_records.extend(company_records)
        print(f"[INFO] {company} | total records accumulated: {len(company_records)}")

    save_records_to_json(all_records, args.output)
    print(f"[INFO] Saved {len(all_records)} total records to {args.output}")


if __name__ == "__main__":
    main()