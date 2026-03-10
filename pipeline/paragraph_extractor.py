from typing import Dict, List, Any

import argparse
import hashlib
import re
from fetch_filing_text import get_filing_text
from section_extractor import extract_risk_section


def is_boilerplate(text: str) -> bool:
    lowered = text.lower().strip()
    boilerplate_terms = [
        "table of contents",
        "index",
        "page",
    ]
    return any(term in lowered for term in boilerplate_terms)


def extract_paragraphs(section_text: str, min_length: int = 80) -> List[str]:
    raw_paragraphs = re.split(r"\n\s*\n", section_text)
    cleaned = []

    for p in raw_paragraphs:
        p = " ".join(p.split())
        if len(p) >= min_length and not is_boilerplate(p):
            cleaned.append(p)

    return cleaned


def normalize_text(text: str) -> str:
    """Normalize paragraph text for exact-match deduplication."""
    return " ".join(text.lower().split())


def paragraph_hash(text: str) -> str:
    """Generate a stable hash for a normalized paragraph."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_paragraphs(paragraphs: List[str]) -> List[Dict[str, str]]:
    """Deduplicate paragraphs using normalized text hashes."""
    seen_hashes = set()
    unique_paragraphs: List[Dict[str, str]] = []

    for paragraph in paragraphs:
        current_hash = paragraph_hash(paragraph)
        if current_hash in seen_hashes:
            continue

        seen_hashes.add(current_hash)
        unique_paragraphs.append(
            {
                "text": paragraph,
                "paragraph_hash": current_hash,
                "normalized_text": normalize_text(paragraph),
            }
        )

    return unique_paragraphs


def build_paragraph_records(
    paragraphs: List[str],
    company: str,
    form_type: str,
    filing_index: int,
    dedupe: bool = False,
) -> List[Dict[str, Any]]:
    """Attach metadata to each paragraph so it can be stored as a dataset row."""

    records: List[Dict[str, Any]] = []

    if dedupe:
        unique_paragraphs = deduplicate_paragraphs(paragraphs)
        for p in unique_paragraphs:
            records.append(
                {
                    "company": company,
                    "form": form_type,
                    "filing_index": filing_index,
                    "paragraph": p["text"],
                    "paragraph_hash": p["paragraph_hash"],
                    "normalized_text": p["normalized_text"],
                }
            )
    else:
        for p in paragraphs:
            records.append(
                {
                    "company": company,
                    "form": form_type,
                    "filing_index": filing_index,
                    "paragraph": p,
                    "paragraph_hash": paragraph_hash(p),
                    "normalized_text": normalize_text(p),
                }
            )

    return records


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract paragraphs from SEC risk sections")

    parser.add_argument("company", help="Ticker, CIK, or company identifier")
    parser.add_argument("--form", default="10-K", help="SEC form type (default: 10-K)")
    parser.add_argument("--index", type=int, default=0, help="Filing index (0 = most recent)")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Deduplicate extracted paragraphs using normalized text hashes",
    )

    args = parser.parse_args()

    filing_text = get_filing_text(args.company, form_type=args.form, index=args.index)

    risk_text = extract_risk_section(filing_text, form_type=args.form)

    if not risk_text:
        print("Risk section not found.")
    else:
        paragraphs = extract_paragraphs(risk_text)
        records = build_paragraph_records(
            paragraphs,
            company=args.company,
            form_type=args.form,
            filing_index=args.index,
            dedupe=args.dedupe,
        )

        print(f"Generated {len(records)} paragraph records.")
        print()

        if records:
            print("First record:")
            print()
            print(records[0])