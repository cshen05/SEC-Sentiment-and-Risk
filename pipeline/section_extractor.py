import re
import argparse
from typing import Optional

from fetch_filing_text import get_filing_text


def extract_section(text: str, start_pattern: str, end_pattern: str) -> Optional[str]:
    """Extract text between two regex section markers, choosing the longest plausible match."""
    pattern = re.compile(
        start_pattern + r"(.*?)" + end_pattern,
        re.IGNORECASE | re.DOTALL,
    )
    matches = pattern.findall(text)
    if not matches:
        return None

    cleaned_matches = [match.strip() for match in matches if match and match.strip()]
    if not cleaned_matches:
        return None

    # Prefer the longest plausible section because SEC filings often include
    # table-of-contents references before the real section body.
    return max(cleaned_matches, key=len)


def normalize_heading_pattern(heading: str) -> str:
    """Make heading regexes more tolerant of punctuation and whitespace variants."""
    tokens = heading.strip().split()
    escaped_tokens = [re.escape(token) for token in tokens]
    return r"[\s\W]*".join(escaped_tokens)


def extract_risk_section(text: str, form_type: str = "10-K") -> Optional[str]:
    """Extract risk-related sections from common SEC filings."""

    form_type = form_type.upper()

    risk_heading = normalize_heading_pattern("item 1a risk factors")

    if form_type == "10-K":
        end_pattern = r"(?:item[\s\W]*1b[\s\W]*|item[\s\W]*2[\s\W]*)"
        return extract_section(text, risk_heading, end_pattern)

    if form_type == "10-Q":
        end_pattern = r"(?:item[\s\W]*2[\s\W]*)"
        return extract_section(text, risk_heading, end_pattern)

    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract risk sections from SEC filings")

    parser.add_argument("company", help="Ticker, CIK, or company identifier")
    parser.add_argument("--form", default="10-K", help="SEC form type (default: 10-K)")
    parser.add_argument("--index", type=int, default=0, help="Filing index (0 = most recent)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional debug information about the extracted section",
    )

    args = parser.parse_args()

    filing_text = get_filing_text(args.company, form_type=args.form, index=args.index)

    risk_text = extract_risk_section(filing_text, form_type=args.form)

    if risk_text:
        print("Found risk section.")
        print()
        print(f"Extracted {len(risk_text)} characters.")
        print()
        print(risk_text[:3000])

        if args.debug:
            print()
            print("--- DEBUG ---")
            print(
                "Contains 'item 1a'? ",
                bool(re.search(r"item[\s\W]*1a", filing_text, re.IGNORECASE)),
            )
            print(
                "Contains 'risk factors'? ",
                bool(re.search(r"risk[\s\W]*factors", filing_text, re.IGNORECASE)),
            )
            print(f"Section preview tail (last 1000 chars):\n{risk_text[-1000:]}")
    else:
        print("Risk section not found for this filing type.")
