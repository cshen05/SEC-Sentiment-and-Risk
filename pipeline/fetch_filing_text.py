import os
import re
import html
import requests
from typing import Any, Dict
from dotenv import load_dotenv
from edgar import Company, set_identity

load_dotenv()

edgar_identity = os.getenv("EDGAR_IDENTITY")
if not edgar_identity:
    raise ValueError("EDGAR_IDENTITY is not set")

set_identity(edgar_identity)
HEADERS = {"User-Agent": edgar_identity}


def _extract_primary_document(submission_text: str, expected_form: str | None = None) -> str:
    """Extract the main filing document from an SEC submission .txt file.

    SEC submission text files often contain multiple <DOCUMENT> blocks. For
    10-K / 10-Q ingestion we want the primary block whose <TYPE> matches the
    filing form when possible.
    """
    document_pattern = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.IGNORECASE | re.DOTALL)
    type_pattern = re.compile(r"<TYPE>\s*([^\n\r<]+)", re.IGNORECASE)
    text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.IGNORECASE | re.DOTALL)

    documents = document_pattern.findall(submission_text)
    if not documents:
        return submission_text

    normalized_expected = expected_form.upper() if expected_form else None

    best_doc_text = None
    fallback_doc_text = None

    for document in documents:
        type_match = type_pattern.search(document)
        text_match = text_pattern.search(document)
        if not text_match:
            continue

        document_type = type_match.group(1).strip().upper() if type_match else ""
        document_text = text_match.group(1)

        if fallback_doc_text is None:
            fallback_doc_text = document_text

        if normalized_expected and document_type == normalized_expected:
            best_doc_text = document_text
            break

    return best_doc_text or fallback_doc_text or submission_text



def _clean_filing_text(raw_text: str) -> str:
    """Normalize SEC filing text while preserving paragraph boundaries."""
    cleaned = html.unescape(raw_text)

    # Preserve structure from common block-level HTML tags before stripping tags.
    cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?i)</p\s*>", "\n\n", cleaned)
    cleaned = re.sub(r"(?i)</div\s*>", "\n\n", cleaned)
    cleaned = re.sub(r"(?i)</tr\s*>", "\n", cleaned)
    cleaned = re.sub(r"(?i)</li\s*>", "\n", cleaned)
    cleaned = re.sub(r"(?i)</h[1-6]\s*>", "\n\n", cleaned)

    # Opening block tags can also mark a natural boundary.
    cleaned = re.sub(r"(?i)<p\b[^>]*>", "\n\n", cleaned)
    cleaned = re.sub(r"(?i)<div\b[^>]*>", "\n\n", cleaned)
    cleaned = re.sub(r"(?i)<li\b[^>]*>", "\n", cleaned)
    cleaned = re.sub(r"(?i)<tr\b[^>]*>", "\n", cleaned)
    cleaned = re.sub(r"(?i)<h[1-6]\b[^>]*>", "\n\n", cleaned)

    # Strip remaining markup after block boundaries have been preserved.
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    cleaned = cleaned.replace("\xa0", " ")
    cleaned = re.sub(r"\r", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def get_filing_text(company_identifier: str, form_type: str = "10-K", index: int = 0) -> str:
    """Fetch text for a filing for a given company and form type.

    Parameters
    ----------
    company_identifier : str
        Company identifier (ticker, CIK, or company string supported by edgartools)
    form_type : str
        SEC form type (e.g., 10-K, 10-Q, 8-K, S-1)
    index : int
        Which filing to retrieve (0 = most recent)
    """
    company = Company(company_identifier)
    filings = company.get_filings(form=form_type)

    if not filings:
        raise ValueError(f"No filings found for {company_identifier} with form {form_type}")

    if index >= len(filings):
        raise IndexError(
            f"Requested filing index {index}, but only {len(filings)} filings were found."
        )

    filing = filings[index]
    return filing.text()


def get_filing_text_from_hit(hit: Dict[str, Any]) -> str:
    """Fetch the exact filing text using metadata from an SEC search hit.

    This avoids relying on list-order alignment between SEC search results
    and edgartools filing lists.
    """

    source = hit.get("_source", {})
    expected_form = source.get("form")

    # SEC search hits are not fully consistent across response shapes, so try a
    # few common path / URL fields before failing.
    raw_location = (
        source.get("file_name")
        or source.get("fileName")
        or source.get("linkToTxt")
        or source.get("linkToHtml")
        or source.get("file_path")
        or source.get("filePath")
    )

    if not raw_location:
        accession_number = (
            source.get("adsh")
            or source.get("accessionNo")
            or source.get("accessionNumber")
        )
        ciks = source.get("ciks")

        primary_cik = None
        if isinstance(ciks, list) and ciks:
            primary_cik = str(ciks[0])
        elif ciks is not None:
            primary_cik = str(ciks)

        if accession_number and primary_cik:
            cik_digits = "".join(ch for ch in primary_cik if ch.isdigit()).lstrip("0") or "0"
            accession_compact = str(accession_number).replace("-", "")
            raw_location = f"edgar/data/{cik_digits}/{accession_compact}/{accession_number}.txt"
        else:
            available_keys = ", ".join(sorted(source.keys()))
            raise ValueError(
                "SEC search hit does not include a usable filing path and could not be "
                "derived from accession metadata. "
                f"Available _source keys: {available_keys}"
            )

    if isinstance(raw_location, str) and raw_location.startswith("http"):
        filing_url = raw_location
    else:
        normalized_location = str(raw_location).lstrip("/")
        if normalized_location.startswith("Archives/"):
            filing_url = f"https://www.sec.gov/{normalized_location}"
        else:
            filing_url = f"https://www.sec.gov/Archives/{normalized_location}"

    print(f"[DEBUG] Fetching filing URL: {filing_url}")
    response = requests.get(filing_url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    primary_document_text = _extract_primary_document(response.text, expected_form=expected_form)
    return _clean_filing_text(primary_document_text)


if __name__ == "__main__":
    text = get_filing_text("AAPL", form_type="10-K", index=0)
    print(text[:5000])