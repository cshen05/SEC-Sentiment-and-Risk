import os
import requests
import time
from typing import Any, Dict, Iterable, Optional, Sequence
from dotenv import load_dotenv

load_dotenv()

EDGAR_IDENTITY = os.getenv("EDGAR_IDENTITY")

if not EDGAR_IDENTITY:
    raise ValueError(
        "EDGAR_IDENTITY environment variable not set. "
        "Add it to your .env file (e.g., 'Your Name your_email@example.com')."
    )

HEADERS = {
    "User-Agent": EDGAR_IDENTITY,
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://www.sec.gov/edgar/search/",
    "Origin": "https://www.sec.gov",
}

DEFAULT_FORMS = ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"]

BASE_URL = "https://efts.sec.gov/LATEST/search-index"


def _normalize_forms(form_type: Optional[str] = None, form_types: Optional[Iterable[str]] = None) -> str:
    """Normalize one or more SEC form types into the EFTS API format."""
    if form_types is not None:
        forms = [form.strip() for form in form_types if form and form.strip()]
    elif form_type is not None:
        forms = [form_type.strip()]
    else:
        forms = DEFAULT_FORMS

    if not forms:
        raise ValueError("At least one SEC form type must be provided.")

    return ",".join(forms)


def _normalize_company_identifiers(companies: Sequence[str]) -> list[str]:
    """Clean and validate a list of company identifiers."""
    normalized = [company.strip() for company in companies if company and company.strip()]
    if not normalized:
        raise ValueError("At least one company identifier must be provided.")
    return normalized


def get_filings(
    form_type: Optional[str] = None,
    form_types: Optional[Iterable[str]] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    query: Optional[str] = None,
    company: Optional[str] = None,
    ticker: Optional[str] = None,
    from_index: int = 0,
    size: int = 50,
    sort: Optional[list[dict[str, dict[str, str]]]] = None,
) -> Dict[str, Any]:
    """Fetch SEC filing metadata from the EDGAR full-text search API.

    Parameters
    ----------
    form_type : Optional[str]
        Single SEC form type.
    form_types : Optional[Iterable[str]]
        Multiple SEC form types.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    query : Optional[str]
        Raw SEC full-text query. If provided, this takes precedence.
    company : Optional[str]
        Company name to search for.
    ticker : Optional[str]
        Ticker symbol to search for.
    from_index : int
        Starting offset for pagination.
    size : int
        Number of results to return.
    sort : Optional[list[dict[str, dict[str, str]]]]
        Custom sort payload for the SEC search API.
    """
    forms = _normalize_forms(form_type=form_type, form_types=form_types)

    if query is not None and query.strip():
        normalized_query = query.strip()
    elif ticker is not None and ticker.strip():
        normalized_query = ticker.strip().upper()
    elif company is not None and company.strip():
        normalized_query = company.strip()
    else:
        normalized_query = "*"

    params = {
        "q": normalized_query,
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
        "forms": forms,
        "from": from_index,
        "size": size,
    }

    # Use a GET request here. The SEC full-text search endpoint is publicly
    # documented as a search interface, and GET has been more reliable than
    # POST for scripted access to this endpoint.
    response = requests.get(
        BASE_URL,
        params=params,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    time.sleep(0.1)
    return response.json()


def extract_hits(data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return the list of filing hits from the SEC search response."""
    hits = data.get("hits", {})
    if isinstance(hits, dict):
        hit_list = hits.get("hits", [])
        if isinstance(hit_list, list):
            return hit_list
    return []


# Filter hits to prefer filings that actually belong to the requested company
def filter_hits_for_company(hits: list[Dict[str, Any]], company_identifier: str) -> list[Dict[str, Any]]:
    """Filter SEC search hits so they more likely belong to the requested company."""
    normalized_identifier = company_identifier.strip().lower()
    filtered_hits: list[Dict[str, Any]] = []

    for hit in hits:
        source = hit.get("_source", {})
        display_names = source.get("display_names") or []
        ciks = source.get("ciks") or []

        if not isinstance(display_names, list):
            display_names = [display_names]
        if not isinstance(ciks, list):
            ciks = [ciks]

        display_text = " ".join(str(name).lower() for name in display_names)
        cik_values = {str(cik).strip() for cik in ciks if cik is not None}

        if normalized_identifier in display_text or normalized_identifier in cik_values:
            filtered_hits.append(hit)

    return filtered_hits


def get_total_hits(data: Dict[str, Any]) -> int:
    """Return the total number of matching filings from the SEC search response."""
    hits = data.get("hits", {})
    if not isinstance(hits, dict):
        return 0

    total = hits.get("total", {})
    if isinstance(total, dict):
        value = total.get("value", 0)
        if isinstance(value, int):
            return value

    if isinstance(total, int):
        return total

    return 0


def print_hit_summary(hit: Dict[str, Any]) -> None:
    """Print a compact summary for one filing result."""
    source = hit.get("_source", {})
    print("Company:", source.get("display_names") or source.get("companyName") or source.get("entityName"))
    print("Form:", source.get("form"))
    print("Filed:", source.get("filedAt"))
    print("CIK:", source.get("ciks"))
    print("Accession:", source.get("adsh") or source.get("accessionNo") or source.get("accessionNumber"))
    print("File name:", source.get("file_name"))


def print_results_summary(data: Dict[str, Any]) -> None:
    """Print a compact summary of the SEC search response."""
    total_hits = get_total_hits(data)
    results = extract_hits(data)
    print(f"Returned {len(results)} filings out of {total_hits} total matches.")


# Helper functions for multi-company ingestion

def get_filings_for_company(
    company_identifier: str,
    form_type: Optional[str] = None,
    form_types: Optional[Iterable[str]] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    from_index: int = 0,
    size: int = 50,
    sort: Optional[list[dict[str, dict[str, str]]]] = None,
) -> Dict[str, Any]:
    """Fetch filings for a single company identifier."""
    company_identifier = company_identifier.strip()
    if not company_identifier:
        raise ValueError("company_identifier cannot be empty.")

    data = get_filings(
        form_type=form_type,
        form_types=form_types,
        start_date=start_date,
        end_date=end_date,
        company=company_identifier,
        from_index=from_index,
        size=size,
        sort=sort,
    )

    filtered_hits = filter_hits_for_company(extract_hits(data), company_identifier)

    if isinstance(data.get("hits"), dict):
        data["hits"]["hits"] = filtered_hits

    return data


def get_all_filings_for_company(
    company_identifier: str,
    form_type: Optional[str] = None,
    form_types: Optional[Iterable[str]] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    page_size: int = 50,
    sort: Optional[list[dict[str, dict[str, str]]]] = None,
) -> list[Dict[str, Any]]:
    """Fetch all available filing hits for a single company using pagination."""
    company_identifier = company_identifier.strip()
    if not company_identifier:
        raise ValueError("company_identifier cannot be empty.")

    if page_size <= 0:
        raise ValueError("page_size must be greater than 0.")

    all_hits: list[Dict[str, Any]] = []
    from_index = 0

    while True:
        data = get_filings_for_company(
            company_identifier=company_identifier,
            form_type=form_type,
            form_types=form_types,
            start_date=start_date,
            end_date=end_date,
            from_index=from_index,
            size=page_size,
            sort=sort,
        )

        hits = extract_hits(data)
        if hits and from_index == 0:
            print("[DEBUG] RAW FIRST HIT KEYS:", list(hits[0].keys()))
            source = hits[0].get("_source", hits[0])
            if isinstance(source, dict):
                print("[DEBUG] RAW FIRST HIT _source KEYS:", list(source.keys()))
                print("[DEBUG] RAW FIRST HIT filedAt:", source.get("filedAt"))
                print("[DEBUG] RAW FIRST HIT periodOfReport:", source.get("periodOfReport"))
                print("[DEBUG] RAW FIRST HIT accessionNo:", source.get("accessionNo"))
                print("[DEBUG] RAW FIRST HIT accessionNumber:", source.get("accessionNumber"))
                print("[DEBUG] RAW FIRST HIT adsh:", source.get("adsh"))
            print("[DEBUG] RAW FIRST HIT FULL:", hits[0])

        if not hits:
            break

        all_hits.extend(hits)

        if len(hits) < page_size:
            break

        total_hits = get_total_hits(data)
        from_index += page_size

        if total_hits and from_index >= total_hits:
            break

    return all_hits


def get_filings_for_companies(
    companies: Sequence[str],
    form_type: Optional[str] = None,
    form_types: Optional[Iterable[str]] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    size_per_company: int = 50,
    paginate: bool = False,
    sort: Optional[list[dict[str, dict[str, str]]]] = None,
) -> Dict[str, list[Dict[str, Any]]]:
    """Fetch filings for multiple companies and return grouped hit lists."""
    normalized_companies = _normalize_company_identifiers(companies)
    grouped_results: Dict[str, list[Dict[str, Any]]] = {}

    for company_identifier in normalized_companies:
        if paginate:
            grouped_results[company_identifier] = get_all_filings_for_company(
                company_identifier=company_identifier,
                form_type=form_type,
                form_types=form_types,
                start_date=start_date,
                end_date=end_date,
                page_size=size_per_company,
                sort=sort,
            )
        else:
            data = get_filings_for_company(
                company_identifier=company_identifier,
                form_type=form_type,
                form_types=form_types,
                start_date=start_date,
                end_date=end_date,
                from_index=0,
                size=size_per_company,
                sort=sort,
            )
            grouped_results[company_identifier] = extract_hits(data)

    return grouped_results


if __name__ == "__main__":
    companies = ["Apple", "Microsoft", "NVIDIA"]

    grouped_results = get_filings_for_companies(
        companies=companies,
        form_types=["10-K", "10-Q"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        size_per_company=5,
        paginate=True,
    )

    for company_name, results in grouped_results.items():
        print(f"{company_name}: {len(results)} filings returned across all pages")
        if results:
            print_hit_summary(results[0])
            print()