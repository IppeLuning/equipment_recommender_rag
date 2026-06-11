from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pypdf import PdfReader


load_dotenv()

UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL")

HEADERS = {
    "User-Agent": (
        f"equipment_recommender_rag/0.1 "
        f"contact: {UNPAYWALL_EMAIL or 'no-email-provided'}"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def is_rug_cover_page(text: str) -> bool:
    text_lower = text.lower()

    cover_page_signals = [
        "university of groningen",
        "important note: you are advised to consult the publisher's version",
        "downloaded from the university of groningen",
        "take-down policy",
        "citation for published version",
        "document version",
    ]

    matches = sum(signal in text_lower for signal in cover_page_signals)
    return matches >= 3


def extract_text_from_pdf_bytes(
    pdf_bytes: bytes,
    remove_rug_cover_page: bool = True,
) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        reader = PdfReader(str(tmp_path))
        pages = []

        for page_index, page in enumerate(reader.pages):
            page_number = page_index + 1
            text = page.extract_text() or ""

            if page_index == 0 and remove_rug_cover_page and is_rug_cover_page(text):
                continue

            pages.append(f"\n\n--- PAGE {page_number} ---\n{text}")

        return "\n".join(pages).strip()

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def fetch_unpaywall(doi: str) -> dict[str, Any] | None:
    if not UNPAYWALL_EMAIL:
        return None

    try:
        response = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": UNPAYWALL_EMAIL},
            headers=HEADERS,
            timeout=(10, 30),
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def get_candidate_pdf_urls(unpaywall_data: dict[str, Any]) -> list[str]:
    urls: list[str] = []

    best_location = unpaywall_data.get("best_oa_location") or {}
    if best_location.get("url_for_pdf"):
        urls.append(best_location["url_for_pdf"])

    for location in unpaywall_data.get("oa_locations") or []:
        pdf_url = location.get("url_for_pdf")
        if pdf_url and pdf_url not in urls:
            urls.append(pdf_url)

    def priority(url: str) -> int:
        url_lower = url.lower()
        if "pure.rug.nl" in url_lower:
            return 0
        if "research.rug.nl" in url_lower:
            return 1
        if "arxiv.org" in url_lower:
            return 2
        if "nature.com" in url_lower:
            return 3
        return 4

    return sorted(urls, key=priority)


def get_candidate_html_urls(unpaywall_data: dict[str, Any], doi: str | None = None) -> list[str]:
    urls: list[str] = []

    best_location = unpaywall_data.get("best_oa_location") or {}
    for key in ["url", "url_for_landing_page"]:
        value = best_location.get(key)
        if value and value not in urls:
            urls.append(value)

    for location in unpaywall_data.get("oa_locations") or []:
        for key in ["url", "url_for_landing_page"]:
            value = location.get(key)
            if value and value not in urls:
                urls.append(value)

    if doi:
        urls.append(f"https://doi.org/{doi}")

    filtered_urls = []
    for url in urls:
        url_lower = url.lower()
        if "pdfdirect" in url_lower:
            continue
        if url_lower.endswith(".pdf"):
            continue
        if url not in filtered_urls:
            filtered_urls.append(url)

    return filtered_urls


def parse_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None

    value = value.strip()

    url_match = re.search(r"arxiv\.org/(abs|pdf)/([^/?#]+)", value, flags=re.IGNORECASE)
    if url_match:
        arxiv_id = url_match.group(2)
        arxiv_id = arxiv_id.replace(".pdf", "")
        return arxiv_id

    ar5iv_match = re.search(r"ar5iv\.labs\.arxiv\.org/html/([^/?#]+)", value, flags=re.IGNORECASE)
    if ar5iv_match:
        return ar5iv_match.group(1)

    plain_match = re.fullmatch(r"(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(v\d+)?", value, flags=re.IGNORECASE)
    if plain_match:
        return value

    return None


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_="ltx_page_content")
        or soup.find("body")
        or soup
    )

    parts: list[str] = []

    for elem in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = elem.get_text(" ", strip=True)
        if not text:
            continue

        if elem.name.startswith("h"):
            parts.append(f"\n## {text}\n")
        else:
            parts.append(text)

    cleaned = "\n".join(parts)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def is_probably_full_text_html(text: str) -> tuple[bool, str]:
    """
    Heuristic filter to reject abstract-only or landing pages.
    """
    if not text:
        return False, "empty text"

    text_lower = text.lower()

    # Too short is almost never a real full-text article
    if len(text) < 5000:
        return False, f"text too short ({len(text)} chars)"

    landing_page_signals = [
        "download pdf",
        "buy article",
        "access through your institution",
        "view full text",
        "purchase access",
        "rights and permissions",
    ]
    landing_hits = sum(signal in text_lower for signal in landing_page_signals)
    if landing_hits >= 2:
        return False, "looks like publisher landing/access page"

    section_keywords = [
        "introduction",
        "background",
        "methods",
        "materials and methods",
        "experimental",
        "results",
        "discussion",
        "conclusion",
        "references",
    ]
    section_hits = sum(keyword in text_lower for keyword in section_keywords)

    if section_hits < 2:
        return False, "not enough full-text section signals"

    # Abstract-only patterns
    has_abstract = "abstract" in text_lower
    has_methods = ("methods" in text_lower) or ("materials and methods" in text_lower) or ("experimental" in text_lower)
    has_results = "results" in text_lower
    has_references = "references" in text_lower

    if has_abstract and not (has_methods or has_results or has_references):
        return False, "looks like abstract-only page"

    # Real full text often references figures/tables
    figure_table_hits = sum(
        token in text_lower
        for token in ["fig.", "figure 1", "table 1", "supplementary", "et al."]
    )

    if section_hits < 3 and figure_table_hits == 0:
        return False, "weak full-text structure"

    return True, "passes full-text HTML heuristic"


def try_fetch_html(url: str) -> str | None:
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=(10, 60),
            allow_redirects=True,
        )

        if response.status_code == 403:
            return None

        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type and "xml" not in content_type:
            return None

        return response.text

    except requests.RequestException:
        return None


def try_fetch_pdf(url: str) -> bytes | None:
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=(10, 120),
            allow_redirects=True,
        )

        if response.status_code == 403:
            return None

        response.raise_for_status()

        if not response.content.startswith(b"%PDF"):
            return None

        return response.content

    except requests.RequestException:
        return None


def fetch_from_ar5iv(arxiv_id: str) -> dict[str, Any] | None:
    ar5iv_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    html = try_fetch_html(ar5iv_url)

    if not html:
        return None

    text = extract_text_from_html(html)
    if not text:
        return None

    ok, reason = is_probably_full_text_html(text)
    if not ok:
        return None

    return {
        "status": "success",
        "source_type": "ar5iv_html",
        "source_url": ar5iv_url,
        "full_text": text,
        "validation_note": reason,
    }


def fetch_from_html_candidates(urls: list[str]) -> dict[str, Any] | None:
    for url in urls:
        html = try_fetch_html(url)
        if not html:
            continue

        text = extract_text_from_html(html)
        if not text:
            continue

        ok, reason = is_probably_full_text_html(text)
        if not ok:
            continue

        return {
            "status": "success",
            "source_type": "publisher_html",
            "source_url": url,
            "full_text": text,
            "validation_note": reason,
        }

    return None


def fetch_from_pdf_candidates(urls: list[str]) -> dict[str, Any] | None:
    for url in urls:
        pdf_bytes = try_fetch_pdf(url)
        if not pdf_bytes:
            continue

        text = extract_text_from_pdf_bytes(pdf_bytes)
        if not text:
            continue

        if len(text) < 3000:
            continue

        return {
            "status": "success",
            "source_type": "pdf",
            "source_url": url,
            "full_text": text,
            "validation_note": "pdf text extracted",
        }

    return None


def retrieve_full_text(
    doi: str | None = None,
    arxiv_id: str | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    """
    Best-effort full-text retrieval.

    Retrieval priority:
    1. ar5iv / arXiv HTML
    2. PDF via Unpaywall
    3. publisher HTML via Unpaywall, but only if it passes a full-text check
    """
    derived_arxiv_id = arxiv_id or parse_arxiv_id(url) or parse_arxiv_id(doi)

    if derived_arxiv_id:
        ar5iv_result = fetch_from_ar5iv(derived_arxiv_id)
        if ar5iv_result:
            ar5iv_result["doi"] = doi
            ar5iv_result["arxiv_id"] = derived_arxiv_id
            ar5iv_result["error"] = None
            return ar5iv_result

    if not doi:
        return {
            "status": "failed",
            "source_type": None,
            "source_url": None,
            "doi": doi,
            "arxiv_id": derived_arxiv_id,
            "full_text": None,
            "validation_note": None,
            "error": "No DOI provided and ar5iv retrieval failed or was unavailable.",
        }

    unpaywall_data = fetch_unpaywall(doi)
    if not unpaywall_data:
        return {
            "status": "failed",
            "source_type": None,
            "source_url": None,
            "doi": doi,
            "arxiv_id": derived_arxiv_id,
            "full_text": None,
            "validation_note": None,
            "error": "Unpaywall lookup failed.",
        }

    # Prefer PDF before publisher HTML, because publisher HTML may be landing/abstract only
    pdf_urls = get_candidate_pdf_urls(unpaywall_data)
    pdf_result = fetch_from_pdf_candidates(pdf_urls)
    if pdf_result:
        pdf_result["doi"] = doi
        pdf_result["arxiv_id"] = derived_arxiv_id
        pdf_result["error"] = None
        return pdf_result

    html_urls = get_candidate_html_urls(unpaywall_data, doi=doi)
    html_result = fetch_from_html_candidates(html_urls)
    if html_result:
        html_result["doi"] = doi
        html_result["arxiv_id"] = derived_arxiv_id
        html_result["error"] = None
        return html_result

    return {
        "status": "failed",
        "source_type": None,
        "source_url": None,
        "doi": doi,
        "arxiv_id": derived_arxiv_id,
        "full_text": None,
        "validation_note": None,
        "error": "No accessible full text found via ar5iv, PDF, or verified HTML.",
    }


if __name__ == "__main__":
    result = retrieve_full_text(
        doi="10.1038/s41467-020-15678-y",
        arxiv_id=None,
        url=None,
    )

    print("status:", result["status"])
    print("source_type:", result["source_type"])
    print("source_url:", result["source_url"])
    print("doi:", result["doi"])
    print("arxiv_id:", result["arxiv_id"])
    print("validation_note:", result.get("validation_note"))
    print("text length:", len(result["full_text"]) if result["full_text"] else 0)
    print("text preview:", result["full_text"][:50000] if result["full_text"] else "None")
    if result["error"]:
        print("error:", result["error"])