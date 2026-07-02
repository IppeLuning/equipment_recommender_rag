import os
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests
from dotenv import load_dotenv
from openai import OpenAI

from equipment_recommender_rag.literature_search.semantic_scholar_cache import (
    get_cached_response,
    save_response_to_cache,
)

load_dotenv()

S2_API_KEY = os.environ["S2_API_KEY"].strip()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

SEMANTIC_SCHOLAR_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_S2_REQUEST_SPACING_SECONDS = 1.1
DEFAULT_S2_INITIAL_BACKOFF_SECONDS = 1.0
DEFAULT_S2_MAX_BACKOFF_SECONDS = 16.0
DEFAULT_S2_MAX_RETRIES = 4
DEFAULT_S2_JITTER_SECONDS = 0.25


# Keep this as a constant so the cache key is stable and easy to reuse.
# These fields support richer fallback when an abstract or full text is unavailable.
SEMANTIC_SCHOLAR_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "year",
        "abstract",
        "authors.name",
        "citationCount",
        "url",
        "externalIds",
        "venue",
        "publicationVenue",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
        "tldr",
        "openAccessPdf",
        "isOpenAccess",
        "publicationTypes",
    ]
)


keyword_extraction_prompt = """
Suggest Semantic Scholar search queries to retrieve relevant papers that can help answer the question.
The search queries must be short and comma separated.

Important rules:
- Preserve the material/domain from the original question when it matters.
- Preserve the problem or failure mode from the original question when it matters.
- Preserve the analysis or measurement intent.
- Avoid overly generic standalone queries such as "phase separation analysis", "morphological analysis", or "SEM microstructure".
- Prefer specific literature-search phrases over full sentences.


Question: {question}
Search queries:
"""


def generate_search_queries(
    question: str,
    model: str = "gpt-5.4-mini",
) -> list[str]:
    """
    Use OpenAI to generate short Semantic Scholar search queries.
    """
    prompt = keyword_extraction_prompt.format(question=question)

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    raw_output = response.output_text.strip()

    if "Search queries:" in raw_output:
        raw_output = raw_output.split("Search queries:", 1)[1].strip()

    queries = [
        q.strip()
        for q in raw_output.split(",")
        if q.strip()
    ]

    return queries

def get_publication_types(paper: dict[str, Any]) -> list[str]:
    """
    Return Semantic Scholar publication types, for example:
    ['Review'] or ['JournalArticle'].
    """
    publication_types = paper.get("publicationTypes") or []

    if isinstance(publication_types, list):
        return [
            item.strip()
            for item in publication_types
            if isinstance(item, str) and item.strip()
        ]

    if isinstance(publication_types, str) and publication_types.strip():
        return [publication_types.strip()]

    return []


def is_review_paper(paper: dict[str, Any]) -> bool:
    """
    Heuristic: mark as review if Semantic Scholar publicationTypes contains review.
    """
    publication_types = get_publication_types(paper)
    return any("review" in item.lower() for item in publication_types)

def _get_tldr_text(paper: dict[str, Any]) -> str | None:
    """
    Semantic Scholar returns tldr as an object, usually {'model': ..., 'text': ...}.
    This helper safely extracts the text when present.
    """
    tldr = paper.get("tldr")
    if isinstance(tldr, dict):
        text = tldr.get("text")
        return text.strip() if isinstance(text, str) and text.strip() else None
    if isinstance(tldr, str):
        return tldr.strip() or None
    return None


def _get_publication_venue_name(paper: dict[str, Any]) -> str | None:
    publication_venue = paper.get("publicationVenue")
    if isinstance(publication_venue, dict):
        name = publication_venue.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    venue = paper.get("venue")
    if isinstance(venue, str) and venue.strip():
        return venue.strip()

    return None


def _get_s2_fields_of_study(paper: dict[str, Any]) -> list[str]:
    s2_fields = paper.get("s2FieldsOfStudy") or []
    names: list[str] = []

    for item in s2_fields:
        if isinstance(item, dict):
            category = item.get("category")
            if isinstance(category, str) and category.strip():
                names.append(category.strip())
        elif isinstance(item, str) and item.strip():
            names.append(item.strip())

    fields_of_study = paper.get("fieldsOfStudy") or []
    for field in fields_of_study:
        if isinstance(field, str) and field.strip():
            names.append(field.strip())

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(names))


def paper_metadata_text(paper: dict[str, Any]) -> str | None:
    """
    Build the best available fallback text from a Semantic Scholar paper record.

    Priority:
    - title is usually present
    - abstract if available
    - TLDR if available
    - venue, year, DOI, field-of-study metadata

    This allows the downstream evidence pipeline to use metadata when full text
    and abstract are missing, instead of dropping a potentially relevant paper.
    """
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    tldr_text = _get_tldr_text(paper)
    year = paper.get("year")
    citation_count = paper.get("citationCount")
    external_ids = paper.get("externalIds") or {}
    doi = external_ids.get("DOI")
    authors = paper.get("authors") or []
    venue_name = _get_publication_venue_name(paper)
    fields_of_study = _get_s2_fields_of_study(paper)
    publication_types = get_publication_types(paper)

    author_names = [
        author.get("name")
        for author in authors
        if isinstance(author, dict) and author.get("name")
    ]

    parts: list[str] = []

    if title:
        parts.append(f"Title: {title}")

    if abstract:
        parts.append(f"Abstract: {abstract}")

    if tldr_text:
        parts.append(f"TLDR: {tldr_text}")

    if publication_types:
        parts.append(f"Publication types: {', '.join(publication_types)}")
        parts.append(f"Is review paper: {is_review_paper(paper)}")

    if year:
        parts.append(f"Year: {year}")

    if venue_name:
        parts.append(f"Venue: {venue_name}")

    if doi:
        parts.append(f"DOI: {doi}")

    if citation_count is not None:
        parts.append(f"Citation count: {citation_count}")

    if author_names:
        parts.append(f"Authors: {', '.join(author_names[:8])}")

    if fields_of_study:
        parts.append(f"Fields of study: {', '.join(fields_of_study[:8])}")

    text = "\n".join(parts).strip()
    return text if text else None


def paper_fallback_source_type(paper: dict[str, Any]) -> str:
    """
    Label how strong the fallback evidence is.
    """
    if paper.get("abstract"):
        return "abstract"
    if _get_tldr_text(paper):
        return "tldr_metadata"
    return "metadata"



def _parse_retry_after_seconds(retry_after: str | None) -> float | None:
    """
    Parse a Retry-After header.

    Retry-After can be either:
    - a number of seconds, e.g. "3"
    - an HTTP-date

    Returns None if the header is missing or cannot be parsed.
    """
    if not retry_after:
        return None

    retry_after = retry_after.strip()

    try:
        wait_seconds = float(retry_after)
        return max(0.0, wait_seconds)
    except ValueError:
        pass

    try:
        retry_datetime = parsedate_to_datetime(retry_after)
        if retry_datetime.tzinfo is None:
            retry_datetime = retry_datetime.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        return max(0.0, (retry_datetime - now).total_seconds())
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _calculate_backoff_seconds(
    attempt: int,
    response: requests.Response | None = None,
    initial_backoff_seconds: float = DEFAULT_S2_INITIAL_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_S2_MAX_BACKOFF_SECONDS,
    jitter_seconds: float = DEFAULT_S2_JITTER_SECONDS,
) -> float:
    """
    Calculate capped exponential backoff.

    attempt is 0 for the first retry wait, 1 for the second, etc.

    If Semantic Scholar returns Retry-After, use it as a lower bound when possible,
    but never wait longer than max_backoff_seconds.
    """
    exponential_wait = initial_backoff_seconds * (2 ** attempt)

    retry_after_wait = None
    if response is not None:
        retry_after_wait = _parse_retry_after_seconds(response.headers.get("Retry-After"))

    wait_time = exponential_wait
    if retry_after_wait is not None:
        wait_time = max(wait_time, retry_after_wait)

    if jitter_seconds > 0:
        wait_time += random.uniform(0, jitter_seconds)

    return min(wait_time, max_backoff_seconds)


def _get_with_exponential_backoff(
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
    timeout: int = 30,
    max_retries: int = DEFAULT_S2_MAX_RETRIES,
    initial_backoff_seconds: float = DEFAULT_S2_INITIAL_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_S2_MAX_BACKOFF_SECONDS,
    jitter_seconds: float = DEFAULT_S2_JITTER_SECONDS,
) -> requests.Response:
    """
    GET request with capped exponential backoff for temporary failures.

    Retries:
    - 429 Too Many Requests
    - 5xx temporary/server errors

    Does not retry permanent client errors such as 400 or 403.
    """
    last_response: requests.Response | None = None

    for attempt in range(max_retries + 1):
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        last_response = response

        if response.status_code not in SEMANTIC_SCHOLAR_RETRYABLE_STATUS_CODES:
            return response

        if attempt >= max_retries:
            return response

        wait_time = _calculate_backoff_seconds(
            attempt=attempt,
            response=response,
            initial_backoff_seconds=initial_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
            jitter_seconds=jitter_seconds,
        )

        retry_after = response.headers.get("Retry-After")
        retry_after_text = f", Retry-After={retry_after}" if retry_after else ""

        print(
            f"Semantic Scholar returned {response.status_code}{retry_after_text}. "
            f"Retry {attempt + 1}/{max_retries} in {wait_time:.1f}s."
        )

        time.sleep(wait_time)

    assert last_response is not None
    return last_response


def search_paper_via_query(
    query: str,
    max_paper_num: int = 10,
    use_cache: bool = True,
    max_retries: int = DEFAULT_S2_MAX_RETRIES,
    initial_backoff_seconds: float = DEFAULT_S2_INITIAL_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_S2_MAX_BACKOFF_SECONDS,
) -> list[dict[str, Any]] | None:
    """
    Search Semantic Scholar for papers using a single query string.

    Uses a local SQLite cache:
    - if the query has already been searched, return the cached response
    - otherwise call the Semantic Scholar API and save the result

    Retry behavior:
    - 429 and temporary 5xx errors use capped exponential backoff
    - the maximum wait between retries is max_backoff_seconds, default 16s
    - successful responses are cached
    - failed/rate-limited responses are not cached
    """
    if "Search queries:" in query:
        query = query.split("Search queries:", 1)[1].strip()

    min_citation_count = 0
    sort = ""
    fields = SEMANTIC_SCHOLAR_FIELDS

    if use_cache:
        cached_papers = get_cached_response(
            query=query,
            max_paper_num=max_paper_num,
            min_citation_count=min_citation_count,
            sort=sort,
            fields=fields,
        )

        if cached_papers is not None:
            print(f"Using cached Semantic Scholar response for query: {query}")
            return cached_papers

    query_params = {
        "query": query,
        "limit": max_paper_num,
        "minCitationCount": min_citation_count,
        "fields": fields,
    }

    print("Calling Semantic Scholar query:", query)
    print("Using S2 key:", S2_API_KEY[:6] + "..." + S2_API_KEY[-4:])

    headers = {"x-api-key": S2_API_KEY}

    response = _get_with_exponential_backoff(
        url=SEMANTIC_SCHOLAR_SEARCH_URL,
        params=query_params,
        headers=headers,
        timeout=30,
        max_retries=max_retries,
        initial_backoff_seconds=initial_backoff_seconds,
        max_backoff_seconds=max_backoff_seconds,
    )

    if response.status_code == 200:
        response_data = response.json()
        if response_data is None or "data" not in response_data:
            print(f"retrieval failed for query as no papers showed up: {query}")
            return None

        papers = response_data["data"]

        if use_cache:
            save_response_to_cache(
                query=query,
                max_paper_num=max_paper_num,
                min_citation_count=min_citation_count,
                sort=sort,
                fields=fields,
                papers=papers,
            )

        return papers

    if response.status_code == 429:
        print("S2 status: 429 Too Many Requests")
        print("S2 Retry-After:", response.headers.get("Retry-After"))
        print(
            "Semantic Scholar still rate-limited this query after retries. "
            "Skipping this query without caching the failed response."
        )
        return None

    print(f"Request failed with status code {response.status_code}: {response.text}")
    return None

def search_papers_for_question(
    question: str,
    max_queries: int = 4,
    max_paper_num_per_query: int = 10,
    model: str = "gpt-5.4-mini",
    min_seconds_between_queries: float = DEFAULT_S2_REQUEST_SPACING_SECONDS,
) -> dict[str, Any]:
    """
    Full literature-search step:
    1. Use OpenAI to generate Semantic Scholar search queries.
    2. Search Semantic Scholar for each query.
    3. Return both the queries and retrieved papers.
    """
    queries = generate_search_queries(question=question, model=model)
    queries = queries[:max_queries]

    results = []
    seen_paper_ids = set()

    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(min_seconds_between_queries)

        papers = search_paper_via_query(
            query,
            max_paper_num=max_paper_num_per_query,
            use_cache=True,
        )

        if not papers:
            continue

        for paper in papers:
            external_ids = paper.get("externalIds") or {}
            paper_id = (
                paper.get("paperId")
                or external_ids.get("CorpusId")
                or external_ids.get("DOI")
                or paper.get("url")
                or paper.get("title")
            )

            if paper_id in seen_paper_ids:
                continue

            seen_paper_ids.add(paper_id)
            results.append(paper)

    return {
        "question": question,
        "generated_queries": queries,
        "papers": results,
    }


if __name__ == "__main__":
    question = (
        "Our experimental medical-grade PEEK variant is much softer than expected. "
        "We suspect morphology or phase separation may be involved. Which equipment "
        "could help analyze this?"
    )

    output = search_papers_for_question(question)

    print("Generated queries:")
    for q in output["generated_queries"]:
        print("-", q)

    print(f"\nRetrieved {len(output['papers'])} unique papers.")
    for paper in output["papers"][:5]:
        print("-", paper.get("title"), f"({paper.get('year')})")
        print("  publication types:", get_publication_types(paper))
        print("  is review paper:", is_review_paper(paper))
        print("  fallback source type:", paper_fallback_source_type(paper))
        print("  has metadata text:", bool(paper_metadata_text(paper)))