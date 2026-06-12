import os
import time
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


def search_paper_via_query(
    query: str,
    max_paper_num: int = 10,
    min_seconds_between_requests: float = 1.2,
    use_cache: bool = True,
) -> list[dict[str, Any]] | None:
    """
    Search Semantic Scholar for papers using a single query string.

    Uses a local SQLite cache:
    - if the query has already been searched, return the cached response
    - otherwise call the Semantic Scholar API and save the result
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

    response = requests.get(
        SEMANTIC_SCHOLAR_SEARCH_URL,
        params=query_params,
        headers=headers,
        timeout=30,
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
        print("S2 status:", response.status_code)
        print("S2 headers:", dict(response.headers))
        print("S2 body:", response.text)
        print("retry after:", response.headers.get("Retry-After"))

        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                wait_time = float(retry_after)
            except ValueError:
                wait_time = min_seconds_between_requests
        else:
            wait_time = min_seconds_between_requests

        print(f"Rate limited on query '{query}'. Retrying once in {wait_time:.1f}s.")
        time.sleep(wait_time)

        retry_response = requests.get(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            params=query_params,
            headers=headers,
            timeout=30,
        )

        if retry_response.status_code == 200:
            response_data = retry_response.json()
            if response_data is None or "data" not in response_data:
                print(f"retrieval failed after retry for query: {query}")
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

        print(
            f"Retry failed with status code {retry_response.status_code}: "
            f"{retry_response.text}"
        )
        return None

    print(f"Request failed with status code {response.status_code}: {response.text}")
    return None


def search_papers_for_question(
    question: str,
    max_queries: int = 4,
    max_paper_num_per_query: int = 10,
    model: str = "gpt-5.4-mini",
    min_seconds_between_queries: float = 10.0,
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
            min_seconds_between_requests=min_seconds_between_queries,
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
        print("  fallback source type:", paper_fallback_source_type(paper))
        print("  has metadata text:", bool(paper_metadata_text(paper)))