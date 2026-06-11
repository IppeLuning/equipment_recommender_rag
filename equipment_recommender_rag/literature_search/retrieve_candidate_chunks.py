from typing import Any

import pandas as pd

from equipment_recommender_rag.embeddings.retrieve_top_k_chunks import (
    retrieve_top_k_chunks_from_full_text,
)
from equipment_recommender_rag.full_text_chunking.retrieve_full_text import (
    retrieve_full_text,
)
from equipment_recommender_rag.literature_search.semantic_scholar import (
    paper_fallback_source_type,
    paper_metadata_text,
)


def extract_doi_from_paper(paper: dict[str, Any]) -> str | None:
    external_ids = paper.get("externalIds") or {}
    return external_ids.get("DOI")


def build_paper_label(paper: dict[str, Any]) -> str:
    title = paper.get("title") or "unknown title"
    year = paper.get("year")
    doi = extract_doi_from_paper(paper)
    return f"{title} ({year}) | DOI: {doi}"


def retrieve_candidate_chunks_from_papers(
    query: str,
    papers: list[dict[str, Any]],
    max_papers: int = 5,
    top_k_per_paper: int = 8,
    chunk_sz: int = 250,
    min_chunk_sz: int = 80,
    keep_last: bool = True,
    use_metadata_fallback: bool = True,
) -> pd.DataFrame:
    """
    Convert Semantic Scholar paper results into candidate evidence chunks.

    For each paper:
    1. Try to retrieve full text.
    2. If full text fails and fallback is enabled, use the best available
       Semantic Scholar metadata text:
       title + abstract, or title + TLDR, or title/venue/DOI/fields metadata.
    3. Retrieve top-k chunks from the available text using embeddings.
    4. Attach paper metadata to each chunk.

    Returns:
        DataFrame with candidate evidence chunks across papers.
    """
    selected_papers = papers[:max_papers]
    all_chunk_dfs: list[pd.DataFrame] = []

    print(f"\nCandidate papers: {len(selected_papers)}")

    for idx, paper in enumerate(selected_papers, start=1):
        title = paper.get("title")
        doi = extract_doi_from_paper(paper)
        url = paper.get("url")

        print(f"\n[{idx}/{len(selected_papers)}] Trying paper: {title}")
        print("DOI:", doi)
        print("URL:", url)

        full_text_result = retrieve_full_text(
            doi=doi,
            arxiv_id=None,
            url=url,
        )

        print("  retrieval status:", full_text_result["status"])
        print("  source type:", full_text_result["source_type"])
        print("  source url:", full_text_result["source_url"])

        if full_text_result["status"] == "success" and full_text_result["full_text"]:
            source_text = full_text_result["full_text"]
            source_type = full_text_result["source_type"]
            source_url = full_text_result["source_url"]

        elif use_metadata_fallback:
            fallback_text = paper_metadata_text(paper)

            if not fallback_text:
                print("  skipping paper: no full text and no usable metadata")
                continue

            source_type = paper_fallback_source_type(paper)

            if source_type == "abstract":
                print("  using abstract fallback")
            elif source_type == "tldr_metadata":
                print("  using TLDR/metadata fallback")
            else:
                print("  using title/metadata fallback")

            source_text = fallback_text
            source_url = url

        else:
            print("  skipping paper: no usable full text")
            continue

        top_chunks = retrieve_top_k_chunks_from_full_text(
            query=query,
            full_text=source_text,
            top_k=top_k_per_paper,
            chunk_sz=chunk_sz,
            min_chunk_sz=min_chunk_sz,
            keep_last=keep_last,
            return_embeddings=False,
        )

        if top_chunks.empty:
            print("  skipping paper: no chunks retrieved")
            continue

        top_chunks = top_chunks.copy()
        top_chunks["paper_title"] = title
        top_chunks["doi"] = doi
        top_chunks["paper_url"] = url
        top_chunks["source_type"] = source_type
        top_chunks["source_url"] = source_url
        top_chunks["paper_label"] = build_paper_label(paper)

        all_chunk_dfs.append(top_chunks)

    if not all_chunk_dfs:
        return pd.DataFrame()

    return pd.concat(all_chunk_dfs, ignore_index=True)
