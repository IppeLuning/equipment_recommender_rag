from __future__ import annotations

from typing import Any

import pandas as pd

from equipment_recommender_rag.literature_search.semantic_scholar import (
    is_review_paper,
    paper_fallback_source_type,
    paper_metadata_text,
)
from equipment_recommender_rag.reranking.rerank_chunks import rerank_top_k_chunks


DEFAULT_PAPER_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def get_paper_identifier(paper: dict[str, Any]) -> str | None:
    """
    Return a stable identifier for a Semantic Scholar paper record.
    """
    external_ids = paper.get("externalIds") or {}
    identifier = (
        paper.get("paperId")
        or external_ids.get("CorpusId")
        or external_ids.get("DOI")
        or paper.get("url")
        or paper.get("title")
    )
    return str(identifier) if identifier is not None else None


def get_paper_doi(paper: dict[str, Any]) -> str | None:
    external_ids = paper.get("externalIds") or {}
    doi = external_ids.get("DOI")
    return str(doi) if doi else None


def _truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0:
        return text
    return text[:max_chars]


def _paper_record_for_reranking(
    paper: dict[str, Any],
    original_index: int,
    text_max_chars: int | None,
) -> dict[str, Any] | None:
    rerank_text = paper_metadata_text(paper)
    if not rerank_text:
        return None

    rerank_text = _truncate_text(rerank_text, text_max_chars).strip()
    if not rerank_text:
        return None

    return {
        "original_index": original_index,
        "paper": paper,
        "paper_id": get_paper_identifier(paper),
        "title": paper.get("title"),
        "year": paper.get("year"),
        "doi": get_paper_doi(paper),
        "citation_count": paper.get("citationCount"),
        "is_open_access": bool(paper.get("isOpenAccess")),
        "has_open_access_pdf": bool(paper.get("openAccessPdf")),
        "is_review_paper": is_review_paper(paper),
        "reranking_text_type": paper_fallback_source_type(paper),
        "paper_rerank_text": rerank_text,
    }


def rerank_papers_by_metadata(
    query: str,
    papers: list[dict[str, Any]],
    top_n: int | None = None,
    model_name: str = DEFAULT_PAPER_RERANKER_MODEL,
    normalize: bool = True,
    text_max_chars: int | None = 4000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Rerank Semantic Scholar paper records using title/abstract/TLDR/metadata text.

    This should run after Semantic Scholar search and before full-text retrieval.
    It lets the pipeline search broadly, then process only the most query-relevant
    papers according to the same BGE reranker used for chunk reranking.

    Parameters
    ----------
    query:
        Original user query or subproblem query.
    papers:
        Unique Semantic Scholar paper records.
    top_n:
        Number of reranked papers to return. If None, return all reranked papers.
    model_name:
        Cross-encoder reranker model name.
    normalize:
        Whether to sigmoid-normalize reranker logits.
    text_max_chars:
        Maximum number of metadata characters per paper sent to the reranker.
        Keeps paper-level reranking fast and avoids very long abstracts/metadata.

    Returns
    -------
    selected_papers:
        Reranked paper dictionaries, truncated to top_n when top_n is set.
    paper_reranking_records:
        JSON-serializable records with rank, score and metadata for debugging.
    """
    if not papers:
        return [], []

    records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []

    for original_index, paper in enumerate(papers):
        record = _paper_record_for_reranking(
            paper=paper,
            original_index=original_index,
            text_max_chars=text_max_chars,
        )
        if record is None:
            skipped_records.append(
                {
                    "original_index": original_index,
                    "paper_id": get_paper_identifier(paper),
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "doi": get_paper_doi(paper),
                    "skipped_reason": "missing_title_abstract_tldr_metadata_text",
                    "selected_for_processing": False,
                }
            )
            continue

        records.append(record)

    if not records:
        fallback_papers = papers[:top_n] if top_n is not None else papers
        return fallback_papers, skipped_records

    papers_df = pd.DataFrame(records)

    reranked_df = rerank_top_k_chunks(
        query=query,
        retrieved_chunks_df=papers_df,
        text_column="paper_rerank_text",
        model_name=model_name,
        normalize=normalize,
    )

    if top_n is not None:
        selected_df = reranked_df.head(top_n).copy()
    else:
        selected_df = reranked_df.copy()

    selected_original_indices = set(selected_df["original_index"].tolist())
    selected_papers = selected_df["paper"].tolist()

    paper_reranking_records: list[dict[str, Any]] = []
    for rank, row in enumerate(reranked_df.itertuples(index=False), start=1):
        original_index = int(row.original_index)
        paper_reranking_records.append(
            {
                "paper_rerank_rank": rank,
                "paper_reranker_score": float(row.reranker_score),
                "selected_for_processing": original_index in selected_original_indices,
                "original_index": original_index,
                "paper_id": row.paper_id,
                "title": row.title,
                "year": row.year,
                "doi": row.doi,
                "citation_count": row.citation_count,
                "is_open_access": bool(row.is_open_access),
                "has_open_access_pdf": bool(row.has_open_access_pdf),
                "is_review_paper": bool(row.is_review_paper),
                "reranking_text_type": row.reranking_text_type,
            }
        )

    paper_reranking_records.extend(skipped_records)
    return selected_papers, paper_reranking_records


def summarize_paper_reranking_records(
    paper_reranking_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compact summary for saved pipeline output.
    """
    selected_records = [
        record
        for record in paper_reranking_records
        if record.get("selected_for_processing")
    ]
    scored_records = [
        record
        for record in paper_reranking_records
        if "paper_reranker_score" in record
    ]
    skipped_records = [
        record
        for record in paper_reranking_records
        if record.get("skipped_reason")
    ]

    text_type_counts: dict[str, int] = {}
    for record in scored_records:
        text_type = record.get("reranking_text_type") or "unknown"
        text_type_counts[text_type] = text_type_counts.get(text_type, 0) + 1

    selected_text_type_counts: dict[str, int] = {}
    for record in selected_records:
        text_type = record.get("reranking_text_type") or "unknown"
        selected_text_type_counts[text_type] = selected_text_type_counts.get(text_type, 0) + 1

    return {
        "paper_reranking_enabled": True,
        "num_papers_scored": len(scored_records),
        "num_papers_selected": len(selected_records),
        "num_papers_skipped_before_reranking": len(skipped_records),
        "reranking_text_type_counts": text_type_counts,
        "selected_reranking_text_type_counts": selected_text_type_counts,
        "top_selected_papers": [
            {
                "rank": record.get("paper_rerank_rank"),
                "score": record.get("paper_reranker_score"),
                "title": record.get("title"),
                "year": record.get("year"),
                "doi": record.get("doi"),
            }
            for record in selected_records[:5]
        ],
    }
