from __future__ import annotations

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


def _log(verbose: bool, *args: Any, **kwargs: Any) -> None:
    if verbose:
        print(*args, **kwargs)


def extract_doi_from_paper(paper: dict[str, Any]) -> str | None:
    external_ids = paper.get("externalIds") or {}
    return external_ids.get("DOI")


def build_paper_label(paper: dict[str, Any]) -> str:
    title = paper.get("title") or "unknown title"
    year = paper.get("year")
    doi = extract_doi_from_paper(paper)
    return f"{title} ({year}) | DOI: {doi}"


def _initial_paper_retrieval_record(
    paper: dict[str, Any],
    paper_index: int,
    num_selected_papers: int,
    abstract_only: bool,
) -> dict[str, Any]:
    """
    Create a paper-level retrieval record before retrieval starts.

    The record is updated throughout retrieval so that we can later inspect:
    - whether full text was attempted
    - whether full text succeeded
    - whether abstract/TLDR/metadata fallback was used
    - whether the paper was skipped
    - how many chunks were retrieved
    """
    external_ids = paper.get("externalIds") or {}

    return {
        "paper_index": paper_index,
        "num_selected_papers": num_selected_papers,
        "paper_title": paper.get("title"),
        "year": paper.get("year"),
        "doi": external_ids.get("DOI"),
        "paper_id": paper.get("paperId"),
        "corpus_id": paper.get("corpusId"),
        "semantic_scholar_url": paper.get("url"),
        "open_access_pdf_url": (
            (paper.get("openAccessPdf") or {}).get("url")
            if isinstance(paper.get("openAccessPdf"), dict)
            else None
        ),
        "abstract_available": bool(paper.get("abstract")),
        "tldr_available": bool(
            isinstance(paper.get("tldr"), dict)
            and bool((paper.get("tldr") or {}).get("text"))
        ),
        "metadata_fallback_available": bool(paper_metadata_text(paper)),
        "fallback_source_type": paper_fallback_source_type(paper),
        "abstract_only_mode": abstract_only,
        "full_text_attempted": False,
        "full_text_status": None,
        "full_text_source_type": None,
        "full_text_source_url": None,
        "full_text_validation_note": None,
        "full_text_error": None,
        "used_source_type": None,
        "used_source_url": None,
        "used_fallback": False,
        "retrieval_status": "not_started",
        "skip_reason": None,
        "num_chunks_retrieved": 0,
        "top_chunk_ids": [],
        "max_embedding_similarity": None,
        "mean_embedding_similarity": None,
    }


def _update_record_from_full_text_result(
    record: dict[str, Any],
    full_text_result: dict[str, Any],
) -> None:
    record["full_text_attempted"] = True
    record["full_text_status"] = full_text_result.get("status")
    record["full_text_source_type"] = full_text_result.get("source_type")
    record["full_text_source_url"] = full_text_result.get("source_url")
    record["full_text_validation_note"] = full_text_result.get("validation_note")
    record["full_text_error"] = full_text_result.get("error")


def _update_record_from_chunks(
    record: dict[str, Any],
    top_chunks: pd.DataFrame,
) -> None:
    record["num_chunks_retrieved"] = int(len(top_chunks))

    if "chunk_id" in top_chunks.columns:
        record["top_chunk_ids"] = [
            int(chunk_id)
            for chunk_id in top_chunks["chunk_id"].dropna().tolist()
        ]

    if "similarity" in top_chunks.columns and not top_chunks.empty:
        record["max_embedding_similarity"] = float(top_chunks["similarity"].max())
        record["mean_embedding_similarity"] = float(top_chunks["similarity"].mean())


def summarize_paper_retrieval_records(
    paper_retrieval_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Create a compact summary of paper-level retrieval outcomes.
    """
    if not paper_retrieval_records:
        return {
            "num_papers_attempted": 0,
            "num_papers_with_chunks": 0,
            "num_full_text_success": 0,
            "num_fallback_used": 0,
            "num_skipped": 0,
            "retrieval_status_counts": {},
            "used_source_type_counts": {},
        }

    retrieval_status_counts: dict[str, int] = {}
    used_source_type_counts: dict[str, int] = {}
    fallback_source_type_counts: dict[str, int] = {}
    full_text_source_type_counts: dict[str, int] = {}

    for record in paper_retrieval_records:
        retrieval_status = record.get("retrieval_status") or "unknown"
        retrieval_status_counts[retrieval_status] = (
            retrieval_status_counts.get(retrieval_status, 0) + 1
        )

        used_source_type = record.get("used_source_type") or "none"
        used_source_type_counts[used_source_type] = (
            used_source_type_counts.get(used_source_type, 0) + 1
        )

        if bool(record.get("used_fallback")):
            fallback_source_type_counts[used_source_type] = (
                fallback_source_type_counts.get(used_source_type, 0) + 1
            )
        elif record.get("retrieval_status") == "full_text_success":
            full_text_source_type_counts[used_source_type] = (
                full_text_source_type_counts.get(used_source_type, 0) + 1
            )

    num_papers_with_chunks = sum(
        int(record.get("num_chunks_retrieved") or 0) > 0
        for record in paper_retrieval_records
    )
    num_full_text_success = sum(
        record.get("retrieval_status") == "full_text_success"
        for record in paper_retrieval_records
    )
    num_fallback_used = sum(
        bool(record.get("used_fallback"))
        for record in paper_retrieval_records
    )
    num_skipped = sum(
        str(record.get("retrieval_status") or "").startswith("skipped")
        or bool(record.get("skip_reason"))
        for record in paper_retrieval_records
    )

    denominator = len(paper_retrieval_records)

    return {
        "num_papers_attempted": denominator,
        "num_papers_with_chunks": num_papers_with_chunks,
        "num_full_text_success": num_full_text_success,
        "num_fallback_used": num_fallback_used,
        "num_skipped": num_skipped,
        "full_text_success_fraction": num_full_text_success / denominator,
        "fallback_used_fraction": num_fallback_used / denominator,
        "skipped_fraction": num_skipped / denominator,
        "retrieval_status_counts": retrieval_status_counts,
        "used_source_type_counts": used_source_type_counts,
        "full_text_source_type_counts": full_text_source_type_counts,
        "fallback_source_type_counts": fallback_source_type_counts,
        "num_pdf_used": used_source_type_counts.get("pdf", 0),
        "num_html_used": used_source_type_counts.get("html", 0),
        "num_ar5iv_used": used_source_type_counts.get("ar5iv", 0),
        "num_abstract_fallback_used": fallback_source_type_counts.get("abstract", 0),
        "num_tldr_metadata_fallback_used": fallback_source_type_counts.get("tldr_metadata", 0),
        "num_metadata_fallback_used": fallback_source_type_counts.get("metadata", 0),
    }


def retrieve_candidate_chunks_from_papers(
    query: str,
    papers: list[dict[str, Any]],
    max_papers: int = 5,
    top_k_per_paper: int = 8,
    chunk_sz: int = 250,
    min_chunk_sz: int = 80,
    keep_last: bool = True,
    use_metadata_fallback: bool = True,
    abstract_only: bool = False,
    return_paper_retrieval_records: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Convert Semantic Scholar paper results into candidate evidence chunks.

    For each paper:
    1. Try to retrieve full text, unless abstract_only=True.
    2. If full text fails and fallback is enabled, use the best available
       Semantic Scholar metadata text:
       title + abstract, or title + TLDR, or title/venue/DOI/fields metadata.
    3. Retrieve top-k chunks from the available text using embeddings.
    4. Attach paper metadata to each chunk.
    5. Optionally return paper-level retrieval records for reporting.

    Returns:
        By default:
            DataFrame with candidate evidence chunks across papers.

        If return_paper_retrieval_records=True:
            (candidate_chunks_df, paper_retrieval_records)
    """
    selected_papers = papers[:max_papers]
    all_chunk_dfs: list[pd.DataFrame] = []
    paper_retrieval_records: list[dict[str, Any]] = []

    _log(verbose, f"\nCandidate papers: {len(selected_papers)}")

    for idx, paper in enumerate(selected_papers, start=1):
        title = paper.get("title")
        doi = extract_doi_from_paper(paper)
        url = paper.get("url")

        record = _initial_paper_retrieval_record(
            paper=paper,
            paper_index=idx,
            num_selected_papers=len(selected_papers),
            abstract_only=abstract_only,
        )

        _log(verbose, f"\n[{idx}/{len(selected_papers)}] Trying paper: {title}")
        _log(verbose, "DOI:", doi)
        _log(verbose, "URL:", url)

        source_text: str | None = None
        source_type: str | None = None
        source_url: str | None = None

        if abstract_only:
            fallback_text = paper_metadata_text(paper)

            if not fallback_text:
                _log(verbose, "  skipping paper: no usable abstract/metadata")
                record["retrieval_status"] = "skipped_no_metadata"
                record["skip_reason"] = "abstract_only enabled, but no usable metadata text"
                paper_retrieval_records.append(record)
                continue

            source_text = fallback_text
            source_type = paper_fallback_source_type(paper)
            source_url = url

            record["used_source_type"] = source_type
            record["used_source_url"] = source_url
            record["used_fallback"] = True
            record["retrieval_status"] = "abstract_only_fallback"

            _log(verbose, f"  using Semantic Scholar fallback source: {source_type}")

        else:
            full_text_result = retrieve_full_text(
                doi=doi,
                arxiv_id=None,
                url=url,
            )

            _update_record_from_full_text_result(record, full_text_result)

            _log(verbose, "  retrieval status:", full_text_result.get("status"))
            _log(verbose, "  source type:", full_text_result.get("source_type"))
            _log(verbose, "  source url:", full_text_result.get("source_url"))

            if (
                full_text_result.get("status") == "success"
                and full_text_result.get("full_text")
            ):
                source_text = full_text_result["full_text"]
                source_type = full_text_result.get("source_type")
                source_url = full_text_result.get("source_url")

                record["used_source_type"] = source_type
                record["used_source_url"] = source_url
                record["used_fallback"] = False
                record["retrieval_status"] = "full_text_success"

            elif use_metadata_fallback:
                fallback_text = paper_metadata_text(paper)

                if not fallback_text:
                    _log(verbose, "  skipping paper: no full text and no usable metadata")
                    record["retrieval_status"] = "skipped_no_full_text_or_metadata"
                    record["skip_reason"] = "full text failed and no usable metadata fallback"
                    paper_retrieval_records.append(record)
                    continue

                source_text = fallback_text
                source_type = paper_fallback_source_type(paper)
                source_url = url

                record["used_source_type"] = source_type
                record["used_source_url"] = source_url
                record["used_fallback"] = True
                record["retrieval_status"] = "metadata_fallback_after_full_text_failure"

                _log(verbose, f"  using Semantic Scholar fallback source: {source_type}")

            else:
                _log(verbose, "  skipping paper: no usable full text")
                record["retrieval_status"] = "skipped_no_full_text"
                record["skip_reason"] = (
                    "full text failed and metadata fallback is disabled"
                )
                paper_retrieval_records.append(record)
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
            _log(verbose, "  skipping paper: no chunks retrieved")
            record["retrieval_status"] = (
                f"{record['retrieval_status']}_but_no_chunks"
            )
            record["skip_reason"] = "source text was available, but no chunks were retrieved"
            paper_retrieval_records.append(record)
            continue

        top_chunks = top_chunks.copy()
        top_chunks["paper_title"] = title
        top_chunks["doi"] = doi
        top_chunks["paper_url"] = url
        top_chunks["source_type"] = source_type
        top_chunks["source_url"] = source_url
        top_chunks["paper_label"] = build_paper_label(paper)

        _update_record_from_chunks(record, top_chunks)

        all_chunk_dfs.append(top_chunks)
        paper_retrieval_records.append(record)

    if all_chunk_dfs:
        candidate_chunks_df = pd.concat(all_chunk_dfs, ignore_index=True)
    else:
        candidate_chunks_df = pd.DataFrame()

    if return_paper_retrieval_records:
        return candidate_chunks_df, paper_retrieval_records

    return candidate_chunks_df
