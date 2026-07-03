from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from equipment_recommender_rag.equipment_inventory.extract_equipment_from_text import (
    extract_equipment_from_reranked_chunks,
)
from equipment_recommender_rag.literature_search.retrieve_candidate_chunks import (
    retrieve_candidate_chunks_from_papers,
    summarize_paper_retrieval_records,
)
from equipment_recommender_rag.literature_search.semantic_scholar import (
    search_papers_for_question,
)
from equipment_recommender_rag.reranking.rerank_chunks import (
    rerank_top_k_chunks,
)
from equipment_recommender_rag.reranking.rerank_papers import (
    DEFAULT_PAPER_RERANKER_MODEL,
    rerank_papers_by_metadata,
    summarize_paper_reranking_records,
)
from equipment_recommender_rag.utils.save_pipeline_results import (
    DEFAULT_RESULTS_PATH,
    save_run_record,
)


def print_paper_retrieval_summary(
    paper_retrieval_summary: dict[str, Any],
) -> None:
    """
    Print only a compact source-availability overview.

    This intentionally does not print one line per paper, because that becomes
    too noisy for large runs. The detailed paper records can still be enabled
    through include_paper_retrieval_records=True if you need them for debugging.
    """
    print("\nPaper retrieval overview:")
    print(
        "  attempted:",
        paper_retrieval_summary.get("num_papers_attempted"),
        "| with chunks:",
        paper_retrieval_summary.get("num_papers_with_chunks"),
        "| skipped:",
        paper_retrieval_summary.get("num_skipped"),
    )
    print(
        "  full text:",
        paper_retrieval_summary.get("num_full_text_success"),
        "| fallback:",
        paper_retrieval_summary.get("num_fallback_used"),
    )
    print(
        "  source types used:",
        paper_retrieval_summary.get("used_source_type_counts", {}),
    )
    print(
        "  full-text source types:",
        paper_retrieval_summary.get("full_text_source_type_counts", {}),
    )
    print(
        "  fallback source types:",
        paper_retrieval_summary.get("fallback_source_type_counts", {}),
    )


def print_paper_reranking_summary(
    paper_reranking_summary: dict[str, Any],
) -> None:
    """
    Print a compact overview of the paper-level reranking step.
    """
    if not paper_reranking_summary.get("paper_reranking_enabled"):
        print("\nPaper reranking disabled.")
        return

    print("\nPaper reranking overview:")
    print(
        "  scored:",
        paper_reranking_summary.get("num_papers_scored"),
        "| selected:",
        paper_reranking_summary.get("num_papers_selected"),
        "| skipped before reranking:",
        paper_reranking_summary.get("num_papers_skipped_before_reranking"),
    )
    print(
        "  selected text types:",
        paper_reranking_summary.get("selected_reranking_text_type_counts", {}),
    )

    top_selected = paper_reranking_summary.get("top_selected_papers", [])
    if top_selected:
        print("  top selected papers:")
        for paper in top_selected[:5]:
            title = paper.get("title") or "Untitled"
            year = paper.get("year") or "n.d."
            score = paper.get("score")
            score_text = f"{score:.4f}" if isinstance(score, float) else str(score)
            print(f"    {paper.get('rank')}. {title} ({year}) | score={score_text}")


def run(
    query: str,
    max_queries: int = 4,
    max_paper_num_per_query: int = 8,
    max_papers: int = 8,
    top_k_per_paper: int = 8,
    final_top_n_chunks: int = 8,
    chunk_sz: int = 250,
    min_chunk_sz: int = 80,
    use_metadata_fallback: bool = True,
    save_results: bool = True,
    abstract_only: bool = False,
    results_output_path: str | Path = DEFAULT_RESULTS_PATH,
    include_paper_retrieval_records: bool = False,
    retrieval_verbose: bool = False,
    print_debug_tables: bool = False,
    rerank_papers: bool = False,
    include_paper_reranking_records: bool = False,
    paper_reranker_model_name: str = DEFAULT_PAPER_RERANKER_MODEL,
    paper_reranker_text_max_chars: int | None = 4000,
) -> dict[str, Any]:
    """
    Main pipeline:
    1. Search papers with Semantic Scholar.
    2. Rerank all retrieved papers by title/abstract/TLDR/metadata.
    3. Keep the top max_papers papers for full text or metadata fallback retrieval.
    4. Retrieve top chunks per paper using embeddings.
    5. Save paper-level retrieval records in the result.
    6. Rerank chunks globally.
    7. Extract query-relevant equipment.
    8. Optionally save proposed equipment for the run.
    """
    pipeline_config = {
        "max_queries": max_queries,
        "max_paper_num_per_query": max_paper_num_per_query,
        "max_papers": max_papers,
        "top_k_per_paper": top_k_per_paper,
        "final_top_n_chunks": final_top_n_chunks,
        "chunk_sz": chunk_sz,
        "min_chunk_sz": min_chunk_sz,
        "use_metadata_fallback": use_metadata_fallback,
        "abstract_only": abstract_only,
        "include_paper_retrieval_records": include_paper_retrieval_records,
        "retrieval_verbose": retrieval_verbose,
        "print_debug_tables": print_debug_tables,
        "rerank_papers": rerank_papers,
        "include_paper_reranking_records": include_paper_reranking_records,
        "paper_reranker_model_name": paper_reranker_model_name,
        "paper_reranker_text_max_chars": paper_reranker_text_max_chars,
    }

    search_result = search_papers_for_question(
        question=query,
        max_queries=max_queries,
        max_paper_num_per_query=max_paper_num_per_query,
    )

    print("\nGenerated Semantic Scholar queries:")
    for q in search_result["generated_queries"]:
        print("-", q)

    all_candidate_papers = search_result["papers"]

    if rerank_papers:
        papers, paper_reranking_records = rerank_papers_by_metadata(
            query=query,
            papers=all_candidate_papers,
            top_n=max_papers,
            model_name=paper_reranker_model_name,
            text_max_chars=paper_reranker_text_max_chars,
        )
        paper_reranking_summary = summarize_paper_reranking_records(
            paper_reranking_records
        )
    else:
        papers = all_candidate_papers
        paper_reranking_records = []
        paper_reranking_summary = {
            "paper_reranking_enabled": False,
            "num_papers_scored": 0,
            "num_papers_selected": min(len(papers), max_papers),
            "num_papers_skipped_before_reranking": 0,
            "top_selected_papers": [],
        }

    print_paper_reranking_summary(paper_reranking_summary)

    candidate_chunks, paper_retrieval_records = retrieve_candidate_chunks_from_papers(
        query=query,
        papers=papers,
        max_papers=max_papers,
        top_k_per_paper=top_k_per_paper,
        chunk_sz=chunk_sz,
        min_chunk_sz=min_chunk_sz,
        keep_last=True,
        use_metadata_fallback=use_metadata_fallback,
        abstract_only=abstract_only,
        return_paper_retrieval_records=True,
        verbose=retrieval_verbose,
    )

    paper_retrieval_summary = summarize_paper_retrieval_records(
        paper_retrieval_records
    )

    print_paper_retrieval_summary(
        paper_retrieval_summary=paper_retrieval_summary,
    )

    if candidate_chunks.empty:
        result = {
            "query": query,
            "status": "no_candidate_chunks",
            "generated_queries": search_result["generated_queries"],
            "query_relevant_equipment": [],
            "num_candidate_papers_before_paper_reranking": len(all_candidate_papers),
            "num_candidate_papers": len(papers),
            "num_candidate_chunks": 0,
            "paper_reranking_summary": paper_reranking_summary,
            "paper_retrieval_summary": paper_retrieval_summary,
        }

        if include_paper_reranking_records:
            result["paper_reranking_records"] = paper_reranking_records

        if include_paper_retrieval_records:
            result["paper_retrieval_records"] = paper_retrieval_records

        if save_results:
            saved_path = save_run_record(
                result=result,
                pipeline_config=pipeline_config,
                output_path=results_output_path,
            )
            result["saved_run_record_path"] = str(saved_path)
            print(f"\nSaved run record to: {saved_path}")

        return result

    if print_debug_tables:
        print("\nCombined embedding-retrieved chunks across papers:")
        print(
            candidate_chunks[
                ["paper_title", "chunk_id", "similarity", "source_type"]
            ].head(20).to_string(index=False)
        )

    reranked_chunks = rerank_top_k_chunks(
        query=query,
        retrieved_chunks_df=candidate_chunks,
        text_column="chunk_text",
    )

    if print_debug_tables:
        print("\nTop globally reranked chunks:")
        print(
            reranked_chunks[
                ["paper_title", "chunk_id", "similarity", "reranker_score", "source_type"]
            ].head(15).to_string(index=False)
        )

    extracted_result = extract_equipment_from_reranked_chunks(
        query=query,
        reranked_chunks_df=reranked_chunks,
        source_label="main_pipeline_semantic_scholar_search",
        top_n_chunks=final_top_n_chunks,
        text_column="chunk_text",
        model="gpt-5.4-mini",
    )

    extracted_result["generated_queries"] = search_result["generated_queries"]
    extracted_result["num_candidate_papers_before_paper_reranking"] = len(all_candidate_papers)
    extracted_result["num_candidate_papers"] = len(papers)
    extracted_result["num_candidate_chunks"] = len(candidate_chunks)
    extracted_result["paper_reranking_summary"] = paper_reranking_summary
    extracted_result["paper_retrieval_summary"] = paper_retrieval_summary

    if include_paper_reranking_records:
        extracted_result["paper_reranking_records"] = paper_reranking_records

    if include_paper_retrieval_records:
        extracted_result["paper_retrieval_records"] = paper_retrieval_records

    if save_results:
        saved_path = save_run_record(
            result=extracted_result,
            pipeline_config=pipeline_config,
            output_path=results_output_path,
        )
        extracted_result["saved_run_record_path"] = str(saved_path)
        print(f"\nSaved run record to: {saved_path}")

    return extracted_result


if __name__ == "__main__":
    query = (
        "I'm developing a clear food-packaging film, but the barrier performance "
        "and haze are worse than expected. Why might this be happening, and what "
        "equipment could help measure whether the polymer structure and "
        "crystallization state are causing the problem?"
    )

    result = run(
        query=query,
        max_queries=4,
        max_paper_num_per_query=15,
        max_papers=80,
        top_k_per_paper=2,
        final_top_n_chunks=20,
        chunk_sz=250,
        min_chunk_sz=80,
        use_metadata_fallback=True,
        save_results=True,
        abstract_only=True,
        results_output_path=DEFAULT_RESULTS_PATH,
        include_paper_retrieval_records=False,
        retrieval_verbose=False,
        print_debug_tables=False,
        rerank_papers=True,
        include_paper_reranking_records=False,
    )

    print("\nQuery-relevant equipment result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
