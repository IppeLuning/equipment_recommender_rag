from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from equipment_recommender_rag.equipment_inventory.extract_equipment_from_text import (
    extract_equipment_from_reranked_chunks,
)
from equipment_recommender_rag.literature_search.retrieve_candidate_chunks import (
    retrieve_candidate_chunks_from_papers,
)
from equipment_recommender_rag.literature_search.semantic_scholar import (
    search_papers_for_question,
)
from equipment_recommender_rag.reranking.rerank_chunks import (
    rerank_top_k_chunks,
)
from equipment_recommender_rag.utils.save_pipeline_results import (
    DEFAULT_RESULTS_PATH,
    save_run_record,
)


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
) -> dict[str, Any]:
    """
    Main pipeline:
    1. Search papers with Semantic Scholar.
    2. Retrieve full text or use Semantic Scholar metadata fallback.
    3. Retrieve top chunks per paper using embeddings.
    4. Rerank chunks globally.
    5. Extract query-relevant equipment.
    6. Optionally save proposed equipment for the run.
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
    }

    search_result = search_papers_for_question(
        question=query,
        max_queries=max_queries,
        max_paper_num_per_query=max_paper_num_per_query,
    )

    print("\nGenerated Semantic Scholar queries:")
    for q in search_result["generated_queries"]:
        print("-", q)

    papers = search_result["papers"]

    candidate_chunks = retrieve_candidate_chunks_from_papers(
        query=query,
        papers=papers,
        max_papers=max_papers,
        top_k_per_paper=top_k_per_paper,
        chunk_sz=chunk_sz,
        min_chunk_sz=min_chunk_sz,
        keep_last=True,
        use_metadata_fallback=use_metadata_fallback,
        abstract_only=abstract_only,
    )

    if candidate_chunks.empty:
        result = {
            "query": query,
            "status": "no_candidate_chunks",
            "generated_queries": search_result["generated_queries"],
            "query_relevant_equipment": [],
            "num_candidate_papers": len(papers),
            "num_candidate_chunks": 0,
        }

        if save_results:
            saved_path = save_run_record(
                result=result,
                pipeline_config=pipeline_config,
                output_path=results_output_path,
            )
            result["saved_run_record_path"] = str(saved_path)
            print(f"\nSaved run record to: {saved_path}")

        return result

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
    extracted_result["num_candidate_papers"] = len(papers)
    extracted_result["num_candidate_chunks"] = len(candidate_chunks)

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
        "I work at Polyvation. We are developing a new polymer, which is a variation on PEEK for medical grade purposes. The material turns out to be way softer than anticipated. Which researchers at the RUG could help me understand why our experimental material turns out so soft? Which equipment could be used for supportive analysis? I am mainly interestested in the techniques Calorimetry, Rheology, Spectroscopy and Microscopy."
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
    )

    print("\nQuery-relevant equipment result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))