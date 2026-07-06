from __future__ import annotations

import argparse
import threading
import hashlib
import inspect
import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from equipment_recommender_rag import main_pipeline
from equipment_recommender_rag.problem_decomposition.subproblem_generation import (
    generate_subproblems,
)


DEFAULT_OUTPUT_PATH = Path("data/processed/decomposed_pipeline_runs.jsonl")


COMPLETED_STATUS = "completed"
FAILED_STATUS = "failed"
INTERRUPTED_STATUS = "interrupted"


def utc_now_iso() -> str:
    """
    Return an ISO timestamp in UTC for run bookkeeping.
    """
    return datetime.now(timezone.utc).isoformat()


def _is_missing_scalar(value: Any) -> bool:
    """
    Treat None and pandas/numpy missing scalar values as missing.
    """
    if value is None:
        return True

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False

    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _optional_text(value: Any) -> str | None:
    """
    Convert a scalar value to clean text, returning None for missing/empty values.
    """
    if _is_missing_scalar(value):
        return None

    text = str(value).strip()
    return text or None


def make_query_run_id(query_record: dict[str, Any]) -> str:
    """
    Create a stable ID for resume/checkpointing.

    The ID deliberately combines query_id with source metadata and the query text.
    This avoids accidentally skipping a benchmark item when two different papers
    happen to use the same local query_id.
    """
    identity_payload = {
        "query_id": _optional_text(query_record.get("query_id")),
        "query": _optional_text(query_record.get("query")) or "",
        "source_pdf_path": _optional_text(query_record.get("source_pdf_path")),
        "source_doi": _optional_text(query_record.get("source_doi")),
    }

    identity_text = json.dumps(
        identity_payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()[:16]

    query_id = _optional_text(query_record.get("query_id"))
    if query_id:
        return f"query_id:{query_id}:sha256:{digest}"

    return f"sha256:{digest}"


def add_input_metadata(record: dict[str, Any], query_record: dict[str, Any]) -> None:
    """
    Add the original benchmark/input fields to a result or failure record.
    """
    record["input_run_id"] = make_query_run_id(query_record)
    record["input_query_id"] = query_record.get("query_id")
    record["input_source_pdf_path"] = query_record.get("source_pdf_path")
    record["input_source_doi"] = query_record.get("source_doi")
    record["input_source_is_review_paper"] = query_record.get("source_is_review_paper")
    record["input_raw_benchmark_item"] = query_record.get("raw_benchmark_item", {})


def record_run_id(record: dict[str, Any]) -> str | None:
    """
    Recover a run ID from a current or older output record.
    """
    run_id = _optional_text(record.get("input_run_id"))
    if run_id:
        return run_id

    original_query = _optional_text(record.get("original_query"))
    if not original_query:
        return None

    query_record = {
        "query": original_query,
        "query_id": record.get("input_query_id"),
        "source_pdf_path": record.get("input_source_pdf_path"),
        "source_doi": record.get("input_source_doi"),
    }
    return make_query_run_id(query_record)


def record_status(record: dict[str, Any]) -> str:
    """
    Return the status of a record, with backwards compatibility for old outputs.
    """
    status = _optional_text(record.get("run_status"))
    if status:
        return status

    if record.get("error"):
        return FAILED_STATUS

    return COMPLETED_STATUS


def remove_records_for_run_id(
    records: list[dict[str, Any]],
    run_id: str,
) -> list[dict[str, Any]]:
    """
    Remove old records for the same input item.

    This keeps checkpoint files clean when a previously failed/interrupted query
    is retried and succeeds later. It also prevents stale failures from remaining
    beside a newer completed result.
    """
    return [
        record
        for record in records
        if record_run_id(record) != run_id
    ]


def rebuild_flat_rows_from_completed_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Rebuild flattened CSV rows from completed records only.

    This is safer than appending incrementally when records can be replaced.
    """
    rows: list[dict[str, Any]] = []

    for record in records:
        if record_status(record) == COMPLETED_STATUS:
            rows.extend(_record_to_flat_rows(record))

    return rows


def load_existing_records(output_path: str | Path) -> list[dict[str, Any]]:
    """
    Load previous output records so the run can resume instead of starting over.
    """
    output_path = Path(output_path)

    if not output_path.exists():
        return []

    if output_path.stat().st_size == 0:
        return []

    try:
        if output_path.suffix.lower() == ".jsonl":
            records: list[dict[str, Any]] = []
            with output_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if isinstance(data, dict):
                        records.append(data)
            return records

        with output_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return [record for record in data if isinstance(record, dict)]

        if isinstance(data, dict):
            return [data]

    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"Warning: could not load existing output file {output_path}: {exc}. "
            "Starting with an empty in-memory record list."
        )

    return []


def make_failure_record(
    query_record: dict[str, Any],
    exc: BaseException,
    started_at_utc: str,
    status: str = FAILED_STATUS,
) -> dict[str, Any]:
    """
    Store enough information to inspect and optionally retry a failed query later.
    """
    failed_at_utc = utc_now_iso()
    record: dict[str, Any] = {
        "run_timestamp_utc": failed_at_utc,
        "run_status": status,
        "started_at_utc": started_at_utc,
        "failed_at_utc": failed_at_utc,
        "original_query": query_record.get("query"),
        "decomposition": None,
        "used_decomposition": None,
        "num_subproblems_run": 0,
        "subproblem_results": [],
        "aggregated_equipment": [],
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        },
    }
    add_input_metadata(record, query_record)
    return record

RELEVANCE_SCORE = {
    "best_match": 3,
    "acceptable_alternative": 2,
    "supporting_but_not_ideal": 1,
    "clearly_not_suitable": 0,
}


def normalize_equipment_name(name: str | None) -> str:
    """
    Normalize equipment names for simple duplicate merging.
    """
    if not name:
        return ""

    normalized = name.lower()
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace("–", " ")
    normalized = normalized.replace("—", " ")
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^a-z0-9µμ\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _deduplicate_strings(values: list[Any]) -> list[str]:
    """
    Deduplicate string-like values while preserving order.
    """
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        if value is None:
            continue

        text = str(value).strip()
        if not text:
            continue

        key = text.lower()
        if key in seen:
            continue

        seen.add(key)
        output.append(text)

    return output


def _deduplicate_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate supporting papers by DOI if available, otherwise by title.
    """
    seen: set[str] = set()
    output: list[dict[str, Any]] = []

    for paper in papers:
        doi = paper.get("doi")
        title = paper.get("paper_title")
        key = str(doi or title or "").lower().strip()

        if not key or key in seen:
            continue

        seen.add(key)
        output.append(
            {
                "paper_title": title,
                "doi": doi,
            }
        )

    return output


def _equipment_relevance_score(equipment: dict[str, Any]) -> int:
    label = equipment.get("relevance_label")
    return RELEVANCE_SCORE.get(str(label), -1)


def aggregate_equipment_across_subproblems(
    subproblem_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge equipment recommendations from all subproblem pipeline runs.

    Ranking priority:
    1. highest relevance label
    2. highest confidence score
    3. number of supporting subproblems
    4. number of supporting papers
    """
    grouped: dict[str, dict[str, Any]] = {}

    for subproblem_result in subproblem_results:
        subproblem_id = subproblem_result.get("subproblem_id")
        subproblem_description = subproblem_result.get("subproblem_description")
        result = subproblem_result.get("pipeline_result", {})

        for equipment in result.get("query_relevant_equipment", []):
            equipment_name = equipment.get("equipment_name")
            normalized_name = normalize_equipment_name(equipment_name)

            if not normalized_name:
                continue

            confidence = equipment.get("confidence_score")
            try:
                confidence_float = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence_float = None

            relevance_label = equipment.get("relevance_label")
            relevance_score = _equipment_relevance_score(equipment)

            if normalized_name not in grouped:
                grouped[normalized_name] = {
                    "equipment_name": equipment_name,
                    "normalized_equipment_name": normalized_name,
                    "aliases": [],
                    "equipment_type": equipment.get("equipment_type"),
                    "best_relevance_label": relevance_label,
                    "best_relevance_score": relevance_score,
                    "max_confidence_score": confidence_float,
                    "supporting_subproblem_ids": [],
                    "supporting_subproblem_descriptions": [],
                    "explanations": [],
                    "query_specific_uses": [],
                    "measurement_outputs": [],
                    "evidence_text": [],
                    "supporting_papers": [],
                    "raw_mentions": [],
                }

            record = grouped[normalized_name]

            # Keep the strongest label seen for this equipment.
            if relevance_score > record["best_relevance_score"]:
                record["best_relevance_score"] = relevance_score
                record["best_relevance_label"] = relevance_label
                record["equipment_name"] = equipment_name or record["equipment_name"]
                record["equipment_type"] = equipment.get("equipment_type") or record["equipment_type"]

            # Keep the highest confidence score seen.
            if confidence_float is not None:
                current_conf = record.get("max_confidence_score")
                if current_conf is None or confidence_float > current_conf:
                    record["max_confidence_score"] = confidence_float

            record["aliases"].extend(equipment.get("aliases", []) or [])
            record["measurement_outputs"].extend(equipment.get("measurement_outputs", []) or [])
            record["evidence_text"].extend(equipment.get("evidence_text", []) or [])
            record["supporting_papers"].extend(equipment.get("supporting_papers", []) or [])

            reason = equipment.get("reason")
            if reason:
                record["explanations"].append(reason)

            query_specific_use = equipment.get("query_specific_use")
            if query_specific_use:
                record["query_specific_uses"].append(query_specific_use)

            if subproblem_id:
                record["supporting_subproblem_ids"].append(subproblem_id)

            if subproblem_description:
                record["supporting_subproblem_descriptions"].append(subproblem_description)

            record["raw_mentions"].append(
                {
                    "subproblem_id": subproblem_id,
                    "equipment_name": equipment.get("equipment_name"),
                    "relevance_label": relevance_label,
                    "confidence_score": confidence_float,
                    "reason": reason,
                }
            )

    aggregated: list[dict[str, Any]] = []

    for record in grouped.values():
        record["aliases"] = _deduplicate_strings(record["aliases"])
        record["measurement_outputs"] = _deduplicate_strings(record["measurement_outputs"])
        record["evidence_text"] = _deduplicate_strings(record["evidence_text"])
        record["explanations"] = _deduplicate_strings(record["explanations"])
        record["query_specific_uses"] = _deduplicate_strings(record["query_specific_uses"])
        record["supporting_subproblem_ids"] = _deduplicate_strings(
            record["supporting_subproblem_ids"]
        )
        record["supporting_subproblem_descriptions"] = _deduplicate_strings(
            record["supporting_subproblem_descriptions"]
        )
        record["supporting_papers"] = _deduplicate_papers(record["supporting_papers"])
        record["num_supporting_subproblems"] = len(record["supporting_subproblem_ids"])
        record["num_supporting_papers"] = len(record["supporting_papers"])
        aggregated.append(record)

    aggregated.sort(
        key=lambda item: (
            item.get("best_relevance_score", -1),
            item.get("max_confidence_score") or 0.0,
            item.get("num_supporting_subproblems", 0),
            item.get("num_supporting_papers", 0),
        ),
        reverse=True,
    )

    return aggregated


def _call_main_pipeline(query: str, **kwargs: Any) -> dict[str, Any]:
    """
    Call main_pipeline.run while only passing keyword arguments supported by the
    current version of that function.

    This makes the root main.py less fragile while you are still changing
    main_pipeline.py during development.
    """
    signature = inspect.signature(main_pipeline.run)
    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return main_pipeline.run(query=query, **supported_kwargs)


def run_decomposed_pipeline(
    query: str,
    use_decomposition: bool = True,
    max_subproblems: int = 5,
    max_queries: int = 4,
    max_paper_num_per_query: int = 8,
    max_papers: int = 8,
    top_k_per_paper: int = 8,
    final_top_n_chunks: int = 8,
    chunk_sz: int = 250,
    min_chunk_sz: int = 80,
    use_metadata_fallback: bool = True,
    abstract_only: bool = False,
    save_subproblem_results: bool = False,
    include_paper_retrieval_records: bool = False,
    retrieval_verbose: bool = False,
    print_debug_tables: bool = False,
    rerank_papers: bool = False,
) -> dict[str, Any]:
    """
    Root orchestration:
    1. Optionally classify/decompose the original problem.
    2. Run the existing main pipeline for each subproblem.
    3. Aggregate the equipment recommendations across subproblems.
    """
    decomposition: dict[str, Any] | None = None

    if use_decomposition:
        decomposition = generate_subproblems(
            query=query,
            max_subproblems=max_subproblems,
        )
        should_decompose = bool(decomposition.get("should_decompose"))
        subproblems = decomposition.get("subproblems", [])
    else:
        should_decompose = False
        subproblems = []

    if not should_decompose or not subproblems:
        subproblems_to_run = [
            {
                "subproblem_id": "original_query",
                "subproblem_description": query,
                "analysis_goal": "Answer the original query directly.",
                "hypothesis_or_branch": "direct_query",
                "expected_method_families": [],
            }
        ]
    else:
        subproblems_to_run = subproblems

    subproblem_results: list[dict[str, Any]] = []

    for index, subproblem in enumerate(subproblems_to_run, start=1):
        subproblem_id = subproblem.get("subproblem_id") or f"subproblem_{index}"
        subproblem_query = subproblem.get("subproblem_description") or query

        print("\n" + "=" * 80)
        print(f"Running pipeline for subproblem {index}/{len(subproblems_to_run)}")
        print("Subproblem ID:", subproblem_id)
        print("Subproblem query:", subproblem_query)
        print("=" * 80)

        pipeline_result = _call_main_pipeline(
            query=subproblem_query,
            max_queries=max_queries,
            max_paper_num_per_query=max_paper_num_per_query,
            max_papers=max_papers,
            top_k_per_paper=top_k_per_paper,
            final_top_n_chunks=final_top_n_chunks,
            chunk_sz=chunk_sz,
            min_chunk_sz=min_chunk_sz,
            use_metadata_fallback=use_metadata_fallback,
            use_abstract_fallback=use_metadata_fallback,
            abstract_only=abstract_only,
            save_results=save_subproblem_results,
            include_paper_retrieval_records=include_paper_retrieval_records,
            retrieval_verbose=retrieval_verbose,
            print_debug_tables=print_debug_tables,
            rerank_papers=rerank_papers,
        )

        subproblem_results.append(
            {
                "subproblem_id": subproblem_id,
                "subproblem_description": subproblem_query,
                "analysis_goal": subproblem.get("analysis_goal"),
                "hypothesis_or_branch": subproblem.get("hypothesis_or_branch"),
                "expected_method_families": subproblem.get("expected_method_families", []),
                "pipeline_result": pipeline_result,
            }
        )

    aggregated_equipment = aggregate_equipment_across_subproblems(subproblem_results)

    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "original_query": query,
        "decomposition": decomposition,
        "used_decomposition": bool(use_decomposition and should_decompose and subproblems),
        "num_subproblems_run": len(subproblems_to_run),
        "subproblem_results": subproblem_results,
        "aggregated_equipment": aggregated_equipment,
    }


def save_json_records(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """
    Save all full run records to a readable JSON file by default.

    The file is written atomically via a temporary file and then replaced. This
    reduces the chance of corrupting the existing checkpoint if the process is
    interrupted while saving.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")

    if output_path.suffix.lower() == ".jsonl":
        with tmp_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
    else:
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2, ensure_ascii=False, default=str)
            file.write("\n")

    tmp_path.replace(output_path)
    return output_path




def _extract_problem_descriptions_from_json_object(
    data: Any,
) -> list[dict[str, Any]]:
    """
    Extract benchmark queries from a JSON object.

    Supports:
    1. Your benchmark format:
       [
         {
           "pdf_path": "...",
           "doi": "...",
           "benchmark_items": [
             {"query_id": "...", "problem_description": "..."}
           ]
         }
       ]

    2. A direct list of items:
       [
         {"query_id": "...", "problem_description": "..."}
       ]

    3. A single dict:
       {"problem_description": "..."}
    """
    query_records: list[dict[str, Any]] = []

    def add_query_record(
        problem_description: Any,
        query_id: Any = None,
        source_pdf_path: Any = None,
        source_doi: Any = None,
        source_is_review_paper: Any = None,
        raw_item: dict[str, Any] | None = None,
    ) -> None:
        if problem_description is None:
            return

        query = str(problem_description).strip()
        if not query:
            return

        query_records.append(
            {
                "query": query,
                "query_id": query_id,
                "source_pdf_path": source_pdf_path,
                "source_doi": source_doi,
                "source_is_review_paper": source_is_review_paper,
                "raw_benchmark_item": raw_item or {},
            }
        )

    if isinstance(data, list):
        for paper_or_item in data:
            if not isinstance(paper_or_item, dict):
                continue

            # Your nested paper-level benchmark format.
            benchmark_items = paper_or_item.get("benchmark_items")
            if isinstance(benchmark_items, list):
                for item in benchmark_items:
                    if not isinstance(item, dict):
                        continue

                    add_query_record(
                        problem_description=item.get("problem_description"),
                        query_id=item.get("query_id"),
                        source_pdf_path=paper_or_item.get("pdf_path"),
                        source_doi=paper_or_item.get("doi"),
                        source_is_review_paper=paper_or_item.get("is_review_paper"),
                        raw_item=item,
                    )

            # Direct list of benchmark items.
            elif "problem_description" in paper_or_item:
                add_query_record(
                    problem_description=paper_or_item.get("problem_description"),
                    query_id=paper_or_item.get("query_id"),
                    source_pdf_path=paper_or_item.get("pdf_path"),
                    source_doi=paper_or_item.get("doi"),
                    source_is_review_paper=paper_or_item.get("is_review_paper"),
                    raw_item=paper_or_item,
                )

    elif isinstance(data, dict):
        # Single benchmark item.
        if "problem_description" in data:
            add_query_record(
                problem_description=data.get("problem_description"),
                query_id=data.get("query_id"),
                source_pdf_path=data.get("pdf_path"),
                source_doi=data.get("doi"),
                source_is_review_paper=data.get("is_review_paper"),
                raw_item=data,
            )

        # Single paper-level record.
        benchmark_items = data.get("benchmark_items")
        if isinstance(benchmark_items, list):
            for item in benchmark_items:
                if not isinstance(item, dict):
                    continue

                add_query_record(
                    problem_description=item.get("problem_description"),
                    query_id=item.get("query_id"),
                    source_pdf_path=data.get("pdf_path"),
                    source_doi=data.get("doi"),
                    source_is_review_paper=data.get("is_review_paper"),
                    raw_item=item,
                )

    return query_records


def load_query_records_from_input_file(input_file: str | Path) -> list[dict[str, Any]]:
    """
    Load input queries from CSV, JSON, or JSONL.

    CSV requirements:
    - Must contain a 'problem_description' column.
    - Optional useful columns: query_id, pdf_path, doi, is_review_paper.

    JSON support:
    - Supports the nested problem_answer_pairs.json format produced by your benchmark generator.
    """
    input_path = Path(input_file)
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        input_df = pd.read_csv(input_path)

        if "problem_description" not in input_df.columns:
            raise ValueError("Input CSV must contain a 'problem_description' column.")

        records: list[dict[str, Any]] = []

        for _, row in input_df.iterrows():
            problem_description = row.get("problem_description")
            if pd.isna(problem_description):
                continue

            records.append(
                {
                    "query": str(problem_description),
                    "query_id": row.get("query_id") if "query_id" in row else None,
                    "source_pdf_path": row.get("pdf_path") if "pdf_path" in row else None,
                    "source_doi": row.get("doi") if "doi" in row else None,
                    "source_is_review_paper": (
                        row.get("is_review_paper")
                        if "is_review_paper" in row
                        else None
                    ),
                    "raw_benchmark_item": row.to_dict(),
                }
            )

        return records

    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        records = _extract_problem_descriptions_from_json_object(data)

        if not records:
            raise ValueError(
                "No problem descriptions found in JSON file. Expected either "
                "'problem_description' fields or nested 'benchmark_items'."
            )

        return records

    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []

        with input_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                records.extend(_extract_problem_descriptions_from_json_object(data))

        if not records:
            raise ValueError(
                "No problem descriptions found in JSONL file. Expected either "
                "'problem_description' fields or nested 'benchmark_items'."
            )

        return records

    raise ValueError(
        f"Unsupported input file type: {suffix}. Use .csv, .json, or .jsonl."
    )


def _record_to_flat_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten aggregated equipment recommendations into rows for CSV output.
    """
    rows: list[dict[str, Any]] = []

    for rank, equipment in enumerate(record.get("aggregated_equipment", []), start=1):
        rows.append(
            {
                "input_query_id": record.get("input_query_id"),
                "input_source_pdf_path": record.get("input_source_pdf_path"),
                "input_source_doi": record.get("input_source_doi"),
                "input_source_is_review_paper": record.get("input_source_is_review_paper"),
                "original_query": record.get("original_query"),
                "used_decomposition": record.get("used_decomposition"),
                "num_subproblems_run": record.get("num_subproblems_run"),
                "rank": rank,
                "equipment_name": equipment.get("equipment_name"),
                "equipment_type": equipment.get("equipment_type"),
                "best_relevance_label": equipment.get("best_relevance_label"),
                "max_confidence_score": equipment.get("max_confidence_score"),
                "num_supporting_subproblems": equipment.get("num_supporting_subproblems"),
                "num_supporting_papers": equipment.get("num_supporting_papers"),
                "supporting_subproblem_ids": " | ".join(
                    equipment.get("supporting_subproblem_ids", [])
                ),
                "aliases": " | ".join(equipment.get("aliases", [])),
                "measurement_outputs": " | ".join(
                    equipment.get("measurement_outputs", [])
                ),
                "explanations": " | ".join(equipment.get("explanations", [])),
                "supporting_papers": " | ".join(
                    [
                        f"{paper.get('paper_title')} ({paper.get('doi')})"
                        for paper in equipment.get("supporting_papers", [])
                    ]
                ),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the equipment recommender with optional subproblem decomposition."
        )
    )

    parser.add_argument("--query", type=str, help="Single problem description to run.")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Optional CSV, JSON, or JSONL file with problem descriptions.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=(
            "Output file for full records. Defaults to JSONL, which is safer "
            "for resumable multi-worker runs."
        ),
    )
    parser.add_argument(
        "--csv_output_file",
        type=str,
        default=None,
        help="Optional CSV output with flattened aggregated equipment rows.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore existing output_file contents and start a fresh run.",
    )
    parser.add_argument(
        "--skip_failed",
        action="store_true",
        help=(
            "When resuming, skip records that previously failed or were interrupted. "
            "By default, completed records are skipped and failed records are retried."
        ),
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help=(
            "Stop the whole run after saving a failure record. By default, the script "
            "logs the failure and continues with the next query."
        ),
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help=(
            "Number of original benchmark queries to run in parallel. Start with 2 "
            "on a MacBook Air. Semantic Scholar calls are still globally throttled "
            "inside semantic_scholar.py."
        ),
    )

    parser.add_argument("--max_subproblems", type=int, default=5)
    parser.add_argument("--max_queries", type=int, default=4)
    parser.add_argument("--max_paper_num_per_query", type=int, default=8)
    parser.add_argument("--max_papers", type=int, default=8)
    parser.add_argument("--top_k_per_paper", type=int, default=8)
    parser.add_argument("--final_top_n_chunks", type=int, default=8)
    parser.add_argument("--chunk_sz", type=int, default=250)
    parser.add_argument("--min_chunk_sz", type=int, default=80)

    parser.add_argument(
        "--no_decomposition",
        action="store_true",
        help="Skip subproblem generation and run the original query directly.",
    )
    parser.add_argument(
        "--no_metadata_fallback",
        action="store_true",
        help="Disable metadata fallback in the underlying pipeline where supported.",
    )
    parser.add_argument(
        "--abstract_only",
        action="store_true",
        help="Use abstract/metadata evidence only in the underlying pipeline where supported.",
    )
    parser.add_argument(
        "--save_subproblem_results",
        action="store_true",
        help="Let main_pipeline save each subproblem run separately as well.",
    )
    parser.add_argument(
        "--include_paper_retrieval_records",
        action="store_true",
        help=(
            "Include the full per-paper retrieval records in the JSON output. "
            "By default only the compact paper_retrieval_summary is saved."
        ),
    )
    parser.add_argument(
        "--retrieval_verbose",
        action="store_true",
        help="Print one retrieval status line per paper. Off by default to keep output compact.",
    )
    parser.add_argument(
        "--print_debug_tables",
        action="store_true",
        help="Print candidate chunk and reranked chunk debug tables. Off by default.",
    )

    parser.add_argument(
        "--no_paper_reranking",
        action="store_true",
        help="Disable paper-level reranking after Semantic Scholar search.",
    )

    args = parser.parse_args()

    if not args.query and not args.input_file:
        parser.error("Provide either --query or --input_file.")

    query_records: list[dict[str, Any]] = []

    if args.query:
        query_records.append(
            {
                "query": args.query,
                "query_id": None,
                "source_pdf_path": None,
                "source_doi": None,
                "source_is_review_paper": None,
                "raw_benchmark_item": {},
            }
        )

    if args.input_file:
        query_records.extend(load_query_records_from_input_file(args.input_file))

    if args.no_resume:
        full_records: list[dict[str, Any]] = []
        print("Resume disabled: starting with an empty output record list.")
    else:
        full_records = load_existing_records(args.output_file)
        if full_records:
            completed_count = sum(
                1 for record in full_records if record_status(record) == COMPLETED_STATUS
            )
            failed_count = sum(
                1 for record in full_records if record_status(record) == FAILED_STATUS
            )
            interrupted_count = sum(
                1 for record in full_records if record_status(record) == INTERRUPTED_STATUS
            )
            print(
                f"Loaded {len(full_records)} existing output records from {args.output_file} "
                f"({completed_count} completed, {failed_count} failed, "
                f"{interrupted_count} interrupted)."
            )

    completed_run_ids = {
        run_id
        for record in full_records
        if record_status(record) == COMPLETED_STATUS
        for run_id in [record_run_id(record)]
        if run_id
    }
    failed_or_interrupted_run_ids = {
        run_id
        for record in full_records
        if record_status(record) in {FAILED_STATUS, INTERRUPTED_STATUS}
        for run_id in [record_run_id(record)]
        if run_id
    }

    flat_rows: list[dict[str, Any]] = []
    for existing_record in full_records:
        if record_status(existing_record) == COMPLETED_STATUS:
            flat_rows.extend(_record_to_flat_rows(existing_record))

    skipped_completed = 0
    skipped_failed = 0
    records_to_run: list[tuple[int, dict[str, Any], str]] = []

    for query_index, query_record in enumerate(query_records, start=1):
        run_id = make_query_run_id(query_record)

        if run_id in completed_run_ids:
            skipped_completed += 1
            print("\n" + "-" * 100)
            print(f"Skipping already completed query {query_index}/{len(query_records)}")
            if query_record.get("query_id"):
                print("Query ID:", query_record.get("query_id"))
            if query_record.get("source_pdf_path"):
                print("Source PDF:", query_record.get("source_pdf_path"))
            print("Run ID:", run_id)
            print("-" * 100)
            continue

        if args.skip_failed and run_id in failed_or_interrupted_run_ids:
            skipped_failed += 1
            print("\n" + "-" * 100)
            print(f"Skipping previously failed/interrupted query {query_index}/{len(query_records)}")
            if query_record.get("query_id"):
                print("Query ID:", query_record.get("query_id"))
            if query_record.get("source_pdf_path"):
                print("Source PDF:", query_record.get("source_pdf_path"))
            print("Run ID:", run_id)
            print("-" * 100)
            continue

        records_to_run.append((query_index, query_record, run_id))

    max_workers = max(1, args.max_workers)
    if max_workers > 1 and len(records_to_run) > 1:
        print(
            f"\nRunning {len(records_to_run)} query records with max_workers={max_workers}. "
            "Semantic Scholar requests are throttled globally in semantic_scholar.py."
        )

    output_lock = threading.Lock()

    def execute_query_record(
        query_index: int,
        query_record: dict[str, Any],
        run_id: str,
    ) -> tuple[dict[str, Any], BaseException | None]:
        query = query_record["query"]

        print("\n" + "#" * 100)
        print(f"Running original query {query_index}/{len(query_records)}")
        if query_record.get("query_id"):
            print("Query ID:", query_record.get("query_id"))
        if query_record.get("source_pdf_path"):
            print("Source PDF:", query_record.get("source_pdf_path"))
        print("Run ID:", run_id)
        print(query)
        print("#" * 100)

        started_at_utc = utc_now_iso()

        try:
            record = run_decomposed_pipeline(
                rerank_papers=not args.no_paper_reranking,
                query=query,
                use_decomposition=not args.no_decomposition,
                max_subproblems=args.max_subproblems,
                max_queries=args.max_queries,
                max_paper_num_per_query=args.max_paper_num_per_query,
                max_papers=args.max_papers,
                top_k_per_paper=args.top_k_per_paper,
                final_top_n_chunks=args.final_top_n_chunks,
                chunk_sz=args.chunk_sz,
                min_chunk_sz=args.min_chunk_sz,
                use_metadata_fallback=not args.no_metadata_fallback,
                abstract_only=args.abstract_only,
                save_subproblem_results=args.save_subproblem_results,
                include_paper_retrieval_records=args.include_paper_retrieval_records,
                retrieval_verbose=args.retrieval_verbose,
                print_debug_tables=args.print_debug_tables,
            )

            record["run_status"] = COMPLETED_STATUS
            record["started_at_utc"] = started_at_utc
            record["completed_at_utc"] = utc_now_iso()
            add_input_metadata(record, query_record)
            return record, None

        except KeyboardInterrupt as exc:
            failure_record = make_failure_record(
                query_record=query_record,
                exc=exc,
                started_at_utc=started_at_utc,
                status=INTERRUPTED_STATUS,
            )
            return failure_record, exc

        except Exception as exc:
            failure_record = make_failure_record(
                query_record=query_record,
                exc=exc,
                started_at_utc=started_at_utc,
                status=FAILED_STATUS,
            )
            return failure_record, exc

    def store_record(record: dict[str, Any], run_id: str) -> Path:
        nonlocal full_records, flat_rows

        with output_lock:
            full_records = remove_records_for_run_id(full_records, run_id)
            full_records.append(record)

            if record_status(record) == COMPLETED_STATUS:
                completed_run_ids.add(run_id)
                failed_or_interrupted_run_ids.discard(run_id)
            else:
                completed_run_ids.discard(run_id)
                failed_or_interrupted_run_ids.add(run_id)

            flat_rows = rebuild_flat_rows_from_completed_records(full_records)
            return save_json_records(full_records, args.output_file)

    def report_saved_result(
        record: dict[str, Any],
        exc: BaseException | None,
        saved_path: Path,
    ) -> None:
        status = record_status(record)

        if status == COMPLETED_STATUS:
            print(f"\nSaved checkpoint JSONL output to: {saved_path}")
            return

        if status == INTERRUPTED_STATUS:
            print(f"\nInterrupted. Saved checkpoint JSONL output to: {saved_path}")
            return

        print(f"\nQuery failed but progress was saved to: {saved_path}")
        if exc is not None:
            print(f"Error type: {type(exc).__name__}")
            print(f"Error message: {exc}")
        print("Continuing with the next query...")

    if max_workers == 1 or len(records_to_run) <= 1:
        for query_index, query_record, run_id in records_to_run:
            record, exc = execute_query_record(query_index, query_record, run_id)
            saved_path = store_record(record, run_id)
            report_saved_result(record, exc, saved_path)

            if record_status(record) == INTERRUPTED_STATUS and exc is not None:
                raise exc

            if args.stop_on_error and record_status(record) == FAILED_STATUS:
                if exc is not None:
                    raise exc
                raise RuntimeError(f"Query failed: {run_id}")
    else:
        future_to_item = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for item in records_to_run:
                future = executor.submit(execute_query_record, *item)
                future_to_item[future] = item

            for future in as_completed(future_to_item):
                query_index, query_record, run_id = future_to_item[future]

                try:
                    record, exc = future.result()
                except BaseException as exc:
                    # This should be rare because execute_query_record catches normal
                    # failures, but keep it so every item still checkpoints.
                    record = make_failure_record(
                        query_record=query_record,
                        exc=exc,
                        started_at_utc=utc_now_iso(),
                        status=(
                            INTERRUPTED_STATUS
                            if isinstance(exc, KeyboardInterrupt)
                            else FAILED_STATUS
                        ),
                    )

                saved_path = store_record(record, run_id)
                report_saved_result(record, exc, saved_path)

                if record_status(record) == INTERRUPTED_STATUS and exc is not None:
                    for pending_future in future_to_item:
                        pending_future.cancel()
                    raise exc

                if args.stop_on_error and record_status(record) == FAILED_STATUS:
                    for pending_future in future_to_item:
                        pending_future.cancel()
                    if exc is not None:
                        raise exc
                    raise RuntimeError(f"Query failed: {run_id}")

    if args.csv_output_file:
        csv_path = Path(args.csv_output_file)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
        print(f"Saved flattened CSV output to: {csv_path}")

    print("\nRun summary:")
    print(f"  Completed records in output: {len(completed_run_ids)}")
    print(f"  Failed/interrupted records in output: {len(failed_or_interrupted_run_ids)}")
    print(f"  Skipped already completed this run: {skipped_completed}")
    print(f"  Skipped failed/interrupted this run: {skipped_failed}")

    if len(query_records) == 1:
        requested_run_id = make_query_run_id(query_records[0])
        matching_completed_records = [
            record
            for record in full_records
            if record_status(record) == COMPLETED_STATUS
            and record_run_id(record) == requested_run_id
        ]

        if matching_completed_records:
            print("\nFinal aggregated equipment result:")
            print(
                json.dumps(
                    matching_completed_records[-1]["aggregated_equipment"],
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )


if __name__ == "__main__":
    main()