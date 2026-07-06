#!/usr/bin/env python3
"""
Create an LLM-only baseline output file from an existing RAG output JSONL file.

The script reads a RAG output JSONL file, takes only the original user query from
that file, asks an LLM to recommend equipment from general scientific knowledge,
and writes a new JSONL file with the same high-level structure as the RAG output.

Important for avoiding benchmark leakage:
- The prompt uses only `original_query`.
- Fields such as `input_raw_benchmark_item`, expected answers, source DOI, and
  source PDF path are copied only as metadata into the output record.
- They are never included in the LLM prompt.

Example:
    uv run python create_llm_only_benchmark_from_rag_output.py \
      --input_file data/processed/test_problem_answer_output.jsonl \
      --output_file data/processed/test_problem_answer_output_llm_only.jsonl \
      --model gpt-5.4-mini \
      --max_items 5

For a quick test:
    uv run python create_llm_only_benchmark_from_rag_output.py \
      --input_file data/processed/test_problem_answer_output.jsonl \
      --output_file data/processed/test_problem_answer_output_llm_only_sample.jsonl \
      --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RELEVANCE_TO_SCORE = {
    "best_match": 3,
    "acceptable_alternative": 2,
    "supporting_but_not_ideal": 1,
    "clearly_not_suitable": 0,
}

ALLOWED_RELEVANCE_LABELS = set(RELEVANCE_TO_SCORE)
ALLOWED_CERTAINTY_LABELS = {"explicit", "strongly_implied"}


_client = None


def get_openai_client():
    """Create the OpenAI client lazily so utility functions can be imported without API deps."""
    global _client
    if _client is not None:
        return _client
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "This script needs the `openai` and `python-dotenv` packages. "
            "Install them in your uv environment, for example: `uv add openai python-dotenv`."
        ) from exc
    load_dotenv()
    _client = OpenAI()
    return _client


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_equipment_name(name: str) -> str:
    """Small normalization function to mimic the grouping used in RAG outputs."""
    normalized = str(name or "").lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[^a-z0-9+\-/() ]", "", normalized)
    return normalized.strip()


def unique_preserve_order(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def strip_code_fences(text: str) -> str:
    """Handle occasional ```json fenced output from the model."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def build_llm_only_prompt(query: str, max_items: int) -> str:
    return f"""
You are a careful scientific equipment recommendation system.

Task:
Given a user query, identify physical specialized equipment that could help answer or solve the query using only general scientific knowledge.

Important:
- This is an LLM-only baseline. Do NOT rely on retrieved papers, local inventory, source papers, citations, or benchmark answers.
- Recommend only PHYSICAL specialized equipment.
- Do NOT include software, simulation codes, computational models, scripts, code stacks, or generic lab supplies.
- Focus on physical instruments, analyzers, microscopes, spectrometers, detectors, lasers, chromatography systems, imaging systems, beamlines, reactors, testing systems, and other tangible scientific equipment.
- Return at most {max_items} equipment items.
- Rank the equipment from most to least useful for the query.
- Do not include clearly unsuitable equipment unless the query specifically asks for exclusions or comparisons.
- Because this is not evidence-grounded RAG, leave evidence_text, source_chunk_ids, and supporting_papers empty.
- Return valid JSON only. No markdown.

Suitability labels:
- "best_match": directly solves the measurement / analysis problem in the query
- "acceptable_alternative": relevant and usable, but not the clearest or best match
- "supporting_but_not_ideal": related to the setup or workflow, but does not directly answer the query
- "clearly_not_suitable": does not solve the query

Return exactly this JSON structure:
{{
  "query": {json.dumps(query)},
  "query_relevant_equipment": [
    {{
      "equipment_name": "normalized common name",
      "aliases": ["alias 1", "alias 2"],
      "manufacturer": null,
      "model": null,
      "equipment_type": "broad category",
      "relevance_label": "best_match",
      "short_description": "short practical description",
      "reason": "why this equipment is suitable for answering the query",
      "measurement_outputs": [
        "output 1",
        "output 2"
      ],
      "query_specific_use": "how it relates to the user query",
      "certainty": "strongly_implied",
      "confidence_score": 0.0,
      "evidence_text": [],
      "source_chunk_ids": [],
      "supporting_papers": []
    }}
  ]
}}

Allowed values:
- "relevance_label": "best_match", "acceptable_alternative", "supporting_but_not_ideal", "clearly_not_suitable"
- "certainty": "explicit" or "strongly_implied"

If no query-relevant physical specialized equipment can be reliably identified, return:
{{
  "query": {json.dumps(query)},
  "query_relevant_equipment": []
}}

User query:
{query}
""".strip()


def call_llm_only_equipment_recommender(
    query: str,
    model: str = "gpt-5.4-mini",
    max_items: int = 5,
    max_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
) -> dict[str, Any]:
    """Ask the LLM for general-knowledge equipment recommendations."""
    prompt = build_llm_only_prompt(query=query, max_items=max_items)

    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = get_openai_client().responses.create(
                model=model,
                input=prompt,
            )
            raw_output = strip_code_fences(response.output_text)
            parsed = json.loads(raw_output)

            if "query_relevant_equipment" not in parsed:
                return {
                    "query": query,
                    "status": "missing_query_relevant_equipment_key",
                    "query_relevant_equipment": [],
                    "raw_output": raw_output,
                }

            parsed["query"] = parsed.get("query", query)
            parsed["query_relevant_equipment"] = validate_query_relevant_equipment(
                parsed.get("query_relevant_equipment", []),
                max_items=max_items,
            )
            parsed["status"] = "query_relevant_equipment_extracted_from_general_knowledge"
            parsed["source_label"] = "llm_only_general_knowledge"
            return parsed

        except Exception as exc:  # noqa: BLE001 - preserve error in output record
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds * attempt)

    return {
        "query": query,
        "status": "failed",
        "query_relevant_equipment": [],
        "error": last_error,
    }


def validate_query_relevant_equipment(items: Any, max_items: int) -> list[dict[str, Any]]:
    """Make the model output safer and more consistent for downstream evaluation."""
    if not isinstance(items, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for raw in items[:max_items]:
        if not isinstance(raw, dict):
            continue

        equipment_name = str(raw.get("equipment_name") or "").strip()
        if not equipment_name:
            continue

        relevance_label = str(raw.get("relevance_label") or "acceptable_alternative").strip()
        if relevance_label not in ALLOWED_RELEVANCE_LABELS:
            relevance_label = "acceptable_alternative"

        certainty = str(raw.get("certainty") or "strongly_implied").strip()
        if certainty not in ALLOWED_CERTAINTY_LABELS:
            certainty = "strongly_implied"

        try:
            confidence_score = float(raw.get("confidence_score", 0.0))
        except (TypeError, ValueError):
            confidence_score = 0.0
        confidence_score = min(max(confidence_score, 0.0), 1.0)

        cleaned.append(
            {
                "equipment_name": equipment_name,
                "aliases": [str(x).strip() for x in ensure_list(raw.get("aliases")) if str(x).strip()],
                "manufacturer": raw.get("manufacturer"),
                "model": raw.get("model"),
                "equipment_type": str(raw.get("equipment_type") or "specialized scientific equipment"),
                "relevance_label": relevance_label,
                "short_description": str(raw.get("short_description") or ""),
                "reason": str(raw.get("reason") or ""),
                "measurement_outputs": [
                    str(x).strip() for x in ensure_list(raw.get("measurement_outputs")) if str(x).strip()
                ],
                "query_specific_use": str(raw.get("query_specific_use") or ""),
                "certainty": certainty,
                "confidence_score": confidence_score,
                # LLM-only baseline has no retrieved evidence.
                "evidence_text": [],
                "source_chunk_ids": [],
                "supporting_papers": [],
            }
        )

    return cleaned


def aggregate_equipment_from_single_result(
    query: str,
    query_relevant_equipment: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert the inner `query_relevant_equipment` schema to the same aggregated
    equipment schema used by the RAG output.
    """
    grouped: dict[str, dict[str, Any]] = {}

    for item in query_relevant_equipment:
        name = str(item.get("equipment_name") or "").strip()
        normalized_name = normalize_equipment_name(name)
        if not normalized_name:
            continue

        relevance_label = item.get("relevance_label", "acceptable_alternative")
        relevance_score = RELEVANCE_TO_SCORE.get(str(relevance_label), 2)
        confidence_score = float(item.get("confidence_score") or 0.0)

        if normalized_name not in grouped:
            grouped[normalized_name] = {
                "equipment_name": name,
                "normalized_equipment_name": normalized_name,
                "aliases": [],
                "equipment_type": item.get("equipment_type"),
                "best_relevance_label": relevance_label,
                "best_relevance_score": relevance_score,
                "max_confidence_score": confidence_score,
                "supporting_subproblem_ids": ["original_query"],
                "supporting_subproblem_descriptions": [query],
                "explanations": [],
                "query_specific_uses": [],
                "measurement_outputs": [],
                "evidence_text": [],
                "supporting_papers": [],
                "raw_mentions": [],
                "num_supporting_subproblems": 1,
                "num_supporting_papers": 0,
            }

        aggregate = grouped[normalized_name]
        aggregate["aliases"] = unique_preserve_order(
            aggregate["aliases"] + ensure_list(item.get("aliases"))
        )
        aggregate["explanations"] = unique_preserve_order(
            aggregate["explanations"] + [item.get("reason", "")]
        )
        aggregate["query_specific_uses"] = unique_preserve_order(
            aggregate["query_specific_uses"] + [item.get("query_specific_use", "")]
        )
        aggregate["measurement_outputs"] = unique_preserve_order(
            aggregate["measurement_outputs"] + ensure_list(item.get("measurement_outputs"))
        )
        aggregate["evidence_text"] = []
        aggregate["supporting_papers"] = []
        aggregate["raw_mentions"].append(
            {
                "subproblem_id": "original_query",
                "equipment_name": name,
                "relevance_label": relevance_label,
                "confidence_score": confidence_score,
                "reason": item.get("reason", ""),
            }
        )

        if relevance_score > int(aggregate["best_relevance_score"]):
            aggregate["best_relevance_label"] = relevance_label
            aggregate["best_relevance_score"] = relevance_score
        aggregate["max_confidence_score"] = max(
            float(aggregate["max_confidence_score"]), confidence_score
        )

    aggregated = list(grouped.values())
    aggregated.sort(
        key=lambda x: (x["best_relevance_score"], x["max_confidence_score"]),
        reverse=True,
    )
    return aggregated


def build_output_record(
    input_record: dict[str, Any],
    llm_result: dict[str, Any],
    model: str,
    max_items: int,
    started_at_utc: str,
    completed_at_utc: str,
) -> dict[str, Any]:
    """Create one output JSONL record matching the current RAG output structure."""
    query = input_record.get("original_query") or input_record.get("query") or ""
    query_relevant_equipment = llm_result.get("query_relevant_equipment", [])
    aggregated_equipment = aggregate_equipment_from_single_result(
        query=query,
        query_relevant_equipment=query_relevant_equipment,
    )

    run_status = "completed" if llm_result.get("status") != "failed" else "failed"

    output_record = {
        "run_timestamp_utc": completed_at_utc,
        "original_query": query,
        "decomposition": None,
        "used_decomposition": False,
        "num_subproblems_run": 1,
        "subproblem_results": [
            {
                "subproblem_id": "original_query",
                "subproblem_description": query,
                "analysis_goal": "Answer the original query directly using general scientific knowledge.",
                "hypothesis_or_branch": "llm_only_general_knowledge",
                "expected_method_families": [],
                "pipeline_result": llm_result,
            }
        ],
        "aggregated_equipment": aggregated_equipment,
        "run_status": run_status,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        # Extra fields that make it explicit that this is not RAG.
        "baseline_type": "llm_only_general_knowledge",
        "model": model,
        "max_items": max_items,
        "retrieval_used": False,
        "literature_evidence_used": False,
        "inventory_used": False,
        # Preserve input metadata so existing evaluation code can still group by query type,
        # source paper, review/non-review, etc. These are copied after the LLM call and are
        # not included in the prompt.
        "input_run_id": input_record.get("input_run_id"),
        "input_query_id": input_record.get("input_query_id"),
        "input_source_pdf_path": input_record.get("input_source_pdf_path"),
        "input_source_doi": input_record.get("input_source_doi"),
        "input_source_is_review_paper": input_record.get("input_source_is_review_paper"),
        "input_raw_benchmark_item": input_record.get("input_raw_benchmark_item"),
    }

    if llm_result.get("error"):
        output_record["error"] = llm_result.get("error")
    if llm_result.get("raw_output"):
        output_record["raw_output"] = llm_result.get("raw_output")

    return output_record


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc


def get_record_key(record: dict[str, Any]) -> str:
    """Stable key for resume/skip behavior."""
    return str(record.get("input_query_id") or record.get("original_query") or record.get("query"))


def load_completed_keys(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()

    completed: set[str] = set()
    for _, record in iter_jsonl(output_file):
        key = get_record_key(record)
        if key:
            completed.add(key)
    return completed


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an LLM-only general-knowledge baseline JSONL from RAG output JSONL."
    )
    parser.add_argument("--input_file", required=True, help="Existing RAG output JSONL file.")
    parser.add_argument("--output_file", required=True, help="Output JSONL file for LLM-only baseline.")
    parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model name.")
    parser.add_argument("--max_items", type=int, default=5, help="Maximum equipment items per query.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of input records to process.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file instead of resuming/skipping existing records.",
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls to reduce rate-limit pressure.",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    if args.overwrite and output_file.exists():
        output_file.unlink()

    completed_keys = load_completed_keys(output_file)
    processed_this_run = 0
    skipped = 0

    for _, input_record in iter_jsonl(input_file):
        if args.limit is not None and processed_this_run >= args.limit:
            break

        query = input_record.get("original_query") or input_record.get("query")
        if not query:
            continue

        key = get_record_key(input_record)
        if key in completed_keys:
            skipped += 1
            continue

        started_at_utc = utc_now_iso()
        llm_result = call_llm_only_equipment_recommender(
            query=query,
            model=args.model,
            max_items=args.max_items,
        )
        completed_at_utc = utc_now_iso()

        output_record = build_output_record(
            input_record=input_record,
            llm_result=llm_result,
            model=args.model,
            max_items=args.max_items,
            started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
        )
        write_jsonl_record(output_file, output_record)
        completed_keys.add(key)
        processed_this_run += 1

        print(
            f"[{processed_this_run}] {output_record['run_status']}: "
            f"{len(output_record['aggregated_equipment'])} equipment items | {query[:90]}"
        )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print(f"Done. Processed: {processed_this_run}. Skipped existing: {skipped}. Output: {output_file}")


if __name__ == "__main__":
    main()
