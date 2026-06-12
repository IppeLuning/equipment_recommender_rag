from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_PATH = Path("data/processed/proposed_equipment_runs.jsonl")


def equipment_to_saved_record(equipment: dict[str, Any]) -> dict[str, Any]:
    """
    Convert one extracted equipment item into the compact format we want to save.

    Note: the extractor currently returns a confidence_score, not a statistical
    confidence interval. This field is therefore saved as confidence_score.
    """
    return {
        "equipment_name": equipment.get("equipment_name"),
        "aliases": equipment.get("aliases", []),
        "equipment_type": equipment.get("equipment_type"),
        "relevance_label": equipment.get("relevance_label"),
        "confidence_score": equipment.get("confidence_score"),
        "explanation": equipment.get("reason"),
        "query_specific_use": equipment.get("query_specific_use"),
        "measurement_outputs": equipment.get("measurement_outputs", []),
        "evidence_text": equipment.get("evidence_text", []),
        "supporting_papers": equipment.get("supporting_papers", []),
    }


def build_run_record(
    result: dict[str, Any],
    pipeline_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a compact JSON-serializable record for one pipeline run.
    """
    proposed_equipment = [
        equipment_to_saved_record(equipment)
        for equipment in result.get("query_relevant_equipment", [])
    ]

    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "original_query": result.get("query"),
        "status": result.get("status"),
        "source_label": result.get("source_label"),
        "generated_queries": result.get("generated_queries", []),
        "num_candidate_papers": result.get("num_candidate_papers"),
        "num_candidate_chunks": result.get("num_candidate_chunks"),
        "pipeline_config": pipeline_config,
        "proposed_equipment": proposed_equipment,
    }


def save_run_record(
    result: dict[str, Any],
    pipeline_config: dict[str, Any],
    output_path: str | Path = DEFAULT_RESULTS_PATH,
) -> Path:
    """
    Append one compact run record to a JSONL file.

    JSONL means one JSON object per line. This is convenient because each
    pipeline run can be appended without loading or rewriting the whole file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_record = build_run_record(
        result=result,
        pipeline_config=pipeline_config,
    )

    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(run_record, ensure_ascii=False) + "\n")

    return output_path