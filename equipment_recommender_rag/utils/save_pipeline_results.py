from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_PATH = Path("data/processed/proposed_equipment_runs.json")


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
        "paper_retrieval_summary": result.get("paper_retrieval_summary"),
        "proposed_equipment": proposed_equipment,
    }


def save_run_record(
    result: dict[str, Any],
    pipeline_config: dict[str, Any],
    output_path: str | Path = DEFAULT_RESULTS_PATH,
) -> Path:
    """
    Append one compact run record to a readable JSON file.

    By default this writes a JSON array to proposed_equipment_runs.json. If the
    output path ends with .jsonl, it writes line-delimited JSON for backwards
    compatibility. The compact record includes paper_retrieval_summary, but not
    the full per-paper retrieval records.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_record = build_run_record(
        result=result,
        pipeline_config=pipeline_config,
    )

    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(run_record, ensure_ascii=False) + "\n")
        return output_path

    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as file:
                existing_records = json.load(file)
        except json.JSONDecodeError:
            existing_records = []
    else:
        existing_records = []

    if isinstance(existing_records, dict):
        existing_records = [existing_records]
    elif not isinstance(existing_records, list):
        existing_records = []

    existing_records.append(run_record)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(existing_records, file, indent=2, ensure_ascii=False)
        file.write("\n")

    return output_path
