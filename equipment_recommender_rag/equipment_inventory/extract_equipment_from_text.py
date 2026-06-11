import json
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI()


def extract_equipment_from_reranked_chunks(
    query: str,
    reranked_chunks_df: pd.DataFrame,
    source_label: str | None = None,
    top_n_chunks: int = 8,
    text_column: str = "chunk_text",
    model: str = "gpt-5.4-mini",
) -> dict[str, Any]:
    """
    Extract only query-relevant physical equipment from already-reranked chunks.

    This module should not search papers, retrieve full text, chunk text, or rerank.
    It only converts reranked evidence chunks into query-relevant equipment answers.
    """
    if reranked_chunks_df.empty:
        return {
            "source_label": source_label,
            "query": query,
            "status": "no_chunks",
            "query_relevant_equipment": [],
        }

    if text_column not in reranked_chunks_df.columns:
        raise ValueError(f"Missing text column '{text_column}' in reranked_chunks_df")

    top_chunks_df = reranked_chunks_df.head(top_n_chunks).copy()

    chunk_blocks = []
    for _, row in top_chunks_df.iterrows():
        chunk_id = row.get("chunk_id")
        sim = row.get("similarity")
        rerank_score = row.get("reranker_score")
        chunk_text = str(row[text_column])
        paper_title = row.get("paper_title")
        doi = row.get("doi")
        source_type = row.get("source_type")

        block = (
            f"[CHUNK {chunk_id}]\n"
            f"Paper title: {paper_title}\n"
            f"DOI: {doi}\n"
            f"Source type: {source_type}\n"
            f"Similarity: {sim}\n"
            f"Reranker score: {rerank_score}\n"
            f"Text:\n{chunk_text}"
        )
        chunk_blocks.append(block)

    combined_context = "\n\n".join(chunk_blocks)

    prompt = f"""
You are a careful scientific query-answering extraction system.

Task:
Given a user query and top reranked text chunks from multiple papers, identify only the PHYSICAL specialized equipment that is relevant for answering the query.

Important:
- Only use the provided chunks as evidence.
- The goal is NOT to extract all equipment mentioned in the chunks.
- The goal IS to identify which equipment can answer the user query and how suitable each item is.
- Do NOT infer equipment that is not explicitly mentioned or strongly implied.
- Do NOT include software, simulation codes, computational models, scripts, or code stacks.
- Do NOT include generic lab supplies or minor accessories unless they are clearly central.
- Focus on physical instruments, analyzers, microscopes, spectrometers, detectors, lasers,
  chromatography systems, imaging systems, beamlines, reactors, testing systems, and other tangible scientific equipment.
- Merge duplicate mentions of the same equipment across chunks and across papers when they clearly refer to the same instrument type.
- Evidence with source_type="metadata" is weak evidence. Do not infer specific equipment from title-only metadata unless the equipment or method is explicitly named in the title or metadata.
- Evidence with source_type="tldr_metadata" is weaker than abstract or full-text evidence. Use it cautiously.
- Prefer full_text/pdf/html and abstract evidence over metadata-only evidence when deciding relevance.
- Return valid JSON only.

Suitability labels:
- "best_match": directly solves the measurement / analysis problem in the query
- "acceptable_alternative": relevant and usable, but not the clearest or best match
- "supporting_but_not_ideal": related to the setup or workflow, but does not directly answer the query
- "clearly_not_suitable": mentioned in the chunks but does not solve the query

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
      "reason": "why this equipment is or is not suitable for answering the query",
      "measurement_outputs": [
        "output 1",
        "output 2"
      ],
      "query_specific_use": "how it relates to the user query",
      "certainty": "explicit",
      "confidence_score": 0.0,
      "evidence_text": [
        "short quote 1",
        "short quote 2"
      ],
      "source_chunk_ids": [0, 1],
      "supporting_papers": [
        {{
          "paper_title": "title",
          "doi": "doi or null"
        }}
      ]
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

Top reranked chunks:
\"\"\"
{combined_context}
\"\"\"
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    raw_output = response.output_text.strip()

    try:
        parsed_output = json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "source_label": source_label,
            "query": query,
            "status": "invalid_json",
            "query_relevant_equipment": [],
            "raw_output": raw_output,
        }

    if "query_relevant_equipment" not in parsed_output:
        return {
            "source_label": source_label,
            "query": query,
            "status": "missing_query_relevant_equipment_key",
            "query_relevant_equipment": [],
            "raw_output": raw_output,
        }

    parsed_output["source_label"] = source_label
    parsed_output["query"] = parsed_output.get("query", query)
    parsed_output["status"] = "query_relevant_equipment_extracted_from_chunks"
    return parsed_output