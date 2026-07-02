from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Add OPENAI_API_KEY=... to your .env file."
    )

OPENAI_API_KEY = OPENAI_API_KEY.strip()

if "\n" in OPENAI_API_KEY or "\r" in OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY contains a newline. Check your .env file.")

if len(OPENAI_API_KEY) > 300:
    raise RuntimeError(
        f"OPENAI_API_KEY looks too long ({len(OPENAI_API_KEY)} characters)."
    )

if not OPENAI_API_KEY.startswith(("sk-", "sk-proj-")):
    raise RuntimeError("OPENAI_API_KEY does not look like a valid OpenAI API key.")


http_client = httpx.Client(
    timeout=60.0,
    trust_env=False,
)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=http_client,
)


ALLOWED_QUERY_TYPES = {
    "specific",
    "broad_diagnostic",
    "method_selection",
    "process_optimization",
}


def _validate_and_normalize_decomposition(
    parsed: dict[str, Any],
    query: str,
    max_subproblems: int,
) -> dict[str, Any]:
    """
    Make the model output safer and more predictable for downstream code.
    """
    parsed["status"] = "subproblems_generated"
    parsed["original_query"] = parsed.get("original_query") or query

    query_type = parsed.get("query_type")
    if query_type not in ALLOWED_QUERY_TYPES:
        parsed["query_type"] = "broad_diagnostic"

    if "should_decompose" not in parsed:
        parsed["should_decompose"] = parsed["query_type"] != "specific"

    if not parsed.get("decomposition_reason"):
        parsed["decomposition_reason"] = "No decomposition reason provided."

    subproblems = parsed.get("subproblems")
    if not isinstance(subproblems, list):
        subproblems = []

    normalized_subproblems: list[dict[str, Any]] = []
    for index, subproblem in enumerate(subproblems[:max_subproblems], start=1):
        if not isinstance(subproblem, dict):
            continue

        subproblem_id = subproblem.get("subproblem_id") or f"subproblem_{index}"
        subproblem_description = subproblem.get("subproblem_description") or ""
        analysis_goal = subproblem.get("analysis_goal") or ""
        hypothesis_or_branch = subproblem.get("hypothesis_or_branch") or ""
        expected_method_families = subproblem.get("expected_method_families") or []

        if not isinstance(expected_method_families, list):
            expected_method_families = [str(expected_method_families)]

        normalized_subproblems.append(
            {
                "subproblem_id": str(subproblem_id),
                "subproblem_description": str(subproblem_description),
                "analysis_goal": str(analysis_goal),
                "hypothesis_or_branch": str(hypothesis_or_branch),
                "expected_method_families": [
                    str(method_family) for method_family in expected_method_families
                ],
            }
        )

    parsed["subproblems"] = normalized_subproblems
    return parsed


def generate_subproblems(
    query: str,
    model: str = "gpt-5.4-mini",
    max_subproblems: int = 5,
) -> dict[str, Any]:
    """
    Classify and decompose a research equipment recommendation query.

    This function only creates query type and subproblem descriptions. It does
    not retrieve papers, recommend final equipment, or generate Semantic Scholar
    search queries. Search-query generation should be a separate downstream step.
    """
    prompt = f"""
You are a scientific problem-decomposition module for a literature-grounded
research equipment recommender.

The user gives a natural-language research, measurement, or analysis problem.
Your task is to classify the query and, when useful, decompose it into smaller
analysis-oriented subproblems before retrieval.

Important:
- Do NOT answer the user directly.
- Do NOT recommend final equipment as the answer.
- Do NOT generate literature search queries in this step.
- Do NOT invent local availability.
- Decompose only when the original problem is broad, diagnostic, causal,
  exploratory, or has multiple possible scientific explanations.
- If the query is already specific, such as asking for one instrument to perform
  one measurement, return query_type="specific", should_decompose=false, and
  one subproblem that restates the specific measurement need.
- For broad diagnostic questions, create 3 to {max_subproblems} subproblems.
- Each subproblem must remain anchored to the original material, sample,
  domain, failure mode, or measurement problem.
- Keep subproblems distinct. Avoid heavy overlap.
- For polymer-softness questions, include a viscoelastic/mechanical behavior
  branch when relevant.
- Split crystallinity/thermal transitions and morphology/phase separation into
  separate branches when both are relevant.
- Only include processing history if the user mentions synthesis, molding,
  annealing, printing, cooling, solvent, moisture, manufacturing, or process
  conditions.

Return valid JSON only.

JSON schema:
{{
  "original_query": "original user query",
  "query_type": "specific | broad_diagnostic | method_selection | process_optimization",
  "should_decompose": true,
  "decomposition_reason": "brief reason",
  "subproblems": [
    {{
      "subproblem_id": "short_snake_case_id",
      "subproblem_description": "natural-language subproblem description",
      "analysis_goal": "what this subproblem tries to investigate",
      "hypothesis_or_branch": "the possible cause, mechanism, or analysis branch",
      "expected_method_families": [
        "method family 1",
        "method family 2"
      ]
    }}
  ]
}}

Guidance:
- expected_method_families may contain broad families such as spectroscopy,
  rheology, calorimetry, microscopy, X-ray scattering, mechanical testing,
  chromatography, mass spectrometry, electrochemical analysis, etc.
- Method families are only expected directions for later retrieval, not final
  recommendations.
- Do not include a field named "search_queries".
- Do not include exact equipment unless it is necessary to describe the
  subproblem at a high level.

User query:
{query}
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    raw_output = response.output_text.strip()

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "original_query": query,
            "status": "invalid_json",
            "raw_output": raw_output,
            "query_type": None,
            "should_decompose": None,
            "decomposition_reason": None,
            "subproblems": [],
        }

    return _validate_and_normalize_decomposition(
        parsed=parsed,
        query=query,
        max_subproblems=max_subproblems,
    )


def get_subproblem_descriptions(
    decomposition: dict[str, Any],
) -> list[str]:
    """
    Return only the natural-language subproblem descriptions.
    This is useful if a downstream module generates search queries separately.
    """
    descriptions: list[str] = []

    for subproblem in decomposition.get("subproblems", []):
        description = subproblem.get("subproblem_description")
        if description and description not in descriptions:
            descriptions.append(description)

    return descriptions


if __name__ == "__main__":
    test_query = (
        "I work at Polyvation. We are developing a new polymer, which is a "
        "variation on PEEK for medical grade purposes. The material turns out "
        "to be way softer than anticipated. Why does our experimental material "
        "turn out so soft? Which equipment could be used for supportive analysis?"
    )

    result = generate_subproblems(
        query=test_query,
        max_subproblems=5,
    )

    print("\nGenerated subproblem decomposition:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\nSubproblem descriptions:")
    for description in get_subproblem_descriptions(result):
        print("-", description)
