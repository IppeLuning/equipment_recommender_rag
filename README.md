# Equipment Recommender RAG

This repository contains a literature-grounded Retrieval-Augmented Generation (RAG) system for recommending scientific research equipment from a natural-language problem description. The system retrieves relevant scientific papers, optionally retrieves full text, chunks and reranks the evidence, and uses an LLM to extract equipment that may help solve the user's problem.

The project was developed for a bachelor thesis on connecting user problems to university research equipment. It can be run for a single query or for a batch of benchmark problem descriptions stored in CSV, JSON, or JSONL format.

## Requirements

- Python managed with [`uv`](https://docs.astral.sh/uv/)
- An OpenAI API key
- A Semantic Scholar API key
- An email address for Unpaywall API requests

## Setup

Clone the repository and enter the project folder:

```bash
git clone https://github.com/IppeLuning/equipment_recommender_rag.git
cd equipment_recommender_rag
```

Install the environment with `uv`:

```bash
uv sync
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-proj-xxxxxx
S2_API_KEY=s2k-xxxxxxxxx
UNPAYWALL_EMAIL=your@email.com
```

Do not commit your `.env` file or API keys to GitHub.

## Running a single query

Run the pipeline with one natural-language problem description:

```bash
uv run python -m main \
  --query "I need to measure whether an antifouling coating prevents protein attachment." \
  --output_file data/processed/example_output.jsonl \
  --csv_output_file data/processed/example_output.csv
```

The JSONL output contains the full pipeline records, including retrieved evidence and aggregated equipment recommendations. The optional CSV output contains a flattened table of recommended equipment.

## Running a benchmark file

The pipeline can also be run on a file with multiple problem descriptions:

```bash
uv run python -m main \
  --input_file data/processed/problem_answer_pairs.json \
  --output_file data/processed/pipeline_output.jsonl \
  --csv_output_file data/processed/pipeline_output.csv
```

Supported input formats are:

- `.csv` with a `problem_description` column
- `.json` with either direct `problem_description` fields or nested `benchmark_items`
- `.jsonl` with the same supported JSON structures per line

Useful optional input columns or fields include:

- `query_id`
- `pdf_path`
- `doi`
- `is_review_paper`

## Common run used for evaluation

An example configuration for running the system without decomposition is:

```bash
uv run python -m main \
  --input_file data/processed/heldout_test_problem_answer_pairs_clean.json \
  --output_file data/processed/test_problem_answer_output.jsonl \
  --csv_output_file data/processed/test_problem_answer_output.csv \
  --no_decomposition \
  --max_workers 3 \
  --max_queries 4 \
  --max_paper_num_per_query 8 \
  --max_papers 10 \
  --top_k_per_paper 8 \
  --final_top_n_chunks 16 \
  --chunk_sz 250 \
  --min_chunk_sz 80
```

## Important command-line options

| Option | Description |
|---|---|
| `--query` | Run one problem description directly. |
| `--input_file` | Run multiple problem descriptions from a CSV, JSON, or JSONL file. |
| `--output_file` | Path for the full JSON or JSONL output. Defaults to `data/processed/decomposed_pipeline_runs.jsonl`. |
| `--csv_output_file` | Optional path for a flattened CSV output. |
| `--no_decomposition` | Disable subproblem generation and run the original query directly. |
| `--max_workers` | Number of original queries to run in parallel. Start low to avoid memory/API issues. |
| `--no_resume` | Ignore an existing output file and start a fresh run. |
| `--skip_failed` | When resuming, skip previously failed or interrupted records instead of retrying them. |
| `--stop_on_error` | Stop the full run when one query fails. |
| `--no_metadata_fallback` | Disable metadata fallback when full text is unavailable. |
| `--abstract_only` | Use only abstract/metadata evidence where supported. |
| `--no_paper_reranking` | Disable paper-level reranking after Semantic Scholar retrieval. |
| `--retrieval_verbose` | Print detailed paper retrieval logs. |
| `--print_debug_tables` | Print debugging tables for candidate and reranked chunks. |

## Output and resume behavior

The recommended output format is JSONL, because it is safer for long-running runs and supports checkpointing. When an output file already exists, completed queries are skipped automatically. Failed or interrupted queries are retried by default unless `--skip_failed` is used.

Each output record includes metadata such as the original query, query ID, source PDF path, DOI, decomposition information, subproblem results, and aggregated equipment recommendations.

## Notes

- Running the full system can be slow because it uses external APIs, full-text retrieval, chunking, reranking, and LLM calls.
- Start with a small number of queries before running a full benchmark file.
- On a laptop, use a low value for `--max_workers`, such as `1`, `2`, or `3`.
- Full-text access is not always available. When enabled, the system falls back to metadata such as titles and abstracts.
