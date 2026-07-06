from __future__ import annotations

from functools import lru_cache
import re
from typing import Any

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# PDF extraction can sometimes produce isolated UTF-16 surrogate characters
# such as \ud835. These are not valid standalone Unicode characters and cannot
# be encoded as UTF-8 by tokenizers / JSON / CSV writers.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def clean_text_for_utf8(value: Any) -> str:
    """Return a string that is safe for UTF-8 tokenization and file writing."""
    if value is None:
        return ""

    # Handle pandas missing values without turning them into the literal string "nan".
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value)

    # Remove isolated surrogate code points directly.
    text = _SURROGATE_RE.sub("", text)

    # Defensive final pass: remove anything else that still cannot be UTF-8 encoded.
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


@lru_cache(maxsize=2)
def load_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def rerank_pairs(
    query: str,
    passages: list[str],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    normalize: bool = True,
) -> list[tuple[float, str]]:
    tokenizer, model = load_reranker(model_name)

    clean_query = clean_text_for_utf8(query)
    clean_passages = [clean_text_for_utf8(passage) for passage in passages]
    pairs = [[clean_query, passage] for passage in clean_passages]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        scores = model(**inputs, return_dict=True).logits.view(-1).float()

        if normalize:
            scores = torch.sigmoid(scores)

    # Return the cleaned passages, so downstream JSONL/CSV writing is also safe.
    scored_passages = list(zip(scores.tolist(), clean_passages))
    scored_passages.sort(key=lambda x: x[0], reverse=True)
    return scored_passages


def rerank_top_k_chunks(
    query: str,
    retrieved_chunks_df: pd.DataFrame,
    text_column: str = "chunk_text",
    model_name: str = "BAAI/bge-reranker-v2-m3",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Rerank a DataFrame of retrieved chunks using BAAI/bge-reranker-v2-m3.

    This version sanitizes PDF-extracted text before tokenization and also
    writes the sanitized text back into the returned DataFrame. This prevents
    crashes such as: UnicodeEncodeError: surrogates not allowed.
    """
    if retrieved_chunks_df.empty:
        return retrieved_chunks_df.copy()

    if text_column not in retrieved_chunks_df.columns:
        raise ValueError(f"Missing text column '{text_column}' in retrieved_chunks_df")

    reranked_df = retrieved_chunks_df.copy()
    reranked_df[text_column] = reranked_df[text_column].map(clean_text_for_utf8)

    clean_query = clean_text_for_utf8(query)
    passages = reranked_df[text_column].tolist()

    tokenizer, model = load_reranker(model_name)
    pairs = [[clean_query, passage] for passage in passages]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        scores = model(**inputs, return_dict=True).logits.view(-1).float()

        if normalize:
            scores = torch.sigmoid(scores)

    reranked_df["reranker_score"] = scores.tolist()
    reranked_df = reranked_df.sort_values("reranker_score", ascending=False).reset_index(drop=True)
    return reranked_df


if __name__ == "__main__":
    query = (
        "We need to measure the EUV emission spectrum from a laser-produced tin "
        "microdroplet plasma around 13.5 nm. What instrument should we use?"
    )

    retrieved_chunks_df = pd.DataFrame(
        {
            "chunk_id": [0, 1],
            "chunk_text": [
                "A transmission grating spectrometer enables unraveling the EUV spectrum.",
                "The experimental spectra were obtained with an Nd:YAG laser pulse. Bad surrogate: \ud835",
            ],
            "similarity": [0.82, 0.79],
        }
    )

    reranked_df = rerank_top_k_chunks(query, retrieved_chunks_df)
    print(reranked_df.to_string(index=False))
