from __future__ import annotations

from functools import lru_cache

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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

    pairs = [[query, passage] for passage in passages]

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

    scored_passages = list(zip(scores.tolist(), passages))
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

    Parameters
    ----------
    query : str
        User query.
    retrieved_chunks_df : pd.DataFrame
        Output of the embedding retriever, must contain a text column.
    text_column : str
        Column containing chunk text.
    model_name : str
        Hugging Face reranker model name.
    normalize : bool
        Whether to apply sigmoid to logits.

    Returns
    -------
    pd.DataFrame
        Same rows as input, reordered by reranker score descending.
    """
    if retrieved_chunks_df.empty:
        return retrieved_chunks_df.copy()

    if text_column not in retrieved_chunks_df.columns:
        raise ValueError(f"Missing text column '{text_column}' in retrieved_chunks_df")

    passages = retrieved_chunks_df[text_column].fillna("").astype(str).tolist()

    tokenizer, model = load_reranker(model_name)

    pairs = [[query, passage] for passage in passages]

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

    reranked_df = retrieved_chunks_df.copy()
    reranked_df["reranker_score"] = scores.tolist()
    reranked_df = reranked_df.sort_values("reranker_score", ascending=False).reset_index(drop=True)
    return reranked_df


if __name__ == "__main__":
    import pandas as pd

    query = (
        "We need to measure the EUV emission spectrum from a laser-produced tin "
        "microdroplet plasma around 13.5 nm. What instrument should we use?"
    )

    retrieved_chunks_df = pd.DataFrame(
        {
            "chunk_id": [0, 1],
            "chunk_text": [
                "A transmission grating spectrometer enables unraveling the EUV spectrum.",
                "The experimental spectra were obtained with an Nd:YAG laser pulse.",
            ],
            "similarity": [0.82, 0.79],
        }
    )

    reranked_df = rerank_top_k_chunks(query, retrieved_chunks_df)

    print(reranked_df.to_string(index=False))