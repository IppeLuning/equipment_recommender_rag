from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from equipment_recommender_rag.embeddings.create_embeddings import create_embedding_openai
from equipment_recommender_rag.full_text_chunking.chunk_full_text import split_data_into_chunks
from equipment_recommender_rag.full_text_chunking.retrieve_full_text import retrieve_full_text


def _cosine_similarity(query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    query_norm = np.linalg.norm(query_embedding)
    doc_norm = np.linalg.norm(doc_embedding)

    if query_norm == 0.0 or doc_norm == 0.0:
        return 0.0

    return float(np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm))


def retrieve_top_k_chunks_from_full_text(
    query: str,
    full_text: str,
    top_k: int = 10,
    chunk_sz: int = 400,
    min_chunk_sz: int = 100,
    keep_last: bool = True,
    return_embeddings: bool = False,
) -> pd.DataFrame:
    """
    Split a full text into chunks in memory, embed each chunk, and return the top-k most relevant chunks.
    """
    if not full_text or not full_text.strip():
        return pd.DataFrame(
            columns=[
                "chunk_id",
                "chunk_text",
                "similarity",
                "chunk_length_words",
            ]
        )

    chunks = split_data_into_chunks(
        text=full_text,
        chunk_sz=chunk_sz,
        min_chunk_sz=min_chunk_sz,
        keep_last=keep_last,
    )

    if not chunks:
        return pd.DataFrame(
            columns=[
                "chunk_id",
                "chunk_text",
                "similarity",
                "chunk_length_words",
            ]
        )

    query_embedding = np.array(create_embedding_openai(query), dtype=np.float32)

    rows: list[dict[str, Any]] = []

    for chunk_id, chunk_text in enumerate(chunks):
        chunk_embedding = np.array(create_embedding_openai(chunk_text), dtype=np.float32)
        similarity = _cosine_similarity(query_embedding, chunk_embedding)

        row = {
            "chunk_id": chunk_id,
            "chunk_text": chunk_text,
            "similarity": similarity,
            "chunk_length_words": len(chunk_text.split()),
        }

        if return_embeddings:
            row["query_embedding"] = query_embedding.tolist()
            row["chunk_embedding"] = chunk_embedding.tolist()

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)
    return df


if __name__ == "__main__":
    query = (
        "We need to measure the EUV emission spectrum from a laser-produced tin "
        "microdroplet plasma and compare it against calculated opacity spectra "
        "around 13.5 nm. What instrument should we use?"
    )

    result = retrieve_full_text(
        doi="10.1038/s41467-020-15678-y",
        arxiv_id=None,
        url=None,
    )

    print("Full-text retrieval status:", result["status"])
    print("Source type:", result["source_type"])
    print("Source URL:", result["source_url"])
    print("Validation note:", result.get("validation_note"))

    if result["status"] != "success" or not result["full_text"]:
        print("Could not retrieve full text.")
    else:
        top_chunks = retrieve_top_k_chunks_from_full_text(
            query=query,
            full_text=result["full_text"],
            top_k=5,
            chunk_sz=400,
            min_chunk_sz=100,
            keep_last=True,
            return_embeddings=False,
        )

        print("\nTop retrieved chunks:")
        print(top_chunks[["chunk_id", "similarity", "chunk_text"]].to_string(index=False))