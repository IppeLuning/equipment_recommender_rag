from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from equipment_recommender_rag.embeddings.create_embeddings import (
    create_embedding_openai,
    create_embeddings_openai,
)
from equipment_recommender_rag.full_text_chunking.chunk_full_text import split_data_into_chunks
from equipment_recommender_rag.full_text_chunking.retrieve_full_text import retrieve_full_text


MAX_CHARS_PER_EMBEDDING = 12_000
BATCH_SIZE = 32


def _cosine_similarity(query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    query_norm = np.linalg.norm(query_embedding)
    doc_norm = np.linalg.norm(doc_embedding)

    if query_norm == 0.0 or doc_norm == 0.0:
        return 0.0

    return float(np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm))


def _empty_result_df(return_embeddings: bool = False) -> pd.DataFrame:
    columns = [
        "chunk_id",
        "chunk_text",
        "similarity",
        "chunk_length_words",
        "chunk_length_chars",
    ]

    if return_embeddings:
        columns.extend(["query_embedding", "chunk_embedding"])

    return pd.DataFrame(columns=columns)


def _clean_text_for_embedding(text: str) -> str:
    """
    Remove null bytes and collapse whitespace before sending text to the embedding API.
    """
    return " ".join(text.replace("\x00", " ").split()).strip()


def _prepare_chunks_for_embedding(chunks: list[str]) -> list[tuple[int, str]]:
    """
    Clean chunks and remove empty or suspiciously large chunks.
    """
    clean_chunks: list[tuple[int, str]] = []

    for chunk_id, chunk_text in enumerate(chunks):
        cleaned_chunk = _clean_text_for_embedding(chunk_text)

        if not cleaned_chunk:
            continue

        if len(cleaned_chunk) > MAX_CHARS_PER_EMBEDDING:
            print(
                f"Skipping chunk {chunk_id}: too long "
                f"({len(cleaned_chunk)} characters)"
            )
            continue

        clean_chunks.append((chunk_id, cleaned_chunk))

    return clean_chunks


def retrieve_top_k_chunks_from_full_text(
    query: str,
    full_text: str,
    top_k: int = 10,
    chunk_sz: int = 400,
    min_chunk_sz: int = 100,
    keep_last: bool = True,
    return_embeddings: bool = False,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Split a full text into chunks in memory, embed chunks in batches,
    and return the top-k most relevant chunks.

    The function keeps the same behaviour as the earlier version, but reduces
    the number of OpenAI embedding requests by embedding chunk batches instead
    of one chunk at a time.
    """
    if not full_text or not full_text.strip():
        return _empty_result_df(return_embeddings=return_embeddings)

    chunks = split_data_into_chunks(
        text=full_text,
        chunk_sz=chunk_sz,
        min_chunk_sz=min_chunk_sz,
        keep_last=keep_last,
    )

    if not chunks:
        return _empty_result_df(return_embeddings=return_embeddings)

    cleaned_query = _clean_text_for_embedding(query)
    if not cleaned_query:
        raise ValueError("Query is empty after cleaning; cannot create query embedding.")

    query_embedding = np.array(create_embedding_openai(cleaned_query), dtype=np.float32)
    clean_chunks = _prepare_chunks_for_embedding(chunks)

    if not clean_chunks:
        return _empty_result_df(return_embeddings=return_embeddings)

    rows: list[dict[str, Any]] = []

    for start in range(0, len(clean_chunks), batch_size):
        batch = clean_chunks[start : start + batch_size]
        batch_texts = [chunk_text for _, chunk_text in batch]

        print(
            f"Embedding chunk batch {start // batch_size + 1}: "
            f"{len(batch_texts)} chunks"
        )

        batch_embeddings = create_embeddings_openai(batch_texts)

        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                "Embedding API returned a different number of embeddings than inputs: "
                f"{len(batch_embeddings)} embeddings for {len(batch)} chunks."
            )

        for (chunk_id, chunk_text), embedding in zip(batch, batch_embeddings):
            chunk_embedding = np.array(embedding, dtype=np.float32)
            similarity = _cosine_similarity(query_embedding, chunk_embedding)

            row: dict[str, Any] = {
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "similarity": similarity,
                "chunk_length_words": len(chunk_text.split()),
                "chunk_length_chars": len(chunk_text),
            }

            if return_embeddings:
                row["query_embedding"] = query_embedding.tolist()
                row["chunk_embedding"] = chunk_embedding.tolist()

            rows.append(row)

    if not rows:
        return _empty_result_df(return_embeddings=return_embeddings)

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
