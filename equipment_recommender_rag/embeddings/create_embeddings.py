from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing.")

api_key = api_key.strip()

if "\n" in api_key or "\r" in api_key:
    raise RuntimeError("OPENAI_API_KEY contains a newline.")

if len(api_key) > 300:
    raise RuntimeError(f"OPENAI_API_KEY looks too long: {len(api_key)} characters.")

if not api_key.startswith(("sk-", "sk-proj-")):
    raise RuntimeError("OPENAI_API_KEY does not look like a valid OpenAI API key.")

http_client = httpx.Client(
    timeout=60.0,
    trust_env=False,
)

client = OpenAI(
    api_key=api_key,
    http_client=http_client,
)


def create_embedding_openai(text: str) -> list[float]:
    cleaned = " ".join(text.replace("\x00", " ").split()).strip()

    if not cleaned:
        raise ValueError("Cannot create embedding for empty text.")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=cleaned,
    )

    return response.data[0].embedding


def create_embeddings_openai(texts: list[str]) -> list[list[float]]:
    cleaned_texts = [
        " ".join(text.replace("\x00", " ").split()).strip()
        for text in texts
    ]
    cleaned_texts = [text for text in cleaned_texts if text]

    if not cleaned_texts:
        return []

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=cleaned_texts,
    )

    return [item.embedding for item in response.data]