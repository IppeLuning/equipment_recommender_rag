import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")


def create_embedding_openai(text: str) -> list[float]:
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "input": text,
            "model": "text-embedding-3-small",
        },
        timeout=60.0,
    )

    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]