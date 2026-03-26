import os
import httpx
from dotenv import load_dotenv

# Put .env in environment
load_dotenv()

API_KEY = os.getenv("OPENAI_KEY")


def create_embedding_openai(text: str) -> list[float]:
    print(text)

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
    )

    # Throws error if HTTP error occurs
    response.raise_for_status()
    data = response.json()

    return data["data"][0]["embedding"]