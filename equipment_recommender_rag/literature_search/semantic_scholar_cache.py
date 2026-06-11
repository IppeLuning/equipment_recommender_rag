import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CACHE_PATH = Path("data/cache/semantic_scholar_cache.db")


def normalize_query(query: str) -> str:
    """
    Normalize query strings so tiny whitespace/casing differences do not create
    unnecessary duplicate cache entries.
    """
    return " ".join(query.strip().lower().split())


def get_connection(db_path: Path = DEFAULT_CACHE_PATH) -> sqlite3.Connection:
    """
    Open the SQLite database. The file is created automatically if it does not exist.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_cache(db_path: Path = DEFAULT_CACHE_PATH) -> None:
    """
    Create the cache table if it does not already exist.
    """
    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_scholar_query_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                max_paper_num INTEGER NOT NULL,
                min_citation_count INTEGER NOT NULL,
                sort TEXT NOT NULL,
                fields TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(query, max_paper_num, min_citation_count, sort, fields)
            )
            """
        )
        connection.commit()


def get_cached_response(
    query: str,
    max_paper_num: int,
    min_citation_count: int,
    sort: str,
    fields: str,
    db_path: Path = DEFAULT_CACHE_PATH,
) -> list[dict[str, Any]] | None:
    """
    Return cached Semantic Scholar papers if this exact request was seen before.
    """
    initialize_cache(db_path)

    normalized_query = normalize_query(query)

    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT response_json
            FROM semantic_scholar_query_cache
            WHERE query = ?
              AND max_paper_num = ?
              AND min_citation_count = ?
              AND sort = ?
              AND fields = ?
            """,
            (
                normalized_query,
                max_paper_num,
                min_citation_count,
                sort,
                fields,
            ),
        ).fetchone()

    if row is None:
        return None

    return json.loads(row["response_json"])


def save_response_to_cache(
    query: str,
    max_paper_num: int,
    min_citation_count: int,
    sort: str,
    fields: str,
    papers: list[dict[str, Any]],
    db_path: Path = DEFAULT_CACHE_PATH,
) -> None:
    """
    Save a successful Semantic Scholar response.
    """
    initialize_cache(db_path)

    normalized_query = normalize_query(query)
    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO semantic_scholar_query_cache (
                query,
                max_paper_num,
                min_citation_count,
                sort,
                fields,
                response_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized_query,
                max_paper_num,
                min_citation_count,
                sort,
                fields,
                json.dumps(papers),
                created_at,
            ),
        )
        connection.commit()