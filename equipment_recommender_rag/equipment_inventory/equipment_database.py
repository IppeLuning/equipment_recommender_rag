from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DB_PATH = Path("data/processed/equipment_inventory.sqlite")


def normalize_text(value: Any) -> str:
    """
    Normalize names for duplicate detection and matching.
    This intentionally keeps the logic simple and transparent.
    """
    if value is None or pd.isna(value):
        return ""

    text = str(value).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9µμ/+. ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_multi_value(value: Any) -> list[str]:
    """
    Split fields that use pipes or semicolons to store multiple values.
    """
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    parts = re.split(r"\s*\|\s*|\s*;\s*", text)
    return [part.strip() for part in parts if part.strip()]


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """
    Create database tables.

    Tables:
    - equipment: one deduplicated row per normalized equipment identity
    - equipment_aliases: searchable aliases for each equipment item
    - equipment_mentions: every paper-level mention from the CSV
    """
    with get_connection(db_path) as connection:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS equipment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                manufacturer TEXT,
                manufacturer_key TEXT NOT NULL DEFAULT '',
                model TEXT,
                model_key TEXT NOT NULL DEFAULT '',
                equipment_type TEXT,
                aliases_json TEXT NOT NULL DEFAULT '[]',
                mention_count INTEGER NOT NULL DEFAULT 0,
                primary_mention_count INTEGER NOT NULL DEFAULT 0,
                supporting_mention_count INTEGER NOT NULL DEFAULT 0,
                average_confidence_score REAL,
                UNIQUE(normalized_name, manufacturer_key, model_key)
            );

            CREATE TABLE IF NOT EXISTS equipment_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id INTEGER NOT NULL,
                alias TEXT NOT NULL,
                normalized_alias TEXT NOT NULL,
                UNIQUE(equipment_id, normalized_alias),
                FOREIGN KEY(equipment_id) REFERENCES equipment(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS equipment_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id INTEGER NOT NULL,
                pdf_path TEXT,
                pdf_filename TEXT,
                doi TEXT,
                study_domain TEXT,
                study_goal TEXT,
                equipment_role TEXT,
                short_description TEXT,
                typical_applications TEXT,
                measurement_outputs TEXT,
                paper_specific_use TEXT,
                sample_context TEXT,
                shared_or_local TEXT,
                contact_person TEXT,
                used_in_study INTEGER,
                certainty TEXT,
                confidence_score REAL,
                evidence_text TEXT,
                location_type TEXT,
                notes TEXT,
                FOREIGN KEY(equipment_id) REFERENCES equipment(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_equipment_normalized_name
                ON equipment(normalized_name);

            CREATE INDEX IF NOT EXISTS idx_equipment_type
                ON equipment(equipment_type);

            CREATE INDEX IF NOT EXISTS idx_equipment_aliases_normalized_alias
                ON equipment_aliases(normalized_alias);

            CREATE INDEX IF NOT EXISTS idx_equipment_mentions_equipment_id
                ON equipment_mentions(equipment_id);

            CREATE INDEX IF NOT EXISTS idx_equipment_mentions_doi
                ON equipment_mentions(doi);
            """
        )
        connection.commit()


def reset_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    with get_connection(db_path) as connection:
        connection.executescript(
            """
            DROP TABLE IF EXISTS equipment_mentions;
            DROP TABLE IF EXISTS equipment_aliases;
            DROP TABLE IF EXISTS equipment;
            """
        )
        connection.commit()

    initialize_database(db_path)


def _clean_cell(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def _equipment_identity(row: pd.Series) -> tuple[str, str | None, str | None]:
    normalized_name = normalize_text(row.get("equipment_name"))
    manufacturer = _clean_cell(row.get("manufacturer"))
    model = _clean_cell(row.get("model"))
    return normalized_name, manufacturer, model


def _upsert_equipment(connection: sqlite3.Connection, row: pd.Series) -> int:
    canonical_name = str(row.get("equipment_name")).strip()
    normalized_name, manufacturer, model = _equipment_identity(row)
    manufacturer_key = normalize_text(manufacturer)
    model_key = normalize_text(model)
    equipment_type = _clean_cell(row.get("equipment_type"))
    aliases = split_multi_value(row.get("aliases"))

    connection.execute(
        """
        INSERT OR IGNORE INTO equipment (
            canonical_name,
            normalized_name,
            manufacturer,
            manufacturer_key,
            model,
            model_key,
            equipment_type,
            aliases_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            canonical_name,
            normalized_name,
            manufacturer,
            manufacturer_key,
            model,
            model_key,
            equipment_type,
            json.dumps(aliases, ensure_ascii=False),
        ),
    )

    equipment_id = connection.execute(
        """
        SELECT id
        FROM equipment
        WHERE normalized_name = ?
          AND manufacturer_key = ?
          AND model_key = ?
        """,
        (normalized_name, normalize_text(manufacturer), normalize_text(model)),
    ).fetchone()["id"]

    # Add canonical name as searchable alias too.
    all_aliases = [canonical_name] + aliases
    for alias in all_aliases:
        normalized_alias = normalize_text(alias)
        if not normalized_alias:
            continue
        connection.execute(
            """
            INSERT OR IGNORE INTO equipment_aliases (
                equipment_id,
                alias,
                normalized_alias
            )
            VALUES (?, ?, ?)
            """,
            (equipment_id, alias, normalized_alias),
        )

    return int(equipment_id)


def _insert_mention(connection: sqlite3.Connection, equipment_id: int, row: pd.Series) -> None:
    used_in_study = row.get("used_in_study")
    if isinstance(used_in_study, str):
        used_in_study_int = 1 if used_in_study.strip().lower() in {"true", "1", "yes"} else 0
    else:
        used_in_study_int = int(bool(used_in_study))

    connection.execute(
        """
        INSERT INTO equipment_mentions (
            equipment_id,
            pdf_path,
            pdf_filename,
            doi,
            study_domain,
            study_goal,
            equipment_role,
            short_description,
            typical_applications,
            measurement_outputs,
            paper_specific_use,
            sample_context,
            shared_or_local,
            contact_person,
            used_in_study,
            certainty,
            confidence_score,
            evidence_text,
            location_type,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            equipment_id,
            _clean_cell(row.get("pdf_path")),
            _clean_cell(row.get("pdf_filename")),
            _clean_cell(row.get("doi")),
            _clean_cell(row.get("study_domain")),
            _clean_cell(row.get("study_goal")),
            _clean_cell(row.get("equipment_role")),
            _clean_cell(row.get("short_description")),
            _clean_cell(row.get("typical_applications")),
            _clean_cell(row.get("measurement_outputs")),
            _clean_cell(row.get("paper_specific_use")),
            _clean_cell(row.get("sample_context")),
            _clean_cell(row.get("shared_or_local")),
            _clean_cell(row.get("contact_person")),
            used_in_study_int,
            _clean_cell(row.get("certainty")),
            float(row.get("confidence_score")) if not pd.isna(row.get("confidence_score")) else None,
            _clean_cell(row.get("evidence_text")),
            _clean_cell(row.get("location_type")),
            _clean_cell(row.get("notes")),
        ),
    )


def _refresh_equipment_summary(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        UPDATE equipment
        SET
            mention_count = (
                SELECT COUNT(*)
                FROM equipment_mentions
                WHERE equipment_mentions.equipment_id = equipment.id
            ),
            primary_mention_count = (
                SELECT COUNT(*)
                FROM equipment_mentions
                WHERE equipment_mentions.equipment_id = equipment.id
                  AND equipment_mentions.equipment_role = 'primary'
            ),
            supporting_mention_count = (
                SELECT COUNT(*)
                FROM equipment_mentions
                WHERE equipment_mentions.equipment_id = equipment.id
                  AND equipment_mentions.equipment_role = 'supporting'
            ),
            average_confidence_score = (
                SELECT AVG(confidence_score)
                FROM equipment_mentions
                WHERE equipment_mentions.equipment_id = equipment.id
            )
        """
    )


def build_equipment_database_from_csv(
    csv_path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    reset: bool = True,
) -> Path:
    """
    Build a local SQLite equipment inventory database from the extracted equipment CSV.
    """
    csv_path = Path(csv_path)
    db_path = Path(db_path)

    if reset:
        reset_database(db_path)
    else:
        initialize_database(db_path)

    df = pd.read_csv(csv_path)

    required_columns = {"equipment_name", "equipment_type", "confidence_score"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    with get_connection(db_path) as connection:
        for _, row in df.iterrows():
            equipment_id = _upsert_equipment(connection, row)
            _insert_mention(connection, equipment_id, row)

        _refresh_equipment_summary(connection)
        connection.commit()

    return db_path


def find_equipment_matches(
    equipment_name: str,
    aliases: list[str] | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Match a RAG-proposed equipment item against the local equipment database.

    Matching order:
    1. exact normalized name / alias match
    2. partial normalized name / alias match
    """
    aliases = aliases or []
    search_terms = [equipment_name] + aliases
    normalized_terms = [normalize_text(term) for term in search_terms if normalize_text(term)]

    if not normalized_terms:
        return []

    with get_connection(db_path) as connection:
        exact_rows: list[sqlite3.Row] = []
        seen_ids: set[int] = set()

        for term in normalized_terms:
            rows = connection.execute(
                """
                SELECT DISTINCT
                    e.id,
                    e.canonical_name,
                    e.normalized_name,
                    e.manufacturer,
                    e.model,
                    e.equipment_type,
                    e.aliases_json,
                    e.mention_count,
                    e.primary_mention_count,
                    e.supporting_mention_count,
                    e.average_confidence_score,
                    'exact' AS match_type
                FROM equipment e
                LEFT JOIN equipment_aliases a ON a.equipment_id = e.id
                WHERE e.normalized_name = ?
                   OR a.normalized_alias = ?
                ORDER BY e.mention_count DESC
                """,
                (term, term),
            ).fetchall()
            for row in rows:
                if row["id"] not in seen_ids:
                    exact_rows.append(row)
                    seen_ids.add(row["id"])

        partial_rows: list[sqlite3.Row] = []
        if len(exact_rows) < limit:
            for term in normalized_terms:
                pattern = f"%{term}%"
                rows = connection.execute(
                    """
                    SELECT DISTINCT
                        e.id,
                        e.canonical_name,
                        e.normalized_name,
                        e.manufacturer,
                        e.model,
                        e.equipment_type,
                        e.aliases_json,
                        e.mention_count,
                        e.primary_mention_count,
                        e.supporting_mention_count,
                        e.average_confidence_score,
                        'partial' AS match_type
                    FROM equipment e
                    LEFT JOIN equipment_aliases a ON a.equipment_id = e.id
                    WHERE e.normalized_name LIKE ?
                       OR a.normalized_alias LIKE ?
                       OR ? LIKE '%' || e.normalized_name || '%'
                       OR ? LIKE '%' || a.normalized_alias || '%'
                    ORDER BY e.mention_count DESC
                    """,
                    (pattern, pattern, term, term),
                ).fetchall()
                for row in rows:
                    if row["id"] not in seen_ids:
                        partial_rows.append(row)
                        seen_ids.add(row["id"])

        rows = (exact_rows + partial_rows)[:limit]

    return [
        {
            "equipment_id": row["id"],
            "canonical_name": row["canonical_name"],
            "manufacturer": row["manufacturer"],
            "model": row["model"],
            "equipment_type": row["equipment_type"],
            "aliases": json.loads(row["aliases_json"] or "[]"),
            "mention_count": row["mention_count"],
            "primary_mention_count": row["primary_mention_count"],
            "supporting_mention_count": row["supporting_mention_count"],
            "average_confidence_score": row["average_confidence_score"],
            "match_type": row["match_type"],
        }
        for row in rows
    ]


def get_equipment_mentions(
    equipment_id: int,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Return paper-level mentions for a deduplicated equipment item.
    """
    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM equipment_mentions
            WHERE equipment_id = ?
            ORDER BY confidence_score DESC
            LIMIT ?
            """,
            (equipment_id, limit),
        ).fetchall()

    return [dict(row) for row in rows]


def compare_rag_result_to_database(
    rag_result: dict[str, Any],
    db_path: str | Path = DEFAULT_DB_PATH,
    limit_per_item: int = 5,
) -> dict[str, Any]:
    """
    Compare query-relevant equipment returned by the RAG pipeline to the local inventory.
    """
    compared_items = []

    for item in rag_result.get("query_relevant_equipment", []):
        matches = find_equipment_matches(
            equipment_name=item.get("equipment_name", ""),
            aliases=item.get("aliases", []),
            db_path=db_path,
            limit=limit_per_item,
        )

        compared_items.append(
            {
                "rag_equipment_name": item.get("equipment_name"),
                "rag_aliases": item.get("aliases", []),
                "rag_relevance_label": item.get("relevance_label"),
                "rag_confidence_score": item.get("confidence_score"),
                "database_match_found": bool(matches),
                "database_matches": matches,
            }
        )

    return {
        "query": rag_result.get("query"),
        "status": "rag_result_compared_to_equipment_database",
        "database_path": str(db_path),
        "items": compared_items,
    }


if __name__ == "__main__":
    csv_path = Path("data/processed/all_equipment_from_papers.csv")
    db_path = DEFAULT_DB_PATH

    build_equipment_database_from_csv(
        csv_path=csv_path,
        db_path=db_path,
        reset=True,
    )

    print(f"Built equipment database: {db_path}")

    example_matches = find_equipment_matches(
        "transmission grating spectrometer",
        aliases=["TGS", "spectrometer"],
        db_path=db_path,
    )

    print(json.dumps(example_matches, indent=2, ensure_ascii=False))
