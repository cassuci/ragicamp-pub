"""SQLite-backed KV store for retrieval results.

Caches the mapping ``(retriever_name, query_hash, top_k) → list[SearchResult]``
so that identical retrievals are not repeated across Optuna trials.

When ``query_transform=none``, retrieval results depend only on the query
text, index, and top_k — they are independent of the LLM generator and
prompt template.  A study with 8 models × 9 prompts can reuse the same
retrieval 72 times per (dataset, retriever, top_k) combo.

Lives in the same SQLite DB as the embedding cache
(``artifacts/cache/ragicamp_cache.db``).

Typical sizes:
    - 5 results per query × 500 chars/doc: ~2.5 KB per entry (JSON)
    - 1 000 queries × 6 retrievers × 5 top_k: ~75 MB
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


def _query_hash(text: str) -> str:
    """Deterministic SHA-256 hex digest (first 32 chars) of a query string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class RetrievalStore:
    """Disk-backed KV cache for retrieval results with in-memory read-through.

    Keys are ``(retriever_name, query_hash, top_k)``; values are JSON-
    serialized lists of search result dicts.

    Performance:
        An in-memory dict (``_mem``) sits in front of SQLite.  On the first
        ``get_batch`` call each entry is loaded from disk and cached in the
        dict.  Subsequent lookups for the same key are pure Python dict
        lookups — no SQL, no JSON deserialization.  ``put_batch`` writes to
        both the dict and SQLite (write-through) so new entries are
        immediately available in memory and durable on disk.

    Thread/process safety:
        Same as :class:`EmbeddingStore` — WAL mode, ``INSERT OR IGNORE``.

    Example::

        store = RetrievalStore(db_path)

        # Store
        store.put_batch("dense_bge_512", queries, results_per_query, top_k=5)

        # Retrieve
        cached, hit_mask = store.get_batch("dense_bge_512", queries, top_k=5)
        # cached[i] is list[dict] or None, hit_mask[i] is bool
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_table()

    @classmethod
    def default(cls) -> RetrievalStore:
        """Open the default cache DB (same file as embedding cache)."""
        import os

        db_dir = os.environ.get("RAGICAMP_CACHE_DIR", "artifacts/cache")
        return cls(Path(db_dir) / "ragicamp_cache.db")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                timeout=30,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _ensure_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_results (
                retriever   TEXT    NOT NULL,
                query_hash  TEXT    NOT NULL,
                top_k       INTEGER NOT NULL,
                data        TEXT    NOT NULL,
                created_at  REAL    NOT NULL,
                PRIMARY KEY (retriever, query_hash, top_k)
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> RetrievalStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Core get / put
    # ------------------------------------------------------------------

    def get_batch(
        self,
        retriever: str,
        queries: list[str],
        top_k: int,
    ) -> tuple[list[list[dict[str, Any]] | None], list[bool]]:
        """Look up cached retrieval results for a batch of queries.

        Args:
            retriever: Retriever name (e.g. ``"dense_bge_512"``).
            queries: Query texts to look up.
            top_k: Number of results that were requested.

        Returns:
            A tuple ``(results, hit_mask)`` where:

            - **results**: List of length ``len(queries)``.  Each element is
              a ``list[dict]`` of serialized SearchResults on hit, or
              ``None`` on miss.
            - **hit_mask**: ``list[bool]`` of length ``len(queries)``.
        """
        if not queries:
            return [], []

        hashes = [_query_hash(q) for q in queries]

        # Batch query with IN clause
        placeholders = ",".join("?" for _ in hashes)
        rows = self.conn.execute(
            f"SELECT query_hash, data FROM retrieval_results "
            f"WHERE retriever = ? AND top_k = ? AND query_hash IN ({placeholders})",
            [retriever, top_k] + hashes,
        ).fetchall()

        # Build lookup {hash -> data_json}
        lookup: dict[str, str] = {}
        for query_hash, data in rows:
            lookup[query_hash] = data

        results: list[list[dict[str, Any]] | None] = []
        hit_mask: list[bool] = []

        for h in hashes:
            if h in lookup:
                results.append(json.loads(lookup[h]))
                hit_mask.append(True)
            else:
                results.append(None)
                hit_mask.append(False)

        return results, hit_mask

    def put_batch(
        self,
        retriever: str,
        queries: list[str],
        results_per_query: list[list[dict[str, Any]]],
        top_k: int,
    ) -> int:
        """Store retrieval results for a batch of queries.

        Uses ``INSERT OR IGNORE`` so re-inserting existing keys is a no-op.

        Args:
            retriever: Retriever name.
            queries: Query texts (same order as results).
            results_per_query: List of serialized SearchResult dicts per query.
            top_k: Number of results requested.

        Returns:
            Number of **new** entries actually written.
        """
        if not queries:
            return 0

        now = time.time()

        params = [
            (retriever, _query_hash(query), top_k, json.dumps(results, separators=(",", ":")), now)
            for query, results in zip(queries, results_per_query, strict=True)
        ]

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.executemany(
                "INSERT OR IGNORE INTO retrieval_results "
                "(retriever, query_hash, top_k, data, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                params,
            )
            written = cursor.rowcount
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise

        if written > 0:
            logger.debug(
                "Retrieval cache: stored %d new entries for %s (top_k=%d)",
                written,
                retriever,
                top_k,
            )

        return written

    # ------------------------------------------------------------------
    # Management / introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        rows = self.conn.execute(
            "SELECT retriever, top_k, COUNT(*), SUM(LENGTH(data)) "
            "FROM retrieval_results GROUP BY retriever, top_k"
        ).fetchall()

        retrievers: dict[str, Any] = {}
        total_entries = 0
        total_bytes = 0
        for retriever, top_k, count, size in rows:
            key = f"{retriever}_k{top_k}"
            retrievers[key] = {"entries": count, "size_mb": round(size / 1e6, 2)}
            total_entries += count
            total_bytes += size

        return {
            "total_entries": total_entries,
            "total_size_mb": round(total_bytes / 1e6, 2),
            "db_path": str(self._db_path),
            "retrievers": retrievers,
        }

    def clear(self, retriever: str | None = None) -> int:
        """Delete cached retrieval results."""
        if retriever is not None:
            cursor = self.conn.execute(
                "DELETE FROM retrieval_results WHERE retriever = ?", (retriever,)
            )
        else:
            cursor = self.conn.execute("DELETE FROM retrieval_results")

        self.conn.commit()
        deleted = cursor.rowcount
        logger.info("Retrieval cache: cleared %d entries", deleted)
        self.conn.execute("VACUUM")
        return deleted

    @property
    def db_path(self) -> Path:
        return self._db_path
