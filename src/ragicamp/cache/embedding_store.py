"""SQLite-backed KV store for embedding vectors.

Stores embeddings keyed by (model_name, text_hash) with raw float32 blobs.
Designed for subprocess safety (WAL mode) and fast batch operations.

The DB lives at ``artifacts/cache/ragicamp_cache.db`` by default.
Future cache layers (retrieval results, reranker scores, etc.) can add
their own tables to the same DB file.

Typical sizes:
    - 384-dim model: ~1.5 KB per entry
    - 3 000 queries:  ~4.5 MB
    - 100 000 queries: ~150 MB
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)

# Default DB path relative to CWD (alongside artifacts/)
_DEFAULT_DB_DIR = "artifacts/cache"
_DEFAULT_DB_NAME = "ragicamp_cache.db"


def _text_hash(text: str) -> str:
    """Deterministic SHA-256 hex digest (first 32 chars) of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class EmbeddingStore:
    """Disk-backed KV cache for embedding vectors.

    Keys are ``(model_name, sha256(text))``; values are raw ``float32`` blobs.

    Thread/process safety:
        - Uses SQLite WAL mode for concurrent readers + single writer.
        - ``put_batch`` wraps inserts in a single transaction.
        - ``INSERT OR IGNORE`` makes concurrent writes of the same key safe.

    Example::

        store = EmbeddingStore.default()

        # Store
        store.put_batch("all-MiniLM-L6-v2", texts, embeddings)

        # Retrieve
        embs, mask = store.get_batch("all-MiniLM-L6-v2", texts)
        # embs.shape == (len(texts), dim), mask is bool array of hits
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_table()

    @classmethod
    def default(cls) -> EmbeddingStore:
        """Open (or create) the default cache DB at ``artifacts/cache/ragicamp_cache.db``."""
        db_dir = os.environ.get("RAGICAMP_CACHE_DIR", _DEFAULT_DB_DIR)
        return cls(Path(db_dir) / _DEFAULT_DB_NAME)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                timeout=30,  # wait up to 30s for write lock
            )
            # WAL mode: allows concurrent readers while writing
            self._conn.execute("PRAGMA journal_mode=WAL")
            # Sync less often for speed (data is a cache, not source of truth)
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _ensure_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                model      TEXT    NOT NULL,
                text_hash  TEXT    NOT NULL,
                dim        INTEGER NOT NULL,
                data       BLOB    NOT NULL,
                created_at REAL    NOT NULL,
                PRIMARY KEY (model, text_hash)
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> EmbeddingStore:
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
        model: str,
        texts: list[str],
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Look up cached embeddings for a list of texts.

        Args:
            model: Embedding model name (e.g. ``"all-MiniLM-L6-v2"``).
            texts: Query texts to look up.

        Returns:
            A tuple ``(embeddings, hit_mask)`` where:

            - **embeddings**: ``np.ndarray`` of shape ``(len(texts), dim)`` with
              cached vectors filled in and zeros for misses.  ``None`` if there
              are no hits at all (dimension unknown).
            - **hit_mask**: ``np.ndarray[bool]`` of shape ``(len(texts),)``.
              ``True`` where the cache had a hit.
        """
        if not texts:
            return None, np.zeros(0, dtype=bool)

        hashes = [_text_hash(t) for t in texts]

        # Batch query with IN clause
        placeholders = ",".join("?" for _ in hashes)
        rows = self.conn.execute(
            f"SELECT text_hash, dim, data FROM embeddings "
            f"WHERE model = ? AND text_hash IN ({placeholders})",
            [model] + hashes,
        ).fetchall()

        if not rows:
            return None, np.zeros(len(texts), dtype=bool)

        # Build lookup {hash -> (dim, data)}
        lookup: dict[str, tuple[int, bytes]] = {}
        for text_hash, dim, data in rows:
            lookup[text_hash] = (dim, data)

        # Determine dimension from first hit
        first_dim = next(iter(lookup.values()))[0]

        embeddings = np.zeros((len(texts), first_dim), dtype=np.float32)
        hit_mask = np.zeros(len(texts), dtype=bool)

        for i, h in enumerate(hashes):
            if h in lookup:
                dim, data = lookup[h]
                embeddings[i] = np.frombuffer(data, dtype=np.float32)
                hit_mask[i] = True

        return embeddings, hit_mask

    def put_batch(
        self,
        model: str,
        texts: list[str],
        embeddings: np.ndarray,
    ) -> int:
        """Store embedding vectors for a list of texts.

        Uses ``INSERT OR IGNORE`` so re-inserting existing keys is a no-op.

        Args:
            model: Embedding model name.
            texts: Source texts (same order as embeddings rows).
            embeddings: ``np.ndarray`` of shape ``(len(texts), dim)``.

        Returns:
            Number of **new** entries actually written.
        """
        if len(texts) == 0:
            return 0

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]
        now = time.time()

        params = [
            (model, _text_hash(text), dim, emb_row.tobytes(), now)
            for text, emb_row in zip(texts, embeddings, strict=True)
        ]

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.executemany(
                "INSERT OR IGNORE INTO embeddings "
                "(model, text_hash, dim, data, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                params,
            )
            written = cursor.rowcount
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise

        if written > 0:
            logger.debug("Embedding cache: stored %d new entries for %s", written, model)

        return written

    # ------------------------------------------------------------------
    # Dimension helper
    # ------------------------------------------------------------------

    def get_dimension(self, model: str) -> int | None:
        """Return the embedding dimension stored for *model*, or ``None``."""
        row = self.conn.execute(
            "SELECT dim FROM embeddings WHERE model = ? LIMIT 1",
            (model,),
        ).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Management / introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dict with keys ``total_entries``, ``total_size_mb``,
            ``db_path``, and ``models`` (a dict of model -> entry count).
        """
        rows = self.conn.execute(
            "SELECT model, COUNT(*), SUM(LENGTH(data)) FROM embeddings GROUP BY model"
        ).fetchall()

        models = {}
        total_entries = 0
        total_bytes = 0
        for model, count, size in rows:
            models[model] = {"entries": count, "size_mb": round(size / 1e6, 2)}
            total_entries += count
            total_bytes += size

        return {
            "total_entries": total_entries,
            "total_size_mb": round(total_bytes / 1e6, 2),
            "db_path": str(self._db_path),
            "models": models,
        }

    def clear(self, model: str | None = None) -> int:
        """Delete cached embeddings.

        Args:
            model: If given, only delete entries for this model.
                   If ``None``, delete everything.

        Returns:
            Number of entries deleted.
        """
        if model is not None:
            cursor = self.conn.execute("DELETE FROM embeddings WHERE model = ?", (model,))
        else:
            cursor = self.conn.execute("DELETE FROM embeddings")

        self.conn.commit()
        deleted = cursor.rowcount
        logger.info("Embedding cache: cleared %d entries", deleted)

        # Reclaim disk space
        self.conn.execute("VACUUM")

        return deleted

    @property
    def db_path(self) -> Path:
        return self._db_path
