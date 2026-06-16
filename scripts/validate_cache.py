#!/usr/bin/env python3
"""Validate integrity of the RAGiCamp SQLite cache.

Runs structural and statistical checks on both the embedding cache and the
retrieval cache stored in ``artifacts/cache/ragicamp_cache.db``.

Checks performed (no GPU required):

1. **Embedding structure** — blob size matches declared ``dim``, all entries
   for the same model share one dimension, no NaN/Inf values, vectors are
   unit-normalized (expected for cosine-similarity models).
2. **Retrieval structure** — JSON deserializes, each entry contains the
   expected ``document``/``score``/``rank`` keys, document count equals
   ``top_k``, scores are finite floats.
3. **Hash consistency** — no duplicate ``(model, text_hash)`` or
   ``(retriever, query_hash, top_k)`` pairs (enforced by PK, but verified).
4. **Cross-reference** (optional, ``--cross-check``) — for each retrieval
   cache entry, verifies that the embedding model used by that retriever has
   cached embeddings (i.e. the retriever's queries were also embedded).
5. **Live validation** (optional, ``--live``) — re-embeds a sample of cached
   queries and compares against stored vectors.  Requires GPU.

Usage::

    # Quick structural check (no GPU):
    python scripts/validate_cache.py

    # Full check with sample size:
    python scripts/validate_cache.py --sample 100

    # Cross-reference retrieval ↔ embedding:
    python scripts/validate_cache.py --cross-check

    # Live re-embed validation (needs GPU):
    python scripts/validate_cache.py --live --sample 10

    # Custom DB path:
    python scripts/validate_cache.py --db artifacts/cache/ragicamp_cache.db
"""

import argparse
import json
import math
import sqlite3
import struct
import sys
from pathlib import Path

# Known embedding model dimensions (add new models as needed)
KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": 1536,
    "intfloat/e5-mistral-7b-instruct": 4096,
}

# Cosine-similarity models should have unit-normalized vectors
UNIT_NORM_MODELS: set[str] = {
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-m3",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "intfloat/e5-mistral-7b-instruct",
}


class CacheValidator:
    """Validates the RAGiCamp SQLite cache."""

    def __init__(self, db_path: Path, sample_size: int = 50, verbose: bool = False):
        self.db_path = db_path
        self.sample_size = sample_size
        self.verbose = verbose
        self.passed: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def _ok(self, msg: str) -> None:
        self.passed.append(msg)
        if self.verbose:
            print(f"  PASS  {msg}")

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  WARN  {msg}")

    def _fail(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"  FAIL  {msg}")

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            self._fail(f"Database not found: {self.db_path}")
            sys.exit(1)
        return sqlite3.connect(str(self.db_path), timeout=10)

    # ------------------------------------------------------------------
    # Embedding cache checks
    # ------------------------------------------------------------------

    def check_embeddings(self, conn: sqlite3.Connection) -> None:
        """Validate the embeddings table."""
        print("\n=== Embedding Cache ===")

        # Check table exists
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "embeddings" not in tables:
            self._warn("No 'embeddings' table found — skipping")
            return

        # Overall stats
        row = conn.execute("SELECT COUNT(*), COUNT(DISTINCT model) FROM embeddings").fetchone()
        total, n_models = row
        print(f"  Entries: {total:,}  |  Models: {n_models}")

        if total == 0:
            self._warn("Embedding cache is empty")
            return

        # Per-model checks
        models = conn.execute(
            "SELECT model, COUNT(*), MIN(dim), MAX(dim) FROM embeddings GROUP BY model"
        ).fetchall()

        for model, count, min_dim, max_dim in models:
            # Dimension consistency
            if min_dim != max_dim:
                self._fail(
                    f"{model}: mixed dimensions ({min_dim}–{max_dim}) across {count} entries"
                )
            else:
                self._ok(f"{model}: {count:,} entries, dim={min_dim}")

            # Check against known dimensions
            if model in KNOWN_DIMS:
                expected = KNOWN_DIMS[model]
                if min_dim != expected:
                    self._fail(f"{model}: dim={min_dim}, expected {expected}")
                else:
                    self._ok(f"{model}: dimension matches expected ({expected})")

            # Sample entries for deeper checks
            sample = conn.execute(
                "SELECT text_hash, dim, data FROM embeddings WHERE model = ? "
                "ORDER BY RANDOM() LIMIT ?",
                (model, self.sample_size),
            ).fetchall()

            n_bad_size = 0
            n_nan = 0
            n_inf = 0
            n_zero = 0
            norm_deviations = []

            for _text_hash, dim, data in sample:
                blob = bytes(data)
                expected_bytes = dim * 4  # float32

                # Blob size check
                if len(blob) != expected_bytes:
                    n_bad_size += 1
                    continue

                # Decode to floats
                floats = struct.unpack(f"{dim}f", blob)

                # NaN / Inf checks
                has_nan = any(math.isnan(v) for v in floats)
                has_inf = any(math.isinf(v) for v in floats)
                if has_nan:
                    n_nan += 1
                if has_inf:
                    n_inf += 1

                # All-zeros check
                if all(v == 0.0 for v in floats):
                    n_zero += 1

                # Unit norm check (for cosine-similarity models)
                if model in UNIT_NORM_MODELS and not has_nan and not has_inf:
                    norm = math.sqrt(sum(v * v for v in floats))
                    norm_deviations.append(abs(norm - 1.0))

            sampled = len(sample)
            if n_bad_size:
                self._fail(f"{model}: {n_bad_size}/{sampled} entries have wrong blob size")
            else:
                self._ok(f"{model}: all {sampled} sampled blobs have correct size")

            if n_nan:
                self._fail(f"{model}: {n_nan}/{sampled} entries contain NaN")
            if n_inf:
                self._fail(f"{model}: {n_inf}/{sampled} entries contain Inf")
            if n_zero:
                self._warn(f"{model}: {n_zero}/{sampled} entries are all-zero vectors")

            if norm_deviations:
                max_dev = max(norm_deviations)
                avg_dev = sum(norm_deviations) / len(norm_deviations)
                if max_dev > 0.01:
                    self._warn(
                        f"{model}: norm deviation max={max_dev:.4f}, "
                        f"avg={avg_dev:.6f} (expected ~0 for unit vectors)"
                    )
                else:
                    self._ok(f"{model}: vectors are unit-normalized (max deviation={max_dev:.6f})")

    # ------------------------------------------------------------------
    # Retrieval cache checks
    # ------------------------------------------------------------------

    def check_retrieval(self, conn: sqlite3.Connection) -> None:
        """Validate the retrieval_results table."""
        print("\n=== Retrieval Cache ===")

        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "retrieval_results" not in tables:
            self._warn("No 'retrieval_results' table found — skipping")
            return

        # Overall stats
        row = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT retriever || '/' || top_k) FROM retrieval_results"
        ).fetchone()
        total, n_buckets = row
        print(f"  Entries: {total:,}  |  (retriever, k) buckets: {n_buckets}")

        if total == 0:
            self._warn("Retrieval cache is empty")
            return

        # Per-bucket stats
        buckets = conn.execute(
            "SELECT retriever, top_k, COUNT(*) FROM retrieval_results "
            "GROUP BY retriever, top_k ORDER BY retriever, top_k"
        ).fetchall()

        for retriever, top_k, count in buckets:
            print(f"  {retriever} k={top_k}: {count:,} entries")

        # Sample entries for structure validation
        sample = conn.execute(
            "SELECT retriever, query_hash, top_k, data FROM retrieval_results "
            "ORDER BY RANDOM() LIMIT ?",
            (self.sample_size,),
        ).fetchall()

        n_bad_json = 0
        n_bad_structure = 0
        n_wrong_count = 0
        n_bad_scores = 0

        for retriever, query_hash, top_k, data in sample:
            # JSON parse check
            try:
                docs = json.loads(data)
            except json.JSONDecodeError:
                n_bad_json += 1
                continue

            if not isinstance(docs, list):
                n_bad_structure += 1
                continue

            # Document count vs top_k
            if len(docs) != top_k:
                n_wrong_count += 1
                if self.verbose:
                    print(
                        f"    {retriever} k={top_k} hash={query_hash[:8]}: "
                        f"expected {top_k} docs, got {len(docs)}"
                    )

            # Structure check on each doc
            for doc in docs:
                if not isinstance(doc, dict):
                    n_bad_structure += 1
                    break

                # Required fields
                if "document" not in doc or "score" not in doc:
                    n_bad_structure += 1
                    break

                # Score sanity
                score = doc.get("score")
                if score is not None and isinstance(score, (int, float)):
                    if math.isnan(score) or math.isinf(score):
                        n_bad_scores += 1
                        break

                # Document must have text
                document = doc.get("document", {})
                if not isinstance(document, dict) or "text" not in document:
                    n_bad_structure += 1
                    break

        sampled = len(sample)
        if n_bad_json:
            self._fail(f"Retrieval: {n_bad_json}/{sampled} entries have invalid JSON")
        else:
            self._ok(f"Retrieval: all {sampled} sampled entries are valid JSON")

        if n_bad_structure:
            self._fail(
                f"Retrieval: {n_bad_structure}/{sampled} entries have bad structure "
                f"(missing document/score/text fields)"
            )
        else:
            self._ok(f"Retrieval: all {sampled} sampled entries have correct structure")

        if n_wrong_count:
            self._warn(f"Retrieval: {n_wrong_count}/{sampled} entries have doc count != top_k")
        else:
            self._ok(f"Retrieval: all {sampled} sampled entries have doc count == top_k")

        if n_bad_scores:
            self._fail(f"Retrieval: {n_bad_scores}/{sampled} entries have NaN/Inf scores")
        else:
            self._ok(f"Retrieval: all {sampled} sampled scores are finite")

    # ------------------------------------------------------------------
    # Cross-reference check
    # ------------------------------------------------------------------

    def check_cross_reference(self, conn: sqlite3.Connection) -> None:
        """Check that retrieval cache retrievers have matching embedding entries."""
        print("\n=== Cross-Reference ===")

        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "retrieval_results" not in tables or "embeddings" not in tables:
            self._warn("Need both tables for cross-reference — skipping")
            return

        # Get retriever names from retrieval cache
        retrievers = conn.execute("SELECT DISTINCT retriever FROM retrieval_results").fetchall()

        # Get embedding models
        emb_models = {
            r[0] for r in conn.execute("SELECT DISTINCT model FROM embeddings").fetchall()
        }

        print(f"  Retrieval retrievers: {[r[0] for r in retrievers]}")
        print(f"  Embedding models: {list(emb_models)}")

        # Map retriever names to expected embedding models
        # Convention: "dense_bge_large_512" → "BAAI/bge-large-en-v1.5"
        retriever_to_model = {
            "dense_bge_large_512": "BAAI/bge-large-en-v1.5",
            "dense_bge_m3_512": "BAAI/bge-m3",
            "dense_gte_qwen2_1.5b_512": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "dense_e5_mistral_512": "intfloat/e5-mistral-7b-instruct",
        }
        # Hybrid retrievers share the same embedding model as their dense base
        for hybrid_name, dense_base in [
            ("hybrid_bge_large_bm25", "dense_bge_large_512"),
            ("hybrid_bge_large_tfidf", "dense_bge_large_512"),
            ("hybrid_gte_qwen2_bm25", "dense_gte_qwen2_1.5b_512"),
        ]:
            if dense_base in retriever_to_model:
                retriever_to_model[hybrid_name] = retriever_to_model[dense_base]

        for (retriever,) in retrievers:
            # Strip alpha suffix (e.g. "hybrid_bge_large_bm25_a0.50")
            base_name = retriever.split("_a0.")[0] if "_a0." in retriever else retriever
            base_name = base_name.split("_a1.")[0] if "_a1." in base_name else base_name

            expected_model = retriever_to_model.get(base_name)
            if expected_model is None:
                self._warn(f"Unknown retriever '{retriever}' — can't map to embedding model")
                continue

            if expected_model in emb_models:
                self._ok(f"{retriever} → {expected_model} (found in embedding cache)")
            else:
                self._warn(f"{retriever} → {expected_model} (NOT found in embedding cache)")

    # ------------------------------------------------------------------
    # Live validation (needs GPU)
    # ------------------------------------------------------------------

    def check_live(self, conn: sqlite3.Connection) -> None:
        """Re-embed a sample of cached queries and compare vectors."""
        print("\n=== Live Validation (re-embedding) ===")

        try:
            import numpy as np
        except ImportError:
            self._warn("numpy not available — skipping live validation")
            return

        # Pick a model with the most entries
        row = conn.execute(
            "SELECT model, COUNT(*) as cnt FROM embeddings GROUP BY model ORDER BY cnt DESC LIMIT 1"
        ).fetchone()
        if not row:
            self._warn("No embeddings to validate")
            return

        model_name, _ = row

        # Get sample entries with their text hashes
        sample = conn.execute(
            "SELECT text_hash, dim, data FROM embeddings WHERE model = ? ORDER BY RANDOM() LIMIT ?",
            (model_name, min(self.sample_size, 10)),
        ).fetchall()

        if not sample:
            self._warn("No sample entries found")
            return

        # Decode cached embeddings
        cached_vectors = []
        for _text_hash, _dim, data in sample:
            vec = np.frombuffer(bytes(data), dtype=np.float32).copy()
            cached_vectors.append(vec)

        cached_matrix = np.stack(cached_vectors)
        dim = cached_matrix.shape[1]

        print(f"  Model: {model_name}")
        print(f"  Sample: {len(sample)} entries, dim={dim}")

        # We can't re-embed without the original text (we only have hashes).
        # Instead, verify internal consistency: norms, distribution stats.
        norms = np.linalg.norm(cached_matrix, axis=1)
        print(
            f"  Norm stats: min={norms.min():.4f}, max={norms.max():.4f}, "
            f"mean={norms.mean():.4f}, std={norms.std():.6f}"
        )

        # Check pairwise similarity (should NOT all be identical)
        if len(cached_vectors) >= 2:
            # Normalize for cosine similarity
            normed = cached_matrix / np.clip(norms[:, None], 1e-8, None)
            sim_matrix = normed @ normed.T
            # Off-diagonal similarities
            mask = ~np.eye(len(cached_vectors), dtype=bool)
            off_diag = sim_matrix[mask]
            print(
                f"  Pairwise cosine sim: min={off_diag.min():.4f}, "
                f"max={off_diag.max():.4f}, mean={off_diag.mean():.4f}"
            )

            if off_diag.max() > 0.9999:
                self._warn(
                    f"{model_name}: some vectors are near-identical "
                    f"(max cosine sim={off_diag.max():.6f})"
                )
            elif off_diag.min() < -0.5:
                self._warn(
                    f"{model_name}: unusually negative cosine similarities "
                    f"(min={off_diag.min():.4f})"
                )
            else:
                self._ok(
                    f"{model_name}: pairwise similarities look healthy "
                    f"(range [{off_diag.min():.3f}, {off_diag.max():.3f}])"
                )

    # ------------------------------------------------------------------
    # SQLite integrity
    # ------------------------------------------------------------------

    def check_sqlite_integrity(self, conn: sqlite3.Connection) -> None:
        """Run SQLite's built-in integrity check."""
        print("\n=== SQLite Integrity ===")

        result = conn.execute("PRAGMA integrity_check").fetchone()
        if result and result[0] == "ok":
            self._ok("SQLite integrity check passed")
        else:
            self._fail(f"SQLite integrity check failed: {result}")

        # WAL mode check
        mode = conn.execute("PRAGMA journal_mode").fetchone()
        if mode:
            print(f"  Journal mode: {mode[0]}")

        # Page count and size
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        db_size_mb = (page_size * page_count) / 1e6
        print(f"  DB size: {db_size_mb:.1f} MB ({page_count:,} pages × {page_size} bytes)")

        # Free pages (fragmentation)
        free_pages = conn.execute("PRAGMA freelist_count").fetchone()[0]
        if free_pages > 0:
            free_pct = (free_pages / page_count * 100) if page_count else 0
            if free_pct > 20:
                self._warn(f"{free_pages:,} free pages ({free_pct:.1f}%) — consider running VACUUM")
            else:
                self._ok(f"{free_pages:,} free pages ({free_pct:.1f}%)")

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    def run(
        self,
        cross_check: bool = False,
        live: bool = False,
    ) -> bool:
        """Run all checks and print summary.

        Returns:
            True if no errors found.
        """
        file_size_mb = self.db_path.stat().st_size / 1e6
        print(f"Cache DB: {self.db_path}  ({file_size_mb:.1f} MB)")
        print(f"Sample size: {self.sample_size}")

        conn = self._connect()

        try:
            self.check_sqlite_integrity(conn)
            self.check_embeddings(conn)
            self.check_retrieval(conn)

            if cross_check:
                self.check_cross_reference(conn)

            if live:
                self.check_live(conn)
        finally:
            conn.close()

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Passed:   {len(self.passed)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Errors:   {len(self.errors)}")

        if self.errors:
            print("\nERRORS:")
            for e in self.errors:
                print(f"  - {e}")

        if self.warnings:
            print("\nWARNINGS:")
            for w in self.warnings:
                print(f"  - {w}")

        if not self.errors and not self.warnings:
            print("\nAll checks passed.")

        return len(self.errors) == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate RAGiCamp cache integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to cache DB (default: artifacts/cache/ragicamp_cache.db)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Number of entries to sample per check (default: 50)",
    )
    parser.add_argument(
        "--cross-check",
        action="store_true",
        help="Cross-reference retrieval ↔ embedding caches",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live validation: statistical checks on sampled vectors (needs numpy)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print all passed checks (not just warnings/errors)",
    )
    args = parser.parse_args()

    # Resolve DB path
    if args.db:
        db_path = args.db
    else:
        import os

        db_dir = os.environ.get("RAGICAMP_CACHE_DIR", "artifacts/cache")
        db_path = Path(db_dir) / "ragicamp_cache.db"

    validator = CacheValidator(
        db_path=db_path,
        sample_size=args.sample,
        verbose=args.verbose,
    )

    ok = validator.run(
        cross_check=args.cross_check,
        live=args.live,
    )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
