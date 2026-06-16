"""Simple artifact management for saving/loading agents and retrievers."""

import json
import pickle
from pathlib import Path
from typing import Any

from ragicamp.core.logging import get_logger

logger = get_logger(__name__)


class ArtifactManager:
    """Simple manager for saving/loading artifacts (indices, configs, etc.)."""

    def __init__(self, base_dir: str = "artifacts"):
        """Initialize artifact manager.

        Args:
            base_dir: Base directory for all artifacts
        """
        self.base_dir = Path(base_dir)
        self.retrievers_dir = self.base_dir / "retrievers"
        self.agents_dir = self.base_dir / "agents"
        self.indexes_dir = self.base_dir / "indexes"  # Shared embedding indexes

        # Create directories if they don't exist
        self.retrievers_dir.mkdir(parents=True, exist_ok=True)
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

    def get_retriever_path(self, name: str) -> Path:
        """Get path for a retriever artifact.

        Args:
            name: Retriever artifact name (e.g., 'wikipedia_nq_v1')

        Returns:
            Path to retriever directory
        """
        path = self.retrievers_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_agent_path(self, name: str) -> Path:
        """Get path for an agent artifact.

        Args:
            name: Agent artifact name (e.g., 'fixed_rag_wikipedia_v1')

        Returns:
            Path to agent directory
        """
        path = self.agents_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, data: dict[str, Any], path: Path) -> None:
        """Save dictionary as JSON (atomic write).

        Args:
            data: Dictionary to save
            path: Path to save to
        """
        from ragicamp.utils.experiment_io import atomic_write_json

        atomic_write_json(data, path)

    def load_json(self, path: Path) -> dict[str, Any]:
        """Load JSON file.

        Args:
            path: Path to load from

        Returns:
            Loaded dictionary
        """
        with open(path) as f:
            return json.load(f)

    def save_pickle(self, obj: Any, path: Path) -> None:
        """Save object as pickle.

        Args:
            obj: Object to save
            path: Path to save to
        """
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load_pickle(self, path: Path) -> Any:
        """Load pickle file.

        Args:
            path: Path to load from

        Returns:
            Loaded object
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def list_retrievers(self) -> list:
        """List all saved retrievers.

        Returns:
            List of retriever names
        """
        return [d.name for d in self.retrievers_dir.iterdir() if d.is_dir()]

    def index_exists(self, name: str) -> bool:
        """Check if a retriever exists and is valid.

        In the new architecture, retrievers are thin configs that reference indexes.
        - Dense: config.json with embedding_index reference
        - Hybrid: config.json + sparse_*.pkl (BM25 index)
        - Hierarchical: config.json with hierarchical_index reference

        Args:
            name: Retriever artifact name

        Returns:
            True if retriever exists with required files
        """
        path = self.retrievers_dir / name
        config_path = path / "config.json"

        if not config_path.exists():
            return False

        # Check config to determine retriever type
        try:
            config = self.load_json(config_path)
            retriever_type = config.get("type", "dense")
        except Exception:
            return False

        if retriever_type == "hybrid":
            # Hybrid needs sparse index reference (new format) or legacy inline sparse
            sparse_index_name = config.get("sparse_index")
            if sparse_index_name:
                # New format: references shared sparse index
                return self.sparse_index_exists(sparse_index_name)
            else:
                # Legacy format: sparse stored with retriever
                return (path / "sparse_matrix.pkl").exists() and (
                    path / "sparse_vectorizer.pkl"
                ).exists()
        else:
            # Dense/hierarchical just need config with index reference
            has_index_ref = (
                config.get("embedding_index") is not None
                or config.get("hierarchical_index") is not None
            )
            return has_index_ref

    # =========================================================================
    # Shared embedding indexes
    # =========================================================================

    def get_embedding_index_path(self, name: str) -> Path:
        """Get path for a shared embedding index.

        Args:
            name: Embedding index name

        Returns:
            Path to embedding index directory
        """
        path = self.indexes_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def embedding_index_exists(self, name: str) -> bool:
        """Check if a shared index exists (embedding or hierarchical).

        Args:
            name: Index name

        Returns:
            True if index exists with required files
        """
        path = self.indexes_dir / name
        if not path.exists():
            return False

        config_path = path / "config.json"
        if not config_path.exists():
            return False

        # Check config to determine index type
        try:
            config = self.load_json(config_path)
            index_type = config.get("type", "embedding")
        except Exception:
            return False

        if index_type == "hierarchical":
            # Hierarchical index files
            return (
                (path / "child_index.faiss").exists()
                and (path / "parent_docs.pkl").exists()
                and (path / "child_docs.pkl").exists()
                and (path / "child_to_parent.pkl").exists()
            )
        else:
            # Standard embedding index files
            return (path / "index.faiss").exists() and (path / "documents.pkl").exists()

    def list_embedding_indexes(self) -> list:
        """List all saved embedding indexes.

        Returns:
            List of embedding index names
        """
        if not self.indexes_dir.exists():
            return []
        return [d.name for d in self.indexes_dir.iterdir() if d.is_dir()]

    # =========================================================================
    # Shared sparse indexes
    # =========================================================================

    def get_sparse_index_path(self, name: str) -> Path:
        """Get path for a shared sparse index.

        Sparse indexes are stored in indexes/ alongside embedding indexes.

        Args:
            name: Sparse index name (e.g., 'en_bge_large_c512_sparse_tfidf')

        Returns:
            Path to sparse index directory
        """
        path = self.indexes_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sparse_index_exists(self, name: str) -> bool:
        """Check if a sparse index exists.

        Args:
            name: Sparse index name

        Returns:
            True if sparse index exists with required files
        """
        path = self.indexes_dir / name
        if not path.exists():
            return False

        config_path = path / "config.json"
        if not config_path.exists():
            return False

        try:
            config = self.load_json(config_path)
            method = config.get("method", "tfidf")
        except Exception:
            return False

        if method == "tfidf":
            return (path / "vectorizer.pkl").exists() and (path / "doc_vectors.pkl").exists()
        else:  # bm25
            return (path / "bm25.pkl").exists()

    @staticmethod
    def compute_index_name(
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        corpus_version: str = "en",
    ) -> str:
        """Compute a canonical name for an embedding index.

        This ensures indexes with the same config share the same name.

        Args:
            embedding_model: Embedding model name
            chunk_size: Chunk size in characters
            chunk_overlap: Chunk overlap in characters
            corpus_version: Corpus version string

        Returns:
            Canonical index name
        """
        # Normalize model name (remove special chars)
        model_short = embedding_model.replace("/", "_").replace("-", "_").lower()
        # Take last part if it's a path-like name
        if "_" in model_short:
            parts = model_short.split("_")
            # Keep meaningful parts (e.g., "bge_large_en_v1.5" -> "bge_large")
            model_short = "_".join(parts[-2:]) if len(parts) > 2 else model_short

        return f"{corpus_version}_{model_short}_c{chunk_size}_o{chunk_overlap}"

    def list_agents(self) -> list:
        """List all saved agents.

        Returns:
            List of agent names
        """
        return [d.name for d in self.agents_dir.iterdir() if d.is_dir()]


# Global artifact manager instance
_artifact_manager: ArtifactManager | None = None


def get_artifact_manager(base_dir: str = "artifacts") -> ArtifactManager:
    """Get or create global artifact manager.

    Args:
        base_dir: Base directory for artifacts

    Returns:
        ArtifactManager instance
    """
    global _artifact_manager
    if _artifact_manager is None:
        _artifact_manager = ArtifactManager(base_dir)
    elif str(_artifact_manager.base_dir) != base_dir:
        logger.warning(
            "ArtifactManager already initialized with base_dir='%s', ignoring '%s'",
            _artifact_manager.base_dir,
            base_dir,
        )
    return _artifact_manager
