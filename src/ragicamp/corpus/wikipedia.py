"""Wikipedia corpus implementation."""

import heapq
from collections.abc import Iterator
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from ragicamp.core.logging import get_logger
from ragicamp.core.types import Document
from ragicamp.corpus.base import CorpusConfig, DocumentCorpus

logger = get_logger(__name__)


def _get_wikirank_cache_path(language: str) -> Path:
    """Get path for cached WikiRank scores."""

    cache_dir = Path.home() / ".cache" / "ragicamp" / "wikirank"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{language}_scores.pkl"


def _load_wikirank_scores(language: str) -> dict:
    """Load WikiRank scores for a language, with caching.

    Returns:
        Dict mapping normalized titles to scores
    """
    import pickle

    import pandas as pd

    cache_path = _get_wikirank_cache_path(language)

    # Try cache first
    if cache_path.exists():
        logger.info("  Loading cached WikiRank scores from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Download from HuggingFace
    url = f"https://huggingface.co/datasets/lewoniewski/wikipedia_quality_wikirank/resolve/main/languages/{language}.csv"

    try:
        logger.info("  Downloading from: %s", url)
        df = pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load WikiRank for language '{language}': {e}\n"
            f"Available languages: en, simple, de, fr, es, etc."
        ) from e

    logger.info("  Loaded %s article scores", len(df))

    # Identify columns - the dataset has: page_id, title, wikirank_score
    title_col = "title" if "title" in df.columns else df.columns[1]
    score_col = next(
        (c for c in ["wikirank_score", "quality", "score"] if c in df.columns),
        df.columns[-1],
    )

    logger.info("  Using columns: title='%s', score='%s'", title_col, score_col)

    # Build title -> score map (normalize titles)
    scores = {}
    for _, row in df.iterrows():
        title = str(row[title_col]).replace("_", " ").strip().lower()
        scores[title] = float(row[score_col])

    # Cache for next time
    with open(cache_path, "wb") as f:
        pickle.dump(scores, f)
    logger.info("  Cached scores to %s", cache_path)

    return scores


def _load_wikirank_top_titles(top_k: int, language: str = "en") -> set[str]:
    """Load top-K article titles from WikiRank dataset.

    WikiRank combines pageviews, edit frequency, references, and link structure
    into a quality/popularity score. Higher scores = more important articles.

    Dataset: lewoniewski/wikipedia_quality_wikirank
    See: https://huggingface.co/datasets/lewoniewski/wikipedia_quality_wikirank

    Args:
        top_k: Number of top articles to keep
        language: Wikipedia language code (e.g., "en", "simple", "de")

    Returns:
        Set of normalized titles (lowercase, spaces not underscores)
    """
    logger.info("Loading WikiRank scores for top %s %s articles...", top_k, language)

    scores = _load_wikirank_scores(language)

    # Find threshold for top-K
    if top_k >= len(scores):
        logger.info("  Requested %s but only %s available, using all", top_k, len(scores))
        return set(scores.keys())

    top_scores = heapq.nlargest(top_k, scores.values())
    threshold = top_scores[-1] if top_scores else 0.0

    # Build allowed set
    allowed = {t for t, s in scores.items() if s >= threshold}
    logger.info("  Filter threshold: %.6f, passing: %s", threshold, len(allowed))

    return allowed


def _normalize_title(title: str) -> str:
    """Normalize title for WikiRank matching."""
    return title.replace("_", " ").strip().lower()


class WikipediaCorpus(DocumentCorpus):
    """Wikipedia corpus using Wikimedia datasets.

    Loads Wikipedia articles from the wikimedia/wikipedia dataset on
    HuggingFace. Supports different language configurations.

    Common configurations:
    - "20231101.simple" - Simple English Wikipedia (~200k articles)
    - "20231101.en" - Full English Wikipedia (~6M articles)

    Example:
        >>> config = CorpusConfig(
        ...     name="wikipedia_simple",
        ...     source="wikimedia/wikipedia",
        ...     version="20231101.simple"
        ... )
        >>> corpus = WikipediaCorpus(config)
        >>> docs = list(corpus.load(max_docs=100))
        >>> print(f"Loaded {len(docs)} articles")
    """

    def __init__(self, config: CorpusConfig):
        """Initialize Wikipedia corpus.

        Args:
            config: Corpus configuration with:
                - source: Should be "wikimedia/wikipedia"
                - version: Wikipedia config (e.g., "20231101.simple")
                - metadata.wikirank_top_k: Optional, filter to top K important articles
        """
        super().__init__(config)
        self._dataset = None
        self._allowed_titles: set[str] | None = None

        # Load WikiRank filter if configured
        top_k = self.config.metadata.get("wikirank_top_k")
        if top_k:
            # Extract language from version (e.g., "20231101.en" -> "en")
            language = self._get_language_from_version()
            self._allowed_titles = _load_wikirank_top_titles(int(top_k), language)

    def load(self, max_docs: int | None = None) -> Iterator[Document]:
        """Load Wikipedia articles.

        Args:
            max_docs: Maximum number of documents to load.
                     Uses config.max_docs if not specified.

        Yields:
            Document objects containing Wikipedia article text.
            Each document has:
            - id: wiki_{index}
            - text: Full article text
            - metadata: {title, url, source}

        Note:
            First call will download the dataset if not cached.
            Set metadata.streaming=False to load full dataset into RAM (faster but ~20GB for en).
        """
        # Determine document limit
        limit = max_docs if max_docs is not None else self.config.max_docs

        # Load dataset - streaming mode by default, can disable for high-RAM machines
        use_streaming = self.config.metadata.get("streaming", True)
        num_proc = self.config.metadata.get("num_proc", 128)  # For parallel loading/filtering

        if self._dataset is None:
            try:
                if use_streaming:
                    logger.info("  Loading Wikipedia (streaming mode)...")
                else:
                    logger.info("  Loading Wikipedia into RAM with %s workers...", num_proc)

                # num_proc only works for non-streaming (sharded loading)
                load_kwargs = {
                    "path": self.config.source,
                    "name": self.config.version,
                    "split": "train",
                    "streaming": use_streaming,
                    "trust_remote_code": False,
                }
                if not use_streaming:
                    load_kwargs["num_proc"] = num_proc

                self._dataset = load_dataset(**load_kwargs)

                if not use_streaming:
                    logger.info("  Loaded %s articles into RAM", len(self._dataset))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Wikipedia corpus '{self.config.version}': {e}\n"
                    f"Available configs: '20231101.simple', '20231101.en', etc."
                ) from e

        # If we have WikiRank filter and non-streaming, use parallel filter
        dataset_to_iterate = self._dataset
        if not use_streaming and self._allowed_titles is not None:
            logger.info("  Filtering with %s workers (this parallelizes the scan)...", num_proc)

            # Need to pass allowed_titles to filter function
            allowed_titles = self._allowed_titles

            def wikirank_filter(example):
                title = example.get("title", "")
                return _normalize_title(title) in allowed_titles

            dataset_to_iterate = self._dataset.filter(
                wikirank_filter,
                num_proc=num_proc,
                desc="WikiRank filter",
            )
            logger.info("  %s articles pass WikiRank filter", len(dataset_to_iterate))

        # Yield documents
        seen_titles = set()
        min_chars = self.config.metadata.get("min_chars", 100)
        collected = 0

        desc = f"Loading {self.config.name}"

        # Determine total for progress bar
        # Use leave=False so bar clears when we pause for batch processing
        if not use_streaming and hasattr(dataset_to_iterate, "__len__"):
            total = min(len(dataset_to_iterate), limit) if limit else len(dataset_to_iterate)
            pbar = tqdm(enumerate(dataset_to_iterate), desc=desc, total=total, leave=False)
        else:
            pbar = tqdm(enumerate(dataset_to_iterate), desc=desc, leave=False)

        for i, article in pbar:
            title = article.get("title", "")
            text = article.get("text", "")

            # WikiRank filter already applied above for non-streaming
            # But for streaming mode, we still need to filter here
            if use_streaming and self._allowed_titles is not None:
                if _normalize_title(title) not in self._allowed_titles:
                    continue

            # Skip if too short or duplicate
            if len(text) < min_chars or title in seen_titles:
                continue

            seen_titles.add(title)
            collected += 1

            # Update progress with collected count when filtering (streaming only)
            if use_streaming and self._allowed_titles is not None:
                pbar.set_postfix(collected=collected, scanned=i + 1)

            # Create document WITHOUT answer information
            yield Document(
                id=f"wiki_{collected}",
                text=text,
                metadata={
                    "title": title,
                    "url": article.get("url", ""),
                    "source": "wikipedia",
                    "corpus": self.config.name,
                },
            )

            # Check limit based on COLLECTED docs, not scanned
            if limit and collected >= limit:
                break

    def get_info(self) -> dict:
        """Get Wikipedia corpus metadata."""
        info = super().get_info()
        info.update(
            {
                "corpus_type": "wikipedia",
                "language": self._get_language_from_version(),
                "streaming": True,
                "wikirank_filter": self._allowed_titles is not None,
                "wikirank_top_k": self.config.metadata.get("wikirank_top_k"),
            }
        )
        return info

    def _get_language_from_version(self) -> str:
        """Extract language code from version string."""
        # Format: "YYYYMMDD.lang" -> extract lang
        parts = self.config.version.split(".")
        return parts[1] if len(parts) > 1 else "unknown"
