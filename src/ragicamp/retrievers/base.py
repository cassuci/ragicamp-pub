"""Backward compatibility shim for pickled Document objects.

Old indexes stored Document objects pickled as 'ragicamp.retrievers.base.Document'.
This module re-exports from the new location so unpickling works.

TODO: Run migrate-indexes to convert old indexes, then remove this shim.
"""

from ragicamp.core.types import Document, SearchResult

__all__ = ["Document", "SearchResult"]
