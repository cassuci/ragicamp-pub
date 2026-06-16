"""Formatting utilities for RAG components."""

from ragicamp.core.types import Document

# Metrics that should be formatted as percentages (0-1 range → 0-100%)
PERCENTAGE_METRICS = frozenset(
    {
        "f1",
        "exact_match",
        "bertscore_f1",
        "bleurt",
        "llm_judge_qa",
    }
)


def format_metrics_summary(metrics: dict[str, float]) -> str:
    """Format a metrics dict into a human-readable summary string.

    Metrics in PERCENTAGE_METRICS are displayed as percentages (e.g. 85.3%),
    all others as raw floats (e.g. 0.123).

    Args:
        metrics: Dict of metric_name -> float value.

    Returns:
        Space-separated summary string, or "no metrics" if empty.
    """
    parts: list[str] = []
    for key, val in metrics.items():
        if not isinstance(val, (int, float)):
            continue
        if key in PERCENTAGE_METRICS:
            parts.append(f"{key}={val * 100:.1f}%")
        else:
            parts.append(f"{key}={val:.3f}")
    return " ".join(parts) if parts else "no metrics"


class ContextFormatter:
    """Utility class for formatting retrieved documents into context strings.

    This centralizes the document formatting logic that was previously
    duplicated across multiple agent implementations.
    """

    @staticmethod
    def format_documents(
        docs: list[Document],
        template: str = "[{idx}] {text}",
        separator: str = "\n\n",
        empty_message: str = "No relevant context found.",
        max_length: int | None = None,
        include_metadata: bool = False,
    ) -> str:
        """Format a list of documents into a context string.

        Args:
            docs: List of Document objects to format
            template: Format template with placeholders: {idx}, {text}, {score}, {id}
            separator: String to join formatted documents
            empty_message: Message to return if no documents
            max_length: Maximum length of each document text (None = no limit)
            include_metadata: Whether to include document metadata

        Returns:
            Formatted context string

        Examples:
            >>> docs = [Document(id="1", text="The sky is blue", metadata={}, score=0.9)]
            >>> ContextFormatter.format_documents(docs)
            '[1] The sky is blue'

            >>> ContextFormatter.format_documents(docs, template="Doc {idx} (score={score:.2f}): {text}")
            'Doc 1 (score=0.90): The sky is blue'
        """
        if not docs:
            return empty_message

        formatted = []
        for i, doc in enumerate(docs, 1):
            # Truncate text if needed
            text = doc.text
            if max_length and len(text) > max_length:
                text = text[:max_length] + "..."

            # Prepare template variables
            template_vars = {
                "idx": i,
                "text": text,
                "id": doc.id,
                "score": doc.score if doc.score is not None else 0.0,
            }

            # Add metadata if requested
            if include_metadata and doc.metadata:
                template_vars.update(doc.metadata)

            # Format this document
            formatted_doc = template.format(**template_vars)
            formatted.append(formatted_doc)

        return separator.join(formatted)

    @staticmethod
    def format_with_scores(
        docs: list[Document], show_score: bool = True, score_format: str = "{score:.3f}"
    ) -> str:
        """Format documents with retrieval scores.

        Args:
            docs: List of Document objects
            show_score: Whether to show scores
            score_format: Format string for scores

        Returns:
            Formatted string with scores

        Example:
            >>> docs = [Document(id="1", text="Example", metadata={}, score=0.856)]
            >>> ContextFormatter.format_with_scores(docs)
            '[1] (score: 0.856) Example'
        """
        if show_score:
            template = f"[{{idx}}] (score: {score_format}) {{text}}"
        else:
            template = "[{idx}] {text}"

        return ContextFormatter.format_documents(docs, template=template)

    @staticmethod
    def format_numbered(docs: list[Document]) -> str:
        """Format documents as clearly labeled retrieved passages.

        Args:
            docs: List of Document objects

        Returns:
            Numbered list of documents with clear passage labels

        Example:
            '--- Passage 1 ---\\nFirst document\\n\\n--- Passage 2 ---\\nSecond document'
        """
        if not docs:
            return "No relevant passages found."

        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"--- Passage {i} ---\n{doc.text}")

        return "\n\n".join(formatted)

    @staticmethod
    def format_numbered_from_docs(docs: list) -> str:
        """Format documents with .text attribute as numbered passages.

        Works with any object that has a .text attribute.

        Args:
            docs: List of document-like objects with .text attribute

        Returns:
            Numbered list of documents
        """
        if not docs:
            return "No relevant passages found."

        formatted = []
        for i, doc in enumerate(docs, 1):
            text = doc.text if hasattr(doc, "text") else str(doc)
            formatted.append(f"--- Passage {i} ---\n{text}")

        return "\n\n".join(formatted)

    @staticmethod
    def format_with_titles(docs: list[Document], title_key: str = "title") -> str:
        """Format documents with titles from metadata.

        Args:
            docs: List of Document objects
            title_key: Metadata key for title

        Returns:
            Formatted string with titles

        Example:
            '[1] Title: Document Title\\nContent: Document text'
        """
        formatted = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get(title_key, f"Document {i}")
            formatted.append(f"[{i}] Title: {title}\nContent: {doc.text}")

        return "\n\n".join(formatted) if formatted else "No relevant context found."
