"""Agent factory for creating RAG agents from configuration.

Creates agents with the clean provider-based architecture where:
- Agents receive providers (not loaded models)
- Agents manage their own GPU lifecycle
- Agents use VectorIndex (not Retriever)
"""

from typing import TYPE_CHECKING, Any

from ragicamp.agents import (
    Agent,
    DirectLLMAgent,
    FixedRAGAgent,
    IterativeRAGAgent,
    SelfRAGAgent,
)
from ragicamp.core.logging import get_logger
from ragicamp.indexes.vector_index import VectorIndex
from ragicamp.models.providers import EmbedderProvider, GeneratorProvider

if TYPE_CHECKING:
    from ragicamp.rag.query_transform import QueryTransformer
    from ragicamp.spec import ExperimentSpec

logger = get_logger(__name__)


class AgentFactory:
    """Factory for creating RAG agents from configuration.

    Creates agents with provider-based architecture:

        # Create providers
        embedder = ProviderFactory.create_embedder("BAAI/bge-large-en")
        generator = ProviderFactory.create_generator("vllm:meta-llama/Llama-3.2-3B")

        # Load index
        index = VectorIndex.load("my_index")

        # Create agent
        agent = AgentFactory.create_rag(
            agent_type="fixed_rag",
            embedder_provider=embedder,
            generator_provider=generator,
            index=index,
        )

        # Run (agent manages GPU internally)
        results = agent.run(queries)
    """

    # Custom agent registry
    _custom_agents: dict[str, type] = {}

    # Agent type aliases
    _aliases: dict[str, str] = {
        "pipeline_rag": "fixed_rag",
        "direct": "direct_llm",
        "rag": "fixed_rag",
    }

    @classmethod
    def register(cls, name: str):
        """Register a custom agent type."""

        def decorator(agent_class: type) -> type:
            cls._custom_agents[name] = agent_class
            logger.debug("Registered agent type: %s", name)
            return agent_class

        return decorator

    @classmethod
    def create_direct(
        cls,
        name: str,
        generator_provider: GeneratorProvider,
        **kwargs: Any,
    ) -> DirectLLMAgent:
        """Create a DirectLLMAgent (no retrieval).

        Args:
            name: Agent name
            generator_provider: Generator provider
            **kwargs: Additional agent config

        Returns:
            DirectLLMAgent
        """
        from ragicamp.utils.prompts import PromptBuilder, PromptConfig

        prompt_builder = kwargs.pop("prompt_builder", None) or PromptBuilder(PromptConfig())

        return DirectLLMAgent(
            name=name,
            generator_provider=generator_provider,
            prompt_builder=prompt_builder,
        )

    @classmethod
    def create_rag(
        cls,
        agent_type: str,
        name: str,
        embedder_provider: EmbedderProvider,
        generator_provider: GeneratorProvider,
        index: VectorIndex,
        top_k: int = 5,
        **kwargs: Any,
    ) -> Agent:
        """Create a RAG agent.

        Args:
            agent_type: Type of agent (fixed_rag, iterative_rag, self_rag)
            name: Agent name
            embedder_provider: Embedder provider
            generator_provider: Generator provider
            index: Vector index
            top_k: Number of documents to retrieve
            **kwargs: Additional agent config

        Returns:
            Agent instance
        """
        from ragicamp.utils.prompts import PromptBuilder, PromptConfig

        # Resolve aliases
        agent_type = cls._aliases.get(agent_type, agent_type)

        prompt_builder = kwargs.pop("prompt_builder", None) or PromptBuilder(PromptConfig())

        common_kwargs = {
            "name": name,
            "embedder_provider": embedder_provider,
            "generator_provider": generator_provider,
            "index": index,
            "top_k": top_k,
            "prompt_builder": prompt_builder,
        }
        common_kwargs.update(kwargs)

        if agent_type == "fixed_rag":
            return FixedRAGAgent(**common_kwargs)
        elif agent_type == "iterative_rag":
            return IterativeRAGAgent(**common_kwargs)
        elif agent_type == "self_rag":
            return SelfRAGAgent(**common_kwargs)
        elif agent_type in cls._custom_agents:
            return cls._custom_agents[agent_type](**common_kwargs)
        else:
            available = ["fixed_rag", "iterative_rag", "self_rag"] + list(cls._custom_agents.keys())
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {available}")

    @classmethod
    def from_spec(
        cls,
        spec: "ExperimentSpec",
        embedder_provider: EmbedderProvider,
        generator_provider: GeneratorProvider,
        index: VectorIndex | None = None,
        reranker_provider: Any | None = None,
    ) -> Agent:
        """Create an agent from an ExperimentSpec.

        Args:
            spec: Experiment specification
            embedder_provider: Embedder provider
            generator_provider: Generator provider
            index: Vector index (required for RAG agents)
            reranker_provider: Optional RerankerProvider for cross-encoder reranking

        Returns:
            Configured Agent
        """
        from ragicamp.utils.prompts import PromptBuilder

        # Determine agent type
        agent_type = spec.agent_type
        if not agent_type:
            agent_type = "direct_llm" if spec.exp_type == "direct" else "fixed_rag"

        agent_type = cls._aliases.get(agent_type, agent_type)

        # Build prompt builder
        prompt_builder = PromptBuilder.from_config(spec.prompt, dataset=spec.dataset)

        kwargs: dict[str, Any] = {
            "prompt_builder": prompt_builder,
        }

        # Add agent-specific params from spec
        if spec.agent_params:
            kwargs.update(dict(spec.agent_params))

        if agent_type == "direct_llm":
            return cls.create_direct(
                name=spec.name,
                generator_provider=generator_provider,
                **kwargs,
            )
        else:
            if index is None:
                raise ValueError(f"RAG agent '{agent_type}' requires an index")

            # Create query transformer if configured
            qt = spec.query_transform
            if qt and qt != "none":
                transformer = cls._create_query_transformer(qt, generator_provider)
                kwargs["query_transformer"] = transformer
                logger.info("Query transform enabled: %s", qt)

            # Always enable retrieval cache — the cache keys on query text,
            # so transformed queries get unique cache entries naturally.
            kwargs.update(cls._get_retrieval_cache_kwargs(spec))

            # Pass reranker if configured
            if reranker_provider is not None:
                kwargs["reranker_provider"] = reranker_provider
                # When reranking, retrieve fetch_k docs and rerank down to top_k
                fetch_k = spec.fetch_k
                if fetch_k and fetch_k > spec.top_k:
                    kwargs["fetch_k"] = fetch_k

            return cls.create_rag(
                agent_type=agent_type,
                name=spec.name,
                embedder_provider=embedder_provider,
                generator_provider=generator_provider,
                index=index,
                top_k=spec.top_k,
                **kwargs,
            )

    @staticmethod
    def _create_query_transformer(
        qt_name: str,
        generator_provider: GeneratorProvider,
    ) -> "QueryTransformer":
        """Create a query transformer from its name.

        Args:
            qt_name: Transformer type ('hyde' or 'multiquery').
            generator_provider: Generator provider for LLM calls.

        Returns:
            Configured QueryTransformer instance.
        """
        from ragicamp.rag.query_transform import HyDETransformer, MultiQueryTransformer

        if qt_name == "hyde":
            return HyDETransformer(generator_provider)
        elif qt_name == "multiquery":
            return MultiQueryTransformer(generator_provider)
        else:
            raise ValueError(
                f"Unknown query_transform: '{qt_name}'. Available: 'hyde', 'multiquery', 'none'"
            )

    @staticmethod
    def _get_retrieval_cache_kwargs(spec: "ExperimentSpec") -> dict[str, Any]:
        """Build retrieval cache kwargs if caching is enabled.

        Returns ``retrieval_store`` and ``retriever_name`` kwargs for the
        agent constructor, or an empty dict if caching is disabled.
        """
        import os

        if os.environ.get("RAGICAMP_CACHE", "1") != "1":
            return {}

        try:
            from ragicamp.cache.retrieval_store import RetrievalStore

            store = RetrievalStore.default()
            retriever_name = spec.retriever or "unknown"
            # Include alpha in cache key for hybrid retrievers to avoid stale results
            if spec.alpha is not None:
                retriever_name = f"{retriever_name}_a{spec.alpha:.2f}"
            logger.info(
                "Retrieval cache enabled (retriever=%s, db=%s)",
                retriever_name,
                store.db_path,
            )
            return {
                "retrieval_store": store,
                "retriever_name": retriever_name,
            }
        except Exception:
            logger.warning(
                "Failed to enable retrieval cache, falling back to uncached",
                exc_info=True,
            )
            return {}

    @classmethod
    def get_available_agents(cls) -> list[str]:
        """Get list of all available agent types."""
        built_in = ["direct_llm", "fixed_rag", "iterative_rag", "self_rag"]
        custom = list(cls._custom_agents.keys())
        return built_in + custom
