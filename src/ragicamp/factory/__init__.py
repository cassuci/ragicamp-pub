"""Factory package for creating RAGiCamp components.

Clean architecture:
- ProviderFactory: Creates EmbedderProvider and GeneratorProvider
- AgentFactory: Creates agents with providers + index
- DatasetFactory: Creates evaluation datasets
- MetricFactory: Creates evaluation metrics

Example:
    from ragicamp.factory import ProviderFactory, AgentFactory
    from ragicamp.indexes import VectorIndex

    # Create providers
    embedder = ProviderFactory.create_embedder("BAAI/bge-large-en")
    generator = ProviderFactory.create_generator("vllm:meta-llama/Llama-3.2-3B")

    # Load index
    index = VectorIndex.load("my_index")

    # Create agent
    agent = AgentFactory.create_rag(
        agent_type="fixed_rag",
        name="my_agent",
        embedder_provider=embedder,
        generator_provider=generator,
        index=index,
    )

    # Run
    results = agent.run(queries)
"""

from ragicamp.factory.agents import AgentFactory
from ragicamp.factory.datasets import DatasetFactory
from ragicamp.factory.metrics import MetricFactory
from ragicamp.factory.providers import ProviderFactory

__all__ = [
    # New architecture
    "ProviderFactory",
    "AgentFactory",
    "DatasetFactory",
    "MetricFactory",
]
