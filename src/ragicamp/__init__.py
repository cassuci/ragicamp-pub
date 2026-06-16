"""RAGiCamp: A modular framework for experimenting with RAG approaches.

Clean Architecture:
- factory: ProviderFactory, AgentFactory, DatasetFactory, MetricFactory
- agents: DirectLLMAgent, FixedRAGAgent, IterativeRAGAgent, SelfRAGAgent
- models: EmbedderProvider, GeneratorProvider
- indexes: VectorIndex, IndexBuilder
- retrievers: HybridSearcher, HierarchicalSearcher
- datasets: QA datasets (NQ, HotpotQA, TriviaQA)
- metrics: Evaluation metrics (F1, EM, BERTScore, LLM-as-judge)

Quick start:
    from ragicamp import Experiment
    from ragicamp.factory import ProviderFactory, AgentFactory
    from ragicamp.indexes import VectorIndex

    # Create providers (lazy loading)
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

# ============================================================================
# Configure TensorFlow BEFORE any library imports
# ============================================================================
import os as _os

if "TF_FORCE_GPU_ALLOW_GROWTH" not in _os.environ:
    _os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
if "TF_CPP_MIN_LOG_LEVEL" not in _os.environ:
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

__version__ = "0.5.0"

from ragicamp.execution import ResilientExecutor
from ragicamp.experiment import (
    Experiment,
    ExperimentCallbacks,
    ExperimentResult,
    run_experiments,
)

# Import from canonical locations
from ragicamp.state import (
    ExperimentHealth,
    ExperimentPhase,
    ExperimentState,
    check_health,
    detect_state,
)

__all__ = [
    # Core
    "Experiment",
    "ExperimentCallbacks",
    "ExperimentResult",
    "run_experiments",
    # State management
    "ExperimentPhase",
    "ExperimentState",
    "ExperimentHealth",
    "check_health",
    "detect_state",
    # Execution
    "ResilientExecutor",
    # Version
    "__version__",
]
