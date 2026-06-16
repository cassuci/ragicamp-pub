"""RAG agents - different strategies for retrieval-augmented generation.

Architecture:
- Agent: Base class with run() interface
- Query: Input data type
- Step: Intermediate step capture
- AgentResult: Output with steps for analysis

All agents use model providers for clean lifecycle management.
"""

from ragicamp.agents.base import (
    Agent,
    AgentResult,
    Query,
    RetrievedDocInfo,
    Step,
    StepTimer,
)
from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.agents.fixed_rag import FixedRAGAgent
from ragicamp.agents.iterative_rag import IterativeRAGAgent
from ragicamp.agents.self_rag import SelfRAGAgent

__all__ = [
    # Base types
    "Agent",
    "AgentResult",
    "Query",
    "RetrievedDocInfo",
    "Step",
    "StepTimer",
    # Agents
    "DirectLLMAgent",
    "FixedRAGAgent",
    "IterativeRAGAgent",
    "SelfRAGAgent",
]
