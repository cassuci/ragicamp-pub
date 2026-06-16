"""Optimization module for intelligent experiment search.

Provides Optuna-based hyperparameter optimization as an alternative
to random/stratified sampling for RAG experiment configuration.
"""

from ragicamp.optimization.optuna_search import run_optuna_study

__all__ = ["run_optuna_study"]
