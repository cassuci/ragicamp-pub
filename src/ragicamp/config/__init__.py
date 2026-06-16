"""Configuration schemas and validation for experiments."""

from ragicamp.config.schemas import (
    AgentConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    MetricConfig,
    ModelConfig,
    OutputConfig,
    RetrieverConfig,
    TrainingConfig,
)
from ragicamp.config.validation import (
    VALID_DATASETS,
    VALID_PROVIDERS,
    ConfigError,
    validate_config,
    validate_dataset,
    validate_model_spec,
)

__all__ = [
    # Schemas
    "ExperimentConfig",
    "ModelConfig",
    "DatasetConfig",
    "AgentConfig",
    "MetricConfig",
    "RetrieverConfig",
    "EvaluationConfig",
    "OutputConfig",
    "TrainingConfig",
    # Validation
    "ConfigError",
    "validate_config",
    "validate_dataset",
    "validate_model_spec",
    "VALID_DATASETS",
    "VALID_PROVIDERS",
]
