"""Configuration schemas for RAGiCamp experiments.

This module provides Pydantic models for validating and standardizing
experiment configurations.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragicamp.core.constants import Defaults


class ModelConfig(BaseModel):
    """Model configuration.

    Supports three model types:
    - huggingface: Standard HuggingFace transformers (with optional BitsAndBytes quantization)
    - vllm: vLLM backend with PagedAttention (recommended for long context)
    - openai: OpenAI API models
    """

    type: str = Field(default="huggingface", description="Model type: huggingface, vllm, openai")
    model_name: str = Field(..., description="Model identifier")
    device: str = Field(default="cuda", description="Device (cuda/cpu) - for huggingface only")

    # HuggingFace-specific quantization (BitsAndBytes)
    load_in_8bit: bool = Field(default=False, description="Use 8-bit quantization (HuggingFace)")
    load_in_4bit: bool = Field(default=False, description="Use 4-bit quantization (HuggingFace)")

    # vLLM-specific options
    dtype: str = Field(
        default="bfloat16",
        description="Model dtype for vLLM: 'bfloat16', 'float16', 'float32', 'auto'",
    )
    quantization: str | None = Field(
        default=None,
        description="vLLM quantization: 'awq', 'gptq', 'squeezellm', or None (no quantization)",
    )
    gpu_memory_utilization: float = Field(
        default=0.50,
        description="Fraction of GPU memory to use (vLLM). Default 0.50 leaves room for embedder (0.25).",
    )
    max_model_len: int | None = Field(
        default=None,
        description="Maximum context length (vLLM). If None, uses model's default.",
    )
    tensor_parallel_size: int = Field(
        default=1,
        description="Number of GPUs for tensor parallelism (vLLM).",
    )
    kv_cache_dtype: str | None = Field(
        default=None,
        description="KV cache dtype (vLLM): 'auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'. "
        "Use 'fp8' for ~2x KV cache memory reduction without weight quantization.",
    )
    enable_prefix_caching: bool = Field(
        default=True,
        description="Enable automatic prefix caching for efficiency (vLLM).",
    )

    # Generation parameters
    max_tokens: int | None = Field(default=None, description="Max tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")

    model_config = ConfigDict(extra="allow")


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str = Field(..., description="Dataset name")
    split: str = Field(default="validation", description="Dataset split")
    num_examples: int | None = Field(default=None, description="Number of examples to use")
    filter_no_answer: bool = Field(default=True, description="Filter questions without answers")
    cache_dir: Path = Field(default=Path("data/datasets"), description="Cache directory")

    model_config = ConfigDict(extra="allow")


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""

    strategy: str = Field(
        default="recursive",
        description="Chunking strategy: 'fixed', 'sentence', 'paragraph', 'recursive'",
    )
    chunk_size: int = Field(
        default=Defaults.CHUNK_SIZE, description="Target chunk size in characters"
    )
    chunk_overlap: int = Field(default=Defaults.CHUNK_OVERLAP, description="Overlap between chunks")
    min_chunk_size: int = Field(default=50, description="Minimum chunk size (discard smaller)")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate chunking strategy."""
        allowed = ["fixed", "sentence", "paragraph", "recursive"]
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}, got '{v}'")
        return v


class RetrieverConfig(BaseModel):
    """Retriever configuration."""

    type: str = Field(default="dense", description="Retriever type")
    name: str = Field(default="retriever", description="Retriever name")
    embedding_model: str = Field(default=Defaults.EMBEDDING_MODEL, description="Embedding model")
    index_type: str = Field(default="flat", description="Index type")
    artifact_path: str | None = Field(default=None, description="Path to saved artifact")
    chunking: ChunkingConfig | None = Field(
        default=None, description="Document chunking config (for indexing)"
    )

    model_config = ConfigDict(extra="allow")


class AgentConfig(BaseModel):
    """Agent configuration."""

    type: str = Field(..., description="Agent type")
    name: str = Field(..., description="Agent name")
    system_prompt: str | None = Field(default=None, description="System prompt")
    top_k: int | None = Field(default=5, description="Top-k documents for RAG")

    model_config = ConfigDict(extra="allow")


class MetricConfig(BaseModel):
    """Metric configuration (can be string or dict)."""

    name: str = Field(..., description="Metric name")
    params: dict[str, Any] = Field(default_factory=dict, description="Metric parameters")


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    mode: str = Field(
        default="both",
        description="Evaluation mode: 'generate' (predictions only), 'evaluate' (metrics only), or 'both'",
    )
    batch_size: int | None = Field(default=None, description="Batch size for evaluation")
    num_examples: int | None = Field(default=None, description="Number of examples to evaluate")
    predictions_file: str | None = Field(
        default=None, description="Path to predictions file (required for 'evaluate' mode)"
    )

    # Checkpointing (existing)
    checkpoint_every: int | None = Field(
        default=None, description="Save checkpoint every N examples"
    )
    resume_from_checkpoint: bool = Field(
        default=False, description="Resume from checkpoint if exists"
    )
    retry_failures: bool = Field(default=False, description="Retry failed examples on resume")

    # State management (new)
    save_state: bool = Field(
        default=True, description="Save experiment state for phase-level resumption"
    )
    state_file: str | None = Field(default=None, description="Path to state file")
    force_rerun_phases: list[str] = Field(
        default_factory=list, description="List of phase names to force rerun (e.g., ['metrics'])"
    )

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate mode is one of the allowed values."""
        allowed = ["generate", "evaluate", "both"]
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_predictions_file(self) -> "EvaluationConfig":
        """Ensure predictions_file is set for evaluate mode."""
        if self.mode == "evaluate" and not self.predictions_file:
            raise ValueError("predictions_file must be set when mode='evaluate'")
        return self

    model_config = ConfigDict(extra="allow")


class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""

    enabled: bool = Field(default=True, description="Enable MLflow tracking")
    experiment_name: str | None = Field(default=None, description="MLflow experiment name")
    tracking_uri: str | None = Field(default=None, description="MLflow tracking URI")
    run_name: str | None = Field(default=None, description="MLflow run name")
    tags: dict[str, str] = Field(default_factory=dict, description="MLflow run tags")
    log_artifacts: bool = Field(default=True, description="Log artifacts to MLflow")
    log_models: bool = Field(default=False, description="Log models to MLflow (slower)")


class OutputConfig(BaseModel):
    """Output configuration."""

    save_predictions: bool = Field(default=False, description="Save predictions to file")
    output_path: str | None = Field(default=None, description="Output file path")
    output_dir: Path = Field(default=Path("outputs"), description="Output directory")

    @model_validator(mode="after")
    def validate_output_path(self) -> "OutputConfig":
        """Ensure output_path is set if save_predictions is True."""
        if self.save_predictions and not self.output_path:
            raise ValueError("output_path must be set when save_predictions is True")
        return self


class TrainingConfig(BaseModel):
    """Training configuration."""

    num_epochs: int = Field(default=1, description="Number of training epochs")
    eval_interval: int = Field(default=100, description="Evaluation interval")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    batch_size: int = Field(default=32, description="Training batch size")

    model_config = ConfigDict(extra="allow")


class OptunaConfig(BaseModel):
    """Optuna hyperparameter optimization configuration."""

    enabled: bool = Field(default=False, description="Enable Optuna optimization")
    n_trials: int = Field(default=20, description="Number of optimization trials")
    study_name: str | None = Field(default=None, description="Optuna study name")
    storage: str | None = Field(default=None, description="Optuna storage URI (for persistence)")
    direction: str = Field(
        default="maximize", description="Optimization direction (maximize/minimize)"
    )
    metric_to_optimize: str = Field(default="f1", description="Metric name to optimize")

    # Parameter search spaces (examples)
    search_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter search spaces (e.g., {'top_k': [1, 20], 'temperature': [0.1, 2.0]})",
    )


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    # Fields required only for generate/both modes
    agent: AgentConfig | None = Field(None, description="Agent configuration")
    model: ModelConfig | None = Field(None, description="Model configuration")
    dataset: DatasetConfig | None = Field(None, description="Dataset configuration")

    # Always required
    metrics: list[str | dict[str, Any]] = Field(..., description="Metrics configuration")

    # Optional fields
    judge_model: ModelConfig | None = Field(default=None, description="Judge model for LLM metrics")
    retriever: RetrieverConfig | None = Field(default=None, description="Retriever configuration")
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Evaluation settings"
    )
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output settings")
    training: TrainingConfig | None = Field(default=None, description="Training settings")
    mlflow: MLflowConfig = Field(
        default_factory=MLflowConfig, description="MLflow tracking settings"
    )
    optuna: OptunaConfig | None = Field(default=None, description="Optuna optimization settings")

    @model_validator(mode="after")
    def validate_cross_field_deps(self) -> "ExperimentConfig":
        """Validate cross-field dependencies."""
        mode = self.evaluation.mode if self.evaluation else "both"
        if mode in ["generate", "both"]:
            if not self.agent:
                raise ValueError("agent is required for 'generate' and 'both' modes")
            if not self.model:
                raise ValueError("model is required for 'generate' and 'both' modes")
            if not self.dataset:
                raise ValueError("dataset is required for 'generate' and 'both' modes")

        # Note: retriever requirement for RAG agents is validated at runtime
        # (factory/agents.py), not at config time, since retriever config may
        # come from a separate source (e.g., artifact path).

        return self

    @field_validator("metrics", mode="before")
    @classmethod
    def normalize_metrics(cls, v: list) -> list:
        """Normalize metrics to list of MetricConfig objects."""
        normalized = []
        for metric in v:
            if isinstance(metric, str):
                normalized.append({"name": metric, "params": {}})
            elif isinstance(metric, dict):
                if "name" not in metric:
                    raise ValueError("Metric dict must have 'name' field")
                normalized.append({"name": metric["name"], "params": metric.get("params", {})})
            else:
                raise ValueError(f"Invalid metric type: {type(metric)}")
        return normalized

    model_config = ConfigDict(extra="forbid")


def parse_metric_config(metric: str | dict[str, Any]) -> MetricConfig:
    """Parse a metric configuration.

    Args:
        metric: Either a string (metric name) or dict with name and params

    Returns:
        MetricConfig object
    """
    if isinstance(metric, str):
        return MetricConfig(name=metric, params={})
    elif isinstance(metric, dict):
        return MetricConfig(**metric)
    else:
        raise ValueError(f"Invalid metric type: {type(metric)}")
