"""Configuration validation utilities.

This module provides validation for study configuration files.
"""

from typing import Any


class ConfigError(ValueError):
    """Configuration validation error."""

    pass


# Valid configuration values
VALID_DATASETS = {"nq", "triviaqa", "hotpotqa", "techqa", "pubmedqa"}
VALID_PROVIDERS = {"hf", "openai", "vllm"}


def validate_model_spec(spec: str) -> None:
    """Validate model specification format.

    Args:
        spec: Model spec like 'hf:google/gemma-2b-it' or 'openai:gpt-4o-mini'

    Raises:
        ConfigError: If spec format is invalid
    """
    if ":" not in spec:
        raise ConfigError(
            f"Invalid model spec: '{spec}'. "
            f"Expected format: 'provider:model_name' "
            f"(e.g., 'hf:google/gemma-2b-it', 'openai:gpt-4o-mini')"
        )
    provider = spec.split(":")[0]
    if provider not in VALID_PROVIDERS:
        raise ConfigError(
            f"Unknown model provider: '{provider}'. Valid providers: {VALID_PROVIDERS}"
        )


def validate_dataset(name: str) -> None:
    """Validate dataset name.

    Args:
        name: Dataset name like 'nq', 'triviaqa', 'hotpotqa'

    Raises:
        ConfigError: If dataset name is invalid
    """
    if name not in VALID_DATASETS:
        raise ConfigError(f"Unknown dataset: '{name}'. Valid datasets: {VALID_DATASETS}")


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate study configuration.

    Args:
        config: Study config dict

    Returns:
        List of warning messages (empty if all valid)

    Raises:
        ConfigError: If required fields are missing or invalid
    """
    warnings = []

    # Required fields
    if "name" not in config:
        raise ConfigError("Config missing required field: 'name'")

    # Validate datasets
    datasets = config.get("datasets", [])
    if not datasets:
        warnings.append("No datasets specified")
    for ds in datasets:
        validate_dataset(ds)

    # Validate direct experiments
    direct = config.get("direct", {})
    if direct.get("enabled"):
        if not direct.get("models"):
            warnings.append("Direct experiments enabled but no models specified")
        for model in direct.get("models", []):
            validate_model_spec(model)

    # Validate RAG experiments
    rag = config.get("rag", {})
    singleton_experiments = config.get("experiments", [])

    if rag.get("enabled"):
        if not rag.get("models"):
            # Only warn if no singleton experiments defined either
            if not singleton_experiments:
                warnings.append(
                    "RAG enabled but no models specified and no singleton experiments defined"
                )
            # If singleton experiments exist, this is intentional (grid search disabled)
        if not rag.get("retrievers"):
            warnings.append("RAG enabled but no retrievers specified")
        for model in rag.get("models", []):
            validate_model_spec(model)

    return warnings
