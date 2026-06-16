"""Path and directory utilities."""

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to directory (or file - will use parent dir)

    Returns:
        Path object for the directory

    Example:
        >>> ensure_dir("outputs/experiments/run1")
        PosixPath('outputs/experiments/run1')

        >>> ensure_dir("outputs/results.json")  # Creates outputs/
        PosixPath('outputs')
    """
    path = Path(path)

    # If path looks like a file (has extension), use parent
    if path.suffix:
        directory = path.parent
    else:
        directory = path

    # Create directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def ensure_output_dirs() -> None:
    """Ensure common output directories exist.

    Creates standard directories used by RAGiCamp:
    - outputs/
    - outputs/experiments/
    - outputs/comparisons/
    - artifacts/
    - artifacts/retrievers/
    - artifacts/agents/
    - data/
    - data/datasets/
    """
    common_dirs = [
        "outputs",
        "outputs/experiments",
        "outputs/comparisons",
        "artifacts",
        "artifacts/retrievers",
        "artifacts/agents",
        "data",
        "data/datasets",
    ]

    for dir_path in common_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def safe_write_json(data: dict, path: str | Path, **kwargs) -> Path:
    """Write JSON to file atomically, ensuring directory exists.

    Uses atomic write (temp file + rename) to prevent corruption on crash.

    Args:
        data: Dictionary to write as JSON
        path: Path to output file
        **kwargs: Additional arguments passed to json.dump

    Returns:
        Path object for the written file

    Example:
        >>> safe_write_json({"key": "value"}, "outputs/data.json")
        PosixPath('outputs/data.json')
    """
    from ragicamp.utils.experiment_io import atomic_write_json

    path = Path(path)
    atomic_write_json(data, path, **kwargs)
    return path


def get_project_root() -> Path:
    """Get the project root directory.

    Looks for the directory containing pyproject.toml or setup.py.

    Returns:
        Path to project root
    """
    current = Path.cwd()

    # Look for project markers
    markers = ["pyproject.toml", "setup.py", ".git"]

    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent

    # Fallback to current directory
    return current
