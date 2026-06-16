"""RAGiCamp CLI - Unified command-line interface.

Usage:
    ragicamp run conf/study/comprehensive_baseline.yaml
    ragicamp index --corpus simple --embedding minilm
    ragicamp compare outputs/
    ragicamp evaluate predictions.json --metrics f1 exact_match
"""

from ragicamp.cli.main import main

__all__ = ["main"]
