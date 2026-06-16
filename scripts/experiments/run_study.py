#!/usr/bin/env python3
"""Run a study from a YAML config file.

This script is a thin wrapper around the ragicamp CLI.
For the full implementation, see: src/ragicamp/cli/study.py

Usage:
    python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml
    python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml --dry-run
    python scripts/experiments/run_study.py conf/study/comprehensive_baseline.yaml --skip-existing

Or use the CLI directly:
    ragicamp run conf/study/comprehensive_baseline.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp.cli.study import run_study


def main():
    parser = argparse.ArgumentParser(description="Run study from config")
    parser.add_argument("config", type=Path, help="Study config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--skip-existing", action="store_true", help="Skip completed")

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_study(config, dry_run=args.dry_run, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
