#!/usr/bin/env python3
"""
Evaluate Qwen 2.5 Math 1.5B zero-shot (direct prediction) on GSM8K test set.

Usage:
    python scripts/alignment/gsm_baseline.py [--output OUTPUT]
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "gsm_direct_baseline.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct-prediction GSM8K baseline")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write JSON results (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(REPO_ROOT))

    from alignment.eval import run_direct_baseline
    run_direct_baseline(args.output)


if __name__ == "__main__":
    main()
