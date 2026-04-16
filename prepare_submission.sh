#!/usr/bin/env bash
set -euo pipefail

uv run pytest -v ./tests --junitxml=test_results.xml || true
echo "Done running tests"

output_file="eecs-148b-hw2-submission.zip"
rm -f "$output_file"

zip -r "$output_file" . \
    -x '*egg-info*' \
    -x '*mypy_cache*' \
    -x '*pytest_cache*' \
    -x '*build*' \
    -x '*ipynb_checkpoints*' \
    -x '*__pycache__*' \
    -x '*.pkl' \
    -x '*.pickle' \
    -x '*.txt' \
    -x '*.log' \
    -x '*.json' \
    -x '*.out' \
    -x '*.err' \
    -x '.git*' \
    -x '.venv/*' \
    -x '*.bin' \
    -x '*.pt' \
    -x '*.pth' \
    -x '*.safetensors'

echo "All files have been compressed into $output_file"
