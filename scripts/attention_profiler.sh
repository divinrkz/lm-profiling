#!/usr/bin/env bash
# Compare eager vs torch.compile'd attention across the full Section 2.7 grid.
# Output is saved to artifacts/attention_compile.log for the write-up.
set -e

mkdir -p artifacts
uv run python systems/attention_benchmark.py --compile-attention \
    | tee artifacts/attention_compile.log
