#!/usr/bin/env bash
# Compare FP32 vs BF16 mixed-precision across all model sizes.
set -e

for size in small medium large; do
  for mode in forward forward-backward; do
    echo "=== ${size} | ${mode} | fp32 ==="
    uv run python systems/benchmark.py --model-size "$size" --mode "$mode"

    echo "=== ${size} | ${mode} | bf16 ==="
    uv run python systems/benchmark.py --model-size "$size" --mode "$mode" --use-bf16
  done
done
