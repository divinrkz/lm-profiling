# inference-only (forward pass only)
uv run python systems/benchmark.py --model-size large --mode forward \
    --use-memory-profiler --measure-steps 3 --warmup-steps 3

# full training step (forward + backward + optimizer)
uv run python systems/benchmark.py --model-size large --mode train-step \
    --use-memory-profiler --measure-steps 3 --warmup-steps 3