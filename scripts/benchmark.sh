# small model in forward mode
uv run python systems/benchmark.py --model-size small --mode forward

# small model in forward-backward mode
uv run python systems/benchmark.py --model-size small --mode forward-backward

# small model in train-step mode
uv run python systems/benchmark.py --model-size small --mode train-step

# medium model in forward mode
uv run python systems/benchmark.py --model-size medium --mode forward

# medium model in forward-backward mode
uv run python systems/benchmark.py --model-size medium --mode forward-backward

# large model in forward mode
uv run python systems/benchmark.py --model-size large --mode forward

# large model in forward-backward mode
uv run python systems/benchmark.py --model-size large --mode forward-backward