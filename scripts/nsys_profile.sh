uv run nsys profile -o nsys/small_forward -t cuda,nvtx \
    python systems/benchmark.py --model-size small --mode forward
