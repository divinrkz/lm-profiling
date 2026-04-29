uv run nsys profile -o nsys/small_forward -t cuda,nvtx --force-overwrite=true \
    python systems/benchmark.py --model-size small --mode forward
