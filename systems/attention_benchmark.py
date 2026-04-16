from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    head_dims: tuple[int, ...] = (16, 32, 64, 128)
    sequence_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = 8
    forward_passes: int = 100
    backward_passes: int = 100
    compile_attention: bool = False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations.")
    parser.add_argument("--compile-attention", action="store_true")
    return parser


def iter_benchmark_shapes(config: AttentionBenchmarkConfig) -> Iterable[tuple[int, int]]:
    for head_dim in config.head_dims:
        for sequence_length in config.sequence_lengths:
            yield head_dim, sequence_length


def make_qkv(batch_size: int, sequence_length: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Create random Q, K, and V tensors for the attention benchmark."""
    raise NotImplementedError


def benchmark_attention_once(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict[str, float]:
    """Time the forward and backward pass for a single attention configuration."""
    raise NotImplementedError


def benchmark_attention_grid(config: AttentionBenchmarkConfig) -> list[dict[str, float | int | str]]:
    """Run the attention benchmark over the Section 2.7 Cartesian product of scales."""
    raise NotImplementedError


def main() -> None:
    args = build_argparser().parse_args()
    config = AttentionBenchmarkConfig(compile_attention=args.compile_attention)
    benchmark_attention_grid(config)


if __name__ == "__main__":
    main()
