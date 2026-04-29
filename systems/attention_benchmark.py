from __future__ import annotations

import argparse
import gc
import timeit
from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from basics.model import scaled_dot_product_attention


# Type alias: f(Q, K, V) -> attention output.
AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    head_dims: tuple[int, ...] = (16, 32, 64, 128)
    sequence_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = 8
    forward_passes: int = 100
    backward_passes: int = 100
    warmup_passes: int = 5
    # Number of extra warmup passes to run when the attention fn was produced
    # by torch.compile -- the first call triggers compilation (can take seconds)
    # and the second call sometimes recompiles, so we want both behind us.
    compile_warmup_passes: int = 10
    compile_attention: bool = False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations.")
    parser.add_argument(
        "--compile-attention",
        action="store_true",
        help="Also benchmark torch.compile'd attention and print side-by-side.",
    )
    return parser


def iter_benchmark_shapes(config: AttentionBenchmarkConfig) -> Iterable[tuple[int, int]]:
    for head_dim in config.head_dims:
        for sequence_length in config.sequence_lengths:
            yield head_dim, sequence_length


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_qkv(
    batch_size: int,
    sequence_length: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random Q, K, and V tensors for the attention benchmark.

    Shapes are (batch_size, sequence_length, head_dim) -- num_heads = 1 by
    omitting the head dimension entirely. requires_grad=True so the backward
    pass has something to differentiate.
    """
    Q = torch.randn(batch_size, sequence_length, head_dim, device=device, requires_grad=True)
    K = torch.randn(batch_size, sequence_length, head_dim, device=device, requires_grad=True)
    V = torch.randn(batch_size, sequence_length, head_dim, device=device, requires_grad=True)
    return Q, K, V


def benchmark_attention_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_fn: AttentionFn,
    forward_passes: int,
    backward_passes: int,
    warmup_passes: int,
) -> dict[str, float]:
    """Time the forward and backward pass for a single attention configuration."""
    for _ in range(warmup_passes):
        out = attention_fn(q, k, v)
        out.sum().backward()
        q.grad = k.grad = v.grad = None
    _sync()

    fwd_times: list[float] = []
    for _ in range(forward_passes):
        start = timeit.default_timer()
        out = attention_fn(q, k, v)
        _sync()
        fwd_times.append(timeit.default_timer() - start)
        del out

    out = attention_fn(q, k, v)
    loss = out.sum()
    _sync()
    pre_bwd_mem_mib = (
        torch.cuda.memory_allocated(q.device) / (1024**2)
        if q.is_cuda
        else 0.0
    )

    # Backward timing. Autograd frees the graph after each backward, so we
    # rebuild the forward each iteration; only the .backward() call is timed.
    bwd_times: list[float] = []
    for i in range(backward_passes):
        if i > 0:
            out = attention_fn(q, k, v)
            loss = out.sum()
        start = timeit.default_timer()
        loss.backward()
        _sync()
        bwd_times.append(timeit.default_timer() - start)
        q.grad = k.grad = v.grad = None

    return {
        "fwd_mean_ms": 1e3 * sum(fwd_times) / len(fwd_times),
        "fwd_min_ms": 1e3 * min(fwd_times),
        "bwd_mean_ms": 1e3 * sum(bwd_times) / len(bwd_times),
        "bwd_min_ms": 1e3 * min(bwd_times),
        "pre_bwd_mem_mib": pre_bwd_mem_mib,
    }


def _benchmark_one_config(
    config: AttentionBenchmarkConfig,
    head_dim: int,
    seq_len: int,
    device: torch.device,
    attention_fn: AttentionFn,
    backend: str,
) -> dict[str, float | int | str]:
    """Run one (head_dim, seq_len, backend) config; returns a row dict."""
    warmup = config.warmup_passes + (
        config.compile_warmup_passes if backend == "compiled" else 0
    )
    try:
        q, k, v = make_qkv(config.batch_size, seq_len, head_dim, device)
        stats = benchmark_attention_once(
            q,
            k,
            v,
            attention_fn=attention_fn,
            forward_passes=config.forward_passes,
            backward_passes=config.backward_passes,
            warmup_passes=warmup,
        )
        return {"d_model": head_dim, "seq_len": seq_len, "backend": backend, **stats, "error": ""}
    except torch.cuda.OutOfMemoryError as e:
        return {
            "d_model": head_dim,
            "seq_len": seq_len,
            "backend": backend,
            "error": f"OOM: {str(e).splitlines()[0]}",
        }
    finally:
        for name in ("q", "k", "v"):
            if name in locals():
                del locals()[name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def benchmark_attention_grid(
    config: AttentionBenchmarkConfig,
) -> list[dict[str, float | int | str]]:
    """Run the attention benchmark over the Section 2.7 Cartesian product of scales.

    If `config.compile_attention` is True, each (d_model, seq_len) is benchmarked
    twice -- once with eager attention and once with `torch.compile`'d attention --
    so the two can be compared side-by-side.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: running on CPU; results will not reflect GPU behaviour.")

    backends: list[tuple[str, AttentionFn]] = [("eager", scaled_dot_product_attention)]
    if config.compile_attention:
        # `dynamic=False` lets torch.compile specialize for each shape, which
        # tends to produce the fastest fused kernels at the cost of recompiling
        # when shapes change. Recompile is fine here because we sweep shapes
        # explicitly.
        compiled_attention = torch.compile(scaled_dot_product_attention, dynamic=False)
        backends.append(("compiled", compiled_attention))

    rows: list[dict[str, float | int | str]] = []
    for head_dim, seq_len in iter_benchmark_shapes(config):
        for backend, attention_fn in backends:
            print(
                f"--> d_model={head_dim:>4d}  seq_len={seq_len:>5d}  backend={backend} ...",
                flush=True,
            )
            rows.append(
                _benchmark_one_config(config, head_dim, seq_len, device, attention_fn, backend)
            )

    _print_table(rows, include_compiled=config.compile_attention)
    return rows


def _print_table(
    rows: list[dict[str, float | int | str]],
    include_compiled: bool,
) -> None:
    if not include_compiled:
        header = (
            f"{'d_model':>8} {'seq_len':>8} "
            f"{'fwd (ms)':>10} {'bwd (ms)':>10} {'mem (MiB)':>11}  notes"
        )
        print()
        print(header)
        print("-" * len(header))
        for row in rows:
            if row.get("error"):
                print(
                    f"{row['d_model']:>8} {row['seq_len']:>8} "
                    f"{'-':>10} {'-':>10} {'-':>11}  {row['error']}"
                )
            else:
                print(
                    f"{row['d_model']:>8} {row['seq_len']:>8} "
                    f"{row['fwd_mean_ms']:>10.3f} {row['bwd_mean_ms']:>10.3f} "
                    f"{row['pre_bwd_mem_mib']:>11.1f}"
                )
        return

    # Side-by-side eager vs compiled.
    by_key: dict[tuple[int, int], dict[str, dict]] = {}
    for row in rows:
        key = (int(row["d_model"]), int(row["seq_len"]))
        by_key.setdefault(key, {})[str(row["backend"])] = row

    header = (
        f"{'d_model':>7} {'seq_len':>7} | "
        f"{'fwd eager':>10} {'fwd comp':>10} {'fwd x':>6} | "
        f"{'bwd eager':>10} {'bwd comp':>10} {'bwd x':>6} | "
        f"{'mem (MiB)':>10}"
    )
    print()
    print(header)
    print("-" * len(header))

    def fmt(value: float | str | None, width: int, prec: int = 3) -> str:
        if value is None or value == "":
            return f"{'-':>{width}}"
        if isinstance(value, str):
            return f"{value:>{width}}"
        return f"{value:>{width}.{prec}f}"

    for (d_model, seq_len), entries in sorted(by_key.items()):
        eager = entries.get("eager")
        compiled = entries.get("compiled")

        if eager and eager.get("error"):
            print(f"{d_model:>7} {seq_len:>7} | {eager['error']}")
            continue
        if compiled and compiled.get("error"):
            note = compiled["error"]
        else:
            note = ""

        f_e = eager["fwd_mean_ms"] if eager and not eager.get("error") else None
        f_c = compiled["fwd_mean_ms"] if compiled and not compiled.get("error") else None
        b_e = eager["bwd_mean_ms"] if eager and not eager.get("error") else None
        b_c = compiled["bwd_mean_ms"] if compiled and not compiled.get("error") else None
        mem = eager["pre_bwd_mem_mib"] if eager and not eager.get("error") else None

        f_speedup = (f_e / f_c) if (f_e and f_c) else None
        b_speedup = (b_e / b_c) if (b_e and b_c) else None

        print(
            f"{d_model:>7} {seq_len:>7} | "
            f"{fmt(f_e, 10)} {fmt(f_c, 10)} {fmt(f_speedup, 6, 2)} | "
            f"{fmt(b_e, 10)} {fmt(b_c, 10)} {fmt(b_speedup, 6, 2)} | "
            f"{fmt(mem, 10, 1)}"
            + (f"  {note}" if note else "")
        )


def main() -> None:
    args = build_argparser().parse_args()
    config = AttentionBenchmarkConfig(compile_attention=args.compile_attention)
    benchmark_attention_grid(config)


if __name__ == "__main__":
    main()
