from __future__ import annotations

import argparse
import statistics
import timeit
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from basics.model import BasicsTransformerLM
import torch.cuda.nvtx as nvtx

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=512, d_ff=2048, num_layers=8, num_heads=8),
    "medium": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "large": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}

@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    context_length: int = 128
    batch_size: int = 4
    vocab_size: int = 10_000
    warmup_steps: int = 5
    measure_steps: int = 10
    mode: Literal["forward", "forward-backward", "train-step"] = "forward"
    use_bf16: bool = False
    use_memory_profiler: bool = False
    compile_model: bool = False
    output_dir: Path = Path("artifacts")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and profile the Basics transformer.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward-backward", "train-step"], default="forward")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-memory-profiler", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def build_model(config: BenchmarkConfig) -> torch.nn.Module:
    """Instantiate the staff Basics transformer for the requested model size."""
    rope_theta = 10000.0
    model_spec = MODEL_SPECS[config.model_size]
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=model_spec.d_model,
        num_layers=model_spec.num_layers,
        num_heads=model_spec.num_heads,
        d_ff=model_spec.d_ff,  
        rope_theta=rope_theta,
    )
    return model


def make_random_batch(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    """Construct a random token batch for benchmarking and profiling."""
    return torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        dtype=torch.long,
        device=device,
    )


def run_single_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mode: Literal["forward", "forward-backward", "train-step"],
    autocast_context,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """Execute one benchmark step and synchronize CUDA before returning."""
    if mode == "train-step":
        assert optimizer is not None, "train-step mode requires an optimizer"
        optimizer.zero_grad(set_to_none=True)

    with autocast_context:
        logits = model(batch)
        if mode != "forward":
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = batch[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_targets.reshape(-1),
            )

    if mode in ("forward-backward", "train-step"):
        loss.backward()

    if mode == "train-step":
        optimizer.step()

    torch.cuda.synchronize()


def benchmark_model(config: BenchmarkConfig) -> dict[str, float]:
    """Run warmup steps followed by timed measurement steps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config).to(device)
    if config.compile_model:
        model = torch.compile(model)
    if config.mode == "train-step":
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    else:
        if config.mode == "forward":
            model.eval()
        else:
            model.train()
        optimizer = None

    batch = make_random_batch(config, device)
    autocast_context = make_autocast_context(config.use_bf16)

    grad_context = torch.no_grad() if config.mode == "forward" else nullcontext()

    with grad_context:
        for _ in range(config.warmup_steps):
            run_single_step(model, batch, config.mode, autocast_context, optimizer)

        # Start memory recording AFTER warmup so the snapshot reflects
        # steady-state allocation behaviour, not one-time setup.
        maybe_start_memory_history(config.use_memory_profiler)

        step_times: list[float] = []
        with nvtx.range("measure"):
            for i in range(config.measure_steps):
                with nvtx.range(f"step_{i}"):
                    start = timeit.default_timer()
                    run_single_step(model, batch, config.mode, autocast_context, optimizer)
                    end = timeit.default_timer()
                    step_times.append(end - start)

        snapshot_name = (
            f"memory_{config.model_size}_{config.mode}"
            f"_ctx{config.context_length}_bs{config.batch_size}"
            f"_{'bf16' if config.use_bf16 else 'fp32'}.pickle"
        )
        maybe_dump_memory_snapshot(
            config.use_memory_profiler,
            config.output_dir / snapshot_name,
        )

    mean_s = statistics.fmean(step_times)
    stdev_s = statistics.stdev(step_times) if len(step_times) > 1 else 0.0
    results = {
        "mean_step_s": mean_s,
        "stdev_step_s": stdev_s,
        "min_step_s": min(step_times),
        "max_step_s": max(step_times),
        "total_s": sum(step_times),
        "steps_per_s": 1.0 / mean_s if mean_s > 0 else float("inf"),
    }

    print(
        f"[{config.model_size} | {config.mode} | "
        f"ctx={config.context_length} bs={config.batch_size}] "
        f"mean={mean_s * 1e3:.3f} ms  std={stdev_s * 1e3:.3f} ms  "
        f"min={results['min_step_s'] * 1e3:.3f} ms  "
        f"max={results['max_step_s'] * 1e3:.3f} ms  "
        f"({config.measure_steps} steps after {config.warmup_steps} warmup)"
    )
    return results


def annotated_scaled_dot_product_attention(*args, **kwargs):
    """Optional NVTX-annotated attention path for Nsight Systems profiling."""
    raise NotImplementedError


def maybe_start_memory_history(enabled: bool) -> None:
    if enabled:
        if not torch.cuda.is_available():
            print("[memory profiler] CUDA not available; skipping.")
            return
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)


def maybe_dump_memory_snapshot(enabled: bool, output_path: Path) -> None:
    if enabled:
        if not torch.cuda.is_available():
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(output_path))
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"[memory profiler] snapshot written to {output_path}")


def make_autocast_context(use_bf16: bool):
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main() -> None:
    args = build_argparser().parse_args()
    config = BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        use_bf16=args.use_bf16,
        use_memory_profiler=args.use_memory_profiler,
        compile_model=args.compile_model,
        output_dir=args.output_dir,
    )
    benchmark_model(config)


if __name__ == "__main__":
    main()
