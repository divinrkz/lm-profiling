#!/usr/bin/env python3
"""
GRPO training on GSM8K — Sections 3.5 and 3.6.

Trains Qwen2.5-Math-1.5B with GRPO-Clip for 50 steps, comparing
use_std_normalization=True vs. False.

Usage:
    uv run python scripts/alignment/train_grpo.py
    uv run python scripts/alignment/train_grpo.py --num-steps 50 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "grpo"
DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training on GSM8K")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Number of questions per step")
    p.add_argument("--microbatch-size", type=int, default=None,
                   help="Microbatch size for gradient accumulation (default: batch*group)")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--cliprange", type=float, default=0.2)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--val-interval", type=int, default=5)
    p.add_argument("--val-size", type=int, default=64)
    p.add_argument("--val-batch-size", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reward-fn", default="answer_tag",
                   choices=["answer_tag", "r1_zero"])
    p.add_argument("--skip-with-std", action="store_true",
                   help="Skip the use_std_normalization=True run")
    p.add_argument("--skip-no-std", action="store_true",
                   help="Skip the use_std_normalization=False run")
    return p.parse_args()


def run_one(
    *,
    model_name: str,
    use_std_normalization: bool,
    output_dir: Path,
    num_steps: int,
    group_size: int,
    batch_size: int,
    microbatch_size: int | None,
    lr: float,
    cliprange: float,
    max_new_tokens: int,
    val_interval: int,
    val_size: int,
    val_batch_size: int,
    temperature: float,
    grad_clip: float,
    seed: int,
    reward_fn_name: str,
) -> dict:
    """Set up model, data, optimizer and run train_grpo for one configuration."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from alignment.eval import load_gsm8k_examples
    from alignment.grpo import train_grpo
    from alignment.prompts import COT_PROMPT_TEMPLATE
    from alignment.rewards import answer_tag_reward_fn
    from alignment.drgrpo_grader import r1_zero_reward_fn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    label = "with_std" if use_std_normalization else "no_std"
    print(f"Starting GRPO run: use_std_normalization={use_std_normalization}  ({label})")
    print(f"  device={device}  model={model_name}")
    print(f"  steps={num_steps}  batch={batch_size}  group={group_size}")
    print(f"  lr={lr}  cliprange={cliprange}  max_new_tokens={max_new_tokens}")
    print(f"{'='*60}")

    # ── Load model & tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
    except (ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)

    # Enable gradient checkpointing to reduce activation memory during backward.
    model.gradient_checkpointing_enable()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    # ── Data ──────────────────────────────────────────────────────────────────
    rng = random.Random(seed)
    train_examples = load_gsm8k_examples("train")
    test_examples = load_gsm8k_examples("test")

    rng.shuffle(test_examples)
    val_examples = test_examples[:val_size]
    print(f"  train examples={len(train_examples)}  val examples={len(val_examples)}")

    # ── Reward function ────────────────────────────────────────────────────────
    reward_fn = answer_tag_reward_fn if reward_fn_name == "answer_tag" else r1_zero_reward_fn

    # ── Train ──────────────────────────────────────────────────────────────────
    results = train_grpo(
        model=model,
        tokenizer=tokenizer,
        train_examples=train_examples,
        val_examples=val_examples,
        reward_fn=reward_fn,
        prompt_template=COT_PROMPT_TEMPLATE,
        optimizer=optimizer,
        device=device,
        num_steps=num_steps,
        group_size=group_size,
        batch_size=batch_size,
        microbatch_size=microbatch_size,
        cliprange=cliprange,
        advantage_eps=1e-6,
        use_std_normalization=use_std_normalization,
        max_new_tokens=max_new_tokens,
        val_interval=val_interval,
        val_batch_size=val_batch_size,
        temperature=temperature,
        stop_string="</answer>",
        log_rollouts_interval=10,
        grad_clip=grad_clip,
        seed=seed,
    )

    # ── Save results ───────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"grpo_{label}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {out_path}")

    return results


def make_plots(
    results_with: dict,
    results_no: dict,
    output_dir: Path,
) -> None:
    """Generate comparison plots and rollout display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract series ─────────────────────────────────────────────────────────
    def _extract(results, key, source="train_logs"):
        return (
            [e["step"] for e in results[source]],
            [e[key] for e in results[source]],
        )

    # ── Figure 1: Validation reward + training reward comparison ──────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GRPO Training: with-std vs. no-std normalization", fontsize=14, fontweight="bold")

    # Val answer reward
    ax = axes[0, 0]
    x_w, y_w = _extract(results_with, "mean_answer_reward", "val_logs")
    x_n, y_n = _extract(results_no, "mean_answer_reward", "val_logs")
    ax.plot(x_w, y_w, "b-o", markersize=4, label="with std")
    ax.plot(x_n, y_n, "r-o", markersize=4, label="no std")
    ax.set_xlabel("Step")
    ax.set_ylabel("Val Answer Reward")
    ax.set_title("Validation Answer Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Val format reward
    ax = axes[0, 1]
    x_w, y_w = _extract(results_with, "mean_format_reward", "val_logs")
    x_n, y_n = _extract(results_no, "mean_format_reward", "val_logs")
    ax.plot(x_w, y_w, "b-o", markersize=4, label="with std")
    ax.plot(x_n, y_n, "r-o", markersize=4, label="no std")
    ax.set_xlabel("Step")
    ax.set_ylabel("Val Format Reward")
    ax.set_title("Validation Format Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Training mean reward
    ax = axes[1, 0]
    x_w, y_w = _extract(results_with, "mean_reward", "train_logs")
    x_n, y_n = _extract(results_no, "mean_reward", "train_logs")
    ax.plot(x_w, y_w, "b-", alpha=0.8, label="with std")
    ax.plot(x_n, y_n, "r-", alpha=0.8, label="no std")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Mean Reward")
    ax.set_title("Training Mean Reward (rollout)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[1, 1]
    x_w, y_w = _extract(results_with, "grad_norm", "train_logs")
    x_n, y_n = _extract(results_no, "grad_norm", "train_logs")
    ax.plot(x_w, y_w, "b-", alpha=0.8, label="with std")
    ax.plot(x_n, y_n, "r-", alpha=0.8, label="no std")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm (stability)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "grpo_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {fig_path}")

    # ── Figure 2: Extra metrics ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("GRPO Additional Metrics", fontsize=13, fontweight="bold")

    # Loss
    ax = axes[0]
    x_w, y_w = _extract(results_with, "loss", "train_logs")
    x_n, y_n = _extract(results_no, "loss", "train_logs")
    ax.plot(x_w, y_w, "b-", alpha=0.8, label="with std")
    ax.plot(x_n, y_n, "r-", alpha=0.8, label="no std")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title("GRPO-Clip Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Clip fraction
    ax = axes[1]
    x_w, y_w = _extract(results_with, "clip_fraction", "train_logs")
    x_n, y_n = _extract(results_no, "clip_fraction", "train_logs")
    ax.plot(x_w, y_w, "b-", alpha=0.8, label="with std")
    ax.plot(x_n, y_n, "r-", alpha=0.8, label="no std")
    ax.set_xlabel("Step"); ax.set_ylabel("Clip Fraction"); ax.set_title("Clip Fraction")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Mean reward std
    ax = axes[2]
    x_w, y_w = _extract(results_with, "std_reward", "train_logs")
    x_n, y_n = _extract(results_no, "std_reward", "train_logs")
    ax.plot(x_w, y_w, "b-", alpha=0.8, label="with std")
    ax.plot(x_n, y_n, "r-", alpha=0.8, label="no std")
    ax.set_xlabel("Step"); ax.set_ylabel("Reward Std"); ax.set_title("Rollout Reward Std")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    extra_path = output_dir / "grpo_extra_metrics.png"
    plt.savefig(extra_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved extra metrics plot to {extra_path}")

    # ── Figure 3: Validation reward only (clean) ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x_w, y_w = _extract(results_with, "mean_answer_reward", "val_logs")
    x_n, y_n = _extract(results_no, "mean_answer_reward", "val_logs")
    ax.plot(x_w, y_w, "b-o", markersize=5, linewidth=2, label="with std normalization")
    ax.plot(x_n, y_n, "r-o", markersize=5, linewidth=2, label="no std normalization")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Validation Answer Reward", fontsize=12)
    ax.set_title("GRPO Validation Reward: Effect of Std Normalization", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    val_path = output_dir / "grpo_val_reward.png"
    plt.savefig(val_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved validation reward plot to {val_path}")


def print_rollout_examples(results: dict, label: str, n_snapshots: int = 3) -> None:
    """Pretty-print rollout examples at a few training snapshots."""
    rollout_logs = results.get("rollout_logs", [])
    if not rollout_logs:
        return

    indices = [0]
    if len(rollout_logs) > 2:
        indices.append(len(rollout_logs) // 2)
    if len(rollout_logs) > 1:
        indices.append(len(rollout_logs) - 1)

    print(f"\n{'='*70}")
    print(f"Example Rollouts — {label}")
    print(f"{'='*70}")

    for idx in sorted(set(indices)):
        entry = rollout_logs[idx]
        step = entry["step"]
        examples = entry["examples"][:2]  # show 2 examples per snapshot
        print(f"\n--- Step {step} ---")
        for i, ex in enumerate(examples):
            print(f"  [Example {i+1}]")
            print(f"  Question: {ex['question'][:100]}...")
            print(f"  Ground truth: {ex['ground_truth']}")
            resp = ex["response"]
            resp_preview = resp[:300] + ("..." if len(resp) > 300 else "")
            print(f"  Response: {resp_preview}")
            print(f"  Reward: {ex['reward']:.1f}  (format={ex['format_reward']:.1f}, answer={ex['answer_reward']:.1f})")


def main() -> None:
    args = _parse_args()

    results_with: dict | None = None
    results_no: dict | None = None

    # ── Run with std normalization ─────────────────────────────────────────────
    if not args.skip_with_std:
        results_with = run_one(
            model_name=args.model,
            use_std_normalization=True,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            group_size=args.group_size,
            batch_size=args.batch_size,
            microbatch_size=args.microbatch_size,
            lr=args.lr,
            cliprange=args.cliprange,
            max_new_tokens=args.max_new_tokens,
            val_interval=args.val_interval,
            val_size=args.val_size,
            val_batch_size=args.val_batch_size,
            temperature=args.temperature,
            grad_clip=args.grad_clip,
            seed=args.seed,
            reward_fn_name=args.reward_fn,
        )
        print_rollout_examples(results_with, "use_std_normalization=True")
    else:
        # Try to load from disk
        saved = args.output_dir / "grpo_with_std.json"
        if saved.exists():
            with saved.open() as f:
                results_with = json.load(f)
            print(f"Loaded with_std results from {saved}")

    # ── Run without std normalization ──────────────────────────────────────────
    if not args.skip_no_std:
        results_no = run_one(
            model_name=args.model,
            use_std_normalization=False,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            group_size=args.group_size,
            batch_size=args.batch_size,
            microbatch_size=args.microbatch_size,
            lr=args.lr,
            cliprange=args.cliprange,
            max_new_tokens=args.max_new_tokens,
            val_interval=args.val_interval,
            val_size=args.val_size,
            val_batch_size=args.val_batch_size,
            temperature=args.temperature,
            grad_clip=args.grad_clip,
            seed=args.seed,
            reward_fn_name=args.reward_fn,
        )
        print_rollout_examples(results_no, "use_std_normalization=False")
    else:
        saved = args.output_dir / "grpo_no_std.json"
        if saved.exists():
            with saved.open() as f:
                results_no = json.load(f)
            print(f"Loaded no_std results from {saved}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if results_with is not None and results_no is not None:
        make_plots(results_with, results_no, args.output_dir)
    elif results_with is not None:
        _save_single_plot(results_with, "with_std", args.output_dir)
    elif results_no is not None:
        _save_single_plot(results_no, "no_std", args.output_dir)


def _save_single_plot(results: dict, label: str, output_dir: Path) -> None:
    """Save a single-run validation reward plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    val_logs = results.get("val_logs", [])
    if not val_logs:
        return

    steps = [e["step"] for e in val_logs]
    rewards = [e["mean_answer_reward"] for e in val_logs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, rewards, "b-o", markersize=5, linewidth=2)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Validation Answer Reward", fontsize=12)
    ax.set_title(f"GRPO Validation Reward ({label})", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"grpo_val_reward_{label}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
