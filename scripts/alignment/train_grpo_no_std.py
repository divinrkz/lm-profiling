#!/usr/bin/env python3
"""Run GRPO with use_std_normalization=False on GPU 1."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from scripts.alignment.train_grpo import run_one, print_rollout_examples

import json

results = run_one(
    model_name="Qwen/Qwen2.5-Math-1.5B",
    use_std_normalization=False,
    output_dir=REPO_ROOT / "artifacts" / "grpo",
    num_steps=50,
    group_size=8,
    batch_size=4,
    microbatch_size=2,
    lr=1e-6,
    cliprange=0.2,
    max_new_tokens=512,
    val_interval=5,
    val_size=32,
    val_batch_size=4,
    temperature=1.0,
    grad_clip=1.0,
    seed=42,
    reward_fn_name="answer_tag",
)

print_rollout_examples(results, "use_std_normalization=False")
print("DONE no_std")
