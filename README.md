# Profiling and Reasoning

## Repository Layout

- `basics/`: parts of the staff transformer implementation carried forward from the earlier assignment.
- `systems/benchmark.py`: starter scaffold for Sections 2.3-2.6 (benchmarking, Nsight profiling, mixed precision, memory profiling).
- `systems/attention_benchmark.py`: starter scaffold for Sections 2.7-2.8 (attention profiling and `torch.compile`).
- `alignment/prompts.py`: prompt templates for the GSM8K experiments in Section 3.
- `alignment/rewards.py`: provided reward/parsing utilities for tagged GSM8K answers.
- `alignment/eval.py`: starter scaffold for the direct-prediction and zero-shot prompting baselines.
- `alignment/grpo.py`: starter scaffold for the helper methods and GRPO training loop.
- `tests/`: public tests for the Section 3 helper methods that are explicitly assigned in the handout.

## Setup

We use `uv` to manage dependencies. To set up the repository environment, run:

```sh
uv sync
```

The outer `pyproject.toml` points at the local `basics` package, so `uv run ...` should make both the systems and alignment starter code available.

## Notes
- Public tests are only provided for the helper utilities in Sections 3.3 and 3.5.
- The public tests are self-contained; they use lightweight toy fixtures rather than downloading a pretrained model.
- The profiling and evaluation scripts in Section 2 and Section 3.1-3.2 are scaffolded but intentionally unimplemented.
- The provided reward utility in `alignment/rewards.py` expects model outputs to include `<answer>...</answer>` tags, matching the prompts in the handout.
- The Colab notebook for Section 3 installs GPU-only packages such as `vllm` separately; they are intentionally not part of the base `uv sync` environment.


