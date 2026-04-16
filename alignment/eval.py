from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets."""
    raise NotImplementedError


def build_prompts(examples: Sequence[dict[str, Any]], prompt_template: str) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    raise NotImplementedError


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
) -> dict[str, Any]:
    """Generate model outputs, score them, and return serializable evaluation artifacts."""
    raise NotImplementedError


def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    """Serialize generations and scores for later analysis."""
    raise NotImplementedError


def run_direct_baseline(output_path: Path) -> None:
    """Evaluate the direct-prediction GSM8K baseline from Section 3.1."""
    raise NotImplementedError


def run_cot_baseline(output_path: Path) -> None:
    """Evaluate the chain-of-thought baseline from Section 3.2."""
    raise NotImplementedError


def run_self_consistency_baseline(output_path: Path, k: int = 5) -> None:
    """Evaluate the self-consistency baseline from Section 3.2."""
    raise NotImplementedError


def get_prompt_template(use_cot: bool) -> str:
    return COT_PROMPT_TEMPLATE if use_cot else DIRECT_PROMPT_TEMPLATE
