from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from alignment.grpo import (
    compute_entropy,
    compute_group_normalized_rewards,
    compute_grpo_clip_loss,
    get_response_log_probs,
    grpo_microbatch_train_step,
    masked_normalize,
    tokenize_prompt_and_output,
)


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    return tokenize_prompt_and_output(prompt_strs=prompt_strs, output_strs=output_strs, tokenizer=tokenizer)


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    return compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
    )


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    return compute_entropy(logits)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    return get_response_log_probs(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=return_token_entropy,
    )


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )


def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return grpo_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    return masked_normalize(
        tensor=tensor,
        mask=mask,
        dim=dim,
        normalize_constant=normalize_constant,
    )
