from __future__ import annotations

import torch

from .adapters import (
    run_compute_group_normalized_rewards as compute_group_normalized_rewards,
    run_compute_grpo_clip_loss as compute_grpo_clip_loss,
    run_grpo_microbatch_train_step as grpo_microbatch_train_step,
)


def _expected_group_normalized_rewards(
    rollout_responses: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_rewards = torch.tensor(
        [int(response.rsplit("_", maxsplit=1)[-1]) / 10.0 for response in rollout_responses],
        dtype=torch.float32,
    )
    grouped = raw_rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        normalized = centered / (grouped.std(dim=1, keepdim=True, unbiased=False) + advantage_eps)
    else:
        normalized = centered
    return normalized.reshape(-1), raw_rewards


def _expected_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> torch.Tensor:
    ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1 - cliprange, 1 + cliprange)
    broadcast_advantages = advantages.expand_as(policy_log_probs)
    return -torch.minimum(ratios * broadcast_advantages, clipped_ratios * broadcast_advantages)


def _expected_microbatch_loss(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> torch.Tensor:
    per_token_loss = _expected_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    masked_loss = per_token_loss * response_mask.to(per_token_loss.dtype)
    per_example_loss = masked_loss.sum(dim=1) / response_mask.sum(dim=1)
    return per_example_loss.mean() / gradient_accumulation_steps


def test_compute_group_normalized_rewards_normalize_by_std(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    advantage_eps,
    group_size,
):
    actual_normalized, actual_raw, metadata = compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=True,
    )
    expected_normalized, expected_raw = _expected_group_normalized_rewards(
        rollout_responses=rollout_responses,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=True,
    )
    torch.testing.assert_close(actual_normalized, expected_normalized)
    torch.testing.assert_close(actual_raw, expected_raw)
    assert isinstance(metadata, dict)


def test_compute_group_normalized_rewards_no_normalize_by_std(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    advantage_eps,
    group_size,
):
    actual_normalized, actual_raw, metadata = compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=False,
    )
    expected_normalized, expected_raw = _expected_group_normalized_rewards(
        rollout_responses=rollout_responses,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=False,
    )
    torch.testing.assert_close(actual_normalized, expected_normalized)
    torch.testing.assert_close(actual_raw, expected_raw)
    assert isinstance(metadata, dict)


def test_compute_grpo_clip_loss_large_cliprange(advantages, policy_log_probs, old_log_probs):
    actual, metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=10.0,
    )
    expected = _expected_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=10.0,
    )
    torch.testing.assert_close(actual, expected)
    assert isinstance(metadata, dict)


def test_compute_grpo_clip_loss_small_cliprange(advantages, policy_log_probs, old_log_probs):
    actual, metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=0.1,
    )
    expected = _expected_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=0.1,
    )
    torch.testing.assert_close(actual, expected)
    assert isinstance(metadata, dict)


def test_grpo_microbatch_train_step_grpo_clip(
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
    advantages,
    old_log_probs,
    cliprange,
):
    actual_policy_log_probs = policy_log_probs.clone().requires_grad_(True)
    actual_loss, metadata = grpo_microbatch_train_step(
        policy_log_probs=actual_policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    expected_policy_log_probs = policy_log_probs.clone().requires_grad_(True)
    expected_loss = _expected_microbatch_loss(
        policy_log_probs=expected_policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    expected_loss.backward()

    torch.testing.assert_close(actual_loss, expected_loss.detach())
    torch.testing.assert_close(actual_policy_log_probs.grad, expected_policy_log_probs.grad)
    assert isinstance(metadata, dict)


def test_grpo_microbatch_train_step_grpo_clip_10_steps(
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
    advantages,
    old_log_probs,
    cliprange,
):
    actual_policy_log_probs = policy_log_probs.clone().requires_grad_(True)
    actual_losses = []
    for _ in range(10):
        loss, _ = grpo_microbatch_train_step(
            policy_log_probs=actual_policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
            advantages=advantages,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        actual_losses.append(loss.detach())

    expected_policy_log_probs = policy_log_probs.clone().requires_grad_(True)
    expected_loss = _expected_microbatch_loss(
        policy_log_probs=expected_policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    for _ in range(10):
        expected_loss.backward(retain_graph=True)

    for loss in actual_losses:
        torch.testing.assert_close(loss, expected_loss.detach())
    torch.testing.assert_close(actual_policy_log_probs.grad, expected_policy_log_probs.grad)
