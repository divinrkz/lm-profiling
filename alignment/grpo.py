from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build a response mask over the labels.

    Encodes prompts and responses separately (no special tokens), concatenates
    them, and pads to the longest sequence in the batch.

    Returns a dict with:
        input_ids     (B, L-1)  — prompt+response tokens, last token dropped.
        labels        (B, L-1)  — same tokens shifted left by one.
        response_mask (B, L-1)  — True at every *label* position that belongs
                                   to the response (not the prompt or padding).
    """
    pad_id = tokenizer.pad_token_id

    full_sequences: list[list[int]] = [
        tokenizer.encode(p, add_special_tokens=False)
        + tokenizer.encode(r, add_special_tokens=False)
        for p, r in zip(prompt_strs, output_strs, strict=True)
    ]
    prompt_lens: list[int] = [
        len(tokenizer.encode(p, add_special_tokens=False)) for p in prompt_strs
    ]
    response_lens: list[int] = [
        len(tokenizer.encode(r, add_special_tokens=False)) for r in output_strs
    ]

    # After the causal shift: sequence of length L becomes L-1 (input_ids / labels).
    max_len = max(len(seq) - 1 for seq in full_sequences)

    input_ids_list: list[list[int]] = []
    labels_list: list[list[int]] = []
    response_mask_list: list[list[bool]] = []

    for seq, p_len, r_len in zip(full_sequences, prompt_lens, response_lens):
        seq_len = len(seq) - 1  # length after shift
        pad = [pad_id] * (max_len - seq_len)

        input_ids_list.append(seq[:-1] + pad)
        labels_list.append(seq[1:] + pad)

        # In the labels tensor, position p_len-1 is the first response token
        # (it predicts the 1st response token given the full prompt).
        response_mask_list.append(
            [False] * (p_len - 1)
            + [True] * r_len
            + [False] * (max_len - seq_len)
        )

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "response_mask": torch.tensor(response_mask_list, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropy of the next-token distribution.

    Uses the log-softmax / logsumexp path for numerical stability.

    Args:
        logits: (B, T, V) unnormalised logits.

    Returns:
        (B, T) per-token entropy values.
    """
    log_probs = F.log_softmax(logits, dim=-1)   # (B, T, V)
    probs = torch.exp(log_probs)                 # (B, T, V)
    return -(probs * log_probs).sum(dim=-1)      # (B, T)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Return per-token conditional log-probabilities (and optionally entropy).

    Args:
        model:               HuggingFace causal LM.
        input_ids:           (B, T) token ids fed to the model.
        labels:              (B, T) target token ids (shifted input_ids).
        return_token_entropy: whether to also return per-token entropy.

    Returns dict with:
        "log_probs"     (B, T)  — log p_theta(x_t | x_{<t})
        "token_entropy" (B, T)  — optional, per-token entropy
    """
    logits = model(input_ids).logits              # (B, T, V)
    log_probs_all = F.log_softmax(logits, dim=-1) # (B, T, V)
    log_probs = log_probs_all.gather(             # (B, T)
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)

    out: dict[str, Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)
    return out


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum masked elements along ``dim`` and divide by ``normalize_constant``.

    Positions where ``mask == 0`` do not contribute to the sum.

    Args:
        tensor:             The tensor to aggregate.
        mask:               Boolean / integer mask, same shape as ``tensor``.
        normalize_constant: Denominator for the normalisation.
        dim:                Dimension to sum along; if None, sum over all dims.

    Returns:
        Normalised sum tensor.
    """
    masked = tensor * mask.to(tensor.dtype)
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Score responses, then compute per-group normalised advantages (GRPO).

    The ``rollout_responses`` list is assumed to be arranged as
    [g0_r0, g0_r1, ..., g0_r{K-1},  g1_r0, ...,  g{N-1}_r{K-1}]
    where K == ``group_size``.

    Returns:
        advantages   (B,)  — mean-centred (and optionally std-normalised)
                             reward for each rollout.
        raw_rewards  (B,)  — raw scalar reward for each rollout.
        metadata     dict  — mean/std of rewards and format/answer rates.
    """
    scores: list[dict[str, float]] = [
        reward_fn(resp, gt)
        for resp, gt in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(
        [s["reward"] for s in scores], dtype=torch.float32
    )

    # Group and centre.
    grouped = raw_rewards.view(-1, group_size)          # (N, K)
    centered = grouped - grouped.mean(dim=1, keepdim=True)

    if normalize_by_std:
        std = grouped.std(dim=1, keepdim=True, unbiased=False)
        advantages = centered / (std + advantage_eps)
    else:
        advantages = centered

    advantages = advantages.reshape(-1)

    metadata: dict[str, float] = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std(unbiased=False).item(),
        "mean_format_reward": sum(s.get("format_reward", 0.0) for s in scores) / len(scores),
        "mean_answer_reward": sum(s.get("answer_reward", 0.0) for s in scores) / len(scores),
    }
    return advantages, raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-token GRPO-Clip loss (before masking / averaging).

    Loss = -min(r * A,  clip(r, 1-ε, 1+ε) * A)
    where r = exp(log π_θ - log π_θ_old).

    Args:
        advantages:       (B, 1) or (B, T) group-normalised advantages.
        policy_log_probs: (B, T) log-probs under the current policy.
        old_log_probs:    (B, T) log-probs under the reference (rollout) policy.
        cliprange:        ε for probability-ratio clipping.

    Returns:
        per_token_loss (B, T), metadata dict.
    """
    ratios = torch.exp(policy_log_probs - old_log_probs)          # (B, T)
    clipped = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)
    broadcast_adv = advantages.expand_as(policy_log_probs)        # (B, T)
    per_token_loss = -torch.minimum(
        ratios * broadcast_adv,
        clipped * broadcast_adv,
    )

    metadata: dict[str, Tensor] = {
        "mean_ratio": ratios.mean().detach(),
        "clip_fraction": ((ratios - 1.0).abs() > cliprange).float().mean().detach(),
    }
    return per_token_loss, metadata


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute GRPO-Clip loss for one microbatch and call .backward().

    Gradients are **accumulated** (not zeroed) so this can be called
    ``gradient_accumulation_steps`` times before an optimiser step.
    The loss is divided by ``gradient_accumulation_steps`` so that the
    accumulated gradient is equivalent to a single step on the full batch.

    Args:
        policy_log_probs:           (B, T) log-probs under current policy.
        response_mask:              (B, T) True on response token positions.
        gradient_accumulation_steps: divisor for normalisation.
        advantages:                 (B, 1) per-example advantages.
        old_log_probs:              (B, T) log-probs under rollout policy.
        cliprange:                  ε for ratio clipping.

    Returns:
        loss     scalar loss value (detached).
        metadata dict of diagnostic tensors.
    """
    per_token_loss, clip_metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Mask out non-response tokens; normalise per example by response length,
    # then average over examples; finally divide by gradient_accumulation_steps.
    mask_f = response_mask.to(per_token_loss.dtype)
    per_example_loss = (per_token_loss * mask_f).sum(dim=1) / mask_f.sum(dim=1)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()

    metadata: dict[str, Tensor] = {
        **clip_metadata,
        "loss": loss.detach(),
    }
    return loss.detach(), metadata


def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Build per-example generation logs plus aggregate statistics.

    Logs for each example:
        prompt, response, ground_truth, format_reward, answer_reward, reward,
        token_entropy (mean over response tokens, if provided),
        response_length (tokens, approximated by whitespace split).

    Appended as a final summary entry:
        mean_reward, mean_format_reward, mean_answer_reward,
        mean_response_length, mean_response_length_correct,
        mean_response_length_incorrect, mean_token_entropy (if provided).
    """
    records: list[dict[str, Any]] = []
    lengths: list[int] = []
    correct_lengths: list[int] = []
    incorrect_lengths: list[int] = []
    all_entropies: list[float] = []

    for i, (prompt, response, gt, rewards) in enumerate(
        zip(prompts, responses, ground_truths, reward_infos)
    ):
        resp_len = len(response.split())
        lengths.append(resp_len)

        avg_entropy: float | None = None
        if token_entropies is not None:
            avg_entropy = float(token_entropies[i])
            all_entropies.append(avg_entropy)

        record: dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "ground_truth": gt,
            "format_reward": rewards.get("format_reward", 0.0),
            "answer_reward": rewards.get("answer_reward", 0.0),
            "reward": rewards.get("reward", 0.0),
            "response_length": resp_len,
        }
        if avg_entropy is not None:
            record["token_entropy"] = avg_entropy

        if rewards.get("reward", 0.0) > 0:
            correct_lengths.append(resp_len)
        else:
            incorrect_lengths.append(resp_len)

        records.append(record)

    def _mean(lst: list[float | int]) -> float | None:
        return sum(lst) / len(lst) if lst else None

    summary: dict[str, Any] = {
        "_summary": True,
        "n": len(records),
        "mean_reward": _mean([r["reward"] for r in records]),
        "mean_format_reward": _mean([r["format_reward"] for r in records]),
        "mean_answer_reward": _mean([r["answer_reward"] for r in records]),
        "mean_response_length": _mean(lengths),
        "mean_response_length_correct": _mean(correct_lengths),
        "mean_response_length_incorrect": _mean(incorrect_lengths),
    }
    if all_entropies:
        summary["mean_token_entropy"] = _mean(all_entropies)

    records.append(summary)
    return records


def _generate_responses(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    stop_string: str,
    device: Any,
    gen_batch_size: int = 8,
) -> list[str]:
    """Generate one response per prompt, processing in batches."""
    model.eval()
    stop_ids = tokenizer.encode(stop_string, add_special_tokens=False)
    eos_id = stop_ids[-1] if stop_ids else tokenizer.eos_token_id

    responses: list[str] = []
    for i in range(0, len(prompts), gen_batch_size):
        batch = prompts[i : i + gen_batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)

        for gen_ids in out_ids:
            resp = tokenizer.decode(gen_ids[input_len:], skip_special_tokens=True)
            if stop_string in resp:
                resp = resp[: resp.index(stop_string) + len(stop_string)]
            responses.append(resp)

    return responses


def _validate(
    model: torch.nn.Module,
    tokenizer,
    examples: list[dict[str, Any]],
    reward_fn: Callable[[str, str], dict[str, float]],
    prompt_template: Any,
    device: Any,
    max_new_tokens: int,
    stop_string: str,
    val_batch_size: int = 8,
) -> dict[str, float]:
    """Evaluate answer reward on a validation set (greedy decoding)."""
    prompts = [prompt_template.format(question=ex["question"]) for ex in examples]
    ground_truths = [ex["ground_truth"] for ex in examples]

    if hasattr(model, "config"):
        model.config.use_cache = True
    responses = _generate_responses(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        stop_string=stop_string,
        device=device,
        gen_batch_size=val_batch_size,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    scores = [reward_fn(r, gt) for r, gt in zip(responses, ground_truths)]
    n = len(scores)
    return {
        "mean_reward": sum(s["reward"] for s in scores) / n,
        "mean_answer_reward": sum(s.get("answer_reward", 0.0) for s in scores) / n,
        "mean_format_reward": sum(s.get("format_reward", 0.0) for s in scores) / n,
    }


def train_grpo(
    *,
    model: torch.nn.Module,
    tokenizer,
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    reward_fn: Callable[[str, str], dict[str, float]],
    prompt_template: Any,
    optimizer: torch.optim.Optimizer,
    device: Any = "cuda",
    num_steps: int = 50,
    group_size: int = 8,
    batch_size: int = 4,
    microbatch_size: int | None = None,
    cliprange: float = 0.2,
    advantage_eps: float = 1e-6,
    use_std_normalization: bool = True,
    max_new_tokens: int = 512,
    val_interval: int = 5,
    val_batch_size: int = 8,
    temperature: float = 1.0,
    stop_string: str = "</answer>",
    log_rollouts_interval: int = 10,
    grad_clip: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5.

    For each step:
      1. Sample ``batch_size`` questions; repeat each ``group_size`` times.
      2. Generate one rollout per repeated prompt.
      3. Compute group-normalised advantages via ``compute_group_normalized_rewards``.
      4. Compute old log-probs (no_grad) under the current policy.
      5. Update the policy using ``grpo_microbatch_train_step`` with microbatch
         gradient accumulation.
      6. Periodically run greedy validation and snapshot example rollouts.

    Returns a dict with keys ``train_logs``, ``val_logs``, and ``rollout_logs``.
    """
    import random

    rng = random.Random(seed)
    total_rollouts = batch_size * group_size
    mb_size = microbatch_size if microbatch_size is not None else total_rollouts
    num_microbatches = (total_rollouts + mb_size - 1) // mb_size

    train_logs: list[dict[str, Any]] = []
    val_logs: list[dict[str, Any]] = []
    rollout_logs: list[dict[str, Any]] = []

    for step in range(num_steps):
        # ── 1. Sample questions ────────────────────────────────────────────────
        batch_examples = rng.sample(train_examples, k=batch_size)
        questions = [ex["question"] for ex in batch_examples]
        ground_truths = [ex["ground_truth"] for ex in batch_examples]

        # Repeat each question group_size times (layout: [q0]*K, [q1]*K, ...)
        prompts_rep = [
            prompt_template.format(question=q)
            for q in questions
            for _ in range(group_size)
        ]
        gts_rep = [gt for gt in ground_truths for _ in range(group_size)]

        # ── 2. Generate rollouts ───────────────────────────────────────────────
        # Re-enable KV cache for fast generation (gradient checkpointing disables it).
        if hasattr(model, "config"):
            model.config.use_cache = True
        rollout_responses = _generate_responses(
            model, tokenizer, prompts_rep,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_string=stop_string,
            device=device,
            gen_batch_size=mb_size,
        )
        if hasattr(model, "config"):
            model.config.use_cache = False

        # ── 3. Tokenize & compute old log-probs (in microbatches) ─────────────
        tokenized = tokenize_prompt_and_output(prompts_rep, rollout_responses, tokenizer)
        # Keep on CPU; move microbatches to device on demand.
        input_ids_cpu = tokenized["input_ids"]
        labels_cpu = tokenized["labels"]
        response_mask_cpu = tokenized["response_mask"]

        # Free CUDA memory held by the KV cache from generation before forward pass.
        torch.cuda.empty_cache()

        model.eval()
        old_log_probs_list: list[Tensor] = []
        with torch.no_grad():
            for mb_s in range(0, total_rollouts, mb_size):
                mb_e = min(mb_s + mb_size, total_rollouts)
                mb_in = input_ids_cpu[mb_s:mb_e].to(device)
                mb_lb = labels_cpu[mb_s:mb_e].to(device)
                out = get_response_log_probs(model, mb_in, mb_lb, return_token_entropy=False)
                old_log_probs_list.append(out["log_probs"].cpu())
                del mb_in, mb_lb, out
        old_log_probs_cpu = torch.cat(old_log_probs_list, dim=0)  # (B, T) on CPU
        del old_log_probs_list
        torch.cuda.empty_cache()

        # ── 4. Compute advantages ──────────────────────────────────────────────
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=gts_rep,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )
        # Keep on CPU; slice and move to device per microbatch.
        advantages_cpu = advantages.unsqueeze(1)  # (B, 1)

        # ── 5. Policy update via microbatch accumulation ───────────────────────
        model.train()
        optimizer.zero_grad()

        step_meta_accum: dict[str, list[float]] = {
            "loss": [], "mean_ratio": [], "clip_fraction": []
        }
        for mb_start in range(0, total_rollouts, mb_size):
            mb_end = min(mb_start + mb_size, total_rollouts)
            mb_input = input_ids_cpu[mb_start:mb_end].to(device)
            mb_labels = labels_cpu[mb_start:mb_end].to(device)
            mb_mask = response_mask_cpu[mb_start:mb_end].to(device)
            mb_adv = advantages_cpu[mb_start:mb_end].to(device)
            mb_old = old_log_probs_cpu[mb_start:mb_end].to(device)

            mb_policy = get_response_log_probs(model, mb_input, mb_labels)
            mb_log_probs = mb_policy["log_probs"]

            _, mb_meta = grpo_microbatch_train_step(
                policy_log_probs=mb_log_probs,
                response_mask=mb_mask,
                gradient_accumulation_steps=num_microbatches,
                advantages=mb_adv,
                old_log_probs=mb_old,
                cliprange=cliprange,
            )
            for k in step_meta_accum:
                step_meta_accum[k].append(float(mb_meta[k]))
            del mb_input, mb_labels, mb_mask, mb_adv, mb_old, mb_policy, mb_log_probs

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        )
        optimizer.step()

        # ── 6. Logging ─────────────────────────────────────────────────────────
        log_entry: dict[str, Any] = {
            "step": step,
            **reward_meta,
            "loss": sum(step_meta_accum["loss"]),
            "mean_ratio": sum(step_meta_accum["mean_ratio"]) / num_microbatches,
            "clip_fraction": sum(step_meta_accum["clip_fraction"]) / num_microbatches,
            "grad_norm": grad_norm,
        }
        train_logs.append(log_entry)

        # ── 7. Validation ──────────────────────────────────────────────────────
        if step % val_interval == 0 or step == num_steps - 1:
            val_metrics = _validate(
                model, tokenizer, val_examples, reward_fn,
                prompt_template, device, max_new_tokens, stop_string,
                val_batch_size=val_batch_size,
            )
            val_logs.append({"step": step, **val_metrics})
            print(
                f"[step {step:3d}] train_reward={reward_meta['mean_reward']:.3f}  "
                f"val_answer_reward={val_metrics['mean_answer_reward']:.3f}  "
                f"grad_norm={grad_norm:.3f}  "
                f"clip_frac={log_entry['clip_fraction']:.3f}"
            )

        # ── 8. Rollout snapshots ───────────────────────────────────────────────
        if step % log_rollouts_interval == 0 or step == num_steps - 1:
            scores_all = [
                reward_fn(r, gt) for r, gt in zip(rollout_responses, gts_rep)
            ]
            rollout_entry: dict[str, Any] = {
                "step": step,
                "examples": [
                    {
                        "question": questions[i // group_size],
                        "ground_truth": gts_rep[i],
                        "response": rollout_responses[i],
                        "reward": scores_all[i]["reward"],
                        "format_reward": scores_all[i].get("format_reward", 0.0),
                        "answer_reward": scores_all[i].get("answer_reward", 0.0),
                    }
                    for i in range(min(group_size, total_rollouts))
                ],
            }
            rollout_logs.append(rollout_entry)

    return {
        "train_logs": train_logs,
        "val_logs": val_logs,
        "rollout_logs": rollout_logs,
    }
