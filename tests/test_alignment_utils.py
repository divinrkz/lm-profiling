from __future__ import annotations

import torch

from .adapters import (
    run_compute_entropy as compute_entropy,
    run_get_response_log_probs as get_response_log_probs,
    run_masked_normalize as masked_normalize,
    run_tokenize_prompt_and_output as tokenize_prompt_and_output,
)


def _manual_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def test_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    output = tokenize_prompt_and_output(
        prompt_strs=prompt_strs,
        output_strs=output_strs,
        tokenizer=tokenizer,
    )

    assert set(output) == {"input_ids", "labels", "response_mask"}
    assert output["input_ids"].dtype == torch.long
    assert output["labels"].dtype == torch.long
    assert output["response_mask"].dtype in {torch.bool, torch.uint8, torch.int64, torch.int32}

    full_sequences = [
        tokenizer.encode(prompt, add_special_tokens=False) + tokenizer.encode(response, add_special_tokens=False)
        for prompt, response in zip(prompt_strs, output_strs, strict=True)
    ]
    max_len = max(len(seq) - 1 for seq in full_sequences)

    expected_input_ids = []
    expected_labels = []
    expected_response_mask = []
    for prompt, response, full_sequence in zip(prompt_strs, output_strs, full_sequences, strict=True):
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        response_len = len(tokenizer.encode(response, add_special_tokens=False))
        sequence_len = len(full_sequence) - 1
        pad = [tokenizer.pad_token_id] * (max_len - sequence_len)

        expected_input_ids.append(full_sequence[:-1] + pad)
        expected_labels.append(full_sequence[1:] + pad)
        response_mask = [False] * (prompt_len - 1) + [True] * response_len + [False] * (max_len - sequence_len)
        expected_response_mask.append(response_mask)

    torch.testing.assert_close(output["input_ids"], torch.tensor(expected_input_ids, dtype=torch.long))
    torch.testing.assert_close(output["labels"], torch.tensor(expected_labels, dtype=torch.long))
    torch.testing.assert_close(output["response_mask"].to(torch.bool), torch.tensor(expected_response_mask))


def test_compute_entropy(logits):
    actual = compute_entropy(logits)
    expected = _manual_entropy(logits)
    torch.testing.assert_close(actual, expected)


def test_get_response_log_probs(model, input_ids, labels):
    actual = get_response_log_probs(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=True,
    )

    logits = model(input_ids).logits
    expected_log_probs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    expected_entropy = _manual_entropy(logits)

    assert set(actual) == {"log_probs", "token_entropy"}
    torch.testing.assert_close(actual["log_probs"], expected_log_probs)
    torch.testing.assert_close(actual["token_entropy"], expected_entropy)


def test_masked_normalize_dim0(tensor, mask, normalize_constant):
    actual = masked_normalize(tensor=tensor, mask=mask, normalize_constant=normalize_constant, dim=0)
    expected = (tensor * mask).sum(dim=0) / normalize_constant
    torch.testing.assert_close(actual, expected)


def test_masked_normalize_dim1(tensor, mask, normalize_constant):
    actual = masked_normalize(tensor=tensor, mask=mask, normalize_constant=normalize_constant, dim=1)
    expected = (tensor * mask).sum(dim=1) / normalize_constant
    torch.testing.assert_close(actual, expected)


def test_masked_normalize_dimlast(tensor, mask, normalize_constant):
    actual = masked_normalize(tensor=tensor, mask=mask, normalize_constant=normalize_constant, dim=-1)
    expected = (tensor * mask).sum(dim=-1) / normalize_constant
    torch.testing.assert_close(actual, expected)


def test_masked_normalize_dimNone(tensor, mask, normalize_constant):
    actual = masked_normalize(tensor=tensor, mask=mask, normalize_constant=normalize_constant)
    expected = (tensor * mask).sum() / normalize_constant
    torch.testing.assert_close(actual, expected)
