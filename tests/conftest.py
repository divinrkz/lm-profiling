from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch


@dataclass
class ToyTokenizer:
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        self._vocab: dict[str, int] = {
            "<pad>": self.pad_token_id,
            "<bos>": self.bos_token_id,
            "<eos>": self.eos_token_id,
        }

    def _token_to_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self._token_to_id(token) for token in text.split()]
        if add_special_tokens:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def __call__(
        self,
        text: str | list[str],
        add_special_tokens: bool = False,
        padding: bool | str = False,
        return_tensors: str | None = None,
    ) -> dict[str, list[int] | list[list[int]] | torch.Tensor]:
        is_single = isinstance(text, str)
        texts = [text] if is_single else list(text)
        encoded = [self.encode(item, add_special_tokens=add_special_tokens) for item in texts]

        if padding:
            max_len = max(len(item) for item in encoded)
            padded = [item + [self.pad_token_id] * (max_len - len(item)) for item in encoded]
            attention_mask = [[1] * len(item) + [0] * (max_len - len(item)) for item in encoded]
        else:
            padded = encoded
            attention_mask = [[1] * len(item) for item in encoded]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

        if is_single:
            return {"input_ids": padded[0], "attention_mask": attention_mask[0]}
        return {"input_ids": padded, "attention_mask": attention_mask}


class ToyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        vocab = torch.arange(self.vocab_size, device=input_ids.device, dtype=torch.float32)
        targets = ((input_ids + 1) % self.vocab_size).to(torch.float32)
        logits = -((vocab.view(1, 1, -1) - targets.unsqueeze(-1)) ** 2) / 5.0
        return SimpleNamespace(logits=logits)


@pytest.fixture
def tokenizer() -> ToyTokenizer:
    return ToyTokenizer()


@pytest.fixture
def model() -> ToyCausalLM:
    return ToyCausalLM(vocab_size=32)


@pytest.fixture
def prompt_strs() -> list[str]:
    return [
        "alpha beta",
        "gamma",
        "delta epsilon zeta",
    ]


@pytest.fixture
def output_strs() -> list[str]:
    return [
        "one two",
        "three four",
        "five",
    ]


@pytest.fixture
def reward_fn():
    def fn(response: str, ground_truth: str) -> dict[str, float]:
        reward = int(response.rsplit("_", maxsplit=1)[-1]) / 10.0
        return {
            "reward": reward,
            "format_reward": reward,
            "answer_reward": reward,
        }

    return fn


@pytest.fixture
def rollout_responses() -> list[str]:
    return [f"resp_{idx}" for idx in range(8)]


@pytest.fixture
def repeated_ground_truths() -> list[str]:
    return ["42"] * 8


@pytest.fixture
def group_size() -> int:
    return 4


@pytest.fixture
def advantage_eps() -> float:
    return 1e-6


@pytest.fixture
def logits() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.0, 1.0, -1.0], [2.0, 0.0, -2.0]],
            [[1.5, -0.5, 0.0], [0.25, 0.75, -1.0]],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def input_ids() -> torch.Tensor:
    return torch.tensor(
        [
            [4, 7, 2, 5],
            [3, 1, 6, 0],
        ],
        dtype=torch.long,
    )


@pytest.fixture
def labels(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [input_ids[:, 1:], torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype)],
        dim=1,
    )


@pytest.fixture
def tensor() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def mask() -> torch.Tensor:
    return torch.tensor(
        [
            [True, False, True],
            [False, True, True],
        ]
    )


@pytest.fixture
def normalize_constant() -> float:
    return 2.5


@pytest.fixture
def policy_log_probs() -> torch.Tensor:
    return torch.tensor(
        [
            [0.2, -0.1, 0.4],
            [-0.3, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def old_log_probs() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, -0.2, 0.1],
            [-0.1, 0.1, 0.4],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def advantages() -> torch.Tensor:
    return torch.tensor([[1.5], [-0.75]], dtype=torch.float32)


@pytest.fixture
def response_mask() -> torch.Tensor:
    return torch.tensor(
        [
            [False, True, True],
            [True, False, True],
        ]
    )


@pytest.fixture
def gradient_accumulation_steps() -> int:
    return 2


@pytest.fixture
def cliprange() -> float:
    return 0.2
