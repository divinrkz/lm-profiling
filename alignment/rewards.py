from __future__ import annotations

from collections import Counter
from typing import Iterable

from .drgrpo_grader import grade


def extract_answer_from_tags(response: str) -> str | None:
    """Extract the final answer from a model response containing <answer> tags."""
    if "<answer>" not in response or "</answer>" not in response:
        return None
    answer = response.rsplit("<answer>", maxsplit=1)[-1].split("</answer>", maxsplit=1)[0].strip()
    return answer or None


def answer_tag_reward_fn(response: str, ground_truth: str | float | int | list[str], fast: bool = True) -> dict[str, float]:
    """Score a response that is expected to end with a tagged final answer."""
    model_answer = extract_answer_from_tags(response)
    if model_answer is None:
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

    if isinstance(ground_truth, (float, int)):
        ground_truth = str(ground_truth)

    if isinstance(ground_truth, str):
        is_correct = grade(model_answer, ground_truth, fast)
    else:
        is_correct = any(grade(model_answer, candidate, fast) for candidate in ground_truth)

    return {
        "format_reward": 1.0,
        "answer_reward": 1.0 if is_correct else 0.0,
        "reward": 1.0 if is_correct else 0.0,
    }


def majority_vote_tagged_answers(responses: Iterable[str]) -> str | None:
    """Return the most common tagged answer across a self-consistency sample."""
    answers = [answer for answer in (extract_answer_from_tags(response) for response in responses) if answer is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]
