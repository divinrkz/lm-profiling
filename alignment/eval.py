"""GSM8K zero-shot evaluation: direct, CoT, and self-consistency baselines.

Usage:
    # Section 3.1 - direct prediction:
    uv run python -m alignment.eval --output-path artifacts/gsm8k_direct.json

    # Section 3.2 - chain-of-thought:
    uv run python -m alignment.eval --use-cot --output-path artifacts/gsm8k_cot.json

    # Section 3.2 - self-consistency (CoT + K=5 majority vote):
    uv run python -m alignment.eval --use-cot --self-consistency --k 5 \
        --output-path artifacts/gsm8k_self_consistency.json

    # Quick smoke test (32 examples):
    uv run python -m alignment.eval --num-examples 32
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .drgrpo_grader import grade, r1_zero_reward_fn
from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE
from .rewards import answer_tag_reward_fn, extract_answer_from_tags

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"

REWARD_FNS: dict[str, Callable[[str, str], dict[str, float]]] = {
    "r1_zero": r1_zero_reward_fn,
    "answer_tag": answer_tag_reward_fn,
}


def _extract_gsm8k_answer(answer_text: str) -> str:
    """Pull the numeric answer after the '####' delimiter in a GSM8K answer field."""
    return answer_text.split("####")[-1].strip()


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets."""
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    return [
        {
            "question": ex["question"],
            "answer": ex["answer"],
            "ground_truth": _extract_gsm8k_answer(ex["answer"]),
        }
        for ex in dataset
    ]


def build_prompts(examples: Sequence[dict[str, Any]], prompt_template: str) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    return [prompt_template.format(question=ex["question"]) for ex in examples]


def _bucket(scores: dict[str, float]) -> str:
    fmt = scores.get("format_reward", 0.0)
    ans = scores.get("answer_reward", 0.0)
    if fmt == 1.0 and ans == 1.0:
        return "correct"
    if fmt == 1.0 and ans == 0.0:
        return "format_only"
    return "neither"


def _summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    counts = {"correct": 0, "format_only": 0, "neither": 0}
    total_format = total_answer = total_reward = 0.0
    for r in records:
        counts[_bucket(r)] += 1
        total_format += r["format_reward"]
        total_answer += r["answer_reward"]
        total_reward += r["reward"]
    return {
        "n": n,
        "accuracy": total_reward / n if n else 0.0,
        "format_accuracy": total_format / n if n else 0.0,
        "answer_given_format": (total_answer / total_format) if total_format else 0.0,
        "counts": counts,
    }


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
    ground_truths: Sequence[str] | None = None,
    questions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate outputs with vLLM, score them, and return serializable artifacts."""
    if ground_truths is None:
        raise ValueError("ground_truths is required to score generations")
    questions = questions if questions is not None else [""] * len(prompts)

    outputs = vllm_model.generate(prompts, eval_sampling_params)

    records: list[dict[str, Any]] = []
    for output, prompt, gt, question in zip(outputs, prompts, ground_truths, questions):
        generation = output.outputs[0].text
        scores = reward_fn(generation, gt)
        records.append(
            {
                "question": question,
                "prompt": prompt,
                "generation": generation,
                "ground_truth": gt,
                "format_reward": float(scores.get("format_reward", 0.0)),
                "answer_reward": float(scores.get("answer_reward", 0.0)),
                "reward": float(scores.get("reward", 0.0)),
            }
        )

    return {"records": records, "metrics": _summarize(records)}


def evaluate_transformers(
    model_name: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    ground_truths: Sequence[str],
    questions: Sequence[str] | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop_string: str = "</answer>",
) -> dict[str, Any]:
    """Transformers fallback for environments without vLLM."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    questions = questions if questions is not None else [""] * len(prompts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    model.eval()

    records: list[dict[str, Any]] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating"):
        batch_prompts = list(prompts[i : i + batch_size])
        batch_gts = list(ground_truths[i : i + batch_size])
        batch_qs = list(questions[i : i + batch_size])

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.encode(stop_string, add_special_tokens=False)[-1],
            )

        input_len = inputs["input_ids"].shape[1]
        for gen_ids, prompt, gt, question in zip(output_ids, batch_prompts, batch_gts, batch_qs):
            generation = tokenizer.decode(gen_ids[input_len:], skip_special_tokens=True)
            if stop_string in generation:
                generation = generation[: generation.index(stop_string) + len(stop_string)]
            scores = reward_fn(generation, gt)
            records.append(
                {
                    "question": question,
                    "prompt": prompt,
                    "generation": generation,
                    "ground_truth": gt,
                    "format_reward": float(scores.get("format_reward", 0.0)),
                    "answer_reward": float(scores.get("answer_reward", 0.0)),
                    "reward": float(scores.get("reward", 0.0)),
                }
            )

    return {"records": records, "metrics": _summarize(records)}


def _summarize_self_consistency(records: Sequence[dict[str, Any]], k: int) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0, "k": k}

    n_correct_vote = sum(1 for r in records if r["answer_reward"] == 1.0)
    n_format_vote = sum(1 for r in records if r["format_reward"] == 1.0)
    n_tie = sum(1 for r in records if r["is_tie"])
    n_no_parse = sum(1 for r in records if r["majority_answer"] is None)

    per_sample_correct = sum(r["per_sample_answer_rate"] for r in records) / n
    per_sample_format = sum(r["per_sample_format_rate"] for r in records) / n

    mode_dist = Counter(r["mode_count"] for r in records)
    return {
        "n": n,
        "k": k,
        "vote_accuracy": n_correct_vote / n,
        "vote_format_accuracy": n_format_vote / n,
        "per_sample_accuracy": per_sample_correct,
        "per_sample_format_accuracy": per_sample_format,
        "tie_rate": n_tie / n,
        "no_parse_rate": n_no_parse / n,
        "mode_count_distribution": dict(sorted(mode_dist.items())),
    }


def _vote_score(
    answers: Sequence[str | None],
    ground_truth: str,
) -> dict[str, Any]:
    """Tally extracted answers, run majority vote, and grade the winner."""
    parsed = [a for a in answers if a is not None and a.strip()]
    if not parsed:
        return {
            "majority_answer": None,
            "vote_counts": {},
            "mode_count": 0,
            "is_tie": False,
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "reward": 0.0,
        }

    counts = Counter(parsed)
    most_common = counts.most_common()
    top_count = most_common[0][1]
    tied = [a for a, c in most_common if c == top_count]
    majority = tied[0]

    is_correct = grade(majority, ground_truth, fast=True)
    return {
        "majority_answer": majority,
        "vote_counts": dict(counts),
        "mode_count": top_count,
        "is_tie": len(tied) > 1,
        "format_reward": 1.0,
        "answer_reward": 1.0 if is_correct else 0.0,
        "reward": 1.0 if is_correct else 0.0,
    }


def evaluate_self_consistency_transformers(
    model_name: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    ground_truths: Sequence[str],
    questions: Sequence[str] | None = None,
    k: int = 5,
    batch_size: int = 4,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop_string: str = "</answer>",
) -> dict[str, Any]:
    """Transformers fallback for self-consistency using num_return_sequences=k."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    questions = questions if questions is not None else [""] * len(prompts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    model.eval()

    records: list[dict[str, Any]] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating (self-consistency)"):
        batch_prompts = list(prompts[i : i + batch_size])
        batch_gts = list(ground_truths[i : i + batch_size])
        batch_qs = list(questions[i : i + batch_size])

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                num_return_sequences=k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.encode(stop_string, add_special_tokens=False)[-1],
            )

        input_len = inputs["input_ids"].shape[1]
        for j, (prompt, gt, question) in enumerate(zip(batch_prompts, batch_gts, batch_qs)):
            samples: list[str] = []
            for s in range(k):
                gen_ids = output_ids[j * k + s]
                generation = tokenizer.decode(gen_ids[input_len:], skip_special_tokens=True)
                if stop_string in generation:
                    generation = generation[: generation.index(stop_string) + len(stop_string)]
                samples.append(generation)

            per_sample_scores = [reward_fn(s, gt) for s in samples]
            extracted = [extract_answer_from_tags(s) for s in samples]
            vote = _vote_score(extracted, gt)

            records.append(
                {
                    "question": question,
                    "prompt": prompt,
                    "samples": samples,
                    "extracted_answers": extracted,
                    "ground_truth": gt,
                    "majority_answer": vote["majority_answer"],
                    "vote_counts": vote["vote_counts"],
                    "mode_count": vote["mode_count"],
                    "is_tie": vote["is_tie"],
                    "format_reward": vote["format_reward"],
                    "answer_reward": vote["answer_reward"],
                    "reward": vote["reward"],
                    "per_sample_format_rate": (
                        sum(s["format_reward"] for s in per_sample_scores) / k
                    ),
                    "per_sample_answer_rate": (
                        sum(s["answer_reward"] for s in per_sample_scores) / k
                    ),
                }
            )

    return {"records": records, "metrics": _summarize_self_consistency(records, k)}


def evaluate_self_consistency_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
    ground_truths: Sequence[str],
    questions: Sequence[str] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Self-consistency with vLLM: sample K generations per prompt, majority-vote."""
    questions = questions if questions is not None else [""] * len(prompts)
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    records: list[dict[str, Any]] = []
    for output, prompt, gt, question in zip(outputs, prompts, ground_truths, questions):
        samples = [s.text for s in output.outputs]
        per_sample_scores = [reward_fn(s, gt) for s in samples]
        extracted = [extract_answer_from_tags(s) for s in samples]
        vote = _vote_score(extracted, gt)

        records.append(
            {
                "question": question,
                "prompt": prompt,
                "samples": samples,
                "extracted_answers": extracted,
                "ground_truth": gt,
                "majority_answer": vote["majority_answer"],
                "vote_counts": vote["vote_counts"],
                "mode_count": vote["mode_count"],
                "is_tie": vote["is_tie"],
                "format_reward": vote["format_reward"],
                "answer_reward": vote["answer_reward"],
                "reward": vote["reward"],
                "per_sample_format_rate": (
                    sum(s["format_reward"] for s in per_sample_scores) / k
                ),
                "per_sample_answer_rate": (
                    sum(s["answer_reward"] for s in per_sample_scores) / k
                ),
            }
        )

    return {"records": records, "metrics": _summarize_self_consistency(records, k)}


def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    """Serialize generations and scores for later analysis."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    records_path = output_path.with_suffix(".jsonl")
    with records_path.open("w", encoding="utf-8") as f:
        for record in results["records"]:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results["metrics"], f, indent=2)

    print(f"Wrote full results to    {output_path}")
    print(f"Wrote per-example JSONL  {records_path}")
    print(f"Wrote metrics summary to {metrics_path}")


def get_prompt_template(use_cot: bool) -> str:
    return COT_PROMPT_TEMPLATE if use_cot else DIRECT_PROMPT_TEMPLATE


def run_baseline(
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    use_cot: bool = False,
    reward_fn_name: str = "r1_zero",
    num_examples: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    seed: int | None = None,
    self_consistency: bool = False,
    k: int = 5,
) -> dict[str, Any]:
    """Evaluate the GSM8K test split for the direct, CoT, or self-consistency baseline."""
    examples = load_gsm8k_examples("test")
    if num_examples is not None:
        examples = examples[:num_examples]

    prompt_template = get_prompt_template(use_cot)
    prompts = build_prompts(examples, prompt_template)
    ground_truths = [ex["ground_truth"] for ex in examples]
    questions = [ex["question"] for ex in examples]
    reward_fn = REWARD_FNS[reward_fn_name]

    mode_label = (
        f"self-consistency (k={k}, CoT)" if self_consistency
        else ("CoT" if use_cot else "direct")
    )
    print(
        f"Evaluating {len(prompts)} GSM8K test examples with model={model_name!r}, "
        f"mode={mode_label}, reward_fn={reward_fn_name}"
    )

    try:
        from vllm import LLM, SamplingParams

        llm = LLM(model=model_name, seed=seed) if seed is not None else LLM(model=model_name)
        sampling_kwargs: dict[str, Any] = dict(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        if self_consistency:
            sampling_kwargs["n"] = k
            sampling_params = SamplingParams(**sampling_kwargs)
            results = evaluate_self_consistency_vllm(
                llm,
                reward_fn,
                prompts,
                sampling_params,
                ground_truths=ground_truths,
                questions=questions,
                k=k,
            )
        else:
            sampling_params = SamplingParams(**sampling_kwargs)
            results = evaluate_vllm(
                llm,
                reward_fn,
                prompts,
                sampling_params,
                ground_truths=ground_truths,
                questions=questions,
            )
    except ModuleNotFoundError:
        print("vLLM not installed — falling back to transformers for generation.")
        if self_consistency:
            results = evaluate_self_consistency_transformers(
                model_name,
                reward_fn,
                prompts,
                ground_truths,
                questions=questions,
                k=k,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            results = evaluate_transformers(
                model_name,
                reward_fn,
                prompts,
                ground_truths,
                questions=questions,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

    metrics = results["metrics"]
    print("\n=== GSM8K Evaluation ===")
    if self_consistency:
        print(f"  n                              = {metrics['n']}")
        print(f"  k (samples per prompt)         = {metrics['k']}")
        print(f"  vote accuracy                  = {metrics['vote_accuracy']:.4f}")
        print(f"  vote format accuracy           = {metrics['vote_format_accuracy']:.4f}")
        print(f"  per-sample accuracy (avg)      = {metrics['per_sample_accuracy']:.4f}")
        print(f"  per-sample format acc (avg)    = {metrics['per_sample_format_accuracy']:.4f}")
        print(f"  tie rate                       = {metrics['tie_rate']:.4f}")
        print(f"  no-parse rate                  = {metrics['no_parse_rate']:.4f}")
        print(f"  mode-count distribution        = {metrics['mode_count_distribution']}")
    else:
        counts = metrics["counts"]
        print(f"  n                              = {metrics['n']}")
        print(f"  accuracy (mean reward)         = {metrics['accuracy']:.4f}")
        print(f"  format accuracy                = {metrics['format_accuracy']:.4f}")
        print(f"  answer | format=1              = {metrics['answer_given_format']:.4f}")
        print(f"  (1) format=1 & answer=1        = {counts['correct']}")
        print(f"  (2) format=1 & answer=0        = {counts['format_only']}")
        print(f"  (3) format=0 & answer=0        = {counts['neither']}")

    write_evaluation_results(results, output_path)
    return results


def run_direct_baseline(output_path: Path) -> None:
    """Evaluate the direct-prediction GSM8K baseline from Section 3.1."""
    run_baseline(output_path, use_cot=False)


def run_cot_baseline(output_path: Path) -> None:
    """Evaluate the chain-of-thought baseline from Section 3.2."""
    run_baseline(output_path, use_cot=True)


def run_self_consistency_baseline(output_path: Path, k: int = 5) -> None:
    """Evaluate the self-consistency baseline from Section 3.2."""
    run_baseline(output_path, use_cot=True, self_consistency=True, k=k)


def _parse_args() -> argparse.Namespace:
    description = (__doc__ or "GSM8K evaluation").splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR / "gsm8k_direct.json",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--use-cot", action="store_true")
    parser.add_argument(
        "--reward-fn",
        default="r1_zero",
        choices=sorted(REWARD_FNS.keys()),
    )
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--self-consistency",
        action="store_true",
        help="Sample K generations per prompt and majority-vote (Section 3.2).",
    )
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_baseline(
        output_path=args.output_path,
        model_name=args.model,
        use_cot=args.use_cot,
        reward_fn_name=args.reward_fn,
        num_examples=args.num_examples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        self_consistency=args.self_consistency,
        k=args.k,
    )


if __name__ == "__main__":
    main()
