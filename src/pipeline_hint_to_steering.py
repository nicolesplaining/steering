#!/usr/bin/env python3
"""
End-to-end pipeline:
1) Sample ~N wrong examples
2) Generate hints and re-run model
3) Build steering vector from improved-after-hint items
4) Evaluate vanilla vs steered on the sampled set
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import SteeringVector, train_steering_vector

from baseline_eval import (
    extract_boxed_answer,
    extract_ground_truth,
    generate_response,
    is_correct,
    load_model,
)
from hint_and_rerun import build_hint_prompt, build_hinted_prompt, generate_hint

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    if TYPE_CHECKING:
        from openai import OpenAI  # pragma: no cover
    else:
        OpenAI = Any  # type: ignore[misc,assignment]


SYSTEM_PROMPT = "You are a helpful math assistant."


@dataclass
class Example:
    item: Dict[str, Any]
    hint: str
    rerun: Dict[str, Any]
    improved: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline: hints -> rerun -> steering -> eval."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input wrong JSON (e.g. baseline_eval_..._wrong.json).",
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Base directory for outputs (run subfolder will be created).",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=40,
        help="Number of wrong examples to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4.1",
        help="OpenAI model for hint generation (default: gpt-4.1).",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Qwen model for rerun + steering.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep between OpenAI requests (seconds).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Force model device (default: auto).",
    )
    parser.add_argument(
        "--mock-hints",
        action="store_true",
        help="Use a placeholder hint instead of calling OpenAI.",
    )
    parser.add_argument(
        "--mock-hint-text",
        default="Focus on the key equation or symmetry; re-check arithmetic.",
        help="Hint text used when --mock-hints is set.",
    )
    parser.add_argument(
        "--use-solution-hint",
        action="store_true",
        help="Use the ground-truth solution text directly as the hint.",
    )
    parser.add_argument(
        "--layers",
        default="",
        help="Comma-separated layer indices (default: mid-to-last).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Steering multiplier strength.",
    )
    parser.add_argument(
        "--start-after-tokens",
        type=int,
        default=0,
        help="Activate steering after this many tokens beyond the prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate.",
    )
    parser.add_argument(
        "--eval-on-improved-only",
        action="store_true",
        default=True,
        help="Evaluate only on improved-after-hint examples.",
    )
    parser.add_argument(
        "--stream-results",
        action="store_true",
        help="Stream results to JSONL instead of storing full arrays in memory.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N examples (default: 1).",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id for output folder. If empty, a timestamped id is used.",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def parse_layers(layers_arg: str, num_layers: int) -> List[int]:
    if not layers_arg:
        return list(range(num_layers // 2, num_layers))
    return [int(layer.strip()) for layer in layers_arg.split(",") if layer.strip()]


def build_messages(problem: str, hint: Optional[str] = None) -> List[Dict[str, str]]:
    if hint:
        user_content = (
            f"Problem: {problem}\n\n"
            f"Hint: {hint}\n\n"
            "Solve step by step. Final answer in \\boxed{}."
        )
    else:
        user_content = (
            f"Problem: {problem}\n\n"
            "Solve step by step. Final answer in \\boxed{}."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_prompt_text(tokenizer: AutoTokenizer, problem: str, hint: Optional[str]) -> str:
    messages = build_messages(problem, hint)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_with_steering(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    steering_vector: Optional[SteeringVector],
    strength: float,
    start_after_tokens: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    min_token_index = prompt_len + max(0, start_after_tokens)

    with torch.no_grad():
        if steering_vector is None:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            with steering_vector.apply(model, multiplier=strength, min_token_index=min_token_index):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    pad_token_id=tokenizer.eos_token_id,
                )

    output_tokens = outputs[0].shape[0]
    generated_tokens = output_tokens - prompt_len
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return {
        "text": response,
        "input_tokens": prompt_len,
        "generated_tokens": generated_tokens,
    }


def sample_wrong_items(items: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)


def main() -> None:
    args = parse_args()

    data = load_json(args.input)
    wrong_items = data.get("results", [])
    sampled = sample_wrong_items(wrong_items, args.num_examples, args.seed)
    if not sampled:
        raise RuntimeError("No wrong examples found or num_examples <= 0.")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / "wrong_subset.json"
    write_json(subset_path, {"config": data.get("config", {}), "results": sampled})

    client: Optional[OpenAI] = None
    if not args.mock_hints and not args.use_solution_hint:
        if not HAS_OPENAI:
            raise RuntimeError("openai package not installed. Please install it first.")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        client = OpenAI(api_key=api_key)

    model, tokenizer = load_model(args.qwen_model)
    if args.device != "auto":
        model = model.to(torch.device(args.device))
        model.eval()

    results_with_hints: List[Dict[str, Any]] = []
    improved_examples: List[Example] = []
    hint_errors = 0
    hints_jsonl_path = output_dir / "wrong_subset_with_hints.jsonl"
    hints_jsonl_handle = open(hints_jsonl_path, "w", encoding="utf-8") if args.stream_results else None

    total_items = len(sampled)
    for idx, item in enumerate(sampled, start=1):
        if args.use_solution_hint:
            hint_data = {
                "hint": item.get("solution", ""),
                "error": None,
                "openai_model": "solution",
                "openai_usage": None,
            }
        elif args.mock_hints:
            hint_data = {
                "hint": args.mock_hint_text,
                "error": None,
                "openai_model": "mock",
                "openai_usage": None,
            }
        else:
            hint_data = generate_hint(client, args.openai_model, item)

        hint = hint_data.get("hint", "")
        if hint_data.get("error"):
            hint_errors += 1

        messages = build_hinted_prompt(item.get("problem", ""), hint)
        gen_result = generate_response(model, tokenizer, messages)
        if gen_result.get("skipped"):
            rerun = {
                "skipped": True,
                "skip_reason": gen_result.get("skip_reason"),
                "input_tokens": gen_result.get("input_tokens"),
                "generated_tokens": gen_result.get("generated_tokens", 0),
                "generation_time_sec": None,
                "response": "",
                "predicted": "",
                "ground_truth": item.get("ground_truth", ""),
                "correct": False,
                "match_type": "skipped",
            }
        else:
            response_text = gen_result["text"]
            predicted = extract_boxed_answer(response_text)
            ground_truth = extract_ground_truth(item.get("solution", "")) or item.get("ground_truth", "")
            correct, match_debug = is_correct(predicted, ground_truth)
            rerun = {
                "skipped": False,
                "response": response_text,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": correct,
                "match_type": match_debug.get("match_type"),
                "input_tokens": gen_result.get("input_tokens"),
                "generated_tokens": gen_result.get("generated_tokens"),
                "generation_time_sec": None,
            }

        original_correct = bool(item.get("correct"))
        improved = bool(rerun.get("correct")) and not original_correct
        if improved:
            improved_examples.append(Example(item=item, hint=hint, rerun=rerun, improved=True))

        result_row = {
            "idx": item.get("idx"),
            "problem": item.get("problem"),
            "original": item,
            "hint": hint,
            "hint_error": hint_data.get("error"),
            "hint_openai_usage": hint_data.get("openai_usage"),
            "rerun": rerun,
            "delta": {
                "correct_before": original_correct,
                "correct_after": bool(rerun.get("correct")),
                "predicted_before": item.get("predicted"),
                "predicted_after": rerun.get("predicted"),
                "match_type_before": item.get("match_type"),
                "match_type_after": rerun.get("match_type"),
            },
        }
        if args.stream_results and hints_jsonl_handle:
            hints_jsonl_handle.write(json.dumps(result_row, ensure_ascii=True) + "\n")
        else:
            results_with_hints.append(result_row)

        if args.sleep > 0:
            time.sleep(args.sleep)

        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"[Hints] {idx}/{total_items} processed")

    if hints_jsonl_handle:
        hints_jsonl_handle.close()

    hints_output = {
        "config": {
            "source_file": str(Path(args.input).resolve()),
            "subset_file": str(subset_path.resolve()),
            "hint_model": args.openai_model,
            "qwen_model": args.qwen_model,
            "device": args.device,
            "mock_hints": args.mock_hints,
            "use_solution_hint": args.use_solution_hint,
            "stream_results": args.stream_results,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": {
            "total": len(sampled),
            "hint_errors": hint_errors,
            "improved": len(improved_examples),
        },
        "results": [] if args.stream_results else results_with_hints,
        "results_file": str(hints_jsonl_path.resolve()) if args.stream_results else None,
    }

    hints_path = output_dir / "wrong_subset_with_hints.json"
    write_json(hints_path, hints_output)

    if not improved_examples:
        raise RuntimeError("No improved examples found after hinting; cannot build steering vector.")

    layers = parse_layers(args.layers, model.config.num_hidden_layers)
    training_samples: List[Tuple[str, str]] = []
    for ex in improved_examples:
        problem = ex.item.get("problem", "")
        hinted = build_prompt_text(tokenizer, problem, ex.hint)
        unhinted = build_prompt_text(tokenizer, problem, None)
        training_samples.append((hinted, unhinted))

    steering_vector = train_steering_vector(
        model=model,
        tokenizer=tokenizer,
        training_samples=training_samples,
        show_progress=True,
        layers=layers,
    )

    vector_path = output_dir / "hint_pipeline_steering_vector.pt"
    torch.save(steering_vector, vector_path)

    eval_items = improved_examples if args.eval_on_improved_only else [
        Example(item=item, hint="", rerun={}, improved=False) for item in sampled
    ]

    eval_results: List[Dict[str, Any]] = []
    vanilla_correct = 0
    steered_correct = 0
    improved_count = 0
    eval_jsonl_path = output_dir / "hint_pipeline_eval.jsonl"
    eval_jsonl_handle = open(eval_jsonl_path, "w", encoding="utf-8") if args.stream_results else None

    total_eval = len(eval_items)
    for idx, ex in enumerate(eval_items, start=1):
        problem = ex.item.get("problem", "")
        prompt_text = build_prompt_text(tokenizer, problem, None)

        vanilla = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            steering_vector=None,
            strength=args.strength,
            start_after_tokens=args.start_after_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        steered = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            steering_vector=steering_vector,
            strength=args.strength,
            start_after_tokens=args.start_after_tokens,
            max_new_tokens=args.max_new_tokens,
        )

        ground_truth = extract_ground_truth(ex.item.get("solution", "")) or ex.item.get("ground_truth", "")
        predicted_vanilla = extract_boxed_answer(vanilla["text"])
        predicted_steered = extract_boxed_answer(steered["text"])
        correct_vanilla, _ = is_correct(predicted_vanilla, ground_truth)
        correct_steered, _ = is_correct(predicted_steered, ground_truth)

        if correct_vanilla:
            vanilla_correct += 1
        if correct_steered:
            steered_correct += 1
        if correct_steered and not correct_vanilla:
            improved_count += 1

        eval_row = {
            "problem": problem,
            "ground_truth": ground_truth,
            "vanilla": {
                "response": vanilla["text"],
                "predicted": predicted_vanilla,
                "correct": correct_vanilla,
            },
            "steered": {
                "response": steered["text"],
                "predicted": predicted_steered,
                "correct": correct_steered,
            },
        }
        if args.stream_results and eval_jsonl_handle:
            eval_jsonl_handle.write(json.dumps(eval_row, ensure_ascii=True) + "\n")
        else:
            eval_results.append(eval_row)

        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"[Eval] {idx}/{total_eval} processed")

    total = len(eval_results)
    if eval_jsonl_handle:
        eval_jsonl_handle.close()

    eval_output = {
        "config": {
            "input_file": str(Path(args.input).resolve()),
            "subset_file": str(subset_path.resolve()),
            "hints_file": str(hints_path.resolve()),
            "vector_file": str(vector_path.resolve()),
            "model": args.qwen_model,
            "layers": layers,
            "strength": args.strength,
            "num_sampled": len(sampled),
            "num_improved": len(improved_examples),
            "eval_on_improved_only": args.eval_on_improved_only,
            "stream_results": args.stream_results,
            "start_after_tokens": args.start_after_tokens,
            "max_new_tokens": args.max_new_tokens,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": {
            "total": total,
            "vanilla_correct": vanilla_correct,
            "steered_correct": steered_correct,
            "vanilla_accuracy": vanilla_correct / total if total else 0.0,
            "steered_accuracy": steered_correct / total if total else 0.0,
            "improved": improved_count,
        },
        "results": [] if args.stream_results else eval_results,
        "results_file": str(eval_jsonl_path.resolve()) if args.stream_results else None,
    }

    eval_path = output_dir / "hint_pipeline_eval.json"
    write_json(eval_path, eval_output)

    print("Done.")
    print(f"Subset: {subset_path}")
    print(f"Hints: {hints_path}")
    print(f"Vector: {vector_path}")
    print(f"Eval:   {eval_path}")


if __name__ == "__main__":
    main()
