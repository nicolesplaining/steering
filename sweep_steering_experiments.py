#!/usr/bin/env python3
"""
Sweep steering strengths and layer sets using existing hinted reruns.
Supports two training baselines:
  1) hinted prompt vs unhinted prompt
  2) hinted prompt vs empty prompt (no prompt)
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import SteeringVector, train_steering_vector

from baseline_eval import extract_boxed_answer, extract_ground_truth, load_model
from hint_and_rerun import is_correct


SYSTEM_PROMPT = "You are a helpful math assistant."


@dataclass
class ImprovedExample:
    problem: str
    hint: str
    ground_truth: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep steering strengths/layers on improved-after-hint items."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to wrong_subset_with_hints.json (from a pipeline run).",
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Output directory for sweep results.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Model name for steering/eval.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Force model device (default: auto).",
    )
    parser.add_argument(
        "--layers-sets",
        default="14-27",
        help="Semicolon-separated layer ranges (e.g., '14-27;12-23;16-27').",
    )
    parser.add_argument(
        "--strengths",
        default="0.3,0.5,1.0,1.5",
        help="Comma-separated strengths to evaluate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate.",
    )
    parser.add_argument(
        "--start-after-tokens",
        type=int,
        default=0,
        help="Activate steering after this many tokens beyond the prompt.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N examples (default: 1).",
    )
    parser.add_argument(
        "--save-vectors",
        action="store_true",
        help="Save steering vectors for each layer set/baseline.",
    )
    parser.add_argument(
        "--baseline-filter",
        type=str,
        default=None,
        choices=["hinted_minus_unhinted", "hinted_minus_empty"],
        help="Filter to test only this baseline type (default: test both).",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def parse_layer_sets(raw: str) -> List[List[int]]:
    sets: List[List[int]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            sets.append(list(range(start, end + 1)))
        else:
            sets.append([int(x.strip()) for x in chunk.split(",") if x.strip()])
    return sets


def parse_strengths(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


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

    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return {
        "text": response,
        "input_tokens": prompt_len,
        "generated_tokens": outputs[0].shape[0] - prompt_len,
    }


def load_improved_examples(path: str) -> List[ImprovedExample]:
    data = load_json(path)
    results = data.get("results", [])
    improved: List[ImprovedExample] = []
    for item in results:
        original = item.get("original", {})
        rerun = item.get("rerun", {})
        if bool(rerun.get("correct")) and not bool(original.get("correct")):
            problem = item.get("problem") or original.get("problem", "")
            hint = item.get("hint", "")
            solution = original.get("solution", "")
            ground_truth = original.get("ground_truth", "") or extract_ground_truth(solution)
            if problem and hint and ground_truth:
                improved.append(
                    ImprovedExample(problem=problem, hint=hint, ground_truth=ground_truth)
                )
    return improved


def main() -> None:
    args = parse_args()

    improved = load_improved_examples(args.input)
    if not improved:
        raise RuntimeError("No improved-after-hint examples found in input.")

    model, tokenizer = load_model(args.model)
    if args.device != "auto":
        model = model.to(torch.device(args.device))
        model.eval()

    layer_sets = parse_layer_sets(args.layers_sets)
    strengths = parse_strengths(args.strengths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_results: List[Dict[str, Any]] = []

    for layers in layer_sets:
        # Build two vectors per layer set: hinted vs unhinted, hinted vs empty prompt
        training_hinted_vs_unhinted: List[Tuple[str, str]] = []
        training_hinted_vs_empty: List[Tuple[str, str]] = []

        for ex in improved:
            hinted = build_prompt_text(tokenizer, ex.problem, ex.hint)
            unhinted = build_prompt_text(tokenizer, ex.problem, None)
            # Use a minimal prompt to avoid empty-token edge cases in activation extraction.
            empty_prompt = tokenizer.bos_token or " "
            training_hinted_vs_unhinted.append((hinted, unhinted))
            training_hinted_vs_empty.append((hinted, empty_prompt))

        vector_unhinted = train_steering_vector(
            model=model,
            tokenizer=tokenizer,
            training_samples=training_hinted_vs_unhinted,
            show_progress=True,
            layers=layers,
        )
        vector_empty = train_steering_vector(
            model=model,
            tokenizer=tokenizer,
            training_samples=training_hinted_vs_empty,
            show_progress=True,
            layers=layers,
        )

        if args.save_vectors:
            torch.save(vector_unhinted, output_dir / f"vector_hinted_minus_unhinted_{layers[0]}_{layers[-1]}.pt")
            torch.save(vector_empty, output_dir / f"vector_hinted_minus_empty_{layers[0]}_{layers[-1]}.pt")

        # Filter baselines if --baseline-filter is specified
        baseline_pairs = [
            ("hinted_minus_unhinted", vector_unhinted),
            ("hinted_minus_empty", vector_empty),
        ]
        if args.baseline_filter:
            baseline_pairs = [
                (name, vec) for name, vec in baseline_pairs
                if name == args.baseline_filter
            ]
            if not baseline_pairs:
                raise ValueError(f"Invalid baseline filter: {args.baseline_filter}")

        for baseline_name, vector in baseline_pairs:
            for strength in strengths:
                vanilla_correct = 0
                steered_correct = 0
                improved_count = 0

                total = len(improved)
                for idx, ex in enumerate(improved, start=1):
                    prompt_text = build_prompt_text(tokenizer, ex.problem, None)

                    vanilla = generate_with_steering(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt_text,
                        steering_vector=None,
                        strength=strength,
                        start_after_tokens=args.start_after_tokens,
                        max_new_tokens=args.max_new_tokens,
                    )
                    steered = generate_with_steering(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt_text,
                        steering_vector=vector,
                        strength=strength,
                        start_after_tokens=args.start_after_tokens,
                        max_new_tokens=args.max_new_tokens,
                    )

                    predicted_vanilla = extract_boxed_answer(vanilla["text"])
                    predicted_steered = extract_boxed_answer(steered["text"])
                    correct_vanilla, _ = is_correct(predicted_vanilla, ex.ground_truth)
                    correct_steered, _ = is_correct(predicted_steered, ex.ground_truth)

                    if correct_vanilla:
                        vanilla_correct += 1
                    if correct_steered:
                        steered_correct += 1
                    if correct_steered and not correct_vanilla:
                        improved_count += 1

                    if args.progress_every > 0 and idx % args.progress_every == 0:
                        print(
                            f"[{baseline_name} | layers {layers[0]}-{layers[-1]} | α={strength}] "
                            f"{idx}/{total}"
                        )

                sweep_results.append(
                    {
                        "baseline": baseline_name,
                        "layers": layers,
                        "strength": strength,
                        "total": total,
                        "vanilla_correct": vanilla_correct,
                        "steered_correct": steered_correct,
                        "vanilla_accuracy": vanilla_correct / total if total else 0.0,
                        "steered_accuracy": steered_correct / total if total else 0.0,
                        "improved": improved_count,
                    }
                )

    output = {
        "config": {
            "input": str(Path(args.input).resolve()),
            "model": args.model,
            "layer_sets": layer_sets,
            "strengths": strengths,
            "num_examples": len(improved),
            "start_after_tokens": args.start_after_tokens,
            "max_new_tokens": args.max_new_tokens,
            "timestamp": datetime.now().isoformat(),
        },
        "results": sweep_results,
    }

    out_path = output_dir / "sweep_results.json"
    write_json(out_path, output)
    print(f"Saved sweep results to {out_path}")


if __name__ == "__main__":
    main()
