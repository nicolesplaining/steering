import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import SteeringVector, train_steering_vector

from baseline_eval import (
    extract_boxed_answer,
    extract_ground_truth,
    is_correct,
    load_model,
)


SYSTEM_PROMPT = "You are a helpful math assistant."


@dataclass
class Example:
    problem: str
    hint: str
    solution: str
    ground_truth: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hint-vs-no-hint steering vector and evaluate improvements."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input JSON with hints. Pass multiple times to combine files.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for outputs.",
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
        "--num-train",
        type=int,
        default=10,
        help="Number of examples to build the steering vector.",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=100,
        help="Number of examples to evaluate.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "--train-from-improved-only",
        action="store_true",
        help="Train only on examples that improved after hinting in the input file.",
    )
    parser.add_argument(
        "--eval-on-train",
        action="store_true",
        help="Evaluate on the training examples instead of a separate test split.",
    )
    return parser.parse_args()


def load_examples(paths: List[str], improved_only: bool) -> List[Example]:
    examples: List[Example] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        results = data.get("results", [])
        for item in results:
            original = item.get("original", item)
            if improved_only:
                rerun = item.get("rerun", {})
                original_correct = bool(original.get("correct"))
                improved = bool(rerun.get("correct")) and not original_correct
                if not improved:
                    continue

            problem = item.get("problem") or original.get("problem", "")
            hint = item.get("hint") or original.get("hint", "")
            solution = original.get("solution", "")
            ground_truth = original.get("ground_truth", "")
            if not ground_truth:
                ground_truth = extract_ground_truth(solution)

            if not problem or not hint:
                continue

            examples.append(
                Example(
                    problem=problem,
                    hint=hint,
                    solution=solution,
                    ground_truth=ground_truth,
                )
            )
    return examples


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


def parse_layers(layers_arg: str, num_layers: int) -> List[int]:
    if not layers_arg:
        return list(range(num_layers // 2, num_layers))
    return [int(layer.strip()) for layer in layers_arg.split(",") if layer.strip()]


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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    examples = load_examples(args.input, args.train_from_improved_only)
    if not examples:
        raise RuntimeError("No usable examples with hints found in input file.")

    random.shuffle(examples)
    train_examples = examples[: args.num_train]
    if args.eval_on_train:
        test_examples = train_examples
    else:
        test_examples = examples[args.num_train : args.num_train + args.num_test]

    if len(test_examples) < args.num_test:
        print(f"Warning: only {len(test_examples)} test examples available.")

    print(f"Training examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")

    model, tokenizer = load_model(args.model)
    if args.device != "auto":
        model = model.to(torch.device(args.device))
        model.eval()

    layers = parse_layers(args.layers, model.config.num_hidden_layers)

    training_samples: List[Tuple[str, str]] = []
    for ex in train_examples:
        hinted = build_prompt_text(tokenizer, ex.problem, ex.hint)
        unhinted = build_prompt_text(tokenizer, ex.problem, None)
        training_samples.append((hinted, unhinted))

    print("Training steering vector (hinted vs unhinted)...")
    steering_vector = train_steering_vector(
        model=model,
        tokenizer=tokenizer,
        training_samples=training_samples,
        show_progress=True,
        layers=layers,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = output_dir / "hint_steering_vector.pt"
    torch.save(steering_vector, vector_path)
    print(f"Saved steering vector to {vector_path}")

    print("Evaluating on test examples...")
    results: List[Dict[str, Any]] = []
    vanilla_correct = 0
    steered_correct = 0
    improved = 0

    for ex in test_examples:
        prompt_text = build_prompt_text(tokenizer, ex.problem, None)

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

        predicted_vanilla = extract_boxed_answer(vanilla["text"])
        predicted_steered = extract_boxed_answer(steered["text"])
        correct_vanilla, _ = is_correct(predicted_vanilla, ex.ground_truth)
        correct_steered, _ = is_correct(predicted_steered, ex.ground_truth)

        if correct_vanilla:
            vanilla_correct += 1
        if correct_steered:
            steered_correct += 1
        if correct_steered and not correct_vanilla:
            improved += 1

        results.append(
            {
                "problem": ex.problem,
                "ground_truth": ex.ground_truth,
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
        )

    total = len(test_examples)
    summary = {
        "total": total,
        "vanilla_correct": vanilla_correct,
        "steered_correct": steered_correct,
        "vanilla_accuracy": vanilla_correct / total if total else 0.0,
        "steered_accuracy": steered_correct / total if total else 0.0,
        "improved": improved,
    }

    output = {
        "config": {
            "input_files": [str(Path(path).resolve()) for path in args.input],
            "model": args.model,
            "layers": layers,
            "strength": args.strength,
            "num_train": len(train_examples),
            "num_test": total,
            "train_from_improved_only": args.train_from_improved_only,
            "eval_on_train": args.eval_on_train,
            "start_after_tokens": args.start_after_tokens,
            "max_new_tokens": args.max_new_tokens,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": summary,
        "results": results,
    }

    results_path = output_dir / "hint_steering_eval.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    print("Done.")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
