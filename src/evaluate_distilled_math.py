#!/usr/bin/env python3
"""
Evaluate a context-distilled LoRA adapter on the math benchmark.

This mirrors `baseline_eval.py` but assumes you have already trained a
LoRA adapter (e.g., via `context_distillation.py` / `finetune_student`)
and want to measure its accuracy on open-r1/OpenThoughts-114k-math.

Typical usage
-------------

# 1) Baseline (no adapter) – existing script:
    python baseline_eval.py \
        --model Qwen/Qwen2.5-Math-1.5B-Instruct \
        --n_samples 200 \
        --seed 42 \
        --output baseline_qwen_math_200.json

# 2) Distilled student (with adapter) – this script:
    python evaluate_distilled_math.py \
        --model Qwen/Qwen2.5-Math-1.5B-Instruct \
        --adapter-path context_distillation_adapter \
        --n_samples 200 \
        --seed 42 \
        --output distilled_qwen_math_200.json

You can then compare the two JSONs' summary accuracy.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from openai import OpenAI

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # optional; script will fail later if adapter is used

from baseline_eval import (
    extract_boxed_answer,
    extract_ground_truth,
    generate_response,
    load_model,
    llm_judge_correctness,
    _is_correct_local,
    _set_seeds,
    _save_results,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA-adapted math model on OpenThoughts-114k-math "
        "(mirrors baseline_eval.py)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Base model ID (must match the one used to train the adapter).",
    )
    parser.add_argument(
        "--adapter-path",
        default="artifacts/context_distillation_adapter",
        help="Path to the saved LoRA adapter (from finetune_student).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (must match baseline to compare).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Number of samples to evaluate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. If omitted, a timestamped name is used.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Use local answer matching instead of OpenAI judge. No API key needed.",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM for inference (with LoRA). Faster on GPU.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="OpenAI model for LLM judge (e.g. gpt-4o, gpt-4o-mini). Default: gpt-4o.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seed = args.seed
    n_samples = args.n_samples

    _set_seeds(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output
    if output_path is None:
        output_path = (
            f"distilled_eval_{args.model.split('/')[-1].lower()}_{timestamp}.json"
        )

    print(f"Distilled Evaluation (with LoRA adapter): {n_samples} samples")
    print(f"Base model: {args.model}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Output: {output_path}")
    print()

    # Load dataset (same as baseline_eval)
    print("Loading open-r1/OpenThoughts-114k-math...")
    ds = load_dataset("open-r1/OpenThoughts-114k-math", split="train")
    print(f"Total examples: {len(ds)}")

    # Sample randomly (no filtering), same pattern as baseline_eval
    indices = list(range(len(ds)))
    random.shuffle(indices)
    sample_indices = indices[:n_samples]

    # Load model: vLLM+LoRA or HuggingFace+Peft
    use_vllm = args.use_vllm
    lora_request = None
    if use_vllm:
        from vllm.lora.request import LoRARequest
        base_model, tokenizer = load_model(args.model, use_vllm=True, enable_lora=True)
        lora_request = LoRARequest("distilled_adapter", 1, args.adapter_path)
        model = base_model
        print(f"LoRA adapter will be applied per-request: {args.adapter_path}")
    else:
        if PeftModel is None:
            raise ImportError("peft is required to load the LoRA adapter. Install with: pip install peft")
        adapter_path = os.path.abspath(os.path.expanduser(args.adapter_path))
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(
                f"Adapter path does not exist: {adapter_path}\n"
                "Train an adapter first, e.g. with context_distillation.py "
                "(build_dataset_from_hint_outputs + finetune_student) or use --adapter-path /path/to/your/adapter"
            )
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Not a valid PEFT adapter directory (no adapter_config.json): {adapter_path}"
            )
        base_model, tokenizer = load_model(args.model)
        print(f"Loading LoRA adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)
        model.eval()
        print(f"LoRA-adapted model loaded on {model.device}")

    judge_client = None if args.no_llm_judge else OpenAI()

    results = []
    correct_count = 0

    system_prompt = "You are a helpful math assistant."

    print(f"\n{'='*60}")
    print("RUNNING DISTILLED EVALUATION")
    print('='*60)

    for i, idx in enumerate(sample_indices):
        example = ds[idx]
        problem = example["problem"]

        # Simple prompt (same as baseline_eval)
        user_content = (
            f"Problem: {problem}\n\nSolve step by step. Final answer in \\boxed{{}}."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Generate
        start_time = time.time()
        gen_result = generate_response(
            model,
            tokenizer,
            messages,
            max_new_tokens=args.max_new_tokens,
            use_vllm=use_vllm,
            lora_request=lora_request,
        )
        gen_time = time.time() - start_time

        # Handle skipped (too long)
        if gen_result.get("skipped"):
            print(f"{i+1}/{n_samples} [SKIP] {gen_result.get('skip_reason')}")
            results.append(
                {
                    "idx": i,
                    "dataset_idx": idx,
                    "problem": problem,
                    "solution": example["solution"],
                    "source": example.get("source", "unknown"),
                    "skipped": True,
                    "skip_reason": gen_result.get("skip_reason"),
                    "input_tokens": gen_result["input_tokens"],
                }
            )
            continue

        response = gen_result["text"]

        # Extract and evaluate
        predicted = extract_boxed_answer(response)
        ground_truth = extract_ground_truth(example["solution"])

        if judge_client is not None:
            correct, llm_judgment = llm_judge_correctness(
                judge_client,
                problem=problem,
                model_response=response,
                ground_truth_solution=example["solution"],
                seed=seed,
                model=args.judge_model,
            )
            match_type = "llm_judge"
        else:
            correct, match_type = _is_correct_local(predicted, ground_truth)
            llm_judgment = None
        if correct:
            correct_count += 1

        evaluated = sum(1 for r in results if not r.get("skipped")) + 1
        accuracy_so_far = correct_count / evaluated
        status = "✓" if correct else "✗"

        print(
            f"{i+1}/{n_samples} [{status}] acc={accuracy_so_far*100:.1f}% "
            f"| pred={predicted[:40]}... | gt={ground_truth[:40]}... | {gen_time:.1f}s"
        )

        # Log everything
        results.append(
            {
                "idx": i,
                "dataset_idx": idx,
                "problem": problem,
                "solution": example["solution"],
                "source": example.get("source", "unknown"),
                "skipped": False,
                "response": response,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": correct,
                "match_type": match_type,
                "llm_judgment": llm_judgment,
                "input_tokens": gen_result["input_tokens"],
                "generated_tokens": gen_result["generated_tokens"],
                "generation_time_sec": gen_time,
            }
        )

        # Save incrementally every 10 examples
        if (i + 1) % 10 == 0:
            _save_results(
                output_path,
                results,
                correct_count,
                i + 1,
                n_samples,
                seed,
                f"{args.model}+adapter:{args.adapter_path}",
            )

    # Final save
    _save_results(
        output_path,
        results,
        correct_count,
        n_samples,
        n_samples,
        seed,
        f"{args.model}+adapter:{args.adapter_path}",
    )

    # Summary
    evaluated = sum(1 for r in results if not r.get("skipped"))
    skipped = sum(1 for r in results if r.get("skipped"))
    final_accuracy = correct_count / evaluated if evaluated > 0 else 0

    print(f"\n{'='*60}")
    print("FINAL DISTILLED RESULTS")
    print('='*60)
    print(f"Evaluated: {evaluated}/{n_samples} (skipped {skipped} due to context length)")
    print(f"Accuracy: {final_accuracy*100:.2f}% ({correct_count}/{evaluated})")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
