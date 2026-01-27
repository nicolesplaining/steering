#!/usr/bin/env python3
"""
Baseline Evaluation: Simple dry run on MATH dataset.

Evaluates Qwen2.5-Math-1.5B-Instruct on 500 random examples from OpenThoughts-114k-math.
No ICL, no hints - just direct evaluation.
"""

import argparse
import json
import random
import re
import time
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# ANSWER EXTRACTION & COMPARISON (from positive_control.py)
# ============================================================================

def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} using brace counting for nested braces."""
    start_idx = text.find("\\boxed{")
    if start_idx == -1:
        return ""
    
    content_start = start_idx + 7  # len("\\boxed{")
    brace_count = 1
    current_idx = content_start
    
    while current_idx < len(text) and brace_count > 0:
        char = text[current_idx]
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
        current_idx += 1
    
    if brace_count == 0:
        return text[content_start:current_idx - 1]
    return ""


def extract_ground_truth(solution: str) -> str:
    """Extract ground truth answer from solution field."""
    boxed = extract_boxed_answer(solution)
    if boxed:
        return boxed
    
    answer_match = re.search(r'(?:answer|result)\s*(?:is|=)\s*[:\s]*([^\n\.]+)', solution, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    return ""


def llm_judge_correctness(
    client: OpenAI,
    problem: str,
    model_response: str,
    ground_truth_solution: str,
    seed: int,
) -> tuple[bool, str]:
    """Use GPT-5 to judge if the model's answer is correct."""
    prompt = (
        "You are a math answer grader. Given a math problem, a model's response, and the "
        "ground truth solution, determine if the model's final answer is mathematically "
        "equivalent to the correct answer.\n\n"
        f"Problem: {problem}\n\n"
        f"Model's Response: {model_response}\n\n"
        f"Ground Truth Solution: {ground_truth_solution}\n\n"
        'Is the model\'s final answer correct? Respond with ONLY "CORRECT" or "INCORRECT" '
        "followed by a brief explanation."
    )

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        seed=seed,
    )

    judgment = response.choices[0].message.content.strip()
    is_correct = judgment.upper().startswith("CORRECT")
    return is_correct, judgment


# ============================================================================
# MODEL
# ============================================================================

def load_model(model_name: str):
    """Load the math model."""
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print(f"Model loaded on {model.device}")
    return model, tokenizer


MAX_CONTEXT_LENGTH = 4096  # Qwen2.5-Math context limit


def generate_response(model, tokenizer, messages: list, max_new_tokens: int = 8192) -> dict:
    """Generate response from the model."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    input_tokens = inputs.input_ids.shape[1]
    
    # Check context length
    if input_tokens > MAX_CONTEXT_LENGTH:
        return {
            "text": "",
            "input_tokens": input_tokens,
            "generated_tokens": 0,
            "skipped": True,
            "skip_reason": f"input_tokens ({input_tokens}) > MAX_CONTEXT_LENGTH ({MAX_CONTEXT_LENGTH})",
        }
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    
    output_tokens = outputs[0].shape[0]
    generated_tokens = output_tokens - input_tokens
    
    response = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)
    
    return {
        "text": response,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "skipped": False,
    }


# ============================================================================
# MAIN
# ============================================================================

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _short_model_name(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline evaluation on OpenThoughts-114k-math."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID to evaluate (e.g., Qwen/Qwen3-4B).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
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
    return parser.parse_args()


def main():
    args = _parse_args()
    seed = args.seed
    n_samples = args.n_samples

    _set_seeds(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output
    if output_path is None:
        output_path = f"baseline_eval_{_short_model_name(args.model)}_{timestamp}.json"
    
    print(f"Baseline Evaluation: {n_samples} samples")
    print(f"Output: {output_path}")
    print()
    
    # Load dataset
    print("Loading open-r1/OpenThoughts-114k-math...")
    ds = load_dataset("open-r1/OpenThoughts-114k-math", split="train")
    print(f"Total examples: {len(ds)}")
    
    # Sample randomly (no filtering)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    sample_indices = indices[:n_samples]
    
    # Load model
    model, tokenizer = load_model(args.model)
    judge_client = OpenAI()
    
    # Run evaluation
    results = []
    correct_count = 0
    
    system_prompt = "You are a helpful math assistant."
    
    print(f"\n{'='*60}")
    print("RUNNING BASELINE EVALUATION")
    print('='*60)
    
    for i, idx in enumerate(sample_indices):
        example = ds[idx]
        problem = example['problem']
        
        # Simple prompt
        user_content = f"Problem: {problem}\n\nSolve step by step. Final answer in \\boxed{{}}."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Generate
        start_time = time.time()
        gen_result = generate_response(
            model,
            tokenizer,
            messages,
            max_new_tokens=args.max_new_tokens,
        )
        gen_time = time.time() - start_time
        
        # Handle skipped (too long)
        if gen_result.get("skipped"):
            print(f"{i+1}/{n_samples} [SKIP] {gen_result.get('skip_reason')}")
            results.append({
                "idx": i,
                "dataset_idx": idx,
                "problem": problem,
                "solution": example['solution'],
                "source": example.get('source', 'unknown'),
                "skipped": True,
                "skip_reason": gen_result.get("skip_reason"),
                "input_tokens": gen_result["input_tokens"],
            })
            continue
        
        response = gen_result["text"]
        
        # Extract and evaluate
        predicted = extract_boxed_answer(response)
        ground_truth = extract_ground_truth(example['solution'])

        correct, llm_judgment = llm_judge_correctness(
            judge_client,
            problem=problem,
            model_response=response,
            ground_truth_solution=example['solution'],
            seed=seed,
        )
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
        results.append({
            "idx": i,
            "dataset_idx": idx,
            "problem": problem,
            "solution": example['solution'],
            "source": example.get('source', 'unknown'),
            "skipped": False,
            "response": response,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": correct,
            "match_type": "llm_judge",
            "llm_judgment": llm_judgment,
            "input_tokens": gen_result["input_tokens"],
            "generated_tokens": gen_result["generated_tokens"],
            "generation_time_sec": gen_time,
        })
        
        # Save incrementally
        if (i + 1) % 10 == 0:
            _save_results(
                output_path,
                results,
                correct_count,
                i + 1,
                n_samples,
                seed,
                args.model,
            )
    
    # Final save
    _save_results(output_path, results, correct_count, n_samples, n_samples, seed, args.model)
    
    # Summary
    evaluated = sum(1 for r in results if not r.get("skipped"))
    skipped = sum(1 for r in results if r.get("skipped"))
    final_accuracy = correct_count / evaluated if evaluated > 0 else 0
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print('='*60)
    print(f"Evaluated: {evaluated}/{n_samples} (skipped {skipped} due to context length)")
    print(f"Accuracy: {final_accuracy*100:.2f}% ({correct_count}/{evaluated})")
    print(f"Results saved to: {output_path}")


def _save_results(
    path: str,
    results: list,
    correct: int,
    done: int,
    total: int,
    seed: int,
    model_name: str,
):
    """Save results to JSON."""
    evaluated = sum(1 for r in results if not r.get("skipped"))
    skipped = sum(1 for r in results if r.get("skipped"))
    
    data = {
        "config": {
            "model": model_name,
            "dataset": "open-r1/OpenThoughts-114k-math",
            "n_samples": total,
            "max_context_length": MAX_CONTEXT_LENGTH,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        },
        "progress": {
            "done": done,
            "total": total,
            "evaluated": evaluated,
            "skipped": skipped,
            "correct": correct,
            "accuracy": correct / evaluated if evaluated > 0 else 0,
        },
        "results": results,
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()

