#!/usr/bin/env python3
"""
Baseline Evaluation: Simple dry run on MATH dataset.

Evaluates Qwen2.5-Math-1.5B-Instruct on 500 random examples from OpenThoughts-114k-math.
No ICL, no hints - just direct evaluation.
"""

import json
import random
import re
import time
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: latex2sympy2 for symbolic math comparison
try:
    from latex2sympy2 import latex2sympy
    from sympy import simplify, N
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    print("WARNING: latex2sympy2 not installed. Math comparison will be string-only.")


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


def normalize_answer(s: str) -> str:
    """Basic normalization for comparison."""
    s = s.strip().lower()
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = s.replace(" ", "").replace(",", "")
    return s


def is_correct(predicted: str, ground_truth: str) -> tuple:
    """Multi-stage comparison for math answers."""
    debug = {
        "pred_raw": predicted,
        "gt_raw": ground_truth,
        "match_type": None,
    }
    
    if not predicted:
        debug["match_type"] = "empty_prediction"
        return False, debug
    
    if not ground_truth:
        debug["match_type"] = "empty_ground_truth"
        return False, debug
    
    # Stage 1: Exact string match
    pred_clean = normalize_answer(predicted)
    gt_clean = normalize_answer(ground_truth)
    
    if pred_clean == gt_clean:
        debug["match_type"] = "exact_string"
        return True, debug
    
    # Stage 2 & 3: Symbolic comparison
    if HAS_SYMPY:
        try:
            pred_sym = latex2sympy(predicted)
            gt_sym = latex2sympy(ground_truth)
            
            pred_val = float(N(pred_sym))
            gt_val = float(N(gt_sym))
            
            if abs(pred_val - gt_val) < 1e-6:
                debug["match_type"] = "numeric_sympy"
                return True, debug
        except:
            pass
        
        try:
            pred_sym = latex2sympy(predicted)
            gt_sym = latex2sympy(ground_truth)
            
            if simplify(pred_sym - gt_sym) == 0:
                debug["match_type"] = "symbolic_equiv"
                return True, debug
        except:
            pass
    
    debug["match_type"] = "no_match"
    return False, debug


# ============================================================================
# MODEL
# ============================================================================

def load_model(model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"):
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

def main():
    seed = 42
    n_samples = 500
    
    random.seed(seed)
    torch.manual_seed(seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"baseline_eval_{timestamp}.json"
    
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
    model, tokenizer = load_model()
    
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
        gen_result = generate_response(model, tokenizer, messages)
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
        
        correct, match_debug = is_correct(predicted, ground_truth)
        if correct:
            correct_count += 1
        
        evaluated = sum(1 for r in results if not r.get("skipped")) + 1
        accuracy_so_far = correct_count / evaluated
        status = "✓" if correct else "✗"
        
        print(f"{i+1}/{n_samples} [{status}] acc={accuracy_so_far*100:.1f}% | pred={predicted[:40]}... | gt={ground_truth[:40]}... | {gen_time:.1f}s")
        
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
            "match_type": match_debug.get("match_type"),
            "input_tokens": gen_result["input_tokens"],
            "generated_tokens": gen_result["generated_tokens"],
            "generation_time_sec": gen_time,
        })
        
        # Save incrementally
        if (i + 1) % 10 == 0:
            _save_results(output_path, results, correct_count, i + 1, n_samples, seed)
    
    # Final save
    _save_results(output_path, results, correct_count, n_samples, n_samples, seed)
    
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


def _save_results(path: str, results: list, correct: int, done: int, total: int, seed: int):
    """Save results to JSON."""
    evaluated = sum(1 for r in results if not r.get("skipped"))
    skipped = sum(1 for r in results if r.get("skipped"))
    
    data = {
        "config": {
            "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
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

