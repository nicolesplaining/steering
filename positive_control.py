#!/usr/bin/env python3
"""
Positive Control Experiment: Validate that CoT reasoning helps model performance.

Three approaches:
1. Progressive hints - give 0-100% of the actual solution as hints
2. GPT-5 curated hints - use GPT-5 to extract helpful reasoning steps
3. Traditional ICL - use solved examples from other problems

Model: Qwen/Qwen2.5-Math-7B-Instruct
Dataset: open-r1/OpenThoughts-114k-math
"""

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: OpenAI for GPT-5 hints
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("WARNING: openai not installed. Approach 2 (GPT-5 hints) unavailable.")


# ============================================================================
# ANSWER EXTRACTION
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
    # Try boxed first
    boxed = extract_boxed_answer(solution)
    if boxed:
        return boxed
    
    # Fallback: look for "answer is" pattern or last number
    answer_match = re.search(r'(?:answer|result)\s*(?:is|=)\s*[:\s]*([^\n\.]+)', solution, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    return ""


def normalize_answer(s: str) -> str:
    """Basic normalization for comparison."""
    s = s.strip().lower()
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\(?:frac|dfrac)\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', s)
    s = s.replace(" ", "").replace(",", "")
    return s


def is_correct(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    if not predicted or not ground_truth:
        return False
    
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    if pred_norm == gt_norm:
        return True
    
    # Try numeric comparison
    try:
        pred_val = float(eval(pred_norm.replace("^", "**")))
        gt_val = float(eval(gt_norm.replace("^", "**")))
        return abs(pred_val - gt_val) < 1e-6
    except:
        pass
    
    return False


# ============================================================================
# DATA LOADING
# ============================================================================

def load_openthoughts_math(num_icl: int = 500, num_test: int = 100, seed: int = 42) -> Tuple[List[dict], List[dict], dict]:
    """Load and split the OpenThoughts-114k-math dataset."""
    print("Loading open-r1/OpenThoughts-114k-math...")
    ds = load_dataset("open-r1/OpenThoughts-114k-math", split="train")
    
    total_before_filter = len(ds)
    
    # Filter to verified correct solutions
    ds = ds.filter(lambda x: x.get('correct', False))
    total_after_filter = len(ds)
    print(f"Filtered to {total_after_filter} verified correct examples (from {total_before_filter})")
    
    # Random sample
    random.seed(seed)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    icl_indices = indices[:num_icl]
    test_indices = indices[num_icl:num_icl + num_test]
    
    # Add index to each example for tracking
    icl_pool = []
    for idx in icl_indices:
        ex = dict(ds[idx])
        ex['__index__'] = idx
        icl_pool.append(ex)
    
    test_set = []
    for idx in test_indices:
        ex = dict(ds[idx])
        ex['__index__'] = idx
        test_set.append(ex)
    
    print(f"ICL pool: {len(icl_pool)}, Test set: {len(test_set)}")
    
    # Return metadata for logging
    metadata = {
        "dataset": "open-r1/OpenThoughts-114k-math",
        "total_examples": total_before_filter,
        "correct_examples": total_after_filter,
        "seed": seed,
        "icl_pool_size": len(icl_pool),
        "test_set_size": len(test_set),
        "icl_indices": icl_indices,
        "test_indices": test_indices
    }
    
    return icl_pool, test_set, metadata


# ============================================================================
# MODEL SETUP
# ============================================================================

def load_model(model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"):
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


def generate_response(model, tokenizer, messages: List[dict], max_new_tokens: int = 2048) -> Dict:
    """Generate response from the model with full metrics."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    input_tokens = inputs.input_ids.shape[1]
    
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
    hit_token_limit = generated_tokens >= max_new_tokens
    
    response = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)
    
    return {
        "text": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generated_tokens": generated_tokens,
        "hit_token_limit": hit_token_limit,
        "max_new_tokens": max_new_tokens
    }


# ============================================================================
# APPROACH 1: PROGRESSIVE HINTS
# ============================================================================

def split_solution_paragraphs(solution_text: str) -> List[str]:
    """Split R1 solution into paragraphs, handling special tags."""
    # Remove special tags but keep the content
    text = solution_text
    text = text.replace('<|begin_of_thought|>', '')
    text = text.replace('<|end_of_thought|>', '')
    text = text.replace('<|begin_of_solution|>', '')
    text = text.replace('<|end_of_solution|>', '')
    
    # Split by double newlines
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs


def get_partial_hint(paragraphs: List[str], percentage: float) -> str:
    """Return first X% of paragraphs as hint."""
    if percentage <= 0:
        return ""
    n_paragraphs = max(1, int(len(paragraphs) * percentage))
    return '\n\n'.join(paragraphs[:n_paragraphs])


def evaluate_progressive_hints(
    model, tokenizer, test_set: List[dict], 
    percentages: List[float], save_path: str
) -> Dict[float, dict]:
    """Evaluate with progressive hint percentages."""
    
    results = {}
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    
    for pct in percentages:
        print(f"\n{'='*60}")
        print(f"APPROACH 1: Progressive Hints - {int(pct*100)}%")
        print('='*60)
        
        correct_count = 0
        format_ok_count = 0
        total_input_tokens = 0
        total_generated_tokens = 0
        truncation_count = 0
        samples = []
        
        for i, example in enumerate(test_set):
            problem = example['problem']
            dataset_idx = example.get('__index__', i)  # Track original index
            source = example.get('source', 'unknown')
            
            # Get the R1 reasoning trace from conversations
            r1_solution = ""
            for conv in example.get('conversations', []):
                if conv.get('from') == 'assistant':
                    r1_solution = conv.get('value', '')
                    break
            
            # Create hint from the solution
            paragraphs = split_solution_paragraphs(r1_solution)
            hint = get_partial_hint(paragraphs, pct)
            hint_token_estimate = len(hint.split()) if hint else 0
            
            # Build prompt
            if hint:
                user_content = f"Problem: {problem}\n\nHere's some helpful reasoning:\n{hint}\n\nNow solve the problem completely."
            else:
                user_content = f"Problem: {problem}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            # Generate
            start_time = time.time()
            gen_result = generate_response(model, tokenizer, messages)
            gen_time = time.time() - start_time
            
            response = gen_result["text"]
            
            # Extract and evaluate
            predicted = extract_boxed_answer(response)
            ground_truth = extract_ground_truth(example['solution'])
            
            # Format compliance
            has_boxed = "\\boxed{" in response
            if has_boxed:
                format_ok_count += 1
            
            correct = is_correct(predicted, ground_truth)
            if correct:
                correct_count += 1
            
            # Aggregate stats
            total_input_tokens += gen_result["input_tokens"]
            total_generated_tokens += gen_result["generated_tokens"]
            if gen_result["hit_token_limit"]:
                truncation_count += 1
            
            status = "✓" if correct else "✗"
            trunc_flag = " [TRUNC]" if gen_result["hit_token_limit"] else ""
            print(f"{i+1}/{len(test_set)} [{status}]{trunc_flag} pred={predicted[:30]}... gt={ground_truth[:30]}... ({gen_time:.1f}s, {gen_result['generated_tokens']} tok)")
            
            samples.append({
                # Problem info
                "dataset_idx": dataset_idx,
                "source": source,
                "problem": problem,
                "problem_truncated": problem[:500],
                
                # Hint info
                "hint_pct": pct,
                "hint_paragraphs_total": len(paragraphs),
                "hint_paragraphs_used": int(len(paragraphs) * pct) if pct > 0 else 0,
                "hint_token_estimate": hint_token_estimate,
                
                # Generation info
                "generation": response,
                "generation_truncated": response[:1000],
                "input_tokens": gen_result["input_tokens"],
                "generated_tokens": gen_result["generated_tokens"],
                "hit_token_limit": gen_result["hit_token_limit"],
                "generation_time_sec": gen_time,
                
                # Answer extraction
                "predicted_raw": predicted,
                "predicted_normalized": normalize_answer(predicted) if predicted else "",
                "ground_truth_raw": ground_truth,
                "ground_truth_normalized": normalize_answer(ground_truth) if ground_truth else "",
                "has_boxed_format": has_boxed,
                
                # Evaluation
                "correct": correct
            })
        
        accuracy = correct_count / len(test_set)
        format_rate = format_ok_count / len(test_set)
        truncation_rate = truncation_count / len(test_set)
        
        results[pct] = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(test_set),
            "format_ok_count": format_ok_count,
            "format_rate": format_rate,
            "truncation_count": truncation_count,
            "truncation_rate": truncation_rate,
            "avg_input_tokens": total_input_tokens / len(test_set),
            "avg_generated_tokens": total_generated_tokens / len(test_set),
            "samples": samples
        }
        
        print(f"\n{int(pct*100)}% hints: {accuracy*100:.1f}% ({correct_count}/{len(test_set)})")
        print(f"  Format OK: {format_rate*100:.1f}% | Truncated: {truncation_rate*100:.1f}%")
        
        # Save incrementally
        _save_results({"approach1_progressive_hints": results}, save_path)
    
    return results


# ============================================================================
# APPROACH 2: GPT-5 CURATED HINTS
# ============================================================================

HINT_SELECTION_PROMPT = """Given this math problem and its full solution trace, identify the most helpful 
intermediate reasoning steps that could guide a student WITHOUT revealing the final answer.

Problem: {problem}

Full Solution Trace:
{full_solution}

Extract 2-3 key reasoning steps that:
1. Set up the problem correctly
2. Identify the right approach/technique
3. Do NOT include the final answer or final calculation

Return ONLY the extracted helpful hints, nothing else."""


def get_gpt5_hint(client, problem: str, full_solution: str) -> str:
    """Use GPT-5 to extract helpful hints from the solution."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using gpt-4o as GPT-5 proxy
            messages=[
                {"role": "user", "content": HINT_SELECTION_PROMPT.format(
                    problem=problem,
                    full_solution=full_solution
                )}
            ],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT-5 API error: {e}")
        return ""


def evaluate_gpt5_hints(
    model, tokenizer, test_set: List[dict], 
    openai_api_key: str, save_path: str
) -> Dict[str, dict]:
    """Evaluate with and without GPT-5 curated hints."""
    
    if not HAS_OPENAI:
        print("ERROR: openai package not installed")
        return {}
    
    client = OpenAI(api_key=openai_api_key)
    results = {"no_hint": {}, "with_hint": {}}
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    
    for use_hint in [False, True]:
        mode = "with_hint" if use_hint else "no_hint"
        print(f"\n{'='*60}")
        print(f"APPROACH 2: GPT-5 Hints - {mode}")
        print('='*60)
        
        correct_count = 0
        format_ok_count = 0
        total_input_tokens = 0
        total_generated_tokens = 0
        truncation_count = 0
        samples = []
        
        for i, example in enumerate(test_set):
            problem = example['problem']
            dataset_idx = example.get('__index__', i)
            source = example.get('source', 'unknown')
            
            # Get hint if needed
            hint = ""
            if use_hint:
                r1_solution = ""
                for conv in example.get('conversations', []):
                    if conv.get('from') == 'assistant':
                        r1_solution = conv.get('value', '')
                        break
                hint = get_gpt5_hint(client, problem, r1_solution)
            
            # Build prompt
            if hint:
                user_content = f"Problem: {problem}\n\nHelpful hints:\n{hint}\n\nNow solve the problem."
            else:
                user_content = f"Problem: {problem}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            # Generate
            start_time = time.time()
            gen_result = generate_response(model, tokenizer, messages)
            gen_time = time.time() - start_time
            
            response = gen_result["text"]
            
            # Extract and evaluate
            predicted = extract_boxed_answer(response)
            ground_truth = extract_ground_truth(example['solution'])
            
            has_boxed = "\\boxed{" in response
            if has_boxed:
                format_ok_count += 1
            
            correct = is_correct(predicted, ground_truth)
            if correct:
                correct_count += 1
            
            total_input_tokens += gen_result["input_tokens"]
            total_generated_tokens += gen_result["generated_tokens"]
            if gen_result["hit_token_limit"]:
                truncation_count += 1
            
            status = "✓" if correct else "✗"
            trunc_flag = " [TRUNC]" if gen_result["hit_token_limit"] else ""
            print(f"{i+1}/{len(test_set)} [{status}]{trunc_flag} pred={predicted[:30]}... ({gen_time:.1f}s)")
            
            samples.append({
                "dataset_idx": dataset_idx,
                "source": source,
                "problem": problem,
                "problem_truncated": problem[:500],
                "gpt5_hint": hint,
                "gpt5_hint_truncated": hint[:500] if hint else "",
                "generation": response,
                "generation_truncated": response[:1000],
                "input_tokens": gen_result["input_tokens"],
                "generated_tokens": gen_result["generated_tokens"],
                "hit_token_limit": gen_result["hit_token_limit"],
                "generation_time_sec": gen_time,
                "predicted_raw": predicted,
                "predicted_normalized": normalize_answer(predicted) if predicted else "",
                "ground_truth_raw": ground_truth,
                "ground_truth_normalized": normalize_answer(ground_truth) if ground_truth else "",
                "has_boxed_format": has_boxed,
                "correct": correct
            })
        
        accuracy = correct_count / len(test_set)
        format_rate = format_ok_count / len(test_set)
        truncation_rate = truncation_count / len(test_set)
        
        results[mode] = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(test_set),
            "format_ok_count": format_ok_count,
            "format_rate": format_rate,
            "truncation_count": truncation_count,
            "truncation_rate": truncation_rate,
            "avg_input_tokens": total_input_tokens / len(test_set),
            "avg_generated_tokens": total_generated_tokens / len(test_set),
            "samples": samples
        }
        
        print(f"\n{mode}: {accuracy*100:.1f}% ({correct_count}/{len(test_set)})")
        print(f"  Format OK: {format_rate*100:.1f}% | Truncated: {truncation_rate*100:.1f}%")
        
        _save_results({"approach2_gpt5_hints": results}, save_path)
    
    return results


# ============================================================================
# APPROACH 3: TRADITIONAL ICL
# ============================================================================

def evaluate_traditional_icl(
    model, tokenizer, test_set: List[dict], icl_pool: List[dict],
    k_values: List[int], save_path: str
) -> Dict[int, dict]:
    """Evaluate with k ICL examples from other problems."""
    
    results = {}
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"APPROACH 3: Traditional ICL - k={k}")
        print('='*60)
        
        # Sample k ICL examples (same for all test problems in this k)
        icl_examples = random.sample(icl_pool, min(k, len(icl_pool))) if k > 0 else []
        icl_indices = [icl_pool.index(ex) for ex in icl_examples] if icl_examples else []
        
        correct_count = 0
        format_ok_count = 0
        total_input_tokens = 0
        total_generated_tokens = 0
        truncation_count = 0
        samples = []
        
        for i, example in enumerate(test_set):
            problem = example['problem']
            dataset_idx = example.get('__index__', i)
            source = example.get('source', 'unknown')
            
            # Build prompt with ICL examples
            messages = [{"role": "system", "content": system_prompt}]
            
            icl_info = []
            for icl_ex in icl_examples:
                icl_problem = icl_ex['problem']
                icl_source = icl_ex.get('source', 'unknown')
                # Get the solution from conversations
                icl_solution = ""
                for conv in icl_ex.get('conversations', []):
                    if conv.get('from') == 'assistant':
                        icl_solution = conv.get('value', '')
                        break
                
                messages.append({"role": "user", "content": f"Problem: {icl_problem}"})
                messages.append({"role": "assistant", "content": icl_solution})
                
                icl_info.append({
                    "problem_preview": icl_problem[:200],
                    "source": icl_source,
                    "solution_length": len(icl_solution)
                })
            
            messages.append({"role": "user", "content": f"Problem: {problem}"})
            
            # Generate
            start_time = time.time()
            gen_result = generate_response(model, tokenizer, messages)
            gen_time = time.time() - start_time
            
            response = gen_result["text"]
            
            # Extract and evaluate
            predicted = extract_boxed_answer(response)
            ground_truth = extract_ground_truth(example['solution'])
            
            has_boxed = "\\boxed{" in response
            if has_boxed:
                format_ok_count += 1
            
            correct = is_correct(predicted, ground_truth)
            if correct:
                correct_count += 1
            
            total_input_tokens += gen_result["input_tokens"]
            total_generated_tokens += gen_result["generated_tokens"]
            if gen_result["hit_token_limit"]:
                truncation_count += 1
            
            status = "✓" if correct else "✗"
            trunc_flag = " [TRUNC]" if gen_result["hit_token_limit"] else ""
            print(f"{i+1}/{len(test_set)} [{status}]{trunc_flag} pred={predicted[:30]}... gt={ground_truth[:30]}... ({gen_time:.1f}s, in={gen_result['input_tokens']}, out={gen_result['generated_tokens']})")
            
            samples.append({
                "dataset_idx": dataset_idx,
                "source": source,
                "problem": problem,
                "problem_truncated": problem[:500],
                "k": k,
                "icl_indices": icl_indices,
                "icl_examples_info": icl_info,
                "generation": response,
                "generation_truncated": response[:1000],
                "input_tokens": gen_result["input_tokens"],
                "generated_tokens": gen_result["generated_tokens"],
                "hit_token_limit": gen_result["hit_token_limit"],
                "generation_time_sec": gen_time,
                "predicted_raw": predicted,
                "predicted_normalized": normalize_answer(predicted) if predicted else "",
                "ground_truth_raw": ground_truth,
                "ground_truth_normalized": normalize_answer(ground_truth) if ground_truth else "",
                "has_boxed_format": has_boxed,
                "correct": correct
            })
        
        accuracy = correct_count / len(test_set)
        format_rate = format_ok_count / len(test_set)
        truncation_rate = truncation_count / len(test_set)
        
        results[k] = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(test_set),
            "format_ok_count": format_ok_count,
            "format_rate": format_rate,
            "truncation_count": truncation_count,
            "truncation_rate": truncation_rate,
            "avg_input_tokens": total_input_tokens / len(test_set),
            "avg_generated_tokens": total_generated_tokens / len(test_set),
            "icl_indices_used": icl_indices,
            "samples": samples
        }
        
        print(f"\nk={k}: {accuracy*100:.1f}% ({correct_count}/{len(test_set)})")
        print(f"  Format OK: {format_rate*100:.1f}% | Truncated: {truncation_rate*100:.1f}% | Avg input: {total_input_tokens/len(test_set):.0f} tok")
        
        _save_results({"approach3_traditional_icl": results}, save_path)
    
    return results


# ============================================================================
# UTILITIES
# ============================================================================

def _save_results(results: dict, save_path: str):
    """Save results to JSON, merging with existing data."""
    existing = {}
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r') as f:
                existing = json.load(f)
        except:
            pass
    
    existing.update(results)
    existing["last_updated"] = datetime.now().isoformat()
    
    with open(save_path, 'w') as f:
        json.dump(existing, f, indent=2, default=str)


def print_summary(results: dict):
    """Print final summary."""
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if "approach1_progressive_hints" in results:
        print("\nApproach 1 - Progressive Hints:")
        for pct, data in sorted(results["approach1_progressive_hints"].items()):
            if isinstance(data, dict) and "accuracy" in data:
                print(f"  {int(float(pct)*100):3d}%: {data['accuracy']*100:.1f}%")
    
    if "approach2_gpt5_hints" in results:
        print("\nApproach 2 - GPT-5 Hints:")
        for mode, data in results["approach2_gpt5_hints"].items():
            if isinstance(data, dict) and "accuracy" in data:
                print(f"  {mode}: {data['accuracy']*100:.1f}%")
    
    if "approach3_traditional_icl" in results:
        print("\nApproach 3 - Traditional ICL:")
        for k, data in sorted(results["approach3_traditional_icl"].items(), key=lambda x: int(x[0])):
            if isinstance(data, dict) and "accuracy" in data:
                print(f"  k={int(k)}: {data['accuracy']*100:.1f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Positive Control: Validate CoT helps performance")
    parser.add_argument("--approach", type=int, choices=[1, 2, 3], nargs="+", default=[1, 3],
                       help="Which approach(es) to run: 1=progressive hints, 2=GPT-5 hints, 3=traditional ICL")
    parser.add_argument("-n", "--num-test", type=int, default=50, help="Number of test problems")
    parser.add_argument("--num-icl", type=int, default=500, help="Size of ICL pool")
    parser.add_argument("--openai-key", type=str, default=None, help="OpenAI API key for approach 2")
    parser.add_argument("--test", action="store_true", help="Quick test with 5 problems")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Quick test mode
    if args.test:
        args.num_test = 5
        print("TEST MODE: 5 problems only")
    
    # Output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"positive_control_{timestamp}.json"
    
    print(f"Output: {args.output}")
    print(f"Approaches: {args.approach}")
    print(f"Test problems: {args.num_test}")
    
    # Load data
    icl_pool, test_set, data_metadata = load_openthoughts_math(args.num_icl, args.num_test, args.seed)
    
    # Load model
    model, tokenizer = load_model()
    
    # Initialize results with comprehensive config
    all_results = {
        "config": {
            "approaches": args.approach,
            "num_test": args.num_test,
            "num_icl_pool": args.num_icl,
            "seed": args.seed,
            "model": "Qwen/Qwen2.5-Math-7B-Instruct",
            "max_new_tokens": 2048,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "do_sample": True,
            "timestamp": datetime.now().isoformat()
        },
        "data_metadata": data_metadata
    }
    
    # Run approaches
    if 1 in args.approach:
        percentages = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        results1 = evaluate_progressive_hints(model, tokenizer, test_set, percentages, args.output)
        all_results["approach1_progressive_hints"] = results1
    
    if 2 in args.approach:
        if args.openai_key:
            results2 = evaluate_gpt5_hints(model, tokenizer, test_set, args.openai_key, args.output)
            all_results["approach2_gpt5_hints"] = results2
        else:
            print("\nSkipping Approach 2: --openai-key not provided")
    
    if 3 in args.approach:
        k_values = [0, 1, 3, 5, 7, 9]
        results3 = evaluate_traditional_icl(model, tokenizer, test_set, icl_pool, k_values, args.output)
        all_results["approach3_traditional_icl"] = results3
    
    # Final save and summary
    _save_results(all_results, args.output)
    print_summary(all_results)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

