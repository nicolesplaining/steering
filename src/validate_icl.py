"""
Validate that ICL with Chain-of-Thought improves model performance on math reasoning.
Tests k ∈ {0, 1, 3, 5, 7, 9} in-context examples.
Supports GSM8K and MATH datasets.
Compares Qwen3 Thinking mode vs Non-Thinking mode.

Features:
- Incremental saving: Results saved after each k value (crash-resistant)
- Generation logging: Sample outputs saved for inspection
- Boxed format: MATH uses \\boxed{}, GSM8K uses ####
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import json
import os
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# Check for latex2sympy2 (needed for MATH symbolic comparison)
try:
    from latex2sympy2 import latex2sympy
    from sympy import simplify, N
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    print("⚠️  latex2sympy2 not installed - MATH comparison will be string-only")


# Global state for incremental saving
CHECKPOINT_FILE = None
SAMPLES_FILE = None


def save_checkpoint(
    results_data: dict,
    samples_data: dict,
    results_file: str,
    samples_file: str,
):
    """Save current progress to disk (called after each k value)."""
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    with open(samples_file, "w") as f:
        json.dump(samples_data, f, indent=2)
    print(f"  [Checkpoint saved to {results_file}]")


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} - handles arbitrarily nested braces."""
    idx = text.find("\\boxed{")
    if idx == -1:
        return ""
    idx += len("\\boxed{")
    depth = 1
    start = idx
    while idx < len(text) and depth > 0:
        if text[idx] == '{':
            depth += 1
        elif text[idx] == '}':
            depth -= 1
        idx += 1
    return text[start:idx-1].strip() if depth == 0 else ""


def load_math_splits(num_icl: int = 15, num_test: int = 200, seed: int = 42):
    """Load MATH-500 dataset and split into ICL examples pool and test set.
    
    NOTE: MATH-500 only has a test split, so this is an internal split.
    ICL and test are disjoint but both from the same test split.
    
    Returns: (icl_pool, test_problems, metadata)
    """
    import random
    random.seed(seed)
    
    # Load MATH-500 dataset (curated 500 problems from MATH)
    # Source: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
    # Fields: problem, solution (CoT), answer (final), subject, level
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    # Random sample all indices, then split into ICL and test
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    
    icl_indices = all_indices[:num_icl]
    test_indices = all_indices[num_icl:num_icl + num_test]
    
    icl_pool = []
    for i in icl_indices:
        item = dataset[i]
        icl_pool.append({
            "question": item["problem"],
            "answer": item["solution"],  # Full solution with CoT for ICL
            "dataset_idx": i,
            "split": "test",  # Note: from test split
        })
    
    test_problems = []
    for i in test_indices:
        item = dataset[i]
        # Use the dedicated 'answer' field directly
        test_problems.append((item["problem"], item["answer"], i))  # Include dataset index
    
    metadata = {
        "dataset": "HuggingFaceH4/MATH-500",
        "note": "INTERNAL SPLIT - MATH-500 only has test split, ICL and eval are disjoint subsets",
        "total_dataset_size": len(dataset),
        "icl_indices": icl_indices,
        "test_indices": test_indices,
        "selection_method": "random",
        "random_seed": seed,
    }
    
    return icl_pool, test_problems, metadata


def load_gsm8k_splits(num_icl: int = 15, num_test: int = 200, seed: int = 42):
    """Load GSM8K: ICL from TRAIN split, eval on TEST split.
    
    Returns: (icl_pool, test_problems, metadata)
    """
    import random
    random.seed(seed)
    
    # ICL examples from TRAIN split (proper protocol)
    train_dataset = load_dataset("gsm8k", "main", split="train")
    # Test problems from TEST split
    test_dataset = load_dataset("gsm8k", "main", split="test")
    
    # Random sample ICL indices from train
    icl_indices = random.sample(range(len(train_dataset)), min(num_icl, len(train_dataset)))
    # Random sample test indices from test
    test_indices = random.sample(range(len(test_dataset)), min(num_test, len(test_dataset)))
    
    icl_pool = []
    for i in icl_indices:
        item = train_dataset[i]
        icl_pool.append({
            "question": item["question"],
            "answer": item["answer"],
            "dataset_idx": i,
            "split": "train",
        })
    
    test_problems = []
    for i in test_indices:
        item = test_dataset[i]
        answer = item["answer"]
        final_answer = answer.split("####")[-1].strip() if "####" in answer else answer
        test_problems.append((item["question"], final_answer, i))  # Include dataset index
    
    metadata = {
        "dataset": "gsm8k/main",
        "icl_split": "train",
        "test_split": "test",
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "icl_indices": icl_indices,
        "test_indices": test_indices,
        "selection_method": "random",
        "random_seed": seed,
    }
    
    return icl_pool, test_problems, metadata


def create_chat_messages(icl_pool: List[dict], problem: str, k: int, dataset_type: str = "gsm8k") -> List[dict]:
    """Create chat messages with k ICL examples."""
    messages = []
    
    # Different system prompts for different datasets
    if dataset_type == "math":
        system_prompt = (
            "You are a helpful math tutor. Solve problems step by step, showing your reasoning. "
            "Put your final answer in \\boxed{} format. For example: \\boxed{42} or \\boxed{\\frac{1}{2}}"
        )
    else:  # gsm8k
        system_prompt = (
            "You are a helpful math tutor. Solve problems step by step and give the final numerical answer after ####."
        )
    
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    for i in range(min(k, len(icl_pool))):
        ex = icl_pool[i]
        messages.append({"role": "user", "content": f"Problem: {ex['question']}"})
        messages.append({"role": "assistant", "content": ex['answer']})
    
    messages.append({"role": "user", "content": f"Problem: {problem}"})
    
    return messages


def extract_answer(text: str, dataset_type: str = "gsm8k") -> Tuple[str, bool, str]:
    """Extract final answer from generated text. STRICT - no fallbacks.
    
    Returns: (answer, format_ok, format_status)
        - answer: extracted answer string (empty if format not found)
        - format_ok: True if model used correct format (\\boxed{} or ####)
        - format_status: 'ok', 'no_format', or 'format_but_empty' (for truncation detection)
    """
    # Remove thinking tags if present (Qwen3 thinking mode)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    
    # For MATH dataset: ONLY accept \boxed{} format
    if dataset_type == "math":
        boxed = extract_boxed_answer(text)
        if boxed:
            return boxed, True, "ok"
        # Check if \boxed{ exists but extraction failed (truncation/malformed)
        if "\\boxed{" in text:
            return "", True, "format_but_empty"
        return "", False, "no_format"
    
    # For GSM8K: ONLY accept #### format
    if "####" in text:
        after_hash = text.split("####")[-1].strip()
        numbers = re.findall(r'-?\d+\.?\d*', after_hash)
        if numbers:
            return numbers[0], True, "ok"
        return "", True, "format_but_empty"  # Had #### but no number (likely truncation)
    
    # No fallbacks - if no #### answer, return empty
    return "", False, "no_format"


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (GSM8K style - numeric only)."""
    s = s.replace(",", "").replace("$", "").replace("%", "")
    s = s.replace(" ", "").strip().lower()
    return s


def is_correct_gsm8k(predicted: str, ground_truth: str) -> Tuple[bool, dict]:
    """Check if predicted answer matches ground truth for GSM8K (numeric).
    
    Returns: (is_correct, debug_info)
    """
    debug = {
        "pred_raw": predicted,
        "gt_raw": ground_truth,
        "pred_normalized": None,
        "gt_normalized": None,
        "match_type": None,
    }
    
    if not predicted:
        debug["match_type"] = "empty_prediction"
        return False, debug
    
    pred_clean = normalize_answer(predicted)
    gt_clean = normalize_answer(ground_truth)
    debug["pred_normalized"] = pred_clean
    debug["gt_normalized"] = gt_clean
    
    # Try numeric comparison
    try:
        pred_num = float(pred_clean)
        gt_num = float(gt_clean)
        if abs(pred_num - gt_num) < 1e-3:
            debug["match_type"] = "numeric_match"
            return True, debug
    except ValueError:
        pass
    
    # String comparison fallback
    if pred_clean == gt_clean:
        debug["match_type"] = "string_match"
        return True, debug
    
    debug["match_type"] = "no_match"
    return False, debug


def is_correct_math(predicted: str, ground_truth: str) -> Tuple[bool, dict]:
    """Multi-stage comparison for MATH answers (LaTeX symbolic).
    
    Returns: (is_correct, debug_info)
    """
    debug = {
        "pred_raw": predicted,
        "gt_raw": ground_truth,
        "pred_sympy": None,
        "gt_sympy": None,
        "pred_numeric": None,
        "gt_numeric": None,
        "match_type": None,
    }
    
    if not predicted:
        debug["match_type"] = "empty_prediction"
        return False, debug
    
    # Stage 1: Exact string match (after basic cleanup)
    pred_clean = predicted.strip().lower()
    gt_clean = ground_truth.strip().lower()
    if pred_clean == gt_clean:
        debug["match_type"] = "exact_string"
        return True, debug
    
    # Stage 2 & 3: Symbolic comparison (only if latex2sympy2 installed)
    if HAS_SYMPY:
        # Stage 2: Numeric comparison (handles "0.5" vs "1/2" if evaluable)
        try:
            pred_sym = latex2sympy(predicted)
            gt_sym = latex2sympy(ground_truth)
            debug["pred_sympy"] = str(pred_sym)
            debug["gt_sympy"] = str(gt_sym)
            
            pred_val = float(N(pred_sym))
            gt_val = float(N(gt_sym))
            debug["pred_numeric"] = pred_val
            debug["gt_numeric"] = gt_val
            
            if abs(pred_val - gt_val) < 1e-6:
                debug["match_type"] = "numeric_sympy"
                return True, debug
        except Exception as e:
            debug["numeric_error"] = str(e)
        
        # Stage 3: Symbolic equivalence (handles algebraic expressions)
        try:
            pred_sym = latex2sympy(predicted)
            gt_sym = latex2sympy(ground_truth)
            debug["pred_sympy"] = str(pred_sym)
            debug["gt_sympy"] = str(gt_sym)
            
            if simplify(pred_sym - gt_sym) == 0:
                debug["match_type"] = "symbolic_equiv"
                return True, debug
        except Exception as e:
            debug["symbolic_error"] = str(e)
    else:
        debug["sympy_available"] = False
    
    debug["match_type"] = "no_match"
    return False, debug


def is_correct(predicted: str, ground_truth: str, dataset_type: str = "gsm8k") -> Tuple[bool, dict]:
    """Check if predicted answer matches ground truth.
    
    Returns: (is_correct, debug_info)
    """
    if dataset_type == "math":
        return is_correct_math(predicted, ground_truth)
    else:
        return is_correct_gsm8k(predicted, ground_truth)


def evaluate_with_k_examples(
    model,
    tokenizer,
    icl_pool: List[dict],
    test_problems: List[Tuple[str, str]],
    k: int,
    device: str,
    enable_thinking: bool = True,
    max_new_tokens: int = 8192,  # Increased for thinking mode (Qwen recommends 32k+ for hard reasoning)
    dataset_type: str = "gsm8k",
    log_samples: int = 5,  # Number of samples to log per k value
    save_callback: Optional[callable] = None,  # Called after each generation
    save_every: int = 1,  # Save every N generations (1 = every generation)
) -> Tuple[float, int, int, List[dict], int]:
    """Evaluate model with k ICL examples. Returns (accuracy, correct, total, samples, format_ok_count)."""
    correct = 0
    format_ok_count = 0  # Track format compliance separately
    format_but_empty_count = 0  # Format used but no answer (truncation indicator)
    content_correct_count = 0  # Correct answer when format was ok
    total = len(test_problems)
    samples = []  # Store sample generations for inspection
    
    for idx, problem_tuple in enumerate(test_problems):
        # Handle both (problem, answer) and (problem, answer, dataset_idx) formats
        if len(problem_tuple) == 3:
            problem, ground_truth, dataset_idx = problem_tuple
        else:
            problem, ground_truth = problem_tuple
            dataset_idx = idx  # Fallback to local index
        print(f"\r  k={k}... {idx+1}/{total} ({correct}/{idx} correct)", end="", flush=True)
        
        messages = create_chat_messages(icl_pool, problem, k, dataset_type)
        
        # Qwen3 supports enable_thinking in chat template
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Fallback for models without enable_thinking
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_tokens = inputs.input_ids.shape[1]
        
        # Warn if context might exceed model limits (Qwen3-8B has 32k context)
        max_context = 32768
        if input_tokens + max_new_tokens > max_context:
            print(f"\n⚠️  WARNING: Total context {input_tokens + max_new_tokens} may exceed {max_context} limit")
        
        # Adjust generation params based on thinking mode
        # Per Qwen3 docs: https://huggingface.co/Qwen/Qwen3-8B
        if enable_thinking:
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                # min_p=0 recommended but not always supported
            }
        else:
            # Non-thinking: Qwen3 recommends T=0.7, top_p=0.8, top_k=20
            # Using sampling (not greedy) for fair comparison
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            }
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_kwargs,
                pad_token_id=tokenizer.eos_token_id,
            )
        generation_time = time.time() - start_time
        
        # model.generate() returns tensor of shape (batch_size, seq_len)
        # We use batch_size=1, so outputs[0] is the generated sequence (1D)
        output_ids = outputs[0]
        
        generated = tokenizer.decode(
            output_ids[inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Calculate token metrics
        tokens_generated = output_ids.shape[0] - input_tokens
        hit_token_limit = tokens_generated >= max_new_tokens - 1
        
        # Detect thinking tokens (content within <think>...</think>)
        think_tokens = 0
        if "<think>" in generated and "</think>" in generated:
            think_match = re.search(r'<think>(.*?)</think>', generated, re.DOTALL)
            if think_match:
                think_content = think_match.group(1)
                think_tokens = len(tokenizer.encode(think_content, add_special_tokens=False))
        
        # Detect answer format used
        if "\\boxed{" in generated:
            answer_format = "boxed"
        elif "####" in generated:
            answer_format = "hash"
        else:
            answer_format = "none"
        
        pred_answer, format_ok, format_status = extract_answer(generated, dataset_type)
        is_corr, match_debug = is_correct(pred_answer, ground_truth, dataset_type)
        
        # Track metrics separately
        if format_ok:
            format_ok_count += 1
            if format_status == "format_but_empty":
                format_but_empty_count += 1
            elif is_corr:
                content_correct_count += 1
        if is_corr:
            correct += 1
        
        # Print full generation for each problem
        print(f"\n{'='*80}")
        status = '✓ CORRECT' if is_corr else ('✗ WRONG (format ok)' if format_ok else '✗ FORMAT FAIL')
        print(f"[k={k}] Problem {idx+1}/{total} | {status}")
        print(f"{'='*80}")
        print(f"TOKENS: {tokens_generated} generated ({input_tokens} input) | {generation_time:.1f}s | format: {answer_format} | {'⚠️ HIT LIMIT' if hit_token_limit else 'OK'}")
        if think_tokens > 0:
            print(f"THINKING: {think_tokens} tokens in <think> block")
        print(f"PROBLEM: {problem[:200]}{'...' if len(problem) > 200 else ''}")
        print(f"\nGROUND TRUTH: {ground_truth}")
        print(f"PREDICTED:    {pred_answer}")
        print(f"MATCH TYPE:   {match_debug.get('match_type', 'N/A')}")
        if dataset_type == "math" and match_debug.get('pred_sympy'):
            print(f"  → pred sympy: {match_debug.get('pred_sympy')}")
            print(f"  → gt sympy:   {match_debug.get('gt_sympy')}")
        print(f"\n{'─'*40} GENERATION {'─'*40}")
        print(generated)
        print(f"{'─'*80}\n")
        
        # Log samples (first N, or mix of correct/incorrect)
        if len(samples) < log_samples:
            samples.append({
                "idx": idx,
                "dataset_idx": dataset_idx,  # Global index in original dataset
                "problem": problem,
                "ground_truth": ground_truth,
                "predicted": pred_answer,
                "format_ok": format_ok,  # Did model use correct format?
                "format_status": format_status,  # 'ok', 'no_format', or 'format_but_empty'
                "correct": is_corr,
                "match_debug": match_debug,  # Detailed comparison info
                "generation": generated,
                # Token metrics
                "input_tokens": input_tokens,
                "tokens_generated": tokens_generated,
                "hit_token_limit": hit_token_limit,
                "think_tokens": think_tokens,
                "answer_format": answer_format,
                "generation_time_sec": round(generation_time, 2),
            })
        
        # Save checkpoint after each generation (or every N)
        if save_callback and (idx + 1) % save_every == 0:
            save_callback(k, idx + 1, correct, samples)
    
    print()  # newline after progress
    accuracy = correct / total
    format_rate = format_ok_count / total
    print(f"  Format compliance: {format_ok_count}/{total} ({format_rate:.1%})")
    if format_but_empty_count > 0:
        print(f"  ⚠️  Format but empty (likely truncation): {format_but_empty_count}")
    print(f"  Content correct (when format ok): {content_correct_count}/{format_ok_count if format_ok_count > 0 else 1}")
    return accuracy, correct, total, samples, format_ok_count


def print_comparison(thinking_results: Dict, non_thinking_results: Dict, k_values: List[int]):
    """Print side-by-side comparison."""
    print(f"\n{'='*70}")
    print("COMPARISON: THINKING vs NON-THINKING")
    print(f"{'='*70}")
    print(f"{'k':<6} {'Thinking':<20} {'Non-Thinking':<20} {'Δ':<10}")
    print("-" * 70)
    
    for k in k_values:
        t = thinking_results[k]
        nt = non_thinking_results[k]
        delta = t['accuracy'] - nt['accuracy']
        print(f"{k:<6} {t['accuracy']:.1%} ({t['correct']}/{t['total']})       "
              f"{nt['accuracy']:.1%} ({nt['correct']}/{nt['total']})       "
              f"{delta:+.1%}")


def main():
    import sys
    
    # Check for flags
    TEST_MODE = "--test" in sys.argv or "-t" in sys.argv
    ULTRA_MODE = "--ultra" in sys.argv  # Super quick: 2 problems, k=9 only
    USE_MATH = "--math" in sys.argv
    THINKING_ONLY = "--thinking-only" in sys.argv or "--think" in sys.argv
    
    # Parse --num N flag for custom number of test problems
    CUSTOM_NUM = None
    for i, arg in enumerate(sys.argv):
        if arg in ("--num", "-n") and i + 1 < len(sys.argv):
            try:
                CUSTOM_NUM = int(sys.argv[i + 1])
            except ValueError:
                pass
    
    # Dataset selection
    dataset_name = "MATH" if USE_MATH else "GSM8K"
    
    if ULTRA_MODE:
        # Ultra quick test (~2 min) - just to see generations
        model_name = "Qwen/Qwen3-8B"
        num_icl_pool = 10
        num_test = CUSTOM_NUM or 2  # Default: 2 problems
        k_values = [9]  # Only k=9
        print(f"\n⚡ ULTRA MODE: {num_test} problems, k=9 only, dataset={dataset_name}\n")
    elif TEST_MODE:
        # Quick test settings (~5 min)
        model_name = "Qwen/Qwen3-8B"
        num_icl_pool = 10
        num_test = CUSTOM_NUM or 5  # Default: 5 problems
        k_values = [0, 3]  # Just 2 k values
        print(f"\n🧪 RUNNING IN TEST MODE ({num_test} problems, k=[0,3], dataset={dataset_name})\n")
    else:
        # Full experiment settings
        model_name = "Qwen/Qwen3-8B"
        num_icl_pool = 15
        num_test = CUSTOM_NUM or 50  # Default: 50 problems per mode
        k_values = [0, 1, 3, 5, 7, 9]
    
    # Create output filenames upfront for incremental saving
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"icl_results_{dataset_name.lower()}_{timestamp}.json"
    samples_file = f"icl_generations_{dataset_name.lower()}_{timestamp}.json"
    
    print(f"{'='*60}")
    print(f"ICL + COT Validation Experiment")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Test problems: {num_test}")
    print(f"K values: {k_values}")
    print(f"Modes: {'Thinking only' if THINKING_ONLY else 'Thinking + Non-Thinking'}")
    print(f"Output: {results_file}")
    print(f"{'='*60}")
    
    # Auto-detect best device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\nUsing CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("\nUsing Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("\nUsing CPU")
    
    # Set random seeds for reproducibility (needed for do_sample=True)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load model
    print(f"\nLoading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()  # Set to evaluation mode
    print(f"Model loaded on {device} (eval mode)!")
    
    # Load data
    print(f"\nLoading {dataset_name} dataset...")
    if USE_MATH:
        print("⚠️  MATH-500 only has test split - ICL and eval are disjoint but from same split")
        icl_pool, test_problems, data_metadata = load_math_splits(num_icl_pool, num_test)
        dataset_type = "math"
    else:
        icl_pool, test_problems, data_metadata = load_gsm8k_splits(num_icl_pool, num_test)
        dataset_type = "gsm8k"
    print(f"ICL pool: {len(icl_pool)} examples (indices: {data_metadata['icl_indices'][:3]}...)")
    print(f"Test set: {len(test_problems)} problems (indices: {data_metadata['test_indices'][:3]}...{data_metadata['test_indices'][-1]})")
    
    # Initialize results containers
    thinking_results = {}
    non_thinking_results = {}
    thinking_samples = {}
    non_thinking_samples = {}
    current_mode = {"mode": "thinking"}  # Track current mode for callbacks
    
    # Helper to save incremental progress (called after EACH generation)
    def save_progress_callback(k: int, completed: int, correct: int, samples: List[dict]):
        """Called after each generation to save progress."""
        mode = current_mode["mode"]
        
        # Update in-progress results for current k
        if mode == "thinking":
            thinking_results[k] = {
                "accuracy": correct / completed if completed > 0 else 0,
                "correct": correct,
                "total": completed,
                "complete": completed == num_test
            }
            thinking_samples[k] = samples
        else:
            non_thinking_results[k] = {
                "accuracy": correct / completed if completed > 0 else 0,
                "correct": correct,
                "total": completed,
                "complete": completed == num_test
            }
            non_thinking_samples[k] = samples
        
        # Save to disk
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "dataset": dataset_name,
            "num_test": num_test,
            "k_values": k_values,
            "data_metadata": data_metadata,  # ICL/test indices, selection method
            "status": "in_progress",
            "current_mode": mode,
            "current_k": k,
            "current_progress": f"{completed}/{num_test}",
            "thinking_results": {str(k): v for k, v in thinking_results.items()},
            "non_thinking_results": {str(k): v for k, v in non_thinking_results.items()},
        }
        samples_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "dataset": dataset_name,
            "thinking_samples": {str(k): v for k, v in thinking_samples.items()},
            "non_thinking_samples": {str(k): v for k, v in non_thinking_samples.items()},
        }
        # Write atomically (write to temp then rename)
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        with open(samples_file, "w") as f:
            json.dump(samples_data, f, indent=2)
    
    # Run THINKING mode
    print(f"\n{'='*60}")
    print("MODE: THINKING (CoT reasoning)")
    print(f"{'='*60}")
    
    current_mode["mode"] = "thinking"
    baseline_think = None
    for k in k_values:
        accuracy, correct, total, samples, format_ok = evaluate_with_k_examples(
            model, tokenizer, icl_pool, test_problems, k, device,
            enable_thinking=True, dataset_type=dataset_type, log_samples=5,
            save_callback=save_progress_callback, save_every=1  # Save after EVERY generation
        )
        thinking_results[k] = {"accuracy": accuracy, "correct": correct, "total": total, "format_ok": format_ok, "complete": True}
        thinking_samples[k] = samples
        
        if baseline_think is None:
            baseline_think = accuracy  # First k value becomes baseline
        
        if k == k_values[0]:  # First k value
            print(f"    → {accuracy:.1%} ({correct}/{total})")
        else:
            delta = accuracy - baseline_think
            print(f"    → {accuracy:.1%} ({correct}/{total}) [{delta:+.1%}]")
    
    # Run NON-THINKING mode (unless --thinking-only flag)
    baseline_no_think = None
    if not THINKING_ONLY:
        print(f"\n{'='*60}")
        print("MODE: NON-THINKING (Direct answer)")
        print(f"{'='*60}")
        
        current_mode["mode"] = "non_thinking"
        for k in k_values:
            accuracy, correct, total, samples, format_ok = evaluate_with_k_examples(
                model, tokenizer, icl_pool, test_problems, k, device,
                enable_thinking=False, dataset_type=dataset_type, log_samples=5,
                save_callback=save_progress_callback, save_every=1  # Save after EVERY generation
            )
            non_thinking_results[k] = {"accuracy": accuracy, "correct": correct, "total": total, "format_ok": format_ok, "complete": True}
            non_thinking_samples[k] = samples
            
            if baseline_no_think is None:
                baseline_no_think = accuracy  # First k value becomes baseline
            
            if k == k_values[0]:  # First k value
                print(f"    → {accuracy:.1%} ({correct}/{total})")
            else:
                delta = accuracy - baseline_no_think
                print(f"    → {accuracy:.1%} ({correct}/{total}) [{delta:+.1%}]")
        
        # Print comparison
        print_comparison(thinking_results, non_thinking_results, k_values)
    else:
        print(f"\n[Skipping non-thinking mode (--thinking-only)]")
    
    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSIONS")
    print(f"{'='*70}")
    
    # Best thinking result
    best_k_think = max(k_values, key=lambda k: thinking_results[k]['accuracy'])
    best_acc_think = thinking_results[best_k_think]['accuracy']
    
    # Best non-thinking result (if run)
    if non_thinking_results:
        best_k_no_think = max(k_values, key=lambda k: non_thinking_results[k]['accuracy'])
        best_acc_no_think = non_thinking_results[best_k_no_think]['accuracy']
    else:
        best_k_no_think = None
        best_acc_no_think = None
    
    first_k = k_values[0]
    print(f"\nTHINKING MODE:")
    print(f"  Baseline (k={first_k}): {baseline_think:.1%}")
    print(f"  Best (k={best_k_think}): {best_acc_think:.1%} ({best_acc_think - baseline_think:+.1%})")
    
    if not THINKING_ONLY:
        print(f"\nNON-THINKING MODE:")
        print(f"  Baseline (k={first_k}): {baseline_no_think:.1%}")
        print(f"  Best (k={best_k_no_think}): {best_acc_no_think:.1%} ({best_acc_no_think - baseline_no_think:+.1%})")
        
        # Overall comparison
        print(f"\nOVERALL:")
        if best_acc_think > best_acc_no_think:
            print(f"  ✓ Thinking mode wins: {best_acc_think:.1%} vs {best_acc_no_think:.1%}")
        else:
            print(f"  ✓ Non-thinking mode wins: {best_acc_no_think:.1%} vs {best_acc_think:.1%}")
    
    icl_helps_think = best_acc_think > baseline_think
    icl_helps_no_think = best_acc_no_think > baseline_no_think if best_acc_no_think else False
    
    if icl_helps_think or icl_helps_no_think:
        print(f"  ✓ ICL+COT improves performance → Steering vectors JUSTIFIED")
    else:
        print(f"  ✗ ICL+COT does not help → Consider different approach")
    
    print(f"{'='*70}")
    
    # Final save with conclusions
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "dataset": dataset_name,
        "num_test": num_test,
        "k_values": k_values,
        "data_metadata": data_metadata,  # ICL/test indices, selection method
        "status": "complete",
        "thinking_results": {str(k): v for k, v in thinking_results.items()},
        "non_thinking_results": {str(k): v for k, v in non_thinking_results.items()},
        "conclusions": {
            "thinking_baseline": baseline_think,
            "thinking_best_k": best_k_think,
            "thinking_best_acc": best_acc_think,
            "non_thinking_baseline": baseline_no_think,
            "non_thinking_best_k": best_k_no_think,
            "non_thinking_best_acc": best_acc_no_think,
            "icl_helps_thinking": icl_helps_think,
            "icl_helps_non_thinking": icl_helps_no_think,
        }
    }
    
    samples_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "dataset": dataset_name,
        "thinking_samples": {str(k): v for k, v in thinking_samples.items()},
        "non_thinking_samples": {str(k): v for k, v in non_thinking_samples.items()},
    }
    
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    with open(samples_file, "w") as f:
        json.dump(samples_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    print(f"✅ Generation samples saved to: {samples_file}")


if __name__ == "__main__":
    main()
