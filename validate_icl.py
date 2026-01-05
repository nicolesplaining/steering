"""
Validate that ICL with Chain-of-Thought improves model performance on math reasoning.
Tests k ∈ {0, 1, 3, 5, 7, 9} in-context examples.
Supports GSM8K and MATH datasets.
Compares Qwen3 Thinking mode vs Non-Thinking mode.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import json
from datetime import datetime
from typing import List, Tuple, Dict


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} format used in MATH dataset."""
    # Handle nested braces
    match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if match:
        return match.group(1).strip()
    return ""


def load_math_splits(num_icl: int = 15, num_test: int = 200):
    """Load MATH-500 dataset and split into ICL examples pool and test set."""
    # Load MATH-500 dataset (curated 500 problems from MATH)
    # Source: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
    # Fields: problem, solution (CoT), answer (final), subject, level
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    icl_pool = []
    for i in range(num_icl):
        item = dataset[i]
        icl_pool.append({
            "question": item["problem"],
            "answer": item["solution"]  # Full solution with CoT for ICL
        })
    
    test_problems = []
    for i in range(num_icl, min(num_icl + num_test, len(dataset))):
        item = dataset[i]
        # Use the dedicated 'answer' field directly
        test_problems.append((item["problem"], item["answer"]))
    
    return icl_pool, test_problems


def load_gsm8k_splits(num_icl: int = 15, num_test: int = 200):
    """Load GSM8K and split into ICL examples pool and test set."""
    dataset = load_dataset("gsm8k", "main", split="test")
    
    icl_pool = []
    for i in range(num_icl):
        item = dataset[i]
        icl_pool.append({
            "question": item["question"],
            "answer": item["answer"]
        })
    
    test_problems = []
    for i in range(num_icl, num_icl + num_test):
        item = dataset[i]
        answer = item["answer"]
        final_answer = answer.split("####")[-1].strip() if "####" in answer else answer
        test_problems.append((item["question"], final_answer))
    
    return icl_pool, test_problems


def create_chat_messages(icl_pool: List[dict], problem: str, k: int) -> List[dict]:
    """Create chat messages with k ICL examples."""
    messages = []
    
    messages.append({
        "role": "system",
        "content": "You are a helpful math tutor. Solve problems step by step and give the final numerical answer after ####."
    })
    
    for i in range(min(k, len(icl_pool))):
        ex = icl_pool[i]
        messages.append({"role": "user", "content": f"Problem: {ex['question']}"})
        messages.append({"role": "assistant", "content": ex['answer']})
    
    messages.append({"role": "user", "content": f"Problem: {problem}"})
    
    return messages


def extract_answer(text: str, dataset_type: str = "gsm8k") -> str:
    """Extract final answer from generated text."""
    # Remove thinking tags if present (Qwen3 thinking mode)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    
    # For MATH dataset, prioritize boxed format
    if dataset_type == "math":
        boxed = extract_boxed_answer(text)
        if boxed:
            return boxed
    
    if "####" in text:
        after_hash = text.split("####")[-1].strip()
        numbers = re.findall(r'-?\d+\.?\d*', after_hash)
        return numbers[0] if numbers else ""
    
    answer_match = re.search(r'answer is[:\s]*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Look for boxed answers
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    
    # Fallback: last number in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else ""


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.replace(",", "").replace("$", "").replace("%", "")
    s = s.replace("\\frac", "").replace("\\", "").replace("{", "").replace("}", "")
    s = s.replace(" ", "").strip().lower()
    return s


def is_correct(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    pred_clean = normalize_answer(predicted)
    gt_clean = normalize_answer(ground_truth)
    
    if not pred_clean:
        return False
    
    # Try numeric comparison first
    try:
        pred_num = float(pred_clean)
        gt_num = float(gt_clean)
        return abs(pred_num - gt_num) < 1e-3
    except:
        pass
    
    # String comparison
    return pred_clean == gt_clean


def evaluate_with_k_examples(
    model,
    tokenizer,
    icl_pool: List[dict],
    test_problems: List[Tuple[str, str]],
    k: int,
    device: str,
    enable_thinking: bool = True,
    max_new_tokens: int = 2048,
    dataset_type: str = "gsm8k",
) -> Tuple[float, int, int]:
    """Evaluate model with k ICL examples."""
    correct = 0
    total = len(test_problems)
    
    for idx, (problem, ground_truth) in enumerate(test_problems):
        print(f"\r  k={k}... {idx+1}/{total}", end="", flush=True)
        
        messages = create_chat_messages(icl_pool, problem, k)
        
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
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,  # Greedy for non-thinking
            }
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_kwargs,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        pred_answer = extract_answer(generated, dataset_type)
        if is_correct(pred_answer, ground_truth):
            correct += 1
    
    print()  # newline after progress
    accuracy = correct / total
    return accuracy, correct, total


def run_experiment(
    model,
    tokenizer,
    icl_pool: List[dict],
    test_problems: List[Tuple[str, str]],
    k_values: List[int],
    device: str,
    enable_thinking: bool,
    mode_name: str,
    dataset_type: str = "gsm8k",
) -> Dict[int, dict]:
    """Run full experiment for one mode."""
    print(f"\n{'='*60}")
    print(f"MODE: {mode_name}")
    print(f"{'='*60}")
    
    results = {}
    baseline_acc = None
    
    for k in k_values:
        accuracy, correct, total = evaluate_with_k_examples(
            model, tokenizer, icl_pool, test_problems, k, device, enable_thinking,
            dataset_type=dataset_type
        )
        results[k] = {"accuracy": accuracy, "correct": correct, "total": total}
        
        if k == 0:
            baseline_acc = accuracy
            print(f"    → {accuracy:.1%} ({correct}/{total})")
        else:
            delta = accuracy - baseline_acc
            print(f"    → {accuracy:.1%} ({correct}/{total}) [{delta:+.1%}]")
    
    return results


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
    USE_MATH = "--math" in sys.argv
    
    # Dataset selection
    dataset_name = "MATH" if USE_MATH else "GSM8K"
    
    if TEST_MODE:
        # Quick test settings (~5 min)
        model_name = "Qwen/Qwen3-8B"
        num_icl_pool = 10
        num_test = 5  # Just 5 problems
        k_values = [0, 3]  # Just 2 k values
        print(f"\n🧪 RUNNING IN TEST MODE (5 problems, k=[0,3], dataset={dataset_name})\n")
    else:
        # Full experiment settings (~3-6 hours)
        model_name = "Qwen/Qwen3-8B"
        num_icl_pool = 15
        num_test = 200  # Large sample for statistical significance
        k_values = [0, 1, 3, 5, 7, 9]
    
    print(f"{'='*60}")
    print(f"ICL + COT Validation Experiment")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Test problems: {num_test}")
    print(f"K values: {k_values}")
    print(f"Modes: Thinking + Non-Thinking")
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
    
    # Load model
    print(f"\nLoading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)
    print(f"Model loaded on {device}!")
    
    # Load data
    print(f"\nLoading {dataset_name} dataset...")
    if USE_MATH:
        icl_pool, test_problems = load_math_splits(num_icl_pool, num_test)
        dataset_type = "math"
    else:
        icl_pool, test_problems = load_gsm8k_splits(num_icl_pool, num_test)
        dataset_type = "gsm8k"
    print(f"ICL pool: {len(icl_pool)} examples")
    print(f"Test set: {len(test_problems)} problems")
    
    # Run experiments for both modes
    thinking_results = run_experiment(
        model, tokenizer, icl_pool, test_problems, k_values, device,
        enable_thinking=True, mode_name="THINKING (CoT reasoning)",
        dataset_type=dataset_type
    )
    
    non_thinking_results = run_experiment(
        model, tokenizer, icl_pool, test_problems, k_values, device,
        enable_thinking=False, mode_name="NON-THINKING (Direct answer)",
        dataset_type=dataset_type
    )
    
    # Print comparison
    print_comparison(thinking_results, non_thinking_results, k_values)
    
    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSIONS")
    print(f"{'='*70}")
    
    # Best thinking result
    best_k_think = max(k_values, key=lambda k: thinking_results[k]['accuracy'])
    best_acc_think = thinking_results[best_k_think]['accuracy']
    baseline_think = thinking_results[0]['accuracy']
    
    # Best non-thinking result
    best_k_no_think = max(k_values, key=lambda k: non_thinking_results[k]['accuracy'])
    best_acc_no_think = non_thinking_results[best_k_no_think]['accuracy']
    baseline_no_think = non_thinking_results[0]['accuracy']
    
    print(f"\nTHINKING MODE:")
    print(f"  Baseline (k=0): {baseline_think:.1%}")
    print(f"  Best (k={best_k_think}): {best_acc_think:.1%} ({best_acc_think - baseline_think:+.1%})")
    
    print(f"\nNON-THINKING MODE:")
    print(f"  Baseline (k=0): {baseline_no_think:.1%}")
    print(f"  Best (k={best_k_no_think}): {best_acc_no_think:.1%} ({best_acc_no_think - baseline_no_think:+.1%})")
    
    # Overall
    print(f"\nOVERALL:")
    if best_acc_think > best_acc_no_think:
        print(f"  ✓ Thinking mode wins: {best_acc_think:.1%} vs {best_acc_no_think:.1%}")
    else:
        print(f"  ✓ Non-thinking mode wins: {best_acc_no_think:.1%} vs {best_acc_think:.1%}")
    
    icl_helps_think = best_acc_think > baseline_think
    icl_helps_no_think = best_acc_no_think > baseline_no_think
    
    if icl_helps_think or icl_helps_no_think:
        print(f"  ✓ ICL+COT improves performance → Steering vectors JUSTIFIED")
    else:
        print(f"  ✗ ICL+COT does not help → Consider different approach")
    
    print(f"{'='*70}")
    
    # Save results to JSON
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "dataset": dataset_name,
        "num_test": num_test,
        "k_values": k_values,
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
    
    results_file = f"icl_results_{dataset_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
