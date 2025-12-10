import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import SteeringVector
import json
import re
from collections import defaultdict


def extract_answer(solution: str) -> str:
    numbers = re.findall(r'-?\d+\.?\d*', solution)
    if numbers:
        return numbers[-1]
    return ""


def is_correct(predicted: str, ground_truth: str) -> bool:
    pred_clean = extract_answer(predicted).strip()
    gt_clean = extract_answer(ground_truth).strip()
    
    try:
        pred_num = float(pred_clean)
        gt_num = float(gt_clean)
        return abs(pred_num - gt_num) < 1e-6
    except:
        return pred_clean.lower() == gt_clean.lower()


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def generate_rollouts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    k: int,
    steering_vector: Optional[SteeringVector] = None,
    steering_strengths: Optional[List[float]] = None,
    start_after_tokens: int = 0,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    rollouts = []
    
    for i in range(k):
        if steering_vector is not None:
            if steering_strengths is not None:
                alpha = steering_strengths[i]
            else:
                alpha = 1.0 
            
            with steering_vector.apply(model, multiplier=alpha, min_token_index=start_after_tokens):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1,
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
            )
        
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        rollouts.append(generated_text)
    
    return rollouts


def evaluate_pass_at_k(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problems: List[Tuple[str, str]], 
    k: int = 10,
    steering_vector: Optional[SteeringVector] = None,
    steering_strengths: Optional[List[float]] = None,
    start_after_tokens: int = 0,
    use_steering: bool = False,
) -> Dict[str, float]:
    results = defaultdict(list)
    
    for problem, ground_truth in problems:
        rollouts = generate_rollouts(
            model=model,
            tokenizer=tokenizer,
            prompt=problem,
            k=k,
            steering_vector=steering_vector if use_steering else None,
            steering_strengths=steering_strengths if use_steering else None,
            start_after_tokens=start_after_tokens if use_steering else 0,
        )
        
        correct = [is_correct(rollout, ground_truth) for rollout in rollouts]
        num_correct = sum(correct)
        
        for k_val in range(1, k + 1):
            pass_score = pass_at_k(len(rollouts), num_correct, k_val)
            results[k_val].append(pass_score)
    
    avg_pass_at_k = {
        k_val: np.mean(scores) for k_val, scores in results.items()
    }
    
    return avg_pass_at_k


def compare_vanilla_vs_steered(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problems: List[Tuple[str, str]],
    steering_vector: SteeringVector,
    k_max: int = 10,
    steering_strengths: Optional[List[float]] = None,
    start_after_tokens: int = 0,
) -> Dict[str, Dict[int, float]]:
    print("Evaluating vanilla rollouts...")
    vanilla_results = evaluate_pass_at_k(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        k=k_max,
        use_steering=False,
    )
    
    print("Evaluating steered rollouts...")
    steered_results = evaluate_pass_at_k(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        k=k_max,
        steering_vector=steering_vector,
        steering_strengths=steering_strengths,
        start_after_tokens=start_after_tokens,
        use_steering=True,
    )
    
    return {
        "vanilla": vanilla_results,
        "steered": steered_results,
    }


def compute_diversity(rollouts: List[str]) -> float:
    unique_solutions = set([extract_answer(r) for r in rollouts])
    return len(unique_solutions) / len(rollouts) if rollouts else 0.0


if __name__ == "__main__":
    model_name = "Qwen/Qwen2-1.5B"  #small model rn
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # example problems for now
    problems = [
        ("What is 2 + 2?", "4"),
        ("What is 3 * 5?", "15"),
    ]
    
    steering_vector_path = "steering_vector_neutral.pt"

    def load_steering_vector(path: str):
        with torch.serialization.safe_globals([SteeringVector]):
            return torch.load(path, weights_only=False, map_location="cpu")

    steering_vector = (
        load_steering_vector(steering_vector_path)
        if isinstance(steering_vector_path, str)
        else steering_vector_path
    )
    
    results = compare_vanilla_vs_steered(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        steering_vector=steering_vector,
        k_max=10,
    )
    
    print("Evaluation framework ready")

