import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector
from typing import List, Tuple, Optional
import numpy as np


def create_good_icl_prompt(examples: List[Tuple[str, str]], current_problem: str) -> str:
    prompt_parts = []
    for problem, solution in examples:
        prompt_parts.append(f"Problem: {problem}")
        prompt_parts.append(f"Solution: {solution}")
        prompt_parts.append("")  
    
    prompt_parts.append(f"Problem: {current_problem}")
    prompt_parts.append("Solution:")
    return "\n".join(prompt_parts)


def create_neutral_prompt(current_problem: str) -> str:
    return f"Problem: {current_problem}\nSolution:"


def create_no_prompt() -> str:
    return " "


def build_reasoning_steering_vector(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    good_icl_examples: List[Tuple[str, str]], 
    layers: Optional[List[int]] = None,
    num_samples: int = 10,
    baseline_type: str = "neutral",  # "neutral" or "empty"
) -> dict:
    if layers is None:
        num_layers = model.config.num_hidden_layers
        layers = list(range(num_layers // 2, num_layers))
    
    training_samples = []
    sample_problems = good_icl_examples[:num_samples]
    
    for problem, solution in sample_problems:
        other_examples = [ex for ex in good_icl_examples if ex != (problem, solution)][:3] 
        good_prompt = create_good_icl_prompt(other_examples, problem)
        
        if baseline_type == "neutral":
            baseline_prompt = create_neutral_prompt(problem)
        else:  # "empty"
            baseline_prompt = create_no_prompt()
        
        training_samples.append((good_prompt, baseline_prompt))
    
    steering_vector = train_steering_vector(
        model=model,
        tokenizer=tokenizer,
        training_samples=training_samples,
        show_progress=True,
        layers=layers,
    )
    
    return steering_vector


def build_reasoning_vectors_from_gsm8k(
    model_name: str = "Qwen/QwQ-32B-Preview",  # or Qwen/Qwen2.5-32B-Instruct
    num_examples: int = 10,
    layers: Optional[List[int]] = None,
    hf_token: Optional[str] = None,
) -> dict:
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    
    from datasets import load_dataset
    gsm8k = load_dataset("gsm8k", "main")
    train_subset = gsm8k["train"].select(range(min(num_examples, len(gsm8k["train"]))))
    good_icl_examples = [(item["question"], item["answer"]) for item in train_subset]
    
    print("Building reasoning steering vectors...")
    steering_vector = build_reasoning_steering_vector(
        model=model,
        tokenizer=tokenizer,
        good_icl_examples=good_icl_examples,
        layers=layers,
        num_samples=num_examples,
    )
    
    return steering_vector, model, tokenizer


if __name__ == "__main__":
    model_name = "Qwen/Qwen2-1.5B"
    num_examples = 5
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    from datasets import load_dataset
    gsm8k = load_dataset("gsm8k", "main")
    train_subset = gsm8k["train"].select(range(min(num_examples, len(gsm8k["train"]))))
    good_icl_examples = [(item["question"], item["answer"]) for item in train_subset]
    
    print("\n" + "="*60)
    print("BUILDING STEERING VECTORS")
    print("="*60)
    
    print("\n[1] ICL vs NEUTRAL (problem only)")
    sv_neutral = build_reasoning_steering_vector(
        model, tokenizer, good_icl_examples,
        num_samples=num_examples, baseline_type="neutral"
    )
    
    print("\n[2] ICL vs EMPTY (no prompt)")
    sv_empty = build_reasoning_steering_vector(
        model, tokenizer, good_icl_examples,
        num_samples=num_examples, baseline_type="empty"
    )
    
    print("\n" + "="*60)
    print("VECTOR COMPARISON")
    print("="*60)
    
    for layer_idx in sv_neutral.layer_activations.keys():
        norm_neutral = torch.norm(sv_neutral.layer_activations[layer_idx]).item()
        norm_empty = torch.norm(sv_empty.layer_activations[layer_idx]).item()
        print(f"Layer {layer_idx}: neutral={norm_neutral:.2f}, empty={norm_empty:.2f}")
    
    print("\n" + "="*60)
    print("GENERATION TEST")
    print("="*60)
    
    test_prompt = "Problem: What is 15 + 27?\nSolution:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    print("\n--- Without steering ---")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    
    for name, sv in [("NEUTRAL", sv_neutral), ("EMPTY", sv_empty)]:
        print(f"\n{'='*40}")
        print(f"Testing: ICL vs {name}")
        print("="*40)
        for strength in [0.1, 0.3, 0.5, 1.0]:
            print(f"\n--- α={strength} ---")
            with sv.apply(model, multiplier=strength):
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            print(tokenizer.decode(output[0], skip_special_tokens=True))
    
    torch.save(sv_neutral, "steering_vector_neutral.pt")
    torch.save(sv_empty, "steering_vector_empty.pt")
    print("\nSaved: steering_vector_neutral.pt, steering_vector_empty.pt")



