#!/usr/bin/env python3
"""
Gist Validation: Pre-trained LLaMA-7B Gist Model on Math Hints

Self-contained script — no imports from the steering repo.
Imports from the gisting repo via sys.path for model loading/inference.

Pipeline:
  Phase 1 (baseline): Run ~200 problems without hints, find wrong ones
  Phase 2 (hints):    GPT generates behavior hints for wrong ones
  Phase 3 (full_hint): Rerun with full hints, keep >=25 that improved
  Phase 4 (gist_eval): Compress hints through gist token, evaluate

Usage:
  export OPENAI_API_KEY=sk-...
  python src/gist_llama_validation.py \
      --gisting-repo /workspace/gisting \
      --model-path /workspace/gisting/llama-7b-gist-reconstructed \
      --phase all \
      --n-problems 200 \
      --target-improved 25 \
      --hint-model gpt-4.1 \
      --seed 42 \
      --output /workspace/gist_validation_results.json
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset

# ============================================================================
# Answer Utilities (self-contained, copied from hint_and_rerun.py)
# ============================================================================

try:
    from latex2sympy2 import latex2sympy
    from sympy import N, simplify

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


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
        return text[content_start : current_idx - 1]
    return ""


def extract_ground_truth(solution: str) -> str:
    """Extract ground truth answer from solution field."""
    boxed = extract_boxed_answer(solution)
    if boxed:
        return boxed
    answer_match = re.search(
        r"(?:answer|result)\s*(?:is|=)\s*[:\s]*([^\n\.]+)", solution, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).strip()
    return ""


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace(" ", "").replace(",", "")
    return s


def is_correct(predicted: str, ground_truth: str) -> Tuple[bool, Dict]:
    debug: Dict[str, Any] = {
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
    pred_clean = normalize_answer(predicted)
    gt_clean = normalize_answer(ground_truth)
    debug["pred_normalized"] = pred_clean
    debug["gt_normalized"] = gt_clean
    if pred_clean == gt_clean:
        debug["match_type"] = "exact_string"
        return True, debug
    if HAS_SYMPY:
        try:
            pred_sym = latex2sympy(predicted)
            gt_sym = latex2sympy(ground_truth)
            pred_val = float(N(pred_sym))
            gt_val = float(N(gt_sym))
            if abs(pred_val - gt_val) < 1e-6:
                debug["match_type"] = "numeric_sympy"
                return True, debug
        except Exception:
            pass
        try:
            pred_sym = latex2sympy(predicted)
            gt_sym = latex2sympy(ground_truth)
            if simplify(pred_sym - gt_sym) == 0:
                debug["match_type"] = "symbolic_equiv"
                return True, debug
        except Exception:
            pass
    debug["match_type"] = "no_match"
    return False, debug


# ============================================================================
# Hint Generation (OpenAI client, copied templates from hint_and_rerun.py)
# ============================================================================

STRONG_REFLECTION_PROMPT_TEMPLATE = """Here is the definition of a behavior:
•A behavior is a note or skill to keep in mind while solving math problems.
•It can be a strategy, a trick, or a technique.
•It can also be a general rule or a common sense principle.
•A behavior is not a solution to the problem, but it can be used to solve the problem.

For example - if the problem is "Find the area of a circle with radius 4", one useful behaviour could be {{"behavior_area_of_circle": area of a circle is pi*r^2}}.

Given a problem and the corresponding solution, reflect and critique the solutions along the following dimensions:
1. Correctness Analysis: Is the answer mathematically correct? Are there calculation errors? Is the reasoning logically sound? Are all steps properly justified? What specific mistakes were made?
2. Missing Behaviors Analysis: What behaviors should have been used but weren't? Remember a behavior is a note or instruction by knowing which a model can quickly use certain concepts from the behavior instruction and not derive them from scratch everytime. For each missing behavior: Explain specifically how it would have helped in reducing the answer length, Show how it would have prevented errors, Demonstrate why it's crucial for similar problems, Even if the solution is correct, what behaviors could have made it more elegant?
3. New Behavior Suggestions: Suggest specific new behaviors that will help with similar problems. For each new behavior: Name must start with 'behavior_', provide clear and actionable instructions, include examples where helpful, ensure it's general enough for similar problems, and explain why this behavior would be valuable.

To make your behavior suggestions stronger, you can:
- explicitly diagnose mistakes and prescribe the exact fix
- name the exact theorem/technique to use and how to apply it
- state a specific error to avoid (what not to do)
- include one concrete intermediate step or relationship (no final result)

Problem:
{problem}

Model attempt:
{response}

Model predicted answer:
{predicted}

Ground-truth answer (do NOT reveal this in behaviors):
{ground_truth}

Ground-truth reasoning (for your reference; do NOT copy verbatim):
{solution}
"""

STRONG_BEHAVIOR_PROMPT_TEMPLATE = """{reflection_prompt}

<reflection>
{reflection}
</reflection>

Now, given this reflection generate a list of behaviors and corresponding instructions/ explanations in json format. Requirements:
- include only behaviors that directly address the diagnosed mistakes or missing skills
- each behavior must be actionable and specific (not vague)
- include at least one concrete intermediate step or relationship in each behavior
- avoid giving the final numeric answer

The format of each behavior is "behavior_[name]: [description]". The list should be in json format, and each behavior should be a key-value pair, where the key is the behavior name and the value is the description.
"""


def _format_behaviors_as_lines(hint: str) -> str:
    """Convert JSON (or ```json ... ```) behaviors to 'key: value' lines."""
    text = hint.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].lower().startswith("```json"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return "\n".join(f"{k}: {v}" for k, v in obj.items())
    except (json.JSONDecodeError, TypeError):
        pass
    return hint


SIMPLE_HINT_PROMPT_TEMPLATE = """A student is trying to solve the math problem below but got it wrong.

Problem:
{problem}

Student's attempt:
{response}

Student's answer: {predicted}

Write a hint that helps the student solve this correctly. You should:
- Name the exact theorem, formula, or technique needed
- Point out specifically where the student's reasoning went wrong
- Give key intermediate steps or relationships (e.g. "rewrite the expression as X", "apply identity Y to get Z")
- Do NOT reveal the final numeric/symbolic answer

Be specific and detailed — vague advice like "be more careful" is useless. 3-6 sentences.
"""


def generate_hint_simple(
    client: Any,
    model: str,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    """Single-call hint generation: one prompt, one response."""
    result: Dict[str, Any] = {"hint": "", "error": None}

    prompt = SIMPLE_HINT_PROMPT_TEMPLATE.format(
        problem=item.get("problem", ""),
        response=item.get("response", ""),
        predicted=item.get("predicted", ""),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=8192,
        )
        hint_text = (response.choices[0].message.content or "").strip()
        if not hint_text:
            result["error"] = "Empty response (reasoning model used all tokens?)"
            return result
        result["hint"] = hint_text
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def generate_hint_for_item(
    client: Any,
    model: str,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    """Two-stage hint generation: reflection -> behavior extraction."""
    result: Dict[str, Any] = {"hint": "", "error": None}

    reflection_prompt = STRONG_REFLECTION_PROMPT_TEMPLATE.format(
        problem=item.get("problem", ""),
        response=item.get("response", ""),
        predicted=item.get("predicted", ""),
        ground_truth=item.get("ground_truth", ""),
        solution=item.get("solution", ""),
    )

    try:
        reflection_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": reflection_prompt}],
            max_completion_tokens=8192,
        )
        reflection_text = (reflection_response.choices[0].message.content or "").strip()
        if not reflection_text:
            result["error"] = "Empty reflection response (reasoning model used all tokens?)"
            return result

        behavior_prompt = STRONG_BEHAVIOR_PROMPT_TEMPLATE.format(
            reflection_prompt=reflection_prompt,
            reflection=reflection_text,
        )
        behavior_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": behavior_prompt}],
            max_completion_tokens=8192,
        )
        behavior_text = (behavior_response.choices[0].message.content or "").strip()
        if not behavior_text:
            result["error"] = "Empty behavior response (reasoning model used all tokens?)"
            return result

        result["hint"] = behavior_text
        result["reflection"] = reflection_text
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


# ============================================================================
# LLaMA Inference (imports from gisting repo via sys.path)
# ============================================================================


def load_gist_model(
    gisting_repo: str, model_path: str, precision: str = "fp16"
):
    """Load pre-reconstructed GistLlamaForCausalLM and tokenizer."""
    # Add gisting repo to sys.path so we can import its modules
    if gisting_repo not in sys.path:
        sys.path.insert(0, gisting_repo)

    from src import gist_llama
    from src.gist_llama import GistLlamaForCausalLM
    from transformers import AutoConfig, LlamaTokenizer

    config = AutoConfig.from_pretrained(model_path)

    print(f"Loading GistLlamaForCausalLM from {model_path}...")
    model = GistLlamaForCausalLM.from_pretrained(model_path, config=config)

    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float,
    }
    model = model.to(dtypes[precision]).cuda().eval()

    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    assert len(tokenizer) == gist_llama.PRETRAINED_VOCAB_SIZE + 1
    gist_token = tokenizer.additional_special_tokens_ids[-1]

    return model, tokenizer, gist_token


@torch.inference_mode()
def generate_no_hint(
    model: Any,
    tokenizer: Any,
    problem: str,
    max_new_tokens: int = 1024,
) -> str:
    """Generate with gisting repo's Alpaca format (Instruction/Input/Output)."""
    prompt = (
        f"Instruction: Solve the following math problem step by step. "
        f"Put your final answer in \\boxed{{}}.\n"
        f"Input: {problem}\n"
        f"Output:"
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    # GistLlamaModel.forward() always uses attention_mask_gist (no None guard),
    # so pass all-ones to disable gist masking for non-gist generation.
    attn_mask = torch.ones_like(input_ids)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        attention_mask_gist=attn_mask[None, None],
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    # Strip at token level — string comparison is unreliable after decode
    output = tokenizer.decode(
        generated[0][input_ids.shape[1] :], skip_special_tokens=True
    )
    return output.strip()


@torch.inference_mode()
def generate_with_hint(
    model: Any,
    tokenizer: Any,
    problem: str,
    behaviors_block: str,
    max_new_tokens: int = 1024,
) -> str:
    """Generate with gisting repo's Alpaca format (Instruction/Input/Output)."""
    prompt = (
        f"Instruction: When solving this math problem, use the following hint:\n"
        f"{behaviors_block}\n\n"
        f"Solve the problem step by step. Put your final answer in \\boxed{{}}.\n"
        f"Input: {problem}\n"
        f"Output:"
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    attn_mask = torch.ones_like(input_ids)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        attention_mask_gist=attn_mask[None, None],
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    output = tokenizer.decode(
        generated[0][input_ids.shape[1] :], skip_special_tokens=True
    )
    return output.strip()


@torch.inference_mode()
def generate_with_gist(
    model: Any,
    tokenizer: Any,
    gist_token: int,
    problem: str,
    behaviors_block: str,
    num_gist_tokens: int = 1,
    max_new_tokens: int = 1024,
) -> str:
    """Compress behaviors through gist token, then generate.

    Follows compress.py's pipeline:
    1. Tokenize instruction with <GIST> suffix, forward to get gist activations
    2. Tokenize input with <GIST> prefix (for consistent tokenization), strip it
    3. Generate with past_key_values from gist activations
    """
    gist_str = "<GIST>" * num_gist_tokens

    # -- Step 1: Compress instruction --
    instruction_text = (
        f"Instruction: When solving this math problem, use the following hint:\n"
        f"{behaviors_block}\n\n"
        f"Solve the problem step by step. Put your final answer in \\boxed{{}}."
    )
    prepped_instruction = f"{instruction_text}\n{gist_str}"
    instruction_input_ids = tokenizer.encode(prepped_instruction)
    instruction_ids_tensor = (
        torch.tensor(instruction_input_ids).unsqueeze(0).cuda()
    )

    gist_activations = model.get_gist_activations(
        gist_token=gist_token,
        num_gist_tokens=num_gist_tokens,
        input_ids=instruction_ids_tensor,
        attention_mask=torch.ones_like(instruction_ids_tensor),
        attention_mask_gist=torch.ones_like(instruction_ids_tensor)[None, None],
    )

    # -- Step 2: Prepare input --
    # Add dummy <GIST> prefix for tokenization consistency, then strip it.
    # This ensures tokens after <GIST> are tokenized identically to how they
    # would be mid-sequence (tokenizers behave differently at string start).
    prepped_input = f"<GIST>\nInput: {problem}\nOutput:"
    input_ids_full = tokenizer.encode(prepped_input)
    try:
        gist_pos = input_ids_full.index(gist_token)
    except ValueError:
        raise RuntimeError(
            f"Gist token {gist_token} not found in encoded input. "
            f"Tokenizer may not have <GIST> as a special token. "
            f"Encoded ids: {input_ids_full[:20]}..."
        )
    input_ids = input_ids_full[gist_pos + 1 :]
    input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).cuda()

    attention_mask_with_gist = (
        torch.tensor([1] * (len(input_ids) + num_gist_tokens)).unsqueeze(0).cuda()
    )

    # -- Step 3: Generate --
    generated = model.generate(
        input_ids=input_ids_tensor,
        attention_mask=attention_mask_with_gist,
        attention_mask_gist=attention_mask_with_gist[None, None],
        past_key_values=gist_activations.past_key_values,
        gist_offset=gist_activations.gist_indices,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    # Strip at token level — generated[0] = input_ids + new tokens
    output = tokenizer.decode(
        generated[0][input_ids_tensor.shape[1] :], skip_special_tokens=True
    )
    return output.strip()


# ============================================================================
# Pipeline Phases
# ============================================================================


def load_math_problems(n_problems: int, seed: int) -> List[Dict[str, Any]]:
    """Load and sample problems from OpenThoughts-114k-math."""
    print("Loading open-r1/OpenThoughts-114k-math...")
    ds = load_dataset("open-r1/OpenThoughts-114k-math", split="train")
    print(f"Total examples: {len(ds)}")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n_problems, len(ds)))
    problems = []
    for idx in indices:
        row = ds[idx]
        gt = extract_ground_truth(row.get("solution", ""))
        if not gt:
            continue
        problems.append(
            {
                "idx": idx,
                "problem": row["problem"],
                "solution": row.get("solution", ""),
                "ground_truth": gt,
            }
        )
    print(f"Sampled {len(problems)} problems with extractable ground truth")
    return problems


def phase_baseline(
    model: Any,
    tokenizer: Any,
    problems: List[Dict[str, Any]],
    save_path: str,
    max_new_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """Phase 1: Run problems without hints, find wrong ones."""
    print(f"\n{'='*60}")
    print("PHASE 1: Baseline (no hint)")
    print(f"{'='*60}")

    results = []
    n_correct = 0
    n_empty = 0
    for i, prob in enumerate(problems):
        response = generate_no_hint(model, tokenizer, prob["problem"], max_new_tokens)
        predicted = extract_boxed_answer(response)
        if not predicted:
            n_empty += 1
        correct, debug = is_correct(predicted, prob["ground_truth"])
        if correct:
            n_correct += 1
        item = {
            **prob,
            "response": response,
            "predicted": predicted,
            "correct": correct,
            "match_type": debug.get("match_type", ""),
        }
        results.append(item)
        if (i + 1) % 10 == 0 or i == len(problems) - 1:
            print(
                f"  [{i+1}/{len(problems)}] correct={n_correct}, "
                f"wrong={i+1-n_correct}, empty={n_empty}"
            )
            # Periodic checkpoint
            with open(save_path, "w") as f:
                json.dump(
                    {"phase": "baseline", "results": results, "n_correct": n_correct,
                     "partial": i + 1 < len(problems)},
                    f, indent=2,
                )

    if n_empty > len(problems) * 0.5:
        print(
            f"WARNING: {n_empty}/{len(problems)} responses had no \\boxed{{}} answer. "
            f"Consider increasing --max-new-tokens."
        )

    wrong = [r for r in results if not r["correct"]]
    print(f"\nBaseline: {n_correct}/{len(problems)} correct, {len(wrong)} wrong")

    with open(save_path, "w") as f:
        json.dump(
            {"phase": "baseline", "results": results, "n_correct": n_correct},
            f,
            indent=2,
        )
    print(f"Saved to {save_path}")
    return results


def phase_hints(
    baseline_results: List[Dict[str, Any]],
    hint_model: str,
    save_path: str,
    simple_hint: bool = False,
) -> List[Dict[str, Any]]:
    """Phase 2: Generate hints for wrong problems."""
    from openai import OpenAI

    print(f"\n{'='*60}")
    print("PHASE 2: Generate hints")
    print(f"{'='*60}")

    hint_fn = generate_hint_simple if simple_hint else generate_hint_for_item
    mode = "simple (1 call)" if simple_hint else "two-stage (2 calls)"

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    wrong = [r for r in baseline_results if not r["correct"]]
    print(f"Generating hints for {len(wrong)} wrong problems using {hint_model} [{mode}]...")

    hinted = []
    for i, item in enumerate(wrong):
        hint_result = hint_fn(client, hint_model, item)
        hint_text = hint_result["hint"]
        item_with_hint = {
            **item,
            "hint_raw": hint_text,
            "hint_error": hint_result.get("error"),
            "behaviors_block": hint_text if simple_hint else _format_behaviors_as_lines(hint_text),
        }
        hinted.append(item_with_hint)
        status = "ok" if not hint_result.get("error") else "ERROR"
        snippet = hint_text[:150].replace("\n", " ") if hint_text else "(empty)"
        print(f"  [{i+1}/{len(wrong)}] {status} | {snippet}...")
        if (i + 1) % 10 == 0 or i == len(wrong) - 1:
            with open(save_path, "w") as f:
                json.dump({"phase": "hints", "results": hinted}, f, indent=2)

    valid = [h for h in hinted if not h.get("hint_error")]
    print(f"\nGenerated hints: {len(valid)} valid, {len(hinted)-len(valid)} errors")
    print(f"Saved to {save_path}")
    return hinted


def phase_full_hint(
    model: Any,
    tokenizer: Any,
    hinted_results: List[Dict[str, Any]],
    target_improved: int,
    save_path: str,
    max_new_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """Phase 3: Rerun with full hints, keep ones that improved."""
    print(f"\n{'='*60}")
    print("PHASE 3: Full hint evaluation")
    print(f"{'='*60}")

    valid = [h for h in hinted_results if not h.get("hint_error")]
    print(f"Evaluating {len(valid)} problems with full hints...")

    improved = []
    all_results = []
    n_empty = 0
    for i, item in enumerate(valid):
        response = generate_with_hint(
            model, tokenizer, item["problem"], item["behaviors_block"], max_new_tokens
        )
        predicted = extract_boxed_answer(response)
        if not predicted:
            n_empty += 1
        correct, debug = is_correct(predicted, item["ground_truth"])
        item_result = {
            **item,
            "full_hint_response": response,
            "full_hint_predicted": predicted,
            "full_hint_correct": correct,
            "full_hint_match_type": debug.get("match_type", ""),
        }
        all_results.append(item_result)
        if correct:
            improved.append(item_result)
        status = "✓" if correct else "✗"
        print(f"\n  [{i+1}/{len(valid)}] {status} pred={predicted!r} | gt={item['ground_truth']!r}")
        print(f"  HINT:     {item['behaviors_block'][:200].replace(chr(10), ' ')}...")
        print(f"  RESPONSE: {response[:400] if response else '(empty)'}")
        print(f"  {'─'*60}")

    print(
        f"\nFull hint: {len(improved)}/{len(valid)} improved "
        f"(need {target_improved}) | {n_empty} empty responses"
    )
    if len(improved) < target_improved:
        print(
            f"WARNING: Only {len(improved)} improved, target was {target_improved}. "
            f"Consider increasing --n-problems."
        )

    with open(save_path, "w") as f:
        json.dump(
            {
                "phase": "full_hint",
                "improved": improved,
                "all_results": all_results,
                "n_improved": len(improved),
                "n_evaluated": len(valid),
                "n_empty": n_empty,
            },
            f,
            indent=2,
        )
    print(f"Saved to {save_path}")
    return improved


def phase_gist_eval(
    model: Any,
    tokenizer: Any,
    gist_token: int,
    improved_results: List[Dict[str, Any]],
    target_improved: int,
    save_path: str,
    num_gist_tokens: int = 1,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Phase 4: Compress hints through gist token, evaluate."""
    print(f"\n{'='*60}")
    print("PHASE 4: Gisted hint evaluation")
    print(f"{'='*60}")

    # Cap at target_improved for a clean evaluation set
    eval_set = improved_results[:target_improved]
    print(f"Evaluating {len(eval_set)} problems with gisted hints...")

    n_correct = 0
    n_empty = 0
    results = []
    for i, item in enumerate(eval_set):
        response = generate_with_gist(
            model,
            tokenizer,
            gist_token,
            item["problem"],
            item["behaviors_block"],
            num_gist_tokens=num_gist_tokens,
            max_new_tokens=max_new_tokens,
        )
        predicted = extract_boxed_answer(response)
        if not predicted:
            n_empty += 1
        correct, debug = is_correct(predicted, item["ground_truth"])
        if correct:
            n_correct += 1

        item_result = {
            **item,
            "gist_response": response,
            "gist_predicted": predicted,
            "gist_correct": correct,
            "gist_match_type": debug.get("match_type", ""),
        }
        results.append(item_result)
        if (i + 1) % 5 == 0 or i == len(eval_set) - 1:
            print(f"  [{i+1}/{len(eval_set)}] gist correct: {n_correct}, empty: {n_empty}")

    if n_empty > 0:
        print(
            f"\nWARNING: {n_empty}/{len(eval_set)} gisted responses had no \\boxed{{}} answer."
        )
        if n_empty > len(eval_set) * 0.5:
            print(
                "  High empty rate may indicate gist compression is producing "
                "gibberish or the model cannot follow the compressed instruction."
            )

    # Final summary
    n_eval = len(eval_set)
    summary = {
        "n_evaluated": n_eval,
        "no_hint_correct": 0,
        "no_hint_accuracy": 0.0,
        "full_hint_correct": n_eval,
        "full_hint_accuracy": 1.0,
        "gist_correct": n_correct,
        "gist_accuracy": n_correct / n_eval if n_eval > 0 else 0.0,
        "num_gist_tokens": num_gist_tokens,
    }

    print(f"\n{'='*60}")
    print("=== Gist Validation Results ===")
    print(
        f"Problems evaluated: {n_eval} "
        f"(cherry-picked: wrong without hint, correct with full hint)"
    )
    print()
    print(f"{'Condition':<20} {'Correct':>8}  {'Accuracy':>8}")
    print(f"{'─'*20} {'─'*8}  {'─'*8}")
    print(
        f"{'No hint':<20} {'0/' + str(n_eval):>8}  {'0.0%':>8}     (by construction)"
    )
    print(
        f"{'Full hint':<20} {str(n_eval) + '/' + str(n_eval):>8}  "
        f"{'100.0%':>8}     (by construction)"
    )
    print(
        f"{'Gisted hint (k=' + str(num_gist_tokens) + ')':<20} "
        f"{str(n_correct) + '/' + str(n_eval):>8}  "
        f"{summary['gist_accuracy']*100:.1f}%".rjust(8)
        + "     ← the key result"
    )
    print(f"{'='*60}")

    final_output = {
        "phase": "gist_eval",
        "summary": summary,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(save_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved to {save_path}")
    return final_output


# ============================================================================
# CLI
# ============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gist Validation: Pre-trained LLaMA-7B Gist Model on Math Hints"
    )
    parser.add_argument(
        "--gisting-repo",
        type=str,
        default="/workspace/gisting",
        help="Path to cloned gisting repo",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/workspace/gisting/llama-7b-gist-reconstructed",
        help="Path to reconstructed gist model",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["baseline", "hints", "full_hint", "gist_eval", "all"],
        help="Which phase to run",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=200,
        help="Number of problems to sample for baseline",
    )
    parser.add_argument(
        "--target-improved",
        type=int,
        default=25,
        help="Target number of improved problems for gist evaluation",
    )
    parser.add_argument(
        "--hint-model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model for hint generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gist_validation_results.json",
        help="Output path for final results",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max new tokens for LLaMA generation",
    )
    parser.add_argument(
        "--num-gist-tokens",
        type=int,
        default=1,
        help="Number of gist tokens (must match model training)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Model precision",
    )
    parser.add_argument(
        "--simple-hint",
        action="store_true",
        help="Use single-call hint generation instead of two-stage reflection+behavior",
    )
    # Paths to intermediate results (for resuming phases)
    parser.add_argument("--baseline-file", type=str, default=None)
    parser.add_argument("--hints-file", type=str, default=None)
    parser.add_argument("--full-hint-file", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_all = args.phase == "all"
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Derive intermediate file paths
    base = os.path.splitext(args.output)[0]
    baseline_path = args.baseline_file or f"{base}_baseline.json"
    hints_path = args.hints_file or f"{base}_hints.json"
    full_hint_path = args.full_hint_file or f"{base}_full_hint.json"

    # Phases that need the model
    needs_model = args.phase in ("baseline", "full_hint", "gist_eval", "all")

    model, tokenizer, gist_token = None, None, None
    if needs_model:
        model, tokenizer, gist_token = load_gist_model(
            args.gisting_repo, args.model_path, args.precision
        )

    # ── Phase 1: Baseline ──
    baseline_results = None
    if run_all or args.phase == "baseline":
        problems = load_math_problems(args.n_problems, args.seed)
        baseline_results = phase_baseline(
            model, tokenizer, problems, baseline_path, args.max_new_tokens
        )
    elif args.phase in ("hints", "all"):
        # Only later phases that need baseline results should load it
        if os.path.exists(baseline_path):
            print(f"Loading baseline results from {baseline_path}")
            with open(baseline_path) as f:
                baseline_results = json.load(f)["results"]
        else:
            print(f"ERROR: Baseline file {baseline_path} not found. Run --phase baseline first.")
            sys.exit(1)

    if args.phase == "baseline":
        return

    # ── Phase 2: Hints ──
    hinted_results = None
    if run_all or args.phase == "hints":
        if baseline_results is None:
            print(f"ERROR: Baseline results required. Run --phase baseline first.")
            sys.exit(1)
        hinted_results = phase_hints(baseline_results, args.hint_model, hints_path, simple_hint=args.simple_hint)
    elif args.phase in ("full_hint", "all") or (args.phase == "gist_eval" and not os.path.exists(full_hint_path)):
        if os.path.exists(hints_path):
            print(f"Loading hints results from {hints_path}")
            with open(hints_path) as f:
                hinted_results = json.load(f)["results"]
        elif args.phase == "full_hint":
            print(f"ERROR: Hints file {hints_path} not found. Run --phase hints first.")
            sys.exit(1)

    if args.phase == "hints":
        return

    # ── Phase 3: Full hint ──
    improved_results = None
    if run_all or args.phase == "full_hint":
        if hinted_results is None:
            print(f"ERROR: Hints results required. Run --phase hints first.")
            sys.exit(1)
        improved_results = phase_full_hint(
            model,
            tokenizer,
            hinted_results,
            args.target_improved,
            full_hint_path,
            args.max_new_tokens,
        )
    elif args.phase == "gist_eval":
        if os.path.exists(full_hint_path):
            print(f"Loading full hint results from {full_hint_path}")
            with open(full_hint_path) as f:
                improved_results = json.load(f)["improved"]
        else:
            print(f"ERROR: Full hint file {full_hint_path} not found. Run --phase full_hint first.")
            sys.exit(1)

    if args.phase == "full_hint":
        return

    # ── Phase 4: Gist eval ──
    if run_all or args.phase == "gist_eval":
        if improved_results is None:
            print(f"ERROR: Full hint results required. Run --phase full_hint first.")
            sys.exit(1)
        phase_gist_eval(
            model,
            tokenizer,
            gist_token,
            improved_results,
            args.target_improved,
            args.output,
            num_gist_tokens=args.num_gist_tokens,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
