import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

from baseline_eval import (
    extract_boxed_answer,
    extract_ground_truth,
    generate_response,
    load_model,
)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    if TYPE_CHECKING:
        from openai import OpenAI  # pragma: no cover
    else:
        OpenAI = Any  # type: ignore[misc,assignment]

try:
    from latex2sympy2 import latex2sympy
    from sympy import simplify, N
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = s.replace(" ", "").replace(",", "")
    return s


def is_correct(predicted: str, ground_truth: str) -> tuple[bool, Dict]:
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
            debug["pred_sympy"] = str(pred_sym)
            debug["gt_sympy"] = str(gt_sym)

            pred_val = float(N(pred_sym))
            gt_val = float(N(gt_sym))
            debug["pred_numeric"] = pred_val
            debug["gt_numeric"] = gt_val

            if abs(pred_val - gt_val) < 1e-6:
                debug["match_type"] = "numeric_sympy"
                return True, debug
        except Exception as exc:
            debug["numeric_error"] = str(exc)

        try:
            pred_sym = latex2sympy(predicted)
            gt_sym = latex2sympy(ground_truth)
            debug["pred_sympy"] = str(pred_sym)
            debug["gt_sympy"] = str(gt_sym)

            if simplify(pred_sym - gt_sym) == 0:
                debug["match_type"] = "symbolic_equiv"
                return True, debug
        except Exception as exc:
            debug["symbolic_error"] = str(exc)
    else:
        debug["sympy_available"] = False

    debug["match_type"] = "no_match"
    return False, debug


REFLECTION_PROMPT_TEMPLATE = """Here is the definition of a behavior:
•A behavior is a note or skill to keep in mind while solving math problems.
•It can be a strategy, a trick, or a technique.
•It can also be a general rule or a common sense principle.
•A behavior is not a solution to the problem, but it can be used to solve the problem.

For example - if the problem is "Find the area of a circle with radius 4", one useful behaviour could be {{"behavior_area_of_circle": area of a circle is pi*r^2}}.

Given a problem and the corresponding solution, reflect and critique the solutions along the following dimensions:
1. Correctness Analysis: Is the answer mathematically correct? Are there calculation errors? Is the reasoning logically sound? Are all steps properly justified? What specific mistakes were made?
2. Missing Behaviors Analysis: What behaviors should have been used but weren't? Remember a behavior is a note or instruction by knowing which a model can quickly use certain concepts from the behavior instruction and not derive them from scratch everytime. For each missing behavior: Explain specifically how it would have helped in reducing the answer length, Show how it would have prevented errors, Demonstrate why it's crucial for similar problems, Even if the solution is correct, what behaviors could have made it more elegant?
3. New Behavior Suggestions: Suggest specific new behaviors that will help with similar problems. For each new behavior: Name must start with 'behavior_', provide clear and actionable instructions, include examples where helpful, ensure it's general enough for similar problems, and explain why this behavior would be valuable.

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

BEHAVIOR_PROMPT_TEMPLATE = """{reflection_prompt}

<reflection>
{reflection}
</reflection>

Now, given this reflection generate a list of behaviors and corresponding instructions/ explanations in json format. Each behavior should be a single line, and the format is "behavior_[name]: [description]". The list should be in json format, and each behavior should be a key-value pair, where the key is the behavior name and the value is the description.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GPT-5 hints and re-run Qwen on wrong items."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Input JSON file(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output JSONs.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4.1",
        help="OpenAI model for hint generation (default: gpt-4.1).",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use an OpenAI model to judge correctness instead of local matching.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1",
        help="OpenAI model for judging correctness (default: gpt-4.1).",
    )
    parser.add_argument(
        "--strong-hints",
        action="store_true",
        help="Use stronger, multi-hint instructions for OpenAI hints.",
    )
    parser.add_argument(
        "--num-hints",
        type=int,
        default=1,
        help="Number of hints to request per item (default: 1).",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Qwen model to re-run with hints.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep between OpenAI requests (seconds).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Force model device (default: auto).",
    )
    parser.add_argument(
        "--mock-hints",
        action="store_true",
        help="Use a placeholder hint instead of calling OpenAI.",
    )
    parser.add_argument(
        "--mock-hint-text",
        default="Focus on the key equation or symmetry; re-check arithmetic.",
        help="Hint text used when --mock-hints is set.",
    )
    parser.add_argument(
        "--use-solution-hint",
        action="store_true",
        help="Use the ground-truth solution text directly as the hint.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print progress every N items (default: 10).",
    )
    parser.add_argument(
        "--log-prompt",
        action="store_true",
        help="Print the hinted prompt and hint for each item.",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def build_reflection_prompt(item: Dict[str, Any]) -> str:
    problem = item.get("problem", "")
    response = item.get("response", "")
    predicted = item.get("predicted", "")
    ground_truth = item.get("ground_truth", "")
    match_type = item.get("match_type", "")
    solution = item.get("solution", "")
    return REFLECTION_PROMPT_TEMPLATE.format(
        problem=problem,
        response=response,
        predicted=predicted,
        ground_truth=ground_truth,
        solution=solution,
    )


def build_behavior_prompt(reflection_prompt: str, reflection_text: str) -> str:
    return BEHAVIOR_PROMPT_TEMPLATE.format(
        reflection_prompt=reflection_prompt,
        reflection=reflection_text,
    )


def parse_hints(text: str, num_hints: int) -> List[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    hints: List[str] = []
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    current: List[str] = []
    for line in lines:
        lower = line.lower()
        if lower.startswith("hint ") or lower.startswith("hint:") or lower.startswith("hint1") or lower.startswith("hint2"):
            if current:
                hints.append(" ".join(current).strip())
                current = []
            line = line.split(":", 1)[-1].strip() if ":" in line else line
            if line:
                current.append(line)
        else:
            current.append(line)
    if current:
        hints.append(" ".join(current).strip())

    if not hints:
        hints = [line for line in lines if line]

    if num_hints > 0:
        hints = hints[:num_hints]

    return [hint for hint in hints if hint]


def generate_hint(
    client: OpenAI,
    model: str,
    item: Dict[str, Any],
    strong_hints: bool = False,
    num_hints: int = 1,
) -> Dict[str, Any]:
    reflection_prompt = build_reflection_prompt(item)
    result: Dict[str, Any] = {
        "hint": "",
        "hints": [],
        "error": None,
        "openai_model": model,
        "openai_usage": None,
    }
    try:
        reflection_response = client.responses.create(
            model=model,
            input=reflection_prompt,
            max_output_tokens=512,
        )
        reflection_text = (reflection_response.output_text or "").strip()

        behavior_prompt = build_behavior_prompt(reflection_prompt, reflection_text)
        behavior_response = client.responses.create(
            model=model,
            input=behavior_prompt,
            max_output_tokens=512,
        )
        behavior_text = (behavior_response.output_text or "").strip()

        result["hint"] = behavior_text
        result["hints"] = [behavior_text] if behavior_text else []
        if hasattr(behavior_response, "usage") and behavior_response.usage is not None:
            result["openai_usage"] = {
                "input_tokens": behavior_response.usage.input_tokens,
                "output_tokens": behavior_response.usage.output_tokens,
                "total_tokens": behavior_response.usage.total_tokens,
            }
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_hinted_prompt(problem: str, hint: str) -> List[Dict[str, str]]:
    system_prompt = "You are a helpful math assistant."
    user_content = (
        f"Problem: {problem}\n\n"
        "A behavior is a note or skill to keep in mind while solving math problems. "
        "It can be a strategy, a trick, or a technique. It can also be a general rule "
        "or a common sense principle. The behavior is not a solution to the problem, "
        "but it can be used to solve the problem. Here is a list of behaviors:\n"
        f"{hint}\n\n"
        "Now, solve the following math problem efficiently and clearly. Use the behaviors above "
        "to solve the problem. In your reasoning, when you use a behavior explicitly refer to the "
        "behaviors when you use them. Please reason step by step and put the final answer in \\boxed{}."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def llm_judge_correctness(
    client: OpenAI,
    model: str,
    problem: str,
    model_response: str,
    ground_truth_solution: str,
) -> tuple[bool, str]:
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

    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=256,
    )
    judgment = (response.output_text or "").strip()
    is_correct_flag = judgment.upper().startswith("CORRECT")
    return is_correct_flag, judgment


def rerun_with_hint(
    model,
    tokenizer,
    item: Dict[str, Any],
    hint: str,
    use_llm_judge: bool,
    judge_model: str,
    judge_client: Optional[OpenAI],
) -> Dict[str, Any]:
    messages = build_hinted_prompt(item.get("problem", ""), hint)
    start_time = time.time()
    gen_result = generate_response(model, tokenizer, messages)
    gen_time = time.time() - start_time

    if gen_result.get("skipped"):
        return {
            "skipped": True,
            "skip_reason": gen_result.get("skip_reason"),
            "input_tokens": gen_result.get("input_tokens"),
            "generated_tokens": gen_result.get("generated_tokens", 0),
            "generation_time_sec": gen_time,
            "response": "",
            "predicted": "",
            "ground_truth": item.get("ground_truth", ""),
            "correct": False,
            "match_type": "skipped",
        }

    response_text = gen_result["text"]
    predicted = extract_boxed_answer(response_text)
    ground_truth = extract_ground_truth(item.get("solution", "")) or item.get("ground_truth", "")
    if use_llm_judge:
        if judge_client is None:
            raise RuntimeError("LLM judge requested but OpenAI client is not configured.")
        correct, llm_judgment = llm_judge_correctness(
            judge_client,
            judge_model,
            problem=item.get("problem", ""),
            model_response=response_text,
            ground_truth_solution=item.get("solution", ""),
        )
        match_debug = {"match_type": "llm_judge", "llm_judgment": llm_judgment}
    else:
        correct, match_debug = is_correct(predicted, ground_truth)

    return {
        "skipped": False,
        "response": response_text,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "correct": correct,
        "match_type": match_debug.get("match_type"),
        "llm_judgment": match_debug.get("llm_judgment"),
        "input_tokens": gen_result.get("input_tokens"),
        "generated_tokens": gen_result.get("generated_tokens"),
        "generation_time_sec": gen_time,
    }


def process_file(
    input_path: str,
    output_dir: str,
    openai_model: str,
    qwen_model: str,
    sleep_s: float,
    device: str,
    mock_hints: bool,
    mock_hint_text: str,
    use_solution_hint: bool,
    strong_hints: bool,
    num_hints: int,
    use_llm_judge: bool,
    judge_model: str,
    log_every: int,
    log_prompt: bool,
) -> str:
    data = load_json(input_path)
    results = data.get("results", [])

    client: Optional[OpenAI] = None
    needs_openai = (not mock_hints and not use_solution_hint) or use_llm_judge
    if needs_openai:
        if not HAS_OPENAI:
            raise RuntimeError("openai package not installed. Please install it first.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

        client = OpenAI(api_key=api_key)

    model, tokenizer = load_model(qwen_model)
    if device != "auto":
        model = model.to(torch.device(device))
        model.eval()

    output_items: List[Dict[str, Any]] = []
    hinted_correct = 0
    skipped = 0
    hint_errors = 0
    improved = 0
    unchanged = 0
    still_wrong = 0

    for idx, item in enumerate(results, start=1):
        original_predicted = item.get("predicted", "")
        original_correct = bool(item.get("correct"))
        original_match_type = item.get("match_type")
        original_response = item.get("response", "")

        if use_solution_hint:
            hint_data = {
                "hint": item.get("solution", ""),
                "hints": [item.get("solution", "")] if item.get("solution", "") else [],
                "error": None,
                "openai_model": "solution",
                "openai_usage": None,
            }
        elif mock_hints:
            hint_data = {
                "hint": mock_hint_text,
                "hints": [mock_hint_text] * max(1, num_hints) if mock_hint_text else [],
                "error": None,
                "openai_model": "mock",
                "openai_usage": None,
            }
        else:
            hint_data = generate_hint(client, openai_model, item, strong_hints, num_hints)
        hint = hint_data.get("hint", "")
        hints_list = hint_data.get("hints", [])
        if hint_data.get("error"):
            hint_errors += 1

        if log_prompt:
            messages = build_hinted_prompt(item.get("problem", ""), hint)
            print(
                "\n".join(
                    [
                        "=" * 80,
                        f"ITEM {idx}/{len(results)}",
                        "HINT:",
                        hint or "<empty hint>",
                        "PROMPT:",
                        messages[-1].get("content", ""),
                        "=" * 80,
                    ]
                ),
                flush=True,
            )

        rerun = rerun_with_hint(
            model,
            tokenizer,
            item,
            hint,
            use_llm_judge,
            judge_model,
            client,
        )
        if rerun.get("skipped"):
            skipped += 1
        if rerun.get("correct"):
            hinted_correct += 1
        predicted_changed = (original_predicted or "") != (rerun.get("predicted") or "")
        response_changed = (original_response or "").strip() != (rerun.get("response") or "").strip()

        if rerun.get("correct") and not original_correct:
            status = "improved"
            improved += 1
        elif not rerun.get("correct"):
            if not predicted_changed:
                status = "unchanged"
                unchanged += 1
            else:
                status = "still_wrong"
                still_wrong += 1
        else:
            status = "unchanged"
            unchanged += 1

        output_items.append(
            {
                "idx": item.get("idx"),
                "problem": item.get("problem"),
                "original": item,
                "hint": hint,
                "hints": hints_list,
                "hint_text": hint,
                "hint_error": hint_data.get("error"),
                "hint_openai_usage": hint_data.get("openai_usage"),
                "rerun": rerun,
                "delta": {
                    "status": status,
                    "correct_before": original_correct,
                    "correct_after": bool(rerun.get("correct")),
                    "predicted_before": original_predicted,
                    "predicted_after": rerun.get("predicted"),
                    "predicted_changed": predicted_changed,
                    "response_changed": response_changed,
                    "match_type_before": original_match_type,
                    "match_type_after": rerun.get("match_type"),
                },
            }
        )

        if sleep_s > 0:
            time.sleep(sleep_s)

        if log_every > 0 and (idx % log_every == 0 or idx == len(results)):
            print(
                f"[{idx}/{len(results)}] "
                f"correct={hinted_correct} improved={improved} "
                f"unchanged={unchanged} still_wrong={still_wrong} "
                f"skipped={skipped} hint_errors={hint_errors}",
                flush=True,
            )

    total = len(output_items)
    accuracy = hinted_correct / total if total else 0.0
    output = {
        "config": {
            "source_file": os.path.abspath(input_path),
            "hint_model": openai_model,
            "qwen_model": qwen_model,
            "device": device,
            "mock_hints": mock_hints,
            "use_solution_hint": use_solution_hint,
            "strong_hints": strong_hints,
            "num_hints": num_hints,
            "use_llm_judge": use_llm_judge,
            "judge_model": judge_model,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": {
            "total": total,
            "hinted_correct": hinted_correct,
            "accuracy": accuracy,
            "skipped": skipped,
            "hint_errors": hint_errors,
            "improved": improved,
            "unchanged": unchanged,
            "still_wrong": still_wrong,
        },
        "results": output_items,
    }

    base_name = os.path.basename(input_path)
    output_name = base_name.replace(".json", "_with_hints.json")
    output_path = os.path.join(output_dir, output_name)
    write_json(output_path, output)
    return output_path


def main() -> None:
    args = parse_args()
    if not args.input:
        args.input = [
            "baseline_eval_wrong_general.json",
            "baseline_eval_wrong_geometry.json",
        ]

    output_paths = []
    for input_path in args.input:
        output_path = process_file(
            input_path=input_path,
            output_dir=args.output_dir,
            openai_model=args.openai_model,
            qwen_model=args.qwen_model,
            sleep_s=args.sleep,
            device=args.device,
            mock_hints=args.mock_hints,
            mock_hint_text=args.mock_hint_text,
            use_solution_hint=args.use_solution_hint,
            strong_hints=args.strong_hints,
            num_hints=args.num_hints,
            use_llm_judge=args.use_llm_judge,
            judge_model=args.judge_model,
            log_every=args.log_every,
            log_prompt=args.log_prompt,
        )
        output_paths.append(output_path)

    print("Wrote outputs:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
