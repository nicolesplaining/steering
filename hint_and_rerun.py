#!/usr/bin/env python3
"""
Generate GPT-5 hints for wrong baseline eval items, re-run Qwen with hints,
and report improvement metrics.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

from baseline_eval import (
    extract_boxed_answer,
    extract_ground_truth,
    generate_response,
    is_correct,
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


HINT_PROMPT_TEMPLATE = """You are helping improve a math solver.

Given the problem and the model's incorrect attempt, write a short hint that nudges
the solver toward the right method. The hint should:
- be concise (1-3 sentences)
- focus on the key idea or a mistaken step
- avoid giving the final answer or final numeric result
- avoid copying the ground-truth answer verbatim

Problem:
{problem}

Model attempt (incorrect):
{response}

Model predicted answer:
{predicted}

Match type (if provided):
{match_type}

Ground-truth answer (do NOT reveal this):
{ground_truth}

Write ONLY the hint:"""


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
        default="gpt-5",
        help="OpenAI model for hint generation (default: gpt-5).",
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
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def build_hint_prompt(item: Dict[str, Any]) -> str:
    problem = item.get("problem", "")
    response = item.get("response", "")
    predicted = item.get("predicted", "")
    ground_truth = item.get("ground_truth", "")
    match_type = item.get("match_type", "")
    return HINT_PROMPT_TEMPLATE.format(
        problem=problem,
        response=response,
        predicted=predicted,
        match_type=match_type,
        ground_truth=ground_truth,
    )


def generate_hint(client: OpenAI, model: str, item: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_hint_prompt(item)
    result: Dict[str, Any] = {
        "hint": "",
        "error": None,
        "openai_model": model,
        "openai_usage": None,
    }
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "minimal"},
            max_output_tokens=512,
        )
        hint_text = (response.output_text or "").strip()
        result["hint"] = hint_text
        if hasattr(response, "usage") and response.usage is not None:
            result["openai_usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_hinted_prompt(problem: str, hint: str) -> List[Dict[str, str]]:
    system_prompt = "You are a helpful math assistant."
    user_content = (
        f"Problem: {problem}\n\n"
        f"Hint: {hint}\n\n"
        "Solve step by step. Final answer in \\boxed{}."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def rerun_with_hint(model, tokenizer, item: Dict[str, Any], hint: str) -> Dict[str, Any]:
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
    correct, match_debug = is_correct(predicted, ground_truth)

    return {
        "skipped": False,
        "response": response_text,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "correct": correct,
        "match_type": match_debug.get("match_type"),
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
) -> str:
    data = load_json(input_path)
    results = data.get("results", [])

    client: Optional[OpenAI] = None
    if not mock_hints:
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

    for item in results:
        original_predicted = item.get("predicted", "")
        original_correct = bool(item.get("correct"))
        original_match_type = item.get("match_type")
        original_response = item.get("response", "")

        if mock_hints:
            hint_data = {
                "hint": mock_hint_text,
                "error": None,
                "openai_model": "mock",
                "openai_usage": None,
            }
        else:
            hint_data = generate_hint(client, openai_model, item)
        hint = hint_data.get("hint", "")
        if hint_data.get("error"):
            hint_errors += 1

        rerun = rerun_with_hint(model, tokenizer, item, hint)
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

    total = len(output_items)
    accuracy = hinted_correct / total if total else 0.0
    output = {
        "config": {
            "source_file": os.path.abspath(input_path),
            "hint_model": openai_model,
            "qwen_model": qwen_model,
            "device": device,
            "mock_hints": mock_hints,
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
        )
        output_paths.append(output_path)

    print("Wrote outputs:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
