#!/usr/bin/env python3
"""
Filter wrong items using an LLM judge and write a new JSON with only truly wrong items.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    if TYPE_CHECKING:
        from openai import OpenAI  # pragma: no cover
    else:
        OpenAI = Any  # type: ignore[misc,assignment]


JUDGE_PROMPT_TEMPLATE = """You are a strict grader for math problems.

Determine whether the model's response should be considered CORRECT given the problem
and the ground-truth solution. Judge semantic equivalence, not formatting.
Mark CORRECT only if the final answer is mathematically equivalent to the ground truth.
If units or form are required by the solution, they must be correct.

Return a JSON object with keys:
- verdict: "correct" or "incorrect"
- explanation: brief (1-2 sentences)

Problem:
{problem}

Ground-truth solution:
{solution}

Model response:
{response}

JSON:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-judge wrong items and filter to truly wrong ones."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON (e.g. baseline_eval_..._wrong.json).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path. Default: input with _llm_judged_wrong.json suffix.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4.1",
        help="OpenAI model for judging (default: gpt-4.1).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep between OpenAI requests (seconds).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of items to judge (0 = no limit).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N items (default: 10).",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def build_prompt(item: Dict[str, Any]) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        problem=item.get("problem", ""),
        solution=item.get("solution", ""),
        response=item.get("response", ""),
    )


def parse_judge_json(text: str) -> Dict[str, str]:
    text = (text or "").strip()
    if not text:
        return {"verdict": "incorrect", "explanation": "Empty judge response."}

    # Try strict JSON parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: extract JSON substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    lowered = text.lower()
    if "correct" in lowered and "incorrect" not in lowered:
        return {"verdict": "correct", "explanation": "Heuristic parse: mentions correct."}
    return {"verdict": "incorrect", "explanation": "Heuristic parse: defaulted to incorrect."}


def judge_item(client: OpenAI, model: str, item: Dict[str, Any]) -> Dict[str, str]:
    prompt = build_prompt(item)
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=256,
    )
    return parse_judge_json(response.output_text or "")


def main() -> None:
    args = parse_args()
    if not HAS_OPENAI:
        raise RuntimeError("openai package not installed. Please install it first.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    data = load_json(args.input)
    results = data.get("results", [])
    if args.limit and args.limit > 0:
        results = results[: args.limit]

    client = OpenAI(api_key=api_key)

    filtered: List[Dict[str, Any]] = []
    total = len(results)
    for idx, item in enumerate(results, start=1):
        verdict = judge_item(client, args.openai_model, item)
        is_correct = verdict.get("verdict", "").strip().lower() == "correct"
        if not is_correct:
            filtered.append(item)

        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"[Judge] {idx}/{total} processed")

        if args.sleep > 0:
            time.sleep(args.sleep)

    output_path = args.output or args.input.replace(".json", "_llm_judged_wrong.json")
    output = {
        "config": data.get("config", {}),
        "progress": {
            "total": len(filtered),
            "wrong": len(filtered),
            "correct": 0,
            "skipped": 0,
        },
        "results": filtered,
    }
    write_json(output_path, output)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
