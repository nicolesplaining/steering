import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = Any  # type: ignore[misc,assignment]

from hint_and_rerun import llm_judge_correctness


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply LLM judging to an existing hint_and_rerun output."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON produced by hint_and_rerun.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: add _llm_judged suffix).",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1",
        help="OpenAI model to use for judging (default: gpt-4.1).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep between OpenAI requests (seconds).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print progress every N items (default: 10).",
    )
    return parser.parse_args()


def update_with_llm_judge(
    data: Dict[str, Any],
    client: OpenAI,
    judge_model: str,
    sleep_s: float,
    log_every: int,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = data.get("results", [])
    hinted_correct = 0
    improved = 0
    unchanged = 0
    still_wrong = 0
    skipped = 0

    for idx, item in enumerate(results, start=1):
        original = item.get("original", {})
        rerun = item.get("rerun", {})

        if rerun.get("skipped"):
            skipped += 1
            if log_every > 0 and (idx % log_every == 0 or idx == len(results)):
                print(
                    f"[{idx}/{len(results)}] skipped={skipped} hinted_correct={hinted_correct}",
                    flush=True,
                )
            continue

        problem = item.get("problem", "")
        model_response = rerun.get("response", "")
        ground_truth_solution = original.get("solution", "")

        correct, llm_judgment = llm_judge_correctness(
            client,
            judge_model,
            problem=problem,
            model_response=model_response,
            ground_truth_solution=ground_truth_solution,
        )

        rerun["correct"] = correct
        rerun["match_type"] = "llm_judge"
        rerun["llm_judgment"] = llm_judgment

        original_correct = bool(original.get("correct"))
        original_predicted = original.get("predicted", "")
        original_response = original.get("response", "")
        predicted_changed = (original_predicted or "") != (rerun.get("predicted") or "")
        response_changed = (original_response or "").strip() != (model_response or "").strip()

        if correct:
            hinted_correct += 1

        if correct and not original_correct:
            status = "improved"
            improved += 1
        elif not correct:
            if not predicted_changed:
                status = "unchanged"
                unchanged += 1
            else:
                status = "still_wrong"
                still_wrong += 1
        else:
            status = "unchanged"
            unchanged += 1

        item["delta"] = {
            "status": status,
            "correct_before": original_correct,
            "correct_after": bool(correct),
            "predicted_before": original_predicted,
            "predicted_after": rerun.get("predicted"),
            "predicted_changed": predicted_changed,
            "response_changed": response_changed,
            "match_type_before": original.get("match_type"),
            "match_type_after": "llm_judge",
        }

        if sleep_s > 0:
            time.sleep(sleep_s)

        if log_every > 0 and (idx % log_every == 0 or idx == len(results)):
            print(
                f"[{idx}/{len(results)}] correct={hinted_correct} "
                f"improved={improved} unchanged={unchanged} still_wrong={still_wrong} "
                f"skipped={skipped}",
                flush=True,
            )

    total = len(results)
    accuracy = hinted_correct / total if total else 0.0
    data["summary"] = {
        "total": total,
        "hinted_correct": hinted_correct,
        "accuracy": accuracy,
        "skipped": skipped,
        "hint_errors": data.get("summary", {}).get("hint_errors", 0),
        "improved": improved,
        "unchanged": unchanged,
        "still_wrong": still_wrong,
    }
    config = data.get("config", {})
    config["use_llm_judge"] = True
    config["judge_model"] = judge_model
    config["timestamp"] = datetime.now().isoformat()
    data["config"] = config
    return data


def main() -> None:
    args = parse_args()
    if not HAS_OPENAI:
        raise RuntimeError("openai package not installed. Please install it first.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    input_path = args.input
    output_path = args.output
    if output_path is None:
        if input_path.endswith(".json"):
            output_path = input_path.replace(".json", "_llm_judged.json")
        else:
            output_path = input_path + "_llm_judged.json"

    client = OpenAI(api_key=api_key)
    data = load_json(input_path)
    updated = update_with_llm_judge(
        data=data,
        client=client,
        judge_model=args.judge_model,
        sleep_s=args.sleep,
        log_every=args.log_every,
    )
    write_json(output_path, updated)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
