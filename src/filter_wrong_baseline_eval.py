import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a baseline_eval JSON file for incorrect results."
    )
    parser.add_argument(
        "--input",
        default="results/baseline_eval_20260121_022109.json",
        help="Path to the baseline_eval JSON file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path for filtered JSON. "
            "Defaults to '<input>_wrong.json'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input)

    with open(input_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    results = data.get("results", [])
    wrong = [item for item in results if item.get("correct") is False]

    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_wrong{ext or '.json'}"
    else:
        output_path = os.path.abspath(output_path)

    output = {
        "config": data.get("config", {}),
        "progress": {
            "total": len(results),
            "wrong": len(wrong),
            "correct": sum(1 for item in results if item.get("correct") is True),
            "skipped": sum(1 for item in results if item.get("skipped") is True),
        },
        "results": wrong,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    print(f"Wrote {len(wrong)} wrong results to {output_path}")


if __name__ == "__main__":
    main()
