#!/usr/bin/env python3
"""
Aggregate sweep_results.json from multiple replicates (e.g. run_1..run_10)
into replicate_summary.json with mean ± std for vanilla/steered accuracy and improved count.

Usage:
  python aggregate_replicate_sweeps.py <parent_dir>
  e.g. python aggregate_replicate_sweeps.py results/run_10replicates_hinted_empty_16-27_20260203_120000
"""

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python aggregate_replicate_sweeps.py <parent_dir>", file=sys.stderr)
        sys.exit(1)
    parent = Path(sys.argv[1])
    if not parent.is_dir():
        print(f"Not a directory: {parent}", file=sys.stderr)
        sys.exit(1)

    accs_vanilla, accs_steered, improved_list = [], [], []
    config_input = None
    for i in range(1, 11):
        p = parent / f"run_{i}" / "sweep_results.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        if config_input is None and "config" in data:
            config_input = data["config"].get("input")
        for r in data.get("results", []):
            if r.get("baseline") == "hinted_minus_empty" and r.get("strength") == 1.0:
                accs_vanilla.append(r["vanilla_accuracy"])
                accs_steered.append(r["steered_accuracy"])
                improved_list.append(r["improved"])
                break

    n = len(accs_vanilla)
    if n == 0:
        print("No results to aggregate (no run_*/sweep_results.json with hinted_minus_empty, strength=1.0)")
        sys.exit(1)

    def mean(x):
        return sum(x) / n

    def std(x):
        m = mean(x)
        return (sum((v - m) ** 2 for v in x) / n) ** 0.5

    summary = {
        "config": {
            "input": config_input,
            "baseline": "hinted_minus_empty",
            "layers": list(range(16, 28)),
            "strength": 1.0,
            "num_replicates": n,
        },
        "vanilla_accuracy": {"mean": mean(accs_vanilla), "std": std(accs_vanilla), "values": accs_vanilla},
        "steered_accuracy": {"mean": mean(accs_steered), "std": std(accs_steered), "values": accs_steered},
        "improved": {"mean": mean(improved_list), "std": std(improved_list), "values": improved_list},
        "steered_better_count": sum(1 for v, s in zip(accs_vanilla, accs_steered) if s > v),
    }
    out = parent / "replicate_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    print(f"Vanilla accuracy: {mean(accs_vanilla):.4f} ± {std(accs_vanilla):.4f}")
    print(f"Steered accuracy: {mean(accs_steered):.4f} ± {std(accs_steered):.4f}")
    print(f"Steered better than vanilla in {summary['steered_better_count']}/{n} replicates")


if __name__ == "__main__":
    main()
