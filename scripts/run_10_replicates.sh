#!/bin/bash
# Run 10 replicates of sweep_steering_experiments with:
#   baseline=hinted_minus_empty, layers=16-27, strength=1.0
# (non-deterministic; each run gets unique results dir under results/)

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

INPUT="${1:-/home/ubuntu/steering/baseline_eval_20260121_022109_wrong_llm_judged_wrong_with_hints.json}"
PARENT_DIR="./results/run_10replicates_hinted_empty_16-27_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PARENT_DIR"

echo "Input: $INPUT"
echo "Output parent: $PARENT_DIR"
echo "Running 10 replicates (hinted_minus_empty, layers 16-27, strength 1.0)..."
echo ""

for i in $(seq 1 10); do
  RUN_DIR="$PARENT_DIR/run_$i"
  mkdir -p "$RUN_DIR"
  echo "[$i/10] Running replicate $i -> $RUN_DIR"
  python src/sweep_steering_experiments.py \
    --input "$INPUT" \
    --output-dir "$RUN_DIR" \
    --layers-sets "16-27" \
    --strengths "1.0" \
    --baseline-filter "hinted_minus_empty" \
    --max-new-tokens 4096 \
    --progress-every 1
  echo "  Done. sweep_results.json in $RUN_DIR"
  echo ""
done

echo "Aggregating 10 replicates..."
python - "$PARENT_DIR" "$INPUT" << 'PYEOF'
import json
import sys
from pathlib import Path

parent = Path(sys.argv[1])
input_path = sys.argv[2]
accs_vanilla, accs_steered, improved_list = [], [], []
for i in range(1, 11):
    p = parent / f'run_{i}' / 'sweep_results.json'
    if not p.exists():
        continue
    data = json.loads(p.read_text())
    for r in data.get('results', []):
        if r.get('baseline') == 'hinted_minus_empty' and r.get('strength') == 1.0:
            accs_vanilla.append(r['vanilla_accuracy'])
            accs_steered.append(r['steered_accuracy'])
            improved_list.append(r['improved'])
            break

n = len(accs_vanilla)
if n == 0:
    print('No results to aggregate')
    sys.exit(1)

def mean(x): return sum(x) / n
def std(x):
    m = mean(x)
    return (sum((v - m) ** 2 for v in x) / n) ** 0.5

summary = {
    'config': {
        'input': input_path,
        'baseline': 'hinted_minus_empty',
        'layers': list(range(16, 28)),
        'strength': 1.0,
        'num_replicates': n,
    },
    'vanilla_accuracy': {'mean': mean(accs_vanilla), 'std': std(accs_vanilla), 'values': accs_vanilla},
    'steered_accuracy': {'mean': mean(accs_steered), 'std': std(accs_steered), 'values': accs_steered},
    'improved': {'mean': mean(improved_list), 'std': std(improved_list), 'values': improved_list},
    'steered_better_count': sum(1 for v, s in zip(accs_vanilla, accs_steered) if s > v),
}
out = parent / 'replicate_summary.json'
out.write_text(json.dumps(summary, indent=2))
print(f'Wrote {out}')
print(f'Vanilla accuracy: {mean(accs_vanilla):.4f} ± {std(accs_vanilla):.4f}')
print(f'Steered accuracy: {mean(accs_steered):.4f} ± {std(accs_steered):.4f}')
print(f'Steered better than vanilla in {summary["steered_better_count"]}/{n} replicates')
PYEOF

echo ""
echo "All 10 replicates done. Results in: $PARENT_DIR"
echo "Summary: $PARENT_DIR/replicate_summary.json"
