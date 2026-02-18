#!/usr/bin/env bash
# Run context distillation on all metacognitive *_with_hints.json runs.
# Only files with config.mock_hints=false and config.use_solution_hint=false
# are used (--metacognitive-only). Duplicate problems across files are deduped.
#
# Each run uses a timestamped output dir so concurrent runs don't overwrite.
# For nohup, use a unique log file, e.g.:
#   nohup ./run_hint_distillation_expanded.sh > nohup_distill_expanded_$(date +%Y%m%d_%H%M%S).out 2>&1 &

set -euo pipefail
cd "$(dirname "$0")"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="results/run_distill_expanded_${TIMESTAMP}"
mkdir -p "$RUN_DIR"
echo "Output directory: $RUN_DIR"

INPUTS=(
  results/run_20260203_052646/baseline_eval_20260121_022109_wrong_llm_judged_wrong_with_hints.json
  results/run_20260203_064154/baseline_eval_20260121_022109_wrong_llm_judged_wrong_with_hints.json
  results/baseline_eval_20260121_022109_wrong_llm_judged_wrong_with_hints.json
  results/run_hints5_20260128_090143/baseline_eval_20260121_022109_wrong_llm_judged_wrong_with_hints.json
  results/run_20260128_055816/baseline_eval_20260121_022109_wrong_llm_judged_wrong_with_hints.json
  results/run_20260128_055816/baseline_eval_20260121_022109_wrong_llm_judged_wrong_with_hints_llm_judged.json
  results/run_20260125_070321/wrong_subset_with_hints.json
  results/run_20260125_065836/wrong_subset_with_hints.json
  results/run_20260125_062653/wrong_subset_with_hints.json
  baseline_eval_wrong_geometry_with_hints.json
  baseline_eval_wrong_general_with_hints.json
  baseline_eval_20260121_022109_wrong_llm_judged_wrong_with_hints.json
)

# Build --input args (skip missing files)
ARGS=()
for f in "${INPUTS[@]}"; do
  if [[ -f "$f" ]]; then
    ARGS+=( --input "$f" )
  else
    echo "Skipping missing: $f" >&2
  fi
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "No input files found." >&2
  exit 1
fi

python src/run_hint_distillation.py \
  "${ARGS[@]}" \
  --dataset-out "$RUN_DIR/math_hints_distill.json" \
  --adapter-path "$RUN_DIR/adapter" \
  "$@"

echo "Done. Dataset: $RUN_DIR/math_hints_distill.json  Adapter: $RUN_DIR/adapter"
