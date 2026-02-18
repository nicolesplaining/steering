#!/bin/bash
# Run all positive-delta combinations with nohup

INPUT="results/run_20260125_065836/wrong_subset_with_hints.json"
OUTPUT_DIR="./results/run_positive_deltas_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting positive delta experiments at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo ""

# hinted_minus_empty | 12-23 | strengths [0.5, 0.8]
echo "[1/6] Running hinted_minus_empty | 12-23 | strengths [0.5, 0.8]"
nohup python src/sweep_steering_experiments.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR/run_1_hinted_minus_empty_12-23" \
  --layers-sets "12-23" \
  --strengths "0.5,0.8" \
  --baseline-filter "hinted_minus_empty" \
  > "$OUTPUT_DIR/run_1.log" 2>&1 &
PID1=$!
echo "  Started with PID $PID1"

# hinted_minus_empty | 14-27 | strengths [0.3, 0.5, 0.8, 1.2]
echo "[2/6] Running hinted_minus_empty | 14-27 | strengths [0.3, 0.5, 0.8, 1.2]"
nohup python src/sweep_steering_experiments.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR/run_2_hinted_minus_empty_14-27" \
  --layers-sets "14-27" \
  --strengths "0.3,0.5,0.8,1.2" \
  --baseline-filter "hinted_minus_empty" \
  > "$OUTPUT_DIR/run_2.log" 2>&1 &
PID2=$!
echo "  Started with PID $PID2"

# hinted_minus_empty | 16-27 | strengths [0.8, 1.0]
echo "[3/6] Running hinted_minus_empty | 16-27 | strengths [0.8, 1.0]"
nohup python src/sweep_steering_experiments.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR/run_3_hinted_minus_empty_16-27" \
  --layers-sets "16-27" \
  --strengths "0.8,1.0" \
  --baseline-filter "hinted_minus_empty" \
  > "$OUTPUT_DIR/run_3.log" 2>&1 &
PID3=$!
echo "  Started with PID $PID3"

# hinted_minus_unhinted | 12-23 | strengths [0.3, 1.0, 1.2, 1.5]
echo "[4/6] Running hinted_minus_unhinted | 12-23 | strengths [0.3, 1.0, 1.2, 1.5]"
nohup python src/sweep_steering_experiments.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR/run_4_hinted_minus_unhinted_12-23" \
  --layers-sets "12-23" \
  --strengths "0.3,1.0,1.2,1.5" \
  --baseline-filter "hinted_minus_unhinted" \
  > "$OUTPUT_DIR/run_4.log" 2>&1 &
PID4=$!
echo "  Started with PID $PID4"

# hinted_minus_unhinted | 14-27 | strengths [0.3, 0.5, 1.2]
echo "[5/6] Running hinted_minus_unhinted | 14-27 | strengths [0.3, 0.5, 1.2]"
nohup python src/sweep_steering_experiments.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR/run_5_hinted_minus_unhinted_14-27" \
  --layers-sets "14-27" \
  --strengths "0.3,0.5,1.2" \
  --baseline-filter "hinted_minus_unhinted" \
  > "$OUTPUT_DIR/run_5.log" 2>&1 &
PID5=$!
echo "  Started with PID $PID5"

# hinted_minus_unhinted | 16-27 | strengths [0.3, 1.0]
echo "[6/6] Running hinted_minus_unhinted | 16-27 | strengths [0.3, 1.0]"
nohup python src/sweep_steering_experiments.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR/run_6_hinted_minus_unhinted_16-27" \
  --layers-sets "16-27" \
  --strengths "0.3,1.0" \
  --baseline-filter "hinted_minus_unhinted" \
  > "$OUTPUT_DIR/run_6.log" 2>&1 &
PID6=$!
echo "  Started with PID $PID6"

echo ""
echo "All 6 experiments started in background"
echo "PIDs: $PID1 $PID2 $PID3 $PID4 $PID5 $PID6"
echo ""
echo "Monitor progress with:"
echo "  tail -f $OUTPUT_DIR/run_*.log"
echo ""
echo "Check if still running with:"
echo "  ps aux | grep -E '($PID1|$PID2|$PID3|$PID4|$PID5|$PID6)'"
echo ""
echo "Results will be in: $OUTPUT_DIR"
