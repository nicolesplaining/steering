#!/usr/bin/env bash
#SBATCH --job-name=baseline_qwen3_4b
#SBATCH --partition=matx
#SBATCH --account=matx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --exclude=matx-amd-1
#SBATCH --output=/matx/u/singhh/steering/logs/baseline_qwen3_4b_%j.out
#SBATCH --error=/matx/u/singhh/steering/logs/baseline_qwen3_4b_%j.err

set -euo pipefail

cd /matx/u/singhh/steering

source /matx/u/singhh/venvs/conf/bin/activate
export HF_HOME=/matx/u/singhh/huggingface
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

mkdir -p /matx/u/singhh/steering/logs
mkdir -p /matx/u/singhh/steering/outputs

MODEL="Qwen/Qwen3-4B"
MODEL_TAG=$(echo "${MODEL}" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
OUTPUT="/matx/u/singhh/steering/outputs/baseline_eval_${MODEL_TAG}_${SLURM_JOB_ID}.json"

python -u baseline_eval.py \
  --model "${MODEL}" \
  --seed 42 \
  --n_samples 500 \
  --max_new_tokens 8192 \
  --output "${OUTPUT}"
