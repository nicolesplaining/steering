#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
cd "$WORKDIR"

# ── 1. Clone repos ──
git clone https://github.com/jayelm/gisting.git || echo "gisting already cloned"
git clone https://github.com/nicolesplaining/steering.git || echo "steering already cloned"

# ── 2. Install gisting deps (inference-only, skip deepspeed/hydra/wandb) ──
# RunPod images ship with PyTorch pre-installed. We only need the pinned
# transformers + a few small packages.
pip install git+https://github.com/huggingface/transformers@fb366b9a
pip install accelerate==0.18.0 fire==0.5.0 sentencepiece==0.1.98
pip install openai datasets latex2sympy2

# ── 3. Reconstruct model from weight diff ──
# Downloads base LLaMA-7B (~27GB) + weight diff (~27GB), reconstructs,
# saves to disk. Only needs to run once.
if [ ! -d "$WORKDIR/gisting/llama-7b-gist-reconstructed" ]; then
    cd "$WORKDIR/gisting"
    python -m src.weight_diff recover \
        --path_raw huggyllama/llama-7b \
        --path_diff jayelm/llama-7b-gist-1 \
        --path_tuned ./llama-7b-gist-reconstructed
    echo "Model reconstructed at $WORKDIR/gisting/llama-7b-gist-reconstructed"
else
    echo "Model already reconstructed, skipping."
fi

# ── 4. Quick sanity check ──
cd "$WORKDIR/gisting"
python -m src.compress \
    --model_name_or_path ./llama-7b-gist-reconstructed \
    --instruction "Summarize the main idea of the text." \
    --input "The quick brown fox jumps over the lazy dog." \
    --precision fp16 \
    --max_new_tokens 64

echo "Setup complete."
