#!/usr/bin/env bash
# Setup script for a fresh GPU instance: install Miniconda and create steering-env.
# Run from the repo root: bash scripts/setup_gpu_instance.sh

set -e

MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)  INSTALLER="Miniconda3-latest-Linux-x86_64.sh" ;;
  aarch64) INSTALLER="Miniconda3-latest-Linux-aarch64.sh" ;;
  *)       echo "Unsupported arch: $ARCH"; exit 1 ;;
esac
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/environment.yml"

echo "==> Repo root: $REPO_ROOT"
cd "$REPO_ROOT"

# --- Install Miniconda if not present ---
if command -v conda &>/dev/null; then
  echo "==> conda already installed: $(conda --version)"
elif [[ -d "$MINICONDA_DIR" ]]; then
  echo "==> Miniconda already installed at $MINICONDA_DIR (not in PATH)"
else
  echo "==> Installing Miniconda into $MINICONDA_DIR ..."
  if [[ ! -f "$INSTALLER" ]]; then
    wget -q "https://repo.anaconda.com/miniconda/$INSTALLER" -O "$INSTALLER"
  fi
  bash "$INSTALLER" -b -p "$MINICONDA_DIR"
  rm -f "$INSTALLER"
  echo "==> Miniconda installed."
fi

# --- Ensure conda is available in this script ---
if [[ -z "$CONDA_EXE" ]]; then
  if [[ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]]; then
    set +e
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
    set -e
  else
    echo "==> Conda not found. If you just installed Miniconda, run: source ~/.bashrc && bash $0"
    exit 1
  fi
fi

# --- Create or update steering-env ---
if conda env list | grep -q '^steering-env '; then
  echo "==> Updating existing conda env steering-env ..."
  conda env update -f "$ENV_FILE" --prune
else
  echo "==> Creating conda env steering-env ..."
  conda env create -f "$ENV_FILE"
fi

echo ""
echo "==> Done. Activate the environment with:"
echo "    conda activate steering-env"
echo ""
echo "If conda is not in your PATH, run first:"
echo "    source $MINICONDA_DIR/etc/profile.d/conda.sh"
