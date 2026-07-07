#!/bin/bash
#SBATCH --job-name=train_bias_capy
#SBATCH --output=out/%x_%j.log
#SBATCH --error=out/%x_%j.log
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=128G
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#
# Stage-1 CAPY bias model: train on ChromBPNet non-peaks, then Tier-A evaluate.
# Usage: sbatch submit_bias_fold.sh [timestamp] [cell_type] [fold] [gpu]
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v job_notify_slurm >/dev/null 2>&1; then
  source "$(command -v job_notify_slurm)"
  notify_job_start || true
fi

# SLURM copies this script to a spool dir, so BASH_SOURCE can't locate the repo.
# Default to the known checkout; override with REPO_ROOT=... sbatch ...
REPO_ROOT="${REPO_ROOT:-/grid/koo/home/ykang/elongation/CAPYBARA}"

timestamp="${1:-$(date +%y%m%d_%H%M%S)}"
cell_type="${2:-K562}"
fold="${3:-0}"
gpu="${4:-0}"

proj_dir="${ATAC_PROJ_DIR:-/grid/koo/home/ykang/elongation/CAPYBARA/results/runs/models/chrombpnet_benchmark}"
params="${BIAS_PARAMS:-${REPO_ROOT}/configs/atac_capy_bias.yaml}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"

if [[ -n "$gpu" ]]; then
  export CUDA_VISIBLE_DEVICES="$gpu"
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "proj_dir=${proj_dir}  cell_type=${cell_type}  fold=${fold}  timestamp=${timestamp}"

"$PYTHON" "${REPO_ROOT}/examples/atac/bias_capy/train_bias.py" \
  --proj_dir "$proj_dir" \
  --params "$params" \
  --cell_type "$cell_type" \
  --fold "$fold" \
  --timestamp "$timestamp"

"$PYTHON" "${REPO_ROOT}/examples/atac/bias_capy/evaluate_bias.py" \
  --proj_dir "$proj_dir" \
  --cell_type "$cell_type" \
  --fold "$fold" \
  --timestamp "$timestamp" \
  --split test
