#!/bin/bash
#SBATCH --job-name=footprint_factorized_capy
#SBATCH --output=out/%x_%j.log
#SBATCH --error=out/%x_%j.log
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=128G
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=04:00:00
#
# Tier-C marginal footprinting of a trained composed CAPY run: footprints the
# bias-corrected nobias model (expect flat) and the frozen bias branch (positive
# control, expect a strong TN5 footprint). Usage:
#   sbatch submit_factorized_footprint.sh [timestamp] [cell_type] [fold] [bias_timestamp] [gpu]
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v job_notify_slurm >/dev/null 2>&1; then
  source "$(command -v job_notify_slurm)"
  notify_job_start || true
fi

REPO_ROOT="${REPO_ROOT:-/grid/koo/home/ykang/elongation/CAPYBARA}"

timestamp="${1:-cw50}"
cell_type="${2:-K562}"
fold="${3:-0}"
bias_timestamp="${4:-chead}"
gpu="${5:-0}"

proj_dir="${ATAC_PROJ_DIR:-/grid/koo/home/ykang/elongation/CAPYBARA/results/runs/models/chrombpnet_benchmark}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"

if [[ -n "$gpu" ]]; then
  export CUDA_VISIBLE_DEVICES="$gpu"
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "proj_dir=${proj_dir}  cell_type=${cell_type}  fold=${fold}  timestamp=${timestamp}  bias=${bias_timestamp}"

DIR="${REPO_ROOT}/examples/atac/bias_factorized_capy"

"$PYTHON" "${DIR}/marginal_footprint.py" \
  --proj_dir "$proj_dir" \
  --cell_type "$cell_type" \
  --fold "$fold" \
  --timestamp "$timestamp" \
  --bias_timestamp "$bias_timestamp" \
  --split test \
  --models nobias bias \
  --verbose
