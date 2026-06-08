#!/bin/bash
#SBATCH --job-name=train_atac_capy
#SBATCH --output=out/%x_%j.log
#SBATCH --error=out/%x_%j.log
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=128G
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=24:00:00

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v job_notify_slurm >/dev/null 2>&1; then
  source "$(command -v job_notify_slurm)"
  notify_job_start || true
fi

REPO_ROOT="/grid/koo/home/nagai/projects/capybara"

timestamp="${1:-}"
proj_dir="${ATAC_PROJ_DIR:-/grid/koo/home/shared/capybara/chrombpnet}"
params="${2:-${REPO_ROOT}/configs/atac_default.yaml}"
cell_type="${3:-K562}"
fold="${4:-1}"
gpu="${5:-0}"

script="${REPO_ROOT}/examples/atac/train_capy.py"
PYTHON="${REPO_ROOT}/.venv/bin/python"

cmd=("$PYTHON"
  "$script"
  --proj_dir "$proj_dir"
  --params "$params"
  --cell_type "$cell_type"
  --fold "$fold"
)

if [[ -n "$timestamp" ]]; then
  cmd+=(--timestamp "$timestamp")
fi

if [[ -n "$gpu" ]]; then
  export CUDA_VISIBLE_DEVICES="$gpu"
fi

echo "Running command:"
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
