#!/bin/bash
#SBATCH --job-name=train_procapnet_proseq
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

# Train ProCapNet on PRO-seq / Rogers-timecourse gene-body data (single strand).
#
# Usage:
#   sbatch examples/proseq/run_train_proseq_procapnet.sh [timestamp] [params] [treatment] [fold] [gpu] [stage]
# [fold] is optional; if empty the fold is taken from the config's dataset.fold.
# Defaults shown below; PROSEQ_PROJ_DIR can override the output root.

set -euo pipefail

REPO_ROOT="/grid/koo/home/ykang/elongation/CAPYBARA"
PYTHON="${REPO_ROOT}/.venv/bin/python"
script="${REPO_ROOT}/examples/proseq/train_proseq_procapnet.py"

timestamp="${1:-}"
params="${2:-${REPO_ROOT}/configs/proseq_procapnet.yaml}"
treatment="${3:-GSM8306530_0m}"
fold="${4:-}"   # empty -> use dataset.fold from the config
gpu="${5:-0}"
stage="${6:-both}"   # train, finetune, both

# Output root for trained models (models/ is created under here).
proj_dir="${PROSEQ_PROJ_DIR:-${REPO_ROOT}/results/runs}"

mkdir -p "${REPO_ROOT}/out"

cmd=("$PYTHON"
  "$script"
  --proj_dir "$proj_dir"
  --params "$params"
  --treatment "$treatment"
  --stage "$stage"
  --device gpu)

if [[ -n "$fold" ]]; then
  cmd+=(--fold "$fold")
fi
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
