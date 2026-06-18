#!/bin/bash
#SBATCH --job-name=eval_proseq
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
#SBATCH --time=2:00:00

# Evaluate a trained CAPY or ProCapNet PRO-seq model on a split, saving metrics
# and predictions (consumed by examples/proseq/benchmark_proseq.ipynb).
#
# Usage:
#   sbatch examples/proseq/run_evaluate_proseq.sh <model_name> <timestamp> \
#       [treatment] [fold] [split] [gpu] [reverse_complement]
# model_name: capy | procapnet ; reverse_complement: 1 (on) or 0 (off).
# split: train|val|test (positives) or neg_train|neg_val|neg_test (negatives).
# PROSEQ_PROJ_DIR can override the output/model root.

set -euo pipefail

REPO_ROOT="/grid/koo/home/ykang/elongation/CAPYBARA"
PYTHON="${REPO_ROOT}/.venv/bin/python"
script="${REPO_ROOT}/examples/proseq/evaluate.py"

proj_dir="${PROSEQ_PROJ_DIR:-${REPO_ROOT}/results/runs}"
model_name="${1:-procapnet}"
timestamp="${2:?timestamp is required}"
treatment="${3:-GSM8306530_0m}"
fold="${4:-fold1}"
split="${5:-test}"
gpu="${6:-0}"
reverse_complement="${7:-0}"

mkdir -p "${REPO_ROOT}/out"

cmd=("$PYTHON"
  "$script"
  --proj_dir "$proj_dir"
  --model_name "$model_name"
  --treatment "$treatment"
  --fold "$fold"
  --timestamp "$timestamp"
  --split "$split"
  --save_predictions)

case "$reverse_complement" in
  1)
    cmd+=(--reverse_complement)
    ;;
  0)
    ;;
  *)
    echo "Invalid reverse_complement value: $reverse_complement" >&2
    echo "Use 1 to enable RC or 0 to disable it." >&2
    exit 1
    ;;
esac

if [[ -n "$gpu" ]]; then
  export CUDA_VISIBLE_DEVICES="$gpu"
fi

echo "Running command:"
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
