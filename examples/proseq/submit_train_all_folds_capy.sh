#!/bin/bash
set -euo pipefail

# Submit one CAPY (Y-net or U-Net) PRO-seq training job per fold (no SLURM array).
# Loops the fold list and sbatch's run_train_proseq.sh once per fold; all folds share
# the given timestamp. Pass configs/proseq_unet.yaml as <params> to train the U-Net
# variant instead of the Y-net default.
#
# Usage:
#   examples/proseq/submit_train_all_folds_capy.sh [timestamp] [params] [treatment] [gpu] [stage]
# Override the fold list with FOLDS, e.g. FOLDS="fold1 fold3" ...
# PROSEQ_PROJ_DIR (read by run_train_proseq.sh) can override the output root.

REPO_ROOT="/grid/koo/home/ykang/elongation/CAPYBARA"

timestamp="${1:-$(date +%y%m%d_%H%M%S)}"
params="${2:-$REPO_ROOT/configs/proseq_default.yaml}"
treatment="${3:-GSM8306530_0m}"
gpu="${4:-}"
stage="${5:-both}"
folds="${FOLDS:-fold1 fold2 fold3 fold4 fold5 fold6 fold7}"

mkdir -p "${REPO_ROOT}/out"

for fold in $folds; do
  sbatch --job-name "capy_${treatment}_${fold}" \
    "$REPO_ROOT/examples/proseq/run_train_proseq.sh" "$timestamp" "$params" "$treatment" "$fold" "$gpu" "$stage"
  sleep 1  # Sleep for a short time to avoid overwhelming the scheduler
done
