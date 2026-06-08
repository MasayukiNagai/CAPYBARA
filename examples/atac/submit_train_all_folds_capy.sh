#!/bin/bash
set -euo pipefail

REPO_ROOT="/grid/koo/home/nagai/projects/capybara"

timestamp="${1:-$(date +%y%m%d_%H%M%S)}"
params="${2:-$REPO_ROOT/configs/atac_default.yaml}"
cell_type="${3:-K562}"
gpu="${4:-}"
folds="${FOLDS:-1 2 3 4 5}"

for fold in $folds; do
  sbatch --job-name "atac_capy_${cell_type}_f${fold}" \
    "$REPO_ROOT/examples/atac/run_train_capy.sh" "$timestamp" "$params" "$cell_type" "$fold"
  sleep 1
done
