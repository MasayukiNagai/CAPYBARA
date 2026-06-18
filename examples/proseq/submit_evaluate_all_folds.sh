#!/bin/bash
set -euo pipefail

# Submit one PRO-seq evaluation job per fold (no SLURM array).
# Loops the fold list and sbatch's run_evaluate_proseq.sh once per fold for a single
# model/timestamp/split.
#
# Because training is stage=both (base + fine-tuned) and we evaluate two splits, full
# coverage is a few invocations -- evaluate the base run and the fine-tuned (_ft) run on
# both the positive (test) and negative (neg_test) splits:
#   submit_evaluate_all_folds.sh capy <ts>     "" test
#   submit_evaluate_all_folds.sh capy <ts>     "" neg_test
#   submit_evaluate_all_folds.sh capy <ts>_ft  "" test
#   submit_evaluate_all_folds.sh capy <ts>_ft  "" neg_test
#
# Usage:
#   examples/proseq/submit_evaluate_all_folds.sh <model_name> <timestamp> [treatment] [split] [gpu] [reverse_complement]
#   model_name: capy | procapnet ; split: test | neg_test | val | ... ; reverse_complement: 1 or 0.
# Override the fold list with FOLDS, e.g. FOLDS="fold1 fold3" ...
# PROSEQ_PROJ_DIR (read by run_evaluate_proseq.sh) can override the model/output root.

REPO_ROOT="/grid/koo/home/ykang/elongation/CAPYBARA"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model_name> <timestamp> [treatment] [split] [gpu] [reverse_complement]" >&2
  echo "Example: $0 capy 260605_120000 GSM8306530_0m test 0 0" >&2
  echo "Set FOLDS='fold1 fold3' to override the default fold list." >&2
  exit 1
fi

model_name="$1"
timestamp="$2"
treatment="${3:-GSM8306530_0m}"
split="${4:-test}"
gpu="${5:-}"
reverse_complement="${6:-0}"
folds="${FOLDS:-fold1 fold2 fold3 fold4 fold5 fold6 fold7}"

mkdir -p "${REPO_ROOT}/out"

for fold in $folds; do
  sbatch --job-name "eval_${model_name}_${treatment}_${fold}" \
    "$REPO_ROOT/examples/proseq/run_evaluate_proseq.sh" "$model_name" "$timestamp" "$treatment" "$fold" "$split" "$gpu" "$reverse_complement"
  # sleep 1
done
