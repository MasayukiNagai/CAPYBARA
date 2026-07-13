#!/bin/bash
#SBATCH --job-name=attri_bias_capy
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
# Tier-B for the CAPY bias model: contribution scores (profile+counts) ->
# TF-MoDISco. Confirms the bias model learned only Tn5/repeat motifs.
# Usage: sbatch submit_attribution_bias.sh [timestamp] [method] [cell_type] [fold] [gpu] [--no_gradient_correction]
#   method: gradientshap (default) | deeplift
#   --no_gradient_correction: disable the Majdandzic simplex correction (gradientshap only;
#     default ON). Output/modisco namespaced under attribution/gradientshap_uncorrected/.
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v job_notify_slurm >/dev/null 2>&1; then
  source "$(command -v job_notify_slurm)"
  notify_job_start || true
fi

REPO_ROOT="${REPO_ROOT:-/grid/koo/home/ykang/elongation/CAPYBARA}"

# Pull the optional --no_gradient_correction flag out of the args; rest stay positional.
gc_flag=""
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no_gradient_correction) gc_flag="--no_gradient_correction"; shift ;;
    *) positional+=("$1"); shift ;;
  esac
done
set -- "${positional[@]}"

timestamp="${1:-chead}"
method="${2:-gradientshap}"
cell_type="${3:-K562}"
fold="${4:-0}"
gpu="${5:-0}"

proj_dir="${ATAC_PROJ_DIR:-/grid/koo/home/ykang/elongation/CAPYBARA/results/runs/models/chrombpnet_benchmark}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"
[[ -n "$gpu" ]] && export CUDA_VISIBLE_DEVICES="$gpu"

DIR="${REPO_ROOT}/examples/atac/attribution"
# Mirror attribution_bias.py's out_dir namespacing: an uncorrected gradientshap run
# writes to attribution/gradientshap_uncorrected/ so it never clobbers the corrected one.
method_subdir="${method}"
[[ "$method" == "gradientshap" && -n "$gc_flag" ]] && method_subdir="${method}_uncorrected"
attr_dir="${proj_dir}/capy_bias/atac/${cell_type}/fold${fold}/${timestamp}/attribution/${method_subdir}"

echo "REPO_ROOT=${REPO_ROOT} proj_dir=${proj_dir} cell=${cell_type} fold=${fold} ts=${timestamp} method=${method} gc_flag=${gc_flag:-on}"

"$PYTHON" "${DIR}/attribution_bias.py" \
  --proj_dir "$proj_dir" \
  --cell_type "$cell_type" \
  --fold "$fold" \
  --timestamp "$timestamp" \
  --method "$method" \
  --heads profile counts $gc_flag

for head in profile counts; do
  "${DIR}/run_modisco.sh" \
    "${attr_dir}/${cell_type}.fold_${fold}.${head}_scores.h5" \
    "${attr_dir}/modisco/${head}"
done
