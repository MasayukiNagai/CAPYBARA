#!/bin/bash
#SBATCH --job-name=attri_nobias_capy
#SBATCH --output=out/%x_%j.log
#SBATCH --error=out/%x_%j.log
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=128G
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --time=48:00:00
#
# Tier-B for the CAPY accessibility (nobias) model: contribution scores -> TF-MoDISco.
# Recovers TF motifs to compare vs ChromBPNet.
# Usage: sbatch submit_attribution_nobias.sh [timestamp] [method] [cell_type] [fold] [bias_timestamp] [gpu] [--heads "<heads>"]
#   method: gradientshap (default, recommended for the attention model) | deeplift
#   --heads: profile (default) | counts | "profile counts"  (modisco writes to modisco/<head>)
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v job_notify_slurm >/dev/null 2>&1; then
  source "$(command -v job_notify_slurm)"
  notify_job_start || true
fi

REPO_ROOT="${REPO_ROOT:-/grid/koo/home/ykang/elongation/CAPYBARA}"

# Pull an optional --heads flag out of the args (default profile); rest stay positional.
heads="profile"
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --heads) heads="$2"; shift 2 ;;
    --heads=*) heads="${1#*=}"; shift ;;
    *) positional+=("$1"); shift ;;
  esac
done
set -- "${positional[@]}"

timestamp="${1:?timestamp (Stage-2 run) required}"
method="${2:-gradientshap}"
cell_type="${3:-K562}"
fold="${4:-0}"
bias_timestamp="${5:-chead}"
gpu="${6:-0}"

proj_dir="${ATAC_PROJ_DIR:-/grid/koo/home/ykang/elongation/CAPYBARA/results/runs/models/chrombpnet_benchmark}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python"
[[ -n "$gpu" ]] && export CUDA_VISIBLE_DEVICES="$gpu"

DIR="${REPO_ROOT}/examples/atac/attribution"
attr_dir="${proj_dir}/capy_chrombpnet/atac/${cell_type}/fold${fold}/${timestamp}/attribution/${method}"

echo "REPO_ROOT=${REPO_ROOT} proj_dir=${proj_dir} cell=${cell_type} fold=${fold} ts=${timestamp} bias=${bias_timestamp} method=${method} heads=${heads}"

"$PYTHON" "${DIR}/attribution_nobias.py" \
  --proj_dir "$proj_dir" \
  --cell_type "$cell_type" \
  --fold "$fold" \
  --timestamp "$timestamp" \
  --bias_timestamp "$bias_timestamp" \
  --method "$method" \
  --heads $heads

for head in $heads; do
  "${DIR}/run_modisco.sh" \
    "${attr_dir}/${cell_type}.fold_${fold}.${head}_scores.h5" \
    "${attr_dir}/modisco/${head}"
done
