#!/usr/bin/env bash
# Run TF-MoDISco (tfmodisco-lite) on a CAPY contribution-score .h5 using the
# ChromBPNet container UNCHANGED. MoDISco is model-agnostic: it reads only the
# .h5 (raw/shap/projected_shap), so the exact same tool ChromBPNet uses runs on
# CAPY scores, giving an apples-to-apples motif comparison.
#
# Mirrors chrombpnet's pipeline: `modisco motifs -n 50000 -w 500` then
# `modisco report -m <bundled motifs.meme.txt>` (TOMTOM annotation + logos).
#
# Usage:
#   run_modisco.sh <scores.h5> <output_dir> [max_seqlets] [window] [n_matches]
#
# Env overrides: CHROMBPNET_SIF, MEME_DB (path inside the container).
set -euo pipefail

H5="${1:?Usage: run_modisco.sh <scores.h5> <output_dir> [max_seqlets] [window] [n_matches]}"
OUT_DIR="${2:?output_dir required}"
MAX_SEQLETS="${3:-50000}"
WINDOW="${4:-500}"
N_MATCHES="${5:-3}"

CHROMBPNET_SIF="${CHROMBPNET_SIF:-/grid/koo/home/shared/capybara/chrombpnet/chrombpnet_latest.sif}"
# Bundled MEME DB lives inside the image; resolvable without a bind.
MEME_DB="${MEME_DB:-/scratch/chrombpnet/chrombpnet/data/motifs.meme.txt}"

command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found" >&2; exit 1; }
[[ -f "${H5}" ]] || { echo "ERROR: scores h5 not found: ${H5}" >&2; exit 1; }
[[ -f "${CHROMBPNET_SIF}" ]] || { echo "ERROR: container not found: ${CHROMBPNET_SIF}" >&2; exit 1; }

H5="$(readlink -f "${H5}")"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(readlink -f "${OUT_DIR}")"
MODISCO_H5="${OUT_DIR}/modisco_results.h5"
REPORTS_DIR="${OUT_DIR}/reports"
mkdir -p "${REPORTS_DIR}"

BINDS=(--bind "$(dirname "${H5}"):$(dirname "${H5}")" --bind "${OUT_DIR}:${OUT_DIR}")

echo "== modisco motifs (max_seqlets=${MAX_SEQLETS}, window=${WINDOW}) =="
apptainer exec --nv "${BINDS[@]}" "${CHROMBPNET_SIF}" \
  modisco motifs -i "${H5}" -n "${MAX_SEQLETS}" -w "${WINDOW}" -o "${MODISCO_H5}" -v

echo "== modisco report (TOMTOM vs bundled MEME DB) =="
apptainer exec "${BINDS[@]}" "${CHROMBPNET_SIF}" \
  modisco report -i "${MODISCO_H5}" -o "${REPORTS_DIR}" -m "${MEME_DB}" -n "${N_MATCHES}"

echo "Done."
echo "  motifs h5 : ${MODISCO_H5}"
echo "  report    : ${REPORTS_DIR}/motifs.html (+ per-motif logos)"
