#!/usr/bin/env bash
# prepare_data.sh
# One-time preprocessing pipeline for a single ENCODE ATAC-seq cell line.
#
# Usage:
#   bash prepare_data.sh                        # all 5 folds
#   FOLD=1 bash prepare_data.sh                 # single fold
#   PROJ_DIR=/path CELL_TYPE=K562 bash prepare_data.sh
#
# Requirements:
#   Python packages: pysam, pybigtools  (pip install "capybara[atac]")
#   download_data.sh must be run first to populate genome/ and data/atac/raw/.

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

# Use the repo's .venv if present (bash subshell doesn't inherit activated venv PATH)
if [ -x "$REPO_ROOT/.venv/bin/python3" ]; then
    PYTHON=${PYTHON:-$REPO_ROOT/.venv/bin/python3}
else
    PYTHON=${PYTHON:-python3}
fi

# ── Configure ──────────────────────────────────────────────────────────────────
PROJ_DIR=${PROJ_DIR:-/path/to/proj}
CELL_TYPE=${CELL_TYPE:-K562}
FOLD=${FOLD:-all}       # a single fold number (1–5), or "all" to process all folds
THREADS=${THREADS:-4}   # threads for BAM decompression in prep_bam.py

# Paths created by download_data.sh (change only if you placed files elsewhere)
INPUT_BAM=$PROJ_DIR/data/atac/raw/$CELL_TYPE/$CELL_TYPE.bam
RAW_PEAKS=$PROJ_DIR/data/atac/raw/$CELL_TYPE/peaks.narrowPeak.gz
BLACKLIST=$PROJ_DIR/genome/hg38.blacklist.bed.gz
CHROM_SIZES=$PROJ_DIR/genome/hg38.withrDNA.chrom.sizes
# ───────────────────────────────────────────────────────────────────────────────

OUT_DIR=$PROJ_DIR/data/atac/processed/$CELL_TYPE
SPLITS_DIR=$PROJ_DIR/data/atac/splits
mkdir -p "$OUT_DIR" "$SPLITS_DIR"

# ── Step 1: Tn5-shifted BigWig + pseudoreplicates (fold-independent, run once) ─

if [ -f "$OUT_DIR/atac.bw" ] && [ -f "$OUT_DIR/rep1.bw" ] && [ -f "$OUT_DIR/rep2.bw" ]; then
    echo "=== Step 1: BigWigs already present; skipping ==="
else
    echo "=== Step 1: Tn5-shifted BigWig + pseudoreplicates ==="
    # Tn5 shift: +4 bp (forward strand), -4 bp (reverse strand)
    $PYTHON "$SCRIPT_DIR/prep_bam.py" \
      --bam         "$INPUT_BAM" \
      --chrom_sizes "$CHROM_SIZES" \
      --out_dir     "$OUT_DIR" \
      --split \
      --threads     "$THREADS"
    mv "$OUT_DIR/unstranded.bw"      "$OUT_DIR/atac.bw"
    mv "$OUT_DIR/unstranded.rep1.bw" "$OUT_DIR/rep1.bw"
    mv "$OUT_DIR/unstranded.rep2.bw" "$OUT_DIR/rep2.bw"
fi

# ── Step 2: Chromosome splits + blacklist-filtered peak BEDs (per fold) ────────

echo "=== Step 2: Chromosome splits + blacklist-filtered peak BEDs ==="

FOLD_ARG=""
[ "$FOLD" != "all" ] && FOLD_ARG="--fold $FOLD"

$PYTHON "$SCRIPT_DIR/make_splits.py" \
  --chrom_sizes "$CHROM_SIZES" \
  --peaks       "$RAW_PEAKS" \
  --blacklist   "$BLACKLIST" \
  --splits_dir  "$SPLITS_DIR" \
  --peaks_dir   "$OUT_DIR" \
  $FOLD_ARG

echo "=== Done ==="
echo "Outputs in: $OUT_DIR"
echo "Splits in:  $SPLITS_DIR"
