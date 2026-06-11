# ATAC-seq Benchmark

This folder contains training, evaluation, and benchmarking code for CAPY on ENCODE ATAC-seq data, replicating the core ChromBPNet profile-prediction benchmark (Fig 1c–1d of Pampari et al., 2024).

The workflow generalises across all five ENCODE Tier-1 cell lines (K562, HepG2, GM12878, IMR90, H1-hESC) by changing a single `--cell_type` argument.

---

## Directory layout

After running all steps the project directory looks like this:

```
proj_dir/
  genome/
    hg38.withrDNA.fasta          # hg38 + rDNA (created by download_data.sh)
    hg38.withrDNA.chrom.sizes
    hg38.blacklist.bed.gz        # ENCODE blacklist ENCFF356LFX
  data/atac/
    raw/{cell_type}/
      {cell_type}.bam            # merged replicate BAM (created by download_data.sh)
      {cell_type}.bam.bai
    processed/{cell_type}/
      atac.bw                    # Tn5-shifted unstranded BigWig
      rep1.bw  rep2.bw           # pseudoreplicate BigWigs (for JSD upper bound)
      peaks.bed.gz               # IDR peaks
      peaks_fold{0-4}_{train,valid,test,train_and_valid}.bed.gz
    splits/
      fold_{0-4}.json             # chromosome split definitions
  models/capy/atac/{cell_type}/fold{fold}/{timestamp}/
    checkpoints/best.pt  last.pt
    params_saved.yaml
    evals/
      {cell_type}_metrics_summary_test.csv
      {cell_type}_metrics_profile_test.csv
      {cell_type}_log_{pred,true}_counts_test.npy
      fig1c_{cell_type}_fold{fold}.pdf
      fig1d_{cell_type}_fold{fold}.pdf
  models/chrombpnet/atac/{cell_type}/fold{fold}/{timestamp}/
    params_saved.yaml              # benchmark metadata for pretrained original ChromBPNet
    config_saved.yaml
    evals/                         # same file contract as CAPY evals
```

---

## Step 1 — Download data

```bash
bash examples/atac/download_data.sh K562
```

This downloads shared reference files (genome, blacklist) once, and per-cell-type replicate BAMs which are merged into a single `{cell_type}.bam`. Re-running is safe; already-present files are skipped.

**Requirements:** `wget`, `samtools` (for merging replicate BAMs)

**ENCODE experiments used** (ChromBPNet paper Table 1):

| Cell type | Experiment | Replicates |
|-----------|------------|------------|
| K562      | ENCSR868FGK | 3 |
| HepG2     | ENCSR042AWH | 2 |
| GM12878   | ENCSR637XSC | 3 |
| IMR90     | ENCSR200OML | 2 |
| H1-hESC   | GEO GSE267154 | 2 (see note below) |

> **H1-hESC** was generated in-house by the Kundaje lab and is not on the ENCODE portal. Run `bash download_data.sh H1-hESC` for instructions on downloading from SRA (GSM8260976, GSM8260977).

---

## Step 2 — Prepare data

```bash
bash examples/atac/prepare_data.sh
```

Edit the variables at the top of `prepare_data.sh` (set `PROJ_DIR`, `CELL_TYPE`, `FOLD`). The default `INPUT_BAM` and `BLACKLIST` paths point to where `download_data.sh` places files.

**Requirements:** `pip install "capybara[atac]"` (installs pysam, pybigtools, pyfaidx, etc.)

This runs two steps using scripts in this directory:

1. `prep_bam.py` — Tn5-shifted BigWigs (+4/−4 bp, ChromBPNet convention):
   - `atac.bw` — merged-replicate coverage (for training)
   - `rep1.bw`, `rep2.bw` — pseudoreplicate coverage (for JSD upper bound)
2. `make_splits.py` — blacklist-filters the ENCODE IDR peaks, then creates:
   - `peaks.bed.gz` — all-chromosome filtered peaks
   - `splits/fold_{N}.json` — chromosome assignments
   - `peaks_fold{N}_{train,valid,test,train_and_valid}.bed.gz` — fold-specific peak BEDs

Repeat with `FOLD=0` through `FOLD=4` to prepare all folds. Step 1 runs once per cell type; step 2 is fast and fold-specific.

**Chromosome splits** (official ChromBPNet zero-based folds):

| Fold | Test chromosomes | Validation |
|------|-----------------|------------|
| 0 | chr1, chr3, chr6 | chr8, chr20 |
| 1 | chr2, chr8, chr9, chr16 | chr12, chr17 |
| 2 | chr4, chr11, chr12, chr15, chrY | chr7, chr22 |
| 3 | chr5, chr10, chr14, chr18, chr20, chr22 | chr6, chr21 |
| 4 | chr7, chr13, chr17, chr19, chr21, chrX | chr10, chr18 |

If you already generated local ATAC folds before this zero-based ChromBPNet convention, do not rename those BEDs. Back them up and regenerate official folds from each cell type's `peaks.bed.gz`:

```bash
PROJ_DIR=/path/to/proj
for CELL_TYPE in K562 HepG2 GM12878 IMR90 H1-hESC; do
  PEAKS_DIR="$PROJ_DIR/data/atac/processed/$CELL_TYPE"
  test -f "$PEAKS_DIR/peaks.bed.gz" || continue

  mkdir -p "$PEAKS_DIR/legacy_pre_chrombpnet_folds" "$PROJ_DIR/data/atac/splits/legacy_pre_chrombpnet_folds"
  mv "$PEAKS_DIR"/peaks_fold*_*.bed.gz "$PEAKS_DIR/legacy_pre_chrombpnet_folds/" 2>/dev/null || true
  mv "$PROJ_DIR"/data/atac/splits/fold*.json "$PROJ_DIR/data/atac/splits/legacy_pre_chrombpnet_folds/" 2>/dev/null || true

  .venv/bin/python examples/atac/make_splits.py \
    --chrom_sizes "$PROJ_DIR/genome/hg38.withrDNA.chrom.sizes" \
    --peaks "$PEAKS_DIR/peaks.bed.gz" \
    --splits_dir "$PROJ_DIR/data/atac/splits" \
    --peaks_dir "$PEAKS_DIR"
done
```

Existing CAPY checkpoints trained on the previous local folds should not be relabeled as official ChromBPNet folds; retrain them for exact fold-matched comparisons.

---

## Step 3 — Train

```bash
python examples/atac/train_capy.py \
    --proj_dir /path/to/proj \
    --cell_type K562 \
    --fold 0
```

Key arguments:

- `--proj_dir`: project directory root (same across steps)
- `--cell_type`: one of `K562`, `HepG2`, `GM12878`, `IMR90`, `H1-hESC`
- `--fold`: 0-4
- `--params`: YAML config file (default: `configs/atac_default.yaml`)
- `--device`: `gpu`, `cpu`, `auto`, or a PyTorch device string
- `--no_wandb`: disable Weights & Biases logging

Training outputs are written under:
```
models/capy/atac/{cell_type}/fold{fold}/{timestamp}/
```

---

## Step 4 — Evaluate

```bash
python examples/atac/evaluate.py \
    --proj_dir /path/to/proj \
    --cell_type K562 \
    --fold 0 \
    --timestamp 250101_120000 \
    --split test \
    --save_predictions
```

If `rep1.bw`/`rep2.bw` are present, the script automatically computes the pseudoreplicate JSD upper bound needed for Fig 1d. If they are absent it skips that step gracefully.

Add `--reverse_complement` for test-time augmentation (averages forward and RC predictions).

Outputs are written to `models/capy/atac/{cell_type}/fold{fold}/{timestamp}/evals/`.

---

## Step 5 — Evaluate pretrained original ChromBPNet

Original ChromBPNet model archives use zero-based fold directories (`fold_0`-`fold_4`), and this repo now uses the same zero-based local folds (`fold0`-`fold4`). By default, `--fold 0` reads `fold_0` from the tarball; `--chrombpnet_fold` remains available only as an override for unusual archives.

```bash
python examples/atac/evaluate_chrombpnet.py \
    --proj_dir /path/to/proj \
    --cell_type GM12878 \
    --fold 0 \
    --timestamp original_fold0 \
    --weights_tar /path/to/ENCFF142IOR.tar.gz \
    --model_accession ENCSR637XSC \
    --split test \
    --save_predictions
```

This loads `model.bias_scaled.fold_{N}.*.h5` and
`model.chrombpnet_nobias.fold_{N}.*.h5` directly from the tarball with
`bpnet-lite`, evaluates on the same local truth BigWigs and peak split, and writes
outputs under:

```
models/chrombpnet/atac/{cell_type}/fold{fold}/{timestamp}/evals/
```

ChromBPNet uses its original `2114` bp input window and `1000` bp output window.
The saved output profile/count arrays have the same shapes and filenames as CAPY
evaluation outputs, so the benchmark notebook can compare models by adding a
`model_specs` entry with `model_name: "chrombpnet"`.

---

## Step 6 — Plot

```bash
python examples/atac/plot_fig1cd.py \
    --eval_dir /path/to/proj/models/capy/atac/K562/fold0/{timestamp}/evals \
    --cell_type K562 \
    --fold 0
```

Produces:

- **`fig1c_{cell_type}_fold{fold}.pdf`** — density scatter of predicted vs. observed log total counts (Pearson r in title)
- **`fig1d_{cell_type}_fold{fold}.pdf`** — JSD distribution histograms with pseudoreplicate upper bound

In the upstream `chrombpnet-figures` repository, Fig 1c is generated from one
K562 `fold_0` run and the `test` chromosomes from `fold_0.json`; it is not
aggregated across all five folds. The local benchmark notebook therefore uses
`fig1c_folds = [0]` for the exact Fig 1c-style count scatter and
`folds = [0, 1, 2, 3, 4]` for aggregate views.

## Baseline to beat (ChromBPNet, K562 ATAC-seq)

| Metric | ChromBPNet | Target |
|--------|-----------|--------|
| Pearson r (log total counts) | 0.70 ± 0.02 | ≥ 0.70 |
| JSD (predicted vs. observed) | approaches pseudorep bound | < pseudorep bound |
| Pseudorep JSD upper bound | — | reference ceiling |

Results across all five cell lines: median Pearson r = 0.69 (ChromBPNet paper Extended Fig 2b).
