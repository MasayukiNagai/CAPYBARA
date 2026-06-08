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
      peaks_fold{1-5}_{train,val,test,train_and_val}.bed.gz
    splits/
      fold{1-5}.json             # chromosome split definitions
  models/capy/atac/{cell_type}/fold{fold}/{timestamp}/
    checkpoints/best.pt  last.pt
    params_saved.yaml
    evals/
      {cell_type}_metrics_summary_test.csv
      {cell_type}_jsd_{pred,shuffled,mean,pseudorep}_test.npy
      {cell_type}_log_{pred,true}_counts_test.npy
      fig1c_{cell_type}_fold{fold}.pdf
      fig1d_{cell_type}_fold{fold}.pdf
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
   - `splits/fold{N}.json` — chromosome assignments
   - `peaks_fold{N}_{train,val,test,train_and_val}.bed.gz` — fold-specific peak BEDs

Repeat with `FOLD=1` through `FOLD=5` to prepare all folds. Step 1 runs once per cell type; step 2 is fast and fold-specific.

**Chromosome splits** (ChromBPNet paper Table 2):

| Fold | Test chromosomes | Validation |
|------|-----------------|------------|
| 1 | chr1, chr8 | chr10 |
| 2 | chr2, chr6 | chr22 |
| 3 | chr3, chr13 | chr21 |
| 4 | chr4, chr7 | chr19 |
| 5 | chr5, chr16 | chr20 |

---

## Step 3 — Train

```bash
python examples/atac/train_capy.py \
    --proj_dir /path/to/proj \
    --cell_type K562 \
    --fold 1
```

Key arguments:

- `--proj_dir`: project directory root (same across steps)
- `--cell_type`: one of `K562`, `HepG2`, `GM12878`, `IMR90`, `H1-hESC`
- `--fold`: 1–5
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
    --fold 1 \
    --timestamp 250101_120000 \
    --split test \
    --save_predictions
```

If `rep1.bw`/`rep2.bw` are present, the script automatically computes the pseudoreplicate JSD upper bound needed for Fig 1d. If they are absent it skips that step gracefully.

Add `--reverse_complement` for test-time augmentation (averages forward and RC predictions).

Outputs are written to `models/capy/atac/{cell_type}/fold{fold}/{timestamp}/evals/`.

---

## Step 5 — Plot

```bash
python examples/atac/plot_fig1cd.py \
    --eval_dir /path/to/proj/models/capy/atac/K562/fold1/{timestamp}/evals \
    --cell_type K562 \
    --fold 1
```

Produces:

- **`fig1c_{cell_type}_fold{fold}.pdf`** — density scatter of predicted vs. observed log total counts (Pearson r in title)
- **`fig1d_{cell_type}_fold{fold}.pdf`** — JSD distribution histograms with pseudoreplicate upper bound

---

## Baseline to beat (ChromBPNet, K562 ATAC-seq)

| Metric | ChromBPNet | Target |
|--------|-----------|--------|
| Pearson r (log total counts) | 0.70 ± 0.02 | ≥ 0.70 |
| JSD (predicted vs. observed) | approaches pseudorep bound | < pseudorep bound |
| Pseudorep JSD upper bound | — | reference ceiling |

Results across all five cell lines: median Pearson r = 0.69 (ChromBPNet paper Extended Fig 2b).
