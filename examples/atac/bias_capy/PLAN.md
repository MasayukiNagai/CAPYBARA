# Stage 1 — CAPY Tn5-bias model

> **The shared design, ChromBPNet mechanics, results, and open items now live in
> [`../BENCHMARK_0713.md`](../BENCHMARK_0713.md).** This file only covers what is specific to Stage 1.

## Purpose

Learn *pure Tn5 sequence-insertion bias* from **non-peaks only** — regions with no TF-driven signal,
so anything the model learns must be enzyme bias. The model is deliberately **small and myopic**
(≈212K params, 2 encoder stages, local `residual_conv` bottleneck) so it can learn the Tn5 motif but
**not** distal TF grammar. See BENCHMARK_0713.md §4 for why that matters.

Config: `configs/atac_capy_bias.yaml`. Sweep variants: `configs/atac_search/bias_*.yaml`.

Hyperparameters come from ChromBPNet's own TSVs, not re-tuned: `counts_weight 5.8`, `max_jitter 0`,
all-negatives, Adam 1e-3, batch 64, seed 1234.

## Files

| File | Purpose |
|---|---|
| `file_config.py` | `BiasCapyFiles`: resolve read-only shared inputs (filtered bias beds, Tn5 BigWig, params TSV, fold split) + CAPY output dirs. |
| `data.py` | `BiasDataModule` — non-peaks only, split by fold chroms, `max_jitter 0`, no negative subsampling. |
| `train_bias.py` | Stage-1 trainer. |
| `evaluate_bias.py` | Tier-A eval on peaks AND nonpeaks; ChromBPNet metric schema; enforces the peak-Pearson > −0.5 QC gate. Defines `chrombpnet_profile_jsd`, which Stage 2 imports. |
| `submit_bias_fold.sh` | Train + evaluate one fold. |
| `submit_bias_eval.sh` | Re-evaluate an existing checkpoint. |

## Run

```bash
# train + Tier-A evaluate (a sweep config is selected via BIAS_PARAMS)
sbatch submit_bias_fold.sh [timestamp] [cell_type] [fold] [gpu]
BIAS_PARAMS=configs/atac_search/bias_relu.yaml sbatch submit_bias_fold.sh relu K562 0

# eval only, on an existing checkpoint
sbatch submit_bias_eval.sh <timestamp> [cell_type] [fold] [split] [gpu]
```

Outputs: `results/runs/models/chrombpnet_benchmark/capy_bias/atac/{cell}/fold{N}/{ts}/`
(`checkpoints/best.pt`, `metrics.tsv`, `evals/{cell}_bias_capy_metrics_{split}.json`).

## Status (K562 / fold_0)

- [x] **Tier A** — 8-run sweep done; **`chead` is the winner** (nonpeak counts r **0.625** vs
      ChromBPNet's 0.606; peak r −0.116, QC gate PASS). Numbers in BENCHMARK_0713.md §6.
- [x] **Tier B** — attribution + TF-MoDISco, built in `../attribution/`. Canonical run is
      `attribution/deeplift/` (post-`SameMaxPool1d` fix). **Profile head: pure TN5 on both CAPY and
      ChromBPNet, zero negative patterns — the bias model learned Tn5 and nothing else, which is the
      precondition for the whole factorization being sound.** Counts head is diffuse/GC-dominated on
      both sides and disagrees in polarity — least trustworthy artifact in the set.
      **Motif comparison: BENCHMARK_0713.md §8a. Directory-naming caveats and the counts-head
      convergence residual: §7.**
- [ ] **Tier C** — marginal footprinting. Not built.
- [ ] Folds 1–4 and the other four cell lines.
