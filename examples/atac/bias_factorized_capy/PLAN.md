# Plan: CAPY Stage-2 — Bias-Factorized Accessibility Model (ChromBPNet mirror)

## Context

Stage 1 (the CAPY Tn5-bias model) is done and benchmarked; the `chead` variant
(nonpeak counts r=0.625, beats ChromBPNet's 0.606; peak r=−0.116, passes the QC
gate) is the winner. This folder implements **Stage 2**: the bias-factorized
accessibility model, mirroring ChromBPNet's `chrombpnet_train_pipeline` with
**CAPY as the only changed variable** — CAPY backbone replaces BPNet in the
accessibility slot; CAPY-`chead` is scaled + frozen in the bias slot.

Target (fair benchmark): reproduce ChromBPNet's composed **peak** metrics on the
same data/fold. Reference K562/fold_0 (`.../models/chrombpnet/K562/fold_0/evaluation/K562.fold_0_chrombpnet_metrics.json`):
counts **pearsonr=0.692**, median JSD **0.341**.

All ChromBPNet mechanics were reverse-engineered from the container source
(`chrombpnet_latest.sif` → `/scratch/chrombpnet/chrombpnet/`):
`helpers/hyperparameters/find_chrombpnet_hyperparams.py`,
`training/models/chrombpnet_with_bias_model.py`,
`training/data_generators/batchgen_generator.py`, `training/train.py`,
`training/predict.py`, `training/metrics.py`.

---

## Locked-in decisions

- **Bias branch = `chead`** (scaled with δ, then frozen). Input checkpoint:
  `results/runs/models/chrombpnet_benchmark/capy_bias/atac/K562/fold0/chead/{checkpoints/best.pt, params_saved.yaml}`.
- **Accessibility = CAPY's full backbone** (`configs/atac_capy_accessibility.yaml`:
  encoder `[128,192,256]`, `hybrid_attention` bottleneck, ynet count head) at
  **input_length 2114 / output 1000**.
- **Training regimen (user choice)**: Adam lr 1e-3, **100 epochs, patience 10,
  ReduceLROnPlateau (factor 0.5, patience 9, min_lr 1e-6)**.
- **ChromBPNet-fixed data/composition**: `counts_weight = 76` (read from the fold's
  `chrombpnet_model_params.tsv`), `max_jitter = 500`, `negative_sampling_ratio = 0.1`,
  the +4/−4 Tn5 BigWig and the exact `filtered.{peaks,nonpeaks}.bed`.
- **Reuse ChromBPNet's exact Stage-2 files** — no re-derivation. Same `filtered.peaks.bed`,
  `filtered.nonpeaks.bed`, `data_unstranded.bw`, fold split.
- **Evaluation mirrors ChromBPNet exactly** — ported `counts_metrics` +
  `profile_metrics` definitions (below), not CAPY's `compute_performance_metrics`.
- **K562 / fold_0** first (only processed fold); everything parameterized by cell/fold.

---

## ChromBPNet Stage-2 mechanics reproduced

**Shared inputs** (read-only) under
`/grid/koo/home/shared/capybara/chrombpnet/models/chrombpnet/{cell}/fold_{fold}/`:
- `auxiliary/{cell}.fold_{fold}_filtered.peaks.bed` — train∪valid∪test peaks (outlier-filtered), 10-col (summit col 10).
- `auxiliary/{cell}.fold_{fold}_filtered.nonpeaks.bed` — negatives pool (subsampled to #peaks, random_state=1), all splits.
- `auxiliary/{cell}.fold_{fold}_data_unstranded.bw` — the +4/−4 Tn5 BigWig.
- `logs/{cell}.fold_{fold}_chrombpnet_model_params.tsv` — `counts_loss_weight=76`, `negative_sampling_ratio=0.1`, `max_jitter=500`, `inputlen=2114`, `outputlen=1000`, `chr_fold_path`.
- `logs/{cell}.fold_{fold}_chrombpnet_data_params.tsv` — `counts_sum_min_thresh`, `counts_sum_max_thresh` (δ-scaling in-range bounds), `trainings_pts_post_thresh`.
- `evaluation/{cell}.fold_{fold}_chrombpnet_metrics.json` — reference.

**Bias δ-scaling** (`find_chrombpnet_hyperparams.py::adjust_bias_model_logcounts`):
over training (train+valid chroms) non-peaks with `min_thresh < Σcounts < max_thresh`,
`δ = mean(log(1+Σcounts) − bias_pred_logcount)`, added to the bias count-head's
**final Linear bias only** (weights unchanged) → read-depth recalibration.

**Composition** (`chrombpnet_with_bias_model.py`), bias fully frozen:
- `profile_logits = acc_logits + bias_logits`  (element-wise, `(B,1,1000)`)
- `log_counts   = logsumexp([acc_logcount, bias_logcount])`  (`(B,1)`)
- Loss on the **composed** output: `profile_mnll` (w=1) + `76·count_log1p_mse`;
  only accessibility weights update.

**Data generator** (`batchgen_generator.py`):
- train = peaks (loaded at `inputlen+2·jitter`, random-cropped per epoch) + negatives
  resampled to `0.1·#peaks` **each epoch**, revcomp on.
- valid = peaks + negatives fixed once (seeded `0.1·#peaks`), jitter 0, revcomp off.
- Count target `log(1+Σcounts)`; profile target = raw per-base counts (multinomial).

**Tier-A evaluation** (`predict.py` + `metrics.py`; composed model on **peaks**, test chroms):
- `counts_metrics` = `spearmanr`, `pearsonr`, `mse` on log-count sums.
- `profile_metrics` (`pseudocount=0.001`):
  - `median_jsd` = median of `jensenshannon(true/(0.001+Σtrue), softmax(pred_logits))`.
  - `median_norm_jsd` = median of `clip((jsd − max_jsd)/(0 − max_jsd), 0, 1)`,
    `max_jsd = jensenshannon(true_prob, uniform)`.
- JSON schema identical to `chrombpnet_metrics.json`.

---

## Files in this folder

| File | Purpose |
|---|---|
| `PLAN.md` | This doc. |
| `__init__.py` | Package marker. |
| `../../../configs/atac_capy_accessibility.yaml` | Full CAPY @ 2114, jitter 500, counts_weight 76, 100 ep, patience 10, ReduceLROnPlateau. |
| `file_config.py` | `FactorizedCapyFiles`: resolve Stage-2 shared assets + chead bias checkpoint + CAPY outputs (`bias_scaled.pt`, composed `checkpoints/best.pt`, `nobias.pt`, `evals/`). |
| `scale_bias.py` | Compute δ over in-range train+valid non-peaks; add to chead count-head bias; save `bias_scaled.pt`. |
| `factorized_model.py` | `BiasFactorizedCAPY(accessibility, frozen_scaled_bias)`: add logits, logsumexp counts. |
| `data.py` | `FactorizedDataModule`: peaks + negatives split by fold chroms; **train IterableDataset** (per-epoch neg resample + jitter 500 + RC), **valid** map-style (fixed negatives, jitter 0). |
| `train_factorized.py` | Freeze bias, Adam over accessibility params + ReduceLROnPlateau, `run_training_loop`; save composed + `nobias.pt`. |
| `evaluate_factorized.py` | Predict composed model on peaks; emit ChromBPNet-schema metrics with ported metric defs. |
| `submit_factorized_fold.sh` | SLURM wrapper: `scale_bias` → `train_factorized` → `evaluate_factorized`. |

### Output layout
```
{proj_dir}/capy_chrombpnet/atac/{cell}/fold{N}/{ts}/
  bias_scaled.pt              # δ-adjusted frozen bias (Stage-2 input)
  checkpoints/best.pt         # composed accessibility (BiasFactorizedCAPY state)
  checkpoints/last.pt
  nobias.pt                   # accessibility-only (≙ chrombpnet_nobias)
  metrics.tsv                 # training log
  params_saved.yaml / config_saved.yaml
  evals/{cell}_chrombpnet_capy_metrics_{split}.json
```

---

## Phased to-do

- [x] **Tier A (NOW)** — scale → compose → train → evaluate; composed peak metrics
      mirroring `chrombpnet_metrics.json` (counts pearson/spearman/mse +
      median_jsd/median_norm_jsd).
- [x] **Tier B — attribution & TF-MoDISco parity** — BUILT in the shared folder
      `examples/atac/attribution/`. Contribution scoring on the `nobias` CAPY
      (profile head = weighted-sum mean-normed logits; counts head optional) → same
      `.h5` schema (`raw`/`shap`/`projected_shap`, (N,4,L)) → the container's
      `modisco motifs -n 50000 -w 500` / `modisco report` unchanged.
      **Engine (`--method`, default `gradientshap`):** the accessibility net uses a
      `hybrid_attention` bottleneck where DeepLIFT/DeepSHAP is unreliable through
      self-attention, so the nobias model uses **`captum` GradientShap** (gradient
      method, robust to attention); `--method deeplift` exists for experimentation
      but is not recommended here. Scalar targets + `.h5` schema are identical
      across engines, so `modisco` is unchanged. Outputs namespaced under
      `attribution/<method>/`. See `attribution_nobias.py`, `run_modisco.sh`,
      `submit_attribution_nobias.sh`.
- [ ] **Tier C (deferred) — marginal footprinting**: reproduce the Tn5-motif
      marginal-footprint response `< 0.003` on the `nobias` CAPY (mirrors
      `evaluation/marginal_footprints/marginal_footprinting.py`).
- [ ] Extend beyond K562/fold_0 once other cell lines / folds are processed.

## Verification (see repo plan)
δ-scaling sanity · composition unit test · region-count parity vs
`chrombpnet_data_params.tsv` · small end-to-end run · benchmark vs reference
(target composed peak r ≈ 0.69, JSD ≈ 0.34).
