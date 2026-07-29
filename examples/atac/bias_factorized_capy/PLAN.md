# Stage 2 — Bias-factorized accessibility CAPY

> **The shared design, ChromBPNet mechanics, results, and open items now live in
> [`../BENCHMARK_0713.md`](../BENCHMARK_0713.md).** This file only covers what is specific to Stage 2.

## Purpose

Mirror ChromBPNet's `chrombpnet_train_pipeline` with **CAPY as the only changed variable**: a full
CAPY backbone in the accessibility slot, and the Stage-1 CAPY bias model (`chead`) δ-scaled and
**frozen** in the bias slot.

Composition, with the bias branch frozen:

    profile_logits = acc_logits + bias_logits
    log_counts     = logsumexp([acc_logcount, bias_logcount])

Loss is on the **composed** output; only accessibility weights update. This forces the accessibility
net to learn only what the bias model cannot already explain — which is what makes the exported
`nobias.pt` "bias-corrected". See BENCHMARK_0713.md §4.

Config: `configs/atac_capy_accessibility.yaml` (encoder [128,192,256], `hybrid_attention` bottleneck,
ynet count head, 2114/1000). ChromBPNet-fixed: `counts_weight 76`, `max_jitter 500`,
`negative_sampling_ratio 0.1`. Training regimen (our choice, deviates from ChromBPNet): Adam 1e-3,
100 epochs, patience 10, ReduceLROnPlateau (0.5, patience 9, min_lr 1e-6).

## Files

| File | Purpose |
|---|---|
| `file_config.py` | `FactorizedCapyFiles`: Stage-2 shared assets + the Stage-1 `chead` bias checkpoint + Stage-2 outputs. |
| `scale_bias.py` | δ recalibration — `δ = mean(log1p(Σcounts) − bias_pred_logcount)` over in-range train+valid non-peaks, added to the bias count-head's **final Linear bias only**. Writes `bias_scaled.pt`. |
| `factorized_model.py` | `BiasFactorizedCAPY(accessibility, frozen_scaled_bias)`. Bias frozen *and* kept in `.eval()` so BatchNorm stats never update. |
| `data.py` | `FactorizedDataModule` — train `IterableDataset` (per-epoch negative resample, jitter 500, RC); valid map-style, fixed seeded negatives, jitter 0. |
| `train_factorized.py` | Freeze bias, Adam over accessibility params + ReduceLROnPlateau; saves composed `best.pt` + accessibility-only `nobias.pt`. |
| `evaluate_factorized.py` | Composed model on peaks; ChromBPNet-schema metrics JSON. |
| `submit_factorized_fold.sh` | `scale_bias` → `train_factorized` → `evaluate_factorized`. |

## Run

```bash
sbatch submit_factorized_fold.sh [timestamp] [cell_type] [fold] [bias_timestamp] [gpu]
sbatch submit_factorized_fold.sh chead_test1 K562 0 chead
```

Outputs: `results/runs/models/chrombpnet_benchmark/capy_chrombpnet/atac/{cell}/fold{N}/{ts}/`
(`bias_scaled.pt`, `checkpoints/best.pt`, `nobias.pt`, `metrics.tsv`,
`evals/{cell}_chrombpnet_capy_metrics_{split}.json`).

## Status (K562 / fold_0)

- [x] **Tier A** — one run, `chead_test1`. Composed peak metrics vs ChromBPNet's `nobias`:
      counts pearson **0.709 vs 0.692**, spearman **0.614 vs 0.597**, `median_jsd` 0.348 vs 0.341.
      **CAPY slightly wins on counts, essentially ties on profile shape.** Details: BENCHMARK_0713.md §6.
- [~] **Tier B** — built in `../attribution/`; GradientShap is the default engine here because
      DeepLIFT is not sound through the `hybrid_attention` bottleneck. **MoDISco on
      `gradientshap_uncorrected/` recovers the same K562 TF vocabulary as ChromBPNet's nobias model —
      CTCF (6,081 vs 6,202 seqlets), GATA, KLF/SP, BACH/NFE2, NFY, ETS, NRF1, YY1 — motif-for-motif at
      the top, with a faint residual TN5_6 trace ChromBPNet lacks.** Comparison: BENCHMARK_0713.md §8a.
      **Incomplete: the Majdandzic-corrected run was never produced** — only `gradientshap_0710_buggy/`
      and `gradientshap_uncorrected/` exist, so §8a's verdict is provisional. See §7 and §9 item 1.
- [~] **Tier C** — marginal footprinting. **Pipeline built + sanity-checked on K562/fold_0
      (`chead_test1`, `cw50`); this is not the final evaluation run.** The nobias `max_bias_response`
      is `corrected` in both runs (all five TN5 rounded maxima 0.001); the frozen bias branch is
      `uncorrected` (TN5 maxima 0.042–0.082), confirming the insertion/loader path works. Descriptive
      numbers and the CAPY-vs-ChromBPNet table: BENCHMARK_0713.md §8b. **§8a stays open** — the
      footprint shape is not yet used to adjudicate over-subtraction.
      Still open for the real evaluation (not done here): (a) bootstrap CIs on the per-motif maxima;
      (b) confirm whether the δ-scaling adjusts the profile logits or the counts head — must be
      settled before Tier C is used to speak to §8a.
- [ ] Folds 1–4 and the other four cell lines.
