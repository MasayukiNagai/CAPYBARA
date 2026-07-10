# Plan: Fair CAPY vs. ChromBPNet ATAC Benchmark (Full Pipeline Parity)

## Context

We want an apples-to-apples benchmark where the **only** variable is the model
architecture — CAPY replacing BPNet — while data, folds, and the entire
bias-factorized training regimen are held identical to ChromBPNet.

ChromBPNet is not a single model: it is a **two-stage, bias-factorized**
pipeline (train a Tn5-bias model on background regions → freeze+scale it → train
an accessibility model on top). The current `examples/atac/train_capy.py` trains
a single naive model directly on observed signal and is **out of scope** for this
benchmark. This document (a) inventories what ChromBPNet assets we already have,
(b) specifies ChromBPNet's exact training workflow (reverse-engineered from the
container source), and (c) defines how CAPY will mirror every step on the same
dataset. **No code is written today — this is the design.**

Reference implementation lives inside the container and was read directly:
`/grid/koo/home/shared/capybara/chrombpnet/chrombpnet_latest.sif`
→ `/scratch/chrombpnet/chrombpnet/` (`pipelines.py`,
`training/models/chrombpnet_with_bias_model.py`, `training/models/bpnet_model.py`,
`training/train.py`, `training/data_generators/batchgen_generator.py`,
`helpers/hyperparameters/find_{bias,chrombpnet}_hyperparams.py`,
`helpers/hyperparameters/param_utils.py`).

---

## Part 1 — What we already have from ChromBPNet (shared, K562 only)

Root: `/grid/koo/home/shared/capybara/chrombpnet/` (`data/`, `genome/` are
symlinks into `/grid/koo/home/shared/data/chrombpnet/`).

**Reference / genome**
- `genome/hg38.fasta` (+ `.chrom.sizes`), ENCODE blacklist, and Zenodo fold
  splits `data/splits/fold_{0..4}.json` (zero-based ChromBPNet chrom splits).

**Processed data — `data/atac/`** (K562 only so far)
- `raw/K562/`: 3 ENCODE replicate BAMs → merged `K562.merged.bam` (+`.bai`), 22.9 GB.
- `processed/K562/peaks/`: `K562.raw.narrowPeak.gz` (ENCODE IDR peaks) →
  `K562.peaks.filtered.bed` → `K562.peaks_no_blacklist.bed` (10-col narrowPeak,
  summit in col 10).
- `processed/K562/nonpeaks/fold_{0..4}/K562.fold_N_negatives.bed`: GC-matched
  background regions (from `chrombpnet prep nonpeaks`), per fold, 10-col.

**Trained models — `models/`** (K562, all 5 folds)
- `bias/K562/fold_N/models/K562.fold_N_bias.h5` — trained Tn5-bias model.
- `chrombpnet/K562/fold_N/`:
  - `models/K562.fold_N_bias_model_scaled.h5` — depth-scaled frozen bias.
  - `models/K562.fold_N_chrombpnet.h5` — full composed model.
  - `models/K562.fold_N_chrombpnet_nobias.h5` — accessibility-only (bias-corrected).
  - `auxiliary/K562.fold_N_data_unstranded.bw` — **the +4/-4 Tn5 BigWig ChromBPNet
    actually trained on** (this is the canonical "same data" signal to reuse).
  - `auxiliary/K562.fold_N_filtered.{peaks,nonpeaks}.bed` — the exact
    post-filter train/test regions used.
  - `logs/*_{model,data}_params.tsv`, `*.args.json` — full hyperparameter provenance.

**Verified hyperparameters (K562 fold_0)**
- Bias stage: `inputlen 2114`, `outputlen 1000`, `max_jitter 0`, `128 filters /
  4 dil`, `counts_loss_weight 5.8`, nonpeak upper count cutoff `101`, `190,863`
  training points, `negative_sampling_ratio 1.0`.
- Main stage: `inputlen 2114`, `outputlen 1000`, `max_jitter 500`, `512 filters /
  8 dil`, `counts_loss_weight 76`, `negative_sampling_ratio 0.1`,
  `outlier_threshold 0.9999`.
- Both: Adam `lr 1e-3`, `50 epochs`, EarlyStopping(`val_loss`, patience `5`,
  restore-best), `batch 64`, `seed 1234`, no LR scheduler (commented out upstream).

---

## Part 2 — ChromBPNet's training workflow (exact)

### Stage 0 — Data processing (already done, inside `chrombpnet`)
`reads_to_bigwig` shifts the merged BAM by **+4/−4** (ATAC Tn5) into an
unstranded BigWig. Peaks come from ENCODE IDR (blacklist-filtered); GC-matched
non-peaks come from `chrombpnet prep nonpeaks`. All regions are **summit-centered**
(`start + col10`).

### Stage 1 — Bias model (`train_bias_pipeline`)
Purpose: learn *pure Tn5 sequence bias* from regions with no TF-driven signal.
1. `find_bias_hyperparams`: from non-peaks, keep those with total count
   `< quantile(peak_counts, 0.01) × 0.5` (the `-b 0.5` factor), drop 0.9999
   outliers; `counts_weight = median(kept counts)/10`. Writes
   `filtered.bias_nonpeaks.bed` + params.
2. `train.py` trains **BPNet(128 filters, 4 dil, conv1=21, profile kernel=75)**
   on **non-peaks only** (jitter 0). Two heads: profile logits (`out=1000`) +
   `logcount` (GAP→Dense). Loss = `multinomial_nll(profile)` +
   `counts_weight·MSE(logcount, log(1+Σcounts))`.

### Stage 2 — Bias-factorized model (`chrombpnet_train_pipeline`)
1. `find_chrombpnet_hyperparams`:
   - Filter peaks+negatives (edge + 0.9999 outlier on peaks∪sampled-negatives).
   - **Scale bias** (`adjust_bias_model_logcounts`): add constant
     `δ = mean(log(1+cts) − bias_pred_logcount)` over training non-peaks to the
     bias count-head's Dense bias → `bias_model_scaled.h5`. Corrects read-depth.
   - `counts_weight = median(in-range counts)/10` (= 76).
2. `train.py` trains **accessibility BPNet(512 filters, 8 dil)** on
   **peaks + 1:10 GC-matched negatives**, jitter 500, with the frozen scaled bias
   composed in:
   - `profile_logits = acc_logits + bias_scaled_logits`  (add logits)
   - `logcount = logsumexp([acc_logcount, bias_scaled_logcount])`  (counts add linearly)
   - Loss on the **composed** output; only accessibility weights update.
3. Saves `chrombpnet.h5` (composed) and `chrombpnet_nobias.h5` (accessibility alone).

### Data generator behavior (both stages)
`ChromBPNetBatchGenerator`: loads summit-centered windows `inputlen+2·jitter` /
`outputlen+2·jitter`; **each epoch** random-crops (jitter), reverse-complements,
shuffles, and **re-subsamples negatives** to `ratio×#peaks` (`on_epoch_end`).
Count target = `log(1 + Σ counts)`; profile target = raw count vector (multinomial).

---

## Part 3 — How CAPY mirrors every step (same data, fair benchmark)

Key enabler: **CAPY's `forward(x) → (profile_logits, log_counts)`
(`capybara/model.py:176`) already matches BPNet's two-head signature**, so CAPY
drops into both the bias slot and the accessibility slot unchanged.

### Locked-in design decisions (from user)
- **Bias model = a small CAPY variant** (reduced encoder channels / shallower),
  the CAPY-native analog of ChromBPNet's shallow 128/4 bias net.
- **Same data = reuse ChromBPNet's exact files**: train on the per-fold
  `auxiliary/K562.fold_N_data_unstranded.bw` and the shared
  `nonpeaks/fold_N/..._negatives.bed` + `peaks_no_blacklist.bed` (bit-identical
  signal, no re-derivation).
- **Count loss = keep CAPY's `count_log1p_mse_loss`** (`capybara/losses.py`),
  which already targets `log(1+Σ)` — matches ChromBPNet's MSE-on-log1p in practice.

### Step-by-step mapping

| ChromBPNet step | CAPY mirror | Reuse / basis |
|---|---|---|
| Tn5 BigWig + peaks + GC negatives | **Reuse shared files as-is** | `data/atac/processed/K562/…`, model `auxiliary/*.bw` |
| `find_bias_hyperparams` (nonpeak count filter, counts_weight) | New `prep_hyperparams.py` (bias mode) | port `find_bias_hyperparams` logic; `extract_loci` (`capybara/data.py:188`) to get counts |
| Bias BPNet(128/4) | **small-CAPY** config `configs/atac_capy_bias.yaml` | `CAPY` + `normalize_config` |
| Bias training loop | New `train_bias_capy.py` | reuse `run_training_loop`, `compute_losses` (`examples/shared/train_utils.py`) |
| `find_chrombpnet_hyperparams` (peak/neg filter, outlier, counts_weight) | New `prep_hyperparams.py` (main mode) | port logic |
| `adjust_bias_model_logcounts` (bias scaling) | New `scale_bias.py`: add δ to CAPY bias count-head bias | `final_count_layer` (`train_utils.py:392`) locates `count_head.mlp[-1]` |
| `chrombpnet_with_bias_model` composition | New `factorized_model.py`: `BiasFactorizedCAPY(acc, frozen_bias)` — add logits, `logsumexp` counts | wraps two `CAPY` instances |
| Accessibility BPNet(512/8) | full-size `configs/atac_capy_accessibility.yaml` | existing `configs/atac_default.yaml` as base |
| Peak+1:10 negative generator, per-epoch jitter/RC/resample | New `datamodule.py` | extend `AtacDataModule`/`ProfileDataset` (`examples/atac/data.py`, `capybara/data.py`) to two region sets + `on_epoch`-style negative resampling |
| Stage-2 training | New `train_factorized_capy.py` (freeze bias, train accessibility) | `run_training_loop`; freezing pattern like `configure_count_finetune_parameters` |
| Save composed + nobias | checkpoint both `BiasFactorizedCAPY` and its `accessibility` submodule | `save_training_checkpoint` |

### New files — all under one new folder `examples/atac/bias_capy/`

Full per-step create/modify list is in **Part 5** below.

### Output layout (parallel to ChromBPNet's)
```
models/capy/atac/K562/fold{N}/{ts}/
  bias/checkpoints/best.pt                # Stage 1 CAPY bias
  bias_scaled.pt                          # δ-adjusted
  checkpoints/best.pt                     # Stage 2 composed accessibility
  nobias.pt                               # accessibility-only (≙ chrombpnet_nobias)
  evals/…                                 # same file contract as existing evaluate.py
```

---

## Part 4 — Bias model deep dive & CAPY bias mirror

### 4.1 What the ChromBPNet bias model is (reverse-engineered)

**Size** (K562 fold_0, byte-for-byte on disk):

| Model | `.h5` | Trainable params | Config |
|---|---|---|---|
| **Bias** `K562.fold_0_bias.h5` | **2.69 MB** | **≈ 217,730** | 128 filters, 4 dilated layers, RF ≈ 81 bp |
| Accessibility `_chrombpnet_nobias.h5` | 25.6 MB | ≈ 6.38 M | 512 filters, 8 dilated layers, RF ≈ 1041 bp |
| Full composed `_chrombpnet.h5` | 77.5 MB | ≈ 6.38 M (+ frozen bias) | acc + frozen scaled bias |

Bias net is ~29× smaller. Two deliberate design choices make it small AND unable
to learn accessibility: **narrow/shallow** (128/4 vs 512/8) **and a tiny ~81 bp
receptive field** (only local Tn5 sequence context, no distal TF grammar).

**Architecture** (`training/models/bpnet_model.py`, same body for both; only
`filters`/`n_dil_layers` differ):
```
Input(2114,4)
 → Conv1D k=21 valid ReLU "bpnet_1st_conv"           (conv1_kernel_size=21, fixed)
 → for i in 1..N_dil: Conv1D k=3 dilation=2**i ReLU + symmetric Crop + residual add
      bias: dil 2,4,8,16  → RF 81 bp     acc: dil 2..256 → RF ≈ 1041 bp
 → Profile head: Conv1D k=75 (profile_kernel_size, fixed) → Crop → logits (len 1000)
 → Count head:   GlobalAvgPool1D → Dense(1) → log-count scalar
Outputs: [profile_logits(1000), logcount(1)]
```
Bias params: conv1 10,880 + 4×49,280 + profile 9,601 + count 129 ≈ **217,730**.
Loss: `multinomial_nll(profile)` (w=1) + `MSE(logcount, log1p(Σcts))` (w=counts_loss_weight).

**Training** (`train_bias_pipeline` → `find_bias_hyperparams` → `train.py`):
- Data: **non-peaks only** (`peaks="None"`); signal = the +4/−4 Tn5 BigWig.
- Nonpeak filter: keep `count < quantile(peak_cts,0.01)×0.5` (`-b 0.5`) → cutoff **101**;
  trim 0.9999/0.0001 outliers → **190,863** training points (fold_0).
- `counts_loss_weight = median(retained nonpeak cts)/10 = 5.8` (floored at 1.0).
- inputlen 2114, outputlen 1000, **max_jitter 0**, all-negatives (ratio 1.0),
  Adam 1e-3, ≤50 epochs, batch 64, seed 1234, EarlyStopping(val_loss, patience 5,
  restore-best); ReduceLROnPlateau present but commented out.

**Evaluation** (three tiers — see 4.3 for phasing):
- **(A) Prediction** (`training/predict.py` → `bias_metrics.json`): scored on **both
  peaks and non-peaks**. Counts = Spearman/Pearson/MSE on log-counts; Profile =
  `median_jsd` + `median_norm_jsd` (+ shuffled baseline). Observed fold_0: nonpeaks
  Pearson **0.61** / peaks Pearson **−0.17** — good on background, deliberately bad
  on peaks. (Upstream omits MNLL, profile-Pearson, count-R²; repo `capybara/metrics.py` adds them.)
- **(B) QC gate**: hard assert **bias peak-counts Pearson > −0.5** (else the bias
  net captured AT/GC composition, not pure Tn5; fix by raising `-b`).
- **(C) Attribution + TF identification**: DeepSHAP on count head (`sum logcount`)
  and profile head (weighted-sum mean-normed logits) on ≤30K subsampled peaks →
  `.{counts,profile}_scores.h5` (schema `raw`/`shap`/`projected_shap`, (N,4,L)) →
  **modisco-lite** (`modisco motifs -n 50000 -w 500`) → `modisco report` (TOMTOM vs
  bundled MEME DB) → motif logos. Confirms only Tn5/repeat motifs, no TF (visual;
  TF q-values should be > 1e-4). Plus **marginal footprinting**: Tn5 response < 0.003
  on the corrected model.

### 4.2 CAPY bias mirror — decisions locked

**Reuse ChromBPNet's exact prep** — no re-derivation needed:
- Regions: `models/bias/K562/fold_N/auxiliary/K562.fold_N_filtered.bias_nonpeaks.bed`
  (train) and `..._filtered.bias_peaks.bed` (eval). Signal:
  `.../auxiliary/K562.fold_N_data_unstranded.bw`. `counts_loss_weight = 5.8` read
  from `.../logs/K562.fold_N_bias_model_params.tsv`. Fold split from
  `data/splits/fold_N.json` (split those regions by chrom into train/valid).

**Bias-CAPY = small capacity + constrained receptive field.** CAPY is a pooling
U-Net so exact 81 bp RF isn't reproducible, but we make it as myopic as CAPY allows:
fewer encoder stages **and a local `residual_conv` bottleneck (NOT `hybrid_attention`,
which is globally connected)**. Proposed `configs/atac_capy_bias.yaml` starting point
(tune to ~200–400K params; validate via the 4.1(B) QC gate + later 4.1(C) modisco):
```yaml
model:
  input_length: 2114        # native — SameMaxPool1d (ceil+pad) + center_crop_1d handle it
  output_length: 1000
  dna_embedder_channels: 32
  encoder_channels: [48, 64]        # 2 stages (÷8); [64] (÷4) = even more myopic alt
  pool_size: 2
  decoder_channels: [64, 48, 32]    # len == len(encoder_channels)+1
  output_embedder_channels: 32
  embedding_projector: {enabled: false}
  norm_type: batch
  profile_head: {source: decoder, num_outputs: 1, kernel_size: 75}
  count_head: {type: ynet, source: bottleneck, conv_hidden_dims: [32], mlp_hidden_dims: [32]}
  bottleneck: {type: residual_conv, depth: 1, kernel_size: 5}   # LOCAL → constrains RF
```

**Stage-1 training** (`train_bias_capy.py`): non-peaks only, `max_jitter 0`,
all-negatives, counts_weight 5.8, Adam 1e-3, ≤50 ep, batch 64, patience 5, seed 1234.
Reuse `run_training_loop`/`compute_losses` (`examples/shared/train_utils.py`),
`count_log1p_mse_loss` + `profile_mnll_loss` (`capybara/losses.py`).

**Stage-1 evaluation NOW (Tier A only)** (`evaluate_bias_capy.py`): score the CAPY
bias model on both `filtered.bias_peaks.bed` and `filtered.bias_nonpeaks.bed`;
report counts Pearson/Spearman/MSE + profile median JSD for **peaks, nonpeaks,
peaks_and_nonpeaks** (mirror `bias_metrics.json` schema) via
`capybara.metrics.compute_performance_metrics`; reproduce the **peak-Pearson > −0.5**
QC assertion.

### 4.3 Phased to-do (record — Tiers B & C are AFTER training path works)

- [ ] **Now (Tier A):** bias config + `train_bias_capy.py` + `evaluate_bias_capy.py`
      (prediction metrics on peaks/nonpeaks + peak-Pearson>−0.5 gate).
- [x] **Tier B — attribution & TF-MoDISco parity** — BUILT in the shared folder
      `examples/atac/attribution/` (not this folder). CAPY contribution scoring on
      the count head (`sum(logcount)`) and profile head (weighted-sum mean-normed
      logits) writes the **same `.h5` schema** (`raw`/`shap`/`projected_shap`,
      (N,4,L)); then the container's `modisco motifs -n 50000 -w 500` / `modisco
      report` run **unchanged** (model-agnostic) to confirm the CAPY bias model
      learned only Tn5 motifs. **Engine (`--method`, default `gradientshap`):**
      GradientShap (`captum`) is the default; DeepLIFT/DeepSHAP via
      `tangermeme.deep_lift_shap` (`--method deeplift`) is the closest-to-ChromBPNet
      comparison (same estimator: Rescale rule averaged over 20 dinuc-shuffled refs =
      ChromBPNet's `TFDeepExplainer`). Outputs namespaced under `attribution/<method>/`.
      See `examples/atac/attribution/` (`attribution_bias.py`, `run_modisco.sh`,
      `submit_attribution_bias.sh`).
      **DeepLIFT convergence deltas fixed (`nonlinear_ops.py`):** the earlier high deltas
      (~1–4) came from CAPY's `SameMaxPool1d`, a custom max-pool class tangermeme could not
      recognize by type (its rule is keyed to `torch.nn.MaxPool1d`) so it treated it as
      linear. Registered via `additional_nonlinear_ops` (reuse-safe via
      `track_shared_maxpools`, since the encoder shares one pool instance across resolutions;
      `maxpool1d` *is* in Captum's `SUPPORTED_NON_LINEAR`, so this is standard). Result:
      **profile delta 2.43→0.135** (residual is inherent GELU-rescale numerical noise).
      The count-head `LayerNorm` is deliberately **left at the Captum/tangermeme default**
      (autograd-gradient passthrough — neither library registers LayerNorm; the generic
      elementwise Rescale rule is the wrong tool for a cross-dim op), so the **counts head
      keeps a ~0.46 gap** — standard-tool behavior, accepted. GELU/BatchNorm(eval)/avg-pool
      need nothing (already registered or linear).
- [ ] **To-do (Tier C) — marginal footprinting**, once corrected/factorized CAPY
      exists: reproduce the Tn5-motif marginal-footprint response `< 0.003` check.

---

## Part 5 — Files to create / modify, per step

**Everything new lives in one new folder: `examples/atac/bias_capy/`** (plus two
YAML configs under `configs/`). Nothing under `examples/atac/` is overwritten;
the old `train_capy.py`/`data.py` are left untouched (out of scope).

### Build NOW — Stage 1 (CAPY bias model) + Tier-A evaluation

| # | File | New/Mod | Purpose | Reuses |
|---|---|---|---|---|
| 0 | `examples/atac/bias_capy/PLAN.md` | **new** | repo-tracked copy of this plan (phased to-dos) | — |
| 1 | `examples/atac/bias_capy/__init__.py` | **new** | package marker | — |
| 2 | `configs/atac_capy_bias.yaml` | **new** | small + constrained-RF CAPY (dims in §4.2): 2 encoder stages, `residual_conv` bottleneck, 1-out profile k=75, inputlen 2114 | `capybara.config.normalize_config` |
| 3 | `examples/atac/bias_capy/file_config.py` | **new** | resolve per-fold shared paths: filtered `bias_{peaks,nonpeaks}.bed`, `data_unstranded.bw`, `bias_model_params.tsv` (counts_weight 5.8), `splits/fold_N.json`, genome, chrom.sizes; CAPY output dirs | pattern from `examples/atac/file_config.py` |
| 4 | `examples/atac/bias_capy/data.py` | **new** | Stage-1 datamodule: **nonpeaks only**, split by fold chroms, `max_jitter 0`, reverse-complement, all-negatives (no subsample); count target `log1p(Σ)` | `extract_loci`, `ProfileDataset`, `load_chrom_names` (`capybara/data.py`) |
| 5 | `examples/atac/bias_capy/train_bias.py` | **new** | Stage-1 trainer (Adam 1e-3, ≤50 ep, batch 64, patience 5, seed 1234, counts_weight from tsv); saves `bias/checkpoints/best.pt` + params | `run_training_loop`, `compute_losses`, `save_training_checkpoint` (`examples/shared/train_utils.py`); `count_log1p_mse_loss`, `profile_mnll_loss` (`capybara/losses.py`) |
| 6 | `examples/atac/bias_capy/evaluate_bias.py` | **new** | Tier-A eval: score bias model on **peaks AND nonpeaks**, emit counts Pearson/Spearman/MSE + profile median JSD for peaks / nonpeaks / peaks_and_nonpeaks (mirror `bias_metrics.json`); assert **peak-Pearson > −0.5** | `capybara.metrics.compute_performance_metrics`; predict loop from `examples/atac/evaluate.py` |
| 7 | `examples/atac/bias_capy/submit_bias_fold.sh` | **new** | sbatch wrapper (one fold); user launches (see [[prepare-dont-submit-jobs]]) | pattern from `run_train_capy.sh` |

### FUTURE — Stage 2 (bias-factorized accessibility CAPY)

| # | File | New/Mod | Purpose |
|---|---|---|---|
| 8 | `configs/atac_capy_accessibility.yaml` | **new** | full CAPY (accessibility), inputlen 2114; base = `configs/atac_default.yaml` |
| 9 | `examples/atac/bias_capy/scale_bias.py` | **new** | `adjust_bias_model_logcounts`: add δ to CAPY bias count-head bias → `bias_scaled.pt` (`final_count_layer` locates `count_head.mlp[-1]`) |
| 10 | `examples/atac/bias_capy/factorized_model.py` | **new** | `BiasFactorizedCAPY(acc, frozen_scaled_bias)`: profile = add logits, counts = `logsumexp` |
| 11 | `examples/atac/bias_capy/data.py` | **mod** | extend to peaks + 1:10 GC negatives, jitter 500, per-epoch negative resample |
| 12 | `examples/atac/bias_capy/train_factorized.py` | **new** | Stage-2 trainer (freeze scaled bias, train accessibility); save composed + `nobias.pt` |
| 13 | `examples/atac/bias_capy/evaluate_factorized.py` | **new** | reuse existing `evaluate.py` metric/JSD contract on the composed model |
| 14 | `examples/atac/bias_capy/submit_factorized_fold.sh` | **new** | sbatch wrapper |

### FUTURE — Tier B (attribution & TF-MoDISco) and Tier C (marginal footprinting)

| # | File | New/Mod | Purpose |
|---|---|---|---|
| 15 | `examples/atac/bias_capy/attribution.py` | **new** | Captum DeepLIFT/DeepLiftShap on count head (`sum logcount`) + profile head (weighted-sum mean-normed logits) → write **same `.h5` schema** (`raw`/`shap`/`projected_shap`, (N,4,L)) |
| 16 | `examples/atac/bias_capy/run_modisco.sh` | **new** | feed the `.h5` to the container's `modisco motifs`/`modisco report` **unchanged** (model-agnostic) → confirm bias-CAPY learned only Tn5 motifs |
| 17 | `examples/atac/bias_capy/marginal_footprint.py` | **new** | Tier C: Tn5-motif marginal-footprint response `< 0.003` on corrected CAPY |

### Reused unchanged (no edits)
`capybara/model.py` (CAPY), `capybara/data.py` (`extract_loci`, `ProfileDataset`),
`capybara/losses.py`, `capybara/metrics.py`, `examples/shared/train_utils.py`,
`examples/shared/metrics.py`. Shared data/models under
`/grid/koo/home/shared/capybara/chrombpnet/` are read-only inputs.

---

## Open items / caveats

1. **Bias-CAPY exact param count / RF** — dims in 4.2 are a proposed starting
   point; confirm ~200–400K params and validate myopia empirically via the
   peak-Pearson>−0.5 gate (and later modisco). If peak-Pearson is too negative or
   modisco shows TF motifs, reduce depth to `encoder_channels: [64]` (÷4).
2. ~~Receptive field vs. inputlen 2114~~ **RESOLVED**: CAPY accepts 2114/1000
   natively (`SameMaxPool1d` ceil+pad, `center_crop_1d`); no arch change.
3. **Bias jitter 0**: Stage-1 datamodule must support `max_jitter 0` (no crop).
4. **Composition math on CAPY profile shape** (Stage 2): CAPY profile is `(B, 1, L)`;
   ensure logit-add + `logsumexp` counts broadcast correctly and MNLL is over `L`.
5. **Only K562 processed so far** — benchmark starts K562; other 4 cell lines
   need the `chrombpnet_setup` steps re-run before extending.
6. **Repo-tracked plan doc**: first implementation step is to copy this plan into
   the repo (`examples/atac/bias_capy/PLAN.md`) so the phased to-dos
   above are version-controlled alongside the code.

---

## Verification (once built, later)

- **Data parity check**: confirm CAPY reads the same `data_unstranded.bw` and
  `filtered.{peaks,nonpeaks}.bed` ChromBPNet used (compare region counts to
  `*_data_params.tsv`: e.g. bias fold_0 = 190,863 pts).
- **Bias model sanity (Tier A)**: after Stage-1, the CAPY bias model should mirror
  ChromBPNet's signature — good nonpeak counts (Pearson ≈ 0.5–0.6) but poor/negative
  peak counts, with peak-Pearson **> −0.5** (QC gate passes). Compare against the
  reference `K562.fold_N_bias_metrics.json`.
- **Bias-scaling sanity**: reproduce δ and verify mean predicted logcount ≈ mean
  `log(1+cts)` on training nonpeaks (mirrors `adjust_bias_model_logcounts`).
- **Composition unit test**: for a batch, assert
  `logcount == logsumexp([acc, bias])` and `profile == acc+bias` element-wise.
- **End-to-end small run**: 1 fold, few epochs on CPU/1 GPU to confirm both
  trainers converge and checkpoints/evals write.
- **Benchmark**: run existing `evaluate.py` on the composed CAPY model and on
  ChromBPNet (`evaluate_chrombpnet.py`) over the same `test` split; compare
  count Pearson r and profile JSD (vs. pseudorep bound) via `plot_fig1cd.py` /
  `benchmark_chrombpnet.ipynb`. Target: match Fig 1c/1d (r ≈ 0.70).
