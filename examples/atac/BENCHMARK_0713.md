# CAPY vs. ChromBPNet ATAC Benchmark — first run-through (2026-07-13)

## 1. Scope and status

This is a **record of the first end-to-end test run**, not the finished benchmark.

- **Cell line / fold:** K562, fold_0 **only**.
- **Completed:** Stage 1 (CAPY Tn5-bias model) and Stage 2 (bias-factorized accessibility
  model), each through **Tier A** (prediction metrics) and **Tier B** (attribution + TF-MoDISco).
- **Not done:** Tier C (marginal footprinting), folds 1–4, the other four cell lines, and
  several attribution gaps listed in §9.

Final, all-fold results belong in `RESULTS.md`, which is deliberately still empty. Nothing in
this document should be quoted as a final number.

This doc supersedes the shared/overlapping content of `bias_capy/PLAN.md` and
`bias_factorized_capy/PLAN.md`, which are now thin stage-specific pointers.

---

## 2. What this benchmark is

We want an apples-to-apples comparison in which the **only** variable is the model architecture —
CAPY replacing BPNet — while the data, the folds, and the entire bias-factorized training regimen
are held identical to ChromBPNet.

The key point, easy to miss: **ChromBPNet is not a single model.** It is a two-stage,
bias-factorized pipeline — train a Tn5-bias model on background regions, freeze and depth-scale it,
then train an accessibility model *on top of* the frozen bias. Comparing a single naive model
against it is not a fair test.

Consequently `examples/atac/train_capy.py` (naive single model trained directly on observed
signal) is **out of scope** for this benchmark. It is left untouched.

All ChromBPNet mechanics below were reverse-engineered by reading the container source directly:
`/grid/koo/home/shared/capybara/chrombpnet/chrombpnet_latest.sif` → `/scratch/chrombpnet/chrombpnet/`
(`pipelines.py`, `training/models/{bpnet_model,chrombpnet_with_bias_model}.py`, `training/train.py`,
`training/data_generators/batchgen_generator.py`, `training/{predict,metrics}.py`,
`helpers/hyperparameters/find_{bias,chrombpnet}_hyperparams.py`).

---

## 3. Shared, read-only ChromBPNet assets

Root: `/grid/koo/home/shared/capybara/chrombpnet/`. We **reuse these files bit-for-bit** rather than
re-deriving anything — that is what makes "same data" literally true.

**Genome / splits**
- `genome/hg38.fasta` (+ `.chrom.sizes`), ENCODE blacklist.
- `data/splits/fold_{0..4}.json` — the Zenodo chromosome splits. Fold-0 test chroms: chr1, chr3, chr6.

**Stage-1 inputs** — `models/bias/K562/fold_0/`
- `auxiliary/K562.fold_0_filtered.bias_nonpeaks.bed` — the training regions (non-peaks).
- `auxiliary/K562.fold_0_filtered.bias_peaks.bed` — eval-only peaks.
- `auxiliary/K562.fold_0_data_unstranded.bw` — **the +4/−4 Tn5 BigWig ChromBPNet actually trained on.**
- `logs/K562.fold_0_bias_model_params.tsv` — `counts_loss_weight 5.8`, filters 128, n_dil 4,
  inputlen 2114, outputlen 1000, `max_jitter 0`, `negative_sampling_ratio 1.0`.
- `logs/K562.fold_0_bias_data_params.tsv` — count cutoff `101`, **190,863** training points.
- `evaluation/K562.fold_0_bias_metrics.json` — the reference numbers in §6.

**Stage-2 inputs** — `models/chrombpnet/K562/fold_0/`
- `auxiliary/K562.fold_0_filtered.{peaks,nonpeaks}.bed`, `..._data_unstranded.bw`.
- `logs/K562.fold_0_chrombpnet_model_params.tsv` — `counts_loss_weight 76`, filters 512, n_dil 8,
  `max_jitter 500`, `negative_sampling_ratio 0.1`.
- `logs/K562.fold_0_chrombpnet_data_params.tsv` — `counts_sum_{min,max}_thresh` (the δ-scaling
  in-range bounds), 223,080 training points.
- `evaluation/K562.fold_0_chrombpnet_metrics.json` — the reference numbers in §6.
- `auxiliary/motif_to_pwm.tsv` — the 5 TN5 motif seeds needed for Tier C.

**Reference models:** `K562.fold_0_bias.h5` (2.7 MB, ≈217,730 params, RF ≈ 81 bp),
`K562.fold_0_chrombpnet_nobias.h5` (25.6 MB, ≈6.38 M), `K562.fold_0_chrombpnet.h5` (77.5 MB, composed).

---

## 4. The pipeline, step by step — and what each step *means*

### Stage 0 — Data (already done, by ChromBPNet)
`reads_to_bigwig` shifts the merged BAM by **+4/−4** (the ATAC Tn5 offset) into an unstranded BigWig.
Peaks are ENCODE IDR, blacklist-filtered; non-peaks are GC-matched background from
`chrombpnet prep nonpeaks`. All regions are **summit-centered**.

### Stage 1 — The bias model
**What it is for:** learning *pure Tn5 sequence-insertion bias*, isolated from any TF-driven signal.
Trained on **non-peaks only** — regions where, by construction, there is no accessibility signal to
learn, so anything the model does learn must be enzyme bias.

**Why it must be myopic.** ChromBPNet's bias net is small (128 filters / 4 dilations, ≈218K params)
*and* has a tiny **~81 bp receptive field**. Both properties are deliberate: a model that can only
see 81 bp of local context can learn the Tn5 insertion motif but **cannot** learn distal TF grammar
or enhancer syntax. If it could, the bias model would start absorbing real accessibility signal, and
Stage 2 would then subtract that signal away — corrupting the whole benchmark.

**CAPY's mirror.** CAPY is a pooling U-Net, so an exact 81 bp RF is not reproducible. We approximate
myopia with **2 encoder stages** and a **local `residual_conv` bottleneck** (explicitly *not*
`hybrid_attention`, which is globally connected and would blow the RF wide open). Size is matched to
ChromBPNet: **≈212K params** vs 217.7K. Config: `configs/atac_capy_bias.yaml`.

Hyperparameters are taken from ChromBPNet's TSVs, not re-tuned: `counts_weight 5.8`, `max_jitter 0`,
all-negatives, Adam 1e-3, batch 64, seed 1234, early-stop on val_loss.

**Loss** (both stages): `multinomial_nll(profile)` + `counts_weight · MSE(logcount, log(1+Σcounts))`.
CAPY's existing `profile_mnll_loss` + `count_log1p_mse_loss` (`capybara/losses.py`) already match this.

### The δ bias-scaling step (between the stages)
`adjust_bias_model_logcounts`: over training non-peaks whose counts fall inside
`[min_thresh, max_thresh]`, compute

    δ = mean( log(1 + Σcounts) − bias_predicted_logcount )

and add δ to the **final Linear bias of the count head only** (weights untouched). This is a
**read-depth recalibration** — the bias model was trained on background regions, so its absolute
count scale is wrong for the peak-containing data Stage 2 sees. Implemented in
`bias_factorized_capy/scale_bias.py`.

### Stage 2 — The bias-factorized accessibility model
The frozen, scaled bias model is composed with a fresh full-size accessibility model:

    profile_logits = acc_logits + bias_logits           # add in logit space
    log_counts     = logsumexp([acc_logcount, bias_logcount])   # counts add linearly

The loss is computed on the **composed** output, but **only the accessibility weights update**. The
meaning: the accessibility net is forced to learn *only what the bias model cannot already explain* —
i.e. the residual, genuinely regulatory signal. That is what makes the exported accessibility-only
model (`nobias.pt`, ChromBPNet's `chrombpnet_nobias.h5`) "bias-corrected".

Trained on **peaks + 1:10 GC-matched negatives**, `max_jitter 500`, `counts_weight 76`.
Implemented in `bias_factorized_capy/factorized_model.py` (`BiasFactorizedCAPY`).

### Data generator semantics (mirrors `ChromBPNetBatchGenerator`)
Train: windows loaded at `inputlen + 2·jitter`, then **each epoch** random-cropped (jitter),
reverse-complemented, shuffled, and **negatives re-subsampled** to `ratio × #peaks`. Valid: fixed
seeded negatives, jitter 0, no reverse-complement. CAPY's Stage-2 train set is an `IterableDataset`
to support the per-epoch resampling; valid is map-style.

### The three evaluation tiers — what each one proves
- **Tier A — prediction metrics.** Does the model predict the signal? Counts (Spearman/Pearson/MSE on
  log-counts) + profile (`median_jsd`, `median_norm_jsd`). Includes the **QC gate**: the bias model's
  *peak* counts Pearson must be **> −0.5**. A strongly negative peak correlation would mean the bias
  net latched onto AT/GC composition rather than pure Tn5.
- **Tier B — attribution + TF-MoDISco.** *What did it learn?* Contribution scores over 30K subsampled
  peaks → TF-MoDISco → TOMTOM motif matches. The bias model should show **only TN5/repeat motifs and
  no real TFs**; the bias-corrected model should show **real TF motifs** (CTCF, GATA1::TAL1, SP1, NRF1…).
- **Tier C — marginal footprinting.** *Was the bias actually removed?* Insert TN5 motifs into
  background sequence and measure the corrected model's response; ChromBPNet's is ≤ 0.002.
  **Not yet built.**

### Metric definitions — exact ChromBPNet parity (a real bug we fixed)
CAPY's in-repo `jensen_shannon_distance` returns the JS **divergence**; ChromBPNet uses
`scipy.spatial.distance.jensenshannon`, the JS **distance** = √divergence. An early comparison made
CAPY look *better* on profile JSD purely as a units artifact. The evaluators now port ChromBPNet's
definitions verbatim (`chrombpnet_profile_jsd` in `bias_capy/evaluate_bias.py`, imported by the
Stage-2 evaluator), with `pseudocount = 0.001`, base e:

    true_prob = true / (0.001 + Σtrue);  pred_prob = softmax(logits)
    jsd       = jensenshannon(true_prob, pred_prob)              # lower = better
    max_jsd   = jensenshannon(true_prob, uniform)
    norm_jsd  = clip((jsd − max_jsd) / (0 − max_jsd), 0, 1)      # higher = better

---

## 5. Code map

Stage-2 code lives in `bias_factorized_capy/`, **not** in `bias_capy/` — the old `bias_capy/PLAN.md`
Part-5 table claimed otherwise and was stale.

### `examples/atac/bias_capy/` — Stage 1
| File | Purpose |
|---|---|
| `file_config.py` | `BiasCapyFiles`: resolves the read-only shared inputs (filtered bias beds, BigWig, params TSV, fold split) and the CAPY output dirs. |
| `data.py` | `BiasDataModule` — non-peaks only, split by fold chroms, `max_jitter 0`, no negative subsampling. |
| `train_bias.py` | Stage-1 trainer. |
| `evaluate_bias.py` | Tier-A eval on peaks AND nonpeaks; ChromBPNet metric schema; enforces the peak-Pearson > −0.5 QC gate. Also defines `chrombpnet_profile_jsd`, reused by Stage 2. |
| `submit_bias_fold.sh` | `sbatch submit_bias_fold.sh [timestamp] [cell_type] [fold] [gpu]` — train then evaluate. `BIAS_PARAMS=<config>` selects a sweep config. |
| `submit_bias_eval.sh` | `sbatch submit_bias_eval.sh <timestamp> [cell_type] [fold] [split] [gpu]` — eval only. |

### `examples/atac/bias_factorized_capy/` — Stage 2
| File | Purpose |
|---|---|
| `file_config.py` | `FactorizedCapyFiles`: Stage-2 shared assets + the Stage-1 `chead` bias checkpoint + Stage-2 outputs. |
| `scale_bias.py` | The δ recalibration; writes `bias_scaled.pt`. |
| `factorized_model.py` | `BiasFactorizedCAPY(acc, frozen_scaled_bias)` — add logits, logsumexp counts. Bias kept frozen *and* in `.eval()` so BatchNorm stats never update. |
| `data.py` | `FactorizedDataModule` — train `IterableDataset` (per-epoch negative resample, jitter 500, RC); valid map-style, fixed. |
| `train_factorized.py` | Freeze bias, Adam over accessibility params + ReduceLROnPlateau; saves composed `best.pt` and accessibility-only `nobias.pt`. |
| `evaluate_factorized.py` | Composed model on peaks; ChromBPNet-schema metrics JSON. |
| `submit_factorized_fold.sh` | `sbatch submit_factorized_fold.sh [timestamp] [cell_type] [fold] [bias_timestamp] [gpu]` — `scale_bias` → `train_factorized` → `evaluate_factorized`. |

### `examples/atac/attribution/` — Tier B, shared by both stages
| File | Purpose |
|---|---|
| `wrappers.py` | Scalarizers: **profile** = `get_weightedsum_meannormed_logits`; **counts** = `sum(log_counts)`. Both return `(B,1)`. |
| `contribs.py` | Region subsampling (30K, `random_state=1234`), `deeplift_attributions`, `gradientshap_attributions`, `center_channels` (the Majdandzic correction, §8), and `write_chrombpnet_h5` (schema `raw`/`shap`/`projected_shap`, `(N,4,L)`). |
| `nonlinear_ops.py` | The DeepLIFT `SameMaxPool1d` fix (§7). |
| `attribution_bias.py` | CLI: attribution on the Stage-1 bias model. |
| `attribution_nobias.py` | CLI: attribution on the Stage-2 `nobias` model. Has `--no_gradient_correction`. |
| `run_modisco.sh` | Feeds the `.h5` to the **container's own** `modisco motifs -n 50000 -w 500` + `modisco report`, unchanged — MoDISco is model-agnostic, so this is exact parity by construction. |
| `submit_attribution_{bias,nobias}.sh` | SLURM wrappers; `[timestamp] [method] [cell_type] [fold] …`, `--heads`, `--no_gradient_correction`. |

**Engine choice.** `--method gradientshap` (captum) is the default; `--method deeplift`
(`tangermeme.deep_lift_shap`) is the closest analogue of ChromBPNet's DeepSHAP. For the **bias**
model DeepLIFT is the better choice and is what we use. For the **factorized** model, the
accessibility net's `hybrid_attention` bottleneck makes DeepLIFT unreliable (DeepLIFT's rules are not
sound through self-attention), so **GradientShap is the default there**.

---

## 6. Results from this run — K562, fold_0

### Stage 1 — bias model
An 8-run sweep was done under hard constraints: receptive field unchanged, size held at ≈212K params,
`counts_weight` fixed at 5.8. Runs: `test_1` (baseline), `do05`, `do20` (dropout), `relu`, `ln`
(activation/norm), `chead`, `prof`, `bneck` (fixed-budget capacity redistribution).

**`chead` won** — capacity shifted into the count head (`conv [48,32]`, `mlp [48]`; encoder [46,62],
decoder [62,46,32]). Source: `.../capy_bias/atac/K562/fold0/chead/evals/K562_bias_capy_metrics_test.json`
vs `models/bias/K562/fold_0/evaluation/K562.fold_0_bias_metrics.json`.

| metric (test split) | CAPY `chead` | ChromBPNet bias |
|---|---|---|
| counts pearson — **nonpeaks** | **0.625** | 0.606 |
| counts spearman — nonpeaks | **0.494** | 0.419 |
| counts pearson — **peaks** (want ≈0/negative) | **−0.116** | −0.172 |
| profile `median_jsd` — peaks (lower better) | 0.402 | **0.394** |
| profile `median_norm_jsd` — peaks (higher better) | 0.354 | **0.367** |

**Reading:** CAPY is *better* on the metric that matters for a bias model (nonpeak counts) and
near-identical on profile shape. The negative peak correlation is **correct and desirable** — a bias
model is supposed to fail on peaks. QC gate (peak Pearson > −0.5): **PASS** for all 8 runs.

### Stage 2 — bias-factorized model
One run, `chead_test1` (`.../capy_chrombpnet/atac/K562/fold0/chead_test1/`). Accessibility net:
encoder [128,192,256], `hybrid_attention` bottleneck depth 2, ynet count head; `counts_weight 76`,
jitter 500, neg ratio 0.1, 100 epochs / patience 10 / ReduceLROnPlateau; best epoch 73; ~9.7 h wall.

Evaluated on **peaks, test split, 66,474 regions** — the same region set ChromBPNet scored.

| metric | CAPY factorized | ChromBPNet nobias |
|---|---|---|
| counts pearson | **0.709** | 0.692 |
| counts spearman | **0.614** | 0.597 |
| counts mse | **0.621** | 0.702 |
| profile `median_jsd` | 0.348 | **0.341** |
| profile `median_norm_jsd` | 0.440 | **0.450** |

**Headline: on this fold CAPY slightly beats ChromBPNet on counts (r 0.709 vs 0.692) and is within
~0.007 JSD on profile shape.** This is the core result of the run-through. It is a **single fold** —
treat it as a promising test-run result, not a claim.

### Region-set parity (verified)
Both Tier-A evals score exactly ChromBPNet's regions: 66,474 peaks + 66,474 nonpeaks on the fold-0
test chroms (chr1/chr3/chr6), summit-centered, 2114/1000, jitter 0, no RC. Note the metrics were
**not** computed on the 30K subsample — that set is the Tier-B interpretation set only.

---

## 7. Attribution provenance — reading the results directory names

**The word "corrected" means two different things in the two stages.** This is the single most
confusing thing in the results tree, and the main reason this document exists.

### Bias model — `capy_bias/atac/K562/fold0/chead/attribution/`
Here **"corrected" = DeepLIFT completeness** (the `SameMaxPool1d` fix). It has **nothing to do with
the Majdandzic gradient correction**.

| dir | date | status | what it is |
|---|---|---|---|
| `gradientshap/` | Jul 9 | **stale** | Predates both the `multiply_by_inputs=False` semantics fix and the gradient correction. Do not use. |
| `deeplift_uncorrected_0709/` | Jul 9 | **superseded** | Ran *before* `nonlinear_ops.py` existed. tangermeme keys its max-pool rule on `torch.nn.MaxPool1d`, so it silently treated CAPY's custom `SameMaxPool1d` as **linear**, leaking passthrough gradient mass. Profile convergence delta ≈ **2.2–2.9**; \|shap\| ≈ 3.3× inflated. |
| `deeplift/` | Jul 10 | ✅ **canonical** | Ran *after* commit `5c9896b` (Rescale rule for `SameMaxPool1d` + `track_shared_maxpools`, needed because `SequenceEncoder` reuses one pool instance across resolutions). Profile delta ≈ **0.05–0.38**, a ~20× improvement. **This is the bias attribution to use.** |

The `_uncorrected_0709` directory was renamed **by hand** to preserve the pre-fix output — no
`--uncorrected` flag has ever existed in `attribution_bias.py`. Verified: the two runs used
character-identical command lines and an identical `interpreted_regions.bed` (same md5); only the
DeepLIFT backward rule differs. Neither h5 is zero-sum across channels, confirming no centering in
either.

**Motif sanity:** all bias MoDISco reports are dominated by TN5 hits (TN5_1…TN5_8) — the bias model
learned Tn5, as intended. The superseded `deeplift_uncorrected_0709` counts report shows noticeable
TF contamination (E2F, NRF1, SP/KLF, TFAP2, p53); the corrected `deeplift` counts report is cleaner —
independent evidence that the maxpool fix mattered.

### Factorized model — `capy_chrombpnet/atac/K562/fold0/chead_test1/attribution/`
Here **"uncorrected" *does* mean the Majdandzic gradient correction.**

| dir | date | status | what it is |
|---|---|---|---|
| `gradientshap_0710_buggy/` | Jul 10 | **buggy** | Produced before `multiply_by_inputs=False`: captum multiplied the expected gradient by `(input − baseline)`, so the array stored as "hypothetical" was **not** hypothetical. Its `modisco/counts/reports/` is also incomplete (trimmed logos, no `motifs.html`). |
| `gradientshap_uncorrected/` | Jul 10 | **valid but uncorrected** | Semantics fixed; Majdandzic centering **off**. Profile head only — no counts scores, no counts MoDISco. |
| `gradientshap/` | — | ❌ **MISSING** | The corrected run **was never produced.** See §9 item 1. |

The `gradientshap_uncorrected` profile MoDISco report does recover real TF motifs, which is the
expected qualitative signature of a successfully bias-corrected model.

### Known residual — the counts head is not fully conservation-clean
Even in the good `deeplift/` run, the **counts head keeps a ~0.46 convergence gap**, because
`LayerNorm` is deliberately left as an autograd passthrough: neither captum nor tangermeme registers
LayerNorm, and the generic elementwise Rescale rule is the wrong tool for a cross-dimensional op.
This is standard-tool behavior and was accepted, but it means **the profile head is the clean one**
and bias *counts* attributions should be read with that caveat. GELU, BatchNorm (in eval), and
average-pooling need nothing — they are already registered or linear.

---

## 8. The Majdandzic gradient correction

From Majdandzic, Rajesh & Koo (2023), *Genome Biology* 24:109. Landed in commit `89a474e`.

Gradients of a model taking one-hot input live in the full 4-channel space, but only the **zero-sum
(simplex-tangent) subspace** is meaningful — a constant added to all 4 channels at a position is a
gauge freedom that changes nothing about the model's behavior on valid one-hot input, yet it shows up
as spurious attribution mass. The correction is an orthogonal projection onto that subspace, i.e. a
**subtraction** of the per-position mean across the 4 nucleotide channels:

    corrected[n, :, l] = raw[n, :, l] − raw[n, :, l].mean()        # mean over the 4-channel axis

**Subtraction, not division.** Division is not a projection: it is unstable when the channel mean is
near zero and can flip the sign of the attribution. The projection is applied in float32 (zero-sum
holds to ~1e-6; ~1e-2 after the float16 storage cast).

**It is GradientShap-only, deliberately.** The DeepLIFT path with `hypothetical=True` already strips a
reference-weighted channel-constant gauge; applying the centering there would **double-apply** it.
`test_gradient_correction.py` asserts by introspection that `deeplift_attributions` does not even
accept the flag.

**Interaction with `multiply_by_inputs=False`.** GradientShap is now constructed with
`multiply_by_inputs=False`, so it returns `E[grad]` — a genuinely *hypothetical* score. Previously
captum multiplied by `(input − baseline)` internally, so the array we stored as "hypothetical" was not.
`write_chrombpnet_h5` now forms `projected_shap = onehot * hyp` in exactly one place (which is also
what MoDISco recomputes internally). Because both the `(input − baseline)` multiply and the centering
are linear, correcting the *expected* gradient post-hoc is exactly equivalent to correcting each
sampled gradient before averaging.

The correction defaults to **ON** in `generate_scores`.

---

## 8a. Tier-B comparison — what the models actually learned

Tier A says CAPY predicts the signal as well as ChromBPNet. This section asks the different question:
**did it learn the same things?** It is an initial, qualitative read of the six MoDISco reports —
not a publication figure.

**Why the comparison is valid as-is.** CAPY's contribution scores were fed to the **container's own
`modisco motifs` / `modisco report`, unchanged**, against the **same bundled MEME database**, on the
**same 30K peak subsample** (CAPY's regions are a strict subset of ChromBPNet's — same seed, minus a
handful of ambiguous-base windows; §7). MoDISco is model-agnostic, so both sides' reports are
produced by identical machinery and can be read side by side without any renormalization.

Reports compared (CAPY `deeplift` for bias, `gradientshap_uncorrected` for factorized — the stale and
buggy runs of §7 are excluded):

| report | pos | neg | total seqlets |
|---|---|---|---|
| CAPY bias / profile — `…/chead/attribution/deeplift/modisco/profile/reports/motifs.html` | 15 | 0 | 29,747 |
| ChromBPNet bias / profile — `…/models/bias/K562/fold_0/evaluation/modisco_profile/motifs.html` | 20 | 0 | 30,457 |
| CAPY bias / counts — `…/deeplift/modisco/counts/reports/motifs.html` | 12 | 19 | 25,356 |
| ChromBPNet bias / counts — `…/evaluation/modisco_counts/motifs.html` | 16 | 40 | 21,520 |
| CAPY factorized / profile — `…/gradientshap_uncorrected/modisco/profile/reports/motifs.html` | 22 | 17 | 24,743 |
| ChromBPNet nobias / profile — `…/models/chrombpnet/K562/fold_0/evaluation/modisco_profile/motifs.html` | 31 | 12 | 24,181 |

### Bias / profile — both models learned Tn5, and only Tn5 ✅

This is the **real test of the myopic-bias design**. If the bias model had learned TF motifs, it would
be absorbing genuine accessibility signal, and Stage 2 would then subtract that signal away — which
would quietly corrupt the entire benchmark. It didn't.

| rank | CAPY bias (deeplift) | seqlets | ChromBPNet bias | seqlets |
|---|---|---|---|---|
| 1 | TN5_8 | 10,480 | TN5_1 | 8,478 |
| 2 | TN5_2 | 9,851 | TN5_4 | 5,100 |
| 3 | TN5_1 | 4,823 | TN5_1 | 4,408 |
| 4 | TN5_3 | 2,179 | TN5_3 | 2,662 |
| 5 | TN5_6 | 695 | TN5_2 | 2,541 |
| 6 | TN5_3 | 512 | TN5_1 | 1,879 |
| 7 | TN5_3 | 312 | TN5_3 | 1,045 |

Both sides are **pure Tn5 at the top, with zero negative patterns**. CAPY's top 7 and ChromBPNet's
top 12 are all TN5 variants. The only non-Tn5 hits are low-count stragglers on both sides — CAPY:
ZNF384 (221 seqlets), RREB1 (109); ChromBPNet: PRDM6 (295), EGR2 (200) — all GC-rich zinc-finger
artifacts, not TF grammar. Note CAPY concentrates its seqlets into fewer, larger patterns (15 vs 20
patterns for a similar seqlet total), i.e. a slightly more consolidated Tn5 representation.

**Verdict: parity. The CAPY bias model is a legitimate Tn5-bias model.**

### Bias / counts — the messy head, and the two sides disagree in polarity ⚠️

This is the one comparison that is **not** clean, in both directions.

| | CAPY bias / counts | ChromBPNet bias / counts |
|---|---|---|
| where the signal lives | **positive** patterns (top: 5,878; 3,292) | **negative** patterns (top pos = only **99**; top neg = **2,967**) |
| Tn5 present? | **yes** — TN5_2 (3,292 pos), TN5_6, TN5_1; TN5_2/4/7 in neg | **barely** — one TN5_2 at 86 seqlets, otherwise none |
| dominant motifs | GC-rich SP2/MAZ/KLF + Tn5 | GC/composition artifacts: DNASE_4, ZNF76, ZFX ×3, NFKB2, SP1/2/3, KLF |
| top pattern | 5,878 seqlets, **no TOMTOM match at all** | DNASE_4 (neg, 2,967), q = 1.0 |

Two things are going on. First, the **count head is a single global scalar** (global-average-pool →
dense), so its attributions are inherently diffuse and dominated by overall base composition rather
than by localized motifs — that is why *both* sides degenerate into GC/SP/KLF-type artifacts. Second,
the **sign conventions differ**: ChromBPNet's count attributions are predominantly negative where
CAPY's are positive, so a naive pos-vs-pos reading would be misleading.

Read this together with the **known ~0.46 convergence residual** on CAPY's counts head (LayerNorm left
as an autograd passthrough, §7). **CAPY's counts attributions are the least trustworthy artifact in
the whole set.** Interestingly, CAPY's counts head *does* still recover Tn5 while ChromBPNet's does
not — but given the convergence gap and the polarity mismatch, this is not a result to lean on. The
profile head is the one to trust, and it is the one that shows clean parity.

### Factorized / profile — both models learned the same TF vocabulary ✅

The headline Tier-B result. Top motifs line up almost one-to-one, with only the KLF↔GATA rank order
swapping:

| rank | CAPY factorized | seqlets | q | ChromBPNet nobias | seqlets | q |
|---|---|---|---|---|---|---|
| 1 | **CTCF** | 6,081 | 1.6e-13 | **CTCF** | 6,202 | 6.2e-13 |
| 2 | **GATA3** | 3,370 | 2.6e-01 | KLF12 *(KLF/SP)* | 4,288 | 4.7e-06 |
| 3 | **KLF3** *(KLF/SP)* | 3,304 | 6.4e-06 | **GATA3** | 2,651 | 3.6e-01 |
| 4 | **BACH2** *(NFE2/AP-1)* | 2,321 | 1.3e-04 | **BACH2** | 2,341 | 3.3e-06 |
| 5 | **NFYA** | 1,390 | 4.4e-01 | **NFYB** | 1,585 | 1.9e-05 |
| 6 | **ETV6** *(ETS)* | 808 | 3.7e-05 | ELF1 *(ETS)* | 1,152 | 4.6e-06 |

Below the top 6 both models independently recover **NRF1, YY1, MITF, AP-1 (JUN/FOS), ZBTB33/KAISO,
CTCFL, SP1/2/3** — and CTCF seqlet counts agree to within 2% (6,081 vs 6,202). These are precisely the
regulators one expects in K562 (GATA1/TAL1 erythroid program, NFE2/BACH, CTCF insulators). **The two
models learned the same regulatory grammar, not merely the same numbers.**

**Polarity: no inversion, but CAPY carries more negative mass.** Unlike bias/counts, both models here
are positive-dominated — CAPY **84%** of seqlet mass in pos_patterns, ChromBPNet **94%**. But CAPY's
negative share is ~2.6× larger (**15.9% vs 6.1%**; 17 neg patterns vs 12). Most of that is benign: the
same motif family appearing in both pos and neg is normal in MoDISco (flanking / repressive context),
and ChromBPNet does it too (NRF1 33% neg, SP2 41%, CTCF 2%). CAPY is in the same range for the main
families (CTCF 3% neg, GATA 13%, KLF 5%, NFY 4%). Two are anomalous:

| family | CAPY neg share | ChromBPNet neg share |
|---|---|---|
| ZBT7A | **56%** (649 neg / 511 pos) | absent |
| SP2 | **78%** (153 neg / 42 pos) | 41% |

A majority-negative motif is a soft smell — net-negative contribution assigned to an element the model
also scores positively elsewhere. **The likely cause is the missing gradient correction**: this run is
`gradientshap_uncorrected`, so the scores still carry the channel-constant gauge component that
Majdandzic centering removes (§8), and that gauge is exactly the kind of thing that manufactures
spurious negative attribution mass. This is a **falsifiable prediction** for the corrected run:
negative mass should fall toward ChromBPNet's ~6%, and ZBT7A/SP2 should stop being majority-negative.

Two caveats:
- **CAPY retains one Tn5 remnant**: TN5_6 at 80 seqlets (rank 16 of 22) plus a DNASE_2 hit at 74 —
  a small trace of incomplete bias removal that ChromBPNet's nobias model does not show (it has **no**
  TN5 hits at all). It is tiny (0.3% of seqlets) but it is the one asymmetry, and Tier C (marginal
  footprinting) is the test that would quantify it properly.
- **This uses the *uncorrected* GradientShap run**, since the Majdandzic-corrected one was never
  produced (§9 item 1). The finding is therefore **provisional** — it should be re-checked against
  the corrected scores.

### What CAPY captured — synthesis

Putting the three panels together: the CAPY bias model learned **Tn5 insertion bias and nothing else**,
which is exactly its job and the precondition for the whole factorization being sound. The CAPY
bias-factorized model then learned the **real K562 regulatory vocabulary** — CTCF, GATA, KLF/SP,
NFE2/BACH, NFY, ETS, NRF1, YY1 — matching ChromBPNet motif-for-motif at the top, with a faint residual
Tn5 signature that ChromBPNet lacks. Combined with Tier A (CAPY slightly ahead on counts, level on
profile shape), the picture is that **swapping BPNet → CAPY preserves both the predictive performance
and the learned biology.** The weakest link is not the model but our *counts-head attribution
machinery*, which is diffuse on both sides and additionally convergence-limited on CAPY's.

---

## 9. What's next — open items

1. **Run the corrected GradientShap attribution — QUEUED, both models, both heads.** ← highest
   priority. No corrected run exists for either model yet, so we have no canonical GradientShap
   Tier-B artifact. The commands are prepared (see below); the jobs write to a fresh `gradientshap/`
   dir on each side and touch nothing existing. **§8a is provisional until these land** — re-run the
   comparison against the corrected reports and check the two predictions it makes: (a) factorized
   negative seqlet mass falls from 15.9% toward ChromBPNet's 6.1%, with ZBT7A/SP2 no longer
   majority-negative; (b) whether the residual TN5_6 trace survives.

   ```bash
   cd /grid/koo/home/ykang/elongation/CAPYBARA

   # one-time: preserve the stale Jul-9 bias run so the new one can take the `gradientshap/` name
   mv results/runs/models/chrombpnet_benchmark/capy_bias/atac/K562/fold0/chead/attribution/gradientshap \
      results/runs/models/chrombpnet_benchmark/capy_bias/atac/K562/fold0/chead/attribution/gradientshap_stale_0709

   # bias CAPY — corrected GradientShap, profile + counts
   sbatch examples/atac/attribution/submit_attribution_bias.sh chead gradientshap K562 0 0

   # factorized CAPY (nobias) — corrected GradientShap, profile + counts
   sbatch examples/atac/attribution/submit_attribution_nobias.sh chead_test1 gradientshap K562 0 chead 0 --heads "profile counts"
   ```
   Correction is ON by default; omitting `--no_gradient_correction` is what selects the corrected run.
2. ~~`attribution_bias.py` has no `--no_gradient_correction` flag~~ — **FIXED.** The flag and the
   `_uncorrected` namespacing now mirror `attribution_nobias.py` in both `attribution_bias.py` and
   `submit_attribution_bias.sh`, so a corrected and an uncorrected bias run can no longer collide.
3. **No counts attribution for the factorized model** — being produced by the queued run above. Note
   ChromBPNet itself ran *profile-only* for its `nobias` model (`interpret.args.json`:
   `profile_or_counts: ["profile"]`), so there is **no ChromBPNet counterpart to compare against** on
   the counts head; the factorized counts scores are a CAPY-only artifact, useful for internal
   diagnosis but outside strict ChromBPNet parity.
4. **`gradientshap_0710_buggy/modisco/counts/reports/` is incomplete** — trimmed logos written, no
   `motifs.html`; the report step died partway. Moot if the dir is discarded.
5. **Tier C (marginal footprinting) is unbuilt** for both stages. Inputs are already available:
   `auxiliary/motif_to_pwm.tsv` (5 TN5 seeds). Target: reproduce ChromBPNet's
   `max_bias_response ≤ 0.002` on the corrected model.
6. **Only K562 / fold_0.** Folds 1–4 need the Stage-1 and Stage-2 runs; the other four cell lines need
   the `chrombpnet_setup/` steps re-run first. `RESULTS.md` stays empty until these land.
7. **Stale scripts:** `run_train_capy.sh` and `submit_train_all_folds_capy.sh` hardcode another user's
   checkout (`/grid/koo/home/nagai/projects/capybara`). They belong to the out-of-scope naive
   `train_capy.py` path.
8. **Only one Stage-2 run exists** (`chead_test1`). No accessibility-side architecture sweep has been
   done, and the Stage-2 hyperparameters (100 ep / patience 10 / ReduceLROnPlateau) deviate from
   ChromBPNet's (50 ep / patience 5 / no scheduler) by explicit choice — worth revisiting when
   deciding what the final, defensible configuration is.
