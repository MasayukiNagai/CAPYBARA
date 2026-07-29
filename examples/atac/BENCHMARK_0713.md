# CAPY vs. ChromBPNet ATAC Benchmark — first run-through (2026-07-13)

## 1. Scope and status

This is a **record of the first end-to-end test run**, not the finished benchmark.

- **Cell line / fold:** K562, fold_0 **only**.
- **Completed:** Stage 1 (CAPY Tn5-bias model) and Stage 2 (bias-factorized accessibility
  model), each through **Tier A** (prediction metrics) and **Tier B** (attribution + TF-MoDISco).
- **Updated 2026-07-20:** added Stage-2 round-2 accessibility sweep metrics + `cw50` repeats (§6)
  and the `cw50` factorized profile attribution tables (§8a).
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
  **Pipeline built and sanity-checked on K562/fold_0 (not the final eval) — descriptive results in §8b.**

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

⚠️ **Caveat on that selection, added retrospectively.** The sweep ranked single runs on counts metrics
that we now know carry a **±0.033–0.046 run-to-run sd** (see reproducibility below). Most of the
separation it saw between the mid-field runs was therefore **within noise**, and `chead`'s margin over
the runners-up is not established. Only the *large* counts movements (e.g. a run dropping to ~0.50)
and the *profile* differences (sd ±0.0005) are interpretable at n=1. `chead` remains the anchor — it is
what every Stage-2 run is built on, and nothing here suggests it is a *bad* choice — but it should not
be described as a demonstrated winner over the other 212K-class configs. Re-ranking the top candidates
with repeats would be needed to make that claim.

Test split; region counts identical on both sides: **66,474 peaks + 66,474 nonpeaks = 132,948**.

#### How to read these two subsets — they are not the same kind of measurement

This is the single easiest thing to get wrong about the bias model, so it is stated before the numbers:

- **Nonpeaks are the task.** The bias model is trained *only* here. These are the numbers on which
  "is CAPY's bias model as good as ChromBPNet's?" is decided. Better = better.
- **Peaks are a leakage reference, not a scoreboard.** The model never trains here, and by design
  **must not** be able to predict them. **Scoring poorly on peaks is the desired outcome.** A model that
  predicted peak *counts* well would have learned accessibility, not Tn5 — the failure mode §4 exists to
  prevent, which would make Stage 2 subtract real signal away. So peak metrics are read as
  "confirmed myopic?", and a CAPY-vs-ChromBPNet gap on peaks is **not** a performance difference.

The one metric that would expose leakage is a **positive peak counts correlation**. Note that the
`peaks_and_nonpeaks` rows mix the task with the leakage reference and therefore have no clean
interpretation for the bias model; they are retained only for continuity with ChromBPNet's schema.

#### Run-to-run reproducibility (4 runs)

`chead` was rerun 3× (`chead_rep{1,2,3}`, `params_saved.yaml` byte-identical to `chead`, same seed
1234), so the spread below is pure nondeterminism — cuDNN kernel selection, dataloader worker order,
GPU atomics. All four early-stopped normally and landed within 0.2 on best valid loss (159.9–160.1).
**This is the error bar for every Stage-1 number in this document**, and the two metric families do not
behave alike:

| family | sd across 4 runs | implication |
|---|---|---|
| **profile** (any subset) | **±0.0005 – 0.0010** | differences ≥ ~0.002 are real |
| **counts** (correlations) | **±0.033 – 0.046** | differences < ~0.05 are **not** interpretable |

**`chead_rep1` is an outlier on counts only** (nonpeak pearson 0.559 vs 0.62–0.63 for the other three)
while sitting inside the pack on all six profile metrics and on valid loss. Whatever occasionally goes
wrong is **localized to the count head**, not the shared trunk — a trunk in a worse basin would have
moved profile too. Three runs cluster tightly and one sits well below: not Gaussian scatter, more like
an occasional bad basin for the count head.

#### (1) Nonpeaks — the actual task

| metric (test, nonpeaks) | CAPY mean ± sd (n=4) | `chead` | ChromBPNet bias | verdict |
|---|---|---|---|---|
| counts pearson *(higher better)* | 0.608 ± 0.033 | 0.625 | 0.606 | **tie** |
| counts spearman *(higher better)* | **0.486 ± 0.046** | 0.494 | 0.419 | **CAPY** (3/4 runs clear) |
| counts mse *(lower better)* | 1.383 ± 0.076 | 1.365 | **1.349** | tie (< 0.5 sd) |
| `median_jsd` *(lower better)* | 0.5808 ± 0.0005 | 0.5801 | **0.5769** | ChromBPNet (7.6 sd) |
| `median_norm_jsd` *(higher better)* | 0.2034 ± 0.0007 | 0.2043 | **0.2092** | ChromBPNet (8.6 sd) |

**Counts: a tie, with a real CAPY edge on rank.** The earlier claim in this document — "CAPY wins on
nonpeak counts pearson, 0.625 vs 0.606" — **does not survive four runs**: the mean is 0.608 vs 0.606.
That claim rested on a single run, and ChromBPNet is itself n=1, so its own envelope is unknown.
**Spearman is the defensible claim** (0.486 vs 0.419; even `rep1`, the bad run, only *ties* ChromBPNet
at 0.420 rather than losing). MSE is a tie once the ±0.076 spread is admitted — and MSE penalizes
**absolute count scale** while the correlations measure only ranking, so a calibration gap here is
expected anyway: this is the raw bias model, *before* the δ read-depth recalibration Stage 2 applies
(§4).

**Profile: the one genuine CAPY deficiency, and it is small.** 0.0039 JSD, but 7.6 sd — real, not
noise. Its practical weight is limited: nonpeak profiles are low-count and near-flat, so **both** models
score far worse here (JSD ≈ 0.58) than on peaks (≈ 0.40) — expected, not alarming. There is little
shape to predict and the multinomial is noise-dominated, which is why **on nonpeaks it is the counts
metric that carries the signal**.

#### (2) Peaks — leakage reference (want: no accessibility information)

| metric (test, peaks) | CAPY mean ± sd (n=4) | `chead` | ChromBPNet bias |
|---|---|---|---|
| counts pearson *(want ≈ 0; **positive = leakage**)* | −0.119 ± 0.042 | −0.116 | −0.172 |
| counts spearman *(want ≈ 0)* | −0.057 ± 0.040 | −0.058 | −0.119 |
| counts mse *(want high)* | 10.16 ± 0.22 | 10.196 | 10.104 |
| `median_jsd` *(want poor, i.e. **high**)* | 0.4025 ± 0.0005 | 0.4019 | 0.3940 |
| `median_norm_jsd` *(want **low**)* | 0.3533 ± 0.0010 | 0.3544 | 0.3670 |

**Verdict: no leakage, in either model.** Every run's peak counts correlation is negative — never
positive — and the QC gate (peak pearson > −0.5) **PASSES for all 8 sweep runs and all 4 repeats**.
Peak MSE is ~10 against ~1.4 on nonpeaks.

The **myopia signature** is essentially identical on both sides:

| | counts r: nonpeak → peak | drop | mse: nonpeak → peak |
|---|---|---|---|
| CAPY `chead` | +0.625 → −0.116 | 0.74 | 1.37 → 10.20 (**7.5×** worse) |
| ChromBPNet bias | +0.606 → −0.172 | 0.78 | 1.35 → 10.10 (**7.5×** worse) |

**CAPY's bias model fails on peaks in the same way and to the same degree as ChromBPNet's.** That is
the parity result that matters in this section.

Two traps in this table, both of which earlier drafts fell into:

**CAPY's *worse* peak profile is not a deficit.** CAPY predicts peak profiles less well than ChromBPNet
(JSD 0.4025 vs 0.3940, `norm_jsd` 0.353 vs 0.367). Under the framing above that means CAPY's bias model
is, if anything, **slightly more myopic** — mildly favourable. Do not report it as an axis on which CAPY
"trails". Note also that peak profile JSD is a *weak* leakage probe in the first place: Tn5 inserts with
the same sequence preference inside peaks as outside, so a bias model is **expected** to explain part of
the peak profile shape legitimately. Peak **counts** is the sharp probe — counts are accessibility
magnitude, which the bias model must not know.

**CAPY's peak counts being closer to zero is not a win either.** CAPY's four runs span −0.067 to −0.168,
an envelope that nearly contains ChromBPNet's −0.172. Within noise: a tie.

The strongest leakage evidence is not in this table at all — it is Tier B (§8a): CAPY's bias model
recovers **only TN5 motifs and no real TFs**.

### Stage 2 — bias-factorized model
One run, `chead_test1` (`.../capy_chrombpnet/atac/K562/fold0/chead_test1/`). Accessibility net:
encoder [128,192,256], `hybrid_attention` bottleneck depth 2, ynet count head; `counts_weight 76`,
jitter 500, neg ratio 0.1, 100 epochs / patience 10 / ReduceLROnPlateau; best epoch 73; ~9.7 h wall.

Evaluated on **peaks only, test split, 66,474 regions** — the same region set ChromBPNet scored. (No
nonpeaks block here: ChromBPNet's Stage-2 evaluation scores peaks only, and we mirror it.)

#### Counts metrics

| metric (test split, peaks) | CAPY factorized | ChromBPNet nobias |
|---|---|---|
| pearson *(higher better)* | **0.709** | 0.692 |
| spearman *(higher better)* | **0.614** | 0.597 |
| mse *(lower better)* | **0.621** | 0.702 |

#### Profile metrics

| metric (test split, peaks) | CAPY factorized | ChromBPNet nobias |
|---|---|---|
| `median_jsd` *(lower better)* | 0.348 | **0.341** |
| `median_norm_jsd` *(higher better)* | 0.440 | **0.450** |

**Headline: on this fold CAPY slightly beats ChromBPNet on counts (r 0.709 vs 0.692) and is within
~0.007 JSD on profile shape.** This is the core result of the run-through. It is a **single fold** —
treat it as a promising test-run result, not a claim.

### Stage 2 — round-1 accessibility sweep

The first accessibility-side sweep: 3 configs against the `chead_test1` anchor, all on the same frozen
`chead` bias, K562/fold0. All four **early-stopped cleanly** (each last epoch = best + 10 = patience;
none reached the 100-epoch cap). Configs: `configs/atac_search/acc_{cw50,pool8,bigattn}.yaml`. Source:
`.../capy_chrombpnet/atac/K562/fold0/{chead_test1,cw50,pool8,bigattn}/evals/K562_chrombpnet_capy_metrics_test.json`.

Test split, **peaks only, 66,474 regions** — the same region set as the tables above.

| run | `median_jsd` *(lower better)* | `median_norm_jsd` *(higher better)* | counts pearson *(higher)* | counts spearman *(higher)* | counts mse *(lower)* | params | best ep | wall |
|---|---|---|---|---|---|---|---|---|
| ChromBPNet nobias | 0.341 | 0.450 | 0.692 | 0.597 | 0.702 | 6.38M | — | — |
| `chead_test1` *(parity anchor)* | 0.348 | 0.440 | 0.709 | 0.614 | 0.621 | 5.37M | 73 | 9.7 h |
| **`cw50`** | **0.339** | **0.454** | **0.727** | **0.649** | **0.592** | 5.37M | 57 | 8.0 h |
| `pool8` | 0.346 | 0.444 | 0.726 | 0.642 | 0.609 | 5.25M | 38 | 7.3 h |
| `bigattn` | 0.361 | 0.420 | 0.631 | 0.487 | 0.830 | 7.34M | 81 | 11.6 h |

**`cw50` wins on all five metrics** against both the anchor and ChromBPNet — the first Stage-2 config
to lead ChromBPNet on **profile** (`median_jsd` 0.339 vs 0.341; `median_norm_jsd` crosses above 0.450).

**`counts_weight` is not a profile↔counts trade — 76 was simply mis-set.** The sweep was designed on
the assumption that lowering `counts_weight` would *spend* CAPY's counts lead to buy profile. It did
not: counts improved *too* (pearson 0.709 → 0.727), and the **validation count loss itself fell**
(0.622 → 0.569) **despite being weighted less**. So `counts_weight 76` sits *past* the optimum, not at
it — over-weighting the count term was degrading the shared trunk and costing *both* heads.
ChromBPNet's TSV `counts_loss_weight` is tuned for BPNet's architecture; **parity on that number is not
free for CAPY.** This is a slope, not a floor — the turning point is still unlocated below 50.

**`bigattn` refutes the capacity hypothesis.** Growing the bottleneck 5.37M → 7.34M made *everything*
worse, counts badly (0.709 → 0.631), while burning the most compute. Extra global-mixing capacity is
actively harmful here. Caveat: it moved **three knobs at once** (depth 2→3, heads 4→8, mlp_ratio 2→4),
so *which* one did the damage is unattributed.

**`pool8` supports the resolution hypothesis, weakly.** Profile moved the right way but only
0.348 → 0.346. The notable part is what *didn't* happen: unlike Stage-1's `pool4`, counts did **not**
crash — they rose to 0.726, because the `hybrid_attention` bottleneck keeps the receptive field global
even with 2 encoder stages. Cheapest run of the four.

**Caveat.** Single run, single fold, no seed replicates. `pool8`'s ~0.002 profile gain is plausibly
noise; `cw50`'s margins are several times larger and consistent across all five metrics.

**Parity bookkeeping — two kinds of number.** `chead_test1` remains the **strict-parity anchor**: every
data hyperparameter comes from ChromBPNet's TSV. `cw50` and its descendants are **deliberate
non-parity, CAPY-optimized** runs — they depart from the TSV `counts_loss_weight` and must never be
quoted as the same kind of result as the anchor. The mechanism is opt-in: `train.counts_weight_override`
in the YAML (`train_factorized.py`), which prints `[non-parity] counts_weight override -> …` at startup.
Absent that key the TSV value is used. `max_jitter` and `negative_sampling_ratio` stay TSV-forced in
all runs.

### Stage-2 — round-2 accessibility sweep

Round 2: 6 configs on the same frozen `chead` bias, K562/fold0, plus 2 `cw50` repeats for a Stage-2
error bar. Configs: `configs/atac_search/acc_{cw25,cw35,cw50_mlp1,cw50_depth1,cw50_pool8,cw50_pool8_mlp1}.yaml`.
Same schedule as round 1 (100 ep / patience 10 / ReduceLROnPlateau). Source:
`.../capy_chrombpnet/atac/K562/fold0/{cw25,cw35,cw50_mlp1,cw50_depth1,cw50_pool8,cw50_pool8_mlp1,cw50_rep1,cw50_rep2}/evals/K562_chrombpnet_capy_metrics_test.json`.
Test split, peaks only, 66,474 regions.

Each run moves one knob vs `cw50`:
- `cw25`, `cw35` — `counts_weight` 50 → 25 / 35 (loss lever).
- `cw50_mlp1` — bottleneck `mlp_ratio` 2 → 1.
- `cw50_depth1` — bottleneck attention `depth` 2 → 1.
- `cw50_pool8` — 3 → 2 encoder stages (bottleneck downsampling 16× → 8×).
- `cw50_pool8_mlp1` — pool8 + mlp1 stacked.

| run | lever vs cw50 | `median_jsd` *(↓)* | `median_norm_jsd` *(↑)* | counts pearson *(↑)* | counts spearman *(↑)* | counts mse *(↓)* | params | best ep | wall |
|---|---|---|---|---|---|---|---|---|---|
| ChromBPNet nobias | — | 0.341 | 0.450 | 0.692 | 0.597 | 0.702 | 6.38M | — | — |
| `chead_test1` *(parity anchor, cw76)* | — | 0.348 | 0.440 | 0.709 | 0.614 | 0.621 | 5.38M | 73 | 9.7 h |
| `cw50` *(round-1 best)* | — | 0.339 | 0.454 | 0.727 | 0.649 | 0.592 | 5.38M | 57 | 8.0 h |
| `cw35` | cw 50→35 | 0.339 | 0.453 | 0.725 | 0.645 | 0.618 | 5.38M | 98 | 11.3 h |
| `cw25` | cw 50→25 | 0.339 | 0.454 | 0.721 | 0.640 | 0.662 | 5.38M | 99 | 11.2 h |
| `cw50_mlp1` | mlp_ratio 2→1 | 0.339 | 0.453 | 0.733 | 0.648 | 0.574 | 5.12M | 47 | 6.3 h |
| `cw50_depth1` | attn depth 2→1 | 0.342 | 0.450 | 0.728 | 0.649 | 0.551 | 4.20M | 40 | 5.2 h |
| `cw50_pool8` | 3→2 enc stages | 0.350 | 0.437 | 0.708 | 0.619 | 0.680 | 5.26M | 87 | 14.5 h |
| `cw50_pool8_mlp1` | pool8 + mlp1 | 0.340 | 0.453 | 0.712 | 0.635 | 0.688 | 4.96M | 97 | 14.2 h |

Stage-2 run-to-run reproducibility — `cw50` config run 3× (`cw50`, `cw50_rep1`, `cw50_rep2`; same
config/seed, spread is nondeterminism):

| metric | cw50 | rep1 | rep2 | mean ± sd (n=3) |
|---|---|---|---|---|
| `median_jsd` *(↓)* | 0.339 | 0.338 | 0.346 | 0.341 ± 0.004 |
| `median_norm_jsd` *(↑)* | 0.454 | 0.455 | 0.443 | 0.451 ± 0.006 |
| counts pearson *(↑)* | 0.727 | 0.718 | 0.709 | 0.718 ± 0.009 |
| counts spearman *(↑)* | 0.649 | 0.641 | 0.622 | 0.637 ± 0.014 |
| counts mse *(↓)* | 0.592 | 0.665 | 0.662 | 0.639 ± 0.041 |

Factual notes: `cw25`/`cw35` did not early-stop (best epoch 98–99); `cw50_mlp1` and `cw50_depth1`
early-stopped and are the cheapest runs; `cw50_depth1` is the smallest model (4.20M). The counts-pearson
spread across `cw25`–`cw50_mlp1` (0.721–0.733) sits within the ±0.009 repeat sd.

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
| `gradientshap_uncorrected/` | Jul 10 | **superseded** | Semantics fixed; Majdandzic centering **off**. Profile head only. Kept only for the before/after negative-mass comparison in §8a. |
| `gradientshap/` | Jul 14 | ✅ **canonical** | The corrected run, **both heads**. Correction verified at the array level: the `shap/seq` (hypothetical) dataset is exactly zero-sum across the 4 channels (per-position channel-mean abs = 0.0, max 3.9e-4 = float16 rounding), vs the uncorrected run's non-zero mean (max 0.93). **This is the factorized attribution to use.** |

The corrected `gradientshap` profile MoDISco report recovers the real TF vocabulary and matches
ChromBPNet motif-for-motif — see §8a.

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

Reports compared (CAPY `deeplift` for bias, the **corrected** `gradientshap` for factorized — the
stale, buggy, and uncorrected runs of §7 are excluded):

| report | pos | neg | total seqlets |
|---|---|---|---|
| CAPY bias / profile — `…/chead/attribution/deeplift/modisco/profile/reports/motifs.html` | 15 | 0 | 29,747 |
| ChromBPNet bias / profile — `…/models/bias/K562/fold_0/evaluation/modisco_profile/motifs.html` | 20 | 0 | 30,457 |
| CAPY bias / counts — `…/deeplift/modisco/counts/reports/motifs.html` | 12 | 19 | 25,356 |
| ChromBPNet bias / counts — `…/evaluation/modisco_counts/motifs.html` | 16 | 40 | 21,520 |
| CAPY factorized / profile — `…/gradientshap/modisco/profile/reports/motifs.html` | 25 | 29 | 25,485 |
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

The headline Tier-B result, now on the **corrected** `gradientshap` run. Top motifs line up almost
one-to-one, with only the KLF↔GATA rank order swapping:

| rank | CAPY factorized (corrected) | seqlets | q | ChromBPNet nobias | seqlets | q |
|---|---|---|---|---|---|---|
| 1 | **CTCF** | 5,910 | 1.5e-15 | **CTCF** | 6,202 | 6.2e-13 |
| 2 | **KLF12/KLF3** *(KLF/SP)* | 3,424 | 4.3e-05 | **KLF12** *(KLF/SP)* | 4,288 | 4.7e-06 |
| 3 | **GATA3** | 3,125 | 1.4e-01 | **GATA3** | 2,651 | 3.6e-01 |
| 4 | **BACH2/NFE2** | 2,112 | 3.8e-05 | **BACH2** | 2,341 | 3.3e-06 |
| 5 | **NFYA** | 1,319 | 3.6e-01 | **NFYB** | 1,585 | 1.9e-05 |
| 6 | **NRF1** | 590 | 7.9e-06 | ELF1 *(ETS)* | 1,152 | 4.6e-06 |

Below the top 6 both models independently recover **ETS (GABPA/ELK/ETV6/ELF), ATF4/CEBP, USF1/MITF,
ZNF143/ZNF76, NRF1, YY1, ZBTB33/KAISO, CTCFL, SP1/2/3, AP-1 (JUN/FOS)** — and the CTCF top-pattern
seqlet count agrees to within ~5% (5,910 vs 6,202). These are precisely the regulators one expects in
K562 (GATA1/TAL1 erythroid program, NFE2/BACH, CTCF insulators). **The two models learned the same
regulatory grammar, not merely the same numbers.** The correction was verified applied at the array
level (§7): the hypothetical scores are exactly zero-sum across channels.

**The two §8a predictions the corrected run tested — one held, one was falsified.**

**(b) Residual Tn5 — resolved, gone ✅.** The uncorrected run carried a confident `TN5_6` remnant
(`pos_patterns.pattern_15`, 80 seqlets, q = 8.3e-03). In the corrected report there is **no `TN5_*`
hit anywhere**. The only bias-ish trace left is `DNASE_2`, and only as an insignificant secondary/
tertiary TOMTOM match (q ≥ 0.05, best 0.047 in a neg pattern) — and ChromBPNet's *own* nobias report
carries the same `DNASE_2` trace (`pos_patterns.pattern_22`, match1, q = 0.044). So it is shared, not
CAPY-specific leakage. **The gradient correction cleaned up the last apparent Tn5 signature** — this
is the clear win of the corrected run. (Tier C marginal footprinting is still the test that would
*quantify* residual bias.)

**(a) Negative mass — falsified ✗.** The prediction was that centering would drop CAPY's negative
seqlet mass toward ChromBPNet's ~6%. It moved the **other way**:

| run | pos mass | neg mass | patterns |
|---|---|---|---|
| CAPY uncorrected | 84.1% | 15.9% | 22 + 17 |
| **CAPY corrected** | 79.0% | **21.0%** | 25 + 29 |
| ChromBPNet nobias | 93.9% | **6.1%** | 31 + 12 |

`ZBT7A` and `SP2` are still majority-negative — `ZBT7A` is now *entirely* negative (`neg_patterns.
pattern_0`, 928 seqlets; no positive ZBT7A pattern at all). So CAPY's excess negative attribution mass
is **not** the channel-constant gauge artifact §8a hypothesized: the Majdandzic projection is
orthogonal to it and if anything shifted mass slightly more negative. This is a **real, standing
property of the CAPY scores**, not a tooling artifact — and the one axis on which corrected CAPY still
diverges from ChromBPNet (which stays 94% positive). Most of the shared families remain benignly
split as before (same motif in pos and neg is normal MoDISco flanking/repressive context, and
ChromBPNet does it too), but the overall polarity gap is genuine and unexplained. Tier C would tell
us whether it reflects real over-subtraction of the composed bias or merely a GradientShap idiom.

### Factorized / profile — round-2 `cw50` run (recorded 2026-07-20)

Data only; the analysis above is for the `chead_test1` (default, cw76) run. `cw50`'s corrected
GradientShap profile report: `…/cw50/attribution/gradientshap/modisco/profile/reports/motifs.html`
(same 30K peak subsample, container `modisco`, same bundled MEME db).

| report | pos patterns | neg patterns | pos seqlets | neg seqlets | neg mass |
|---|---|---|---|---|---|
| `cw50` | 27 | 5 | 15,848 | 344 | 2.1% |
| `chead_test1` *(default)* | 25 | 29 | 20,129 | 5,356 | 21.0% |
| ChromBPNet nobias | 31 | 12 | 22,713 | 1,468 | 6.1% |

Top profile motifs — match0 TOMTOM hit *(seqlets, qval0)*:

| rank | `cw50` | `chead_test1` | ChromBPNet nobias |
|---|---|---|---|
| 1 | CTCF *(5,379, q 1.2e-13)* | CTCF *(5,910, q 1.5e-15)* | CTCF *(6,202, q 6.2e-13)* |
| 2 | KLF3 *(2,430, q 1.3e-05)* | KLF12 *(3,424, q 4.3e-05)* | KLF12 *(4,288, q 4.7e-06)* |
| 3 | GATA3 *(1,896, q 3.2e-01)* | GATA3 *(3,125, q 1.4e-01)* | GATA3 *(2,651, q 3.6e-01)* |
| 4 | BACH2 *(1,369, q 1.2e-04)* | BACH2 *(2,112, q 3.8e-05)* | BACH2 *(2,341, q 3.3e-06)* |
| 5 | NFYA *(1,058, q 4.4e-01)* | NFYA *(1,319, q 3.6e-01)* | NFYB *(1,585, q 1.9e-05)* |
| 6 | ELF2/ETS *(537, q 3.6e-06)* | NRF1 *(590, q 7.9e-06)* | ELF1/ETS *(1,152, q 4.6e-06)* |
| 7 | MITF *(451, q 6.0e-05)* | GABPA/ETS *(544, q 7.5e-06)* | MITF *(500, q 1.3e-05)* |
| 8 | JDP2/AP-1 *(436, q 9.2e-06)* | JDP2/AP-1 *(532, q 1.6e-05)* | SP1 *(480, q 1.6e-04)* |
| 9 | ATF4 *(317, q 2.3e-06)* | ATF4 *(452, q 1.6e-04)* | CEBPG *(448, q 2.4e-07)* |
| 10 | NRF1 *(312, q 1.4e-07)* | ZIC1 *(444, q 1.0e-02)* | FOSL2+JUN/AP-1 *(435, q 6.5e-05)* |

Factual notes: `cw50` recovers the same TF families as ChromBPNet (CTCF, KLF/SP, GATA3, BACH2, NFY,
ETS, MITF, AP-1, ATF4, NRF1, YY1, ZNF143, NFIC, ZBTB33/KAISO, CTCFL); no `TN5_*` and no significant
`DNASE_*` patterns. `cw50`'s negative mass is 2.1% (5 small neg patterns, largest 117 seqlets:
ZEB1/SP2/MXI1), vs the default run's 21.0% (§8a above) and ChromBPNet's 6.1%. `cw50` clusters fewer
total pos seqlets (15,848 vs 20,129 / 22,713).

### What CAPY captured — synthesis

Putting the three panels together: the CAPY bias model learned **Tn5 insertion bias and nothing else**,
which is exactly its job and the precondition for the whole factorization being sound. The CAPY
bias-factorized model then learned the **real K562 regulatory vocabulary** — CTCF, GATA, KLF/SP,
NFE2/BACH, NFY, ETS, NRF1, YY1 — matching ChromBPNet motif-for-motif at the top. On the **corrected**
GradientShap run the last faint Tn5 remnant that the uncorrected run showed is **gone**, so the
factorized model now looks as bias-clean as ChromBPNet's on the motif axis. Combined with Tier A —
Stage 1 a **tie** on the nonpeak counts task with matched myopia on peaks, Stage 2 **ahead on counts**
and, once `counts_weight` is tuned off ChromBPNet's TSV value, **ahead on profile too** (§6) — the
picture is that **swapping BPNet → CAPY preserves both the predictive performance and the learned
biology.** The one residual difference is
polarity — CAPY still assigns ~21% of profile seqlet mass to negative patterns vs ChromBPNet's ~6%,
and the gradient correction did **not** close that gap (it slightly widened it), so it is a real
property of the CAPY scores rather than a gauge artifact. Tier C (marginal footprinting) is the test
that would settle whether it matters. The other soft spot remains the *counts-head attribution
machinery*, diffuse on both sides and additionally convergence-limited on CAPY's.

---

## 8b. Tier C — marginal footprinting (pipeline sanity check, K562/fold_0)

**Status: this is a pipeline / sanity-check run, not the final Tier-C evaluation.** The purpose here
is to confirm the footprinting pipeline is correct and its numbers are sensible; the section is
descriptive only and draws no verdict about CAPY vs ChromBPNet or about §8a.

**What the evaluation is.** Marginal footprinting asks *was the Tn5 bias actually removed?* A Tn5
enzyme motif is inserted at the center of background (non-peak) sequence, the model predicts, and we
read the shape of the predicted profile response. A bias-corrected (nobias) model should give a
**flat** footprint (no Tn5 preference left); a model that still carries Tn5 bias produces a footprint.
Two numbers make the readouts interpretable: (i) **uniform baseline = 1/1000 = 0.001** — a perfectly
flat length-1000 footprint puts 0.001 at every base, so the gate (max < 0.003) is only ~3× this
floor; (ii) **profile-shape-only caveat** — each sequence's footprint is normalized to sum 1 before
averaging, so the `(exp(logcounts)−1)` factor cancels within a sequence. Tier C therefore reflects
**profile-head shape**, not counts-head debiasing, and should not be cited as evidence about the
counts head.

**How both did it.** We mirror ChromBPNet's `marginal_footprinting.py`: per-base footprint =
`softmax(profile_logits) · (exp(logcounts) − 1)`, computed on the inserted sequence **and** its
reverse complement, summed, normalized per sequence to sum 1, averaged over all background sequences
→ a length-1000 curve per motif. The per-motif scalar is `round(max(curve), 3)`; the gate is
`all(round(max,3) < 0.003)` (rounds first, then strict `<`), labeling the run `corrected` or
`uncorrected`. CAPY's two-headed PyTorch model (`profile_logits (B,1,1000)`, `log_counts (B,1)`)
maps directly onto ChromBPNet's Keras port; the CAPY script is
`bias_factorized_capy/marginal_footprint.py`. **Background set:** both sides use the *same input BED*,
`K562.fold_0_filtered.nonpeaks.bed` (66,474 non-peaks on the test chroms). CAPY's summary JSON records
that it scored all 66,474 windows (`edge_skipped=0`, 10 windows contain an `N`, frac 1.5e-4).
ChromBPNet's per-run scored count (N / edge-skips / N-windows) is **not** recorded in the artifacts we
have, so we note the input BED is identical but do **not** assert scored-window parity on the CBP side.

**Models footprinted.** (1) the bias-corrected `nobias.pt` — the real test, expected flat; (2) the
frozen scaled bias branch — a CAPY-internal positive control expected to show a strong Tn5 footprint.
ChromBPNet did not footprint its own bias model, so the bias-branch numbers have no ChromBPNet
counterpart. Both K562/fold_0 CAPY runs (`chead_test1`, `cw50`) share the same frozen `chead` bias
branch, so their bias-branch numbers are identical.

**nobias — `max_bias_response` line (both runs):** `corrected_0.001_0.001/0.001/0.001/0.001/0.001`
(all five TN5 rounded maxima 0.001; passes the gate at both 0.003 and 0.002). ChromBPNet's own line:
`corrected_0.001_0.002/0.001/0.001/0.001/0.002`.

Per-motif detail, nobias, over the same background BED. CAPY `raw_max`, `ratio_vs_control`,
`center_minus_control_center` are read directly from each run's `_footprint_summary.json`; the
ChromBPNet columns are read from `K562.fold_0_chrombpnet_nobias_footprints.h5` (`raw_max`, and
`ratio` computed against CBP's own control max 0.001017 from that same file):

| motif | CAPY `cw50` max (ratio; Δctr) | CAPY `chead_test1` max (ratio; Δctr) | ChromBPNet max (ratio) | rounded cw50 / chead / CBP |
|---|---|---|---|---|
| control | 0.001042 (1.000) | 0.001035 (1.000) | 0.001017 (1.000) | 0.001 / 0.001 / 0.001 |
| tn5_1 | 0.001312 (1.258; +0.000294) | 0.001220 (1.179; +0.000114) | 0.001552 (1.526) | 0.001 / 0.001 / 0.002 |
| tn5_2 | 0.001306 (1.253; +0.000119) | 0.001248 (1.206; +0.000094) | 0.001439 (1.415) | 0.001 / 0.001 / 0.001 |
| tn5_3 | 0.001275 (1.223; +0.000057) | 0.001209 (1.168; +0.000140) | 0.001499 (1.474) | 0.001 / 0.001 / 0.001 |
| tn5_4 | 0.001402 (1.345; +0.000287) | 0.001271 (1.228; +0.000164) | 0.001421 (1.397) | 0.001 / 0.001 / 0.001 |
| tn5_5 | 0.001355 (1.300; +0.000263) | 0.001245 (1.203; +0.000146) | 0.001518 (1.493) | 0.001 / 0.001 / 0.002 |

`center_minus_control_center` (nobias footprint value at the exact insertion center minus the control's
center value) is **positive and small for all five motifs in both runs** (+0.00006 … +0.00029), i.e.
no central dip below control was observed in this run. This is reported as a raw observation only —
**§8a stays open**; the footprint shape is not used here to adjudicate over-subtraction.

**bias branch — positive control (identical across both runs, same frozen `chead`):**
`uncorrected_0.063_0.042/0.062/0.056/0.082/0.071`. TN5 raw maxima 0.042–0.082 (ratio-vs-control
40–77×, from the `_footprint_summary.json`), a sharp feature at the insertion center (argmax 495–501).
This confirms the insertion / loader / prediction path is working — a Tn5-carrying model does produce
a large footprint through this code.

**Still open for the real evaluation (not done here):** bootstrap confidence intervals on the
per-motif maxima; and confirming whether the δ-scaling adjusts the profile logits or the counts head —
which must be settled before Tier C is used to speak to §8a.

---

## 9. What's next — open items

1. **Corrected GradientShap attribution — DONE for both models, both heads.** Both corrected
   `gradientshap/` dirs landed (factorized Jul 14, bias Jul 15). Correction verified applied
   (zero-sum hypothetical scores, §7). **The factorized / profile comparison in §8a has been
   re-done against the corrected report** and its two predictions resolved: (a) negative-mass drop
   **falsified** — it rose 15.9% → 21.0%, ZBT7A/SP2 still majority-negative; (b) residual Tn5
   **gone** — no `TN5_*` hit survives. Remaining follow-up: the **bias / profile** and **bias /
   counts** panels of §8a still cite the `deeplift` run — fold the bias corrected `gradientshap` in
   and confirm the bias model's Tn5-only signature is unchanged under GradientShap. The reproduce
   commands (kept for the record):

   ```bash
   cd /grid/koo/home/ykang/elongation/CAPYBARA
   # bias CAPY — corrected GradientShap, profile + counts
   sbatch examples/atac/attribution/submit_attribution_bias.sh chead gradientshap K562 0 0
   # factorized CAPY (nobias) — corrected GradientShap, profile + counts
   sbatch examples/atac/attribution/submit_attribution_nobias.sh chead_test1 gradientshap K562 0 chead 0 --heads "profile counts"
   ```
   Correction is ON by default; omitting `--no_gradient_correction` is what selects the corrected run.
2. ~~`attribution_bias.py` has no `--no_gradient_correction` flag~~ — **FIXED.** The flag and the
   `_uncorrected` namespacing now mirror `attribution_nobias.py` in both `attribution_bias.py` and
   `submit_attribution_bias.sh`, so a corrected and an uncorrected bias run can no longer collide.
3. **Factorized counts attribution — now produced** (corrected `gradientshap/modisco/counts/`, 32 pos
   + 25 neg patterns; top calls ZBT7A/SP1/CTCF/NRF1/GATA3). Note ChromBPNet itself ran *profile-only*
   for its `nobias` model (`interpret.args.json`: `profile_or_counts: ["profile"]`), so there is
   **no ChromBPNet counterpart to compare against** on the counts head; the factorized counts scores
   are a CAPY-only artifact, useful for internal diagnosis but outside strict ChromBPNet parity.
4. **`gradientshap_0710_buggy/modisco/counts/reports/` is incomplete** — trimmed logos written, no
   `motifs.html`; the report step died partway. Moot if the dir is discarded.
5. **Tier C (marginal footprinting) — pipeline built + sanity-checked, real eval pending.** The
   script (`bias_factorized_capy/marginal_footprint.py`) and wrapper are built and were run on
   K562/fold_0 (`chead_test1`, `cw50`); descriptive results are in §8b (nobias `corrected`, bias
   branch `uncorrected` positive control). This was a correctness/sanity check, **not** the final
   evaluation. Still to do for the real run: bootstrap CIs on the per-motif maxima; confirm whether
   the δ-scaling adjusts the profile logits or the counts head before Tier C is used to speak to §8a;
   then extend across folds/cell lines. Target remains ChromBPNet's `max_bias_response ≤ 0.002` on
   the corrected model.
6. **Only K562 / fold_0.** Folds 1–4 need the Stage-1 and Stage-2 runs; the other four cell lines need
   the `chrombpnet_setup/` steps re-run first. `RESULTS.md` stays empty until these land.
7. **Stale scripts:** `run_train_capy.sh` and `submit_train_all_folds_capy.sh` hardcode another user's
   checkout (`/grid/koo/home/nagai/projects/capybara`). They belong to the out-of-scope naive
   `train_capy.py` path.
8. **Stage-2 sweep — rounds 1 and 2 done.** Round 1: `chead_test1`, `cw50`, `pool8`, `bigattn`.
   Round 2: `cw25`, `cw35`, `cw50_mlp1`, `cw50_depth1`, `cw50_pool8`, `cw50_pool8_mlp1`, plus 2 `cw50`
   repeats — metrics in §6, `cw50` profile attribution in §8a. Summary of round 2: `cw50_mlp1` (counts
   pearson 0.733, 5.12M) and `cw50_depth1` (mse 0.551, smallest at 4.20M) match/beat `cw50` while
   costing less; `cw25`/`cw35` are flat vs `cw50`; `cw50_pool8` regressed.

   Still open: the Stage-2 **training schedule** (100 ep / patience 10 / ReduceLROnPlateau) deviates
   from ChromBPNet's (50 ep / patience 5 / no scheduler) by explicit choice — worth revisiting when
   deciding what the final, defensible configuration is. Note this interacts with the
   `counts_weight` finding above: the more CAPY-specific tuning the Stage-2 model carries, the more
   carefully the parity anchor and the optimized runs have to be reported separately.

9. **Run-to-run noise is measured for Stage 1 and unmeasured for Stage 2 — and it is large on counts.**
   The 4 `chead` runs (§6) give **profile sd ±0.0005, counts-correlation sd ±0.033–0.046**. Two
   consequences:

   - **Every single-run counts comparison in this document is weaker than it reads**, including the
     `chead` sweep selection (§6 caveat) and every "CAPY vs ChromBPNet" counts claim — ChromBPNet's
     numbers are n=1 too, so its envelope is entirely unknown. Only ChromBPNet repeats would fix that,
     which we cannot produce without retraining their model.
   - **Stage-2 error bar now measured** (was: "none"). `cw50` run 3× (§6) gives profile `median_jsd`
     sd ±0.004, `median_norm_jsd` sd ±0.006, counts pearson sd ±0.009, counts spearman sd ±0.014,
     counts mse sd ±0.041. So `cw50`'s counts-pearson lead over the anchor (0.709 → 0.727) is ~2 sd,
     and the round-2 counts-pearson spread (0.721 → 0.733) sits within ~1 sd.

10. **All Stage-1 counts metrics carry an occasional bad-basin risk in the count head.** `chead_rep1`
    (§6) trained to a normal valid loss and normal profile metrics but generalized markedly worse on
    counts (nonpeak pearson 0.559 vs 0.62–0.63). Since `valid_loss` is the selection metric, this
    failure mode is **invisible to early stopping** — a run can be selected as "best" while its count
    head sits in a poor basin. Worth watching if a Stage-2 run ever posts anomalous counts.
