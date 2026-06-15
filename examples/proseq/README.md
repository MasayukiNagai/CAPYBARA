# PRO-seq / gene-body workflow

Train and benchmark profile+count models on PRO-seq (Rogers timecourse) gene-body
data. This mirrors `examples/procap` but for **single-strand**, TSS-anchored
gene-body prediction instead of the two-strand PRO-cap setup.

Key differences from `examples/procap`:

| | PRO-cap (`examples/procap`) | PRO-seq (`examples/proseq`) |
|---|---|---|
| Strands | double `(B, 2, L)` | **single** `(B, 1, L)` (transcription strand) |
| Windows | symmetric, peak-centered | TSS-anchored, asymmetric `in=[-2000,4000]`, `out=[0,2000]` |
| Negatives / mask | DNase negatives + unmappability mask | none |
| Signal | integer counts | fractional (normalized BigWig) → relaxed count asserts |
| Splits | `cell_type`, int `fold` 1–7, DNase splits | `treatment`, string `fold` (`fold1`…`fold7`) |

The model, losses, training loop, and ProCapNet builder are **reused unchanged**
from `examples/procap` via `sys.path` (`train_utils.py`, `procapnet.py`); only the
data path (`data.py`), file layout (`proseq_file_config.py`), and metrics
(`performance_metrics.py`, relaxed integer asserts) are PRO-seq-specific. Nothing
under `examples/procap` is modified.

Data splits live under
`/grid/koo/home/ykang/elongation/data_hunting/split_FANTOM5_TSS_rogers_all_times_top_counts_per_gene`
(`ref_all_timepoints_top_count_per_gene_<fold>_{train,val,test}.bed.gz`). The fold
is taken from `dataset.fold` in the config and can be overridden with `--fold`.

## Files

| File | Purpose |
|---|---|
| `data.py` | Single-strand, TSS-anchored, strand-oriented locus extraction + `ProSeqDataModule`. |
| `proseq_file_config.py` | `ProSeqFilesConfig` — input/output paths for a (treatment, fold, model) run. |
| `performance_metrics.py` | PRO-seq copy of the procap metrics (fractional counts; same math). |
| `train_proseq.py` | Train **CAPY** on PRO-seq (joint profile+count, optional count fine-tune). |
| `train_proseq_procapnet.py` | Train **ProCapNet** on PRO-seq (single strand, `n_outputs=1`). |
| `evaluate.py` | Evaluate a trained CAPY **or** ProCapNet run on a split; saves metrics + predictions. |
| `benchmark_proseq.ipynb` | Compare CAPY vs ProCapNet across folds → CSVs + figures. |
| `run_train_proseq.sh` / `run_train_proseq_procapnet.sh` / `run_evaluate_proseq.sh` | SLURM wrappers. |

Configs: `configs/proseq_default.yaml` (CAPY), `configs/proseq_procapnet.yaml`
(ProCapNet), `configs/proseq_smoke.yaml`, `configs/proseq_unet.yaml`.

## 1. Train

```bash
# CAPY (joint train; add --stage both to also count-head fine-tune)
.venv/bin/python examples/proseq/train_proseq.py \
    --proj_dir results/runs --params configs/proseq_default.yaml \
    --treatment GSM8306530_0m --fold fold1 --stage both --device gpu

# ProCapNet baseline, retrained on the same PRO-seq data + task (single strand)
.venv/bin/python examples/proseq/train_proseq_procapnet.py \
    --proj_dir results/runs --params configs/proseq_procapnet.yaml \
    --treatment GSM8306530_0m --fold fold1 --stage both --device gpu
```

`--fold` is optional; if omitted it falls back to `dataset.fold` in the config.
Outputs go to `results/runs/models/<model_name>/proseq/<treatment>/<fold>/<timestamp>/`
(`checkpoints/best.pt`, `params_saved.yaml`, `metrics.tsv`). `--stage both` runs a
count-head fine-tune afterward and writes it under a derived `<timestamp>_ft`.

SLURM: `sbatch examples/proseq/run_train_proseq_procapnet.sh [timestamp] [params] [treatment] [fold] [gpu] [stage]`.

## 2. Evaluate on the test split

The same evaluator scores either model (`--model_name capy|procapnet`). Use the
fine-tuned timestamp if you trained with `--stage both`. Pass the timestamp printed
by training.

```bash
.venv/bin/python examples/proseq/evaluate.py \
    --proj_dir results/runs --model_name procapnet \
    --treatment GSM8306530_0m --fold fold1 --timestamp <ts> \
    --split test --save_predictions
```

Writes to the run's `evals/` directory (prefixed by `<treatment>`):
`*_metrics_summary_test.csv` (mean nll/jsd/profile_pearson/…/count_r2),
`*_metrics_profile_test.csv` (per-example profile metrics), and — with
`--save_predictions` — `*_log_pred_profiles_test.npy`, `*_log_pred_counts_test.npy`,
`*_log_true_counts_test.npy`, and `*_true_profiles_test.npy` (observed profiles, for
the benchmark notebook). Add `--reverse_complement` to average the prediction with
its reverse complement (filenames gain an `_rc` suffix).

SLURM: `sbatch examples/proseq/run_evaluate_proseq.sh <model_name> <timestamp> [treatment] [fold] [split] [gpu] [rc]`.

## 3. Benchmark CAPY vs ProCapNet

Run `evaluate.py … --save_predictions` for both models across all folds, then open
`benchmark_proseq.ipynb`, set the two model `timestamp`s (and `rc_augmented` to match
how you evaluated) in the config cell, and run all cells. It loads the saved per-fold
predictions and writes comparison CSVs + figures to
`examples/proseq/results/benchmark/`:

- `count_metrics_{by_fold,aggregate}.csv`, `profile_metrics_{by_fold,aggregate}.csv`
- `count_scatter_*.pdf` (measured vs predicted reads), `cmp_cdf_profile_jsd.pdf`
  (profile JSD CDF vs an average-profile baseline), `cmp_fold_metrics.pdf`
  (per-fold profile/count metric lines).
