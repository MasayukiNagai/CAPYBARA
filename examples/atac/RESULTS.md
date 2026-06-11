# ATAC Results Report

## Status

The ATAC benchmark code is set up for fold-matched CAPY and original
ChromBPNet comparisons, but this report should be filled after the five
official ChromBPNet folds have been regenerated and evaluated.

## Benchmark Setup

The benchmark uses official zero-based ChromBPNet folds `0..4` on the held-out
test split. Per-fold outputs are expected under:

```text
models/{capy,chrombpnet}/atac/{cell_type}/fold{fold}/{timestamp}/evals/
```

Each run should provide:

- `{cell_type}_metrics_summary[_rc]_test.csv`
- `{cell_type}_metrics_profile[_rc]_test.csv`
- optional `{cell_type}_log_pred_profiles[_rc]_test.npy`
- optional `{cell_type}_log_pred_counts[_rc]_test.npy`
- optional `{cell_type}_log_true_counts[_rc]_test.npy`

## Metrics

Report fold mean +/- sample standard deviation across the five test folds for:

| Category | Metrics |
| --- | --- |
| Profile | Profile JSD, Profile Pearson, Profile Spearman |
| Count | Count R2, Count Pearson, Count Spearman |

Fig 1c-style count scatter and Fig 1d-style JSD distributions should be
reported in two forms:

- fold-specific comparison using fold 0, matching the upstream figure setup
- aggregate comparison concatenating folds 0, 1, 2, 3, and 4

If pseudoreplicate BigWigs are unavailable, `jsd_pseudorep` is expected to be
`NaN` and the pseudoreplicate curve should be omitted.
