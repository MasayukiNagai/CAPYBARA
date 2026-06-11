from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Fig 1c/1d plots from ATAC evaluation outputs.")
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--cell_type", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--rc_suffix", type=str, default="", help="Empty string or '_rc'.")
    return parser.parse_args()


def _load(eval_dir: Path, name: str) -> np.ndarray:
    return np.load(eval_dir / name)


def _load_metric_column(path, column):
    values = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get(column, "nan")
            values.append(float(raw) if raw != "" else float("nan"))
    return np.asarray(values, dtype=np.float64)



def plot_fig1c(
    log_true_counts: np.ndarray,
    log_pred_counts: np.ndarray,
    *,
    cell_type: str,
    fold: int,
    out_path: Path,
) -> None:
    r, _ = pearsonr(log_true_counts, log_pred_counts)

    fig, ax = plt.subplots(figsize=(6, 6))
    h = ax.hist2d(
        log_true_counts,
        log_pred_counts,
        bins=100,
        cmap="Blues",
        norm=matplotlib.colors.LogNorm(),
    )
    fig.colorbar(h[3], ax=ax, label="Count")
    ax.set_xlabel("Observed log counts")
    ax.set_ylabel("Predicted log counts")
    ax.set_title(f"{cell_type} fold{fold} — Pearson r = {r:.3f}")
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=300, transparent=True)
    plt.close(fig)
    print(f"Saved Fig 1c: {out_path}")


def plot_fig1d(
    jsd_model: np.ndarray,
    jsd_shuffled: np.ndarray,
    jsd_mean: np.ndarray,
    jsd_pseudorep: np.ndarray,
    *,
    cell_type: str,
    fold: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 1, 101)

    data = [
        (jsd_model, "blue", "Predicted vs Observed"),
        (jsd_shuffled, "black", "Shuffled Observed vs Observed"),
        (jsd_mean, "green", "Mean Observed vs Observed"),
        (jsd_pseudorep, "red", "Pseudoreplicate upper bound"),
    ]
    for arr, color, label_prefix in data:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        median = float(np.median(finite))
        ax.hist(finite, bins=bins, color=color, alpha=0.5, label=f"{label_prefix} (median={median:.3f})")
        ax.axvline(median, color=color, linestyle="--", linewidth=1.2)

    ax.set_xlabel("Jensen-Shannon Distance")
    ax.set_ylabel("Number of peaks")
    ax.set_title(f"{cell_type} fold{fold} — JSD Distribution")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=300, transparent=True)
    plt.close(fig)
    print(f"Saved Fig 1d: {out_path}")


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir
    ct = args.cell_type
    fold = args.fold
    split = args.split
    suf = args.rc_suffix

    # Fig 1c: counts scatter
    true_counts_path = eval_dir / f"{ct}_log_true_counts{suf}_{split}.npy"
    pred_counts_path = eval_dir / f"{ct}_log_pred_counts{suf}_{split}.npy"
    if true_counts_path.exists() and pred_counts_path.exists():
        log_true = _load(eval_dir, f"{ct}_log_true_counts{suf}_{split}.npy")
        log_pred = _load(eval_dir, f"{ct}_log_pred_counts{suf}_{split}.npy")
        plot_fig1c(
            log_true, log_pred,
            cell_type=ct,
            fold=fold,
            out_path=eval_dir / f"fig1c_{ct}_fold{fold}{suf}.pdf",
        )
    else:
        print(f"Skipping Fig 1c: count .npy files not found in {eval_dir}")

    # Fig 1d: JSD distributions
    profile_path = eval_dir / f"{ct}_metrics_profile{suf}_{split}.csv"
    if profile_path.exists():
        plot_fig1d(
            _load_metric_column(profile_path, "jsd"),
            _load_metric_column(profile_path, "jsd_shuffled"),
            _load_metric_column(profile_path, "jsd_mean"),
            _load_metric_column(profile_path, "jsd_pseudorep"),
            cell_type=ct,
            fold=fold,
            out_path=eval_dir / f"fig1d_{ct}_fold{fold}{suf}.pdf",
        )
    else:
        print(f"Skipping Fig 1d: metrics profile CSV not found: {profile_path}")


if __name__ == "__main__":
    main()
