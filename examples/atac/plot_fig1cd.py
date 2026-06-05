from __future__ import annotations

import argparse
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
    jsd_pred: np.ndarray,
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
        (jsd_pred, "blue", f"Predicted vs Observed (median={np.nanmedian(jsd_pred):.3f})"),
        (jsd_shuffled, "black", f"Shuffled Observed vs Observed (median={np.nanmedian(jsd_shuffled):.3f})"),
        (jsd_mean, "green", f"Mean Observed vs Observed (median={np.nanmedian(jsd_mean):.3f})"),
        (jsd_pseudorep, "red", f"Pseudoreplicate upper bound (median={np.nanmedian(jsd_pseudorep):.3f})"),
    ]
    for arr, color, label in data:
        ax.hist(arr[np.isfinite(arr)], bins=bins, color=color, alpha=0.5, label=label)
        ax.axvline(np.nanmedian(arr), color=color, linestyle="--", linewidth=1.2)

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
    jsd_keys = ["jsd_pred", "jsd_shuffled", "jsd_mean", "jsd_pseudorep"]
    jsd_paths = {k: eval_dir / f"{ct}_{k}{suf}_{split}.npy" for k in jsd_keys}
    if all(p.exists() for p in jsd_paths.values()):
        plot_fig1d(
            _load(eval_dir, f"{ct}_jsd_pred{suf}_{split}.npy"),
            _load(eval_dir, f"{ct}_jsd_shuffled{suf}_{split}.npy"),
            _load(eval_dir, f"{ct}_jsd_mean{suf}_{split}.npy"),
            _load(eval_dir, f"{ct}_jsd_pseudorep{suf}_{split}.npy"),
            cell_type=ct,
            fold=fold,
            out_path=eval_dir / f"fig1d_{ct}_fold{fold}{suf}.pdf",
        )
    else:
        missing = [str(p) for p in jsd_paths.values() if not p.exists()]
        print(f"Skipping Fig 1d: missing JSD arrays: {missing}")


if __name__ == "__main__":
    main()
