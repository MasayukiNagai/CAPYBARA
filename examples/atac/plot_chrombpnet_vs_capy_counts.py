"""Predicted-vs-measured read scatter for ChromBPNet and CAPY on the same regions.

Both models write the same h5 layout (ChromBPNet's ``*_predictions.h5``: coords +
softmax profs + ``logcounts`` in natural-log ``log(1 + counts)`` space), so one
loader serves both. Neither file stores observed signal, so the measured axis is
recomputed from the ATAC BigWig at the saved summit coordinates, exactly as
``chrombpnet/training/utils/data_utils.get_cts`` and ``capybara.data.extract_loci``
do: sum over ``[center - output_length//2, center + output_length//2)``.

Example (K562 fold 0):

    python examples/atac/plot_chrombpnet_vs_capy_counts.py \
      --pred_h5 "ChromBPNet=/grid/koo/home/shared/capybara/chrombpnet/models/chrombpnet/K562/fold_0/evaluation/K562.fold_0_chrombpnet_predictions.h5" \
      --pred_h5 "CAPY=/grid/koo/home/ykang/elongation/CAPYBARA/results/runs/models/chrombpnet_benchmark/capy_chrombpnet/atac/K562/fold0/cw50/evals/K562_chrombpnet_capy_predictions_test.h5" \
      --data_bw /grid/koo/home/shared/capybara/chrombpnet/models/chrombpnet/K562/fold_0/auxiliary/K562.fold_0_data_unstranded.bw \
      --out_prefix /grid/koo/home/ykang/elongation/CAPYBARA/results/runs/models/chrombpnet_benchmark/capy_chrombpnet/atac/K562/fold0/cw50/evals/K562_chrombpnet_vs_capy_count_scatter_test

The figure is written next to the evaluation outputs of the CAPY model being plotted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara.metrics import count_corr_mse_r2

# Plot controls, matching examples/procap/benchmark_procap.ipynb.
PANEL_SIZE = 2.8
DPI = 300
POINT_SIZE = 1
POINT_ALPHA = 0.14
POINT_COLOR = "#1565C0"
PLOT_FLOOR = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pred_h5",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Panel label and prediction h5. Repeat once per model, in panel order.",
    )
    parser.add_argument("--data_bw", type=Path, required=True, help="Observed unstranded ATAC BigWig.")
    parser.add_argument("--out_prefix", type=Path, required=True, help="Output path without extension.")
    parser.add_argument("--output_length", type=int, default=1000)
    parser.add_argument(
        "--regions",
        choices=["peaks", "nonpeaks", "all"],
        default="peaks",
        help="Which coords_peak flag to keep.",
    )
    return parser.parse_args()


def parse_pred_specs(specs: list[str]) -> list[tuple[str, Path]]:
    out = []
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"--pred_h5 expects LABEL=PATH, got: {spec!r}")
        out.append((label, Path(path)))
    return out


def load_predictions(path: Path, regions: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        chrom = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in handle["coords/coords_chrom"][:]])
        center = handle["coords/coords_center"][:].astype(np.int64)
        peak = handle["coords/coords_peak"][:].astype(np.int64)
        logcounts = handle["predictions/logcounts"][:].astype(np.float64)

    if regions == "peaks":
        keep = peak == 1
    elif regions == "nonpeaks":
        keep = peak == 0
    else:
        keep = np.ones(peak.shape, dtype=bool)
    if not keep.any():
        raise SystemExit(f"No regions with regions={regions!r} in {path}")

    return {"chrom": chrom[keep], "center": center[keep], "logcounts": logcounts[keep]}


def measured_counts(bw_path: Path, chrom: np.ndarray, center: np.ndarray, output_length: int) -> np.ndarray:
    """Total observed reads per region, matching extract_loci / get_cts exactly."""
    import pybigtools

    half = int(output_length) // 2
    handle = pybigtools.open(str(bw_path))
    try:
        totals = np.empty(center.shape[0], dtype=np.float64)
        for i, (c, m) in enumerate(zip(chrom, center)):
            values = handle.values(str(c), int(m) - half, int(m) + half, fillna=0)
            totals[i] = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0).sum()
    finally:
        close = getattr(handle, "close", None)
        if close is not None:
            close()
    return totals


def count_metrics(log_true: np.ndarray, log_pred: np.ndarray) -> dict[str, float]:
    """Pearson/Spearman/MSE/R2 in the shared log(1 + counts) space."""
    true_2d = log_true.reshape(-1, 1)
    pred_2d = log_pred.reshape(-1, 1)
    pearson, spearman, mse, r2 = count_corr_mse_r2(true_2d, pred_2d)
    return {
        "pearson": float(np.ravel(pearson)[0]),
        "spearman": float(np.ravel(spearman)[0]),
        "mse": float(np.ravel(mse)[0]),
        "r2": float(np.ravel(r2)[0]),
        "n": int(log_true.size),
    }


def reads_for_plot(measured: np.ndarray, log_pred: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.exp(np.asarray(log_pred, dtype=float))
    measured = np.where(measured > 0, measured, floor)
    predicted = np.where(predicted > 0, predicted, floor)
    finite = np.isfinite(measured) & np.isfinite(predicted)
    return measured[finite], predicted[finite]


def count_limits(pairs: list[tuple[np.ndarray, np.ndarray]], floor: float) -> tuple[float, float]:
    values = np.concatenate([arr for pair in pairs for arr in reads_for_plot(*pair, floor=floor)])
    return max(float(np.nanmin(values)) * 0.8, floor), float(np.nanmax(values)) * 1.2


def annotate_metrics(ax, metrics: dict[str, float], fontsize: float = 6.5) -> None:
    text = (
        f"Pearson $r$ = {metrics['pearson']:.3f}\n"
        f"Spearman $\\rho$ = {metrics['spearman']:.3f}\n"
        f"R2 = {metrics['r2']:.3f}\n"
        f"N = {metrics['n']:,}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", ha="left", fontsize=fontsize, linespacing=1.12)


def style_count_ax(ax, lim: tuple[float, float], title: str, show_ylabel: bool) -> None:
    ax.plot(lim, lim, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Measured Reads", fontsize=7)
    if show_ylabel:
        ax.set_ylabel("Predicted Reads", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, labelsize=6)


def main() -> None:
    args = parse_args()
    specs = parse_pred_specs(args.pred_h5)

    loaded = [(label, load_predictions(path, args.regions)) for label, path in specs]
    reference_label, reference = loaded[0]
    for label, preds in loaded[1:]:
        if not np.array_equal(preds["chrom"], reference["chrom"]) or not np.array_equal(
            preds["center"], reference["center"]
        ):
            raise SystemExit(
                f"Region mismatch: {label} coords differ from {reference_label}. "
                "The panels would not share an x-axis; refusing to plot."
            )

    print(f"Loading measured reads for {reference['center'].size:,} regions from {args.data_bw}", flush=True)
    measured = measured_counts(args.data_bw, reference["chrom"], reference["center"], args.output_length)
    log_true = np.log(measured + 1)

    metrics = {}
    for label, preds in loaded:
        metrics[label] = count_metrics(log_true, preds["logcounts"])
        m = metrics[label]
        print(
            f"  {label:>12}: pearson={m['pearson']:.4f} spearman={m['spearman']:.4f} "
            f"mse={m['mse']:.4f} r2={m['r2']:.4f} n={m['n']}",
            flush=True,
        )

    lim = count_limits([(measured, preds["logcounts"]) for _, preds in loaded], floor=PLOT_FLOOR)
    fig, axes = plt.subplots(
        1,
        len(loaded),
        figsize=(PANEL_SIZE * len(loaded), PANEL_SIZE),
        dpi=DPI,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, (label, preds) in enumerate(loaded):
        ax = axes[0, i]
        x, y = reads_for_plot(measured, preds["logcounts"], floor=PLOT_FLOOR)
        ax.scatter(x, y, alpha=POINT_ALPHA, s=POINT_SIZE, color=POINT_COLOR, linewidths=0)
        style_count_ax(ax, lim, label, i == 0)
        annotate_metrics(ax, metrics[label])

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_path = args.out_prefix.with_suffix(".png")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    print(f"saved: {out_path}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
