from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara.data import ProfileDataset, extract_loci
from examples.atac.bias_capy.file_config import BiasCapyFiles
from examples.atac.evaluate import compute_standard_metrics, finite_mean, load_model, run_model
from examples.shared.train_utils import read_yaml, require_training_dependencies, select_device

SPLITS = ["train", "valid", "test", "all"]
# ChromBPNet aborts full-model training if the bias peak-counts Pearson is <= this.
PEAK_PEARSON_QC_FLOOR = -0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier-A evaluation of a CAPY bias model (mirrors ChromBPNet bias_metrics.json).")
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--shared_root", type=Path, default=None)
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--reverse_complement", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def split_chroms(files: BiasCapyFiles, split: str) -> list[str]:
    fold_split = files.fold_split()
    if split == "all":
        return sorted({c for chroms in fold_split.values() for c in chroms})
    return list(fold_split[split])


def load_region_set(
    *,
    files: BiasCapyFiles,
    bed_path: Path,
    chroms: list[str],
    params: dict[str, Any],
    batch_size: int,
    num_workers: int,
    reverse_complement: bool,
    device,
    verbose: bool,
    model,
) -> dict[str, np.ndarray]:
    dataset_params = params["dataset"]
    seqs, signals, _ = extract_loci(
        genome_path=files.genome_path,
        chroms=chroms,
        bw_paths=[files.data_bw_path],
        bed_path=bed_path,
        input_length=int(dataset_params["input_length"]),
        output_length=int(dataset_params["output_length"]),
        max_jitter=0,
        summits=True,
        verbose=verbose,
    )
    dataset = ProfileDataset(
        sequences=seqs,
        signals=signals,
        masks=None,
        input_length=int(dataset_params["input_length"]),
        output_length=int(dataset_params["output_length"]),
        max_jitter=0,
        reverse_complement=False,
        random_seed=None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return run_model(model, loader, device, reverse_complement=reverse_complement)


def summarize(results: dict[str, np.ndarray]) -> dict[str, float]:
    metrics = compute_standard_metrics(
        results["true_profiles"], results["pred_log_profiles"], results["pred_log_counts"]
    )
    true_counts = results["true_profiles"].sum(axis=(1, 2))
    valid = true_counts > 0
    jsd = np.ravel(metrics["jsd"])
    return {
        "n": int(results["true_profiles"].shape[0]),
        "counts_pearsonr": finite_mean(metrics["count_pearson"]),
        "counts_spearmanr": finite_mean(metrics["count_spearman"]),
        "counts_mse": finite_mean(metrics["count_mse"]),
        "median_jsd": float(np.nanmedian(jsd[valid])) if np.any(valid) else float("nan"),
    }


def concat_results(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([a[key], b[key]], axis=0) for key in a}


def run(args: argparse.Namespace) -> None:
    require_training_dependencies()
    kwargs: dict[str, Any] = dict(
        proj_dir=args.proj_dir,
        cell_type=args.cell_type,
        fold=args.fold,
        timestamp=args.timestamp,
    )
    if args.shared_root is not None:
        kwargs["shared_root"] = args.shared_root
    files = BiasCapyFiles.create(**kwargs)
    files.validate_inputs()
    if not files.best_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {files.best_checkpoint_path}")
    if not files.params_path.exists():
        raise FileNotFoundError(f"Missing saved params: {files.params_path}")

    params = read_yaml(files.params_path)
    batch_size = int(args.batch_size or params["train"]["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else params["dataloader"]["num_workers"])
    device = select_device(args.device)
    chroms = split_chroms(files, args.split)

    print(f"Loading CAPY bias checkpoint: {files.best_checkpoint_path}", flush=True)
    model = load_model(params, files.best_checkpoint_path, device)

    region_sets = {"peaks": files.peaks_bed_path, "nonpeaks": files.nonpeaks_bed_path}
    raw: dict[str, dict[str, np.ndarray]] = {}
    for name, bed_path in region_sets.items():
        print(f"Scoring {name} on split={args.split} ({len(chroms)} chroms)...", flush=True)
        raw[name] = load_region_set(
            files=files,
            bed_path=bed_path,
            chroms=chroms,
            params=params,
            batch_size=batch_size,
            num_workers=num_workers,
            reverse_complement=args.reverse_complement,
            device=device,
            verbose=args.verbose,
            model=model,
        )

    per_set = {name: summarize(res) for name, res in raw.items()}
    per_set["peaks_and_nonpeaks"] = summarize(concat_results(raw["peaks"], raw["nonpeaks"]))

    metrics = {
        "counts_metrics": {
            name: {
                "spearmanr": stats["counts_spearmanr"],
                "pearsonr": stats["counts_pearsonr"],
                "mse": stats["counts_mse"],
            }
            for name, stats in per_set.items()
        },
        "profile_metrics": {name: {"median_jsd": stats["median_jsd"]} for name, stats in per_set.items()},
        "n_regions": {name: stats["n"] for name, stats in per_set.items()},
        "split": args.split,
        "reverse_complement": bool(args.reverse_complement),
    }

    files.eval_dir.mkdir(parents=True, exist_ok=True)
    rc_suffix = "_rc" if args.reverse_complement else ""
    out_path = files.eval_dir / f"{args.cell_type}_bias_capy_metrics{rc_suffix}_{args.split}.json"
    with out_path.open("w") as handle:
        json.dump(metrics, handle, indent=4)

    peak_pearson = per_set["peaks"]["counts_pearsonr"]
    nonpeak_pearson = per_set["nonpeaks"]["counts_pearsonr"]
    print("\n=== CAPY bias model — Tier-A metrics ===", flush=True)
    print(f"  nonpeaks: counts pearson={nonpeak_pearson:.4f}  median_jsd={per_set['nonpeaks']['median_jsd']:.4f}", flush=True)
    print(f"  peaks:    counts pearson={peak_pearson:.4f}  median_jsd={per_set['peaks']['median_jsd']:.4f}", flush=True)
    print(f"  wrote {out_path}", flush=True)

    if files.ref_metrics_path.exists():
        ref = json.loads(files.ref_metrics_path.read_text())
        ref_peak = ref.get("counts_metrics", {}).get("peaks", {}).get("pearsonr")
        ref_nonpeak = ref.get("counts_metrics", {}).get("nonpeaks", {}).get("pearsonr")
        print(f"  ChromBPNet ref: peaks pearson={ref_peak}  nonpeaks pearson={ref_nonpeak}", flush=True)

    # QC gate (mirror ChromBPNet's assertion before full-model training).
    if peak_pearson <= PEAK_PEARSON_QC_FLOOR:
        raise AssertionError(
            f"Bias QC FAILED: peak counts Pearson {peak_pearson:.4f} <= {PEAK_PEARSON_QC_FLOOR}. "
            "The bias model may be capturing AT/GC composition, not pure Tn5 bias."
        )
    print(f"  QC gate PASSED (peak Pearson {peak_pearson:.4f} > {PEAK_PEARSON_QC_FLOOR}).", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
