from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara import CAPY
from capybara.data import ProfileDataset, extract_loci
from examples.atac.bias_capy.evaluate_bias import chrombpnet_profile_jsd
from examples.atac.bias_factorized_capy.factorized_model import BiasFactorizedCAPY
from examples.atac.bias_factorized_capy.file_config import FactorizedCapyFiles
from examples.atac.evaluate import compute_standard_metrics, finite_mean, run_model
from examples.shared.train_utils import read_yaml, require_training_dependencies, select_device

SPLITS = ["train", "valid", "test", "all"]
REGION_CHOICES = ["peaks", "nonpeaks", "both"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tier-A evaluation of the composed CAPY model (mirrors ChromBPNet chrombpnet_metrics.json)."
    )
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--shared_root", type=Path, default=None)
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--bias_timestamp", type=str, default="chead")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--regions", choices=REGION_CHOICES, default="peaks",
                        help="ChromBPNet evaluates the composed model on peaks only.")
    parser.add_argument("--reverse_complement", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def make_files(args: argparse.Namespace) -> FactorizedCapyFiles:
    kwargs: dict[str, Any] = dict(
        proj_dir=args.proj_dir,
        cell_type=args.cell_type,
        fold=args.fold,
        timestamp=args.timestamp,
        bias_timestamp=args.bias_timestamp,
    )
    if args.shared_root is not None:
        kwargs["shared_root"] = args.shared_root
    return FactorizedCapyFiles.create(**kwargs)


def split_chroms(files: FactorizedCapyFiles, split: str) -> list[str]:
    fold_split = files.fold_split()
    if split == "all":
        return sorted({c for chroms in fold_split.values() for c in chroms})
    return list(fold_split[split])


def load_composed_model(files: FactorizedCapyFiles, params: dict[str, Any], device) -> BiasFactorizedCAPY:
    accessibility = CAPY(params)
    scaled = torch.load(files.bias_scaled_path, map_location=device)
    bias = CAPY(scaled["params"])
    model = BiasFactorizedCAPY(accessibility, bias)
    checkpoint = torch.load(files.best_checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Composed checkpoint missing model_state_dict: {files.best_checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def score_region_set(
    *,
    files: FactorizedCapyFiles,
    bed_path: Path,
    chroms: list[str],
    params: dict[str, Any],
    batch_size: int,
    num_workers: int,
    reverse_complement: bool,
    device,
    verbose: bool,
    model,
) -> dict[str, float]:
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
    results = run_model(model, loader, device, reverse_complement=reverse_complement)

    metrics = compute_standard_metrics(
        results["true_profiles"], results["pred_log_profiles"], results["pred_log_counts"]
    )
    median_jsd, median_norm_jsd = chrombpnet_profile_jsd(
        results["true_profiles"], results["pred_log_profiles"]
    )
    return {
        "n": int(results["true_profiles"].shape[0]),
        "spearmanr": finite_mean(metrics["count_spearman"]),
        "pearsonr": finite_mean(metrics["count_pearson"]),
        "mse": finite_mean(metrics["count_mse"]),
        "median_jsd": median_jsd,
        "median_norm_jsd": median_norm_jsd,
    }


def main() -> None:
    args = parse_args()
    require_training_dependencies()

    files = make_files(args)
    files.validate_inputs()
    if not files.best_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing composed checkpoint: {files.best_checkpoint_path}")
    if not files.params_path.exists():
        raise FileNotFoundError(f"Missing saved params: {files.params_path}")
    if not files.bias_scaled_path.exists():
        raise FileNotFoundError(f"Missing scaled bias: {files.bias_scaled_path}")

    params = read_yaml(files.params_path)
    batch_size = int(args.batch_size or params["train"]["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else params["dataloader"]["num_workers"])
    device = select_device(args.device)
    chroms = split_chroms(files, args.split)

    print(f"Loading composed CAPY checkpoint: {files.best_checkpoint_path}", flush=True)
    model = load_composed_model(files, params, device)

    region_sets: dict[str, Path] = {}
    if args.regions in ("peaks", "both"):
        region_sets["peaks"] = files.peaks_bed_path
    if args.regions in ("nonpeaks", "both"):
        region_sets["nonpeaks"] = files.nonpeaks_bed_path

    per_set: dict[str, dict[str, float]] = {}
    for name, bed_path in region_sets.items():
        print(f"Scoring {name} on split={args.split} ({len(chroms)} chroms)...", flush=True)
        per_set[name] = score_region_set(
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

    metrics = {
        "counts_metrics": {
            name: {"spearmanr": s["spearmanr"], "pearsonr": s["pearsonr"], "mse": s["mse"]}
            for name, s in per_set.items()
        },
        "profile_metrics": {
            name: {"median_jsd": s["median_jsd"], "median_norm_jsd": s["median_norm_jsd"]}
            for name, s in per_set.items()
        },
        "n_regions": {name: s["n"] for name, s in per_set.items()},
        "split": args.split,
        "reverse_complement": bool(args.reverse_complement),
    }

    files.eval_dir.mkdir(parents=True, exist_ok=True)
    rc_suffix = "_rc" if args.reverse_complement else ""
    out_path = files.eval_dir / f"{args.cell_type}_chrombpnet_capy_metrics{rc_suffix}_{args.split}.json"
    with out_path.open("w") as handle:
        json.dump(metrics, handle, indent=4)

    print("\n=== Composed CAPY — Tier-A metrics (ChromBPNet-comparable) ===", flush=True)
    for name, s in per_set.items():
        print(
            f"  {name:>10}: counts pearson={s['pearsonr']:.4f} spearman={s['spearmanr']:.4f} "
            f"mse={s['mse']:.4f} | median_jsd={s['median_jsd']:.4f} median_norm_jsd={s['median_norm_jsd']:.4f}",
            flush=True,
        )
    print(f"  wrote {out_path}", flush=True)

    if files.ref_metrics_path.exists():
        ref = json.loads(files.ref_metrics_path.read_text())
        print("  --- ChromBPNet reference (same fold, composed model) ---", flush=True)
        for name in per_set:
            rc = ref.get("counts_metrics", {}).get(name, {})
            rp = ref.get("profile_metrics", {}).get(name, {})
            print(
                f"  {name:>10}: counts pearson={rc.get('pearsonr')} spearman={rc.get('spearmanr')} "
                f"mse={rc.get('mse')} | median_jsd={rp.get('median_jsd')} median_norm_jsd={rp.get('median_norm_jsd')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
