from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara import CAPY
from capybara.data import ProfileDataset, extract_loci, load_chrom_names
from examples.atac.file_config import AtacFoldFilesConfig
from capybara.metrics import compute_performance_metrics
from examples.shared.train_utils import read_yaml, require_training_dependencies, select_device


PROFILE_METRIC_COLUMNS = ["nll", "cross_ent", "jsd", "profile_pearson", "profile_spearman", "profile_mse"]
SUMMARY_METRIC_COLUMNS = [*PROFILE_METRIC_COLUMNS, "count_pearson", "count_spearman", "count_mse", "count_r2"]
SPLITS = ["train", "valid", "val", "test", "all"]
JSD_BASELINE_COLUMNS = ["jsd_shuffled", "jsd_mean", "jsd_pseudorep"]

# Smoothing kernel matching ChromBPNet's pseudoreplicate JSD computation
_SMOOTH_SIGMA = 7
_SMOOTH_TRUNCATE = (81 - 1) / (2 * _SMOOTH_SIGMA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CAPY ATAC-seq model.")
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--reverse_complement", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def canonical_split(split: str) -> str:
    return "valid" if split == "val" else split


def split_peak_path(files: AtacFoldFilesConfig, split: str) -> Path:
    split = canonical_split(split)
    return {
        "valid": files.valid_peak_path,
        "train": files.train_peak_path,
        "val": files.val_peak_path,
        "test": files.test_peak_path,
        "all": files.all_peak_path,
    }[split]


def load_model(params: dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = CAPY(params)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def make_eval_loader(
    *,
    files: AtacFoldFilesConfig,
    params: dict[str, Any],
    split: str,
    batch_size: int,
    num_workers: int,
    verbose: bool,
) -> DataLoader:
    dataset_params = params["dataset"]
    peak_path = split_peak_path(files, split)
    chroms = load_chrom_names(files.chrom_size_path)
    seqs, signals, _ = extract_loci(
        genome_path=files.genome_path,
        chroms=chroms,
        bw_paths=[files.atac_bw_path],
        bed_path=peak_path,
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
        random_seed=dataset_params.get("seed"),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)


def _log_softmax_profiles(logits: torch.Tensor) -> torch.Tensor:
    flat = logits.reshape(logits.shape[0], -1)
    return torch.nn.functional.log_softmax(flat, dim=-1).reshape_as(logits)


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    reverse_complement: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, log_counts = model(x)
    log_probs = _log_softmax_profiles(logits)
    if not reverse_complement:
        return log_probs, log_counts

    rc_x = torch.flip(x, dims=(1, 2))
    rc_logits, rc_log_counts = model(rc_x)
    rc_logits = torch.flip(rc_logits, dims=(1, 2))
    rc_log_probs = _log_softmax_profiles(rc_logits)

    merged_probs = 0.5 * (torch.exp(log_probs) + torch.exp(rc_log_probs))
    merged_log_probs = torch.log(torch.clamp(merged_probs, min=torch.finfo(merged_probs.dtype).tiny))
    merged_log_counts = 0.5 * (log_counts + rc_log_counts)
    return merged_log_probs, merged_log_counts


@torch.no_grad()
def run_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    reverse_complement: bool,
) -> dict[str, np.ndarray]:
    true_profiles, pred_log_profiles, pred_log_counts = [], [], []
    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)
        log_probs, log_counts = predict_batch(model, x, reverse_complement=reverse_complement)
        true_profiles.append(batch["y"].detach().cpu().numpy())
        pred_log_profiles.append(log_probs.detach().cpu().numpy())
        pred_log_counts.append(log_counts.detach().cpu().numpy())

    if not true_profiles:
        raise RuntimeError("Evaluation dataloader produced no batches.")

    return {
        "true_profiles": np.concatenate(true_profiles, axis=0),
        "pred_log_profiles": np.concatenate(pred_log_profiles, axis=0),
        "pred_log_counts": np.concatenate(pred_log_counts, axis=0),
    }


def _jsd_per_peak(obs: np.ndarray, pred: np.ndarray, pseudocount: float = 1e-3) -> np.ndarray:
    """Jensen-Shannon distance between obs and pred for each peak. Both (N, L) arrays."""
    from scipy.spatial.distance import jensenshannon

    jsds = np.empty(obs.shape[0])
    for i in range(obs.shape[0]):
        p = obs[i] + pseudocount
        q = pred[i] + pseudocount
        jsds[i] = jensenshannon(p / p.sum(), q / q.sum())
    return jsds


def compute_jsd_arrays(
    true_profiles: np.ndarray,
    pred_log_profiles: np.ndarray,
    rep1_profiles: np.ndarray | None = None,
    rep2_profiles: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute the four JSD distributions needed for Fig 1d.

    All inputs have shape (N, 1, L). Returns per-peak JSD arrays of shape (N,).
    """
    # Squeeze channel dim; shape becomes (N, L)
    obs = true_profiles[:, 0, :]
    if rep1_profiles is not None and rep2_profiles is not None:
        r1 = rep1_profiles[:, 0, :]
        r2 = rep2_profiles[:, 0, :]
        r1_smooth = scipy.ndimage.gaussian_filter1d(r1.astype(np.float64), _SMOOTH_SIGMA, axis=-1, truncate=_SMOOTH_TRUNCATE)
        r2_smooth = scipy.ndimage.gaussian_filter1d(r2.astype(np.float64), _SMOOTH_SIGMA, axis=-1, truncate=_SMOOTH_TRUNCATE)
        jsd_pseudorep = _jsd_per_peak(r1_smooth, r2_smooth)
    else:
        jsd_pseudorep = np.full(obs.shape[0], np.nan, dtype=np.float64)
    mean_profile = obs.mean(axis=0, keepdims=True).repeat(obs.shape[0], axis=0)

    rng = np.random.default_rng(seed=0)
    shuffled = np.array([rng.permutation(obs[i]) for i in range(obs.shape[0])])

    return {
        "jsd_shuffled": _jsd_per_peak(obs, shuffled),
        "jsd_mean": _jsd_per_peak(obs, mean_profile),
        "jsd_pseudorep": jsd_pseudorep,
    }


def load_replicate_profiles(
    files: AtacFoldFilesConfig,
    peak_path: Path,
    output_length: int,
    chroms: list[str],
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Load rep1 and rep2 signal at the same peaks used for evaluation."""
    _, rep1, _ = extract_loci(
        genome_path=files.genome_path,
        chroms=chroms,
        bw_paths=[files.rep1_bw_path],
        bed_path=peak_path,
        input_length=output_length,
        output_length=output_length,
        max_jitter=0,
        summits=True,
        verbose=verbose,
    )
    _, rep2, _ = extract_loci(
        genome_path=files.genome_path,
        chroms=chroms,
        bw_paths=[files.rep2_bw_path],
        bed_path=peak_path,
        input_length=output_length,
        output_length=output_length,
        max_jitter=0,
        summits=True,
        verbose=verbose,
    )
    return rep1.numpy(), rep2.numpy()


def compute_standard_metrics(
    true_profiles: np.ndarray,
    pred_log_profiles: np.ndarray,
    pred_log_counts: np.ndarray,
) -> dict:
    N = true_profiles.shape[0]
    # Reshape to (N, 1, L, 1) for compute_performance_metrics
    true_flat = true_profiles.reshape(N, -1)
    true_metrics = np.expand_dims(true_flat, (1, 3))
    true_counts = true_metrics.sum(axis=2)

    pred_flat = pred_log_profiles.reshape(N, -1)
    pred_metrics = np.expand_dims(pred_flat, (1, 3))

    pred_counts = pred_log_counts.reshape(N, -1)
    pred_counts = np.expand_dims(pred_counts, 1)

    return compute_performance_metrics(
        true_metrics, pred_metrics, true_counts, pred_counts,
        smooth_true_profs=False, smooth_pred_profs=False,
    )


def finite_mean(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = np.asarray(values)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_outputs(
    *,
    model_results: dict[str, np.ndarray],
    jsd_arrays: dict[str, np.ndarray] | None,
    standard_metrics: dict,
    files: AtacFoldFilesConfig,
    args: argparse.Namespace,
    params: dict[str, Any],
    model_name: str = "capy",
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    eval_dir = files.eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.cell_type
    split = canonical_split(args.split)
    rc_suffix = "_rc" if args.reverse_complement else ""

    true_profiles = model_results["true_profiles"]
    pred_log_profiles = model_results["pred_log_profiles"]
    pred_log_counts = model_results["pred_log_counts"]

    N = true_profiles.shape[0]
    true_log_counts = np.log1p(true_profiles.sum(axis=(1, 2)))
    profile_valid = true_profiles.sum(axis=(1, 2)) > 0
    if jsd_arrays is None:
        jsd_arrays = {key: np.full(N, np.nan, dtype=np.float64) for key in JSD_BASELINE_COLUMNS}

    # Summary metrics CSV
    summary = {
        key: finite_mean(np.ravel(standard_metrics[key]), mask=profile_valid)
        for key in PROFILE_METRIC_COLUMNS
    }
    summary.update({
        key: finite_mean(standard_metrics[key])
        for key in ("count_pearson", "count_spearman", "count_mse", "count_r2")
    })
    summary.update({
        "model_name": model_name,
        "cell_type": args.cell_type,
        "data_type": "atac",
        "fold": int(args.fold),
        "timestamp": args.timestamp,
        "split": split,
        "reverse_complement": bool(args.reverse_complement),
        "num_examples": N,
    })
    for key in JSD_BASELINE_COLUMNS:
        arr = np.asarray(jsd_arrays[key])
        summary[key] = finite_mean(arr)
        finite = arr[np.isfinite(arr)]
        summary[f"{key}_median"] = float(np.median(finite)) if finite.size else float("nan")
    jsd_arr = np.ravel(standard_metrics["jsd"])
    jsd_finite = jsd_arr[np.isfinite(jsd_arr)]
    summary["jsd_median"] = float(np.median(jsd_finite)) if jsd_finite.size else float("nan")

    summary_path = eval_dir / f"{prefix}_metrics_summary{rc_suffix}_{split}.csv"
    profile_path = eval_dir / f"{prefix}_metrics_profile{rc_suffix}_{split}.csv"
    log_path = eval_dir / f"{prefix}_eval_log{rc_suffix}_{split}.txt"

    write_csv(summary_path, [summary], list(summary.keys()))
    profile_rows = []
    for i in range(N):
        row = {"example_index": i}
        row.update({key: float(np.ravel(standard_metrics[key])[i]) for key in PROFILE_METRIC_COLUMNS})
        row.update({key: float(np.asarray(jsd_arrays[key])[i]) for key in JSD_BASELINE_COLUMNS})
        profile_rows.append(row)
    write_csv(profile_path, profile_rows, ["example_index", *PROFILE_METRIC_COLUMNS, *JSD_BASELINE_COLUMNS])

    saved_paths: dict[str, str] = {
        "metrics_summary": str(summary_path),
        "metrics_profile": str(profile_path),
        "eval_log": str(log_path),
    }

    if args.save_predictions:
        pred_profiles_path = eval_dir / f"{prefix}_log_pred_profiles{rc_suffix}_{split}.npy"
        pred_counts_path = eval_dir / f"{prefix}_log_pred_counts{rc_suffix}_{split}.npy"
        true_counts_path = eval_dir / f"{prefix}_log_true_counts{rc_suffix}_{split}.npy"
        np.save(pred_profiles_path, pred_log_profiles)
        np.save(pred_counts_path, pred_log_counts.reshape(N, -1).squeeze(axis=1))
        np.save(true_counts_path, true_log_counts)
        saved_paths.update({
            "log_pred_profiles": str(pred_profiles_path),
            "log_pred_counts": str(pred_counts_path),
            "log_true_counts": str(true_counts_path),
        })


    log_payload = {
        "args": vars(args),
        "checkpoint_path": str(files.best_checkpoint_path) if files.best_checkpoint_path.exists() else None,
        "params_path": str(files.params_path),
        "peak_path": str(split_peak_path(files, split)),
        "output_length": int(params["dataset"]["output_length"]),
        "outputs": saved_paths,
        "summary": summary,
    }
    if source_metadata is not None:
        log_payload["source_metadata"] = source_metadata
    with log_path.open("w") as f:
        f.write(json.dumps(log_payload, default=json_default, indent=2, sort_keys=True) + "\n")

    return saved_paths


def run(args: argparse.Namespace) -> None:
    require_training_dependencies()
    files = AtacFoldFilesConfig.create(
        proj_dir=args.proj_dir,
        cell_type=args.cell_type,
        fold=args.fold,
        model_name="capy",
        timestamp=args.timestamp,
    )
    if not files.best_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {files.best_checkpoint_path}")
    if not files.params_path.exists():
        raise FileNotFoundError(f"Missing saved params: {files.params_path}")

    params = read_yaml(files.params_path)
    batch_size = int(args.batch_size or params["train"]["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else params["dataloader"]["num_workers"])
    device = select_device(args.device)

    print(f"Loading CAPY checkpoint: {files.best_checkpoint_path}", flush=True)
    model = load_model(params, files.best_checkpoint_path, device)

    print(f"Loading {args.split} split data.", flush=True)
    dataloader = make_eval_loader(
        files=files,
        params=params,
        split=args.split,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=args.verbose,
    )

    print(f"Evaluating on {device}; reverse_complement={args.reverse_complement}", flush=True)
    model_results = run_model(model, dataloader, device, reverse_complement=args.reverse_complement)

    standard_metrics = compute_standard_metrics(
        model_results["true_profiles"],
        model_results["pred_log_profiles"],
        model_results["pred_log_counts"],
    )

    # Compute JSD arrays for Fig 1d if replicate BigWigs are present
    jsd_arrays = None
    try:
        files.validate_replicates()
        print("Loading pseudoreplicate BigWigs for JSD upper bound.", flush=True)
        chroms = load_chrom_names(files.chrom_size_path)
        peak_path = split_peak_path(files, args.split)
        rep1, rep2 = load_replicate_profiles(
            files=files,
            peak_path=peak_path,
            output_length=int(params["dataset"]["output_length"]),
            chroms=chroms,
            verbose=args.verbose,
        )
        jsd_arrays = compute_jsd_arrays(
            model_results["true_profiles"],
            model_results["pred_log_profiles"],
            rep1,
            rep2,
        )
        print("JSD arrays computed.", flush=True)
    except FileNotFoundError as exc:
        print(f"Skipping pseudoreplicate JSD (rep BigWigs not found): {exc}", flush=True)

    if jsd_arrays is None:
        jsd_arrays = compute_jsd_arrays(
            model_results["true_profiles"],
            model_results["pred_log_profiles"],
        )
    saved_paths = save_outputs(
        model_results=model_results,
        jsd_arrays=jsd_arrays,
        standard_metrics=standard_metrics,
        files=files,
        args=args,
        params=params,
    )
    print("Saved evaluation outputs:", flush=True)
    for name, path in saved_paths.items():
        print(f"  {name}: {path}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
