from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara import CAPY
from capybara.data import extract_loci
from examples.atac.bias_factorized_capy.file_config import FactorizedCapyFiles
from examples.shared.train_utils import (
    final_count_layer,
    read_yaml,
    require_training_dependencies,
    select_device,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Depth-scale a trained CAPY bias model (add delta to the count-head bias), "
        "mirroring ChromBPNet's adjust_bias_model_logcounts."
    )
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--shared_root", type=Path, default=None)
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--timestamp", type=str, required=True, help="Stage-2 output timestamp/run id.")
    parser.add_argument("--bias_timestamp", type=str, default="chead", help="Stage-1 bias run to scale.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
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


def load_bias_model(files: FactorizedCapyFiles, device: torch.device) -> tuple[CAPY, dict[str, Any]]:
    params = read_yaml(files.bias_params_path)
    model = CAPY(params)
    checkpoint = torch.load(files.bias_checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Bias checkpoint missing model_state_dict: {files.bias_checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, params


@torch.no_grad()
def predict_logcounts(model: CAPY, seqs: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    preds = []
    for start in range(0, seqs.shape[0], batch_size):
        batch = seqs[start : start + batch_size].to(device, non_blocking=True).to(torch.float32)
        log_counts = model.forward_count(batch)  # (B, 1)
        preds.append(log_counts.detach().cpu().numpy().ravel())
    return np.concatenate(preds, axis=0)


def main() -> None:
    args = parse_args()
    require_training_dependencies()

    files = make_files(args)
    files.validate_inputs()
    files.validate_bias_checkpoint()
    device = select_device(args.device)

    bias_params = read_yaml(files.bias_params_path)
    dataset_params = bias_params["dataset"]
    input_length = int(dataset_params["input_length"])
    output_length = int(dataset_params["output_length"])

    split = files.fold_split()
    train_valid_chroms = list(split["train"]) + list(split["valid"])
    min_thresh = files.counts_sum_min_thresh
    max_thresh = files.counts_sum_max_thresh

    print(
        f"Scaling bias '{args.bias_timestamp}' on train+valid non-peaks "
        f"({len(train_valid_chroms)} chroms), in-range ({min_thresh}, {max_thresh})...",
        flush=True,
    )

    # ChromBPNet scales on the training (train+valid) non-peaks used for the main model.
    seqs, signals, _ = extract_loci(
        genome_path=files.genome_path,
        chroms=train_valid_chroms,
        bw_paths=[files.data_bw_path],
        bed_path=files.nonpeaks_bed_path,
        input_length=input_length,
        output_length=output_length,
        max_jitter=0,
        summits=True,
        verbose=args.verbose,
    )
    counts = signals.sum(dim=(1, 2)).numpy().astype(np.float64)  # (N,) summed insertions
    in_range = (counts < max_thresh) & (counts > min_thresh)
    n_used = int(in_range.sum())
    if n_used == 0:
        raise RuntimeError("No in-range non-peaks for bias scaling; check thresholds.")

    model, _ = load_bias_model(files, device)
    pred_logcts = predict_logcounts(model, seqs[in_range], device, args.batch_size)
    obs_log1p = np.log(1.0 + counts[in_range])
    delta = float(np.mean(obs_log1p - pred_logcts))

    mean_pred_before = float(np.mean(pred_logcts))
    mean_obs = float(np.mean(obs_log1p))
    print(
        f"  n_in_range={n_used}  mean_obs_log1p={mean_obs:.4f}  "
        f"mean_pred_before={mean_pred_before:.4f}  delta={delta:.4f}",
        flush=True,
    )

    # Add delta to the count-head's final Linear bias only (weights unchanged).
    layer = final_count_layer(model, "capy")
    with torch.no_grad():
        layer.bias.add_(delta)

    # Sanity: mean predicted logcount should now match mean observed log1p.
    pred_after = predict_logcounts(model, seqs[in_range], device, args.batch_size)
    mean_pred_after = float(np.mean(pred_after))
    print(f"  mean_pred_after={mean_pred_after:.4f}  (target {mean_obs:.4f})", flush=True)

    files.model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "params": bias_params,
        "delta": delta,
        "counts_sum_min_thresh": min_thresh,
        "counts_sum_max_thresh": max_thresh,
        "n_in_range": n_used,
        "mean_obs_log1p": mean_obs,
        "mean_pred_before": mean_pred_before,
        "mean_pred_after": mean_pred_after,
        "bias_timestamp": args.bias_timestamp,
        "source_checkpoint": str(files.bias_checkpoint_path),
    }
    torch.save(payload, files.bias_scaled_path)
    write_yaml(files.config_path, files.as_dict())
    print(f"  wrote scaled bias -> {files.bias_scaled_path}", flush=True)


if __name__ == "__main__":
    main()
