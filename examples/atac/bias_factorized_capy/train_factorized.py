from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara import CAPY, load_config, print_model_summary
from examples.atac.bias_factorized_capy.data import FactorizedDataModule
from examples.atac.bias_factorized_capy.factorized_model import BiasFactorizedCAPY
from examples.atac.bias_factorized_capy.file_config import FactorizedCapyFiles
from examples.shared.train_utils import (
    METRICS_COLUMNS,
    compute_losses,
    read_yaml,
    require_training_dependencies,
    run_training_loop,
    save_training_checkpoint,
    select_device,
    validate,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the bias-factorized CAPY accessibility model (Stage 2).")
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=REPO_ROOT / "configs" / "atac_capy_accessibility.yaml")
    parser.add_argument("--shared_root", type=Path, default=None)
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--timestamp", type=str, required=True, help="Stage-2 run id (shared with scale_bias).")
    parser.add_argument("--bias_timestamp", type=str, default="chead")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
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


def build_datamodule(*, files: FactorizedCapyFiles, params: dict[str, Any], verbose: bool) -> FactorizedDataModule:
    dataset_params = params["dataset"]
    split = files.fold_split()
    data_config = {
        "genome_path": str(files.genome_path),
        "data_bw_path": str(files.data_bw_path),
        "peaks_bed_path": str(files.peaks_bed_path),
        "nonpeaks_bed_path": str(files.nonpeaks_bed_path),
        "train_chroms": split["train"],
        "valid_chroms": split["valid"],
        "input_length": int(dataset_params["input_length"]),
        "output_length": int(dataset_params["output_length"]),
        "max_jitter": int(dataset_params["max_jitter"]),
        "negative_sampling_ratio": float(dataset_params["negative_sampling_ratio"]),
        "reverse_complement": bool(dataset_params["reverse_complement"]),
        "seed": dataset_params.get("seed"),
    }
    return FactorizedDataModule(
        config=data_config,
        batch_size=int(params["train"]["batch_size"]),
        num_workers=int(params["dataloader"]["num_workers"]),
        prefetch_factor=params["dataloader"].get("prefetch_factor", 2),
        pin_memory=bool(params["dataloader"].get("pin_memory", True)),
        verbose=verbose,
    )


def build_composed_model(*, files: FactorizedCapyFiles, model_cfg: dict[str, Any], device) -> BiasFactorizedCAPY:
    accessibility = CAPY(model_cfg)
    scaled = torch.load(files.bias_scaled_path, map_location=device)
    bias = CAPY(scaled["params"])
    bias.load_state_dict(scaled["model_state_dict"])
    model = BiasFactorizedCAPY(accessibility, bias)
    model.to(device)
    return model


def main() -> None:
    args = parse_args()
    require_training_dependencies()

    files = make_files(args)
    files.validate_inputs()
    if not files.bias_scaled_path.exists():
        raise FileNotFoundError(
            f"Missing scaled bias {files.bias_scaled_path}. Run scale_bias.py "
            f"--timestamp {args.timestamp} first."
        )

    params = read_yaml(args.params)
    if args.no_wandb:
        params.setdefault("wandb", {})["enabled"] = False

    # Mirror ChromBPNet's per-fold data params exactly (read from the shared tsv).
    # counts_weight may be overridden per-run for non-parity CAPY-optimization sweeps
    # via train.counts_weight_override in the YAML; absent that key, the TSV value is
    # used (strict parity). max_jitter / neg_ratio stay TSV-forced.
    cw_override = params.get("train", {}).get("counts_weight_override")
    if cw_override is not None:
        counts_weight = float(cw_override)
        print(
            f"[non-parity] counts_weight override -> {counts_weight} "
            f"(TSV {files.counts_loss_weight} ignored)",
            flush=True,
        )
    else:
        counts_weight = files.counts_loss_weight
    params.setdefault("train", {})["counts_weight"] = counts_weight
    params.setdefault("dataset", {})["max_jitter"] = files.max_jitter
    params["dataset"]["negative_sampling_ratio"] = files.negative_sampling_ratio
    print(
        f"Per-fold params: counts_weight={counts_weight} max_jitter={files.max_jitter} "
        f"neg_ratio={files.negative_sampling_ratio} (from {files.model_params_tsv_path.name})",
        flush=True,
    )

    device = select_device(args.device)
    model_cfg = load_config(args.params)
    files.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_dict = files.as_dict()
    write_yaml(files.params_path, params)
    write_yaml(files.config_path, config_dict)

    print("Building datamodule...", flush=True)
    datamodule = build_datamodule(files=files, params=params, verbose=args.verbose)

    print("Building composed model (accessibility + frozen scaled bias)...", flush=True)
    model = build_composed_model(files=files, model_cfg=model_cfg, device=device)
    print_model_summary(model.accessibility)

    counts_weight = float(params["train"]["counts_weight"])
    learning_rate = float(params["train"]["learning_rate"])
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    scheduler = None
    scheduler_cfg = params.get("lr_scheduler", {})
    if scheduler_cfg.get("scheduler_name") == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_cfg.get("scheduler_kwargs", {})
        )

    def train_joint_step(m: nn.Module, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        total_loss, profile_loss, count_loss = compute_losses(m, batch, counts_weight)
        return total_loss, {
            "train_loss": total_loss,
            "train_profile_loss": profile_loss,
            "train_count_loss": count_loss,
        }

    metadata = {"model_name": "capy_chrombpnet", "params": params, "file_config": config_dict}
    print(f"Training Stage-2 CAPY on {device}; outputs: {files.model_dir}", flush=True)
    run_training_loop(
        model=model,
        datamodule=datamodule,
        output_paths=config_dict,
        params=params,
        device=device,
        metadata=metadata,
        optimizer=optimizer,
        train_step_fn=train_joint_step,
        validate_fn=lambda m, vl, d: validate(m, vl, d, counts_weight),
        set_train_mode_fn=lambda m: m.train(),  # BiasFactorizedCAPY keeps bias in eval
        metrics_columns=METRICS_COLUMNS,
        selection_metric="valid_loss",
        best_metric_name="best_valid_loss",
        scheduler=scheduler,
    )

    # Save the accessibility-only ("nobias") model from the best composed checkpoint.
    best = torch.load(files.best_checkpoint_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    save_training_checkpoint(
        files.nobias_path,
        model=model.accessibility,
        optimizer=optimizer,
        epoch=int(best.get("epoch", -1)),
        best_metric_name="best_valid_loss",
        best_metric_value=float(best.get("best_valid_loss", float("nan"))),
        metadata={"model_name": "capy_chrombpnet_nobias", "params": params},
    )
    print(f"Saved accessibility-only model -> {files.nobias_path}", flush=True)


if __name__ == "__main__":
    main()
