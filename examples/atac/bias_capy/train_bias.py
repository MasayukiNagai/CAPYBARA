from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara import CAPY, load_config, print_model_summary
from examples.atac.bias_capy.data import BiasDataModule
from examples.atac.bias_capy.file_config import BiasCapyFiles
from examples.shared.train_utils import (
    read_yaml,
    require_training_dependencies,
    select_device,
    train_model,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CAPY-native bias model (Stage 1) on ChromBPNet non-peaks.")
    parser.add_argument("--proj_dir", type=Path, required=True, help="CAPY output root (models/ written here).")
    parser.add_argument("--params", type=Path, default=REPO_ROOT / "configs" / "atac_capy_bias.yaml")
    parser.add_argument("--shared_root", type=Path, default=None, help="Override shared ChromBPNet asset root.")
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--device", type=str, default="gpu", help="Device: gpu, cpu, auto, or a torch device string.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
    return parser.parse_args()


def make_files(args: argparse.Namespace) -> BiasCapyFiles:
    kwargs: dict[str, Any] = dict(
        proj_dir=args.proj_dir,
        cell_type=args.cell_type,
        fold=args.fold,
        timestamp=args.timestamp,
    )
    if args.shared_root is not None:
        kwargs["shared_root"] = args.shared_root
    return BiasCapyFiles.create(**kwargs)


def build_datamodule(
    *,
    files: BiasCapyFiles,
    params: dict[str, Any],
    batch_size: int,
    num_workers: int,
    verbose: bool,
) -> BiasDataModule:
    dataset_params = params["dataset"]
    split = files.fold_split()
    data_config = {
        "genome_path": str(files.genome_path),
        "data_bw_path": str(files.data_bw_path),
        "nonpeaks_bed_path": str(files.nonpeaks_bed_path),
        "train_chroms": split["train"],
        "valid_chroms": split["valid"],
        "input_length": int(dataset_params["input_length"]),
        "output_length": int(dataset_params["output_length"]),
        "max_jitter": int(dataset_params["max_jitter"]),
        "reverse_complement": bool(dataset_params["reverse_complement"]),
        "random_seed": dataset_params.get("seed"),
    }
    return BiasDataModule(
        config=data_config,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=params["dataloader"].get("prefetch_factor", 2),
        pin_memory=bool(params["dataloader"].get("pin_memory", True)),
        persistent_workers=bool(params["dataloader"].get("persistent_workers", True)),
        verbose=verbose,
    )


def main() -> None:
    args = parse_args()
    require_training_dependencies()

    files = make_files(args)
    files.validate_inputs()

    params = read_yaml(args.params)
    if args.no_wandb:
        params.setdefault("wandb", {})["enabled"] = False

    # Mirror ChromBPNet exactly: use the fold's recorded counts_loss_weight.
    counts_weight = files.counts_loss_weight
    params.setdefault("train", {})["counts_weight"] = counts_weight
    print(f"Using per-fold counts_weight={counts_weight} (from {files.model_params_tsv_path.name})", flush=True)

    device = select_device(args.device)
    model_cfg = load_config(args.params)
    files.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_dict = files.as_dict()
    write_yaml(files.params_path, params)
    write_yaml(files.config_path, config_dict)

    print("Building datamodule...", flush=True)
    datamodule = build_datamodule(
        files=files,
        params=params,
        batch_size=int(params["train"]["batch_size"]),
        num_workers=int(params["dataloader"]["num_workers"]),
        verbose=args.verbose,
    )

    print("Building model...", flush=True)
    model = CAPY(model_cfg)
    metadata = {
        "model_name": "capy_bias",
        "params": params,
        "file_config": config_dict,
    }
    print(f"Training CAPY bias model on {device}; outputs: {files.model_dir}", flush=True)
    print_model_summary(model)
    train_model(
        model=model,
        datamodule=datamodule,
        output_paths=config_dict,
        params=params,
        device=device,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
