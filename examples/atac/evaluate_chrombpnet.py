from __future__ import annotations

import argparse
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara.data import load_chrom_names
from examples.atac.evaluate import (
    SPLITS,
    compute_jsd_arrays,
    compute_standard_metrics,
    load_replicate_profiles,
    make_eval_loader,
    run_model,
    save_outputs,
    split_peak_path,
)
from examples.atac.file_config import AtacFoldFilesConfig
from examples.shared.train_utils import require_training_dependencies, select_device, write_yaml


CHROMBPNET_INPUT_LENGTH = 2114
CHROMBPNET_OUTPUT_LENGTH = 1000
MODEL_NAME = "chrombpnet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate original pretrained ChromBPNet weights on local ATAC splits."
    )
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--fold", type=int, default=0, help="Official zero-based ChromBPNet/CAPY fold, 0-4.")
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--weights_tar", type=Path, required=True)
    parser.add_argument(
        "--chrombpnet_fold",
        type=int,
        default=None,
        help="Zero-based fold directory inside the ChromBPNet tarball. Defaults to --fold.",
    )
    parser.add_argument(
        "--model_accession",
        type=str,
        default=None,
        help="Optional accession suffix used in tar members, e.g. ENCSR637XSC.",
    )
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--reverse_complement", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def source_fold(args: argparse.Namespace) -> int:
    fold = int(args.chrombpnet_fold) if args.chrombpnet_fold is not None else int(args.fold)
    if not (0 <= fold <= 4):
        raise ValueError(f"chrombpnet_fold must be in [0, 4], got {fold}.")
    return fold


def _clean_member_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def _find_member(
    tar: tarfile.TarFile,
    *,
    fold: int,
    kind: str,
    model_accession: str | None,
) -> str:
    fold_prefix = f"fold_{fold}/"
    candidates = []
    for member in tar.getmembers():
        if not member.isfile():
            continue
        name = _clean_member_name(member.name)
        basename = Path(name).name
        if not name.startswith(fold_prefix):
            continue
        if kind == "bias" and not basename.startswith(f"model.bias_scaled.fold_{fold}."):
            continue
        if kind == "accessibility" and not basename.startswith(f"model.chrombpnet_nobias.fold_{fold}."):
            continue
        if not basename.endswith(".h5"):
            continue
        if model_accession is not None and f".{model_accession}.h5" not in basename:
            continue
        candidates.append(member.name)

    if len(candidates) != 1:
        accession_msg = f" and accession {model_accession}" if model_accession else ""
        raise FileNotFoundError(
            f"Expected exactly one {kind} model in fold_{fold}{accession_msg}; "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _read_member(tar: tarfile.TarFile, name: str) -> bytes:
    handle = tar.extractfile(name)
    if handle is None:
        raise FileNotFoundError(f"Could not read tar member: {name}")
    return handle.read()


def load_chrombpnet_from_tar(
    weights_tar: Path,
    *,
    fold: int,
    model_accession: str | None,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    try:
        from bpnetlite import ChromBPNet
    except ImportError as exc:
        raise ImportError(
            "Missing bpnet-lite. Install the ATAC extra before running this evaluator."
        ) from exc

    with tarfile.open(weights_tar, "r:gz") as tar:
        bias_member = _find_member(
            tar,
            fold=fold,
            kind="bias",
            model_accession=model_accession,
        )
        accessibility_member = _find_member(
            tar,
            fold=fold,
            kind="accessibility",
            model_accession=model_accession,
        )
        bias_bytes = _read_member(tar, bias_member)
        accessibility_bytes = _read_member(tar, accessibility_member)

    model = ChromBPNet.from_chrombpnet(BytesIO(bias_bytes), BytesIO(accessibility_bytes))
    model.to(device)
    model.eval()
    metadata = {
        "weights_tar": str(weights_tar),
        "chrombpnet_fold": int(fold),
        "model_accession": model_accession,
        "bias_member": bias_member,
        "accessibility_member": accessibility_member,
        "input_length": CHROMBPNET_INPUT_LENGTH,
        "output_length": CHROMBPNET_OUTPUT_LENGTH,
    }
    return model, metadata


def params_for_chrombpnet(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": {
            "name": MODEL_NAME,
            "input_length": CHROMBPNET_INPUT_LENGTH,
            "output_length": CHROMBPNET_OUTPUT_LENGTH,
        },
        "dataset": {
            "input_length": CHROMBPNET_INPUT_LENGTH,
            "output_length": CHROMBPNET_OUTPUT_LENGTH,
            "max_jitter": 0,
            "reverse_complement": False,
            "seed": None,
        },
        "train": {
            "batch_size": int(args.batch_size),
        },
        "dataloader": {
            "num_workers": int(args.num_workers),
        },
    }


def run(args: argparse.Namespace) -> None:
    require_training_dependencies()
    if not args.weights_tar.exists():
        raise FileNotFoundError(f"Missing ChromBPNet weights tarball: {args.weights_tar}")

    cbp_fold = source_fold(args)
    files = AtacFoldFilesConfig.create(
        proj_dir=args.proj_dir,
        cell_type=args.cell_type,
        fold=args.fold,
        model_name=MODEL_NAME,
        timestamp=args.timestamp,
    )
    params = params_for_chrombpnet(args)
    device = select_device(args.device)

    config_dict = files.as_dict()
    config_dict.update(
        {
            "chrombpnet_fold": cbp_fold,
            "chrombpnet_fold_mapping": "exact_zero_based_fold",
            "weights_tar": str(args.weights_tar),
            "model_accession": args.model_accession,
        }
    )
    write_yaml(files.config_path, config_dict)
    write_yaml(files.params_path, params)

    print(
        f"Loading original ChromBPNet fold_{cbp_fold} for local fold{args.fold}: "
        f"{args.weights_tar}",
        flush=True,
    )
    model, source_metadata = load_chrombpnet_from_tar(
        args.weights_tar,
        fold=cbp_fold,
        model_accession=args.model_accession,
        device=device,
    )

    print(
        f"Loading {args.split} split data with input_length={CHROMBPNET_INPUT_LENGTH}, "
        f"output_length={CHROMBPNET_OUTPUT_LENGTH}.",
        flush=True,
    )
    dataloader = make_eval_loader(
        files=files,
        params=params,
        split=args.split,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        verbose=args.verbose,
    )

    print(f"Evaluating on {device}; reverse_complement={args.reverse_complement}", flush=True)
    model_results = run_model(model, dataloader, device, reverse_complement=args.reverse_complement)
    standard_metrics = compute_standard_metrics(
        model_results["true_profiles"],
        model_results["pred_log_profiles"],
        model_results["pred_log_counts"],
    )

    jsd_arrays = None
    try:
        files.validate_replicates()
        print("Loading pseudoreplicate BigWigs for JSD upper bound.", flush=True)
        chroms = load_chrom_names(files.chrom_size_path)
        peak_path = split_peak_path(files, args.split)
        rep1, rep2 = load_replicate_profiles(
            files=files,
            peak_path=peak_path,
            output_length=CHROMBPNET_OUTPUT_LENGTH,
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
        model_name=MODEL_NAME,
        source_metadata=source_metadata,
    )
    print("Saved evaluation outputs:", flush=True)
    for name, path in saved_paths.items():
        print(f"  {name}: {path}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
