"""Tier-B attribution for the CAPY Stage-2 accessibility (nobias) model.

Computes contribution scores on the profile head over a 30K subsample of the
Stage-2 peaks, writing ChromBPNet's ``.h5`` schema. Feed the resulting ``.h5`` to
``run_modisco.sh`` to recover TF motifs and compare against the ChromBPNet
reference motif set.

``--method`` selects the engine (default ``gradientshap``). GradientShap is
strongly preferred here: the accessibility CAPY uses a ``hybrid_attention``
bottleneck where DeepLIFT/DeepSHAP's rescale rule is unreliable through
self-attention; GradientShap is gradient-based and robust to attention.
``--method deeplift`` is available for experimentation but not recommended for the
attention model. Scalar targets and ``.h5`` schema are identical across engines,
so ``modisco`` runs unchanged. Outputs are namespaced by method under
``.../attribution/<method>/``.

Example:
    python -m examples.atac.attribution.attribution_nobias \\
        --proj_dir results/runs/models/chrombpnet_benchmark \\
        --cell_type K562 --fold 0 --timestamp chead_test1 --bias_timestamp chead --method gradientshap --heads profile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.atac.attribution.contribs import (
    DEFAULT_N_SHUFFLES,
    DEFAULT_N_SUBSAMPLE,
    SUBSAMPLE_RANDOM_STATE,
    generate_scores,
)
from examples.atac.bias_factorized_capy.file_config import FactorizedCapyFiles
from examples.atac.evaluate import load_model
from examples.shared.train_utils import read_yaml, require_training_dependencies, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GradientShap contribution scores for the CAPY nobias model.")
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--shared_root", type=Path, default=None)
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--bias_timestamp", type=str, default="chead")
    parser.add_argument("--method", choices=["gradientshap", "deeplift"], default="gradientshap",
                        help="Attribution engine. Default gradientshap (robust to the attention bottleneck).")
    parser.add_argument("--heads", nargs="+", choices=["profile", "counts"], default=["profile"],
                        help="ChromBPNet's main pipeline runs modisco on the profile head only.")
    parser.add_argument("--n_subsample", type=int, default=DEFAULT_N_SUBSAMPLE)
    parser.add_argument("--n_shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    parser.add_argument("--seed", type=int, default=SUBSAMPLE_RANDOM_STATE)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_training_dependencies()

    kwargs: dict[str, Any] = dict(
        proj_dir=args.proj_dir,
        cell_type=args.cell_type,
        fold=args.fold,
        timestamp=args.timestamp,
        bias_timestamp=args.bias_timestamp,
    )
    if args.shared_root is not None:
        kwargs["shared_root"] = args.shared_root
    files = FactorizedCapyFiles.create(**kwargs)
    files.validate_inputs()
    if not files.nobias_path.exists():
        raise FileNotFoundError(f"Missing accessibility (nobias) checkpoint: {files.nobias_path}")
    if not files.params_path.exists():
        raise FileNotFoundError(f"Missing saved params: {files.params_path}")

    params = read_yaml(files.params_path)
    device = select_device(args.device)
    input_length = int(params["dataset"]["input_length"])

    print(f"Loading CAPY nobias (accessibility) checkpoint: {files.nobias_path}", flush=True)
    model = load_model(params, files.nobias_path, device)

    out_dir = files.model_dir / "attribution" / args.method
    written = generate_scores(
        model,
        bed_path=files.peaks_bed_path,
        genome_path=files.genome_path,
        chrom_size_path=files.chrom_size_path,
        input_length=input_length,
        out_dir=out_dir,
        cell_type=args.cell_type,
        fold=args.fold,
        engine=args.method,
        heads=list(args.heads),
        device=device,
        n_subsample=args.n_subsample,
        n_shuffles=args.n_shuffles,
        seed=args.seed,
        verbose=args.verbose,
    )

    print(f"\n=== CAPY nobias attribution ({args.method}) — wrote ChromBPNet-schema .h5 ===", flush=True)
    for head, path in written.items():
        print(f"  {head:>8}: {path}", flush=True)
    print(f"  interpreted regions: {out_dir / 'interpreted_regions.bed'}", flush=True)
    print("  next: run_modisco.sh on the .h5 to recover TF motifs.", flush=True)


if __name__ == "__main__":
    main()
