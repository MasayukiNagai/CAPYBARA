"""Tier-B attribution for the CAPY Stage-1 bias model.

Computes contribution scores on the profile and counts heads over a 30K subsample
of the bias peaks, writing ChromBPNet's ``.h5`` schema. Feed the resulting ``.h5``
to ``run_modisco.sh`` to confirm the bias model learned only Tn5/repeat motifs.

``--method`` selects the engine (default ``gradientshap``): ``gradientshap``
(``captum``) or ``deeplift`` (``tangermeme.deep_lift_shap``). GradientShap is the
default because DeepLIFT shows high convergence deltas on CAPY's pooling U-Net;
DeepLIFT remains available for the closest-to-ChromBPNet comparison. Outputs are
namespaced by method under ``.../attribution/<method>/``.

The Majdandzic simplex-tangent gradient correction is **ON by default** on the
gradientshap path; ``--no_gradient_correction`` disables it and namespaces the run
under ``.../attribution/gradientshap_uncorrected/`` so the two never collide.

Example:
    python -m examples.atac.attribution.attribution_bias \\
        --proj_dir results/runs/models/chrombpnet_benchmark \\
        --cell_type K562 --fold 0 --timestamp chead --method gradientshap --heads profile counts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from examples.atac.bias_capy.file_config import BiasCapyFiles
from examples.atac.evaluate import load_model
from examples.shared.train_utils import read_yaml, require_training_dependencies, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepLIFT contribution scores for the CAPY bias model.")
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--shared_root", type=Path, default=None)
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--method", choices=["gradientshap", "deeplift"], default="gradientshap",
                        help="Attribution engine. Default gradientshap (DeepLIFT has high convergence deltas on CAPY).")
    parser.add_argument("--no_gradient_correction", action="store_true",
                        help="Disable the Majdandzic simplex-tangent gradient correction (gradientshap "
                             "only; default ON). Use for an uncorrected A/B baseline; output is namespaced "
                             "under attribution/gradientshap_uncorrected/. No effect on the deeplift engine.")
    parser.add_argument("--heads", nargs="+", choices=["profile", "counts"], default=["profile", "counts"])
    parser.add_argument("--n_subsample", type=int, default=DEFAULT_N_SUBSAMPLE)
    parser.add_argument("--n_shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    parser.add_argument("--seed", type=int, default=SUBSAMPLE_RANDOM_STATE)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--print_convergence_deltas", action="store_true",
                        help="Report DeepLIFT summation-to-delta error per batch (deeplift engine only).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_training_dependencies()

    kwargs = dict(proj_dir=args.proj_dir, cell_type=args.cell_type, fold=args.fold, timestamp=args.timestamp)
    if args.shared_root is not None:
        kwargs["shared_root"] = args.shared_root
    files = BiasCapyFiles.create(**kwargs)
    files.validate_inputs()
    if not files.best_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing bias checkpoint: {files.best_checkpoint_path}")
    if not files.params_path.exists():
        raise FileNotFoundError(f"Missing saved params: {files.params_path}")

    params = read_yaml(files.params_path)
    device = select_device(args.device)
    input_length = int(params["dataset"]["input_length"])

    print(f"Loading CAPY bias checkpoint: {files.best_checkpoint_path}", flush=True)
    model = load_model(params, files.best_checkpoint_path, device)

    # Correction applies to gradientshap only; namespace an uncorrected run so it
    # does not clobber the corrected one (cf. the deeplift_uncorrected_* convention).
    gradient_correction = not args.no_gradient_correction
    method_dirname = args.method
    if args.method == "gradientshap" and not gradient_correction:
        method_dirname += "_uncorrected"
    out_dir = files.model_dir / "attribution" / method_dirname
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
        batch_size=args.batch_size,
        gradient_correction=gradient_correction,
        print_convergence_deltas=args.print_convergence_deltas,
        verbose=args.verbose,
    )

    correction_note = ""
    if args.method == "gradientshap":
        correction_note = f", gradient_correction={'on' if gradient_correction else 'off'}"
    print(f"\n=== CAPY bias attribution ({args.method}{correction_note}) — wrote ChromBPNet-schema .h5 ===",
          flush=True)
    for head, path in written.items():
        print(f"  {head:>8}: {path}", flush=True)
    print(f"  interpreted regions: {out_dir / 'interpreted_regions.bed'}", flush=True)
    print("  next: run_modisco.sh on each .h5 to confirm only Tn5/repeat motifs.", flush=True)


if __name__ == "__main__":
    main()
