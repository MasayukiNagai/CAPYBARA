from __future__ import annotations

"""Tier C — marginal footprinting for the bias-factorized CAPY benchmark.

Ports ChromBPNet's ``evaluation/marginal_footprints/marginal_footprinting.py``
(in ``chrombpnet_latest.sif``) verbatim, adapted to CAPY's two-headed PyTorch
model. A TN5 enzyme motif is inserted at the exact center of every background
non-peak sequence; the model is run on the inserted sequence *and* its reverse
complement; the per-base footprint is ``softmax(profile_logits) * (exp(logcounts)
- 1)``, summed fwd+rev, normalized per sequence to sum 1, and averaged over all
backgrounds. The QC scalar is the max of that mean footprint; ChromBPNet's gate
is ``all(round(max, 3) < 0.003)`` (rounds first, then strict ``<`` -> effective
raw gate ``< 0.0025``).

Parity notes settled from disk (see PLAN):
* Background regions = the *identical* BED ChromBPNet footprinted
  (``files.nonpeaks_bed_path`` = ``.../auxiliary/<cell>.fold_<N>_filtered.nonpeaks.bed``),
  subset to the fold's test chroms. 66,474 regions for K562/fold_0.
* ``skip_ambiguous`` is OFF to mirror ChromBPNet's ``get_seq`` (which does not
  drop N-windows); CAPY encodes ``N`` as an all-zero column, fine in a forward
  pass. The N-window count is reported, not dropped.

CRITICAL interpretation caveat (also in the write-up): each sequence's footprint
is normalized to sum 1, so the ``(exp(logcounts) - 1)`` factor cancels *within* a
sequence and only sets the fwd/rev relative weighting. **Tier C therefore
measures profile-head shape almost exclusively and must NOT be cited as evidence
that the counts head is debiased.** The uniform baseline is ``1/outputlen`` =
``1/1000`` = 0.001, so the 0.003 gate is only ~3x the noise floor; the
interpretable quantity is each motif's max relative to the empty-motif control.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara import CAPY
from capybara.data import extract_loci, one_hot_encode
from examples.atac.bias_factorized_capy.evaluate_factorized import load_composed_model
from examples.atac.bias_factorized_capy.file_config import FactorizedCapyFiles
from examples.atac.evaluate import load_model
from examples.shared.train_utils import read_yaml, require_training_dependencies, select_device

MODEL_CHOICES = ["nobias", "bias", "composed"]
SPLITS = ["train", "valid", "test", "all"]

# ChromBPNet's container constants (evaluation/marginal_footprints + make_html).
GATE_RAW = 0.003  # container: np.all(round(max, 3) < 0.003)
GATE_DOC = 0.002  # PLAN.md / BENCHMARK_0713 stated target


# --------------------------------------------------------------------------- #
# ChromBPNet ports
# --------------------------------------------------------------------------- #
def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """ChromBPNet's numerically-stabilized softmax over axis=1 (verbatim)."""
    norm_x = x - np.mean(x, axis=1, keepdims=True)
    return np.exp(temp * norm_x) / np.sum(np.exp(temp * norm_x), axis=1, keepdims=True)


def read_motif_tsv(path: Path) -> list[tuple[str, str]]:
    """Read ChromBPNet's 2-col ``motif_to_pwm.tsv`` (MOTIF_NAME<TAB>MOTIF_PWM_FWD)."""
    motifs: list[tuple[str, str]] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            motifs.append((fields[0].strip(), fields[1].strip()))
    return motifs


# --------------------------------------------------------------------------- #
# Background loading — mirrors ChromBPNet's get_seq (center = start + summit,
# width = inputlen, NO edge/N filtering), plus a test-chrom subset.
# --------------------------------------------------------------------------- #
def load_backgrounds(
    *,
    genome_path: Path,
    bed_path: Path,
    chroms: set[str],
    input_length: int,
    n_subsample: int | None,
    seed: int,
    verbose: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(uint8 (N, 4, input_length), stats)`` for test-chrom non-peaks.

    Mirrors ``chrombpnet/training/utils/data_utils.get_seq``: for each region,
    fetch ``genome[chr][center - width//2 : center - width//2 + width]`` where
    ``center = start + summit``, then one-hot. Stored as ``uint8`` so the full
    background is ~0.5 GB (not 1.7 GB float32) and motif insertion happens in a
    per-batch float32 copy, never on a full inserted array.
    """
    from pyfaidx import Fasta

    half = input_length // 2

    rows: list[tuple[str, int, int]] = []
    with Path(bed_path).open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 10 or fields[0] not in chroms:
                continue
            start = int(fields[1])
            summit = int(fields[9])
            rows.append((fields[0], start, summit))

    n_total_testchrom = len(rows)
    if n_subsample and n_total_testchrom > n_subsample:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(n_total_testchrom, size=n_subsample, replace=False))
        rows = [rows[i] for i in idx]

    fasta = Fasta(str(genome_path), sequence_always_upper=True)
    chrom_lengths = {str(k): len(v) for k, v in fasta.items()}

    onehots: list[np.ndarray] = []
    edge_skipped = 0
    n_windows_with_N = 0
    try:
        for chrom, start, summit in rows:
            center = start + summit
            seq_start = center - half
            seq_end = seq_start + input_length
            chrom_length = chrom_lengths.get(chrom)
            if chrom_length is None or seq_start < 0 or seq_end > chrom_length:
                edge_skipped += 1
                continue
            encoded = one_hot_encode(str(fasta[chrom][seq_start:seq_end]))  # (L, 4)
            if encoded.shape[0] != input_length:
                edge_skipped += 1
                continue
            if not (encoded.sum(axis=1) == 1).all():
                n_windows_with_N += 1  # reported, NOT dropped (mirrors get_seq)
            onehots.append(encoded.T.astype(np.uint8))  # (4, L)
    finally:
        fasta.close()

    if not onehots:
        raise RuntimeError(f"No background regions extracted from {bed_path} on chroms {sorted(chroms)}")

    bg = np.stack(onehots)  # (N, 4, L) uint8
    n = bg.shape[0]
    stats = {
        "n_backgrounds": int(n),
        "n_testchrom_regions": int(n_total_testchrom),
        "n_subsample": int(n_subsample) if n_subsample else None,
        "edge_skipped": int(edge_skipped),
        "n_windows_with_N": int(n_windows_with_N),
        "frac_windows_with_N": float(n_windows_with_N / n) if n else 0.0,
        "background_bed": str(bed_path),
    }
    if verbose:
        print(
            f"Backgrounds: N={n} (test-chrom regions={n_total_testchrom}, "
            f"edge_skipped={edge_skipped}, windows_with_N={n_windows_with_N} "
            f"[{stats['frac_windows_with_N']:.4%}], shape={bg.shape})",
            flush=True,
        )
    return bg, stats


# --------------------------------------------------------------------------- #
# CAPY prediction
# --------------------------------------------------------------------------- #
def batched_predict(model, seqs_f32: np.ndarray, device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Run a CAPY-style model on ``(N, 4, L)`` float32; return ``(profile (N, out), logcount (N, 1))``.

    Raw logits + logcounts (NOT log-softmaxed), unlike ``evaluate.run_model``.
    """
    profiles: list[np.ndarray] = []
    logcounts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, seqs_f32.shape[0], batch_size):
            batch = torch.from_numpy(seqs_f32[i : i + batch_size]).to(device)
            profile_logits, log_counts = model(batch)
            profiles.append(profile_logits.squeeze(1).float().cpu().numpy())  # (B, out)
            logcounts.append(log_counts.float().cpu().numpy().reshape(-1, 1))  # (B, 1)
    return np.concatenate(profiles, axis=0), np.concatenate(logcounts, axis=0)


def get_footprint_for_motif(
    bg_uint8: np.ndarray,
    motif: str,
    model,
    device,
    input_length: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    """ChromBPNet's ``get_footprint_for_motif``, in-place insertion per batch.

    ``motif`` is a DNA string ("" for the control, no insertion). Returns the
    mean per-base footprint ``(outputlen,)`` and the mean total counts.
    """
    midpoint = input_length // 2
    if motif:
        motif_oh = one_hot_encode(motif).T.astype(np.float32)  # (4, len)
        m_len = motif_oh.shape[1]
        m_start = midpoint - m_len // 2
    else:
        motif_oh = None
        m_start = m_len = 0

    prof_fwd_all, lc_fwd_all, prof_rev_all, lc_rev_all = [], [], [], []
    for i in range(0, bg_uint8.shape[0], batch_size):
        batch = bg_uint8[i : i + batch_size].astype(np.float32)  # per-batch copy only
        if motif_oh is not None:
            batch[:, :, m_start : m_start + m_len] = motif_oh
        # Reverse complement: flip length + channel (ACGT order -> channel-flip = complement).
        # ascontiguousarray: [:, ::-1, ::-1] has negative strides that torch rejects.
        batch_rev = np.ascontiguousarray(batch[:, ::-1, ::-1])

        pf, lf = batched_predict(model, batch, device, batch_size)
        pr, lr = batched_predict(model, batch_rev, device, batch_size)
        prof_fwd_all.append(pf)
        lc_fwd_all.append(lf)
        prof_rev_all.append(pr)
        lc_rev_all.append(lr)

    prof_fwd = np.concatenate(prof_fwd_all, axis=0)
    lc_fwd = np.concatenate(lc_fwd_all, axis=0)
    prof_rev = np.concatenate(prof_rev_all, axis=0)
    lc_rev = np.concatenate(lc_rev_all, axis=0)

    footprint_fwd = softmax(prof_fwd) * (np.exp(lc_fwd) - 1)  # (N, out)
    footprint_rev = softmax(prof_rev) * (np.exp(lc_rev) - 1)  # (N, out)
    counts_for_motif = (np.exp(lc_rev) - 1) + (np.exp(lc_fwd) - 1)  # (N, 1)

    footprint_tot = footprint_fwd + footprint_rev[:, ::-1]  # flip rev back
    footprint = footprint_tot / footprint_tot.sum(axis=1, keepdims=True)
    return footprint.mean(0), float(counts_for_motif.mean(0).item())


# --------------------------------------------------------------------------- #
# Model loaders
# --------------------------------------------------------------------------- #
def load_scaled_bias(files: FactorizedCapyFiles, device) -> CAPY:
    """Reconstruct the frozen, δ-scaled Stage-1 bias branch from ``bias_scaled.pt``."""
    scaled = torch.load(files.bias_scaled_path, map_location=device)
    model = CAPY(scaled["params"])
    model.load_state_dict(scaled["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_target_model(name: str, files: FactorizedCapyFiles, params: dict[str, Any], device):
    if name == "nobias":
        return load_model(params, files.nobias_path, device)
    if name == "bias":
        return load_scaled_bias(files, device)
    if name == "composed":
        return load_composed_model(files, params, device)
    raise ValueError(f"Unknown model {name!r}")


# --------------------------------------------------------------------------- #
# Sanity checks (item 6)
# --------------------------------------------------------------------------- #
def assert_onehot_acgt() -> None:
    """one_hot_encode('ACGT') must be I4 — makes the channel-flip a valid complement."""
    got = one_hot_encode("ACGT")
    if not np.array_equal(got, np.eye(4, dtype=got.dtype)):
        raise AssertionError(f"one_hot_encode('ACGT') is not the identity matrix:\n{got}")


def log_logcount_scale(files: FactorizedCapyFiles, params: dict[str, Any], device, n_loci: int = 50) -> None:
    """Log ``exp(logcounts)-1`` vs observed BigWig coverage on a few peaks (item 6b).

    Uses the composed model (fit to observed counts) as the reference for the
    log(1+Σcounts) convention. Non-fatal — logs an order-of-magnitude comparison.
    """
    try:
        model = load_composed_model(files, params, device)
    except Exception as exc:  # noqa: BLE001 — advisory check only
        print(f"[logcount-scale] skipped (composed model unavailable): {exc}", flush=True)
        return
    ds = params["dataset"]
    test_chroms = list(files.fold_split()["test"])
    seqs, signals, _ = extract_loci(
        genome_path=files.genome_path,
        chroms=test_chroms,
        bw_paths=[files.data_bw_path],
        bed_path=files.peaks_bed_path,
        input_length=int(ds["input_length"]),
        output_length=int(ds["output_length"]),
        max_jitter=0,
        summits=True,
        n_loci=n_loci,
        verbose=False,
    )
    seqs_np = seqs.numpy() if torch.is_tensor(seqs) else np.asarray(seqs)
    signals_np = signals.numpy() if torch.is_tensor(signals) else np.asarray(signals)
    _, logcount = batched_predict(model, seqs_np.astype(np.float32), device, batch_size=64)
    pred_counts = np.exp(logcount.reshape(-1)) - 1.0
    obs_counts = signals_np.reshape(signals_np.shape[0], -1).sum(axis=1)
    print(
        "[logcount-scale] composed model on "
        f"{seqs.shape[0]} peaks: median pred(exp(logc)-1)={np.median(pred_counts):.1f} "
        f"vs observed coverage sum median={np.median(obs_counts):.1f} "
        f"(ratio={np.median(pred_counts) / max(np.median(obs_counts), 1e-9):.2f}; "
        "same order of magnitude ⇒ logcounts are natural-log log(1+Σcounts))",
        flush=True,
    )
    del model


# --------------------------------------------------------------------------- #
# ChromBPNet reference (already on disk) for the two-column table
# --------------------------------------------------------------------------- #
def parse_chrombpnet_reference(files: FactorizedCapyFiles) -> dict[str, Any] | None:
    """Parse ChromBPNet's ``*_chrombpnet_nobias_max_bias_response.txt`` if present."""
    ref_txt = (
        files._cbp_fold_dir
        / "evaluation"
        / f"{files.cell_type}.fold_{files.fold}_chrombpnet_nobias_max_bias_response.txt"
    )
    if not ref_txt.exists():
        return None
    content = ref_txt.read_text().strip()
    # Format: "{label}_{mean}_{m1/m2/.../m5}"
    label, mean_str, maxes_str = content.split("_", 2)
    return {
        "path": str(ref_txt),
        "raw": content,
        "label": label,
        "mean": float(mean_str),
        "per_motif_max": [float(v) for v in maxes_str.split("/")],
    }


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def plot_footprint(fp: np.ndarray, control_fp: np.ndarray, outputlen: int, title: str, out_png: Path) -> None:
    lo, hi = outputlen // 2 - 100, outputlen // 2 + 100
    plt.figure()
    plt.plot(range(200), control_fp[lo:hi], color="0.6", lw=1, label="control")
    plt.plot(range(200), fp[lo:hi], color="C0", lw=1.5, label=title)
    plt.axhline(1.0 / outputlen, color="k", ls=":", lw=0.8, label="uniform (1/L)")
    plt.xlabel("200bp around motif insertion", fontsize=11)
    plt.ylabel("Probability", fontsize=11)
    plt.xticks(ticks=[0, 100, 200], labels=[-100, 0, 100])
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def summarize_motif(fp: np.ndarray, control_fp: np.ndarray, outputlen: int) -> dict[str, Any]:
    center = outputlen // 2
    raw_max = float(np.max(fp))
    control_max = float(np.max(control_fp))
    lo, hi = center - 100, center + 100
    return {
        "max": raw_max,
        "raw_max": raw_max,
        "rounded_max": float(np.round(raw_max, 3)),
        "min": float(np.min(fp)),
        "argmax": int(np.argmax(fp)),
        "argmin": int(np.argmin(fp)),
        "center_value": float(fp[center]),
        "center_minus_control_center": float(fp[center] - control_fp[center]),
        "delta_vs_control": raw_max - control_max,
        "ratio_vs_control": raw_max / control_max if control_max > 0 else float("inf"),
        "center_200bp": fp[lo:hi].astype(float).tolist(),
    }


def run_model_footprints(
    *,
    model_name: str,
    model,
    bg_uint8: np.ndarray,
    motifs: list[tuple[str, str]],
    input_length: int,
    batch_size: int,
    device,
    files: FactorizedCapyFiles,
    bg_stats: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    outputlen = None
    footprints: dict[str, np.ndarray] = {}
    counts: dict[str, float] = {}

    # control first (empty motif = no insertion) — the noise floor.
    control_fp, control_counts = get_footprint_for_motif(bg_uint8, "", model, device, input_length, batch_size)
    outputlen = control_fp.shape[0]
    footprints["control"] = control_fp
    counts["control"] = control_counts
    print(f"  [{model_name}] control: max={np.max(control_fp):.6f} (uniform={1.0/outputlen:.6f})", flush=True)

    avg_response_at_tn5: list[float] = []
    per_motif: dict[str, dict[str, Any]] = {"control": summarize_motif(control_fp, control_fp, outputlen)}

    for name, seq in motifs:
        fp, cnt = get_footprint_for_motif(bg_uint8, seq, model, device, input_length, batch_size)
        footprints[name] = fp
        counts[name] = cnt
        summary = summarize_motif(fp, control_fp, outputlen)
        per_motif[name] = summary
        if ("tn5" in name.lower()) or ("dnase" in name.lower()):
            # round in float64 (fp is float32) so the string matches ChromBPNet's "0.001" exactly.
            avg_response_at_tn5.append(round(float(np.max(fp)), 3))
        print(
            f"  [{model_name}] {name}: raw_max={summary['raw_max']:.6f} "
            f"center={summary['center_value']:.6f} Δctr={summary['center_minus_control_center']:+.6f} "
            f"ratio={summary['ratio_vs_control']:.2f}",
            flush=True,
        )
        plot_footprint(fp, control_fp, outputlen, name, out_dir / f"{files.cell_type}_{model_name}.{name}.footprint.png")

    plot_footprint(
        control_fp, control_fp, outputlen, "control",
        out_dir / f"{files.cell_type}_{model_name}.control.footprint.png",
    )

    # ChromBPNet's exact round-then-compare gate + max_bias_response.txt.
    passed_003 = bool(avg_response_at_tn5) and bool(np.all(np.array(avg_response_at_tn5) < GATE_RAW))
    passed_002 = bool(avg_response_at_tn5) and bool(np.all(np.array(avg_response_at_tn5) < GATE_DOC))
    if avg_response_at_tn5:
        label = "corrected" if passed_003 else "uncorrected"
        mean_resp = round(float(np.mean(avg_response_at_tn5)), 3)
        txt = f"{label}_{mean_resp}_" + "/".join(str(v) for v in avg_response_at_tn5)
        (out_dir / f"{files.cell_type}_{model_name}_max_bias_response.txt").write_text(txt)

    # Footprint arrays: deepdish (ChromBPNet-format) if available, else npz.
    fp_payload = {k: [footprints[k], np.array([counts[k]])] for k in footprints}
    h5_path = out_dir / f"{files.cell_type}_{model_name}_footprints.h5"
    try:
        import deepdish as dd

        dd.io.save(str(h5_path), fp_payload, compression="blosc")
        arrays_path = str(h5_path)
    except Exception:  # noqa: BLE001 — deepdish/pytables optional in the CAPY env
        npz_path = out_dir / f"{files.cell_type}_{model_name}_footprints.npz"
        np.savez_compressed(
            npz_path,
            **{f"{k}_footprint": footprints[k] for k in footprints},
            **{f"{k}_counts": np.array([counts[k]]) for k in footprints},
        )
        arrays_path = str(npz_path)

    # Sanity-check pipeline; NOT the final Tier-C eval. Two items still open for the real run
    # (not handled here): (a) bootstrap CIs on the per-motif maxima; (b) confirm whether the
    # delta-scaling adjusts the profile logits or the counts head before Tier C speaks to §8a.
    result: dict[str, Any] = {
        "model": model_name,
        "background": bg_stats,
        "background_set_comparable": True,
        "uniform_baseline": 1.0 / outputlen,
        "outputlen": int(outputlen),
        "gate_raw": GATE_RAW,
        "gate_doc": GATE_DOC,
        "gate_effective_raw_max": 0.0025,
        "gate_semantics": "all(round(max,3) < 0.003)  # rounds first, then strict <",
        "profile_shape_only": True,
        "profile_shape_only_note": (
            "Footprints are sum-1 normalized, so (exp(logcounts)-1) cancels within a sequence; "
            "Tier C measures profile-head shape, NOT counts-head debiasing."
        ),
        "tn5_rounded_max": avg_response_at_tn5,
        "tn5_mean_rounded_max": round(float(np.mean(avg_response_at_tn5)), 3) if avg_response_at_tn5 else None,
        "passed_gate_0.003": passed_003,
        "passed_gate_0.002": passed_002,
        "per_motif": per_motif,
        "arrays_path": arrays_path,
    }
    if model_name == "nobias":
        ref = parse_chrombpnet_reference(files)
        result["chrombpnet_reference"] = ref
        if ref is not None:
            capy_max = [per_motif[name]["rounded_max"] for name, _ in motifs if "tn5" in name.lower()]
            print("  --- CAPY nobias vs ChromBPNet nobias (rounded max) ---", flush=True)
            print(f"      CAPY:       {capy_max}", flush=True)
            print(f"      ChromBPNet: {ref['per_motif_max']}", flush=True)

    return result


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tier-C marginal footprinting of trained CAPY models (mirrors ChromBPNet)."
    )
    parser.add_argument("--proj_dir", type=Path, required=True)
    parser.add_argument("--shared_root", type=Path, default=None)
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--bias_timestamp", type=str, default="chead")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=["nobias", "bias"])
    parser.add_argument("--motif_tsv", type=Path, default=None, help="Default: the fold's motif_to_pwm.tsv.")
    parser.add_argument("--background_bed", type=Path, default=None, help="Default: files.nonpeaks_bed_path.")
    parser.add_argument("--n_subsample", type=int, default=None, help="Speed knob; default = all test-chrom nonpeaks.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--skip_scale_check", action="store_true")
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

    if not files.params_path.exists():
        raise FileNotFoundError(f"Missing saved params: {files.params_path}")
    params = read_yaml(files.params_path)
    ds = params["dataset"]
    input_length = int(ds["input_length"])
    output_length = int(ds["output_length"])
    # Parity guards — this benchmark is fixed to ChromBPNet's dims.
    assert input_length == 2114, f"expected input_length 2114, got {input_length}"
    assert output_length == 1000, f"expected output_length 1000, got {output_length}"
    assert input_length // 2 == 1057, "midpoint must be 1057"
    assert_onehot_acgt()

    batch_size = int(args.batch_size or params["train"]["batch_size"])
    device = select_device(args.device)

    motif_tsv = args.motif_tsv or files.motif_pwm_path
    if not Path(motif_tsv).exists():
        raise FileNotFoundError(f"Missing motif TSV: {motif_tsv}")
    motifs = read_motif_tsv(motif_tsv)
    print(f"Motifs ({len(motifs)}): {[m[0] for m in motifs]}", flush=True)

    if not args.skip_scale_check:
        log_logcount_scale(files, params, device)

    test_chroms = set(files.fold_split()[args.split])
    background_bed = args.background_bed or files.nonpeaks_bed_path
    bg_uint8, bg_stats = load_backgrounds(
        genome_path=files.genome_path,
        bed_path=Path(background_bed),
        chroms=test_chroms,
        input_length=input_length,
        n_subsample=args.n_subsample,
        seed=args.seed,
        verbose=True,
    )

    out_root = files.eval_dir / "footprints"
    all_results: dict[str, Any] = {
        "cell_type": args.cell_type,
        "fold": args.fold,
        "split": args.split,
        "timestamp": args.timestamp,
        "bias_timestamp": args.bias_timestamp,
        "background": bg_stats,
        "background_set_comparable": True,
        "motif_tsv": str(motif_tsv),
        "models": {},
    }

    for model_name in args.models:
        print(f"\n=== Tier C: {model_name} ===", flush=True)
        model = load_target_model(model_name, files, params, device)
        out_dir = out_root / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_model_footprints(
            model_name=model_name,
            model=model,
            bg_uint8=bg_uint8,
            motifs=motifs,
            input_length=input_length,
            batch_size=batch_size,
            device=device,
            files=files,
            bg_stats=bg_stats,
            out_dir=out_dir,
        )
        summary_path = out_dir / f"{args.cell_type}_{model_name}_footprint_summary.json"
        with summary_path.open("w") as handle:
            json.dump(result, handle, indent=2)
        print(f"  wrote {summary_path}", flush=True)
        all_results["models"][model_name] = {
            k: v for k, v in result.items() if k != "per_motif"
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_root.mkdir(parents=True, exist_ok=True)
    combined_path = out_root / f"{args.cell_type}_tierC_footprint_summary_{args.split}.json"
    with combined_path.open("w") as handle:
        json.dump(all_results, handle, indent=2)
    print(f"\nWrote combined summary: {combined_path}", flush=True)


if __name__ == "__main__":
    main()
