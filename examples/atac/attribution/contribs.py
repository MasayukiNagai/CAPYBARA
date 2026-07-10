"""CAPY contribution-scoring core (Tier B).

Produces ChromBPNet's exact interpretation ``.h5`` (``raw``/``shap``/
``projected_shap``, each ``(N, 4, L)``) from a trained CAPY model, so the
container's ``modisco`` runs on the output unchanged. Two attribution engines are
provided and share everything else (region selection, scalar targets, ``.h5``
schema):

* ``engine="deeplift"``  — ``tangermeme.deep_lift_shap`` (used for the bias model).
* ``engine="gradientshap"`` — ``captum.attr.GradientShap`` with per-sequence
  dinucleotide-shuffled baselines (used for the attention-based nobias model).

Regions are summit-centered and subsampled to 30K with ``random_state`` matching
ChromBPNet's ``peaks.sample(30000, random_state=1234)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara.data import load_chrom_names, one_hot_encode
from examples.atac.attribution.wrappers import CountScalarWrapper, ProfileScalarWrapper

WRAPPERS = {"profile": ProfileScalarWrapper, "counts": CountScalarWrapper}
DEFAULT_N_SUBSAMPLE = 30000
DEFAULT_N_SHUFFLES = 20
SUBSAMPLE_RANDOM_STATE = 1234  # matches ChromBPNet peaks.sample(..., random_state=1234)


# --------------------------------------------------------------------------- #
# Dinucleotide shuffle (Altschul-Erikson; vendored from the kundaje deeplift
# implementation) — used for captum GradientShap baselines.
# --------------------------------------------------------------------------- #
def _one_hot_to_tokens(one_hot: np.ndarray) -> np.ndarray:
    """(L, D) one-hot -> (L,) int tokens; ambiguous positions map to D."""
    tokens = np.full(one_hot.shape[0], one_hot.shape[1], dtype=np.int64)
    seq_inds, dim_inds = np.nonzero(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def _tokens_to_one_hot(tokens: np.ndarray, dim: int) -> np.ndarray:
    """(L,) tokens -> (L, dim) one-hot; token==dim (ambiguous) becomes all-zero."""
    identity = np.identity(dim + 1, dtype=np.float32)[:, :-1]
    return identity[tokens]


def dinuc_shuffle(one_hot_ld: np.ndarray, num_shufs: int, rng: np.random.RandomState) -> np.ndarray:
    """Dinucleotide-preserving shuffle of a ``(L, D)`` one-hot -> ``(num, L, D)``."""
    arr = _one_hot_to_tokens(one_hot_ld)
    chars, tokens = np.unique(arr, return_inverse=True)

    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t
        shuf_next_inds.append(np.where(mask)[0] + 1)

    dim = one_hot_ld.shape[1]
    results = np.empty((num_shufs, one_hot_ld.shape[0], dim), dtype=np.float32)
    for i in range(num_shufs):
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            if len(inds) > 1:
                inds[:-1] = rng.permutation(len(inds) - 1)  # keep the last edge fixed
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]
        results[i] = _tokens_to_one_hot(chars[result], dim)
    return results


# --------------------------------------------------------------------------- #
# Region extraction (summit-centered, subsampled, edge-filtered)
# --------------------------------------------------------------------------- #
def extract_onehot_regions(
    *,
    genome_path: Path,
    chrom_size_path: Path,
    bed_path: Path,
    input_length: int,
    n_subsample: int = DEFAULT_N_SUBSAMPLE,
    seed: int = SUBSAMPLE_RANDOM_STATE,
    skip_ambiguous: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Summit-center + subsample + one-hot a narrowPeak BED.

    Returns ``(onehot (N, 4, L) float32, used_bed_lines)`` with rows aligned:
    the i-th one-hot corresponds to the i-th line in ``used_bed_lines`` (the
    subset that survived the chromosome-edge filter). Centering matches
    ``capybara.data._iter_loci(summits=True)``: center = ``chromStart + col10``.

    ``skip_ambiguous`` (default True) drops windows containing any non-ACGT base
    (``N`` -> an all-zero one-hot column). Such windows are unusable by
    ``tangermeme.deep_lift_shap`` (it strictly requires each column to sum to 1)
    and carry no motif signal; the count dropped is logged. Applied for both
    engines so their region sets stay identical.
    """
    from pyfaidx import Fasta

    allowed = set(load_chrom_names(chrom_size_path))
    rows: list[tuple[str, list[str]]] = []
    with Path(bed_path).open() as handle:
        for line in handle:
            line = line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 3 or fields[0] not in allowed:
                continue
            rows.append((line, fields))

    if n_subsample and len(rows) > n_subsample:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(len(rows), size=n_subsample, replace=False))
        rows = [rows[i] for i in idx]

    fasta = Fasta(str(genome_path), sequence_always_upper=True)
    chrom_lengths = {str(k): len(v) for k, v in fasta.items()}
    half = input_length // 2

    onehots: list[np.ndarray] = []
    used: list[str] = []
    skipped = 0
    skipped_ambiguous = 0
    try:
        for line, fields in rows:
            chrom = fields[0]
            start, end = int(fields[1]), int(fields[2])
            summit = int(fields[9]) if len(fields) >= 10 else (end - start) // 2
            center = start + summit
            seq_start = center - half
            seq_end = center + half + (input_length % 2)
            chrom_length = chrom_lengths.get(chrom)
            if chrom_length is None or seq_start < 0 or seq_end > chrom_length:
                skipped += 1
                continue
            encoded = one_hot_encode(str(fasta[chrom][seq_start:seq_end]))  # (L, 4)
            if encoded.shape[0] != input_length:
                skipped += 1
                continue
            if skip_ambiguous and not (encoded.sum(axis=1) == 1).all():
                skipped_ambiguous += 1
                continue
            onehots.append(encoded.T)  # (4, L)
            used.append(line)
    finally:
        fasta.close()

    if not onehots:
        raise RuntimeError(f"No valid loci extracted from {bed_path}")
    onehot = np.stack(onehots).astype(np.float32)  # (N, 4, L)
    if verbose:
        print(
            f"Extracted {onehot.shape[0]} regions from {bed_path} "
            f"(subsample={n_subsample}, skipped_edges={skipped}, "
            f"skipped_ambiguous_N={skipped_ambiguous}, shape={onehot.shape})",
            flush=True,
        )
    return onehot, used


# --------------------------------------------------------------------------- #
# Attribution engines (both return hypothetical scores, shape (N, 4, L))
# --------------------------------------------------------------------------- #
def deeplift_attributions(
    model: torch.nn.Module,
    head: str,
    onehot: Tensor,
    *,
    device: torch.device,
    batch_size: int = 16,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    verbose: bool = True,
) -> np.ndarray:
    """DeepLIFT/DeepSHAP hypothetical attributions via ``tangermeme.deep_lift_shap``.

    Requires strictly one-hot ``onehot`` (every column sums to 1): tangermeme
    validates this and its internal dinucleotide-shuffle re-validates, so
    ``N``/ambiguous (all-zero) columns raise. ``extract_onehot_regions`` drops such
    windows (``skip_ambiguous=True``), so inputs from ``generate_scores`` are safe.
    """
    try:
        from tangermeme.deep_lift_shap import deep_lift_shap
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tangermeme is required for the DeepLIFT engine. Install with "
            "`.venv/bin/pip install tangermeme`."
        ) from exc

    wrapper = WRAPPERS[head](model).to(device).eval()
    attributions = deep_lift_shap(
        wrapper,
        onehot,
        n_shuffles=n_shuffles,
        hypothetical=True,
        device=str(device),
        batch_size=batch_size,
        verbose=verbose,
    )
    return attributions.detach().cpu().numpy().astype(np.float16)  # (N, 4, L)


def gradientshap_attributions(
    model: torch.nn.Module,
    head: str,
    onehot: Tensor,
    *,
    device: torch.device,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = SUBSAMPLE_RANDOM_STATE,
    verbose: bool = True,
) -> np.ndarray:
    """GradientShap hypothetical attributions with per-sequence dinuc baselines.

    Used for the attention-based nobias model, where DeepLIFT's rescale rule is
    unreliable through self-attention. One sequence at a time so each input gets
    its own dinucleotide-shuffled baseline distribution (mirrors ChromBPNet's /
    ProCapNet's per-sequence reference construction).
    """
    try:
        from captum.attr import GradientShap
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "captum is required for the GradientShap engine. Install with "
            "`.venv/bin/pip install captum`."
        ) from exc

    wrapper = WRAPPERS[head](model).to(device).eval()
    explainer = GradientShap(wrapper)
    onehot_np = onehot.detach().cpu().numpy()
    n, _, length = onehot_np.shape
    out = np.empty((n, 4, length), dtype=np.float16)
    rng = np.random.RandomState(seed)

    iterator = range(n)
    if verbose:
        try:
            from tqdm import trange

            iterator = trange(n, desc=f"gradientshap:{head}")
        except ImportError:
            pass

    for i in iterator:
        x = onehot[i : i + 1].to(device)  # (1, 4, L)
        baselines = dinuc_shuffle(onehot_np[i].T, n_shuffles, rng)  # (S, L, 4)
        baselines = torch.from_numpy(np.transpose(baselines, (0, 2, 1))).to(device)  # (S, 4, L)
        attr = explainer.attribute(x, baselines=baselines, target=0, n_samples=n_shuffles, stdevs=0.0)
        out[i] = attr[0].detach().cpu().numpy().astype(np.float16)
    return out


# --------------------------------------------------------------------------- #
# Output: ChromBPNet .h5 schema + interpreted regions bed
# --------------------------------------------------------------------------- #
def write_chrombpnet_h5(path: Path, onehot: np.ndarray, hypothetical: np.ndarray) -> None:
    """Write the ChromBPNet interpretation schema (raw/shap/projected_shap)."""
    import h5py

    onehot_i8 = onehot.astype(np.int8)
    hyp_f16 = hypothetical.astype(np.float16)
    projected = (onehot.astype(np.float32) * hypothetical.astype(np.float32)).astype(np.float16)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("raw/seq", data=onehot_i8, compression="gzip")
        handle.create_dataset("shap/seq", data=hyp_f16, compression="gzip")
        handle.create_dataset("projected_shap/seq", data=projected, compression="gzip")


def write_regions_bed(path: Path, lines: list[str]) -> None:
    with Path(path).open("w") as handle:
        for line in lines:
            handle.write(line + "\n")


# --------------------------------------------------------------------------- #
# Orchestrator shared by both CLIs
# --------------------------------------------------------------------------- #
def generate_scores(
    model: torch.nn.Module,
    *,
    bed_path: Path,
    genome_path: Path,
    chrom_size_path: Path,
    input_length: int,
    out_dir: Path,
    cell_type: str,
    fold: int,
    engine: str,
    heads: list[str],
    device: torch.device,
    n_subsample: int = DEFAULT_N_SUBSAMPLE,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = SUBSAMPLE_RANDOM_STATE,
    batch_size: int = 16,
    verbose: bool = True,
) -> dict[str, Path]:
    """Subsample -> extract one-hot -> attribute -> write ChromBPNet ``.h5`` per head."""
    if engine not in ("deeplift", "gradientshap"):
        raise ValueError(f"Unknown engine {engine!r}; expected 'deeplift' or 'gradientshap'.")
    for head in heads:
        if head not in WRAPPERS:
            raise ValueError(f"Unknown head {head!r}; expected one of {sorted(WRAPPERS)}.")

    out_dir.mkdir(parents=True, exist_ok=True)
    onehot_np, used_lines = extract_onehot_regions(
        genome_path=genome_path,
        chrom_size_path=chrom_size_path,
        bed_path=bed_path,
        input_length=input_length,
        n_subsample=n_subsample,
        seed=seed,
        verbose=verbose,
    )
    write_regions_bed(out_dir / "interpreted_regions.bed", used_lines)
    onehot_t = torch.from_numpy(onehot_np).to(torch.float32)

    written: dict[str, Path] = {}
    for head in heads:
        print(f"Computing {engine} attributions for '{head}' head over {onehot_np.shape[0]} regions...", flush=True)
        if engine == "deeplift":
            hyp = deeplift_attributions(
                model, head, onehot_t, device=device, batch_size=batch_size, n_shuffles=n_shuffles, verbose=verbose
            )
        else:
            hyp = gradientshap_attributions(
                model, head, onehot_t, device=device, n_shuffles=n_shuffles, seed=seed, verbose=verbose
            )
        h5_path = out_dir / f"{cell_type}.fold_{fold}.{head}_scores.h5"
        write_chrombpnet_h5(h5_path, onehot_np, hyp)
        written[head] = h5_path
        print(f"  wrote {h5_path}", flush=True)
    return written
