from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara.data import DNA_ALPHABET, DNA_IGNORE, load_chrom_names


@dataclass(frozen=True)
class BenchmarkPaths:
    genome_path: Path
    chrom_size_path: Path
    bed_path: Path
    signal_paths: tuple[Path, ...]
    summits: bool


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def print_package_report() -> None:
    packages = ["tangermeme", "pybigtools", "pyBigWig", "pyfaidx", "pandas", "torch", "numpy"]
    print("Package availability:")
    for name in packages:
        print(f"  {name}: {'ok' if package_available(name) else 'missing'}")


def one_hot_encode_local(sequence: str, dtype: np.dtype = np.float32) -> np.ndarray:
    sequence = sequence.upper()
    encoded = np.zeros((len(sequence), 4), dtype=dtype)
    lookup = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, base in enumerate(sequence):
        j = lookup.get(base)
        if j is not None:
            encoded[i, j] = 1
    return encoded


def load_bed_array(path: str | Path, *, chroms: list[str], summits: bool) -> dict[str, np.ndarray]:
    usecols = (0, 1, 2, 9) if summits else (0, 1, 2)
    raw = np.loadtxt(path, dtype=str, usecols=usecols)
    if raw.ndim == 1:
        raw = raw[None, :]

    chrom = raw[:, 0]
    start = raw[:, 1].astype(int)
    end = raw[:, 2].astype(int)
    if summits:
        summit = raw[:, 3].astype(int)
        width = end - start
        mid = start + summit
        start = mid - width // 2
        end = mid + width // 2

    keep = np.isin(chrom, chroms)
    return {"chrom": chrom[keep], "start": start[keep], "end": end[keep]}


class BigWigReader:
    def __init__(self, path: str | Path, backend: str) -> None:
        self.backend = backend
        self.path = str(path)
        if backend == "pyBigWig":
            import pyBigWig

            self.handle = pyBigWig.open(self.path, "r")
            if self.handle is None:
                raise OSError(f"Could not open BigWig: {path}")
        elif backend == "pybigtools":
            import pybigtools

            self.handle = pybigtools.open(self.path)
        else:
            raise ValueError(f"Unsupported BigWig backend: {backend}")

    def values(self, chrom: str, start: int, end: int) -> np.ndarray:
        if self.backend == "pyBigWig":
            values = self.handle.values(chrom, start, end, numpy=True)
        else:
            values = self.handle.values(chrom, start, end, fillna=0)
        return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def close(self) -> None:
        close = getattr(self.handle, "close", None)
        if close is not None:
            close()


def choose_local_backend(requested: str) -> str:
    if requested == "pyBigWig":
        if not package_available("pyBigWig"):
            raise ImportError("pyBigWig is required for --backend pyBigWig but is not installed.")
        return "pyBigWig"
    if requested == "pybigtools":
        if not package_available("pybigtools"):
            raise ImportError("pybigtools is required for --backend pybigtools but is not installed.")
        return "pybigtools"
    if package_available("pyBigWig"):
        return "pyBigWig"
    if package_available("pybigtools"):
        return "pybigtools"
    raise ImportError("Need either pyBigWig or pybigtools for the local extraction benchmark.")


def extract_local(
    *,
    genome_path: Path,
    chroms: list[str],
    bed_path: Path,
    signal_paths: tuple[Path, ...],
    input_window: int,
    output_window: int,
    summits: bool,
    n_loci: int | None,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    from pyfaidx import Fasta

    fasta = Fasta(str(genome_path), sequence_always_upper=True)
    bigwigs = [BigWigReader(path, backend) for path in signal_paths]
    loci = load_bed_array(bed_path, chroms=chroms, summits=summits)

    seqs = []
    signals = []
    half_in = input_window // 2
    half_out = output_window // 2
    try:
        for chrom, start, end in zip(loci["chrom"], loci["start"], loci["end"]):
            mid = int(start) + (int(end) - int(start)) // 2
            seq_start = mid - half_in
            seq_end = mid + half_in + (input_window % 2)
            sig_start = mid - half_out
            sig_end = mid + half_out + (output_window % 2)
            if seq_start < 0 or sig_start < 0:
                continue

            seq = one_hot_encode_local(str(fasta[str(chrom)][seq_start:seq_end])).T
            signal = np.stack([bw.values(str(chrom), sig_start, sig_end) for bw in bigwigs])
            if seq.shape != (4, input_window) or signal.shape != (len(bigwigs), output_window):
                continue

            seqs.append(seq)
            signals.append(signal)
            if n_loci is not None and len(seqs) >= n_loci:
                break
    finally:
        fasta.close()
        for bw in bigwigs:
            bw.close()

    if not seqs:
        raise RuntimeError(f"No valid loci extracted from {bed_path}")

    return (
        torch.from_numpy(np.stack(seqs)).to(torch.float32),
        torch.from_numpy(np.stack(signals)).to(torch.float32),
    )


def extract_tangermeme(
    *,
    genome_path: Path,
    chroms: list[str],
    bed_path: Path,
    signal_paths: tuple[Path, ...],
    input_window: int,
    output_window: int,
    summits: bool,
    n_loci: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from tangermeme.io import extract_loci

    seqs, signals = extract_loci(
        loci=str(bed_path),
        sequences=str(genome_path),
        signals=[str(path) for path in signal_paths],
        chroms=chroms,
        in_window=input_window,
        out_window=output_window,
        max_jitter=0,
        summits=summits,
        alphabet=DNA_ALPHABET,
        ignore=DNA_IGNORE,
        n_loci=n_loci,
        verbose=False,
    )
    return seqs.to(torch.float32), signals.to(torch.float32)


def paths_for_dataset(args: argparse.Namespace) -> BenchmarkPaths:
    proj_dir = Path(args.proj_dir)
    if args.dataset == "atac":
        processed = proj_dir / "data" / "atac" / "processed" / args.cell_type
        return BenchmarkPaths(
            genome_path=proj_dir / "genome" / "hg38.withrDNA.fasta",
            chrom_size_path=proj_dir / "genome" / "hg38.withrDNA.chrom.sizes",
            bed_path=processed / f"peaks_fold{args.fold}_{args.split}.bed.gz",
            signal_paths=(processed / "atac.bw",),
            summits=True,
        )

    processed = proj_dir / "data" / args.data_type / "processed" / args.cell_type
    return BenchmarkPaths(
        genome_path=proj_dir / "genome" / "hg38.withrDNA.fasta",
        chrom_size_path=proj_dir / "genome" / "hg38.withrDNA.chrom.sizes",
        bed_path=processed / f"peaks_fold{args.fold}_{args.split}.bed.gz",
        signal_paths=(processed / "5prime.pos.bigWig", processed / "5prime.neg.bigWig"),
        summits=False,
    )


def validate_paths(paths: BenchmarkPaths) -> None:
    required = [paths.genome_path, paths.chrom_size_path, paths.bed_path, *paths.signal_paths]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark input(s):\n  " + "\n  ".join(missing))


def time_call(name: str, fn, repeats: int) -> list[dict[str, Any]]:
    rows = []
    for rep in range(1, repeats + 1):
        gc.collect()
        start = time.perf_counter()
        seqs, signals = fn()
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "method": name,
                "repeat": rep,
                "seconds": elapsed,
                "n": int(seqs.shape[0]),
                "seq_shape": tuple(seqs.shape),
                "signal_shape": tuple(signals.shape),
                "seq_sum": float(seqs.sum()),
                "signal_sum": float(signals.sum()),
            }
        )
        del seqs, signals
    return rows


def print_results(rows: list[dict[str, Any]]) -> None:
    print("\nResults:")
    print("method\trepeat\tseconds\tn\tseq_shape\tsignal_shape\tseq_sum\tsignal_sum")
    for row in rows:
        print(
            f"{row['method']}\t{row['repeat']}\t{row['seconds']:.3f}\t{row['n']}\t"
            f"{row['seq_shape']}\t{row['signal_shape']}\t{row['seq_sum']:.1f}\t{row['signal_sum']:.1f}"
        )

    by_method: dict[str, list[float]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(float(row["seconds"]))
    print("\nSummary:")
    for method, values in by_method.items():
        print(f"  {method}: mean={np.mean(values):.3f}s min={np.min(values):.3f}s max={np.max(values):.3f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local vs tangermeme locus extraction.")
    parser.add_argument("--dataset", choices=["atac", "procap"], default="atac")
    parser.add_argument("--proj-dir", type=Path, default=None)
    parser.add_argument("--cell-type", default="K562")
    parser.add_argument("--data-type", default="procap")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--split", choices=["train", "val", "test", "train_and_val"], default="train")
    parser.add_argument("--n-loci", type=int, default=1000, help="Number of kept loci to extract; use 0 for all.")
    parser.add_argument("--input-window", type=int, default=2448)
    parser.add_argument("--output-window", type=int, default=1400)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--backend", choices=["auto", "pyBigWig", "pybigtools"], default="auto")
    parser.add_argument("--method", choices=["both", "local", "tangermeme"], default="both")
    args = parser.parse_args()
    if args.proj_dir is None:
        args.proj_dir = Path(
            "/grid/koo/home/shared/data/chrombpnet"
            if args.dataset == "atac"
            else "/grid/koo/home/shared/data/procapnet"
        )
    args.n_loci = None if args.n_loci == 0 else args.n_loci
    return args


def main() -> None:
    args = parse_args()
    print_package_report()
    paths = paths_for_dataset(args)
    validate_paths(paths)

    chroms = load_chrom_names(paths.chrom_size_path)
    print("\nInputs:")
    print(f"  dataset: {args.dataset}")
    print(f"  bed: {paths.bed_path}")
    print(f"  signals: {', '.join(str(path) for path in paths.signal_paths)}")
    print(f"  genome: {paths.genome_path}")
    print(f"  windows: input={args.input_window} output={args.output_window}")
    print(f"  n_loci: {args.n_loci if args.n_loci is not None else 'all'}")

    rows: list[dict[str, Any]] = []
    if args.method in {"both", "local"}:
        backend = choose_local_backend(args.backend)
        print(f"\nLocal backend: {backend}")
        rows.extend(
            time_call(
                f"local-{backend}",
                lambda: extract_local(
                    genome_path=paths.genome_path,
                    chroms=chroms,
                    bed_path=paths.bed_path,
                    signal_paths=paths.signal_paths,
                    input_window=args.input_window,
                    output_window=args.output_window,
                    summits=paths.summits,
                    n_loci=args.n_loci,
                    backend=backend,
                ),
                args.repeats,
            )
        )

    if args.method in {"both", "tangermeme"}:
        rows.extend(
            time_call(
                "tangermeme",
                lambda: extract_tangermeme(
                    genome_path=paths.genome_path,
                    chroms=chroms,
                    bed_path=paths.bed_path,
                    signal_paths=paths.signal_paths,
                    input_window=args.input_window,
                    output_window=args.output_window,
                    summits=paths.summits,
                    n_loci=args.n_loci,
                ),
                args.repeats,
            )
        )

    print_results(rows)


if __name__ == "__main__":
    main()
