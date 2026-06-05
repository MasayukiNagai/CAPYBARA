#!/usr/bin/env python3
"""
prep_bam.py
Generate Tn5-shift-corrected ATAC-seq coverage BigWig(s) from a BAM file.

Replaces: chrombpnet prep bam / chrombpnet prep pseudoreplicates

Tn5 shift convention (ChromBPNet / symmetric +4/-4):
  forward-strand cut site = read.reference_start + tn5_plus
  reverse-strand cut site = read.reference_end   - tn5_minus

Outputs (in --out_dir):
  unstranded.bw            always
  unstranded.rep1.bw       only with --split
  unstranded.rep2.bw       only with --split

Rename to atac.bw / rep1.bw / rep2.bw after this script completes.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pybigtools
import pysam


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bam",         required=True, type=Path, help="Input BAM (must be indexed).")
    p.add_argument("--chrom_sizes", required=True, type=Path, help="Chromosome sizes file.")
    p.add_argument("--out_dir",     required=True, type=Path, help="Output directory.")
    p.add_argument("--split",       action="store_true", help="Also generate pseudoreplicate BigWigs.")
    p.add_argument("--tn5_plus",    type=int, default=4,  help="Tn5 shift for forward strand (default 4).")
    p.add_argument("--tn5_minus",   type=int, default=4,  help="Tn5 shift for reverse strand (default 4).")
    p.add_argument("--min_mapq",    type=int, default=30, help="Minimum mapping quality (default 30).")
    p.add_argument("--seed",        type=int, default=0,  help="Random seed for pseudoreplicate assignment.")
    p.add_argument("--threads",     type=int, default=1,  help="Threads for BAM decompression (default 1).")
    return p.parse_args()


def load_chrom_sizes(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                sizes[parts[0]] = int(parts[1])
    return sizes


def canonical_chroms(chrom_sizes: dict[str, int]) -> list[str]:
    """Return chromosomes that look like canonical hg38 autosomes/sex chromosomes."""
    skip = ("_", "M", "Un", "EBV", ".")
    return [c for c, _ in chrom_sizes.items()
            if c.startswith("chr") and not any(tok in c for tok in skip)]


def _rep_idx(query_name: str | None, seed: int) -> int:
    """Deterministically assign a read to replicate 0 or 1 by name hash."""
    key = f"{seed}\t{query_name or ''}".encode()
    return hashlib.md5(key).digest()[0] & 1


def _chrom_entries(chrom: str, cov: np.ndarray):
    nz = np.flatnonzero(cov)
    for pos in nz:
        yield (chrom, int(pos), int(pos) + 1, float(cov[pos]))


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_sizes = load_chrom_sizes(args.chrom_sizes)
    chroms = canonical_chroms(all_sizes)

    bam = pysam.AlignmentFile(str(args.bam), "rb", threads=args.threads)
    bam_chroms = set(bam.references)
    present = [c for c in chroms if c in bam_chroms]

    if not present:
        print("No canonical chromosomes found in BAM; nothing to do.", file=sys.stderr)
        bam.close()
        return

    chrom_sizes_dict = {c: all_sizes[c] for c in present}

    # Collect sparse coverage entries per output file across all chromosomes.
    entries_full: list[tuple] = []
    entries_r1: list[tuple] = []
    entries_r2: list[tuple] = []

    try:
        for chrom in present:
            size = all_sizes[chrom]
            cov = np.zeros(size, dtype=np.float32)
            r1 = np.zeros(size, dtype=np.float32) if args.split else None
            r2 = np.zeros(size, dtype=np.float32) if args.split else None

            print(f"  {chrom}", flush=True)
            for read in bam.fetch(chrom):
                if (read.is_unmapped or read.is_duplicate or
                        read.is_qcfail or read.is_secondary or
                        read.mapping_quality < args.min_mapq):
                    continue

                pos = (read.reference_start + args.tn5_plus
                       if not read.is_reverse
                       else read.reference_end - args.tn5_minus)

                if not (0 <= pos < size):
                    continue

                cov[pos] += 1
                if args.split:
                    rep = _rep_idx(read.query_name, args.seed)
                    (r1 if rep == 0 else r2)[pos] += 1

            entries_full.extend(_chrom_entries(chrom, cov))
            if args.split:
                entries_r1.extend(_chrom_entries(chrom, r1))
                entries_r2.extend(_chrom_entries(chrom, r2))
    finally:
        bam.close()

    out_full = args.out_dir / "unstranded.bw"
    pybigtools.pybigtools.open(str(out_full), "w").write(chrom_sizes_dict, iter(entries_full))
    print(f"Wrote {out_full}", flush=True)

    if args.split:
        out_rep1 = args.out_dir / "unstranded.rep1.bw"
        out_rep2 = args.out_dir / "unstranded.rep2.bw"
        pybigtools.pybigtools.open(str(out_rep1), "w").write(chrom_sizes_dict, iter(entries_r1))
        print(f"Wrote {out_rep1}", flush=True)
        pybigtools.pybigtools.open(str(out_rep2), "w").write(chrom_sizes_dict, iter(entries_r2))
        print(f"Wrote {out_rep2}", flush=True)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
