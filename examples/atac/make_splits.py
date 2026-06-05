#!/usr/bin/env python3
"""
make_splits.py
Create chromosome split JSONs and per-fold peak BED files from a narrowPeak file.

Replaces: chrombpnet helpers make_chr_splits / chrombpnet helpers get_fold_split_bed

Steps performed:
  1. Apply blacklist filtering to the input peaks (if --blacklist is given).
  2. Write blacklist-filtered all-chromosome peaks to <peaks_dir>/peaks.bed.gz.
  3. For each requested fold, write a JSON defining train/val/test chromosomes.
  4. Write per-fold peak BED files filtered to the appropriate chromosomes.

Chromosome assignments follow ChromBPNet paper Table 2 (5-fold CV on hg38):
  fold 1: test=chr1,chr8    val=chr10
  fold 2: test=chr2,chr6    val=chr22
  fold 3: test=chr3,chr13   val=chr21
  fold 4: test=chr4,chr7    val=chr19
  fold 5: test=chr5,chr16   val=chr20
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


FOLD_SPLITS: dict[int, dict[str, list[str]]] = {
    1: {"test": ["chr1",  "chr8"],  "val": ["chr10"]},
    2: {"test": ["chr2",  "chr6"],  "val": ["chr22"]},
    3: {"test": ["chr3",  "chr13"], "val": ["chr21"]},
    4: {"test": ["chr4",  "chr7"],  "val": ["chr19"]},
    5: {"test": ["chr5",  "chr16"], "val": ["chr20"]},
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chrom_sizes", required=True, type=Path)
    p.add_argument("--peaks",       required=True, type=Path,
                   help="Input peaks file (narrowPeak, optionally .gz). "
                        "Must have summit offset in column 10.")
    p.add_argument("--blacklist",   type=Path, default=None,
                   help="Blacklist BED file (.gz OK). Overlapping peaks are excluded.")
    p.add_argument("--splits_dir",  required=True, type=Path,
                   help="Directory where fold JSON files are written.")
    p.add_argument("--peaks_dir",   required=True, type=Path,
                   help="Directory where filtered peak BED files are written.")
    p.add_argument("--fold",        type=int, default=None,
                   help="Process only this fold (1–5). Default: all five folds.")
    return p.parse_args()


def _load_canonical_chroms(chrom_sizes_path: Path) -> list[str]:
    chroms = []
    with chrom_sizes_path.open() as f:
        for line in f:
            chrom = line.strip().split()[0]
            if not chrom.startswith("chr"):
                continue
            if any(tok in chrom for tok in ("_", "M", "Un", "EBV", ".")):
                continue
            chroms.append(chrom)
    return chroms


def _open(path: Path, mode: str = "rt"):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def _load_blacklist(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load blacklist BED into per-chromosome sorted (starts, ends) arrays."""
    regions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with _open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            regions[parts[0]].append((int(parts[1]), int(parts[2])))
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, ivs in regions.items():
        ivs.sort()
        result[chrom] = (
            np.array([s for s, _ in ivs], dtype=np.int32),
            np.array([e for _, e in ivs], dtype=np.int32),
        )
    return result


def _is_blacklisted(chrom: str, start: int, end: int,
                    blacklist: dict[str, tuple[np.ndarray, np.ndarray]]) -> bool:
    if chrom not in blacklist:
        return False
    bl_starts, bl_ends = blacklist[chrom]
    # All blacklist regions with bl_start < peak_end
    idx = int(np.searchsorted(bl_starts, end))
    if idx == 0:
        return False
    # Any of those with bl_end > peak_start is an overlap
    return bool(np.any(bl_ends[:idx] > start))


def _load_peaks(path: Path, canonical_chroms: set[str],
                blacklist: dict | None) -> list[str]:
    """Return lines from peaks file on canonical, non-blacklisted peaks."""
    kept: list[str] = []
    with _open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if chrom not in canonical_chroms:
                continue
            if blacklist and _is_blacklisted(chrom, int(parts[1]), int(parts[2]), blacklist):
                continue
            kept.append(line if line.endswith("\n") else line + "\n")
    return kept


def _write_peaks(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.writelines(lines)


def _filter_by_chroms(lines: list[str], chroms: set[str]) -> list[str]:
    return [l for l in lines if l.split("\t")[0] in chroms]


def main() -> None:
    args = _parse_args()
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    args.peaks_dir.mkdir(parents=True, exist_ok=True)

    canonical = _load_canonical_chroms(args.chrom_sizes)
    canonical_set = set(canonical)

    blacklist = _load_blacklist(args.blacklist) if args.blacklist else None

    print("Loading and filtering peaks...", flush=True)
    all_peaks = _load_peaks(args.peaks, canonical_set, blacklist)
    n_bl = sum(1 for l in all_peaks)
    print(f"  Kept {len(all_peaks)} peaks (blacklist-filtered on canonical chroms)", flush=True)

    # Write all-chromosome filtered peaks
    all_peaks_out = args.peaks_dir / "peaks.bed.gz"
    _write_peaks(all_peaks, all_peaks_out)
    print(f"  Wrote {all_peaks_out}", flush=True)

    folds = [args.fold] if args.fold else list(FOLD_SPLITS.keys())

    for fold in folds:
        test_chroms = FOLD_SPLITS[fold]["test"]
        val_chroms  = FOLD_SPLITS[fold]["val"]
        reserved    = set(test_chroms) | set(val_chroms)
        train_chroms = [c for c in canonical if c not in reserved]

        split_def = {
            "train":         train_chroms,
            "val":           val_chroms,
            "test":          test_chroms,
            "train_and_val": train_chroms + val_chroms,
        }

        json_path = args.splits_dir / f"fold{fold}.json"
        with json_path.open("w") as f:
            json.dump(split_def, f, indent=2)
        print(f"Fold {fold}: test={test_chroms}, val={val_chroms} → {json_path.name}", flush=True)

        for split_name, chroms in split_def.items():
            out = args.peaks_dir / f"peaks_fold{fold}_{split_name}.bed.gz"
            subset = _filter_by_chroms(all_peaks, set(chroms))
            _write_peaks(subset, out)
            print(f"  {out.name}: {len(subset)} peaks", flush=True)


if __name__ == "__main__":
    main()
