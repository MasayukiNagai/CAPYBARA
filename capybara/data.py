from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


DNA_ALPHABET = ["A", "C", "G", "T"]
DNA_IGNORE = [
    chr(code)
    for code in range(ord("A"), ord("Z") + 1)
    if chr(code) not in DNA_ALPHABET
] + ["-", "."]
_DNA_LOOKUP = np.full(256, 4, dtype=np.uint8)
_DNA_LOOKUP[ord("A")] = 0
_DNA_LOOKUP[ord("C")] = 1
_DNA_LOOKUP[ord("G")] = 2
_DNA_LOOKUP[ord("T")] = 3
_DNA_LOOKUP[ord("a")] = 0
_DNA_LOOKUP[ord("c")] = 1
_DNA_LOOKUP[ord("g")] = 2
_DNA_LOOKUP[ord("t")] = 3
__all__ = [
    "DNA_ALPHABET",
    "DNA_IGNORE",
    "ProfileDataset",
    "extract_loci",
    "load_chrom_names",
    "one_hot_encode",
]



def one_hot_encode(sequence: str, dtype: np.dtype = np.float32) -> np.ndarray:
    encoded = np.zeros((len(sequence), 4), dtype=dtype)
    if not sequence:
        return encoded

    bases = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    idx = _DNA_LOOKUP[bases]
    valid = idx < 4
    positions = np.nonzero(valid)[0]
    encoded[positions, idx[valid]] = 1
    return encoded


class ProfileDataset(Dataset):
    """PyTorch Dataset with per-sample jitter and reverse-complement augmentation.

    Accepts profile signals with any number of channels. Reverse-complement
    augmentation flips both channel and position axes, which is a no-op on the
    channel axis for single-channel unstranded signals.
    """

    def __init__(
        self,
        *,
        sequences: Tensor,
        signals: Tensor,
        masks: Tensor | None,
        input_length: int,
        output_length: int,
        max_jitter: int,
        reverse_complement: bool,
        random_seed: int | None = None,
    ) -> None:
        self.sequences = sequences
        self.signals = signals
        self.masks = masks
        self.input_length = int(input_length)
        self.output_length = int(output_length)
        self.max_jitter = int(max_jitter)
        self.reverse_complement = bool(reverse_complement)
        self.rng = np.random.RandomState(random_seed)
        self._check_shapes()

    def _check_shapes(self) -> None:
        expected_seq_len = self.input_length + 2 * self.max_jitter
        expected_signal_len = self.output_length + 2 * self.max_jitter
        if self.sequences.ndim != 3 or self.sequences.shape[1:] != (4, expected_seq_len):
            raise ValueError(f"Unexpected sequence shape {tuple(self.sequences.shape)}")
        if self.signals.ndim != 3 or self.signals.shape[-1] != expected_signal_len:
            raise ValueError(f"Unexpected signal shape {tuple(self.signals.shape)}")
        if self.masks is not None and tuple(self.masks.shape) != tuple(self.signals.shape):
            raise ValueError(
                f"Mask shape {tuple(self.masks.shape)} does not match signals "
                f"{tuple(self.signals.shape)}"
            )

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        jitter = int(self.rng.randint(0, 2 * self.max_jitter)) if self.max_jitter > 0 else 0

        x = self.sequences[idx, :, jitter : jitter + self.input_length]
        y = self.signals[idx, :, jitter : jitter + self.output_length]
        mask = (
            self.masks[idx, :, jitter : jitter + self.output_length]
            if self.masks is not None
            else None
        )

        if self.reverse_complement and self.rng.rand() < 0.5:
            x = torch.flip(x, dims=(0, 1))
            y = torch.flip(y, dims=(0, 1))
            if mask is not None:
                mask = torch.flip(mask, dims=(0, 1))

        item = {"x": x.to(torch.float32), "y": y.to(torch.float32)}
        if mask is not None:
            item["mask"] = mask.to(torch.bool)
        return item



def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def _iter_loci(
    path: str | Path,
    *,
    chroms: set[str],
    summits: bool,
) -> Iterator[tuple[str, int, int]]:
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"Expected at least 3 BED columns in {path}:{line_number}")

            chrom = fields[0]
            if chrom not in chroms:
                continue
            start = int(fields[1])
            end = int(fields[2])

            if summits:
                if len(fields) < 10:
                    raise ValueError(
                        f"summits=True requires BED10/narrowPeak input: {path}:{line_number}"
                    )
                summit = int(fields[9])
                if summit < 0:
                    raise ValueError(f"Summit cannot be negative in {path}:{line_number}")
                if start + summit > end:
                    raise ValueError(f"Summit + start exceeds end in {path}:{line_number}")
                width = end - start
                mid = start + summit
                start = mid - width // 2
                end = mid + width // 2

            yield chrom, start, end


class _BigWig:
    def __init__(self, path: str | Path) -> None:
        import pybigtools

        self.path = str(path)
        self.handle = pybigtools.open(self.path)

    def values(self, chrom: str, start: int, end: int) -> np.ndarray:
        try:
            values = self.handle.values(chrom, start, end, fillna=0)
        except Exception:
            values = np.zeros(end - start, dtype=np.float32)
        return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def close(self) -> None:
        close = getattr(self.handle, "close", None)
        if close is not None:
            close()


def extract_loci(
    *,
    genome_path: str | Path,
    chroms: list[str],
    bw_paths: list[str | Path],
    bed_path: str | Path,
    input_length: int,
    output_length: int,
    max_jitter: int = 0,
    mask_bw_path: str | Path | None = None,
    summits: bool = False,
    n_loci: int | None = None,
    verbose: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None]:
    from pyfaidx import Fasta

    input_window = int(input_length) + 2 * int(max_jitter)
    output_window = int(output_length) + 2 * int(max_jitter)
    half_input = input_window // 2
    half_output = output_window // 2

    fasta = Fasta(str(genome_path), sequence_always_upper=True)
    chrom_lengths = {str(key): len(value) for key, value in fasta.items()}
    allowed_chroms = set(chroms)
    bigwigs = [_BigWig(path) for path in bw_paths]
    mask_bw = _BigWig(mask_bw_path) if mask_bw_path is not None else None

    seqs: list[np.ndarray] = []
    signals: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    seen = 0
    skipped = 0

    try:
        for chrom, start, end in _iter_loci(bed_path, chroms=allowed_chroms, summits=summits):
            seen += 1
            mid = start + (end - start) // 2
            seq_start = mid - half_input
            seq_end = mid + half_input + (input_window % 2)
            sig_start = mid - half_output
            sig_end = mid + half_output + (output_window % 2)

            chrom_length = chrom_lengths.get(chrom)
            if (
                chrom_length is None
                or seq_start < 0
                or sig_start < 0
                or seq_end > chrom_length
                or sig_end > chrom_length
            ):
                skipped += 1
                continue

            seq = one_hot_encode(str(fasta[chrom][seq_start:seq_end])).T
            signal = np.stack([bw.values(chrom, sig_start, sig_end) for bw in bigwigs])
            mask = None
            if mask_bw is not None:
                mask_values = mask_bw.values(chrom, sig_start, sig_end)
                mask = np.stack([(mask_values > 0)] * len(bigwigs))

            if seq.shape != (4, input_window) or signal.shape != (len(bigwigs), output_window):
                skipped += 1
                continue
            if mask is not None and mask.shape != signal.shape:
                skipped += 1
                continue

            seqs.append(seq)
            signals.append(signal)
            if mask is not None:
                masks.append(mask)
            if n_loci is not None and len(seqs) >= int(n_loci):
                break
    finally:
        fasta.close()
        for bw in bigwigs:
            bw.close()
        if mask_bw is not None:
            mask_bw.close()

    if not seqs:
        raise RuntimeError(f"No valid loci extracted from {bed_path}")

    seq_tensor = torch.from_numpy(np.stack(seqs)).to(torch.float32)
    signal_tensor = torch.from_numpy(np.stack(signals)).to(torch.float32)
    mask_tensor = (
        torch.from_numpy(np.stack(masks)).to(torch.bool)
        if mask_bw_path is not None
        else None
    )

    if verbose:
        print(
            f"Extracted {seq_tensor.shape[0]} examples from {bed_path}; "
            f"seen={seen}, skipped={skipped}, seqs={tuple(seq_tensor.shape)}, "
            f"signals={tuple(signal_tensor.shape)}, "
            f"masks={tuple(mask_tensor.shape) if mask_tensor is not None else None}",
            flush=True,
        )

    return seq_tensor, signal_tensor, mask_tensor


def load_chrom_names(
    chrom_sizes: str | Path,
    filter_out: Iterable[str] = ("_", "M", "Un", "EBV"),
) -> list[str]:
    chroms = []
    with Path(chrom_sizes).open() as handle:
        for line in handle:
            chrom = line.strip().split()[0]
            if not chrom.startswith("chr"):
                continue
            if any(token in chrom for token in filter_out):
                continue
            chroms.append(chrom)
    return chroms
