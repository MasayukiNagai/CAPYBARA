from __future__ import annotations

"""PRO-seq / gene-body data module for CAPY.

This mirrors ``examples/procap/data.py`` but replaces the PRO-cap-style locus
extraction (symmetric window centered on the peak midpoint, both strands) with
the PRO-seq / ag_mini semantics:

* **single strand** — only the strand matching the locus is read, so signals
  have shape ``(1, L)`` and the model uses ``profile_head.num_outputs: 1``.
* **TSS-anchored, asymmetric windows** — windows are placed relative to the TSS
  coordinate in the BED (column ``start`` for ``+`` loci, ``end`` for ``-``),
  using explicit offsets ``in_window`` / ``out_window`` (e.g. ``[-2000, 4000]``
  and ``[0, 2000]``). They are NOT centered on the midpoint.
* **transcription orientation** — minus-strand loci have their sequence
  reverse-complemented and their signal reversed, so every example is read
  5'->3' along transcription.

The extraction logic is ported from
``ProCapNet/src/2_train_models/data_loading_genebody_strand.py::extract_peaks``.
The ``ProfileDataset`` / ``ProSeqDataModule`` keep the same public interface as
the PRO-cap versions, so ``examples/procap/train_utils.py`` drives them
unchanged (batches are ``{"x": (B,4,Lin), "y": (B,1,Lout)}`` dicts).
"""

import gzip
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler


def one_hot_encode(sequence: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """One-hot encode a DNA string to shape ``(len, 4)`` with A,C,G,T order."""
    sequence = sequence.upper()
    encoded = np.zeros((len(sequence), 4), dtype=dtype)
    lookup = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, base in enumerate(sequence):
        j = lookup.get(base)
        if j is not None:
            encoded[i, j] = 1
    return encoded


def load_chrom_names(chrom_sizes: str | Path, filter_out: Iterable[str] = ("_", "M", "Un", "EBV")) -> list[str]:
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


def load_bed_lines(path: str | Path) -> list[list[str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode) as handle:
        return [line.strip().split() for line in handle if line.strip()]


def load_stranded_bed(path: str | Path, strand_col: int = 4) -> dict[str, np.ndarray]:
    """Load a stranded TSS BED.

    Columns 0,1,2 are (chrom, start, end); column ``strand_col`` (default 4) is
    the strand. Rows are typically 1-bp TSS anchors. Matches the columns read by
    ProCapNet's ``extract_peaks`` (``usecols=(0, 1, 2, 4)``).
    """
    chroms, starts, ends, strands = [], [], [], []
    for fields in load_bed_lines(path):
        if len(fields) <= strand_col:
            raise ValueError(
                f"BED row has {len(fields)} columns but strand is expected at index "
                f"{strand_col}: {path}"
            )
        chroms.append(fields[0])
        starts.append(int(fields[1]))
        ends.append(int(fields[2]))
        strands.append(fields[strand_col])
    return {
        "chrom": np.asarray(chroms),
        "start": np.asarray(starts, dtype=int),
        "end": np.asarray(ends, dtype=int),
        "strand": np.asarray(strands),
    }


def _open_bigwig(path: str | Path):
    import pyBigWig

    bw = pyBigWig.open(str(path), "r")
    if bw is None:
        raise OSError(f"Could not open BigWig: {path}")
    return bw


def _get_signal(bw, chrom: str, start: int, end: int) -> np.ndarray:
    values = bw.values(chrom, start, end, numpy=True)
    return np.nan_to_num(values).astype(np.float32, copy=False)


def _extract_one_locus_stranded(
    *,
    fasta,
    plus_bw,
    minus_bw,
    chrom: str,
    start: int,
    end: int,
    strand: str,
    in_window: tuple[int, int],
    out_window: tuple[int, int],
    max_jitter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one strand-oriented, TSS-anchored locus.

    Returns (seq, signal):
        seq    : (4, (in_window[1]-in_window[0]) + 2*max_jitter)
        signal : (1, (out_window[1]-out_window[0]) + 2*max_jitter)
    Both oriented 5'->3' along transcription (minus strand revcomp'd / reversed).
    """
    J = int(max_jitter)

    if strand == "+":
        seq_start = start + in_window[0] - J
        seq_end = start + in_window[1] + J
        sig_start = start + out_window[0] - J
        sig_end = start + out_window[1] + J
        if seq_start < 0 or sig_start < 0:
            raise ValueError(f"Negative window start for {chrom}:{start}-{end} (+)")
        seq = one_hot_encode(str(fasta[chrom][seq_start:seq_end])).T  # (4, Lin)
        signal = _get_signal(plus_bw, chrom, sig_start, sig_end)       # (Lsig,)
    elif strand == "-":
        seq_start = end - in_window[1] - J
        seq_end = end - in_window[0] + J
        sig_start = end - out_window[1] - J
        sig_end = end - out_window[0] + J
        if seq_start < 0 or sig_start < 0:
            raise ValueError(f"Negative window start for {chrom}:{start}-{end} (-)")
        seq = one_hot_encode(str(fasta[chrom][seq_start:seq_end])).T  # (4, Lin)
        # reverse-complement: reverse channel order (A<->T, C<->G) and positions
        seq = seq[::-1, ::-1]
        signal = _get_signal(minus_bw, chrom, sig_start, sig_end)[::-1]  # reversed
    else:
        raise ValueError(f"Invalid strand {strand!r} for {chrom}:{start}-{end}; must be '+' or '-'.")

    # Force positive signal (ProCapNet fix for negative-strand bigwig values).
    signal = np.absolute(signal).astype(np.float32, copy=False)
    signal = signal.reshape(1, -1)

    expected_seq_len = (in_window[1] - in_window[0]) + 2 * J
    expected_sig_len = (out_window[1] - out_window[0]) + 2 * J
    if seq.shape != (4, expected_seq_len):
        raise ValueError(f"Unexpected sequence shape {seq.shape}; expected (4, {expected_seq_len})")
    if signal.shape != (1, expected_sig_len):
        raise ValueError(f"Unexpected signal shape {signal.shape}; expected (1, {expected_sig_len})")
    return np.ascontiguousarray(seq), np.ascontiguousarray(signal)


def extract_loci_stranded(
    *,
    genome_path: str | Path,
    chroms: list[str],
    plus_bw_path: str | Path,
    minus_bw_path: str | Path,
    bed_path: str | Path,
    in_window: tuple[int, int],
    out_window: tuple[int, int],
    max_jitter: int,
    verbose: bool = True,
) -> tuple[Tensor, Tensor]:
    from pyfaidx import Fasta
    from tqdm import tqdm

    fasta = Fasta(str(genome_path), sequence_always_upper=True)
    plus_bw = _open_bigwig(plus_bw_path)
    minus_bw = _open_bigwig(minus_bw_path)

    loci = load_stranded_bed(bed_path)
    keep = np.isin(loci["chrom"], chroms)
    kept_chroms = loci["chrom"][keep]
    starts = loci["start"][keep]
    ends = loci["end"][keep]
    strands = loci["strand"][keep]
    if verbose:
        print(f"Loaded {len(kept_chroms)} loci from {bed_path}")

    seqs, signals = [], []
    n_skipped = 0
    try:
        iterator = zip(kept_chroms, starts, ends, strands)
        for chrom, start, end, strand in tqdm(
            iterator, total=len(kept_chroms), disable=not verbose, desc="Extracting loci"
        ):
            try:
                seq, signal = _extract_one_locus_stranded(
                    fasta=fasta,
                    plus_bw=plus_bw,
                    minus_bw=minus_bw,
                    chrom=str(chrom),
                    start=int(start),
                    end=int(end),
                    strand=str(strand),
                    in_window=in_window,
                    out_window=out_window,
                    max_jitter=max_jitter,
                )
            except (ValueError, RuntimeError) as exc:
                n_skipped += 1
                if verbose:
                    print(f"Skipping {chrom}:{start}-{end} ({strand}): {exc}")
                continue
            seqs.append(seq)
            signals.append(signal)
    finally:
        fasta.close()
        plus_bw.close()
        minus_bw.close()

    if not seqs:
        raise RuntimeError(f"No valid loci extracted from {bed_path}")

    seq_tensor = torch.from_numpy(np.stack(seqs)).to(torch.float32)
    signal_tensor = torch.from_numpy(np.stack(signals)).to(torch.float32)
    if verbose:
        print(
            f"Extracted {seq_tensor.shape[0]} examples "
            f"({n_skipped} skipped): seqs={tuple(seq_tensor.shape)}, "
            f"signals={tuple(signal_tensor.shape)}"
        )
    return seq_tensor, signal_tensor


class ProfileDataset(Dataset):
    """Single-strand profile dataset with jitter + reverse-complement augmentation.

    Mirrors ``examples/procap/data.py::ProfileDataset`` but the signal strand
    dimension is ``num_outputs`` (1 here) rather than a fixed 2.
    """

    def __init__(
        self,
        *,
        sequences: Tensor,
        signals: Tensor,
        input_length: int,
        output_length: int,
        max_jitter: int,
        reverse_complement: bool,
        num_outputs: int = 1,
        random_seed: int | None = None,
    ) -> None:
        self.sequences = sequences
        self.signals = signals
        self.input_length = int(input_length)
        self.output_length = int(output_length)
        self.max_jitter = int(max_jitter)
        self.reverse_complement = bool(reverse_complement)
        self.num_outputs = int(num_outputs)
        self.rng = np.random.RandomState(random_seed)
        self._check_shapes()

    def _check_shapes(self) -> None:
        expected_seq_len = self.input_length + 2 * self.max_jitter
        expected_signal_len = self.output_length + 2 * self.max_jitter
        if self.sequences.ndim != 3 or self.sequences.shape[1:] != (4, expected_seq_len):
            raise ValueError(f"Unexpected sequence shape {tuple(self.sequences.shape)}")
        if self.signals.ndim != 3 or self.signals.shape[1:] != (self.num_outputs, expected_signal_len):
            raise ValueError(
                f"Unexpected signal shape {tuple(self.signals.shape)}; "
                f"expected (N, {self.num_outputs}, {expected_signal_len})"
            )

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if self.max_jitter == 0:
            jitter = 0
        else:
            jitter = int(self.rng.randint(0, 2 * self.max_jitter))

        x = self.sequences[idx, :, jitter : jitter + self.input_length]
        y = self.signals[idx, :, jitter : jitter + self.output_length]

        if self.reverse_complement and np.random.rand() < 0.5:
            # flip channel + length; for a single strand the channel flip on a
            # size-1 dim is a no-op, the length flip reverses the profile.
            x = torch.flip(x, dims=(0, 1))
            y = torch.flip(y, dims=(0, 1))

        return {"x": x.to(torch.float32), "y": y.to(torch.float32)}


class MultiSourceBatchSampler(Sampler[list[int]]):
    """Yield batches mixing several datasets at fixed per-batch fractions.

    Ported verbatim from ``examples/procap/data.py``. The first source (positives)
    is the "primary": it is iterated by a ``randperm`` so each item is seen ~once per
    epoch, and the epoch length is ``len(primary) // batch_sizes[0]``. The remaining
    sources (negatives) are sampled *with replacement* via ``torch.randint`` to fill
    their per-batch slots, so they need not match the primary in count.
    """

    def __init__(self, dataset_lengths: list[int], source_fracs: list[float], batch_size: int, seed: int | None = None) -> None:
        if len(dataset_lengths) != len(source_fracs):
            raise ValueError("dataset_lengths and source_fracs must have the same length.")
        if abs(sum(source_fracs) - 1.0) > 1e-6:
            raise ValueError(f"source_fracs must sum to 1, got {sum(source_fracs)}.")
        self.dataset_lengths = [int(length) for length in dataset_lengths]
        self.source_fracs = [float(frac) for frac in source_fracs]
        self.batch_size = int(batch_size)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(int(seed))

        self.batch_sizes = [int(torch.ceil(torch.tensor(self.batch_size * frac)).item()) for frac in self.source_fracs]
        overflow = sum(self.batch_sizes) - self.batch_size
        self.batch_sizes[0] -= overflow
        if self.batch_sizes[0] < 1:
            raise ValueError(f"Primary source batch size must be at least 1, got {self.batch_sizes[0]}.")

        self.offsets = [0]
        for length in self.dataset_lengths[:-1]:
            self.offsets.append(self.offsets[-1] + length)
        self.num_batches = self.dataset_lengths[0] // self.batch_sizes[0]

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        primary = torch.randperm(self.dataset_lengths[0], generator=self.generator)
        primary = primary[: self.num_batches * self.batch_sizes[0]].view(self.num_batches, self.batch_sizes[0])

        for batch_idx in range(self.num_batches):
            parts = [primary[batch_idx]]
            for source_idx in range(1, len(self.dataset_lengths)):
                start = self.offsets[source_idx]
                end = start + self.dataset_lengths[source_idx]
                count = self.batch_sizes[source_idx]
                parts.append(torch.randint(start, end, (count,), generator=self.generator))
            batch = torch.cat(parts)
            yield batch[torch.randperm(batch.numel(), generator=self.generator)].tolist()


class ProSeqDataModule:
    """PRO-seq data module with optional negative-region mixing (no mask).

    Public interface matches ``examples/procap/data.py::ProCapDataModule`` so
    ``examples/procap/train_utils.run_training_loop`` consumes it unchanged.

    When ``use_negatives`` is set, negative-region loci (``neg_train_path``) are mixed
    into each *training* batch at ``source_fracs`` via ``MultiSourceBatchSampler``,
    mirroring procap's DNase-negative handling. Validation stays positives-only.

    Expected ``config`` keys: genome_path, chrom_size_path, plus_bw_path,
    minus_bw_path, train_peak_path, val_peak_path, input_length, output_length,
    in_window, out_window, max_jitter, reverse_complement, random_seed,
    num_outputs. Optional: use_negatives, source_fracs, neg_train_path.
    """

    def __init__(
        self,
        *,
        config: dict,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int | None = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = prefetch_factor
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self.verbose = bool(verbose)
        self.chroms = load_chrom_names(config["chrom_size_path"])
        self.num_outputs = int(config.get("num_outputs", 1))
        self.train_dataset = None
        self.valid_dataset = None

    def setup(self) -> None:
        self.train_dataset = self._make_train_dataset()
        self.valid_dataset = self._make_dataset(
            self.config["val_peak_path"], jitter=False, reverse_complement=False
        )

    def _make_train_dataset(self):
        peak_dataset = self._make_dataset(
            self.config["train_peak_path"], jitter=True, reverse_complement=bool(self.config["reverse_complement"])
        )
        if not self.config.get("use_negatives"):
            return peak_dataset
        neg_dataset = self._make_dataset(
            self.config["neg_train_path"], jitter=True, reverse_complement=bool(self.config["reverse_complement"])
        )
        return [peak_dataset, neg_dataset]

    def _make_dataset(self, bed_path: str, *, jitter: bool, reverse_complement: bool) -> ProfileDataset:
        max_jitter = int(self.config["max_jitter"]) if jitter else 0
        in_window = tuple(int(v) for v in self.config["in_window"])
        out_window = tuple(int(v) for v in self.config["out_window"])
        seqs, signals = extract_loci_stranded(
            genome_path=self.config["genome_path"],
            chroms=self.chroms,
            plus_bw_path=self.config["plus_bw_path"],
            minus_bw_path=self.config["minus_bw_path"],
            bed_path=bed_path,
            in_window=in_window,
            out_window=out_window,
            max_jitter=max_jitter,
            verbose=self.verbose,
        )
        return ProfileDataset(
            sequences=seqs,
            signals=signals,
            input_length=int(self.config["input_length"]),
            output_length=int(self.config["output_length"]),
            max_jitter=max_jitter,
            reverse_complement=reverse_complement,
            num_outputs=self.num_outputs,
            random_seed=self.config.get("random_seed"),
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        if self.valid_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._loader(self.valid_dataset, shuffle=False, drop_last=False)

    def _loader(self, dataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)

        if isinstance(dataset, list):
            sampler = MultiSourceBatchSampler(
                dataset_lengths=[len(ds) for ds in dataset],
                source_fracs=list(self.config["source_fracs"]),
                batch_size=self.batch_size,
                seed=self.config.get("random_seed"),
            )
            return DataLoader(ConcatDataset(dataset), batch_sampler=sampler, **kwargs)

        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs
        )
