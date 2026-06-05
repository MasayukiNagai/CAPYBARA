from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Sampler

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from shared.data import ProfileDataset, load_chrom_names

__all__ = [
    "ProfileDataset",
    "extract_loci",
    "load_chrom_names",
    "MultiSourceBatchSampler",
    "ProCapDataModule",
]


def _extract_mask_signal(
    bw_path: str | Path,
    bed_path: str | Path,
    chroms: list[str],
    kept_mask: torch.Tensor,
    output_window: int,
    n_channels: int,
) -> Tensor:
    import pandas
    import pybigtools

    bw = pybigtools.open(str(bw_path))
    loci = pandas.read_csv(
        str(bed_path), sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"]
    )
    loci = loci[np.isin(loci["chrom"], chroms)].reset_index(drop=True)
    loci = loci[kept_mask.numpy()].reset_index(drop=True)

    half = output_window // 2
    odd = output_window % 2

    masks = []
    for _, row in loci.iterrows():
        mid = int(row["start"]) + (int(row["end"]) - int(row["start"])) // 2
        start = mid - half
        end = mid + half + odd
        try:
            values = np.array(bw.values(row["chrom"], start, end, fillna=0), dtype=np.float32)
        except Exception:
            values = np.zeros(output_window, dtype=np.float32)
        values = np.nan_to_num(values)
        one_strand = (values > 0)
        masks.append(np.stack([one_strand] * n_channels))

    return torch.from_numpy(np.stack(masks))


def extract_loci(
    *,
    genome_path: str | Path,
    chroms: list[str],
    bw_paths: list[str | Path],
    bed_path: str | Path,
    mask_bw_path: str | Path | None = None,
    input_length: int,
    output_length: int,
    max_jitter: int,
    verbose: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None]:
    from tangermeme.io import extract_loci as _tg_extract_loci

    in_window = input_length + 2 * max_jitter
    out_window = output_length + 2 * max_jitter

    result = _tg_extract_loci(
        loci=str(bed_path),
        sequences=str(genome_path),
        signals=[str(p) for p in bw_paths],
        chroms=chroms,
        in_window=in_window,
        out_window=out_window,
        max_jitter=0,
        return_mask=(mask_bw_path is not None),
        verbose=verbose,
    )

    if mask_bw_path is not None:
        seqs, signals, kept_mask = result
        mask = _extract_mask_signal(
            bw_path=mask_bw_path,
            bed_path=bed_path,
            chroms=chroms,
            kept_mask=kept_mask,
            output_window=out_window,
            n_channels=int(signals.shape[1]),
        )
        return seqs, signals, mask
    else:
        seqs, signals = result
        return seqs, signals, None


class MultiSourceBatchSampler(Sampler[list[int]]):
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


class ProCapDataModule:
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
        self.train_dataset = None
        self.valid_dataset = None

    def setup(self) -> None:
        self.train_dataset = self._make_train_dataset()
        self.valid_dataset = self._make_dataset(
            self.config["val_peak_path"],
            mask=False,
            jitter=False,
            reverse_complement=False,
        )

    def _make_dataset(self, bed_path: str, *, mask: bool, jitter: bool, reverse_complement: bool) -> ProfileDataset:
        max_jitter = int(self.config["max_jitter"]) if jitter else 0
        seqs, signals, masks = extract_loci(
            genome_path=self.config["genome_path"],
            chroms=self.chroms,
            bw_paths=[self.config["plus_bw_path"], self.config["minus_bw_path"]],
            bed_path=bed_path,
            mask_bw_path=self.config["mask_bw_path"] if mask else None,
            input_length=int(self.config["input_length"]),
            output_length=int(self.config["output_length"]),
            max_jitter=max_jitter,
            verbose=self.verbose,
        )
        return ProfileDataset(
            sequences=seqs,
            signals=signals,
            masks=masks,
            input_length=int(self.config["input_length"]),
            output_length=int(self.config["output_length"]),
            max_jitter=max_jitter,
            reverse_complement=reverse_complement,
            random_seed=self.config.get("random_seed"),
        )

    def _make_train_dataset(self):
        peak_dataset = self._make_dataset(
            self.config["train_peak_path"],
            mask=bool(self.config["mask_bw_path"]),
            jitter=True,
            reverse_complement=bool(self.config["reverse_complement"]),
        )
        if not self.config["use_dnase"]:
            return peak_dataset
        dnase_dataset = self._make_dataset(
            self.config["dnase_train_path"],
            mask=bool(self.config["mask_bw_path"]),
            jitter=True,
            reverse_complement=bool(self.config["reverse_complement"]),
        )
        return [peak_dataset, dnase_dataset]

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        if self.valid_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._loader(self.valid_dataset, shuffle=False, drop_last=False)

    def _loader(self, dataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        kwargs: dict = {
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

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)
