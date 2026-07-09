from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capybara.data import ProfileDataset, extract_loci


def _rc(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    return torch.flip(x, dims=(0, 1)), torch.flip(y, dims=(0, 1))


class FactorizedTrainDataset(IterableDataset):
    """Train stream mirroring ChromBPNet's ``ChromBPNetBatchGenerator`` (train mode).

    Each epoch (one ``__iter__`` call) the negatives are **resampled** to
    ``negative_sampling_ratio * n_peaks`` without replacement, concatenated with
    all peaks, and shuffled; peaks are random-cropped (jitter) and both are
    reverse-complemented with p=0.5. Negatives are loaded at ``input_length``
    exactly and are not jittered (matching ChromBPNet).

    Per-epoch resampling is driven by an internal epoch counter that increments
    on each ``__iter__``; with ``persistent_workers=True`` every worker advances
    its counter in lockstep and uses an epoch-only seed for the sampling/shuffle
    plan, so shards partition the same combined set with no overlap.
    """

    def __init__(
        self,
        *,
        peak_seqs: Tensor,
        peak_signals: Tensor,
        nonpeak_seqs: Tensor,
        nonpeak_signals: Tensor,
        input_length: int,
        output_length: int,
        max_jitter: int,
        negative_sampling_ratio: float,
        reverse_complement: bool,
        base_seed: int = 0,
    ) -> None:
        super().__init__()
        self.peak_seqs = peak_seqs
        self.peak_signals = peak_signals
        self.nonpeak_seqs = nonpeak_seqs
        self.nonpeak_signals = nonpeak_signals
        self.input_length = int(input_length)
        self.output_length = int(output_length)
        self.max_jitter = int(max_jitter)
        self.negative_sampling_ratio = float(negative_sampling_ratio)
        self.reverse_complement = bool(reverse_complement)
        self.base_seed = int(base_seed)
        self._epoch = 0

        self.n_peaks = int(peak_seqs.shape[0])
        self.n_nonpeaks = int(nonpeak_seqs.shape[0])
        self.n_neg = min(int(self.negative_sampling_ratio * self.n_peaks), self.n_nonpeaks)

    def __len__(self) -> int:
        return self.n_peaks + self.n_neg

    def _epoch_plan(self, epoch: int) -> list[tuple[int, int]]:
        plan_rng = np.random.RandomState(self.base_seed + epoch)
        neg_idx = plan_rng.choice(self.n_nonpeaks, size=self.n_neg, replace=False)
        items: list[tuple[int, int]] = [(1, i) for i in range(self.n_peaks)]
        items += [(0, int(j)) for j in neg_idx]
        order = plan_rng.permutation(len(items))
        return [items[k] for k in order]

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        epoch = self._epoch
        self._epoch += 1

        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers

        items = self._epoch_plan(epoch)
        items = items[worker_id::num_workers]
        aug_rng = np.random.RandomState((self.base_seed + epoch) * 100003 + worker_id + 1)

        for is_peak, idx in items:
            if is_peak:
                jitter = int(aug_rng.randint(0, 2 * self.max_jitter)) if self.max_jitter > 0 else 0
                x = self.peak_seqs[idx, :, jitter : jitter + self.input_length]
                y = self.peak_signals[idx, :, jitter : jitter + self.output_length]
            else:
                x = self.nonpeak_seqs[idx]
                y = self.nonpeak_signals[idx]

            if self.reverse_complement and aug_rng.rand() < 0.5:
                x, y = _rc(x, y)

            yield {"x": x.to(torch.float32), "y": y.to(torch.float32)}


class FactorizedDataModule:
    """Stage-2 datamodule: peaks + GC-matched negatives split by fold chromosomes.

    Train uses :class:`FactorizedTrainDataset` (per-epoch negative resampling +
    jitter + reverse-complement). Valid fixes its negative subset once (seeded)
    with jitter 0 and no reverse-complement, matching ChromBPNet's valid generator.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        batch_size: int,
        num_workers: int,
        prefetch_factor: int | None = 2,
        pin_memory: bool = True,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = prefetch_factor
        self.pin_memory = bool(pin_memory)
        self.verbose = bool(verbose)
        self.train_dataset: FactorizedTrainDataset | None = None
        self.valid_dataset: ProfileDataset | None = None

    def _extract(self, *, bed_path: str, chroms: list[str], max_jitter: int) -> tuple[Tensor, Tensor]:
        seqs, signals, _ = extract_loci(
            genome_path=self.config["genome_path"],
            chroms=chroms,
            bw_paths=[self.config["data_bw_path"]],
            bed_path=bed_path,
            input_length=int(self.config["input_length"]),
            output_length=int(self.config["output_length"]),
            max_jitter=max_jitter,
            summits=True,
            verbose=self.verbose,
        )
        return seqs, signals

    def setup(self) -> None:
        input_length = int(self.config["input_length"])
        output_length = int(self.config["output_length"])
        max_jitter = int(self.config["max_jitter"])
        ratio = float(self.config["negative_sampling_ratio"])
        seed = int(self.config.get("seed") or 0)

        # ---- train ----
        peak_seqs, peak_sigs = self._extract(
            bed_path=self.config["peaks_bed_path"],
            chroms=list(self.config["train_chroms"]),
            max_jitter=max_jitter,
        )
        neg_seqs, neg_sigs = self._extract(
            bed_path=self.config["nonpeaks_bed_path"],
            chroms=list(self.config["train_chroms"]),
            max_jitter=0,
        )
        self.train_dataset = FactorizedTrainDataset(
            peak_seqs=peak_seqs,
            peak_signals=peak_sigs,
            nonpeak_seqs=neg_seqs,
            nonpeak_signals=neg_sigs,
            input_length=input_length,
            output_length=output_length,
            max_jitter=max_jitter,
            negative_sampling_ratio=ratio,
            reverse_complement=bool(self.config["reverse_complement"]),
            base_seed=seed,
        )

        # ---- valid (fixed negatives, jitter 0, no RC) ----
        v_peak_seqs, v_peak_sigs = self._extract(
            bed_path=self.config["peaks_bed_path"],
            chroms=list(self.config["valid_chroms"]),
            max_jitter=0,
        )
        v_neg_seqs, v_neg_sigs = self._extract(
            bed_path=self.config["nonpeaks_bed_path"],
            chroms=list(self.config["valid_chroms"]),
            max_jitter=0,
        )
        n_val_neg = min(int(ratio * v_peak_seqs.shape[0]), int(v_neg_seqs.shape[0]))
        val_rng = np.random.RandomState(seed)
        keep = val_rng.choice(int(v_neg_seqs.shape[0]), size=n_val_neg, replace=False)
        keep_t = torch.as_tensor(np.sort(keep), dtype=torch.long)
        seqs = torch.cat([v_peak_seqs, v_neg_seqs.index_select(0, keep_t)], dim=0)
        signals = torch.cat([v_peak_sigs, v_neg_sigs.index_select(0, keep_t)], dim=0)
        self.valid_dataset = ProfileDataset(
            sequences=seqs,
            signals=signals,
            masks=None,
            input_length=input_length,
            output_length=output_length,
            max_jitter=0,
            reverse_complement=False,
            random_seed=seed,
        )
        if self.verbose:
            print(
                f"  train peaks={self.train_dataset.n_peaks} neg_pool={self.train_dataset.n_nonpeaks} "
                f"neg/epoch={self.train_dataset.n_neg} | valid peaks={int(v_peak_seqs.shape[0])} neg={n_val_neg}",
                flush=True,
            )

    def _loader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"num_workers": self.num_workers, "pin_memory": self.pin_memory}
        if self.num_workers > 0:
            # Persistent workers are required for the IterableDataset per-epoch
            # resampling to advance (the worker's epoch counter must survive).
            kwargs["persistent_workers"] = True
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return kwargs

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        # IterableDataset: no shuffle arg; drop_last trims uneven final batch.
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.valid_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **self._loader_kwargs(),
        )
