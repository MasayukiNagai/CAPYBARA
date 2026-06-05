from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


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


class ProfileDataset(Dataset):
    """PyTorch Dataset with per-sample jitter and reverse-complement augmentation.

    Accepts signals with any number of channels (e.g. 1 for unstranded ATAC,
    2 for stranded PRO-cap). RC augmentation flips both channel and position
    axes, which is a no-op on the single-channel axis for unstranded signals.
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
                f"Mask shape {tuple(self.masks.shape)} does not match signals {tuple(self.signals.shape)}"
            )

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        jitter = int(self.rng.randint(0, 2 * self.max_jitter)) if self.max_jitter > 0 else 0

        x = self.sequences[idx, :, jitter : jitter + self.input_length]
        y = self.signals[idx, :, jitter : jitter + self.output_length]
        mask = self.masks[idx, :, jitter : jitter + self.output_length] if self.masks is not None else None

        if self.reverse_complement and np.random.rand() < 0.5:
            x = torch.flip(x, dims=(0, 1))
            y = torch.flip(y, dims=(0, 1))
            if mask is not None:
                mask = torch.flip(mask, dims=(0, 1))

        item = {"x": x.to(torch.float32), "y": y.to(torch.float32)}
        if mask is not None:
            item["mask"] = mask.to(torch.bool)
        return item
