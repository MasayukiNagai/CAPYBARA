from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from capybara.data import ProfileDataset, extract_loci, load_chrom_names


class AtacDataModule:
    """DataModule for unstranded ATAC-seq signal from a single BigWig."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
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
        self.train_dataset: ProfileDataset | None = None
        self.valid_dataset: ProfileDataset | None = None

    def setup(self) -> None:
        self.train_dataset = self._make_dataset(
            self.config["train_peak_path"],
            jitter=True,
            reverse_complement=bool(self.config["reverse_complement"]),
        )
        self.valid_dataset = self._make_dataset(
            self.config["val_peak_path"],
            jitter=False,
            reverse_complement=False,
        )

    def _make_dataset(self, bed_path: str, *, jitter: bool, reverse_complement: bool) -> ProfileDataset:
        max_jitter = int(self.config["max_jitter"]) if jitter else 0
        seqs, signals, _ = extract_loci(
            genome_path=self.config["genome_path"],
            chroms=self.chroms,
            bw_paths=[self.config["atac_bw_path"]],
            bed_path=bed_path,
            input_length=int(self.config["input_length"]),
            output_length=int(self.config["output_length"]),
            max_jitter=max_jitter,
            summits=True,
            verbose=self.verbose,
        )
        return ProfileDataset(
            sequences=seqs,
            signals=signals,
            masks=None,
            input_length=int(self.config["input_length"]),
            output_length=int(self.config["output_length"]),
            max_jitter=max_jitter,
            reverse_complement=reverse_complement,
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

    def _loader(self, dataset: ProfileDataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        kwargs: dict = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)
