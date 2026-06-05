from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


VALID_CELL_TYPES = {"K562", "HepG2", "GM12878", "IMR90", "H1-hESC"}


@dataclass(frozen=True)
class AtacFoldFilesConfig:
    proj_dir: Path
    cell_type: str
    fold: int
    model_name: str
    timestamp: str

    @classmethod
    def create(
        cls,
        *,
        proj_dir: str | Path,
        cell_type: str = "K562",
        fold: int | str = 1,
        model_name: str,
        timestamp: str | None = None,
    ) -> "AtacFoldFilesConfig":
        cell_type = str(cell_type)
        if cell_type not in VALID_CELL_TYPES:
            raise ValueError(f"Unsupported cell_type={cell_type!r}. Expected one of {sorted(VALID_CELL_TYPES)}.")

        fold = int(fold)
        if fold < 1 or fold > 5:
            raise ValueError(f"fold must be in [1, 5], got {fold}.")

        timestamp = timestamp or datetime.now().strftime("%y%m%d_%H%M%S")
        cfg = cls(
            proj_dir=Path(proj_dir),
            cell_type=cell_type,
            fold=fold,
            model_name=str(model_name),
            timestamp=timestamp,
        )
        cfg.validate()
        return cfg

    @property
    def genome_path(self) -> Path:
        return self.proj_dir / "genome" / "hg38.withrDNA.fasta"

    @property
    def chrom_size_path(self) -> Path:
        return self.proj_dir / "genome" / "hg38.withrDNA.chrom.sizes"

    @property
    def processed_dir(self) -> Path:
        return self.proj_dir / "data" / "atac" / "processed" / self.cell_type

    @property
    def all_peak_path(self) -> Path:
        return self.processed_dir / "peaks.bed.gz"

    @property
    def atac_bw_path(self) -> Path:
        return self.processed_dir / "atac.bw"

    @property
    def rep1_bw_path(self) -> Path:
        return self.processed_dir / "rep1.bw"

    @property
    def rep2_bw_path(self) -> Path:
        return self.processed_dir / "rep2.bw"

    @property
    def train_peak_path(self) -> Path:
        return self.processed_dir / f"peaks_fold{self.fold}_train.bed.gz"

    @property
    def val_peak_path(self) -> Path:
        return self.processed_dir / f"peaks_fold{self.fold}_val.bed.gz"

    @property
    def test_peak_path(self) -> Path:
        return self.processed_dir / f"peaks_fold{self.fold}_test.bed.gz"

    @property
    def train_val_peak_path(self) -> Path:
        return self.processed_dir / f"peaks_fold{self.fold}_train_and_val.bed.gz"

    @property
    def model_dir(self) -> Path:
        return self.proj_dir / "models" / self.model_name / "atac" / self.cell_type / f"fold{self.fold}" / self.timestamp

    @property
    def checkpoint_dir(self) -> Path:
        return self.model_dir / "checkpoints"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "last.pt"

    @property
    def metrics_path(self) -> Path:
        return self.model_dir / "metrics.tsv"

    @property
    def params_path(self) -> Path:
        return self.model_dir / "params_saved.yaml"

    @property
    def config_path(self) -> Path:
        return self.model_dir / "config_saved.yaml"

    @property
    def eval_dir(self) -> Path:
        return self.model_dir / "evals"

    def validate(self) -> None:
        required = [
            self.genome_path,
            self.chrom_size_path,
            self.all_peak_path,
            self.atac_bw_path,
            self.train_peak_path,
            self.val_peak_path,
            self.test_peak_path,
            self.train_val_peak_path,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            joined = "\n  ".join(missing)
            raise FileNotFoundError(f"Missing required ATAC file(s):\n  {joined}")

    def validate_replicates(self) -> None:
        missing = [str(p) for p in [self.rep1_bw_path, self.rep2_bw_path] if not p.exists()]
        if missing:
            joined = "\n  ".join(missing)
            raise FileNotFoundError(f"Missing pseudoreplicate BigWig(s) (needed for evaluate.py):\n  {joined}")

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "proj_dir": str(self.proj_dir),
            "cell_type": self.cell_type,
            "data_type": "atac",
            "fold": self.fold,
            "model_name": self.model_name,
            "genome_path": str(self.genome_path),
            "chrom_size_path": str(self.chrom_size_path),
            "all_peak_path": str(self.all_peak_path),
            "atac_bw_path": str(self.atac_bw_path),
            "rep1_bw_path": str(self.rep1_bw_path),
            "rep2_bw_path": str(self.rep2_bw_path),
            "train_peak_path": str(self.train_peak_path),
            "val_peak_path": str(self.val_peak_path),
            "test_peak_path": str(self.test_peak_path),
            "train_val_peak_path": str(self.train_val_peak_path),
            "model_dir": str(self.model_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "last_checkpoint_path": str(self.last_checkpoint_path),
            "metrics_path": str(self.metrics_path),
            "params_path": str(self.params_path),
            "config_path": str(self.config_path),
            "eval_dir": str(self.eval_dir),
        }
