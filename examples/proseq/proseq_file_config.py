from __future__ import annotations

"""File-path configuration for the PRO-seq / Rogers-timecourse workflow.

This is the PRO-seq analogue of ``examples/procap/file_config.py::FoldFilesConfig``.
The PRO-cap version hard-codes a single ``proj_dir`` layout, a cell-type
whitelist, integer folds in ``[1, 7]``, and requires PRO-cap + DNase + mask +
train_and_val files. The PRO-seq data does not follow that layout: genome,
BigWigs, and peak BEDs live in three different trees, folds are strings (``fold1``),
the "cell type" is a GEO timepoint sample (``treatment``), and there are no
DNase/mask/train_and_val files.

``ProSeqFilesConfig`` therefore takes explicit input directories (with defaults
pointing at the known Rogers locations) plus a ``treatment`` and string ``fold``
to build the BigWig and peak filenames, and uses ``proj_dir`` only as the output
root for trained models.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# --- Default input locations (PRO-seq / Rogers timecourse) ---
DEFAULT_GENOME_PATH = "/grid/koo/home/ykang/elongation/reference/samtools/GRCh38.p13.genome.fa"
DEFAULT_CHROM_SIZE_PATH = "/grid/koo/home/ykang/elongation/reference/samtools/GRCh38.p13.genome.chrominfo.txt"
DEFAULT_BIGWIG_DIR = "/grid/koo/home/ykang/elongation/data_hunting/rogers_timecourse_data"
DEFAULT_PEAK_DIR = (
    "/grid/koo/home/ykang/elongation/data_hunting/"
    "split_FANTOM5_TSS_rogers_all_times_top_counts_per_gene"
)
# Peak BED naming: ref_all_timepoints_top_count_per_gene_{fold}_{split}.bed.gz
PEAK_PREFIX = "ref_all_timepoints_top_count_per_gene"

# Negative-region BEDs (PRO-seq analogue of procap's DNase negatives). Same per-fold
# split layout as the positive peaks:
# rogers_negative_samples_{fold}_{split}.bed.gz
DEFAULT_NEG_DIR = (
    "/grid/koo/home/ykang/elongation/ProCapNet/Rogers_negative_samples/"
    "split_ref_condition_negative_samples"
)
NEG_PREFIX = "rogers_negative_samples"


@dataclass(frozen=True)
class ProSeqFilesConfig:
    proj_dir: Path                 # output root (models/ written here)
    treatment: str                 # GEO sample / timepoint, e.g. "GSM8306530_0m"
    fold: str                      # string fold, e.g. "fold1"
    model_name: str                # "capy"
    data_type: str                 # namespace for output dir, e.g. "proseq"
    timestamp: str
    genome_path: Path
    chrom_size_path: Path
    bigwig_dir: Path
    peak_dir: Path
    neg_dir: Path
    require_negatives: bool

    @classmethod
    def create(
        cls,
        *,
        proj_dir: str | Path,
        treatment: str = "GSM8306530_0m",
        fold: str = "fold1",
        model_name: str = "capy",
        data_type: str = "proseq",
        timestamp: str | None = None,
        genome_path: str | Path = DEFAULT_GENOME_PATH,
        chrom_size_path: str | Path = DEFAULT_CHROM_SIZE_PATH,
        bigwig_dir: str | Path = DEFAULT_BIGWIG_DIR,
        peak_dir: str | Path = DEFAULT_PEAK_DIR,
        neg_dir: str | Path = DEFAULT_NEG_DIR,
        require_negatives: bool = False,
    ) -> "ProSeqFilesConfig":
        timestamp = timestamp or datetime.now().strftime("%y%m%d_%H%M%S")
        cfg = cls(
            proj_dir=Path(proj_dir),
            treatment=str(treatment),
            fold=str(fold),
            model_name=str(model_name),
            data_type=str(data_type),
            timestamp=timestamp,
            genome_path=Path(genome_path),
            chrom_size_path=Path(chrom_size_path),
            bigwig_dir=Path(bigwig_dir),
            peak_dir=Path(peak_dir),
            neg_dir=Path(neg_dir),
            require_negatives=bool(require_negatives),
        )
        cfg.validate()
        return cfg

    # --- inputs ---
    @property
    def plus_bw_path(self) -> Path:
        return self.bigwig_dir / f"{self.treatment}_F.bw"

    @property
    def minus_bw_path(self) -> Path:
        return self.bigwig_dir / f"{self.treatment}_R.bw"

    @property
    def train_peak_path(self) -> Path:
        return self.peak_dir / f"{PEAK_PREFIX}_{self.fold}_train.bed.gz"

    @property
    def val_peak_path(self) -> Path:
        return self.peak_dir / f"{PEAK_PREFIX}_{self.fold}_val.bed.gz"

    @property
    def test_peak_path(self) -> Path:
        return self.peak_dir / f"{PEAK_PREFIX}_{self.fold}_test.bed.gz"

    # --- negatives ---
    @property
    def neg_train_path(self) -> Path:
        return self.neg_dir / f"{NEG_PREFIX}_{self.fold}_train.bed.gz"

    @property
    def neg_val_path(self) -> Path:
        return self.neg_dir / f"{NEG_PREFIX}_{self.fold}_val.bed.gz"

    @property
    def neg_test_path(self) -> Path:
        return self.neg_dir / f"{NEG_PREFIX}_{self.fold}_test.bed.gz"

    # --- outputs ---
    @property
    def model_dir(self) -> Path:
        return (
            self.proj_dir / "models" / self.model_name / self.data_type
            / self.treatment / self.fold / self.timestamp
        )

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
            self.plus_bw_path,
            self.minus_bw_path,
            self.train_peak_path,
            self.val_peak_path,
        ]
        if self.require_negatives:
            required += [self.neg_train_path, self.neg_val_path, self.neg_test_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            joined = "\n  ".join(missing)
            raise FileNotFoundError(f"Missing required PRO-seq file(s):\n  {joined}")

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "timestamp": self.timestamp,
            "proj_dir": str(self.proj_dir),
            "treatment": self.treatment,
            # WandbLogger builds the run name/group from "cell_type"; reuse the
            # treatment so PRO-seq runs are named/grouped per timepoint, e.g.
            # capy-GSM8306530_0m-fold<fold>-<timestamp>.
            "cell_type": self.treatment,
            "fold": self.fold,
            "data_type": self.data_type,
            "model_name": self.model_name,
            "genome_path": str(self.genome_path),
            "chrom_size_path": str(self.chrom_size_path),
            "plus_bw_path": str(self.plus_bw_path),
            "minus_bw_path": str(self.minus_bw_path),
            "train_peak_path": str(self.train_peak_path),
            "val_peak_path": str(self.val_peak_path),
            "test_peak_path": str(self.test_peak_path),
            "neg_train_path": str(self.neg_train_path),
            "neg_val_path": str(self.neg_val_path),
            "neg_test_path": str(self.neg_test_path),
            "model_dir": str(self.model_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "last_checkpoint_path": str(self.last_checkpoint_path),
            "metrics_path": str(self.metrics_path),
            "params_path": str(self.params_path),
            "config_path": str(self.config_path),
            "eval_dir": str(self.eval_dir),
        }
