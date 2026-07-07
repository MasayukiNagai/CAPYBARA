from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


VALID_CELL_TYPES = {"K562", "HepG2", "GM12878", "IMR90", "H1-hESC"}

# Root of the shared ChromBPNet assets (read-only inputs for the benchmark).
DEFAULT_SHARED_ROOT = Path("/grid/koo/home/shared/capybara/chrombpnet")


@dataclass(frozen=True)
class BiasCapyFiles:
    """Paths for CAPY Stage-1 bias training.

    Reuses ChromBPNet's exact per-fold filtered regions, +4/-4 BigWig, fold
    split, and recorded hyperparameters (read-only), and writes CAPY outputs
    under ``proj_dir/models/capy_bias/...``.
    """

    proj_dir: Path
    shared_root: Path
    cell_type: str
    fold: int
    timestamp: str
    model_name: str = "capy_bias"

    @classmethod
    def create(
        cls,
        *,
        proj_dir: str | Path,
        shared_root: str | Path = DEFAULT_SHARED_ROOT,
        cell_type: str = "K562",
        fold: int | str = 0,
        timestamp: str | None = None,
    ) -> "BiasCapyFiles":
        cell_type = str(cell_type)
        if cell_type not in VALID_CELL_TYPES:
            raise ValueError(f"Unsupported cell_type={cell_type!r}. Expected one of {sorted(VALID_CELL_TYPES)}.")
        fold = int(fold)
        if not (0 <= fold <= 4):
            raise ValueError(f"fold must be in [0, 4], got {fold}.")
        timestamp = timestamp or datetime.now().strftime("%y%m%d_%H%M%S")
        cfg = cls(
            proj_dir=Path(proj_dir),
            shared_root=Path(shared_root),
            cell_type=cell_type,
            fold=fold,
            timestamp=timestamp,
        )
        return cfg

    # ------------------------------------------------------------------ #
    # Shared read-only inputs (ChromBPNet assets)
    # ------------------------------------------------------------------ #
    @property
    def genome_path(self) -> Path:
        return self.shared_root / "genome" / "hg38.fasta"

    @property
    def chrom_size_path(self) -> Path:
        return self.shared_root / "genome" / "hg38.chrom.sizes"

    @property
    def _bias_fold_dir(self) -> Path:
        return self.shared_root / "models" / "bias" / self.cell_type / f"fold_{self.fold}"

    @property
    def _aux_prefix(self) -> Path:
        return self._bias_fold_dir / "auxiliary" / f"{self.cell_type}.fold_{self.fold}"

    @property
    def nonpeaks_bed_path(self) -> Path:
        return Path(f"{self._aux_prefix}_filtered.bias_nonpeaks.bed")

    @property
    def peaks_bed_path(self) -> Path:
        return Path(f"{self._aux_prefix}_filtered.bias_peaks.bed")

    @property
    def data_bw_path(self) -> Path:
        return Path(f"{self._aux_prefix}_data_unstranded.bw")

    @property
    def model_params_tsv_path(self) -> Path:
        return self._bias_fold_dir / "logs" / f"{self.cell_type}.fold_{self.fold}_bias_model_params.tsv"

    @property
    def ref_metrics_path(self) -> Path:
        return self._bias_fold_dir / "evaluation" / f"{self.cell_type}.fold_{self.fold}_bias_metrics.json"

    def read_model_params(self) -> dict[str, str]:
        """Parse the ChromBPNet bias ``*_bias_model_params.tsv`` (2-col TSV)."""
        params: dict[str, str] = {}
        with self.model_params_tsv_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                key, _, value = line.partition("\t")
                params[key.strip()] = value.strip()
        return params

    @property
    def counts_loss_weight(self) -> float:
        return float(self.read_model_params()["counts_loss_weight"])

    @property
    def fold_split_path(self) -> Path:
        return Path(self.read_model_params()["chr_fold_path"])

    def fold_split(self) -> dict[str, list[str]]:
        with self.fold_split_path.open() as handle:
            return json.load(handle)

    # ------------------------------------------------------------------ #
    # CAPY writable outputs
    # ------------------------------------------------------------------ #
    @property
    def model_dir(self) -> Path:
        return (
            self.proj_dir
            / self.model_name
            / "atac"
            / self.cell_type
            / f"fold{self.fold}"
            / self.timestamp
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

    def validate_inputs(self) -> None:
        required = [
            self.genome_path,
            self.chrom_size_path,
            self.nonpeaks_bed_path,
            self.peaks_bed_path,
            self.data_bw_path,
            self.model_params_tsv_path,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            joined = "\n  ".join(missing)
            raise FileNotFoundError(f"Missing required shared bias input(s):\n  {joined}")
        if not self.fold_split_path.exists():
            raise FileNotFoundError(f"Missing fold split referenced by TSV: {self.fold_split_path}")

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "proj_dir": str(self.proj_dir),
            "shared_root": str(self.shared_root),
            "cell_type": self.cell_type,
            "data_type": "atac",
            "fold": self.fold,
            "model_name": self.model_name,
            "genome_path": str(self.genome_path),
            "chrom_size_path": str(self.chrom_size_path),
            "nonpeaks_bed_path": str(self.nonpeaks_bed_path),
            "peaks_bed_path": str(self.peaks_bed_path),
            "data_bw_path": str(self.data_bw_path),
            "model_params_tsv_path": str(self.model_params_tsv_path),
            "fold_split_path": str(self.fold_split_path),
            "ref_metrics_path": str(self.ref_metrics_path),
            "model_dir": str(self.model_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "last_checkpoint_path": str(self.last_checkpoint_path),
            "metrics_path": str(self.metrics_path),
            "params_path": str(self.params_path),
            "config_path": str(self.config_path),
            "eval_dir": str(self.eval_dir),
        }
