#!/usr/bin/env bash

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "Required file not found: $1"
}

ensure_dir() {
  mkdir -p "$1"
}

ensure_apptainer_image() {
  local image_path="$1"
  local image_uri="$2"

  [[ -n "${image_path}" ]] || die "Apptainer image path is empty."
  [[ -n "${image_uri}" ]] || die "Apptainer image URI is empty."

  if [[ -f "${image_path}" ]]; then
    return 0
  fi

  require_cmd apptainer
  ensure_dir "$(dirname "${image_path}")"
  apptainer pull "${image_path}" "${image_uri}"
}

check_base_dir_config() {
  [[ -n "${BASE_DIR:-}" ]] || die "BASE_DIR is not set."
  [[ -n "${DATA_DIR:-}" ]] || die "DATA_DIR is not set."
  [[ -n "${GENOME_DIR:-}" ]] || die "GENOME_DIR is not set."
  [[ -n "${SPLITS_DIR:-}" ]] || die "SPLITS_DIR is not set."
}

check_env_config() {
  [[ -n "${CELL_TYPE:-}" ]] || die "CELL_TYPE is not set."
}

check_fold_config() {
  [[ -n "${FOLD:-}" ]] || die "FOLD is not set."
  [[ "${FOLD}" =~ ^[0-4]$ ]] || die "FOLD must be an integer from 0 to 4; got ${FOLD}."
}

refresh_env_derived_vars() {
  THREADS="${THREADS:-4}"
  BLACKLIST_WINDOW="${BLACKLIST_WINDOW:-1057}"
  CALL_PEAKS="${CALL_PEAKS:-0}"

  BASE_DIR="${BASE_DIR:-${PWD}/data/chrombpnet}"
  ENCODE_BASE_URL="${ENCODE_BASE_URL:-https://www.encodeproject.org/files}"
  DATA_DIR="${DATA_DIR:-${BASE_DIR}/data}"
  GENOME_DIR="${GENOME_DIR:-${BASE_DIR}/genome}"
  REF_DIR="${REF_DIR:-${GENOME_DIR}}"
  SPLITS_DIR="${SPLITS_DIR:-${DATA_DIR}/splits}"
  ATAC_RAW_DIR="${ATAC_RAW_DIR:-${DATA_DIR}/atac/raw}"
  ATAC_PROCESSED_DIR="${ATAC_PROCESSED_DIR:-${DATA_DIR}/atac/processed}"
  WORK_DIR="${WORK_DIR:-${BASE_DIR}}"
  IMAGES_DIR="${BASE_DIR}"  #"${IMAGES_DIR:-${BASE_DIR}/images}"
  MODELS_DIR="${MODELS_DIR:-${BASE_DIR}/models}"

  RAW_SAMPLE_DIR="${RAW_SAMPLE_DIR:-${ATAC_RAW_DIR}/${CELL_TYPE}}"
  PROCESSED_SAMPLE_DIR="${PROCESSED_SAMPLE_DIR:-${ATAC_PROCESSED_DIR}/${CELL_TYPE}}"
  BAM_DIR="${BAM_DIR:-${RAW_SAMPLE_DIR}}"
  PEAKS_DIR="${PEAKS_DIR:-${PROCESSED_SAMPLE_DIR}/peaks}"
  if [[ -n "${FOLD:-}" ]]; then
    NONPEAKS_DIR="${NONPEAKS_DIR:-${PROCESSED_SAMPLE_DIR}/nonpeaks/fold_${FOLD}}"
  fi

  HG38_FA="${HG38_FA:-${GENOME_DIR}/hg38.fasta}"
  HG38_FA_GZ="${HG38_FA_GZ:-${HG38_FA}.gz}"
  CHROM_SIZES="${CHROM_SIZES:-${GENOME_DIR}/hg38.chrom.sizes}"
  WITH_RDNA_CHROM_SIZES="${WITH_RDNA_CHROM_SIZES:-${GENOME_DIR}/hg38.withrDNA.chrom.sizes}"
  BLACKLIST="${BLACKLIST:-${GENOME_DIR}/hg38.blacklist.bed.gz}"
  BLACKLIST_EXTENDED="${BLACKLIST_EXTENDED:-${GENOME_DIR}/hg38.blacklist.slop${BLACKLIST_WINDOW}.bed}"

  BAM_MERGED="${BAM_MERGED:-${BAM_DIR}/${CELL_TYPE}.merged.bam}"
  BAM_MERGED_UNSORTED="${BAM_MERGED_UNSORTED:-${BAM_DIR}/${CELL_TYPE}.merged.unsorted.bam}"

  if ! declare -p BAM_REPLICATES >/dev/null 2>&1 && declare -p BAM_ACCESSIONS >/dev/null 2>&1; then
    BAM_REPLICATES=()
    for accession in "${BAM_ACCESSIONS[@]}"; do
      BAM_REPLICATES+=("${BAM_DIR}/${accession}.bam")
    done
  fi

  PEAKS_RAW="${PEAKS_RAW:-${PEAKS_DIR}/${CELL_TYPE}.raw.narrowPeak.gz}"
  PEAKS_FILTERED="${PEAKS_FILTERED:-${PEAKS_DIR}/${CELL_TYPE}.peaks.filtered.bed}"
  PEAKS_NO_BLACKLIST="${PEAKS_NO_BLACKLIST:-${PEAKS_DIR}/${CELL_TYPE}.peaks_no_blacklist.bed}"
  PEAKS="${PEAKS:-${PEAKS_RAW}}"
  PEAKS_TA="${PEAKS_TA:-${PEAKS_DIR}/${CELL_TYPE}.tagAlign.gz}"
  if [[ -z "${PEAKS_URL:-}" && -n "${PEAKS_ACCESSION:-}" ]]; then
    PEAKS_URL="${ENCODE_BASE_URL}/${PEAKS_ACCESSION}/@@download/${PEAKS_ACCESSION}.bed.gz"
  fi

  if [[ -n "${FOLD:-}" ]]; then
    FOLD_JSON="${FOLD_JSON:-${SPLITS_DIR}/fold_${FOLD}.json}"
  fi
  SPLITS_BASE_URL="${SPLITS_BASE_URL:-https://zenodo.org/record/7445373/files}"

  if [[ -n "${FOLD:-}" ]]; then
    NONPEAKS_PREFIX="${NONPEAKS_PREFIX:-${NONPEAKS_DIR}/${CELL_TYPE}.fold_${FOLD}}"
    NEGATIVE_BED="${NEGATIVE_BED:-${NONPEAKS_PREFIX}_negatives.bed}"
    SAMPLE_PREFIX="${SAMPLE_PREFIX:-${CELL_TYPE}.fold_${FOLD}}"

    BIAS_MODEL_DIR="${BIAS_MODEL_DIR:-${MODELS_DIR}/bias/${CELL_TYPE}/fold_${FOLD}}"
    BIAS_MODEL_PATH="${BIAS_MODEL_PATH:-${BIAS_MODEL_DIR}/models/bias.h5}"
    CHROMBPNET_MODEL_DIR="${CHROMBPNET_MODEL_DIR:-${MODELS_DIR}/chrombpnet/${CELL_TYPE}/fold_${FOLD}}"
  fi

  CHROMBPNET_IMAGE_URI="${CHROMBPNET_IMAGE_URI:-docker://kundajelab/chrombpnet:latest}"
  CHROMBPNET_SIF="${CHROMBPNET_SIF:-${IMAGES_DIR}/chrombpnet_latest.sif}"
  ATAC_PIPELINE_IMAGE_URI="${ATAC_PIPELINE_IMAGE_URI:-docker://encodedcc/atac-seq-pipeline:v2.2.3}"
  ATAC_PIPELINE_SIF="${ATAC_PIPELINE_SIF:-${IMAGES_DIR}/encode_atac_pipeline_v2.2.3.sif}"
}
