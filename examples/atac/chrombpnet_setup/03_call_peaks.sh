#!/usr/bin/env bash
set -euo pipefail

# Require an explicit config path.
ENV_FILE="${1:-}"
[[ -f "${ENV_FILE}" ]] || {
  echo "Usage: $0 <path/to/sample.env>" >&2
  echo "Example: $0 examples/atac/chrombpnet_setup/envs/k562.env" >&2
  exit 1
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/chrombpnet_env.sh"
source "${ENV_FILE}"
refresh_env_derived_vars

# Validate config and required host tools.
check_base_dir_config
check_env_config
require_cmd awk
require_cmd gzip
# Ensure output location exists.
ensure_dir "${PEAKS_DIR}"

locate_task_in_container() {
  local image="$1"
  local task_name="$2"
  local task_path
  task_path="$(apptainer exec "${image}" bash -lc "find / -type f -name '${task_name}' 2>/dev/null | head -n 1")"
  [[ -n "${task_path}" ]] || die "Could not locate ${task_name} inside ${image}"
  echo "${task_path}"
}

run_bam2ta() {
  require_cmd apptainer
  # Pull ENCODE ATAC pipeline image if missing.
  ensure_apptainer_image "${ATAC_PIPELINE_SIF}" "${ATAC_PIPELINE_IMAGE_URI}"
  # Locate BAM->tagAlign task script inside the container.
  local bam2ta_script
  bam2ta_script="$(locate_task_in_container "${ATAC_PIPELINE_SIF}" "encode_task_bam2ta.py")"
  # Convert merged BAM to Tn5-shifted tagAlign.
  apptainer exec \
    --bind "${WORK_DIR}" \
    --bind "${DATA_DIR}" \
    "${ATAC_PIPELINE_SIF}" \
    python "${bam2ta_script}" \
      "${BAM_MERGED}" \
      --out-dir "${PEAKS_DIR}" \
      --mito-chr-name chrM
}

run_macs2() {
  require_cmd apptainer
  # Pull ENCODE ATAC pipeline image if missing.
  ensure_apptainer_image "${ATAC_PIPELINE_SIF}" "${ATAC_PIPELINE_IMAGE_URI}"
  # Locate MACS2 ATAC task script inside the container.
  local macs2_script
  macs2_script="$(locate_task_in_container "${ATAC_PIPELINE_SIF}" "encode_task_macs2_atac.py")"
  # Run permissive MACS2 peak calling.
  apptainer exec \
    --bind "${WORK_DIR}" \
    --bind "${DATA_DIR}" \
    "${ATAC_PIPELINE_SIF}" \
    python "${macs2_script}" \
      "${PEAKS_TA}" \
      --gensz hs \
      --chrsz "${CHROM_SIZES}" \
      --pval-thresh 0.01 \
      --smooth-win 150 \
      --cap-num-peak 300000 \
      --out-dir "${PEAKS_DIR}"
}

if [[ "${CALL_PEAKS}" == "1" ]]; then
  if [[ ! -f "${PEAKS_TA}" ]]; then
    # Ensure merged BAM exists before tagAlign generation.
    require_file "${BAM_MERGED}"
    run_bam2ta
  else
    echo "Skipping bam2ta: ${PEAKS_TA} already exists."
  fi

  if [[ ! -f "${PEAKS}" ]]; then
    # Ensure inputs exist before MACS2 peak calling.
    require_file "${PEAKS_TA}"
    require_file "${CHROM_SIZES}"
    run_macs2
  else
    echo "Skipping MACS2: ${PEAKS} already exists."
  fi
else
  require_file "${PEAKS_RAW}"
  PEAKS="${PEAKS_RAW}"
fi

if [[ -f "${PEAKS_FILTERED}" ]]; then
  echo "Skipping peak filtering: ${PEAKS_FILTERED} already exists."
else
  require_file "${PEAKS}"
  if [[ "${PEAKS}" == *.gz ]]; then
    # Keep standard chromosomes and a 10-column narrowPeak shape.
    gzip -dc "${PEAKS}" \
      | awk -v OFS="\t" '$1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X|Y)$/ && NF >= 10 {print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10}' \
      > "${PEAKS_FILTERED}"
  else
    # Keep standard chromosomes and a 10-column narrowPeak shape.
    awk -v OFS="\t" '$1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X|Y)$/ && NF >= 10 {print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10}' "${PEAKS}" \
      > "${PEAKS_FILTERED}"
  fi
fi

echo "Peaks prepared at ${PEAKS_FILTERED}"
