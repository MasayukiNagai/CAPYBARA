#!/usr/bin/env bash
set -euo pipefail

# Require an explicit config path.
ENV_FILE="${1:-}"
[[ -f "${ENV_FILE}" ]] || {
  echo "Usage: $0 <path/to/sample.env> [fold]" >&2
  echo "Example: $0 examples/atac/chrombpnet_setup/envs/k562.env" >&2
  exit 1
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/chrombpnet_env.sh"
source "${ENV_FILE}"
FOLD="${2:-${FOLD:?fold must be specified}}"
refresh_env_derived_vars

# Validate config and required inputs.
check_base_dir_config
check_env_config
check_fold_config
require_cmd apptainer
require_file "${BAM_MERGED}"
require_file "${HG38_FA}"
require_file "${CHROM_SIZES}"
require_file "${PEAKS_NO_BLACKLIST}"
require_file "${NEGATIVE_BED}"
require_file "${FOLD_JSON}"
require_file "${BIAS_MODEL_PATH}"

if [[ -d "${CHROMBPNET_MODEL_DIR}" ]] && [[ -n "$(ls -A "${CHROMBPNET_MODEL_DIR}" 2>/dev/null)" ]] && [[ "${OVERWRITE:-0}" != "1" ]]; then
  die "Output exists: ${CHROMBPNET_MODEL_DIR}. Set OVERWRITE=1 to allow rerun into existing directory."
fi
ensure_dir "${CHROMBPNET_MODEL_DIR}"

# Pull chrombpnet image if missing.
ensure_apptainer_image "${CHROMBPNET_SIF}" "${CHROMBPNET_IMAGE_URI}"

APPTAINER_BIND_ARGS=()
add_bind() {
  local path="$1"
  [[ -e "${path}" ]] || die "Cannot bind missing path: ${path}"
  APPTAINER_BIND_ARGS+=(--bind "${path}:${path}")
}

CONTAINER_BAM_MERGED="$(readlink -f "${BAM_MERGED}")"
CONTAINER_HG38_FA="$(readlink -f "${HG38_FA}")"
CONTAINER_CHROM_SIZES="$(readlink -f "${CHROM_SIZES}")"
CONTAINER_PEAKS_NO_BLACKLIST="$(readlink -f "${PEAKS_NO_BLACKLIST}")"
CONTAINER_NEGATIVE_BED="$(readlink -f "${NEGATIVE_BED}")"
CONTAINER_FOLD_JSON="$(readlink -f "${FOLD_JSON}")"
CONTAINER_BIAS_MODEL_PATH="$(readlink -f "${BIAS_MODEL_PATH}")"
CONTAINER_CHROMBPNET_MODEL_DIR="$(readlink -f "${CHROMBPNET_MODEL_DIR}")"

add_bind "$(readlink -f "${BAM_DIR}")"
add_bind "$(readlink -f "${DATA_DIR}")"
add_bind "$(readlink -f "${GENOME_DIR}")"
add_bind "$(dirname "${CONTAINER_BIAS_MODEL_PATH}")"
add_bind "${CONTAINER_CHROMBPNET_MODEL_DIR}"

# Train bias-factorized ChromBPNet using a pretrained or locally trained bias model.
apptainer_cmd=(
  apptainer exec
  "${APPTAINER_BIND_ARGS[@]}"
  --nv
  "${CHROMBPNET_SIF}"
  chrombpnet pipeline
  -ibam "${CONTAINER_BAM_MERGED}"
  -d "ATAC"
  -g "${CONTAINER_HG38_FA}"
  -c "${CONTAINER_CHROM_SIZES}"
  -p "${CONTAINER_PEAKS_NO_BLACKLIST}"
  -n "${CONTAINER_NEGATIVE_BED}"
  -fl "${CONTAINER_FOLD_JSON}"
  -b "${CONTAINER_BIAS_MODEL_PATH}"
  -o "${CONTAINER_CHROMBPNET_MODEL_DIR}"
  -fp "${SAMPLE_PREFIX}"
)
echo "${apptainer_cmd[*]}"
"${apptainer_cmd[@]}"
