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

# Validate config and required host tools.
check_base_dir_config
check_env_config
check_fold_config
require_cmd apptainer
require_file "${HG38_FA}"
require_file "${PEAKS_NO_BLACKLIST}"
require_file "${CHROM_SIZES}"
require_file "${FOLD_JSON}"
require_file "${BLACKLIST}"
ensure_dir "$(dirname "${NONPEAKS_PREFIX}")"

if [[ -f "${NEGATIVE_BED}" ]]; then
  echo "Skipping non-peak generation: ${NEGATIVE_BED} already exists."
  exit 0
fi

# Pull chrombpnet container image if missing.
ensure_apptainer_image "${CHROMBPNET_SIF}" "${CHROMBPNET_IMAGE_URI}"

# Optional: suppress tqdm progress output from chrombpnet.
APPTAINER_ENV_ARGS=()
if [[ "${CHROMBPNET_QUIET_PROGRESS:-0}" == "1" ]]; then
  APPTAINER_ENV_ARGS+=(--env TQDM_DISABLE=1)
fi

APPTAINER_BIND_ARGS=()
add_bind() {
  local path="$1"
  [[ -e "${path}" ]] || die "Cannot bind missing path: ${path}"
  APPTAINER_BIND_ARGS+=(--bind "${path}:${path}")
}

CONTAINER_HG38_FA="$(readlink -f "${HG38_FA}")"
CONTAINER_PEAKS_NO_BLACKLIST="$(readlink -f "${PEAKS_NO_BLACKLIST}")"
CONTAINER_CHROM_SIZES="$(readlink -f "${CHROM_SIZES}")"
CONTAINER_FOLD_JSON="$(readlink -f "${FOLD_JSON}")"
CONTAINER_BLACKLIST="$(readlink -f "${BLACKLIST}")"
CONTAINER_NONPEAKS_DIR="$(readlink -f "$(dirname "${NONPEAKS_PREFIX}")")"
CONTAINER_NONPEAKS_PREFIX="${CONTAINER_NONPEAKS_DIR}/$(basename "${NONPEAKS_PREFIX}")"

add_bind "$(readlink -f "${DATA_DIR}")"
add_bind "$(readlink -f "${GENOME_DIR}")"
add_bind "${CONTAINER_NONPEAKS_DIR}"

# Generate GC-matched background regions from non-peak loci.
apptainer_cmd=(
  apptainer exec
  "${APPTAINER_ENV_ARGS[@]}"
  "${APPTAINER_BIND_ARGS[@]}"
  --nv
  "${CHROMBPNET_SIF}"
  chrombpnet prep nonpeaks
  -g "${CONTAINER_HG38_FA}"
  -p "${CONTAINER_PEAKS_NO_BLACKLIST}"
  -c "${CONTAINER_CHROM_SIZES}"
  -fl "${CONTAINER_FOLD_JSON}"
  -br "${CONTAINER_BLACKLIST}"
  -o "${CONTAINER_NONPEAKS_PREFIX}"
)
echo "${apptainer_cmd[*]}"
"${apptainer_cmd[@]}"

require_file "${NEGATIVE_BED}"
echo "Non-peak regions generated: ${NEGATIVE_BED}"
