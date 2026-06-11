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
require_cmd bedtools
require_file "${BLACKLIST}"
require_file "${CHROM_SIZES}"
require_file "${PEAKS_FILTERED}"

if [[ -f "${PEAKS_NO_BLACKLIST}" ]]; then
  echo "Skipping blacklist removal: ${PEAKS_NO_BLACKLIST} already exists."
  exit 0
fi

if [[ ! -f "${BLACKLIST_EXTENDED}" ]]; then
  # Expand blacklist regions by configured flank length.
  bedtools slop -i "${BLACKLIST}" -g "${CHROM_SIZES}" -b "${BLACKLIST_WINDOW}" > "${BLACKLIST_EXTENDED}"
else
  echo "Skipping blacklist extension: ${BLACKLIST_EXTENDED} already exists."
fi

# Remove peaks overlapping extended blacklist regions.
bedtools intersect -v -a "${PEAKS_FILTERED}" -b "${BLACKLIST_EXTENDED}" > "${PEAKS_NO_BLACKLIST}"

echo "Blacklist-filtered peaks written to ${PEAKS_NO_BLACKLIST}"
