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
require_cmd samtools
# Ensure BAM output directory exists.
ensure_dir "${BAM_DIR}"

# Validate replicate BAM inputs.
(( ${#BAM_REPLICATES[@]} > 0 )) || die "BAM_REPLICATES is empty in sample config."
for bam in "${BAM_REPLICATES[@]}"; do
  require_file "${bam}"
done

if [[ -f "${BAM_MERGED}" && -f "${BAM_MERGED}.bai" ]]; then
  echo "Skipping merge: ${BAM_MERGED} and index already exist."
  exit 0
fi

if [[ ! -f "${BAM_MERGED_UNSORTED}" ]]; then
  # Merge replicate BAMs into one pooled unsorted BAM.
  samtools merge -@ "${THREADS}" -f "${BAM_MERGED_UNSORTED}" "${BAM_REPLICATES[@]}"
else
  echo "Skipping merge command: ${BAM_MERGED_UNSORTED} already exists."
fi

if [[ ! -f "${BAM_MERGED}" ]]; then
  # Coordinate-sort pooled BAM for downstream indexing and access.
  samtools sort -@ "${THREADS}" "${BAM_MERGED_UNSORTED}" -o "${BAM_MERGED}"
else
  echo "Skipping sort: ${BAM_MERGED} already exists."
fi

if [[ ! -f "${BAM_MERGED}.bai" ]]; then
  # Build BAM index for random-access queries.
  samtools index -@ "${THREADS}" "${BAM_MERGED}"
else
  echo "Skipping index: ${BAM_MERGED}.bai already exists."
fi

echo "Merged BAM created at ${BAM_MERGED}"
