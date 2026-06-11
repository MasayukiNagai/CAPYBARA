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

# Validate basic config and required host tools.
check_base_dir_config
check_env_config
require_cmd wget
require_cmd gunzip
require_cmd awk
# Ensure reference and split directories exist.
ensure_dir "${REF_DIR}"
ensure_dir "${SPLITS_DIR}"
ensure_dir "${RAW_SAMPLE_DIR}"
ensure_dir "${PEAKS_DIR}"

resolve_encode_peak_url() {
  local experiment="$1"
  require_cmd python3
  python3 - "${experiment}" <<'PY'
import json
import sys
import urllib.request

experiment = sys.argv[1]
url = f"https://www.encodeproject.org/experiments/{experiment}/?format=json"
request = urllib.request.Request(url, headers={"Accept": "application/json"})
with urllib.request.urlopen(request) as response:
    data = json.load(response)

priority = [
    "pseudoreplicated IDR thresholded peaks",
    "optimal IDR thresholded peaks",
    "IDR thresholded peaks",
]
files = [
    f for f in data.get("files", [])
    if f.get("file_format") == "bed"
    and f.get("assembly") == "GRCh38"
    and f.get("status") == "released"
    and f.get("output_type") in priority
]
files.sort(key=lambda f: priority.index(f["output_type"]))
if files:
    print("https://www.encodeproject.org" + files[0]["href"])
PY
}

if [[ ! -f "${HG38_FA}" ]]; then
  # Download the hg38 no-alt reference FASTA.
  wget -O "${HG38_FA_GZ}" \
    "https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz"
  # Decompress FASTA for downstream genome access.
  gunzip -f "${HG38_FA_GZ}"
fi

if [[ ! -f "${CHROM_SIZES}" ]]; then
  if [[ -f "${WITH_RDNA_CHROM_SIZES}" ]]; then
    # Reuse the local chrom sizes but remove only the appended rDNA contig.
    awk '$1 != "U13369.1"' "${WITH_RDNA_CHROM_SIZES}" > "${CHROM_SIZES}"
  else
    # Download chromosome sizes for peak calling and nonpeak generation.
    wget -O "${CHROM_SIZES}" \
      "https://www.encodeproject.org/files/GRCh38_EBV.chrom.sizes/@@download/GRCh38_EBV.chrom.sizes.tsv"
  fi
fi

if [[ ! -f "${BLACKLIST}" ]]; then
  # Download ENCODE blacklist regions for artifact filtering.
  wget -O "${BLACKLIST}" \
    "https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz"
fi

for fold in 0 1 2 3 4; do
  fold_json_path="${SPLITS_DIR}/fold_${fold}.json"
  if [[ -f "${fold_json_path}" ]]; then
    echo "Skipping split download: ${fold_json_path} already exists."
  else
    # Download predefined ChromBPNet split JSON from Zenodo.
    wget -O "${fold_json_path}" "${SPLITS_BASE_URL}/fold_${fold}.json"
  fi
done

if declare -p BAM_ACCESSIONS >/dev/null 2>&1; then
  for accession in "${BAM_ACCESSIONS[@]}"; do
    bam_path="${BAM_DIR}/${accession}.bam"
    if [[ -f "${bam_path}" ]]; then
      echo "Skipping BAM download: ${bam_path} already exists."
    else
      wget -O "${bam_path}" "${ENCODE_BASE_URL}/${accession}/@@download/${accession}.bam"
    fi
  done
fi

if [[ -z "${PEAKS_URL:-}" && -n "${PEAKS_EXPERIMENT:-}" && ! -f "${PEAKS_RAW}" ]]; then
  PEAKS_URL="$(resolve_encode_peak_url "${PEAKS_EXPERIMENT}")"
  [[ -n "${PEAKS_URL}" ]] || die "Could not resolve an ENCODE peak URL for ${PEAKS_EXPERIMENT}."
fi

if [[ -n "${PEAKS_URL:-}" && ! -f "${PEAKS_RAW}" ]]; then
  wget -O "${PEAKS_RAW}" "${PEAKS_URL}"
else
  echo "Skipping peak download: ${PEAKS_RAW} already exists or PEAKS_URL is unset."
fi

if declare -p BAM_REPLICATES >/dev/null 2>&1; then
  for bam in "${BAM_REPLICATES[@]}"; do
    if [[ -f "${bam}" ]]; then
      echo "Found BAM: ${bam}"
    else
      echo "Missing BAM listed in config: ${bam}" >&2
    fi
  done
fi

echo "Reference, split, and peak download checks complete."
