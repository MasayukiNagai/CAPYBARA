#!/usr/bin/env bash
# download_data.sh
# Download shared reference files (genome, blacklist) and per-cell-type ATAC-seq BAMs.
# Shared files are downloaded once and skipped on re-runs if already present.
#
# Requirements: wget, samtools
#
# Usage:
#   bash download_data.sh                  # shared files only
#   bash download_data.sh K562             # shared + K562 BAMs
#   CELL_TYPE=K562 bash download_data.sh   # same, via env var

set -euo pipefail

CELL_TYPE=${1:-${CELL_TYPE:-""}}

# ── Configure ──────────────────────────────────────────────────────────────────
PROJ_DIR=${PROJ_DIR:-/path/to/proj}
THREADS=${THREADS:-4}   # threads for samtools merge/index
# ───────────────────────────────────────────────────────────────────────────────

ENCODE="https://www.encodeproject.org/files"
GENOME_DIR=$PROJ_DIR/genome
RAW_DIR=$PROJ_DIR/data/atac/raw

mkdir -p "$GENOME_DIR"

# ── Shared: reference genome ───────────────────────────────────────────────────

echo "=== Reference genome ==="

if [ ! -f "$GENOME_DIR/hg38.withrDNA.fasta" ]; then
    echo "Downloading hg38 (ENCODE GRCh38 no-alt analysis set)..."
    wget "https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz" \
        -O "$GENOME_DIR/hg38.fasta.gz"
    gunzip "$GENOME_DIR/hg38.fasta.gz"

    echo "Downloading rDNA (U13369.1 from NCBI)..."
    wget "https://www.ncbi.nlm.nih.gov/sviewer/viewer.cgi?tool=portal&save=file&log\$=seqview&db=nuccore&report=fasta&id=555853&conwithfeat=on&withparts=on&hide-cdd=on" \
        -O - | sed '$ d' > "$GENOME_DIR/rDNA_U13369.1.fasta"

    cat "$GENOME_DIR/hg38.fasta" "$GENOME_DIR/rDNA_U13369.1.fasta" \
        > "$GENOME_DIR/hg38.withrDNA.fasta"
    echo "Created hg38.withrDNA.fasta"
else
    echo "hg38.withrDNA.fasta already exists; skipping."
fi

if [ ! -f "$GENOME_DIR/hg38.withrDNA.chrom.sizes" ]; then
    wget "${ENCODE}/GRCh38_EBV.chrom.sizes/@@download/GRCh38_EBV.chrom.sizes.tsv" \
        -O - | grep -v "_alt" > "$GENOME_DIR/hg38.withrDNA.chrom.sizes"
    rDNA_LEN=$(grep -v ">" "$GENOME_DIR/rDNA_U13369.1.fasta" | tr -d '\n' | wc -c)
    printf "U13369.1\t%s\n" "$rDNA_LEN" >> "$GENOME_DIR/hg38.withrDNA.chrom.sizes"
    echo "Created hg38.withrDNA.chrom.sizes"
else
    echo "hg38.withrDNA.chrom.sizes already exists; skipping."
fi

# ── Shared: blacklist ──────────────────────────────────────────────────────────

echo "=== Blacklist ==="

if [ ! -f "$GENOME_DIR/hg38.blacklist.bed.gz" ]; then
    wget "${ENCODE}/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz" \
        -O "$GENOME_DIR/hg38.blacklist.bed.gz"
    echo "Downloaded hg38.blacklist.bed.gz (ENCFF356LFX)"
else
    echo "hg38.blacklist.bed.gz already exists; skipping."
fi

# ── Cell-type-specific BAMs ────────────────────────────────────────────────────

if [ -z "$CELL_TYPE" ]; then
    echo ""
    echo "Shared files done. To download BAMs, re-run with a CELL_TYPE argument:"
    echo "  bash download_data.sh K562"
    exit 0
fi

# ENCODE BAM file accessions (output_type=alignments, GRCh38, latest pipeline).
# Experiment accessions correspond to ChromBPNet paper Table 1.
#
# K562   ENCSR868FGK  |  HepG2   ENCSR042AWH  |  GM12878  ENCSR637XSC
# IMR90  ENCSR200OML  |  H1-hESC GEO GSE267154 (handled separately below)
#
case "$CELL_TYPE" in
    K562)    EXP=ENCSR868FGK; REPS="ENCFF534DCE ENCFF128WZG ENCFF077FBI" ;;
    HepG2)   EXP=ENCSR042AWH; REPS="ENCFF239RGZ ENCFF394BBD"             ;;
    GM12878) EXP=ENCSR637XSC; REPS="ENCFF981FXV ENCFF962FMH ENCFF440GRZ" ;;
    IMR90)   EXP=ENCSR200OML; REPS="ENCFF848XMR ENCFF715NAV"             ;;
    H1-hESC)
        echo ""
        echo "H1-hESC ATAC-seq data is from GEO GSE267154 (in-house; not on the ENCODE portal)."
        echo "Download the two replicate BAMs from SRA:"
        echo "  GSM8260976 → rep1"
        echo "  GSM8260977 → rep2"
        echo ""
        echo "Using sra-tools (prefetch + samtools):"
        echo "  prefetch <SRR_accession>"
        echo "  samtools sort -n <sra_file> | samtools fastq ... | bowtie2 ... | samtools sort > rep.bam"
        echo ""
        echo "Then place the merged BAM at:"
        echo "  $RAW_DIR/H1-hESC/H1-hESC.bam"
        echo "and index it:"
        echo "  samtools index $RAW_DIR/H1-hESC/H1-hESC.bam"
        exit 0
        ;;
    *)
        echo "Unknown CELL_TYPE: $CELL_TYPE"
        echo "Supported: K562, HepG2, GM12878, IMR90, H1-hESC"
        exit 1
        ;;
esac

CT_RAW_DIR=$RAW_DIR/$CELL_TYPE
mkdir -p "$CT_RAW_DIR"

MERGED_BAM=$CT_RAW_DIR/${CELL_TYPE}.bam

echo "=== Downloading ENCODE IDR peaks for $CELL_TYPE ($EXP) ==="
PEAKS_OUT=$CT_RAW_DIR/peaks.narrowPeak.gz
if [ ! -f "$PEAKS_OUT" ]; then
    # Query the ENCODE REST API to find the pseudoreplicated IDR narrowPeak file
    PEAKS_HREF=$(python3 - <<PYEOF
import urllib.request, json, sys
url = "https://www.encodeproject.org/experiments/$EXP/?format=json"
req = urllib.request.Request(url, headers={"Accept": "application/json"})
try:
    with urllib.request.urlopen(req) as r:
        data = json.load(r)
except Exception as e:
    print("", file=sys.stderr)
    sys.exit(0)
priority = [
    "pseudoreplicated IDR thresholded peaks",
    "optimal IDR thresholded peaks",
    "IDR thresholded peaks",
]
files = [f for f in data.get("files", [])
         if f.get("file_format") == "bed"
         and f.get("assembly") == "GRCh38"
         and f.get("status") == "released"
         and f.get("output_type") in priority]
# Sort by priority order
files.sort(key=lambda f: priority.index(f["output_type"]))
if files:
    print("https://www.encodeproject.org" + files[0]["href"])
PYEOF
)
    if [ -z "$PEAKS_HREF" ]; then
        echo "WARNING: Could not find IDR peaks for $CELL_TYPE ($EXP) via ENCODE API."
        echo "         Download the narrowPeak file manually and place it at:"
        echo "         $PEAKS_OUT"
    else
        wget "$PEAKS_HREF" -O "$PEAKS_OUT"
        echo "Downloaded peaks → $PEAKS_OUT"
    fi
else
    echo "peaks.narrowPeak.gz already present; skipping."
fi

if [ -f "$MERGED_BAM" ]; then
    echo "$MERGED_BAM already exists; skipping BAM download and merge."
else
    echo "=== Downloading $CELL_TYPE replicate BAMs ==="
    REP_BAMS=()
    for ACC in $REPS; do
        OUT=$CT_RAW_DIR/${ACC}.bam
        if [ ! -f "$OUT" ]; then
            echo "  Downloading $ACC..."
            wget "${ENCODE}/${ACC}/@@download/${ACC}.bam" -O "$OUT"
        else
            echo "  $ACC already present; skipping."
        fi
        REP_BAMS+=("$OUT")
    done

    echo "=== Merging replicates → $MERGED_BAM ==="
    if [ ${#REP_BAMS[@]} -eq 1 ]; then
        cp "${REP_BAMS[0]}" "$MERGED_BAM"
    else
        samtools merge -@ "$THREADS" -f "$MERGED_BAM" "${REP_BAMS[@]}"
    fi
    samtools index -@ "$THREADS" "$MERGED_BAM"
    echo "Merged and indexed: $MERGED_BAM"
fi

echo ""
echo "=== Done ==="
echo "Next step: run prepare_data.sh with PROJ_DIR=$PROJ_DIR CELL_TYPE=$CELL_TYPE"
