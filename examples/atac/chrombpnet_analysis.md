# ChromBPNet Benchmark Analysis
## Reference for CAPY vs. ChromBPNet Comparison

> **Context:** ChromBPNet (Pampari et al., 2024) is a bias-factorized BPNet model for predicting base-resolution chromatin accessibility profiles from DNA sequence. This document catalogs every benchmark/analysis in the paper, the datasets and metrics used, and which figures are worth reproducing for a direct CAPY comparison. The plan is to load pretrained ChromBPNet weights into bpnet-lite (PyTorch) and compare against a CAPY model trained on the same ATAC-seq data.

---

## 1. Model Overview

| Property | Value |
|---|---|
| Input | 2114 bp one-hot encoded DNA sequence |
| Output | 1000 bp base-resolution coverage profile |
| Heads | Count head (log total counts, MSE loss) + Profile head (multinomial probability, MNLL loss) |
| Architecture | 512 filters, 1 conv (21 bp) + 8 dilated conv (3 bp, dilation ×2 each), effective receptive field 1041 bp |
| Bias model | Shallow BPNet (128 filters, 4 dilated layers, 81 bp receptive field), frozen during ChromBPNet training |
| Training | 5-fold cross-validation, Adam lr=0.001, random jitter ±500 bp around summit, 1:10 non-peak:peak ratio |

---

## 2. All Benchmarks and Analyses

### 2.1 Profile Prediction Performance (Main)
**What:** Evaluate how well ChromBPNet predicts total counts and profile shape on held-out test chromosomes.

**Datasets:**
- K562 ATAC-seq: ENCODE, 572M reads (primary cell line for development)
- K562 DNase-seq: ENCODE, 68M reads
- HepG2, GM12878, IMR90, H1-hESC: ATAC-seq and DNase-seq from ENCODE (Table 1 in paper)
- Peaks: pseudoreplicated IDR peaks from ENCODE; peak-calling via MACS2 for DNase-seq
- Background: GC-content-matched non-peak bins (2× peak count)

**Metrics:**
- **Pearson r** between predicted and observed log(total counts) across peak regions in test chromosomes
- **Jensen-Shannon Distance (JSD)** between predicted and observed base-resolution probability profiles
- Upper bound: JSD between pseudo-replicate profiles of the same experiment
- Baselines: JSD vs. average profile across all peaks; JSD vs. scrambled profiles

**Results (K562):**
- ATAC-seq: Pearson r = 0.70 ± 0.02 (counts); JSD approaching pseudo-replicate upper bound
- DNase-seq: Pearson r = 0.71 ± 0.02 (counts); similar JSD performance
- 5 cell lines median: Pearson r = 0.69 (counts), median JSD = 0.62 (profile)

**Key figures:** Fig 1c (counts scatter plot), Fig 1d (JSD distribution vs. pseudo-replicate upper bound), Extended Fig 2b-c

---

### 2.2 Genome-wide Peak vs. Background Discrimination
**What:** Assess whether ChromBPNet predictions can discriminate accessible peaks from background across whole chromosomes.

**Datasets:** Same ENCODE cell lines; positive = IDR peaks (100 bp centered on summit); negatives = genome-wide 100 bp bins excluding peaks and blacklist regions.

**Metrics:**
- **auROC** (Area Under ROC Curve)
- **Average Precision (AP)** (area under precision-recall curve)

**Results (K562):**
- ATAC-seq: auROC = 0.98 ± 0.001, AP = 0.42 ± 0.01
- DNase-seq: auROC = 0.98 ± 0.001, AP = 0.46 ± 0.02

---

### 2.3 Bias Correction Benchmarks
**What:** Compare ChromBPNet's bias correction against HINT-ATAC, TOBIAS, naked DNA bias models, and no-correction baselines. Evaluated on GM12878 ATAC-seq and DNase-seq.

**Datasets:** GM12878 ATAC-seq (ENCODE), GM12878 DNase-seq (ENCODE); naked DNA ATAC-seq (SRA: SRX030445), naked DNA DNase-seq (SRA: SRR1565781, SRR1565782)

**Metrics:**
- TF-MODISCO motif discovery from profile contribution scores: presence/absence of Tn5/DNase-I bias motifs vs. TF motifs
- Marginal footprint profiles at Tn5/DNase-I bias motifs and TF motifs
- Visual inspection of contribution scores at exemplar loci

**Key figures:** Fig 2b (exemplar locus tracks), Fig 2e (TF-MODISCO seqlet frequencies), Fig 2f (marginal footprints at Tn5 bias motifs), Fig 2g (marginal footprints at TF motifs), Extended Figs 4, 5

---

### 2.4 ATAC-seq vs. DNase-seq Concordance (Bias Correction Validation)
**What:** Measure how bias correction improves agreement between independently trained ATAC-seq and DNase-seq models.

**Dataset:** K562 ATAC-seq and DNase-seq, 30,000 randomly sampled overlapping peaks.

**Metrics:**
- JSD between observed ATAC-seq and DNase-seq profiles
- JSD between uncorrected predicted profiles (ATAC-seq model vs. DNase-seq model)
- JSD between bias-corrected predicted profiles
- JSD between count/profile contribution scores from both models
- Pearson r between marginal footprint depths across TF motifs (ATAC vs. DNase)

**Results:**
- Measured profiles: JSD = 0.81 ± 0.08
- Uncorrected predicted: JSD = 0.58 ± 0.03
- Bias-corrected predicted: JSD = 0.26 ± 0.08
- Marginal footprint depth correlation: Pearson r = 0.98

**Key figures:** Fig 3a (exemplar locus), Fig 3b (JSD distributions), Fig 3c (contribution score JSD), Fig 3d (motif frequency comparison), Fig 3e-h (marginal footprints)

---

### 2.5 Performance at Low Sequencing Depth (Subsampling Analysis)
**What:** Evaluate ChromBPNet's ability to impute profiles and motifs from sparse datasets.

**Dataset:** GM12878 ATAC-seq subsampled to 250M, 100M, 50M, 25M, and 5M reads (from 572M full dataset).

**Metrics:**
- JSD between observed profiles at each depth vs. full-depth (raw data degradation)
- JSD between ChromBPNet predicted profiles at each depth vs. full-depth reference model
- JSD for count and profile contribution scores vs. full-depth
- **Motif recall**: fraction of TF-MODISCO motif instances from the full-depth model recovered at each subsampled depth (using FiNeMo)
- Marginal footprint fidelity across depths

**Results:**
- Observed profiles degrade rapidly (JSD 0.3→0.9 from 250M→5M)
- ChromBPNet predicted profiles remain stable (JSD ~0.15 across 572M→25M; slight degradation at 5M: JSD = 0.19)
- Motif recall stable down to 25M; substantial loss for rare motifs at 5M

**Key figures:** Fig 4a (exemplar locus tracks), Fig 4b (JSD distributions), Fig 4c (motif recall), Fig 4d (marginal footprints)

---

### 2.6 TF Motif Lexicon Discovery (5 ENCODE Cell Lines)
**What:** Identify the full compendium of TF motifs that drive chromatin accessibility in each cell line.

**Dataset:** ATAC-seq and DNase-seq from 5 ENCODE Tier-1 cell lines (K562, HepG2, GM12878, IMR90, H1-hESC). All peaks per cell line. TF-MODISCO run with 1,000,000 seqlets.

**Metrics / Analysis:**
- Number of TF-MODISCO motifs from count and profile contribution scores (26–47 from count head, 49–114 from profile head per cell line × assay)
- Unified non-redundant motif lexicon per cell line (41–60 motifs per cell line)
- TOMTOM annotation against JASPAR database
- Fisher's exact test enrichment against ENCODE TF ChIP-seq peaks

**Key figures:** Fig 5a (top 10 count-head motifs per cell line), Extended Fig 6a (top 10 profile-head motifs), Extended Fig 7a-b (DNase-seq motifs)

---

### 2.7 Cooperative Motif Syntax Analysis
**What:** Test whether ChromBPNet learns strict spacing/orientation constraints in composite TF motifs.

**Dataset:** IMR90 ATAC-seq/DNase-seq (FOS-TEAD composite); GM12878 (SPI1-IRF, AP1-IRF composites).

**Metrics:**
- Marginal footprint strength as a function of inter-motif spacing
- Comparison of composite footprint vs. sum of individual motif footprints (super-additivity)
- Corroboration with BPNet models trained on FOS ChIP-seq in IMR90

**Key figures:** Fig 5c (FOS-TEAD spacing and cooperativity), Extended Fig 6b (composite motifs), Extended Fig 6c (FOS ChIP-seq corroboration)

---

### 2.8 Motif Contribution Score vs. TF ChIP-seq Correlation (vs. TOBIAS)
**What:** Benchmark whether ChromBPNet contribution scores at motif instances reflect TF occupancy better than total accessibility or footprinting methods.

**Dataset:**
- 72 unique TF-MODISCO motif × TF ChIP-seq pairs across GM12878, HepG2, K562, H1-hESC
- 138 motif × experiment pairs total
- BPNet models trained on TF ChIP-seq data as reference occupancy scores

**Metrics:**
- Pearson r between ChromBPNet count contribution scores at motif instances vs. ChIP-seq BPNet contribution scores
- Compared against: (1) total peak-level accessibility (ATAC/DNase counts), (2) TOBIAS footprint scores

**Results (CTCF in HepG2):**
- ChromBPNet count contributions: r > 0.89
- Total accessibility: r = -0.1
- TOBIAS footprint scores: r = 0.2

**Key figures:** Fig 5e (CTCF scatter), Fig 5f (comparison bar for CTCF), Fig 5g (systematic comparison across motifs)

---

### 2.9 Variant Effect Prediction: DNase-seq QTLs in Yoruba LCLs (Primary Benchmark)
**What:** Predict effects of variants on chromatin accessibility and compare to known dsQTLs.

**Dataset:** ~560 significant dsQTLs from 70 Yoruban LCLs (from Degner et al. 2012 / deltaSVM paper); ~27K matched control variants within DNase-seq peaks. Models trained on GM12878 (European ancestry reference).

**Variant Scores (ChromBPNet):**
1. **logFC**: log fold-change of predicted total counts (ref vs. alt allele)
2. **JSD**: Jensen-Shannon Distance between ref and alt allelic profiles (profile shape change)
3. **AAQ**: Active Allele Quantile (percentile of the stronger allele's predicted coverage among all peaks)
4. **IES**: Integrative Effect Size = mean(logFC) × mean(JSD) across 5 folds
5. **IPS**: Integrative Prioritization Score = mean(|logFC|) × mean(JSD) × mean(AAQ) across 5 folds

**Metrics:**
- **Average Precision (AP)** on precision-recall curve (variant classification)
- **Pearson r** between predicted logFC and observed QTL effect sizes (betas)

**Comparison models:** gkm-SVM (deltaSVM), Enformer (reported scores), Enformer (recomputed local scores)

**Results (GM12878 ATAC-seq 572M reads):**
| Model | AP | r (effect size) |
|---|---|---|
| ChromBPNet ATAC-seq (572M) | **0.54** | **0.76** |
| ChromBPNet ATAC-seq (50M) | 0.46 | 0.73 |
| ChromBPNet DNase-seq (68M) | 0.43 | 0.73 |
| gkm-SVM DNase-seq (68M) | 0.19 | — |
| Enformer reported | 0.33 | 0.56 |
| Enformer recomputed local | 0.53 | 0.73 |

**Key figures:** Fig 6a (PR curves), Fig 6b (effect size scatter), Extended Fig 8a (score comparison), Extended Fig 8b (read depth vs. AP), Extended Fig 8c (read depth vs. r)

---

### 2.10 Variant Effect Prediction: caQTLs in European LCLs
**What:** Replicate dsQTL benchmark with an independent ATAC-seq caQTL study.

**Dataset:** ~7,900 significant caQTLs (-log10 p > 6) from ~91 European LCLs; ~87K control variants. Processed with nf-core/atacseq pipeline.

**Metrics:** AP (variant classification), Pearson r (effect size), evaluated at multiple significance thresholds.

**Key figures:** Fig 6d (PR curves), Fig 6e (effect size scatter), Extended Fig 9a-b

---

### 2.11 Variant Effect Prediction: caQTLs in African LCLs
**What:** Test generalization of models trained on a single European reference (GM12878) to African ancestry QTLs.

**Dataset:** ~6,821 significant caQTLs (-log10 p > 5) from ~100 African individuals (6 sub-populations: Esan, Maasai, Mende, Gambian, Luhya, Yoruba); ~72K controls.

**Also:** 5,220 allele-specific chromatin accessibility (ASC) sites for allelic imbalance analysis.

**Metrics:**
- AP (variant classification), Pearson r (effect size)
- **Cross-ancestry logFC/JSD correlation** between models trained on each of 6 African sub-populations + European reference (correlation matrix)
- Pearson r between predicted logFC and observed allelic imbalance effect sizes

**Results (cross-ancestry):**
- Between African subgroup models: r = 0.96-0.97
- African vs. European (GM12878) model: r = 0.93-0.95

**Key figures:** Fig 6f (PR curves), Fig 6g (effect size scatter), Fig 6h (allelic imbalance scatter), Fig 6i (cross-ancestry correlation matrix), Extended Fig 9c-f

---

### 2.12 Variant Effect Prediction: scATAC-seq Primary Cells (Microglia + SMCs)
**What:** Extend variant effect prediction to disease-relevant primary cells using pseudobulk scATAC-seq.

**Datasets:**
- **Microglia**: pseudobulk from GEO GSE147672; benchmarked against 877 unique microglia caQTLs
- **Smooth Muscle Cells (SMC)**: GEO GSE175621; benchmarked against 386 SMC caQTLs

**Metrics:** Pearson r between predicted logFC and observed QTL effect sizes

**Results:** r = 0.6 (microglia), r = 0.7 (SMC)

**Key figures:** Fig 6j-k, Extended Fig 9g-h

---

### 2.13 Variant Effect Prediction: SPI1 Binding QTLs (Pioneer TF)
**What:** Test whether chromatin accessibility variant scores predict upstream effects on pioneer TF occupancy.

**Dataset:** 447 high-confidence SPI1 bQTLs (from 999,799 tested across 60 Yoruban LCLs via pooled ChIP-seq); 57,612 control variants (from African caQTL dataset, -log10 p > 1).

**Metrics:** AP (variant classification), Pearson r (effect size = log fold change in allele ChIP frequencies)

**Results (GM12878):**
| Model | AP | r |
|---|---|---|
| ChromBPNet ATAC-seq (572M) | **0.37** | **0.59** |
| ChromBPNet DNase-seq (68M) | 0.33 | 0.59 |
| Enformer reported | 0.14 | — |
| Enformer recomputed local | 0.34 | 0.53 |

**Key figures:** Fig 7a (exemplar locus), Fig 7b (PR curves), Fig 7c (effect size scatter), Extended Fig 10a-c

---

### 2.14 Variant Effect Prediction: MPRA Saturated Mutagenesis (CAGI5)
**What:** Compare predicted variant effects on chromatin accessibility to experimentally measured reporter activity across >30,000 mutations in 20 disease-associated regulatory elements.

**Dataset:** CAGI5 competition MPRA data; 9 promoters (e.g. IRF4, IRF6, MYC, SORT1) and 5 enhancers (e.g. TERT, LDLR, F9, HBG1) in matched cell lines.

**ChromBPNet models used:** Cell-type matched DNase-seq models (HepG2, K562, HEK293, pancreas, glioblastoma, keratinocyte, SK-MEL — see Table 4).

**Metrics:** Pearson r between predicted effect sizes and observed MPRA effect sizes, per cRE locus. Compared to Enformer.

**Results:** ChromBPNet outperforms Enformer at 5 of 14 loci (IRF4, SORT1, HNF4A, MSMB, HBG1), matches within ±0.05 at 9 loci.

**Key figures:** Fig 7d (scatter per cRE, exemplar scatter for HBG1), Extended Fig 10d (IRF4, SORT1 examples)

---

### 2.15 GWAS Fine-Mapped Variant Enrichment (Blood Traits)
**What:** Test whether ChromBPNet effect scores enrich for causally fine-mapped GWAS variants.

**Dataset:**
- 12,951 fine-mapped variants from 9 UK Biobank blood traits (Hb, HbA1c, Plt, RBC, WBC, MCH, MCV, MCHC) enriched in K562 accessible regions
- 11,903,173 background non-coding variants as control
- ChromBPNet model: K562 ATAC-seq and DNase-seq

**Metrics:** Enrichment of ChromBPNet high-scoring variants (at varying thresholds) overlapping fine-mapped variants at various PIP thresholds.

**Results:** Up to **70-fold enrichment** at PIP ≥ 0.4 and stringent ChromBPNet score thresholds.

**Key figures:** Fig 7e (enrichment curves), Extended Fig 10e

---

### 2.16 Rare Disease / De Novo Variant Analysis (Transgenic Validation)
**What:** Showcase interpretation of de novo variant in a rare neurodevelopmental disorder patient; validate with transgenic mouse reporter assay.

**Dataset:** Fetal brain scATAC-seq pseudobulk from glutamatergic neurons (DDD cohort patient variant chr6:14501369 A>G).

**Analysis:** Predicted accessibility change, contribution score comparison (reference vs. alternate allele), identification of disrupted TF motif (NR2F1 repressor site created by risk allele). Validated in transgenic E11.5 mice.

**Key figures:** Fig 7g (predicted profiles and contribution scores), Fig 7h (transgenic mouse reporter images)

---

## 3. Summary of Datasets Used

| Dataset | Source | Cell Type / Context | Assay | Purpose |
|---|---|---|---|---|
| ENCODE Tier-1 cell lines | ENCODE portal | K562, HepG2, GM12878, IMR90, H1-hESC | ATAC-seq + DNase-seq | Model training & prediction benchmarks |
| H1-hESC ATAC-seq | GEO GSE267154 | H1 hESC | ATAC-seq | H1-hESC model (generated in-house) |
| GM12878 subsampled | ENCODE ENCSR637XSC | GM12878 | ATAC-seq | Read depth analysis (250M→5M) |
| Naked DNA | SRA SRX030445 (ATAC), SRR1565781-82 (DNase) | None (naked genomic DNA) | ATAC-seq, DNase-seq | Bias model comparison |
| Smooth muscle cells | GEO GSE175621 | Coronary artery SMC | scATAC-seq pseudobulk | caQTL benchmark |
| Microglia | GEO GSE147672 | Microglia | scATAC-seq pseudobulk | caQTL benchmark |
| African ancestry LCLs | ENCODE portal (6 populations) | LCLs (Gambian, Luhya, Esan, Maasai, Mende, Yoruba) | ATAC-seq | Cross-ancestry model training |
| Fetal brain scATAC-seq | Synapse syn63395628 | Glutamatergic neurons | scATAC-seq pseudobulk | Rare disease variant analysis |
| dsQTLs (Yoruba LCL) | Degner et al. (GEO GSE31388) | 70 Yoruban LCLs | DNase-seq QTLs | Primary variant prediction benchmark |
| caQTLs (European LCL) | Kuningas et al. (Zenodo) | ~91 European LCLs | ATAC-seq QTLs | Variant prediction replication |
| caQTLs (African LCL) | Alasoo et al. (obtained from authors) | ~100 African LCLs (6 subgroups) | ATAC-seq QTLs | Cross-ancestry variant benchmark |
| SMC caQTLs | Miller et al. (Supp Data 6) | Coronary artery SMC | ATAC-seq QTLs | Primary cell variant benchmark |
| Microglia caQTLs | Kosoy et al. (Synapse syn30863713) | Microglia | ATAC-seq QTLs | Primary cell variant benchmark |
| SPI1 bQTLs | Tehranchi et al. (obtained from authors) | 60 Yoruban LCLs | ChIP-seq QTLs | Pioneer TF variant benchmark |
| CAGI5 MPRA | Kircher et al. (personal communication) | Multiple cell lines | MPRA | Reporter activity correlation |
| GWAS blood traits | UK Biobank + Finucane lab finemap | Blood cells | Fine-mapping | GWAS enrichment |
| DDD cohort variants | DDD cohort | Fetal brain | Patient de novo SNV | Rare disease analysis |

---

## 4. Core Evaluation Metrics Summary

| Metric | What it measures | Used in |
|---|---|---|
| Pearson r (log total counts) | Total accessibility prediction accuracy | Benchmarks 2.1, 2.9–2.16 |
| JSD (Jensen-Shannon Distance) | Profile shape prediction accuracy | Benchmarks 2.1, 2.3–2.5 |
| auROC | Peak vs. background discrimination | Benchmark 2.2 |
| Average Precision (AP) | Peak vs. background + variant classification | Benchmarks 2.2, 2.9–2.13 |
| logFC | Predicted allelic effect size (total counts) | Benchmarks 2.9–2.16 |
| JSD (allelic) | Predicted allelic profile shape difference | Benchmarks 2.9–2.16 |
| AAQ | Percentile rank of stronger allele in peak distribution | Benchmarks 2.9–2.13 |
| IPS | Integrative prioritization (logFC × JSD × AAQ) | Benchmarks 2.9–2.13 |
| Enrichment fold | GWAS fine-mapped variant overlap enrichment | Benchmark 2.15 |

---

## 5. Figures to Reproduce for CAPY Comparison

These are the figures that directly compare ChromBPNet to baselines and would be natural to reproduce with CAPY on the same data/tasks.

### Tier 1 (Core — must reproduce)

| Figure | What it shows | Why reproduce |
|---|---|---|
| **Fig 1c** | Pearson r: predicted vs. observed log total counts on test chromosomes (K562 ATAC-seq, scatter plot) | Direct head-to-head: CAPY vs. ChromBPNet counts prediction |
| **Fig 1d** | JSD distribution: predicted vs. observed profiles (blue) vs. pseudo-replicate upper bound (red) vs. baselines | Core profile prediction metric; pseudo-replicate bound is the ceiling to aim for |
| **Fig 6a** | PR curves for variant classification (dsQTLs, Yoruba LCLs): gkm-SVM vs. ChromBPNet vs. Enformer | The primary variant effect prediction benchmark |
| **Fig 6b** | Scatter: predicted logFC vs. observed dsQTL effect sizes (Pearson r) | Effect size prediction quality |

### Tier 2 (Important — reproduce if feasible)

| Figure | What it shows | Why reproduce |
|---|---|---|
| **Extended Fig 2b-c** | Pearson r and JSD across all 5 ENCODE cell lines (box plots per fold) | Shows generalization; use same cell lines |
| **Fig 4b** | JSD distributions at different read depths (250M→5M) | Shows robustness; relevant if training on shallow ATAC data |
| **Fig 4c** | Motif recall vs. read depth | Shows sensitivity at low coverage |
| **Fig 5g** | Systematic motif contribution score vs. ChIP-seq BPNet correlation across motifs | Shows whether CAPY contributon scores better capture TF occupancy than ChromBPNet |
| **Fig 6d-e** | European LCL caQTL classification + effect size scatter | Independent replication of dsQTL benchmark |

### Tier 3 (Nice to have)

| Figure | What it shows | Effort vs. payoff |
|---|---|---|
| **Fig 3b** | ATAC-seq vs. DNase-seq JSD before/after bias correction | High effort (need both assays); shows concordance |
| **Fig 3h** | Marginal footprint depth correlation ATAC vs. DNase | Requires both assays + TF-MODISCO |
| **Fig 7b** | SPI1 bQTL variant classification | Requires SPI1 bQTL dataset |
| **Fig 7d** | MPRA correlation per cRE locus | Requires CAGI5 MPRA data |
| **Fig 7e** | GWAS enrichment curves | High effort; requires UK Biobank fine-mapping |

---

## 6. Practical Notes for the CAPY Benchmark

### What data to obtain (same as ChromBPNet training data)

1. **Primary training/eval:** ENCODE ATAC-seq for K562, GM12878, HepG2, IMR90, H1-hESC
   - All available at ENCODE portal (see ENCODE file IDs in paper Table 1)
   - Use pseudoreplicated peaks from ENCODE portal
   - Apply same read shifts: +4/-4 bp for ATAC-seq

2. **Variant benchmark:** dsQTL dataset (Yoruba LCL, Degner et al.)
   - Available from GEO GSE31388 (effect sizes) + deltaSVM paper Supp Table 1 (positive/negative sets)
   - Standardized benchmark dataset at Synapse syn64126763

3. **Reference variant scores from ChromBPNet** are available at Synapse syn59449898

### Key preprocessing details from the paper

- Tn5 strand shifts: +4 bp (+ strand), **-4 bp (- strand)** — NOT the conventional +4/-5
- DNase-I shifts: 0 bp (+ strand), +1 bp (- strand) — NOT 0/0
- GC-matched non-peak background regions: 2× peak count, exclude blacklist regions (ENCFF356LFX)
- Exclude peaks beyond 0.99 quantile of total counts during training
- 5-fold cross-validation: specific chromosome splits in paper Table 2

### Baseline to beat

For the primary dsQTL benchmark (Fig 6a, 6b):
- gkm-SVM: AP = 0.19 (easy to beat)
- ChromBPNet DNase-seq (68M): AP = 0.43, r = 0.73
- ChromBPNet ATAC-seq (572M): AP = 0.54, r = 0.76
- Enformer (recomputed local): AP = 0.53, r = 0.73

### What ChromBPNet reports that CAPY should match or exceed

For profile prediction on K562 ATAC-seq (held-out test chromosomes):
- Pearson r ≥ 0.70 for total counts
- JSD approaching pseudo-replicate concordance

For variant classification (dsQTLs):
- AP ≥ 0.43 at matched read depth (~68M)

---

## 7. Code and Data Availability

| Resource | Location |
|---|---|
| ChromBPNet code | https://github.com/kundajelab/chrombpnet |
| Figure reproduction code | https://github.com/kundajelab/chrombpnet-figures |
| Models + outputs (ENCODE cell lines) | ENCODE portal http://encodeproject.org |
| All other models + variant scores | Synapse syn59449898 |
| Standardized benchmark datasets | Synapse syn64126763 |
| African ancestry models | Synapse syn59651013 |
| SMC models | Synapse syn59479965 |
| Microglia models | Synapse syn59479966 |
| Fetal brain models | Synapse syn63395628 |
| H1-hESC ATAC-seq data | GEO GSE267154 |
| Variant scorer code | https://github.com/kundajelab/variant-scorer |
