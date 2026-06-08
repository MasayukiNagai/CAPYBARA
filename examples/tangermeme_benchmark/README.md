# Tangermeme Data-Processing Benchmark

Benchmark the current `tangermeme.io.extract_loci` path against the older local
FASTA/BigWig extraction loop used in the main-branch PRO-cap example.

Examples:

```bash
.venv/bin/python examples/tangermeme_benchmark/benchmark_data_processing.py \
  --dataset atac \
  --proj-dir /grid/koo/home/shared/data/chrombpnet \
  --cell-type K562 \
  --fold 1 \
  --n-loci 1000
```

For an exact old-loader comparison, install `pyBigWig`. If it is unavailable,
the benchmark can run the local loop with `pybigtools` as a fallback backend,
which is useful for separating local-loop overhead from tangermeme overhead.
