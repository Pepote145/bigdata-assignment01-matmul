# Assignment 1 – Basic Matrix Multiplication in C, Java and Python

BIG DATA · Grado en Ciencia e Ingeniería de Datos · ULPGC · 2026-2027

**Author:** Jose Jesus Perez Suarez  
**Repository:** https://github.com/Pepote145/bigdata-assignment01-matmul

**Research question.** How does matrix size *n* affect the runtime and peak memory of the
same straightforward triple-loop (i-j-k) dense matrix multiplication implemented in C, Java
and Python, on one documented machine?

The report is `Assignment01_PerezSuarez_Report.pdf` (source: `report/report.md`).

## Repository layout

```
config.json                 every experimental parameter (seed, sizes, reps, budgets, flags)
src/c/matmul.c              C implementation  (contiguous row-major double array)
src/java/MatMul.java        Java implementation (double[][])
src/python/matmul.py        Python implementation (list of lists, standard library only)
src/c/diag_leading_dim.c    supplementary diagnostic: C kernel with padded rows (not part of the comparison)
scripts/generate_inputs.py  shared input matrices -> data/A_n{n}.bin, data/B_n{n}.bin
scripts/validate.py         correctness tests (identity, zero, hand-computed, vs NumPy, cross-language)
scripts/env_info.py         records CPU, cores, RAM, OS, compiler/JVM/Python versions
scripts/run_benchmarks.py   benchmark harness -> results/raw_times.csv, results/runs.csv
scripts/analyze.py          summary tables, slope fits and figures
results/                    raw and summarised measurements (CSV), environment.json, validation.csv
figures/                    figures used in the report (PNG and PDF)
report/report.md            report source (pandoc)
```

## Requirements

Versions used for the submitted results (see `results/environment.json` for the full record):

| Tool | Version |
|---|---|
| GCC | 13.3.0 (`-O2 -std=c11 -ffp-contract=off`) |
| Java (JDK, `javac` + `java`) | 21 (OpenJDK, `-Xmx2g -XX:+UseSerialGC`) |
| Python | 3.13 (CPython) |
| NumPy / pandas / matplotlib | 2.5.3 / 3.0.5 / 3.11.2 – only for input generation, validation and analysis |
| pandoc + XeLaTeX | only to rebuild the PDF report |

Linux or macOS. On Windows use WSL (the harness uses `os.wait4` and the programs read
`/proc/self/status` for memory on Linux).

```bash
pip install numpy pandas matplotlib
```

## Reproduce everything

```bash
make build        # gcc -O2 -std=c11 -ffp-contract=off  +  javac
make inputs       # python3 scripts/generate_inputs.py   (~170 MB of .bin files, seed in config.json)
make env          # python3 scripts/env_info.py          -> results/environment.json
make validate     # python3 scripts/validate.py          -> results/validation.csv (must say 34/34)
make bench        # python3 scripts/run_benchmarks.py    -> results/raw_times.csv, results/runs.csv
make diag         # bash scripts/run_diagnostic.sh       -> results/diag_leading_dim.csv (supplementary, ~5 min)
make analyze      # python3 scripts/analyze.py           -> results/summary.csv, fits.csv, tables.md, figures/
make report       # pandoc -> Assignment01_PerezSuarez_Report.pdf
```

`make all` runs build → analyze in order. `make report` needs pandoc, XeLaTeX and the
FreeSerif and DejaVu Sans Mono fonts (`apt install fonts-freefont-ttf fonts-dejavu`). The full benchmark takes roughly 20–30 minutes on
the 2-vCPU machine documented in the report; close other programs while it runs.

The input `.bin` files are not committed (they are large); `make inputs` regenerates them
bit-for-bit. Their SHA-256 hashes are recorded in `data/manifest.json` so you can check that.

### Running a single configuration by hand

```bash
# arguments: A.bin B.bin n warmups reps min_batch_seconds max_seconds_per_multiplication
build/matmul_c                    bench data/A_n512.bin data/B_n512.bin 512 1 5 0.05 60
java -Xmx2g -XX:+UseSerialGC -cp build/java MatMul bench data/A_n512.bin data/B_n512.bin 512 5 5 0.05 60
python3 src/python/matmul.py       bench data/A_n512.bin data/B_n512.bin 512 1 5 0.05 60
```

Each prints `WARMUP`, `META`, `REP,<rep>,<batch>,<seconds per multiplication>,<checksum>`,
`PEAKRSS,<bytes>` and `STATUS` lines.

## Measurement protocol (summary)

* **Inputs**: identical binary files for all languages; uniform values in [-1, 1), float64,
  PCG64 seeded with `SeedSequence([20261014, n])`.
* **Timer boundary**: the multiplication kernel only. Loading inputs and allocating A, B and C
  happen before timing. Monotonic timers: `clock_gettime(CLOCK_MONOTONIC)`, `System.nanoTime()`,
  `time.perf_counter_ns()`.
* **Warm-ups** (excluded from samples): C 1, Java 5 (JIT), Python 1.
* **Repetitions**: 5 measured repetitions per configuration, each in a fresh process per
  (language, n). Kernels shorter than 50 ms are batched; the reported value is time per
  multiplication.
* **Result consumed**: a checksum of C is computed and printed after every repetition.
* **Budgets**: 60 s per multiplication; 4 GB estimated array storage. A configuration over
  the time budget is recorded as `exceeded_time_budget` and larger sizes are not attempted for
  that language.
* **Memory metric**: peak resident set size (VmHWM) of the benchmark process, read by the
  program itself. It includes the language runtime (JVM, interpreter).

## Raw data columns

`results/raw_times.csv`: `language, n, repetition, batch, time_per_mult_s, checksum, dtype, timestamp_utc`
– one row per measured repetition, nothing removed.

`results/runs.csv`: one row per (language, n): status, warm-up times, batch size, peak RSS,
estimated array bytes, process wall time, Java heap in use, exit code, note.
