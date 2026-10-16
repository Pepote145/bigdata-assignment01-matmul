"""Summarise raw measurements and draw the report figures.

Inputs : results/raw_times.csv, results/runs.csv
Outputs: results/summary.csv     median, IQR, min, max, GFLOP/s, slowdown vs C, peak RSS
         results/fits.csv        log-log slope of median time vs n
         results/tables.md       Markdown tables included in the report
         figures/*.png / *.pdf   runtime, normalised time, memory, slowdown

Central measure: median of the R measured repetitions; variability: IQR (Q3-Q1,
linear interpolation) plus min/max. Every raw repetition is kept; nothing is excluded.
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGURES, RESULTS  # noqa: E402

LANGS = ["c", "java", "python"]
LABEL = {"c": "C (gcc -O2)", "java": "Java (HotSpot JIT)", "python": "Python (CPython)"}
# validated categorical slots 1-3 (blue, orange, aqua) + distinct markers, so identity
# never relies on colour alone
COLOR = {"c": "#2a78d6", "java": "#eb6834", "python": "#1baf7a"}
MARKER = {"c": "o", "java": "s", "python": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def set_n_axis(ax, ns):
    """log2 x-axis with one tick per tested n; 1000 and 1024 share a label."""
    ns = sorted(int(v) for v in ns)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    labels = []
    for v in ns:
        if v == 1000 and 1024 in ns:
            labels.append("1000/1024")
        elif v == 1024 and 1000 in ns:
            labels.append("")
        else:
            labels.append(str(v))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.minorticks_off()


def summarise(raw, runs):
    g = raw.groupby(["language", "n"])["time_per_mult_s"]
    s = pd.DataFrame({
        "reps": g.size(),
        "batch": raw.groupby(["language", "n"])["batch"].first(),
        "median_s": g.median(),
        "q1_s": g.quantile(0.25),
        "q3_s": g.quantile(0.75),
        "min_s": g.min(),
        "max_s": g.max(),
    }).reset_index()
    s["iqr_s"] = s["q3_s"] - s["q1_s"]
    s["rel_iqr_pct"] = 100 * s["iqr_s"] / s["median_s"]
    s["gflops"] = 2 * s["n"].astype(float) ** 3 / s["median_s"] / 1e9
    s["ns_per_inner_iter"] = 1e9 * s["median_s"] / s["n"].astype(float) ** 3
    c = s[s.language == "c"].set_index("n")["median_s"]
    s["slowdown_vs_c"] = s.apply(lambda r: r.median_s / c.get(r.n, np.nan), axis=1)
    mem = runs[["language", "n", "peak_rss_bytes", "est_array_bytes", "java_heap_used_bytes"]]
    s = s.merge(mem, on=["language", "n"], how="left")
    s["peak_rss_mb"] = s["peak_rss_bytes"] / 1e6
    s["est_array_mb"] = s["est_array_bytes"] / 1e6
    return s.sort_values(["language", "n"])


def fit_slopes(s):
    """Least-squares slope of log(median time) vs log(n) on three declared subsets."""
    rows = []
    for lang in LANGS:
        d = s[s.language == lang]
        subsets = (("128 <= n <= 384", d.n.between(128, 384)),
                   ("all n >= 256", d.n >= 256),
                   ("n >= 256, no multiple of 512", (d.n >= 256) & (d.n % 512 != 0)))
        for subset, mask in subsets:
            dd = d[mask.values]
            if len(dd) < 2:
                continue
            x, y = np.log(dd.n.astype(float)), np.log(dd.median_s)
            slope, icpt = np.polyfit(x, y, 1)
            r2 = np.corrcoef(x, y)[0, 1] ** 2
            rows.append({"language": lang, "subset": subset, "sizes": " ".join(map(str, dd.n)),
                         "slope": slope, "r2": r2})
    return pd.DataFrame(rows)


def fig_runtime(s):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for lang in LANGS:
        d = s[s.language == lang]
        ax.errorbar(d.n, d.median_s, yerr=[d.median_s - d.min_s, d.max_s - d.median_s],
                    color=COLOR[lang], marker=MARKER[lang], ms=5, lw=2, capsize=2,
                    label=LABEL[lang])
    # cubic reference anchored at C's n=256 median
    c256 = s[(s.language == "c") & (s.n == 256)].median_s.iloc[0]
    nn = np.array([s.n.min(), s.n.max()], float)
    ax.plot(nn, c256 * (nn / 256) ** 3, ls="--", lw=1, color=INK2, label=r"$\propto n^3$ reference")
    set_n_axis(ax, s.n.unique())
    ax.set_yscale("log")
    ax.set_xlabel("matrix size n (n × n)")
    ax.set_ylabel("time per multiplication (s, log scale)")
    ax.set_title("Median runtime vs matrix size (whiskers: min–max of 5 reps)", fontsize=10, color=INK)
    ax.legend(loc="upper left")
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig1_runtime.{ext}")
    plt.close(fig)


def fig_normalised(s):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for lang in LANGS:
        d = s[s.language == lang]
        ax.plot(d.n, d.ns_per_inner_iter, color=COLOR[lang], marker=MARKER[lang], ms=5, lw=2,
                label=LABEL[lang])
    set_n_axis(ax, s.n.unique())
    ax.set_yscale("log")
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("ns per inner-loop iteration\n(median time / n³, log scale)")
    ax.set_title("Normalised cost: flat = pure cubic growth", fontsize=10, color=INK)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig2_normalised.{ext}")
    plt.close(fig)


def fig_memory(s, runs):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ok = runs[runs.peak_rss_bytes.notna() & (runs.peak_rss_bytes != "")].copy()
    ok["peak_rss_bytes"] = ok["peak_rss_bytes"].astype(float)
    for lang in LANGS:
        d = ok[ok.language == lang].sort_values("n")
        good, over = d[d.status == "ok"], d[d.status != "ok"]
        ax.plot(good.n, good.peak_rss_bytes / 1e6, color=COLOR[lang], marker=MARKER[lang], ms=5,
                lw=2, label=LABEL[lang])
        # runs stopped by the time budget: peak RSS after one multiplication, hollow marker
        ax.plot(over.n, over.peak_rss_bytes / 1e6, ls="none", marker=MARKER[lang], ms=7,
                mfc="none", mec=COLOR[lang], mew=1.5)
    nn = np.array(sorted(ok.n.unique()), float)
    ax.plot(nn, 3 * 8 * nn ** 2 / 1e6, ls="--", lw=1, color=INK2, label="3 × 8n² bytes (raw doubles)")
    ax.plot(nn, 3 * 32 * nn ** 2 / 1e6, ls=":", lw=1.2, color=INK2, label="3 × 32n² bytes (boxed floats)")
    set_n_axis(ax, nn)
    ax.set_yscale("log")
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("MB (log scale)")
    ax.plot([], [], ls="none", marker="o", mfc="none", mec=INK2, label="stopped by time budget (1 multiplication)")
    ax.set_title("Peak resident memory (VmHWM) of the benchmark process", fontsize=10, color=INK)
    ax.legend(loc="upper left", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig3_memory.{ext}")
    plt.close(fig)


def fmt_s(x):
    if pd.isna(x):
        return "–"
    if x < 1e-3:
        return f"{x * 1e6:.0f} µs"
    if x < 1:
        return f"{x * 1e3:.1f} ms"
    return f"{x:.2f} s"


def tables(s, runs, fits):
    out = ["<!-- generated by scripts/analyze.py -->", ""]
    out += ["| n | language | median | IQR | min – max | batch | GFLOP/s | × C | peak RSS (MB) |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in s.sort_values(["n", "language"]).iterrows():
        out.append(f"| {r.n} | {r.language} | {fmt_s(r.median_s)} | {fmt_s(r.iqr_s)} "
                   f"({r.rel_iqr_pct:.1f}%) | {fmt_s(r.min_s)} – {fmt_s(r.max_s)} | {r.batch} | "
                   f"{r.gflops:.3f} | {r.slowdown_vs_c:.1f} | {r.peak_rss_mb:.1f} |")
    out += ["", "", "| language | n | status | note |", "|---|---:|---|---|"]
    for _, r in runs[runs.status != "ok"].iterrows():
        note = r.note if isinstance(r.note, str) else ""
        out.append(f"| {r.language} | {r.n} | {r.status} | {note} |")
    out += ["", "", "| language | sizes used in fit | slope | R² |", "|---|---|---:|---:|"]
    for _, r in fits.iterrows():
        out.append(f"| {r.language} | {r.sizes} | {r.slope:.2f} | {r.r2:.3f} |")
    diag_path = RESULTS / "diag_leading_dim.csv"
    if diag_path.exists():
        d = pd.read_csv(diag_path)
        g = d.groupby(["n", "ld"]).time_s.agg(["median", "min", "max", "count"]).reset_index()
        out += ["", "", "| n | leading dim ld | median | min – max | reps | speed-up vs ld = n |",
                "|---:|---:|---:|---:|---:|---:|"]
        for _, r in g.iterrows():
            base = g[(g.n == r.n) & (g.ld == r.n)]["median"].iloc[0]
            out.append(f"| {int(r.n)} | {int(r.ld)} | {fmt_s(r['median'])} | {fmt_s(r['min'])} – "
                       f"{fmt_s(r['max'])} | {int(r['count'])} | {base / r['median']:.2f} |")
        g.to_csv(RESULTS / "diag_leading_dim_summary.csv", index=False, float_format="%.6g")
    (RESULTS / "tables.md").write_text("\n".join(out) + "\n")


def compact_tables(s, runs):
    """Wide tables (one row per n) used in the report."""
    status = {(r.language, r.n): r.status for r in runs.itertuples()}

    def cell(lang, n, col, fmt):
        row = s[(s.language == lang) & (s.n == n)]
        if row.empty:
            st = status.get((lang, n), "")
            return "over budget" if st == "exceeded_time_budget" else ("not run" if st else "–")
        return fmt(row.iloc[0])

    ns = sorted(runs.n.unique())
    t = ["| n | C | Java | Python | Java / C | Python / C |", "|---:|---:|---:|---:|---:|---:|"]
    for n in ns:
        f = lambda r: f"{fmt_s(r.median_s)} ({r.rel_iqr_pct:.1f}%)"
        t.append(f"| {n} | {cell('c', n, 0, f)} | {cell('java', n, 0, f)} | {cell('python', n, 0, f)} | "
                 f"{cell('java', n, 0, lambda r: f'{r.slowdown_vs_c:.2f}')} | "
                 f"{cell('python', n, 0, lambda r: f'{r.slowdown_vs_c:.1f}')} |")
    m = ["| n | C | Java | Python | 3 × 8n² (raw doubles) |", "|---:|---:|---:|---:|---:|"]
    for n in ns:
        vals = []
        for lang in LANGS:
            r = runs[(runs.language == lang) & (runs.n == n)].iloc[0]
            if r.status == "ok":
                vals.append(f"{r.peak_rss_bytes / 1e6:.1f}")
            elif r.status == "exceeded_time_budget":
                vals.append(f"({r.peak_rss_bytes / 1e6:.1f})†")
            else:
                vals.append("not run")
        m.append(f"| {n} | {' | '.join(vals)} | {3 * 8 * n * n / 1e6:.1f} |")
    (RESULTS / "report_tables.md").write_text("\n".join(t) + "\n\n" + "\n".join(m) + "\n")


def main():
    raw = pd.read_csv(RESULTS / "raw_times.csv")
    runs = pd.read_csv(RESULTS / "runs.csv")
    FIGURES.mkdir(exist_ok=True)
    s = summarise(raw, runs)
    s.to_csv(RESULTS / "summary.csv", index=False, float_format="%.6g")
    fits = fit_slopes(s)
    fits.to_csv(RESULTS / "fits.csv", index=False, float_format="%.4f")
    # marginal memory: peak RSS = a + b * (3 n^2) over completed runs with n >= 256
    ok = runs[(runs.status == "ok") & (runs.n >= 256)]
    mem_rows = []
    for lang, d in ok.groupby("language"):
        b, a = np.polyfit(3 * d.n.astype(float) ** 2, d.peak_rss_bytes.astype(float), 1)
        mem_rows.append({"language": lang, "bytes_per_element": b, "intercept_mb": a / 1e6,
                         "sizes": " ".join(map(str, d.n))})
    pd.DataFrame(mem_rows).to_csv(RESULTS / "memory_fit.csv", index=False, float_format="%.2f")
    fig_runtime(s)
    fig_normalised(s)
    fig_memory(s, runs)
    tables(s, runs, fits)
    compact_tables(s, runs)
    print((RESULTS / "tables.md").read_text())


if __name__ == "__main__":
    main()
