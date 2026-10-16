"""Benchmark harness: runs every (language, n) configuration in its own process.

Protocol (see config.json for the values):
  * sizes are run in increasing order; for each size, C -> Java -> Python
  * each process: load shared inputs, W warm-ups (not reported), then R measured
    repetitions of the kernel only (monotonic timer inside each program)
  * very short kernels are batched: batch = ceil(min_batch_seconds / last warm-up),
    and each repetition reports time per multiplication = batch time / batch
  * time budget: if one multiplication exceeds max_seconds_per_multiplication the
    program stops and the configuration is recorded as exceeded_time_budget;
    larger sizes are then not attempted for that language
  * memory budget: a configuration is not attempted if its estimated array storage
    exceeds memory_budget_bytes
  * memory metric: peak resident set size of the benchmark process, reported by
    the program itself (Linux: VmHWM from /proc/self/status). It includes the
    language runtime. The wait4() ru_maxrss is also stored, but on Linux it is
    inherited across execve from the spawning Python harness (~11 MB floor), so it
    is NOT used for analysis

Writes results/raw_times.csv (one row per measured repetition, never filtered)
and results/runs.csv (one row per configuration, incl. warm-ups and status).
"""
import csv
import datetime as dt
import os
import subprocess
import sys
import tempfile
import threading
import time

from common import DATA, RESULTS, command, load_config


def run_process(argv, hard_timeout):
    """Run argv, return (stdout_text, exit_code, peak_rss_bytes, wall_s, killed)."""
    with tempfile.TemporaryFile(mode="w+") as out:
        t0 = time.monotonic()
        proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.DEVNULL)
        killed = {"flag": False}

        def kill():
            killed["flag"] = True
            proc.kill()

        timer = threading.Timer(hard_timeout, kill)
        timer.start()
        _, status, rusage = os.wait4(proc.pid, 0)
        timer.cancel()
        proc.returncode = os.waitstatus_to_exitcode(status)
        wall = time.monotonic() - t0
        # ru_maxrss is KiB on Linux and bytes on macOS
        rss = rusage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        out.seek(0)
        return out.read(), proc.returncode, rss, wall, killed["flag"]


def main():
    cfg = load_config()
    reps = cfg["repetitions"]
    limit = cfg["max_seconds_per_multiplication"]
    stopped = {}  # language -> reason
    RESULTS.mkdir(exist_ok=True)
    raw_path, runs_path = RESULTS / "raw_times.csv", RESULTS / "runs.csv"
    raw_f = open(raw_path, "w", newline="")
    runs_f = open(runs_path, "w", newline="")
    raw = csv.writer(raw_f)
    runs = csv.writer(runs_f)
    raw.writerow(["language", "n", "repetition", "batch", "time_per_mult_s", "checksum",
                  "dtype", "timestamp_utc"])
    runs.writerow(["language", "n", "status", "warmups", "warmup_times_s", "batch",
                   "reps_measured", "peak_rss_bytes", "rusage_maxrss_bytes", "est_array_bytes", "process_wall_s",
                   "java_heap_used_bytes", "exit_code", "note"])

    for n in cfg["sizes"]:
        for lang in cfg["languages"]:
            est = 3 * n * n * cfg["bytes_per_element_estimate"][lang]
            base = [lang, n]
            if lang in stopped:
                runs.writerow(base + ["not_attempted", "", "", "", 0, "", "", est, "", "", "",
                                      f"stopped after {stopped[lang]}"])
                continue
            if est > cfg["memory_budget_bytes"]:
                runs.writerow(base + ["not_attempted_memory_budget", "", "", "", 0, "", "", est,
                                      "", "", "", "estimate exceeds budget"])
                stopped[lang] = f"memory budget at n={n}"
                continue

            warm = cfg["warmup"][lang]
            a, b = DATA / f"A_n{n}.bin", DATA / f"B_n{n}.bin"
            argv = command(lang, cfg) + ["bench", str(a), str(b), str(n), str(warm), str(reps),
                                         str(cfg["min_batch_seconds"]), str(limit)]
            # generous hard limit: every warm-up and rep at the budget, plus batching and startup
            hard = limit * (warm + reps + 2) + 60
            print(f"[{dt.datetime.now():%H:%M:%S}] {lang:<6} n={n:<5} ...", end=" ", flush=True)
            text, code, rss, wall, killed = run_process(argv, hard)

            warmups, batch, heap, status, note, nrep, peak = [], "", "", "error", "", 0, ""
            stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            for line in text.splitlines():
                p = line.split(",")
                if p[0] == "WARMUP":
                    warmups.append(p[2])
                elif p[0] == "META":
                    batch = p[4]
                elif p[0] == "REP":
                    raw.writerow([lang, n, p[1], p[2], p[3], p[4], cfg["dtype"], stamp])
                    nrep += 1
                elif p[0] == "PEAKRSS":
                    peak = int(p[1])
                elif p[0] == "JAVAHEAP":
                    heap = p[1]
                elif p[0] == "STATUS":
                    status = p[1]
                    if status == "exceeded_time_budget":
                        note = f"one warm-up multiplication took {float(p[2]):.1f} s > {limit} s"
            if killed:
                status, note = "killed_hard_timeout", f"process exceeded {hard:.0f} s"
            if status != "ok":
                stopped[lang] = f"{status} at n={n}"
            runs.writerow(base + [status, len(warmups), ";".join(warmups), batch, nrep, peak, rss, est,
                                  f"{wall:.3f}", heap, code, note])
            raw_f.flush()
            runs_f.flush()
            print(f"{status} ({nrep} reps, batch={batch}, peak_rss={(peak or 0) / 1e6:.1f} MB, wall={wall:.1f} s)")

    raw_f.close()
    runs_f.close()
    print(f"wrote {raw_path} and {runs_path}")


if __name__ == "__main__":
    main()
