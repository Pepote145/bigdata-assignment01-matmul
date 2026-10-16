"""
Assignment 1 - Basic dense matrix multiplication (Python implementation)

Algorithm : straightforward triple loop, i-j-k order, scalar accumulator.
            C[i][j] = sum_{k=0}^{n-1} A[i][k] * B[k][j]   -> O(n^3) work.
Numeric   : Python float = IEEE-754 binary64 (same as C double / Java double).
Storage   : list of n row lists; every element is a pointer to a boxed float
            object (8-byte pointer + 24-byte object on 64-bit CPython).
Libraries : standard library only. NumPy is NOT used here (it is used only as a
            correctness reference in scripts/validate.py).

Usage:
  python3 matmul.py bench    A.bin B.bin n warmup reps min_batch_s max_mult_s
  python3 matmul.py validate A.bin B.bin n C_out.bin

Timed region (bench mode): ONLY the multiplication kernel (same boundary as C and Java).
"""
import math
import sys
import time
from array import array


def matmul(A, B, C, n):
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s


def load(path, n):
    if sys.byteorder != "little":
        raise SystemExit("input files are little-endian; big-endian host not supported")
    flat = array("d")
    with open(path, "rb") as f:
        flat.fromfile(f, n * n)
    return [flat[i * n:(i + 1) * n].tolist() for i in range(n)]


def peak_rss_bytes():
    """Peak RSS of this process: VmHWM on Linux, getrusage elsewhere (bytes on macOS)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def checksum(C):
    s = 0.0
    for row in C:
        for v in row:
            s += v
    return s


def main(argv):
    if len(argv) == 6 and argv[1] == "validate":
        n = int(argv[4])
        A, B = load(argv[2], n), load(argv[3], n)
        C = [[0.0] * n for _ in range(n)]
        matmul(A, B, C, n)
        out = array("d", (v for row in C for v in row))
        with open(argv[5], "wb") as f:
            out.tofile(f)
        return 0

    if len(argv) == 9 and argv[1] == "bench":
        n = int(argv[4])
        warmup = max(1, int(argv[5]))
        reps = int(argv[6])
        min_batch_s = float(argv[7])
        max_mult_s = float(argv[8])

        A, B = load(argv[2], n), load(argv[3], n)
        C = [[0.0] * n for _ in range(n)]

        last = 0.0
        for w in range(warmup):
            t0 = time.perf_counter_ns()
            matmul(A, B, C, n)
            last = (time.perf_counter_ns() - t0) * 1e-9
            print(f"WARMUP,{w},{last:.9e}", flush=True)
            if last > max_mult_s:
                print(f"PEAKRSS,{peak_rss_bytes()}", flush=True)
                print(f"STATUS,exceeded_time_budget,{last:.6f}", flush=True)
                return 0
        batch = max(1, math.ceil(min_batch_s / max(last, 1e-9)))
        print(f"META,python,{n},{warmup},{batch}", flush=True)

        for r in range(reps):
            t0 = time.perf_counter_ns()
            for _ in range(batch):
                matmul(A, B, C, n)
            per = (time.perf_counter_ns() - t0) * 1e-9 / batch
            print(f"REP,{r},{batch},{per:.9e},{checksum(C)!r}", flush=True)
        print(f"PEAKRSS,{peak_rss_bytes()}", flush=True)
        print("STATUS,ok", flush=True)
        return 0

    print("bad arguments", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
