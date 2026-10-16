"""Correctness tests for the three implementations (run BEFORE benchmarking).

Cases
  1. Hand-computed 2x2 product              -> exact equality
  2. A x I and I x A (n = 17)                -> exact equality with A
  3. Zero matrix (n = 33)                    -> exact zeros
  4. Random matrices, several n              -> within a rounding-error bound of NumPy
  5. Cross-language agreement on case 4      -> bit-for-bit identical

Tolerance for case 4. Each C[i][j] is a length-n dot product. For recursive
summation the classical forward error bound (Higham, Accuracy and Stability of
Numerical Algorithms, 2nd ed., Sec. 3.1) is
    |fl(C) - C| <= gamma_n * (|A| |B|),   gamma_n = n*u / (1 - n*u),  u = 2^-53.
NumPy's BLAS reference uses a different summation order with its own error of the
same form, so we accept |C_impl - C_numpy| <= 2 * gamma_n * (|A||B|) elementwise.
This tolerance scales with n and with the magnitudes involved, unlike a fixed rtol.

Bitwise agreement (case 5) is expected because all three implementations use the
same i-j-k order, binary64 and no fused multiply-add (C is built with
-ffp-contract=off; Java and CPython never contract).
"""
import csv
import subprocess
import tempfile

import numpy as np

from common import RESULTS, command, load_config

U = 2.0 ** -53


def run_impl(lang, cfg, A, B, workdir):
    n = A.shape[0]
    pa, pb, pc = workdir / "A.bin", workdir / "B.bin", workdir / f"C_{lang}.bin"
    A.astype("<f8").tofile(pa)
    B.astype("<f8").tofile(pb)
    subprocess.run(command(lang, cfg) + ["validate", str(pa), str(pb), str(n), str(pc)],
                   check=True, capture_output=True)
    return np.fromfile(pc, dtype="<f8").reshape(n, n)


def main():
    cfg = load_config()
    langs = cfg["languages"]
    rng = np.random.default_rng(12345)
    rows = []

    def record(case, lang, n, passed, max_abs_err, detail):
        rows.append({"case": case, "language": lang, "n": n, "passed": passed,
                     "max_abs_error": f"{max_abs_err:.3e}", "detail": detail})
        print(f"{'PASS' if passed else 'FAIL'}  {case:<22} {lang:<7} n={n:<4} "
              f"max|err|={max_abs_err:.3e}  {detail}")

    import pathlib
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        # 1. hand-computed
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        for lang in langs:
            C = run_impl(lang, cfg, A, B, tmp)
            record("hand_2x2", lang, 2, np.array_equal(C, expected),
                   float(np.max(np.abs(C - expected))), "exact")

        # 2. identity
        n = 17
        A = rng.uniform(-1, 1, (n, n))
        I = np.eye(n)
        for lang in langs:
            C1 = run_impl(lang, cfg, A, I, tmp)
            C2 = run_impl(lang, cfg, I, A, tmp)
            ok = np.array_equal(C1, A) and np.array_equal(C2, A)
            err = max(np.max(np.abs(C1 - A)), np.max(np.abs(C2 - A)))
            record("identity_AI_IA", lang, n, ok, float(err), "exact")

        # 3. zero
        n = 33
        A = rng.uniform(-1, 1, (n, n))
        Z = np.zeros((n, n))
        for lang in langs:
            C = run_impl(lang, cfg, A, Z, tmp)
            record("zero", lang, n, not C.any(), float(np.max(np.abs(C))), "exact")

        # 4 + 5. random vs NumPy, and cross-language bitwise agreement
        for n in (1, 3, 64, 100, 129):
            A = rng.uniform(-1, 1, (n, n))
            B = rng.uniform(-1, 1, (n, n))
            ref = A @ B
            bound = 2 * (n * U / (1 - n * U)) * (np.abs(A) @ np.abs(B))
            results = {}
            for lang in langs:
                C = run_impl(lang, cfg, A, B, tmp)
                results[lang] = C
                err = np.abs(C - ref)
                record("random_vs_numpy", lang, n, bool(np.all(err <= bound)),
                       float(err.max()), f"bound max={bound.max():.3e}")
            base = results[langs[0]]
            for lang in langs[1:]:
                same = np.array_equal(base.view(np.uint64), results[lang].view(np.uint64))
                record("bitwise_vs_" + langs[0], lang, n, same,
                       float(np.max(np.abs(results[lang] - base))), "identical bits")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    failed = [r for r in rows if not r["passed"]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} checks passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
