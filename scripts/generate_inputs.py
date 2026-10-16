"""Generate the shared input matrices used by ALL three implementations.

Every language reads the same binary files, so the logical inputs are identical
across methods (cross-language random generators would otherwise differ).

Format : n*n IEEE-754 float64 values, little-endian, row-major, no header.
Values : uniform on [value_low, value_high) from NumPy's PCG64 generator,
         seeded with SeedSequence([seed, n]) so each size is independently
         reproducible. A is drawn first, then B, from the same stream.
Output : data/A_n{n}.bin, data/B_n{n}.bin and data/manifest.json (SHA-256 hashes).
"""
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(n, cfg, out_dir):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg["seed"], n])))
    lo, hi = cfg["value_low"], cfg["value_high"]
    A = rng.uniform(lo, hi, size=(n, n)).astype("<f8")
    B = rng.uniform(lo, hi, size=(n, n)).astype("<f8")
    pa, pb = out_dir / f"A_n{n}.bin", out_dir / f"B_n{n}.bin"
    A.tofile(pa)
    B.tofile(pb)
    return {"n": n, "A": pa.name, "B": pb.name, "sha256_A": sha256(pa), "sha256_B": sha256(pb)}


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    sizes = [int(s) for s in sys.argv[1:]] or cfg["sizes"]
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    entries = [generate(n, cfg, out_dir) for n in sizes]
    manifest = {
        "generator": "numpy.random.PCG64, SeedSequence([seed, n]); A then B via uniform()",
        "numpy_version": np.__version__,
        "seed": cfg["seed"],
        "value_range": [cfg["value_low"], cfg["value_high"]],
        "dtype": "float64 little-endian, row-major",
        "files": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(entries)} input pairs to {out_dir}")


if __name__ == "__main__":
    main()
