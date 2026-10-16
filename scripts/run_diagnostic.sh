#!/usr/bin/env bash
# Supplementary diagnostic (reported separately from the main comparison):
# C kernel with leading dimension ld = n (unpadded) vs ld = n + 8 (rows shifted by 64 B).
# Output: results/diag_leading_dim.csv
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build results
gcc -O2 -std=c11 -ffp-contract=off -o build/diag_leading_dim src/c/diag_leading_dim.c
out=results/diag_leading_dim.csv
echo "tag,n,ld,repetition,time_s,checksum" > "$out"
for n in 512 768 1000 1024 1536; do
  for pad in 0 8; do
    echo "n=$n pad=$pad"
    build/diag_leading_dim "data/A_n$n.bin" "data/B_n$n.bin" "$n" "$pad" 3 >> "$out"
  done
done
sed -i 's/^DIAG,/c_ld_diag,/' "$out"
echo "wrote $out"
