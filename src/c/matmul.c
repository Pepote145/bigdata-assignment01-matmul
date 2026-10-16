/*
 * Assignment 1 - Basic dense matrix multiplication (C implementation)
 *
 * Algorithm : straightforward triple loop, i-j-k order, scalar accumulator.
 *             C[i][j] = sum_{k=0}^{n-1} A[i][k] * B[k][j]   -> O(n^3) work.
 * Numeric   : IEEE-754 binary64 (double).
 * Storage   : one contiguous row-major block per matrix, element (i,j) at i*n + j.
 *
 * Usage:
 *   matmul bench    A.bin B.bin n warmup reps min_batch_s max_mult_s
 *   matmul validate A.bin B.bin n C_out.bin
 *
 * Input files: n*n little-endian float64 values, row-major (see scripts/generate_inputs.py).
 *
 * Timed region (bench mode): ONLY the multiplication kernel. Input loading and
 * allocation of A, B and C happen before timing. The kernel overwrites every
 * C[i][j], so C needs no re-initialisation between runs.
 *
 * Output (stdout, CSV-like lines parsed by scripts/run_benchmarks.py):
 *   META,c,<n>,<warmup>,<batch>
 *   WARMUP,<idx>,<seconds>
 *   REP,<rep>,<batch>,<seconds_per_multiplication>,<checksum>
 *   PEAKRSS,<bytes>
 *   STATUS,ok | STATUS,exceeded_time_budget,<seconds>
 */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

/* Peak resident set size of THIS process in bytes.
 * Linux: VmHWM from /proc/self/status (per address space, so it is not
 * inflated by the parent process that spawned us; getrusage's ru_maxrss is
 * carried across execve on Linux and would include the harness).
 * Other POSIX (macOS): getrusage, where ru_maxrss is already in bytes. */
static long long peak_rss_bytes(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        long long kb = -1;
        while (fgets(line, sizeof line, f))
            if (sscanf(line, "VmHWM: %lld kB", &kb) == 1) break;
        fclose(f);
        if (kb >= 0) return kb * 1024;
    }
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return (long long)r.ru_maxrss;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void matmul(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            }
            C[(size_t)i * n + j] = s;
        }
    }
}

static double *load_matrix(const char *path, int n) {
    size_t count = (size_t)n * (size_t)n;
    double *m = malloc(count * sizeof(double));
    if (!m) { fprintf(stderr, "allocation failed\n"); exit(2); }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    if (fread(m, sizeof(double), count, f) != count) {
        fprintf(stderr, "short read on %s\n", path); exit(2);
    }
    fclose(f);
    return m;
}

static double checksum(const double *C, int n) {
    double s = 0.0;
    for (size_t i = 0; i < (size_t)n * n; i++) s += C[i];
    return s;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: see source header\n"); return 1; }

    if (strcmp(argv[1], "validate") == 0 && argc == 6) {
        int n = atoi(argv[4]);
        double *A = load_matrix(argv[2], n), *B = load_matrix(argv[3], n);
        double *C = malloc((size_t)n * n * sizeof(double));
        matmul(A, B, C, n);
        FILE *f = fopen(argv[5], "wb");
        fwrite(C, sizeof(double), (size_t)n * n, f);
        fclose(f);
        free(A); free(B); free(C);
        return 0;
    }

    if (strcmp(argv[1], "bench") == 0 && argc == 9) {
        int n = atoi(argv[4]);
        int warmup = atoi(argv[5]);
        int reps = atoi(argv[6]);
        double min_batch_s = atof(argv[7]);
        double max_mult_s = atof(argv[8]);
        if (warmup < 1) warmup = 1; /* one warm-up is needed for batch calibration */

        double *A = load_matrix(argv[2], n), *B = load_matrix(argv[3], n);
        double *C = malloc((size_t)n * n * sizeof(double));
        if (!C) { fprintf(stderr, "allocation failed\n"); return 2; }

        double last = 0.0;
        for (int w = 0; w < warmup; w++) {
            double t0 = now_s();
            matmul(A, B, C, n);
            last = now_s() - t0;
            printf("WARMUP,%d,%.9e\n", w, last);
            fflush(stdout);
            if (last > max_mult_s) {
                printf("PEAKRSS,%lld\n", peak_rss_bytes());
                printf("STATUS,exceeded_time_budget,%.6f\n", last);
                return 0;
            }
        }
        int batch = (int)ceil(min_batch_s / (last > 0 ? last : 1e-9));
        if (batch < 1) batch = 1;
        printf("META,c,%d,%d,%d\n", n, warmup, batch);

        for (int r = 0; r < reps; r++) {
            double t0 = now_s();
            for (int b = 0; b < batch; b++) matmul(A, B, C, n);
            double per = (now_s() - t0) / batch;
            /* the checksum consumes the result so the kernel cannot be elided */
            printf("REP,%d,%d,%.9e,%.17g\n", r, batch, per, checksum(C, n));
            fflush(stdout);
        }
        printf("PEAKRSS,%lld\n", peak_rss_bytes());
        printf("STATUS,ok\n");
        free(A); free(B); free(C);
        return 0;
    }

    fprintf(stderr, "bad arguments\n");
    return 1;
}
