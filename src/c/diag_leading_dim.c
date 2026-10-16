/*
 * SUPPLEMENTARY DIAGNOSTIC - not part of the controlled three-language comparison.
 *
 * Same i-j-k kernel as matmul.c, but each matrix row is stored with a leading
 * dimension `ld` (elements between the starts of consecutive rows). With ld = n
 * this is exactly matmul.c; with ld = n + pad the rows are shifted so that a
 * column walk B[0][j], B[1][j], ... no longer has a stride that is a multiple of
 * 4096 bytes. Used to test the hypothesis that the slow-downs at n = 512, 1024
 * and 1536 come from cache-set conflicts rather than from the cubic work itself.
 *
 * Usage: diag_leading_dim A.bin B.bin n pad reps
 * Output: DIAG,<n>,<ld>,<rep>,<seconds>,<checksum>
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double *load(const char *path, int n, int ld) {
    double *m = calloc((size_t)n * ld, sizeof(double));
    FILE *f = fopen(path, "rb");
    if (!m || !f) { fprintf(stderr, "load failed\n"); exit(2); }
    for (int i = 0; i < n; i++)
        if (fread(m + (size_t)i * ld, sizeof(double), n, f) != (size_t)n) exit(2);
    fclose(f);
    return m;
}

static void matmul(const double *A, const double *B, double *C, int n, int ld) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += A[(size_t)i * ld + k] * B[(size_t)k * ld + j];
            C[(size_t)i * ld + j] = s;
        }
}

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: A.bin B.bin n pad reps\n"); return 1; }
    int n = atoi(argv[3]), ld = n + atoi(argv[4]), reps = atoi(argv[5]);
    double *A = load(argv[1], n, ld), *B = load(argv[2], n, ld);
    double *C = calloc((size_t)n * ld, sizeof(double));
    matmul(A, B, C, n, ld); /* warm-up, not reported */
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        matmul(A, B, C, n, ld);
        double t = now_s() - t0, s = 0.0;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) s += C[(size_t)i * ld + j];
        printf("DIAG,%d,%d,%d,%.9e,%.17g\n", n, ld, r, t, s);
        fflush(stdout);
    }
    return 0;
}
