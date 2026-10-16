/*
 * Assignment 1 - Basic dense matrix multiplication (Java implementation)
 *
 * Algorithm : straightforward triple loop, i-j-k order, scalar accumulator.
 *             C[i][j] = sum_{k=0}^{n-1} A[i][k] * B[k][j]   -> O(n^3) work.
 * Numeric   : IEEE-754 binary64 (double). Java forbids FMA contraction, so the
 *             summation order and rounding match the C build (-ffp-contract=off).
 * Storage   : double[][] - an array of n row arrays. Each row is contiguous, but
 *             rows are separate heap objects (one extra indirection per row).
 *
 * Usage:
 *   java MatMul bench    A.bin B.bin n warmup reps min_batch_s max_mult_s
 *   java MatMul validate A.bin B.bin n C_out.bin
 *
 * Timed region (bench mode): ONLY the multiplication kernel (same boundary as C
 * and Python). Warm-up runs let the JIT compile the kernel and are not reported
 * as samples.
 */
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;

public class MatMul {

    static void matmul(double[][] A, double[][] B, double[][] C, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += A[i][k] * B[k][j];
                }
                C[i][j] = s;
            }
        }
    }

    static double[][] load(String path, int n) throws IOException {
        byte[] bytes = Files.readAllBytes(Paths.get(path));
        if (bytes.length != (long) n * n * 8) {
            throw new IOException("unexpected size for " + path);
        }
        DoubleBuffer db = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) db.get(m[i]);
        return m;
    }

    static double checksum(double[][] C) {
        double s = 0.0;
        for (double[] row : C) for (double v : row) s += v;
        return s;
    }

    /** Peak RSS of this JVM process (VmHWM, Linux only); -1 where unavailable. */
    static long peakRssBytes() {
        try {
            for (String line : Files.readAllLines(Paths.get("/proc/self/status"))) {
                if (line.startsWith("VmHWM:")) {
                    return Long.parseLong(line.replaceAll("[^0-9]", "")) * 1024L;
                }
            }
        } catch (IOException | RuntimeException e) {
            // not Linux
        }
        return -1;
    }

    public static void main(String[] args) throws IOException {
        if (args.length == 5 && args[0].equals("validate")) {
            int n = Integer.parseInt(args[3]);
            double[][] A = load(args[1], n), B = load(args[2], n);
            double[][] C = new double[n][n];
            matmul(A, B, C, n);
            ByteBuffer out = ByteBuffer.allocate(n * n * 8).order(ByteOrder.LITTLE_ENDIAN);
            for (double[] row : C) for (double v : row) out.putDouble(v);
            Files.write(Paths.get(args[4]), out.array());
            return;
        }
        if (args.length == 8 && args[0].equals("bench")) {
            int n = Integer.parseInt(args[3]);
            int warmup = Math.max(1, Integer.parseInt(args[4]));
            int reps = Integer.parseInt(args[5]);
            double minBatchS = Double.parseDouble(args[6]);
            double maxMultS = Double.parseDouble(args[7]);

            double[][] A = load(args[1], n), B = load(args[2], n);
            double[][] C = new double[n][n];

            double last = 0.0;
            for (int w = 0; w < warmup; w++) {
                long t0 = System.nanoTime();
                matmul(A, B, C, n);
                last = (System.nanoTime() - t0) * 1e-9;
                System.out.printf("WARMUP,%d,%.9e%n", w, last);
                if (last > maxMultS) {
                    System.out.printf("PEAKRSS,%d%n", peakRssBytes());
                    System.out.printf("STATUS,exceeded_time_budget,%.6f%n", last);
                    return;
                }
            }
            int batch = Math.max(1, (int) Math.ceil(minBatchS / Math.max(last, 1e-9)));
            System.out.printf("META,java,%d,%d,%d%n", n, warmup, batch);

            for (int r = 0; r < reps; r++) {
                long t0 = System.nanoTime();
                for (int b = 0; b < batch; b++) matmul(A, B, C, n);
                double per = (System.nanoTime() - t0) * 1e-9 / batch;
                System.out.printf("REP,%d,%d,%.9e,%.17g%n", r, batch, per, checksum(C));
            }
            Runtime rt = Runtime.getRuntime();
            System.out.printf("JAVAHEAP,%d,%d%n", rt.totalMemory() - rt.freeMemory(), rt.totalMemory());
            System.out.printf("PEAKRSS,%d%n", peakRssBytes());
            System.out.println("STATUS,ok");
            return;
        }
        System.err.println("bad arguments");
        System.exit(1);
    }
}
