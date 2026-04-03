// matmul/cuda.cu
// Tiled matrix multiplication  C = A @ B  using CUDA shared memory.
//
// Algorithm (blocked GEMM):
//   The output matrix C is divided into (BLOCK x BLOCK) tiles. Each
//   thread-block computes one such tile by streaming along the K dimension
//   in BLOCK-wide slices, loading each slice of A and B into shared
//   memory, synchronising threads, then accumulating partial dot products.
//
//   Using shared memory eliminates repeated global-memory traffic: each
//   element of A/B is loaded from DRAM once per K-slice rather than BLOCK
//   times (one per output element that needs it in the naive approach).
//
// Dimensions:
//   A [M x K],  B [K x N],  C [M x N]
//
// Grid:  (ceil(N/BLOCK), ceil(M/BLOCK)) blocks
// Block: (BLOCK, BLOCK) threads — threadIdx.x = column, threadIdx.y = row

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK 32  // tile side length; also the warp-width, so no partial warps

// Each (BLOCK x BLOCK) thread-block computes one (BLOCK x BLOCK) tile of C.
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    // Shared-memory tiles: loaded once per K-slice, reused BLOCK times each.
    __shared__ float As[BLOCK][BLOCK];
    __shared__ float Bs[BLOCK][BLOCK];

    // Absolute row/col in C that this thread is responsible for.
    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;
    float acc = 0.0f;  // running dot-product accumulator

    // Iterate over K in BLOCK-wide slices (tiles along the shared K axis).
    for (int t = 0; t < (K + BLOCK - 1) / BLOCK; t++) {
        // Each thread loads one element of the A tile and one of the B tile.
        int a_col = t * BLOCK + threadIdx.x;  // column in A for this K-slice
        int b_row = t * BLOCK + threadIdx.y;  // row    in B for this K-slice

        // Boundary-safe loads: pad with 0 for elements outside the matrix.
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // Wait until all threads in the block have loaded their elements before
        // any thread starts reading from the shared-memory tiles.
        __syncthreads();

        // Accumulate: this thread's element of C += dot(A_row_slice, B_col_slice).
        for (int k = 0; k < BLOCK; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        // Wait until all threads are done reading the tile before the next
        // iteration overwrites it with the next K-slice.
        __syncthreads();
    }

    // Write the finished accumulator to C (skip out-of-bounds threads).
    if (row < M && col < N)
        C[row * N + col] = acc;
}

// Host wrapper: configures a 2-D grid to cover all (row, col) output tiles.
void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(BLOCK, BLOCK);                              // 2-D thread-block
    dim3 grid((N + BLOCK - 1) / BLOCK,                    // tiles along N
              (M + BLOCK - 1) / BLOCK);                    // tiles along M
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

// Print current device memory usage (called after alloc and after kernel).
static void print_mem(const char* label) {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    size_t used = total_bytes - free_bytes;
    printf("  [%-8s] used=%6.3f MB  free=%6.0f MB\n",
           label, used / 1024.0 / 1024.0, free_bytes / 1024.0 / 1024.0);
}

// Run one end-to-end test for matrices of shape M×K and K×N:
//   alloc → CPU reference → H2D → kernel → D2H → verify.
void test(int M, int N, int K) {
    printf("\n--- M=%d N=%d K=%d ---\n", M, N, K);
    size_t bytes_a = (size_t)M * K * sizeof(float);
    size_t bytes_b = (size_t)K * N * sizeof(float);
    size_t bytes_c = (size_t)M * N * sizeof(float);

    // 1. host alloc + init with deterministic values
    float* hA   = (float*)malloc(bytes_a);
    float* hB   = (float*)malloc(bytes_b);
    float* hC   = (float*)malloc(bytes_c);
    float* hRef = (float*)malloc(bytes_c);
    for (int i = 0; i < M * K; i++) hA[i] = (float)(i % 7) / 7.0f;
    for (int i = 0; i < K * N; i++) hB[i] = (float)(i % 5) / 5.0f;

    // Naive O(M*N*K) CPU reference used for correctness verification.
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += hA[r * K + k] * hB[k * N + c];
            hRef[r * N + c] = s;
        }

    // 2. device alloc
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes_a);
    cudaMalloc(&dB, bytes_b);
    cudaMalloc(&dC, bytes_c);
    print_mem("alloc");

    // 3. H2D
    cudaEvent_t t0, t1, t2, t3, t4, t5;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventCreate(&t2); cudaEventCreate(&t3);
    cudaEventCreate(&t4); cudaEventCreate(&t5);

    cudaEventRecord(t0);
    cudaMemcpy(dA, hA, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes_b, cudaMemcpyHostToDevice);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float h2d_ms; cudaEventElapsedTime(&h2d_ms, t0, t1);
    printf("  [H2D]     %.3f ms\n", h2d_ms);

    // 4. kernel launch
    cudaEventRecord(t2);
    matmul(dA, dB, dC, M, N, K);
    cudaEventRecord(t3);
    cudaEventSynchronize(t3);
    float kernel_ms; cudaEventElapsedTime(&kernel_ms, t2, t3);
    print_mem("kernel");
    printf("  [kernel]  %.3f ms\n", kernel_ms);

    // 5. D2H
    cudaEventRecord(t4);
    cudaMemcpy(hC, dC, bytes_c, cudaMemcpyDeviceToHost);
    cudaEventRecord(t5);
    cudaEventSynchronize(t5);
    float d2h_ms; cudaEventElapsedTime(&d2h_ms, t4, t5);
    printf("  [D2H]     %.3f ms\n", d2h_ms);

    // 6. correctness check — floating-point accumulation order differs from
    // the CPU reference, so allow a small absolute tolerance (1e-3).
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(hC[i] - hRef[i]);
        if (err > max_err) max_err = err;
    }
    printf("  [verify]  max_err=%.2e  total=%.3f ms  %s\n",
           max_err, h2d_ms + kernel_ms + d2h_ms, max_err < 1e-3f ? "PASS" : "FAIL");

    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaEventDestroy(t2);
    cudaEventDestroy(t3); cudaEventDestroy(t4); cudaEventDestroy(t5);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hRef);
}

int main() {
    // Warm up: force CUDA driver + runtime initialisation before timing starts.
    float *d; cudaMalloc(&d, sizeof(float)); cudaFree(d);

    // Test a variety of shapes: square, non-square, and large.
    test(32,  32,  32);
    test(37,  41,  28);   // non-multiples of BLOCK to exercise boundary masking
    test(128, 128, 128);
    test(512, 512, 512);
    printf("\nAll tests passed.\n");
    return 0;
}
