// reverse_vector/cuda.cu
// Reverse a vector: out[i] = x[n - 1 - i]  using CUDA.
//
// Parallelisation:
//   One GPU thread per element. Thread i reads x[i] and writes to
//   out[n-1-i]. Because every thread reads from a unique source and
//   writes to a unique destination there are no data hazards.
//
// Grid:  ceil(n / BLOCK_SIZE) blocks   (1-D)
// Block: BLOCK_SIZE (256) threads      (1-D)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Each thread reads position i from x and writes it to the mirror position
// (n-1-i) in out, effectively reversing the array.
__global__ void reverse_vector_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    if (i < n) out[n - 1 - i] = x[i];              // mirror: i ↔ n-1-i
}

// Host wrapper: launches the kernel with a 1-D grid sized for n elements.
void reverse_vector(const float* x, float* out, int n) {
    const int BLOCK_SIZE = 256;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;  // ceil(n / BLOCK_SIZE)
    reverse_vector_kernel<<<grid, BLOCK_SIZE>>>(x, out, n);
}

// Print current device memory usage (called after alloc and after kernel).
static void print_mem(const char* label) {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    size_t used = total_bytes - free_bytes;
    printf("  [%-8s] used=%6.3f MB  free=%6.0f MB\n",
           label, used / 1024.0 / 1024.0, free_bytes / 1024.0 / 1024.0);
}

// Run one end-to-end test for a vector of length n:
//   alloc → H2D → kernel → D2H → verify, timing each phase with CUDA events.
void test(int n) {
    printf("\n--- n=%d ---\n", n);
    size_t bytes = n * sizeof(float);

    // 1. host alloc — fill with sequential values so reversal is easy to verify
    float* hx   = (float*)malloc(bytes);
    float* hout = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) hx[i] = (float)i;

    // 2. device alloc
    float *dx, *dout;
    cudaMalloc(&dx,   bytes);
    cudaMalloc(&dout, bytes);
    print_mem("alloc");

    // 3. H2D — time the copy with CUDA events (GPU-side timestamps)
    cudaEvent_t t0, t1, t2, t3;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventCreate(&t2); cudaEventCreate(&t3);

    cudaEventRecord(t0);
    cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float h2d_ms; cudaEventElapsedTime(&h2d_ms, t0, t1);
    printf("  [H2D]     %.3f ms\n", h2d_ms);

    // 4. kernel launch — record start/stop events around the dispatch
    cudaEventRecord(t2);
    reverse_vector(dx, dout, n);
    cudaEventRecord(t3);
    cudaEventSynchronize(t3);   // wait for GPU to finish before reading the timer
    float kernel_ms; cudaEventElapsedTime(&kernel_ms, t2, t3);
    print_mem("kernel");
    printf("  [kernel]  %.3f ms\n", kernel_ms);

    // 5. D2H
    cudaEvent_t t4, t5;
    cudaEventCreate(&t4); cudaEventCreate(&t5);
    cudaEventRecord(t4);
    cudaMemcpy(hout, dout, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(t5);
    cudaEventSynchronize(t5);
    float d2h_ms; cudaEventElapsedTime(&d2h_ms, t4, t5);
    printf("  [D2H]     %.3f ms\n", d2h_ms);

    // 6. correctness check — out[i] should equal x[n-1-i]
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(hout[i] - hx[n - 1 - i]);
        if (err > max_err) max_err = err;
    }
    printf("  [verify]  max_err=%.2e  total=%.3f ms  %s\n",
           max_err, h2d_ms + kernel_ms + d2h_ms, max_err == 0.0f ? "PASS" : "FAIL");

    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaEventDestroy(t2); cudaEventDestroy(t3);
    cudaEventDestroy(t4); cudaEventDestroy(t5);
    cudaFree(dx); cudaFree(dout);
    free(hx); free(hout);
}

int main() {
    // Warm up: force CUDA driver + runtime initialisation before timing starts.
    float *d; cudaMalloc(&d, sizeof(float)); cudaFree(d);

    // Test across a range of sizes: tiny, near-power-of-2, large.
    int sizes[] = {1, 127, 1024, 10000, 1048576};
    for (int i = 0; i < 5; i++) test(sizes[i]);
    printf("\nAll tests passed.\n");
    return 0;
}
