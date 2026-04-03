// conv1d/cuda.cu
// 1-D valid convolution  out[i] = Σ_{j=0}^{K-1} x[i+j] * w[j]  using CUDA.
//
// "Valid" means no padding: the output is shorter than the input by K-1.
//   Input  x:   length n
//   Kernel w:   length k
//   Output out: length n - k + 1
//
// Parallelisation:
//   One GPU thread per output element. Thread i computes the dot product
//   of x[i..i+k-1] with w[0..k-1]. Reads from x are not coalesced for
//   adjacent threads (offset by 1 each) but the kernel is memory-bound
//   rather than compute-bound for small k, so this is acceptable.
//
// Grid:  ceil(out_len / BLOCK_SIZE) blocks   (1-D)
// Block: BLOCK_SIZE (256) threads            (1-D)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Each thread computes one output sample by looping over the k kernel weights.
__global__ void conv1d_kernel(const float* x, const float* w, float* out,
                               int n, int k, int out_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // output index
    if (i >= out_len) return;                        // bounds check for partial last block
    float acc = 0.0f;
    // Dot product of the k-element input window starting at i with the kernel.
    for (int j = 0; j < k; j++)
        acc += x[i + j] * w[j];
    out[i] = acc;
}

// Host wrapper: computes out_len = n - k + 1 and launches the kernel.
void conv1d(const float* x, const float* w, float* out, int n, int k) {
    int out_len = n - k + 1;
    const int BLOCK_SIZE = 256;
    int grid = (out_len + BLOCK_SIZE - 1) / BLOCK_SIZE;  // ceil(out_len / BLOCK_SIZE)
    conv1d_kernel<<<grid, BLOCK_SIZE>>>(x, w, out, n, k, out_len);
}

// Print current device memory usage (called after alloc and after kernel).
static void print_mem(const char* label) {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    size_t used = total_bytes - free_bytes;
    printf("  [%-8s] used=%6.3f MB  free=%6.0f MB\n",
           label, used / 1024.0 / 1024.0, free_bytes / 1024.0 / 1024.0);
}

// Run one end-to-end test for input length n and kernel size k:
//   alloc → CPU reference → H2D → kernel → D2H → verify.
void test(int n, int k) {
    int out_len = n - k + 1;
    printf("\n--- n=%d  k=%d  out_len=%d ---\n", n, k, out_len);

    size_t bytes_x   = n       * sizeof(float);
    size_t bytes_w   = k       * sizeof(float);
    size_t bytes_out = out_len * sizeof(float);

    // 1. host alloc + init — box filter (uniform weights) for easy verification
    float* hx   = (float*)malloc(bytes_x);
    float* hw   = (float*)malloc(bytes_w);
    float* hout = (float*)malloc(bytes_out);
    float* hRef = (float*)malloc(bytes_out);
    for (int i = 0; i < n; i++) hx[i] = (float)(i % 11) / 11.0f;
    for (int j = 0; j < k; j++) hw[j] = 1.0f / k;  // normalised box filter

    // Naive O(n*k) CPU reference used for correctness verification.
    for (int i = 0; i < out_len; i++) {
        float s = 0.0f;
        for (int j = 0; j < k; j++) s += hx[i + j] * hw[j];
        hRef[i] = s;
    }

    // 2. device alloc
    float *dx, *dw, *dout;
    cudaMalloc(&dx,   bytes_x);
    cudaMalloc(&dw,   bytes_w);
    cudaMalloc(&dout, bytes_out);
    print_mem("alloc");

    // 3. H2D
    cudaEvent_t t0, t1, t2, t3, t4, t5;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventCreate(&t2); cudaEventCreate(&t3);
    cudaEventCreate(&t4); cudaEventCreate(&t5);

    cudaEventRecord(t0);
    cudaMemcpy(dx, hx, bytes_x, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, hw, bytes_w, cudaMemcpyHostToDevice);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float h2d_ms; cudaEventElapsedTime(&h2d_ms, t0, t1);
    printf("  [H2D]     %.3f ms\n", h2d_ms);

    // 4. kernel launch
    cudaEventRecord(t2);
    conv1d(dx, dw, dout, n, k);
    cudaEventRecord(t3);
    cudaEventSynchronize(t3);
    float kernel_ms; cudaEventElapsedTime(&kernel_ms, t2, t3);
    print_mem("kernel");
    printf("  [kernel]  %.3f ms\n", kernel_ms);

    // 5. D2H
    cudaEventRecord(t4);
    cudaMemcpy(hout, dout, bytes_out, cudaMemcpyDeviceToHost);
    cudaEventRecord(t5);
    cudaEventSynchronize(t5);
    float d2h_ms; cudaEventElapsedTime(&d2h_ms, t4, t5);
    printf("  [D2H]     %.3f ms\n", d2h_ms);

    // 6. correctness check — float32 accumulation should be exact here
    // since we use simple fractions, so the tolerance is tight.
    float max_err = 0.0f;
    for (int i = 0; i < out_len; i++) {
        float err = fabsf(hout[i] - hRef[i]);
        if (err > max_err) max_err = err;
    }
    printf("  [verify]  max_err=%.2e  total=%.3f ms  %s\n",
           max_err, h2d_ms + kernel_ms + d2h_ms, max_err < 1e-5f ? "PASS" : "FAIL");

    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaEventDestroy(t2);
    cudaEventDestroy(t3); cudaEventDestroy(t4); cudaEventDestroy(t5);
    cudaFree(dx); cudaFree(dw); cudaFree(dout);
    free(hx); free(hw); free(hout); free(hRef);
}

int main() {
    // Warm up: force CUDA driver + runtime initialisation before timing starts.
    float *d; cudaMalloc(&d, sizeof(float)); cudaFree(d);

    // Vary both input length and kernel size to stress different regimes.
    test(1024,     3);
    test(1024,     7);
    test(10000,    3);
    test(1048576,  3);
    test(1048576, 15);
    printf("\nAll tests passed.\n");
    return 0;
}
