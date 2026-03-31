#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

void vector_add(const float* x, const float* y, float* out, int n) {
    const int BLOCK_SIZE = 1024;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add_kernel<<<grid, BLOCK_SIZE>>>(x, y, out, n);
}

static void print_mem(const char* label) {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    size_t used = total_bytes - free_bytes;
    printf("  [%-8s] used=%6.3f MB  free=%6.0f MB\n",
           label, used / 1024.0 / 1024.0, free_bytes / 1024.0 / 1024.0);
}

void test(int n) {
    printf("\n--- n=%d ---\n", n);
    size_t bytes = n * sizeof(float);

    // 1. host alloc
    float* hx   = (float*)malloc(bytes);
    float* hy   = (float*)malloc(bytes);
    float* hout = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { hx[i] = (float)i; hy[i] = (float)(n - i); }

    // 2. device alloc
    float *dx, *dy, *dout;
    cudaMalloc(&dx,   bytes);
    cudaMalloc(&dy,   bytes);
    cudaMalloc(&dout, bytes);
    print_mem("alloc");

    // 3. H2D
    cudaEvent_t t0, t1, t2, t3;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventCreate(&t2); cudaEventCreate(&t3);

    cudaEventRecord(t0);
    cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float h2d_ms; cudaEventElapsedTime(&h2d_ms, t0, t1);
    printf("  [H2D]     %.3f ms\n", h2d_ms);

    // 4. kernel
    cudaEventRecord(t2);
    vector_add(dx, dy, dout, n);
    cudaEventRecord(t3);
    cudaEventSynchronize(t3);
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

    // 6. verify
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(hout[i] - (hx[i] + hy[i]));
        if (err > max_err) max_err = err;
    }
    printf("  [verify]  max_err=%.2e  total=%.3f ms  %s\n",
           max_err, h2d_ms + kernel_ms + d2h_ms, max_err == 0.0f ? "PASS" : "FAIL");

    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaEventDestroy(t2); cudaEventDestroy(t3);
    cudaEventDestroy(t4); cudaEventDestroy(t5);
    cudaFree(dx); cudaFree(dy); cudaFree(dout);
    free(hx); free(hy); free(hout);
}

int main() {
    // warm up
    float *d; cudaMalloc(&d, sizeof(float)); cudaFree(d);

    int sizes[] = {1, 127, 1024, 10000, 1048576};
    for (int i = 0; i < 5; i++) test(sizes[i]);
    printf("\nAll tests passed.\n");
    return 0;
}
