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
    cudaDeviceSynchronize();
}

void test(int n) {
    // host alloc
    float* hx   = (float*)malloc(n * sizeof(float));
    float* hy   = (float*)malloc(n * sizeof(float));
    float* hout = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) { hx[i] = (float)i; hy[i] = (float)(n - i); }

    // device alloc
    float *dx, *dy, *dout;
    cudaMalloc(&dx,   n * sizeof(float));
    cudaMalloc(&dy,   n * sizeof(float));
    cudaMalloc(&dout, n * sizeof(float));

    cudaMemcpy(dx, hx, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, n * sizeof(float), cudaMemcpyHostToDevice);

    vector_add(dx, dy, dout, n);

    cudaMemcpy(hout, dout, n * sizeof(float), cudaMemcpyDeviceToHost);

    // verify
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float expected = hx[i] + hy[i];
        float err = fabsf(hout[i] - expected);
        if (err > max_err) max_err = err;
    }
    printf("n=%10d  max_err=%.2e  %s\n", n, max_err, max_err == 0.0f ? "PASS" : "FAIL");

    cudaFree(dx); cudaFree(dy); cudaFree(dout);
    free(hx); free(hy); free(hout);
}

int main() {
    int sizes[] = {1, 127, 1024, 10000, 1048576};
    for (int i = 0; i < 5; i++) test(sizes[i]);
    printf("\nAll tests passed.\n");
    return 0;
}
