# cuda-triton-kernels

GPU kernels written in both CUDA (`.cu`) and Triton (`.py`), tested on Modal.

## Setup

```bash
pip install modal
modal setup  # authenticate with Modal
```

## Running tests on Modal

Pass any kernel file to [`test.py`](test.py) — it detects `.cu` vs `.py` automatically.

```bash
# Triton
modal run test.py --kernel vector_add/triton_kernel.py

# CUDA
modal run test.py --kernel vector_add/cuda.cu

# Different GPU (default: T4)
modal run test.py --kernel vector_add/triton_kernel.py --gpu A10G
modal run test.py --kernel vector_add/triton_kernel.py --gpu A100-40GB
modal run test.py --kernel vector_add/triton_kernel.py --gpu H100
```

## Kernels

| Kernel | Triton | CUDA |
|--------|--------|------|
| Vector Add | [`vector_add/triton_kernel.py`](vector_add/triton_kernel.py) | [`vector_add/cuda.cu`](vector_add/cuda.cu) |
| Reverse Vector | [`reverse_vector/triton_kernel.py`](reverse_vector/triton_kernel.py) | [`reverse_vector/cuda.cu`](reverse_vector/cuda.cu) |
| Matrix Multiply | [`matmul/triton_kernel.py`](matmul/triton_kernel.py) | [`matmul/cuda.cu`](matmul/cuda.cu) |
| 1-D Convolution | [`conv1d/triton_kernel.py`](conv1d/triton_kernel.py) | [`conv1d/cuda.cu`](conv1d/cuda.cu) |

### Notes

- **vector_add** — elementwise `out[i] = x[i] + y[i]`; the simplest possible kernel, good starting point.
- **reverse_vector** — `out[i] = x[n-1-i]`; each thread writes to a mirrored index, no data hazards.
- **matmul** — tiled GEMM; CUDA uses `__shared__` memory tiles (32×32), Triton uses `tl.dot` with explicit strides.
- **conv1d** — 1-D valid convolution; one thread/program per output element, loops over the K kernel weights.
