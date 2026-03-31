# cuda-triton-kernels

GPU kernels written in both CUDA (`.cu`) and Triton (`.py`), tested on Modal.

## Structure

Each kernel has two implementations side by side:
- `vector_add.py` — Triton kernel
- `vector_add.cu` — CUDA kernel

## Running tests on Modal

```bash
# Triton
modal run test.py --kernel vector_add.py

# CUDA
modal run test.py --kernel vector_add.cu

# Different GPU (T4 default)
modal run test.py --kernel vector_add.py --gpu A100-40GB
modal run test.py --kernel vector_add.py --gpu H100
```

## Kernels

| Kernel | Triton | CUDA |
|--------|--------|------|
| Vector Add | `vector_add.py` | `vector_add.cu` |
