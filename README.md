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
modal run test.py --kernel vector_add.py

# CUDA
modal run test.py --kernel vector_add.cu

# Different GPU (default: T4)
modal run test.py --kernel vector_add.py --gpu A10G
modal run test.py --kernel vector_add.py --gpu A100-40GB
modal run test.py --kernel vector_add.py --gpu H100
```

## Kernels

| Kernel | Triton | CUDA |
|--------|--------|------|
| Vector Add | [`vector_add.py`](vector_add.py) | [`vector_add.cu`](vector_add.cu) |
