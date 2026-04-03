# matmul/triton_kernel.py
# Tiled matrix multiplication  C = A @ B  using Triton.
#
# Each kernel instance handles one (BLOCK x BLOCK) output tile of C.
# The grid is 2-D: axis-0 steps over row-tiles, axis-1 over col-tiles.
#
#   A  [M x K]          B  [K x N]          C  [M x N]
#   ┌──────────┐         ┌────┬────┐         ┌────┬────┐
#   │          │         │ B0 │ B1 │         │ C0 │ C1 │
#   ├──────────┤    @    ├────┼────┤    =    ├────┼────┤
#   │          │         │ B2 │ B3 │         │ C2 │ C3 │
#   └──────────┘         └────┴────┘         └────┴────┘
#
# This kernel computes one Ci tile by streaming K in BLOCK-wide slices
# and accumulating partial dot products into `acc`.

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,  # how many elements to skip to move one row in A
    stride_ak,  # how many elements to skip to move one col in A
    stride_bk,  # how many elements to skip to move one row in B
    stride_bn,  # how many elements to skip to move one col in B
    stride_cm,
    stride_cn,
    BLOCK: tl.constexpr,
):
    # Which output tile are we?
    #   pid_m selects the row-block, pid_n the col-block.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Row/col indices this tile is responsible for.
    #   rm: [BLOCK] — absolute row indices into A and C
    #   rn: [BLOCK] — absolute col indices into B and C
    #   rk: [BLOCK] — relative offsets within one K-slice (reset each iteration)
    rm = pid_m * BLOCK + tl.arange(0, BLOCK)
    rn = pid_n * BLOCK + tl.arange(0, BLOCK)
    rk = tl.arange(0, BLOCK)

    # Accumulator for this output tile, zero-initialised.
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    # Stream along the K dimension in BLOCK-wide slices.
    #
    #   iteration 0        iteration 1        ...
    #   A tile: rows rm, cols  0..BLOCK-1
    #   B tile: rows  0..BLOCK-1, cols rn
    #                          A tile: rows rm, cols BLOCK..2*BLOCK-1
    #                          B tile: rows BLOCK..2*BLOCK-1, cols rn
    for kk in range(tl.cdiv(K, BLOCK)):
        k = kk * BLOCK + rk  # absolute K indices for this slice

        # Load an (BLOCK x BLOCK) slice from A: shape [BLOCK, BLOCK]
        #   rm[:, None]  → column vector of row indices   [BLOCK, 1]
        #   k[None, :]   → row vector of col indices      [1, BLOCK]
        #   mask zeros out out-of-bounds rows (rm >= M) or cols (k >= K)
        a = tl.load(
            a_ptr + rm[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )

        # Load a matching (BLOCK x BLOCK) slice from B: shape [BLOCK, BLOCK]
        #   k[:, None]   → column vector of row indices   [BLOCK, 1]
        #   rn[None, :]  → row vector of col indices      [1, BLOCK]
        b = tl.load(
            b_ptr + k[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(k[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )

        # Accumulate: acc += A_slice @ B_slice
        # allow_tf32=False forces full float32 precision (no TF32 rounding).
        acc += tl.dot(a, b, allow_tf32=False)

    # Write the finished (BLOCK x BLOCK) tile to C.
    tl.store(
        c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0]
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb
    # Cast to float32 for the kernel; restore original dtype at the end.
    a_f = a.float()
    b_f = b.float()
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    BLOCK = 32
    grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))  # 2-D program grid
    matmul_kernel[grid](
        a_f, b_f, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK=BLOCK,
    )
    return c.to(a.dtype)


def test_matmul():
    torch.manual_seed(0)
    # Include non-multiples of BLOCK to exercise boundary masking.
    sizes = [(32, 32, 32), (37, 41, 28), (128, 128, 128), (512, 512, 512)]

    # Warm up every size so the Triton JIT compile does not inflate timings.
    for M, N, K in sizes:
        _a = torch.ones(M, K, device="cuda")
        _b = torch.ones(K, N, device="cuda")
        matmul(_a, _b)
    torch.cuda.synchronize()

    for M, N, K in sizes:
        print(f"\n--- M={M} N={N} K={K} ---")
        torch.cuda.reset_peak_memory_stats()

        # 1. allocate inputs on the GPU
        mem0 = torch.cuda.memory_allocated()
        a = torch.randn(M, K, device="cuda")
        b = torch.randn(K, N, device="cuda")
        mem1 = torch.cuda.memory_allocated()
        print(f"  [alloc]   {mem1/1024**2:.3f} MB  (+{(mem1-mem0)/1024**2:.3f} MB)")

        # 2. time the kernel with CUDA events (GPU-side, sub-millisecond precision)
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        out = matmul(a, b)
        end.record()
        torch.cuda.synchronize()  # wait for GPU before reading the timer
        elapsed_ms = start.elapsed_time(end)
        mem2 = torch.cuda.memory_allocated()
        print(f"  [kernel]  {elapsed_ms:.3f} ms   mem={mem2/1024**2:.3f} MB  (+{(mem2-mem1)/1024**2:.3f} MB)")

        # 3. correctness check against PyTorch reference (cuBLAS)
        expected = a @ b
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
        max_err = (out - expected).abs().max().item()
        peak = torch.cuda.max_memory_allocated()
        print(f"  [verify]  max_err={max_err:.2e}  peak={peak/1024**2:.3f} MB  PASS")

    print("\nAll tests passed.")


if __name__ == "__main__":
    test_matmul()
