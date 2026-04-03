# conv1d/triton_kernel.py
# 1-D valid convolution  out[i] = Σ_{j=0}^{K-1} x[i+j] * w[j]  using Triton.
#
# "Valid" means no padding: the output is shorter than the input by K-1.
#   Input  x:   length n
#   Kernel w:   length K
#   Output out: length n - K + 1
#
# Each program handles BLOCK_SIZE consecutive output elements. For each
# output element it loops over the K kernel weights and accumulates the
# dot product with the corresponding input window.
#
# Grid:  ceil(out_len / BLOCK_SIZE) programs   (1-D)
# Each program processes BLOCK_SIZE output elements.

import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,           # pointer to input vector  (length n)
    w_ptr,           # pointer to kernel weights (length K)
    out_ptr,         # pointer to output vector  (length out_len)
    out_len,         # number of output elements = n - K + 1
    K,               # kernel (filter) length; treated as a runtime value
    BLOCK_SIZE: tl.constexpr,  # output elements per program; must be a power of 2
):
    pid = tl.program_id(axis=0)

    # Output indices this program is responsible for.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask out lanes past the end of the output (handles non-multiples of BLOCK_SIZE).
    mask = offsets < out_len

    # Accumulator: one float per output lane.
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over the K kernel weights.
    # For weight j, each output lane i needs input element x[i + j].
    for j in range(K):
        # Load x[offsets + j] — a contiguous read shifted by j.
        x = tl.load(x_ptr + offsets + j, mask=mask, other=0.0)
        # w[j] is a scalar; tl.load of a single address returns a scalar
        # which Triton broadcasts across the BLOCK_SIZE lanes.
        w = tl.load(w_ptr + j)
        acc += x * w

    tl.store(out_ptr + offsets, acc, mask=mask)


def conv1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda
    assert x.dim() == 1 and w.dim() == 1
    n, k = x.numel(), w.numel()
    out_len = n - k + 1
    assert out_len > 0
    out = torch.empty(out_len, device=x.device, dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(out_len, BLOCK_SIZE),)  # 1-D grid: one program per block
    conv1d_kernel[grid](x.float(), w.float(), out, out_len, k, BLOCK_SIZE=BLOCK_SIZE)
    return out.to(x.dtype)


def test_conv1d():
    torch.manual_seed(0)
    test_cases = [
        (1024,      3),
        (1024,      7),
        (10_000,    3),
        (1_048_576, 3),
        (1_048_576, 15),
    ]

    # Warm up every case so the Triton JIT compile does not inflate timings.
    for n, k in test_cases:
        _x = torch.ones(n, device="cuda")
        _w = torch.ones(k, device="cuda") / k
        conv1d(_x, _w)
    torch.cuda.synchronize()

    for n, k in test_cases:
        out_len = n - k + 1
        print(f"\n--- n={n:,}  k={k}  out_len={out_len:,} ---")
        torch.cuda.reset_peak_memory_stats()

        # 1. allocate inputs on the GPU
        mem0 = torch.cuda.memory_allocated()
        x = torch.randn(n, device="cuda")
        w = torch.randn(k, device="cuda")
        mem1 = torch.cuda.memory_allocated()
        print(f"  [alloc]   {mem1/1024**2:.3f} MB  (+{(mem1-mem0)/1024**2:.3f} MB)")

        # 2. time the kernel with CUDA events (GPU-side, sub-millisecond precision)
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        out = conv1d(x, w)
        end.record()
        torch.cuda.synchronize()  # wait for GPU before reading the timer
        elapsed_ms = start.elapsed_time(end)
        mem2 = torch.cuda.memory_allocated()
        print(f"  [kernel]  {elapsed_ms:.3f} ms   mem={mem2/1024**2:.3f} MB  (+{(mem2-mem1)/1024**2:.3f} MB)")

        # 3. correctness check against torch.nn.functional.conv1d.
        # Reshape to (batch=1, channels=1, length) as F.conv1d expects.
        import torch.nn.functional as F
        expected = F.conv1d(x.unsqueeze(0).unsqueeze(0),
                            w.unsqueeze(0).unsqueeze(0)).squeeze()
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
        max_err = (out - expected).abs().max().item()
        peak = torch.cuda.max_memory_allocated()
        print(f"  [verify]  max_err={max_err:.2e}  peak={peak/1024**2:.3f} MB  PASS")

    print("\nAll tests passed.")


if __name__ == "__main__":
    test_conv1d()
