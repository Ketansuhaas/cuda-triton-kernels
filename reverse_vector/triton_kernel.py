# reverse_vector/triton_kernel.py
# Reverse a vector: out[i] = x[n - 1 - i]  using Triton.
#
# Each program reads a block of BLOCK_SIZE elements at increasing offsets
# and writes them to the mirror positions at the other end of the array.
# Because every program reads from a unique source range and writes to a
# unique destination range there are no data hazards.
#
# Grid:  ceil(n / BLOCK_SIZE) programs   (1-D)
# Each program processes BLOCK_SIZE elements.

import torch
import triton
import triton.language as tl


@triton.jit
def reverse_vector_kernel(
    x_ptr,           # pointer to input vector
    out_ptr,         # pointer to output vector
    n_elements,      # total number of elements
    BLOCK_SIZE: tl.constexpr,  # elements per program; must be a power of 2
):
    # Which block of BLOCK_SIZE elements is this program reading from?
    pid = tl.program_id(axis=0)

    # Forward offsets: the input positions this program reads.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mirror offsets: where each element lands in the output.
    # Element at position i goes to position (n-1-i).
    reversed_offsets = (n_elements - 1) - offsets

    # Mask out lanes past the end of the array.
    mask = offsets < n_elements

    # Load from forward positions, store at mirrored positions.
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + reversed_offsets, x, mask=mask)


def reverse_vector(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)  # 1-D grid: one program per block
    reverse_vector_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def test_reverse_vector():
    torch.manual_seed(0)
    sizes = [1, 127, 1024, 10_000, 1_048_576]

    # Warm up every size so the Triton JIT compile does not inflate timings.
    for n in sizes:
        _x = torch.ones(n, device="cuda")
        reverse_vector(_x)
    torch.cuda.synchronize()

    for n in sizes:
        print(f"\n--- n={n:,} ---")
        torch.cuda.reset_peak_memory_stats()

        # 1. allocate input tensor on the GPU
        mem0 = torch.cuda.memory_allocated()
        x = torch.randn(n, device="cuda")
        mem1 = torch.cuda.memory_allocated()
        print(f"  [alloc]   {mem1/1024**2:.3f} MB  (+{(mem1-mem0)/1024**2:.3f} MB)")

        # 2. time the kernel with CUDA events (GPU-side, sub-millisecond precision)
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        out = reverse_vector(x)
        end.record()
        torch.cuda.synchronize()  # wait for GPU before reading the timer
        elapsed_ms = start.elapsed_time(end)
        mem2 = torch.cuda.memory_allocated()
        print(f"  [kernel]  {elapsed_ms:.3f} ms   mem={mem2/1024**2:.3f} MB  (+{(mem2-mem1)/1024**2:.3f} MB)")

        # 3. correctness check: compare against torch.flip reference
        expected = x.flip(0)
        torch.testing.assert_close(out, expected)
        max_err = (out - expected).abs().max().item()
        peak = torch.cuda.max_memory_allocated()
        print(f"  [verify]  max_err={max_err:.2e}  peak={peak/1024**2:.3f} MB  PASS")

    print("\nAll tests passed.")


if __name__ == "__main__":
    test_reverse_vector()
