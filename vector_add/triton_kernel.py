import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.is_cuda and y.is_cuda
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def test_vector_add():
    torch.manual_seed(0)
    sizes = [1, 127, 1024, 10_000, 1_048_576]

    # warm up so first-run JIT compile doesn't pollute timing
    _x = torch.ones(1, device="cuda")
    vector_add(_x, _x)
    torch.cuda.synchronize()

    for n in sizes:
        print(f"\n--- n={n:,} ---")
        torch.cuda.reset_peak_memory_stats()

        # 1. allocate
        mem0 = torch.cuda.memory_allocated()
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        mem1 = torch.cuda.memory_allocated()
        print(f"  [alloc]   {mem1/1024**2:.3f} MB  (+{(mem1-mem0)/1024**2:.3f} MB)")

        # 2. kernel
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        out = vector_add(x, y)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        mem2 = torch.cuda.memory_allocated()
        print(f"  [kernel]  {elapsed_ms:.3f} ms   mem={mem2/1024**2:.3f} MB  (+{(mem2-mem1)/1024**2:.3f} MB)")

        # 3. verify
        expected = x + y
        torch.testing.assert_close(out, expected)
        max_err = (out - expected).abs().max().item()
        peak = torch.cuda.max_memory_allocated()
        print(f"  [verify]  max_err={max_err:.2e}  peak={peak/1024**2:.3f} MB  PASS")

    print("\nAll tests passed.")


if __name__ == "__main__":
    test_vector_add()
