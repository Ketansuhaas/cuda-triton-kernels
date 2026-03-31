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

    for n in sizes:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")

        expected = x + y
        actual = vector_add(x, y)

        torch.testing.assert_close(actual, expected)
        print(f"n={n:>10,}  max_err={( actual - expected).abs().max().item():.2e}  PASS")

    print("\nAll tests passed.")


if __name__ == "__main__":
    test_vector_add()
