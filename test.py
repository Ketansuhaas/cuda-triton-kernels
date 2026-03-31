"""
Run Triton kernel tests on Modal.

Usage:
  modal run modal_test.py --kernel vector_add.py
  modal run modal_test.py --kernel vector_add.py --gpu H100
"""

import modal

app = modal.App("cuda-triton")

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install("triton")
    .add_local_dir(
        ".",
        remote_path="/workspace/cuda",
        ignore=["**/__pycache__/**", "**/*.pyc", ".git/**"],
    )
)

@app.function(image=image, gpu="T4")
def run_on_t4(kernel: str):
    _run(kernel)


@app.function(image=image, gpu="A10G")
def run_on_a10g(kernel: str):
    _run(kernel)


@app.function(image=image, gpu="A100-40GB")
def run_on_a100(kernel: str):
    _run(kernel)


@app.function(image=image, gpu="H100")
def run_on_h100(kernel: str):
    _run(kernel)


def _run(kernel: str):
    import subprocess
    if kernel.endswith(".cu"):
        compile = subprocess.run(
            ["nvcc", "-o", "/tmp/kernel_out", kernel],
            cwd="/workspace/cuda",
            capture_output=True,
            text=True,
        )
        if compile.returncode != 0:
            print(compile.stderr)
            raise RuntimeError(f"nvcc compilation failed for {kernel}")
        result = subprocess.run(["/tmp/kernel_out"], capture_output=True, text=True)
    else:
        result = subprocess.run(
            ["python", kernel],
            cwd="/workspace/cuda",
            capture_output=True,
            text=True,
        )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"{kernel} failed")


_RUNNERS = {
    "T4":   run_on_t4,
    "A10G": run_on_a10g,
    "A100": run_on_a100,
    "H100": run_on_h100,
}


@app.local_entrypoint()
def main(kernel: str, gpu: str = "T4"):
    gpu = gpu.upper()
    if gpu not in _RUNNERS:
        raise ValueError(f"Unsupported GPU '{gpu}'. Choose from: {list(_RUNNERS)}")
    _RUNNERS[gpu].remote(kernel)
