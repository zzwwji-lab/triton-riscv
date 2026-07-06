from pathlib import Path
import os
import re
import shutil
import subprocess

import pytest
import torch
import triton

from triton.backends.triton_shared.riscv import DEFAULT_LLC_FEATURES, DEFAULT_TRIPLE

from ._euclidean_dist import _euclidean_dist, _euclidean_dist_kernel


def _get_llvm_bin_path(bin_name):
    llvm_binary_dir = os.environ.get("LLVM_BINARY_DIR")
    if llvm_binary_dir:
        return str(Path(llvm_binary_dir) / bin_name)

    path = shutil.which(bin_name)
    if path:
        return path

    raise RuntimeError(f"Unable to locate '{bin_name}' via LLVM_BINARY_DIR or PATH.")


@pytest.mark.parametrize("N, M, D", [(8, 10, 16), (16, 8, 32), (32, 64, 128)])
def test_euclidean_dist_kernel_emits_riscv_object(tmp_path, N, M, D):
    x1 = torch.empty((N, D), device="cpu", dtype=torch.float32)
    x2 = torch.empty((M, D), device="cpu", dtype=torch.float32)
    out = torch.empty((N, M), device="cpu", dtype=torch.float32)
    obj_path = tmp_path / "euclidean_dist.o"
    block_d = min(triton.next_power_of_2(D), 1024)

    compiled = _euclidean_dist_kernel.warmup(
        x1,
        x2,
        out,
        N,
        M,
        D,
        x1.stride(0),
        x2.stride(0),
        out.stride(0),
        BLOCK_D=block_d,
        grid=(N, M),
        target_triple=DEFAULT_TRIPLE,
        target_features=DEFAULT_LLC_FEATURES,
    )
    obj_path.write_bytes(compiled.asm["obj"])

    asm = subprocess.check_output(
        [_get_llvm_bin_path("llvm-objdump"), "-d", str(obj_path)],
        text=True,
    )
    assert re.search(r"<_euclidean_dist_kernel>:", asm)
    assert re.search(r"\bvsetivli\b", asm)
    assert re.search(r"\bvle32\.v\b", asm)
    assert re.search(r"\bvse32\.v\b", asm)
    assert re.search(r"\bvfsub\.vv\b", asm)
    assert re.search(r"\bvfmul\.vv\b", asm)


def test_euclidean_dist_shape_assertions():
    x1 = torch.randn(8, 16)
    x2 = torch.randn(10, 32)

    with pytest.raises(AssertionError):
        _euclidean_dist(x1, x2)
