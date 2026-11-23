# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tune which backend serves ``mslk::f8f8bf16_rowwise_preshuffle`` on ROCm, per
shape.

Compares MSLK's kernel (FlyDSL on gfx950) with aiter's ``gemm_a8w8_bpreshuffle``
on every shape in ``bench/aiter/untuned/f8f8bf16_rowwise_preshuffle.csv`` and
updates ``mslk/gemm/dispatch/tables/f8f8bf16_rowwise_preshuffle.csv``. The
tuning itself (incremental runs, multi-GPU, timing, choosing) is
:mod:`mslk.aiter.tuning`; this file holds only what is specific to the op.
Needs the pinned aiter and a gfx950 GPU. ``tune_f8f8f16_rowwise_preshuffle.py``
reuses it for the FP16 op.

Correctness: the reference is the product of the dequantized inputs in FP32,
which the kernels match up to the GPU's FP8 dot-product accumulation and output
rounding. Weights are scaled by 1/sqrt(K), which leaves their FP8 values as
they are (rowwise quantization moves the factor into ``w_scale``) and makes
the outputs unit-scale at every K, so one tolerance is equally tight at every
shape: atol = 1e-3 and rtol = 1e-2 (one BF16 ulp is up to 2^-7). On gfx950 the
worst element of either kernel uses under 40% of it at K = 256 to 28672, while
dropping one 64-wide block of K moves outputs by ~sqrt(64 / K), over 10x the
tolerance at the largest K. The op allocates its output, so free GPU memory is
filled with NaN before each check: elements a kernel leaves unwritten fail it
instead of reading the previous backend's result.

The aiter backend declines the calls aiter would compute with its unreliable
8-wave FlyDSL kernels (see ``mslk/gemm/dispatch/f8f8bf16_rowwise_preshuffle.py``),
which then run MSLK's kernel. Timing that as aiter would compare MSLK's kernel
with itself, so a declined call raises instead: the profile records aiter as
failing on that shape, and the table keeps MSLK.

Example, on four GPUs (run from the repository root):

    HIP_VISIBLE_DEVICES=0,1,2,3 python bench/aiter/tune_f8f8bf16_rowwise_preshuffle.py --mp 4
"""

import math
from pathlib import Path
from typing import Any, Dict

import mslk.gemm  # noqa: F401  (registers mslk::f8f8bf16_rowwise_preshuffle)
import torch
from mslk.aiter import tuning
from mslk.aiter.routing import choose_backend, OpRouting
from mslk.aiter.tuning import OpTuner, Row
from mslk.gemm.dispatch.f8f8bf16_rowwise_preshuffle import BACKENDS, key, OP, ROUTING
from mslk.quantize.shuffle import preshuffle_b_mfma
from mslk.quantize.triton.fp8_quantize import quantize_fp8_row

REPO = Path(__file__).resolve().parents[2]


def unshuffle(WQ: torch.Tensor) -> torch.Tensor:
    """The ``[N, K]`` weight that ``preshuffle_b_mfma`` turned into ``WQ``."""
    N, K = WQ.shape
    blocks = WQ.view(N // 16, K // 64, 4, 16, 16)
    return blocks.permute(0, 3, 1, 2, 4).reshape(N, K)


class F8F8BF16RowwisePreshuffleTuner(OpTuner):
    op = OP
    key_columns = ("N", "K")
    bucket_column = "M_max"
    backends = BACKENDS
    atol = 1e-3
    rtol = 1e-2
    #: The op's output dtype.
    dtype: torch.dtype = torch.bfloat16
    routing: OpRouting = ROUTING

    def make_inputs(self, shape: Row) -> Dict[str, Any]:
        M, N, K = int(shape["M_max"]), int(shape["N"]), int(shape["K"])
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / math.sqrt(K)
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale = quantize_fp8_row(w)
        return {
            "XQ": xq,
            "WQ": preshuffle_b_mfma(wq),
            "x_scale": x_scale,
            "w_scale": w_scale,
        }

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        args = (inputs["XQ"], inputs["WQ"], inputs["x_scale"], inputs["w_scale"])
        XQ, WQ = args[0], args[1]
        # Graph capture, which the timing replays, skips these.
        if not torch.cuda.is_current_stream_capturing():
            forced = choose_backend(
                self.op, lambda: self.routing.table, XQ.get_device(), *key(*args)
            )
            if self.routing.backend_for(*args) != forced:
                raise RuntimeError(f"the {forced} backend declines this call")
            # The output will be allocated where this NaN buffer was.
            torch.cuda.empty_cache()
            torch.full(
                (XQ.shape[0], WQ.shape[0]),
                float("nan"),
                device=XQ.device,
                dtype=self.dtype,
            )
        return getattr(torch.ops.mslk, self.op)(*args)

    def reference(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        xq, wq = inputs["XQ"], unshuffle(inputs["WQ"])
        x_scale, w_scale = inputs["x_scale"], inputs["w_scale"]
        return (xq.float() @ wq.float().T) * x_scale[:, None] * w_scale

    def flops(self, shape: Row) -> float:
        return 2.0 * int(shape["M_max"]) * int(shape["N"]) * int(shape["K"])

    def bytes_moved(self, shape: Row) -> float:
        M, N, K = int(shape["M_max"]), int(shape["N"]), int(shape["K"])
        # FP8 operands, FP32 scales (one per row of each), 16-bit output.
        return (M + N) * K + 4 * (M + N) + 2 * M * N


if __name__ == "__main__":
    tuning.main(
        F8F8BF16RowwisePreshuffleTuner(),
        untune_file=str(
            REPO / "bench" / "aiter" / "untuned" / "f8f8bf16_rowwise_preshuffle.csv"
        ),
        tune_file=str(
            REPO
            / "mslk"
            / "gemm"
            / "dispatch"
            / "tables"
            / "f8f8bf16_rowwise_preshuffle.csv"
        ),
    )
