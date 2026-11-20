# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tune which backend serves ``mslk::f8f8bf16_rowwise_batched`` on ROCm, per shape.

Compares MSLK's kernel with aiter's Triton ``batched_gemm_a8w8`` on every shape
in ``bench/aiter/untuned/fp8_rowwise_batched.csv`` and updates
``mslk/gemm/dispatch/tables/fp8_rowwise_batched.csv``. The tuning itself
(incremental runs, multi-GPU, timing, choosing) is :mod:`mslk.aiter.tuning`;
this file holds only what is specific to the op. Needs the pinned aiter.

As for the other FP8 rowwise ops (``tune_fp8_rowwise.py``), MSLK's kernel runs
autotuned, and the correctness check compares with the FP32 product of the
quantized operands at rtol = 1e-2 and atol = 5e-2, which covers the FP8 matrix
instructions' accumulation (see there).

Example, on four GPUs (run from the repository root):

    HIP_VISIBLE_DEVICES=0,1,2,3 python bench/aiter/tune_fp8_rowwise_batched.py --mp 4
"""

import os
from pathlib import Path
from typing import Any, Dict

import mslk.gemm  # noqa: F401  (registers the ops)
import torch
from mslk.aiter import tuning
from mslk.aiter.tuning import OpTuner, Row
from mslk.gemm.dispatch.fp8_rowwise_batched import BACKENDS, OP
from mslk.quantize.triton.fp8_quantize import quantize_fp8_row

REPO = Path(__file__).resolve().parents[2]

# Read on each call; set at import so the --mp workers have it too.
os.environ.setdefault("MSLK_AUTOTUNE_ENABLE", "1")


class FP8RowwiseBatchedTuner(OpTuner):
    op = OP
    key_columns = ("B", "N", "K")
    bucket_column = "M_max"
    backends = BACKENDS
    atol = 5e-2
    rtol = 1e-2
    outputs = ("out",)

    def make_inputs(self, shape: Row) -> Dict[str, Any]:
        B, M = int(shape["B"]), int(shape["M_max"])
        N, K = int(shape["N"]), int(shape["K"])
        xq, x_scale = quantize_fp8_row(
            torch.randn(B, M, K, device="cuda", dtype=torch.bfloat16)
        )
        wq, w_scale = quantize_fp8_row(
            torch.randn(B, N, K, device="cuda", dtype=torch.bfloat16)
        )
        out = torch.empty(B, M, N, device="cuda", dtype=torch.bfloat16)
        return {
            "xq": xq,
            "wq": wq,
            "x_scale": x_scale.view(B, M),
            "w_scale": w_scale.view(B, N),
            "out": out,
        }

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        return torch.ops.mslk.f8f8bf16_rowwise_batched(
            inputs["xq"],
            inputs["wq"],
            inputs["x_scale"],
            inputs["w_scale"],
            output=inputs["out"],
        )

    def reference(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        product = torch.bmm(inputs["xq"].float(), inputs["wq"].float().transpose(1, 2))
        return product * inputs["x_scale"][:, :, None] * inputs["w_scale"][:, None, :]

    def flops(self, shape: Row) -> float:
        B, M = int(shape["B"]), int(shape["M_max"])
        return 2.0 * B * M * int(shape["N"]) * int(shape["K"])

    def bytes_moved(self, shape: Row) -> float:
        B, M = int(shape["B"]), int(shape["M_max"])
        N, K = int(shape["N"]), int(shape["K"])
        # FP8 operands, FP32 scales, BF16 output.
        return B * ((M + N) * K + 4 * (M + N) + 2 * M * N)


if __name__ == "__main__":
    tuning.main(
        FP8RowwiseBatchedTuner(),
        untune_file=str(
            REPO / "bench" / "aiter" / "untuned" / "fp8_rowwise_batched.csv"
        ),
        tune_file=str(
            REPO / "mslk" / "gemm" / "dispatch" / "tables" / "fp8_rowwise_batched.csv"
        ),
    )
