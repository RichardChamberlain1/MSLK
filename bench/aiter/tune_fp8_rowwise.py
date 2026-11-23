# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tune which backend serves ``mslk::f8f8bf16_rowwise`` and ``f8f8f16_rowwise``
on ROCm, per shape.

Compares MSLK's kernel with aiter's CK and Triton ``gemm_a8w8`` on every shape
in ``bench/aiter/untuned/fp8_rowwise.csv`` and updates
``mslk/gemm/dispatch/tables/fp8_rowwise.csv``, which the two ops share.
``f8f8bf16_rowwise_out`` has a table of its own, tuned by
``tune_fp8_rowwise_out.py``. The tuning itself
(incremental runs, multi-GPU, timing, choosing) is :mod:`mslk.aiter.tuning`;
this file holds only what is specific to the ops. Needs the pinned aiter.

MSLK's kernel runs autotuned: this script sets ``MSLK_AUTOTUNE_ENABLE=1`` unless
it is set already. The tables are tuned for that, and the ops warn when used
without it (see ``mslk/gemm/dispatch/fp8_rowwise.py``). Autotuning happens in
the warm-up, so it is not timed.

Correctness: the reference is the FP32 product of the quantized operands,
scaled. rtol = 1e-2 covers the output's rounding. atol = 5e-2 covers the FP8
matrix instructions' accumulation, which is less precise than FP32 and is the
same in every backend (and in hipBLASLt): it differs from the exact sum by up to
about 1e-2 at K = 28672, where an output near zero sums products of either sign.
Inputs are unit-scale, so typical outputs are about 100, far larger than atol,
and a wrong or partial result fails at every K.

Example, on four GPUs (run from the repository root):

    HIP_VISIBLE_DEVICES=0,1,2,3 python bench/aiter/tune_fp8_rowwise.py --mp 4
"""

import os
from pathlib import Path
from typing import Any, Dict

import mslk.gemm  # noqa: F401  (registers the ops)
import torch
from mslk.aiter import tuning
from mslk.aiter.tuning import OpTuner, Row
from mslk.gemm.dispatch.fp8_rowwise import BACKENDS, OP
from mslk.quantize.triton.fp8_quantize import quantize_fp8_row

REPO = Path(__file__).resolve().parents[2]

# Read on each call; set at import so the --mp workers have it too.
os.environ.setdefault("MSLK_AUTOTUNE_ENABLE", "1")


class FP8RowwiseTuner(OpTuner):
    op = OP
    key_columns = ("N", "K")
    bucket_column = "M_max"
    backends = BACKENDS
    atol = 5e-2
    rtol = 1e-2

    def make_inputs(self, shape: Row) -> Dict[str, Any]:
        M, N, K = int(shape["M_max"]), int(shape["N"]), int(shape["K"])
        xq, x_scale = quantize_fp8_row(
            torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        )
        wq, w_scale = quantize_fp8_row(
            torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        )
        return {"xq": xq, "wq": wq, "x_scale": x_scale, "w_scale": w_scale}

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        return torch.ops.mslk.f8f8bf16_rowwise(
            inputs["xq"], inputs["wq"], inputs["x_scale"], inputs["w_scale"]
        )

    def reference(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        product = inputs["xq"].float() @ inputs["wq"].float().T
        return product * inputs["x_scale"][:, None] * inputs["w_scale"][None, :]

    def flops(self, shape: Row) -> float:
        return 2.0 * int(shape["M_max"]) * int(shape["N"]) * int(shape["K"])

    def bytes_moved(self, shape: Row) -> float:
        M, N, K = int(shape["M_max"]), int(shape["N"]), int(shape["K"])
        # FP8 operands, FP32 scales, BF16 output.
        return (M + N) * K + 4 * (M + N) + 2 * M * N


if __name__ == "__main__":
    tuning.main(
        FP8RowwiseTuner(),
        untune_file=str(REPO / "bench" / "aiter" / "untuned" / "fp8_rowwise.csv"),
        tune_file=str(
            REPO / "mslk" / "gemm" / "dispatch" / "tables" / "fp8_rowwise.csv"
        ),
    )
