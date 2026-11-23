# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tune which backend serves ``mslk::f8f8bf16_rowwise_out`` on ROCm, per shape.

As ``tune_fp8_rowwise.py``, on the same shapes, for the op that writes into the
caller's buffer: compares MSLK's kernel with aiter's Triton ``gemm_a8w8`` (aiter's
CK kernel allocates its output, so it is not a candidate) and updates
``mslk/gemm/dispatch/tables/fp8_rowwise_out.csv``.

Example, on four GPUs (run from the repository root):

    HIP_VISIBLE_DEVICES=0,1,2,3 python bench/aiter/tune_fp8_rowwise_out.py --mp 4
"""

from typing import Any, Dict

import torch
from mslk.aiter import tuning
from mslk.aiter.tuning import Row
from mslk.gemm.dispatch.fp8_rowwise import OUT_BACKENDS, OUT_OP
from tune_fp8_rowwise import FP8RowwiseTuner, REPO


class FP8RowwiseOutTuner(FP8RowwiseTuner):
    op = OUT_OP
    backends = OUT_BACKENDS
    outputs = ("out",)

    def make_inputs(self, shape: Row) -> Dict[str, Any]:
        inputs = super().make_inputs(shape)
        inputs["out"] = torch.empty(
            int(shape["M_max"]), int(shape["N"]), device="cuda", dtype=torch.bfloat16
        )
        return inputs

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        torch.ops.mslk.f8f8bf16_rowwise_out(
            inputs["xq"],
            inputs["wq"],
            inputs["x_scale"],
            inputs["w_scale"],
            inputs["out"],
        )
        return inputs["out"]


if __name__ == "__main__":
    tuning.main(
        FP8RowwiseOutTuner(),
        untune_file=str(REPO / "bench" / "aiter" / "untuned" / "fp8_rowwise.csv"),
        tune_file=str(
            REPO / "mslk" / "gemm" / "dispatch" / "tables" / "fp8_rowwise_out.csv"
        ),
    )
