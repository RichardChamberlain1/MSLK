# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tune which backend serves ``quantize_fp8_tensor`` on ROCm, per shape.

Compares MSLK's Triton kernel with aiter's ``per_tensor_quant_hip`` on every
shape in ``bench/aiter/untuned/quantize_fp8_tensor.csv`` and updates
``mslk/quantize/dispatch/tables/quantize_fp8_tensor.csv``. The tuning itself
(incremental runs, multi-GPU, timing, choosing) is :mod:`mslk.aiter.tuning`;
this file holds only what is specific to the op. Needs the pinned aiter and a
gfx950 GPU.

Correctness: the reference is the exact quantization: the scale
``max(amax, eps) / FP8_MAX`` and each element times its reciprocal, saturated
and rounded to FP8. Both kernels match it up to rounding: their scale can be an
FP32 ulp off, and so a rare element one FP8 step off. The check compares, as
one tensor, every FP8 value and the scale's relative error times 4096, with
atol = 2^-9 and rtol = 2^-3: each value must be within one FP8 step of the
reference (2^-9 is the smallest subnormal step of float8_e4m3fn), and the scale
within 2^-21 of it, about 4 FP32 ulps, so a wrong amax, a wrong or partial
result, or unwritten output fails. Inputs are standard normal, so every shape
has values across FP8's range. The op allocates its output, so free GPU memory
is filled with NaN before each check: elements a kernel leaves unwritten fail
it instead of reading the previous backend's result.

Example, on four GPUs (run from the repository root):

    HIP_VISIBLE_DEVICES=0,1,2,3 python bench/aiter/tune_quantize_fp8_tensor.py --mp 4
"""

from pathlib import Path
from typing import Any, Dict

import mslk.quantize  # noqa: F401
import torch
from mslk.aiter import tuning
from mslk.aiter.tuning import OpTuner, Row
from mslk.quantize.dispatch.quantize_fp8_tensor import BACKENDS, OP
from mslk.quantize.triton.fp8_quantize import quantize_fp8_tensor
from mslk.utils.triton.fp8_utils import get_fp8_constants

REPO = Path(__file__).resolve().parents[2]

FP8, _, FP8_MAX, EPS = get_fp8_constants()
#: Scales the scale's relative error so that atol bounds it by 2^-21.
SCALE_ERROR_WEIGHT = 4096.0


class QuantizeFP8TensorTuner(OpTuner):
    op = OP
    key_columns = ("dtype", "K")
    bucket_column = "M_max"
    backends = BACKENDS
    atol = 2.0**-9
    rtol = 2.0**-3

    def make_inputs(self, shape: Row) -> Dict[str, Any]:
        M, K = int(shape["M_max"]), int(shape["K"])
        a = torch.randn(M, K, device="cuda", dtype=getattr(torch, shape["dtype"]))
        amax = a.abs().max().float().clamp(min=EPS)
        return {"a": a, "scale_ref": amax / FP8_MAX}

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        a = inputs["a"]
        # Graph capture, which the timing replays, runs the op alone.
        if torch.cuda.is_current_stream_capturing():
            return quantize_fp8_tensor(a)[0]
        # The output will be allocated where this NaN buffer was.
        torch.cuda.empty_cache()
        torch.full(a.shape, float("nan"), device=a.device, dtype=FP8)
        a_fp8, scale = quantize_fp8_tensor(a)
        scale_error = scale.reshape(1) / inputs["scale_ref"].reshape(1) - 1
        return torch.cat([a_fp8.float().reshape(-1), scale_error * SCALE_ERROR_WEIGHT])

    def reference(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        a, scale_ref = inputs["a"], inputs["scale_ref"]
        a_fp8 = (a.float() * (1.0 / scale_ref)).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        return torch.cat([a_fp8.float().reshape(-1), scale_ref.new_zeros(1)])

    def bytes_moved(self, shape: Row) -> float:
        M, K = int(shape["M_max"]), int(shape["K"])
        # 16-bit input, FP8 output.
        return 3.0 * M * K


if __name__ == "__main__":
    tuning.main(
        QuantizeFP8TensorTuner(),
        untune_file=str(
            REPO / "bench" / "aiter" / "untuned" / "quantize_fp8_tensor.csv"
        ),
        tune_file=str(
            REPO
            / "mslk"
            / "quantize"
            / "dispatch"
            / "tables"
            / "quantize_fp8_tensor.csv"
        ),
    )
