# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""aiter backend of ``mslk::f8f8f16_rowwise_preshuffle`` (FP8 x FP8 -> FP16,
rowwise scales, preshuffled weight) on ROCm.

The same as ``mslk::f8f8bf16_rowwise_preshuffle``'s
(``mslk/gemm/aiter/f8f8bf16_rowwise_preshuffle.py``), with FP16 output: aiter's
``gemm_a8w8_bpreshuffle`` takes the op's arguments unchanged.

Which calls this backend serves is decided in
``mslk/gemm/dispatch/f8f8f16_rowwise_preshuffle.py``.
"""

import torch
from mslk.gemm.aiter.f8f8bf16_rowwise_preshuffle import rowwise_preshuffle


def f8f8f16_rowwise_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """``mslk::f8f8f16_rowwise_preshuffle`` computed by aiter's
    ``gemm_a8w8_bpreshuffle``, without bias and with fast accumulation, the
    only modes MSLK's own ROCm kernels support."""
    return rowwise_preshuffle(XQ, WQ, x_scale, w_scale, torch.float16)
