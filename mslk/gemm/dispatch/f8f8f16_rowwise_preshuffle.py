# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which backend serves ``mslk::f8f8f16_rowwise_preshuffle`` on ROCm.

The same choice as for ``mslk::f8f8bf16_rowwise_preshuffle`` (see
``mslk/gemm/dispatch/f8f8bf16_rowwise_preshuffle.py``), with FP16 output:

- ``mslk``: MSLK's kernel, FlyDSL on gfx942 and gfx950 and CK elsewhere, the
  default and the fallback.
- ``aiter``: aiter's ``gemm_a8w8_bpreshuffle``
  (``mslk/gemm/aiter/f8f8f16_rowwise_preshuffle.py``), for the shapes
  ``tables/f8f8f16_rowwise_preshuffle.csv`` assigns to it. That table is
  written by ``bench/aiter/tune_f8f8f16_rowwise_preshuffle.py``.
"""

from typing import Optional, Tuple

import torch
from mslk.aiter.routing import OpRouting
from mslk.gemm.dispatch.f8f8bf16_rowwise_preshuffle import make_routing, mslk_kernel

OP: str = "f8f8f16_rowwise_preshuffle"


def _mslk(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    return mslk_kernel(OP, XQ, WQ, x_scale, w_scale, bias, use_fast_accum)


def _aiter(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    from mslk.gemm.aiter import f8f8f16_rowwise_preshuffle as aiter_op

    return aiter_op.f8f8f16_rowwise_preshuffle(XQ, WQ, x_scale, w_scale)


#: ``mslk::f8f8f16_rowwise_preshuffle`` on ROCm: call it with the op's arguments.
ROUTING: OpRouting = make_routing(OP, _mslk, _aiter)
BACKENDS: Tuple[str, ...] = tuple(ROUTING.backends)
