# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Which backend serves ``mslk::f8f8bf16_rowwise_batched`` on ROCm.

- ``mslk``: MSLK's own kernel (FlyDSL where it supports the GPU, else CK; see
  ``mslk/gemm/__init__.py``), the default and the fallback.
- ``aiter_triton``: aiter's Triton ``batched_gemm_a8w8``
  (``mslk/gemm/aiter/fp8_rowwise.py``), for the shapes
  ``tables/fp8_rowwise_batched.csv`` assigns to it. That table is written by
  ``bench/aiter/tune_fp8_rowwise_batched.py``.

The table is keyed by the batch count, N and K, with M bucketed. Like the other
FP8 rowwise ops' tables, it is tuned with MSLK's kernel autotuned, and
:func:`route` warns, once, when it is in use without ``MSLK_AUTOTUNE_ENABLE``.
aiter only serves calls MSLK's kernel accepts; others go to MSLK's kernel,
which raises its usual error.
"""

from typing import Any, Optional, Sequence, Tuple

import torch
from mslk.aiter.common import is_aiter_enabled
from mslk.aiter.routing import MSLK, OpRouting

AITER_TRITON: str = "aiter_triton"
OP: str = "f8f8bf16_rowwise_batched"
BACKENDS: Tuple[str, ...] = (MSLK, AITER_TRITON)


def _mslk(*args: object, **kwargs: object) -> object:
    # Imported on first use: importing mslk.gemm must not pull in a backend.
    from mslk.gemm import _rowwise_rocm

    return _rowwise_rocm("fp8_rowwise_gemm", OP)(*args, **kwargs)


def _aiter_triton(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from mslk.gemm.aiter import fp8_rowwise

    return fp8_rowwise.gemm_triton_batched(XQ, WQ, x_scale, w_scale, output)


def _accepts(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
    output: Optional[torch.Tensor] = None,
) -> bool:
    # What MSLK's kernel accepts (mslk/gemm/flydsl/fp8_rowwise_gemm.py).
    if not (
        bias is None
        and use_fast_accum
        and XQ.dim() == 3
        and WQ.dim() == 3
        and XQ.dtype == WQ.dtype
        and XQ.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        and x_scale.dtype == w_scale.dtype == torch.float32
        and XQ.is_contiguous()
        and WQ.is_contiguous()
        and WQ.device == x_scale.device == w_scale.device == XQ.device
    ):
        return False
    b, m, k = XQ.shape
    n = WQ.shape[1]
    return (
        WQ.shape[0] == b
        and WQ.shape[2] == k
        and x_scale.numel() == b * m
        and w_scale.numel() == b * n
        and (
            output is None
            or (
                tuple(output.shape) == (b, m, n)
                and output.dtype == torch.bfloat16
                and output.is_contiguous()
                and output.device == XQ.device
            )
        )
    )


def _key(
    XQ: torch.Tensor, WQ: torch.Tensor, *args: object, **kwargs: object
) -> Tuple[Sequence[object], int]:
    return (WQ.shape[0], WQ.shape[1], WQ.shape[2]), XQ.shape[1]


#: ``mslk::f8f8bf16_rowwise_batched`` on ROCm: call it with the op's arguments.
ROUTING: OpRouting = OpRouting(
    OP,
    "mslk.gemm.dispatch",
    "tables/fp8_rowwise_batched.csv",
    key_columns=("B", "N", "K"),
    bucket_column="M_max",
    backends={MSLK: _mslk, AITER_TRITON: _aiter_triton},
    key=_key,
    accepts={AITER_TRITON: _accepts},
)


def route(*args: Any, **kwargs: Any) -> Any:
    """Serve a call of the op."""
    from mslk.flydsl.autotune import autotune_enabled, AUTOTUNE_ENV
    from mslk.gemm.dispatch.fp8_rowwise import _warn_untuned

    if not autotune_enabled() and is_aiter_enabled(OP):
        _warn_untuned(AUTOTUNE_ENV)
    return ROUTING(*args, **kwargs)
