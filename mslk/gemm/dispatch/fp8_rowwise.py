# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Which backend serves MSLK's FP8 rowwise GEMM ops on ROCm
(``mslk::f8f8bf16_rowwise``, ``f8f8bf16_rowwise_out`` and ``f8f8f16_rowwise``).

- ``mslk``: MSLK's own kernel (FlyDSL where it supports the GPU, else CK; see
  ``mslk/gemm/__init__.py``), the default and the fallback.
- ``aiter_ck`` and ``aiter_triton``: aiter's ``gemm_a8w8``
  (``mslk/gemm/aiter/fp8_rowwise.py``), for the shapes the op's table assigns
  to them.

The ops that return their output share ``tables/fp8_rowwise.csv``, tuned on
``f8f8bf16_rowwise``: ``f8f8f16_rowwise`` differs only in the output's dtype.
``f8f8bf16_rowwise_out`` writes into the caller's buffer, which aiter's CK
backend cannot, so it has its own table, ``tables/fp8_rowwise_out.csv``, with
aiter's Triton backend as its only aiter candidate. The tables are written by
``bench/aiter/tune_fp8_rowwise.py`` and ``tune_fp8_rowwise_out.py``.

Each op has its own routing under its own name, which
:func:`~mslk.aiter.routing.force_backend` and ``MSLK_DISABLE_AITER_OPS`` take.
aiter only serves calls MSLK's kernel accepts; others go to MSLK's kernel,
which raises its usual error.

The tables are tuned with MSLK's kernel autotuned (``MSLK_AUTOTUNE_ENABLE``):
without it MSLK's kernel runs a fixed default tile, slower on many shapes than
the table assumes. :func:`route` logs a warning, once, when the tables are in
use without it.
"""

import functools
import logging
import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from mslk.aiter.common import is_aiter_enabled
from mslk.aiter.routing import MSLK, OpRouting

logger: logging.Logger = logging.getLogger(__name__)

AITER_CK: str = "aiter_ck"
AITER_TRITON: str = "aiter_triton"

#: The ops each table is tuned on, and their candidate backends.
OP: str = "f8f8bf16_rowwise"
BACKENDS: Tuple[str, ...] = (MSLK, AITER_CK, AITER_TRITON)
OUT_OP: str = "f8f8bf16_rowwise_out"
OUT_BACKENDS: Tuple[str, ...] = (MSLK, AITER_TRITON)


def _mslk(op: str) -> Callable[..., object]:
    # Imported on first use: importing mslk.gemm must not pull in a backend.
    def run(*args: object, **kwargs: object) -> object:
        from mslk.gemm import _rowwise_rocm

        return _rowwise_rocm("fp8_rowwise_gemm", op)(*args, **kwargs)

    return run


def _aiter(backend: str, dtype: torch.dtype) -> Callable[..., torch.Tensor]:
    def run(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        use_fast_accum: bool = True,
    ) -> torch.Tensor:
        from mslk.gemm.aiter import fp8_rowwise

        return getattr(fp8_rowwise, f"gemm_{backend}")(XQ, WQ, x_scale, w_scale, dtype)

    return run


def _aiter_triton_out(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> None:
    from mslk.gemm.aiter import fp8_rowwise

    fp8_rowwise.gemm_triton(XQ, WQ, x_scale, w_scale, torch.bfloat16, output)


def _operands_valid(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    use_fast_accum: bool,
) -> bool:
    # What MSLK's kernel accepts (mslk/gemm/flydsl/fp8_rowwise_gemm.py).
    return (
        bias is None
        and use_fast_accum
        and XQ.dtype == WQ.dtype
        and XQ.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        and x_scale.dtype == w_scale.dtype == torch.float32
        and WQ.dim() == 2
        and XQ.dim() >= 1
        and XQ.shape[-1] == WQ.shape[1]
        and XQ.is_contiguous()
        and WQ.is_contiguous()
        and x_scale.numel() == math.prod(XQ.shape[:-1])
        and w_scale.numel() == WQ.shape[0]
        and WQ.device == x_scale.device == w_scale.device == XQ.device
    )


def _accepts(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> bool:
    return _operands_valid(XQ, WQ, x_scale, w_scale, bias, use_fast_accum)


def _accepts_out(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> bool:
    return (
        _operands_valid(XQ, WQ, x_scale, w_scale, bias, use_fast_accum)
        and output.dtype == torch.bfloat16
        and output.is_contiguous()
        and output.device == XQ.device
        and output.dim() >= 1
        and output.shape[-1] == WQ.shape[0]
        and math.prod(output.shape[:-1]) == math.prod(XQ.shape[:-1])
    )


def _key(
    XQ: torch.Tensor, WQ: torch.Tensor, *args: object, **kwargs: object
) -> Tuple[Sequence[object], int]:
    return (WQ.shape[0], WQ.shape[1]), math.prod(XQ.shape[:-1])


def _routing(
    op: str, table: str, backends: Dict[str, Callable[..., object]]
) -> OpRouting:
    accepts = _accepts_out if op == OUT_OP else _accepts
    return OpRouting(
        op,
        "mslk.gemm.dispatch",
        table,
        key_columns=("N", "K"),
        bucket_column="M_max",
        backends={MSLK: _mslk(op), **backends},
        key=_key,
        accepts={name: accepts for name in backends},
    )


#: Each op's routing: call it with the op's arguments.
ROUTINGS: Dict[str, OpRouting] = {
    "f8f8bf16_rowwise": _routing(
        "f8f8bf16_rowwise",
        "tables/fp8_rowwise.csv",
        {
            AITER_CK: _aiter("ck", torch.bfloat16),
            AITER_TRITON: _aiter("triton", torch.bfloat16),
        },
    ),
    "f8f8f16_rowwise": _routing(
        "f8f8f16_rowwise",
        "tables/fp8_rowwise.csv",
        {
            AITER_CK: _aiter("ck", torch.float16),
            AITER_TRITON: _aiter("triton", torch.float16),
        },
    ),
    OUT_OP: _routing(
        OUT_OP, "tables/fp8_rowwise_out.csv", {AITER_TRITON: _aiter_triton_out}
    ),
}


@functools.lru_cache(maxsize=1)
def _warn_untuned(env: str) -> None:
    logger.warning(
        "MSLK's FP8 rowwise GEMM ops choose between MSLK's kernel and aiter per "
        "shape, from tables tuned with MSLK's kernel autotuned, but %s is not "
        "set: MSLK's kernel runs its default tile, which on many shapes is slower "
        "than the tables assume. Set %s=1 for the performance they were tuned for.",
        env,
        env,
    )


def route(op: str, *args: Any, **kwargs: Any) -> Any:
    """Serve a call of ``op`` (one of :data:`ROUTINGS`)."""
    from mslk.flydsl.autotune import autotune_enabled, AUTOTUNE_ENV

    if not autotune_enabled() and is_aiter_enabled(op):
        _warn_untuned(AUTOTUNE_ENV)
    return ROUTINGS[op](*args, **kwargs)
