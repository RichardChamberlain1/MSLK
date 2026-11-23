# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which backend serves ``mslk::f8f8bf16_rowwise_preshuffle`` on ROCm.

- ``mslk``: MSLK's kernel, FlyDSL on gfx942 and gfx950 and CK elsewhere (chosen
  in ``mslk/gemm/__init__.py``), the default and the fallback.
- ``aiter``: aiter's ``gemm_a8w8_bpreshuffle``
  (``mslk/gemm/aiter/f8f8bf16_rowwise_preshuffle.py``), for the shapes
  ``tables/f8f8bf16_rowwise_preshuffle.csv`` assigns to it. That table is
  written by ``bench/aiter/tune_f8f8bf16_rowwise_preshuffle.py``.

aiter only gets calls that MSLK's kernel computes; calls it rejects (bias, no
fast accumulation, wrong dtypes or scale counts, ...) still go to it, and so
fail as before. Nor does aiter get calls it would compute with its 8-wave
FlyDSL kernels, which are unreliable (see
``mslk.gemm.aiter.f8f8bf16_rowwise_preshuffle.has_reliable_kernel``).

``mslk::f8f8f16_rowwise_preshuffle`` (FP16 output) is routed the same way, by
``mslk/gemm/dispatch/f8f8f16_rowwise_preshuffle.py`` with its own table, from
:func:`make_routing`.
"""

from math import prod
from typing import Callable, Optional, Sequence, Tuple

import torch
from mslk.aiter.routing import device_key, MSLK, OpRouting

OP: str = "f8f8bf16_rowwise_preshuffle"
AITER: str = "aiter"


def mslk_kernel(
    op: str,
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    """MSLK's own kernel for ``op``, as the op used before aiter."""
    from mslk.gemm import _rowwise_preshuffle_impl

    return _rowwise_preshuffle_impl(op, XQ.get_device())(
        XQ, WQ, x_scale, w_scale, bias, use_fast_accum
    )


# Backends are imported on first use: importing mslk.gemm must not pull in
# aiter.
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
    from mslk.gemm.aiter import f8f8bf16_rowwise_preshuffle as aiter_op

    return aiter_op.f8f8bf16_rowwise_preshuffle(XQ, WQ, x_scale, w_scale)


def aiter_accepts(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> bool:
    """True for the calls MSLK's kernel computes, rather than rejects or
    answers with zeros (empty shapes), and that aiter computes reliably."""
    if bias is not None or not use_fast_accum or XQ.dim() == 0 or WQ.dim() != 2:
        return False
    N, K = WQ.shape
    M = prod(XQ.shape[:-1])
    # The GPU's FP8 type, as MSLK's kernel and aiter require.
    fp8 = (
        torch.float8_e4m3fn
        if device_key(XQ.get_device())[0] == "gfx950"
        else torch.float8_e4m3fnuz
    )
    if not (
        M > 0
        and N > 0
        and K > 0
        and N % 16 == 0
        and K % 64 == 0
        and XQ.shape[-1] == K
        and XQ.dtype == fp8
        and WQ.dtype == fp8
        and x_scale.dtype == torch.float32
        and w_scale.dtype == torch.float32
        and x_scale.numel() == M
        and w_scale.numel() == N
        and XQ.is_contiguous()
        and WQ.is_contiguous()
        and XQ.device == WQ.device == x_scale.device == w_scale.device
    ):
        return False
    from mslk.gemm.aiter import f8f8bf16_rowwise_preshuffle as aiter_op

    return aiter_op.has_reliable_kernel(M, N, K)


def key(
    XQ: torch.Tensor, WQ: torch.Tensor, *args: object, **kwargs: object
) -> Tuple[Sequence[object], int]:
    # XQ is [..., K] and WQ [N, K]; M is the product of XQ's leading dims.
    # Malformed calls get a key no table has, so MSLK's kernel rejects them.
    if XQ.dim() == 0 or WQ.dim() == 0:
        return (), 0
    return (WQ.shape[0], XQ.shape[-1]), prod(XQ.shape[:-1])


def make_routing(
    op: str,
    mslk: Callable[..., torch.Tensor],
    aiter: Callable[..., torch.Tensor],
) -> OpRouting:
    """The routing of one of the rowwise preshuffle ops, from its two backends;
    called with the op's arguments."""
    return OpRouting(
        op,
        "mslk.gemm.dispatch",
        f"tables/{op}.csv",
        key_columns=("N", "K"),
        bucket_column="M_max",
        backends={MSLK: mslk, AITER: aiter},
        key=key,
        accepts={AITER: aiter_accepts},
    )


#: ``mslk::f8f8bf16_rowwise_preshuffle`` on ROCm: call it with the op's arguments.
ROUTING: OpRouting = make_routing(OP, _mslk, _aiter)
BACKENDS: Tuple[str, ...] = tuple(ROUTING.backends)
