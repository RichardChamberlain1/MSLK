# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""aiter backends of MSLK's FP8 rowwise GEMM ops on ROCm
(``mslk::f8f8bf16_rowwise``, ``f8f8bf16_rowwise_out``, ``f8f8f16_rowwise`` and
``f8f8bf16_rowwise_batched``).

aiter's ``gemm_a8w8`` computes these ops: ``out = (XQ @ WQ^T) * x_scale *
w_scale`` with one FP32 scale per row of each operand. It takes the scales as
``[M, 1]`` and ``[1, N]``, which are views of MSLK's ``[M]`` and ``[N]``. aiter
exposes it twice, with different kernels behind each:

- ``gemm_ck``: ``aiter.gemm_a8w8``, which runs aiter's CK kernel on CDNA GPUs.
  It allocates its output, so it serves only the ops that return one. It runs
  without split-K: aiter's tuned configs give some shapes a split-K, whose
  partial sums are added in the output's precision, which loses accuracy
  against MSLK's kernel (on 1280 x 8192, 10-17% of elements outside rtol = atol
  = 1e-2); and as the configs cover only some values of M, the tuning could not
  check every M in a bucket for it.
- ``gemm_triton``: aiter's Triton ``gemm_a8w8``, which writes into a given
  output, so it also serves ``f8f8bf16_rowwise_out``.
- ``gemm_triton_batched``: aiter's Triton ``batched_gemm_a8w8``, for
  ``f8f8bf16_rowwise_batched``. aiter's CK ``batched_gemm_a8w8`` takes INT8
  operands only, so it is not a backend.

Which calls these backends serve is decided in
``mslk/gemm/dispatch/fp8_rowwise.py`` and ``fp8_rowwise_batched.py``.
"""

import math
from typing import Optional

import torch
from mslk.aiter.common import import_aiter_module


def gemm_ck(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The rowwise GEMM computed by aiter's CK ``gemm_a8w8``."""
    gemm_a8w8 = import_aiter_module("aiter").gemm_a8w8
    # The ops take XQ as [..., M, K]; aiter as [M, K].
    m = math.prod(XQ.shape[:-1])
    n, k = WQ.shape
    out = gemm_a8w8(
        XQ.view(m, k), WQ, x_scale.reshape(m, 1), w_scale.reshape(1, n), None, dtype, 0
    )
    return out.view(*XQ.shape[:-1], n)


def gemm_triton(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The rowwise GEMM computed by aiter's Triton ``gemm_a8w8``, into
    ``output`` if given."""
    gemm_a8w8 = import_aiter_module("aiter.ops.triton.gemm.basic.gemm_a8w8").gemm_a8w8
    m = math.prod(XQ.shape[:-1])
    n, k = WQ.shape
    if output is None:
        output = XQ.new_empty((*XQ.shape[:-1], n), dtype=dtype)
    gemm_a8w8(
        XQ.view(m, k),
        WQ,
        x_scale.reshape(m, 1),
        w_scale.reshape(1, n),
        None,
        dtype,
        output.view(m, n),
    )
    return output


def gemm_triton_batched(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The batched rowwise GEMM computed by aiter's Triton ``batched_gemm_a8w8``,
    into ``output`` if given."""
    batched_gemm_a8w8 = import_aiter_module(
        "aiter.ops.triton.gemm.batched.batched_gemm_a8w8"
    ).batched_gemm_a8w8
    # The op takes the scales as [B, M] and [B, N]; aiter as [B, M, 1] and
    # [B, 1, N].
    b, m, _ = XQ.shape
    n = WQ.shape[1]
    return batched_gemm_a8w8(
        XQ,
        WQ,
        x_scale.reshape(b, m, 1),
        w_scale.reshape(b, 1, n),
        None,
        torch.bfloat16,
        None,
        output,
    )
