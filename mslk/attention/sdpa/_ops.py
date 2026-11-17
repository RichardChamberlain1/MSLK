# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""SDPA ops matching aten's _scaled_dot_product_{flash,efficient}_attention.

Inputs are [B, H, L, E] (aten convention). Internally the ops convert to
MSLK's [B, L, H, K] layout, run the fmha dispatch, and convert back.

The aten override registers these as ROCm implementations of the two aten
operators, so ``torch.nn.functional.scaled_dot_product_attention`` routes
through MSLK's kernels instead of falling back to the slow math kernel.

LSE convention: aten returns logsumexp in natural log (ln); MSLK's fmha
kernels return log2.  Forward divides by log2(e); backward multiplies back.
"""

import math
from typing import Optional, Tuple

import torch
from mslk.attention import fmha
from mslk.attention.fmha.attn_bias import (
    LowerTriangularFromBottomRightMask,
    LowerTriangularMask,
)

_LOG2E = math.log2(math.e)
_LN2 = math.log(2)

_CAUSAL_SELF = LowerTriangularMask()
_CAUSAL_BR = LowerTriangularFromBottomRightMask()


def _bhld_to_bmhk(t: torch.Tensor) -> torch.Tensor:
    """[B, H, L, D] -> [B, L, H, D] as a view."""
    return t.transpose(1, 2)


def _bmhk_to_bhld(t: torch.Tensor) -> torch.Tensor:
    """[B, L, H, D] -> [B, H, L, D] as a view."""
    return t.transpose(1, 2)


def _causal_bias(is_causal: bool, Lq: int, Lkv: int):
    """Map is_causal to the fmha bias type.

    PyTorch's flash backend uses bottom-right alignment when Lq != Lkv.
    Reuses cached instances to avoid per-call allocation.
    """
    if not is_causal:
        return None
    return _CAUSAL_SELF if Lq == Lkv else _CAUSAL_BR


# ---------------------------------------------------------------------------
# Flash forward / backward
# ---------------------------------------------------------------------------


def sdpa_flash_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[
    torch.Tensor,  # output  [B, H, Lq, D]
    torch.Tensor,  # logsumexp [B, H, Lq] (natural log)
    torch.Tensor,  # cum_seq_q (placeholder)
    torch.Tensor,  # cum_seq_k (placeholder)
    int,  # max_q
    int,  # max_k
    torch.Tensor,  # rng_state (placeholder)
    torch.Tensor,  # unused
    torch.Tensor,  # debug_attn_mask (placeholder)
]:
    if dropout_p > 0.0:
        raise NotImplementedError("MSLK SDPA does not support dropout")

    B, H, Lq, D = query.shape
    _, _, Lkv, _ = key.shape

    Q = _bhld_to_bmhk(query)
    K = _bhld_to_bmhk(key)
    V = _bhld_to_bmhk(value)

    attn_bias = _causal_bias(is_causal, Lq, Lkv)

    out, lse = fmha.memory_efficient_attention_forward_requires_grad(
        Q, K, V, attn_bias=attn_bias, scale=scale
    )

    out_bhld = _bmhk_to_bhld(out)
    lse_nat = (
        lse.mul(_LN2)
        if lse is not None
        else torch.empty(0, device=query.device, dtype=torch.float32)
    )

    _e = torch.empty(0, device=query.device)
    return (
        out_bhld,
        lse_nat,
        _e.to(torch.int32),
        _e.to(torch.int32),
        Lq,
        Lkv,
        _e.to(torch.int64),
        _e.to(query.dtype),
        _e.to(query.dtype),
    )


def sdpa_flash_bwd(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dropout_p > 0.0:
        raise NotImplementedError("MSLK SDPA does not support dropout")

    Q = _bhld_to_bmhk(query)
    K = _bhld_to_bmhk(key)
    V = _bhld_to_bmhk(value)
    dO = _bhld_to_bmhk(grad_out)
    O_bmhk = _bhld_to_bmhk(out)  # noqa: N806

    B, H, Lq, D = query.shape
    _, _, Lkv, _ = key.shape
    attn_bias = _causal_bias(is_causal, Lq, Lkv)

    lse_log2 = logsumexp.mul(_LOG2E)

    dq, dk, dv = fmha.memory_efficient_attention_backward(
        dO, O_bmhk, lse_log2, Q, K, V, attn_bias=attn_bias, scale=scale
    )

    k_is_v = key.data_ptr() == value.data_ptr()
    dQ = _bmhk_to_bhld(dq)
    dK = _bmhk_to_bhld(dk)
    dV = _bmhk_to_bhld(dv)
    if k_is_v:
        dK = dK + dV
        dV = torch.zeros_like(dV)

    return dQ, dK, dV


# ---------------------------------------------------------------------------
# Efficient forward / backward
# ---------------------------------------------------------------------------


def sdpa_efficient_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    compute_log_sumexp: bool,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dropout_p > 0.0:
        raise NotImplementedError("MSLK SDPA does not support dropout")

    B, H, Lq, D = query.shape
    _, _, Lkv, _ = key.shape

    Q = _bhld_to_bmhk(query)
    K = _bhld_to_bmhk(key)
    V = _bhld_to_bmhk(value)

    bias = _causal_bias(is_causal, Lq, Lkv)
    if attn_bias is not None and bias is None:
        bias = attn_bias

    out, lse = fmha.memory_efficient_attention_forward_requires_grad(
        Q, K, V, attn_bias=bias, scale=scale
    )

    out_bhld = _bmhk_to_bhld(out)
    if compute_log_sumexp and lse is not None:
        lse_nat = lse.mul(_LN2)
    else:
        lse_nat = torch.empty(0, device=query.device, dtype=torch.float32)

    _e = torch.empty(0, device=query.device, dtype=torch.int64)
    return out_bhld, lse_nat, _e, _e


def sdpa_efficient_bwd(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    dropout_p: float,
    grad_input_mask: list,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if dropout_p > 0.0:
        raise NotImplementedError("MSLK SDPA does not support dropout")

    Q = _bhld_to_bmhk(query)
    K = _bhld_to_bmhk(key)
    V = _bhld_to_bmhk(value)
    dO = _bhld_to_bmhk(grad_out)
    O_bmhk = _bhld_to_bmhk(out)  # noqa: N806

    B, H, Lq, D = query.shape
    _, _, Lkv, _ = key.shape
    bias = _causal_bias(is_causal, Lq, Lkv)
    if attn_bias is not None and bias is None:
        bias = attn_bias

    lse_log2 = logsumexp.mul(_LOG2E)

    dq, dk, dv = fmha.memory_efficient_attention_backward(
        dO, O_bmhk, lse_log2, Q, K, V, attn_bias=bias, scale=scale
    )

    k_is_v = key.data_ptr() == value.data_ptr()
    dQ = _bmhk_to_bhld(dq)
    dK = _bmhk_to_bhld(dk)
    dV = _bmhk_to_bhld(dv)
    if k_is_v:
        dK = dK + dV
        dV = torch.zeros_like(dV)

    return dQ, dK, dV, None


# ---------------------------------------------------------------------------
# Aten override
# ---------------------------------------------------------------------------

_aten_override_active = False


def enable_aten_override() -> None:
    """Register MSLK's SDPA as the ROCm implementation of aten's SDPA ops.

    After this call, ``torch.nn.functional.scaled_dot_product_attention``
    routes through MSLK's fmha kernels on ROCm.

    Note: ``torch.library`` does not support unregistration, so
    ``disable_aten_override`` cannot undo this within the same process.
    """
    global _aten_override_active
    if _aten_override_active:
        return

    torch.library.impl(
        "aten::_scaled_dot_product_flash_attention",
        "CUDA",
    )(sdpa_flash_fwd)

    torch.library.impl(
        "aten::_scaled_dot_product_flash_attention_backward",
        "CUDA",
    )(sdpa_flash_bwd)

    torch.library.impl(
        "aten::_scaled_dot_product_efficient_attention",
        "CUDA",
    )(sdpa_efficient_fwd)

    torch.library.impl(
        "aten::_scaled_dot_product_efficient_attention_backward",
        "CUDA",
    )(sdpa_efficient_bwd)

    _aten_override_active = True


def disable_aten_override() -> None:
    """Mark the override as inactive.

    ``torch.library`` does not support unregistering implementations, so
    the aten dispatch entries remain.  This flag is provided for forward
    compatibility; a future PyTorch release may add unregistration.
    """
    global _aten_override_active
    _aten_override_active = False
