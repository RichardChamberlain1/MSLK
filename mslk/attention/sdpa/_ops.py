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
"""

from typing import Optional, Tuple

import torch


def _bhld_to_bmhk(t: torch.Tensor) -> torch.Tensor:
    return t.transpose(1, 2)


def _bmhk_to_bhld(t: torch.Tensor) -> torch.Tensor:
    return t.transpose(1, 2)


def _fmha():
    """Lazy import to avoid pulling in FlyDSL at module load."""
    from mslk.attention import fmha

    return fmha


def _flash_causal_bias(is_causal: bool, Lq: int, Lkv: int):
    """Flash path: bottom-right for Lq != Lkv (aten convention)."""
    if not is_causal:
        return None
    from mslk.attention.fmha.attn_bias import (
        LowerTriangularFromBottomRightMask,
        LowerTriangularMask,
    )

    return (
        LowerTriangularMask()
        if Lq == Lkv
        else LowerTriangularFromBottomRightMask()
    )


def _efficient_causal_bias(is_causal: bool, Lq: int, Lkv: int):
    """Efficient path: top-left alignment (aten convention)."""
    if not is_causal:
        return None
    from mslk.attention.fmha.attn_bias import LowerTriangularMask

    return LowerTriangularMask()


def _validate_4d(name: str, t: torch.Tensor) -> None:
    if t.ndim != 4:
        raise ValueError(
            f"MSLK SDPA {name}: expected 4-D [B,H,L,E], got {t.ndim}-D"
        )


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
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    int,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _validate_4d("flash_fwd query", query)

    if dropout_p > 0.0:
        return _native_flash_fwd(
            query,
            key,
            value,
            dropout_p,
            is_causal,
            return_debug_mask,
            scale=scale,
        )

    B, H, Lq, D = query.shape
    _, _, Lkv, _ = key.shape

    Q = _bhld_to_bmhk(query)
    K = _bhld_to_bmhk(key)
    V = _bhld_to_bmhk(value)

    attn_bias = _flash_causal_bias(is_causal, Lq, Lkv)

    fmha = _fmha()
    out, lse = fmha.memory_efficient_attention_forward_requires_grad(
        Q, K, V, attn_bias=attn_bias, scale=scale
    )

    out_bhld = _bmhk_to_bhld(out)
    if lse is None:
        lse = torch.empty(
            B, H, 0, device=query.device, dtype=torch.float32
        )

    dev = query.device
    return (
        out_bhld,
        lse,
        None,  # cum_seq_q
        None,  # cum_seq_k
        Lq,
        Lkv,
        torch.tensor(0, dtype=torch.int64, device=dev),
        torch.tensor(0, dtype=torch.int64, device=dev),
        torch.empty(0, device=dev, dtype=query.dtype),
    )


def sdpa_flash_bwd(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: Optional[torch.Tensor],
    cum_seq_k: Optional[torch.Tensor],
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
        return _native_flash_bwd(
            grad_out,
            query,
            key,
            value,
            out,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            philox_seed,
            philox_offset,
            scale=scale,
        )

    B, H, Lq, D = query.shape
    _, _, Lkv, _ = key.shape
    attn_bias = _flash_causal_bias(is_causal, Lq, Lkv)

    fmha = _fmha()
    dq, dk, dv = fmha.memory_efficient_attention_backward(
        _bhld_to_bmhk(grad_out),
        _bhld_to_bmhk(out),
        logsumexp,
        _bhld_to_bmhk(query),
        _bhld_to_bmhk(key),
        _bhld_to_bmhk(value),
        attn_bias=attn_bias,
        scale=scale,
    )

    return _bmhk_to_bhld(dq), _bmhk_to_bhld(dk), _bmhk_to_bhld(dv)


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
    _validate_4d("efficient_fwd query", query)

    if dropout_p > 0.0:
        return _native_efficient_fwd(
            query,
            key,
            value,
            attn_bias,
            compute_log_sumexp,
            dropout_p,
            is_causal,
            scale=scale,
        )

    if is_causal and attn_bias is not None:
        raise NotImplementedError(
            "MSLK SDPA: is_causal=True with attn_bias is not supported"
        )

    B, H, Lq, D = query.shape
    _, _, Lkv, _ = key.shape

    Q = _bhld_to_bmhk(query)
    K = _bhld_to_bmhk(key)
    V = _bhld_to_bmhk(value)

    bias = _efficient_causal_bias(is_causal, Lq, Lkv)
    if attn_bias is not None and bias is None:
        bias = attn_bias

    fmha = _fmha()
    out, lse = fmha.memory_efficient_attention_forward_requires_grad(
        Q, K, V, attn_bias=bias, scale=scale
    )

    out_bhld = _bmhk_to_bhld(out)
    if not compute_log_sumexp or lse is None:
        lse = torch.empty(
            B, H, 0, device=query.device, dtype=torch.float32
        )

    dev = query.device
    return (
        out_bhld,
        lse,
        torch.tensor(0, dtype=torch.int64, device=dev),
        torch.tensor(0, dtype=torch.int64, device=dev),
    )


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
        return _native_efficient_bwd(
            grad_out,
            query,
            key,
            value,
            attn_bias,
            out,
            logsumexp,
            philox_seed,
            philox_offset,
            dropout_p,
            grad_input_mask,
            is_causal,
            scale=scale,
        )

    if is_causal and attn_bias is not None:
        raise NotImplementedError(
            "MSLK SDPA: is_causal=True with attn_bias is not supported"
        )

    if grad_input_mask[3]:
        raise NotImplementedError("MSLK SDPA: bias gradient not supported")

    B, H, Lq, D = query.shape
    _, _, Lkv, _ = key.shape
    bias = _efficient_causal_bias(is_causal, Lq, Lkv)
    if attn_bias is not None and bias is None:
        bias = attn_bias

    fmha = _fmha()
    dq, dk, dv = fmha.memory_efficient_attention_backward(
        _bhld_to_bmhk(grad_out),
        _bhld_to_bmhk(out),
        logsumexp,
        _bhld_to_bmhk(query),
        _bhld_to_bmhk(key),
        _bhld_to_bmhk(value),
        attn_bias=bias,
        scale=scale,
    )

    return _bmhk_to_bhld(dq), _bmhk_to_bhld(dk), _bmhk_to_bhld(dv), None


# ---------------------------------------------------------------------------
# Native fallbacks (dropout, unsupported configs)
# ---------------------------------------------------------------------------


def _native_flash_fwd(*args, **kwargs):
    return torch.ops.aten._scaled_dot_product_flash_attention(
        *args, **kwargs
    )


def _native_flash_bwd(*args, **kwargs):
    return torch.ops.aten._scaled_dot_product_flash_attention_backward(
        *args, **kwargs
    )


def _native_efficient_fwd(*args, **kwargs):
    return torch.ops.aten._scaled_dot_product_efficient_attention(
        *args, **kwargs
    )


def _native_efficient_bwd(*args, **kwargs):
    return torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        *args, **kwargs
    )


# ---------------------------------------------------------------------------
# Aten override
# ---------------------------------------------------------------------------

_aten_lib: Optional["torch.library.Library"] = None


def enable_aten_override() -> None:
    """Register MSLK's SDPA as the ROCm implementation of aten's SDPA ops.

    Only registers on ROCm (``torch.version.hip``). After this call,
    ``torch.nn.functional.scaled_dot_product_attention`` routes through
    MSLK's fmha kernels. Dropout falls back to the native kernel.
    """
    global _aten_lib
    if _aten_lib is not None:
        return

    if not getattr(torch.version, "hip", None):
        return

    lib = torch.library.Library("aten", "IMPL")
    lib.impl(
        "_scaled_dot_product_flash_attention", sdpa_flash_fwd, "CUDA"
    )
    lib.impl(
        "_scaled_dot_product_flash_attention_backward",
        sdpa_flash_bwd,
        "CUDA",
    )
    lib.impl(
        "_scaled_dot_product_efficient_attention",
        sdpa_efficient_fwd,
        "CUDA",
    )
    lib.impl(
        "_scaled_dot_product_efficient_attention_backward",
        sdpa_efficient_bwd,
        "CUDA",
    )
    _aten_lib = lib


def disable_aten_override() -> None:
    """Unregister MSLK's SDPA aten override, restoring native kernels."""
    global _aten_lib
    if _aten_lib is None:
        return
    _aten_lib._destroy()
    _aten_lib = None
