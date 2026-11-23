# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""aiter backends of the HSTU ops on ROCm.

Functions take the op's arguments in the routing convention (``fn(q, **args)``)
and assume the dispatch module's ``accepts`` predicate took the call.

- Ragged: aiter's Triton and FlyDSL HSTU kernels (SiLU heads). Shared K/V passes
  K as V and sums dK + dV; ``M`` is zeros.
- Cross: ``flash_attn_varlen_func``. ``M`` is its LSE in base 2; the backward
  needs ``Out``.
"""

import math
from typing import Any, List, Optional, Tuple

import torch
from mslk.aiter.common import import_aiter_module

_LN2: float = math.log(2.0)


def _triton_hstu() -> Any:
    return import_aiter_module("aiter.ops.triton.attention.hstu_attention")


def _flydsl_hstu() -> Any:
    return import_aiter_module("aiter.ops.flydsl.hstu_attention")


def _mha() -> Any:
    return import_aiter_module("aiter.ops.mha")


def _silu_lse(q: torch.Tensor) -> torch.Tensor:
    return q.new_zeros((q.shape[0], q.shape[1]), dtype=torch.float32)


def _write(dst: torch.Tensor, src: torch.Tensor) -> None:
    if src is not dst:
        dst.copy_(src)


# ── Ragged: aiter's Triton HSTU kernels ─────────────────────────────────


def ragged_fwd_triton(
    q: torch.Tensor,
    *,
    N: int,
    alpha: float,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    shared_kv: bool = False,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = _triton_hstu().triton_hstu_attention_fwd(
        N,
        alpha,
        q,
        k,
        k if shared_kv else v,
        seq_offsets,
        True,  # causal
        num_targets,
        max_attn_len,
        contextual_seq_len,
        None,  # sort_by_length_indices
    )
    return out, _silu_lse(q)


def ragged_bwd_triton(
    q: torch.Tensor,
    *,
    dout: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    N: int,
    alpha: float,
    max_attn_len: int,
    contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor],
    shared_kv: bool = False,
    **_: Any,
) -> None:
    value = k if shared_kv else v
    dv_buf = torch.empty_like(k) if shared_kv else dv
    assert dv_buf is not None
    gq, gk, gv = _triton_hstu().triton_hstu_attention_bwd(
        dout,
        q,
        k,
        value,
        dq,
        dk,
        dv_buf,
        seq_offsets,
        num_targets,
        N,
        alpha,
        max_attn_len,
        True,  # causal
        contextual_seq_len,
        sort_by_length_indices,
    )
    _write(dq, gq)
    _write(dk, gk)
    if shared_kv:
        dk.add_(gv)
    else:
        _write(dv_buf, gv)


# ── Ragged: aiter's FlyDSL HSTU kernels ─────────────────────────────────
#
# aiter caches compiled FlyDSL kernels without the offsets dtype, so mixing
# int32 and int64 callers reuses the wrong kernel. Always pass int32.


def _int32(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return t if t is None or t.dtype == torch.int32 else t.to(torch.int32)


def ragged_fwd_flydsl(
    q: torch.Tensor,
    *,
    N: int,
    alpha: float,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    shared_kv: bool = False,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = _flydsl_hstu().flydsl_hstu_attention_fwd(
        N,
        alpha,
        q,
        k,
        k if shared_kv else v,
        _int32(seq_offsets),
        True,  # causal
        _int32(num_targets),
        max_attn_len,
        contextual_seq_len,
    )
    return out, _silu_lse(q)


def ragged_bwd_flydsl(
    q: torch.Tensor,
    *,
    dout: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    N: int,
    alpha: float,
    max_attn_len: int,
    contextual_seq_len: int,
    shared_kv: bool = False,
    **_: Any,
) -> None:
    # aiter sorts by length itself; sort_by_length_indices is not needed.
    gq, gk, gv = _flydsl_hstu().flydsl_hstu_attention_bwd(
        N,
        alpha,
        q,
        k,
        k if shared_kv else v,
        dout,
        _int32(seq_offsets),
        True,  # causal
        _int32(num_targets),
        max_attn_len,
        contextual_seq_len,
    )
    dq.copy_(gq)
    if shared_kv:
        torch.add(gk, gv, out=dk)
    else:
        assert dv is not None
        dk.copy_(gk)
        dv.copy_(gv)


# ── Cross: aiter's flash_attn_varlen_func ───────────────────────────────


def _cu_seqlens(offsets: torch.Tensor) -> torch.Tensor:
    return offsets if offsets.dtype == torch.int32 else offsets.to(torch.int32)


def cross_fwd(
    q: torch.Tensor,
    *,
    max_seq_len: int,
    alpha: float,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    max_q_len: int,
    shared_kv: bool = False,
    causal: bool = False,
    **_: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    value = k if (shared_kv or v is None) else v
    out, lse = _mha().flash_attn_varlen_func(
        q,
        k,
        value,
        _cu_seqlens(seq_offsets_q),
        _cu_seqlens(seq_offsets),
        max_q_len,
        max_seq_len,
        softmax_scale=alpha,
        causal=causal,
        return_lse=True,
    )
    # LSE [H, T_q] -> base-2 M [T_q, H].
    M = (lse[:, : q.shape[0]].transpose(0, 1) / _LN2).contiguous()
    return out, M


def cross_bwd(
    q: torch.Tensor,
    *,
    dout: torch.Tensor,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    max_seq_len: int,
    alpha: float,
    max_q_len: Optional[int],
    seq_offsets_q: Optional[torch.Tensor],
    causal: bool,
    shared_kv: bool,
    M: Optional[torch.Tensor] = None,
    Out: Optional[torch.Tensor] = None,
    **_: Any,
) -> List[torch.Tensor]:
    assert M is not None and Out is not None
    assert max_q_len is not None and seq_offsets_q is not None
    shared = shared_kv or v is None
    value = k if shared else v
    assert value is not None
    lse = (M.transpose(0, 1) * _LN2).contiguous()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(value)
    _mha()._flash_attn_varlen_backward(
        dout.contiguous(),
        q,
        k,
        value,
        Out.contiguous(),
        lse,
        dq,
        dk,
        dv,
        _cu_seqlens(seq_offsets_q),
        _cu_seqlens(seq_offsets),
        max_q_len,
        max_seq_len,
        0.0,  # dropout_p
        alpha,
        causal,
        -1,  # window_size_left
        -1,  # window_size_right
        None,  # alibi_slopes
        False,  # deterministic
    )
    if shared:
        return [dq, dk.add_(dv)]
    return [dq, dk, dv]
