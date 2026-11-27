# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MSLK's backend of the HSTU ops: the pure-PyTorch reference
(:mod:`mslk.attention.hstu._torch_ref`).

Functions take the op's arguments in the routing convention (``fn(q, **args)``).
Backwards rerun the reference forward under autograd, so ``M``, ``Out`` and
``Delta`` are ignored. The reference reads offsets on the host, so it cannot be
captured in a CUDA graph.
"""

import warnings
from typing import Any, List, Optional, Set, Tuple

import torch
from mslk.attention.hstu import _torch_ref as ref

# Inputs with at least this many tokens warn once per op.
LARGE_INPUT_TOKENS: int = 1 << 16

_warned: Set[str] = set()


def _warn_if_large(op: str, tokens: int) -> None:
    if tokens >= LARGE_INPUT_TOKENS and op not in _warned:
        _warned.add(op)
        warnings.warn(
            f"mslk::{op} ({tokens} tokens) runs on the pure-PyTorch HSTU "
            "reference: slow, and not CUDA-graph capturable.",
            stacklevel=3,
        )


def _full_lse(M: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The op's ``M`` is ``[T, H]``; the reference's covers softmax heads only."""
    if M.shape[1] == q.shape[1]:
        return M.contiguous()
    return M.new_zeros((q.shape[0], q.shape[1]))


def _leaf(t: torch.Tensor) -> torch.Tensor:
    return t.detach().requires_grad_(True)


def _grads(
    out: torch.Tensor, inputs: List[torch.Tensor], dout: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    return torch.autograd.grad(out, inputs, dout, allow_unused=True)


def _copy_or_zero(dst: torch.Tensor, src: Optional[torch.Tensor]) -> None:
    if src is None:
        dst.zero_()
    else:
        dst.copy_(src)


# ── Ragged ──────────────────────────────────────────────────────────────


def _ragged_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    N: int,
    alpha: float,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    full_attn_size: int,
    num_softmax_heads: int,
    shared_kv: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return ref.ragged_hstu_attention_ref(
        N,
        alpha,
        q,
        k,
        k if shared_kv else v,
        seq_offsets,
        invalid_attn_mask_type,
        num_targets,
        attn_scale,
        max_attn_len,
        contextual_seq_len,
        full_attn_size,
        num_softmax_heads,
        shared_kv,
    )


def ragged_fwd(
    q: torch.Tensor, *, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    _warn_if_large("hstu_ragged_fwd", q.shape[0])
    with torch.no_grad():
        out, M = _ragged_ref(q, k, v, **kwargs)
    return out, _full_lse(M, q)


def ragged_bwd(
    q: torch.Tensor,
    *,
    dout: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: Optional[torch.Tensor],
    sort_by_length_indices: Optional[torch.Tensor] = None,
    M: Optional[torch.Tensor] = None,
    Out: Optional[torch.Tensor] = None,
    shared_kv: bool = False,
    **kwargs: Any,
) -> None:
    _warn_if_large("hstu_ragged_bwd", q.shape[0])
    q_, k_ = _leaf(q), _leaf(k)
    # Shared K/V: the gradient of k is dK + dV.
    v_ = k_ if shared_kv else _leaf(v)
    with torch.enable_grad():
        out, _ = _ragged_ref(q_, k_, v_, shared_kv=shared_kv, **kwargs)
        inputs = [q_, k_] if shared_kv else [q_, k_, v_]
        grads = _grads(out, inputs, dout)
    _copy_or_zero(dq, grads[0])
    _copy_or_zero(dk, grads[1])
    if dv is not None:
        _copy_or_zero(dv, grads[2])


# ── Cross ───────────────────────────────────────────────────────────────


def _cross_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    max_seq_len: int,
    alpha: float,
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    max_q_len: int,
    attn_scale: Optional[torch.Tensor],
    G: int,
    shared_kv: bool,
    num_softmax_heads: int,
    truncate_method: str,
    num_targets: Optional[torch.Tensor],
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return ref.cross_hstu_attention_ref(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        seq_offsets_q,
        max_q_len,
        attn_scale,
        G,
        shared_kv,
        num_softmax_heads,
        truncate_method,
        num_targets,
        causal,
    )


def cross_fwd(
    q: torch.Tensor,
    *,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    shared_kv: bool = False,
    enable_tma: bool = True,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _warn_if_large("hstu_cross_fwd", k.shape[0])
    value = k if (shared_kv or v is None) else v
    with torch.no_grad():
        out, M = _cross_ref(q, k, value, shared_kv=shared_kv, **kwargs)
    return out, _full_lse(M, q)


def _query_lengths_max(seq_offsets_q: torch.Tensor) -> int:
    return int((seq_offsets_q[1:] - seq_offsets_q[:-1]).max().item())


def cross_bwd(
    q: torch.Tensor,
    *,
    dout: torch.Tensor,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    max_q_len: Optional[int],
    seq_offsets_q: Optional[torch.Tensor],
    shared_kv: bool,
    M: Optional[torch.Tensor] = None,
    Delta: Optional[torch.Tensor] = None,
    stride_mm: int = 0,
    enable_tma: bool = True,
    Out: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> List[torch.Tensor]:
    _warn_if_large("hstu_cross_bwd", k.shape[0])
    if seq_offsets_q is None:
        raise ValueError(
            "MSLK's cross attention backward needs seq_offsets_q (the query offsets)"
        )
    if max_q_len is None:
        max_q_len = _query_lengths_max(seq_offsets_q)
    q_, k_ = _leaf(q), _leaf(k)
    shared = shared_kv or v is None
    v_ = k_ if shared else _leaf(v)
    with torch.enable_grad():
        out, _ = _cross_ref(
            q_,
            k_,
            v_,
            max_q_len=max_q_len,
            seq_offsets_q=seq_offsets_q,
            shared_kv=shared,
            **kwargs,
        )
        inputs = [q_, k_] if shared else [q_, k_, v_]
        grads = _grads(out, inputs, dout)
    result = [
        g if g is not None else torch.zeros_like(t) for g, t in zip(grads, inputs)
    ]
    return [g.to(t.dtype) for g, t in zip(result, inputs)]


# ── Grouped ─────────────────────────────────────────────────────────────


def _grouped_scale(
    attn_scale_tensor: Optional[torch.Tensor], num_groups: int
) -> Optional[torch.Tensor]:
    """Op scale ``[G]`` or ``[T, 1]`` -> reference scale ``[G]`` or ``[T]``."""
    if attn_scale_tensor is None or attn_scale_tensor.dim() == 1:
        return attn_scale_tensor
    flat = attn_scale_tensor.reshape(attn_scale_tensor.shape[0], -1)
    if flat.shape[1] != 1:
        raise NotImplementedError(
            "MSLK's grouped HSTU attention takes one scale per query token; got "
            f"attn_scale_tensor of shape {tuple(attn_scale_tensor.shape)}"
        )
    flat = flat[:, 0]
    if flat.shape[0] == num_groups:
        raise NotImplementedError(
            "MSLK's grouped HSTU attention cannot tell a per-token scale from a "
            "per-group one when the token count equals the group count"
        )
    return flat


def _grouped_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    Z: int,
    seq_offsets: torch.Tensor,
    group_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_seq_len_tensor: torch.Tensor,
    max_attn_len_tensor: torch.Tensor,
    full_attn_size_tensor: torch.Tensor,
    contextual_seq_len_tensor: torch.Tensor,
    alpha: float,
    invalid_attn_mask_type: str,
    attn_scale_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return ref.grouped_ragged_hstu_attention_ref(
        Z,
        q,
        k,
        v,
        seq_offsets,
        group_offsets,
        num_targets,
        max_seq_len_tensor,
        max_attn_len_tensor,
        full_attn_size_tensor,
        contextual_seq_len_tensor,
        alpha,
        invalid_attn_mask_type,
        _grouped_scale(attn_scale_tensor, max_seq_len_tensor.numel()),
    )


def grouped_fwd(
    q: torch.Tensor, *, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
) -> torch.Tensor:
    _warn_if_large("hstu_grouped_fwd", q.shape[0])
    with torch.no_grad():
        return _grouped_ref(q, k, v, **kwargs)


def grouped_bwd(
    q: torch.Tensor,
    *,
    dout: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: Optional[torch.Tensor],
    Out: Optional[torch.Tensor] = None,
    shared_kv: bool = False,
    **kwargs: Any,
) -> None:
    _warn_if_large("hstu_grouped_bwd", q.shape[0])
    q_, k_ = _leaf(q), _leaf(k)
    # Shared K/V: pass K as V; the gradient of k is dK + dV.
    v_ = k_ if shared_kv else _leaf(v)
    with torch.enable_grad():
        out = _grouped_ref(q_, k_, v_, **kwargs)
        inputs = [q_, k_] if shared_kv else [q_, k_, v_]
        grads = _grads(out, inputs, dout)
    _copy_or_zero(dq, grads[0])
    _copy_or_zero(dk, grads[1])
    if dv is not None:
        _copy_or_zero(dv, grads[2])
