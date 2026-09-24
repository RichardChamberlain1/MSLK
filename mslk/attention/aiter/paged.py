# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AITER paged-attention decode, wrapped for MSLK.

Thin binding over AITER's paged decode kernels, plus the cache-layout helpers
and the capability gate they need.

Two entry points exist and they are **not** interchangeable:

``forward_decode`` (default)
    ``aiter.paged_attn.PagedAttention.forward_decode`` -> ``paged_attention_rocm``.
    Splits the KV range into ``ceil(seq_len / 256)`` partitions with a combine
    pass, so a single request still fills the device. This is the decode path.

``pa_fwd_asm`` (opt in with ``use_asm=True``)
    The hand-written assembly kernel (``pa_bf16_noquant_gqa8_1tg_4w``). The name
    says it: one thread group, four waves, and **no KV splitting**. It is built
    for concurrency, and at ``requests=1`` it leaves the GPU idle -- measured
    182 us vs 35 us for ``forward_decode`` on ctx=32000/d=128, a 5.1x gap. It
    also refuses ``head_dim=64`` (segfault) and ``q_len>4`` (NaN), neither of
    which constrains ``forward_decode``.

Why this is a module and not an ``fmha.AttentionFwOpBase``
----------------------------------------------------------
The fmha op contract hands the kernel K/V in MSLK's paged pool layout,
``[1, pages * page_size, H_kv, D]``. AITER's asm kernels want a different
physical layout entirely (see below), and at these context lengths converting
per call would move gigabytes and dominate the measurement. A serving stack
would *store* the cache in the backend's native layout, so this module exposes
the conversion as a one-time setup helper and the kernel as a direct call.

Layouts
-------
``k_cache``  ``[pages, H_kv, D // KV_VEC, PAGE_SIZE, KV_VEC]``
``v_cache``  ``[pages, H_kv, PAGE_SIZE // KV_VEC, D, KV_VEC]``  (the asm V shuffle)
``query``    ``[sum(q_len), H_q, D]``  packed, with ``qo_indptr`` when q_len > 1
``block_tables`` ``[requests, pages_per_request]`` int32
``context_lens`` ``[requests]`` int32

Capability envelope
-------------------
Measured on gfx950 with ``amd-aiter 0.1.22``; ``not_supported_reasons`` encodes
it. ``forward_decode`` handles head_dim 64 and 128, q_len 1..16, and contexts to
at least 512k, matching an fp32 streaming-softmax reference to ~2.8e-3 relative
Frobenius error (bf16 rounding) with bottom-right causal masking for q_len > 1.

The asm path is far narrower -- head_dim 128 only and q_len <= 4 -- and those
limits are enforced only when ``use_asm=True``, because outside them it
segfaults or returns NaN rather than raising.

Neither path takes a sliding-window argument, so windowed attention is out of
scope for both.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

# The gfx950 bf16 no-quant asm kernels are built for a 16-token page.
PAGE_SIZE = 16
# K/V are vectorised 16 bytes at a time; 8 elements for a 2-byte dtype.
KV_VEC = 8

# forward_decode (split-KV) envelope, verified against an fp32 reference.
_SUPPORTED_HEAD_DIMS = (64, 128)
_MAX_Q_LEN = 16
# pa_fwd_asm is built per GQA ratio and is much more restrictive.
_ASM_HEAD_DIMS = (128,)
_ASM_GQA_RATIOS = (8, 16)
_ASM_MAX_Q_LEN = 4
# forward_decode's KV partition size; the split count is ceil(seq_len / this).
PARTITION_SIZE = 256


@functools.lru_cache(maxsize=1)
def is_available() -> bool:
    """True when the AITER asm paged kernels can run on this device."""
    try:
        import aiter  # noqa: F401
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        return False
    return arch.startswith("gfx942") or arch.startswith("gfx950")


def not_supported_reasons(
    head_dim: int,
    query_heads: int,
    kv_heads: int,
    q_len: int,
    dtype: torch.dtype,
    window_left: int = -1,
    use_asm: bool = False,
) -> List[str]:
    """Why AITER cannot serve this shape; empty list means it can.

    Refusing up front matters for the asm path in particular: ``head_dim=64``
    takes the process down with a segfault and ``q_len>=5`` returns NaN rather
    than raising, so neither is something a caller could catch.
    """
    reasons: List[str] = []
    if not is_available():
        reasons.append("AITER unavailable on this build/device")
    if dtype is not torch.bfloat16:
        reasons.append(f"dtype {dtype} unsupported (bf16 only)")
    if kv_heads <= 0 or query_heads % kv_heads:
        reasons.append(
            f"query_heads={query_heads} not divisible by kv_heads={kv_heads}"
        )
    if window_left >= 0:
        reasons.append("sliding-window attention unsupported (no window argument)")

    if use_asm:
        if head_dim not in _ASM_HEAD_DIMS:
            reasons.append(
                f"asm path: head_dim={head_dim} unsupported "
                f"(kernels are {_ASM_HEAD_DIMS}; head_dim=64 segfaults)"
            )
        if kv_heads > 0 and query_heads % kv_heads == 0:
            ratio = query_heads // kv_heads
            if ratio not in _ASM_GQA_RATIOS:
                reasons.append(
                    f"asm path: GQA ratio {ratio} unsupported "
                    f"(kernels are {_ASM_GQA_RATIOS})"
                )
        if q_len > _ASM_MAX_Q_LEN:
            reasons.append(
                f"asm path: q_len={q_len} unsupported (>{_ASM_MAX_Q_LEN} returns NaN)"
            )
    else:
        if head_dim not in _SUPPORTED_HEAD_DIMS:
            reasons.append(
                f"head_dim={head_dim} unsupported (supported: {_SUPPORTED_HEAD_DIMS})"
            )
        if q_len > _MAX_Q_LEN:
            reasons.append(f"q_len={q_len} unsupported (verified up to {_MAX_Q_LEN})")
    return reasons


def build_kv_caches(
    k_pages: torch.Tensor, v_pages: torch.Tensor, use_asm: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert page-major ``[pages, H_kv, D, PAGE_SIZE]`` K/V to AITER layout.

    K is always ``[pages, H_kv, D // KV_VEC, PAGE_SIZE, KV_VEC]``. V stays
    page-major for the default decode route and gets the asm shuffle,
    ``[pages, H_kv, PAGE_SIZE // KV_VEC, D, KV_VEC]``, for ``use_asm``.

    Setup-time only -- this materialises a full copy of the cache.
    """
    if k_pages.shape != v_pages.shape or k_pages.ndim != 4:
        raise ValueError(
            "expected matching [pages, H_kv, D, PAGE_SIZE] K/V, got "
            f"{tuple(k_pages.shape)} and {tuple(v_pages.shape)}"
        )
    pages, kv_heads, head_dim, page_size = k_pages.shape
    if page_size != PAGE_SIZE:
        raise ValueError(f"page size must be {PAGE_SIZE}, got {page_size}")
    if head_dim % KV_VEC or page_size % KV_VEC:
        raise ValueError(f"head_dim and page size must be multiples of {KV_VEC}")
    # K: split D into (D/vec, vec) and move the vector axis last, page rows third.
    k_cache = (
        k_pages.view(pages, kv_heads, head_dim // KV_VEC, KV_VEC, page_size)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    if use_asm:
        # The asm shuffle: split the page axis and hoist it ahead of D.
        v_cache = (
            v_pages.view(pages, kv_heads, head_dim, page_size // KV_VEC, KV_VEC)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
    else:
        v_cache = v_pages.contiguous()
    return k_cache, v_cache


@dataclass
class DecodeWorkspace:
    """Preallocated scratch for the split-KV decode kernel.

    ``forward_decode`` allocates these three tensors on every call. Hoisting
    them out is worth 24% at ``requests=1`` (38.1 -> 29.1 us on ctx=32000/d=128,
    bit-identical output) and is free elsewhere -- it is a fixed per-call cost,
    so it only shows up when the kernel itself is short.
    """

    out: torch.Tensor
    exp_sums: torch.Tensor
    max_logits: torch.Tensor
    tmp_out: torch.Tensor
    partition_size: int


def make_decode_workspace(
    query: torch.Tensor,
    max_context_len: int,
    partition_size: int = PARTITION_SIZE,
) -> DecodeWorkspace:
    """Scratch sized for ``query`` ``[rows, H_q, D]`` and this context length."""
    rows, heads, head_dim = query.shape
    partitions = (max_context_len + partition_size - 1) // partition_size
    kw = dict(device=query.device)
    return DecodeWorkspace(
        out=torch.empty_like(query),
        exp_sums=torch.empty(rows, heads, partitions, dtype=torch.float32, **kw),
        max_logits=torch.empty(rows, heads, partitions, dtype=torch.float32, **kw),
        tmp_out=torch.empty(rows, heads, partitions, head_dim, dtype=query.dtype, **kw),
        partition_size=partition_size,
    )


def paged_attention_forward(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_q_len: int = 1,
    qo_indptr: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    max_context_len: Optional[int] = None,
    workspace: Optional[DecodeWorkspace] = None,
    use_asm: bool = False,
) -> torch.Tensor:
    """Run AITER paged attention.

    ``query`` is packed ``[sum(q_len), H_q, D]``; bottom-right causal masking is
    applied within each request's query block when ``max_q_len > 1``.

    Default route is the split-KV decode kernel, which is the one to use unless
    you specifically want the asm kernel's high-concurrency behaviour -- see the
    module docstring for the 5.1x single-request gap between them.

    ``v_cache`` layout differs by route: page-major ``[pages, H_kv, D,
    PAGE_SIZE]`` for the default, and the asm-shuffled variant for ``use_asm``.
    Build both with :func:`build_kv_caches`.

    Pass a ``workspace`` from :func:`make_decode_workspace` to skip the
    per-call scratch allocation; see :class:`DecodeWorkspace`.

    Pass ``max_context_len`` when the call may be captured into a CUDA graph.
    The default route sizes its partition scratch from it, and reading it off
    ``context_lens`` needs a device->host sync, which invalidates capture
    (``hipErrorStreamCaptureInvalidated``).
    """
    import aiter
    from aiter import paged_attn

    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    if use_asm:
        # Takes no scale argument -- it hardcodes 1/sqrt(head_dim).
        return aiter.pa_fwd_asm(
            query,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            block_tables.stride(0),
            max_q_len,
            None,  # K_QScale
            None,  # V_QScale
            out,
            qo_indptr,
        )

    if max_context_len is None:
        # Graph-unsafe: syncs. Callers that capture must pass it explicitly.
        max_context_len = int(context_lens.max().item())
    ones = torch.ones(1, dtype=torch.float32, device=query.device)
    if workspace is not None:
        # Same kernel forward_decode calls, without re-allocating its scratch.
        torch.ops.aiter.paged_attention_rocm(
            workspace.out,
            workspace.exp_sums,
            workspace.max_logits,
            workspace.tmp_out,
            query,
            k_cache,
            v_cache,
            k_cache.shape[1],
            scale,
            block_tables,
            context_lens,
            k_cache.shape[3],
            max_context_len,
            None,  # alibi_slopes
            "auto",
            ones,
            ones,
            None,  # fp8_out_scale
            workspace.partition_size,
            mtp=max_q_len,
            q_scale=None,
        )
        return workspace.out
    return paged_attn.PagedAttention.forward_decode(
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        max_context_len,
        "auto",
        k_cache.shape[1],  # num_kv_heads
        scale,
        None,  # alibi_slopes
        ones,  # k_scale
        ones,  # v_scale
        mtp=max_q_len,
        output_dtype=query.dtype,
    )
