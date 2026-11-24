# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Addressing of the FlyDSL gfx950 backward: inputs of 2^31 or more elements,
empty sequences, and the sizes it must decline.

The kernel addresses each sequence from a 64-bit base, so only in-sequence
offsets are 32-bit. A sequence's gradients depend only on that sequence, so a
few sequences (first, the one crossing 2^31 K/V elements, last) are checked
against an fp32 PyTorch reference. Large cases skip without enough GPU memory.
"""

import itertools
import math
from typing import List, Optional

import pytest
import torch
from mslk.attention import fmha
from mslk.attention.fmha import flydsl
from mslk.attention.fmha.attn_bias import BlockDiagonalMask
from mslk.utils.device import is_gfx950

OPS = (flydsl.FwOp, flydsl.BwOp)
LEN_Q, LEN_KV = 128, 65536
BF16 = dict(device="cuda", dtype=torch.bfloat16)


def _require_gfx950() -> None:
    if not torch.version.hip or not is_gfx950():
        pytest.skip("requires gfx950")
    assert flydsl._is_flydsl_available(), "FlyDSL is unavailable on gfx950"


def _reference_grads(q, k, v, dout):
    """fp32 attention gradients of one sequence; k/v may have fewer heads."""
    q, k, v = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    group = q.shape[1] // k.shape[1]
    kh, vh = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
    scores = torch.einsum("qhd,khd->hqk", q, kh) / math.sqrt(q.shape[-1])
    out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vh)
    return torch.autograd.grad(out, (q, k, v), dout.float())


def _assert_close(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    err = ((got.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    assert err < 2e-2, f"{name}: relative L2 error {err:.3e}"


def _run(
    n: int,
    heads: int = 1,
    kv_heads: int = 1,
    head_dim: int = 128,
    varlen: bool = True,
    deterministic: bool = False,
) -> None:
    kv_elements = n * LEN_KV * kv_heads * head_dim
    # K, V, dK, dV in bf16, plus fp32 dK/dV per query head.
    need = kv_elements * 8 + n * LEN_KV * heads * head_dim * 8
    torch.cuda.empty_cache()  # earlier cases' cached blocks count as used
    if need * 1.25 > torch.cuda.mem_get_info()[0]:
        pytest.skip("not enough GPU memory")
    torch.manual_seed(0)
    if varlen:
        bias: Optional[BlockDiagonalMask] = BlockDiagonalMask.from_seqlens(
            q_seqlen=[LEN_Q] * n, kv_seqlen=[LEN_KV] * n, device="cuda"
        )
        q_shape, kv_shape = (1, n * LEN_Q), (1, n * LEN_KV)
    else:
        bias = None
        q_shape, kv_shape = (n, LEN_Q), (n, LEN_KV)
    q = torch.randn(*q_shape, heads, head_dim, **BF16).requires_grad_(True)
    k0 = torch.randn(*kv_shape, kv_heads, head_dim, **BF16).requires_grad_(True)
    v0 = torch.randn(*kv_shape, kv_heads, head_dim, **BF16).requires_grad_(True)
    group = heads // kv_heads
    k, v = k0, v0
    if group > 1:  # GQA through stride-0 heads, as callers pass it
        k = k0.unsqueeze(3).expand(*kv_shape, kv_heads, group, head_dim).flatten(2, 3)
        v = v0.unsqueeze(3).expand(*kv_shape, kv_heads, group, head_dim).flatten(2, 3)
    out = fmha.memory_efficient_attention(q, k, v, attn_bias=bias, op=OPS)
    dout = torch.randn_like(out)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        dq, dk, dv = torch.autograd.grad(out, (q, k0, v0), dout)
    finally:
        torch.use_deterministic_algorithms(previous)

    crossing = (1 << 31) // (LEN_KV * kv_heads * head_dim)  # first past 2^31
    assert crossing < n, "the case must cross 2^31 K/V elements"
    for i in sorted({0, crossing, n - 1}):
        if varlen:
            sq = (0, slice(i * LEN_Q, (i + 1) * LEN_Q))
            skv = (0, slice(i * LEN_KV, (i + 1) * LEN_KV))
        else:
            sq = skv = (i, slice(None))
        refs = _reference_grads(q[sq], k0[skv], v0[skv], dout[sq])
        for name, got, ref in zip(("dq", "dk", "dv"), (dq[sq], dk[skv], dv[skv]), refs):
            _assert_close(f"sequence {i}/{n} {name}", got, ref)


def _n_for(multiple: float, kv_heads: int = 1, head_dim: int = 128) -> int:
    return math.ceil(multiple * (1 << 31) / (LEN_KV * kv_heads * head_dim))


@pytest.mark.parametrize("multiple", [1.02, 2, 4])
def test_varlen_backward_beyond_2_31_kv_elements(multiple: float) -> None:
    _require_gfx950()
    _run(_n_for(multiple))


def test_deterministic_backward_beyond_2_31_kv_elements() -> None:
    _require_gfx950()
    _run(_n_for(1.02), deterministic=True)


def test_gqa_backward_beyond_2_31_kv_elements() -> None:
    _require_gfx950()
    _run(_n_for(1.02), heads=4, kv_heads=1)


def test_dense_backward_beyond_2_31_kv_elements() -> None:
    _require_gfx950()
    _run(_n_for(1.02), varlen=False)


def test_padded_head_dim_backward_beyond_2_31_kv_elements() -> None:
    _require_gfx950()
    _run(_n_for(1.02, head_dim=96), head_dim=96)  # computed at head dim 128


@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [
        ([64, 64, 64], [0, 128, 128]),
        ([64, 64, 64], [128, 0, 128]),
        ([64, 64, 64], [128, 128, 0]),
        ([64, 0, 64], [128, 128, 128]),
        ([64, 64, 0], [128, 128, 128]),
    ],
)
def test_varlen_backward_empty_sequences(q_lens: List[int], kv_lens: List[int]) -> None:
    _require_gfx950()
    torch.manual_seed(0)
    bias = BlockDiagonalMask.from_seqlens(
        q_seqlen=q_lens, kv_seqlen=kv_lens, device="cuda"
    )
    q = torch.randn(1, sum(q_lens), 2, 128, **BF16).requires_grad_(True)
    k = torch.randn(1, sum(kv_lens), 2, 128, **BF16).requires_grad_(True)
    v = torch.randn(1, sum(kv_lens), 2, 128, **BF16).requires_grad_(True)
    out = fmha.memory_efficient_attention(q, k, v, attn_bias=bias, op=OPS)
    dout = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    torch.cuda.synchronize()
    qs = [0] + list(itertools.accumulate(q_lens))
    ks = [0] + list(itertools.accumulate(kv_lens))
    for i in range(len(q_lens)):
        sq, skv = slice(qs[i], qs[i + 1]), slice(ks[i], ks[i + 1])
        if q_lens[i] == 0 or kv_lens[i] == 0:
            # No query attends to this sequence's keys, or no key exists.
            for t, s in ((dq, sq), (dk, skv), (dv, skv)):
                assert not t[0, s].any(), f"sequence {i}: expected zero gradients"
            continue
        refs = _reference_grads(q[0, sq], k[0, skv], v[0, skv], dout[0, sq])
        for name, got, ref in zip(
            ("dq", "dk", "dv"), (dq[0, sq], dk[0, skv], dv[0, skv]), refs
        ):
            _assert_close(f"sequence {i} {name}", got, ref)


def test_declines_one_sequence_whose_fp32_gradients_exceed_4_gib() -> None:
    # GQA 32/1: the inputs are small, but the per-query-head fp32 dK/dV of the
    # single sequence is 8 GiB, which the kernel cannot address. Dispatch must
    # skip the op, and a forced op must raise rather than return wrong gradients.
    _require_gfx950()
    n_kv, heads, d = 1 << 19, 32, 128
    q = torch.randn(1, 128, heads, d, **BF16).requires_grad_(True)
    k0 = torch.randn(1, n_kv, 1, d, **BF16).requires_grad_(True)
    v0 = torch.randn(1, n_kv, 1, d, **BF16).requires_grad_(True)
    k, v = k0.expand(1, n_kv, heads, d), v0.expand(1, n_kv, heads, d)
    reasons = flydsl.BwOp.not_supported_reasons(
        fmha.common.Inputs(query=q, key=k, value=v)
    )
    assert any("exceed 4 GiB" in r for r in reasons), reasons
    out = fmha.memory_efficient_attention(q, k, v, op=OPS)
    with pytest.raises(NotImplementedError, match="exceed 4 GiB"):
        torch.autograd.grad(out, (q, k0, v0), torch.randn_like(out))


def test_dq_convert_handles_a_full_chunk() -> None:
    _require_gfx950()
    import flydsl.compiler as flyc
    from mslk.attention.flydsl.fmha_bwd_convert_dq import compile_fmha_bwd_convert_dq

    n = flydsl._CONVERT_DQ_CHUNK
    if n * 6 * 1.25 > torch.cuda.mem_get_info()[0]:
        pytest.skip("not enough GPU memory")
    src = torch.ones(n, 1, device="cuda", dtype=torch.float32)
    dst = torch.zeros(n, 1, **BF16)
    stream = torch.cuda.current_stream()
    run = flyc.compile(
        compile_fmha_bwd_convert_dq(dtype_str="bf16"), src, dst, n, stream
    )
    run(src, dst, n, stream)
    torch.cuda.synchronize()
    assert bool((dst == 1).all()), "every element of a full chunk must be converted"


def test_dq_convert_in_chunks_matches_one_launch(monkeypatch) -> None:
    _require_gfx950()
    torch.manual_seed(0)
    bias = BlockDiagonalMask.from_seqlens(
        q_seqlen=[96, 33, 160], kv_seqlen=[256, 70, 512], device="cuda"
    )
    q = torch.randn(1, 289, 2, 128, **BF16).requires_grad_(True)
    k = torch.randn(1, 838, 2, 128, **BF16).requires_grad_(True)
    v = torch.randn(1, 838, 2, 128, **BF16).requires_grad_(True)
    out = fmha.memory_efficient_attention(q, k, v, attn_bias=bias, op=OPS)
    dout = torch.randn_like(out)
    torch.use_deterministic_algorithms(True)
    try:
        whole = torch.autograd.grad(out, (q,), dout, retain_graph=True)[0]
        monkeypatch.setattr(flydsl, "_CONVERT_DQ_CHUNK", 1000)  # not a divisor
        chunked = torch.autograd.grad(out, (q,), dout)[0]
    finally:
        torch.use_deterministic_algorithms(False)
    assert torch.equal(whole, chunked)
