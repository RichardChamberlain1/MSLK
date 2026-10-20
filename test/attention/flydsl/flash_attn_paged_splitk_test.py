# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Automatic paged split-K selection.

Decode shapes (short Q against a long paged cache) walk the whole context as
one serial chain per workgroup. ``_paged_num_kv_splits`` recovers that
parallelism along KV. These tests pin the two properties that matter:
splitting must not change the result, and the split count must stay within
the chain, grid and workspace bounds.
"""

import math

import pytest
import torch
from mslk.attention.flydsl import flash_attn_interface as fai
from mslk.attention.flydsl.flash_attn_interface import (
    _paged_num_kv_splits,
    flydsl_flash_attn_func,
)
from mslk.flydsl.common import is_flydsl_available

PAGE = 64  # _PAGED_PAGE_SIZE: the only page size the native paged kernel takes
H, HKV = 32, 4


def test_target_requires_gfx950_and_flydsl():
    assert torch.version.hip, "requires a ROCm build"
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    assert arch == "gfx950", f"requires gfx950, got {arch}"
    assert is_flydsl_available(), "requires FlyDSL"


def _paged_inputs(B, Sq, ctx, D, device="cuda", dtype=torch.bfloat16):
    pages_per_req = ctx // PAGE
    total = B * pages_per_req
    torch.manual_seed(0)
    q = torch.randn(B, Sq, H, D, device=device, dtype=dtype)
    k = torch.randn(total, PAGE, HKV, D, device=device, dtype=dtype)
    v = torch.randn_like(k)
    block_table = torch.arange(total, device=device, dtype=torch.int32).view(
        B, pages_per_req
    )
    seqlen_k = torch.full((B,), ctx, device=device, dtype=torch.int32)
    return q, k, v, block_table, seqlen_k


def _run(q, k, v, block_table, seqlen_k, num_kv_splits):
    return flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=num_kv_splits,
    )


def _run_single_pass(*args):
    """Reference with the public split count pinned to one."""
    return _run(*args, num_kv_splits=1)


def _windowed_reference(q, k, v, left, right, num_kv_heads):
    batch, seqlen_q, num_heads, head_dim = q.shape
    seqlen_k = k.shape[0] * k.shape[1]
    k = k.reshape(seqlen_k, num_kv_heads, head_dim).repeat_interleave(
        num_heads // num_kv_heads, dim=1
    )
    v = v.reshape(seqlen_k, num_kv_heads, head_dim).repeat_interleave(
        num_heads // num_kv_heads, dim=1
    )
    scores = torch.einsum("bthd,khd->bhtk", q.float(), k.float()) / math.sqrt(head_dim)
    q_pos = torch.arange(seqlen_q, device=q.device) + seqlen_k - seqlen_q
    k_pos = torch.arange(seqlen_k, device=q.device)
    mask = (k_pos[None, :] >= q_pos[:, None] - left) & (
        k_pos[None, :] <= torch.minimum(q_pos[:, None] + right, q_pos[:, None])
    )
    probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    return torch.einsum("bhtk,khd->bthd", probs, v.float()).reshape(
        batch, seqlen_q, num_heads, head_dim
    )


def _paged_reference(q, k, v, block_table, lengths, causal):
    batch, seqlen_q, num_heads, head_dim = q.shape
    outputs = []
    for b in range(batch):
        seqlen_k = int(lengths[b])
        if seqlen_k == 0:
            outputs.append(torch.zeros_like(q[b], dtype=torch.float32))
            continue
        kb = k[block_table[b]].reshape(-1, HKV, head_dim)[:seqlen_k].float()
        vb = v[block_table[b]].reshape(-1, HKV, head_dim)[:seqlen_k].float()
        kb = kb.repeat_interleave(num_heads // HKV, dim=1)
        vb = vb.repeat_interleave(num_heads // HKV, dim=1)
        scores = torch.einsum("thd,khd->htk", q[b].float(), kb) / math.sqrt(head_dim)
        if causal:
            q_pos = torch.arange(seqlen_q, device=q.device) + seqlen_k - seqlen_q
            k_pos = torch.arange(seqlen_k, device=q.device)
            scores.masked_fill_(k_pos[None, None, :] > q_pos[None, :, None], -math.inf)
        probs = scores.softmax(-1)
        outputs.append(torch.einsum("htk,khd->thd", probs, vb))
    return torch.stack(outputs)


def _paged_varlen_reference(q, cu_q, k, v, block_table, lengths, causal):
    outputs = []
    for b in range(block_table.shape[0]):
        start, end = int(cu_q[b]), int(cu_q[b + 1])
        outputs.append(
            _paged_reference(
                q[start:end].unsqueeze(0),
                k,
                v,
                block_table[b : b + 1],
                lengths[b : b + 1],
                causal,
            )[0]
        )
    return torch.cat(outputs)


def _paged_gappy_reference(q, k, v, block_table, starts, lengths, causal):
    batch, seqlen_q, num_heads, head_dim = q.shape
    outputs = []
    for b in range(batch):
        start = int(starts[b])
        seqlen_k = int(lengths[b])
        if seqlen_k == 0:
            outputs.append(torch.zeros_like(q[b], dtype=torch.float32))
            continue
        row_k = k[block_table[b]].reshape(-1, HKV, head_dim)
        row_v = v[block_table[b]].reshape(-1, HKV, head_dim)
        kb = row_k[start : start + seqlen_k].float()
        vb = row_v[start : start + seqlen_k].float()
        kb = kb.repeat_interleave(num_heads // HKV, dim=1)
        vb = vb.repeat_interleave(num_heads // HKV, dim=1)
        scores = torch.einsum("thd,khd->htk", q[b].float(), kb) / math.sqrt(head_dim)
        if causal:
            q_pos = torch.arange(seqlen_q, device=q.device) + seqlen_k - seqlen_q
            k_pos = torch.arange(seqlen_k, device=q.device)
            scores.masked_fill_(k_pos[None, None, :] > q_pos[None, :, None], -math.inf)
        outputs.append(torch.einsum("htk,khd->thd", scores.softmax(-1), vb))
    return torch.stack(outputs)


@pytest.mark.parametrize("B,Sq", [(1, 1), (1, 16), (2, 16), (4, 4)])
@pytest.mark.parametrize("D", [64, 128])
def test_auto_splitk_matches_single_split(B, Sq, D):
    """Auto split-K must be numerically equivalent to the single-pass kernel."""
    args = _paged_inputs(B, Sq, 32768, D)
    ref = _run_single_pass(*args)
    got = _run(*args, num_kv_splits=0)
    # Split-K reorders the softmax reduction, so allow bf16 rounding but nothing
    # structural: bf16 epsilon is ~7.8e-3 and observed deviation is ~2e-4.
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("nks", [2, 4, 8])
def test_forced_splitk_matches_single_split_at_short_q(nks):
    """Explicit split-K at Sq < 384 (previously rejected outright) is correct."""
    args = _paged_inputs(1, 1, 32768, 128)
    ref = _run_single_pass(*args)
    got = _run(*args, num_kv_splits=nks)
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("num_kv_splits,sized", [(0, True), (1, False)])
def test_auto_split_count_reaches_paged_sizing(monkeypatch, num_kv_splits, sized):
    """0 reaches the chain-length sizing and splits a long decode context; 1
    means never split and must not consult it."""
    seen = []
    real = fai._paged_num_kv_splits
    monkeypatch.setattr(
        fai, "_paged_num_kv_splits", lambda *a: (seen.append(real(*a)), seen[-1])[1]
    )
    _run(*_paged_inputs(1, 1, 32768, 128), num_kv_splits)
    if sized:
        assert len(seen) == 1 and seen[0] > 1
    else:
        assert not seen


def test_paged_splits_decline_short_chain():
    """A context that fits in the target chain gains nothing from a combine pass."""
    ctx = fai._PAGED_TARGET_CHAIN * fai._PAGED_BLOCK_N_OUT
    assert _paged_num_kv_splits(1, HKV, 8, ctx, 128) == 1


def test_paged_splits_when_chain_is_long():
    assert _paged_num_kv_splits(1, HKV, 8, 32768, 128) > 1


def test_paged_splits_capped_by_grid():
    """A large base grid must not be split past the workgroup ceiling."""
    B = 256
    blocks = B * H
    splits = _paged_num_kv_splits(B, H, 1, 131072, 128)
    assert splits <= max(1, fai._PAGED_MAX_WG // blocks)


def test_paged_splits_capped_by_workspace(monkeypatch):
    """The fp32 workspace must stay under its budget."""
    monkeypatch.setattr(fai, "_PAGED_WS_BUDGET_MB", 1)
    B, Sq, D = 4, 16, 128
    splits = _paged_num_kv_splits(B, H, Sq, 131072, D)
    elems = splits * B * H * Sq * (D // 2 + 2)
    assert splits == 1 or elems * 4 <= 1024 * 1024


def test_paged_splits_are_powers_of_two():
    """The split count is a compile-time trait; a growing decode context must
    not walk through a new value (and a new JIT) every few hundred tokens."""
    seen = set()
    for ctx in range(1024, 131072, 512):
        for B, Sq in ((1, 8), (4, 32), (16, 8)):
            splits = _paged_num_kv_splits(B, HKV, Sq, ctx, 128)
            assert splits & (splits - 1) == 0, (ctx, B, Sq, splits)
            seen.add(splits)
    assert len(seen) <= 8


def test_paged_splits_respect_user_cap(monkeypatch):
    """Power-of-two rounding must never exceed FLYDSL_PAGED_MAX_SPLITS."""
    monkeypatch.setattr(fai, "_PAGED_MAX_SPLITS", 48)
    for B, Sq, ctx in ((1, 8, 32000), (16, 8, 128000), (1, 1, 512000)):
        blocks = B * HKV * math.ceil(Sq / 64)
        cap = 48 * (2 if blocks < fai._PAGED_STARVED_BLOCKS else 1)
        assert _paged_num_kv_splits(B, HKV, Sq, ctx, 128) <= cap


def test_paged_splits_keep_block_table_window():
    """Rounding down must not leave a split owning more pages than the
    block-table LDS window holds; that shape would raise instead of run."""
    B, ctx = 640, 300000  # by_grid allows 3 splits; 2 would need 2344 pages
    splits = _paged_num_kv_splits(B, HKV, 1, ctx, 128)
    pages = math.ceil(ctx / PAGE)
    assert math.ceil(pages / splits) <= fai._PAGED_BT_LDS_SIZE


@pytest.mark.parametrize("ctx", [128, 512])
@pytest.mark.parametrize("Sq,D", [(1, 64), (4, 128)])
def test_single_split_gqa_packed_matches_reference(ctx, Sq, D):
    """Short contexts run one split with GQA stride packing.

    The packed M tile has padding rows past the real (group, token) rows; they
    sit HEAD_DIM apart in the caller's buffer, so an unmasked store lands on
    the next KV heads' output. Split-K masked them, the single pass did not.
    """
    q, k, v, block_table, seqlen_k = _paged_inputs(1, Sq, ctx, D)
    got = _run(q, k, v, block_table, seqlen_k, 0)
    ref = _windowed_reference(q, k, v, ctx, ctx, HKV)
    torch.testing.assert_close(got[0].float(), ref[0], atol=2e-2, rtol=2e-2)


def test_return_lse_still_works_under_auto():
    """return_lse needs the generic light kernel; auto sizing must keep it."""
    args = _paged_inputs(1, 1, 32768, 128)
    out, lse = flydsl_flash_attn_func(
        args[0],
        args[1],
        args[2],
        causal=True,
        num_kv_heads=HKV,
        block_table=args[3],
        seqlen_k=args[4],
        kv_cache_layout="linear",
        num_kv_splits=0,
        return_lse=True,
    )
    assert out.shape == args[0].shape
    assert lse.shape == (1, H, 1)


def test_auto_split_stride_packed_gqa_matches_single_pass_and_reference(monkeypatch):
    batch, seqlen_q, ctx, head_dim = 1, 64, 4096, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    auto_selection_calls = []

    def choose_two_splits(*args, **kwargs):
        auto_selection_calls.append((args, kwargs))
        return 2

    monkeypatch.setattr(fai, "_paged_num_kv_splits", choose_two_splits)
    guard_elements = 256
    guard_value = 42
    storage = torch.full(
        (q.numel() + 2 * guard_elements,),
        guard_value,
        device=q.device,
        dtype=q.dtype,
    )
    out = storage[guard_elements:-guard_elements].view_as(q)

    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=0,
        out=out,
    )
    single_pass = _run_single_pass(q, k, v, block_table, seqlen_k)
    reference = _paged_reference(q, k, v, block_table, seqlen_k, causal=True)

    assert len(auto_selection_calls) == 1
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(got, single_pass, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(got.float(), reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        storage[:guard_elements],
        torch.full_like(storage[:guard_elements], guard_value),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        storage[-guard_elements:],
        torch.full_like(storage[-guard_elements:], guard_value),
        atol=0,
        rtol=0,
    )


def test_ragged_varlen_does_not_auto_select_split_k(monkeypatch):
    ctx, head_dim = 512, 64
    q_dense, k, v, block_table, seqlen_k = _paged_inputs(2, 384, ctx, head_dim)
    q_lengths = torch.tensor([383, 384], device="cuda", dtype=torch.int32)
    cu_q = torch.nn.functional.pad(q_lengths.cumsum(0), (1, 0))
    q = torch.cat((q_dense[0, :383], q_dense[1]))
    cu_kv = torch.tensor([0, ctx, 2 * ctx], device="cuda", dtype=torch.int32)
    monkeypatch.setattr(fai, "_paged_num_kv_splits", lambda *a, **kw: 2)

    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=384,
        max_seqlen_kv=ctx,
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    ref = _paged_varlen_reference(q, cu_q, k, v, block_table, seqlen_k, causal=True)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_stride_packed_varlen_gqa_uses_token_row_lengths():
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q_dense, k, v, block_table, seqlen_k = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    q = q_dense.reshape(-1, H, head_dim)
    cu_q = torch.arange(
        0, (batch + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    cu_kv = torch.tensor([0, ctx, 2 * ctx], device="cuda", dtype=torch.int32)

    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=ctx,
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    ref = _paged_varlen_reference(q, cu_q, k, v, block_table, seqlen_k, causal=True)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_per_request_seqlen_k_is_honoured():
    """A request shorter than ``max_seqlen_kv`` must not attend to stale cache.

    The paged launch hands the kernel a scalar ``seq_len_kv`` -- the batch
    maximum -- and never the per-request tensor, so a short request reads cache
    past its own end. Error grows with the gap, not with tile misalignment:
    29952 is tile-aligned and still wrong.

    Measured here (B=8, ctx=32000, D=64, Sq=13), varying only the declared
    batch maximum, against an fp32 reference::

        seqlen_k   max_seqlen_kv     error
           32000           32000  2.49e-04
           31999           32000  2.03e-03
           30001           32000  1.29e-02
           29952           32000  1.30e-02

    Tolerance is bf16 epsilon (7.81e-03): it clears split-K's reordering noise
    (~2e-04) by a wide margin while still rejecting the two wrong answers.

    Fixed by forwarding ``seqlen_k`` to the paged launch and reading each
    request's length in the kernel under the KV_LENS trait, so the light route
    is correct for ragged batches without a device-to-host sync or a host-side
    scan, and stays legal under CUDA-graph capture.

    This was xfail(strict) while unfixed; the marker is gone because the
    behaviour it guarded now holds.
    """
    ctx, short, B = 32000, 30001, 8
    q, k, v, block_table, _ = _paged_inputs(B, 13, ctx, 64)
    seqlen_k = torch.full((B,), short, device="cuda", dtype=torch.int32)

    def _attend(max_seqlen_kv):
        return flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=0,
            max_seqlen_kv=max_seqlen_kv,
        )

    # Identical request lengths either way, so the *declared* batch maximum
    # must not change the answer.
    torch.testing.assert_close(
        _attend(ctx).float(), _attend(short).float(), atol=7.81e-3, rtol=7.81e-3
    )


@pytest.mark.parametrize("num_kv_splits", [0, 2])
def test_per_request_seqlen_k_on_dualwave_route(num_kv_splits):
    """bf16 non-causal Sq > 256 on gfx950 takes the dualwave kernel, which has
    no dense per-request KV length; a short request must still stop at its own
    end rather than read the stale cache up to ``max_seqlen_kv``."""
    ctx, Sq, D = 4096, 512, 64
    q, k, v, block_table, _ = _paged_inputs(2, Sq, ctx, D)
    seqlen_k = torch.tensor([ctx, 3001], device="cuda", dtype=torch.int32)
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=False,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=num_kv_splits,
        max_seqlen_kv=ctx,
    )
    for b in range(2):
        n = int(seqlen_k[b])
        pages = block_table[b]
        kb = k[pages].reshape(-1, HKV, D)[:n].float()
        vb = v[pages].reshape(-1, HKV, D)[:n].float()
        kb = kb.repeat_interleave(H // HKV, 1)
        vb = vb.repeat_interleave(H // HKV, 1)
        scores = torch.einsum("thd,khd->htk", q[b].float(), kb) / math.sqrt(D)
        ref = torch.einsum("htk,khd->thd", scores.softmax(-1), vb)
        torch.testing.assert_close(got[b].float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal,Sq,num_kv_splits", [(True, 13, 0), (False, 512, 0)])
def test_per_request_seqlen_k_is_bounded_by_block_table_capacity(
    causal, Sq, num_kv_splits
):
    ctx, D = 512, 64
    q, k, v, block_table, _ = _paged_inputs(3, Sq, ctx, D)
    backing = torch.tensor(
        [ctx, 0, -PAGE, 0, ctx + PAGE, 0], device="cuda", dtype=torch.int64
    )
    seqlen_k = backing[::2]
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=num_kv_splits,
    )
    bounded = torch.tensor([ctx, 0, ctx], device="cuda", dtype=torch.int32)
    ref = _paged_reference(q, k, v, block_table, bounded, causal)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal,Sq", [(True, 13), (False, 512)])
def test_int64_dense_lengths_are_clamped_before_narrowing(causal, Sq):
    ctx, head_dim = 512, 64
    q, k, v, block_table, _ = _paged_inputs(2, Sq, ctx, head_dim)
    backing = torch.tensor([2**32 + 128, 0, 2**31, 0], device="cuda", dtype=torch.int64)
    seqlen_k = backing[::2]
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        max_seqlen_kv=ctx,
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    bounded = torch.full((2,), ctx, device="cuda", dtype=torch.int32)
    ref = _paged_reference(q, k, v, block_table, bounded, causal)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_int64_cumulative_lengths_are_clamped_before_narrowing():
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q_dense, k, v, block_table, seqlen_k = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    q = q_dense.reshape(-1, H, head_dim)
    cu_q = torch.arange(
        0, (batch + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    huge = 2**32 + 128
    cu_kv = torch.tensor([0, huge, 2 * huge], device="cuda", dtype=torch.int64)
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=ctx,
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    bounded = torch.full((batch,), ctx, device="cuda", dtype=torch.int32)
    ref = _paged_varlen_reference(q, cu_q, k, v, block_table, bounded, causal=True)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_int64_gappy_starts_are_clamped_before_narrowing():
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q, k, v, block_table, _ = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    q_packed = q.reshape(-1, H, head_dim)
    cu_q = torch.arange(
        0, (batch + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    huge = 2**32 + 128
    cu_kv = torch.tensor([0, huge, 2 * huge], device="cuda", dtype=torch.int64)
    starts = torch.tensor([huge, 0], device="cuda", dtype=torch.int64)
    got = flydsl_flash_attn_func(
        q_packed,
        k,
        v,
        causal=False,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=None,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        kv_seqstart=starts,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=ctx,
        cross_seqlen=True,
    ).view_as(q)
    normalized_starts = torch.tensor([ctx, 0], device="cuda", dtype=torch.int32)
    bounded = torch.tensor([0, ctx], device="cuda", dtype=torch.int32)
    ref = _paged_gappy_reference(
        q, k, v, block_table, normalized_starts, bounded, causal=False
    )
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal,Sq", [(True, 13), (False, 512)])
def test_explicit_max_seqlen_kv_bounds_each_request(causal, Sq):
    ctx, bound, D = 512, 384, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(2, Sq, ctx, D)
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        max_seqlen_kv=bound,
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    bounded = torch.full_like(seqlen_k, bound)
    ref = _paged_reference(q, k, v, block_table, bounded, causal)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_zero_length_has_empty_attention_result():
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 4, 512, 64)
    seqlen_k.fill_(-1)
    out, lse = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        return_lse=True,
    )
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    assert torch.isneginf(lse).all()


def test_gappy_lengths_are_bounded_by_remaining_block_table_capacity():
    batch, seqlen_q, ctx, head_dim = 3, 4, 512, 64
    q, k, v, block_table, _ = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    starts = torch.tensor([-64, 480, 576], device="cuda", dtype=torch.int32)
    raw_lengths = torch.tensor([128, 128, 128], device="cuda", dtype=torch.int32)
    cu_q = torch.arange(
        0, (batch + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    cu_kv = torch.nn.functional.pad(raw_lengths.cumsum(0), (1, 0))
    got = flydsl_flash_attn_func(
        q.reshape(-1, H, head_dim),
        k,
        v,
        causal=False,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=None,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        kv_seqstart=starts,
        max_seqlen_q=seqlen_q,
        cross_seqlen=True,
    ).view_as(q)
    normalized_starts = torch.tensor([0, 480, 512], device="cuda", dtype=torch.int32)
    bounded = torch.tensor([128, 32, 0], device="cuda", dtype=torch.int32)
    ref = _paged_gappy_reference(
        q, k, v, block_table, normalized_starts, bounded, causal=False
    )
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("max_seqlen_kv", [-1, 513])
def test_rejects_max_seqlen_kv_outside_block_table_capacity(max_seqlen_kv):
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 4, 512, 64)
    with pytest.raises(ValueError, match="max_seqlen_kv must be between"):
        flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            max_seqlen_kv=max_seqlen_kv,
        )


@pytest.mark.parametrize("mode", ["dense", "varlen", "gappy"])
def test_rejects_zero_width_block_table_before_build(mode):
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    kwargs = {}
    q_arg = q
    if mode != "dense":
        q_arg = q.reshape(-1, H, head_dim)
        cu_q = torch.arange(
            0,
            (batch + 1) * seqlen_q,
            seqlen_q,
            device="cuda",
            dtype=torch.int32,
        )
        cu_kv = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
        kwargs.update(
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=seqlen_q,
        )
        if mode == "gappy":
            seqlen_k = None
            kwargs["kv_seqstart"] = torch.zeros(batch, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="at least one page column"):
        flydsl_flash_attn_func(
            q_arg,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table[:, :0],
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=0,
            **kwargs,
        )


@pytest.mark.parametrize("causal,Sq", [(True, 1), (True, 4), (False, 512)])
def test_cuda_graph_sees_updated_seqlen_k(causal, Sq):
    """Capture without a scalar bound must not read ``seqlen_k`` on the host."""
    ctx = 4096
    q, k, v, block_table, seqlen_k = _paged_inputs(2, Sq, ctx, 64)
    out = torch.empty_like(q)

    def step():
        flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=0,
            out=out,
        )

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    seqlen_k.copy_(torch.tensor([ctx, 1000], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    ref = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k.clone(),
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_cuda_graph_clamps_int64_lengths_before_narrowing():
    ctx, seqlen_q, head_dim = 512, 4, 64
    q, k, v, block_table, _ = _paged_inputs(2, seqlen_q, ctx, head_dim)
    seqlen_k = torch.full((2,), ctx, device="cuda", dtype=torch.int64)
    out = torch.empty_like(q)

    def step():
        flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            max_seqlen_kv=ctx,
            kv_cache_layout="linear",
            num_kv_splits=0,
            out=out,
        )

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    seqlen_k.copy_(torch.tensor([2**32 + 128, 2**31], device="cuda", dtype=torch.int64))
    graph.replay()
    torch.cuda.synchronize()

    bounded = torch.full((2,), ctx, device="cuda", dtype=torch.int32)
    ref = _paged_reference(q, k, v, block_table, bounded, causal=True)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_cuda_graph_normalizes_gappy_metadata_without_dense_lengths():
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q, k, v, block_table, _ = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    q_packed = q.reshape(-1, H, head_dim)
    cu_q = torch.arange(
        0, (batch + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    raw_lengths = torch.tensor([128, 128], device="cuda", dtype=torch.int32)
    cu_kv = torch.nn.functional.pad(raw_lengths.cumsum(0), (1, 0))
    starts = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    out = torch.empty_like(q_packed)

    def step():
        flydsl_flash_attn_func(
            q_packed,
            k,
            v,
            causal=False,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=None,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            kv_seqstart=starts,
            max_seqlen_q=seqlen_q,
            cross_seqlen=True,
            out=out,
        )

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    starts.copy_(torch.tensor([64, 480], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    bounded = torch.tensor([128, 32], device="cuda", dtype=torch.int32)
    ref = _paged_gappy_reference(q, k, v, block_table, starts, bounded, causal=False)
    torch.testing.assert_close(out.view_as(q).float(), ref, atol=2e-2, rtol=2e-2)


def test_updater_only_graph_invalidates_dualwave_lengths():
    ctx, Sq, D = 4096, 512, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(2, Sq, ctx, D)
    staging = seqlen_k.clone()

    def attend(lengths):
        return flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=False,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=lengths,
            kv_cache_layout="linear",
            num_kv_splits=0,
            max_seqlen_kv=ctx,
        )

    attend(seqlen_k)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        seqlen_k.copy_(staging)
    staging.copy_(torch.tensor([ctx, 3001], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    got = attend(seqlen_k)
    ref = attend(seqlen_k.clone())
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


def test_dualwave_accepts_inference_tensor_lengths():
    ctx, Sq, D = 4096, 512, 64
    q, k, v, block_table, _ = _paged_inputs(2, Sq, ctx, D)
    with torch.inference_mode():
        seqlen_k = torch.tensor([ctx, 3001], device="cuda", dtype=torch.int32)
        got = flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=False,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=0,
            max_seqlen_kv=ctx,
        )
    assert torch.isfinite(got).all()


def test_rejects_wrong_device_seqlen_k():
    q, k, v, block_table, _ = _paged_inputs(2, 4, 512, 64)
    seqlen_k = torch.tensor([512, 511], device="cpu", dtype=torch.int32)
    with pytest.raises(ValueError, match="seqlen_k must be on"):
        _run(q, k, v, block_table, seqlen_k, 0)


def test_rejects_wrong_device_block_table():
    q, k, v, block_table, seqlen_k = _paged_inputs(2, 4, 512, 64)
    with pytest.raises(ValueError, match="block_table must be on"):
        _run(q, k, v, block_table.cpu(), seqlen_k, 0)


def test_rejects_negative_num_kv_splits():
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 384, 512, 64)
    with pytest.raises(ValueError, match="num_kv_splits must be nonnegative"):
        _run(q, k, v, block_table, seqlen_k, -1)


def test_explicit_stream_launch_matches_default_stream():
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 384, 512, 64)
    launch_stream = torch.cuda.Stream()
    with torch.cuda.stream(launch_stream):
        got = flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=2,
            max_seqlen_kv=512,
            stream=launch_stream,
        )
    launch_stream.synchronize()
    ref = _run(q, k, v, block_table, seqlen_k, 2)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("varlen", [False, True])
def test_materialized_gqa_pack_and_unpack_stay_on_explicit_stream(monkeypatch, varlen):
    batch, seqlen_q, ctx, head_dim = 1, 4, 512, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    monkeypatch.setattr(fai, "_PAGED_GQA_STRIDE", False)
    q_arg = q
    kwargs = {}
    if varlen:
        q_arg = q.reshape(-1, H, head_dim)
        kwargs.update(
            cu_seqlens_q=torch.tensor([0, seqlen_q], device="cuda", dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, ctx], device="cuda", dtype=torch.int32),
            max_seqlen_q=seqlen_q,
        )

    def attend(*, out=None, stream=None):
        return flydsl_flash_attn_func(
            q_arg,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=0,
            out=out,
            stream=stream,
            **kwargs,
        )

    # Compile and initialize the materialized-pack variant before installing the
    # stream gate; otherwise compilation time could let the gate finish early.
    expected = attend()
    out = torch.full_like(q_arg, math.nan)
    torch.cuda.synchronize()

    producer_stream = torch.cuda.current_stream()
    launch_stream = torch.cuda.Stream()
    gate = torch.cuda.Event()
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    for _ in range(128):
        torch.mm(a, b, out=c)
    gate.record(producer_stream)
    assert not gate.query(), "stream blocker completed before the test launch"

    got = attend(out=out, stream=launch_stream)
    launch_stream.synchronize()
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    producer_stream.synchronize()


# ── Paged sliding window ──────────────────────────────────────────────────────
# The window mask lives in the generic kernel's N32 path, which the paged light
# route already builds. The dualwave route masks by its own rules and has no
# window term, so it must decline a windowed shape rather than drop the window
# silently.


def _paged_window_reference(q, k, v, window):
    """Bottom-right causal attention restricted to the last ``window`` keys.

    ``window`` is the kernel's ``window_left``: it keeps ``kv > q_pos - window``,
    so exactly ``window`` keys including the query's own position.
    """
    return _windowed_reference(q, k, v, window - 1, 0, HKV)[0]


@pytest.mark.parametrize("num_kv_splits", [1, 4, 0])
@pytest.mark.parametrize("Sq", [1, 4, 16, 32, 64])
@pytest.mark.parametrize("window", [17, 256, 2048])
def test_paged_window_matches_reference(num_kv_splits, Sq, window):
    """A windowed paged call must match an explicitly windowed reference.

    Parametrised over split counts because split-K partitions the KV range: the
    window bound has to hold inside each partition, not just end to end. Query
    lengths cover GQA packing (rows = group x Sq, 128 rows at Sq=16), and the
    windows span less than one KV tile up to the whole context.
    """
    ctx, B, D = 2048, 1, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(B, Sq, ctx, D)
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=num_kv_splits,
        max_seqlen_kv=ctx,
        window_left=window,
    )
    ref = _paged_window_reference(q, k, v, window)
    torch.testing.assert_close(got[0].float(), ref, atol=2e-2, rtol=2e-2)


def test_paged_window_splits_bounded():
    """Windowed sizing splits by grid, never past the reachable KV tiles, and
    stays a power of two (the split count is compiled in)."""
    for B, rows, reach in ((1, 8, 2049), (64, 32, 2052), (4, 128, 17), (16, 8, 300)):
        s = fai._paged_window_num_kv_splits(B, HKV, rows, reach, 128)
        assert s & (s - 1) == 0
        assert s <= max(1, math.ceil(reach / fai._PAGED_BLOCK_N_OUT))
        assert s <= fai._PAGED_WINDOW_MAX_SPLITS


def test_paged_window_long_context_matches_reference():
    """A window over a context past one split's block-table window (2048 pages)
    still needs enough splits for the pages, even though it reads few keys."""
    ctx, window, D = 262144, 2048, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 1, ctx, D)
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=0,
        max_seqlen_kv=ctx,
        window_left=window,
    )
    ref = _paged_window_reference(q, k, v, window)
    torch.testing.assert_close(got[0].float(), ref, atol=2e-2, rtol=2e-2)


def test_paged_window_keeps_gqa_packing(monkeypatch):
    """A causal window stays on the light kernel under GQA packing: the packed
    rows carry their query position, so the window bound still applies per row."""

    seen = []
    real = fai._build_paged_light
    monkeypatch.setattr(
        fai, "_build_paged_light", lambda **kw: (seen.append(kw), real(**kw))[1]
    )
    args = _paged_inputs(1, 4, 2048, 64)

    _run(*args, num_kv_splits=1)
    assert seen[-1]["window_left"] == -1
    assert seen[-1]["q_pack_qlen"] == 4, "unwindowed shapes should still pack"

    flydsl_flash_attn_func(
        args[0],
        args[1],
        args[2],
        causal=True,
        num_kv_heads=HKV,
        block_table=args[3],
        seqlen_k=args[4],
        kv_cache_layout="linear",
        num_kv_splits=1,
        max_seqlen_kv=2048,
        window_left=255,
    )
    assert seen[-1]["window_left"] == 255, "window must reach the light builder"
    assert seen[-1]["q_pack_qlen"] == 4, "causal windowed shapes should pack"


@pytest.mark.parametrize(
    "Sq,causal", [(1, True), (4, True), (512, False)], ids=["sq1", "sq4", "sq512"]
)
def test_paged_window_is_never_silently_dropped(Sq, causal):
    """The window must change the answer on every route a shape can take.

    This is the sharp form of the routing guard: a route with no window term
    returns a full-context answer rather than failing, so identical output is
    the bug. ``Sq=512`` non-causal is the case that previously reached the
    native kernel and produced bit-identical results with and without a window.
    """
    ctx = 2048
    q, k, v, block_table, seqlen_k = _paged_inputs(1, Sq, ctx, 64)

    def attend(window_left):
        return flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=1,
            max_seqlen_kv=ctx,
            window_left=window_left,
        )

    assert not torch.equal(attend(-1), attend(255))


def test_paged_window_still_rejected_for_gappy_kv():
    """Gappy paged indexes KV differently, so the flat-column window bound does
    not line up. It must fail loudly rather than answer incorrectly."""
    ctx, B = 2048, 1
    q, k, v, block_table, seqlen_k = _paged_inputs(B, 4, ctx, 64)
    with pytest.raises(NotImplementedError, match="gappy"):
        flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_seqstart=torch.zeros(B, device="cuda", dtype=torch.int32),
            kv_cache_layout="linear",
            num_kv_splits=1,
            max_seqlen_kv=ctx,
            window_left=255,
        )
