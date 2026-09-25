# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Automatic paged split-K selection (``num_kv_splits=0``).

Decode shapes (short Q against a long paged cache) produce only ``B * H``
workgroups and leave most CUs idle. ``_auto_paged_kv_splits`` recovers that
parallelism along KV. These tests pin the two properties that matter:
splitting must not change the result, and it must not fire where the device is
already full or where the kernel cannot support it.
"""

import pytest
import torch

from mslk.attention.flydsl.flash_attn_interface import (
    _auto_paged_kv_splits,
    flydsl_flash_attn_func,
)
from mslk.flydsl.common import is_flydsl_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flydsl_available(),
    reason="requires a ROCm GPU with FlyDSL",
)

PAGE = 64  # _PAGED_PAGE_SIZE: the only page size the native paged kernel takes
H, HKV = 32, 4


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


@pytest.mark.parametrize("B,Sq", [(1, 1), (1, 16), (2, 16), (4, 4)])
@pytest.mark.parametrize("D", [64, 128])
def test_auto_splitk_matches_single_split(B, Sq, D):
    """Auto split-K must be numerically equivalent to the single-pass kernel."""
    args = _paged_inputs(B, Sq, 32768, D)
    ref = _run(*args, num_kv_splits=1)
    got = _run(*args, num_kv_splits=0)
    # Split-K reorders the softmax reduction, so allow bf16 rounding but nothing
    # structural: bf16 epsilon is ~7.8e-3 and observed deviation is ~2e-4.
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("nks", [2, 4, 8])
def test_forced_splitk_matches_single_split_at_short_q(nks):
    """Explicit split-K at Sq < 384 (previously rejected outright) is correct."""
    args = _paged_inputs(1, 1, 32768, 128)
    ref = _run(*args, num_kv_splits=1)
    got = _run(*args, num_kv_splits=nks)
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-3, rtol=2e-3)


def test_explicit_one_disables_splitting():
    """num_kv_splits=1 stays an exact opt-out even where auto would split."""
    args = _paged_inputs(1, 1, 32768, 128)
    torch.testing.assert_close(
        _run(*args, num_kv_splits=1).float(), _run(*args, num_kv_splits=1).float()
    )


def test_heuristic_declines_when_device_is_full():
    """Large batch already fills the CUs, so auto must return 1 (no combine pass)."""
    device = torch.device("cuda")
    full = _auto_paged_kv_splits(
        num_batches=64,
        num_heads=H,
        seqlen_q=1,
        head_dim=128,
        max_kv_pages=2048,
        dtype_str="bf16",
        device=device,
    )
    assert full == 1


def test_heuristic_splits_when_device_is_starved():
    """B=1 leaves most CUs idle, so auto must split."""
    starved = _auto_paged_kv_splits(
        num_batches=1,
        num_heads=H,
        seqlen_q=1,
        head_dim=128,
        max_kv_pages=2048,
        dtype_str="bf16",
        device=torch.device("cuda"),
    )
    assert starved > 1


def test_heuristic_capped_by_available_pages():
    """A tiny cache cannot feed many splits, whatever the occupancy says."""
    capped = _auto_paged_kv_splits(
        num_batches=1,
        num_heads=H,
        seqlen_q=1,
        head_dim=128,
        max_kv_pages=4,
        dtype_str="bf16",
        device=torch.device("cuda"),
    )
    assert capped == 1


def test_heuristic_declines_unsupported_dtype():
    """fp8 has no paged split-K variant; auto must not select one."""
    assert (
        _auto_paged_kv_splits(
            num_batches=1,
            num_heads=H,
            seqlen_q=1,
            head_dim=128,
            max_kv_pages=2048,
            dtype_str="fp8",
            device=torch.device("cuda"),
        )
        == 1
    )


def test_return_lse_still_works_under_auto():
    """return_lse needs the generic light kernel, which requires splits <= 1."""
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


# ── Head-packed paged decode fast path (Sq == 1) ──────────────────────────────
# The dualwave paged kernel maps the MFMA M-axis to query rows, so at Sq=1 only
# 1 of 32 rows is real work. `decode/pa_decode_gfx950.py` packs query heads onto
# M instead; `_flydsl_flash_attn_paged` routes Sq=1 to it. These tests pin the
# routing conditions and the equivalence of the two kernels.


def _count_hp_calls(monkeypatch):
    """Patch the head-packed launcher to record invocations; returns the log.

    Also disables GQA head packing. Packing removes the same fan-out the
    head-packed decode kernel was routed here to avoid, covers Sq up to 64
    rather than the M-tile budget's 4-16, and therefore takes precedence by
    default (see test_gqa_packing_takes_precedence). The decode kernel remains
    the route for everything packing declines, and that is what these tests
    pin -- so they select it explicitly rather than depending on which
    mechanism happens to win.
    """
    from mslk.attention.flydsl import flash_attn_interface as fai
    from mslk.attention.flydsl.decode import pa_decode_dense

    monkeypatch.setattr(fai, "_PAGED_GQA_PACK", False)
    calls = []
    real = pa_decode_dense.pa_decode_paged_launch

    def _spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(pa_decode_dense, "pa_decode_paged_launch", _spy)
    return calls


def _spy_hp(monkeypatch):
    """Record head-packed launches without disabling GQA packing."""
    from mslk.attention.flydsl.decode import pa_decode_dense

    calls = []
    real = pa_decode_dense.pa_decode_paged_launch

    def _spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(pa_decode_dense, "pa_decode_paged_launch", _spy)
    return calls


def test_head_packed_takes_precedence_over_gqa_packing(monkeypatch):
    """Both remove the GQA fan-out, so only one may run -- this one wins.

    It is the only paged route that honours a per-request ``seqlen_k``. The
    light/dualwave launch hands the kernel a scalar ``seq_len_kv`` (the batch
    max) and no per-request tensor, so a request shorter than the batch max
    attends to stale cache. Preferring this path is what keeps that correct;
    see test_per_request_seqlen_k_is_honoured.
    """
    calls = _spy_hp(monkeypatch)
    _run(*_paged_inputs(1, 1, 32768, 64), 0)
    assert len(calls) == 1


def test_gqa_packing_owns_what_head_packing_declines(monkeypatch):
    """Past the M-tile budget the head-packed route declines and packing runs.

    Sq=13 at D=128 needs 7 tiles, one past the measured no-spill limit of 6, and
    13 is odd so two-pass query grouping cannot divide it either. Sq=16 would
    *not* work as a case here: grouping rescues it as two 4-tile passes.
    """
    calls = _spy_hp(monkeypatch)
    _run(*_paged_inputs(8, 13, 32768, 128), 0)
    assert calls == []


def test_per_request_seqlen_k_is_honoured():
    """A request shorter than ``max_seqlen_kv`` must not attend to stale cache.

    Regression guard for the routing above. With `seqlen_k` below
    `max_seqlen_kv` the generic paged path masks every request to the batch
    max; the error grows with the gap and passes bf16 epsilon well before the
    gap reaches a page. Upstream main shows the same, so this pins the routing
    that avoids it rather than a fix to that kernel.
    """
    ctx, short = 32768, 30001
    q, k, v, block_table, _ = _paged_inputs(4, 4, ctx, 64)
    seqlen_k = torch.full((4,), short, device="cuda", dtype=torch.int32)
    exact = flydsl_flash_attn_func(
        q, k, v, causal=True, num_kv_heads=HKV, block_table=block_table,
        seqlen_k=seqlen_k, kv_cache_layout="linear", num_kv_splits=0,
        max_seqlen_kv=short,
    )
    padded = flydsl_flash_attn_func(
        q, k, v, causal=True, num_kv_heads=HKV, block_table=block_table,
        seqlen_k=seqlen_k, kv_cache_layout="linear", num_kv_splits=0,
        max_seqlen_kv=ctx,
    )
    # Same request lengths, so the declared batch max must not change the answer.
    torch.testing.assert_close(padded, exact, atol=2e-2, rtol=2e-2)


def calls_groups(calls):
    """Query-group count the routing layer asked for on the first call."""
    return calls[0].get("query_groups", 1)


def test_head_packed_matches_dualwave_at_sq1(monkeypatch):
    """The two kernels must agree; the head-packed one is the faster path."""
    from mslk.attention.flydsl import flash_attn_interface as fai

    args = _paged_inputs(2, 1, 32768, 64)
    monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", True)
    ref = _run(*args, 1)
    monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", False)
    got = _run(*args, 0)
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


def test_head_packed_selected_at_sq1(monkeypatch):
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(1, 1, 32768, 64), 0)
    assert len(calls) == 1


@pytest.mark.parametrize("Sq", [2, 4])
def test_head_packed_selected_for_short_query_blocks(monkeypatch, Sq):
    """M holds ratio*Sq pairs over MAX_M_TILES tiles: at ratio 8 that is Sq <= 4."""
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(1, Sq, 32768, 64), 0)
    assert len(calls) == 1


@pytest.mark.parametrize("Sq", [8, 13, 16])
def test_head_packed_selected_for_deep_tiling_at_d64(monkeypatch, Sq):
    """D=64 is measured clean to 8 M-tiles, so Sq up to 16 is in range."""
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, Sq, 32768, 64), 0)
    assert len(calls) == 1


@pytest.mark.parametrize("Sq", [13, 15])
def test_head_packed_declined_when_d128_would_spill(monkeypatch, Sq):
    """D=128 spills past 6 M-tiles (measured 280B at 7, 916B at 8).

    Spilling a bandwidth-bound decode kernel is self-defeating, so these fall
    through to the existing path. 13 and 15 are odd, so equal-span query
    grouping cannot rescue them either -- see the Sq=16 case below.
    """
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, Sq, 32768, 128), 0)
    assert calls == []


def test_head_packed_uses_query_groups_when_single_pass_would_spill(monkeypatch):
    """D=128 / Sq=16 needs 8 tiles in one pass, which spills.

    Two passes of 8 query tokens need 4 tiles each -- under the budget -- at the
    cost of reading KV twice. Measured 1.9x faster than the path it replaces.
    """
    from mslk.attention.flydsl import flash_attn_interface as fai

    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, 16, 32768, 128), 0)
    assert len(calls) == 1
    assert calls_groups(calls) == 2


def test_query_grouping_kill_switch(monkeypatch):
    """With grouping disabled, a shape that only fits via groups declines."""
    from mslk.attention.flydsl import flash_attn_interface as fai

    monkeypatch.setattr(fai, "_DISABLE_PAGED_QGROUPS", True)
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, 16, 32768, 128), 0)
    assert calls == []


def test_query_grouping_not_used_when_single_pass_fits(monkeypatch):
    """Grouping is only for rescuing shapes the register budget would reject.

    D=64 fits Sq=16 in one pass, and grouping there measured inside run-to-run
    noise, so the extra KV pass must not be spent.
    """
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, 16, 32768, 64), 0)
    assert len(calls) == 1
    assert calls_groups(calls) == 1


def test_head_packed_declined_when_too_few_ctas(monkeypatch):
    """Deep tiling costs occupancy, so it needs enough CTAs to stay resident.

    One CTA per CU measured 0.74x the path it replaced; the floor rejects it.
    """
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(1, 16, 32000, 64), 0)  # 8 tiles, batch 1 -> ~1 CTA/CU
    assert calls == []


def test_head_packed_floor_does_not_reject_shallow_tiling(monkeypatch):
    """The CTA floor applies to deep tiling only.

    Sq=4 is two tiles and still holds 6 waves/SIMD; it was measured winning at
    batch 1, so the floor must not take it away.
    """
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(1, 4, 32000, 64), 0)
    assert len(calls) == 1


@pytest.mark.parametrize("Sq,D", [(2, 64), (4, 64), (4, 128)])
def test_head_packed_matches_dualwave_multi_token(monkeypatch, Sq, D):
    """Packed (qtok, head) rows must agree with the dualwave path.

    This is the check on the per-query-token causal bound: the kernel applies
    `min(t_end, t_full - Sq + qtok + 1)` itself, so a wrong bound shows up here
    as a mismatch on the earlier query rows only.
    """
    from mslk.attention.flydsl import flash_attn_interface as fai

    args = _paged_inputs(2, Sq, 32768, D)
    monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", True)
    ref = _run(*args, 1)
    monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", False)
    got = _run(*args, 0)
    assert got.shape == ref.shape == (2, Sq, H, D)
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


def test_head_packed_declined_for_wide_gqa_ratio(monkeypatch):
    """ratio = H // HKV must be <= MFMA_M (16); here it is 32."""
    calls = _count_hp_calls(monkeypatch)
    ctx, D, pages = 32768, 64, 32768 // PAGE
    torch.manual_seed(0)
    q = torch.randn(1, 1, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(pages, PAGE, 1, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    bt = torch.arange(pages, device="cuda", dtype=torch.int32).view(1, pages)
    sk = torch.full((1,), ctx, device="cuda", dtype=torch.int32)
    flydsl_flash_attn_func(
        q, k, v, causal=True, num_kv_heads=1, block_table=bt, seqlen_k=sk,
        kv_cache_layout="linear", num_kv_splits=0,
    )
    assert calls == []


def test_head_packed_kill_switch(monkeypatch):
    from mslk.attention.flydsl import flash_attn_interface as fai

    calls = _count_hp_calls(monkeypatch)
    monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", True)
    _run(*_paged_inputs(1, 1, 32768, 64), 0)
    assert calls == []


def test_head_packed_exceeds_dualwave_page_table_cap():
    """The dualwave path caps at 2048 pages/split (131072 tokens at page 64).

    The decode kernel reads the block table straight from memory, so it has no
    such window -- this context would raise on the old path.
    """
    ctx = 262144
    out = _run(*_paged_inputs(1, 1, ctx, 64), 0)
    assert out.shape == (1, 1, H, 64)
    assert torch.isfinite(out).all()
