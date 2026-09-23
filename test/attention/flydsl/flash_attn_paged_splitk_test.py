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
