# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level FlyDSL Flash Attention API for gfx950 / gfx942.

Wraps ``flash_attn_generic.build_flash_attn_func_module`` (gfx942-compatible,
dense self/cross-attention) and ``flash_attn_gfx950.build_flash_attn_dualwave_swp_module``
(gfx950 DUALWAVE_SWP, varlen + split-K) behind a single function:

    ``flydsl_flash_attn_func(q, k, v, ...)``

Key features vs calling build_* directly:
- ``@functools.lru_cache`` on the build call so repeated invocations with the
  same (static) config compile only once per process.
- Explicit ``max_seqlen_q`` / ``cross_seqlen`` controls for varlen builds.
- split-K fp32 workspace allocation, zeroing, and the 4 GiB descriptor guard.
- Unified device / stream context (``torch.cuda.device`` + current stream).
- Validates shapes, dtypes, and arch before compiling.
- Accepts ``debug_counts`` tensor to enable the lazy-rescale branch counter
  (gfx950 DUALWAVE_SWP dualwave_swp_debug_lazy_counts=True path).

Known dualwave defects (tracked; routing below escapes affected cases to the
generic light kernel, which moves those users off the faster dualwave path):
  * f16 dualwave softmax overflows at large logits (narrow exponent) -> NaN, so
    f16 is forced to the light path (see ``_paged_light_ok`` / the varlen and
    dense f16 escapes below).
  * dualwave cross-attention NaNs for >=5 KV tiles / hardcodes 1/sqrt(D), so
    cross-length cases are kept off dualwave.
TODO(): Fix the flyDSL dualwave f16 cross-attn NaN
"""

from __future__ import annotations

import functools
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: F401  (imported for callers' convenience)

# Re-export so callers only need to import from this module.
from .flash_attn_utils import dualwave_splitk_workspace_elems

__all__ = ["flydsl_flash_attn_func", "dualwave_splitk_workspace_elems"]

_DTYPE_MAP = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float8_e4m3fn: "fp8"}

# Short varlen/paged cases use the lightweight generic path.
_VARLEN_LIGHT_MAX_SEQ = 256
_DENSE_LIGHT_CU_FALLBACK = 256
_DENSE_DUALWAVE_MIN_SEQ = 256
_DENSE_DUALWAVE_LARGE_BATCH = 8
_DENSE_DUALWAVE_MIN_SEQ_LARGE_BATCH = 192
_DENSE_M256_MIN_TOKENS = 4096


def _dtype_str(t: torch.Tensor) -> str:
    s = _DTYPE_MAP.get(t.dtype)
    if s is None:
        raise ValueError(
            f"flydsl_flash_attn_func only supports bf16/f16/fp8, got {t.dtype!r}"
        )
    return s


def _gpu_arch(device: torch.device) -> str:
    try:
        return torch.cuda.get_device_properties(device.index).gcnArchName.split(":")[0]
    except Exception:
        return ""


def _dense_routes_to_dualwave(batch: int, seq_len: int) -> bool:
    if batch >= _DENSE_DUALWAVE_LARGE_BATCH:
        return seq_len >= _DENSE_DUALWAVE_MIN_SEQ_LARGE_BATCH
    return seq_len >= _DENSE_DUALWAVE_MIN_SEQ


def _dense_light_cu(device: torch.device) -> int:
    try:
        return int(torch.cuda.get_device_properties(device.index).multi_processor_count)
    except Exception:
        return _DENSE_LIGHT_CU_FALLBACK


def _dense_generic_tile(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_str: str,
    device: torch.device,
    has_bias: bool = False,
):
    if head_dim in (64, 128) and dtype_str in ("bf16", "f16"):
        main_blocks = batch * num_heads * ((seq_len + 127) // 128)
        if main_blocks < _dense_light_cu(device):
            # Additive bias forces the N32 path and reloads the bias plane per Q
            # tile; the block_m=64 light tile reloads it twice as often, so prefer
            # block_m=128 when bias is present (ties at small S, wins at large S).
            if has_bias:
                return 128, 256, "N32"
            return 64, 128, "N32"
    if num_heads >= 32 and batch * seq_len >= _DENSE_M256_MIN_TOKENS:
        return 256, 512, "auto"
    return 128, 256, "auto"


# Generic split-K uses BLOCK_M=64, so the "mtile" (Q rows per workgroup) is 64.
_GENERIC_SPLITK_BLOCK_M = 64

# Waves per CTA, from the launch geometry of each paged kernel (confirmed against
# Workgroup_Size_X in rocprofv3 kernel traces). Used to turn a workgroup count
# into a wave count when judging whether the device is full.
_LIGHT_WAVES_PER_CTA = 2  # flash_attn_generic, 128 threads
_DUALWAVE_WAVES_PER_CTA = 8  # flash_attn_dualwave_swp_gfx950, 512 threads

# Resident waves per CU to aim for. Matches the target in auto_split_k_hp
# (decode/pa_decode_dense.py) so the two split heuristics agree.
_TARGET_WAVES_PER_CU = 8


def _generic_splitk_list(i: int) -> int:
    # Mirror CK generate_splits_list: 1,2,4,8,16,32,64,96,128,... .
    if i <= 0:
        return 1
    if i <= 5:
        return 1 << (i - 1)
    return (i - 5) * 32


def _num_kv_splits_heuristic(
    num_batches: int,
    num_heads: int,
    seqlen_q: int,
    head_dim: int,
    num_cu: int,
    max_splits: int = 8,
    waves_per_cta: int = 0,
) -> int:
    """Port of CK get_num_kv_splits_heuristic for the generic BLOCK_M=64 kernel.

    Returns the number of KV splits (1 = no split-K). CK varies the mtile by
    head-dim, but the generic kernel always uses BLOCK_M=64, so the occupancy
    estimate uses mtile=64 (or 16 for tiny q, matching CK's smallq branch).

    ``waves_per_cta`` makes the "is the device full?" test count **waves** rather
    than workgroups. Counting workgroups silently assumes every kernel has the
    same CTA width, which is false here: the light kernel is 128 threads (2
    waves), dualwave is 512 (8). At B=8/H=32 the workgroup test sees
    ``256 >= 0.9*256`` and declines to split, but those are 2-wave CTAs --
    measured ``SQ_WAVES = 512`` on 256 CUs, i.e. 2 waves/CU, the most starved
    band in the benchmark grid. Left at 0 the original workgroup-based test is
    used, so existing callers are unaffected.
    """

    def ceildiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    if head_dim > 256:
        return 1

    # Enough Q tiles to fill ~all CUs at the default pipeline mtile -> no split.
    if seqlen_q >= _GENERIC_SPLITK_BLOCK_M:
        default_blocks = (
            num_batches * num_heads * ceildiv(seqlen_q, _GENERIC_SPLITK_BLOCK_M)
        )
        if default_blocks >= 0.8 * num_cu:
            return 1

    mtile = _GENERIC_SPLITK_BLOCK_M
    if seqlen_q <= 16:
        mtile = 16
    blocks = num_batches * num_heads * ceildiv(seqlen_q, mtile)
    if waves_per_cta > 0:
        if blocks * waves_per_cta >= _TARGET_WAVES_PER_CU * num_cu:
            return 1
    elif blocks >= 0.9 * num_cu:
        return 1

    max_splits = min(max_splits, num_cu)
    max_check = 1
    while _generic_splitk_list(max_check) <= max_splits:
        max_check += 1

    num_splits = 1
    for i in range(2, max_check):
        num_splits = _generic_splitk_list(i)
        if blocks * num_splits >= num_cu:
            break
    return num_splits


# ── build-cache helpers ────────────────────────────────────────────────────


@functools.lru_cache(maxsize=256)
def _build_dense(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    block_m: int,
    flat_work_group_size: int,
    path_tag: str,
    waves_per_eu: int,
    daz: bool,
    return_lse: bool = False,
    window_left: int = -1,
    sm_scale: Optional[float] = None,
    has_bias: bool = False,
    has_dropout: bool = False,
    num_kv_splits: int = 1,
):
    """Build (and cache) one dense generic launcher variant."""
    from .flash_attn_generic import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        cross_seqlen=cross_seqlen,
        block_m=block_m,
        flat_work_group_size=flat_work_group_size,
        path_tag=path_tag,
        waves_per_eu=waves_per_eu,
        daz=daz,
        return_lse=return_lse,
        window_left=window_left,
        sm_scale=sm_scale,
        has_bias=has_bias,
        has_dropout=has_dropout,
        num_kv_splits=num_kv_splits,
    )


@functools.lru_cache(maxsize=256)
def _build_dense_dualwave(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    debug_lazy_counts: bool,
    enable_stagger: bool,
    return_lse: bool = False,
):
    """Build (and cache) the dense gfx950 DUALWAVE_SWP launcher."""
    from .flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    return build_flash_attn_dualwave_swp_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        cross_seqlen=cross_seqlen,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_debug_lazy_counts=debug_lazy_counts,
        dualwave_swp_enable_stagger=enable_stagger,
        return_lse=return_lse,
    )


@functools.lru_cache(maxsize=128)
def _build_dense_fp8(
    num_heads: int,
    num_kv_heads: int,
    causal: bool,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    enable_stagger: bool,
):
    """Build (and cache) the dense gfx950 fp8 launcher."""
    from .flash_attn_fp8_gfx950 import build_flash_attn_dualwave_swp_fp8_module

    return build_flash_attn_dualwave_swp_fp8_module(
        num_heads=num_heads,
        head_dim=128,
        causal=causal,
        dtype_str="fp8",
        num_kv_heads=num_kv_heads,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_enable_stagger=enable_stagger,
    )


@functools.lru_cache(maxsize=256)
def _build_varlen(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    debug_lazy_counts: bool,
    enable_stagger: bool,
    return_lse: bool = False,
):
    """Build (and cache) a varlen-mode launcher (gfx950 DUALWAVE_SWP, varlen=True)."""
    from .flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    return build_flash_attn_dualwave_swp_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        varlen=True,
        cross_seqlen=cross_seqlen,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_debug_lazy_counts=debug_lazy_counts,
        dualwave_swp_enable_stagger=enable_stagger,
        return_lse=return_lse,
    )


@functools.lru_cache(maxsize=256)
def _build_varlen_light(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    debug_lazy_counts: bool,
    enable_stagger: bool,
    return_lse: bool = False,
    causal_top_left: bool = False,
    window_left: int = -1,
    sm_scale: Optional[float] = None,
    gappy_kv: bool = False,
    num_kv_splits: int = 1,
):
    """Build a lightweight packed-varlen launcher for short attention."""
    from .flash_attn_generic import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        sm_scale=sm_scale,
        num_kv_heads=num_kv_heads,
        cross_seqlen=cross_seqlen,
        varlen=True,
        block_m=64,
        flat_work_group_size=128,
        waves_per_eu=waves_per_eu,
        daz=daz,
        return_lse=return_lse,
        causal_top_left=causal_top_left,
        window_left=window_left,
        gappy_kv=gappy_kv,
        num_kv_splits=num_kv_splits,
    )


@functools.lru_cache(maxsize=256)
def _build_splitk(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    num_kv_splits: int,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    enable_stagger: bool,
    return_lse: bool = False,
):
    """Build (and cache) a split-K launcher (gfx950 DUALWAVE_SWP, num_kv_splits>1)."""
    from .flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    return build_flash_attn_dualwave_swp_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        num_kv_splits=num_kv_splits,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_enable_stagger=enable_stagger,
        return_lse=return_lse,
    )


@functools.lru_cache(maxsize=256)
def _build_paged(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    enable_stagger: bool,
    num_kv_splits: int = 1,
    varlen: bool = False,
    kv_cache_layout: str = "linear",
):
    """Build (and cache) a paged-KV launcher (gfx950 DUALWAVE_SWP, paged=True).

    ``num_kv_splits > 1`` builds the paged + split-K variant (KV dimension split
    across grid_z = B*num_kv_splits workgroups + a combine pass), which fills the
    GPU for low-occupancy shapes (small B / few heads).

    ``varlen=True`` builds the packed-Q (cu_seqlens) + paged-KV variant: Q/O are
    ``[total_q, H, D]`` and K/V are the physical page cache, looked up via the
    block table per kv-tile. Mutually exclusive with split-K.

    ``kv_cache_layout`` selects the physical page layout: "linear"
    [NumBlocks,PageSize,Hkv,D] or "vectorized" (aiter 5D).
    """
    from .flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    return build_flash_attn_dualwave_swp_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        paged=True,
        varlen=varlen,
        num_kv_splits=num_kv_splits,
        cross_seqlen=cross_seqlen,
        kv_cache_layout=kv_cache_layout,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_enable_stagger=enable_stagger,
    )


# ── paged-KV native path ────────────────────────────────────────────────────

# gfx950 dualwave paged-KV currently supports exactly one configuration.
_PAGED_PAGE_SIZE = 64
_PAGED_BT_LDS_SIZE = 2048

# Streaming tile of the head-packed decode kernel (TILE_N in decode/pa_decode_gfx950.py).
# A page must be a whole number of tiles so a tile never straddles two pages.
_PAGED_DECODE_TILE_N = 32

# Each KV split must own enough pages to be worth a workgroup + its combine pass.
_PAGED_MIN_PAGES_PER_SPLIT = 4

# Kill-switch for the automatic paged split-K selection below, mirroring
# `MSLK_DISABLE_SPLITK` in csrc/gemm/cutlass/mx6mx6bf16.cu. Forces
# `num_kv_splits=0` (auto) to resolve to 1 so the dense single-pass paged kernel
# is used, for A/B measurement and for bisecting regressions.
_DISABLE_PAGED_AUTO_SPLITK: bool = (
    os.environ.get("MSLK_DISABLE_PAGED_SPLITK", "0") != "0"
)

# Kill-switch for routing paged Sq=1 decode to the head-packed decode kernel
# (`decode/pa_decode_gfx950.py`). Set to fall back to the dualwave paged path,
# for A/B measurement and for bisecting regressions.
_DISABLE_PAGED_DECODE_HP: bool = (
    os.environ.get("MSLK_DISABLE_PAGED_DECODE_HP", "0") != "0"
)

# The head-packed decode kernel maps the MFMA M-axis to (query-token, head) pairs,
# so it needs `ratio` to divide MFMA_M and `ratio * Sq` to fit the M-tile budget.
# Mirrors MFMA_M / MAX_M_TILES_BY_HEAD_DIM in decode/pa_decode_gfx950.py.
_HP_MFMA_M = 16

# One CTA is one wave here, so CTA count is the wave count. Below this the kernel
# cannot hide memory latency: more M-tiles cost occupancy (9 waves/SIMD at 1 tile
# down to 2 at 8), and with too few CTAs there is nothing else resident to cover
# the stall. Measured at D=64/Sq=16: 1.0 CTA/CU ran 0.74x the path it replaced,
# while every shape at >= 4.0 CTA/CU won (1.27x-3.23x).
_HP_MIN_CTAS_PER_CU = 2

# Tile count at which the occupancy cost becomes worth guarding. Measured
# waves/SIMD at D=64: 9 at one tile, 6 at two, then 4 and below. Shallow tiling
# keeps enough waves resident to hide latency on its own, and Sq<=4 was measured
# winning at batch 1 (1.45x vs B200), so the floor must not reject it.
_HP_FLOOR_MIN_TILES = 3

# Query groups to use when one pass would exceed the register budget. Measured at
# D=128/Sq=16: G=2 is 1.92x-1.97x over single-pass, G=4 is slower than G=2 in 9
# of 10 shapes (the extra launches cost more than the occupancy they buy), so
# there is no reason to go beyond 2.
_HP_QUERY_GROUPS = 2

# Kill-switch for query grouping, mirroring MSLK_DISABLE_PAGED_DECODE_HP. With
# this set, shapes that only fit via grouping decline to the dualwave path as
# they did before.
_DISABLE_PAGED_QGROUPS: bool = os.environ.get("MSLK_DISABLE_PAGED_QGROUPS", "0") != "0"


def _hp_decode_ok(
    num_heads: int,
    num_kv_heads: int,
    seqlen_q: int,
    head_dim: int,
    num_batches: int,
    max_kv_pages: int,
    device,
) -> int:
    """How should this shape run on the head-packed decode kernel?

    Returns the number of query groups to split the block into, or 0 to decline.
    1 is the ordinary single pass.

    Three derived gates -- no tuned table:

    1. **Register budget.** `ratio * Sq` pairs must fit in `MFMA_M` slots across
       at most `max_m_tiles(head_dim)` tiles, the measured point before the
       compiler spills.
    2. **Query grouping.** A block too deep for that budget can instead be run as
       `_HP_QUERY_GROUPS` shallower passes (see `query_group_seqlen`). Costs one
       extra KV pass per group but avoids the spill, measured 1.5x-2.0x at
       D=128/Sq=16. Only used to rescue a shape the budget would reject: where
       single-pass already fits, it was inside run-to-run noise (1.03x-1.28x at
       D=64) and is not worth the extra traffic.
    3. **Parallelism floor.** The resulting CTA count must reach
       `_HP_MIN_CTAS_PER_CU * CUs`. More tiles buy fewer KV passes but cost
       occupancy, and that trade only pays when enough CTAs are resident.
    """
    from .decode.pa_decode_dense import auto_split_k_hp
    from .decode.pa_decode_gfx950 import max_m_tiles

    if _DISABLE_PAGED_QGROUPS:
        groups_allowed = 1
    else:
        groups_allowed = _HP_QUERY_GROUPS

    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        return 0
    ratio = num_heads // num_kv_heads
    if not (1 <= ratio <= _HP_MFMA_M) or _HP_MFMA_M % ratio != 0:
        return 0
    t_pack = _HP_MFMA_M // ratio  # query tokens per M-tile
    budget = t_pack * max_m_tiles(head_dim)

    groups = 1
    if seqlen_q > budget:
        # Too deep for one pass: can grouping bring it under the budget?
        if (
            groups_allowed > 1
            and seqlen_q % groups_allowed == 0
            and seqlen_q // groups_allowed <= budget
        ):
            groups = groups_allowed
        else:
            return 0
    if seqlen_q < 1:
        return 0

    # Only deep tiling trades enough occupancy away to need the floor; shallow
    # tiling still leaves 6+ waves/SIMD resident. Grouping makes each pass
    # shallower, so the depth that matters is the per-group one.
    tiles = -(-(seqlen_q // groups) // t_pack)
    if tiles < _HP_FLOOR_MIN_TILES:
        return groups
    num_cu = _dense_light_cu(device)
    split_k = auto_split_k_hp(
        num_batches, 1, num_heads, num_kv_heads, max_kv_pages * _PAGED_PAGE_SIZE
    )
    ctas = num_batches * num_kv_heads * split_k
    return groups if ctas >= _HP_MIN_CTAS_PER_CU * num_cu else 0


def _auto_paged_kv_splits(
    *,
    num_batches: int,
    num_heads: int,
    seqlen_q: int,
    head_dim: int,
    max_kv_pages: int,
    dtype_str: str,
    device,
) -> int:
    """Pick ``num_kv_splits`` for the dense paged path from occupancy.

    Decode shapes (``Sq`` of 1..16 against a long paged cache) produce only
    ``B * H`` workgroups, which leaves most CUs idle: at B=1/H=32 that is 32 of
    256 CUs on MI350X. The KV dimension is the only parallelism left, so reuse
    the same occupancy heuristic the dense path already applies
    (``_num_kv_splits_heuristic``) instead of silently running single-split.

    Returns 1 when the shape is ineligible or already fills the device, so the
    caller can treat the result as unconditional.
    """
    if _DISABLE_PAGED_AUTO_SPLITK:
        return 1
    if head_dim not in (64, 128) or dtype_str not in ("bf16", "f16"):
        return 1
    # Judge occupancy in waves: at splits<=1 the paged path runs the light kernel
    # (_paged_light_ok), whose CTAs are 2 waves, so a workgroup count understates
    # how empty the device is by 4x against the 8 waves/CU target.
    splits = _num_kv_splits_heuristic(
        num_batches,
        num_heads,
        seqlen_q,
        head_dim,
        _dense_light_cu(device),
        waves_per_cta=_LIGHT_WAVES_PER_CTA,
    )
    # Never split finer than the cache can feed: each split needs its own pages.
    splits = min(splits, max(1, max_kv_pages // _PAGED_MIN_PAGES_PER_SPLIT))
    return max(1, splits)

# Paged split-K sizing. `_num_kv_splits_heuristic` (CK's) stops as soon as the
# workgroup count covers the CUs once, because it assumes a workgroup that has
# started is a workgroup making progress. That does not hold for the paged light
# kernel: each workgroup walks its KV range as a serial chain of dependent loads
# at roughly one memory latency per BLOCK_N_OUT tile, so a full GPU can still be
# idle-stalled. Size by chain length instead, then bound the workgroup count so
# the fp32 workspace and combine pass stay cheap.
_PAGED_BLOCK_N_OUT = 64  # generic paged builds with path_tag="N32"
# Target KV tiles per workgroup. Measured on MI350X (gfx950, bf16, ctx 32k-128k,
# D=64/128): the optimum sits at 16 tiles across r=1..64 and q_len=1..16, and the
# curve is flat between 8 and 32 before combine overhead takes over past ~64.
_PAGED_TARGET_CHAIN = int(os.getenv("FLYDSL_PAGED_TARGET_CHAIN", "8"))
_PAGED_MAX_SPLITS = int(os.getenv("FLYDSL_PAGED_MAX_SPLITS", "64"))
# A base grid this small cannot fill the device even at MAX_SPLITS, so it is
# allowed twice the cap. Measured at r=1 (4 base blocks after GQA packing):
# 128 splits beats 64 by 27% at ctx=512000 and 3% at ctx=128000, while at
# ctx=32000 the chain is already short enough that the split cost dominates.
_PAGED_STARVED_BLOCKS = int(os.getenv("FLYDSL_PAGED_STARVED_BLOCKS", "8"))
# Ceiling on total workgroups (blocks * splits). Past this the combine pass and
# scheduling cost outweigh the shorter chain; measured knee at r=64 on MI350X.
_PAGED_MAX_WG = int(os.getenv("FLYDSL_PAGED_MAX_WG", "8192"))
# The fp32 split-K workspace is B*splits*H*Sq*(D/2+2) elements; cap it so a
# large-batch speculative-decode call cannot quietly allocate gigabytes.
_PAGED_WS_BUDGET_MB = int(os.getenv("FLYDSL_PAGED_WS_BUDGET_MB", "512"))
# Debug/tuning override: force an exact split count (0 = use the heuristic).
_PAGED_FORCE_SPLITS = int(os.getenv("FLYDSL_PAGED_FORCE_SPLITS", "0"))
# Fold GQA query heads into the M dimension at q_len==1 (see the pack block in
# _flydsl_flash_attn_paged). Set to 0 to fall back to one workgroup per query head.
_PAGED_GQA_PACK = os.getenv("FLYDSL_PAGED_GQA_PACK", "1") == "1"
_PAGED_LIGHT_BLOCK_M = int(os.getenv("FLYDSL_PAGED_BLOCK_M", "64"))


@functools.lru_cache(maxsize=256)
def _uniform_cu_seqlens(count: int, step: int, device: str) -> torch.Tensor:
    """Cached prefix sums for uniform sequence lengths.

    Pure function of (count, step, device), so caching is safe. Saves an arange
    dispatch per call; decode replays the same shape indefinitely.
    """
    return torch.arange(0, (count + 1) * step, step, dtype=torch.int32, device=device)


# Two-pass MTP packing (q_len > 1). See _paged_mtp_two_pass.
_PAGED_MTP_PACK = os.getenv("FLYDSL_PAGED_MTP_PACK", "0") == "1"
# Below this much distinct KV the fixed cost of the tail pass outweighs the
# bandwidth it saves. Measured on gfx950 over the full matrix: the 125 MiB band
# regresses (0.82x geomean, worst 0.75x) while every case at >=126 MiB gains
# (1.19x geomean at 126-500 MiB, rising to 3.21x above 2 GiB). 192 sits between
# the two with margin.
_PAGED_MTP_MIN_KV_MB = int(os.getenv("FLYDSL_PAGED_MTP_MIN_KV_MB", "192"))


def _paged_mtp_two_pass(
    q,
    k,
    v,
    *,
    block_table,
    seqlen_k,
    num_kv_heads,
    page_size,
    q_len,
    batch,
    heads,
    out,
    max_seqlen_kv,
    sm_scale,
    kv_cache_layout,
    waves_per_eu,
    daz,
    dualwave_swp_lazy_rescale,
    dualwave_swp_setprio,
    dualwave_swp_enable_stagger,
    stream,
):
    """Bottom-right causal paged attention at q_len>1, without the GQA fan-out.

    The grid is one workgroup per (request, QUERY head), so every query head in
    a GQA group re-streams its KV head's whole range: measured 7.3 TB/s of
    issued loads for 0.9 TB/s of distinct DRAM traffic at q_len=4, against
    5.2 TB/s when the same kernel runs MHA and has no fan-out to pay.

    Packing the group's query heads into M fixes that, but only at q_len==1 --
    with several query tokens the packed (head, token) rows need a mask no
    bottom-right causal can express. So split the KV range instead:

      pass 1  rows = (head, token) packed, **non-causal** over ``[0, kv-q_len)``
              -- every packed row sees exactly that prefix, so the mask is
              uniform and the fan-out is gone. Returns partial O and LSE.
      pass 2  the last ``q_len`` keys only, causal, ``q_len x q_len`` per
              request. Tiny, and done in torch.
      merge   standard log-sum-exp combine of the two partials.

    Worth 4.3x at r=16/ctx=128k/q_len=4 (4.54 -> 1.06 ms, 0.92 -> 3.96 TB/s)
    and 2.6x at q_len=16. Callers gate on ``_PAGED_MTP_MIN_KV_MB``; below that
    the tail pass dominates and this is a slowdown.
    """
    group = heads // num_kv_heads
    rows = q_len * group
    head_dim = q.shape[-1]
    dev = q.device

    # ---- pass 1: packed (head, token) rows over the shared prefix ----
    # Works for both layouts: dense [B, q_len, H, D] and varlen [B*q_len, H, D]
    # have the same element order.
    packed_q = (
        q.view(batch, q_len, num_kv_heads, group, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch * rows, num_kv_heads, head_dim)
        .contiguous()
    )
    prefix_k = (seqlen_k.to(torch.int32) - q_len).clamp_min(0)
    out1, lse1 = _flydsl_flash_attn_paged(
        packed_q,
        k,
        v,
        causal=False,
        num_kv_heads=num_kv_heads,
        block_table=block_table,
        seqlen_k=prefix_k,
        max_seqlen_kv=int(max_seqlen_kv) - q_len,
        kv_cache_layout=kv_cache_layout,
        cu_seqlens_q=torch.arange(
            0, (batch + 1) * rows, rows, dtype=torch.int32, device=dev
        ),
        cu_seqlens_kv=torch.nn.functional.pad(
            prefix_k.cumsum(0, dtype=torch.int32), (1, 0)
        ),
        kv_seqstart=None,
        max_seqlen_q=rows,
        cross_seqlen=True,
        num_kv_splits=1,
        return_lse=True,
        sm_scale=sm_scale,
        out=None,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=dualwave_swp_lazy_rescale,
        dualwave_swp_setprio=dualwave_swp_setprio,
        dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
        stream=stream,
    )
    o1 = (
        out1.view(batch, q_len, group, num_kv_heads, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, q_len, heads, head_dim)
        .float()
    )
    l1 = (
        lse1.view(batch, num_kv_heads, q_len, group)
        .permute(0, 2, 1, 3)
        .reshape(batch, q_len, heads)
        .float()
    )

    # ---- pass 2: the last q_len keys, causal ----
    steps = torch.arange(q_len, device=dev, dtype=torch.int32)
    tail = seqlen_k.to(torch.int32).view(batch, 1) - q_len + steps.view(1, q_len)
    phys = (
        block_table.gather(1, (tail // page_size).long()).long() * page_size
        + (tail % page_size).long()
    )
    kf = k.reshape(-1, num_kv_heads, head_dim)
    vf = v.reshape(-1, num_kv_heads, head_dim)
    kt, vt = kf[phys].float(), vf[phys].float()
    qf = q.view(batch, q_len, num_kv_heads, group, head_dim).float()
    scores = torch.einsum("rtgpd,rkgd->rtgpk", qf, kt) * sm_scale
    causal = steps.view(q_len, 1) >= steps.view(1, q_len)
    scores = scores.masked_fill(~causal[None, :, None, None, :], -float("inf"))
    m2 = scores.amax(-1)
    p2 = torch.exp(scores - m2[..., None])
    denom = p2.sum(-1)
    o2 = (torch.einsum("rtgpk,rkgd->rtgpd", p2, vt) / denom[..., None]).reshape(
        batch, q_len, heads, head_dim
    )
    l2 = (m2 + torch.log(denom)).reshape(batch, q_len, heads)

    # ---- merge ----
    top = torch.maximum(l1, l2)
    w1, w2 = torch.exp(l1 - top), torch.exp(l2 - top)
    merged = ((o1 * w1[..., None] + o2 * w2[..., None]) / (w1 + w2)[..., None]).to(
        q.dtype
    )
    if out is not None:
        out.view(batch, q_len, heads, head_dim).copy_(merged)
        return out
    return merged.reshape(q.shape)


def _paged_num_kv_splits(
    num_batches: int, num_heads: int, seqlen_q: int, seqlen_kv: int, head_dim: int
) -> int:
    """KV splits for the generic paged (light) kernel.

    Sized by chain length, not by workgroup count. The KV loop is a serial chain
    of dependent loads costing roughly one memory latency per BLOCK_N_OUT tile,
    so a grid that already covers every CU can still be latency-stalled -- at
    r=64 (2048 workgroups, 8 per CU) splitting 32 ways is still worth 22%.
    Pick the smallest split count that brings the chain to _PAGED_TARGET_CHAIN
    tiles, then clamp to the workspace budget.
    """
    if _PAGED_FORCE_SPLITS > 0:
        return _PAGED_FORCE_SPLITS

    def ceildiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    kv_tiles = ceildiv(max(seqlen_kv, 1), _PAGED_BLOCK_N_OUT)
    if kv_tiles <= _PAGED_TARGET_CHAIN:
        return 1  # chain is already short; splitting only adds a combine pass

    blocks = num_batches * num_heads * ceildiv(max(seqlen_q, 1), 64)
    cap = _PAGED_MAX_SPLITS * (2 if blocks < _PAGED_STARVED_BLOCKS else 1)
    want = min(ceildiv(kv_tiles, max(_PAGED_TARGET_CHAIN, 1)), cap)

    # Don't let the grid blow past the workgroup ceiling: with a large base grid
    # (high concurrency) the extra splits stop buying latency hiding and the
    # combine pass starts to dominate.
    by_grid = max(1, _PAGED_MAX_WG // max(blocks, 1))

    # rows = B*splits*H*Sq; elems = rows*(D//2) + 2*rows  (see
    # dualwave_splitk_workspace_elems). Solve for the largest affordable split.
    per_split_elems = num_batches * num_heads * max(seqlen_q, 1) * (head_dim // 2 + 2)
    budget_elems = _PAGED_WS_BUDGET_MB * 1024 * 1024 // 4
    affordable = max(1, budget_elems // max(per_split_elems, 1))
    return max(1, min(want, by_grid, affordable, kv_tiles))


def _flydsl_flash_attn_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    num_kv_heads: Optional[int],
    block_table: Optional[torch.Tensor],
    seqlen_k: Optional[torch.Tensor],
    max_seqlen_kv: Optional[int],
    kv_cache_layout: str,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_kv: Optional[torch.Tensor],
    kv_seqstart: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    cross_seqlen: Optional[bool],
    num_kv_splits: int,
    return_lse: bool,
    sm_scale: Optional[float],
    out: Optional[torch.Tensor],
    waves_per_eu: int,
    daz: bool,
    dualwave_swp_lazy_rescale: bool,
    dualwave_swp_setprio: bool,
    dualwave_swp_enable_stagger: bool,
    stream,
) -> torch.Tensor:
    """Native paged-KV attention on the gfx950 dualwave kernel.

    Supported config ONLY (anything else raises): linear/vectorized cache layout
    [NumBlocks, PageSize=64, NumKVHeads, HeadDim], vLLM lookup (block_table +
    seqlen_k), causal, D=64/128, dtype bf16/f16.
    - Dense 4D Q ``[B, Sq, H, D]``: split-K supported at any Sq. ``num_kv_splits=0``
      selects a count from occupancy; ``1`` disables; ``>1`` forces.
    - Varlen packed Q ``[total_q, H, D]`` (cu_seqlens_q given): paged K/V looked up
      per kv-tile via block_table; split-K not supported (matches dense varlen).
    """
    if kv_cache_layout not in ("linear", "vectorized"):
        raise NotImplementedError(
            f"flydsl_flash_attn_func: native paged KV supports kv_cache_layout in ('linear','vectorized'), "
            f"got {kv_cache_layout!r}"
        )
    # Gappy paged uses kv_seqstart + cu_seqlens_kv (per-seq lengths) instead of the
    # vLLM seqlen_k bound.
    if block_table is None or (seqlen_k is None and kv_seqstart is None):
        raise ValueError(
            "flydsl_flash_attn_func: native paged KV (vllm) requires block_table and "
            "seqlen_k (or kv_seqstart for gappy paged)"
        )
    vectorized = kv_cache_layout == "vectorized"
    if vectorized:
        # aiter 5D: K [NumBlocks, Hkv, D/kVS, PageSize, kVS], V [NumBlocks, Hkv, PageSize/kVS, D, kVS].
        if k.dim() != 5 or v.dim() != 5:
            raise ValueError(
                f"flydsl_flash_attn_func: vectorized paged K/V must be 5D, got K{k.dim()}D V{v.dim()}D"
            )
    elif k.dim() != 4:
        raise ValueError(
            f"flydsl_flash_attn_func: linear paged K/V must be 4D [NumBlocks,PageSize,Hkv,D], got {k.dim()}D"
        )

    varlen = cu_seqlens_q is not None
    dtype_str = _dtype_str(q)
    if varlen:
        # Packed varlen Q: [total_q, H, D]. Per-batch ranges come from cu_seqlens
        # inside the kernel; grid_y is sized by max_seqlen_q.
        if cu_seqlens_kv is None:
            raise ValueError(
                "flydsl_flash_attn_func: varlen paged KV requires cu_seqlens_kv"
            )
        if max_seqlen_q is None:
            raise ValueError(
                "flydsl_flash_attn_func: varlen paged KV requires max_seqlen_q"
            )
        if q.dim() != 3:
            raise ValueError(
                f"flydsl_flash_attn_func: varlen paged q must be 3D [total_q,H,D], got {q.dim()}D"
            )
        _total_q, H, D = q.shape
        B = cu_seqlens_q.numel() - 1
        Sq = int(max_seqlen_q)
        # Split-K's combine pass addresses O as a dense [B, max_seqlen_q, H, D]
        # block (it is handed batch_size/seq_len, not cu_seqlens), which matches
        # the packed varlen layout only when every sequence has the same length.
        # sum(q_seqlens) == B*max_seqlen_q iff they are all equal, so this is an
        # exact test and needs no host sync. Ragged q with split-K silently
        # returns garbage (measured rel_err ~1.0), hence the hard error.
        _varlen_uniform_q = _total_q == B * Sq
        if num_kv_splits > 1 and not _varlen_uniform_q:
            raise NotImplementedError(
                "flydsl_flash_attn_func: varlen paged split-K requires uniform "
                f"q_seqlens; got total_q={_total_q} for B={B}, max_seqlen_q={Sq}"
            )
    else:
        _varlen_uniform_q = True
        if q.dim() != 4:
            raise ValueError(
                f"flydsl_flash_attn_func: paged dense q must be 4D [B,Sq,H,D], got {q.dim()}D"
            )
        B, Sq, H, D = q.shape

    # ── GQA head packing ────────────────────────────────────────────────────
    # The grid is one workgroup per (request, QUERY head), so the group_size
    # query heads sharing a KV head each stream that head's entire KV range
    # independently. Measured on GQA 32x4: 7.3 TB/s of issued loads for
    # 0.9 TB/s of distinct DRAM traffic, while the same kernel at MHA 32x32
    # reaches 5.2 TB/s DRAM -- it is L2-bound on redundant reads, not slow.
    #
    # Fold the group's query heads into M instead: one workgroup per
    # (request, KV head), KV read once and shared across the group. Rows are
    # laid out head-major, row = h * q_len + t, and the kernel is built with
    # Q_PACK_QLEN = q_len so its causal bound uses t = row % q_len rather than
    # the row index. Without that trait this is only expressible at q_len == 1.
    _gqa_packed = False
    _gqa_group = 0
    _gqa_qlen = 0
    _gqa_B = B
    _gqa_heads = H
    _gqa_varlen = varlen
    _gqa_user_out = None
    if (
        _PAGED_GQA_PACK
        and not return_lse
        and kv_seqstart is None
        and 1 <= Sq <= 64
        and _varlen_uniform_q
        and num_kv_heads is not None
        and num_kv_heads > 0
        and H > num_kv_heads
        and H % num_kv_heads == 0
        and (Sq == 1 or causal)
    ):
        _gqa_group = H // num_kv_heads
        _gqa_qlen = Sq
        rows = _gqa_group * Sq
        # [.., q_len, Hkv, group, D] -> [.., group, q_len, Hkv, D]; the flat M
        # index is h * q_len + t, matching Q_PACK_QLEN's row % q_len.
        q = (
            q.view(B, Sq, num_kv_heads, _gqa_group, D)
            .permute(0, 3, 1, 2, 4)
            .reshape(
                (B * rows, num_kv_heads, D) if varlen else (B, rows, num_kv_heads, D)
            )
            .contiguous()
        )
        if varlen:
            cu_seqlens_q = _uniform_cu_seqlens(B, rows, str(q.device))
            max_seqlen_q = rows
            _total_q = B * rows
        # The caller's `out` is in the unpacked layout, so compute into a fresh
        # packed buffer and write the result back at the end.
        _gqa_user_out, out = out, None
        _gqa_heads = H
        H = num_kv_heads
        Sq = rows
        _gqa_packed = True

    if vectorized:
        kvs = 16 // q.element_size()
        Hkv = int(k.shape[1])
        page_size = int(k.shape[3])
        k_head_dim = int(k.shape[2]) * int(k.shape[4])  # (D/kVS) * kVS
        if int(k.shape[4]) != kvs:
            raise ValueError(
                f"flydsl_flash_attn_func: vectorized K last dim ({k.shape[4]}) must equal kVS={kvs}"
            )
    else:
        page_size = int(k.shape[1])
        Hkv = int(k.shape[2])
        k_head_dim = int(k.shape[3])
    # Gappy paged uses the generic kernel (any D the generic path builds; the op
    # reshapes the cache to 64-row sub-pages), so skip the native-paged D/page gates.
    _gappy_paged = kv_seqstart is not None
    if page_size != _PAGED_PAGE_SIZE and not _gappy_paged:
        raise NotImplementedError(
            f"flydsl_flash_attn_func: native paged KV supports page_size={_PAGED_PAGE_SIZE} only, got {page_size}"
        )
    if D not in (64, 128) and not _gappy_paged:
        raise NotImplementedError(
            f"flydsl_flash_attn_func: native paged KV supports head_dim=64 or 128, got {D}"
        )
    if k_head_dim != D:
        raise ValueError(
            f"flydsl_flash_attn_func: paged K head_dim ({k_head_dim}) must match q head_dim ({D})"
        )

    if num_kv_heads is None:
        num_kv_heads = Hkv
    if H % num_kv_heads != 0:
        raise ValueError(
            f"flydsl_flash_attn_func: num_heads ({H}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    # ── MTP two-pass (q_len > 1) ────────────────────────────────────────────
    # Same motivation as the q_len==1 packing below -- kill the GQA fan-out --
    # but q_len>1 needs the KV range split to keep the mask expressible. See
    # _paged_mtp_two_pass. Gated on distinct-KV size: below the threshold the
    # tail pass costs more than the bandwidth it saves.
    if (
        _PAGED_MTP_PACK
        and 1 < Sq <= 64
        and _varlen_uniform_q
        and not return_lse
        and kv_seqstart is None
        and num_kv_heads is not None
        and num_kv_heads > 0
        and H > num_kv_heads
        and H % num_kv_heads == 0
        and D in (64, 128)
        and dtype_str in ("bf16", "f16")
        and max_seqlen_kv is not None
        and int(max_seqlen_kv) > 2 * Sq
        and seqlen_k is not None
        and causal
    ):
        bpe = q.element_size()
        kv_mib = 2 * B * int(max_seqlen_kv) * num_kv_heads * D * bpe / 2**20
        if kv_mib >= _PAGED_MTP_MIN_KV_MB:
            return _paged_mtp_two_pass(
                q,
                k,
                v,
                block_table=block_table,
                seqlen_k=seqlen_k,
                num_kv_heads=num_kv_heads,
                page_size=page_size,
                q_len=Sq,
                batch=B,
                heads=H,
                out=out,
                max_seqlen_kv=max_seqlen_kv,
                sm_scale=(
                    float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(D)
                ),
                kv_cache_layout=kv_cache_layout,
                waves_per_eu=waves_per_eu,
                daz=daz,
                dualwave_swp_lazy_rescale=dualwave_swp_lazy_rescale,
                dualwave_swp_setprio=dualwave_swp_setprio,
                dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
                stream=stream,
            )

    # ── route + split-K sizing ──────────────────────────────────────────────
    # Two paged kernels: the generic "light" one and the gfx950 dualwave one. The
    # dualwave paged softmax overflows for f16 (narrow exponent) and on the causal
    # path at large logits, so those route to light regardless of seqlen; bf16
    # non-causal keeps the faster dualwave kernel. Decided here (rather than at the
    # launch site) because split-K sizing depends on which kernel runs.
    _arch = _gpu_arch(q.device)
    _gappy = kv_seqstart is not None
    _splitk_dtype_ok = D in (64, 128) and dtype_str in ("bf16", "f16")
    _paged_light_ok = _gappy or (
        _splitk_dtype_ok
        and (
            dtype_str == "f16"
            or causal
            or not _arch.startswith("gfx950")
            or Sq <= _VARLEN_LIGHT_MAX_SEQ
        )
    )

    # Split-K partitions the KV loop across extra workgroups + a combine pass. A
    # decode-shaped paged call (Sq=1..16) launches only B*H workgroups and each one
    # walks the whole context as a serial ~ctx/BLOCK_N_OUT chain of dependent
    # loads, so wall time is (ctx/64) x memory latency no matter how idle the GPU
    # is. Splitting shortens that chain, which is the entire lever on these shapes.
    # No caller sizes num_kv_splits for paged, so do it here.
    if (
        num_kv_splits <= 1
        and _paged_light_ok
        and _splitk_dtype_ok
        and not _gappy
        and _varlen_uniform_q
    ):
        _kv_tiles = (
            int(max_seqlen_kv) if max_seqlen_kv is not None else int(seqlen_k.max())
        )
        num_kv_splits = _paged_num_kv_splits(B, H, Sq, _kv_tiles, D)

    splitk = num_kv_splits > 1
    # The dualwave split-K route needs enough Q rows to amortise its pipeline; the
    # generic light route has no such floor, and short-q is exactly where splitting
    # pays. Keep the old requirement only for the route it was written for.
    if splitk and not _splitk_dtype_ok:
        raise ValueError(
            f"flydsl_flash_attn_func: paged split-K requires D=64/128, dtype "
            f"bf16/f16; got D={D}, dtype={dtype_str}"
        )
    if splitk and not _paged_light_ok and Sq < 384:
        raise ValueError(
            f"flydsl_flash_attn_func: dualwave paged split-K requires seq_len>=384; "
            f"got seq_len={Sq}"
        )

    # Per-batch KV lengths differ in general → bottom-right cross-length masking. Varlen
    # paged always uses cross masking (per-batch seqlen_q/seqlen_kv come from cu_seqlens).
    skv = (
        int(max_seqlen_kv) if max_seqlen_kv is not None else int(seqlen_k.max().item())
    )
    max_kv_pages = (skv + page_size - 1) // page_size

    # ── Paged decode fast path (short query blocks) ───────────────────────────
    # The dualwave kernel below maps the MFMA M-axis to query *rows*, so at Sq=1
    # only 1 of 32 M-rows carries work: measured on MI350X it issues 64x the MFMA
    # instructions of an equivalent head-packed kernel for the same maths, and
    # spends ~50% of its stall cycles on the barriers needed to assemble a tile
    # that is 31/32 padding. `decode/pa_decode_gfx950.py` packs (query-token,
    # head) pairs onto M instead, so the matrix core stays full.
    #
    # `_hp_decode_ok` bounds that: M holds `ratio * Sq` pairs across at most
    # `_HP_MAX_M_TILES` tiles, so at GQA ratio 8 this covers Sq <= 4. Longer
    # query blocks keep the dualwave path until the M-tile budget is raised.
    #
    # Everything else excluded here also keeps the dualwave path: varlen and
    # gappy have no decode kernel, `return_lse` is not exposed by it, and the
    # vectorized cache layout is a different memory format.
    #
    # `causal` is deliberately not a condition: the decode kernel applies the
    # bottom-right causal bound per query token itself (query i attends to
    # [0, seqlen_kv - Sq + i + 1)), which degenerates to the full range at Sq=1.
    _hp_groups = 0
    if (
        not _DISABLE_PAGED_DECODE_HP
        # The GQA head-packing above removes the same fan-out this kernel was
        # routed here to avoid, and covers Sq up to 64 rather than the M-tile
        # budget's 4-16. When it fires it has already reshaped q, so this path
        # must not also run. It stays as the route for shapes that packing
        # declines (FLYDSL_PAGED_GQA_PACK=0, Sq>64, non-causal multi-token, MHA).
        and not _gqa_packed
        and not varlen
        and kv_seqstart is None
        and not return_lse
        and not vectorized
        and D in (64, 128)
        and dtype_str in ("bf16", "f16")
        and page_size % _PAGED_DECODE_TILE_N == 0
        and _gpu_arch(q.device).startswith("gfx950")
    ):
        # 0 declines; otherwise the number of query groups to cover Sq with.
        _hp_groups = _hp_decode_ok(H, num_kv_heads, Sq, D, B, max_kv_pages, q.device)
    if _hp_groups:
        from .decode.pa_decode_dense import pa_decode_paged_launch

        # Kernel Q layout is [B, Sq, G, H_q, D]; the paged ABI has no G axis (G=1).
        hp_out = pa_decode_paged_launch(
            q.view(B, Sq, 1, H, D),
            k,
            v,
            block_table,
            seqlen_k,
            float(sm_scale) if sm_scale is not None else float(D**-0.5),
            page_size=page_size,
            max_seqlen_kv=skv,
            split_k=0 if num_kv_splits == 0 else num_kv_splits,
            output_dtype=q.dtype,
            query_groups=_hp_groups,
        ).view(B, Sq, H, D)
        if out is not None:
            out.copy_(hp_out)
            return out
        return hp_out

    # Split-K (paged, dense only): split the KV dimension across grid_z = B*num_kv_splits
    # workgroups + a combine pass. Fills the GPU for low-occupancy shapes (small B / few
    # heads), where single-split paged underutilizes the device.
    #
    # `num_kv_splits == 0` means "choose for me" (same sentinel as
    # `pa_decode_launch(split_k=0)` and `mx6mx6bf16(splits=0)`). Auto-selection is
    # restricted to the dense native-paged kernel:
    #   - varlen packed Q has no split-K variant (rejected below), and
    #   - gappy paged and `return_lse` both require the generic light kernel, which
    #     `_paged_light_ok` only selects when num_kv_splits <= 1.
    # Anything else keeps the caller's explicit value, so `num_kv_splits=1` remains
    # an exact opt-out.
    # `_paged_num_kv_splits` above has already sized this for the paged routes it
    # covers, and it sizes by KV-chain length rather than occupancy -- a grid that
    # fills every CU can still be latency-stalled, which occupancy cannot see. Only
    # resolve the `0` sentinel for what it left alone, and never overwrite a count
    # it chose: doing so pushed shapes off the light kernel onto dualwave, which
    # carries a masking defect at non-tile-aligned seqlen_k (1e-2, above bf16 eps).
    if num_kv_splits == 0:
        num_kv_splits = 1

    splitk = num_kv_splits > 1
    # NOTE: the dense (non-paged) path additionally requires seq_len >= 384. That floor
    # does not apply here: the dualwave native-paged kernel splits along KV, not Q, so
    # short-Q decode shapes are exactly the ones that need it. Verified on MI350X across
    # B in {1,2,4}, Sq in {1,4,16}, D in {64,128}, ctx in {32k,128k}: max deviation from
    # the single-split result was 2e-4, well inside bf16 epsilon (~7.8e-3).
    if splitk and (D not in (64, 128) or dtype_str not in ("bf16", "f16")):
        raise ValueError(
            f"flydsl_flash_attn_func: paged split-K requires D=64/128, dtype bf16/f16; "
            f"got D={D}, dtype={dtype_str}"
        )
    max_pages_per_split = (max_kv_pages + int(num_kv_splits) - 1) // int(num_kv_splits)
    if max_pages_per_split > _PAGED_BT_LDS_SIZE:
        max_supported_kv = _PAGED_BT_LDS_SIZE * int(num_kv_splits) * page_size
        raise NotImplementedError(
            f"flydsl_flash_attn_func: paged KV length {skv} exceeds block-table LDS window "
            f"({_PAGED_BT_LDS_SIZE} pages/split, max_kv_len={max_supported_kv} for "
            f"num_kv_splits={num_kv_splits}, page_size={page_size})"
        )
    if varlen:
        cross = bool(cross_seqlen) if cross_seqlen is not None else True
    else:
        cross = skv != Sq
    block_table_stride = int(block_table.shape[1])
    # Flatten so the kernel's flat row-major index addresses block_table correctly.
    block_table_i32 = (
        (
            block_table
            if block_table.dtype == torch.int32
            else block_table.to(torch.int32)
        )
        .contiguous()
        .reshape(-1)
    )

    with torch.cuda.device(q.device.index):
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        # Route (_paged_light_ok) and split count were decided above, before the
        # block-table LDS check, since that check is per-split.
        if return_lse and not _paged_light_ok:
            # Only the generic light path produces LSE; the dualwave native-paged
            # kernel does not. (Short-q / gfx942 / paged-gappy take the light path.)
            raise NotImplementedError(
                "flydsl_flash_attn_func: return_lse for native paged KV is only "
                "supported on the generic (short-seq / gfx942) path"
            )
        if _paged_light_ok:
            exe = _build_paged_light(
                num_heads=H,
                num_kv_heads=num_kv_heads,
                head_dim=D,
                causal=causal,
                dtype_str=dtype_str,
                cross_seqlen=cross,
                varlen=varlen,
                kv_cache_layout=kv_cache_layout,
                waves_per_eu=waves_per_eu,
                daz=daz,
                lazy_rescale=dualwave_swp_lazy_rescale,
                setprio=dualwave_swp_setprio,
                debug_lazy_counts=False,
                enable_stagger=dualwave_swp_enable_stagger,
                gappy_kv=_gappy,
                return_lse=return_lse,
                sm_scale=sm_scale,
                num_kv_splits=int(num_kv_splits),
                q_pack_qlen=_gqa_qlen if (_gqa_packed and causal) else 0,
                # Packing multiplies the M rows by the GQA group size. Rows that
                # overflow BLOCK_M spill into a second Q tile, and each tile
                # re-reads the whole KV range -- measured 4.08 -> 2.72 TB/s going
                # from 64 rows to 88. Widen the tile instead. The reverse costs
                # 19% when the rows do fit, so only widen when they do not.
                block_m=128 if (_gqa_packed and Sq > 64) else 0,
            )
        else:
            exe = _build_paged(
                num_heads=H,
                num_kv_heads=num_kv_heads,
                head_dim=D,
                causal=causal,
                dtype_str=dtype_str,
                cross_seqlen=cross,
                waves_per_eu=waves_per_eu,
                daz=daz,
                lazy_rescale=dualwave_swp_lazy_rescale,
                setprio=dualwave_swp_setprio,
                enable_stagger=dualwave_swp_enable_stagger,
                num_kv_splits=int(num_kv_splits),
                varlen=varlen,
                kv_cache_layout=kv_cache_layout,
            )
        if out is None:
            out = torch.empty_like(q)
        # Keep tensors in natural shape; flattening can overflow int32 C-ABI dims.
        # The paged kernel rebuilds per-batch/page descriptors from base pointers.
        q_flat = q.contiguous()
        k_flat = k.contiguous()
        v_flat = v.contiguous()
        o_flat = out.contiguous()
        kwargs = dict(
            block_table=block_table_i32,
            block_table_stride=block_table_stride,
            stream=launch_stream,
        )
        if varlen:
            kwargs["cu_seqlens_q"] = cu_seqlens_q
            kwargs["cu_seqlens_kv"] = cu_seqlens_kv
        if kv_seqstart is not None:
            kwargs["kv_seqstart"] = kv_seqstart
        if cross:
            kwargs["seq_len_kv"] = skv
        lse = (
            torch.empty((B, H, Sq), dtype=torch.float32, device=q.device)
            if return_lse
            else None
        )
        if return_lse:
            kwargs["lse"] = lse
        if splitk:
            ws_elems = dualwave_splitk_workspace_elems(
                B, H, Sq, int(num_kv_splits), head_dim=D
            )
            _ws = torch.empty(ws_elems, dtype=torch.float32, device=q.device)
            kwargs["workspace"] = _ws
        exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)
        if o_flat.data_ptr() != out.data_ptr():
            out.copy_(o_flat)

    if _gqa_packed:
        # Invert the pack permutation: (group, token, kv-head) -> (token, head).
        unpacked = out.view(_gqa_B, _gqa_group, _gqa_qlen, H, D).permute(0, 2, 3, 1, 4)
        out = unpacked.reshape(
            (_gqa_B * _gqa_qlen, _gqa_heads, D)
            if _gqa_varlen
            else (_gqa_B, _gqa_qlen, _gqa_heads, D)
        )
        if _gqa_user_out is not None:
            _gqa_user_out.view(out.shape).copy_(out)
            out = _gqa_user_out

    return (out, lse) if return_lse else out


@functools.lru_cache(maxsize=256)
def _build_paged_light(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    varlen: bool,
    kv_cache_layout: str,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    debug_lazy_counts: bool,
    enable_stagger: bool,
    gappy_kv: bool = False,
    return_lse: bool = False,
    sm_scale: Optional[float] = None,
    num_kv_splits: int = 1,
    q_pack_qlen: int = 0,
    block_m: int = 0,
):
    """Build a lightweight paged-varlen launcher for short attention.

    ``num_kv_splits > 1`` partitions the KV loop across grid.y. Decode-shaped
    paged calls (q_len 1-16) otherwise walk the whole context in one workgroup,
    which is latency-bound end to end; splitting shortens that serial chain.
    """
    from .flash_attn_generic import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        cross_seqlen=cross_seqlen,
        varlen=varlen,
        paged=True,
        kv_cache_layout=kv_cache_layout,
        block_m=block_m or _PAGED_LIGHT_BLOCK_M,
        flat_work_group_size=(block_m or _PAGED_LIGHT_BLOCK_M) * 2,
        path_tag="N32",
        waves_per_eu=waves_per_eu,
        daz=daz,
        gappy_kv=gappy_kv,
        return_lse=return_lse,
        sm_scale=sm_scale,
        num_kv_splits=num_kv_splits,
        q_pack_qlen=q_pack_qlen,
    )


# ── public API ─────────────────────────────────────────────────────────────


def flydsl_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    num_kv_heads: Optional[int] = None,
    # Varlen (packed cu_seqlens): pass both to enable the varlen path.
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    # Gappy KV: per-seq absolute KV start [B] into an un-repacked K/V store. With
    # cu_seqlens_kv giving per-seq lengths, the kernel gathers each seq in-place.
    kv_seqstart: Optional[torch.Tensor] = None,
    # Max per-batch Q seqlen (varlen only). Required for varlen to size grid_y
    # without synchronizing on cu_seqlens_q.
    max_seqlen_q: Optional[int] = None,
    # Max per-batch KV seqlen (varlen cross-attn only). Used to size the KV grid
    # when seqlen_q != seqlen_kv per batch.
    max_seqlen_kv: Optional[int] = None,
    # Whether per-batch Sq and Skv can differ. Dense mode infers this from shapes;
    # varlen mode requires it explicitly to choose the correct build variant.
    cross_seqlen: Optional[bool] = None,
    # Paged KV cache ABI: vLLM-style block_table + seqlen_k.
    block_table: Optional[torch.Tensor] = None,
    seqlen_k: Optional[torch.Tensor] = None,
    kv_cache_layout: str = "linear",
    # Split-K (gfx950 only, D=64/128, bf16/f16). The dense path additionally
    # requires seq_len >= 384. 0 = auto (paged dense only; see below), 1 = never
    # split, >1 = force that many splits.
    num_kv_splits: int = 0,
    # fp8 dense ABI: per-tensor descales for pre-quantized e4m3fn Q/K/V.
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    # Output tensor; allocated if None.
    out: Optional[torch.Tensor] = None,
    # Kernel build options.
    waves_per_eu: int = 2,
    daz: bool = True,
    dualwave_swp_lazy_rescale: bool = True,
    dualwave_swp_setprio: bool = True,
    dualwave_swp_enable_stagger: bool = True,
    # Debug: pass a pre-allocated float32[2] tensor to enable the lazy-rescale
    # branch counter (dualwave_swp_debug_lazy_counts=True). Only for dense mode.
    debug_counts: Optional[torch.Tensor] = None,
    # When True, also return per-row log-sum-exp (natural log, sm_scale folded in,
    # shape [B, H, Sq]; varlen: [B, H, max_seqlen_q]). Not supported for fp8/paged KV.
    return_lse: bool = False,
    # Sliding-window: keep KV in (q_row - window_left, q_row]; -1 disables.
    window_left: int = -1,
    # Softmax scale (None -> 1/sqrt(head_dim)); non-default forces the generic path.
    sm_scale: Optional[float] = None,
    # Additive bias broadcastable to [B, H, Sq, Skv], added to QK*scale.
    bias: Optional[torch.Tensor] = None,
    # Top-left causal (kv_col <= q_row) vs default bottom-right; cross-length only.
    causal_top_left: bool = False,
    # Dropout mask broadcastable to [B, H, Sq, Skv]; 0 or 1/(1-p), applied to post-
    # softmax P.
    dropout_mask: Optional[torch.Tensor] = None,
    # CUDA/HIP stream; defaults to the current stream for q.device.
    stream: Optional[torch.cuda.Stream] = None,
):
    """Run FlyDSL Flash Attention (gfx950 DUALWAVE_SWP / gfx942 generic fallback).

    Args:
        q: Query tensor. Dense: ``[B, Sq, H, D]`` (BSHD).
           Varlen: ``[total_q, H, D]`` (packed, cu_seqlens_q required).
        k: Key tensor. Dense: ``[B, Skv, Hkv, D]``.
           Varlen: ``[total_kv, Hkv, D]``.
        v: Value tensor, same shape as k.
           Paged KV cache (future ABI): physical K/V cache tensors. Supported
           ``kv_cache_layout`` values:
           - ``linear``: 4D paged K/V, ``[NumBlocks, PageSize, NumKVHeads, HeadDim]``.
           - ``linear3d``: page_size=1 special case,
             ``[NumBlocks, NumKVHeads, HeadDim]``.
           - ``vectorized``: aiter-style 5D K/V, where
             ``K = [NumBlocks, NumKVHeads, HeadDim / kVectorSize, PageSize, kVectorSize]``
             and
             ``V = [NumBlocks, NumKVHeads, PageSize / kVectorSize, HeadDim, kVectorSize]``.
             Here ``kVectorSize = 16 / element_size`` (bf16/fp16: 8, fp8: 16);
             page_size and head_dim must be divisible by it.
        causal: Bottom-right aligned causal mask when True.
        num_kv_heads: KV head count for GQA/MQA; defaults to q num_heads (MHA).
        cu_seqlens_q: Int32 ``[B+1]`` cumulative Q token counts (varlen).
        cu_seqlens_kv: Int32 ``[B+1]`` cumulative KV token counts (varlen).
        max_seqlen_q: Maximum per-batch Q seqlen (varlen). Required in varlen mode.
        max_seqlen_kv: Maximum per-batch KV seqlen (varlen cross-attn). Required when
            seqlen_q != seqlen_kv per batch.
        cross_seqlen: Whether seqlen_q and seqlen_kv differ. Required in varlen mode;
            dense mode infers it from ``q.shape[1] != k.shape[1]``.
        block_table / seqlen_k: vLLM-style 2D block table metadata.
        num_kv_splits: Split-K factor. ``0`` (default) means auto: the dense paged
            path picks a split count from occupancy via ``_auto_paged_kv_splits``
            and every other path resolves it to 1, so behaviour is unchanged
            outside paged decode. ``1`` forces the single-pass kernel everywhere
            (also reachable with ``MSLK_DISABLE_PAGED_SPLITK=1``). ``>1`` forces
            that many splits (gfx950 only, D=64/128, bf16/f16; the dense
            non-paged path additionally requires seq>=384).
        q_descale / k_descale / v_descale: fp32 shape-[1] descales required
            for dense fp8 e4m3fn inputs.
        out: Optional pre-allocated output tensor. For fp8, output is bf16;
            otherwise it has the same dtype as q.
        waves_per_eu: Kernel occupancy hint.
        daz: Enable denormals-are-zero.
        dualwave_swp_lazy_rescale: Enable lazy online softmax rescale.
        dualwave_swp_setprio: Enable s_setprio scheduling hints.
        dualwave_swp_enable_stagger: Enable wave-group phase stagger.
        debug_counts: Float32[2] tensor; when given, counts lazy-rescale branches
            (debug_counts[0] = all-below-true, debug_counts[1] = all-below-false).
        return_lse: When True, also compute and return per-row log-sum-exp
            (natural log, sm_scale folded in). Not supported for fp8 or paged KV.
        stream: CUDA/HIP stream to launch on.

    Returns:
        Output tensor with the same shape as q (dtype bf16 for fp8 inputs,
        otherwise the same dtype as q). When ``return_lse=True``, returns a
        ``(out, lse)`` tuple instead, where ``lse`` is float32 ``[B, H, Sq]``
        (varlen: ``[B, H, max_seqlen_q]``; padded rows undefined).
    """
    # ── validation ──────────────────────────────────────────────────────────
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_func: q/k/v must be CUDA tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(
            f"flydsl_flash_attn_func: q/k/v must share device; got {q.device}/{k.device}/{v.device}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            f"flydsl_flash_attn_func: q/k/v must share dtype; got {q.dtype}/{k.dtype}/{v.dtype}"
        )

    dtype_str = _dtype_str(q)
    paged_kv = any(x is not None for x in (block_table, seqlen_k))
    if dtype_str == "fp8" and paged_kv:
        raise NotImplementedError(
            "flydsl_flash_attn_func: fp8 flash_attn does not support paged KV"
        )
    if return_lse and dtype_str == "fp8":
        raise NotImplementedError(
            "flydsl_flash_attn_func: return_lse is not supported for fp8"
        )
    # LSE support for paged KV depends on which kernel the shape routes to: the
    # generic light path (paged-gappy, or short-q / gfx942 native paged) produces
    # LSE; the dualwave native-paged path does not. The precise check lives in
    # _flydsl_flash_attn_paged where the light-vs-dualwave decision is made.
    if window_left >= 0:
        # Window lives in apply_kv_mask (dense + generic varlen); fp8/paged lack it.
        if dtype_str == "fp8" or paged_kv:
            raise NotImplementedError(
                "flydsl_flash_attn_func: sliding-window (window_left>=0) is not "
                "supported for fp8 or paged KV"
            )
    if sm_scale is not None:
        # fp8 and native paged hardcode 1/sqrt(head_dim); gappy paged uses the
        # generic kernel and folds sm_scale.
        _gappy_paged = paged_kv and kv_seqstart is not None
        if dtype_str == "fp8" or (paged_kv and not _gappy_paged):
            raise NotImplementedError(
                "flydsl_flash_attn_func: custom sm_scale is only supported on the "
                "dense/varlen f16/bf16 paths (fp8/paged hardcode 1/sqrt(head_dim))"
            )
    if bias is not None:
        _varlen_req = cu_seqlens_q is not None
        if dtype_str == "fp8" or paged_kv or _varlen_req:
            raise NotImplementedError(
                "flydsl_flash_attn_func: attention bias is only supported on the "
                "dense f16/bf16 path (not fp8/paged/varlen)"
            )
    if dropout_mask is not None:
        _varlen_req = cu_seqlens_q is not None
        if dtype_str == "fp8" or paged_kv or _varlen_req:
            raise NotImplementedError(
                "flydsl_flash_attn_func: dropout is only supported on the "
                "dense f16/bf16 path (not fp8/paged/varlen)"
            )
    if paged_kv:
        return _flydsl_flash_attn_paged(
            q,
            k,
            v,
            causal=causal,
            num_kv_heads=num_kv_heads,
            block_table=block_table,
            seqlen_k=seqlen_k,
            max_seqlen_kv=max_seqlen_kv,
            kv_cache_layout=kv_cache_layout,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            kv_seqstart=kv_seqstart,
            max_seqlen_q=max_seqlen_q,
            cross_seqlen=cross_seqlen,
            num_kv_splits=num_kv_splits,
            return_lse=return_lse,
            sm_scale=sm_scale,
            out=out,
            waves_per_eu=waves_per_eu,
            daz=daz,
            dualwave_swp_lazy_rescale=dualwave_swp_lazy_rescale,
            dualwave_swp_setprio=dualwave_swp_setprio,
            dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
            stream=stream,
        )

    varlen = cu_seqlens_q is not None

    # `0` (auto) only selects a split count on the dense paged path, which has
    # already returned above. Everything from here on is non-paged, where the
    # existing per-shape `generic_splitk` selection is the auto mechanism, so
    # collapse the sentinel to the historical default before anything reads it.
    if num_kv_splits == 0:
        num_kv_splits = 1

    if dtype_str == "fp8":
        if varlen:
            raise NotImplementedError(
                "flydsl_flash_attn_func: fp8 flash_attn does not support varlen"
            )
        if num_kv_splits > 1:
            raise NotImplementedError(
                "flydsl_flash_attn_func: fp8 flash_attn does not support split-K"
            )
        if debug_counts is not None:
            raise NotImplementedError(
                "flydsl_flash_attn_func: fp8 flash_attn does not support debug_counts"
            )
        if any(x is None for x in (q_descale, k_descale, v_descale)):
            raise ValueError(
                "flydsl_flash_attn_func: fp8 requires q_descale, k_descale, and v_descale"
            )
        for name, scale in (
            ("q_descale", q_descale),
            ("k_descale", k_descale),
            ("v_descale", v_descale),
        ):
            if not scale.is_cuda:
                raise ValueError(
                    f"flydsl_flash_attn_func: {name} must be a CUDA tensor"
                )
            if scale.device != q.device:
                raise ValueError(
                    f"flydsl_flash_attn_func: {name} must be on {q.device}, got {scale.device}"
                )
            if scale.dtype != torch.float32 or scale.numel() != 1:
                raise ValueError(
                    f"flydsl_flash_attn_func: {name} must be a shape-[1] float32 tensor"
                )

    if varlen and cu_seqlens_kv is None:
        raise ValueError(
            "flydsl_flash_attn_func: cu_seqlens_kv required when cu_seqlens_q is given"
        )
    if not varlen and cu_seqlens_kv is not None:
        raise ValueError(
            "flydsl_flash_attn_func: cu_seqlens_q required when cu_seqlens_kv is given"
        )
    if varlen and num_kv_splits > 1:
        raise ValueError(
            "flydsl_flash_attn_func: varlen + split-K (num_kv_splits>1) is not supported"
        )

    # ── shape inference ─────────────────────────────────────────────────────
    if varlen:
        if q.dim() != 3:
            raise ValueError(
                f"flydsl_flash_attn_func: varlen q must be 3D [total,H,D], got {q.dim()}D"
            )
        _total_q, H, D = q.shape
        Hkv = k.shape[1]
        B = cu_seqlens_q.numel() - 1
        if max_seqlen_q is None:
            raise ValueError(
                "flydsl_flash_attn_func: max_seqlen_q is required in varlen mode"
            )
        if cross_seqlen is None:
            raise ValueError(
                "flydsl_flash_attn_func: cross_seqlen is required in varlen mode"
            )
        Sq = int(max_seqlen_q)
        cross = bool(cross_seqlen)
        if cross and max_seqlen_kv is None:
            raise ValueError(
                "flydsl_flash_attn_func: max_seqlen_kv is required when varlen cross_seqlen=True"
            )
    else:
        if q.dim() != 4:
            raise ValueError(
                f"flydsl_flash_attn_func: dense q must be 4D [B,Sq,H,D], got {q.dim()}D"
            )
        B, Sq, H, D = q.shape
        Skv = k.shape[1]
        Hkv = k.shape[2]
        cross = Sq != Skv if cross_seqlen is None else bool(cross_seqlen)

    if num_kv_heads is None:
        num_kv_heads = Hkv
    if H % num_kv_heads != 0:
        raise ValueError(
            f"flydsl_flash_attn_func: num_heads ({H}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    if D < 64 or D % 32 != 0:
        raise ValueError(
            f"flydsl_flash_attn_func: head_dim ({D}) must be >= 64 and a multiple of 32"
        )

    splitk = num_kv_splits > 1
    # generic_splitk is chosen internally per-shape (below); it uses the generic
    # BLOCK_M=64 kernel + dense combine, distinct from the dualwave `splitk` above.
    generic_splitk = False

    # ── split-K eligibility guard (SKIP analogous to run_splitk_config) ────
    if splitk:
        if D not in (64, 128) or dtype_str not in ("bf16", "f16") or Sq < 384:
            raise ValueError(
                f"flydsl_flash_attn_func: split-K requires D=64/128, dtype bf16/f16, seq_len>=384; "
                f"got D={D}, dtype={dtype_str}, seq_len={Sq}"
            )
        ws_elems = dualwave_splitk_workspace_elems(
            B, H, Sq, int(num_kv_splits), head_dim=D
        )

    # ── build (cached) ──────────────────────────────────────────────────────
    debug_lazy = debug_counts is not None

    with torch.cuda.device(q.device.index):
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )

        if splitk:
            exe = _build_splitk(
                num_heads=H,
                num_kv_heads=num_kv_heads,
                head_dim=D,
                causal=causal,
                dtype_str=dtype_str,
                num_kv_splits=int(num_kv_splits),
                waves_per_eu=waves_per_eu,
                daz=daz,
                lazy_rescale=dualwave_swp_lazy_rescale,
                setprio=dualwave_swp_setprio,
                enable_stagger=dualwave_swp_enable_stagger,
                return_lse=return_lse,
            )
        elif varlen:
            # Generic light for short/cross/D256 varlen; long self-attn uses dualwave.
            # (The dualwave cross path NaNs for >=5 KV tiles; D=256 is generic-only.)
            _arch = _gpu_arch(q.device)
            _prefer_light = (
                (not debug_lazy)
                and D in (64, 128, 256)
                and dtype_str in ("bf16", "f16")
                and (
                    cross
                    or D == 256
                    or window_left >= 0
                    # dualwave varlen hardcodes 1/sqrt(head_dim), so custom scale
                    # must use light.
                    or sm_scale is not None
                    # f16 overflows the dualwave softmax; use light.
                    or dtype_str == "f16"
                    or not _arch.startswith("gfx950")
                    or Sq <= _VARLEN_LIGHT_MAX_SEQ
                    # Gappy KV is only implemented on the generic (light) path.
                    or kv_seqstart is not None
                )
            )
            if _prefer_light:
                exe = _build_varlen_light(
                    num_heads=H,
                    num_kv_heads=num_kv_heads,
                    head_dim=D,
                    causal=causal,
                    dtype_str=dtype_str,
                    cross_seqlen=cross,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    lazy_rescale=dualwave_swp_lazy_rescale,
                    setprio=dualwave_swp_setprio,
                    debug_lazy_counts=debug_lazy,
                    enable_stagger=dualwave_swp_enable_stagger,
                    return_lse=return_lse,
                    causal_top_left=causal_top_left,
                    window_left=window_left,
                    sm_scale=sm_scale,
                    gappy_kv=kv_seqstart is not None,
                )
            else:
                exe = _build_varlen(
                    num_heads=H,
                    num_kv_heads=num_kv_heads,
                    head_dim=D,
                    causal=causal,
                    dtype_str=dtype_str,
                    cross_seqlen=cross,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    lazy_rescale=dualwave_swp_lazy_rescale,
                    setprio=dualwave_swp_setprio,
                    debug_lazy_counts=debug_lazy,
                    enable_stagger=dualwave_swp_enable_stagger,
                    return_lse=return_lse,
                )
        else:
            _arch = _gpu_arch(q.device)
            if dtype_str == "fp8":
                if not _arch.startswith("gfx950"):
                    raise ValueError(
                        f"flydsl_flash_attn_func: fp8 requires gfx950, got '{_arch or 'unknown'}'"
                    )
                exe = _build_dense_fp8(
                    num_heads=H,
                    num_kv_heads=num_kv_heads,
                    causal=causal,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    lazy_rescale=dualwave_swp_lazy_rescale,
                    setprio=dualwave_swp_setprio,
                    enable_stagger=dualwave_swp_enable_stagger,
                )
            else:
                # f16 excluded from dualwave: its narrow exponent overflows the
                # dualwave softmax at large logits -> NaN; bf16 is safe.
                can_dualwave = (
                    D in (64, 128)
                    and dtype_str == "bf16"
                    and _arch.startswith("gfx950")
                )
                if debug_lazy and not can_dualwave:
                    raise NotImplementedError(
                        "flydsl_flash_attn_func: debug_counts requires the gfx950 DUALWAVE_SWP path"
                    )
                # Window / custom scale / bias / dropout / cross are generic-only;
                # never route them to dualwave (hardcodes 1/sqrt(D), NaNs on cross).
                _windowed = window_left >= 0
                _custom_scale = sm_scale is not None
                _has_bias = bias is not None
                _has_dropout = dropout_mask is not None
                if (
                    not _windowed
                    and not _custom_scale
                    and not _has_bias
                    and not _has_dropout
                    and not cross
                    and (
                        debug_lazy
                        or (can_dualwave and _dense_routes_to_dualwave(B, Sq))
                    )
                ):
                    exe = _build_dense_dualwave(
                        num_heads=H,
                        num_kv_heads=num_kv_heads,
                        head_dim=D,
                        causal=causal,
                        dtype_str=dtype_str,
                        cross_seqlen=cross,
                        waves_per_eu=waves_per_eu,
                        daz=daz,
                        lazy_rescale=dualwave_swp_lazy_rescale,
                        setprio=dualwave_swp_setprio,
                        debug_lazy_counts=debug_lazy,
                        enable_stagger=dualwave_swp_enable_stagger,
                        return_lse=return_lse,
                    )
                else:
                    # Generic split-K for short-q dense self-attention that underfills
                    # the GPU: BLOCK_M=64, gated by CK's occupancy heuristic. Dropout
                    # disables it (matches CK); bias/window/scale compose fine.
                    _gen_splits = 1
                    if (
                        not _has_dropout
                        and not cross
                        and D in (64, 128, 256)
                        and dtype_str in ("bf16", "f16")
                    ):
                        _gen_splits = _num_kv_splits_heuristic(
                            B, H, Sq, D, _dense_light_cu(q.device)
                        )
                    if _gen_splits > 1:
                        generic_splitk = True
                        num_kv_splits = _gen_splits
                        # BLOCK_M=64 short-q tile; N128 only for causal D128, else
                        # N32 (matches the generic builder's own path selection).
                        _sk_tag = "N128" if (causal and D == 128) else "N32"
                        exe = _build_dense(
                            num_heads=H,
                            num_kv_heads=num_kv_heads,
                            head_dim=D,
                            causal=causal,
                            dtype_str=dtype_str,
                            cross_seqlen=cross,
                            block_m=64,
                            flat_work_group_size=128,
                            path_tag=_sk_tag,
                            waves_per_eu=waves_per_eu,
                            daz=daz,
                            return_lse=return_lse,
                            window_left=window_left,
                            sm_scale=sm_scale,
                            has_bias=_has_bias,
                            has_dropout=_has_dropout,
                            num_kv_splits=_gen_splits,
                        )
                    else:
                        block_m, flat_work_group_size, path_tag = _dense_generic_tile(
                            B, Sq, H, D, dtype_str, q.device, has_bias=_has_bias
                        )
                        exe = _build_dense(
                            num_heads=H,
                            num_kv_heads=num_kv_heads,
                            head_dim=D,
                            causal=causal,
                            dtype_str=dtype_str,
                            cross_seqlen=cross,
                            block_m=block_m,
                            flat_work_group_size=flat_work_group_size,
                            path_tag=path_tag,
                            waves_per_eu=waves_per_eu,
                            daz=daz,
                            return_lse=return_lse,
                            window_left=window_left,
                            sm_scale=sm_scale,
                            has_bias=_has_bias,
                            has_dropout=_has_dropout,
                        )

        # ── allocate output ─────────────────────────────────────────────────
        if out is None:
            out_dtype = torch.bfloat16 if dtype_str == "fp8" else q.dtype
            out = torch.empty(q.shape, dtype=out_dtype, device=q.device)
        elif dtype_str == "fp8" and out.dtype != torch.bfloat16:
            raise ValueError(
                f"flydsl_flash_attn_func: fp8 output must be bf16, got {out.dtype}"
            )
        elif dtype_str != "fp8" and out.dtype != q.dtype:
            raise ValueError(
                f"flydsl_flash_attn_func: output dtype must match q dtype {q.dtype}, got {out.dtype}"
            )
        # Keep natural shape; flattening can overflow int32 C-ABI dims.
        # Kernels rebuild per-batch descriptors from base pointers and strides.
        if dtype_str == "fp8":
            # The fp8 gfx950 module preserves the original dense ABI from 711.diff:
            # flattened Q/K/V/O tensors plus descale kwargs.
            q_flat = q.contiguous().view(-1)
            k_flat = k.contiguous().view(-1)
            v_flat = v.contiguous().view(-1)
            o_flat = out.contiguous().view(-1)
        else:
            q_flat = q.contiguous()
            k_flat = k.contiguous()
            v_flat = v.contiguous()
            o_flat = out.contiguous()

        lse = (
            torch.empty((B, H, Sq), dtype=torch.float32, device=q.device)
            if return_lse
            else None
        )
        lse_kwargs = dict(lse=lse) if return_lse else {}

        # ── launch ──────────────────────────────────────────────────────────
        if splitk:
            _ws = torch.empty(ws_elems, dtype=torch.float32, device=q.device)
            exe(
                q_flat,
                k_flat,
                v_flat,
                o_flat,
                B,
                Sq,
                workspace=_ws,
                stream=launch_stream,
                **lse_kwargs,
            )
        elif varlen:
            kwargs = dict(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                stream=launch_stream,
                **lse_kwargs,
            )
            if kv_seqstart is not None:
                kwargs["kv_seqstart"] = kv_seqstart
            if cross:
                kwargs["seq_len_kv"] = int(max_seqlen_kv)
            if debug_lazy:
                exe(
                    q_flat,
                    k_flat,
                    v_flat,
                    o_flat,
                    B,
                    Sq,
                    debug_counts=debug_counts,
                    **kwargs,
                )
            else:
                exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)
        else:
            kwargs: dict = dict(stream=launch_stream, **lse_kwargs)
            if cross:
                kwargs["seq_len_kv"] = Skv
            if debug_lazy:
                kwargs["debug_counts"] = debug_counts
            if dtype_str == "fp8":
                kwargs.update(
                    q_descale=q_descale, k_descale=k_descale, v_descale=v_descale
                )
            if bias is not None:
                # Contiguous + broadcast to [B, H, Sq, Skv] (unit kv stride; a
                # size-1 head axis gives stride 0 => head-broadcast).
                bias_e = bias.contiguous().expand(B, H, Sq, Skv)
                assert bias_e.stride(3) == 1, "bias kv axis must be unit-stride"
                kwargs.update(
                    bias=bias_e,
                    bias_stride_b=int(bias_e.stride(0)),
                    bias_stride_h=int(bias_e.stride(1)),
                    bias_stride_q=int(bias_e.stride(2)),
                )
            if dropout_mask is not None:
                # Same ABI as bias: unit kv stride, broadcast to [B, H, Sq, Skv].
                drop_e = dropout_mask.contiguous().expand(B, H, Sq, Skv)
                assert drop_e.stride(3) == 1, "dropout kv axis must be unit-stride"
                kwargs.update(
                    dropout=drop_e,
                    dropout_stride_b=int(drop_e.stride(0)),
                    dropout_stride_h=int(drop_e.stride(1)),
                    dropout_stride_q=int(drop_e.stride(2)),
                )
            if generic_splitk:
                # Dense combine writes O in [B, Sq, H, D] layout (stride_q_n = H*D).
                _ws = torch.empty(
                    dualwave_splitk_workspace_elems(
                        B, H, Sq, int(num_kv_splits), head_dim=D
                    ),
                    dtype=torch.float32,
                    device=q.device,
                )
                kwargs["workspace"] = _ws
                kwargs["stride_q_n"] = H * D
            exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)

    return (out, lse) if return_lse else out
