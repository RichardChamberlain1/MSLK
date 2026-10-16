# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL sliding-window paged decode (gfx950), single launch.

Each (request, KV head) is handled by num_splits CTAs whose waves each walk a
contiguous slice of the window, using the head-packed MFMA layout of
pa_decode_gfx950 (16 M rows hold the GQA group's (query token, head) pairs) and
a private LDS region, so the KV loop needs no CTA barrier. The waves merge
through LDS; with num_splits > 1 the CTAs then merge in-kernel via a workspace.

Query row i (bottom-right causal) attends keys in (pos - window_left, pos],
pos = seqlen_kv - seqlen_q + i, matching the generic paged kernel.
"""

from __future__ import annotations

import functools
from typing import Any

import flydsl.compiler as flyc  # pyre-ignore[21]
import flydsl.expr as fx  # pyre-ignore[21]
import torch
from flydsl._mlir.dialects import llvm as _llvm  # pyre-ignore[21]
from flydsl.expr import (  # pyre-ignore[21]
    arith,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr.typing import T  # pyre-ignore[21]
from flydsl.runtime.device import get_rocm_arch  # pyre-ignore[21]
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr  # pyre-ignore[21]
from mslk.flydsl import buffer_ops, vector_ops as vector

from .utils import (
    dpp_xor_f32,
    exp2_f32 as _exp2_fast,
    maxnumf as _mxf,
    rcp_f32,
    smem_bytes,
    WARP_SIZE,
)

MFMA_M = 16  # (query token, head) pairs packed on the QK MFMA M-axis
MFMA_N = 16  # tokens per QK sub-tile
MFMA_K_QK = 32  # head-dim elements per QK MFMA
TILE_N = 32  # tokens per tile (= PV MFMA K-dim)
N_SUBTILE = TILE_N // MFMA_N
LOG2E: float = 1.4426950408889634
_FX_DTYPE = {"f16": fx.Float16, "bf16": fx.BFloat16}


@functools.lru_cache(maxsize=64)
def compile_pa_decode_swa(
    *,
    head_size: int,
    kv_dtype_str: str,
    page_size: int,
    gqa_ratio: int,
    seqlen_q: int,
    num_waves: int,
    num_splits: int = 1,
    arch: str = "",
) -> Any:  # pyre-ignore[3]
    if not arch:
        arch = get_rocm_arch()
    assert arch.startswith("gfx950"), f"pa_decode_swa requires gfx950, got {arch}"
    assert head_size % MFMA_K_QK == 0 and head_size % MFMA_N == 0
    assert page_size % TILE_N == 0
    assert 1 <= gqa_ratio <= MFMA_M and MFMA_M % gqa_ratio == 0

    _HEAD = head_size
    _PAGE = page_size
    _RATIO = gqa_ratio
    _SQ = seqlen_q
    _NW = num_waves
    _S = num_splits  # CTAs per (request, KV head); > 1 merges through `ws`
    _BLOCK = _NW * WARP_SIZE
    _T_PACK = MFMA_M // _RATIO
    _M_TILES = (_SQ + _T_PACK - 1) // _T_PACK
    _FX_KV = _FX_DTYPE[kv_dtype_str]
    _QK_GRP = _HEAD // MFMA_K_QK
    _DN = _HEAD // MFMA_N
    _N_ACC = _DN * 4
    _mfma = (
        rocdl.mfma_f32_16x16x32_bf16
        if kv_dtype_str == "bf16"
        else rocdl.mfma_f32_16x16x32_f16
    )

    # Per-wave LDS region: the loop's P planes + V tile, later reused by that
    # wave for its merge partials (acc rows, then max and sum).
    _NUM_LD_V = (TILE_N * _HEAD // 8) // WARP_SIZE  # vec8 V loads per lane
    _P_F32 = _M_TILES * MFMA_M * TILE_N
    _V_F16 = TILE_N * _HEAD
    _K_F16 = TILE_N * _HEAD  # K tile, row-major with a 16B-chunk XOR swizzle
    _LOOP_BYTES = _P_F32 * 4 + (_V_F16 + _K_F16) * 2
    _C8 = _HEAD // 8  # 16-byte chunks per K/V row
    _ACC_F32 = _M_TILES * MFMA_M * _HEAD
    _MERGE_BYTES = (_ACC_F32 + 2 * _M_TILES * MFMA_M) * 4
    _REGION = max(_LOOP_BYTES, _MERGE_BYTES)
    _REGION = (_REGION + 15) // 16 * 16
    _LDS_TOTAL = _NW * _REGION
    if _LDS_TOTAL > smem_bytes(arch):
        raise ValueError(f"pa_decode_swa LDS {_LDS_TOTAL}B > {arch!r} cap")
    _R_F32 = _REGION // 4

    alloc = SmemAllocator(
        None,
        arch=arch,
        global_sym_name=(
            f"pa_swa_h{_HEAD}_{kv_dtype_str}_pg{_PAGE}_r{_RATIO}q{_SQ}_w{_NW}s{_S}"
        ),
    )
    alloc.ptr = _LDS_TOTAL

    @flyc.kernel(known_block_size=(_BLOCK, 1, 1))
    def pa_decode_swa_kernel(
        out_ptr: fx.Tensor,
        q_ptr: fx.Tensor,
        k_ptr: fx.Tensor,
        v_ptr: fx.Tensor,
        seq_ptr: fx.Tensor,
        bt_ptr: fx.Tensor,
        stride_qb: fx.Int32,
        stride_qs: fx.Int32,
        stride_qh: fx.Int32,
        stride_kb: fx.Int32,  # page stride of [num_pages, page_size, H_kv, D]
        stride_km: fx.Int32,  # token stride within a page
        stride_kh: fx.Int32,
        stride_btb: fx.Int32,
        num_hq: fx.Int32,
        num_hkv: fx.Int32,
        window_left: fx.Int32,
        softmax_scale: fx.Float32,
        ws_ptr: fx.Tensor,
    ) -> None:
        tid = fx.Int32(gpu.thread_idx.x)
        lane = tid % fx.Int32(WARP_SIZE)
        wave = fx.Int32(
            rocdl.readfirstlane(T.i32, arith.unwrap(tid // fx.Int32(WARP_SIZE)))
        )
        tok_lane = lane % fx.Int32(MFMA_N)
        grp = lane // fx.Int32(MFMA_N)

        flat = fx.Int32(gpu.block_idx.x)
        split = flat % fx.Int32(_S)
        head_flat = flat // fx.Int32(_S)
        hkv_abs = head_flat % num_hkv
        b_idx = head_flat // num_hkv
        hq_base = hkv_abs * fx.Int32(_RATIO)

        c_zero = arith.constant(0.0, type=T.f32)
        c_one = arith.constant(1.0, type=T.f32)
        c_neginf = arith.constant(float("-inf"), type=T.f32)
        zero_v4 = arith.constant_vector(0.0, T.vec(4, T.f32))
        zero_v8h = arith.constant_vector(0.0, T.vec(8, _FX_KV.ir_type))

        q_rsrc = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(v_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        seq_rsrc = buffer_ops.create_buffer_resource(seq_ptr, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(bt_ptr, max_size=False)

        seq_len = buffer_ops.buffer_load(seq_rsrc, b_idx, vec_width=1, dtype=T.i32)
        t_full = fx.Int32(arith.select(seq_len > fx.Int32(0), seq_len, fx.Int32(0)))
        # Keys any row of this CTA can see: from the first row's window start
        # (aligned down to a tile) to the end of the sequence.
        lo = t_full - fx.Int32(_SQ - 1) - window_left
        lo = fx.Int32(arith.select(lo > fx.Int32(0), lo, fx.Int32(0)))
        t_lo = (lo // fx.Int32(TILE_N)) * fx.Int32(TILE_N)
        n_tiles = (t_full - t_lo + fx.Int32(TILE_N - 1)) // fx.Int32(TILE_N)
        chunk = (n_tiles + fx.Int32(_S * _NW - 1)) // fx.Int32(_S * _NW)
        w_start = t_lo + (split * fx.Int32(_NW) + wave) * chunk * fx.Int32(TILE_N)
        w_end_raw = w_start + chunk * fx.Int32(TILE_N)
        w_end = fx.Int32(arith.select(w_end_raw < t_full, w_end_raw, t_full))
        w_start = fx.Int32(arith.select(w_start < w_end, w_start, w_end))

        # One page id per lane: the window spans at most WARP_SIZE pages.
        page0 = t_lo // fx.Int32(_PAGE)
        pid_reg = buffer_ops.buffer_load(
            bt_rsrc, b_idx * stride_btb + page0 + lane, vec_width=1, dtype=T.i32
        )

        smem = alloc.get_base()
        lds_f32 = SmemPtr(smem, 0, T.f32, shape=(_LDS_TOTAL // 4,)).get()
        lds_base = buffer_ops.extract_base_index(smem, address_space=3)
        my_f32 = fx.Int32(wave) * fx.Int32(_R_F32)  # this wave's region, f32 units
        v_lds_f16 = lds_base + fx.Index(my_f32 * fx.Int32(2) + fx.Int32(_P_F32 * 2))
        k_lds_f16 = v_lds_f16 + fx.Index(_V_F16)

        # Q fragments, loop-invariant. M slot m = qtok * ratio + head.
        q_head = tok_lane % fx.Int32(_RATIO)
        q_tok_local = tok_lane // fx.Int32(_RATIO)
        q_base_bh = b_idx * stride_qb + (hq_base + q_head) * stride_qh
        q_frags = []
        for t in range_constexpr(_M_TILES):
            q_tok = q_tok_local + fx.Int32(t * _T_PACK)
            q_tok = fx.Int32(
                arith.select(q_tok < fx.Int32(_SQ), q_tok, fx.Int32(_SQ - 1))
            )
            frags = []
            for g in range_constexpr(_QK_GRP):
                q_off = (
                    q_base_bh
                    + q_tok * stride_qs
                    + fx.Int32(g * MFMA_K_QK)
                    + grp * fx.Int32(8)
                )
                frags.append(
                    buffer_ops.buffer_load(q_rsrc, q_off, vec_width=8, dtype=_FX_KV)
                )
            q_frags.append(frags)

        kv_base = hkv_abs * stride_kh
        _TILE_STATE = 8 + _N_ACC

        def _load_tile(tile_start):
            """Issue one tile's K and V loads, coalesced: a lane loads one 16B
            chunk and _C8 consecutive lanes cover a whole token row."""
            page = rocdl.readlane(
                T.i32,
                arith.unwrap(fx.Int32(pid_reg)),
                arith.unwrap(tile_start // fx.Int32(_PAGE) - page0),
            )
            org = (
                kv_base
                + (tile_start % fx.Int32(_PAGE)) * stride_km
                + fx.Int32(page) * stride_kb
            )
            ks = []
            vs = []
            for _r in range_constexpr(_NUM_LD_V):
                _lin = lane + fx.Int32(_r * WARP_SIZE)
                _off = (
                    org
                    + (_lin // fx.Int32(_C8)) * stride_km
                    + (_lin % fx.Int32(_C8)) * fx.Int32(8)
                )
                ks.append(
                    buffer_ops.buffer_load(k_rsrc, _off, vec_width=8, dtype=_FX_KV)
                )
                vs.append(
                    buffer_ops.buffer_load(v_rsrc, _off, vec_width=8, dtype=_FX_KV)
                )
            return ks, vs

        # Register double buffer: the next tile's K/V loads are issued before
        # this tile's math. A wave's last tile re-issues itself (harmless).
        _k0, _v0 = _load_tile(w_start)
        _init = (
            ([c_neginf] * 4 + [c_zero] * 4 + [c_zero] * _N_ACC) * _M_TILES + _k0 + _v0
        )
        _KV_OFF = _TILE_STATE * _M_TILES

        for _tile_i, state in range(
            fx.Index(w_start), fx.Index(w_end), arith.index(TILE_N), init=_init
        ):
            rmax = [
                [fx.Float32(state[t * _TILE_STATE + i]) for i in range(4)]
                for t in range(_M_TILES)
            ]
            rsum = [
                [fx.Float32(state[t * _TILE_STATE + 4 + i]) for i in range(4)]
                for t in range(_M_TILES)
            ]
            acc = [
                [fx.Float32(state[t * _TILE_STATE + 8 + i]) for i in range(_N_ACC)]
                for t in range(_M_TILES)
            ]
            tile_start = fx.Int32(arith.index_cast(T.i32, _tile_i))
            cur_k = [state[_KV_OFF + i] for i in range(_NUM_LD_V)]
            cur_v = [state[_KV_OFF + _NUM_LD_V + i] for i in range(_NUM_LD_V)]
            nxt = tile_start + fx.Int32(TILE_N)
            nxt = fx.Int32(arith.select(nxt < w_end, nxt, tile_start))
            nxt_k, nxt_v = _load_tile(nxt)

            # Park this tile's K (swizzled rows) and V (transpose layout) in LDS.
            _v8s = cur_v
            _v8dst = []
            for _r in range_constexpr(_NUM_LD_V):
                _lin = lane + fx.Int32(_r * WARP_SIZE)
                _tok = _lin // fx.Int32(_C8)
                _c8 = _lin % fx.Int32(_C8)
                _v8dst.append(
                    v_lds_f16
                    + fx.Index(_c8 // fx.Int32(2)) * fx.Index(TILE_N * 16)
                    + fx.Index(_tok) * fx.Index(16)
                    + fx.Index(_c8 % fx.Int32(2)) * fx.Index(8)
                )
                _kdst = k_lds_f16 + fx.Index(
                    _tok * fx.Int32(_HEAD)
                    + (_c8 ^ (_tok % fx.Int32(_C8))) * fx.Int32(8)
                )
                _llvm.StoreOp(
                    cur_k[_r],
                    buffer_ops.create_llvm_ptr(
                        fx.Int64(_kdst * fx.Index(2)), address_space=3
                    ),
                    alignment=16,
                )
            fx.llvm.memory_fence(
                syncscope=rocdl.SyncScope.Wavefront, ordering=fx.AtomicOrdering.AcqRel
            )

            qk_st = [[zero_v4 for _ in range(N_SUBTILE)] for _ in range(_M_TILES)]
            for st in range_constexpr(N_SUBTILE):
                _row = fx.Int32(st * MFMA_N) + tok_lane
                for g in range_constexpr(_QK_GRP):
                    _c = fx.Int32(g * 4) + grp
                    _ksrc = k_lds_f16 + fx.Index(
                        _row * fx.Int32(_HEAD)
                        + (_c ^ (_row % fx.Int32(_C8))) * fx.Int32(8)
                    )
                    k8 = _llvm.LoadOp(
                        T.vec(8, _FX_KV.ir_type),
                        buffer_ops.create_llvm_ptr(
                            fx.Int64(_ksrc * fx.Index(2)), address_space=3
                        ),
                        alignment=16,
                    ).result
                    for t in range_constexpr(_M_TILES):
                        qk_st[t][st] = _mfma(
                            T.vec(4, T.f32), [q_frags[t][g], k8, qk_st[t][st], 0, 0, 0]
                        )

            new_max = [[] for _ in range(_M_TILES)]
            alpha = [[] for _ in range(_M_TILES)]
            for t in range_constexpr(_M_TILES):
                for e in range_constexpr(4):
                    m_idx = grp * fx.Int32(4) + fx.Int32(e)
                    qtok = fx.Int32(t * _T_PACK) + m_idx // fx.Int32(_RATIO)
                    end_m = t_full - fx.Int32(_SQ) + qtok + fx.Int32(1)
                    end_m = fx.Int32(arith.select(w_end < end_m, w_end, end_m))
                    begin_m = t_full - fx.Int32(_SQ) + qtok + fx.Int32(1) - window_left
                    loc = fx.Float32(c_neginf)
                    for st in range_constexpr(N_SUBTILE):
                        s = fx.Float32(
                            vector.extract(
                                qk_st[t][st], static_position=[e], dynamic_position=[]
                            )
                        )
                        s = fx.Float32(
                            arith.mulf(arith.unwrap(s), arith.unwrap(softmax_scale))
                        )
                        tok_abs = tile_start + fx.Int32(st * MFMA_N) + tok_lane
                        ok = (tok_abs < end_m) & (tok_abs >= begin_m)
                        s = fx.Float32(
                            arith.select(arith.unwrap(ok), arith.unwrap(s), c_neginf)
                        )
                        loc = _mxf(loc, s)
                        qk_st[t][st] = vector.insert(
                            arith.unwrap(s),
                            qk_st[t][st],
                            static_position=[e],
                            dynamic_position=[],
                        )
                    for sh in (1, 2, 4, 8):
                        loc = _mxf(loc, dpp_xor_f32(loc, sh))
                    nm = _mxf(rmax[t][e], loc)
                    new_max[t].append(nm)
                    a = _exp2_fast(
                        fx.Float32(
                            arith.mulf(
                                arith.subf(arith.unwrap(rmax[t][e]), arith.unwrap(nm)),
                                arith.constant(LOG2E, type=T.f32),
                            )
                        )
                    )
                    # A row with nothing seen yet has rmax = nm = -inf: exp2(NaN).
                    a = fx.Float32(
                        arith.select(
                            arith.unwrap(nm) > c_neginf, arith.unwrap(a), c_one
                        )
                    )
                    alpha[t].append(a)

            for t in range_constexpr(_M_TILES):
                tile_sum = [fx.Float32(c_zero) for _ in range(4)]
                for e in range_constexpr(4):
                    m_idx = grp * fx.Int32(4) + fx.Int32(e)
                    for st in range_constexpr(N_SUBTILE):
                        s = fx.Float32(
                            vector.extract(
                                qk_st[t][st], static_position=[e], dynamic_position=[]
                            )
                        )
                        p = _exp2_fast(
                            fx.Float32(
                                arith.mulf(
                                    arith.subf(
                                        arith.unwrap(s), arith.unwrap(new_max[t][e])
                                    ),
                                    arith.constant(LOG2E, type=T.f32),
                                )
                            )
                        )
                        p = fx.Float32(
                            arith.select(
                                arith.unwrap(new_max[t][e]) > c_neginf,
                                arith.unwrap(p),
                                c_zero,
                            )
                        )
                        tile_sum[e] = fx.Float32(
                            arith.addf(arith.unwrap(tile_sum[e]), arith.unwrap(p))
                        )
                        tok = fx.Int32(st * MFMA_N) + tok_lane
                        vector.store(
                            fx.Vector.from_elements(
                                [arith.unwrap(p)], dtype=fx.Float32
                            ),
                            lds_f32,
                            [
                                fx.Index(
                                    my_f32
                                    + fx.Int32(t * MFMA_M * TILE_N)
                                    + m_idx * fx.Int32(TILE_N)
                                    + tok
                                )
                            ],
                        )
                for e in range_constexpr(4):
                    for sh in (1, 2, 4, 8):
                        tile_sum[e] = fx.Float32(
                            arith.addf(
                                arith.unwrap(tile_sum[e]),
                                arith.unwrap(dpp_xor_f32(tile_sum[e], sh)),
                            )
                        )
                    rsum[t][e] = fx.Float32(
                        arith.addf(
                            arith.mulf(
                                arith.unwrap(alpha[t][e]), arith.unwrap(rsum[t][e])
                            ),
                            arith.unwrap(tile_sum[e]),
                        )
                    )
                    rmax[t][e] = new_max[t][e]

            for _r in range_constexpr(_NUM_LD_V):
                _sp = buffer_ops.create_llvm_ptr(
                    fx.Int64(_v8dst[_r] * fx.Index(2)), address_space=3
                )
                _llvm.StoreOp(_v8s[_r], _sp, alignment=16)
            # This wave alone reads what it just wrote, and a wavefront's LDS
            # operations complete in order: only the compiler needs fencing.
            fx.llvm.memory_fence(
                syncscope=rocdl.SyncScope.Wavefront, ordering=fx.AtomicOrdering.AcqRel
            )

            for t in range_constexpr(_M_TILES):
                for dpass in range_constexpr(_DN):
                    for e in range_constexpr(4):
                        acc[t][dpass * 4 + e] = fx.Float32(
                            arith.mulf(
                                arith.unwrap(acc[t][dpass * 4 + e]),
                                arith.unwrap(alpha[t][e]),
                            )
                        )
            p_frags = []
            for t in range_constexpr(_M_TILES):
                p_frag = zero_v8h
                for j in range_constexpr(8):
                    pv = fx.Vector.load(
                        T.vec(1, T.f32),
                        lds_f32,
                        [
                            fx.Index(
                                my_f32
                                + fx.Int32(t * MFMA_M * TILE_N)
                                + tok_lane * fx.Int32(TILE_N)
                                + grp * fx.Int32(8)
                                + fx.Int32(j)
                            )
                        ],
                    )[0]
                    p_frag = vector.insert(
                        arith.truncf(_FX_KV.ir_type, arith.unwrap(fx.Float32(pv))),
                        p_frag,
                        static_position=[j],
                        dynamic_position=[],
                    )
                p_frags.append(p_frag)

            _v4h = T.vec(4, _FX_KV.ir_type)
            for dpass in range_constexpr(_DN):
                _gb = fx.Int32(dpass * (TILE_N * 16)) + grp * fx.Int32(8 * 16)
                _off_lo = v_lds_f16 + fx.Index(_gb) + fx.Index(tok_lane) * fx.Index(4)
                _vlo = rocdl.ds_read_tr16_b64(
                    _v4h,
                    buffer_ops.create_llvm_ptr(
                        fx.Int64(_off_lo * fx.Index(2)), address_space=3
                    ),
                ).result
                _off_hi = _off_lo + fx.Index(4 * 16)
                _vhi = rocdl.ds_read_tr16_b64(
                    _v4h,
                    buffer_ops.create_llvm_ptr(
                        fx.Int64(_off_hi * fx.Index(2)), address_space=3
                    ),
                ).result
                v_frag = vector.shuffle(_vlo, _vhi, [0, 1, 2, 3, 4, 5, 6, 7])
                for t in range_constexpr(_M_TILES):
                    c_in = zero_v4
                    for e in range_constexpr(4):
                        c_in = vector.insert(
                            arith.unwrap(acc[t][dpass * 4 + e]),
                            c_in,
                            static_position=[e],
                            dynamic_position=[],
                        )
                    c_out = _mfma(T.vec(4, T.f32), [p_frags[t], v_frag, c_in, 0, 0, 0])
                    for e in range_constexpr(4):
                        acc[t][dpass * 4 + e] = fx.Float32(
                            vector.extract(
                                c_out, static_position=[e], dynamic_position=[]
                            )
                        )
            fx.llvm.memory_fence(
                syncscope=rocdl.SyncScope.Wavefront, ordering=fx.AtomicOrdering.AcqRel
            )

            state_out = (
                [
                    x
                    for t in range(_M_TILES)
                    for x in (
                        [arith.unwrap(rmax[t][i]) for i in range(4)]
                        + [arith.unwrap(rsum[t][i]) for i in range(4)]
                        + [arith.unwrap(acc[t][i]) for i in range(_N_ACC)]
                    )
                ]
                + list(nxt_k)
                + list(nxt_v)
            )
            results = yield state_out

        # ── Merge: each wave parks its partial in its own region, then the CTA
        # combines the waves per output element. ──
        _M_OFF = _ACC_F32  # max rows, then sum rows, after the acc rows
        for t in range_constexpr(_M_TILES):
            for e in range_constexpr(4):
                m_idx = grp * fx.Int32(4) + fx.Int32(e)
                row = fx.Int32(t * MFMA_M) + m_idx
                for dpass in range_constexpr(_DN):
                    vector.store(
                        fx.Vector.from_elements(
                            [results[t * _TILE_STATE + 8 + dpass * 4 + e]],
                            dtype=fx.Float32,
                        ),
                        lds_f32,
                        [
                            fx.Index(
                                my_f32
                                + row * fx.Int32(_HEAD)
                                + fx.Int32(dpass * MFMA_N)
                                + tok_lane
                            )
                        ],
                    )
                if tok_lane == fx.Int32(0):
                    vector.store(
                        fx.Vector.from_elements(
                            [results[t * _TILE_STATE + e]], dtype=fx.Float32
                        ),
                        lds_f32,
                        [fx.Index(my_f32 + fx.Int32(_M_OFF) + row)],
                    )
                    vector.store(
                        fx.Vector.from_elements(
                            [results[t * _TILE_STATE + 4 + e]], dtype=fx.Float32
                        ),
                        lds_f32,
                        [fx.Index(my_f32 + fx.Int32(_M_OFF + _M_TILES * MFMA_M) + row)],
                    )
        gpu.barrier()

        _OUTS = _M_TILES * MFMA_M * _HEAD
        _M16 = _M_TILES * MFMA_M
        _NPASS = (_OUTS + _BLOCK - 1) // _BLOCK
        _FX_OUT = _FX_KV

        def _out_coords(i):
            idx = tid + fx.Int32(i * _BLOCK)
            row = idx // fx.Int32(_HEAD)
            d = idx % fx.Int32(_HEAD)
            m = row % fx.Int32(MFMA_M)
            qtok = (row // fx.Int32(MFMA_M)) * fx.Int32(_T_PACK) + m // fx.Int32(_RATIO)
            head_abs = hq_base + m % fx.Int32(_RATIO)
            ok = (idx < fx.Int32(_OUTS)) & (qtok < fx.Int32(_SQ)) & (head_abs < num_hq)
            return row, d, qtok, head_abs, ok

        def _lse_merge(parts):
            """(max, sum, numerator) parts -> merged; a part with max -inf is empty."""
            mx = parts[0][0]
            for pm, _, _ in parts[1:]:
                mx = _mxf(mx, pm)
            den = fx.Float32(c_zero)
            num = fx.Float32(c_zero)
            for pm, pl, pn in parts:
                sc = _exp2_fast(
                    fx.Float32(
                        arith.mulf(
                            arith.subf(arith.unwrap(pm), arith.unwrap(mx)),
                            arith.constant(LOG2E, type=T.f32),
                        )
                    )
                )
                sc = fx.Float32(
                    arith.select(arith.unwrap(pm) > c_neginf, arith.unwrap(sc), c_zero)
                )
                den = fx.Float32(
                    arith.addf(
                        arith.unwrap(den),
                        arith.mulf(arith.unwrap(sc), arith.unwrap(pl)),
                    )
                )
                num = fx.Float32(
                    arith.addf(
                        arith.unwrap(num),
                        arith.mulf(arith.unwrap(sc), arith.unwrap(pn)),
                    )
                )
            return mx, den, num

        def _lds_f(idx):
            return fx.Float32(
                fx.Vector.load(T.vec(1, T.f32), lds_f32, [fx.Index(idx)])[0]
            )

        def _store_out(row, d, qtok, head_abs, den, num):
            safe = fx.Float32(
                arith.select(arith.unwrap(den) > c_zero, arith.unwrap(den), c_one)
            )
            val = fx.Float32(arith.mulf(arith.unwrap(num), arith.unwrap(rcp_f32(safe))))
            buffer_ops.buffer_store(
                arith.unwrap(_FX_OUT(arith.unwrap(val))),
                out_rsrc,
                b_idx * stride_qb + qtok * stride_qs + head_abs * stride_qh + d,
            )

        if const_expr(_S > 1):
            # Split CTAs hand their merged partial to whichever finishes last,
            # using agent-coherent (SC1) stores/loads and a monotonic agent
            # counter; the counter is reset by that CTA.
            _CPOL_SC1 = 16
            _PART = _M16 * (_HEAD + 2)
            _COUNTERS = 1 << 16
            ws_rsrc = buffer_ops.create_buffer_resource(ws_ptr, max_size=True)
            part0 = fx.Int32(_COUNTERS) + head_flat * fx.Int32(_S * _PART)
            my_part = part0 + split * fx.Int32(_PART)

        for i in range_constexpr(_NPASS):
            row, d, qtok, head_abs, ok = _out_coords(i)
            if ok:
                parts = []
                for w in range_constexpr(_NW):
                    base = fx.Int32(w * _R_F32 + _M_OFF) + row
                    parts.append(
                        (
                            _lds_f(base),
                            _lds_f(base + fx.Int32(_M16)),
                            _lds_f(fx.Int32(w * _R_F32) + row * fx.Int32(_HEAD) + d),
                        )
                    )
                mx, den, num = _lse_merge(parts)
                if const_expr(_S == 1):
                    _store_out(row, d, qtok, head_abs, den, num)
                else:
                    buffer_ops.buffer_store(
                        arith.unwrap(num),
                        ws_rsrc,
                        my_part + row * fx.Int32(_HEAD) + d,
                        cache_modifier=_CPOL_SC1,
                    )
                    if d == fx.Int32(0):
                        _mo = my_part + fx.Int32(_M16 * _HEAD) + row
                        buffer_ops.buffer_store(
                            arith.unwrap(mx), ws_rsrc, _mo, cache_modifier=_CPOL_SC1
                        )
                        buffer_ops.buffer_store(
                            arith.unwrap(den),
                            ws_rsrc,
                            _mo + fx.Int32(_M16),
                            cache_modifier=_CPOL_SC1,
                        )

        if const_expr(_S > 1):
            rocdl.s_waitcnt(0)
            gpu.barrier()  # every thread's partial stores done; LDS reads done
            ws_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ws_ptr)))
            counter_ptr = buffer_ops.create_llvm_ptr(
                ws_i64 + fx.Int64(head_flat * fx.Int32(4)), address_space=1
            )
            flag_ptr = buffer_ops.create_llvm_ptr(fx.Int64(0), address_space=3)
            if tid == fx.Int32(0):
                old = fx.llvm.atomic_add(
                    counter_ptr,
                    fx.Int32(1),
                    syncscope=rocdl.SyncScope.Agent,
                    ordering=fx.AtomicOrdering.Monotonic,
                )
                last = (fx.Int32(old) == fx.Int32(_S - 1)).select(
                    fx.Int32(1), fx.Int32(0)
                )
                _llvm.StoreOp(arith.unwrap(last), flag_ptr)
            gpu.barrier()
            is_last = fx.Int32(
                rocdl.readfirstlane(T.i32, _llvm.LoadOp(T.i32, flag_ptr).result)
            )
            if is_last != fx.Int32(0):
                for i in range_constexpr(_NPASS):
                    row, d, qtok, head_abs, ok = _out_coords(i)
                    if ok:
                        parts = []
                        for sp in range_constexpr(_S):
                            pb = part0 + fx.Int32(sp * _PART)
                            _mo = pb + fx.Int32(_M16 * _HEAD) + row
                            parts.append(
                                tuple(
                                    fx.Float32(
                                        buffer_ops.buffer_load(
                                            ws_rsrc,
                                            off,
                                            vec_width=1,
                                            dtype=T.f32,
                                            cache_modifier=_CPOL_SC1,
                                        )
                                    )
                                    for off in (
                                        _mo,
                                        _mo + fx.Int32(_M16),
                                        pb + row * fx.Int32(_HEAD) + d,
                                    )
                                )
                            )
                        mx, den, num = _lse_merge(parts)
                        _store_out(row, d, qtok, head_abs, den, num)
                if tid == fx.Int32(0):
                    _llvm.StoreOp(arith.unwrap(fx.Int32(0)), counter_ptr)

    return pa_decode_swa_kernel, alloc


@functools.lru_cache(maxsize=64)
def _make_swa_launcher(
    head_size, kv_dtype_str, page_size, gqa_ratio, seqlen_q, num_waves, num_splits
):
    kernel, _alloc = compile_pa_decode_swa(
        head_size=head_size,
        kv_dtype_str=kv_dtype_str,
        page_size=page_size,
        gqa_ratio=gqa_ratio,
        seqlen_q=seqlen_q,
        num_waves=num_waves,
        num_splits=num_splits,
    )
    block = num_waves * WARP_SIZE

    @flyc.jit
    def _launcher(
        out_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        seq_ptr,
        bt_ptr,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_kb,
        stride_km,
        stride_kh,
        stride_btb,
        num_hq,
        num_hkv,
        window_left,
        scale,
        ws_ptr,
        grid_x,
        stream: fx.Stream = fx.Stream(None),
    ):
        from flydsl._mlir import ir as _ir
        from flydsl.compiler.kernel_function import CompilationContext

        _alloc.finalized = False
        ctx = CompilationContext.get_current()
        with _ir.InsertionPoint(ctx.gpu_module_body):
            _alloc.finalize()
        kernel(
            out_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            seq_ptr,
            bt_ptr,
            stride_qb,
            stride_qs,
            stride_qh,
            stride_kb,
            stride_km,
            stride_kh,
            stride_btb,
            num_hq,
            num_hkv,
            window_left,
            scale,
            ws_ptr,
        ).launch(grid=(grid_x, 1, 1), block=(block, 1, 1), stream=stream)

    return _launcher


# (device, stream) -> persistent workspace whose leading counters stay zero.
_WORKSPACES: dict = {}


def _workspace(elems: int, device: torch.device, stream: Any) -> torch.Tensor:
    key = (device, stream.cuda_stream)
    buf = _WORKSPACES.get(key)
    if buf is None or buf.numel() < elems:
        buf = torch.zeros(elems, dtype=torch.float32, device=device)
        _WORKSPACES[key] = buf
    return buf


def pa_decode_swa_supported(
    *,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    seqlen_q: int,
    page_size: int,
    window_left: int,
    dtype: torch.dtype,
) -> bool:
    """Shapes pa_decode_swa_launch handles."""
    ratio = num_heads // num_kv_heads if num_kv_heads else 0
    if not (ratio and num_heads % num_kv_heads == 0 and MFMA_M % ratio == 0):
        return False
    t_pack = MFMA_M // ratio
    pages = -(-(window_left + seqlen_q + TILE_N) // page_size) + 1
    return (
        get_rocm_arch().startswith("gfx950")
        and dtype in (torch.float16, torch.bfloat16)
        and head_dim in (64, 128)
        and window_left >= 0
        and page_size % TILE_N == 0
        and pages <= WARP_SIZE
        and -(-seqlen_q // t_pack) <= 2
    )


def pa_decode_swa_launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: torch.Tensor,
    seqlen_k: torch.Tensor,
    window_left: int,
    softmax_scale: float,
    out: torch.Tensor,
    num_waves: int = 8,
    num_splits: int = 1,
) -> torch.Tensor:
    """Sliding-window paged decode. q/out are [B, Sq, H_q, D]; k/v are
    [num_pages, page_size, H_kv, D]; block_table is [B, pages] int32."""
    from mslk.flydsl.jit import run_compiled

    B, Sq, H_q, D = q.shape
    _, page_size, H_kv, _ = k.shape
    assert k.numel() < 2**31 and v.numel() < 2**31, "32-bit KV offsets"
    assert out.stride() == q.stride()
    kv_str = {torch.float16: "f16", torch.bfloat16: "bf16"}[k.dtype]
    launcher = _make_swa_launcher(
        D, kv_str, page_size, H_q // H_kv, Sq, num_waves, num_splits
    )
    if num_splits > 1:
        assert B * H_kv <= 1 << 16
        m16 = -(-Sq // (MFMA_M // (H_q // H_kv))) * MFMA_M
        ws = _workspace(
            (1 << 16) + B * H_kv * num_splits * m16 * (D + 2),
            q.device,
            torch.cuda.current_stream(),
        )
    else:
        ws = out
    sq = q.stride()
    ks = k.stride()
    assert v.stride() == ks
    run_compiled(
        launcher,
        out,
        q,
        k,
        v,
        seqlen_k,
        block_table,
        sq[0],
        sq[1],
        sq[2],
        ks[0],
        ks[1],
        ks[2],
        block_table.stride(0),
        H_q,
        H_kv,
        int(window_left),
        float(softmax_scale),
        ws,
        B * H_kv * num_splits,
        torch.cuda.current_stream(),
    )
    return out
