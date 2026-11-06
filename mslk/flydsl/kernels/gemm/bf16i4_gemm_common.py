# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Building blocks for the BF16 x INT4 GEMM kernel (bf16i4_gemm.py).

The kernel takes its A path, its MFMA issue and its epilogue from the grouped
GEMM (fp8_grouped_gemm_common.py). What it adds is the B side: INT4 weights in
the layout of mslk/gemm/flydsl/bf16i4bf16_layout.py, loaded straight from HBM
into registers one 128-k window at a time, and the per-group scale and zero
each weight is dequantised with.

A window is 128 k of one 16-row block (1 KiB), or the 64 k left over when K is
an odd multiple of 64 (512 B). Lane L of a wave owns the 16 bytes at 16 * L of
a full block, or the 8 bytes at 8 * L of a half one. Its dword j holds the
eight k from 32 * j + 8 * (L // 16), which is what dword j's MFMA issue wants
from that lane on both gfx950 and gfx942.
"""

from collections import namedtuple

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm, math as math_dialect
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Vector
from mslk.flydsl import buffer_ops, vector_ops as vector

K_PER_WINDOW = 128
K_PER_HALF_WINDOW = 64
BYTES_PER_FULL_BLOCK = 1024
ROWS_PER_BLOCK = 16
DWORDS_PER_LANE_FULL = 4
DWORDS_PER_LANE_HALF = 2
DWORDS_PER_HALF_WINDOW = 2
HALVES_PER_WINDOW = 2
WEIGHTS_PER_DWORD = 8

SCALE_DTYPES = ("f32", "bf16")

# scale_x16 is 16 * scale: v_cvt_off_f32_i4 returns q / 16, and folding the 16
# into the scale makes fma(q / 16, 16 * s, z) = q * s + z with one rounding.
GroupParams = namedtuple("GroupParams", ["scale", "zero", "scale_x16"])


def make_int4_b_window_loader(
    *,
    b_rsrc,
    row_block_bytes,
    by_n,
    n_tile_base,
    lane_id,
    num_acc_n,
):
    """Build the loader of one window of reordered INT4 weights.

    Returns ``load_b_window(window, *, half_window=False)``, which issues one
    buffer load per 16-column block the wave owns and returns, per block, the
    lane's raw dwords: an i32x4 for a full window, an i32x2 for the half one.
    ``window`` is a Python int or a runtime index; ``half_window`` must be known
    while tracing, since it sets the load width.

    ``row_block_bytes`` is ``16 * K / 2``: a row block is a permutation of its
    16 stored rows, so full and half windows alike start at
    ``row_block * row_block_bytes + window * 1024``.
    """
    lane = fx.Index(lane_id)
    # by_n and n_tile_base are multiples of 16, so the wave's first column block
    # is a whole row block.
    first_row_block = (by_n + n_tile_base) // fx.Index(ROWS_PER_BLOCK)

    def load_b_window(window, *, half_window=False):
        dwords_per_lane = DWORDS_PER_LANE_HALF if half_window else DWORDS_PER_LANE_FULL
        raw_dwords = []
        for ni in range_constexpr(num_acc_n):
            row_block = first_row_block + fx.Index(ni)
            block_start_bytes = row_block * fx.Index(row_block_bytes) + fx.Index(
                window
            ) * fx.Index(BYTES_PER_FULL_BLOCK)
            lane_start_dwords = block_start_bytes // fx.Index(4) + lane * fx.Index(
                dwords_per_lane
            )
            words = buffer_ops.buffer_load(
                b_rsrc, lane_start_dwords, vec_width=dwords_per_lane, dtype=T.i32
            )
            raw_dwords.append(Vector(words))
        return raw_dwords

    return load_b_window


def make_group_param_loader(
    *,
    scale_rsrc,
    zero_rsrc,
    scale_dtype,
    group_size,
    n_in,
    by_n,
    n_tile_base,
    lane_mod_16,
    num_acc_n,
):
    """Build the loader of the scale and zero a lane needs for one window.

    Scales and zeros are ``[K / group_size, N]``, so a lane reads those of its own
    column. ``group_size`` is a multiple of 64, so each 64-k half of a window
    lies inside one group: dwords 0 and 1 of a lane share the first half's
    group, dwords 2 and 3 the second's.

    Returns ``load_group_params(window, *, half_window=False)``, which gives,
    per 16-column block, a list indexed by window half of ``GroupParams`` holding
    f32 values. When ``group_size`` is a multiple of 128 both halves are one
    group, loaded once and listed twice. A half window has only the first half.
    """
    if scale_dtype not in SCALE_DTYPES:
        raise ValueError(
            f"scale_dtype must be one of {SCALE_DTYPES}, got {scale_dtype!r}"
        )
    if group_size % K_PER_HALF_WINDOW != 0:
        raise ValueError(f"group_size ({group_size}) must be a multiple of 64")
    one_group_per_window = group_size % K_PER_WINDOW == 0

    columns = [
        by_n + n_tile_base + fx.Index(ni * 16) + lane_mod_16
        for ni in range_constexpr(num_acc_n)
    ]

    def load_as_f32(rsrc, element_index):
        if scale_dtype == "f32":
            return buffer_ops.buffer_load(rsrc, element_index, vec_width=1, dtype=T.f32)
        # A bf16 is the upper half of the f32 with the same value.
        bf16_bits = buffer_ops.buffer_load(
            rsrc, element_index, vec_width=1, dtype=T.i16
        )
        f32_bits = ArithValue(bf16_bits).extui(T.i32) << fx.Int32(16)
        return arith.bitcast(T.f32, f32_bits)

    def load_group_params(window, *, half_window=False):
        window_first_k = fx.Index(window) * fx.Index(K_PER_WINDOW)
        loaded_halves = 1 if (half_window or one_group_per_window) else 2
        params = []
        for ni in range_constexpr(num_acc_n):
            per_half = []
            for half in range_constexpr(loaded_halves):
                half_first_k = window_first_k + fx.Index(half * K_PER_HALF_WINDOW)
                group = half_first_k // fx.Index(group_size)
                element_index = group * n_in + columns[ni]
                scale = ArithValue(load_as_f32(scale_rsrc, element_index))
                per_half.append(
                    GroupParams(
                        scale=scale,
                        zero=ArithValue(load_as_f32(zero_rsrc, element_index)),
                        scale_x16=scale * fx.Float32(16.0),
                    )
                )
            if not half_window and loaded_halves == 1:
                per_half.append(per_half[0])
            params.append(per_half)
        return params

    return load_group_params


def make_int4_dequantizer(*, is_gfx950):
    """Build the dequant of reordered INT4 weights, one 64-byte step of A at a time.

    Returns ``dequantize_dwords(dwords, scales_x16, zeros) -> [(b0, b1), ...]``,
    one entry per 16-column block. Each dword holds eight weights of consecutive
    k: k + t in the low nibble of byte t and k + 4 + t in its high nibble. ``b0``
    and ``b1`` are i64s of four bf16 each, k to k + 3 and k + 4 to k + 7, which is
    the pair of B packs ``compute_tile`` consumes for that step.

    Each weight is ``fma(q / 16, 16 * s, z)``, one f32 rounding of ``q * s + z``.
    The round to bf16 is to nearest on gfx950, with v_cvt_pk_bf16_f32, as the
    Triton kernel rounds. gfx942 has no such instruction and keeps the upper 16
    bits, which rounds towards zero by up to one bf16 ULP.

    Both conversions are emitted as LLVM operations, not through FlyDSL's
    inline-asm wrappers for the same instructions. LLVM does not look inside
    inline asm, so it cannot insert the wait states CDNA requires between an
    MFMA reading a VGPR as SrcC and a VALU write to it (CDNA4 ISA 7.6; LLVM pads
    this MFMA's SrcC reads to 3 wait states). With the dequant interleaved with
    the MFMAs, the inline forms overwrote a live accumulator.

    WORKAROUND (gfx950 packed FP32 hazard): the FMAs are written as two-wide
    vector FMAs that pair column block ni with ni + 1, so that the multiplier is
    a genuine pair (16 * s of each block) rather than one scalar shared by both
    halves. Written as scalars, LLVM pairs them itself into v_pk_fma_f32 and can
    address a shared scalar held in an odd VGPR as the high half of a pair, with
    op_sel selecting that high half for the low result. On gfx950, a packed FP32
    op of that form issued while an MFMA executes and followed by another MFMA
    within 2 wait states reads that operand as 0 on lanes 48-63, in about half
    the runs; LLVM does not model it and the CDNA4 ISA does not list it. A block
    with no partner (an odd number of blocks per wave) still pairs consecutive
    weights over one shared multiplier, which leaves that form possible.
    """
    c16 = fx.Int32(16)
    upper_half_mask = fx.Int32(0xFFFF0000)

    def pack_two_bf16(low, high):
        if is_gfx950:
            pair = Vector.from_elements([low, high], fx.Float32)
            as_bf16x2 = arith.trunc_f(T.vec(2, T.bf16), pair)
            return vector.extract(
                vector.bitcast(T.vec(1, T.i32), as_bf16x2),
                static_position=[0],
                dynamic_position=[],
            )
        low_bits = ArithValue(arith.bitcast(T.i32, low))
        high_bits = ArithValue(arith.bitcast(T.i32, high))
        return (low_bits >> c16) | (high_bits & upper_half_mask)

    def pack_four_bf16(weights):
        low_pair = pack_two_bf16(weights[0], weights[1])
        high_pair = pack_two_bf16(weights[2], weights[3])
        as_i32x2 = vector.from_elements(T.vec(2, T.i32), [low_pair, high_pair])
        as_i64x1 = vector.bitcast(T.vec(1, T.i64), as_i32x2)
        return fx.Int64(
            vector.extract(as_i64x1, static_position=[0], dynamic_position=[])
        )

    def low_nibble_of_byte(source, byte):
        # v_cvt_off_f32_i4 reads bits 3:0. These are the forms LLVM folds into
        # an SDWA byte or word select instead of a separate shift.
        if byte == 0:
            return source
        if byte == 1:
            return (source >> fx.Int32(8)) & fx.Int32(0xFF)
        return source >> fx.Int32(8 * byte)

    def int4_to_f32_over_16(source):
        return llvm.call_intrinsic(
            T.f32, "llvm.amdgcn.cvt.off.f32.i4", [arith._to_raw(source)], [], []
        )

    def quarters_of_dword(dword):
        """The dword's eight q / 16, in k order: low nibbles of bytes 0-3, then high."""
        high_nibbles = dword >> fx.Int32(4)
        return [
            int4_to_f32_over_16(low_nibble_of_byte(source, byte))
            for source in (dword, high_nibbles)
            for byte in range_constexpr(4)
        ]

    def fma_pair(multiplicands, multipliers, addends):
        """Two FMAs as one two-wide vector FMA; returns the two results."""
        result = Vector(
            math_dialect.fma(
                Vector.from_elements(multiplicands, fx.Float32),
                Vector.from_elements(multipliers, fx.Float32),
                Vector.from_elements(addends, fx.Float32),
            )
        )
        return result[0], result[1]

    def dequantize_dwords(dwords, scales_x16, zeros):
        q_over_16 = [quarters_of_dword(dword) for dword in dwords]
        weights = [[None] * WEIGHTS_PER_DWORD for _ in dwords]
        paired_blocks = len(dwords) - len(dwords) % 2
        for first in range_constexpr(0, paired_blocks, 2):
            second = first + 1
            for k in range_constexpr(WEIGHTS_PER_DWORD):
                weights[first][k], weights[second][k] = fma_pair(
                    [q_over_16[first][k], q_over_16[second][k]],
                    [scales_x16[first], scales_x16[second]],
                    [zeros[first], zeros[second]],
                )
        if len(dwords) % 2:
            last = len(dwords) - 1
            for k in range_constexpr(0, WEIGHTS_PER_DWORD, 2):
                weights[last][k], weights[last][k + 1] = fma_pair(
                    [q_over_16[last][k], q_over_16[last][k + 1]],
                    [scales_x16[last], scales_x16[last]],
                    [zeros[last], zeros[last]],
                )
        return [(pack_four_bf16(w[0:4]), pack_four_bf16(w[4:8])) for w in weights]

    return dequantize_dwords


def make_b_tile_dequantizer(*, dequantize_dwords, num_acc_n):
    """Build the assembler of one A tile's B operand from a loaded window.

    Returns ``dequantize_b_tile(raw_window, params, first_dword, dwords_in_tile)``.
    ``raw_window`` and ``params`` are what ``load_b_window`` and
    ``load_group_params`` returned for the window this A tile lies in;
    ``first_dword`` is the window dword holding the tile's first k (0, or 2 for
    the second 64-k tile of a window). The result is a list over the tile's
    64-byte steps ``ku`` of ``(packs0[ni], packs1[ni])``, the form
    ``compute_tile`` takes as its B tile. All column blocks of a step are
    dequantised together, so that ``dequantize_dwords`` can pair them.
    """

    def dequantize_b_tile(raw_window, params, first_dword, dwords_in_tile):
        b_tile = []
        for ku in range_constexpr(dwords_in_tile):
            dword_index = first_dword + ku
            window_half = dword_index // DWORDS_PER_HALF_WINDOW
            group_params = [
                params[ni][window_half] for ni in range_constexpr(num_acc_n)
            ]
            packs = dequantize_dwords(
                [
                    ArithValue(Vector(raw_window[ni])[dword_index])
                    for ni in range_constexpr(num_acc_n)
                ],
                [gp.scale_x16 for gp in group_params],
                [gp.zero for gp in group_params],
            )
            b_tile.append(([b0 for b0, _ in packs], [b1 for _, b1 in packs]))
        return b_tile

    return dequantize_b_tile


def make_window_step(
    *,
    tiles_per_window,
    k_unroll,
    prefetch_a_tile,
    store_a_tile_to_lds,
    compute_tile,
    dequantize_b_tile,
    lds_base,
):
    """Build the K-loop body for one window of B.

    Returns ``window_step(accs, a_regs, b_raw, params, first_tile, *,
    half_window=False, is_last_window=False, load_next_window=None)``, which
    walks the A tiles the window spans and returns ``(accs, a_regs,
    next_window)``.

    The A side is the grouped GEMM's plain K loop, tile by tile: publish the
    tile already in registers to the single LDS buffer, issue the next tile's
    global loads, compute, and fence before the buffer is overwritten. B differs
    in where it comes from: the window's dwords are already in registers, and
    ``load_next_window`` issues the next window's loads first, so they overlap
    this window's tiles as the next A tile's loads do.

    A window spans ``tiles_per_window`` A tiles of ``k_unroll`` dwords each. The
    half window spans one tile; where that tile is wider than the 64 k left, the
    A loads past K are masked to zero and the B steps past it are zero packs.
    """

    def window_step(
        accs,
        a_regs,
        b_raw,
        params,
        first_tile,
        *,
        half_window=False,
        is_last_window=False,
        load_next_window=None,
    ):
        next_window = load_next_window() if load_next_window is not None else None
        tiles = 1 if half_window else tiles_per_window
        dwords_in_window = DWORDS_PER_LANE_HALF if half_window else DWORDS_PER_LANE_FULL
        zero_pack = fx.Int64(arith.constant(0, type=T.i64))
        for tile in range_constexpr(tiles):
            k_tile = first_tile + tile
            store_a_tile_to_lds(a_regs, lds_base)
            gpu.barrier()
            if not (is_last_window and tile == tiles - 1):
                a_regs = prefetch_a_tile(k_tile + 1)
            first_dword = tile * k_unroll
            dwords_in_tile = min(k_unroll, dwords_in_window - first_dword)
            b_tile = dequantize_b_tile(b_raw, params, first_dword, dwords_in_tile)
            num_acc_n = len(b_raw)
            for _ in range_constexpr(k_unroll - dwords_in_tile):
                b_tile.append(([zero_pack] * num_acc_n, [zero_pack] * num_acc_n))
            accs = compute_tile(accs, k_tile, lds_base, b_tile, None)
            gpu.barrier()
        return accs, a_regs, next_window

    return window_step


def flatten_window(b_raw, params):
    """List a window's B registers and group parameters as loop state."""
    values = list(b_raw)
    for per_half in params:
        for group_params in per_half:
            values.extend(group_params)
    return values


def unflatten_window(values, num_acc_n):
    """Inverse of ``flatten_window`` for a full window's state."""
    b_raw = [Vector(v) for v in values[:num_acc_n]]
    rest = values[num_acc_n:]
    fields = len(GroupParams._fields)
    params = []
    for ni in range_constexpr(num_acc_n):
        per_half = []
        for half in range_constexpr(HALVES_PER_WINDOW):
            start = (ni * HALVES_PER_WINDOW + half) * fields
            per_half.append(
                GroupParams(*[ArithValue(v) for v in rest[start : start + fields]])
            )
        params.append(per_half)
    return b_raw, params
