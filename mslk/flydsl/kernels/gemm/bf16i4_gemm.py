# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BF16 x INT4 GEMM kernel with groupwise scale and zero.

Derived from the grouped GEMM (fp8_grouped_gemm.py). The A path -- global loads
into registers, an XOR-swizzled LDS tile, 16-byte LDS reads per lane --, the
MFMA issue, the group resolution and the epilogue are that kernel's bf16 path,
reused through fp8_grouped_gemm_common. B is different: INT4 weights reordered
by mslk/gemm/flydsl/bf16i4bf16_layout.py are loaded straight from HBM into
registers, one 128-k window per wave and 16-column block, and dequantised in
registers just before the MFMA. See bf16i4_gemm_common.py.

    w[n, k] = q[n, k] * scale[k // group_size, n] + zero[k // group_size, n]
    out[m, n] = sum_k A[m, k] * w[n, k]

with q the signed INT4 value, (nibble XOR 8) - 8.

Tensors:
  - A: [M_total, K] bf16, the rows of every group, packed or in per-group slabs.
  - B: [num_groups, B_bytes] uint8, each group's weights in the reordered layout,
    B_bytes = n_padded * K / 2 with N padded to 16.
  - scale, zero: [num_groups, K / group_size, N], f32 or bf16 (``scale_dtype``).
  - m_sizes: [num_groups] int64 row counts for the "sizes" layout; unread for
    "batched".
  - D: [M_total, N] bf16 output.

``layout`` is the grouped GEMM's: "batched" gives each group a fixed slab of
M_total / num_groups rows on grid z, which with one group is a plain GEMM;
"sizes" packs the groups along M and resolves the owner of each M tile on the
device from m_sizes.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T, Vector
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from mslk.flydsl import buffer_ops, vector_ops as vector
from mslk.flydsl.kernels.common.kernels_common import default_f8_type
from mslk.flydsl.kernels.gemm.bf16i4_gemm_common import (
    flatten_window,
    K_PER_HALF_WINDOW,
    K_PER_WINDOW,
    make_b_tile_dequantizer,
    make_group_param_loader,
    make_int4_b_window_loader,
    make_int4_dequantizer,
    make_window_step,
    ROWS_PER_BLOCK,
    unflatten_window,
)
from mslk.flydsl.kernels.gemm.fp8_grouped_gemm import CSHUFFLE_NLANE
from mslk.flydsl.kernels.gemm.fp8_grouped_gemm_common import (
    _check_lds_budget,
    compute_compile_constants,
    compute_mfma_tiling,
    init_accumulators,
    make_a_tile_loaders,
    make_compute_tile,
    make_epilogue_writers,
    make_k_tail_mask,
    make_lds_loader,
    out_mlir_for,
    resolve_group_rows,
    SCALING_NONE,
    validate_params,
)
from mslk.flydsl.kernels.mma.mfma_epilogues import mfma_epilog

LAYOUTS = ("batched", "sizes")
TILE_KS = (64, 128)
SCALE_BYTES = {"f32": 4, "bf16": 2}
BF16_BYTES = 2


@functools.lru_cache(maxsize=128)
def compile_bf16i4_gemm(
    *,
    n: int,
    k: int,
    num_groups: int,
    group_size: int,
    scale_dtype: str,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    waves_m: int = 1,
    waves_n: int = 4,
    waves_per_eu: int | None = None,
    layout: str = "batched",
    n_padding: bool = False,
    roll_k: bool = False,
    out_dtype: str = "bf16",
):
    """Compile the BF16 x INT4 GEMM kernel and return the JIT launcher.

    Args:
        n, k: output columns and reduction length per group. K is a multiple of
            64; when it is an odd multiple the last window is a half window.
        num_groups: groups in the stack; 1 for a plain GEMM.
        group_size: K extent of one scale and zero, a multiple of 64 dividing K.
        scale_dtype: "f32" or "bf16", the dtype of scale and zero.
        tile_m, tile_n: output tile. tile_m is a multiple of 16 * waves_m.
        tile_k: 64 or 128, the K extent of one A tile. A window is 128 k, so it
            spans 128 / tile_k A tiles and its B registers serve all of them.
        waves_m, waves_n: the wave grid over the output tile, as in the grouped
            GEMM; waves_n * 16 divides tile_n.
        waves_per_eu: minimum waves per EU asked of the register allocator.
        layout: "batched" or "sizes"; see the module docstring.
        n_padding: mask the store past N, so N need only be a multiple of 8.
        roll_k: emit the K loop once as a loop over windows rather than unrolled
            over every window. The unrolled form folds each window's index into
            its addresses; the rolled one holds the IR and the compile time
            constant in K, carrying the accumulators, the next A tile and the
            next window's B as loop state.
        out_dtype: "bf16" or "f16".
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if tile_k not in TILE_KS:
        raise ValueError(f"tile_k must be one of {TILE_KS}, got {tile_k}")
    if scale_dtype not in SCALE_BYTES:
        raise ValueError(
            f"scale_dtype must be one of {tuple(SCALE_BYTES)}, got {scale_dtype!r}"
        )
    if k % K_PER_HALF_WINDOW != 0:
        raise ValueError(f"k ({k}) must be a multiple of {K_PER_HALF_WINDOW}")
    if group_size % K_PER_HALF_WINDOW != 0 or k % group_size != 0:
        raise ValueError(
            f"group_size ({group_size}) must be a multiple of {K_PER_HALF_WINDOW} "
            f"dividing k ({k})"
        )
    if waves_m < 1 or waves_n < 1:
        raise ValueError(f"wave grid must be positive, got {waves_m}x{waves_n}")
    if tile_m % (waves_m * 16) or tile_n % (waves_n * 16):
        raise ValueError(
            f"tile {tile_m}x{tile_n} does not divide into a {waves_m}x{waves_n} "
            "wave grid of whole 16x16 MFMA tiles"
        )
    num_waves = waves_m * waves_n

    full_windows, k_past_full_windows = divmod(k, K_PER_WINDOW)
    has_half_window = k_past_full_windows != 0
    tiles_per_window = K_PER_WINDOW // tile_k
    # Only a tile wider than the half window reaches past K; its A loads there
    # are masked to zero.
    k_padding = k % tile_k != 0

    gpu_arch = get_hip_arch()
    is_gfx950 = str(gpu_arch).startswith("gfx95")

    validate_params(
        n=n,
        k=k,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=tile_k,
        scale_block_n=tile_n,
        out_dtype=out_dtype,
        scaling=SCALING_NONE,
        in_dtype="bf16",
        k_padding=k_padding,
        n_padding=n_padding,
    )
    _c = compute_compile_constants(
        n=n,
        k=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=tile_k,
        scale_block_n=tile_n,
        k_padding=k_padding,
        in_dtype="bf16",
        scaling=SCALING_NONE,
        num_waves=num_waves,
    )
    total_threads = _c.total_threads
    elem_bytes = _c.elem_bytes
    num_k_tiles = _c.num_k_tiles
    k_unroll = _c.k_unroll
    tile_k_bytes = _c.tile_k_bytes

    # B never touches LDS, and the K loop uses one A buffer; the epilogue output
    # aliases the same arena once the loop is done. Checked before tracing: an
    # overflow during tracing kills the process.
    lds_a_bytes = tile_m * tile_k_bytes
    lds_out_bytes = tile_m * tile_n * BF16_BYTES
    _check_lds_budget(
        variant="bf16i4",
        total=max(lds_a_bytes, lds_out_bytes),
        detail=f"A {lds_a_bytes}, epilogue {lds_out_bytes}",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        arch=gpu_arch,
    )
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_bf16i4_gemm")
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + max(lds_a_bytes, lds_out_bytes)
    out_mlir = out_mlir_for(out_dtype)

    n_padded = -(-n // ROWS_PER_BLOCK) * ROWS_PER_BLOCK
    row_block_bytes = ROWS_PER_BLOCK * k // 2
    b_group_bytes = n_padded * k // 2
    num_scale_groups = k // group_size
    params_group_bytes = num_scale_groups * n * SCALE_BYTES[scale_dtype]
    slab_layout = layout == "batched"

    _npad = "_npad" if n_padding else ""
    _roll = "_rollk" if roll_k else ""
    _wpe = f"_wpe{int(waves_per_eu)}" if waves_per_eu else ""
    _wg = f"_w{waves_m}x{waves_n}" if (waves_m, waves_n) != (1, 4) else ""
    module_name = (
        f"bf16i4_gemm_{layout}_{scale_dtype}_{out_dtype}_n{n}_k{k}_g{num_groups}"
        f"_gs{group_size}_t{tile_m}x{tile_n}x{tile_k}{_npad}{_roll}{_wpe}{_wg}"
    )
    _kernel_attrs = (
        {"known_block_size": [total_threads, 1, 1]} if total_threads > 256 else {}
    )

    @flyc.kernel(name=module_name, **_kernel_attrs)
    def bf16i4_gemm_kernel(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale: fx.Tensor,
        arg_zero: fx.Tensor,
        arg_m_sizes: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_num_groups: fx.Int32,
    ):
        m_in = fx.Index(i32_m)
        n_in = fx.Index(i32_n)
        k_in = fx.Index(i32_k)
        num_groups_in = fx.Index(i32_num_groups)

        tx = gpu.thread_id("x")
        by = gpu.block_id("x")  # N block
        bx = gpu.block_id("y")  # M tile
        bz = gpu.block_id("z")  # group, in the slab layout
        by_n = by * fx.Index(tile_n)

        layout_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1))
        coord_wave_lane = fx.idx2crd(fx.Int32(tx), layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)
        if const_expr(waves_m > 1):
            m_wave_base = (wave_id // fx.Index(waves_n)) * fx.Index(tile_m // waves_m)
        else:
            m_wave_base = None
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        coord_lane16 = fx.idx2crd(fx.Int32(lane_id), layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        # The arena is addressed as raw bytes, as in the grouped GEMM.
        base_ptr = allocator.get_base()
        lds_a = SmemPtr(
            base_ptr, lds_alloc_offset, default_f8_type(), shape=(lds_a_bytes,)
        ).get()
        layout_lds = fx.make_layout((tile_m, tile_k_bytes), stride=(tile_k_bytes, 1))
        lds_base = fx.Index(0)
        lds_out = SmemPtr(
            base_ptr, lds_alloc_offset, out_mlir(), shape=(tile_m * tile_n,)
        ).get()

        _grp = resolve_group_rows(
            arg_m_sizes=arg_m_sizes,
            num_groups_in=num_groups_in,
            m_in=m_in,
            m_tile_idx=bx,
            slab_idx=bz,
            tile_m=tile_m,
            num_groups=num_groups,
            layout=layout,
        )
        if _grp.is_valid:
            group_idx = fx.Index(_grp.group_id)
            bx_m = fx.Index(_grp.row_start)
            row_limit_i32 = _grp.row_limit

            k_bytes_in = k_in * fx.Index(elem_bytes)
            a_rsrc = buffer_ops.create_buffer_resource(
                arg_a, max_size=False, num_records_bytes=m_in * k_bytes_in
            )
            d_rsrc = buffer_ops.create_buffer_resource(
                arg_d, max_size=False, num_records_bytes=m_in * n_in * fx.Index(2)
            )

            # B, scale and zero are based at this block's group, so the group
            # term is a 64-bit base offset and every load stays inside one group.
            def group_resource(arg, group_bytes):
                return buffer_ops.create_buffer_resource(
                    arg,
                    max_size=False,
                    num_records_bytes=fx.Index(group_bytes),
                    base_byte_offset=group_idx * fx.Index(group_bytes),
                )

            b_rsrc = group_resource(arg_b, b_group_bytes)
            scale_rsrc = group_resource(arg_scale, params_group_bytes)
            zero_rsrc = group_resource(arg_zero, params_group_bytes)

            _t = compute_mfma_tiling(
                tile_m=tile_m, tile_n=tile_n, waves_m=waves_m, waves_n=waves_n
            )
            m_repeat = _t.m_repeat
            num_acc_n = _t.num_acc_n
            acc_init, accs = init_accumulators(_t.num_accs)
            n_tile_base = (wave_id % fx.Index(waves_n)) * fx.Index(_t.n_per_wave)

            k_tail_mask = make_k_tail_mask(
                k_padding=k_padding,
                num_k_tiles=num_k_tiles,
                k=k,
                tile_k=tile_k,
                k_in=k_in,
                elem_bytes=elem_bytes,
                always=roll_k,
            )
            (
                prefetch_a_tile,
                store_a_tile_to_lds,
                _a_row_local,
                _a_col_local_i32,
                k_blocks16,
            ) = make_a_tile_loaders(
                a_rsrc=a_rsrc,
                lds_a=lds_a,
                layout_lds=layout_lds,
                bx_m=bx_m,
                tx=tx,
                tile_m=tile_m,
                tile_k=tile_k,
                tile_k_bytes=tile_k_bytes,
                tile_k_dwords=_c.tile_k_dwords,
                chunk_i32_a=_c.chunk_i32_a,
                num_a_loads=_c.num_a_loads,
                total_threads=total_threads,
                elem_bytes=elem_bytes,
                k_in=k_in,
                k_tail_mask=k_tail_mask,
            )
            lds_load_packs_k64 = make_lds_loader(
                lds_a=lds_a, layout_lds=layout_lds, k_blocks16=k_blocks16
            )
            compute_tile = make_compute_tile(
                m_wave_base=m_wave_base,
                _use_hw_scale=False,
                _is_gfx950=is_gfx950,
                lds_load_packs_k64=lds_load_packs_k64,
                sa_rsrc=None,
                sb_rsrc=None,
                group_idx=group_idx,
                scale_n=0,
                scale_k=0,
                c_scale_k=None,
                n_block_for_scale=None,
                bx_m=bx_m,
                lane_mod_16=lane_mod_16,
                lane_div_16=lane_div_16,
                m_in=m_in,
                sb_per_tile=_c.sb_per_tile,
                m_repeat=m_repeat,
                num_acc_n=num_acc_n,
                ku_per_sb=_c.ku_per_sb,
                col_offset_base_bytes=lane_div_16 * fx.Index(16),
                mfma_res_ty=T.f32x4,
                acc_init=acc_init,
                scaling=SCALING_NONE,
                in_dtype="bf16",
            )

            load_b_window = make_int4_b_window_loader(
                b_rsrc=b_rsrc,
                row_block_bytes=row_block_bytes,
                by_n=by_n,
                n_tile_base=n_tile_base,
                lane_id=lane_id,
                num_acc_n=num_acc_n,
            )
            load_group_params = make_group_param_loader(
                scale_rsrc=scale_rsrc,
                zero_rsrc=zero_rsrc,
                scale_dtype=scale_dtype,
                group_size=group_size,
                n_in=n_in,
                by_n=by_n,
                n_tile_base=n_tile_base,
                lane_mod_16=lane_mod_16,
                num_acc_n=num_acc_n,
            )
            dequantize_b_tile = make_b_tile_dequantizer(
                dequantize_dwords=make_int4_dequantizer(is_gfx950=is_gfx950),
                num_acc_n=num_acc_n,
            )
            window_step = make_window_step(
                tiles_per_window=tiles_per_window,
                k_unroll=k_unroll,
                prefetch_a_tile=prefetch_a_tile,
                store_a_tile_to_lds=store_a_tile_to_lds,
                compute_tile=compute_tile,
                dequantize_b_tile=dequantize_b_tile,
                lds_base=lds_base,
            )

            def load_window(window, half_window=False):
                return (
                    load_b_window(window, half_window=half_window),
                    load_group_params(window, half_window=half_window),
                )

            # The window after `window`, as loaded by the step before it.
            def next_window_loader(window):
                following = window + 1
                if const_expr(following < full_windows):
                    return lambda: load_window(following)
                if const_expr(following == full_windows and has_half_window):
                    return lambda: load_window(following, half_window=True)
                return None

            a_regs = prefetch_a_tile(0)
            b_raw, params = load_window(0, half_window=(full_windows == 0))

            # Windows whose successor is a full window, then the last full one,
            # then the half window. The rolled form iterates over the first
            # group and walks the rest unrolled, so its body has no tail tests.
            steady_windows = max(full_windows - 1, 0)
            if const_expr(roll_k and steady_windows > 0):
                _n_acc = len(accs)
                _n_a = len(a_regs)
                for _w, _st in fx.range(
                    0,
                    steady_windows,
                    1,
                    init=list(accs) + list(a_regs) + flatten_window(b_raw, params),
                ):
                    _accs = [Vector(v) for v in _st[:_n_acc]]
                    _a_cur = [Vector(v) for v in _st[_n_acc : _n_acc + _n_a]]
                    _b_cur, _p_cur = unflatten_window(_st[_n_acc + _n_a :], num_acc_n)
                    _accs, _a_nxt, (_b_nxt, _p_nxt) = window_step(
                        _accs,
                        _a_cur,
                        _b_cur,
                        _p_cur,
                        _w * fx.Index(tiles_per_window),
                        load_next_window=lambda: load_window(_w + fx.Index(1)),
                    )
                    _res = yield (
                        list(_accs) + list(_a_nxt) + flatten_window(_b_nxt, _p_nxt)
                    )
                accs = [Vector(v) for v in _res[:_n_acc]]
                a_regs = [Vector(v) for v in _res[_n_acc : _n_acc + _n_a]]
                b_raw, params = unflatten_window(_res[_n_acc + _n_a :], num_acc_n)
                first_unrolled_window = steady_windows
            else:
                first_unrolled_window = 0

            for window in range_constexpr(first_unrolled_window, full_windows):
                is_last = window == full_windows - 1 and not has_half_window
                accs, a_regs, following = window_step(
                    accs,
                    a_regs,
                    b_raw,
                    params,
                    window * tiles_per_window,
                    is_last_window=is_last,
                    load_next_window=next_window_loader(window),
                )
                if const_expr(following is not None):
                    b_raw, params = following
            if const_expr(has_half_window):
                accs, a_regs, _ = window_step(
                    accs,
                    a_regs,
                    b_raw,
                    params,
                    full_windows * tiles_per_window,
                    half_window=True,
                    is_last_window=True,
                )

            # ===== Epilogue: the grouped GEMM's CShuffle vectorised stores =====
            e_vec = 4 if (tile_n % (CSHUFFLE_NLANE * 4)) == 0 else 2
            write_row_to_lds, store_pair = make_epilogue_writers(
                accs=accs,
                d_rsrc=d_rsrc,
                out_mlir=out_mlir,
                e_vec=e_vec,
                c_n=n_in,
                n_padding=n_padding,
            )

            def precompute_row(*, row_local, row):
                row_i32 = arith.index_cast(T.i32, row)
                row_valid = arith.cmpi(arith.CmpIPredicate.ult, row_i32, row_limit_i32)
                return (None, row_valid)

            mfma_epilog(
                use_cshuffle=True,
                m_wave_base=m_wave_base,
                arith=arith,
                vector=vector,
                gpu=gpu,
                scf=scf,
                range_constexpr=range_constexpr,
                tile_m=tile_m,
                tile_n=tile_n,
                e_vec=e_vec,
                block_size=total_threads,
                m_repeat=m_repeat,
                num_acc_n=num_acc_n,
                tx=tx,
                lane_div_16=lane_div_16,
                lane_mod_16=lane_mod_16,
                bx_m=bx_m,
                by_n=by_n,
                n_tile_base=n_tile_base,
                lds_out=lds_out,
                frag_elem_type=out_mlir(),
                write_row_to_lds=write_row_to_lds,
                precompute_row=precompute_row,
                store_pair=store_pair,
            )

    @flyc.jit
    def launch_bf16i4_gemm(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale: fx.Tensor,
        arg_zero: fx.Tensor,
        arg_m_sizes: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_num_groups: fx.Int32,
        i32_num_m_tiles: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_in = fx.Index(i32_n)
        if const_expr(n_padding):
            gx = (n_in + fx.Index(tile_n - 1)) // fx.Index(tile_n)
        else:
            gx = n_in // fx.Index(tile_n)
        gy = fx.Index(i32_num_m_tiles)
        gz = fx.Index(i32_num_groups) if const_expr(slab_layout) else fx.Index(1)

        launcher = bf16i4_gemm_kernel(
            arg_d,
            arg_a,
            arg_b,
            arg_scale,
            arg_zero,
            arg_m_sizes,
            i32_m,
            i32_n,
            i32_k,
            i32_num_groups,
        )
        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, _wpe
                        )
        launcher.launch(grid=(gx, gy, gz), block=(total_threads, 1, 1), stream=stream)

    return launch_bf16i4_gemm
