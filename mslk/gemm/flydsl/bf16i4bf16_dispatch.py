# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Host dispatch for the FlyDSL BF16 x INT4 rowwise GEMM.

The op modules call ``is_supported`` to decide between this kernel and Triton,
then ``bf16i4bf16_rowwise_flydsl`` to run it. This module checks the operand
contract, looks up the reordered weight, picks the tile, and launches.

Tile selection goes through mslk.flydsl.autotune, as for the grouped GEMM
(grouped_dispatch.py): tuning runs only when MSLK_AUTOTUNE_ENABLE is set, and
otherwise the default comes from ``default_tile_config``, which depends on M.

Tensor contract (bf16i4bf16_gemm.py):
  x     : [M, K] or [B, M, K] bf16
  w     : [N, K / 2] int8, two INT4 per byte
  scale : [K / group_size, N] bf16 or fp32
  zero  : same shape and dtype as scale
  out   : [M, N] or [B, M, N] bf16
"""

import functools

import torch
from mslk.flydsl.autotune import next_pow2, prune_by_divisibility, tunable
from mslk.flydsl.jit import run_compiled
from mslk.gemm.flydsl.bf16i4bf16_gemm import reordered_weight

SUPPORTED_ARCHS = ("gfx942", "gfx950")

# The kernel's K granularity: a weight window is 128 k, and the last one may be a
# half window. group_size has to be a multiple of it too, so that a scale group
# never splits a half window.
K_GRANULE = 64

# The reordered weight pads N to 16 rows, and the store masks the tail, so N
# itself only needs to be a multiple of 8 (one 16-byte bf16 store per lane pair).
N_GRANULE = 8
WEIGHT_ROW_PADDING = 16

SCALE_DTYPES = {torch.bfloat16: "bf16", torch.float32: "f32"}

#: What one AMD buffer descriptor can address: num_records and the voffset a
#: buffer_load adds are both 32-bit, so every operand must stay below this.
BUFFER_LIMIT_BYTES = 1 << 32

# PENDING fork 2 (Deferred D4): provisional per-architecture tile_k for
# tile_m >= 32. To be set from the unit 7 tile_k x roll_k sweep
# (designs/probes/bf16i4bf16-flydsl-port/unit7_tilek_roll).
DEFAULT_TILE_K_BY_ARCH = {"gfx950": 128, "gfx942": 64}
DEFAULT_TILE_N = 128

# The A loader copies the tile into LDS with all 256 threads in whole 16-byte
# loads, so a bf16 A tile must hold a multiple of 4096 bytes: tile_m = 16 reaches
# that only at tile_k = 128.
SMALLEST_TILE_M = 16
SMALLEST_TILE_M_TILE_K = 128
LARGEST_TILE_M = 256

_CANDIDATE_TILE_M = (16, 32, 64, 128, 256)
_CANDIDATE_TILE_N = (64, 128, 256)
_CANDIDATE_TILE_K = (64, 128)
# Swept as in grouped_dispatch (_WAVES_PER_EU, _WAVE_GRIDS): 0 leaves the
# occupancy hint to the compiler, and the four waves of a block can split the
# tile 1x4, 2x2 or 4x1.
_CANDIDATE_WAVES_PER_EU = (0, 2)
_CANDIDATE_WAVE_GRIDS = ((1, 4), (2, 2), (4, 1))
MFMA_TILE = 16
CANDIDATE_TILES = tuple(
    {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "waves_m": waves_m,
        "waves_n": waves_n,
        "waves_per_eu": waves_per_eu,
    }
    for tile_m in _CANDIDATE_TILE_M
    for tile_n in _CANDIDATE_TILE_N
    for tile_k in _CANDIDATE_TILE_K
    for waves_m, waves_n in _CANDIDATE_WAVE_GRIDS
    for waves_per_eu in _CANDIDATE_WAVES_PER_EU
    if not (tile_m == SMALLEST_TILE_M and tile_k != SMALLEST_TILE_M_TILE_K)
    # The wave grid has to cut the tile into whole 16x16 MFMA tiles, the rule
    # the kernel factory enforces.
    and tile_m % (waves_m * MFMA_TILE) == 0
    and tile_n % (waves_n * MFMA_TILE) == 0
)

# A tile that overruns N or K still compiles, masked, but wastes part of its work;
# when nothing divides, prune_by_divisibility falls back to the whole list.
_PRUNE = prune_by_divisibility({"tile_n": "n", "tile_k": "k"})
# roll_k is policy (should_roll_k), not tuned, as in grouped_dispatch: a fully
# unrolled candidate costs a compile that grows with K. The architecture is not
# in the key because FlyDSL's tuning cache already records the device.
_KEY = ["m_bucket", "n", "k", "group_size", "scale_dtype"]


def should_roll_k(k: int, tile_k: int) -> bool:
    """Whether to compile the K loop rolled rather than unrolled.

    PENDING fork 3: provisional policy, always unrolled (the kernel factory's
    default). To be set from the unit 7 tile_k x roll_k sweep.
    """
    return False


def default_tile_config(m: int, arch: str) -> dict:
    """The untuned tile for M rows on ``arch``.

    tile_m follows M up to the largest tile, so a decode-sized M pads at most to
    16 rows. The smallest tile needs tile_k = 128 (see SMALLEST_TILE_M_TILE_K).
    """
    tile_m = min(LARGEST_TILE_M, max(SMALLEST_TILE_M, next_pow2(m)))
    if tile_m == SMALLEST_TILE_M:
        tile_k = SMALLEST_TILE_M_TILE_K
    else:
        tile_k = DEFAULT_TILE_K_BY_ARCH[arch]
    return {"tile_m": tile_m, "tile_n": DEFAULT_TILE_N, "tile_k": tile_k}


@functools.lru_cache(maxsize=8)
def _device_arch(device: torch.device) -> str:
    """The gfx target of ``device``, without feature suffixes (gfx942:sramecc+ -> gfx942)."""
    return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]


@functools.lru_cache(maxsize=8)
def _unused_m_sizes(device: torch.device) -> torch.Tensor:
    """Stand-in for the kernel's m_sizes operand, which the batched layout never reads.

    Cached so that a call allocates nothing and the address stays stable under
    CUDA-graph capture, as grouped_dispatch.unused_group_meta.
    """
    return torch.zeros((1,), dtype=torch.int64, device=device)


def _operand_bytes(m: int, n: int, k: int, group_size: int, scale_bytes: int) -> dict:
    """Bytes each operand spans, for the buffer-descriptor limit."""
    n_padded = -(-n // WEIGHT_ROW_PADDING) * WEIGHT_ROW_PADDING
    return {
        "x": m * k * 2,
        "out": m * n * 2,
        "w (reordered)": n_padded * k // 2,
        "scale": (k // group_size) * n * scale_bytes,
    }


def is_supported(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int,
) -> bool:
    """True when the FlyDSL kernel can run this call; otherwise the caller uses Triton."""
    if not (x.is_cuda and torch.version.hip is not None):
        return False
    if _device_arch(x.device) not in SUPPORTED_ARCHS:
        return False
    if x.dtype != torch.bfloat16 or x.dim() not in (2, 3):
        return False
    if w.dtype != torch.int8 or w.dim() != 2:
        return False
    if scale.dtype not in SCALE_DTYPES or zero.dtype != scale.dtype:
        return False
    k = x.shape[-1]
    n = w.shape[0]
    m = x.numel() // k if k else 0
    if w.shape[1] * 2 != k:
        return False
    if k % K_GRANULE != 0 or n % N_GRANULE != 0:
        return False
    if group_size <= 0 or group_size % K_GRANULE != 0 or k % group_size != 0:
        return False
    expected_scale_shape = (k // group_size, n)
    if (
        tuple(scale.shape) != expected_scale_shape
        or tuple(zero.shape) != expected_scale_shape
    ):
        return False
    scale_bytes = scale.element_size()
    operand_bytes = _operand_bytes(m, n, k, group_size, scale_bytes)
    return all(nbytes < BUFFER_LIMIT_BYTES for nbytes in operand_bytes.values())


def launch(
    x,
    w_reordered,
    scale,
    zero,
    out,
    m_bucket,
    n,
    k,
    group_size,
    scale_dtype,
    *,
    tile_m,
    tile_n,
    tile_k,
    waves_m=1,
    waves_n=4,
    waves_per_eu=0,
):
    """Compile (cached) and launch the kernel for one tile config.

    ``x`` is [M, K] and ``out`` [M, N]. ``m_bucket``, ``n``, ``k``,
    ``group_size`` and ``scale_dtype`` form the autotune key; ``m_bucket``
    also selects the default tile. The wave defaults (a 1x4 grid, occupancy
    left to the compiler) are what the untuned path runs.
    """
    from mslk.flydsl.kernels.gemm.bf16i4_gemm import compile_bf16i4_gemm

    m = x.shape[0]
    num_m_tiles = -(-m // tile_m)
    launcher = compile_bf16i4_gemm(
        n=n,
        k=k,
        num_groups=1,
        group_size=group_size,
        scale_dtype=scale_dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        waves_m=waves_m,
        waves_n=waves_n,
        # 0 means the compiler picks.
        waves_per_eu=None if waves_per_eu <= 0 else waves_per_eu,
        layout="batched",
        n_padding=(n % tile_n != 0),
        roll_k=should_roll_k(k, tile_k),
    )
    # Every operand is addressed as flat bytes by the kernel, so each is handed
    # over as an int8 view in its natural shape (a flattened view would overflow
    # the int32 extents the argument marshalling packs).
    run_compiled(
        launcher,
        out.view(torch.int8),
        x.view(torch.int8),
        w_reordered.view(torch.int8),
        scale.view(torch.int8),
        zero.view(torch.int8),
        _unused_m_sizes(x.device),
        m,
        n,
        k,
        1,
        num_m_tiles,
        torch.cuda.current_stream(x.device),
    )
    return out


def _default_for_call(args) -> dict:
    return default_tile_config(args["m_bucket"], _device_arch(args["x"].device))


_tuned_launch = tunable(
    configs=CANDIDATE_TILES, default=_default_for_call, key=_KEY, prune=_PRUNE
)(launch)


def bf16i4bf16_rowwise_flydsl(
    x: torch.Tensor,
    w: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """out = x @ dequant(w).T in bf16, for a call ``is_supported`` accepted.

    Returns [M, N], or [B, M, N] for a 3D ``x``.
    """
    k = x.shape[-1]
    n = w.shape[0]
    leading_shape = x.shape[:-1]
    x_rows = x.reshape(-1, k).contiguous()
    m = x_rows.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if m == 0 or n == 0:
        return out.view(*leading_shape, n)
    if k == 0:
        # The kernel does not launch, and a contraction over nothing is zero.
        return out.zero_().view(*leading_shape, n)

    _tuned_launch(
        x_rows,
        reordered_weight(w),
        scale.contiguous(),
        zero.contiguous(),
        out,
        next_pow2(m),
        n,
        k,
        group_size,
        SCALE_DTYPES[scale.dtype],
    )
    return out.view(*leading_shape, n)
