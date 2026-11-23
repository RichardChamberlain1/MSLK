# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""INT4 weight layout read by the FlyDSL BF16 x INT4 GEMM.

The op stores weights as ``W: int8 [N, K/2]``, with k even in the low nibble of
each byte. The kernel instead reads a copy in which each lane of a wave finds
the weights of one MFMA operand in one contiguous 16-byte run, so that one
``buffer_load_dwordx4`` per lane per 128-k window replaces four 4-byte loads.

The copy is a sequence of blocks, row block first, then K window. A block holds
16 weight rows and one 128-k window (1 KiB), or the last 64 k when K is an odd
multiple of 64 (512 B). Inside a block, with ``n = 16 * row_block + row`` and
``k = 128 * window + 32 * dword + 8 * lane_group + 4 * nibble + byte``:

* lane ``16 * lane_group + row`` owns bytes ``[bytes_per_lane * lane, ...)``,
  where ``bytes_per_lane`` is 16 in a full block and 8 in a half block;
* its dword ``dword`` holds k ``32 * dword + 8 * lane_group`` to ``+ 7``, so the
  lane's k slots follow the bf16 MFMA operand rule on gfx950 and gfx942 alike;
* inside that dword, byte ``byte`` holds ``k + byte`` in its low nibble and
  ``k + 4 + byte`` in its high nibble, which is the order
  ``unpack_b_w4a16`` unpacks into consecutive k without an interleave.

Every row block is a permutation of its 16 stored rows, so it occupies
``16 * K / 2`` bytes and the copy is the size of ``W`` with N padded to 16.

This module imports no FlyDSL, so it can be used and tested without it.
"""

from typing import NamedTuple

import torch

ROWS_PER_BLOCK = 16
K_PER_WINDOW = 128
BYTES_PER_FULL_BLOCK = 1024
LANES = 64
LANE_GROUPS = 4
K_PER_DWORD = 8


class CachedLayout(NamedTuple):
    """Shape facts the kernel needs to address the reordered copy."""

    n_padded: int
    row_block_bytes: int
    full_windows: int
    has_half_window: bool


def cached_layout(n: int, k: int) -> CachedLayout:
    assert k % 64 == 0, f"K ({k}) must be a multiple of 64"
    row_blocks = -(-n // ROWS_PER_BLOCK)
    full_windows, k_remainder = divmod(k, K_PER_WINDOW)
    return CachedLayout(
        n_padded=row_blocks * ROWS_PER_BLOCK,
        row_block_bytes=ROWS_PER_BLOCK * k // 2,
        full_windows=full_windows,
        has_half_window=k_remainder != 0,
    )


def reorder_int4_for_mfma(W: torch.Tensor) -> torch.Tensor:
    """Return the reordered copy of ``W`` as a flat uint8 tensor on W's device.

    Rows past N are filled with nibble 0, which decodes to q = 0; their outputs
    are never stored.
    """
    assert W.dim() == 2 and W.dtype == torch.int8, (
        f"W must be int8 [N, K/2], got {W.dtype} {list(W.shape)}"
    )
    n, k_bytes = W.shape
    k = 2 * k_bytes
    layout = cached_layout(n, k)

    stored = W.contiguous().view(torch.uint8)
    low_nibbles = stored & 0xF
    high_nibbles = stored >> 4
    nibbles = torch.stack((low_nibbles, high_nibbles), dim=-1).reshape(n, k)
    nibbles = torch.nn.functional.pad(nibbles, (0, 0, 0, layout.n_padded - n))

    k_in_full_windows = layout.full_windows * K_PER_WINDOW
    row_blocks = [_pack_blocks(nibbles[:, :k_in_full_windows], dwords_per_lane=4)]
    if layout.has_half_window:
        row_blocks.append(
            _pack_blocks(nibbles[:, k_in_full_windows:], dwords_per_lane=2)
        )
    return torch.cat(row_blocks, dim=1).reshape(-1)


def _pack_blocks(nibbles: torch.Tensor, dwords_per_lane: int) -> torch.Tensor:
    """Pack ``[n_padded, windows * k_per_block]`` nibbles into blocks.

    Returns ``[row_blocks, windows * block_bytes]`` uint8, one row per row block.
    """
    n_padded, k_span = nibbles.shape
    k_per_block = dwords_per_lane * LANE_GROUPS * K_PER_DWORD
    row_blocks = n_padded // ROWS_PER_BLOCK
    windows = k_span // k_per_block

    # Split n and k into the coordinates named in the module docstring.
    by_coordinate = nibbles.view(
        row_blocks,
        ROWS_PER_BLOCK,  # row
        windows,
        dwords_per_lane,  # dword
        LANE_GROUPS,  # lane_group
        2,  # nibble: k + byte (low) or k + 4 + byte (high)
        4,  # byte
    )
    # Block order, then lane = 16 * lane_group + row, then the lane's bytes.
    in_block_order = by_coordinate.permute(0, 2, 4, 1, 3, 6, 5)
    low = in_block_order[..., 0]
    high = in_block_order[..., 1]
    packed = low | (high << 4)
    return packed.reshape(row_blocks, -1)
