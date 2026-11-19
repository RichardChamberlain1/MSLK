# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from mslk.gemm.flydsl.bf16i4bf16_layout import cached_layout, reorder_int4_for_mfma
from parameterized import parameterized


def cached_position(n: int, k: int, n_rows: int, k_total: int) -> tuple[int, int]:
    """Byte offset and nibble (0 low, 1 high) of weight (n, k) in the copy."""
    layout = cached_layout(n_rows, k_total)
    row_block, row = divmod(n, 16)
    window, k_in_window = divmod(k, 128)
    in_half_window = window == layout.full_windows
    bytes_per_lane = 8 if in_half_window else 16
    dword, k_in_dword_run = divmod(k_in_window, 32)
    lane_group, k_in_lane_run = divmod(k_in_dword_run, 8)
    nibble, byte = divmod(k_in_lane_run, 4)
    lane = 16 * lane_group + row
    offset = (
        row_block * layout.row_block_bytes
        + window * 1024
        + bytes_per_lane * lane
        + 4 * dword
        + byte
    )
    return offset, nibble


def stored_nibble(W: torch.Tensor, n: int, k: int) -> int:
    byte = int(W[n, k // 2]) & 0xFF
    return (byte >> 4) if k % 2 else (byte & 0xF)


class BF16Int4LayoutTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("one_block", 16, 128),
            ("half_window_only", 16, 64),
            ("full_and_half_windows", 32, 320),
            ("n_tail", 24, 256),
            ("n_tail_and_half_window", 17, 192),
            ("several_row_blocks", 48, 512),
        ]
    )
    def test_matches_cached_position(self, _name: str, n: int, k: int) -> None:
        torch.manual_seed(0)
        W = torch.randint(-128, 128, (n, k // 2), dtype=torch.int8)
        layout = cached_layout(n, k)

        reordered = reorder_int4_for_mfma(W)

        self.assertEqual(reordered.dtype, torch.uint8)
        self.assertEqual(reordered.numel(), layout.n_padded * k // 2)
        nibble_written = torch.zeros(reordered.numel(), 2, dtype=torch.bool)
        for row in range(layout.n_padded):
            for col in range(k):
                offset, nibble = cached_position(row, col, n, k)
                byte = int(reordered[offset])
                got = (byte >> 4) if nibble else (byte & 0xF)
                want = stored_nibble(W, row, col) if row < n else 0
                self.assertEqual(got, want, f"weight (n={row}, k={col})")
                self.assertFalse(nibble_written[offset, nibble], "slot reused")
                nibble_written[offset, nibble] = True
        self.assertTrue(bool(nibble_written.all()), "copy has unwritten nibbles")

    def test_lane_dword_holds_consecutive_k(self) -> None:
        # Lane 37 of a full block serves row 5 and lane group 2: its first dword
        # holds k 16..23, byte t with k = 16 + t low and 20 + t high (design §3.3).
        n, k = 16, 128
        W = torch.zeros((n, k // 2), dtype=torch.int8)
        row = 37 % 16
        for col in range(16, 24):
            nibble = col - 16 + 1  # distinct non-zero values 1..8
            byte_index = col // 2
            shift = 4 * (col % 2)
            current = int(W[row, byte_index]) & 0xFF
            W[row, byte_index] = (
                torch.tensor(current | (nibble << shift))
                .to(torch.uint8)
                .view(torch.int8)
            )

        lane_bytes = reorder_int4_for_mfma(W)[16 * 37 : 16 * 37 + 4].tolist()

        self.assertEqual([b & 0xF for b in lane_bytes], [1, 2, 3, 4])
        self.assertEqual([b >> 4 for b in lane_bytes], [5, 6, 7, 8])

    def test_rejects_k_not_multiple_of_64(self) -> None:
        with self.assertRaises(AssertionError):
            reorder_int4_for_mfma(torch.zeros((16, 48), dtype=torch.int8))


if __name__ == "__main__":
    unittest.main()
