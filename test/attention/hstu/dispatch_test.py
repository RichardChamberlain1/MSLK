# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Each routed HSTU op serves its calls with the backend its tuning table names
(the cheapest row per backend, forward and backward), and
``MSLK_DISABLE_AITER_OPS`` sends them back to MSLK.
"""

import math
import os
import unittest
from typing import Any, Callable, Dict, List, Tuple
from unittest import mock

import mslk.attention.hstu  # noqa: F401  (registers the HSTU ops)
import torch
from mslk.aiter import common
from mslk.aiter.common import is_aiter_available
from mslk.aiter.routing import device_key, MSLK
from mslk.attention.dispatch import hstu_cross, hstu_ragged
from mslk.testing.device import skipUnlessRocm

Row = Dict[str, str]


def _offsets(lengths: List[int]) -> torch.Tensor:
    return torch.tensor([0] + lengths, device="cuda").cumsum(0)


def _ragged_call(row: Row) -> None:
    H, D, DV, N = (int(row[c]) for c in ("H", "D", "DV", "max_seq_len"))
    shared_kv = row["shared_kv"] == "1"
    lengths = [N, max(1, N // 2)]
    total = sum(lengths)
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    q, k = torch.randn(total, H, D, **bf16), torch.randn(total, H, D, **bf16)
    v = k if shared_kv else torch.randn(total, H, DV, **bf16)
    seq_offsets = _offsets(lengths)
    window = int(row["max_attn_len"])
    softmax_heads = H if row["softmax"] == "1" else 0
    out, M = torch.ops.mslk.hstu_ragged_fwd(
        N, D**-0.5, q, k, v, seq_offsets, row["mask"], None, None, window, 0, 0,
        softmax_heads, shared_kv,
    )  # fmt: skip
    dq, dk = torch.empty_like(q), torch.empty_like(k)
    dv = None if shared_kv else torch.empty_like(v)
    torch.ops.mslk.hstu_ragged_bwd(
        torch.randn_like(out), q, k, v, dq, dk, dv, seq_offsets, None, None, N,
        D**-0.5, window, row["mask"], 0, None, 0, softmax_heads, M, out, shared_kv,
    )  # fmt: skip


def _cross_call(row: Row) -> None:
    H, D, DV = (int(row[c]) for c in ("H", "D", "DV"))
    max_kv, max_q = int(row["max_kv_len"]), int(row["max_q_len_p2"])
    shared_kv = row["shared_kv"] == "1"
    causal = row["causal"] == "1"
    lengths_kv, lengths_q = [max_kv, max(1, max_kv // 2)], [max_q, max(1, max_q // 2)]
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    q = torch.randn(sum(lengths_q), H, D, **bf16)
    k = torch.randn(sum(lengths_kv), H, D, **bf16)
    v = None if shared_kv else torch.randn(sum(lengths_kv), H, DV, **bf16)
    softmax_heads = H if row["softmax"] == "1" else 0
    so_kv, so_q = _offsets(lengths_kv), _offsets(lengths_q)
    scale = torch.ones((), device="cuda")
    out, M = torch.ops.mslk.hstu_cross_fwd(
        max_kv, D**-0.5, q, k, v, so_kv, so_q, max_q, scale, 1, shared_kv,
        softmax_heads, True, "none", None, causal,
    )  # fmt: skip
    torch.ops.mslk.hstu_cross_bwd(
        torch.randn_like(out), q, k, v, so_kv, scale, max_kv, D**-0.5, max_q, so_q,
        None, causal, shared_kv, 1, softmax_heads, M, None, M.stride(0), "none",
        True, out,
    )  # fmt: skip


# (op module, forward + backward call for a table row)
CASES: List[Tuple[Any, Callable[[Row], None]]] = [
    (hstu_ragged, _ragged_call),
    (hstu_cross, _cross_call),
]


def _cheapest_rows(module: Any) -> Dict[str, Row]:
    """The cheapest row of the op's table per backend, on this device."""
    gfx, cu_num = device_key(torch.cuda.current_device())
    table = module.FWD_ROUTING.table
    columns = table.key_columns + (table.bucket_column,)
    rows = [r for r in table.rows if (r["gfx"], r["cu_num"]) == (gfx, str(cu_num))]
    rows.sort(key=lambda r: math.prod(int(r[c]) for c in columns if r[c].isdigit()))
    cheapest: Dict[str, Row] = {}
    for row in rows:
        cheapest.setdefault(row["backend"], row)
    return cheapest


@skipUnlessRocm()
class HstuDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        if not is_aiter_available():
            self.skipTest("requires the pinned aiter")

    def _spied(self, module: Any, call: Callable[[Row], None], row: Row) -> Dict:
        spies = {}
        patches = []
        for direction in ("FWD", "BWD"):
            routing = getattr(module, f"{direction}_ROUTING")
            spies[direction] = {
                b: mock.Mock(wraps=fn) for b, fn in routing.backends.items()
            }
            patches.append(mock.patch.dict(routing.backends, spies[direction]))
        for p in patches:
            p.start()
        try:
            call(row)
        finally:
            for p in patches:
                p.stop()
        return spies

    def test_follows_its_tuning_table(self) -> None:
        for module, call in CASES:
            cheapest = _cheapest_rows(module)
            if not cheapest:
                continue
            for expected, row in cheapest.items():
                with self.subTest(op=module.OP, backend=expected):
                    spies = self._spied(module, call, row)
                    for direction, by_backend in spies.items():
                        for backend, spy in by_backend.items():
                            self.assertEqual(
                                spy.call_count,
                                int(backend == expected),
                                f"{direction} {backend} {row}",
                            )

    def test_uncovered_calls_use_the_default_backend(self) -> None:
        # Keys no table lists: window 7 (ragged), 2 heads (cross).
        ragged = dict(
            H="1",
            D="64",
            DV="64",
            mask="lower_triangular",
            max_attn_len="7",
            softmax="0",
            shared_kv="0",
            max_seq_len="256",
        )
        cross = dict(H="2", D="64", DV="64", causal="0", softmax="1", shared_kv="0",
                     max_q_len_p2="64", max_kv_len="256")  # fmt: skip
        for module, call, row in (
            (hstu_ragged, _ragged_call, ragged),
            (hstu_cross, _cross_call, cross),
        ):
            with self.subTest(op=module.OP):
                expected = getattr(module, "UNTUNED_DEFAULT", None)
                spies = self._spied(module, call, row)
                for direction, by_backend in spies.items():
                    self.assertEqual(
                        by_backend[expected or module.FWD_ROUTING.default].call_count,
                        1,
                        direction,
                    )

    def test_untuned_head_counts_take_the_nearest_tuned_ones_choice(self) -> None:
        """A ragged call with a head count the table does not list takes the
        backend of the same shape at the nearest tuned head count."""
        gfx, cu_num = device_key(torch.cuda.current_device())
        table = hstu_ragged.FWD_ROUTING.table
        rows = [r for r in table.rows if (r["gfx"], r["cu_num"]) == (gfx, str(cu_num))]
        h1 = [r for r in rows if r["H"] == "1" and r["backend"] != MSLK]
        if not h1:
            self.skipTest(f"no tuning-table rows for {gfx} with {cu_num} CUs")
        # The cheapest H=1 row of each backend, called with H=2 (nearest: H=1).
        for backend in sorted({r["backend"] for r in h1}):
            row = min(
                (r for r in h1 if r["backend"] == backend),
                key=lambda r: int(r["max_seq_len"]) * int(r["D"]),
            )
            with self.subTest(backend=backend):
                spies = self._spied(hstu_ragged, _ragged_call, {**row, "H": "2"})
                for direction, by_backend in spies.items():
                    self.assertEqual(
                        by_backend[backend].call_count, 1, f"{direction} {row}"
                    )

    def test_disable_switch_sends_the_op_back_to_mslk(self) -> None:
        for module, call in CASES:
            rows = {b: r for b, r in _cheapest_rows(module).items() if b != MSLK}
            for backend, row in rows.items():
                with self.subTest(op=module.OP, backend=backend):
                    try:
                        with mock.patch.dict(
                            os.environ, {common.DISABLE_OPS_ENV: module.OP}
                        ):
                            common.reset()
                            spies = self._spied(module, call, row)
                    finally:
                        common.reset()
                    for direction, by_backend in spies.items():
                        self.assertEqual(by_backend[MSLK].call_count, 1, direction)
                        self.assertEqual(by_backend[backend].call_count, 0, direction)


if __name__ == "__main__":
    unittest.main()
