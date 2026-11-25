# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Each routed quantization op serves its calls with the backend its tuning
table names.

End to end, on the GPU: for each op, the cheapest row of its table per backend
is called through the op, and the test checks which backend served it. Adding
a routed op means adding one entry to ``CASES``. The quantization counterpart
of ``test/gemm/dispatch_test.py``.
"""

import math
import unittest
from typing import Any, Callable, Dict, List, Tuple
from unittest import mock

import mslk.quantize  # noqa: F401
import torch
from mslk.aiter.common import is_aiter_enabled
from mslk.aiter.routing import device_key, OpRouting
from mslk.quantize.dispatch import quantize_fp8_tensor
from mslk.quantize.triton.fp8_quantize import (
    quantize_fp8_tensor as quantize_fp8_tensor_op,
)
from mslk.testing.device import skipUnlessRocm
from parameterized import parameterized

Args = Tuple[Tuple[Any, ...], Dict[str, Any]]


def _quantize_fp8_tensor_args(row: Dict[str, str]) -> Args:
    M, K = int(row["M_max"]), int(row["K"])
    a = torch.randn(M, K, device="cuda", dtype=getattr(torch, row["dtype"]))
    return (a,), {}


# (name, routing, the op, arguments for a table row)
CASES: List[Tuple[str, OpRouting, Callable[..., Any], Callable[..., Args]]] = [
    (
        "quantize_fp8_tensor",
        quantize_fp8_tensor.ROUTING,
        quantize_fp8_tensor_op,
        _quantize_fp8_tensor_args,
    ),
]


@skipUnlessRocm()
class DispatchTest(unittest.TestCase):
    @parameterized.expand(CASES, skip_on_empty=True)
    def test_follows_its_tuning_table(
        self,
        _name: str,
        routing: OpRouting,
        op: Callable[..., Any],
        make_args: Callable[..., Args],
    ) -> None:
        # Off when aiter is not installed, or switched off for the op (e.g.
        # MSLK_DISABLE_AITER=1); MSLK's kernel then serves every call.
        if not is_aiter_enabled(routing.op):
            self.skipTest("requires the pinned aiter, enabled for the op")
        gfx, cu_num = device_key(torch.cuda.current_device())
        table = routing.table
        columns = table.key_columns + (table.bucket_column,)
        rows = [r for r in table.rows if (r["gfx"], r["cu_num"]) == (gfx, str(cu_num))]
        # The cheapest shape for each backend is enough to show the routing.
        rows.sort(key=lambda r: math.prod(int(r[c]) for c in columns if r[c].isdigit()))
        cheapest = {}
        for row in rows:
            cheapest.setdefault(row["backend"], row)
        if not cheapest:
            self.skipTest(f"no tuning-table rows for {gfx} with {cu_num} CUs")
        for expected, row in cheapest.items():
            args, kwargs = make_args(row)
            spies = {b: mock.Mock(wraps=fn) for b, fn in routing.backends.items()}
            with mock.patch.dict(routing.backends, spies):
                op(*args, **kwargs)
            for backend, spy in spies.items():
                self.assertEqual(spy.call_count, int(backend == expected), row)


if __name__ == "__main__":
    unittest.main()
