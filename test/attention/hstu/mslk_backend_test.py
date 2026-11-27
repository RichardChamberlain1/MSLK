# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MSLK's backend of the HSTU ops (the pure-PyTorch reference) through the
registered ops and autograd: outputs, ``M`` and gradients must match the
reference called directly. Needs a GPU.
"""

import unittest
import warnings
from typing import Any, Dict

import mslk.attention.hstu as hstu
import torch
from hstu_test_utils import make_cross_inputs, make_grouped_inputs, make_ragged_inputs
from mslk.aiter.routing import force_backend, MSLK
from mslk.attention.dispatch import hstu_cross, hstu_grouped, hstu_ragged
from mslk.attention.hstu import _reference_backend, _torch_ref as ref

CUDA = torch.device("cuda")


def _close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a.float(), b.float(), atol=1e-5, rtol=1e-4)


def _leaves(inputs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(inputs)
    shared = inputs["v"] is inputs["k"]
    for name in ("q", "k") + (() if shared else ("v",)):
        out[name] = inputs[name].detach().clone().requires_grad_(True)
    if shared:
        out["v"] = out["k"]
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class MslkBackendTest(unittest.TestCase):
    def test_ragged(self) -> None:
        for softmax in (False, True):
            for shared_kv in (False, True):
                with self.subTest(softmax=softmax, shared_kv=shared_kv):
                    base = make_ragged_inputs(
                        4,
                        32,
                        2,
                        16,
                        16,
                        0.5,
                        dtype=torch.float32,
                        device=CUDA,
                        max_targets=4,
                        shared_kv=shared_kv,
                    )
                    options = dict(
                        invalid_attn_mask_type="lower_triangular",
                        num_targets=base["num_targets"],
                        attn_scale=None,
                        max_attn_len=8,
                        contextual_seq_len=0,
                        full_attn_size=0,
                        num_softmax_heads=2 if softmax else 0,
                        shared_kv=shared_kv,
                    )
                    a, b = _leaves(base), _leaves(base)
                    common = ("N", "alpha", "q", "k", "v", "seq_offsets")
                    with force_backend(hstu_ragged.OP, MSLK):
                        out, M = hstu.ragged_hstu_attention(
                            **{n: a[n] for n in common}, **options
                        )
                        dout = torch.randn_like(out)
                        out.backward(dout)
                    ref_out, ref_M = ref.ragged_hstu_attention_ref(
                        **{n: b[n] for n in common}, **options
                    )
                    ref_out.backward(dout)
                    _close(out, ref_out)
                    self.assertEqual(M.shape, (out.shape[0], 2))
                    if softmax:
                        _close(M, ref_M)
                    for name in ("q", "k") + (() if shared_kv else ("v",)):
                        _close(a[name].grad, b[name].grad)

    def test_cross(self) -> None:
        for shared_kv in (False, True):
            with self.subTest(shared_kv=shared_kv):
                base = make_cross_inputs(
                    4,
                    48,
                    16,
                    2,
                    1,
                    16,
                    16,
                    0.5,
                    dtype=torch.float32,
                    device=CUDA,
                    shared_kv=shared_kv,
                )
                base["attn_scale"] = torch.ones((), device=CUDA)
                a, b = _leaves(base), _leaves(base)
                names = (
                    "max_seq_len",
                    "alpha",
                    "q",
                    "k",
                    "v",
                    "seq_offsets",
                    "seq_offsets_q",
                    "max_q_len",
                    "attn_scale",
                    "G",
                    "shared_kv",
                )
                with force_backend(hstu_cross.OP, MSLK):
                    out, M = hstu.cross_hstu_attention(
                        **{n: a[n] for n in names}, num_softmax_heads=2, causal=True
                    )
                    dout = torch.randn_like(out)
                    out.backward(dout)
                ref_out, ref_M = ref.cross_hstu_attention_ref(
                    **{n: b[n] for n in names}, num_softmax_heads=2, causal=True
                )
                ref_out.backward(dout)
                _close(out, ref_out)
                finite = torch.isfinite(ref_M)
                _close(M[finite], ref_M[finite])
                for name in ("q", "k") + (() if shared_kv else ("v",)):
                    _close(a[name].grad, b[name].grad)

    def test_cross_bwd_takes_out(self) -> None:
        base = make_cross_inputs(
            2, 32, 8, 2, 1, 16, 16, 0.5, dtype=torch.float32, device=CUDA
        )
        attn_scale = torch.ones((), device=CUDA)
        with force_backend(hstu_cross.OP, MSLK):
            out, M = hstu.cross_hstu_attention_fwd(
                base["max_seq_len"],
                base["alpha"],
                base["q"],
                base["k"],
                base["v"],
                base["seq_offsets"],
                base["seq_offsets_q"],
                base["max_q_len"],
                attn_scale,
                1,
                num_softmax_heads=2,
            )
            dout = torch.randn_like(out)
            with_out = hstu.cross_hstu_attention_bwd(
                dout,
                base["q"],
                base["k"],
                base["v"],
                base["seq_offsets"],
                attn_scale,
                base["max_seq_len"],
                base["alpha"],
                base["max_q_len"],
                base["seq_offsets_q"],
                None,
                False,
                False,
                1,
                2,
                M,
                Out=out,
            )
            without = hstu.cross_hstu_attention_bwd(
                dout,
                base["q"],
                base["k"],
                base["v"],
                base["seq_offsets"],
                attn_scale,
                base["max_seq_len"],
                base["alpha"],
                None,  # max_q_len is optional in the backward
                base["seq_offsets_q"],
                None,
                False,
                False,
                1,
                2,
            )
        for x, y in zip(with_out, without):
            _close(x, y)

    def test_grouped(self) -> None:
        base = make_grouped_inputs(
            3, [24, 16], 2, 16, 16, 0.5, dtype=torch.float32, device=CUDA
        )
        total = base["q"].shape[0]
        per_token = torch.rand(total, 1, device=CUDA)
        for shared_kv in (False, True):
            for scale in (None, per_token):
                with self.subTest(shared_kv=shared_kv, per_token=scale is not None):
                    inputs = dict(base)
                    if shared_kv:
                        inputs["v"] = inputs["k"]
                    a, b = _leaves(inputs), _leaves(inputs)
                    rest = {
                        n: inputs[n]
                        for n in (
                            "seq_offsets",
                            "group_offsets",
                            "max_seq_len_tensor",
                            "max_attn_len_tensor",
                            "full_attn_size_tensor",
                            "contextual_seq_len_tensor",
                            "alpha",
                            "invalid_attn_mask_type",
                        )
                    }
                    with force_backend(hstu_grouped.OP, MSLK):
                        out = hstu.grouped_ragged_hstu_attention(
                            inputs["Z"],
                            a["q"],
                            a["k"],
                            a["v"],
                            group_num_targets=None,
                            attn_scale_tensor=scale,
                            **rest,
                        )
                        dout = torch.randn_like(out)
                        out.backward(dout)
                    ref_out = ref.grouped_ragged_hstu_attention_ref(
                        inputs["Z"],
                        b["q"],
                        b["k"],
                        b["v"],
                        num_targets=None,
                        attn_scale_tensor=None if scale is None else scale[:, 0],
                        **rest,
                    )
                    ref_out.backward(dout)
                    _close(out, ref_out)
                    for name in ("q", "k") + (() if shared_kv else ("v",)):
                        _close(a[name].grad, b[name].grad)

    def test_warns_once_on_large_inputs(self) -> None:
        q = torch.zeros(2, 1, 16, device=CUDA)
        _reference_backend._warned.discard("hstu_ragged_fwd")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                _reference_backend._warn_if_large(
                    "hstu_ragged_fwd", _reference_backend.LARGE_INPUT_TOKENS
                )
            _reference_backend._warn_if_large("hstu_ragged_fwd", q.shape[0])
        self.assertEqual(len(caught), 1)
        self.assertIn("pure-PyTorch HSTU reference", str(caught[0].message))


if __name__ == "__main__":
    unittest.main()
