# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""aiter backends of the HSTU ops against the pure-PyTorch reference: forced
backends must take the call and match the output and every gradient; declined
calls must go to MSLK. Needs ROCm, gfx942/gfx950 and the pinned aiter.
``force_backend`` is process-wide: do not run these tests in parallel threads.
"""

import unittest
import unittest.mock
from typing import Any, Dict, Optional

import mslk.attention.hstu as hstu
import torch
from hstu_test_utils import assert_hstu_close, make_cross_inputs, make_ragged_inputs
from mslk.aiter.common import is_aiter_available
from mslk.aiter.routing import device_key, force_backend, MSLK
from mslk.attention.dispatch import hstu_cross, hstu_ragged
from mslk.attention.hstu import _torch_ref as ref


def _aiter_gpu() -> bool:
    return (
        torch.version.hip is not None
        and torch.cuda.is_available()
        and device_key(torch.cuda.current_device())[0] in ("gfx942", "gfx950")
        and is_aiter_available()
    )


def _leaves(
    inputs: Dict[str, Any], names: Any, dtype: Optional[torch.dtype] = None
) -> Dict[str, Any]:
    """Fresh leaf copies (V stays K when shared), optionally cast."""
    out = dict(inputs)
    shared = inputs["v"] is inputs["k"]
    for name in names:
        if name == "v" and shared:
            continue
        t = inputs[name].detach().clone()
        out[name] = (t if dtype is None else t.to(dtype)).requires_grad_(True)
    if shared:
        out["v"] = out["k"]
    return out


def _assert_flash_close(
    out: torch.Tensor, ref: torch.Tensor, name: str, gradient: bool = False
) -> None:
    """Flash kernels round probabilities to bf16, so they need more than TLX's
    atol 8e-4 (aiter reaches 1.4e-3); gradients scale atol by their magnitude."""
    atol = 4e-3
    if gradient:
        atol *= max(1.0, ref.abs().max().item())
    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=atol,
        rtol=1.6e-2,
        msg=lambda formatted: f"{name} (max |ref| {ref.abs().max().item():.3g}): "
        + formatted,
    )


def _grads(inputs: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
    shared = inputs["v"] is inputs["k"]
    grads = {"q": inputs["q"].grad, "k": inputs["k"].grad}
    if not shared:
        grads["v"] = inputs["v"].grad
    return grads


@unittest.skipUnless(_aiter_gpu(), "requires ROCm, gfx942/gfx950 and the pinned aiter")
class RaggedAiterTest(unittest.TestCase):
    BACKENDS = (hstu_ragged.AITER_TRITON, hstu_ragged.AITER_FLYDSL)

    def _call_args(self, inputs: Dict[str, Any], **options: Any) -> Dict[str, Any]:
        return dict(
            N=inputs["N"],
            alpha=inputs["alpha"],
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            seq_offsets=inputs["seq_offsets"],
            invalid_attn_mask_type=options.get(
                "invalid_attn_mask_type", "lower_triangular"
            ),
            num_targets=inputs.get("num_targets"),
            attn_scale=options.get("attn_scale"),
            max_attn_len=options.get("max_attn_len", 0),
            contextual_seq_len=options.get("contextual_seq_len", 0),
            full_attn_size=options.get("full_attn_size", 0),
            num_softmax_heads=options.get("num_softmax_heads", 0),
            shared_kv=inputs["shared_kv"],
        )

    def _routed(self, backend: str, args: Dict[str, Any]) -> str:
        args = dict(args)
        q = args.pop("q")
        with force_backend(hstu_ragged.OP, backend):
            return hstu_ragged.FWD_ROUTING.backend_for(q, **args)

    def _check(self, backend: str, head_dim: int, **case: Any) -> None:
        shared_kv = case.pop("shared_kv", False)
        max_targets = case.pop("max_targets", 0)
        base = make_ragged_inputs(
            8,
            case.pop("max_seq_len", 256),
            2,
            head_dim,
            head_dim,
            0.5,
            device=torch.device("cuda"),
            seed=head_dim,
            max_targets=max_targets,
            shared_kv=shared_kv,
        )
        mine = _leaves(base, ("q", "k", "v"))
        theirs = _leaves(base, ("q", "k", "v"), torch.float32)
        args = self._call_args(mine, **case)
        self.assertEqual(self._routed(backend, args), backend, case)
        with force_backend(hstu_ragged.OP, backend):
            out, _ = hstu.ragged_hstu_attention(**args)
        dout = torch.randn_like(out)
        with force_backend(hstu_ragged.OP, backend):
            out.backward(dout)
        ref_args = self._call_args(theirs, **case)
        ref_out, _ = ref.ragged_hstu_attention_ref(**ref_args)
        ref_out.backward(dout.float())
        assert_hstu_close(out, ref_out, "silu")
        ref_grads = _grads(theirs)
        for name, grad in _grads(mine).items():
            assert_hstu_close(grad, ref_grads[name], "silu", f"d{name}")

    def test_matches_reference(self) -> None:
        cases = [
            {},
            {"max_attn_len": 16},
            {"max_targets": 8},
            {"contextual_seq_len": 4},
            {"max_attn_len": 32, "max_targets": 8},
            {"max_targets": 8, "contextual_seq_len": 4},
            {"shared_kv": True},
            {"shared_kv": True, "max_attn_len": 16, "max_targets": 8},
        ]
        for backend in self.BACKENDS:
            for head_dim in (64, 128, 256):
                for case in cases:
                    with self.subTest(backend=backend, head_dim=head_dim, **case):
                        self._check(backend, head_dim, **dict(case))

    def test_declines_what_aiter_cannot_compute(self) -> None:
        base = make_ragged_inputs(4, 64, 2, 64, 64, 0.5, device=torch.device("cuda"))
        declined = [
            {"num_softmax_heads": 2},
            {"attn_scale": torch.tensor(0.5, device="cuda")},
            {"max_attn_len": 8, "full_attn_size": 4},
            {"invalid_attn_mask_type": "upper_triangular"},
            # aiter windows the contextual prefix; the reference does not.
            {"max_attn_len": 8, "contextual_seq_len": 4},
        ]
        for backend in self.BACKENDS:
            for case in declined:
                with self.subTest(backend=backend, case=str(case)):
                    args = self._call_args(base, **case)
                    self.assertEqual(self._routed(backend, args), MSLK)
        fp32 = make_ragged_inputs(
            4, 64, 2, 64, 64, 0.5, dtype=torch.float32, device=torch.device("cuda")
        )
        for backend in self.BACKENDS:
            self.assertEqual(self._routed(backend, self._call_args(fp32)), MSLK)
        odd = make_ragged_inputs(4, 64, 2, 96, 96, 0.5, device=torch.device("cuda"))
        self.assertEqual(
            self._routed(hstu_ragged.AITER_TRITON, self._call_args(odd)), MSLK
        )

    def test_empty_call_goes_to_mslk(self) -> None:
        """N == 0 is never tuned; aiter's kernels reject it, so the call
        goes to MSLK whatever the default backend is."""
        q = torch.empty(0, 2, 64, dtype=torch.bfloat16, device="cuda")
        offsets = torch.zeros(2, dtype=torch.int64, device="cuda")
        inputs = {
            "N": 0,
            "alpha": 1.0,
            "q": q,
            "k": q,
            "v": q,
            "seq_offsets": offsets,
            "shared_kv": False,
        }
        args = self._call_args(inputs)
        args.pop("q")
        self.assertEqual(hstu_ragged.FWD_ROUTING.backend_for(q, **args), MSLK)
        out, _ = hstu.ragged_hstu_attention(**self._call_args(inputs))
        self.assertEqual(out.shape, (0, 2, 64))


class FlydslOffsetsDtypeTest(unittest.TestCase):
    """aiter's FlyDSL kernel cache ignores the offsets dtype: always pass int32."""

    def test_offsets_and_targets_reach_aiter_as_int32(self) -> None:
        from mslk.attention.aiter import hstu as aiter_hstu

        calls = []

        class _Fake:
            def flydsl_hstu_attention_fwd(self, *args: Any) -> torch.Tensor:
                calls.append(args)
                return args[4]

            def flydsl_hstu_attention_bwd(self, *args: Any) -> Any:
                calls.append(args)
                return args[2], args[3], args[4]

        q = torch.zeros(4, 1, 16)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        targets = torch.tensor([1], dtype=torch.int64)
        common = dict(
            N=4, alpha=1.0, k=q, v=q, seq_offsets=offsets, num_targets=targets,
            max_attn_len=0, contextual_seq_len=0,
        )  # fmt: skip
        with unittest.mock.patch.object(
            aiter_hstu, "_flydsl_hstu", return_value=_Fake()
        ):
            aiter_hstu.ragged_fwd_flydsl(q, **common)
            aiter_hstu.ragged_bwd_flydsl(
                q, dout=q, dq=torch.empty_like(q), dk=torch.empty_like(q),
                dv=torch.empty_like(q), **common,
            )  # fmt: skip
        fwd, bwd = calls
        for t in (fwd[5], fwd[7], bwd[6], bwd[8]):
            self.assertEqual(t.dtype, torch.int32)


@unittest.skipUnless(_aiter_gpu(), "requires ROCm, gfx942/gfx950 and the pinned aiter")
class CrossAiterTest(unittest.TestCase):
    def _call_args(self, inputs: Dict[str, Any], causal: bool) -> Dict[str, Any]:
        q = inputs["q"]
        return dict(
            max_seq_len=inputs["max_seq_len"],
            alpha=inputs["alpha"],
            q=q,
            k=inputs["k"],
            v=inputs["v"],
            seq_offsets=inputs["seq_offsets"],
            seq_offsets_q=inputs["seq_offsets_q"],
            max_q_len=inputs["max_q_len"],
            attn_scale=torch.ones((), device=q.device, dtype=torch.float32),
            G=inputs["G"],
            shared_kv=inputs["shared_kv"],
            num_softmax_heads=q.shape[1],
            truncate_method="none",
            num_targets=None,
            causal=causal,
        )

    def _routed(self, args: Dict[str, Any]) -> str:
        args = dict(args)
        q = args.pop("q")
        with force_backend(hstu_cross.OP, hstu_cross.AITER):
            return hstu_cross.FWD_ROUTING.backend_for(q, **args)

    def _check(
        self, head_dim: int, causal: bool, shared_kv: bool, **sizes: Any
    ) -> None:
        base = make_cross_inputs(
            sizes.get("batch", 8),
            sizes.get("max_kv_len", 512),
            sizes.get("max_q_len", 64),
            2,
            1,
            head_dim,
            head_dim,
            0.5,
            device=torch.device("cuda"),
            seed=head_dim,
            shared_kv=shared_kv,
        )
        mine = _leaves(base, ("q", "k", "v"))
        theirs = _leaves(base, ("q", "k", "v"), torch.float32)
        args = self._call_args(mine, causal)
        self.assertEqual(self._routed(args), hstu_cross.AITER)
        with force_backend(hstu_cross.OP, hstu_cross.AITER):
            out, M = hstu.cross_hstu_attention(**args)
            dout = torch.randn_like(out)
            out.backward(dout)
        ref_out, ref_M = ref.cross_hstu_attention_ref(**self._call_args(theirs, causal))
        ref_out.backward(dout.float())
        _assert_flash_close(out, ref_out, "out")
        finite = torch.isfinite(ref_M)
        torch.testing.assert_close(M[finite], ref_M[finite], atol=1e-2, rtol=1e-3)
        ref_grads = _grads(theirs)
        for name, grad in _grads(mine).items():
            _assert_flash_close(grad, ref_grads[name], f"d{name}", gradient=True)

    def test_matches_reference(self) -> None:
        for head_dim in (64, 128, 256):
            for causal in (False, True) if head_dim <= 128 else (False,):
                for shared_kv in (False, True):
                    with self.subTest(
                        head_dim=head_dim, causal=causal, shared_kv=shared_kv
                    ):
                        self._check(head_dim, causal, shared_kv)

    def test_long_queries(self) -> None:
        # Queries longer than their KV.
        for causal in (False, True):
            with self.subTest(causal=causal):
                self._check(128, causal, False, max_kv_len=64, max_q_len=256)

    def test_declines_what_aiter_cannot_compute(self) -> None:
        base = make_cross_inputs(
            4, 128, 32, 2, 1, 64, 64, 0.5, device=torch.device("cuda")
        )
        silu = self._call_args(base, False)
        silu["num_softmax_heads"] = 0
        truncated = self._call_args(base, False)
        truncated["truncate_method"] = "keep_last"
        targets = self._call_args(base, True)
        targets["num_targets"] = torch.ones(4, dtype=torch.int64, device="cuda")
        wide = make_cross_inputs(
            4, 128, 32, 2, 1, 256, 256, 0.5, device=torch.device("cuda")
        )
        for name, args in (
            ("silu", silu),
            ("truncated", truncated),
            ("causal targets", targets),
            ("causal D=256", self._call_args(wide, True)),
        ):
            with self.subTest(name):
                self.assertEqual(self._routed(args), MSLK)

    def test_backward_without_out_uses_mslk(self) -> None:
        base = make_cross_inputs(
            4, 128, 32, 2, 1, 64, 64, 0.5, device=torch.device("cuda")
        )
        args = self._call_args(base, False)
        bwd = dict(
            dout=torch.empty_like(base["q"]),
            k=args["k"],
            v=args["v"],
            seq_offsets=args["seq_offsets"],
            attn_scale=args["attn_scale"],
            max_seq_len=args["max_seq_len"],
            alpha=args["alpha"],
            max_q_len=args["max_q_len"],
            seq_offsets_q=args["seq_offsets_q"],
            num_targets=None,
            causal=False,
            shared_kv=False,
            G=1,
            num_softmax_heads=args["num_softmax_heads"],
            M=torch.empty(base["q"].shape[:2], device="cuda"),
            Delta=None,
            stride_mm=0,
            truncate_method="none",
            enable_tma=True,
            Out=None,
        )
        with force_backend(hstu_cross.OP, hstu_cross.AITER):
            self.assertEqual(hstu_cross.BWD_ROUTING.backend_for(base["q"], **bwd), MSLK)
            bwd["Out"] = torch.empty_like(base["q"])
            self.assertEqual(
                hstu_cross.BWD_ROUTING.backend_for(base["q"], **bwd), hstu_cross.AITER
            )


if __name__ == "__main__":
    unittest.main()
