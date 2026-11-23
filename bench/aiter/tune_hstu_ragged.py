# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tune which backend serves ``mslk::hstu_ragged_fwd``/``_bwd`` on ROCm, per shape.

Compares MSLK's backend with aiter's Triton and FlyDSL HSTU kernels on the
shapes in ``bench/aiter/untuned/hstu_ragged.csv`` and updates
``mslk/attention/dispatch/tables/hstu_ragged.csv`` (see :mod:`mslk.aiter.tuning`).
Needs the pinned aiter and a gfx950 GPU.

A measured call is forward + backward, since one table row routes both. MSLK's
backend (the pure-PyTorch reference) is checked but not timed. Shapes are synthetic:
about ``TOKENS`` tokens in sequences of length ``[N / 2, N]``, SiLU heads.
Each output is divided by the reference's largest magnitude before comparing
with the public TLX test tolerance (8e-3).

    python bench/aiter/tune_hstu_ragged.py
"""

from pathlib import Path
from typing import Any, Dict, List

import mslk.attention.hstu  # noqa: F401  (registers the HSTU ops)
import torch
from mslk.aiter import tuning
from mslk.aiter.routing import MSLK
from mslk.aiter.tuning import OpTuner, Row
from mslk.attention.dispatch.hstu_ragged import BACKENDS, BUCKET_COLUMN, KEY_COLUMNS, OP
from mslk.attention.hstu import _torch_ref

REPO = Path(__file__).resolve().parents[2]

#: Packed tokens per tuned shape (rounded to whole sequences).
TOKENS: int = 16384


def _scales(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Each reference tensor's largest magnitude (1 if all zeros)."""
    s = torch.stack([t.detach().float().abs().amax() for t in tensors])
    return torch.where(s > 0, s, torch.ones_like(s))


def _joined(tensors: List[torch.Tensor], scales: torch.Tensor) -> torch.Tensor:
    """Tensors divided by the reference scales, flattened and joined."""
    return torch.cat([(t.float() / scales[i]).flatten() for i, t in enumerate(tensors)])


class HstuRaggedTuner(OpTuner):
    op = OP
    key_columns = KEY_COLUMNS
    bucket_column = BUCKET_COLUMN
    backends = BACKENDS
    atol = rtol = 8e-3
    outputs = ("dq", "dk", "dv")
    untimed_backends = (MSLK,)

    def make_inputs(self, shape: Row) -> Dict[str, Any]:
        H, D, DV = int(shape["H"]), int(shape["D"]), int(shape["DV"])
        N = int(shape["max_seq_len"])
        shared_kv = shape["shared_kv"] == "1"
        if shared_kv and D != DV:
            raise ValueError("shared K/V needs D == DV")
        batch = max(1, TOKENS // N)
        lengths = torch.randint(max(1, N // 2), N + 1, (batch,))
        offsets = torch.zeros(batch + 1, dtype=torch.int64)
        offsets[1:] = torch.cumsum(lengths, 0)
        total = int(offsets[-1])
        bf16 = dict(device="cuda", dtype=torch.bfloat16)
        inputs: Dict[str, Any] = {
            "N": N,
            "alpha": D**-0.5,
            "mask": shape["mask"],
            "max_attn_len": int(shape["max_attn_len"]),
            "num_softmax_heads": H if shape["softmax"] == "1" else 0,
            "shared_kv": shared_kv,
            "q": torch.randn(total, H, D, **bf16),
            "k": torch.randn(total, H, D, **bf16),
            "seq_offsets": offsets.to("cuda"),
            "dout": torch.randn(total, H, DV, **bf16),
            "dq": torch.empty(total, H, D, **bf16),
            "dk": torch.empty(total, H, D, **bf16),
            # Unused with shared K/V; present for the NaN fill.
            "dv": torch.empty(total, H, DV, **bf16),
            # Set by reference().
            "scales": torch.ones(4, device="cuda"),
        }
        if not shared_kv:
            # With shared K/V, V is K (not stored: copies would split them).
            inputs["v"] = torch.randn(total, H, DV, **bf16)
        return inputs

    @staticmethod
    def _value(inputs: Dict[str, Any]) -> torch.Tensor:
        return inputs["k"] if inputs["shared_kv"] else inputs["v"]

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        q, k, v = inputs["q"], inputs["k"], self._value(inputs)
        shared_kv = inputs["shared_kv"]
        out, M = torch.ops.mslk.hstu_ragged_fwd(
            inputs["N"],
            inputs["alpha"],
            q,
            k,
            v,
            inputs["seq_offsets"],
            inputs["mask"],
            None,
            None,
            inputs["max_attn_len"],
            0,
            0,
            inputs["num_softmax_heads"],
            shared_kv,
        )
        dv = None if shared_kv else inputs["dv"]
        torch.ops.mslk.hstu_ragged_bwd(
            inputs["dout"],
            q,
            k,
            v,
            inputs["dq"],
            inputs["dk"],
            dv,
            inputs["seq_offsets"],
            None,
            None,
            inputs["N"],
            inputs["alpha"],
            inputs["max_attn_len"],
            inputs["mask"],
            0,
            None,
            0,
            inputs["num_softmax_heads"],
            M,
            out,
            shared_kv,
        )
        grads = [inputs["dq"], inputs["dk"]] + ([] if shared_kv else [inputs["dv"]])
        return _joined([out] + grads, inputs["scales"])

    def reference(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        shared_kv = inputs["shared_kv"]
        q = inputs["q"].detach().float().requires_grad_(True)
        k = inputs["k"].detach().float().requires_grad_(True)
        v = k if shared_kv else inputs["v"].detach().float().requires_grad_(True)
        out, _ = _torch_ref.ragged_hstu_attention_ref(
            inputs["N"],
            inputs["alpha"],
            q,
            k,
            v,
            inputs["seq_offsets"],
            inputs["mask"],
            max_attn_len=inputs["max_attn_len"],
            num_softmax_heads=inputs["num_softmax_heads"],
            shared_kv=shared_kv,
        )
        leaves = [q, k] if shared_kv else [q, k, v]
        grads = list(torch.autograd.grad(out, leaves, inputs["dout"].float()))
        tensors = [out.detach()] + grads
        # reference() runs before the tuner copies the inputs.
        inputs["scales"][: len(tensors)].copy_(_scales(tensors))
        return _joined(tensors, inputs["scales"])


if __name__ == "__main__":
    tuning.main(
        HstuRaggedTuner(),
        untune_file=str(REPO / "bench" / "aiter" / "untuned" / "hstu_ragged.csv"),
        tune_file=str(
            REPO / "mslk" / "attention" / "dispatch" / "tables" / "hstu_ragged.csv"
        ),
    )
