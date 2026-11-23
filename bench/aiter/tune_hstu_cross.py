# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tune which backend serves ``mslk::hstu_cross_fwd``/``_bwd`` on ROCm, per shape.

Compares MSLK's backend with aiter's varlen flash attention on the shapes in
``bench/aiter/untuned/hstu_cross.csv`` and updates
``mslk/attention/dispatch/tables/hstu_cross.csv`` (see :mod:`mslk.aiter.tuning`).
Needs the pinned aiter and a gfx950 GPU.

A measured call is forward + backward, since one table row routes both. MSLK's
backend (the pure-PyTorch reference) is checked but not timed. Shapes are synthetic:
up to ``BATCH`` sequences, lengths in ``[max / 2, max]``, softmax heads. Gradients
are divided by the reference's largest magnitude; the tolerance (atol 4e-3,
rtol 1.6e-2) allows for flash kernels rounding probabilities to bf16.

    python bench/aiter/tune_hstu_cross.py
"""

from pathlib import Path
from typing import Any, Dict, List

import mslk.attention.hstu  # noqa: F401  (registers the HSTU ops)
import torch
from mslk.aiter import tuning
from mslk.aiter.routing import MSLK
from mslk.aiter.tuning import OpTuner, Row
from mslk.attention.dispatch.hstu_cross import BACKENDS, BUCKET_COLUMN, KEY_COLUMNS, OP
from mslk.attention.hstu import _torch_ref

REPO = Path(__file__).resolve().parents[2]

#: Sequences per tuned shape, fewer for the largest so the reference fits.
BATCH: int = 16
#: Score elements (Lq x Lkv x H) per shape that the batch is capped to.
MAX_SCORES: int = 1 << 31


def _offsets(lengths: torch.Tensor) -> torch.Tensor:
    offsets = torch.zeros(lengths.numel() + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, 0)
    return offsets


def _scales(out: torch.Tensor, grads: List[torch.Tensor]) -> torch.Tensor:
    """1 for the output; each reference gradient's largest magnitude (1 if all
    zeros, e.g. dQ and dK with one key per sequence)."""
    maxima = [g.detach().float().abs().amax() for g in grads]
    scales = [torch.ones_like(maxima[0])] + maxima
    s = torch.stack(scales)
    return torch.where(s > 0, s, torch.ones_like(s))


def _joined(tensors: List[torch.Tensor], scales: torch.Tensor) -> torch.Tensor:
    """Tensors divided by the reference scales, flattened and joined."""
    return torch.cat([(t.float() / scales[i]).flatten() for i, t in enumerate(tensors)])


class HstuCrossTuner(OpTuner):
    op = OP
    key_columns = KEY_COLUMNS
    bucket_column = BUCKET_COLUMN
    backends = BACKENDS
    atol, rtol = 4e-3, 1.6e-2
    untimed_backends = (MSLK,)

    def make_inputs(self, shape: Row) -> Dict[str, Any]:
        H, D, DV = int(shape["H"]), int(shape["D"]), int(shape["DV"])
        max_kv = int(shape["max_kv_len"])
        max_q = int(shape["max_q_len_p2"])
        shared_kv = shape["shared_kv"] == "1"
        if shared_kv and D != DV:
            raise ValueError("shared K/V needs D == DV")
        batch = max(1, min(BATCH, MAX_SCORES // (max_q * max_kv * H)))
        lengths_kv = torch.randint(max(1, max_kv // 2), max_kv + 1, (batch,))
        lengths_q = torch.randint(max(1, max_q // 2), max_q + 1, (batch,))
        offsets_kv, offsets_q = _offsets(lengths_kv), _offsets(lengths_q)
        total_kv, total_q = int(offsets_kv[-1]), int(offsets_q[-1])
        bf16 = dict(device="cuda", dtype=torch.bfloat16)
        inputs: Dict[str, Any] = {
            "max_seq_len": max_kv,
            "max_q_len": max_q,
            "alpha": D**-0.5,
            "causal": shape["causal"] == "1",
            "num_softmax_heads": H if shape["softmax"] == "1" else 0,
            "shared_kv": shared_kv,
            "q": torch.randn(total_q, H, D, **bf16),
            "k": torch.randn(total_kv, H, D, **bf16),
            "seq_offsets": offsets_kv.to("cuda"),
            "seq_offsets_q": offsets_q.to("cuda"),
            "attn_scale": torch.ones((), device="cuda", dtype=torch.float32),
            "dout": torch.randn(total_q, H, DV, **bf16),
            # Set by reference().
            "scales": torch.ones(4, device="cuda"),
        }
        if not shared_kv:
            inputs["v"] = torch.randn(total_kv, H, DV, **bf16)
        return inputs

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        shared_kv = inputs["shared_kv"]
        q, k = inputs["q"], inputs["k"]
        v = None if shared_kv else inputs["v"]
        out, M = torch.ops.mslk.hstu_cross_fwd(
            inputs["max_seq_len"],
            inputs["alpha"],
            q,
            k,
            v,
            inputs["seq_offsets"],
            inputs["seq_offsets_q"],
            inputs["max_q_len"],
            inputs["attn_scale"],
            1,
            shared_kv,
            inputs["num_softmax_heads"],
            True,
            "none",
            None,
            inputs["causal"],
        )
        grads = torch.ops.mslk.hstu_cross_bwd(
            inputs["dout"],
            q,
            k,
            v,
            inputs["seq_offsets"],
            inputs["attn_scale"],
            inputs["max_seq_len"],
            inputs["alpha"],
            inputs["max_q_len"],
            inputs["seq_offsets_q"],
            None,
            inputs["causal"],
            shared_kv,
            1,
            inputs["num_softmax_heads"],
            M,
            None,
            M.stride(0),
            "none",
            True,
            out,
        )
        return _joined([out] + list(grads), inputs["scales"])

    def reference(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        shared_kv = inputs["shared_kv"]
        q = inputs["q"].detach().float().requires_grad_(True)
        k = inputs["k"].detach().float().requires_grad_(True)
        v = k if shared_kv else inputs["v"].detach().float().requires_grad_(True)
        out, _ = _torch_ref.cross_hstu_attention_ref(
            inputs["max_seq_len"],
            inputs["alpha"],
            q,
            k,
            v,
            inputs["seq_offsets"],
            inputs["seq_offsets_q"],
            inputs["max_q_len"],
            inputs["attn_scale"],
            1,
            shared_kv,
            inputs["num_softmax_heads"],
            causal=inputs["causal"],
        )
        leaves = [q, k] if shared_kv else [q, k, v]
        grads = list(torch.autograd.grad(out, leaves, inputs["dout"].float()))
        # reference() runs before the tuner copies the inputs.
        scales = _scales(out, grads)
        inputs["scales"][: scales.numel()].copy_(scales)
        return _joined([out.detach()] + grads, inputs["scales"])


if __name__ == "__main__":
    tuning.main(
        HstuCrossTuner(),
        untune_file=str(REPO / "bench" / "aiter" / "untuned" / "hstu_cross.csv"),
        tune_file=str(
            REPO / "mslk" / "attention" / "dispatch" / "tables" / "hstu_cross.csv"
        ),
    )
