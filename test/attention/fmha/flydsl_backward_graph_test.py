# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The FlyDSL varlen backward takes its max sequence lengths from the mask on
the host: no device-to-host sync, so it can be captured in a CUDA graph."""

import pytest
import torch
from mslk.attention import fmha
from mslk.attention.fmha import flydsl
from mslk.attention.fmha.attn_bias import BlockDiagonalCausalMask, BlockDiagonalMask
from mslk.utils.device import is_gfx950

OPS = (flydsl.FwOp, flydsl.BwOp)


def _inputs(bias_type):
    torch.manual_seed(0)
    q_lens, kv_lens = [200, 17, 64] * 4, [4000, 33, 512] * 4
    if bias_type is BlockDiagonalCausalMask:
        kv_lens = q_lens
    bias = bias_type.from_seqlens(q_seqlen=q_lens, kv_seqlen=kv_lens, device="cuda")
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    q = torch.randn(1, sum(q_lens), 2, 64, **bf16).requires_grad_(True)
    k = torch.randn(1, sum(kv_lens), 2, 64, **bf16).requires_grad_(True)
    v = torch.randn(1, sum(kv_lens), 2, 64, **bf16).requires_grad_(True)
    out = fmha.memory_efficient_attention(q, k, v, attn_bias=bias, op=OPS)
    return (q, k, v), out, torch.randn_like(out)


def _require_flydsl() -> None:
    if not torch.version.hip or not is_gfx950():
        pytest.skip("requires gfx950")
    assert flydsl._is_flydsl_available(), "FlyDSL is unavailable on gfx950"


@pytest.mark.parametrize("bias_type", [BlockDiagonalMask, BlockDiagonalCausalMask])
def test_varlen_backward_does_not_sync(bias_type) -> None:
    _require_flydsl()
    leaves, out, dout = _inputs(bias_type)
    torch.autograd.grad(out, leaves, dout, retain_graph=True)  # compile
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        torch.autograd.grad(out, leaves, dout, retain_graph=True)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("bias_type", [BlockDiagonalMask, BlockDiagonalCausalMask])
def test_varlen_backward_cuda_graph_replay_matches_eager(bias_type) -> None:
    _require_flydsl()
    # Forward, warm-up and capture on one stream: autograd nodes keep the
    # stream they ran on, and capture refuses nodes from another stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        leaves, out, dout = _inputs(bias_type)
        eager = torch.autograd.grad(out, leaves, dout, retain_graph=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = torch.autograd.grad(out, leaves, dout, retain_graph=True)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    # dQ is accumulated with atomics, so the order (and last bits) can differ.
    for name, got, ref in zip(("dq", "dk", "dv"), captured, eager):
        torch.testing.assert_close(
            got, ref, atol=1e-2, rtol=1e-2, msg=lambda m, n=name: f"{n}: {m}"
        )


def test_varlen_backward_requires_max_seqlens() -> None:
    _require_flydsl()
    bias = BlockDiagonalMask.from_seqlens(
        q_seqlen=[8, 4], kv_seqlen=[16, 8], device="cuda"
    )
    q = torch.randn(1, 12, 1, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 24, 1, 64, device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros(1, 1, 12, device="cuda", dtype=torch.float32)
    seqstart_q, seqstart_k = bias.q_seqinfo.seqstart, bias.k_seqinfo.seqstart
    with pytest.raises(ValueError, match="max_seqlen_q and max_seqlen_k"):
        torch.ops.mslk_flydsl.fmha_bwd(
            q, k, k, q, lse, q, 0.125, 0, seqstart_q, seqstart_k
        )
