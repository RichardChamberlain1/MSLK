# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark MSLK's own attention kernels, on ROCm and CUDA alike.

Covers the attention entry points that run an MSLK kernel on both vendors:

- ``forward`` / ``backward``: ``memory_efficient_attention_forward`` and
  ``memory_efficient_attention_backward`` pinned to MSLK's kernel on each GPU,
  FlyDSL on ROCm and CUTLASS Blackwell on SM100. Without ``op=``, CUDA's
  default dispatch picks FlashAttention, which is not part of MSLK.
- ``merge``: ``merge_attentions`` (Triton, same source on both).
- ``mla_decode`` / ``mla_prefill``: ``mla_decode_fwd`` and ``mla_prefill_fwd``
  (Triton, same source on both).

Timing goes through ``mslk.bench.common.utils.do_bench`` with the rotating
buffer, so every call reads its inputs from HBM. Each case also reports its
relative error against an fp32 PyTorch reference.

Examples::

    python bench/attn/mslk_attn_bench.py
    python bench/attn/mslk_attn_bench.py --kernels forward,backward --seqlens 1024,4096
"""

import itertools
import math
from typing import Callable, Optional

import click
import torch
import torch.nn.functional as F
from mslk.attention import fmha
from mslk.attention.mla import mla_decode_fwd, mla_prefill_fwd
from mslk.bench.common.utils import BenchOptions, do_bench

ALL_KERNELS = ("forward", "backward", "merge", "mla_decode", "mla_prefill")

# DeepSeek-V3 MLA after weight absorption.
MLA_NUM_HEADS = 128
MLA_KV_LORA_RANK = 512
MLA_QK_HEAD_DIM = MLA_KV_LORA_RANK + 64


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float(
        (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
    )


def _report(
    name: str, shape: str, ms: float, err: float, flops: float, mem_bytes: float
) -> None:
    us = ms * 1e3
    print(
        f"{name:<12} {shape:<34} {us:>10.2f} {flops / (ms * 1e-3) / 1e12:>9.1f} "
        f"{mem_bytes / (ms * 1e-3) / 1e9:>9.1f} {err:>10.2e}"
    )


def _mslk_fmha_ops() -> tuple[type, type]:
    """MSLK's own forward/backward attention kernel on this GPU."""
    if torch.version.hip is not None:
        return fmha.flydsl.FwOp, fmha.flydsl.BwOp
    return fmha.cutlass_blackwell.FwOp, fmha.cutlass_blackwell.BwOp


def _sdpa_ref(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> torch.Tensor:
    """fp32 attention on BMHK tensors, returned in BMHK."""
    out = F.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
        is_causal=causal,
    )
    return out.transpose(1, 2)


def _fmha_inputs(B: int, S: int, H: int, D: int, causal: bool, dtype: torch.dtype):
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=dtype) for _ in range(3))
    bias = fmha.attn_bias.LowerTriangularMask() if causal else None
    return q, k, v, bias


def _fmha_flops(B: int, S: int, H: int, D: int, causal: bool) -> float:
    return 4 * B * H * S * S * D * (0.5 if causal else 1.0)


def _supported(op: type, inp: fmha.Inputs) -> Optional[str]:
    if not op.is_available():
        return f"{op.NAME} unavailable"
    reasons = op.not_supported_reasons(inp)
    return "; ".join(reasons) if reasons else None


def bench_forward(B, S, H, D, causal, dtype, opts) -> None:
    fw, _ = _mslk_fmha_ops()
    q, k, v, bias = _fmha_inputs(B, S, H, D, causal, dtype)
    shape = f"B={B} S={S} H={H} D={D}{' causal' if causal else ''}"
    skip = _supported(fw, fmha.Inputs(query=q, key=k, value=v, attn_bias=bias))
    if skip:
        print(f"{'forward':<12} {shape:<34} skipped: {skip}")
        return

    def fn(q, k, v):
        return fmha.memory_efficient_attention_forward(q, k, v, bias, op=fw)

    err = _rel_err(fn(q, k, v), _sdpa_ref(q, k, v, causal))
    ms = do_bench(fn, (q, k, v), opts)
    mem = 4 * B * S * H * D * q.element_size()
    _report("forward", shape, ms, err, _fmha_flops(B, S, H, D, causal), mem)


def bench_backward(B, S, H, D, causal, dtype, opts) -> None:
    fw, bw = _mslk_fmha_ops()
    q, k, v, bias = _fmha_inputs(B, S, H, D, causal, dtype)
    shape = f"B={B} S={S} H={H} D={D}{' causal' if causal else ''}"
    inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=bias)
    skip = _supported(fw, inp) or _supported(bw, inp)
    if skip:
        print(f"{'backward':<12} {shape:<34} skipped: {skip}")
        return

    out, lse = fmha.memory_efficient_attention_forward_requires_grad(
        q, k, v, bias, op=fw
    )
    grad = torch.randn_like(out)

    def fn(grad, out, lse, q, k, v):
        return fmha.memory_efficient_attention_backward(
            grad, out, lse, q, k, v, bias, op=bw
        )

    dq, dk, dv = fn(grad, out, lse, q, k, v)
    qr, kr, vr = (t.detach().float().requires_grad_() for t in (q, k, v))
    _sdpa_ref(qr, kr, vr, causal).backward(grad.float())
    err = max(_rel_err(a, b.grad) for a, b in ((dq, qr), (dk, kr), (dv, vr)))

    ms = do_bench(fn, (grad, out, lse, q, k, v), opts)
    # The usual 2.5x of the forward: five GEMMs in place of two.
    flops = 2.5 * _fmha_flops(B, S, H, D, causal)
    mem = 8 * B * S * H * D * q.element_size()
    _report("backward", shape, ms, err, flops, mem)


def bench_merge(num_splits: int, B: int, M: int, H: int, D: int, opts) -> None:
    attn = torch.randn(num_splits, B, M, H, D, device="cuda", dtype=torch.float32)
    lse = torch.randn(num_splits, B, H, M, device="cuda", dtype=torch.float32)

    def fn(attn, lse):
        return fmha.merge_attentions(attn, lse, write_lse=True)

    out, _ = fn(attn, lse)
    weights = torch.softmax(lse, dim=0).permute(0, 1, 3, 2).unsqueeze(-1)
    err = _rel_err(out, (weights * attn).sum(0))

    ms = do_bench(fn, (attn, lse), opts)
    shape = f"splits={num_splits} B={B} M={M} H={H} D={D}"
    mem = (attn.numel() + lse.numel()) * 4 + (B * M * H * (D + 1)) * 4
    _report("merge", shape, ms, err, 0.0, mem)


def _mla_paged_kv(batch: int, ctx_len: int):
    """Latent KV cache scattered over pages of one token (as in the tests)."""
    kv = torch.randn(batch, ctx_len, MLA_QK_HEAD_DIM, device="cuda") * 0.1
    kv = kv.to(torch.bfloat16)
    num_pages = batch * ctx_len
    block_tables = torch.randperm(num_pages, device="cuda", dtype=torch.int32).view(
        batch, ctx_len
    )
    kv_buffer = torch.empty(
        num_pages, 1, 1, MLA_QK_HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    kv_buffer[block_tables.reshape(-1).long(), 0, 0] = kv.reshape(-1, MLA_QK_HEAD_DIM)
    seqused_k = torch.full((batch,), ctx_len, device="cuda", dtype=torch.int32)
    return kv, kv_buffer, block_tables, seqused_k


def _mla_ref(
    q: torch.Tensor, kv: torch.Tensor, scale: float, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """q: [batch, q_len, H, Dqk]; kv: [batch, ctx, Dqk]."""
    k = kv.float()
    scores = torch.einsum("bqhd,bkd->bqhk", q.float(), k) * scale
    if mask is not None:
        scores.masked_fill_(mask[None, :, None, :].logical_not(), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bqhk,bkd->bqhd", probs, k[..., :MLA_KV_LORA_RANK])


def bench_mla_decode(batch: int, ctx_len: int, opts) -> None:
    kv, kv_buffer, block_tables, seqused_k = _mla_paged_kv(batch, ctx_len)
    q = (torch.randn(batch, MLA_NUM_HEADS, MLA_QK_HEAD_DIM, device="cuda") * 0.1).to(
        torch.bfloat16
    )
    cu_seqlens_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(MLA_QK_HEAD_DIM)

    def fn(q, kv_buffer, block_tables, cu_seqlens_q, seqused_k):
        return mla_decode_fwd(q, kv_buffer, block_tables, cu_seqlens_q, seqused_k)

    args = (q, kv_buffer, block_tables, cu_seqlens_q, seqused_k)
    err = _rel_err(fn(*args), _mla_ref(q.unsqueeze(1), kv, scale, None).squeeze(1))
    ms = do_bench(fn, args, opts)
    flops = 2 * batch * MLA_NUM_HEADS * ctx_len * (MLA_QK_HEAD_DIM + MLA_KV_LORA_RANK)
    mem = kv_buffer.numel() * kv_buffer.element_size()
    _report("mla_decode", f"batch={batch} ctx={ctx_len}", ms, err, flops, mem)


def bench_mla_prefill(batch: int, ctx_len: int, q_len: int, opts) -> None:
    kv, kv_buffer, block_tables, seqused_k = _mla_paged_kv(batch, ctx_len)
    q = (
        torch.randn(batch * q_len, MLA_NUM_HEADS, MLA_QK_HEAD_DIM, device="cuda") * 0.1
    ).to(torch.bfloat16)
    cu_seqlens_q = torch.arange(
        0, batch * q_len + 1, q_len, device="cuda", dtype=torch.int32
    )
    scale = 1.0 / math.sqrt(MLA_QK_HEAD_DIM)

    def fn(q, kv_buffer, block_tables, cu_seqlens_q, seqused_k):
        return mla_prefill_fwd(q, kv_buffer, block_tables, cu_seqlens_q, seqused_k)

    args = (q, kv_buffer, block_tables, cu_seqlens_q, seqused_k)
    # Query token i sees the context up to its own position, at the end of it.
    mask = torch.ones(q_len, ctx_len, dtype=torch.bool, device="cuda").tril(
        diagonal=ctx_len - q_len
    )
    ref = _mla_ref(q.view(batch, q_len, MLA_NUM_HEADS, -1), kv, scale, mask)
    err = _rel_err(fn(*args), ref.reshape(batch * q_len, MLA_NUM_HEADS, -1))
    ms = do_bench(fn, args, opts)
    visible = q_len * ctx_len - q_len * (q_len - 1) / 2
    flops = 2 * batch * MLA_NUM_HEADS * visible * (MLA_QK_HEAD_DIM + MLA_KV_LORA_RANK)
    mem = (kv_buffer.numel() + q.numel()) * 2 + batch * q_len * MLA_NUM_HEADS * 512 * 2
    shape = f"batch={batch} ctx={ctx_len} q_len={q_len}"
    _report("mla_prefill", shape, ms, err, flops, mem)


@click.command()
@click.option(
    "--kernels",
    default=",".join(ALL_KERNELS),
    help=f"Comma separated subset of: {', '.join(ALL_KERNELS)}.",
)
@click.option("--batch", default="1,8", help="Comma separated batch sizes (fmha).")
@click.option("--seqlens", default="1024,4096", help="Comma separated S (fmha).")
@click.option("--heads", default=16, type=int, help="Number of heads (fmha).")
@click.option("--head-dims", default="64,128", help="Comma separated D (fmha).")
@click.option("--dtype", type=click.Choice(["bf16", "f16"]), default="bf16")
@click.option(
    "--cuda-graph/--no-cuda-graph", default=True, help="Time CUDA-graph replays."
)
@click.option(
    "--rotating-buffer/--no-rotating-buffer",
    default=True,
    help="Rotate input copies so every call reads HBM.",
)
def main(
    kernels: str,
    batch: str,
    seqlens: str,
    heads: int,
    head_dims: str,
    dtype: str,
    cuda_graph: bool,
    rotating_buffer: bool,
) -> None:
    selected = [k.strip() for k in kernels.split(",") if k.strip()]
    unknown = set(selected) - set(ALL_KERNELS)
    if unknown:
        raise click.BadParameter(f"unknown kernels: {sorted(unknown)}")
    opts = BenchOptions(cuda_graph=cuda_graph, rotating_buffer=rotating_buffer)
    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype]
    fmha_cases = list(
        itertools.product(
            [int(b) for b in batch.split(",")],
            [int(s) for s in seqlens.split(",")],
            [int(d) for d in head_dims.split(",")],
            (False, True),
        )
    )

    header = ("Kernel", "Shape", "us", "TFLOPS", "GB/s", "RelErr")
    print("{:<12} {:<34} {:>10} {:>9} {:>9} {:>10}".format(*header))
    benches: dict[str, Callable[[], None]] = {
        "forward": lambda: [
            bench_forward(B, S, heads, D, c, torch_dtype, opts)
            for B, S, D, c in fmha_cases
        ],
        "backward": lambda: [
            bench_backward(B, S, heads, D, c, torch_dtype, opts)
            for B, S, D, c in fmha_cases
        ],
        "merge": lambda: [bench_merge(n, 4, 1024, 16, 128, opts) for n in (2, 8)],
        "mla_decode": lambda: [
            bench_mla_decode(b, ctx, opts)
            for b, ctx in itertools.product((1, 16, 64), (1024, 4096))
        ],
        "mla_prefill": lambda: [
            bench_mla_prefill(b, ctx, ql, opts)
            for b, ctx, ql in ((1, 4096, 4096), (4, 4096, 1024), (16, 2048, 256))
        ],
    }
    for name in selected:
        benches[name]()

    print(f"\nHardware: {torch.cuda.get_device_name()}")
    print("Benchmark Settings:")
    print(f"    CUDA graph: {cuda_graph}")
    print(f"    Buffer rotation: {rotating_buffer}")
    fw, bw = _mslk_fmha_ops()
    print(f"    forward/backward ops: {fw.NAME} / {bw.NAME}")


if __name__ == "__main__":
    main()
