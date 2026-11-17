# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for mslk.attention.sdpa — SDPA ops matching aten contracts."""

import pytest
import torch
from mslk.attention.sdpa import sdpa_efficient_fwd, sdpa_flash_bwd, sdpa_flash_fwd

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/ROCm"
)


def _ref_sdpa(Q, K, V, scale=None, is_causal=False, attn_mask=None):
    """Reference SDPA. Uses an explicit bottom-right causal mask when Lq != Lkv,
    matching the aten flash contract (PyTorch's math kernel uses top-left)."""
    Lq, Lkv = Q.shape[2], K.shape[2]
    if is_causal and Lq != Lkv:
        mask = torch.zeros(Lq, Lkv, device=Q.device, dtype=torch.float32)
        for i in range(Lq):
            for j in range(Lkv):
                if j > i + (Lkv - Lq):
                    mask[i, j] = float("-inf")
        return torch.nn.functional.scaled_dot_product_attention(
            Q.float(),
            K.float(),
            V.float(),
            scale=scale,
            attn_mask=mask,
        )
    return torch.nn.functional.scaled_dot_product_attention(
        Q.float(),
        K.float(),
        V.float(),
        scale=scale,
        is_causal=is_causal,
        attn_mask=attn_mask,
    )


SHAPES = [
    (1, 1, 32, 32, 64),
    (2, 4, 16, 64, 128),
    (1, 2, 32, 32, 256),
    (1, 8, 64, 64, 80),
    (2, 4, 128, 128, 96),
]


@skip_no_cuda
class TestSdpaFlashForward:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("B,H,Lq,Lkv,D", SHAPES)
    @pytest.mark.parametrize("is_causal", [False, True])
    def test_shape_and_correctness(self, dtype, B, H, Lq, Lkv, D, is_causal):
        if is_causal and Lq > Lkv:
            pytest.skip("causal requires Lq <= Lkv for bottom-right mask")
        Q = torch.randn(B, H, Lq, D, device="cuda", dtype=dtype)
        K = torch.randn(B, H, Lkv, D, device="cuda", dtype=dtype)
        V = torch.randn(B, H, Lkv, D, device="cuda", dtype=dtype)

        result = sdpa_flash_fwd(Q, K, V, is_causal=is_causal)
        out = result[0]

        assert out.shape == (B, H, Lq, D)
        assert out.dtype == dtype

        ref = _ref_sdpa(Q, K, V, is_causal=is_causal)
        torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.02)

    def test_rejects_dropout(self):
        Q = torch.randn(1, 1, 16, 64, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(NotImplementedError, match="dropout"):
            sdpa_flash_fwd(Q, Q, Q, dropout_p=0.1)

    @pytest.mark.parametrize("D", [64, 80, 96, 128, 256])
    def test_head_dims(self, D):
        Q = torch.randn(1, 4, 32, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(1, 4, 32, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(1, 4, 32, D, device="cuda", dtype=torch.bfloat16)

        out = sdpa_flash_fwd(Q, K, V)[0]
        ref = _ref_sdpa(Q, K, V)
        torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.02)


@skip_no_cuda
class TestSdpaFlashBackward:
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize(
        "B,H,Lq,Lkv,D",
        [(1, 2, 32, 32, 64), (2, 4, 16, 16, 128), (1, 4, 64, 64, 96)],
    )
    @pytest.mark.parametrize("is_causal", [False, True])
    def test_backward_correctness(self, dtype, B, H, Lq, Lkv, D, is_causal):
        Q = torch.randn(B, H, Lq, D, device="cuda", dtype=dtype, requires_grad=True)
        K = torch.randn(B, H, Lkv, D, device="cuda", dtype=dtype, requires_grad=True)
        V = torch.randn(B, H, Lkv, D, device="cuda", dtype=dtype, requires_grad=True)

        # Forward
        result = sdpa_flash_fwd(Q, K, V, is_causal=is_causal)
        out, lse = result[0], result[1]
        cum_q, cum_k = result[2], result[3]
        max_q, max_k = result[4], result[5]
        rng = result[6]

        # Grad
        grad = torch.randn_like(out)
        dQ, dK, dV = sdpa_flash_bwd(
            grad,
            Q,
            K,
            V,
            out,
            lse,
            cum_q,
            cum_k,
            max_q,
            max_k,
            0.0,
            is_causal,
            rng,
            rng,
            scale=None,
        )

        assert dQ.shape == Q.shape
        assert dK.shape == K.shape
        assert dV.shape == V.shape

        # Reference backward via autograd
        Q_ref = Q.detach().float().requires_grad_(True)
        K_ref = K.detach().float().requires_grad_(True)
        V_ref = V.detach().float().requires_grad_(True)
        ref_out = _ref_sdpa(Q_ref, K_ref, V_ref, is_causal=is_causal)
        ref_out.backward(grad.float())

        torch.testing.assert_close(dQ.float(), Q_ref.grad, atol=0.05, rtol=0.03)
        torch.testing.assert_close(dK.float(), K_ref.grad, atol=0.05, rtol=0.03)
        torch.testing.assert_close(dV.float(), V_ref.grad, atol=0.05, rtol=0.03)


@skip_no_cuda
class TestSdpaEfficientForward:
    @pytest.mark.parametrize("B,H,Lq,Lkv,D", SHAPES[:3])
    def test_efficient_fwd(self, B, H, Lq, Lkv, D):
        Q = torch.randn(B, H, Lq, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, H, Lkv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, H, Lkv, D, device="cuda", dtype=torch.bfloat16)

        out, lse, _, _ = sdpa_efficient_fwd(
            Q, K, V, attn_bias=None, compute_log_sumexp=True
        )
        assert out.shape == (B, H, Lq, D)
        ref = _ref_sdpa(Q, K, V)
        torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.02)


@skip_no_cuda
class TestSdpaCrossAttention:
    def test_noncausal(self):
        Q = torch.randn(1, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(1, 2, 64, 128, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(1, 2, 64, 128, device="cuda", dtype=torch.bfloat16)

        out = sdpa_flash_fwd(Q, K, V, is_causal=False)[0]
        ref = _ref_sdpa(Q, K, V, is_causal=False)
        torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.02)

    def test_causal_bottom_right(self):
        Q = torch.randn(1, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(1, 2, 64, 128, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(1, 2, 64, 128, device="cuda", dtype=torch.bfloat16)

        out = sdpa_flash_fwd(Q, K, V, is_causal=True)[0]
        ref = _ref_sdpa(Q, K, V, is_causal=True)
        torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.02)


@skip_no_cuda
class TestSdpaKisV:
    def test_flash_k_is_v(self):
        Q = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16)
        KV = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16)

        out = sdpa_flash_fwd(Q, KV, KV)[0]
        ref = _ref_sdpa(Q, KV, KV)
        torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.02)


@skip_no_cuda
class TestSdpaEdgeCases:
    def test_lq_1(self):
        Q = torch.randn(1, 4, 1, 64, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)

        out = sdpa_flash_fwd(Q, K, V)[0]
        ref = _ref_sdpa(Q, K, V)
        torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.02)

    def test_large_batch(self):
        Q = torch.randn(8, 32, 16, 64, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(8, 32, 16, 64, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(8, 32, 16, 64, device="cuda", dtype=torch.bfloat16)

        out = sdpa_flash_fwd(Q, K, V)[0]
        ref = _ref_sdpa(Q, K, V)
        torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.02)


@skip_no_cuda
class TestSdpaAtenOverride:
    def test_override_api_exists(self):
        from mslk.attention.sdpa import disable_aten_override, enable_aten_override

        assert callable(enable_aten_override)
        assert callable(disable_aten_override)
