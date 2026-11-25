# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import contextlib
import unittest
from typing import Any, Callable, Dict, List, Tuple

import torch
from mslk.aiter.common import is_aiter_available, is_aiter_enabled
from mslk.aiter.routing import force_backend
from mslk.quantize.dispatch.quantize_fp8_tensor import (
    OP as QUANTIZE_FP8_TENSOR_OP,
    ROUTING as QUANTIZE_FP8_TENSOR_ROUTING,
)
from mslk.quantize.triton.fp8_quantize import (
    quantize_fp8_block,
    quantize_fp8_row,
    quantize_fp8_tensor,
    triton_quantize_fp8_tensor,
)
from mslk.testing.device import skipUnlessRocm
from mslk.utils.triton.fp8_utils import get_fp8_constants
from parameterized import parameterized


def undo_tensorwise_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float) * scale


def undo_rowwise_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float) * scale.unsqueeze(1)


def undo_blockwise_quant(
    x: torch.Tensor,
    scale: torch.Tensor,
    block_m: int = 256,
    block_k: int = 256,
) -> torch.Tensor:
    M, K = x.shape[-2], x.shape[-1]
    block_m = min(block_m, M)
    block_k = min(block_k, K)

    # Pad to be divisible by block sizes
    pad_m = (block_m - M % block_m) % block_m
    pad_k = (block_k - K % block_k) % block_k

    x = x.to(torch.float)
    if pad_m > 0 or pad_k > 0:
        x = torch.nn.functional.pad(x, (0, pad_k, 0, pad_m))

    M_padded, K_padded = x.shape[-2], x.shape[-1]
    grid_m = M_padded // block_m
    grid_k = K_padded // block_k

    # Reshape into blocks and apply per-block scale
    x = x.reshape(grid_m, block_m, grid_k, block_k)
    x = x * scale[:, None, :, None]
    x = x.reshape(M_padded, K_padded)

    # Remove padding
    return x[:M, :K]


INPUT_SHAPES: List[List[int]] = [
    # Small shape
    [16, 16],
    # Non-square shape
    [32, 64],
    # Medium shape
    [1024, 1024],
    # Large shape
    [8192, 4096],
    # Non-power-of-2 shapes
    [333, 777],
    # Small dimensions
    [1, 128],
    [128, 1],
]

TRITON_QUANTIZE_OPS: List[Tuple[Callable[..., Any], Callable[..., Any]]] = [
    (quantize_fp8_row, undo_rowwise_quant),
    (quantize_fp8_block, undo_blockwise_quant),
    (quantize_fp8_tensor, undo_tensorwise_quant),
]

ALL_QUANTIZE_OPS: List[Tuple[Callable[..., Any], Callable[..., Any]]] = (
    TRITON_QUANTIZE_OPS
)


def _generate_test_cases() -> List[
    Tuple[str, List[int], Callable[..., Any], Callable[..., Any]]
]:
    """Generate test cases for parameterized tests."""
    test_cases = []
    for shape in INPUT_SHAPES:
        for quantize_op, dequantize_op in ALL_QUANTIZE_OPS:
            name = f"{quantize_op}_{shape[0]}x{shape[1]}"
            test_cases.append((name, shape, quantize_op, dequantize_op))
    return test_cases


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Operators are only available on CUDA enabled machines",
)
@unittest.skipIf(
    torch.version.cuda is None or torch.cuda.get_device_capability("cuda") < (9, 0),
    "Only support sm90+",
)
class FP8CorrectnessTest(unittest.TestCase):
    """Check that FP8 ops produce correct results."""

    # pyre-ignore[56]: Pyre can't track decorator-modified signatures.
    @parameterized.expand(_generate_test_cases())
    def test_quantize(
        self,
        _name: str,
        input_shape: List[int],
        quantize_op: Callable[..., Any],
        dequantize_op: Callable[..., Any],
    ) -> None:
        test_input = torch.randn(input_shape, device="cuda", dtype=torch.bfloat16)
        quant_out = quantize_op(test_input)
        reconstructed_input = dequantize_op(*quant_out).to(torch.bfloat16)
        torch.testing.assert_close(
            test_input, reconstructed_input, atol=2.5e-1, rtol=1e-3
        )


# The names quantize_fp8_tensor is called by; all reach its routing on ROCm.
QUANTIZE_FP8_TENSOR_NAMES: Dict[str, Callable[..., Any]] = {
    "quantize_fp8_tensor": quantize_fp8_tensor,
    "torch.ops.triton.quantize_fp8_tensor": lambda a: (
        torch.ops.triton.quantize_fp8_tensor(a)
    ),
    "triton_quantize_fp8_tensor": triton_quantize_fp8_tensor,
}

# Shapes that quantize_fp8_tensor's tuning table routes to aiter on gfx950, in
# BF16 and FP16, so that the op's tests cover aiter's kernel as well as MSLK's.
QUANTIZE_FP8_TENSOR_AITER_SHAPES: Tuple[Tuple[int, ...], ...] = (
    (1, 4096),  # decode
    (100, 8192),
    (3000, 16384),  # prefill, large K
    (2, 8, 4096),  # leading dims
)


@skipUnlessRocm()
class QuantizeFP8TensorRocmTest(unittest.TestCase):
    """quantize_fp8_tensor on ROCm, through each of its names, under its routing
    (which sends the shapes in its tuning table to aiter or MSLK's Triton
    kernel) and with MSLK's kernel forced."""

    def assert_quantized(
        self, a: torch.Tensor, a_fp8: torch.Tensor, scale: torch.Tensor
    ) -> None:
        # As the op's tuning script checks: the scale within 2^-21 (~4 ulps) of
        # max(amax, eps) / FP8_MAX, and each value within one FP8 step of the
        # input times its reciprocal, saturated and rounded to FP8.
        fp8, _, fp8_max, eps = get_fp8_constants()
        scale_ref = a.abs().max().float().clamp(min=eps) / fp8_max
        a_fp8_ref = (a.float() * (1.0 / scale_ref)).clamp(-fp8_max, fp8_max).to(fp8)
        self.assertEqual(a_fp8.dtype, fp8)
        self.assertEqual(a_fp8.shape, a.shape)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(scale.shape, ())
        torch.testing.assert_close(scale, scale_ref, atol=0, rtol=2.0**-21)
        torch.testing.assert_close(
            a_fp8.float(), a_fp8_ref.float(), atol=2.0**-9, rtol=2.0**-3
        )

    @parameterized.expand(
        [
            (shape, dtype, name, backend)
            for shape, dtypes in (
                *(
                    (s, (torch.bfloat16, torch.float16))
                    for s in QUANTIZE_FP8_TENSOR_AITER_SHAPES
                ),
                ((4096, 4096), (torch.bfloat16, torch.float16)),  # in the table, MSLK
                # Not in the table, or not taken by aiter.
                ((333, 777), (torch.bfloat16,)),
                ((128, 1), (torch.bfloat16,)),
                ((1000,), (torch.float16,)),
                ((100, 8192), (torch.float32,)),  # a shape routed to aiter
            )
            for dtype in dtypes
            for name in QUANTIZE_FP8_TENSOR_NAMES
            # "default" uses the op's routing; "mslk" forces MSLK's kernel, the
            # fallback, on every shape.
            for backend in ("default", "mslk")
        ]
    )
    def test_quantize(
        self, shape: Tuple[int, ...], dtype: torch.dtype, name: str, backend: str
    ) -> None:
        a = torch.randn(shape, device="cuda", dtype=dtype)
        with (
            force_backend(QUANTIZE_FP8_TENSOR_OP, backend)
            if backend != "default"
            else contextlib.nullcontext()
        ):
            a_fp8, scale = QUANTIZE_FP8_TENSOR_NAMES[name](a)
        self.assert_quantized(a, a_fp8, scale)
        if (
            backend == "default"
            and shape in QUANTIZE_FP8_TENSOR_AITER_SHAPES
            and dtype != torch.float32
            and is_aiter_enabled(QUANTIZE_FP8_TENSOR_OP)
        ):
            self.assertEqual(
                QUANTIZE_FP8_TENSOR_ROUTING.backend_for(a),
                "aiter",
                "the tuning table no longer routes this shape to aiter; replace "
                "it in QUANTIZE_FP8_TENSOR_AITER_SHAPES with one it does",
            )

    @parameterized.expand([("default",), ("mslk",)])
    def test_all_zero_input(self, backend: str) -> None:
        # aiter returns scale 0 (and -FP8_MAX values) where MSLK's kernel
        # clamps the amax to eps (and returns zeros): either way, the tensor
        # dequantizes to zero, with no NaN.
        a = torch.zeros(
            QUANTIZE_FP8_TENSOR_AITER_SHAPES[0], device="cuda", dtype=torch.bfloat16
        )
        with (
            force_backend(QUANTIZE_FP8_TENSOR_OP, backend)
            if backend != "default"
            else contextlib.nullcontext()
        ):
            a_fp8, scale = quantize_fp8_tensor(a)
        self.assertFalse(torch.isnan(a_fp8.float()).any())
        self.assertTrue(torch.isfinite(scale) and scale >= 0)
        torch.testing.assert_close(
            a_fp8.float() * scale,
            torch.zeros_like(a, dtype=torch.float32),
            atol=0,
            rtol=0,
        )

    def test_non_contiguous_input_keeps_its_layout(self) -> None:
        # aiter takes contiguous tensors only, so a transposed one, in a shape
        # the table routes to aiter, is served by MSLK's kernel, whose output
        # keeps the input's strides.
        M, K = QUANTIZE_FP8_TENSOR_AITER_SHAPES[1]
        a = torch.randn(K, M, device="cuda", dtype=torch.bfloat16).t()
        a_fp8, scale = quantize_fp8_tensor(a)
        self.assertEqual(QUANTIZE_FP8_TENSOR_ROUTING.backend_for(a), "mslk")
        self.assertEqual(a_fp8.stride(), a.stride())
        self.assert_quantized(a, a_fp8, scale)

    def test_aiter_declines_what_it_cannot_compute(self) -> None:
        # Even forced: inputs other than contiguous BF16/FP16 with rows of whole
        # 16 bytes (aiter computes rows of 7 or 33 BF16 elements wrong).
        if not is_aiter_available():
            self.skipTest("requires the pinned aiter")
        declined = [
            torch.randn(3, 7, device="cuda", dtype=torch.bfloat16),
            torch.randn(5, 33, device="cuda", dtype=torch.bfloat16),
            torch.randn(16, 4096, device="cuda", dtype=torch.float32),
            torch.randn(4096, 16, device="cuda", dtype=torch.bfloat16).t(),
            torch.randn((), device="cuda", dtype=torch.bfloat16),
            torch.empty(0, 4096, device="cuda", dtype=torch.bfloat16),
        ]
        with force_backend(QUANTIZE_FP8_TENSOR_OP, "aiter"):
            for a in declined:
                with self.subTest(shape=a.shape, dtype=a.dtype, stride=a.stride()):
                    self.assertEqual(QUANTIZE_FP8_TENSOR_ROUTING.backend_for(a), "mslk")
            a = torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16)
            self.assertEqual(QUANTIZE_FP8_TENSOR_ROUTING.backend_for(a), "aiter")


if __name__ == "__main__":
    unittest.main()
