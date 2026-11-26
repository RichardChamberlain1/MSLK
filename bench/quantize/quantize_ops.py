# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
from typing import Any, TypeVar

import torch
from mslk.bench.common.utils import BenchOptions, do_bench
from mslk.quantize.triton.fp4_quantize import (
    calculate_group_max,
    nvfp4_quantize_stacked,
    nvfp4_quantize_stacked_with_token_scale,
    triton_quantize_nvfp4,
)
from mslk.quantize.triton.fp4_utils import (
    dequantize_mx4,
    dequantize_mx4_unswizzled,
    dequantize_nvfp4,
    global_scale_nvfp4,
)
from mslk.quantize.triton.fp8_quantize import (
    dequantize_fp8_block,
    dequantize_fp8_row,
    quantize_fp8_row,
    scale_fp8_row,
    triton_quantize_fp8_block,
    triton_quantize_fp8_group,
    triton_quantize_fp8_tensor,
)
from mslk.quantize.triton.quantize_kernels.mx4 import quantize_mx4
from mslk.quantize.triton.quantize_kernels.mx4_stacked import quantize_mx4_stacked
from mslk.utils.device import is_cuda, is_rocm


class QuantizeOpBase(metaclass=abc.ABCMeta):
    """Helper abstract class to define expected methods of quantize ops."""

    @abc.abstractmethod
    def quantize(self, input: torch.Tensor) -> Any:
        """Function which quantizes inputs."""
        pass

    @abc.abstractmethod
    def dequantize(self, *args: Any) -> torch.Tensor:
        """Function which dequantizes inputs. Used for sanity checking."""
        pass

    @abc.abstractproperty
    # pyrefly: ignore [bad-return]
    def hip(self) -> bool:
        """Whether this operator supports AMD or not."""
        pass

    @abc.abstractproperty
    # pyrefly: ignore [bad-return]
    def cuda(self) -> bool:
        """Whether this operator supports Nvidia or not."""
        pass

    def preprocess(self, input: torch.Tensor, num_groups: int = 1) -> Any:
        """This is used for ops that require additional preprocessing. This method will not be counted in benchmarking."""
        return ()

    @property
    def name(self) -> str:
        """Name of this operator."""
        return self.__class__.__name__

    @property
    def supported(self) -> bool:
        """Whether this op will run on the current device."""
        if is_rocm():
            return self.hip
        elif is_cuda():
            return self.cuda
        else:
            return False

    def benchmark(
        self,
        input: torch.Tensor,
        args: Any,
        opts: BenchOptions,
    ) -> float:
        """Benchmark runtime of this operator using do_bench from common."""
        return do_bench(
            lambda inp, *a: self.quantize(inp, *a),
            (input, *args),
            opts,
        )

    def io_bytes(self, input: torch.Tensor, args: Any, outputs: Any) -> int:
        """Bytes the timed call reads and writes, for the bandwidth column."""
        input_bytes = input.numel() * input.element_size()
        return input_bytes + sum(t.numel() * t.element_size() for t in outputs)


op_registry: dict[str, QuantizeOpBase] = {}

T = TypeVar("T", bound=QuantizeOpBase)


def register_op(op_class: type[T]) -> type[T]:
    """Decorator function for assembling all quantize ops."""
    op_registry[op_class.__name__] = op_class()
    return op_class


def get_ops() -> list[QuantizeOpBase]:
    """Get all registered quantize ops."""
    return list(op_registry.values())


@register_op
class Copy(QuantizeOpBase):
    """Copies input to output.

    Useful for measuring achievable DRAM bandwidth.
    """

    def quantize(self, input: torch.Tensor) -> Any:
        return (input.clone(),)

    def dequantize(self, *args: Any) -> torch.Tensor:
        return args[0]

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonFP8Rowwise(QuantizeOpBase):
    def quantize(self, input: torch.Tensor) -> Any:
        return quantize_fp8_row(input)

    def dequantize(self, *args: Any) -> torch.Tensor:
        input_quantized: torch.Tensor
        scale: torch.Tensor
        input_quantized, scale = args
        return dequantize_fp8_row(input_quantized, scale)

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonFP8Blockwise(QuantizeOpBase):
    def __init__(self) -> None:
        super().__init__()
        self.block_m = 128
        self.block_k = 128

    def quantize(self, input: torch.Tensor) -> Any:
        return triton_quantize_fp8_block(
            input, block_m=self.block_m, block_k=self.block_k
        )

    def dequantize(self, *args: Any) -> torch.Tensor:
        input_quantized: torch.Tensor
        scale: torch.Tensor
        input_quantized, scale = args
        return dequantize_fp8_block(input_quantized, scale, self.block_m, self.block_k)

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonFP8Groupwise(QuantizeOpBase):
    def __init__(self) -> None:
        super().__init__()
        self.group_size = 128

    def quantize(self, input: torch.Tensor) -> Any:
        return triton_quantize_fp8_group(input, group_size=self.group_size)

    def dequantize(self, *args: Any) -> torch.Tensor:
        input_quantized: torch.Tensor
        scale: torch.Tensor
        input_quantized, scale = args

        input_quantized = input_quantized.to(torch.float)
        dequantized = input_quantized.view(
            -1, input_quantized.shape[1] // self.group_size, self.group_size
        ) * scale.unsqueeze(-1)
        return dequantized.view(input_quantized.shape)

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonFP8RowwiseScale(QuantizeOpBase):
    """scale_fp8_row: applies row and column scales to a GEMM output."""

    def preprocess(self, input: torch.Tensor, num_groups: int = 1) -> Any:
        M, K = input.shape
        x_scale = torch.rand(M, device=input.device, dtype=torch.float32) + 0.5
        w_scale = torch.rand(K, device=input.device, dtype=torch.float32) + 0.5
        return (x_scale, w_scale)

    def quantize(self, input: torch.Tensor, *args: Any) -> Any:
        x_scale, w_scale = args
        return (scale_fp8_row(input, x_scale, w_scale), x_scale, w_scale)

    def dequantize(self, *args: Any) -> torch.Tensor:
        scaled, x_scale, w_scale = args
        return scaled.float() / (x_scale[:, None] * w_scale[None, :])

    def io_bytes(self, input: torch.Tensor, args: Any, outputs: Any) -> int:
        # Reads the input and both scale vectors, writes one output.
        scale_bytes = sum(t.numel() * t.element_size() for t in args)
        return 2 * input.numel() * input.element_size() + scale_bytes

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


class _DequantizeOpBase(QuantizeOpBase):
    """Times a dequantize kernel; the quantized operands are prepared untimed.

    Only the quantized operands are timed and rotated: the bf16 input is not
    read by the kernel.
    """

    def benchmark(
        self,
        input: torch.Tensor,
        args: Any,
        opts: BenchOptions,
    ) -> float:
        return do_bench(lambda *a: self._dequantize(*a), tuple(args), opts)

    def _dequantize(self, xq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def quantize(self, input: torch.Tensor, *args: Any) -> Any:
        return (self._dequantize(*args),)

    def dequantize(self, *args: Any) -> torch.Tensor:
        return args[0]

    def io_bytes(self, input: torch.Tensor, args: Any, outputs: Any) -> int:
        operand_bytes = sum(t.numel() * t.element_size() for t in args)
        return operand_bytes + sum(t.numel() * t.element_size() for t in outputs)

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonFP8RowwiseDequant(_DequantizeOpBase):
    """dequantize_fp8_row."""

    def preprocess(self, input: torch.Tensor, num_groups: int = 1) -> Any:
        return quantize_fp8_row(input)

    def _dequantize(self, xq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return dequantize_fp8_row(xq, scale)


@register_op
class TritonFP8BlockwiseDequant(_DequantizeOpBase):
    """dequantize_fp8_block."""

    def __init__(self) -> None:
        super().__init__()
        self.block_m = 128
        self.block_k = 128

    def preprocess(self, input: torch.Tensor, num_groups: int = 1) -> Any:
        return triton_quantize_fp8_block(
            input, block_m=self.block_m, block_k=self.block_k
        )

    def _dequantize(self, xq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return dequantize_fp8_block(xq, scale, self.block_m, self.block_k)


@register_op
class TritonNVFP4(QuantizeOpBase):
    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, input: torch.Tensor, num_groups: int = 1) -> Any:
        global_scale = global_scale_nvfp4(input)
        return (global_scale,)

    def quantize(self, input: torch.Tensor, *args: Any) -> Any:
        global_scale: torch.Tensor
        global_scale = args[0]
        input_quantized, scales = triton_quantize_nvfp4(input, global_scale)
        return input_quantized.view(torch.uint8), scales, global_scale

    def dequantize(self, *args: Any) -> torch.Tensor:
        input_quantized: torch.Tensor
        scale: torch.Tensor
        global_scale: torch.Tensor
        input_quantized, scale, global_scale = args

        return dequantize_nvfp4(input_quantized, scale, global_scale)

    @property
    def hip(self) -> bool:
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_op
class CudaFP8Rowwise(QuantizeOpBase):
    def quantize(self, input: torch.Tensor) -> Any:
        return quantize_fp8_row(input)

    def dequantize(self, *args: Any) -> torch.Tensor:
        input_quantized: torch.Tensor
        scale: torch.Tensor
        input_quantized, scale = args
        return dequantize_fp8_row(input_quantized, scale)

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonFP8Tensorwise(QuantizeOpBase):
    def quantize(self, input: torch.Tensor) -> Any:
        return triton_quantize_fp8_tensor(input)

    def dequantize(self, *args: Any) -> torch.Tensor:
        input_quantized: torch.Tensor
        scale: torch.Tensor
        input_quantized, scale = args
        return input_quantized.to(torch.float32) * scale

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonNVFP4Stacked(QuantizeOpBase):
    def __init__(self) -> None:
        super().__init__()
        self.num_groups: int = 1
        self.input_shape: tuple[int, ...] = (0, 0)

    def preprocess(self, input: torch.Tensor, num_groups: int = 1) -> Any:
        self.num_groups = num_groups
        self.input_shape = tuple(input.shape)
        M = input.shape[0]
        # Split M evenly across num_groups.
        base = M // num_groups
        remainder = M % num_groups
        sizes = [base + (1 if i < remainder else 0) for i in range(num_groups)]
        m_sizes = torch.tensor(sizes, dtype=torch.int64, device=input.device)
        x_global_scale, _ = calculate_group_max(input, m_sizes)
        return (m_sizes, x_global_scale)

    def quantize(self, input: torch.Tensor, *args: Any) -> Any:
        m_sizes: torch.Tensor = args[0]
        x_global_scale: torch.Tensor = args[1]
        xq, scale = nvfp4_quantize_stacked(m_sizes, input, x_global_scale)
        return xq, scale, x_global_scale

    def dequantize(self, *args: Any) -> torch.Tensor:
        # Grouped dequant is complex due to per-segment padding; return zeros.
        return torch.zeros(
            self.input_shape, dtype=torch.bfloat16, device=args[0].device
        )

    @property
    def hip(self) -> bool:
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonNVFP4StackedTokenScale(QuantizeOpBase):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape: tuple[int, ...] = (0, 0)

    def preprocess(self, input: torch.Tensor, num_groups: int = 1) -> Any:
        self.input_shape = tuple(input.shape)
        M = input.shape[0]
        # Split M evenly across num_groups.
        base = M // num_groups
        remainder = M % num_groups
        sizes = [base + (1 if i < remainder else 0) for i in range(num_groups)]
        m_sizes = torch.tensor(sizes, dtype=torch.int64, device=input.device)
        return (m_sizes,)

    def quantize(self, input: torch.Tensor, *args: Any) -> Any:
        m_sizes: torch.Tensor = args[0]
        return nvfp4_quantize_stacked_with_token_scale(m_sizes, input)

    def dequantize(self, *args: Any) -> torch.Tensor:
        # Grouped dequant is complex due to per-segment padding; return zeros.
        return torch.zeros(
            self.input_shape, dtype=torch.bfloat16, device=args[0].device
        )

    @property
    def hip(self) -> bool:
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonMXFP4(QuantizeOpBase):
    """MXFP4 quantize via quantize_kernels.mx4.quantize_mx4."""

    def quantize(self, input: torch.Tensor, *args: Any) -> Any:
        xq, scales = quantize_mx4(input)
        return xq, scales

    def dequantize(self, *args: Any) -> torch.Tensor:
        xq, scales = args[0], args[1]
        # quantize_mx4 returns plain [M, K/32] scales on ROCm and CUTLASS's
        # swizzled layout on CUDA.
        if is_rocm():
            return dequantize_mx4_unswizzled(xq, scales)
        return dequantize_mx4(xq, scales)

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonMXFP4Stacked(QuantizeOpBase):
    """Stacked (MoE) MXFP4 quantize via quantize_kernels.mx4_stacked."""

    def __init__(self) -> None:
        super().__init__()
        self.input_shape: tuple[int, ...] = (0, 0)

    def preprocess(self, input: torch.Tensor, num_groups: int = 1) -> Any:
        self.input_shape = tuple(input.shape)
        M = input.shape[0]
        # Split M evenly across num_groups.
        base = M // num_groups
        remainder = M % num_groups
        sizes = [base + (1 if i < remainder else 0) for i in range(num_groups)]
        m_sizes = torch.tensor(sizes, dtype=torch.int64, device=input.device)
        return (m_sizes,)

    def quantize(self, input: torch.Tensor, *args: Any) -> Any:
        m_sizes: torch.Tensor = args[0]
        xq, scales = quantize_mx4_stacked(m_sizes, input)
        return xq, scales

    def dequantize(self, *args: Any) -> torch.Tensor:
        # ROCm scales are plain [M, K/32] rows, so dequantizing is direct.
        if is_rocm():
            return dequantize_mx4_unswizzled(args[0], args[1])
        # CUDA pads each segment's scales to 128 rows; return zeros.
        return torch.zeros(
            self.input_shape, dtype=torch.bfloat16, device=args[0].device
        )

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True
