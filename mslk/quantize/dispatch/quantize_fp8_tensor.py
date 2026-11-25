# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which backend serves ``quantize_fp8_tensor`` on ROCm.

``quantize_fp8_tensor`` (the ``triton::quantize_fp8_tensor`` op) and
``triton_quantize_fp8_tensor`` both run through
``mslk.quantize.triton.fp8_quantize.triton_quantize_fp8_tensor``, which calls
:data:`ROUTING` on ROCm:

- ``mslk``: MSLK's Triton kernel (``mslk/quantize/triton/fp8_quantize.py``),
  the default and the fallback.
- ``aiter``: aiter's ``per_tensor_quant_hip``
  (``mslk/quantize/aiter/quantize_fp8_tensor.py``), for the shapes
  ``tables/quantize_fp8_tensor.csv`` assigns to it. That table is written by
  ``bench/aiter/tune_quantize_fp8_tensor.py``.

The table is keyed by the input's dtype and its last dimension K, and bucketed
by its number of rows M (all other dimensions): aiter runs one thread block per
row, so its speed depends on both, while MSLK's kernel depends on the size only.
aiter only takes contiguous BF16 or FP16 tensors with rows of whole 16 bytes,
quantized to the GPU's own FP8 type; other calls go to MSLK's kernel.
"""

from math import prod
from typing import Sequence, Tuple

import torch
from mslk.aiter.routing import device_key, MSLK, OpRouting

OP: str = "quantize_fp8_tensor"
AITER: str = "aiter"


# Backends are imported on first use: importing mslk.quantize must not pull in
# aiter.
def _mslk(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    from mslk.quantize.triton import fp8_quantize

    return fp8_quantize._triton_quantize_fp8_tensor(a)


def _aiter(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    from mslk.quantize.aiter import quantize_fp8_tensor as aiter_op

    return aiter_op.quantize_fp8_tensor(a)


def _aiter_accepts(a: torch.Tensor) -> bool:
    if (
        a.dtype not in (torch.bfloat16, torch.float16)
        or a.dim() == 0
        or a.numel() == 0
        or a.shape[-1] % 8 != 0
        or not a.is_contiguous()
    ):
        return False
    from mslk.utils.triton.fp8_utils import get_fp8_constants

    # aiter quantizes to the GPU's own FP8 type (OCP on gfx950, FNUZ on gfx942)
    # only, which MSLK can be told to override.
    native = (
        torch.float8_e4m3fn
        if device_key(a.get_device())[0] == "gfx950"
        else torch.float8_e4m3fnuz
    )
    return get_fp8_constants()[0] == native


def _key(
    a: torch.Tensor, *args: object, **kwargs: object
) -> Tuple[Sequence[object], int]:
    # Malformed calls get a key no table has, so MSLK's kernel handles them.
    if a.dim() == 0:
        return (), 0
    return (str(a.dtype).removeprefix("torch."), a.shape[-1]), prod(a.shape[:-1])


#: ``quantize_fp8_tensor`` on ROCm: call it with the tensor to quantize.
ROUTING: OpRouting = OpRouting(
    OP,
    "mslk.quantize.dispatch",
    "tables/quantize_fp8_tensor.csv",
    key_columns=("dtype", "K"),
    bucket_column="M_max",
    backends={MSLK: _mslk, AITER: _aiter},
    key=_key,
    accepts={AITER: _aiter_accepts},
)
BACKENDS: Tuple[str, ...] = tuple(ROUTING.backends)
