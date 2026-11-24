# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""aiter backend of ``quantize_fp8_tensor`` (one FP8 scale for a whole tensor)
on ROCm.

aiter's ``per_tensor_quant_hip`` computes what MSLK's Triton kernel computes:
the scale ``amax / FP8_MAX`` for the whole tensor, and the input times its
reciprocal, saturated to FP8. On random inputs the two return the same scale
and the same FP8 values, except for a rare value one FP8 step apart (aiter's
reciprocal is approximate; MSLK's scale is sometimes one FP32 ulp off). The
op's argument goes to it unchanged, and its ``[1]`` scale is returned as the
op's 0-dim one, a view.

It differs where the input's amax is below MSLK's epsilon (1e-12), which MSLK
clamps the amax to: an all-zero tensor gets scale 0 and every element -FP8_MAX
instead of scale eps / FP8_MAX and zeros. The dequantized tensor, scale times
values, is the same (zero), and for amax below epsilon aiter's is closer to the
input than MSLK's, which flushes it to zero.

Which calls this backend serves is decided in
``mslk/quantize/dispatch/quantize_fp8_tensor.py``.
"""

from typing import Tuple

import torch
from mslk.aiter.common import import_aiter_module
from mslk.utils.triton.fp8_utils import get_fp8_constants


def quantize_fp8_tensor(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``quantize_fp8_tensor`` computed by aiter's ``per_tensor_quant_hip``:
    the FP8 tensor, of ``a``'s shape, and the scalar FP32 scale that
    dequantizes it.

    ``a`` is a contiguous BF16 or FP16 tensor whose rows (its last dimension)
    are a multiple of 16 bytes; aiter computes other rows wrong.
    """
    per_tensor_quant_hip = import_aiter_module("aiter.ops.quant").per_tensor_quant_hip
    a_fp8, scale = per_tensor_quant_hip(a, quant_dtype=get_fp8_constants()[0])
    return a_fp8, scale.view(())
