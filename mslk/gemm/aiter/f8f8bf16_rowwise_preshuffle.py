# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""aiter backend of ``mslk::f8f8bf16_rowwise_preshuffle`` (FP8 x FP8 -> BF16,
rowwise scales, preshuffled weight) on ROCm.

aiter's ``gemm_a8w8_bpreshuffle`` computes what the op computes. Its weight
layout, ``shuffle_weight(w, layout=(16, 16))``, is byte-identical to the op's
(``ck_preshuffle(w, 16)``, ``preshuffle_b_mfma(w)``), and it reads the FP32 row
and column scales as flat arrays, so the op's arguments go to it unchanged;
only XQ's leading dimensions are flattened, which is a view. aiter picks one of
its CK, CK-tile and FlyDSL kernels per shape from its own tuned configs.

``mslk::f8f8f16_rowwise_preshuffle`` runs the same kernels with FP16 output
(``mslk/gemm/aiter/f8f8f16_rowwise_preshuffle.py``), through
:func:`rowwise_preshuffle`.

Which calls this backend serves is decided in
``mslk/gemm/dispatch/f8f8bf16_rowwise_preshuffle.py``, which leaves out the
calls :func:`has_reliable_kernel` rejects.
"""

import torch
from mslk.aiter.common import import_aiter_module

_GEMM_MODULE = "aiter.ops.gemm_op_a8w8"


def has_reliable_kernel(M: int, N: int, K: int) -> bool:
    """False when aiter would compute an ``[M, K] x [N, K]`` call with one of
    its 8-wave FlyDSL kernels (``flydsl_bpreshuffle_8w_*``), which its configs
    pick for some large shapes.

    In the pinned aiter, those kernels return some wrong elements on some runs
    (on gfx950, 33 of 495 runs over the 33 tuned shapes they serve, with errors
    of the order of the outputs), and reject FP16 output. A single check while
    tuning cannot rule that out, so such calls are left to MSLK's kernel.
    """
    gemm = import_aiter_module(_GEMM_MODULE)
    config = gemm.get_GEMM_config_with_quant_type(
        M,
        N,
        K,
        gemm.dtypes.fp8,
        gemm.AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
    )
    return config is None or not str(config["kernelName"]).startswith(
        "flydsl_bpreshuffle_8w_"
    )


def rowwise_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``XQ @ WQ.T`` scaled by ``x_scale`` per row and ``w_scale`` per column,
    in ``dtype`` (BF16 or FP16), computed by aiter's ``gemm_a8w8_bpreshuffle``.

    XQ is contiguous ``[..., K]``, WQ is ``[N, K]`` in the preshuffle layout,
    and the scales are FP32 with one value per row of XQ and per row of WQ.
    """
    gemm_a8w8_bpreshuffle = import_aiter_module(_GEMM_MODULE).gemm_a8w8_bpreshuffle
    K = XQ.shape[-1]
    N = WQ.shape[0]
    out = gemm_a8w8_bpreshuffle(
        XQ.view(-1, K), WQ, x_scale.reshape(-1), w_scale.reshape(-1), dtype=dtype
    )
    return out.view(*XQ.shape[:-1], N)


def f8f8bf16_rowwise_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """``mslk::f8f8bf16_rowwise_preshuffle`` computed by aiter's
    ``gemm_a8w8_bpreshuffle``, without bias and with fast accumulation, the
    only modes MSLK's own ROCm kernels support."""
    return rowwise_preshuffle(XQ, WQ, x_scale, w_scale, torch.bfloat16)
