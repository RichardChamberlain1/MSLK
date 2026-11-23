# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tune which backend serves ``mslk::f8f8f16_rowwise_preshuffle`` on ROCm, per
shape.

Compares MSLK's kernel (FlyDSL on gfx950) with aiter's ``gemm_a8w8_bpreshuffle``
on every shape in ``bench/aiter/untuned/f8f8f16_rowwise_preshuffle.csv`` and
updates ``mslk/gemm/dispatch/tables/f8f8f16_rowwise_preshuffle.csv``. Inputs,
reference and checks are those of ``tune_f8f8bf16_rowwise_preshuffle.py``, with
FP16 output; one FP16 ulp is up to 2^-10, so rtol is 2e-3 (atol stays 1e-3). On
gfx950 the worst element of either kernel uses under 25% of that tolerance at
K = 256 to 28672. Needs the pinned aiter and a gfx950 GPU.

Example, on four GPUs (run from the repository root):

    HIP_VISIBLE_DEVICES=0,1,2,3 python bench/aiter/tune_f8f8f16_rowwise_preshuffle.py --mp 4
"""

import torch
from mslk.aiter import tuning
from mslk.gemm.dispatch.f8f8f16_rowwise_preshuffle import BACKENDS, OP, ROUTING

# Run as a script, this directory is on sys.path.
from tune_f8f8bf16_rowwise_preshuffle import F8F8BF16RowwisePreshuffleTuner, REPO


class F8F8F16RowwisePreshuffleTuner(F8F8BF16RowwisePreshuffleTuner):
    op = OP
    backends = BACKENDS
    rtol = 2e-3
    dtype = torch.float16
    routing = ROUTING


if __name__ == "__main__":
    tuning.main(
        F8F8F16RowwisePreshuffleTuner(),
        untune_file=str(
            REPO / "bench" / "aiter" / "untuned" / "f8f8f16_rowwise_preshuffle.csv"
        ),
        tune_file=str(
            REPO
            / "mslk"
            / "gemm"
            / "dispatch"
            / "tables"
            / "f8f8f16_rowwise_preshuffle.csv"
        ),
    )
