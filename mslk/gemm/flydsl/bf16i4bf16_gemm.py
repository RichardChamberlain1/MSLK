# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BF16 x INT4 rowwise GEMM via FlyDSL.

* ``mslk::bf16i4bf16_rowwise`` and ``mslk::bf16i4bf16_shuffled`` -- on ROCm
  ``preshuffle_i4`` is the identity, so both ops receive the same weights and
  are served by the same kernel.

The kernel reads the weights in the layout of ``bf16i4bf16_layout``, built from
``W`` on the first call with a given weight tensor and cached for as long as
that tensor lives.

Tensor contract:
  X       : [..., K]        BF16
  W       : [N, K / 2]      int8  -- two INT4 per byte, k even in the low nibble
  w_scale : [K / g, N]      BF16 or FP32
  w_zero  : [K / g, N]      BF16 or FP32, same dtype as w_scale
  Output  : [..., N]        BF16

  out[m, n] = sum_k X[m, k] * (q[n, k] * w_scale[k // g, n] + w_zero[k // g, n])
  with q = (nibble XOR 8) - 8
"""

import weakref

import torch
from mslk.gemm.flydsl.bf16i4bf16_layout import reorder_int4_for_mfma

# Keyed by the weight tensor itself, not its address: a freed weight's address
# is handed to the next allocation of the same size, which would otherwise be
# served the old weight's copy. The version counter catches in-place updates,
# and an entry is dropped once its weight is collected so the cache does not pin
# the copies. Same policy as the FP8 preshuffle cache (preshuffle_gemm.py),
# including under graph capture: a miss there records the reorder into the graph
# and caches its result.
_reordered_cache: dict[int, tuple[weakref.ref, int, torch.Size, torch.Tensor]] = {}


def _evict(ref: weakref.ref, key: int) -> None:
    entry = _reordered_cache.get(key)
    if entry is not None and entry[0] is ref:
        del _reordered_cache[key]


def reordered_weight(W: torch.Tensor) -> torch.Tensor:
    """Return the reordered copy of ``W``, building and caching it on a miss."""
    key = id(W)
    entry = _reordered_cache.get(key)
    if entry is not None:
        ref, version, shape, reordered = entry
        if ref() is W and version == W._version and shape == W.shape:
            return reordered
    reordered = reorder_int4_for_mfma(W)
    ref = weakref.ref(W, lambda r, k=key: _evict(r, k))
    _reordered_cache[key] = (ref, W._version, W.shape, reordered)
    return reordered
