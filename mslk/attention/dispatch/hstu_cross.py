# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which backend serves HSTU cross attention (``mslk::hstu_cross_fwd`` and
``mslk::hstu_cross_bwd``) on ROCm.

- ``mslk``: the default and fallback; the pure-PyTorch reference for now.
- ``aiter``: aiter's varlen flash attention, for the shapes in
  ``tables/hstu_cross.csv`` (written by ``bench/aiter/tune_hstu_cross.py``) and
  for calls the table does not cover. Every KV sequence must fit in
  ``max_seq_len``.

Routings are called as ``ROUTING(q, **op_arguments_except_q)``: ``q`` comes
first because :class:`~mslk.aiter.routing.OpRouting` takes the device from the
first argument, and every other argument is passed by its schema name.

Backward backends return ``[dq, dk]`` when ``shared_kv`` (``dk`` holds dK+dV)
and ``[dq, dk, dv]`` otherwise.

Table key: head count, head dims, causal, softmax (1) or SiLU (0), shared K/V,
and the query length rounded up to a power of two (``max_q_len_p2``); bucket:
``max_kv_len`` (``max_seq_len``).
"""

from typing import Any, List, Optional, Sequence, Tuple

import torch
from mslk.aiter.routing import MSLK, OpRouting

OP: str = "hstu_cross"
AITER: str = "aiter"
KEY_COLUMNS: Tuple[str, ...] = (
    "H",
    "D",
    "DV",
    "causal",
    "softmax",
    "shared_kv",
    "max_q_len_p2",
)
BUCKET_COLUMN: str = "max_kv_len"
_TABLE = ("mslk.attention.dispatch", "tables/hstu_cross.csv")


# Backends are imported on first use.
def _mslk_fwd(q: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    from mslk.attention.hstu import _reference_backend

    return _reference_backend.cross_fwd(q, **kwargs)


def _mslk_bwd(q: torch.Tensor, **kwargs: Any) -> List[torch.Tensor]:
    from mslk.attention.hstu import _reference_backend

    return _reference_backend.cross_bwd(q, **kwargs)


def _aiter_fwd(q: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    from mslk.attention.aiter import hstu

    return hstu.cross_fwd(q, **kwargs)


def _aiter_bwd(q: torch.Tensor, **kwargs: Any) -> List[torch.Tensor]:
    from mslk.attention.aiter import hstu

    return hstu.cross_bwd(q, **kwargs)


def aiter_accepts(
    q: torch.Tensor,
    *,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    G: int,
    num_softmax_heads: int,
    truncate_method: str,
    num_targets: Optional[torch.Tensor],
    causal: bool,
    **kwargs: Any,
) -> bool:
    """aiter computes softmax heads only, without truncation, causal targets or
    GQA, with equal Q/K and V head dims (multiple of 8, <= 256; <= 128 when
    causal: its causal kernels at D=256 get a few outputs wrong). Backward calls
    also need ``M``, ``Out``, ``max_q_len`` and ``seq_offsets_q``."""
    value = k if v is None else v
    head_dim = q.shape[2]
    accepted = (
        num_softmax_heads == q.shape[1]
        and truncate_method == "none"
        and not (causal and num_targets is not None)
        and G == 1
        and q.dtype in (torch.bfloat16, torch.float16)
        and value.shape[2] == head_dim
        and head_dim % 8 == 0
        and head_dim <= (128 if causal else 256)
    )
    if "dout" in kwargs:
        accepted = accepted and all(
            kwargs.get(name) is not None
            for name in ("M", "Out", "max_q_len", "seq_offsets_q")
        )
    return accepted


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def key(
    q: torch.Tensor,
    *,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    max_seq_len: int,
    max_q_len: Optional[int],
    causal: bool,
    num_softmax_heads: int,
    shared_kv: bool,
    **_: Any,
) -> Tuple[Sequence[object], int]:
    """Table key and bucket. Shared by forward and backward, so both directions
    of a call route to the same backend (the backward consumes the forward's
    ``M``)."""
    value = k if v is None else v
    key_values: List[object] = [
        q.shape[1],
        q.shape[2],
        value.shape[2],
        int(causal),
        int(num_softmax_heads > 0),
        int(shared_kv),
        # The backward's max_q_len is optional; without it the call uses MSLK.
        _next_pow2(max_q_len) if max_q_len is not None else 0,
    ]
    return key_values, max_seq_len


FWD_ROUTING: OpRouting = OpRouting(
    OP,
    *_TABLE,
    key_columns=KEY_COLUMNS,
    bucket_column=BUCKET_COLUMN,
    backends={MSLK: _mslk_fwd, AITER: _aiter_fwd},
    key=key,
    accepts={AITER: aiter_accepts},
    default=AITER,
)

BWD_ROUTING: OpRouting = OpRouting(
    OP,
    *_TABLE,
    key_columns=KEY_COLUMNS,
    bucket_column=BUCKET_COLUMN,
    backends={MSLK: _mslk_bwd, AITER: _aiter_bwd},
    key=key,
    accepts={AITER: aiter_accepts},
    default=AITER,
)

BACKENDS: Tuple[str, ...] = tuple(FWD_ROUTING.backends)
