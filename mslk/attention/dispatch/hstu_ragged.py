# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which backend serves ragged HSTU self-attention (``mslk::hstu_ragged_fwd``
and ``mslk::hstu_ragged_bwd``) on ROCm.

- ``mslk``: the pure-PyTorch reference for now. It serves calls the aiter
  backends decline and empty calls (``N == 0``), and the shapes the table gives
  it.
- ``aiter_triton``, ``aiter_flydsl``: aiter's HSTU kernels, for the shapes in
  ``tables/hstu_ragged.csv`` (written by ``bench/aiter/tune_hstu_ragged.py``).
  ``aiter_flydsl`` also serves calls the table does not cover (``aiter_triton``
  misses the tolerance on long full-causal sequences).

Routings are called as ``ROUTING(q, **op_arguments_except_q)``: ``q`` comes
first because :class:`~mslk.aiter.routing.OpRouting` takes the device from the
first argument, and every other argument is passed by its schema name.

Table key: head count, head dims, mask type, window (``max_attn_len``, 0 = no
window), softmax (1) or SiLU (0), shared K/V; bucket: ``max_seq_len`` (``N``).
Window values are matched exactly; windows the table does not list go to the
default backend, like any other call the table does not cover.
"""

from typing import Any, List, Optional, Sequence, Tuple

import torch
from mslk.aiter.routing import device_key, MSLK, OpRouting

OP: str = "hstu_ragged"
AITER_TRITON: str = "aiter_triton"
AITER_FLYDSL: str = "aiter_flydsl"
KEY_COLUMNS: Tuple[str, ...] = (
    "H",
    "D",
    "DV",
    "mask",
    "max_attn_len",
    "softmax",
    "shared_kv",
)
BUCKET_COLUMN: str = "max_seq_len"
_TABLE = ("mslk.attention.dispatch", "tables/hstu_ragged.csv")


# Backends are imported on first use.
def _mslk_fwd(q: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    from mslk.attention.hstu import _reference_backend

    return _reference_backend.ragged_fwd(q, **kwargs)


def _mslk_bwd(q: torch.Tensor, **kwargs: Any) -> None:
    from mslk.attention.hstu import _reference_backend

    _reference_backend.ragged_bwd(q, **kwargs)


def _aiter_triton_fwd(
    q: torch.Tensor, **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    from mslk.attention.aiter import hstu

    return hstu.ragged_fwd_triton(q, **kwargs)


def _aiter_triton_bwd(q: torch.Tensor, **kwargs: Any) -> None:
    from mslk.attention.aiter import hstu

    hstu.ragged_bwd_triton(q, **kwargs)


def _aiter_flydsl_fwd(
    q: torch.Tensor, **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    from mslk.attention.aiter import hstu

    return hstu.ragged_fwd_flydsl(q, **kwargs)


def _aiter_flydsl_bwd(q: torch.Tensor, **kwargs: Any) -> None:
    from mslk.attention.aiter import hstu

    hstu.ragged_bwd_flydsl(q, **kwargs)


_AITER_ARCHS: Tuple[str, ...] = ("gfx942", "gfx950")


def aiter_accepts(
    q: torch.Tensor,
    *,
    v: torch.Tensor,
    invalid_attn_mask_type: str,
    attn_scale: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    full_attn_size: int,
    num_softmax_heads: int,
    **_: Any,
) -> bool:
    """aiter computes SiLU heads with the ``1 / N`` scale and the
    ``lower_triangular`` mask, without ``full_attn_size``, in bf16/fp16 on
    gfx942/gfx950. It also applies the window to the contextual prefix, which
    the reference does not, so a window with a contextual prefix is declined."""
    return (
        num_softmax_heads == 0
        and attn_scale is None
        and full_attn_size == 0
        and invalid_attn_mask_type == "lower_triangular"
        and not (max_attn_len > 0 and contextual_seq_len > 0)
        and q.dtype in (torch.bfloat16, torch.float16)
        and device_key(q.get_device())[0] in _AITER_ARCHS
    )


def _is_pow2(n: int) -> bool:
    return n >= 16 and (n & (n - 1)) == 0


def aiter_triton_accepts(q: torch.Tensor, *, v: torch.Tensor, **kwargs: Any) -> bool:
    """Also needs power-of-two head dims (>= 16)."""
    return (
        aiter_accepts(q, v=v, **kwargs)
        and _is_pow2(q.shape[2])
        and _is_pow2(v.shape[2])
    )


def aiter_flydsl_accepts(q: torch.Tensor, *, v: torch.Tensor, **kwargs: Any) -> bool:
    """Also needs head dims that are multiples of 32, up to 256."""
    return aiter_accepts(q, v=v, **kwargs) and all(
        d % 32 == 0 and 0 < d <= 256 for d in (q.shape[2], v.shape[2])
    )


def key(
    q: torch.Tensor,
    *,
    v: torch.Tensor,
    N: int,
    invalid_attn_mask_type: str,
    max_attn_len: int,
    num_softmax_heads: int,
    shared_kv: bool,
    **_: Any,
) -> Tuple[Sequence[object], int]:
    """Table key and bucket. Shared by forward and backward, so both directions
    of a call route to the same backend (the backward consumes the forward's
    ``M``)."""
    key_values: List[object] = [
        q.shape[1],
        q.shape[2],
        v.shape[2],
        invalid_attn_mask_type,
        max_attn_len,
        int(num_softmax_heads > 0),
        int(shared_kv),
    ]
    return key_values, N


_ACCEPTS = {AITER_TRITON: aiter_triton_accepts, AITER_FLYDSL: aiter_flydsl_accepts}

FWD_ROUTING: OpRouting = OpRouting(
    OP,
    *_TABLE,
    key_columns=KEY_COLUMNS,
    bucket_column=BUCKET_COLUMN,
    backends={
        MSLK: _mslk_fwd,
        AITER_TRITON: _aiter_triton_fwd,
        AITER_FLYDSL: _aiter_flydsl_fwd,
    },
    key=key,
    accepts=_ACCEPTS,
    default=AITER_FLYDSL,
)

BWD_ROUTING: OpRouting = OpRouting(
    OP,
    *_TABLE,
    key_columns=KEY_COLUMNS,
    bucket_column=BUCKET_COLUMN,
    backends={
        MSLK: _mslk_bwd,
        AITER_TRITON: _aiter_triton_bwd,
        AITER_FLYDSL: _aiter_flydsl_bwd,
    },
    key=key,
    accepts=_ACCEPTS,
    default=AITER_FLYDSL,
)

BACKENDS: Tuple[str, ...] = tuple(FWD_ROUTING.backends)
