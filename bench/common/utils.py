# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import contextlib
import copy
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import click
import numpy as np
import torch
import triton  # @manual=//triton:triton
from torch.profiler import profile, ProfilerActivity  # pyre-ignore


@dataclass
class BenchOptions:
    """Common benchmark options used across all benchmark scripts.

    This dataclass encapsulates all configuration options for running benchmarks
    in the MSLK benchmark suite. It is used by gemm_bench, conv_bench, and
    quantize_bench to maintain consistent benchmarking behavior.

    Attributes:
        cuda_graph: Whether to use CUDA graphs for benchmarking. CUDA graphs
            reduce kernel launch overhead and provide more accurate measurements
            for GPU-bound workloads.
        rotating_buffer: Whether to use a rotating buffer during benchmarking.
            Each call reads a different copy of the inputs, cycling through
            enough copies to sweep twice the device's last-level cache, so
            every call reads its inputs from HBM.
        rep_ms: Repetition time in milliseconds for triton.testing.do_bench.
            Controls how long each benchmark runs before measuring.
        trace: Whether to produce a performance trace of the benchmark using
            PyTorch profiler. Traces are saved to Manifold or temp directory.
        fast_accum: Whether to enable fast accumulation for FP8 implementations.
            This is only relevant for Hopper GPUs.
        torch_compile: Whether to use torch.compile for applicable operations.
    """

    cuda_graph: bool = True
    rotating_buffer: bool = False
    rep_ms: int = 200
    trace: bool = False
    fast_accum: bool = False
    torch_compile: bool = False


def _do_bench(
    fn: Callable[[], Any],
    opts: BenchOptions,
) -> float:
    if opts.cuda_graph:
        with torch.cuda.stream(torch.cuda.Stream()):
            return triton.testing.do_bench_cudagraph(fn, rep=opts.rep_ms)
    else:
        return triton.testing.do_bench(fn, rep=opts.rep_ms)


# Last-level cache per ROCm arch. torch reports only the per-XCD L2 there, not
# the Infinity Cache that sits in front of HBM.
_ROCM_LAST_LEVEL_CACHE_BYTES = {
    "gfx942": 256 * 1024 * 1024,
    "gfx950": 256 * 1024 * 1024,
}

# A cyclic sweep over exactly the cache's capacity still leaves hits, since the
# replacement policy is not strict LRU; twice the capacity evicts each set
# before it comes around again.
_ROTATING_BUFFER_CACHE_MULTIPLE = 2

# Bounds the graph a tiny problem would otherwise need: below this many sets,
# small problems fall short of the target and run partly warm.
_MAX_ROTATING_BUFFER_SETS = 10000


# Peak HBM bandwidth per ROCm arch, from AMD's data sheets (MI300X, MI350X/MI355X).
# triton.testing.get_dram_gbps() derives it as memory clock x bus width x 2,
# which gives half the HBM3E figure on gfx950 (3.9 TB/s against 8 TB/s).
_ROCM_HBM_BANDWIDTH_GBPS = {
    "gfx942": 5300.0,
    "gfx950": 8000.0,
}


def hbm_bandwidth_gbps() -> float:
    """Peak HBM bandwidth of the current device in GB/s."""
    if torch.version.hip is not None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = props.gcnArchName.split(":")[0]
        if arch in _ROCM_HBM_BANDWIDTH_GBPS:
            return _ROCM_HBM_BANDWIDTH_GBPS[arch]
    return triton.testing.get_dram_gbps()


def last_level_cache_bytes() -> int:
    """Size of the cache closest to HBM on the current device."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if torch.version.hip is not None:
        arch = props.gcnArchName.split(":")[0]
        if arch in _ROCM_LAST_LEVEL_CACHE_BYTES:
            return _ROCM_LAST_LEVEL_CACHE_BYTES[arch]
    return props.L2_cache_size


def rotating_buffer_target_bytes() -> int:
    """Bytes the rotating buffer sweeps per pass, so every pass reads HBM.

    ``MSLK_BENCH_ROTATING_BUFFER_MB`` overrides the default of twice the
    device's last-level cache.
    """
    override_mb = os.environ.get("MSLK_BENCH_ROTATING_BUFFER_MB")
    if override_mb:
        return int(override_mb) * 1024 * 1024
    return _ROTATING_BUFFER_CACHE_MULTIPLE * last_level_cache_bytes()


def _tensor_bytes(obj: Any, seen: set[int]) -> int:
    """Bytes of the distinct tensor storages in obj, searching lists, tuples
    and dicts, so grouped ops' per-group tensors and in-place outputs count
    once each."""
    if isinstance(obj, torch.Tensor):
        storage = obj.untyped_storage()
        if storage.data_ptr() in seen:
            return 0
        seen.add(storage.data_ptr())
        return storage.nbytes()
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(o, seen) for o in obj)
    if isinstance(obj, dict):
        return sum(_tensor_bytes(o, seen) for o in obj.values())
    return 0


def rotating_buffer_sets(footprint_bytes: int) -> int:
    """Number of input sets to cycle through for a call touching
    footprint_bytes of memory."""
    if footprint_bytes <= 0:
        return 1
    sets = -(-rotating_buffer_target_bytes() // footprint_bytes)
    return min(max(sets, 2), _MAX_ROTATING_BUFFER_SETS)


def do_bench(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    opts: BenchOptions,
) -> float:
    """
    Benchmark a function using triton's benchmarking utilities.

    Args:
        fn: The function to benchmark.
        args: Tuple of arguments to pass to the function.
        opts: Benchmark options

    Returns:
        The runtime in milliseconds.
    """
    if not opts.rotating_buffer:
        return _do_bench(lambda: fn(*args), opts)

    # A call touches its inputs and writes its outputs; both pass through the
    # cache, so both count towards the sweep.
    seen: set[int] = set()
    input_bytes = _tensor_bytes(args, seen)
    if input_bytes == 0:
        return _do_bench(lambda: fn(*args), opts)
    footprint_bytes = input_bytes + _tensor_bytes(fn(*args), seen)

    args_list = [args]
    for _ in range(rotating_buffer_sets(footprint_bytes) - 1):
        args_list.append(copy.deepcopy(args))

    # We benchmark on a different stream, so a sync is required.
    torch.cuda.synchronize()

    def rotating_buffer_fn() -> list[Any]:
        return [fn(*args) for args in args_list]

    return _do_bench(rotating_buffer_fn, opts) / len(args_list)


def profiler(
    enabled: bool,
    with_stack: bool = False,
    record_shapes: bool = False,
):
    """
    Returns a profiler context manager if enabled, otherwise a null context.

    When enabled, profiles CPU and CUDA activities.

    Args:
        enabled: Whether to enable profiling.
        with_stack: Whether to record stack traces.
        record_shapes: Whether to record tensor shapes.

    Returns:
        A context manager - either a torch profiler or nullcontext.
    """

    def _kineto_trace_handler(p: torch.profiler.profile) -> None:
        trace_filename = f"mslk_{os.getpid()}_{uuid.uuid4().hex}.json"

        if os.path.exists("/etc/fbwhoami"):
            trace_url = f"manifold://gpu_traces/tree/accelerator/{trace_filename}"
        else:
            trace_url = os.path.join(tempfile.gettempdir(), trace_filename)

        p.export_chrome_trace(trace_url)

    return (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # pyre-ignore
            on_trace_ready=_kineto_trace_handler,
            with_stack=with_stack,
            record_shapes=record_shapes,
        )
        if enabled
        else contextlib.nullcontext()
    )


@dataclass
class BenchStatResult:
    """Statistical summary of N benchmark iterations."""

    median_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    cov_pct: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    n: int = 0
    raw_ms: list[float] = field(default_factory=list)


def do_bench_statistical(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    opts: BenchOptions,
    n_iterations: int = 5,
) -> BenchStatResult:
    """Run the benchmark n_iterations times and return statistical summary.

    Each iteration uses triton's do_bench internally (which itself runs for
    rep_ms). The distribution across iterations captures run-to-run variance.
    """
    results = [do_bench(fn, args, opts) for _ in range(n_iterations)]
    return summarize_bench_results(results)


def summarize_bench_results(results: list[float]) -> BenchStatResult:
    """Return summary statistics for repeated benchmark measurements."""
    arr = np.array(results)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return BenchStatResult(
        median_ms=float(np.median(arr)),
        mean_ms=mean,
        std_ms=std,
        cov_pct=(std / mean * 100) if mean > 0 else 0.0,
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        n=len(results),
        raw_ms=results,
    )


def common_bench_options(shape_registry):
    """Apply the common Click options shared across bench scripts.

    Args:
        shape_registry: Dict mapping shape name to shape function.
    """

    def decorator(func):
        options = [
            click.option(
                "--output-dir",
                default="/tmp",
                help="Directory to save plots and csvs to",
            ),
            click.option(
                "--export-csv",
                is_flag=True,
                help="Export results to a CSV file.",
            ),
            click.option(
                "--kernels",
                default=None,
                help=(
                    "Comma separated list of kernels to benchmark."
                    " Defaults to all kernels."
                ),
            ),
            click.option(
                "--cuda-graph/--no-cuda-graph",
                default=True,
                help="Use cuda graph for benchmarking.",
            ),
            click.option(
                "--rotating-buffer/--no-rotating-buffer",
                default=True,
                help="Use rotating buffer for benchmarking.",
            ),
            click.option(
                "--shapes",
                default=None,
                help=(
                    "Specific model shapes to use, options: "
                    f"{', '.join(shape_registry.keys())}."
                ),
            ),
            click.option(
                "--trace",
                is_flag=True,
                help="If set, produce a performance trace of the benchmark.",
            ),
            click.option(
                "--rep-ms",
                default=200,
                type=int,
                help="Repetition time in ms for triton.testing.do_bench.",
            ),
        ]
        for option in reversed(options):
            func = option(func)
        return func

    return decorator
