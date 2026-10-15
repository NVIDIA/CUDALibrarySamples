# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Requirements:
#   numpy
#   cupy-cuda12x (or cupy-cuda13x, matching your CUDA toolkit)
#   nvidia.nvcomp

import argparse
import csv
import os
import statistics
import sys
import math
from typing import Any, Callable
from functools import cache

import cupy as cp
import numpy as np
import numpy.typing as npt

import nvidia.nvcomp as nvcomp

HostArray = npt.NDArray[np.uint8]
StagingBuffer = tuple[int, HostArray]  # (stride_bytes, padded_array)

DEFAULT_HLIF_CHUNK_SIZE = 65536

# nvcomp accepts this typestr for FP8 E4M3 (1 byte) for ANS, numpy has no matching dtype.
FLOAT8_E4M3_TYPESTR = "|f1"

ALGORITHMS = (
    "ANS",
    "Bitcomp",
    "Cascaded",
    "Deflate",
    "GDeflate",
    "Gzip",
    "LZ4",
    "Snappy",
    "Zstd",
)


def canonical_algorithm(value: str) -> str:
    return {name.lower(): name for name in ALGORITHMS}.get(value.lower(), value)


def parse_codec_opts(pairs: list[str]) -> dict[str, Any]:
    """Parse `key=value` codec options, coercing true/false to bool and numeric
    strings to int/float; everything else (e.g. a `<i4` typestr) stays a string."""

    def coerce(value: str) -> Any:
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    opts: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            sys.exit(f"error: --codec-opts entry '{pair}' must be key=value")
        opts[key] = coerce(value)
    return opts


# Codec options whose value must be an nvcomp enum instance; accept a member name
# (e.g. checksum_policy=NO_COMPUTE_NO_VERIFY) or its integer value.
ENUM_CODEC_OPTS = {
    "bitshuffle_mode": nvcomp.BitshuffleMode,
    "checksum_policy": nvcomp.ChecksumPolicy,
    "decompress_backend": nvcomp.DecompressBackend,
    "execution_policy": nvcomp.ExecutionPolicy,
}


def resolve_enum_opts(opts: dict[str, Any]) -> None:
    """Convert enum-typed codec options (given as a member name or int) into the
    nvcomp enum instances the Codec requires."""
    for key, enum_cls in ENUM_CODEC_OPTS.items():
        if key not in opts:
            continue
        value = opts[key]
        try:
            opts[key] = (
                getattr(enum_cls, value) if isinstance(value, str) else enum_cls(value)
            )
        except (AttributeError, TypeError, ValueError):
            members = ", ".join(m for m in dir(enum_cls) if m.isupper())
            sys.exit(
                f"error: --codec-opts {key}={value} is not a valid "
                f"{enum_cls.__name__} (choices: {members})"
            )


def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


# Conservative upper bound on every nvCOMP algorithm's required buffer
# alignment (ANS is the strictest, at 16 bytes)
MAX_CODEC_ALIGNMENT = 16


def align_up_to_max_codec_alignment(x: int) -> int:
    return ceil_div(x, MAX_CODEC_ALIGNMENT) * MAX_CODEC_ALIGNMENT


class _L2Handle:
    def __init__(self) -> None:
        # Calls are guarded by `with stream:` so the cached handle will be queried with device selected by the user.
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.get_device_id())
        l2_cache_size = props["l2CacheSize"]

        self.l2_buffer = cp.cuda.Memory(l2_cache_size)

    def flush(self, stream: cp.cuda.Stream):
        if self.l2_buffer.size > 0:
            cp.cuda.runtime.memsetAsync(
                self.l2_buffer.ptr, 0, self.l2_buffer.size, stream.ptr
            )
        stream.synchronize()


@cache
def l2_handle():
    return _L2Handle()


def benchmark_operation(
    call: Callable[[], Any],
    seconds: float,
    stream: cp.cuda.Stream,
    min_samples: int,
) -> list[float]:
    # Warmup
    result = call()
    stream.synchronize()

    start, end = cp.cuda.Event(), cp.cuda.Event()
    samples: list[float] = []

    # Sample until the measured-time budget is spent, but always collect at least
    # min_samples so slow operations still get a stable estimate.
    total_time = 0.0
    while total_time < seconds or len(samples) < min_samples:
        l2_handle().flush(stream)
        start.record(stream)
        result = call()
        end.record(stream)

        stream.synchronize()

        sample_time_s = cp.cuda.get_elapsed_time(start, end) / 1000.0
        samples.append(sample_time_s)
        total_time += sample_time_s

    return samples


def throughput_gbps(samples: list[float], nbytes: int) -> float:
    mean_s = statistics.mean(samples)
    if mean_s <= 0:
        return float("nan")

    return nbytes / mean_s / 1e9


CSV_HEADER = (
    "dataset,algorithm,bitstream_kind,chunk_size,uncompressed_bytes,"
    "compressed_bytes,"
    "compression_ratio,compression_gbps,decompression_gbps,codec_opts"
)


def write_csv_row(
    path: str,
    args: argparse.Namespace,
    uncomp_bytes: int,
    comp_bytes: int,
    enc: list[float],
    dec: list[float],
) -> None:
    fields = [
        os.path.basename(args.input_file),
        args.algorithm.lower(),
        args.bitstream_kind,
        args.chunk_size,
        uncomp_bytes,
        comp_bytes,
        f"{uncomp_bytes / comp_bytes:.4f}",
        f"{throughput_gbps(enc, uncomp_bytes):.4f}",
        f"{throughput_gbps(dec, uncomp_bytes):.4f}",
        args.codec_opts_str,
    ]
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(CSV_HEADER.split(","))
        writer.writerow(fields)


def print_report(
    args: argparse.Namespace,
    num_chunks: int,
    uncomp_bytes: int,
    comp_bytes: int,
    enc: list[float],
    dec: list[float],
) -> None:
    if args.csv:
        write_csv_row(args.csv, args, uncomp_bytes, comp_bytes, enc, dec)
    else:
        print("----------")
        print(f"algorithm: {args.algorithm}")
        print(f"file: {args.input_file}")
        print(f"bitstream kind: {args.bitstream_kind}")
        print(f"codec options: {args.codec_opts_str or '(defaults)'}")
        if args.bitstream_kind == "NVCOMP_NATIVE":
            print("input arrays: 1")
            print(f"HLIF internal chunk size (B): {args.chunk_size}")
        else:
            print(f"input arrays: {num_chunks}")
            print(f"input chunk size (B): {args.chunk_size}")
        print(f"samples: {len(enc)} enc / {len(dec)} dec ")
        print(f"uncompressed (B): {uncomp_bytes}")
        print(
            f"comp_size: {comp_bytes}, compressed ratio: {uncomp_bytes / comp_bytes:.4f}"
        )
        print(
            f"compression throughput (GB/s): {throughput_gbps(enc, uncomp_bytes):.4f}"
        )
        print(
            f"decompression throughput (GB/s): {throughput_gbps(dec, uncomp_bytes):.4f}"
        )


def stage_chunked_input(
    host_input: HostArray,
    chunk_size: int,
    duplicate_data: int,
    stream: cp.cuda.Stream,
) -> tuple[nvcomp.BatchArray, list[int], StagingBuffer]:
    # Replicate contiguously, then ceil-split into chunks.
    scale = max(duplicate_data, 1)
    full = np.tile(host_input, scale) if scale > 1 else host_input
    total = full.size

    num_chunks = ceil_div(total, chunk_size)
    sizes = [min(chunk_size, total - i * chunk_size) for i in range(num_chunks)]

    stride = chunk_size
    host_padded = np.zeros(num_chunks * stride, dtype=np.uint8)
    for i, size in enumerate(sizes):
        host_padded[i * stride : i * stride + size] = full[
            i * chunk_size : i * chunk_size + size
        ]

    with stream:
        d_in = cp.empty(host_padded.size, dtype=cp.uint8)
        d_in.set(host_padded)

    in_batch = nvcomp.as_arrays(d_in, [i * stride for i in range(num_chunks)], sizes)
    return in_batch, sizes, (stride, host_padded)


def run_chunked(
    codec: nvcomp.Codec,
    host_input: HostArray,
    chunk_size: int,
    stream: cp.cuda.Stream,
    args: argparse.Namespace,
) -> None:
    scale = max(args.duplicate_data, 1)
    n = host_input.size * scale
    in_batch, sizes, (stride, host_padded) = stage_chunked_input(
        host_input, chunk_size, args.duplicate_data, stream
    )
    num_chunks = len(sizes)

    comp_cfg = codec.compression_config(sizes)
    decomp_cfg = codec.decompression_config(comp_cfg)

    max_comp_size = codec.get_max_comp_buffer_size(in_batch[0])
    comp_stride = align_up_to_max_codec_alignment(max_comp_size)
    with stream:
        comp_slab = cp.zeros(num_chunks * comp_stride, dtype=cp.uint8)
        out_slab = cp.zeros(num_chunks * stride, dtype=cp.uint8)

    comp_offsets = [i * comp_stride for i in range(num_chunks)]
    out_comps = nvcomp.as_arrays(
        comp_slab,
        comp_offsets,
        [max_comp_size] * num_chunks,
    )

    out_bufs = nvcomp.as_arrays(
        out_slab, [i * stride for i in range(num_chunks)], sizes
    )

    def enc_call() -> nvcomp.BatchArray:
        return codec.encode(in_batch, compression_config=comp_cfg, out=out_comps)

    enc_samples = benchmark_operation(enc_call, args.seconds, stream, args.min_samples)

    d_comps = enc_call()
    comp_sizes = d_comps.buffer_sizes
    comp_bytes = np.sum(comp_sizes)

    dec_ins = nvcomp.as_arrays(comp_slab, comp_offsets, comp_sizes)

    def dec_call() -> nvcomp.BatchArray:
        return codec.decode(dec_ins, decompression_config=decomp_cfg, out=out_bufs)

    dec_samples = benchmark_operation(dec_call, args.seconds, stream, args.min_samples)

    decoded = cp.asnumpy(out_slab)
    for i, size in enumerate(sizes):
        begin = i * stride
        if not np.array_equal(
            decoded[begin : begin + size], host_padded[begin : begin + size]
        ):
            sys.exit(f"error: roundtrip verification failed for chunk {i}")

    print_report(args, num_chunks, n, comp_bytes, enc_samples, dec_samples)


def run_single(
    codec: nvcomp.Codec,
    host_input: HostArray,
    stream: cp.cuda.Stream,
    args: argparse.Namespace,
) -> None:
    if args.duplicate_data > 1:
        host_input = np.tile(host_input, args.duplicate_data)

    n = host_input.size

    with stream:
        d_in = cp.zeros(n, dtype=cp.uint8)
        d_in.set(host_input)

    in_arr = nvcomp.as_array(d_in)

    comp_cfg = codec.compression_config(n)
    decomp_cfg = codec.decompression_config(comp_cfg)

    with stream:
        comp_buf = cp.zeros(codec.get_max_comp_buffer_size(in_arr), dtype=cp.uint8)
        out_buf = cp.zeros(n, dtype=cp.uint8)

    comp_arr = nvcomp.as_array(comp_buf)
    out_arr = nvcomp.as_array(out_buf)

    def enc_call() -> nvcomp.Array:
        return codec.encode(in_arr, compression_config=comp_cfg, out=comp_arr)

    enc_samples = benchmark_operation(enc_call, args.seconds, stream, args.min_samples)

    d_comp = enc_call()
    comp_bytes = d_comp.buffer_size

    def dec_call() -> nvcomp.Array:
        return codec.decode(d_comp, decompression_config=decomp_cfg, out=out_arr)

    dec_samples = benchmark_operation(dec_call, args.seconds, stream, args.min_samples)

    if not np.array_equal(cp.asnumpy(out_buf), host_input):
        sys.exit("error: roundtrip verification failed")

    print_report(args, 1, n, comp_bytes, enc_samples, dec_samples)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-f", "--input_file", required=True, help="input file")
    ap.add_argument(
        "-a",
        "--algorithm",
        required=True,
        type=canonical_algorithm,
        choices=ALGORITHMS,
        help="algorithm",
    )
    ap.add_argument(
        "-p",
        "--chunk_size",
        type=int,
        default=None,
        help="chunk size in bytes (default 64 KiB); for RAW it splits the input, "
        "for NVCOMP_NATIVE it is the HLIF internal chunk size",
    )
    ap.add_argument(
        "-b",
        "--bitstream_kind",
        choices=("NVCOMP_NATIVE", "RAW", "WITH_UNCOMPRESSED_SIZE"),
        default="NVCOMP_NATIVE",
        help="bitstream kind (default NVCOMP_NATIVE)",
    )
    ap.add_argument(
        "-x",
        "--duplicate_data",
        type=int,
        default=0,
        help="replicate the input N times (0 or 1 means no duplication)",
    )
    ap.add_argument(
        "--seconds",
        type=float,
        default=2.0,
        help="per-operation timing budget: sample each operation until the "
        "collected GPU timings sum to this many seconds (default 2s)",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="minimum number of samples per operation, collected even if the "
        "--seconds budget is already spent (default 10)",
    )
    ap.add_argument(
        "--data_type",
        metavar="TYPESTR",
        help="Array-protocol typestr for the element type (e.g. <i4, |u1, |f1 for "
        "fp8); sets the input element size. Cascaded defaults to <i4",
    )
    ap.add_argument(
        "--device_id",
        type=int,
        help="CUDA device id (all algorithms)",
    )
    ap.add_argument(
        "--codec-opts",
        nargs="*",
        default=[],
        metavar="KEY=VAL",
        help="extra options forwarded to nvcomp.Codec as key=value pairs, e.g. "
        "compression_level=2 fine_grained_encodings=3 algorithm_type=3 use_de_sort=true",
    )
    ap.add_argument(
        "--csv", metavar="PATH", help="append a result row (CSV) to this file"
    )
    args = ap.parse_args()

    if args.chunk_size is None:
        args.chunk_size = DEFAULT_HLIF_CHUNK_SIZE
    if args.chunk_size <= 0:
        sys.exit("error: chunk_size must be > 0")
    if args.chunk_size % 8 != 0:
        sys.exit("error: chunk_size must be a multiple of 8 bytes")
    if args.duplicate_data < 0:
        sys.exit("error: duplicate_data must be >= 0")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        sys.exit("error: seconds must be a finite number > 0")
    if args.min_samples < 1:
        sys.exit("error: min-samples must be >= 1")

    codec_kwargs = parse_codec_opts(args.codec_opts)
    resolve_enum_opts(codec_kwargs)

    # data_type and device_id have dedicated flags but may also be given via
    # --codec-opts. The dedicated flag wins; warn if both are provided.
    for key, flag_value in (
        ("data_type", args.data_type),
        ("device_id", args.device_id),
    ):
        if flag_value is not None:
            if key in codec_kwargs:
                print(f"warning: --{key} overrides --codec-opts {key}", file=sys.stderr)
            codec_kwargs[key] = flag_value

    # Human-readable summary of the effective codec options, for the report.
    args.codec_opts_str = " ".join(
        f"{k}={getattr(v, 'name', v)}" for k, v in sorted(codec_kwargs.items())
    )

    with open(args.input_file, "rb") as f:
        host_input = np.frombuffer(f.read(), dtype=np.uint8).copy()
    if host_input.size == 0:
        sys.exit(f"error: input file '{args.input_file}' is empty")

    data_type = codec_kwargs.get("data_type")
    if data_type is not None:
        try:
            item_size = np.dtype(data_type).itemsize
        except (TypeError, ValueError):
            if data_type != FLOAT8_E4M3_TYPESTR:
                sys.exit(f"error: unrecognized data_type '{data_type}'")
            item_size = 1
        if item_size == 0:
            sys.exit(f"error: data_type '{data_type}' has zero element size")
    elif args.algorithm == "Cascaded":
        item_size = np.dtype("<i4").itemsize
    else:
        item_size = 1

    usable = host_input.size - host_input.size % item_size
    if usable == 0:
        sys.exit(f"error: input smaller than one element ({item_size} B)")

    host_input = host_input[:usable]

    if args.chunk_size <= 0:
        sys.exit(f"error: chunk_size must be positive, got {args.chunk_size}")
    if args.chunk_size < item_size:
        sys.exit(f"error: chunk_size must be at least one element ({item_size} B)")

    chunk_alignment = math.lcm(MAX_CODEC_ALIGNMENT, item_size)
    if args.chunk_size % chunk_alignment != 0:
        aligned = args.chunk_size - args.chunk_size % chunk_alignment
        if aligned == 0:
            aligned = chunk_alignment
        print(
            f"note: chunk_size {args.chunk_size} not a multiple of {chunk_alignment} B; "
            f"using {aligned}",
            file=sys.stderr,
        )
        args.chunk_size = aligned

    device_id = codec_kwargs.get("device_id")
    try:
        device = (
            cp.cuda.Device(device_id) if device_id is not None else cp.cuda.Device()
        )
        with device:
            stream = cp.cuda.Stream(non_blocking=True)
            try:
                codec = nvcomp.Codec(
                    algorithm=args.algorithm,
                    cuda_stream=stream.ptr,
                    uncomp_chunk_size=args.chunk_size,
                    bitstream_kind=getattr(nvcomp.BitstreamKind, args.bitstream_kind),
                    **codec_kwargs,
                )
                if args.bitstream_kind == "NVCOMP_NATIVE":
                    run_single(codec, host_input, stream, args)
                else:
                    run_chunked(codec, host_input, args.chunk_size, stream, args)
            finally:
                # Drop the codec before the stream it borrows.
                codec = None
    except (RuntimeError, ValueError, MemoryError, OSError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
