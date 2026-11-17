# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LZ4 compression straight from and to global memory with nvmath.tile.compress.

The chunks, compressed payloads, and restored chunks stay in global-memory arrays.
Compression reads an input slice and writes a payload slot through ``out=``.
Decompression reads that slot and writes the restored chunk through ``out=``.
Only the byte counts are tiles.
"""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
import nvmath.tile.compress as comp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import Runtime

ALGORITHM = "lz4"
DTYPE, NP_DTYPE = ct.uint8, np.uint8
CHUNKS, ELEMENTS = 4, 4096
CHUNK_BYTES = ELEMENTS * np.dtype(NP_DTYPE).itemsize
COMPRESS_SCRATCH = comp.compress_scratch_size(CHUNK_BYTES, ALGORITHM, DTYPE)
DECOMPRESS_SCRATCH = comp.decompress_scratch_size(CHUNK_BYTES, ALGORITHM, DTYPE)
# Exact worst case per chunk; slots are padded so each one stays aligned.
CAPACITY = comp.max_compressed_size(CHUNK_BYTES, ALGORITHM)
SLOT = (CAPACITY + comp.buffer_alignment() - 1) // comp.buffer_alignment() * comp.buffer_alignment()


@ct.kernel
def compress_kernel(data: ct.Array, payload: ct.Array, sizes: ct.Array, scratch: ct.Array):
    chunk = ct.bid(0)
    source = data.slice(0, chunk * ELEMENTS, (chunk + 1) * ELEMENTS)
    slot = payload.slice(0, chunk * SLOT, chunk * SLOT + CAPACITY)
    chunk_scratch = scratch.slice(0, chunk * COMPRESS_SCRATCH, (chunk + 1) * COMPRESS_SCRATCH)
    size = comp.compress(source, ALGORITHM, chunk_scratch, out=slot, num_elements=ELEMENTS)
    ct.store(sizes, (chunk,), size)


@ct.kernel
def decompress_kernel(payload: ct.Array, sizes: ct.Array, restored: ct.Array, scratch: ct.Array):
    chunk = ct.bid(0)
    slot = payload.slice(0, chunk * SLOT, chunk * SLOT + CAPACITY)
    target = restored.slice(0, chunk * ELEMENTS, (chunk + 1) * ELEMENTS)
    size = ct.load(sizes, (chunk,), shape=())
    chunk_scratch = scratch.slice(0, chunk * DECOMPRESS_SCRATCH, (chunk + 1) * DECOMPRESS_SCRATCH)
    comp.decompress(slot, size, ALGORITHM, ELEMENTS, DTYPE, chunk_scratch, out=target)


def main() -> None:
    rt = Runtime()
    index = np.arange(CHUNKS * ELEMENTS) % ELEMENTS
    chunk = np.arange(CHUNKS * ELEMENTS) // ELEMENTS
    data = ((index // 16 + 3 * chunk) % 23).astype(NP_DTYPE)
    payload = rt.zeros(CHUNKS * SLOT, np.uint8)
    sizes = rt.zeros(CHUNKS, np.uint64)
    restored = rt.zeros(data.shape, NP_DTYPE)
    compress_scratch = rt.zeros(CHUNKS * COMPRESS_SCRATCH, np.uint8)
    decompress_scratch = rt.zeros(CHUNKS * DECOMPRESS_SCRATCH, np.uint8)
    rt.launch(compress_kernel, (rt.to_device(data), payload, sizes, compress_scratch), grid=(CHUNKS, 1, 1))
    rt.launch(decompress_kernel, (payload, sizes, restored, decompress_scratch), grid=(CHUNKS, 1, 1))
    rt.sync()
    compressed = int(rt.to_host(sizes).sum())
    exact = np.array_equal(rt.to_host(restored), data)
    print(f"lz4 gmem: {CHUNKS} x {CHUNK_BYTES} B -> {compressed} B, round trip exact: {exact}")
    assert exact and compressed < CHUNKS * CHUNK_BYTES


if __name__ == "__main__":
    main()
