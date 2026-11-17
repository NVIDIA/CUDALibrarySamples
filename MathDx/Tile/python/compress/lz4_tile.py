# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LZ4 compression of tiles with nvmath.tile.compress.

Each block loads one chunk into a tile, compresses it, decompresses the
compressed tile, and stores both results.
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
PAYLOAD = comp.compressed_tile_size(CHUNK_BYTES, ALGORITHM)


@ct.kernel
def compress_kernel(
    data: ct.Array,
    payload: ct.Array,
    sizes: ct.Array,
    restored: ct.Array,
    compress_scratch: ct.Array,
    decompress_scratch: ct.Array,
):
    chunk = ct.bid(0)
    scratch = compress_scratch.slice(0, chunk * COMPRESS_SCRATCH, (chunk + 1) * COMPRESS_SCRATCH)
    compressed, size = comp.compress(ct.load(data, (chunk,), shape=(ELEMENTS,)), ALGORITHM, scratch)
    ct.store(payload, (chunk,), compressed)
    ct.store(sizes, (chunk,), size)
    scratch = decompress_scratch.slice(0, chunk * DECOMPRESS_SCRATCH, (chunk + 1) * DECOMPRESS_SCRATCH)
    data_tile = comp.decompress(compressed, size, ALGORITHM, ELEMENTS, DTYPE, scratch)
    ct.store(restored, (chunk,), data_tile)


def main() -> None:
    rt = Runtime()
    index = np.arange(CHUNKS * ELEMENTS) % ELEMENTS
    chunk = np.arange(CHUNKS * ELEMENTS) // ELEMENTS
    data = ((index // 16 + 3 * chunk) % 23).astype(NP_DTYPE)
    payload = rt.zeros(CHUNKS * PAYLOAD, np.uint8)
    sizes = rt.zeros(CHUNKS, np.uint64)
    restored = rt.zeros(data.shape, NP_DTYPE)
    compress_scratch = rt.zeros(CHUNKS * COMPRESS_SCRATCH, np.uint8)
    decompress_scratch = rt.zeros(CHUNKS * DECOMPRESS_SCRATCH, np.uint8)
    rt.launch(
        compress_kernel,
        (rt.to_device(data), payload, sizes, restored, compress_scratch, decompress_scratch),
        grid=(CHUNKS, 1, 1),
    )
    rt.sync()
    compressed = int(rt.to_host(sizes).sum())
    exact = np.array_equal(rt.to_host(restored), data)
    print(f"lz4 tile: {CHUNKS} x {CHUNK_BYTES} B -> {compressed} B, round trip exact: {exact}")
    assert exact and compressed < CHUNKS * CHUNK_BYTES


if __name__ == "__main__":
    main()
