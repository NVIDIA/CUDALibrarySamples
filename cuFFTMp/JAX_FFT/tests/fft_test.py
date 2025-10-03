# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time
import math
import numpy as np
import sys

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from fft_common import Dist, Dir
from cufftmp_jax import cufftmp
from xfft import xfft

import helpers


def main():

    opt = helpers.parser()

    # Initialize JAX for multi-process runs
    if opt['multiprocess'] is not None:
        if opt['multiprocess'] == 'bootstrap':
            jax.distributed.initialize()
        else:
            coordinator, num_procs, my_proc = opt['multiprocess'].split(',')
            jax.distributed.initialize(
                coordinator_address=coordinator,
                num_processes=int(num_procs),
                process_id=int(my_proc)
            )

    fft_dims = opt['shape']
    cycles = opt['cycles']

    impl = opt['implementation']
    if impl == "cufftmp":
        dist_fft = cufftmp
    elif impl == "xfft":
        dist_fft = xfft
    else:
        raise ValueError(f"Wrong implementation: got {impl}, expected cufftmp or xfft")

    dist = Dist.create(opt['dist'])
    input_shape = dist.slab_shape(fft_dims)
    dtype = jnp.complex64

    mesh = Mesh(np.asarray(jax.devices()), ('gpus',))
    sharding = jax.sharding.NamedSharding(mesh, dist.part_spec)

    with jax.spmd_mode('allow_all'):

        if opt['mode'] == "test":

            seed = 170
            key = jax.random.PRNGKey(seed)
            input = jax.random.normal(key, shape=fft_dims, dtype=dtype)

            with mesh:

                fft = jax.jit(dist_fft,
                              in_shardings=sharding,
                              out_shardings=sharding,
                              static_argnums=[1, 2])

                output = fft(input, dist, Dir.FWD)

            output_ref = jnp.fft.fftn(input)
            error = jnp.linalg.norm(output - output_ref) / \
                jnp.linalg.norm(output_ref)

            if jax.process_index() == 0:
                print(f"{impl} (test): {fft_dims}, dist {dist} --> {dist.opposite}, num GPUs {jax.device_count()}, num processes {jax.process_count()}, L2 rel error {error:.2e}")
                if error < 1e-4:
                    print("&&&& PASSED")
                else:
                    print("&&&& FAILED")
                    sys.exit(1)

        else:

            with mesh:

                # Performance testing only supports 1 device per process
                # because of the way the input `dinput` is generated
                assert jax.local_device_count() == 1

                # Quick generation of the local array
                input = jnp.ones(input_shape, dtype=dtype)

                # Create the global sharded array
                dinput = jax.make_array_from_single_device_arrays(
                    fft_dims,
                    sharding,
                    [input])

                # Function to benchmark. 
                # One forward distributed FFT and one backward distributed FFT.
                def fwd_bwd(x, dist, dir):
                    return dist_fft(dist_fft(x, dist, dir),
                                    dist.opposite,
                                    dir.opposite)

                fwd_bwd_jit = jax.jit(fwd_bwd,
                                      in_shardings=sharding,
                                      out_shardings=sharding,
                                      static_argnums=[1, 2])

                def fwd_bwd_bench(x):
                    return fwd_bwd_jit(x, dist, Dir.FWD)

                # Warmup
                x = fwd_bwd_bench(dinput).block_until_ready()

                # Benchmark
                times = []
                x = dinput
                for _ in range(cycles):
                    cycle_start = time.time()
                    x = fwd_bwd_bench(x)
                    x.block_until_ready()
                    cycle_end = time.time()
                    times.append((cycle_end - cycle_start)/2) # divide by 2 since two FFTs in forward-backward transform
                doutput = x.block_until_ready()

                # Check error
                doutput_ref = dinput
                error = helpers.Frob_error(doutput_ref, doutput, mesh, dist)

                # Calculate statistics, and multiply 1e3 to convert to ms
                mean_time = np.mean(times) * 1e3
                median_time = np.median(times) * 1e3
                std_dev = np.std(times) * 1e3
                min_time = np.min(times) * 1e3
                max_time = np.max(times) * 1e3
                
                # Perf ?
                perf_GFlops = \
                    (5 * math.prod(fft_dims) * math.log2(math.prod(fft_dims))) / 1e9 / median_time
                bandwidth_GBsGPUdir = \
                    (8 * math.prod(fft_dims)) / jax.device_count()             / 1e9 / median_time

                if jax.process_index() == 0:
                    print(f"{impl} (perf): {fft_dims}, num GPUs {jax.device_count()}, num processes {jax.process_count()}, relative L2 error {error:.2e}, cycles {cycles}, time_avg {mean_time:.2e} ms, time_med {median_time:.2e} ms, time_std {std_dev:.2e} ms, time_min {min_time:.2e} ms, time_max {max_time:.2e} ms, perf_median {perf_GFlops:.2e} GFlop/s, bandwidth_median {bandwidth_GBsGPUdir:.2e} GB/s/GPU")
                    if error < 1e-4:
                        print("&&&& PASSED")
                    else:
                        print("&&&& FAILED")
                        sys.exit(1)


if __name__ == "__main__":
    main()
