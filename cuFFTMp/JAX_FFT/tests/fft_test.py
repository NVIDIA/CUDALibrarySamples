# -*- coding: utf-8 -*-

import time
import math
import numpy as np
import sys

import jax
import jax.numpy as jnp
from jax.experimental import maps
from jax.experimental.pjit import pjit

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

    mesh = maps.Mesh(np.asarray(jax.devices()), ('gpus',))

    with jax.spmd_mode('allow_all'):

        if opt['mode'] == "test":

            seed = 170
            key = jax.random.PRNGKey(seed)
            input = jax.random.normal(key, shape=fft_dims, dtype=dtype)

            with mesh:

                fft = pjit(dist_fft,
                           in_axis_resources=None,
                           out_axis_resources=None,
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
                    jax.sharding.NamedSharding(mesh, dist.part_spec),
                    [input])

                # Function to benchmark
                def fwd_bwd(x, dist, dir):
                    return dist_fft(dist_fft(x, dist, dir),
                                    dist.opposite,
                                    dir.opposite)

                fwd_bwd_pjit = pjit(fwd_bwd,
                                    in_axis_resources=dist.part_spec,
                                    out_axis_resources=dist.part_spec,
                                    static_argnums=[1, 2])

                def fwd_bwd_bench(x):
                    return fwd_bwd_pjit(x, dist, Dir.FWD)

                # Warmup
                x = fwd_bwd_bench(dinput).block_until_ready()

                # Benchmark
                start = time.time()
                x = dinput
                for _ in range(cycles):
                    x = fwd_bwd_bench(x)
                doutput = x.block_until_ready()
                stop = time.time()

                # Check error
                doutput_ref = dinput
                error = helpers.Frob_error(doutput_ref, doutput, dist)

                # Perf ?
                time_s = stop - start
                av_time_s = time_s / (2 * cycles)
                perf_GFlops = \
                    (5 * math.prod(fft_dims) * math.log2(math.prod(fft_dims))) / 1e9 / av_time_s
                bandwidth_GBsGPUdir = \
                    (8 * math.prod(fft_dims)) / jax.device_count()             / 1e9 / av_time_s

                if jax.process_index() == 0:
                    print(f"{impl} (perf): {fft_dims}, num GPUs {jax.device_count()}, num processes {jax.process_count()}, relative L2 error {error:.2e}, cycles {cycles}, time {av_time_s * 1e3:.2e} ms, perf {perf_GFlops:.2e} GFlop/s, bandwidth {bandwidth_GBsGPUdir:.2e} GB/s/GPU")
                    if error < 1e-4:
                        print("&&&& PASSED")
                    else:
                        print("&&&& FAILED")
                        sys.exit(1)


if __name__ == "__main__":
    main()
