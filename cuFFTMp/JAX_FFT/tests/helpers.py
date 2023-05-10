import argparse

import jax
from jax.experimental.pjit import pjit
from jax.experimental.maps import xmap


def Frob_error_impl(dtest, dref):
    derr2 = jax.numpy.linalg.norm(dtest - dref) ** 2
    dnorm2 = jax.numpy.linalg.norm(dref) ** 2
    derr2_sum = jax.lax.psum(derr2,  axis_name="gpus")
    dnorm2_sum = jax.lax.psum(dnorm2, axis_name="gpus")
    error = jax.numpy.sqrt(derr2_sum / dnorm2_sum)
    return error


def Frob_error(dtest, dref, dist):

    """Computes the relative error in the Frobenius norm

    Arguments:
    dtest  -- the test array, sharded along the axis `ngpus`
    dref   -- the reference array, sharded along the axis `ngpus`
    dist   -- the sharding of dtest and dref
              Should be an instance of fft_common.Dist

    Returns the relative error in the Frobenius norm, i.e., 
    ||dtest - dref||_F / ||dref||_F
    """

    return pjit(
        xmap(
            Frob_error_impl,
            in_axes=dist.axes_map,
            out_axes={},
            axis_resources={'gpus': 'gpus'}
        ),
        in_axis_resources=dist.part_spec,
        out_axis_resources=None
    )(dtest, dref)


def parser():
    parser = argparse.ArgumentParser(
        description="Test program for distributed FFTs in JAX"
    )
    parser.add_argument(
        "implementation",
        type=str,
        choices=['cufftmp', 'xfft'],
        default='cufftmp',
        help='uses cuFFTMp or pjit+xmap'
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=['test', 'perf'],
        default='test',
        help='test (correctness) or perf (performance)'
    )
    parser.add_argument(
        "-x", "--xsize",
        type=int,
        help="Size along X",
        default=1
    )
    parser.add_argument(
        "-y", "--ysize",
        type=int,
        help="Size along Y",
        default=1
    )
    parser.add_argument(
        "-z", "--zsize",
        type=int,
        help="Size along Z",
        default=None
    )
    parser.add_argument(
        "-n", "--size",
        type=int,
        help="Size along X, Y and Z (takes precedence over xsize, ysize and zsize",
        default=None
    )
    parser.add_argument(
        "-c", "--cycles",
        type=int,
        help="Cycles to benchmark (perf only)",
        default=10
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help="Verbosity level (0 = silent, 2 = debug)"
    )
    parser.add_argument(
        '-d', '--dist',
        type=str,
        choices=['X', 'Y'],
        default='X',
        help="Input distribution (X for SLABS_X or Y for SLABS_Y)"
    )
    parser.add_argument(
        "--multiprocess",
        type=str,
        default=None,
        help="If set, should be of the shape `coordinator,num_procs,proc_id` or `bootstrap` (automatic cluster detection)")
    args = parser.parse_args()
    if args.size:
        shape = args.size, args.size, args.size
    else:
        if args.zsize:
            shape = args.xsize, args.ysize, args.zsize
        else:
            shape = args.xsize, args.ysize
    return {'shape': shape, **vars(args)}
