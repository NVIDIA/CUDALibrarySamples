from functools import partial

import jax
from jax._src.sharding import NamedSharding
from jax.experimental.custom_partitioning import custom_partitioning
from fft_common import Dir


def _fft(x, dist, dir):

    """ Compute a local FFT along the appropriate axes (based on dist), in the
    forward or backward direction """

    if dir == Dir.FWD:
        return jax.numpy.fft.fftn(x, axes=dist.fft_axes(len(x.shape)))
    else:
        return jax.numpy.fft.ifftn(x, axes=dist.fft_axes(len(x.shape)))


def _supported_sharding(sharding, dist):
    return NamedSharding(sharding.mesh, dist.part_spec)


def _partition(arg_shapes,
               arg_shardings,
               result_shape,
               result_sharding,
               dist,
               dir):
    return lambda x: _fft(x, dist, dir), \
           _supported_sharding(arg_shardings[0], dist), \
           [_supported_sharding(arg_shardings[0], dist)]


def _infer_sharding_from_operands(arg_shapes,
                                  arg_shardings,
                                  result_shape,
                                  dist,
                                  dir):
    return _supported_sharding(arg_shardings[0], dist)


def fft(x, dist, dir):

    """ Extends jax.numpy.fft.fftn to support sharding along the first or
    second direction, without intermediate re-sharding """

    @custom_partitioning
    def _fft_(x):
        return _fft(x, dist, dir)

    _fft_.def_partition(
        infer_sharding_from_operands=partial(_infer_sharding_from_operands,
                                             dist=dist,
                                             dir=dir),
        partition=partial(_partition, dist=dist, dir=dir))

    return _fft_(x)


def xfft(x, dist, dir):

    """Compute the discrete Fourier transform using a JAX-only implementation.

    Arguments:
    x    -- the input tensor
    dist -- the data decomposition of x.
            Should be an instance of fft_common.Dist
    dir  -- the direction of the transform.
            Should be an instance of fft_common.Dir

    Returns the transformed tensor.
    The output tensoris distributed according to dist.opposite

    This function should be used with pjit like

        pjit(xfft,
             in_axis_resources=dist.part_spec,
             out_axis_resources=dist.opposite.part_spec,
             static_argnums=[1, 2]
            )(x, dist, dir)
    """

    # If dist == Dist.SLABS_X, FFT along Y and Z
    x = fft(x, dist, dir)

    # Implicitly re-shards to match the required
    # input sharding of the next fft(..., dist.opposite, ...)

    # If dist == Dist.SLABS_X, FFT along X
    x = fft(x, dist.opposite, dir)

    return x
