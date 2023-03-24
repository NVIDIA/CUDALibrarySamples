from enum import Enum

import jax
from jax.experimental import PartitionSpec


class Dist(Enum):

    """Describes a SLAB data decomposition

    For a X*Y*Z array, SLABS_X indicates the array is
    distributed along the first dimension, i.e., each 
    device owns a slab of size (X // nGPUs)*Y*Z
    SLABS_Y indicates the array is distributed along the
    second dimension, with each device owning a slab
    of size X*(Y // nGPUs)*Z.
    """

    SLABS_X = 'SLABS_X'
    SLABS_Y = 'SLABS_Y'

    @staticmethod
    def create(string):
        if string == 'X':
            return Dist.SLABS_X
        elif string == 'Y':
            return Dist.SLABS_Y
        else:
            raise RuntimeError("Wrong dist")

    @property
    def opposite(dist):
        if dist == Dist.SLABS_X:
            return Dist.SLABS_Y
        else:
            return Dist.SLABS_X

    @property
    def _C_enum(dist):
        if dist == Dist.SLABS_X:
            return 0
        else:
            return 1

    def fft_axes(self, fft_rank):
        if self == Dist.SLABS_X:
            return list(range(1, fft_rank))
        else:
            return [0]

    def xmap_shape(self, fft_dims):
        ngpus = jax.device_count()
        if self == Dist.SLABS_X:
            return (
                ngpus,
                fft_dims[0] // ngpus,
                fft_dims[1],
                *fft_dims[2:]
            )
        else:
            return (
                fft_dims[0],
                ngpus,
                fft_dims[1] // ngpus,
                *fft_dims[2:]
            )

    def slab_shape(self, fft_dims):
        ngpus = jax.device_count()
        if self == Dist.SLABS_X:
            return (
                fft_dims[0] // ngpus,
                fft_dims[1],
                *fft_dims[2:]
            )
        else:
            return (
                fft_dims[0],
                fft_dims[1] // ngpus,
                *fft_dims[2:]
            )

    def fft_shape(self, local_shape):
        ngpus = jax.device_count()
        if self == Dist.SLABS_X:
            return (local_shape[0] * ngpus, local_shape[1], *local_shape[2:])
        else:
            return (local_shape[0], local_shape[1] * ngpus, *local_shape[2:])

    @property
    def axes_map(dist):
        if dist == Dist.SLABS_X:
            return {0: "gpus"}
        else:
            return {1: "gpus"}

    @property
    def part_spec(dist):
        if dist == Dist.SLABS_X:
            return PartitionSpec("gpus", None)
        else:
            return PartitionSpec(None, "gpus")


class Dir(Enum):

    """Describe the FFT direction

    FWD is the forward, unnormalized, direction.
    BWD is the backward, normalized by 1/N, direction,
    with N the product of the dimensions.
    """

    FWD = 'FWD'
    INV = 'INV'

    @property
    def _C_enum(dir):
        if dir == Dir.FWD:
            return 0
        else:
            return 1

    @property
    def opposite(dir):
        if dir == Dir.FWD:
            return Dir.INV
        else:
            return Dir.FWD
