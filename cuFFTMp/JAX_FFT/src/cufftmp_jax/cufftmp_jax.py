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


__all__ = ["cufftmp"]

from functools import partial
import math

import jax
from jax.lib import xla_client
from jax import core, dtypes
from jax.interpreters import xla, mlir
from jax.core import ShapedArray
from jax.sharding import NamedSharding
from jax.experimental.custom_partitioning import custom_partitioning
from jaxlib.hlo_helpers import custom_call

from fft_common import Dir, Dist

from . import gpu_ops
for _name, _value in gpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

xops = xla_client.ops

# ************
# * BINDINGS *
# ************


def _cufftmp_bind(input, num_parts, dist, dir):

    # param=val means it's a static parameter
    (output,) = _cufftmp_prim.bind(input,
                                   num_parts=num_parts,
                                   dist=dist,
                                   dir=dir)

    # scale in INVERSE direction
    if dir == Dir.INV:
        fft_dims = dist.fft_shape(input.shape)
        output = jax.numpy.divide(output, math.prod([
            jax.numpy.complex64(f) for f in fft_dims
        ]))

    return output


def _supported_sharding(sharding, dist):
    return NamedSharding(sharding.mesh, dist.part_spec)


def _partition(mesh, 
               arg_shapes,
               result_shape, 
               dist,
               dir):

    """ Describes the required input and output sharding of the op.
    `mesh`, `arg_shapes` and `result_shape` are the shapes provided by the
    user (i.e., in jit).
    Returns:
    - The operation to perform locally on all GPUs
    - The output sharding
    - The input sharding """
    
    arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)

    return mesh, \
        lambda x: _cufftmp_bind(x,
                                num_parts=jax.device_count(),
                                dist=dist,
                                dir=dir), \
        _supported_sharding(arg_shardings[0], dist.opposite), \
        (_supported_sharding(arg_shardings[0], dist),)


def _infer_sharding_from_operands(mesh,
                                  arg_shapes,
                                  result_shape,
                                  dist,
                                  dir):

    arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
    return _supported_sharding(arg_shardings[0], dist)


def cufftmp(x, dist, dir):

    """Compute the DFT using a JAX+cuFFTMp implementation.

    Arguments:
    x    -- the input tensor
    dist -- the data decomposition of x.
            Should be an instance of fft_common.Dist
    dir  -- the direction of the transform.
            Should be an instance of fft_common.Dir

    Returns the transformed tensor.
    The output tensoris distributed according to dist.opposite

    This function should be used with jit like

        jit(
            cufftmp,
            in_shardings=sharding,
            out_shardings=sharding,
            static_argnums=[1, 2]
            )(x, dist, dir)

    """

    # cuFFTMp only supports 1 device per proces
    assert jax.local_device_count() == 1

    @custom_partitioning
    def _cufftmp_(x):
        return _cufftmp_bind(x, num_parts=1, dist=dist, dir=dir)

    _cufftmp_.def_partition(
        infer_sharding_from_operands=partial(
            _infer_sharding_from_operands,
            dist=dist,
            dir=dir),
        partition=partial(
            _partition,
            dist=dist,
            dir=dir))

    return _cufftmp_(x)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# Abstract implementation, i.e., return the shape of the output array
# based on the input array and a number of partitions (ie devices)
def _cufftmp_abstract(input, num_parts, dist, dir):
    dtype = dtypes.canonicalize_dtype(input.dtype)
    input_shape = input.shape
    if dist == Dist.SLABS_X:
        output_shape = (input_shape[0] * num_parts,
                        input_shape[1] // num_parts,
                        *input_shape[2:])
    elif dist == Dist.SLABS_Y:
        output_shape = (input_shape[0] // num_parts,
                        input_shape[1] * num_parts,
                        *input_shape[2:])
    return (ShapedArray(output_shape, dtype),)


# Implementation calling into the C++ bindings
def _cufftmp_lowering(ctx, input, num_parts, dist, dir):

    assert num_parts == jax.device_count()

    input_type = mlir.ir.RankedTensorType(input.type)
    dims_in = input_type.shape
    fft_dims = dist.fft_shape(dims_in)
    dims_out = dist.opposite.slab_shape(fft_dims)
    output_type = mlir.ir.RankedTensorType.get(
        dims_out,
        input_type.element_type
    )

    layout = tuple(range(len(dims_in) - 1, -1, -1))

    if len(fft_dims) == 2:
        opaque = gpu_ops.build_cufftmp_descriptor(
            fft_dims[0],
            fft_dims[1],
            1,
            dist._C_enum,
            dir._C_enum
        )
    elif len(fft_dims) == 3:
        opaque = gpu_ops.build_cufftmp_descriptor(
            fft_dims[0],
            fft_dims[1],
            fft_dims[2],
            dist._C_enum,
            dir._C_enum
        )
    else:
        raise ValueError("Unsupported tensor rank; must be 2 or 3")

    return custom_call(
        "gpu_cufftmp",
        # Output types
        result_types=[output_type],
        # The inputs:
        operands=[input,],
        # Layout specification:
        operand_layouts=[layout,],
        result_layouts=[layout,],
        # GPU specific additional data
        backend_config=opaque
    ).results


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_cufftmp_prim = core.Primitive("cufftmp")
_cufftmp_prim.multiple_results = True
_cufftmp_prim.def_impl(partial(xla.apply_primitive, _cufftmp_prim))
_cufftmp_prim.def_abstract_eval(_cufftmp_abstract)

# Register the op with MLIR
mlir.register_lowering(_cufftmp_prim, _cufftmp_lowering, platform="gpu")
