# cuTENSOR Python Binding Sample

This sample provides a Python package `cutensor` with bindings for PyTorch and Tensorflow that provides an [einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) style interface that leverages cuTENSOR and can be used similar to PyTorch's and Tensorflow's native einsum implementations.


## Installation

1. Download and extract the cuTENSOR library. Visit https://developer.nvidia.com/cutensor/downloads for instructions. For the purpose of this sample, we assume that cuTENSOR is extracted in the current working directory.
2. Start a docker container that has access to the cuTENSOR library and set the `CUTENSOR_ROOT` environment variable appropriately.
   
   ```
   # PyTorch
   host$ docker run -it --rm --gpus all -v $PWD/libcutensor:/cutensor --env CUTENSOR_ROOT=/cutensor nvcr.io/nvidia/pytorch:23.09-py3

   # Tensorflow
   host$ docker run -it --rm --gpus all -v $PWD/libcutensor:/cutensor --env CUTENSOR_ROOT=/cutensor nvcr.io/nvidia/tensorflow:23.03-tf1-py3
   ```
3. Clone this repository in the docker container, and install this package.

   ```
   docker$ git clone https://github.com/NVIDIA/CUDALibrarySamples.git
   docker$ cd CUDALibrarySamples/cuTENSOR/python
   docker$ pip install .
   ```
4. Run the tests.
   ```
   docker$ pip install parameterized
   docker$ TF_CPP_MIN_LOG_LEVEL=3 python cutensor/tensorflow/einsum_test.py
   docker$ python cutensor/torch/einsum_test.py
   ```


## PyTorch Usage

All PyTorch functionality is part of the `cutensor.torch` module.
In particular, it provides `einsum` function that performs a single binary einsum, a `EinsumFunction` that performs an einsum operation with gradient support, a `Einsum` `torch.nn.Module` and an `EinsumGeneral` function that provides support for einsum operations with more than two input tensors.
Among those, `EinsumGeneral` is easiest to use, see the following example (for more samples, see the tests in `cutensor/torch/einsum_test.py`):

    from cutensor.torch import EinsumGeneral

    def batched_1x1_convolution(weight_tensor, activation_tensor)
        return EinsumGeneral('kc,nchw->nkhw', weight_tensor, activation_tensor)


## Tensorflow Usage

All Tensorflow functionality can be found as part of the `cutensor.tensorflow` module.
It provides a single function `einsum` that provides an accelerated unary or binary einsum operation.
The following example shows how that function can be used (for more samples, see the tests in `cutensor/tensorflow/einsum_test.py`):

    from cutensor.tensorflow import einsum
    
    def batched_1x1_convolution(weight_tensor, activation_tensor):
        return einsum('kc,nchw->nkhw', weight_tensor, activation_tensor)
    
