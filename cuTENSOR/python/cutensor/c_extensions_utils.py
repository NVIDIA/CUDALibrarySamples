# ! /usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  - Neither the name(s) of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from setuptools import Extension
from distutils.spawn import find_executable
import os
import subprocess
import re

__all__ = ['CustomExtension']

include_dirs = []
library_dirs = []

cuda_nvcc = find_executable('nvcc')
cuda_root = os.path.join(os.path.dirname(cuda_nvcc), os.pardir)
cuda_version = re.search(
    r'release ([^,]*),',
    subprocess.check_output([cuda_nvcc, '--version']).decode('utf-8')).group(1)
include_dirs.append(os.path.join(cuda_root, 'include'))
library_dirs.append(os.path.join(cuda_root, 'lib64'))

if 'CUTENSOR_ROOT' in os.environ:
    root = os.environ['CUTENSOR_ROOT']
    include_dirs.append(os.path.join(root, 'include'))
    library_dirs.append(os.path.join(root, 'lib'))
    library_dirs.append(os.path.join(root, 'build/lib'))
    versioned_path = os.path.join(root, 'lib', cuda_version)
    if not os.path.exists(versioned_path):
        versioned_path = os.path.join(root, 'lib', cuda_version.split('.')[0])
    library_dirs.append(versioned_path)


class CustomExtension:
    modules = []

    @classmethod
    def Torch(cls, name, sources):
        try:
            import torch
            from torch.utils.cpp_extension import CUDAExtension
            ext = CUDAExtension(name,
                                sources=sources,
                                libraries=['cutensor'],
                                define_macros=[
                                    ('TORCH_API_INCLUDE_EXTENSION_H',),
                                    ('TORCH_EXTENSION_NAME',
                                     name.split('.')[-1]),
                                    ('_GLIBCXX_USE_CXX11_ABI',
                                     str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))
                                ],
                                extra_compile_args=['-std=c++17', '-fopenmp'],
                                extra_link_args=['-std=c++17', '-fopenmp'],
                                include_dirs=include_dirs,
                                library_dirs=library_dirs,
                                runtime_library_dirs=library_dirs)
            cls.modules.append(ext)
            return ext
        except ImportError:
            return None

    @classmethod
    def Tensorflow(cls, name, sources):
        try:
            import tensorflow as tf
            ext = Extension(name,
                            sources=sources,
                            libraries=['cutensor', 'cudart'],
                            extra_compile_args=tf.sysconfig.get_compile_flags(),
                            extra_link_args=tf.sysconfig.get_link_flags() +
                            tf.sysconfig.get_compile_flags(),
                            define_macros=[('GOOGLE_CUDA', '1')],
                            include_dirs=include_dirs,
                            library_dirs=library_dirs,
                            runtime_library_dirs=library_dirs)
            cls.modules.append(ext)
        except ImportError:
            return None
