# ! /usr/bin/python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
