#! /usr/bin/python
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

MAJOR = 0
MINOR = 1
PATCH = 0

VERSION = (MAJOR, MINOR, PATCH)

__version__ = '.'.join(map(str, VERSION))

__package_name__ = 'cutensor-python'
__description__ = 'PyTorch and Tensorflow Python bindings for cuTENSOR',
__homepage__ = 'https://developer.nvidia.com/cutensor',
__download_url__ = 'https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuTENSOR/cutensor',
__license__ = 'BSD'
