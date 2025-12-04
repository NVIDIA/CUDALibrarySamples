# SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.10.0)

project(cublasLtSamples)

add_subdirectory(LtDgemmEmulated)
add_subdirectory(LtSgemm)
add_subdirectory(LtFp8Matmul)
add_subdirectory(LtBlk128x128Fp8Matmul)
add_subdirectory(LtMxfp8Matmul)
add_subdirectory(LtNvfp4Matmul)
add_subdirectory(LtDgemmPresetAlgo)
add_subdirectory(LtIgemmTensor)
add_subdirectory(LtHSHgemmStridedBatchSimple)
add_subdirectory(LtHSHgemmPointerArrayBatchSimple)
add_subdirectory(LtHSHgemmGroupedSimple)
add_subdirectory(LtSgemmCustomFind)
add_subdirectory(LtFp8CustomFind)
add_subdirectory(LtPlanarComplex)
add_subdirectory(LtSgemmSimpleAutoTuning)