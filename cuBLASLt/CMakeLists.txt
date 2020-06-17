# 
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

cmake_minimum_required(VERSION 3.10.0)

project(cublasLtSamples)

add_subdirectory(LtSgemm)
add_subdirectory(LtDgemmPresetAlgo)
add_subdirectory(LtIgemmTensor)
add_subdirectory(LtHSHgemmStridedBatchSimple)
add_subdirectory(LtSgemmCustomFind)
add_subdirectory(LtPlanarComplex)
add_subdirectory(LtSgemmSimpleAutoTuning)