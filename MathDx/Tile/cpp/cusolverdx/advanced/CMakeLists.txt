# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Larger, batched examples. Both check themselves against the batched host APIs
# and report throughput against them, so they link cuBLAS and cuSolver from the
# CUDA toolkit in addition to their Tile tilelibs.
#
# Skipped on Windows: a Tile kernel plus a cuBLAS/cuSolver handle crashes in the
# driver (nvgpucomp64.dll) at process exit, after the results are printed.
# FIXME: build on Windows again once NVBug 6909101 is fixed.
if(NOT WIN32)
    mathdx_add_tile_example(reg_least_squares reg_least_squares.cu)
    target_link_libraries(reg_least_squares PRIVATE mathdx_tile::cusolverdx CUDA::cudart CUDA::cublas CUDA::cusolver)

    mathdx_add_tile_example(blocked_potrf blocked_potrf.cu)
    target_link_libraries(blocked_potrf PRIVATE mathdx_tile::cublasdx mathdx_tile::cusolverdx CUDA::cudart CUDA::cublas CUDA::cusolver)
endif()
