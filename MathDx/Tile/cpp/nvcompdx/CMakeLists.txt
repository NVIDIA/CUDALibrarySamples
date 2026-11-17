# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(example IN ITEMS lz4_tile lz4_gmem ans_tile ans_gmem)
    mathdx_add_tile_example(${example} "${example}.cu")
    target_link_libraries(${example} PRIVATE mathdx_tile::nvcompdx CUDA::cudart)
endforeach()
