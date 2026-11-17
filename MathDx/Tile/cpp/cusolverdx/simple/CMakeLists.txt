# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(example IN ITEMS cholesky_factor cholesky_solve lu_solve qr least_squares_solve eig_hermitian factor_reuse tridiagonal_solve svd generalized_eigh eig_hermitian_tridiagonal lq modified_lu_factor svd_bidiagonal)
    mathdx_add_tile_example(${example} "${example}.cu")
    target_link_libraries(${example} PRIVATE mathdx_tile::cusolverdx CUDA::cudart)
endforeach()
