/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cusolverdx_cluster.hpp>

#include "../common/cudart.hpp"
#include "../common/error_checking.hpp"
#include "../common/random.hpp"
#include "../common/example_sm_runner.hpp"
#include "../common/device_io.hpp"
#include "../common/measure.hpp"
#include "../common/print.hpp"
#include "../common/cusolver_reference_cholesky.hpp"

// This example demonstrates how to use cuSolverDx API with Cluster execution to perform Cholesky factorization for a batched, symmetric, positive-definite matrix A.
// The matrix is too large to fit into shared memory of a single thread block.
// Cluster execution stores the tiled matrix triangle in distributed shared memory and performs blocked Cholesky factorization, using the cluster group features in Hopper and later GPUs.
// The results are compared with the reference values obtained with cuSolver host API.

template<class POTRF, class DataType = typename POTRF::a_data_type>
__global__ __launch_bounds__(POTRF::max_threads_per_block) void potrf_kernel(DataType*                    A,
                                                                             const unsigned int           lda_gmem,
                                                                             typename POTRF::status_type* info,
                                                                             const unsigned int           batches) {
    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(POTRF);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const auto batch_idx = blockIdx.x / POTRF::blocks_per_cluster;
    if (batch_idx >= batches)
        return;

    extern __shared__ __align__(sizeof(DataType)) cusolverdx::byte shared_mem[];
    auto*          tiles_s     = reinterpret_cast<DataType*>(shared_mem);

    constexpr auto tile_size   = POTRF::tile_size;
    const unsigned thread_id   = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    auto* Ag = reinterpret_cast<DataType*>(A + lda_gmem * POTRF::m_size * batch_idx);

    cusolverdx::detail::potrf::load_global_to_cluster_tiles<DataType,
                                                            POTRF::m_size,
                                                            tile_size,
                                                            POTRF::fill_mode,
                                                            POTRF::a_arrangement,
                                                            POTRF::max_threads_per_block,
                                                            POTRF::blocks_per_cluster>(Ag, lda_gmem, tiles_s, thread_id);

    POTRF().execute(reinterpret_cast<DataType*>(tiles_s), &info[batch_idx]);

    cusolverdx::detail::potrf::store_cluster_tiles_to_global<DataType,
                                                             POTRF::m_size,
                                                             tile_size,
                                                             POTRF::fill_mode,
                                                             POTRF::a_arrangement,
                                                             POTRF::max_threads_per_block,
                                                             POTRF::blocks_per_cluster>(Ag, lda_gmem, tiles_s, thread_id);
#endif
}

template<int Arch>
int potrf_batched_cluster() {

    if constexpr (Arch < 900) {
        printf("Arch %d < 900. Cluster POTRF requires Hopper or later. Exiting with no error.\n", Arch);
        return 0;
    } else {
        using namespace cusolverdx;

        using POTRF = decltype(Size<250>() + Precision<double>() + Type<type::real>() + Function<function::potrf>() +
                               FillMode<fill_mode::upper>() + Arrangement<row_major>() + SM<Arch>() + Cluster() +
                               BlocksPerCluster<4>() + TileSize<64>() + BlockDim<256>());

        using data_type = typename POTRF::a_data_type;
        using cuda_data_type = typename POTRF::a_cuda_data_type;

        constexpr auto m            = POTRF::m_size;
        constexpr auto bpc          = POTRF::blocks_per_cluster;
        constexpr bool is_col_maj_a = arrangement_of_v_a<POTRF> == arrangement::col_major;

        constexpr auto lda              = m;
        const auto     one_batch_size_A = lda * m;
        const auto     batches          = 200;

        printf("Matrix Size m = %d, Block Dim = %d, Blocks Per Cluster = %d, Batches = %d\n", m, POTRF::block_dim.x, bpc, batches);

        cudaStream_t stream = nullptr;
        CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::vector<data_type> A(one_batch_size_A * batches);
        std::vector<data_type> L(one_batch_size_A * batches);
        common::fillup_random_diagonal_dominant_matrix<data_type>(
            arrangement_of_v_a<POTRF> == col_major, m, m, A.data(), lda, false, 2, 4, batches);

        std::vector<int> info(batches, 0);
        data_type*       d_A    = nullptr; /* device copy of A */
        int*             d_info = nullptr; /* error info */


        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * batches));

        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

        // Increase max dynamic shared memory for the kernel if needed.
        const auto sm_size = POTRF::shared_memory_size;
        printf("shared memory workspace size needed for POTRF cluster execution = %u bytes, Matrix size = %lu bytes\n", sm_size, m * m * sizeof(data_type));

        const auto kernel = potrf_kernel<POTRF>;
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_size));

        //Invokes kernel
        cudaLaunchAttribute attribute[1];
        attribute[0].id               = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = bpc;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        cudaLaunchConfig_t config = {};
        config.gridDim            = dim3(batches * bpc, 1, 1);
        config.blockDim           = POTRF::block_dim;
        config.dynamicSmemBytes   = sm_size;
        config.numAttrs           = 1;
        config.attrs              = attribute;
        config.stream             = stream;

        // Pass arguments to the kernel, args is void*, so each argument needs to cast to void*
        unsigned lda_arg     = lda;
        unsigned batches_arg = batches;
        void*    args[]      = {reinterpret_cast<void*>(&d_A),
                                reinterpret_cast<void*>(&lda_arg),
                                reinterpret_cast<void*>(&d_info),
                                reinterpret_cast<void*>(&batches_arg)};

        auto run_kernel = [&](cudaStream_t str) {
            config.stream = str;
            CUDA_CHECK_AND_EXIT(cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(kernel), args));
            CUDA_CHECK_AND_EXIT(cudaGetLastError());
        };
        auto reset = [&](cudaStream_t str) {
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, str));
            CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_info, 0, sizeof(int) * batches, str));
        };

        const unsigned int warmup_repeats = 1;
        const unsigned int repeats        = 5;

        reset(stream);
        double ms                     = common::measure::execution(run_kernel, reset, warmup_repeats, repeats, stream) / repeats;
        double seconds_per_giga_batch = ms / 1e3 / batches * 1e9;
        double gb_s                   = one_batch_size_A * sizeof(data_type) * 2 / seconds_per_giga_batch;
        double gflops                 = common::get_flops_potrf<data_type>(m) / seconds_per_giga_batch;

        common::print_perf("cuSolverDx-POTRF-CGA", batches, m, m, 1, gflops, gb_s, ms, POTRF::block_dim.x, bpc);

        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * batches, cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
        int result = 0;
        if (std::accumulate(info.begin(), info.end(), 0) != 0) {
            std::cout << "non-zero d_info returned for at least one of the batches after cuSolverDx kernel \n";
            for (int j = 0; j < batches; j++) {
                if (info[j] != 0)
                    std::cout << "info[" << j << "]=" << info[j] << std::endl;
            }
            result = -1;
        } else {
            //=======================================================
            // cuSolver reference with potrfBatched
            //=======================================================
            std::vector<data_type> dummy_B;
            common::reference_cusolver_cholesky<data_type, cuda_data_type, false /* do_solver */, true /* check_factor_perf */>(
                A,
                dummy_B,
                info.data(),
                m,
                1,
                batches,
                (fill_mode_of_v<POTRF> == fill_mode::lower), /* is_lower? */
                is_col_maj_a,
                true,
                batches);

            auto total_relative_error = common::check_error<data_type, data_type>(L.data(), A.data(), batches * one_batch_size_A);
            printf("BATCHED POTRF: relative error of A between cuSolverDx and cuSolver results: = %e\n", total_relative_error);

            if (common::is_error_acceptable<data_type>(total_relative_error)) {
                std::cout << "Success compared to cuSolver potrfBatched Result " << std::endl;
            } else {
                std::cout << "Failure compared to cuSolver potrfBatched Result " << std::endl;
                result = 1;
            }
        }

        CUDA_CHECK_AND_EXIT(cudaFree(d_A));
        CUDA_CHECK_AND_EXIT(cudaFree(d_info));
        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

        CUDA_CHECK_AND_EXIT(cudaDeviceReset());

        return result;
    }
}

template<int Arch>
struct potrf_batched_cluster_functor {
    int operator()() { return potrf_batched_cluster<Arch>(); }
};


int main() {
    return common::run_example_with_sm<potrf_batched_cluster_functor>();
}
