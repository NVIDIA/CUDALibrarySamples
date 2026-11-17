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
#include "../common/cublas_reference_geqrf_gels.hpp"
#include "../common/numeric.hpp"

// This example demonstrates how to use cuSolverDx API with Cluster execution to perform QR factorization for a batched, tall-skinny matrix A.
// The matrix is too large to fit into shared memory of a single thread block.
// Cluster execution stores the tiled matrix in distributed shared memory and performs blocked QR factorization,
// using the cluster group features in Hopper and later GPUs.
// The results are compared with the reference values obtained with cuSolver host API.

template<class TSQRHR, class DataType = typename TSQRHR::a_data_type>
__global__ __launch_bounds__(TSQRHR::max_threads_per_block) void tsqrhr_cluster_kernel(DataType*          A,
                                                                                       const unsigned int lda_gmem,
                                                                                       DataType*          tau,
                                                                                       const unsigned int batches) {
    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(TSQRHR);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const auto batch_idx = blockIdx.x / TSQRHR::blocks_per_cluster;
    if (batch_idx >= batches) {
        return;
    }

    extern __shared__ __align__(16) cusolverdx::byte shared_mem[];
    const unsigned thread_id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned block_id  = cooperative_groups::this_cluster().block_rank();

    const unsigned one_batch_elems =
        TSQRHR::a_arrangement == cusolverdx::col_major ? lda_gmem * TSQRHR::n_size : TSQRHR::m_size * lda_gmem;
    auto* Ag = reinterpret_cast<DataType*>(A + batch_idx * one_batch_elems);

    constexpr unsigned NB = cusolverdx::detail::geqrf::cluster_tile_rows(TSQRHR::m_size, TSQRHR::blocks_per_cluster);
    auto [a_smem, tau_smem, workspace_smem] = cusolverdx::shared_memory::slice<DataType, DataType, DataType>(shared_mem,
                                                                                                            alignof(DataType),
                                                                                                            NB * TSQRHR::n_size,
                                                                                                            alignof(DataType),
                                                                                                            TSQRHR::n_size,
                                                                                                            alignof(DataType),
                                                                                                            TSQRHR::workspace_size);

    TSQRHR().load_global_to_dsmem(Ag, lda_gmem, a_smem);

    TSQRHR().execute(a_smem, tau_smem, workspace_smem);
    
    TSQRHR().store_dsmem_to_global(Ag, lda_gmem, a_smem);

    if (block_id == 0) {
        for (int i = thread_id; i < TSQRHR::n_size; i += TSQRHR::max_threads_per_block) {
            tau[batch_idx * TSQRHR::n_size + i] = tau_smem[i];
        }
    }
#endif
}

template<int Arch>
int geqrf_batched_cluster() {
    if constexpr (Arch < 900) {
        printf("Arch %d < 900. Cluster GEQRF requires Hopper or later. Exiting with no error.\n", Arch);
        return 0;
    } else {
        using namespace cusolverdx;

        constexpr unsigned m = 1023;
        constexpr unsigned n = 32;
        constexpr unsigned blocks_per_cluster = (Arch == 1200 || Arch == 1210) ? 8 : 2; // SM 1200 and 1210 has smaller shared memory, so use larger bpc

        using GEQRF = decltype(Size<m, n>() + Precision<float>() + Type<type::complex>() + Function<function::geqrf>() +
                               Arrangement<col_major>() + SM<Arch>() + Cluster() + BlocksPerCluster<blocks_per_cluster>() + BlockDim<128>());

        using data_type      = typename GEQRF::a_data_type;
        using cuda_data_type = typename GEQRF::a_cuda_data_type;

        constexpr auto bpc              = GEQRF::blocks_per_cluster;
        constexpr bool is_col_maj_a     = GEQRF::a_arrangement == col_major;
        constexpr auto lda              = GEQRF::default_lda;
        constexpr auto one_batch_size_A = is_col_maj_a ? lda * n : m * lda;
        constexpr auto batches          = 100;

        printf("Matrix Size %u x %u (M x N), Full Size = %lu bytes,Block Dim = %d, Blocks Per Cluster = %d, Batches = %d\n",
               m,
               n,
               one_batch_size_A * sizeof(data_type),
               GEQRF::block_dim.x,
               bpc,
               batches);

        cudaStream_t stream = nullptr;
        CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::vector<data_type> A(one_batch_size_A * batches);
        common::fillup_random_matrix<data_type>(is_col_maj_a, m, n, A.data(), lda, false, false, -2.0, 2.0, batches);
        const std::vector<data_type> A_orig = A;

        std::vector<data_type> tau(batches * n, common::convert<data_type>(0.0));

        data_type* d_A   = nullptr;
        data_type* d_tau = nullptr;

        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_tau), sizeof(data_type) * tau.size()));

        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

        const auto sm_size = GEQRF::get_shared_memory_size();
        printf("shared memory workspace size needed per thread block = %u bytes\n", sm_size);

        const auto kernel = tsqrhr_cluster_kernel<GEQRF>;
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        CUDA_CHECK_AND_EXIT(common::set_max_dynamic_shared_memory_size(kernel, sm_size));

        cudaLaunchAttribute attribute[1];
        attribute[0].id               = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = bpc;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        cudaLaunchConfig_t config = {};
        config.gridDim            = dim3(batches * bpc, 1, 1);
        config.blockDim           = GEQRF::block_dim;
        config.dynamicSmemBytes   = sm_size;
        config.numAttrs           = 1;
        config.attrs              = attribute;
        config.stream             = stream;

        unsigned lda_arg     = lda;
        unsigned batches_arg = batches;
        void*    args[]      = {reinterpret_cast<void*>(&d_A),
                                reinterpret_cast<void*>(&lda_arg),
                                reinterpret_cast<void*>(&d_tau),
                                reinterpret_cast<void*>(&batches_arg)};

        auto run_kernel = [&](cudaStream_t str) {
            config.stream = str;
            CUDA_CHECK_AND_EXIT(cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(kernel), args));
            CUDA_CHECK_AND_EXIT(cudaGetLastError());
        };
        auto reset = [&](cudaStream_t str) {
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A_orig.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, str));
        };

        const unsigned int warmup_repeats = 1;
        const unsigned int repeats        = 5;

        reset(stream);
        double ms                     = common::measure::execution(run_kernel, reset, warmup_repeats, repeats, stream) / repeats;
        double seconds_per_giga_batch = ms / 1e3 / batches * 1e9;
        double gb_s                   = one_batch_size_A * sizeof(data_type) * 2 / seconds_per_giga_batch;
        double gflops                 = common::get_flops_geqrf<data_type>(m, n) / seconds_per_giga_batch;

        common::print_perf("cuSolverDx-GEQRF-TSQRHR-CGA", batches, m, n, 1, gflops, gb_s, ms, GEQRF::block_dim.x, bpc, GEQRF::tile_size);

        reset(stream);
        run_kernel(stream);
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(A.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(tau.data(), d_tau, sizeof(data_type) * tau.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

        std::vector<data_type> A_ref = A_orig;
        std::vector<data_type> tau_ref(batches * n, common::convert<data_type>(0.0));
        std::vector<data_type> dummy_b;
        const bool             ref_ok = common::reference_cublas_geqrf_gels<data_type, cuda_data_type, false, true>(
            A_ref, dummy_b, tau_ref, m, n, 1, batches, is_col_maj_a);

        int result = 0;
        if (!ref_ok) {
            printf("cuBLAS reference GEQRF failed\n");
            result = 1;
        } else {
            const double a_err   = common::check_error<data_type, data_type>(A.data(), A_ref.data(), A.size());
            const double tau_err = common::check_error<data_type, data_type>(tau.data(), tau_ref.data(), tau.size());
            printf("Compact QR relative error vs cuBLAS GEQRF: %.3e\n", a_err);
            printf("Tau relative error vs cuBLAS GEQRF: %.3e\n", tau_err);

            if (common::is_error_acceptable<data_type>(a_err) && common::is_error_acceptable<data_type>(tau_err)) {
                std::cout << "Success compared to cuBLAS GEQRF result" << std::endl;
            } else {
                std::cout << "Failure compared to cuBLAS GEQRF result" << std::endl;
                result = 1;
            }
        }

        CUDA_CHECK_AND_EXIT(cudaFree(d_A));
        CUDA_CHECK_AND_EXIT(cudaFree(d_tau));
        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
        CUDA_CHECK_AND_EXIT(cudaDeviceReset());

        return result;
    }
}

template<int Arch>
struct geqrf_batched_cluster_functor {
    int operator()() { return geqrf_batched_cluster<Arch>(); }
};

int main() {
    return common::run_example_with_sm<geqrf_batched_cluster_functor>();
}
