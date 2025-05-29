#ifndef CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_LU_HPP
#define CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_LU_HPP

namespace common {
    template<typename T, typename cuda_data_type, bool do_solver = false>
    bool reference_cusolver_lu(std::vector<T>&    A,
                               std::vector<T>&    B,
                               int*               info,
                               const unsigned int m,
                               const unsigned int n,
                               const unsigned int nrhs           = 1,
                               const unsigned int padded_batches = 1,
                               const bool         is_pivot       = false,
                               bool               is_col_major_a = true,
                               bool               is_col_major_b = true,
                               bool               is_trans_a     = false,
                               int64_t*           ipiv           = nullptr,
                               const unsigned int actual_batches = 0) {

        const unsigned int a_size = A.size() / padded_batches;
        const unsigned int lda    = a_size / n;
        const unsigned int mn = min(m, n);

        [[maybe_unused]] const unsigned int b_size = B.size() / padded_batches;
        [[maybe_unused]] const unsigned int ldb    = b_size / nrhs; // ldb if b is column major

        const unsigned batches = (actual_batches == 0) ? padded_batches : actual_batches;

        cudaStream_t stream = nullptr;
        CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        cusolverDnHandle_t cusolverH = NULL;
        cusolverDnParams_t params    = nullptr;
        CUSOLVER_CHECK_AND_EXIT(cusolverDnCreate(&cusolverH));
        CUSOLVER_CHECK_AND_EXIT(cusolverDnSetStream(cusolverH, stream));
        CUSOLVER_CHECK_AND_EXIT(cusolverDnCreateParams(&params));

        // if row major, transpose the input A
        if (!is_col_major_a) {
            transpose_matrix<T>(A, lda, n, batches); // fast, second_fast, batch -> after transpose, swap fast and second_fast
        }
        if (!is_col_major_b && do_solver && nrhs > 1) {
            transpose_matrix<T>(B, ldb, nrhs, batches); // fast, second_fast, batch -> after transpose, swap fast and second_fast
        }

        [[maybe_unused]] cublasOperation_t trans = (is_trans_a) ? (common::is_complex<T>() ? CUBLAS_OP_C : CUBLAS_OP_T) : CUBLAS_OP_N;

        // d_info
        int* d_info = nullptr;
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));
        CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_info, 3, sizeof(int), stream));
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

        T* d_A = nullptr; /* device copy of A */
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(T) * a_size));
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(T) * a_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

        [[maybe_unused]] T* d_B = nullptr;
        if constexpr (do_solver) {
            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(T) * b_size));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, B.data(), sizeof(T) * b_size, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
        }

        int64_t* d_ipiv_64 = nullptr;
        if (is_pivot) {
            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_ipiv_64), sizeof(int64_t) * mn));
        }

        size_t            workspaceInBytesOnDevice = 0;       /* size of workspace */
        void*             d_work                   = nullptr; /* device workspace for getrf */
        size_t            workspaceInBytesOnHost   = 0;       /* size of workspace */
        std::vector<char> h_work;                             /* host workspace for getrf */

        // query working space
        CUSOLVER_CHECK_AND_EXIT(cusolverDnXgetrf_bufferSize(cusolverH,
                                                            params,
                                                            int64_t(m),
                                                            int64_t(n),
                                                            common::traits<cuda_data_type>::cuda_data_type,
                                                            d_A,
                                                            lda,
                                                            common::traits<cuda_data_type>::cuda_data_type,
                                                            &workspaceInBytesOnDevice,
                                                            &workspaceInBytesOnHost));

        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_work), workspaceInBytesOnDevice));
        if (0 < workspaceInBytesOnHost) {
            h_work.resize(workspaceInBytesOnHost);
        }


        // LU factorization one batch at a time
        for (unsigned int batch = 0; batch < batches; batch++) {
            if (batch > 0) {
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, &(A[a_size * batch]), sizeof(T) * a_size, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
            }
            CUSOLVER_CHECK_AND_EXIT(cusolverDnXgetrf(cusolverH,
                                                     params,
                                                     int64_t(m),
                                                     int64_t(n),
                                                     common::traits<cuda_data_type>::cuda_data_type,
                                                     d_A,
                                                     lda,
                                                     d_ipiv_64,
                                                     common::traits<cuda_data_type>::cuda_data_type,
                                                     d_work,
                                                     workspaceInBytesOnDevice,
                                                     h_work.data(),
                                                     workspaceInBytesOnHost,
                                                     d_info));

            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&(A[a_size * batch]), d_A, sizeof(T) * a_size, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&info[batch], d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
            if (is_pivot) {
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&ipiv[batch * mn], d_ipiv_64, sizeof(int64_t) * mn, cudaMemcpyDeviceToHost, stream));
            }

            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            //printf("after cusolverDnXgetrf: info = %d\n", info[batch]);
            if (0 > info[batch]) {
                printf("%d-th parameter is wrong \n", -info[batch]);
                return false;
            }

            if constexpr (do_solver) {
                if (batch > 0) {
                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, &B[b_size * batch], sizeof(T) * b_size, cudaMemcpyHostToDevice, stream));
                    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
                }
                CUSOLVER_CHECK_AND_EXIT(cusolverDnXgetrs(
                    cusolverH, params, trans, int64_t(m), nrhs, common::traits<cuda_data_type>::cuda_data_type, d_A, lda, d_ipiv_64, common::traits<cuda_data_type>::cuda_data_type, d_B, ldb, d_info));

                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&B[b_size * batch], d_B, sizeof(T) * b_size, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
            }
        } // end batch loop

        /* free resources */
        CUDA_CHECK_AND_EXIT(cudaFree(d_A));
        CUDA_CHECK_AND_EXIT(cudaFree(d_info));
        CUDA_CHECK_AND_EXIT(cudaFree(d_work));
        if (is_pivot) {
            CUDA_CHECK_AND_EXIT(cudaFree(d_ipiv_64));
        }
        if constexpr (do_solver) {
            CUDA_CHECK_AND_EXIT(cudaFree(d_B));
        }

        // if row major, transpose the result A
        if (!is_col_major_a) {
            transpose_matrix<T>(A, n, lda, batches); // fast, second fast, batch
        }
        if (!is_col_major_b && do_solver && nrhs > 1) {
            transpose_matrix<T>(B, nrhs, ldb, batches); // fast, second fast, batch
        }

        CUSOLVER_CHECK_AND_EXIT(cusolverDnDestroy(cusolverH));
        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
        return true;
    }

} // namespace common

#endif // CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_LU_HPP
