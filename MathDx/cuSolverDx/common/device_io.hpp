#ifndef CUSOLVERDX_EXAMPLE_COMMON_DEVICE_IO_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_DEVICE_IO_HPP

namespace common {
    // Perform cooperative threadblock-wide copy from src
    // (arbitrary memory space) to dst (also arbitrary). This is not
    // an efficient implementation, used just for correctness.
    template<int Size, typename Prec>
    __device__ __forceinline__ void block_copy(const Prec* src, Prec* dst) {
        const auto threads = blockDim.x * blockDim.y * blockDim.z;
        const auto tid     = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;

#pragma unroll
        for (int i = tid; i < Size; i += threads) {
            dst[i] = src[i];
        }
    }

    // Perform single-thread copy from src
    // (arbitrary memory space) to dst (also arbitrary). This is a
    // very efficient implementation, for better results stage through
    // shared memory to achieve higher global memory coalescing.
    template<int Size, typename Prec>
    __device__ __forceinline__ void thread_copy(const Prec* src, Prec* dst) {
#pragma unroll
        for (int i = 0; i < Size; ++i) {
            dst[i] = src[i];
        }
    }

    // this function assumes nBatches % BPB == 0. If not the case, pad nBatches accordingly
    template<class Operation, unsigned BPB = 1>
    struct io {
        using data_type = typename Operation::a_data_type;

        static constexpr unsigned int m        = Operation::m_size;
        static constexpr unsigned int n        = Operation::n_size;
        static constexpr unsigned int nrhs     = Operation::nrhs;
        static constexpr unsigned int nthreads = Operation::max_threads_per_block;

        // Iteration helper to avoid duplicating logic
        // body is a lambda that takes 2 arguments, the indexes in the fast and the slow dimentions, respectively
        template<unsigned int M, unsigned int N, cusolverdx::arrangement Arrange, class Functor>
        static __forceinline__ __device__ void iterate(Functor body) {
            const int tid = threadIdx.x + threadIdx.y * Operation::block_dim.x + threadIdx.z * Operation::block_dim.x * Operation::block_dim.y;
            __builtin_assume(tid < nthreads);

            constexpr unsigned int dim_fast = (Arrange == cusolverdx::arrangement::col_major) ? M : N;
            constexpr unsigned int dim_slow = (Arrange == cusolverdx::arrangement::col_major) ? N : M;

            constexpr unsigned int actual_n      = dim_slow * BPB;
            constexpr unsigned int total_element = M * N * BPB;

            if constexpr (nthreads % dim_fast == 0) {
                constexpr unsigned int stride = nthreads / dim_fast;
                const unsigned int     r      = tid % dim_fast;
                const unsigned int     c      = tid / dim_fast;

                constexpr unsigned cc_end = (actual_n / stride) * stride; // Round down to multiple of stride
                unsigned           cc     = 0;
// Sometimes cc_end is 0, so disable warning about comparing unsigned int with zero
#pragma nv_diag_suppress 186
                for (; cc < cc_end; cc += stride) {
#pragma nv_diag_default 186
                    body(r, cc + c);
                }
                if (cc + c < actual_n) {
                    body(r, cc + c);
                }
            } else {
#pragma unroll
                for (int k = 0; k < total_element; k += nthreads) {
                    if (k + tid < total_element) {
                        unsigned r = (k + tid) % dim_fast;
                        unsigned c = (k + tid) / dim_fast;
                        body(r, c);
                    }
                }
            }
        }

        template<unsigned int M, unsigned int N, cusolverdx::arrangement Arrange = cusolverdx::arrangement_of_v_a<Operation>>
        static inline __device__ void load_matrix(const data_type* A, const int lda, data_type* As, const int ldas) {
            iterate<M, N, Arrange>([&](unsigned r, unsigned c) { As[r + c * ldas] = A[r + c * lda]; });
            __syncthreads();
        }

        //load mxn of A(lda, n) to mxn of As(ldas, n)
        static inline __device__ void load(const data_type* A, const int lda, data_type* As, const int ldas) { load_matrix<m, n>(A, lda, As, ldas); }

        // Load array B(m, 1) to Bs
        static inline __device__ void load_1d(const data_type* B, data_type* Bs) { load_matrix<m, 1>(B, m, Bs, m); }

        // Load array B to Bs
        static inline __device__ void load_rhs(const data_type* B, const int ldb, data_type* Bs, const int ldbs) { load_matrix<n, nrhs, cusolverdx::arrangement_of_v_b<Operation>>(B, ldb, Bs, ldbs); }

        template<unsigned int M, unsigned int N, cusolverdx::arrangement Arrange = cusolverdx::arrangement_of_v_a<Operation>>
        static inline __device__ void store_matrix(const data_type* As, const int ldas, data_type* A, const int lda) {
            __syncthreads();
            iterate<M, N, Arrange>([&](unsigned r, unsigned c) { A[r + c * lda] = As[r + c * ldas]; });
        }

        //store mxn of As(ldas, n) to mxn of A(lda, n)
        static inline __device__ void store(const data_type* As, const int ldas, data_type* A, const int lda) { store_matrix<m, n>(As, ldas, A, lda); }

        // Store array Bs(m, 1) to B
        static inline __device__ void store_1d(const data_type* Bs, data_type* B) { store_matrix<m, 1>(Bs, m, B, m); }

        //store Bs back to B
        static inline __device__ void store_rhs(const data_type* Bs, const int ldbs, data_type* B, const int ldb) {
            store_matrix<n, nrhs, cusolverdx::arrangement_of_v_b<Operation>>(Bs, ldbs, B, ldb);
        }
    };

} // namespace common

#endif // CUSOLVERDX_EXAMPLE_COMMON_KERNELS_HPP
