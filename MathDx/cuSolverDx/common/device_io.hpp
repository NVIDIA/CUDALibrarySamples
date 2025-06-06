#ifndef CUSOLVERDX_EXAMPLE_COMMON_DEVICE_IO_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_DEVICE_IO_HPP

#include <cusolverdx.hpp>

namespace common {

    // Convenience wrapper for cusolverdx::copy_2d functions to use in the examples
    template<class Operation, unsigned BPB = 1>
    struct io {
        using data_type = typename Operation::a_data_type;

        static constexpr unsigned int m        = Operation::m_size;
        static constexpr unsigned int n        = Operation::n_size;
        static constexpr unsigned int nrhs     = Operation::k_size;
        static constexpr unsigned int nthreads = Operation::max_threads_per_block;

        //load BPB batches of A from global memory to shared memory
        static inline __device__ void load_a(const data_type* A, const int lda, data_type* As, const int ldas) { 
            cusolverdx::copy_2d<Operation, m, n, cusolverdx::arrangement_of_v_a<Operation>, BPB>(A, lda, As, ldas);
            __syncthreads();
        }

        //store BPB batches of A from shared memory to global memory
        static inline __device__ void store_a(const data_type* As, const int ldas, data_type* A, const int lda) {
            __syncthreads();
            cusolverdx::copy_2d<Operation, m, n, cusolverdx::arrangement_of_v_a<Operation>, BPB>(As, ldas, A, lda);
        }

        //load BPB batches of B from global memory to shared memory
        //Note that the wrapper function cannot be used for unmlq/unmqr function with right size multiplication
        static inline __device__ void load_b(const data_type* B, const int ldb, data_type* Bs, const int ldbs) {
            cusolverdx::copy_2d<Operation, (m > n ? m : n), nrhs, cusolverdx::arrangement_of_v_b<Operation>, BPB>(B, ldb, Bs, ldbs);
            __syncthreads();
        }

        //store BPB batches of B from shared memory to global memory
        //Note that the wrapper function cannot be used for unmlq/unmqr function with right size multiplication
        static inline __device__ void store_b(const data_type* Bs, const int ldbs, data_type* B, const int ldb) {
            __syncthreads();
            cusolverdx::copy_2d<Operation, (m > n ? m : n), nrhs, cusolverdx::arrangement_of_v_b<Operation>, BPB>(Bs, ldbs, B, ldb);
        }
    };

} // namespace common

#endif // CUSOLVERDX_EXAMPLE_COMMON_DEVICE_IO_HPP
