#ifndef CUBLASDX_EXAMPLE_BLOCK_IO_HPP_
#define CUBLASDX_EXAMPLE_BLOCK_IO_HPP_

#include <cublasdx.hpp>

namespace example {
    // load MxN of A(lda, N) to MxN of As(ldas, N)
    // fast_load/store functions are usually faster than regular versions as they permit loop unrolling
    template<class T, unsigned int M, unsigned int N, unsigned int BlockSize>
    inline __device__ void fast_load(const T* A, const unsigned int lda, T* As, const unsigned int ldas) {
        const unsigned int     tid   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
        constexpr unsigned int ept   = ((M * N) + (BlockSize - 1)) / BlockSize;
        unsigned int           index = tid;
        // Since ept is known at compile time, the compiler can unroll the loop
        for (unsigned int i = 0; i < ept; i++) {
            unsigned int r   = index % M;
            unsigned int c   = index / M;
            if(index < (M*N)) {
                As[r + c * ldas] = A[r + c * lda];
            }
            index += BlockSize;
        }
    }

    // Load two batches of complex<__half> data of size "Size" from shared or global memory to registers as complex<__half2>.
    // The source data is laid out in natural order, first for batch 1 and then for batch 2. The register memory layout follows
    // the interleaved format "(real1, real2), (imag1, imag2), ...", where 1 and 2 represent the first and second batch
    // respectively. This function is useful for performing half-precision FFTs using cuFFTDx.
    template<unsigned int EPT, unsigned int Size, unsigned int Stride, template <class> class T, template <class> class U>
    inline __device__ void load(const T<__half>* A, U<__half2>* thread_data) {
        const unsigned int tid   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
        unsigned int       index = tid;
        for (unsigned int i = 0; i < EPT; i++) {
            if (index < Size) {
                auto v1 = A[index].xy;
                auto v2 = A[index + Size].xy;
                thread_data[i] = {{v1.x, v2.x}, {v1.y,v2.y}};
            }
            index += Stride;
        }
    }


    // Store batched complex<__half2> data from registers to shared or global memory into two batches of complex<__half> data of
    // size "Size". The register memory layout follows the interleaved format "(real1, real2), (imag1, imag2), ...", where
    // 1 and 2 represent the first and second batch respectively. The destination data is laid out in natural order, first for
    // batch 1 and then for batch 2. This function is useful for performing half-precision FFTs using cuFFTDx.
    template<unsigned int EPT, unsigned int Size, unsigned int Stride, template <class> class U, template <class> class T>
    inline __device__ void store(U<__half2>* thread_data, T<__half>* A) {
        const unsigned int tid   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
        unsigned int       index = tid;
        for (unsigned int i = 0; i < EPT; i++) {
            if(index < Size) {
                A[index]        = __lows2half2(thread_data[i].x, thread_data[i].y);
                A[index + Size] = __highs2half2(thread_data[i].x, thread_data[i].y);
            }
            index += Stride;
        }
    }


    // store MxN of As(ldas, N) to MxN of A(lda, N)
    template<class T, unsigned int M, unsigned int N, unsigned int BlockSize>
    inline __device__ void fast_store(const T* As, const unsigned int ldas, T* A, const unsigned int lda) {
        const unsigned int tid      = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
        constexpr unsigned int ept   = ((M * N) + (BlockSize - 1)) / BlockSize;
        unsigned int           index = tid;

        for (unsigned int i = 0; i < ept; i++) {
            unsigned int r = index % M;
            unsigned int c = index / M;
            if(index < (M*N)) {
                A[r + c * lda] = As[r + c * ldas];
            }
            index += BlockSize;
        }
    }

    // load MxN of A(lda, N) to MxN of As(ldas, N)
    template<class T, unsigned int M, unsigned int N>
    inline __device__ void load(const T* A, const unsigned int lda, T* As, const unsigned int ldas) {
        const unsigned int tid      = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
        unsigned int       nthreads = blockDim.x * blockDim.y * blockDim.z;

        for (unsigned int k = tid; k < (M * N); k += nthreads) {
            unsigned int r   = k % M;
            unsigned int c   = k / M;
            As[r + c * ldas] = A[r + c * lda];
        }
    }

    // store MxN of As(ldas, N) to MxN of A(lda, N)
    template<class T, unsigned int M, unsigned int N>
    inline __device__ void store(const T* As, const unsigned int ldas, T* A, const unsigned int lda) {
        const unsigned int tid      = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
        unsigned int       nthreads = blockDim.x * blockDim.y * blockDim.z;

        for (unsigned int k = tid; k < (M * N); k += nthreads) {
            unsigned int r = k % M;
            unsigned int c = k / M;
            A[r + c * lda] = As[r + c * ldas];
        }
    }

    namespace detail {
        template<class T>
        inline __device__ void naive_copy(T* dest, const T* src, unsigned int size) {
            if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
                // Note: This copies values in padding too
                for (unsigned int idx = 0; idx < size; ++idx) {
                    dest[idx] = src[idx];
                }
            }
        }
    } // namespace detail

    template<class BLAS>
    struct io {
        using a_value_type = typename BLAS::a_value_type;
        using b_value_type = typename BLAS::b_value_type;
        using c_value_type = typename BLAS::c_value_type;

        template<class T>
        static inline __device__ void load(T* shared_output, const T* global_input, const unsigned int size) {
            detail::naive_copy(shared_output, global_input, size);
        }

        template<class T>
        static inline __device__ void store(T* global_output, const T* shared_input, const unsigned int size) {
            detail::naive_copy(global_output, shared_input, size);
        }

        template<class T>
        static inline __device__ void load(T*                 shared_output,
                                           const T*  global_input,
                                           const unsigned int m,
                                           const unsigned int n,
                                           const unsigned int ld) {
            detail::naive_copy(shared_output, global_input, (ld * n));
        }

        static inline __device__ void a_load(a_value_type* shared_output, const a_value_type* global_input) {
            constexpr auto m = std::get<0>(BLAS::a_dim);
            constexpr auto n = std::get<1>(BLAS::a_dim);
            example::load<a_value_type, m, n>(global_input, BLAS::lda, shared_output, BLAS::lda);
        }

        static inline __device__ void b_load(b_value_type* shared_output, const b_value_type* global_input) {
            constexpr auto m = std::get<0>(BLAS::b_dim);
            constexpr auto n = std::get<1>(BLAS::b_dim);
            example::load<b_value_type, m, n>(global_input, BLAS::ldb, shared_output, BLAS::ldb);
        }

        static inline __device__ void c_load(c_value_type* shared_output, const c_value_type* global_input) {
            constexpr auto m = std::get<0>(BLAS::c_dim);
            constexpr auto n = std::get<1>(BLAS::c_dim);
            example::load<c_value_type, m, n>(global_input, BLAS::ldc, shared_output, BLAS::ldc);
        }

        template<unsigned int BlockSize>
        static inline __device__ void a_fast_load(a_value_type* shared_output, const a_value_type* global_input) {
            constexpr auto m = std::get<0>(BLAS::a_dim);
            constexpr auto n = std::get<1>(BLAS::a_dim);
            example::fast_load<a_value_type, m, n, BlockSize>(global_input, BLAS::lda, shared_output, BLAS::lda);
        }

        template<unsigned int BlockSize>
        static inline __device__ void b_fast_load(b_value_type* shared_output, const b_value_type* global_input) {
            constexpr auto m = std::get<0>(BLAS::b_dim);
            constexpr auto n = std::get<1>(BLAS::b_dim);
            example::fast_load<b_value_type, m, n, BlockSize>(global_input, BLAS::ldb, shared_output, BLAS::ldb);
        }

        template<unsigned int BlockSize>
        static inline __device__ void c_fast_load(c_value_type* shared_output, const c_value_type* global_input) {
            constexpr auto m = std::get<0>(BLAS::c_dim);
            constexpr auto n = std::get<1>(BLAS::c_dim);
            example::fast_load<c_value_type, m, n, BlockSize>(global_input, BLAS::ldc, shared_output, BLAS::ldc);
        }

        template<class T>
        static inline __device__ void store(c_value_type*      global_output,
                                            const T*           shared_input,
                                            const unsigned int m,
                                            const unsigned int n,
                                            const unsigned int ld) {
            detail::naive_copy(global_output, reinterpret_cast<const c_value_type*>(shared_input), (ld * n));
        }

        static inline __device__ void a_store(a_value_type* global_output, const a_value_type* shared_input) {
            constexpr auto m = std::get<0>(BLAS::a_dim);
            constexpr auto n = std::get<1>(BLAS::a_dim);
            example::store<a_value_type, m, n>(shared_input, BLAS::lda, global_output, BLAS::lda);
        }

        static inline __device__ void b_store(b_value_type* global_output, const b_value_type* shared_input) {
            constexpr auto m = std::get<0>(BLAS::b_dim);
            constexpr auto n = std::get<1>(BLAS::b_dim);
            example::store<b_value_type, m, n>(shared_input, BLAS::ldb, global_output, BLAS::ldb);
        }

        static inline __device__ void c_store(c_value_type* global_output, const c_value_type* shared_input) {
            constexpr auto m = std::get<0>(BLAS::c_dim);
            constexpr auto n = std::get<1>(BLAS::c_dim);
            example::store<c_value_type, m, n>(shared_input, BLAS::ldc, global_output, BLAS::ldc);
        }

        template<unsigned int BlockSize>
        static inline __device__ void c_fast_store(c_value_type* global_output, const c_value_type* shared_input) {
            constexpr auto m = std::get<0>(BLAS::c_dim);
            constexpr auto n = std::get<1>(BLAS::c_dim);
            example::fast_store<c_value_type, m, n, BlockSize>(shared_input, BLAS::ldc, global_output, BLAS::ldc);
        }

    };
} // namespace example

#endif // CUBLASDX_EXAMPLE_BLOCK_IO_HPP_
