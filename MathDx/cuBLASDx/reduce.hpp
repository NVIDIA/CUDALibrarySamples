#ifndef CUBLASDX_EXAMPLE_REDUCE_HPP
#define CUBLASDX_EXAMPLE_REDUCE_HPP

#include <limits>

namespace example {

namespace reducers {

    template <class ValueType>
    struct limits {
        static constexpr ValueType lowest_value = std::numeric_limits<ValueType>::lowest();
    };

    template <>
    struct limits<__half> {
        static constexpr int lowest_value = -65504;
    };

    // Reduction operators.

    template <class ValueType>
    struct maximum {
        __device__ __forceinline__
        ValueType operator()(ValueType a, ValueType b) const {
            return a > b ? a : b;
        }
        ValueType initial_value = limits<ValueType>::lowest_value;
    };

    template <class ValueType>
    struct addition {
        __device__ __forceinline__
        ValueType operator()(ValueType a, ValueType b) const {
            return a + b;
        }
        ValueType initial_value = 0.;
    };

} // namespace "reducers"


// Row reduction helper.
template <unsigned int LD, unsigned int N, class ValueType, class Reducer>
__device__ __forceinline__
void reduce_row_chunk(ValueType *data, Reducer reducer, unsigned int start, unsigned int m, unsigned int n, ValueType *workspace, ValueType *reduced) {
    // Logical thread block of m x n threads, with each thread processing a chunk of items in its row.
    unsigned int tx = threadIdx.x % m;
    unsigned int ty = threadIdx.x / m;

    // Each thread reduces "size" items.
    unsigned int size = N / n + (N % n > 0);
    ValueType partial = reducer.initial_value;
    for (unsigned int s = ty * size; s < (ty + 1) * size; ++s) {
        if (s < N) {
            partial = reducer(partial, data[s * LD + start + tx]);
        }
    }
    workspace[threadIdx.x] = partial;
    __syncthreads();

    // Cooperative final reduction of upto "n" thread columns.
    unsigned int limit = N / size + (N % size > 0);
    unsigned int leftover = limit & 1;
    for (unsigned int s = limit >> 1; s > 0; s >>= 1) {
        if (ty >= s) continue;
        workspace[ty * m + tx] = reducer(workspace[ty * m + tx], workspace[(ty + s) * m + tx]);
        if (leftover && ty == s - 1) {
            workspace[ty * m + tx] = reducer(workspace[ty * m + tx], workspace[(ty + s + 1) * m + tx]);
        }
        leftover = s & 1;
    }
    __syncthreads();

    // The first thread column stores the reduced results.
    if (ty == 0) {
        reduced[start + tx] = workspace[tx];
    }
    __syncthreads();
}


// Row reduction.
template <unsigned int M, unsigned int N, unsigned int LD, class ValueType, class Reducer>
__device__ __forceinline__
void reduce_row(ValueType *data, Reducer reducer, ValueType *workspace, ValueType *reduced) {

    unsigned int block_size = blockDim.x;
    unsigned int start = 0;

    do {
        unsigned int m = 1, n = block_size;
        while (m <= block_size && start + m <= M) {
            m <<= 1, n >>= 1;
        }
        m >>= 1, n <<= 1;

        reduce_row_chunk<LD, N>(data, reducer, start, m, n, workspace, reduced);
        start += m;
    } while (start < M);
}

} // namespace "example"

#endif // CUBLASDX_EXAMPLE_REDUCE_HPP
