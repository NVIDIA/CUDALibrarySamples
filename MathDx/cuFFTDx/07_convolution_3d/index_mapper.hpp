#ifndef CUFFTDX_EXAMPLE_3D_INDEX_MAPPER_HPP
#define CUFFTDX_EXAMPLE_3D_INDEX_MAPPER_HPP

namespace example {
    // clang-format off
    // Index mapper is a structure allowing to create recursive layouts,
    // which can be used for easy offset calculation in multidimensionally
    // batches executions.

    // Index computation:
    // Let's assume the following layout:
    //
    // auto layout = index_mapper<int_pair<5, 1>,
    //                            int_pair<4, 15>,
    //                            int_pair<3, 5>,
    //                            int_pair<2, 60>>{};
    // It can be addressed in 2 modes:
    //   1. Natural
    //      To get a natural coordinate we simply perform a dot product
    //      of coordinate with strides:
    //      const auto index = layout(0, 0, 1, 0);
    //         (0 * 1 + 0 * 15 + 1 * 5 + 0 * 60 == 5)
    //   2. Flattened
    //      To get a flattened result we module and divide until last,
    //      for example for flat 20:
    //          First coordinate: 20 % 5 = 0, next flat -> 20 / 5 = 4
    //          Second coordinate: 4 % 4 = 0, next flat -> 4 / 4 = 1
    //          Third coordinate: 1 % 3 = 1, next flat -> 1 / 3 = 0
    //          Fourth coordinate: 0 % 2 = 0
    //      So the total natural coordinate for flat 20 is:
    //          (0, 0, 1, 0). Now we find offset as in previous point
    //      layout(20) == layout(0, 0, 1, 0)
    //

    // Example of use:
    // Let's assume we have 8 batches of 32-point FFT in shared memory.
    // int_pairs are (Dim, Stride), so:
    // using layout = index_mapper<int_pair<32, 1>,
    //                             int_pair<8, 32>>;
    // Accessing element 0 of all batches would cause 8-way shared memory
    // bank conflicts, so we can pad with 1 element in the batch dimension:
    // using layout = index_mapper<int_pair<32, 1>,
    //                             int_pair<8, 33>>;
    // clang-format on

    template<int Dim, int Stride>
    struct int_pair {
        static constexpr int size = Dim;

        // Translates index to memory offset, based on provided layout.
        // Please refer to index_mapper explanation at the top of this
        // file.
        __device__ __host__ __forceinline__ size_t operator()(int id) {
            return id * Stride;
        }
    };

    template<typename... Ts>
    struct index_mapper;

    template<typename LastDim>
    struct index_mapper<LastDim> {
        static constexpr int size = LastDim::size;

        // Variadic recursive call base case for only 1 dimension.
        // Translates index to memory offset, based on provided layout.
        // Please refer to index_mapper explanation at the top of this
        // file.
        __device__ __host__ __forceinline__ size_t operator()(int id) {
            return LastDim {}(id);
        }
    };

    template<typename ThisDim, typename... NextDims>
    struct index_mapper<ThisDim, NextDims...> {
        static constexpr int size = (ThisDim::size * ... * NextDims::size);

        // Flat coordinate addressing
        __device__ __host__ __forceinline__
            size_t
            operator()(int id) {
            constexpr int this_dim_size = ThisDim::size;
            return ThisDim {}(id % this_dim_size) + index_mapper<NextDims...>()(id / this_dim_size);
        }

        // Natural coordinate addressing
        template<typename... Indices>
        __device__ __host__ __forceinline__
            size_t
            operator()(int id, Indices... indices) {
            static_assert(sizeof...(Indices) == sizeof...(NextDims));
            return ThisDim {}(id) + index_mapper<NextDims...>()(indices...);
        }
    };
} // namespace example

#endif // CUFFTDX_EXAMPLE_3D_INDEX_MAPPER_HPP
