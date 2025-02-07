#ifndef CUSOLVERDX_EXAMPLE_COMMON_MEASURE_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_MEASURE_HPP

namespace common {

    struct measure {
        // Returns execution time in ms.
        template<typename Kernel>
        static float execution(Kernel&& kernel, const unsigned int warm_up_runs, const unsigned int runs, cudaStream_t stream) {
            cudaEvent_t startEvent, stopEvent;
            CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvent));
            CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvent));
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            for (unsigned int i = 0; i < warm_up_runs; i++) {
                kernel(stream);
            }

            CUDA_CHECK_AND_EXIT(cudaGetLastError());
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvent, stream));
            for (unsigned int i = 0; i < runs; i++) {
                kernel(stream);
            }
            CUDA_CHECK_AND_EXIT(cudaEventRecord(stopEvent, stream));
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            float time;
            CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time, startEvent, stopEvent));
            CUDA_CHECK_AND_EXIT(cudaEventDestroy(startEvent));
            CUDA_CHECK_AND_EXIT(cudaEventDestroy(stopEvent));
            return time;
        }

        // Returns execution time in ms.
        // Takes a reset function to allow testing in-place functions
        template<typename Kernel, typename Reset>
        static float execution(Kernel&& kernel, Reset&& reset, const unsigned int warm_up_runs, const unsigned int runs, cudaStream_t stream) {
            std::vector<cudaEvent_t> startEvents (runs), stopEvents(runs);
            for (int i = 0; i < runs; ++i) {
                CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvents[i]));
                CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvents[i]));
            }
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            for (unsigned int i = 0; i < warm_up_runs; i++) {
                kernel(stream);
                reset(stream);
            }

            CUDA_CHECK_AND_EXIT(cudaGetLastError());
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            for (unsigned int i = 0; i < runs; i++) {
                CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvents[i], stream));
                kernel(stream);
                CUDA_CHECK_AND_EXIT(cudaEventRecord(stopEvents[i], stream));
                if (i+1 != runs) {
                    reset(stream);
                }
            }
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            float total_time = 0;
            for (int i = 0; i < runs; ++i) {
                float time;
                CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time, startEvents[i], stopEvents[i]));
                total_time += time;

                CUDA_CHECK_AND_EXIT(cudaEventDestroy(startEvents[i]));
                CUDA_CHECK_AND_EXIT(cudaEventDestroy(stopEvents[i]));
            }
            return total_time;
        }
    };

    //==================================
    // GFLOPS calculation
    //==================================

    template<typename DataType>
    double get_flops_trsm(const unsigned int n, const unsigned int nrhs) {
        auto fmuls = nrhs * n * (n + 1) / 2.;
        auto fadds = nrhs * n * (n - 1) / 2.;

        if constexpr (common::is_complex<DataType>()) {
            return 6. * fmuls + 2. * fadds;
        } else {
            return fmuls + fadds;
        }
    }

    // https://github.com/icl-utk-edu/magma/blob/master/testing/flops.h
    template<typename DataType>
    double get_flops_potrf(const unsigned int n) {
        auto fmuls = n * (((1. / 6.) * n + 0.5) * n + (1. / 3.));
        auto fadds = n * (((1. / 6.) * n) * n - (1. / 6.));

        if constexpr (common::is_complex<DataType>()) {
            return 6. * fmuls + 2. * fadds;
        } else {
            return fmuls + fadds;
        }
    }

    template<typename DataType>
    double get_flops_potrs(const unsigned int n, const unsigned int nrhs) {
        return 2 * get_flops_trsm<DataType>(n, nrhs);
    }

    template<typename DataType>
    double get_flops_getrf(const unsigned int m, const unsigned int n) {
        // FLOP calculation depends on whether m or n is smaller
        auto mn = min(m, n);
        auto mx = max(m, n);

        auto fmuls = 0.5 * mn * (mn * (mx - mn / 3. - 1) + mx) + 2. * mn / 3.;
        auto fadds = 0.5 * mn * (mn * (mx - mn / 3.)     - mx) +      mn / 6.;

        if constexpr (common::is_complex<DataType>()) {
            return 6. * fmuls + 2. * fadds;
        } else {
            return fmuls + fadds;
        }
    }

    inline void print_perf(const std::string msg, const unsigned int batches, const unsigned int M, const unsigned int N, const unsigned int nrhs, const double gflops, const double gb_s, const double ms) {
        printf("%-30s %10u %5u %5u %5u  %7.2f GFLOP/s, %7.2f GB/s, %7.2f ms\n", msg.c_str(), batches, M, N, nrhs, gflops, gb_s, ms);
    }
} // namespace common

#endif
