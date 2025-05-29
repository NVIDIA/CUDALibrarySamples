#ifndef CUSOLVERDX_EXAMPLE_COMMON_MACROS_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_MACROS_HPP

#ifndef CUDA_CHECK_AND_EXIT
#    define CUDA_CHECK_AND_EXIT(error)                                                                      \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#endif // CUDA_CHECK_AND_EXIT

#ifndef CUSOLVER_CHECK_AND_EXIT
#    define CUSOLVER_CHECK_AND_EXIT(error)                                              \
        {                                                                               \
            auto status = static_cast<cusolverStatus_t>(error);                         \
            if (status != CUSOLVER_STATUS_SUCCESS) {                                    \
                std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                      \
            }                                                                           \
        }
#endif // CUSOLVER_CHECK

#ifndef CUBLAS_CHECK_AND_EXIT
#    define CUBLAS_CHECK_AND_EXIT(error)                                              \
        {                                                                               \
            auto status = static_cast<cublasStatus_t>(error);                         \
            if (status != CUBLAS_STATUS_SUCCESS) {                                    \
                std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                      \
            }                                                                           \
        }
#endif // CUSOLVER_CHECK

#endif // CUSOLVERDX_EXAMPLE_COMMON_MACROS_HPP
