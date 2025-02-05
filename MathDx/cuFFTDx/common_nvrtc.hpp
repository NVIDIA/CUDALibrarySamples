
#ifndef CUFFTDX_EXAMPLE_COMMON_NVRTC_HPP
#define CUFFTDX_EXAMPLE_COMMON_NVRTC_HPP

#include <cstdlib>
#include "common.hpp"

#define NVRTC_SAFE_CALL(x)                                                                            \
    do {                                                                                              \
        nvrtcResult result = x;                                                                       \
        if (result != NVRTC_SUCCESS) {                                                                \
            std::cerr << "\nerror: " #x " failed with error " << nvrtcGetErrorString(result) << '\n'; \
            exit(1);                                                                                  \
        }                                                                                             \
    } while (0)

#ifndef CU_CHECK_AND_EXIT
#    define CU_CHECK_AND_EXIT(error)                                                                        \
        {                                                                                                   \
            auto status = static_cast<CUresult>(error);                                                     \
            if (status != CUDA_SUCCESS) {                                                                   \
                const char * pstr; cuGetErrorString(status, &pstr);                                         \
                std::cout << pstr << " " << __FILE__ << ":" << __LINE__ << std::endl;                      \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#endif // CU_CHECK_AND_EXIT

namespace example {
    namespace nvrtc {
        template<class Type>
        inline Type get_global_from_module(CUmodule module, const char* name) {
            CUdeviceptr value_ptr;
            size_t value_size;
            CU_CHECK_AND_EXIT(cuModuleGetGlobal(&value_ptr, &value_size, module, name));
            Type value_host;
            CU_CHECK_AND_EXIT(cuMemcpyDtoH(&value_host, value_ptr, value_size));
            return value_host;
        }

        inline std::vector<std::string> get_cufftdx_include_dirs() {
#ifndef CUFFTDX_INCLUDE_DIRS
            return std::vector<std::string>();
#endif
            std::vector<std::string> cufftdx_include_dirs_array;
            {
                std::string cufftdx_include_dirs = CUFFTDX_INCLUDE_DIRS;
                std::string delim                = ";";
                size_t      start                = 0U;
                size_t      end                  = cufftdx_include_dirs.find(delim);
                while (end != std::string::npos) {
                    cufftdx_include_dirs_array.push_back("--include-path=" +
                                                         cufftdx_include_dirs.substr(start, end - start));
                    start = end + delim.length();
                    end   = cufftdx_include_dirs.find(delim, start);
                }
                cufftdx_include_dirs_array.push_back("--include-path=" +
                                                     cufftdx_include_dirs.substr(start, end - start));
            }
            #ifdef COMMONDX_INCLUDE_DIR
            {
                cufftdx_include_dirs_array.push_back("--include-path=" + std::string(COMMONDX_INCLUDE_DIR));
            }
            #endif
            {
                const char* env_ptr = std::getenv("CUFFTDX_EXAMPLE_CUFFTDX_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    cufftdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                }
            }
            {
                const char* env_ptr = std::getenv("CUFFTDX_EXAMPLE_CUTLASS_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    cufftdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                } else {
                    cufftdx_include_dirs_array.push_back("-DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY");
                }
            }
            {
                const char* env_ptr = std::getenv("CUFFTDX_EXAMPLE_COMMONDX_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    cufftdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                }
            }
            {
                const char* env_ptr = std::getenv("CUFFTDX_EXAMPLE_CUDA_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    cufftdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                }
            }
            return cufftdx_include_dirs_array;
        }

        inline unsigned get_device_architecture(int device) {
            int major = 0;
            int minor = 0;
            CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
            CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
            return major * 10 + minor;
        }

        inline std::string get_device_architecture_option(int device) {
            std::string gpu_architecture_option = "--gpu-architecture=compute_" + std::to_string(get_device_architecture(device));
            return gpu_architecture_option;
        }

        inline void print_program_log(const nvrtcProgram prog) {
            size_t log_size;
            NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));
            char* log = new char[log_size];
            NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
            std::cout << log << '\n';
            delete[] log;
        }
    } // namespace nvrtc
} // namespace example

#endif // CUFFTDX_EXAMPLE_COMMON_NVRTC_HPP
