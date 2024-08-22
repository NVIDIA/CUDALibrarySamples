
#ifndef CUBLASDX_EXAMPLE_COMMON_NVRTC_HPP
#define CUBLASDX_EXAMPLE_COMMON_NVRTC_HPP

#include <cstdlib>
#define CUBLASDX_EXAMPLE_NVRTC
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

        inline std::vector<std::string> get_cublasdx_include_dirs() {
#ifndef CUBLASDX_INCLUDE_DIRS
            return std::vector<std::string>();
#endif
            std::vector<std::string> cublasdx_include_dirs_array;
            {
                std::string cublasdx_include_dirs = CUBLASDX_INCLUDE_DIRS;
                std::string delim                = ";";
                size_t      start                = 0U;
                size_t      end                  = cublasdx_include_dirs.find(delim);
                while (end != std::string::npos) {
                    cublasdx_include_dirs_array.push_back("--include-path=" +
                                                         cublasdx_include_dirs.substr(start, end - start));
                    start = end + delim.length();
                    end   = cublasdx_include_dirs.find(delim, start);
                }
                cublasdx_include_dirs_array.push_back("--include-path=" +
                                                     cublasdx_include_dirs.substr(start, end - start));
            }
            #ifdef COMMONDX_INCLUDE_DIR
            {
                cublasdx_include_dirs_array.push_back("--include-path=" + std::string(COMMONDX_INCLUDE_DIR));
            }
            #endif
            #ifdef CUTLASS_INCLUDE_DIR
            {
                cublasdx_include_dirs_array.push_back("--include-path=" + std::string(CUTLASS_INCLUDE_DIR));
            }
            #endif
            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_COMMONDX_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                }
            }
            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_CUTLASS_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                }
            }
            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_CUBLASDX_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                }
            }
            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_CUDA_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                }
            }
            return cublasdx_include_dirs_array;
        }

        inline unsigned get_device_architecture(int device) {
            int major = 0;
            int minor = 0;
            CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
            CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
            return major * 10 + minor;
        }

        inline std::string get_device_architecture_option(int device) {
            // --gpus-architecture=compute_... will generate PTX, which means NVRTC must be at least as recent as the CUDA driver;
            // --gpus-architecture=sm_... will generate SASS, which will always run on any CUDA driver from the current major
            std::string gpu_architecture_option = "--gpu-architecture=sm_" + std::to_string(get_device_architecture(device));
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

#endif // CUBLASDX_EXAMPLE_COMMON_NVRTC_HPP
