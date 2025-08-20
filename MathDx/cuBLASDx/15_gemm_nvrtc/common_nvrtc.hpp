
#ifndef CUBLASDX_EXAMPLE_COMMON_NVRTC_HPP
#define CUBLASDX_EXAMPLE_COMMON_NVRTC_HPP

#include <cstdlib>
#define CUBLASDX_EXAMPLE_NVRTC
#include "../common/common.hpp"

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
            std::vector<std::string> cublasdx_include_dirs_array;

            auto append_multiple_dirs = [](auto& container, const std::string& semicolon_separated_dirs) {
                if (semicolon_separated_dirs.empty()) return;
                
                std::stringstream ss(semicolon_separated_dirs);
                std::string dir;
                while (std::getline(ss, dir, ';')) {
                    if (!dir.empty()) {  // Skip empty directories
                        container.push_back("--include-path=" + dir);
                    }
                }
            };

            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_COMMONDX_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                } else {
                    #ifdef COMMONDX_INCLUDE_DIR
                    {
                        cublasdx_include_dirs_array.push_back("--include-path=" + std::string(COMMONDX_INCLUDE_DIR));
                    }
                    #endif
                }
            }
            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_CUTLASS_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                } else {
                    #ifdef CUTLASS_INCLUDE_DIR
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(CUTLASS_INCLUDE_DIR));
                    #endif
                }
            }
            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_CUBLASDX_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                } else {
                    #ifdef CUBLASDX_INCLUDE_DIRS
                    append_multiple_dirs(cublasdx_include_dirs_array, std::string(CUBLASDX_INCLUDE_DIRS)); 
                    #endif
                }
            }
            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_CUDA_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                    // CUDA 13 created a separate include folder for CCCL
                    #if CUDA_VERSION >= 13000
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr) + "/cccl");
                    #endif
                } else {
                    #ifdef CUDA_INCLUDE_DIR
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(CUDA_INCLUDE_DIR));
                    // CUDA 13 created a separate include folder for CCCL
                    #if CUDA_VERSION >= 13000
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(CUDA_INCLUDE_DIR) + "/cccl");
                    #endif
                    #endif
                }
            }

            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_USER_DIRECTORIES");
                if(env_ptr != nullptr) {
                    append_multiple_dirs(cublasdx_include_dirs_array, std::string(env_ptr)); 
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
