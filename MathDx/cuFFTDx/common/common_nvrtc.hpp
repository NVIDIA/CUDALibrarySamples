#ifndef CUFFTDX_EXAMPLE_COMMON_NVRTC_HPP
#define CUFFTDX_EXAMPLE_COMMON_NVRTC_HPP

#include <cstdlib>
#include <sstream>
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
                std::cerr << pstr << " " << __FILE__ << ":" << __LINE__ << std::endl;                       \
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

            std::vector<std::string> cufftdx_include_dirs_array;

            const auto path_handler = [&](const std::string& path_name, const std::string& entry) {
                if (!entry.empty()) {
                    cufftdx_include_dirs_array.push_back("--include-path=" + entry);
                } else {
                    std::cerr << "Empty include path in '" << path_name << "'" << std::endl;
                    std::exit(-1);
                }
            };

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
                const char* env_ptr = std::getenv("CUFFTDX_EXAMPLE_COMMONDX_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    path_handler("COMMONDX_INCLUDE_DIR", std::string(env_ptr));
                } else {
                    #ifdef COMMONDX_INCLUDE_DIR
                    {
                        path_handler("COMMONDX_INCLUDE_DIR", std::string(COMMONDX_INCLUDE_DIR));
                    }
                    #endif
                }
            }

            {
                const char* env_ptr = std::getenv("CUFFTDX_EXAMPLE_CUFFTDX_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    path_handler("CUFFTDX_EXAMPLE_CUFFTDX_INCLUDE_DIR", std::string(env_ptr));
                } else {
                    #ifdef CUFFTDX_INCLUDE_DIRS
                    {
                        append_multiple_dirs(cufftdx_include_dirs_array, std::string(CUFFTDX_INCLUDE_DIRS));
                    }
                    #endif
                }
            }
            {
                const char* env_ptr = std::getenv("CUFFTDX_EXAMPLE_CUTLASS_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    path_handler("CUFFTDX_EXAMPLE_CUTLASS_INCLUDE_DIR", std::string(env_ptr));
                } else {
                    cufftdx_include_dirs_array.push_back("-DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY");
                }
            }
            {
                const char* env_ptr = std::getenv("CUFFTDX_EXAMPLE_CUDA_INCLUDE_DIR");
                if(env_ptr != nullptr) {
                    path_handler("CUFFTDX_EXAMPLE_CUDA_INCLUDE_DIR", std::string(env_ptr));
                    // CUDA 13 created a separate include folder for CCCL
                    #if CUDA_VERSION >= 13000
                    path_handler("CUFFTDX_EXAMPLE_CUDA_INCLUDE_DIR_CCCL", std::string(env_ptr) + "/cccl");
                    #endif
                } else {
                    #ifdef CUDA_INCLUDE_DIR
                    {
                        path_handler("CUDA_INCLUDE_DIR", std::string(CUDA_INCLUDE_DIR));
                        // CUDA 13 created a separate include folder for CCCL
                        #if CUDA_VERSION >= 13000
                        path_handler("CUDA_INCLUDE_DIR_CCCL", std::string(CUDA_INCLUDE_DIR) + "/cccl");
                        #endif
                    }
                    #endif
                }
            }
            {
                const char* env_ptr = std::getenv("CUFFTDX_EXAMPLE_USER_DIRECTORIES");
                if(env_ptr != nullptr) {
                    append_multiple_dirs(cufftdx_include_dirs_array, std::string(env_ptr));
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
