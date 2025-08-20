
#ifndef CUSOLVERDX_EXAMPLE_COMMON_NVRTC_HELPER_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_NVRTC_HELPER_HPP

#include <cstdlib>
#define CUSOLVERDX_EXAMPLE_NVRTC

#define NVRTC_SAFE_CALL(x)                                                                            \
    do {                                                                                              \
        nvrtcResult result = x;                                                                       \
        if (result != NVRTC_SUCCESS) {                                                                \
            std::cerr << "\nerror: " #x " failed with error " << nvrtcGetErrorString(result) << '\n'; \
            exit(1);                                                                                  \
        }                                                                                             \
    } while (0)

#define NVJITLINK_SAFE_CALL(h, x)                                                \
    do {                                                                         \
        nvJitLinkResult result = x;                                              \
        if (result != NVJITLINK_SUCCESS) {                                       \
            std::cerr << "\nerror: " #x " failed with error " << result << '\n'; \
            size_t lsize;                                                        \
            result = nvJitLinkGetErrorLogSize(h, &lsize);                        \
            if (result == NVJITLINK_SUCCESS && lsize > 0) {                      \
                std::vector<char> log(lsize);                                    \
                result    = nvJitLinkGetErrorLog(h, log.data());                 \
                if (result == NVJITLINK_SUCCESS) {                               \
                    std::cerr << "error: " << log.data() << '\n';                \
                }                                                                \
            }                                                                    \
            std::exit(result);                                                   \
        }                                                                        \
    } while (0)

#ifndef CU_CHECK_AND_EXIT
    #define CU_CHECK_AND_EXIT(error)                                                  \
        {                                                                             \
            auto status = static_cast<CUresult>(error);                               \
            if (status != CUDA_SUCCESS) {                                             \
                const char* pstr;                                                     \
                cuGetErrorString(status, &pstr);                                      \
                std::cout << pstr << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                    \
            }                                                                         \
        }
#endif // CU_CHECK_AND_EXIT

namespace common {
    namespace nvrtc {
        template<class Type>
        inline Type get_global_from_module(CUmodule module, const char* name) {
            CUdeviceptr value_ptr;
            size_t      value_size;
            CU_CHECK_AND_EXIT(cuModuleGetGlobal(&value_ptr, &value_size, module, name));
            Type value_host;
            CU_CHECK_AND_EXIT(cuMemcpyDtoH(&value_host, value_ptr, value_size));
            return value_host;
        }

        inline std::vector<std::string> get_solver_include_dirs() {
#ifndef CUSOLVERDX_INCLUDE_DIRS
            return std::vector<std::string>();
#endif
            std::vector<std::string> solver_include_dirs_array;
            {
                std::string solver_include_dirs = CUSOLVERDX_INCLUDE_DIRS;
                std::string delim               = ";";
                size_t      start               = 0U;
                size_t      end                 = solver_include_dirs.find(delim);
                while (end != std::string::npos) {
                    solver_include_dirs_array.push_back("--include-path=" + solver_include_dirs.substr(start, end - start));
                    start = end + delim.length();
                    end   = solver_include_dirs.find(delim, start);
                }
                solver_include_dirs_array.push_back("--include-path=" + solver_include_dirs.substr(start, end - start));
            }
#ifdef COMMONDX_INCLUDE_DIR
            { solver_include_dirs_array.push_back("--include-path=" + std::string(COMMONDX_INCLUDE_DIR)); }
#endif
#ifdef CUTLASS_INCLUDE_DIR
            { solver_include_dirs_array.push_back("--include-path=" + std::string(CUTLASS_INCLUDE_DIR)); }
#endif
            {
                const char* env_ptr = std::getenv("CUSOLVERDX_EXAMPLE_COMMONDX_INCLUDE_DIR");
                if (env_ptr != nullptr) {
                    solver_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                }
            }
            {
                const char* env_ptr = std::getenv("CUSOLVERDX_EXAMPLE_CUTLASS_INCLUDE_DIR");
                if (env_ptr != nullptr) {
                    solver_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                }
            }
            {
                const char* env_ptr = std::getenv("CUSOLVERDX_EXAMPLE_CUSOLVERDX_INCLUDE_DIR");
                if (env_ptr != nullptr) {
                    solver_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                }
            }
            {
                const char* env_ptr = std::getenv("CUSOLVERDX_EXAMPLE_CUDA_INCLUDE_DIR");
                if (env_ptr != nullptr) {
                    solver_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                    #if CUDA_VERSION >= 13000
                    solver_include_dirs_array.push_back("--include-path=" + std::string(env_ptr) + "/cccl");
                    #endif
                }
            }
            return solver_include_dirs_array;
        }

        inline unsigned get_device_architecture(int device) {
            int major = 0;
            int minor = 0;
            CU_CHECK_AND_EXIT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
            CU_CHECK_AND_EXIT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
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
} // namespace common

#endif // CUSOLVERDX_EXAMPLE_COMMON_NVRTC_HELPER_HPP
