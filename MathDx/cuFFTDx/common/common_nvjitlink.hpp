
#ifndef CUFFTDX_EXAMPLE_COMMON_NVJITLINK_HPP
#define CUFFTDX_EXAMPLE_COMMON_NVJITLINK_HPP

#include "common_nvrtc.hpp"

#define NVJITLINK_SAFE_CALL(h,x)                                 \
    do {                                                         \
        nvJitLinkResult result = x;                              \
        if (result != NVJITLINK_SUCCESS) {                       \
            std::cerr << "\nerror: " #x " failed with error "    \
                        << result << '\n';                       \
            size_t lsize;                                        \
            result = nvJitLinkGetErrorLogSize(h, &lsize);        \
            if (result == NVJITLINK_SUCCESS && lsize > 0) {      \
                std::vector<char> log(lsize);                    \
                result = nvJitLinkGetErrorLog(h, log.data());    \
                if (result == NVJITLINK_SUCCESS) {               \
                    std::cerr << "error: " << log.data() << '\n';\
                }                                                \
            }                                                    \
            exit(1);                                             \
        }                                                        \
    } while(0)

namespace example {
    namespace nvjitlink {
        inline std::string get_device_architecture_option(int device) {
            std::string gpu_architecture_option = "-arch=sm_" + std::to_string(nvrtc::get_device_architecture(device));
            return gpu_architecture_option;
        }
    }
}
#endif // CUFFTDX_EXAMPLE_COMMON_NVJITLINK_HPP
