#ifndef CUBLASDX_EXAMPLE_ARCH_RUNNER_HPP
#define CUBLASDX_EXAMPLE_ARCH_RUNNER_HPP

#include <type_traits>
#include <optional>
#include "../common/common.hpp"

namespace example {

template<class StatusType, typename = void>
struct default_value;

template<class StatusType>
struct default_value<StatusType, std::enable_if_t<std::is_integral_v<StatusType>>> {
    static auto inline get() {
        return static_cast<StatusType>(1);
    }
};

template<>
struct default_value<void> {
    static auto inline get() {
        return void();
    }
};

template<class Internal>
struct default_value<std::optional<Internal>> {
    static std::optional<Internal> inline get() {
        return std::nullopt;
    }
};

template<class EnableSM, class StatusType, class Functor, class ... Args>
StatusType arch_runner(unsigned cuda_device_arch, Functor example_functor, Args&& ... args) {
    switch (cuda_device_arch) {
        case 700: {
            if constexpr(EnableSM::sm_70) {
                return example_functor(std::integral_constant<int, 700>{}, static_cast<Args&&>(args)...);
            }
            break;
        }
        case 720: {
            if constexpr(EnableSM::sm_72) {
                return example_functor(std::integral_constant<int, 720>{}, static_cast<Args&&>(args)...);
            }
            break;
        }
        case 750: {
            if constexpr(EnableSM::sm_75) {
                return example_functor(std::integral_constant<int, 750>{}, static_cast<Args&&>(args)...);
            }
            break;
        }
        case 800: {
            if constexpr(EnableSM::sm_80) {
                return example_functor(std::integral_constant<int, 800>{}, static_cast<Args&&>(args)...);
            }
            break;
        }
        case 860: {
            if constexpr(EnableSM::sm_86) {
                return example_functor(std::integral_constant<int, 860>{}, static_cast<Args&&>(args)...);
            }
            break;
        }
        case 870: {
            if constexpr(EnableSM::sm_87) {
                return example_functor(std::integral_constant<int, 870>{}, static_cast<Args&&>(args)...);
            }
            break;
        }
        case 890: {
            if constexpr(EnableSM::sm_89) {
                return example_functor(std::integral_constant<int, 890>{}, static_cast<Args&&>(args)...);
            }
            break;
        }
        case 900: {
            if constexpr(EnableSM::sm_90) {
                return example_functor(std::integral_constant<int, 900>{}, static_cast<Args&&>(args)...);
            }
            break;
        }
        case 1000: {
            if constexpr(EnableSM::sm_100) {
                return example_functor(std::integral_constant<int, 1000>{}, static_cast<Args&&>(args)...);
            }
            break;
        }
        case 1010: {
            if constexpr(EnableSM::sm_101) {
                return example_functor(std::integral_constant<int, 1010>{}, static_cast<Args&&>(args)...);
            }
            break;
        } 
        case 1030: {
            if constexpr(EnableSM::sm_103) {
                return example_functor(std::integral_constant<int, 1030>{}, static_cast<Args&&>(args)...);
            }
            break;
        } 
        case 1200: {
            if constexpr(EnableSM::sm_120) {
                return example_functor(std::integral_constant<int, 1200>{}, static_cast<Args&&>(args)...);
            }
            break;
        }
        case 1210: {
            if constexpr(EnableSM::sm_121) {
                return example_functor(std::integral_constant<int, 1210>{}, static_cast<Args&&>(args)...);
            }
            break;
        }
        default: {
            printf("Examples not configured to support SM %u.  Use the CUBLASDX_CUDA_ARCHITECTURES CMake variable to configure the SM support.\n",
                   cuda_device_arch);
            return 1;
        }
    }

    // We cannot check invoke_result_t because that
    // would require instantiating Arch dependent
    // runner
    return default_value<StatusType>::get();
}

}

#endif // CUBLASDX_EXAMPLE_ARCH_RUNNER_HPP
