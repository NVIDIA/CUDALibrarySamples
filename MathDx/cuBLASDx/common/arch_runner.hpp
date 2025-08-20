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

template<class EnableSM>
void print_supported_sm(unsigned cuda_device_arch) {
    auto stream = std::stringstream();

    if constexpr (EnableSM::sm_70) {
        stream << "- SM 700" << std::endl;
    }
    if constexpr (EnableSM::sm_72) {
        stream << "- SM 720" << std::endl;
    }
    if constexpr (EnableSM::sm_75) {
        stream << "- SM 750" << std::endl;
    }
    if constexpr (EnableSM::sm_80) {
        stream << "- SM 800" << std::endl;
    }
    if constexpr (EnableSM::sm_86) {
        stream << "- SM 860" << std::endl;
    }
    if constexpr (EnableSM::sm_87) {
        stream << "- SM 870" << std::endl;
    }
    if constexpr (EnableSM::sm_89) {
        stream << "- SM 890" << std::endl;
    }
    if constexpr (EnableSM::sm_90) {
        stream << "- SM 900" << std::endl;
    }  
    if constexpr (EnableSM::sm_100) {
        stream << "- SM 1000" << std::endl;
    }

  #if CUDA_VERSION < 13000
    if constexpr (EnableSM::sm_101) {
        stream << "- SM 1010" << std::endl;
    }
  #endif

    if constexpr (EnableSM::sm_103) {
        stream << "- SM 1030" << std::endl;
    }

  #if CUDA_VERSION >= 13000
    if constexpr (EnableSM::sm_110) {
        stream << "- SM 1100" << std::endl;
    }
  #endif

    if constexpr (EnableSM::sm_120) {
        stream << "- SM 1200" << std::endl;
    }   
    if constexpr (EnableSM::sm_121) {
        stream << "- SM 1210" << std::endl;
    }

    std::cerr << "Functor failed to run on any supported SM, supported SMs: \n" << stream.str() << std::endl
              << "this device architecture: " << cuda_device_arch << std::endl;
}


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
#if CUDA_VERSION < 13000
        case 1010: {
            if constexpr(EnableSM::sm_101) {
                return example_functor(std::integral_constant<int, 1010>{}, static_cast<Args&&>(args)...);
            }
            break;
        } 
#endif
        case 1030: {
            if constexpr(EnableSM::sm_103) {
                return example_functor(std::integral_constant<int, 1030>{}, static_cast<Args&&>(args)...);
            }
            break;
        } 
#if CUDA_VERSION >= 13000
        case 1100: {
            if constexpr(EnableSM::sm_110) {
                return example_functor(std::integral_constant<int, 1100>{}, static_cast<Args&&>(args)...);
            }
            break;
        } 
#endif
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
    }

    // We cannot check invoke_result_t because that
    // would require instantiating Arch dependent
    // runner
    print_supported_sm<EnableSM>(cuda_device_arch);
    return default_value<StatusType>::get();
}

}

#endif // CUBLASDX_EXAMPLE_ARCH_RUNNER_HPP
