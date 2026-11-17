// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include "error_checks.hpp"

namespace mathdx_examples {

// SM architectures in suggested_tile_config's encoding, 100 * major + 10 * minor.

// The architecture this pass compiles device code for; 0 in the host pass, which
// has no __CUDA_ARCH__. A kernel's compile-time tile configuration comes from it.
constexpr int compiled_sm_arch() {
#ifdef __CUDA_ARCH__
    return __CUDA_ARCH__;
#else
    return 0;
#endif
}

// The compute capability an architecture in that encoding spells: sm_86 (860)
// has major 8 and minor 6.
struct cc_t {
    int major;
    int minor;
    constexpr explicit cc_t(int sm_arch) : major(sm_arch / 100), minor(sm_arch / 10 % 10) {}
};

// The architecture of the kernel image the runtime loaded for `kernel` on the
// current device: the device pass that compiled it, also when a cubin runs on a
// later minor architecture (an sm_80 build on sm_86, say). Host code that must
// agree with the kernel's compile-time configuration looks this one up.
template <typename Kernel>
int loaded_sm_arch(Kernel kernel) {
    cudaFuncAttributes attributes {};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    return 10 * attributes.binaryVersion;
}

} // namespace mathdx_examples
