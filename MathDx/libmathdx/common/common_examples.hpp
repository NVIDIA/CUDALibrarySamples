/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LIBMATHDX_EXAMPLES_COMMON_EXAMPLES_HPP
#define LIBMATHDX_EXAMPLES_COMMON_EXAMPLES_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <libcufftdx.h>
#include <nvJitLink.h>
#include <nvrtc.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "arch.hpp"
#include "macros.hpp"

namespace examples {

inline std::string getenv(const char* name) {
#ifdef _WIN32
    char* ptr = nullptr;
    size_t size = 0;
    auto error = _dupenv_s(&ptr, &size, name);
    ASSERT(!error);
    if (ptr == nullptr) {
        return std::string();
    } else {
        std::string out(ptr);
        free(ptr);
        return out;
    }
#else
    const char* ptr = std::getenv(name);
    return ptr == nullptr ? std::string() : std::string(ptr);
#endif
}

inline std::tuple<int, int> get_nvrtc_version() {
    int major = 0;
    int minor = 0;
    NVRTC_CHECK(nvrtcVersion(&major, &minor));
    return { major, minor };
}

// Typed input for nvJitLink: pairs an input kind (LTOIR, FATBIN, ...) with its bytes.
// Lets examples link a mix of fatbins and LTOIRs (e.g. nvCOMPDx requires both).
struct lto_t {
    nvJitLinkInputType type;
    std::vector<char> code;
};

inline std::vector<char> link(const std::vector<lto_t>& ltos, arch_t sm) {
    nvJitLinkHandle handle {};

    auto gpu_cc = get_current_cc();
    const bool use_cubin = sm.cc == gpu_cc;
    std::vector<std::string> link_options;
    if (use_cubin) {
        link_options = { "-lto", std::string("-arch=sm_") + sm.str() };
    } else {
        link_options = { "-lto", std::string("-arch=compute_") + sm.str(), "-ptx" };
    }

    std::vector<const char*> lto_opts;
    for (const auto& o : link_options) {
        lto_opts.emplace_back(o.c_str());
    }
    NVJITLINK_CHECK(handle, nvJitLinkCreate(&handle, static_cast<int>(lto_opts.size()), lto_opts.data()));
    for (const auto& lto : ltos) {
        NVJITLINK_CHECK(handle, nvJitLinkAddData(handle, lto.type, lto.code.data(), lto.code.size(), "lto_"));
    }
    NVJITLINK_CHECK(handle, nvJitLinkComplete(handle));

    size_t size = 0;
    if (use_cubin) {
        NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubinSize(handle, &size));
    } else {
        NVJITLINK_CHECK(handle, nvJitLinkGetLinkedPtxSize(handle, &size));
    }
    std::vector<char> output(size);
    if (use_cubin) {
        NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubin(handle, output.data()));
    } else {
        NVJITLINK_CHECK(handle, nvJitLinkGetLinkedPtx(handle, output.data()));
    }
    NVJITLINK_CHECK(handle, nvJitLinkDestroy(&handle));

    std::string dump = getenv("LIBMATHDX_EXAMPLE_DUMP_FILENAME");
    if (!dump.empty()) {
        printf("Writing nvJitLink output to %s\n", dump.c_str());
        std::ofstream out(dump.c_str(), std::ios::out | std::ios::binary);
        ASSERT(out.is_open());
        out.write(output.data(), static_cast<long>(output.size()));
        out.close();
    }

    return output;
}

inline std::vector<char> compile_and_link(const std::string& cuda_code, std::vector<lto_t> ltos, arch_t sm) {
    // if LIBMATHDX_EXAMPLE_VERBOSE is set, print generated kernel source
    if (!getenv("LIBMATHDX_EXAMPLE_VERBOSE").empty()) {
        printf("%s\n", cuda_code.c_str());
    }

    nvrtcProgram prog { nullptr };
    NVRTC_CHECK(nvrtcCreateProgram(&prog, cuda_code.c_str(), "test_code.cu", 0, nullptr, nullptr));

    std::vector<std::string> options = { "--relocatable-device-code=true",
                                         "--device-as-default-execution-space",
                                         "--std=c++17",
                                         "-dlto",
                                         std::string("--gpu-architecture=compute_") + sm.str() };

    std::vector<const char*> opts;
    for (const auto& o : options) {
        opts.emplace_back(o.c_str());
    }

    auto compile_result = nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
    if (compile_result != NVRTC_SUCCESS) {
        size_t log_size = 0;
        NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
        std::vector<char> log(log_size);
        NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
        printf("Log: %s\n", log.data());
        NVRTC_CHECK(compile_result);
    }


    size_t lto_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(prog, &lto_size));
    std::vector<char> compiled_code(lto_size);
    NVRTC_CHECK(nvrtcGetLTOIR(prog, compiled_code.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));

    ltos.push_back({ NVJITLINK_INPUT_LTOIR, std::move(compiled_code) });
    return link(ltos, sm);
}

// Convenience overload: a single LTOIR (the common case for cuFFTDx, cuBLASDx, etc).
inline std::vector<char> compile_and_link(const std::string& cuda_code, const std::vector<char>& lto, arch_t sm) {
    return compile_and_link(cuda_code, std::vector<lto_t> { { NVJITLINK_INPUT_LTOIR, lto } }, sm);
}

template <typename... Args>
std::string strprintf(const char* format, Args... args) {
    // size does not include \0
    int size = snprintf(nullptr, 0, format, args...);
    ASSERT(size >= 0);
    std::vector<char> buffer(size + 1, '\0');
    int written = snprintf(buffer.data(), buffer.size(), format, args...);
    ASSERT(written + 1 == static_cast<int>(buffer.size()));
    return std::string(buffer.data());
}

} // namespace examples

#endif // LIBMATHDX_EXAMPLES_COMMON_EXAMPLES_HPP
