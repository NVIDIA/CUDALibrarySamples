/*
 * Copyright (c) 2025 NVIDIA CORPORATION AND AFFILIATES. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the NVIDIA CORPORATION nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <nvrtc.h>
#include <nvJitLink.h>

#ifndef NVRTC_CHECK
#define NVRTC_CHECK(func)                                                     \
  do {                                                                        \
    nvrtcResult rt = (func);                                                  \
    if (rt != NVRTC_SUCCESS) {                                                \
      const char* str = nvrtcGetErrorString(rt);                              \
      std::cerr << "NVRTC call failure \"" #func "\" with " << rt << " at "   \
                << __FILE__ << ":" << __LINE__ << std::endl;                  \
      std::cerr << str << std::endl;                                          \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)
#endif // NVRTC_CHECK

#ifndef NVJITLINK_CHECK
#define NVJITLINK_CHECK(handle, func)                                             \
  do {                                                                            \
    nvJitLinkResult rt = (func);                                                  \
    if (rt != NVJITLINK_SUCCESS) {                                                \
      std::cerr << "NVJITLINK call failure \"" #func "\" with " << rt << " at "   \
                << __FILE__ << ":" << __LINE__ << std::endl;                      \
      size_t log_size;                                                            \
      rt = nvJitLinkGetErrorLogSize(handle, &log_size);                           \
      if (rt == NVJITLINK_SUCCESS && log_size > 0) {                              \
        auto log = std::make_unique<char[]>(log_size);                            \
        rt = nvJitLinkGetErrorLog(handle, log.get());                             \
        if (rt == NVJITLINK_SUCCESS) {                                            \
          std::cerr << log.get() << std::endl;                                    \
        }                                                                         \
      }                                                                           \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)
#endif // NVJITLINK_CHECK

void print_nvrtc_program_log(std::ostream& os, const nvrtcProgram& prog)
{
  size_t log_size;
  NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
  auto log = std::make_unique<char[]>(log_size);
  NVRTC_CHECK(nvrtcGetProgramLog(prog, log.get()));
  os << log.get() << std::endl;
}

unsigned int get_device_architecture(CUdevice& device)
{
  int major = 0;
  int minor = 0;
  CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  return static_cast<unsigned int>(major * 10 + minor);
}
