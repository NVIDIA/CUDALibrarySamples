// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

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
