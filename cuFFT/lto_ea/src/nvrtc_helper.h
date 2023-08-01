/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* A simplified version of the NVRTC helper included with CUDA samples,
 * targeting cuFFT LTO callbacks
 */

#ifndef COMMON_NVRTC_HELPER_H_
#define COMMON_NVRTC_HELPER_H_

#include <cuda.h>
#include <nvrtc.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define NVRTC_SAFE_CALL(Name, x)                                \
  do {                                                          \
    nvrtcResult result = x;                                     \
    if (result != NVRTC_SUCCESS) {                              \
      std::cerr << "\nerror: " << Name << " failed with error " \
                << nvrtcGetErrorString(result) << std::endl;    \
      exit(1);                                                  \
    }                                                           \
  } while (0)

#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)
#define INCLUDE_CUDA_PATH "-I" STRINGIZE(CUDA_PATH) "/include"
#define CUDA_ARCH_FLAG "-arch=compute_" STRINGIZE(CUDA_ARCH)
#define CALLBACK_CODE_PATH(name) STRINGIZE(SOURCE_PATH) "/" name

void compile_file_to_lto(std::vector<char>& cubin_result, const char *filename) {
  std::ifstream inputFile(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if (!inputFile.is_open()) {
    std::cerr << "\nerror: unable to open " << filename << " for reading!\n";
    exit(1);
  }

  std::streampos pos = inputFile.tellg();
  size_t inputSize = (size_t)pos;
  std::vector<char> memBlock(inputSize + 1);

  inputFile.seekg(0, std::ios::beg);
  inputFile.read(memBlock.data(), inputSize);
  inputFile.close();
  memBlock[inputSize] = '\x0';

  const int   num_params       = 6;
  const char *compile_params[] = {INCLUDE_CUDA_PATH,
                                  CUDA_ARCH_FLAG,
                                  "--std=c++11",
                                  "--relocatable-device-code=true",
                                  "-default-device",
                                  "-dlto"};

  // Compile
  nvrtcProgram prog;
  NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&prog, memBlock.data(), filename, 0, NULL, NULL));
  nvrtcResult res = nvrtcCompileProgram(prog, num_params, compile_params);

  // Print log
  size_t logSize;
  NVRTC_SAFE_CALL("nvrtcGetProgramLogSize", nvrtcGetProgramLogSize(prog, &logSize));
  std::vector<char> log(logSize + 1);
  NVRTC_SAFE_CALL("nvrtcGetProgramLog", nvrtcGetProgramLog(prog, log.data()));
  log[logSize] = '\x0';

  if(log.size() > 2) {
    std::cerr << "\n compilation log ---\n";
    std::string s(log.begin(), log.end());
    std::cerr << s;
    std::cerr << "\n end log ---\n";
  }

  NVRTC_SAFE_CALL("nvrtcCompileProgram", res);

  size_t codeSize;
  NVRTC_SAFE_CALL("nvrtcGetLTOIRSize", nvrtcGetLTOIRSize(prog, &codeSize));
  std::vector<char> buffer(codeSize);
  NVRTC_SAFE_CALL("nvrtcGetNVVM", nvrtcGetLTOIR(prog, buffer.data()));
  cubin_result = buffer;
}

#endif  // COMMON_NVRTC_HELPER_H_
