// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <lz4.h>
#include <lz4hc.h>

#include <nvrtc.h>
#include <nvJitLink.h>

#include <nvcompdx.hpp>
#include "../common/batch_data.hpp"
#include "../common/util_nvrtc.hpp"

using namespace nvcompdx;

// This sample demonstrates the usage of the warp-level device API for
// LZ4 GPU decompression. The decompression kernel is compiled and linked
// during runtime. The compression happens through the host-side
// lz4 CPU library.

// LZ4 decompression kernel, using the preconfigured decompressor
// 1 warp per chunk, but multiple chunks per thread block

const char* decomp_kernel = R"kernel(
#include <nvcompdx.hpp>

using namespace nvcompdx;

extern "C" __global__ void decomp_warp_kernel(
    size_t batch_size,
    const void * const * comp_chunks,
    void * const * uncomp_chunks,
    const size_t * comp_chunk_sizes,
    size_t * decomp_chunk_sizes) {

  const unsigned int global_chunk_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const unsigned int local_chunk_id = threadIdx.x / 32;
  if(global_chunk_id >= batch_size) {
    return;
  }

  using decompressor_type =
    decltype(Algorithm<algorithm::ALGORITHM>() +
             DataType<datatype::DATATYPE>() +
             MaxUncompChunkSize<MAX_UNCOMP_CHUNK_SIZE>() +
             Direction<direction::decompress>() +
             Warp() +
             SM<ARCH>());

  auto decompressor = decompressor_type();
  constexpr auto shmem_size_warp = decompressor.shmem_size_group();
  extern __shared__ __align__(decompressor.shmem_alignment()) uint8_t shared_scratch_decomp_buffer[];

  decompressor.execute(
    comp_chunks[global_chunk_id],
    uncomp_chunks[global_chunk_id],
    comp_chunk_sizes[global_chunk_id],
    decomp_chunk_sizes + global_chunk_id,
    shared_scratch_decomp_buffer + shmem_size_warp * local_chunk_id,
    nullptr);
}
)kernel";

static std::string get_device_architecture_option(CUdevice& device)
{
  // Note:
  // -arch=compute_...       will generate PTX
  // -arch=sm_...            will generate SASS
  // -arch=sm_... with -dlto will generate LTO IR
  std::string option = "-arch=sm_" + std::to_string(get_device_architecture(device));
  return option;
}

static std::vector<std::string> get_comp_include_dirs()
{
#ifndef NVCOMPDX_INCLUDE_DIRS
  return std::vector<std::string>();
#else
  std::vector<std::string> comp_include_dirs_array;
  {
    std::string comp_include_dirs = NVCOMPDX_INCLUDE_DIRS;
    std::string delim             = ",";
    size_t      start             = 0U;
    size_t      end               = comp_include_dirs.find(delim);
    while (end != std::string::npos) {
      comp_include_dirs_array.push_back("--include-path=" +
        comp_include_dirs.substr(start, end - start));
      start = end + delim.length();
      end   = comp_include_dirs.find(delim, start);
    }
    comp_include_dirs_array.push_back("--include-path=" +
      comp_include_dirs.substr(start, end - start));
  }
#endif // NVCOMPDX_INCLUDE_DIRS
#ifdef COMMONDX_INCLUDE_DIR
  {
    comp_include_dirs_array.push_back("--include-path=" + std::string(COMMONDX_INCLUDE_DIR));
  }
#endif // COMMONDX_INCLUDE_DIR
#ifdef CUTLASS_INCLUDE_DIR
  {
    comp_include_dirs_array.push_back("--include-path=" + std::string(CUTLASS_INCLUDE_DIR));
  }
#endif // CUTLASS_INCLUDE_DIR
  {
    const char* env_ptr = std::getenv("NVCOMPDX_EXAMPLE_COMMONDX_INCLUDE_DIR");
    if (env_ptr != nullptr) {
        comp_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
    }
  }
  {
    const char* env_ptr = std::getenv("NVCOMPDX_EXAMPLE_CUTLASS_INCLUDE_DIR");
    if (env_ptr != nullptr) {
        comp_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
    }
  }
  {
    const char* env_ptr = std::getenv("NVCOMPDX_EXAMPLE_NVCOMPDX_INCLUDE_DIR");
    if (env_ptr != nullptr) {
        comp_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
    }
  }
  {
    const char* env_ptr = std::getenv("NVCOMPDX_EXAMPLE_CUDA_INCLUDE_DIR");
    if (env_ptr != nullptr) {
        comp_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
        comp_include_dirs_array.push_back("--include-path=" + std::string(env_ptr) + "/cuda/std");
    }
  }
  return comp_include_dirs_array;
}

// Benchmark performance from the binary data file
template<unsigned int Arch>
static int run_nvrtc_example(const std::vector<std::vector<char>>& data)
{
  assert(!data.empty());

  size_t total_bytes = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
  }
  std::cout << "----------" << std::endl;
  std::cout << "files: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;

  // Compile-time (de)compression parameters
  constexpr size_t num_warps_per_chunk = 1;
  constexpr size_t num_chunks_per_block = 4;
  constexpr size_t num_warps_per_block = num_warps_per_chunk * num_chunks_per_block;
  constexpr unsigned int block_size = static_cast<unsigned int>(num_warps_per_block * 32);
  constexpr size_t chunk_size = 1 << 16; // [bytes]

  // Build up input batch on CPU
  BatchDataCPU input_data_cpu(data, chunk_size);
  size_t batch_size = input_data_cpu.batch_size();
  std::cout << "chunks: " << batch_size << std::endl;

  // Allocate and prepare output/compressed batch
  BatchDataCPU compressed_data_cpu(
      LZ4_compressBound(chunk_size), batch_size);

  // Compressing on the CPU
  // loop over chunks on the CPU, compressing each one one by one
  for (size_t i = 0; i < batch_size; ++i) {
    // could use LZ4_compress_default or LZ4_compress_fast instead
    const int size = LZ4_compress_HC(
        static_cast<const char*>(input_data_cpu.chunk_ptrs()[i]),
        static_cast<char*>(compressed_data_cpu.chunk_ptrs()[i]),
        static_cast<int>(input_data_cpu.chunk_sizes()[i]),
        static_cast<int>(compressed_data_cpu.chunk_sizes()[i]),
        12 /* compression level */);
    if (size == 0) {
      throw std::runtime_error(
          "LZ4 CPU failed to compress chunk " + std::to_string(i) + ".");
    }

    // Set the actual compressed size
    compressed_data_cpu.chunk_sizes()[i] = static_cast<size_t>(size);
  }

  // Compute compression ratio
  size_t* compressed_sizes_host = compressed_data_cpu.chunk_sizes();
  size_t comp_bytes =
    std::accumulate(compressed_sizes_host, compressed_sizes_host + batch_size, size_t(0));

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;

  // Configure the GPU decompressor
  using lz4_decompressor_type =
    decltype(Algorithm<algorithm::lz4>() +
             DataType<datatype::uint8>() +
             MaxUncompChunkSize<chunk_size>() +
             Direction<direction::decompress>() +
             Warp() +
             SM<Arch>());

  // Runtime decompression parameters
  const auto block_count =
    static_cast<unsigned int>((batch_size + num_chunks_per_block - 1) / num_chunks_per_block);

  // Global scratch buffer
  // Note: lz4 decompression requires no global scratch buffer
  static_assert(lz4_decompressor_type().tmp_size_group() == 0);

  // Shared scratch buffer
  const auto decomp_shared_memory =
    static_cast<unsigned int>(lz4_decompressor_type().shmem_size_group() * num_chunks_per_block);

  // Copy compressed data to GPU
  BatchData compressed_data(compressed_data_cpu, true, lz4_decompressor_type().input_alignment());

  // Allocate and build up decompression batch on GPU
  BatchData decomp_data(input_data_cpu, false, lz4_decompressor_type().output_alignment());

  // Create an NVRTC program out of the string-defined kernel
  nvrtcProgram program;
  NVRTC_CHECK(nvrtcCreateProgram(&program,
                                 decomp_kernel,
                                 NULL /* CUDA program name */,
                                 0 /* numHeaders */,
                                 NULL /* headers */,
                                 NULL /* includeNames */));

  // Prepare compilation options
  CUdevice cuDevice;
  CU_CHECK(cuInit(0 /* flags */));
  CU_CHECK(cuDeviceGet(&cuDevice, 0 /* by default using the first device */));
  const auto gpu_architecture_option = get_device_architecture_option(cuDevice);
  std::vector<const char*> opts = {
    "--std=c++17",
    "--device-as-default-execution-space",
    "--include-path=" CUDAToolkit_INCLUDE_DIR, // Path to the CUDA include directory
    "--include-path=" CUDAToolkit_INCLUDE_DIR "/cuda/std", // Path to standard headers
    "-dlto",
    "-rdc=true",
#ifdef NVCOMPDX_DISABLE_CUTLASS
    "-DNVCOMPDX_DISABLE_CUTLASS",
#endif // NVCOMPDX_DISABLE_CUTLASS
    gpu_architecture_option.c_str()
  };

  auto opt_convert_define = [](const auto& s1, const auto& s2) {
    return std::string("-D") + s1 + std::string("=") + s2;
  };

  // Compiler definitions
  std::vector<std::string> comp_config_values = {
    opt_convert_define("ALGORITHM", "lz4"),
    opt_convert_define("DATATYPE", "uint8"),
    opt_convert_define("MAX_UNCOMP_CHUNK_SIZE", std::to_string(chunk_size)),
    opt_convert_define("ARCH", std::to_string(Arch))
  };
  for (auto& config : comp_config_values) {
    opts.push_back(config.c_str());
  }

  // Include folder paths
  std::vector<std::string> comp_include_dirs = get_comp_include_dirs();
  for (auto& d : comp_include_dirs) {
    opts.push_back(d.c_str());
  }

  // Compile kernel via nvrtc
  nvrtcResult compileResult = nvrtcCompileProgram(program,
                                                  static_cast<int>(opts.size()),
                                                  opts.data());
  if (compileResult != NVRTC_SUCCESS) {
      // Obtain compilation log from the program if unsuccessful
      for (auto option : opts) {
          std::cout << option << std::endl;
      }
      print_nvrtc_program_log(std::cerr, program);
      std::exit(1);
  }

  // Obtain generated LTO IR from the program
  size_t lto_size;
  NVRTC_CHECK(nvrtcGetLTOIRSize(program, &lto_size));
  auto ltoir = std::make_unique<char[]>(lto_size);
  NVRTC_CHECK(nvrtcGetLTOIR(program, ltoir.get()));
  NVRTC_CHECK(nvrtcDestroyProgram(&program));

  // Load the generated Cubin and get a handle to our kernel
  CUcontext context;
  CU_CHECK(cuCtxCreate(&context, 0 /* flags */, cuDevice));

  // Load the generated LTO IR and the static nvCOMPDx LTO library
  nvJitLinkHandle linker;
  std::vector<const char*> lopts;
  lopts.emplace_back("-lto");
  lopts.emplace_back(gpu_architecture_option.c_str());
  NVJITLINK_CHECK(linker, nvJitLinkCreate(&linker,
                                          static_cast<uint32_t>(lopts.size()),
                                          lopts.data()));

  // Add the runtime-compiled kernel LTO IR
  NVJITLINK_CHECK(linker,
    nvJitLinkAddData(linker, NVJITLINK_INPUT_LTOIR, ltoir.get(), lto_size, "lto_online"));

  // Add nvCOMPDx LTO library or the nvCOMPDx fatbinary
  const char* fatbin_env_ptr = std::getenv("NVCOMPDX_EXAMPLE_NVCOMPDX_FATBIN");
  const char* library_env_ptr = std::getenv("NVCOMPDX_EXAMPLE_NVCOMPDX_LIBRARY");
  if(fatbin_env_ptr) {
    NVJITLINK_CHECK(linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN, fatbin_env_ptr));
  } else if(library_env_ptr) {
    NVJITLINK_CHECK(linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_LIBRARY, library_env_ptr));
  } else {
#if defined(NVCOMPDX_FATBIN)
    NVJITLINK_CHECK(linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN, NVCOMPDX_FATBIN));
#elif defined(NVCOMPDX_LIBRARY)
    NVJITLINK_CHECK(linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_LIBRARY, NVCOMPDX_LIBRARY));
#else
    std::cerr << "Please set one of the environment variables: "
              << "NVCOMPDX_EXAMPLE_NVCOMPDX_LIBRARY, "
              << "NVCOMPDX_EXAMPLE_NVCOMPDX_FATBIN, " << std::endl
              << "or during compilation define NVCOMPDX_LIBRARY or "
              << "NVCOMPDX_FATBIN." << std::endl;
    return 1;
#endif
  }

  // Generate the cubin from the LTO IR sources
  NVJITLINK_CHECK(linker, nvJitLinkComplete(linker));

  // Acquire cubin
  size_t cubin_size;
  NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubinSize(linker, &cubin_size));
  auto cubin = std::make_unique<char[]>(cubin_size);
  NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubin(linker, cubin.get()));
  NVJITLINK_CHECK(linker, nvJitLinkDestroy(&linker));

  // Load cubin
  CUmodule module;
  CUfunction kernel;
  CU_CHECK(cuModuleLoadDataEx(&module, cubin.get(), 0 /* numOptions */, NULL, NULL));
  CU_CHECK(cuModuleGetFunction(&kernel, module, "decomp_warp_kernel"));

  // Set dynamic shared memory needs
  CU_CHECK(cuFuncSetAttribute(kernel,
                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              decomp_shared_memory));

  // Start with the actual decompression
  auto comp_chunks = compressed_data.chunk_ptrs();
  auto uncomp_chunks = decomp_data.chunk_ptrs();
  auto comp_chunk_sizes = compressed_data.chunk_sizes();
  auto decomp_chunk_sizes = decomp_data.chunk_sizes();
  void* args[] = {
    &batch_size,
    &comp_chunks,
    &uncomp_chunks,
    &comp_chunk_sizes,
    &decomp_chunk_sizes
  };
  CU_CHECK(cuLaunchKernel(kernel,
                          block_count /* gridDimX */,
                          1 /* gridDimY */,
                          1 /* gridDimZ */,
                          block_size /* blockDimX */,
                          1 /* blockDimY */,
                          1 /* blockDimZ */,
                          decomp_shared_memory,
                          NULL /* hStream */,
                          args,
                          NULL));
  CU_CHECK(cuCtxSynchronize());

  // Validate decompressed data against input
  if (decomp_data != input_data_cpu) {
    throw std::runtime_error("Failed to validate decompressed data");
  } else {
    std::cout << "decompression validated :)" << std::endl;
  }
  return 0;
}

template<unsigned int Arch>
struct Runner {
  template<typename... Args>
  static int run(Args&&... args)
  {
    return run_nvrtc_example<Arch>(std::forward<Args>(args)...);
  }
};

int main(int argc, char* argv[])
{
  std::vector<std::string> file_names;

  do {
    if (argc < 3) {
      break;
    }

    int i = 1;
    while (i < argc) {
      const char* current_argv = argv[i++];
      if (strcmp(current_argv, "-f") == 0) {
        // parse until next `-` argument
        while (i < argc && argv[i][0] != '-') {
          file_names.emplace_back(argv[i++]);
        }
      } else {
        std::cerr << "Unknown argument: " << current_argv << std::endl;
        return 1;
      }
    }
  } while (0);

  if (file_names.empty()) {
    std::cerr << "Must specify at least one file via '-f <file>'." << std::endl;
    return 1;
  }

  auto data = multi_file(file_names);

  return run_with_current_arch<Runner>(data);
}
