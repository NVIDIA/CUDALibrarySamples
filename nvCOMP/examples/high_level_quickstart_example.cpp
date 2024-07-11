/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include <random>
#include <assert.h>
#include <iostream>

#include "nvcomp/lz4.hpp"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"
#include "util.h"

/*
  To build, execute

  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . --config Release

  To execute,
  ./high_level_quickstart_example
*/

using namespace nvcomp;

/**
 * In this example, we:
 *  1) compress the input data
 *  2) construct a new manager using the input data for demonstration purposes
 *  3) decompress the input data
 */
void decomp_compressed_with_manager_factory_example(uint8_t* device_input_ptrs, const size_t input_buffer_len)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // In this example we are not introducing the scope, because we explicitly deallocate gpu memory
  // and the cuda stream will not be used after that point. If we don't deallocate memory explicitly
  // then it will be deallocated in destructor of the manager, which require cuda stream to exists
  // and this is why we introduce the scope in each of the following examples.

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  nvcompBatchedLZ4Opts_t format_opts{data_type};
  LZ4Manager nvcomp_manager{chunk_size, format_opts, stream};
  CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

  uint8_t* comp_buffer;
  CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));

  nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);

  // Construct a new nvcomp manager from the compressed buffer.
  // Note we could use the nvcomp_manager from above, but here we demonstrate how to create a manager
  // for the use case where a buffer is received and the user doesn't know how it was compressed
  // Also note, creating the manager in this way synchronizes the stream, as the compressed buffer must be read to
  // construct the manager

  auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);

  DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
  uint8_t* res_decomp_buffer;
  CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

  decomp_nvcomp_manager->decompress(res_decomp_buffer, comp_buffer, decomp_config);

  CUDA_CHECK(cudaFree(comp_buffer));
  CUDA_CHECK(cudaFree(res_decomp_buffer));

  // Deallocating the gpu mem before the stream is destroyed for safety
  nvcomp_manager.deallocate_gpu_mem();
  decomp_nvcomp_manager->deallocate_gpu_mem();

  CUDA_CHECK(cudaStreamDestroy(stream));
}

/**
 * In this example, we:
 *  1) construct an nvcompManager
 *  2) compress the input data
 *  3) decompress the input data
 */ 
void comp_decomp_with_single_manager(uint8_t* device_input_ptrs, const size_t input_buffer_len)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  nvcompBatchedLZ4Opts_t format_opts{data_type};

  // We are introducing a scope, so that nvcomp_manager is destructed
  // before we destroy the stream.
  {
    LZ4Manager nvcomp_manager{chunk_size, format_opts, stream};
    CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

    uint8_t* comp_buffer;
    CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));
    
    nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);

    DecompressionConfig decomp_config = nvcomp_manager.configure_decompression(comp_buffer);
    uint8_t* res_decomp_buffer;
    CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

    nvcomp_manager.decompress(res_decomp_buffer, comp_buffer, decomp_config);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(comp_buffer));
    CUDA_CHECK(cudaFree(res_decomp_buffer));
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
}

/**
 * Additionally, we can use the same manager to execute multiple streamed compressions / decompressions
 * In this example we configure the multiple decompressions by inspecting the compressed buffers
 */  
void multi_comp_decomp_example(const std::vector<uint8_t*>& device_input_ptrs, std::vector<size_t>& input_buffer_lengths)
{
  size_t num_buffers = input_buffer_lengths.size();
  
  using namespace std;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  nvcompBatchedLZ4Opts_t format_opts{data_type};

  // Are asynchronous memory (de)allocations supported?
  bool use_async_mem_ops = false;
  {
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    int attribute_res_val;
    cudaError_t result = cudaDeviceGetAttribute(&attribute_res_val, cudaDevAttrMemoryPoolsSupported, device_id);
    if(result == cudaSuccess && attribute_res_val == 1) {
      use_async_mem_ops = true;
    }
  }

  // We are introducing a scope, so that nvcomp_manager is destructed
  // before we destroy the stream.
  {
    LZ4Manager nvcomp_manager{chunk_size, format_opts, stream};

    size_t offset = 8;
    auto alloc_fn = [&stream, offset, use_async_mem_ops](size_t alloc_size){
      void* buffer;
      if(use_async_mem_ops) {
        CUDA_CHECK(cudaMallocAsync(&buffer, alloc_size + offset, stream));
      } else {
        CUDA_CHECK(cudaMalloc(&buffer, alloc_size + offset));
      }
      return reinterpret_cast<void*>(reinterpret_cast<char*>(buffer) + offset);
    };

    auto dealloc_fn = [&stream, offset, use_async_mem_ops](void* buffer, size_t /*alloc_size*/){
      if(use_async_mem_ops) {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(reinterpret_cast<char*>(buffer) - offset), stream));
      } else {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(reinterpret_cast<char*>(buffer) - offset)));
      }
    };

    nvcomp_manager.set_scratch_allocators(alloc_fn, dealloc_fn);

    std::vector<uint8_t*> comp_result_buffers(num_buffers);

    for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      uint8_t* input_data = device_input_ptrs[ix_buffer];
      size_t input_length = input_buffer_lengths[ix_buffer];

      auto comp_config = nvcomp_manager.configure_compression(input_length);

      CUDA_CHECK(cudaMalloc(&comp_result_buffers[ix_buffer], comp_config.max_compressed_buffer_size));
      nvcomp_manager.compress(input_data, comp_result_buffers[ix_buffer], comp_config);
    }

    std::vector<uint8_t*> decomp_result_buffers(num_buffers);
    for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      uint8_t* comp_data = comp_result_buffers[ix_buffer];

      auto decomp_config = nvcomp_manager.configure_decompression(comp_data);

      CUDA_CHECK(cudaMalloc(&decomp_result_buffers[ix_buffer], decomp_config.decomp_data_size));

      nvcomp_manager.decompress(decomp_result_buffers[ix_buffer], comp_data, decomp_config);
    }

    for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      CUDA_CHECK(cudaFree(decomp_result_buffers[ix_buffer]));
      CUDA_CHECK(cudaFree(comp_result_buffers[ix_buffer]));
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
}

/**
 * Additionally, we can use the same manager to execute multiple streamed compressions / decompressions
 * In this example we configure the multiple decompressions by storing the comp_config's and inspecting those
 */
void multi_comp_decomp_example_comp_config(const std::vector<uint8_t*>& device_input_ptrs, std::vector<size_t>& input_buffer_lengths)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // We are introducing a scope, so that nvcomp_manager is destructed
  // before we destroy the stream.
  {
    const int chunk_size = 1 << 16;
    nvcompType_t data_type = NVCOMP_TYPE_CHAR;
    nvcompBatchedLZ4Opts_t format_opts{data_type};
    std::vector<CompressionConfig> comp_configs;

    LZ4Manager nvcomp_manager{chunk_size, format_opts, stream};

    size_t num_buffers = input_buffer_lengths.size();
    comp_configs.reserve(num_buffers);

    std::vector<uint8_t*> comp_result_buffers(num_buffers);

    for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      uint8_t* input_data = device_input_ptrs[ix_buffer];
      size_t input_length = input_buffer_lengths[ix_buffer];

      comp_configs.push_back(nvcomp_manager.configure_compression(input_length));
      auto& comp_config = comp_configs.back();

      CUDA_CHECK(cudaMalloc(&comp_result_buffers[ix_buffer], comp_config.max_compressed_buffer_size));

      nvcomp_manager.compress(input_data, comp_result_buffers[ix_buffer], comp_config);    
    }

    std::vector<uint8_t*> decomp_result_buffers(num_buffers);
    for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      auto decomp_config = nvcomp_manager.configure_decompression(comp_configs[ix_buffer]);

      CUDA_CHECK(cudaMalloc(&decomp_result_buffers[ix_buffer], decomp_config.decomp_data_size));

      nvcomp_manager.decompress(decomp_result_buffers[ix_buffer], comp_result_buffers[ix_buffer], decomp_config);    
    }

    for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      CUDA_CHECK(cudaFree(decomp_result_buffers[ix_buffer]));
      CUDA_CHECK(cudaFree(comp_result_buffers[ix_buffer]));
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
}

/**
 * This example shows how to use batched version of the API.
 */
void multi_comp_decomp_batched(const std::vector<uint8_t*>& device_input_ptrs, std::vector<size_t>& input_buffer_lengths)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // We are introducing a scope, so that nvcomp_manager is destructed
  // before we destroy the stream.
  {
    const int chunk_size = 1 << 16;
    nvcompType_t data_type = NVCOMP_TYPE_CHAR;
    nvcompBatchedLZ4Opts_t format_opts{data_type};

    LZ4Manager nvcomp_manager{chunk_size, format_opts, stream};

    size_t num_buffers = input_buffer_lengths.size();

    auto comp_configs = nvcomp_manager.configure_compression(input_buffer_lengths);

    // Allocate buffer for compressed data
    std::vector<uint8_t*> comp_result_buffers(num_buffers);
    for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      CUDA_CHECK(cudaMalloc(&comp_result_buffers[ix_buffer], comp_configs[ix_buffer].max_compressed_buffer_size));
    }

    nvcomp_manager.compress(device_input_ptrs.data(), comp_result_buffers.data(), comp_configs);

    auto decomp_configs = nvcomp_manager.configure_decompression(comp_result_buffers.data(), num_buffers);

    // Allocate buffer for decompressed data
    std::vector<uint8_t*> decomp_result_buffers(num_buffers);
    for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      CUDA_CHECK(cudaMalloc(&decomp_result_buffers[ix_buffer], decomp_configs[ix_buffer].decomp_data_size));
    }

    nvcomp_manager.decompress(decomp_result_buffers.data(), comp_result_buffers.data(), decomp_configs);

    for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      CUDA_CHECK(cudaFree(decomp_result_buffers[ix_buffer]));
      CUDA_CHECK(cudaFree(comp_result_buffers[ix_buffer]));
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
}

/**
 * This example shows how to use nvcomp manager with RAW BitstreamKind.
 * With BitstreamKind::RAW, the manager doesn't split the input data into smaller chunks like
 * it does with default (NVCOMP_NATIVE) BitstreamKind. No nvcomp header will be added
 * to the compressed data. Such manager is interoperable with the low-level API.
 *
 * We need to provide already chunked data to have efficient compression and decompression.
 * Some of the underlying algorithms may also fail if chunk size is too big.
 * For example how to split input data into chunks, see low_level_quickstart_examples.cpp
 *
 * This mode is useful when we want to compress already chunked data, but we don't want
 * to create and manage temporary buffers as we need with the low-level API
 */
void multi_comp_decomp_raw(const std::vector<uint8_t*>& device_input_ptrs, std::vector<size_t>& input_buffer_lengths)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // We are introducing a scope, so that nvcomp_manager is destructed
  // before we destroy the stream.
  {
    const int chunk_size = 1 << 16;
    nvcompType_t data_type = NVCOMP_TYPE_CHAR;
    nvcompBatchedLZ4Opts_t format_opts{data_type};

    // Chunk_size is ignored when we use BitstreamKind::RAW
    LZ4Manager nvcomp_manager{chunk_size, format_opts, stream, NoComputeNoVerify, BitstreamKind::RAW};

    size_t num_buffers = input_buffer_lengths.size();

    // Configure compression looks exactly the same as with default BitstreamKind.
    auto comp_configs = nvcomp_manager.configure_compression(input_buffer_lengths);

    // Allocate buffer for compressed data
    std::vector<uint8_t*> comp_result_buffers(num_buffers);
    for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      CUDA_CHECK(cudaMalloc(&comp_result_buffers[ix_buffer], comp_configs[ix_buffer].max_compressed_buffer_size));
    }

    // As there is no header, we need to pass an additional output buffer, where the size of each compressed chunk will be stored.
    // Such buffer can also be passed when manager is created with default (NVCOMP_NATIVE) BitstreamKind
    size_t* comp_sizes;
    CUDA_CHECK(cudaMalloc(&comp_sizes, sizeof(size_t) * num_buffers));

    nvcomp_manager.compress(device_input_ptrs.data(), comp_result_buffers.data(), comp_configs, comp_sizes);

    // The same size buffer must be then passed to configure_decompress and compress
    auto decomp_configs = nvcomp_manager.configure_decompression(comp_result_buffers.data(), num_buffers, comp_sizes);

    // Allocate buffer for decompressed data
    std::vector<uint8_t*> decomp_result_buffers(num_buffers);
    for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      CUDA_CHECK(cudaMalloc(&decomp_result_buffers[ix_buffer], decomp_configs[ix_buffer].decomp_data_size));
    }

    nvcomp_manager.decompress(decomp_result_buffers.data(), comp_result_buffers.data(), decomp_configs, comp_sizes);

    for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      CUDA_CHECK(cudaFree(decomp_result_buffers[ix_buffer]));
      CUDA_CHECK(cudaFree(comp_result_buffers[ix_buffer]));
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
}

/**
 * In this example, we:
 *  1) construct an nvcompManager with checksum support enabled
 *  2) compress the input data 
 *  3) decompress the input data
 */ 
void comp_decomp_with_single_manager_with_checksums(uint8_t* device_input_ptrs, const size_t input_buffer_len)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  /* 
   * There are 5 possible modes for checksum processing as
   * described below.
   * 
   * Mode: NoComputeNoVerify
   * Description:
   *   - During compression, do not compute checksums
   *   - During decompression, do not verify checksums
   *
   * Mode: ComputeAndNoVerify
   * Description:
   *   - During compression, compute checksums
   *   - During decompression, do not attempt to verify checksums
   *
   * Mode: NoComputeAndVerifyIfPresent
   * Description:
   *   - During compression, do not compute checksums
   *   - During decompression, verify checksums if they were included
   *
   * Mode: ComputeAndVerifyIfPresent
   * Description:
   *   - During compression, compute checksums
   *   - During decompression, verify checksums if they were included
   *
   * Mode: ComputeAndVerify
   * Description:
   *   - During compression, compute checksums
   *   - During decompression, verify checksums. A runtime error will be 
   *     thrown upon configure_decompression if checksums were not 
   *     included in the compressed buffer.
   */

  // manager constructed with checksum mode as final argument

  // We are introducing a scope, so that nvcomp_manager is destructed
  // before we destroy the stream.
  {
    nvcompBatchedLZ4Opts_t format_opts{data_type};
    LZ4Manager nvcomp_manager{chunk_size, format_opts, stream, ComputeAndVerify};
    CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

    uint8_t* comp_buffer;
    CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));
    
    // Checksums are computed and stored for uncompressed and compressed buffers during compression
    nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);

    DecompressionConfig decomp_config = nvcomp_manager.configure_decompression(comp_buffer);
    uint8_t* res_decomp_buffer;
    CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

    // Checksums are computed for compressed and decompressed buffers and verified against those
    // stored during compression
    nvcomp_manager.decompress(res_decomp_buffer, comp_buffer, decomp_config);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
    * After synchronizing the stream, the nvcomp status can be checked to see if
    * the checksums were successfully verified. Provided no unrelated nvcomp errors occurred,
    * if the checksums were successfully verified, the status will be nvcompSuccess. Otherwise,
    * it will be nvcompErrorBadChecksum.
    */
    nvcompStatus_t final_status = *decomp_config.get_status();
    if(final_status == nvcompErrorBadChecksum) {
      throw std::runtime_error("One or more checksums were incorrect.\n");
    }

    CUDA_CHECK(cudaFree(comp_buffer));
    CUDA_CHECK(cudaFree(res_decomp_buffer));
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
}

void decomp_compressed_with_manager_factory_with_checksums(
  uint8_t* device_input_ptrs, const size_t input_buffer_len)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  // We are introducing a scope, so that nvcomp_manager is destructed
  // before we destroy the stream.
  {
    /*
    * For a full description of the checksum modes, see the above example. Here, the
    * constructed manager will compute checksums on compression, but not verify them
    * on decompression.
    */
    nvcompBatchedLZ4Opts_t format_opts{data_type};
    LZ4Manager nvcomp_manager{chunk_size, format_opts, stream, ComputeAndNoVerify};
    CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

    uint8_t* comp_buffer;
    CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));
    
    nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);

    // Construct a new nvcomp manager from the compressed buffer.
    // Note we could use the nvcomp_manager from above, but here we demonstrate how to create a manager
    // for the use case where a buffer is received and the user doesn't know how it was compressed
    // Also note, creating the manager in this way synchronizes the stream, as the compressed buffer must be read to
    // construct the manager. This manager is configured to verify checksums on decompression if they were
    // supplied in the compressed buffer. For a full description of the checksum modes, see the
    // above example.
    auto decomp_nvcomp_manager = create_manager(comp_buffer, stream, NoComputeAndVerifyIfPresent);

    DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
    uint8_t* res_decomp_buffer;
    CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

    decomp_nvcomp_manager->decompress(res_decomp_buffer, comp_buffer, decomp_config);

    CUDA_CHECK(cudaFree(comp_buffer));
    CUDA_CHECK(cudaFree(res_decomp_buffer));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    /*
    * After synchronizing the stream, the nvcomp status can be checked to see if
    * the checksums were successfully verified. Provided no unrelated nvcomp errors occurred,
    * if the checksums were successfully verified, the status will be nvcompSuccess. Otherwise,
    * it will be nvcompErrorBadChecksum.
    */
    nvcompStatus_t final_status = *decomp_config.get_status();
    if(final_status == nvcompErrorBadChecksum) {
      throw std::runtime_error("One or more checksums were incorrect.\n");
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
}

int main()
{
  // Initialize a random array of chars
  const size_t input_buffer_len = 1000000;
  
  std::mt19937 random_gen(42);

  // char specialization of std::uniform_int_distribution is
  // non-standard, and isn't available on MSVC, so use short instead,
  // but with the range limited, and then cast below.
  std::uniform_int_distribution<short> uniform_dist(0, 255);

  // Single buffer examples
  {
    std::vector<uint8_t> uncompressed_data(input_buffer_len);

    for (size_t ix = 0; ix < input_buffer_len; ++ix) {
      uncompressed_data[ix] = static_cast<uint8_t>(uniform_dist(random_gen));
    }

    uint8_t* device_input_ptrs;
    CUDA_CHECK(cudaMalloc(&device_input_ptrs, input_buffer_len));
    CUDA_CHECK(cudaMemcpy(device_input_ptrs, uncompressed_data.data(), input_buffer_len, cudaMemcpyDefault));

    // Four roundtrip examples
    decomp_compressed_with_manager_factory_example(device_input_ptrs, input_buffer_len);
    decomp_compressed_with_manager_factory_with_checksums(device_input_ptrs, input_buffer_len);
    comp_decomp_with_single_manager(device_input_ptrs, input_buffer_len);
    comp_decomp_with_single_manager_with_checksums(device_input_ptrs, input_buffer_len);

    CUDA_CHECK(cudaFree(device_input_ptrs));
  }

  // Multi buffers examples
  {
    const size_t num_buffers = 10;

    std::vector<uint8_t*> gpu_buffers(num_buffers);
    std::vector<size_t> input_buffer_lengths(num_buffers);

    std::vector<std::vector<uint8_t>> uncompressed_buffers(num_buffers);
    for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      uncompressed_buffers[ix_buffer].resize(input_buffer_len);
      for (size_t ix_byte = 0; ix_byte < input_buffer_len; ++ix_byte) {
        uncompressed_buffers[ix_buffer][ix_byte] = static_cast<uint8_t>(uniform_dist(random_gen));
      }
      CUDA_CHECK(cudaMalloc(&gpu_buffers[ix_buffer], input_buffer_len));
      CUDA_CHECK(cudaMemcpy(gpu_buffers[ix_buffer], uncompressed_buffers[ix_buffer].data(), input_buffer_len, cudaMemcpyDefault));
      input_buffer_lengths[ix_buffer] = input_buffer_len;
    }

    multi_comp_decomp_example(gpu_buffers, input_buffer_lengths);
    multi_comp_decomp_example_comp_config(gpu_buffers, input_buffer_lengths);
    multi_comp_decomp_batched(gpu_buffers, input_buffer_lengths);

    for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      CUDA_CHECK(cudaFree(gpu_buffers[ix_buffer]));
    }
  }

  // Multi buffers examples with BitstreamKind::RAW
  {
    // With BitstreamKind::RAW, the manager doesn't split the input data into smaller chunks like manager with
    // default BitstreamKind (NVCOMP_NATIVE) do. We need to provide already chunked data to have efficient
    // compression and decompression. 
    // For more details see multi_comp_decomp_raw function descriptions or docs.

    const size_t num_buffers = 100;
    const size_t raw_input_buffer_len = 65536;

    std::vector<uint8_t*> gpu_buffers(num_buffers);
    std::vector<size_t> raw_input_buffer_lengths(num_buffers);

    std::vector<std::vector<uint8_t>> uncompressed_buffers(num_buffers);
    for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      uncompressed_buffers[ix_buffer].resize(raw_input_buffer_len);
      for (size_t ix_byte = 0; ix_byte < raw_input_buffer_len; ++ix_byte) {
        uncompressed_buffers[ix_buffer][ix_byte] = static_cast<uint8_t>(uniform_dist(random_gen));
      }
      CUDA_CHECK(cudaMalloc(&gpu_buffers[ix_buffer], raw_input_buffer_len));
      CUDA_CHECK(cudaMemcpy(gpu_buffers[ix_buffer], uncompressed_buffers[ix_buffer].data(), raw_input_buffer_len, cudaMemcpyDefault));
      raw_input_buffer_lengths[ix_buffer] = raw_input_buffer_len;
    }

    multi_comp_decomp_raw(gpu_buffers, raw_input_buffer_lengths);

    for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
      CUDA_CHECK(cudaFree(gpu_buffers[ix_buffer]));
    }
  }

  std::cout << "All examples finished successfully." << std::endl;

  return 0;
}
