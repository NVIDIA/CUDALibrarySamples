/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#include "BatchData.h"
#include "zlib.h"
#include "libdeflate.h"
#include "nvcomp/deflate.h"

BatchDataCPU GetBatchDataCPU(const BatchData& batch_data, bool copy_data)
{
  BatchDataCPU compress_data_cpu(
      batch_data.ptrs(),
      batch_data.sizes(),
      batch_data.data(),
      batch_data.size(),
      copy_data);
  return compress_data_cpu;
}

// Benchmark performance from the binary data file fname
static void run_example(const std::vector<std::vector<char>>& data, int algo)
{
  size_t total_bytes = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
  }

  std::cout << "----------" << std::endl;
  std::cout << "files: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;

  const size_t chunk_size = 1 << 16;

  // build up metadata
  BatchData input_data(data, chunk_size);
  static nvcompBatchedDeflateOpts_t nvcompBatchedDeflateOpts = nvcompBatchedDeflateDefaultOpts;
  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  nvcompStatus_t status = nvcompBatchedDeflateCompressGetTempSize(
      input_data.size(),
      chunk_size,
      nvcompBatchedDeflateOpts,
      &comp_temp_bytes);
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedDeflateCompressGetTempSize() not successful");
  }

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t max_out_bytes;
  status = nvcompBatchedDeflateCompressGetMaxOutputChunkSize(
      chunk_size, nvcompBatchedDeflateOpts, &max_out_bytes);
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedDeflateCompressGetMaxOutputChunkSize() not successful");
  }

  BatchData compress_data(max_out_bytes, input_data.size());

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(start, stream));

  status = nvcompBatchedDeflateCompressAsync(
      input_data.ptrs(),
      input_data.sizes(),
      chunk_size,
      input_data.size(),
      d_comp_temp,
      comp_temp_bytes,
      compress_data.ptrs(),
      compress_data.sizes(),
      nvcompBatchedDeflateOpts,
      stream);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedDeflateCompressAsync() failed.");
  }
  
  CUDA_CHECK(cudaEventRecord(end, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // free compression memory
  CUDA_CHECK(cudaFree(d_comp_temp));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));

  // compute compression ratio
  std::vector<size_t> compressed_sizes_host(compress_data.size());
  CUDA_CHECK(cudaMemcpy(
      compressed_sizes_host.data(),
      compress_data.sizes(),
      compress_data.size() * sizeof(*compress_data.sizes()),
      cudaMemcpyDeviceToHost));

  size_t comp_bytes = 0;
  for (const size_t s : compressed_sizes_host) {
    comp_bytes += s;
  }

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;
  std::cout << "compression throughput (GB/s): "
            << (double)total_bytes / (1.0e6 * ms) << std::endl;

  // Allocate and prepare output/compressed batch
  BatchDataCPU compress_data_cpu = GetBatchDataCPU(compress_data, true);
  BatchDataCPU decompress_data_cpu = GetBatchDataCPU(input_data, false);

  // loop over chunks on the CPU, decompressing each one
  for (size_t i = 0; i < input_data.size(); ++i) {
    if(algo==0){
        struct libdeflate_decompressor  *decompressor;
        decompressor = libdeflate_alloc_decompressor();
        enum libdeflate_result res = libdeflate_deflate_decompress(decompressor, compress_data_cpu.ptrs()[i], compress_data_cpu.sizes()[i], 
                                                   decompress_data_cpu.ptrs()[i], decompress_data_cpu.sizes()[i], NULL);
    
       if (res != LIBDEFLATE_SUCCESS) {
       throw std::runtime_error(
           "libdeflate CPU failed to decompress chunk " + std::to_string(i) + ".");
       }
    }else if (algo==1){
        z_stream zs1;
        zs1.zalloc = NULL;
        zs1.zfree = NULL;
        zs1.msg = NULL;
        zs1.next_in = (Bytef*)compress_data_cpu.ptrs()[i];
        zs1.avail_in = static_cast<uInt>(compress_data_cpu.sizes()[i]);
        zs1.next_out = (Bytef*)decompress_data_cpu.ptrs()[i];
        zs1.avail_out = static_cast<uInt>(decompress_data_cpu.sizes()[i]);

        int ret = inflateInit2(&zs1, -15);
        if (ret != Z_OK) {
           throw std::runtime_error("inflateInit2 error " + std::to_string(ret));
        }
        if ((ret = inflate(&zs1, Z_FINISH)) != Z_STREAM_END) {
           throw std::runtime_error("zlib::inflate operation fail " + std::to_string(ret));;
            if ((ret = inflateEnd(&zs1)) != Z_OK) {
               throw std::runtime_error("Call to inflateEnd failed: " + std::to_string(ret));
            }
        }
        if ((ret = inflateEnd(&zs1)) != Z_OK) {
           throw std::runtime_error("Call to inflateEnd failed: " + std::to_string(ret));
        }
    }
  }
  // Validate decompressed data against input
  if (!(decompress_data_cpu == input_data))
    throw std::runtime_error("Failed to validate CPU decompressed data");
  else
    std::cout << "CPU decompression validated :)" << std::endl;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

std::vector<char> readFile(const std::string& filename)
{
  std::vector<char> buffer(4096);
  std::vector<char> host_data;

  std::ifstream fin(filename, std::ifstream::binary);
  fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  size_t num;
  do {
    num = fin.readsome(buffer.data(), buffer.size());
    host_data.insert(host_data.end(), buffer.begin(), buffer.begin() + num);
  } while (num > 0);

  return host_data;
}

std::vector<std::vector<char>>
multi_file(const std::vector<std::string>& filenames)
{
  std::vector<std::vector<char>> split_data;

  for (auto const& filename : filenames) {
    split_data.emplace_back(readFile(filename));
  }

  return split_data;
}

int main(int argc, char* argv[])
{
 std::vector<std::string> file_names;

 if (argc < 5) {
   std::cerr << "Must choose the algorithm (-a <0>) and specify at least one file (-f <inputfile>)." << std::endl;
   return 1;
 }
 int algo = 0;
 int i = 1; bool choose_algo = false; bool input_file = false;
 do{
  if(strcmp(argv[i], "-a") !=0 && strcmp(argv[i], "-f") != 0){
    std::cerr << "The config only could be -a (choose algorithm: 0 libdeflate, 1 zlib_inflate) or -f (add input files)." << std::endl;
    return 1;
  }else if(strcmp(argv[i], "-a") ==0){
    choose_algo = true;
    i++;
    if( (i < argc) && (atoi(argv[i]) == 0 ||  atoi(argv[i]) == 1)){
      algo = atoi(argv[2]);
      i++;
    }else{
      std::cerr<<"`-a` could only be 0, 1. (0 libdeflate, 1 zlib_inflate)"<<std::endl;
      return 1;
    }
  }else if (strcmp(argv[i], "-f") == 0){
    i++;
    if(i >= argc){
      std::cerr<<"Specify at least one input file." <<std::endl;
      return 1;
    }
    do{
      input_file = true;
      file_names.push_back(argv[i]);
      i++;
    }while(i < argc && strcmp(argv[i], "-a") !=0);
  }
 }while(i < argc);

 if(!choose_algo){
  std::cerr<<"Have to choose an algorithm use `-a`. `-a` could be 0, 1. (0 libdeflate, 1 zlib_inflate)"<<std::endl;
  return 1;
 }

 if(!input_file){
   std::cerr<<"Specify at least one input file by using `-f`"<<std::endl;
   return 1;
 }

  auto data = multi_file(file_names);
  run_example(data, algo);

  return 0;
}

