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

// Benchmark performance from the binary data file fname
static void run_example(const std::vector<std::vector<char>>& data, int algo, int level)
{
  size_t total_bytes = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
  }

  std::cout << "----------" << std::endl;
  std::cout << "files: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;

  const size_t chunk_size = 1 << 16;

  // build up input batch on CPU
  BatchDataCPU input_data_cpu(data, chunk_size);
  std::cout << "chunks: " << input_data_cpu.size() << std::endl;

  // compression

  // Allocate and prepare output/compressed batch
  BatchDataCPU compress_data_cpu(
      chunk_size, input_data_cpu.size());

  // loop over chunks on the CPU, compressing each one
  for (size_t i = 0; i < input_data_cpu.size(); ++i) {
    size_t actual_len = 0;
    if(algo==0){ //libdeflate
      struct libdeflate_compressor *compressor;
      compressor = libdeflate_alloc_compressor(6%13);
      size_t len = libdeflate_deflate_compress(compressor, input_data_cpu.ptrs()[i],
                            input_data_cpu.sizes()[i], compress_data_cpu.ptrs()[i], compress_data_cpu.sizes()[i]);
      if (len == 0) {
        throw std::runtime_error(
            "libdeflate_deflate_compress failed to compress chunk " + std::to_string(i) + ".");
      }
      actual_len = len;
    }else if(algo==1){ //zlib::compress2
     uLongf len = static_cast<uLongf>(input_data_cpu.sizes()[i]);
     int ret = compress2((uint8_t *)compress_data_cpu.ptrs()[i], &len, (const Bytef *) input_data_cpu.ptrs()[i], static_cast<uLong>(input_data_cpu.sizes()[i]), 9);
     if (ret != Z_OK) {
         throw std::runtime_error("ZLIB compress() failed " + std::to_string(ret));
     }
     if (len >= 6) {
       memmove((uint8_t*)compress_data_cpu.ptrs()[i], (uint8_t*)compress_data_cpu.ptrs()[i] + 2, len - 6);
       len -= 6;
     }
     actual_len = static_cast<size_t>(len);
    }else if(algo==2){ //zlib::deflate
     z_stream zs;
     zs.zalloc = NULL; zs.zfree = NULL;
     zs.msg = NULL;
     zs.next_in  = (Bytef *)input_data_cpu.ptrs()[i];
     zs.avail_in = static_cast<uInt>(input_data_cpu.sizes()[i]);
     zs.next_out = (Bytef *)compress_data_cpu.ptrs()[i];
     zs.avail_out = static_cast<uInt>(input_data_cpu.sizes()[i]);
     int strategy=Z_DEFAULT_STRATEGY; //Z_HUFFMAN_ONLY //Z_FIXED, Z_DEFAULT_STRATEGY
     int ret = deflateInit2(&zs, level, Z_DEFLATED, -15, 8, strategy                       ); // -15 to disable zlib header/footer
     if (ret!=Z_OK) {
         throw std::runtime_error("Call to deflateInit2 failed: " + std::to_string(ret));
     }
     if ((ret = deflate(&zs, Z_FINISH)) != Z_STREAM_END) {
         throw std::runtime_error("Deflate operation failed: " + std::to_string(ret));
     }
     if ((ret = deflateEnd(&zs)) != Z_OK) {
         throw std::runtime_error("Call to deflateEnd failed: " + std::to_string(ret));
     }
     actual_len = static_cast<size_t>(zs.total_out);
    }
   // set the actual compressed size
   compress_data_cpu.sizes()[i] = actual_len;
  }

  // compute compression ratio
  size_t* compressed_sizes_host = compress_data_cpu.sizes();
  size_t comp_bytes = 0;
  for (size_t i = 0; i < compress_data_cpu.size(); ++i)
    comp_bytes += compressed_sizes_host[i];

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;

  // Copy compressed data to GPU
  BatchData compress_data(compress_data_cpu, true);

  // Allocate and build up decompression batch on GPU
  BatchData decomp_data(input_data_cpu, false);

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // CUDA events to measure decompression time
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  // deflate GPU decompression
  size_t decomp_temp_bytes;
  nvcompStatus_t status = nvcompBatchedDeflateDecompressGetTempSize(
      compress_data.size(), chunk_size, &decomp_temp_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedDeflateDecompressGetTempSize() failed.");
  }

  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, decomp_data.size() * sizeof(size_t)));

  nvcompStatus_t* d_status_ptrs;
  CUDA_CHECK(cudaMalloc(&d_status_ptrs, decomp_data.size() * sizeof(nvcompStatus_t)));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Run decompression
  status = nvcompBatchedDeflateDecompressAsync(
      compress_data.ptrs(),
      compress_data.sizes(),
      decomp_data.sizes(),
      d_decomp_sizes,
      compress_data.size(),
      d_decomp_temp,
      decomp_temp_bytes,
      decomp_data.ptrs(),
      d_status_ptrs,
      stream);
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedDeflateDecompressAsync() not successful");
  }

  // Validate decompressed data against input
  if (!(input_data_cpu == decomp_data))
    throw std::runtime_error("Failed to validate decompressed data");
  else
    std::cout << "decompression validated :)" << std::endl;

  // Re-run decompression to get throughput
  CUDA_CHECK(cudaEventRecord(start, stream));
  status = nvcompBatchedDeflateDecompressAsync(
    compress_data.ptrs(),
    compress_data.sizes(),
    decomp_data.sizes(),
    d_decomp_sizes,
    compress_data.size(),
    d_decomp_temp,
    decomp_temp_bytes,
    decomp_data.ptrs(),
    d_status_ptrs,
    stream);
  CUDA_CHECK(cudaEventRecord(end, stream));
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedDeflateDecompressAsync() not successful");
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));

  double decompression_throughput = ((double)total_bytes / ms) * 1e-6;
  std::cout << "decompression throughput (GB/s): " << decompression_throughput
            << std::endl;

  CUDA_CHECK(cudaFree(d_decomp_temp));
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
  int algo = 0; int level = 6;
  printf("level %d\n", level); 
  int i = 1; bool choose_algo = false; bool input_file = false;
  do{
   if(strcmp(argv[i], "-a") !=0 && strcmp(argv[i], "-f") != 0){
     std::cerr << "The config only could be -a (choose algorithm: 0 libdeflate, 1 zlib_compress2, 2 zlib_deflate) or -f (add input files)." << std::endl;
     return 1;
   }else if(strcmp(argv[i], "-a") ==0){
     choose_algo = true;
     i++;
     if( (i < argc) && (atoi(argv[i]) == 0 ||  atoi(argv[i]) == 1 || atoi(argv[i]) == 2)){
       algo = atoi(argv[2]);
       i++;
     }else{
       std::cerr<<"`-a` could only be 0, 1, 2. (0 libdeflate, 1 zlib_compress2, 2 zlib_deflate)"<<std::endl;
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
   std::cerr<<"Have to choose an algorithm use `-a`. `-a` could be 0, 1, 2. (0 libdeflate, 1 zlib_compress2, 2 zlib_deflate)"<<std::endl;
   return 1;
  }

  if(!input_file){
   std::cerr<<"Specify at least one input file by using `-f`"<<std::endl;
   return 1;
  }

  auto data = multi_file(file_names);
  run_example(data, algo, level);

  return 0;
}
