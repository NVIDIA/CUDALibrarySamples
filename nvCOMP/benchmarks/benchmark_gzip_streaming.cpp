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

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "benchmark_common.h"
#include "custom_criterion.hpp"

#include <driver_types.h>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
// nvbench's device_info.cuh trips a GCC false positive under -Werror when this
// benchmark is compiled by the host compiler (it is the only .cpp NVBench benchmark).
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "nvbench/nvbench.cuh"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "nvcomp/native/streaming_gzip.hpp"

using namespace nvcomp;

namespace fs = std::filesystem;

NVBENCH_REGISTER_CRITERION(total_time_criterion);

static nvcompBatchedGzipCompressOpts_t nvcompBatchedGzipCompressOpts = nvcompBatchedGzipCompressDefaultOpts;

/**
 * Type of the IO wrapped around streaming operation
 * - Disk: streams from/to on-disk fstreams (temp location for output),
 * - Memory: streams from/to in-memory stringstreams.
 */
enum class InputMode
{
  Memory,
  Disk,
};

struct args_type
{
  std::string filename{};
  size_t duplicate_count = 1;
  InputMode mode = InputMode::Memory;
};

static args_type args;

// streambuf that loops the input stream for a given number of times.
// With disk mode and duplication this enables low memory footprint by looping over the initial stream.
class repeating_file_streambuf : public std::streambuf
{
public:
  repeating_file_streambuf(std::ifstream &file, size_t repeat_count)
      : file_(file)
      , repeat_count_(repeat_count)
  {
    file_.seekg(0, std::ios::end);
    file_size_ = static_cast<std::streamsize>(file_.tellg());
    reset();
  }

  void reset()
  {
    reps_done_ = 0;
    bytes_read_this_pass_ = 0;
    file_.clear();
    file_.seekg(0);

    // set egptr and gptr to the same value so underflow() is immedietly executed.
    // gptr - pointer to the current element
    // egptr - pointer to the element just beyond the last element
    setg(buf_, buf_, buf_);
  }

  // Original file size (without repetitions)
  std::streamsize file_size() const { return file_size_; }

protected:
  // Called when stream buffer is depleted (gptr() == egptr())
  int_type underflow() override
  {
    if (gptr() < egptr())
    {
      return traits_type::to_int_type(*gptr());
    }

    for (;;)
    {
      // Read the next buffer from the file handle.
      file_.read(buf_, sizeof(buf_));
      std::streamsize n = file_.gcount();
      if (n > 0)
      {
        bytes_read_this_pass_ += n;
        setg(buf_, buf_, buf_ + n);
        return traits_type::to_int_type(*gptr());
      }

      // If no bytes were read either we have an error or we drained the file.
      if (file_.bad() || (!file_.eof() && file_.fail()))
      {
        return traits_type::eof();
      }

      reps_done_ += 1;
      if (reps_done_ >= repeat_count_)
      {
        bytes_read_this_pass_ = 0;
        return traits_type::eof();
      }
      // Reset the file so next the underflows read from a looped handle.
      bytes_read_this_pass_ = 0;
      file_.clear();
      file_.seekg(0);
    }
  }

  // Returns position as it would be in a concatenated long stream.
  pos_type seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode which) override
  {
    if (!(which & std::ios_base::in))
    {
      return pos_type(off_type(-1));
    }

    if (way == std::ios_base::beg && off == 0)
    {
      reset();
      return pos_type(0);
    }
    if (way == std::ios_base::cur && off == 0)
    {
      std::streamsize buf_remaining = egptr() - gptr();
      std::streamsize pos_in_pass = bytes_read_this_pass_ - buf_remaining;
      return static_cast<pos_type>(static_cast<std::streamsize>(reps_done_) * file_size_ + pos_in_pass);
    }
    return pos_type(off_type(-1));
  }

private:
  std::ifstream &file_;
  size_t repeat_count_;
  size_t reps_done_ = 0;
  std::streamsize file_size_ = 0;
  std::streamsize bytes_read_this_pass_ = 0;
  static constexpr size_t BUF_SIZE = 65536;
  char buf_[BUF_SIZE];
};

// NOTE: populated by run_compression, consumed by run_decompression for Memory mode.
// Not cleared between NVBench runs - if parameter sweeps are added, this must be reset per configuration.
static std::string data;

static std::string compressed_data_host;

// Temp file paths are PID-qualified to avoid races when multiple instances run concurrently.
static fs::path s_gz_temp_path;
static fs::path s_out_temp_path;
static size_t s_uncompressed_file_size = 0;

struct FileGuard
{
  fs::path path;

  explicit FileGuard(fs::path path)
      : path(path) {};

  ~FileGuard()
  {
    if (!path.empty())
    {
      std::error_code ec;
      fs::remove(path, ec);
      if (ec)
      {
        std::cerr << "Failed to clean up the temporary 'disk' mode file: " << path << " due to: " << ec.message()
                  << std::endl;
      }
    }
  }
};

// Useful for better measurements of Memory mode
// Without pre-allocation a classic list-doubling allocations will be present when writing output data to the stringstream.
void reserve_ostringstream(std::ostringstream &oss, size_t bytes)
{
  oss.str(std::string(bytes, '\0'));
  oss.seekp(0);
}

static void add_compression_ratio_summary(nvbench::state &state, size_t uncompressed_size, size_t compressed_size)
{
  auto &summary = state.add_summary("Compression Ratio");
  summary.set_string("name", "Compression Ratio");
  summary.set_float64("value", static_cast<double>(uncompressed_size) / static_cast<double>(compressed_size));
}

static void reset_streams(std::istream &input, std::ostream &output)
{
  input.clear();
  input.seekg(0);
  output.clear();
  output.seekp(0);
}

template <typename FnT>
void compressStreaming(nvbench::state &state, InputMode mode, void *d_temp, size_t temp_bytes, FnT exec_sync)
{
  size_t compressed_size = 0;
  size_t uncompressed_size = 0;

  if (mode == InputMode::Memory)
  {
    const size_t dup = args.duplicate_count;
    std::string blob;
    {
      const std::vector<char> bytes = readFile(args.filename);
      blob.reserve(bytes.size() * dup);
      for (size_t i = 0; i < dup; ++i)
      {
        blob.append(bytes.data(), bytes.size());
      }
    }
    uncompressed_size = blob.size();
    state.add_element_count(uncompressed_size, "Bytes");

    data.resize(blob.size());
    std::memcpy(data.data(), blob.data(), blob.size());

    std::istringstream source_stream(blob);
    std::ostringstream output_stream;

    // Pre-size the output buffer to the input size so the timed loop never reallocates for compressible data.
    reserve_ostringstream(output_stream, blob.size());

    exec_sync([&](nvbench::launch &launch) {
      reset_streams(source_stream, output_stream);
      nvcompStatus_t st = nvcompGzipStreamingCompress(
        source_stream,
        output_stream,
        temp_bytes,
        d_temp,
        nvcompBatchedGzipCompressOpts,
        launch.get_stream()
      );
      benchmark_assert(st == nvcompSuccess, "nvcompGzipStreamingCompress failed");
    });

    compressed_size = output_stream.tellp();

    // TODO: Move the data instead of copying after transition to c++20
    compressed_data_host = output_stream.str();
    compressed_data_host.resize(compressed_size);
  }
  else
  {
    const size_t dup = args.duplicate_count;
    std::ifstream file(args.filename, std::ios::binary);
    if (!file.is_open())
    {
      throw std::runtime_error("Failed to open input file: " + args.filename);
    }
    repeating_file_streambuf rsb(file, dup);
    std::istream source_stream(&rsb);

    uncompressed_size = static_cast<size_t>(rsb.file_size()) * dup;
    s_uncompressed_file_size = uncompressed_size;

    state.add_element_count(uncompressed_size, "Bytes");

    std::ofstream output_stream(s_gz_temp_path, std::ios::binary | std::ios::trunc);
    if (!output_stream.is_open())
    {
      throw std::runtime_error("Failed to open temp output file: " + s_gz_temp_path.string());
    }

    exec_sync([&](nvbench::launch &launch) {
      rsb.reset();
      source_stream.rdbuf(&rsb); // re-inits the istream (clears all state via init())
      output_stream.clear();
      output_stream.seekp(0);
      nvcompStatus_t st = nvcompGzipStreamingCompress(
        source_stream,
        output_stream,
        temp_bytes,
        d_temp,
        nvcompBatchedGzipCompressOpts,
        launch.get_stream()
      );
      output_stream.flush();
      benchmark_assert(st == nvcompSuccess, "nvcompGzipStreamingCompress failed");
    });

    compressed_size = output_stream.tellp();
  }

  add_compression_ratio_summary(state, uncompressed_size, compressed_size);
}

template <typename FnT>
void decompressStreaming(nvbench::state &state, InputMode mode, void *d_temp, size_t temp_bytes, FnT exec_sync)
{
  size_t uncompressed_size = 0;
  size_t compressed_size = 0;

  if (mode == InputMode::Memory)
  {
    if (compressed_data_host.empty())
    {
      state.skip("No compressed data available; run the compression benchmark first.");
      return;
    }

    uncompressed_size = data.size();
    state.add_element_count(uncompressed_size, "Bytes");

    compressed_size = compressed_data_host.size();

    std::istringstream source_stream(compressed_data_host);
    std::ostringstream output_stream;

    reserve_ostringstream(output_stream, uncompressed_size);

    exec_sync([&](nvbench::launch &launch) {
      reset_streams(source_stream, output_stream);

      nvcompStatus_t st =
        nvcompGzipStreamingDecompress(source_stream, output_stream, temp_bytes, d_temp, launch.get_stream());
      benchmark_assert(st == nvcompSuccess, "nvcompGzipStreamingDecompress failed");
    });

    // Validate decompression
    // TODO: Move the data instead of copying after transition to c++20
    benchmark_assert(
      std::memcmp(data.data(), output_stream.str().c_str(), uncompressed_size) == 0,
      "Decompressed data does not match input"
    );
  }
  else
  {
    std::ifstream source_stream(s_gz_temp_path, std::ios::binary);
    if (!source_stream.is_open())
    {
      state.skip("Compressed temp file not found; run the compression benchmark first.");
      return;
    }

    compressed_size = static_cast<size_t>(fs::file_size(s_gz_temp_path));
    uncompressed_size = s_uncompressed_file_size;

    state.add_element_count(uncompressed_size, "Bytes");

    std::ifstream ref_file(args.filename, std::ios::binary);
    if (!ref_file.is_open())
    {
      throw std::runtime_error("Failed to open input uncompressed file: " + args.filename);
    }

    repeating_file_streambuf rsb(ref_file, args.duplicate_count);

    std::fstream output_stream(s_out_temp_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!output_stream.is_open())
    {
      throw std::runtime_error("Failed to open temp output file: " + s_out_temp_path.string());
    }

    exec_sync([&](nvbench::launch &launch) {
      reset_streams(source_stream, output_stream);
      nvcompStatus_t st =
        nvcompGzipStreamingDecompress(source_stream, output_stream, temp_bytes, d_temp, launch.get_stream());
      output_stream.flush();
      benchmark_assert(st == nvcompSuccess, "nvcompGzipStreamingDecompress failed");
    });

    // Validate decompression
    rsb.reset();
    std::istream ref_stream(&rsb);
    output_stream.seekg(0);
    benchmark_assert(
      std::equal(
        std::istreambuf_iterator<char>(ref_stream),
        std::istreambuf_iterator<char>(), // end-of-stream sentinel
        std::istreambuf_iterator<char>(output_stream),
        std::istreambuf_iterator<char>() // end-of-stream sentinel
      ),
      "Decompressed data does not match input"
    );
  }

  add_compression_ratio_summary(state, uncompressed_size, compressed_size);
}

template <bool DO_COMPRESSION>
void run_benchmark(nvbench::state &state)
{
  if (args.filename.empty())
  {
    state.skip("No input file specified (-f).");
    return;
  }

  auto exec_sync = [&](auto &&launcher) {
    exec_or_waive_on_oom(state, nvbench::exec_tag::sync, std::forward<decltype(launcher)>(launcher));
  };

  if constexpr (DO_COMPRESSION)
  {
    nvcompBatchedGzipCompressOpts_t opts = nvcompBatchedGzipCompressOpts;

    size_t temp_bytes = 0;
    benchmark_assert(
      nvcompGzipStreamingCompressGetTempSize(opts, &temp_bytes) == nvcompSuccess,
      "nvcompGzipStreamingCompressGetTempSize failed"
    );
    void *d_temp = nullptr;
    CUDA_CHECK(cudaMallocSafe(&d_temp, temp_bytes));

    compressStreaming(state, args.mode, d_temp, temp_bytes, exec_sync);

    CUDA_CHECK(cudaFree(d_temp));
  }
  else
  {
    size_t temp_bytes = 0;
    benchmark_assert(
      nvcompGzipStreamingDecompressGetTempSize(&temp_bytes) == nvcompSuccess,
      "nvcompGzipStreamingDecompressGetTempSize failed"
    );
    void *d_temp = nullptr;
    CUDA_CHECK(cudaMallocSafe(&d_temp, temp_bytes));

    decompressStreaming(state, args.mode, d_temp, temp_bytes, exec_sync);

    CUDA_CHECK(cudaFree(d_temp));
  }
}

static void run_benchmark_compress(nvbench::state &state) { run_benchmark<true>(state); }
#ifndef _WIN32
static void run_benchmark_decompress(nvbench::state &state) { run_benchmark<false>(state); }
#endif

NVBENCH_BENCH(run_benchmark_compress)
  .set_name("Gzip Streaming Compression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);

#ifndef _WIN32
NVBENCH_BENCH(run_benchmark_decompress)
  .set_name("Gzip Streaming Decompression")
  .set_stopping_criterion("total-time-criterion")
  .set_timeout(30.0);
#endif

void usage(const std::string &name, const std::vector<parameter_type> &parameters)
{
  std::cout << "Usage: " << name << " [OPTIONS]" << std::endl;

  for (const parameter_type &parameter : parameters)
  {
    const std::string short_prefix = parameter.short_flag.size() > 1 ? "--" : "-";
    std::cout << "  " << short_prefix << parameter.short_flag << ",";
    std::cout << "--" << parameter.long_flag;
    std::cout << "  : " << parameter.description << std::endl;
    if (parameter.default_value.empty())
    {
      // no default value
    }
    else if (parameter.default_value == REQUIRED_PARAMETER)
    {
      std::cout << "    required" << std::endl;
    }
    else
    {
      std::cout << "    default=" << parameter.default_value << std::endl;
    }
  }
}

// Parses our flags from argv, populates global `args`, and returns a vector of the remaining arguments for NVBench.
std::vector<char *> parse_args(int argc, char **argv)
{
  args.duplicate_count = 1;
  args.mode = InputMode::Memory;

  std::vector<parameter_type> params{
    {"h", "help", "Show options.", "", nullptr},
    {"f", "input_file", "The input file to compress/decompress.", REQUIRED_PARAMETER, nullptr},
    {"x",
     "duplicate_data",
     "Clone uncompressed chunks multiple times (scale factor, 1x means no "
     "duplication).",
     std::to_string(args.duplicate_count),
     [](const char *arg) {
       char *end = nullptr;
       unsigned long long val = std::strtoull(arg, &end, 10);
       if (end == arg || *end != '\0' || val < 1 || val > static_cast<unsigned long long>(std::numeric_limits<int>::max()))
       {
         std::cerr << "ERROR: --duplicate_data must be a non-negative integer." << std::endl;
         return false;
       }
       args.duplicate_count = static_cast<size_t>(val);
       return true;
     }},
    {"m",
     "mode",
     "IO mode: 'disk' (stream from/to on-disk fstreams; output goes to a temp location) "
     "or 'memory' (stream from/to in-memory stringstreams).",
     "memory",
     [](const char *arg) {
       std::string_view sv(arg);
       if (sv == "disk")
       {
         args.mode = InputMode::Disk;
       }
       else if (sv == "memory")
       {
         args.mode = InputMode::Memory;
       }
       else
       {
         std::cerr << "ERROR: --mode must be 'disk' or 'memory'." << std::endl;
         return false;
       }
       return true;
     }},
    {"alg", "algorithm", "Deflate algorithm to use (0-5).", "1", [](const char *arg) -> bool {
       char *end = nullptr;
       long algorithm_type = std::strtol(arg, &end, 10);
       if (end == arg || *end != '\0' || algorithm_type < 0 || algorithm_type > 5)
       {
         std::cerr << "ERROR: Deflate algorithm must be 0, 1, 2, 3, 4, or 5." << std::endl;
         return false;
       }
       nvcompBatchedGzipCompressOpts.algorithm = static_cast<int>(algorithm_type);
       return true;
     }}
  };

  const std::string name(argv[0]);
  std::vector<char *> nvbench_argv{argv[0]};

  for (int i = 1; i < argc;)
  {
    char *const cur_arg_ptr = argv[i];
    std::string arg(argv[i++]);
    bool found = false;
    for (const parameter_type &param : params)
    {
      const std::string short_prefix = param.short_flag.size() > 1 ? "--" : "-";
      if (arg == short_prefix + param.short_flag || arg == "--" + param.long_flag)
      {
        found = true;

        if (param.long_flag == "help")
        {
          usage(name, params);
          // NVBench does not recognize "-h" as a help flag, so we add the long one instead.
          static char help_flag[] = "--help";
          nvbench_argv.push_back(help_flag);
          break;
        }

        // everything from here on out requires an extra parameter
        if (i >= argc)
        {
          std::cerr << "ERROR: Missing argument for '" << arg << "'." << std::endl;
          usage(name, params);
          std::exit(1);
        }
        if (param.long_flag == "input_file")
        {
          args.filename = argv[i++];
          break;
        }
        if (!param.handler(argv[i++]))
        {
          std::exit(1);
        }
        break;
      }
    }
    if (!found)
    {
      nvbench_argv.push_back(cur_arg_ptr);
    }
  }

  if (args.filename.empty())
  {
    std::cerr << "WARNING: No input file specified (-f). Benchmarks will not run." << std::endl;
  }

  return nvbench_argv;
}

int main(int argc, char **argv)
{
  std::vector<char *> nvbench_argv = parse_args(argc, argv);

  if (args.mode == InputMode::Disk)
  {
    std::string pid = std::to_string(getpid());
    s_gz_temp_path = fs::temp_directory_path() / ("nvcomp_gzip_streaming_" + pid + ".gz");
    s_out_temp_path = fs::temp_directory_path() / ("nvcomp_gzip_streaming_" + pid);
  }

  FileGuard gz_guard(s_gz_temp_path);
  FileGuard out_guard(s_out_temp_path);

  nvcomp::default_to_single_device(nvbench_argv);
  int nvbench_argc = static_cast<int>(nvbench_argv.size());
  NVBENCH_MAIN_BODY(nvbench_argc, nvbench_argv.data());
}
