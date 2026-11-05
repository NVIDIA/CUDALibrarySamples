/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

  

#include "nvjpegDecoder.h"
#include <charconv>
#include <iomanip>


int decode_images(const FileData &img_data, const std::vector<size_t> &img_len,
                  std::vector<nvjpegImage_t> &out, decode_params_t &params,
                  double &time) {
  CHECK_CUDA(cudaStreamSynchronize(params.stream));
  cudaEvent_t startEvent = NULL, stopEvent = NULL;
  float loopTime = 0; 
  
  CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
  CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));


  std::vector<const unsigned char*> batched_bitstreams;
  std::vector<size_t> batched_bitstreams_size;
  std::vector<nvjpegImage_t>  batched_output;

  // bit-streams that batched decode cannot handle
  std::vector<const unsigned char*> otherdecode_bitstreams;
  std::vector<size_t> otherdecode_bitstreams_size;
  std::vector<nvjpegImage_t> otherdecode_output;

  if(params.hw_decode_available && !params.require_hardware){
    for(int i = 0; i < params.batch_size; i++){
      // extract bitstream meta data to figure out whether a bit-stream can be decoded
      CHECK_NVJPEG(nvjpegJpegStreamParseHeader(params.nvjpeg_handle, (const unsigned char *)img_data[i].data(), img_len[i], params.jpeg_streams[0]));
      int isSupported = -1;
      CHECK_NVJPEG(nvjpegDecodeBatchedSupported(params.nvjpeg_handle, params.jpeg_streams[0], &isSupported));

      // Packed YUV output uses the decoupled API rather than the legacy batch call.
      if(isSupported == 0 && params.fmt != NVJPEG_OUTPUT_NV12 && params.fmt != NVJPEG_OUTPUT_YUY2){
        batched_bitstreams.push_back((const unsigned char *)img_data[i].data());
        batched_bitstreams_size.push_back(img_len[i]);
        batched_output.push_back(out[i]);
      } else {
        otherdecode_bitstreams.push_back((const unsigned char *)img_data[i].data());
        otherdecode_bitstreams_size.push_back(img_len[i]);
        otherdecode_output.push_back(out[i]);
      }
    }
  } else if (!params.require_hardware) {
    for(int i = 0; i < params.batch_size; i++) {
      otherdecode_bitstreams.push_back((const unsigned char *)img_data[i].data());
      otherdecode_bitstreams_size.push_back(img_len[i]);
      otherdecode_output.push_back(out[i]);
    }
  }

  CHECK_CUDA(cudaEventRecord(startEvent, params.stream));

    if (params.require_hardware) {
      // The single-image API supports hardware scaling. Keep each parser/state
      // alive until its decode completes before reusing it for the next image.
      for (int i = 0; i < params.batch_size; ++i) {
        CHECK_NVJPEG(nvjpegJpegStreamParse(params.nvjpeg_handle,
            reinterpret_cast<const unsigned char *>(img_data[i].data()), img_len[i],
            0, 0, params.jpeg_streams[0]));
        CHECK_NVJPEG(nvjpegDecodeJpeg(params.nvjpeg_handle, params.nvjpeg_decoder,
            params.nvjpeg_decoupled_state, params.jpeg_streams[0], &out[i],
            params.nvjpeg_decode_params, params.stream));
        CHECK_CUDA(cudaStreamSynchronize(params.stream));
      }
    }

    if(batched_bitstreams.size() > 0)
     {
          CHECK_NVJPEG(
               nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state,
                                            batched_bitstreams.size(), 1, params.fmt));

         CHECK_NVJPEG(nvjpegDecodeBatched(
             params.nvjpeg_handle, params.nvjpeg_state, batched_bitstreams.data(),
             batched_bitstreams_size.data(), batched_output.data(), params.stream));
     }

    if(otherdecode_bitstreams.size() > 0)
    {
          CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(params.nvjpeg_decoupled_state, params.device_buffer));
          int buffer_index = 0;
          CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(params.nvjpeg_decode_params, params.fmt));
          for (size_t i = 0; i < otherdecode_bitstreams.size(); i++) {
              CHECK_NVJPEG(
                  nvjpegJpegStreamParse(params.nvjpeg_handle, otherdecode_bitstreams[i], otherdecode_bitstreams_size[i],
                  0, 0, params.jpeg_streams[buffer_index]));

              CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(params.nvjpeg_decoupled_state,
                  params.pinned_buffers[buffer_index]));

              CHECK_NVJPEG(nvjpegDecodeJpegHost(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                  params.nvjpeg_decode_params, params.jpeg_streams[buffer_index]));

              CHECK_CUDA(cudaStreamSynchronize(params.stream));

              CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                  params.jpeg_streams[buffer_index], params.stream));

              buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode to avoid an extra sync

              CHECK_NVJPEG(nvjpegDecodeJpegDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                  &otherdecode_output[i], params.stream));

          }
    }
  CHECK_CUDA(cudaEventRecord(stopEvent, params.stream));

  CHECK_CUDA(cudaEventSynchronize(stopEvent));
  CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
  time = 0.001 * static_cast<double>(loopTime); // cudaEventElapsedTime returns milliseconds
  CHECK_CUDA(cudaEventDestroy(startEvent));
  CHECK_CUDA(cudaEventDestroy(stopEvent));

  return EXIT_SUCCESS;
}

int write_images(std::vector<nvjpegImage_t> &iout, std::vector<int> &widths,
                 std::vector<int> &heights, decode_params_t &params,
                 FileNames &filenames) {
  for (int i = 0; i < params.batch_size; i++) {
    if (params.raw_output) {
      std::vector<unsigned char> output;
      for (int c = 0; c < NVJPEG_MAX_COMPONENT; ++c) {
        const size_t rows = params.output_rows[i][c];
        if (!rows) continue;
        const size_t offset = output.size(), pitch = iout[i].pitch[c];
        output.resize(offset + pitch * rows);
        CHECK_CUDA(cudaMemcpy2D(output.data() + offset, pitch, iout[i].channel[c], pitch,
            pitch, rows, cudaMemcpyDeviceToHost));
        // The last YUY2 pair has an unused second Y when the width is odd.
        if (params.fmt == NVJPEG_OUTPUT_YUY2 && widths[i] % 2)
          for (size_t row = 0; row < rows; ++row) output[offset + (row + 1) * pitch - 2] = 0;
      }
      const std::filesystem::path input(filenames[i]);
      auto name = std::filesystem::is_directory(params.input_dir)
          ? input.lexically_relative(params.input_dir) : input.filename();
      name += "." + params.format_name + ".s" + std::to_string(params.scale) + ".raw";
      if (!params.reference_dir.empty()) {
        const auto path = std::filesystem::path(params.reference_dir) / name;
        std::ifstream reference(path, std::ios::binary);
        if (!reference) throw sample_error(1, "Cannot open golden: " + path.string());
        const std::vector<unsigned char> golden((std::istreambuf_iterator<char>(reference)), {});
        if (reference.bad() || golden.empty()) throw sample_error(1, "Cannot read golden: " + path.string());
        if (golden != output) throw sample_error(4, "Golden mismatch: " + path.string());
      }
      if (params.write_decoded) {
        const auto path = std::filesystem::path(params.output_dir) / name;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(output.data()), output.size());
        file.close();
        if (!file) throw sample_error(1, "Cannot write output: " + path.string());
      }
      continue;
    }
    // Get the file name, without extension.
    // This will be used to rename the output file.
    size_t position = filenames[i].find_last_of("/\\");
    std::string sFileName =
        (std::string::npos == position)
            ? filenames[i]
            : filenames[i].substr(position + 1, filenames[i].size());
    position = sFileName.rfind(".");
    sFileName = (std::string::npos == position) ? sFileName
                                                : sFileName.substr(0, position);
    std::string fname(params.output_dir + "/" + sFileName + ".bmp");

    int err = 0;
    if (params.fmt == NVJPEG_OUTPUT_RGB || params.fmt == NVJPEG_OUTPUT_BGR) {
      err = writeBMP(fname.c_str(), iout[i].channel[0], iout[i].pitch[0],
                     iout[i].channel[1], iout[i].pitch[1], iout[i].channel[2],
                     iout[i].pitch[2], widths[i], heights[i]);
    } else if (params.fmt == NVJPEG_OUTPUT_RGBI ||
               params.fmt == NVJPEG_OUTPUT_BGRI) {
      // Write BMP from interleaved data
      err = writeBMPi(fname.c_str(), iout[i].channel[0], iout[i].pitch[0],
                      widths[i], heights[i]);
    }
    if (err) {
      std::cout << "Cannot write output file: " << fname << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Done writing decoded image to file: " << fname << std::endl;
  }
  return EXIT_SUCCESS;
}

int process_images(FileNames &image_names, decode_params_t &params,
                      double &total) {
  // vector for storing raw files and file lengths
  FileData file_data(params.batch_size);
  std::vector<size_t> file_len(params.batch_size);
  FileNames current_names(params.batch_size);
  std::vector<int> widths(params.batch_size);
  std::vector<int> heights(params.batch_size);
  // we wrap over image files to process total_images of files
  FileNames::iterator file_iter = image_names.begin();

  // stream for decoding
  CHECK_CUDA(
      cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));

  int total_processed = 0;

  // output buffers
  std::vector<nvjpegImage_t> iout(params.batch_size);
  // output buffer sizes, for convenience
  std::vector<nvjpegImage_t> isz(params.batch_size);
  params.output_rows.resize(params.batch_size);

  for (size_t i = 0; i < iout.size(); i++) {
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
      iout[i].channel[c] = NULL;
      iout[i].pitch[c] = 0;
      isz[i].pitch[c] = 0;
    }
  }

  double test_time = 0;
  int warmup = 0;
  const int batch_size = params.batch_size;
  try {
    while (total_processed < params.total_images) {
      // Process the final partial batch without changing the requested count.
      params.batch_size = std::min(batch_size, params.total_images - total_processed);
      if (read_next_batch(image_names, params.batch_size, file_iter, file_data,
                          file_len, current_names))
        throw sample_error(1, "Cannot read input batch");

      if (prepare_buffers(file_data, file_len, widths, heights, iout, isz,
                          current_names, params))
        throw sample_error(1, "Cannot prepare output buffers");

      double time;
      if (decode_images(file_data, file_len, iout, params, time))
        throw sample_error(1, "Cannot decode batch");

      if ((params.write_decoded || !params.reference_dir.empty()) &&
          write_images(iout, widths, heights, params, current_names))
        throw sample_error(1, "Cannot write decoded images");
      if (warmup < params.warmup) {
        warmup++;
      } else {
        total_processed += params.batch_size;
        params.decoded += params.batch_size;
        test_time += time;
      }

    }
  } catch (...) {
    // Drain work before file_data and the destination buffers go out of scope.
    cudaStreamSynchronize(params.stream);
    release_buffers(iout);
    throw;
  }
  params.batch_size = batch_size;
  total = test_time;

  release_buffers(iout);

  CHECK_CUDA(cudaStreamDestroy(params.stream));
  params.stream = NULL;

  return EXIT_SUCCESS;
}


int parse_integer(const char *value, int minimum) {
  int result = 0;
  const char *end = value + std::strlen(value);
  const auto parsed = std::from_chars(value, end, result);
  if (parsed.ec != std::errc{} || parsed.ptr != end || result < minimum)
    throw sample_error(2, std::string("Invalid integer: ") + value);
  return result;
}

int run(int argc, const char *argv[], decode_params_t &params) {
  int pidx;

  if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
      (pidx = findParamIndex(argv, argc, "--help")) != -1) {
    std::cout << "Usage: " << argv[0]
              << " -i images_dir [-b batch_size] [-t total_images] "
                 "[-w warmup_iterations] [-o output_dir] "
                 "[-fmt output_format] [-backend default|hardware] "
                 "[-scale 1|2|4|8] [-raw] [-reference directory] [-device id]\n";
    std::cout << "Parameters: " << std::endl;
    std::cout << "\timages_dir\t:\tPath to single image or directory of images"
              << std::endl;
    std::cout << "\tbatch_size\t:\tDecode images from input by batches of "
                 "specified size"
              << std::endl;
    std::cout << "\ttotal_images\t:\tDecode this much images, if there are "
                 "less images \n"
              << "\t\t\t\t\tin the input than total images, decoder will loop "
                 "over the input"
              << std::endl;
    std::cout << "\twarmup_iterations\t:\tRun this amount of batches first "
                 "without measuring performance"
              << std::endl;
    std::cout
        << "\toutput_dir\t:\tWrite decoded images (BMP for RGB/BGR, raw otherwise)"
        << std::endl;
    std::cout << "\toutput_format\t:\tnvJPEG output format for decoding. One "
                 "of [rgb, rgbi, bgr, bgri, yuv, y, unchanged, nv12, yuy2]"
              << std::endl;
    std::cout << "\t-backend hardware requires hardware; unsupported inputs fail without fallback.\n"
              << "\t-scale sets the hardware downscale divisor; requires -backend hardware.\n"
              << "\t-raw writes tightly packed raw output, including RGB/BGR.\n"
              << "\t-reference compares every decode against raw golden files.\n";
    params.help = true;
    return EXIT_SUCCESS;
  }

  params.batch_size = 1;
  params.total_images = -1;
  params.fmt = NVJPEG_OUTPUT_RGB;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "-raw") { params.raw_output = true; continue; }
    // Retain these previously accepted flags; backend routing selects the API.
    if (key == "-pipelined" || key == "-batched") continue;
    if (i + 1 == argc) throw sample_error(2, "Missing value for " + key);
    const char *value = argv[++i];
    if (key == "-i") params.input_dir = value;
    else if (key == "-o") { params.output_dir = value; params.write_decoded = true; }
    else if (key == "-reference") { params.reference_dir = value; params.raw_output = true; }
    else if (key == "-b") params.batch_size = parse_integer(value, 1);
    else if (key == "-t") params.total_images = parse_integer(value, 1);
    else if (key == "-w") params.warmup = parse_integer(value, 0);
    else if (key == "-device") params.dev = parse_integer(value, 0);
    else if (key == "-scale") params.scale = parse_integer(value, 1);
    else if (key == "-fmt") params.format_name = value;
    else if (key == "-backend") {
      if (std::string(value) != "default" && std::string(value) != "hardware")
        throw sample_error(2, "-backend must be default or hardware");
      params.require_hardware = std::string(value) == "hardware";
    } else throw sample_error(2, "Unknown option: " + key);
  }
  if (params.scale != 1 && params.scale != 2 && params.scale != 4 && params.scale != 8)
    throw sample_error(2, "-scale must be 1, 2, 4, or 8");
  if (params.scale != 1 && !params.require_hardware)
    throw sample_error(2, "Hardware scaling requires -backend hardware");
  if (params.input_dir.empty()) {
    // Search in default paths for input images.
     int found = getInputDir(params.input_dir, argv[0]);
    if (!found)
    {
      std::cout << "Please specify input directory with encoded images"<< std::endl;
      throw sample_error(2, "Please specify an input path with -i");
    }
  }

  {
    const std::string &sfmt = params.format_name;
    if (sfmt == "rgb")
      params.fmt = NVJPEG_OUTPUT_RGB;
    else if (sfmt == "bgr")
      params.fmt = NVJPEG_OUTPUT_BGR;
    else if (sfmt == "rgbi")
      params.fmt = NVJPEG_OUTPUT_RGBI;
    else if (sfmt == "bgri")
      params.fmt = NVJPEG_OUTPUT_BGRI;
    else if (sfmt == "yuv")
      params.fmt = NVJPEG_OUTPUT_YUV;
    else if (sfmt == "y")
      params.fmt = NVJPEG_OUTPUT_Y;
    else if (sfmt == "unchanged")
      params.fmt = NVJPEG_OUTPUT_UNCHANGED;
    else if (sfmt == "nv12")
      params.fmt = NVJPEG_OUTPUT_NV12;
    else if (sfmt == "yuy2")
      params.fmt = NVJPEG_OUTPUT_YUY2;
    else {
      throw sample_error(2, "Unknown format: " + sfmt);
    }
  }

  if (params.fmt != NVJPEG_OUTPUT_RGB && params.fmt != NVJPEG_OUTPUT_BGR &&
      params.fmt != NVJPEG_OUTPUT_RGBI && params.fmt != NVJPEG_OUTPUT_BGRI)
    params.raw_output = true;
  if (params.write_decoded) std::filesystem::create_directories(params.output_dir);
  CHECK_CUDA(cudaSetDevice(params.dev));

  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
  nvjpegPinnedAllocator_t pinned_allocator ={&host_malloc, &host_free};

  nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, &dev_allocator,
                                &pinned_allocator,NVJPEG_FLAGS_DEFAULT,  &params.nvjpeg_handle);
  params.hw_decode_available = true;
  if( status == NVJPEG_STATUS_ARCH_MISMATCH && !params.require_hardware) {
    std::cout<<"Hardware Decoder not supported. Falling back to default backend"<<std::endl;
    CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
                              &pinned_allocator, NVJPEG_FLAGS_DEFAULT, &params.nvjpeg_handle));
    params.hw_decode_available = false;
  } else {
    CHECK_NVJPEG(status);
  }

  CHECK_NVJPEG(
      nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));



  create_decoupled_api_handles(params);
  std::cout << "CONFIG backend=" << (params.require_hardware ? "hardware" : "default")
            << " format=" << params.format_name << " scale=" << params.scale << '\n';

  // read source images
  FileNames image_names;
  if (readInput(params.input_dir, image_names) || image_names.empty())
    throw sample_error(1, "No JPEG input images found");

  if (params.total_images == -1) {
    params.total_images = image_names.size();
  }

  std::cout << "Decoding images in directory: " << params.input_dir
            << ", total " << params.total_images << ", batchsize "
            << params.batch_size << std::endl;

  double total;
  if (process_images(image_names, params, total)) return EXIT_FAILURE;
  std::cout << "Total decoding time: " << total << " (s)" << std::endl;
  std::cout << "Avg decoding time per image: " << total / params.total_images 
            << " (s)" << std::endl;
  std::cout << "Avg images per sec: " << params.total_images / total
            << std::endl;
  std::cout << "Avg decoding time per batch: "
            << total / ((params.total_images + params.batch_size - 1) /
                        params.batch_size) 
            << " (s)" << std::endl;

  return EXIT_SUCCESS;
}

int main(int argc, const char *argv[]) {
  decode_params_t params{};
  int result = 0;
  try {
    result = run(argc, argv, params);
  } catch (const sample_error &error) {
    result = error.code;
    std::cerr << "RESULT status=FAIL code=" << result << " decoded=" << params.decoded
              << " message=" << std::quoted(error.what()) << '\n';
  } catch (const std::exception &error) {
    result = 1;
    std::cerr << "RESULT status=FAIL code=1 decoded=" << params.decoded
              << " message=" << std::quoted(error.what()) << '\n';
  }
  if (params.stream) cudaStreamSynchronize(params.stream);
  destroy_decoupled_api_handles(params);
  if (params.nvjpeg_state) nvjpegJpegStateDestroy(params.nvjpeg_state);
  if (params.nvjpeg_handle) nvjpegDestroy(params.nvjpeg_handle);
  if (params.stream) cudaStreamDestroy(params.stream);
  if (result == 0 && !params.help)
    std::cout << "RESULT status=PASS code=0 decoded=" << params.decoded << '\n';
  return result;
}
