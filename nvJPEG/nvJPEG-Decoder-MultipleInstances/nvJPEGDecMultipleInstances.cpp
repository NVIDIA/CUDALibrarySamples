/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
  

#include "nvJPEGDecMultipleInstances.h"
#include "threadpool.h"

float get_scale_factor(nvjpegChromaSubsampling_t chroma_subsampling)
{
  float scale_factor = 3.0; // it should be 3.0 for 444
  if(chroma_subsampling == NVJPEG_CSS_420 || chroma_subsampling == NVJPEG_CSS_411) {
    scale_factor = 1.5;
  }
  else if(chroma_subsampling == NVJPEG_CSS_422 || chroma_subsampling == NVJPEG_CSS_440) {
    scale_factor = 2.0;
  }
  else if(chroma_subsampling == NVJPEG_CSS_410) {
    scale_factor = 1.25;
  }
  else if(chroma_subsampling == NVJPEG_CSS_GRAY){
    scale_factor = 1.0;
  }

  return scale_factor;
}

bool pick_gpu_backend(nvjpegJpegStream_t&  jpeg_stream)
{
  unsigned int frame_width,frame_height;
  nvjpegChromaSubsampling_t chroma_subsampling;
      
  CHECK_NVJPEG(nvjpegJpegStreamGetFrameDimensions(jpeg_stream,
        &frame_width, &frame_height));
  CHECK_NVJPEG(nvjpegJpegStreamGetChromaSubsampling(jpeg_stream,&chroma_subsampling));
  auto scale_factor = get_scale_factor(chroma_subsampling);

  bool use_gpu_backend = false;
  // use NVJPEG_BACKEND_GPU_HYBRID when dimensions are greater than 512x512
  if( frame_height*frame_width * scale_factor > 512*512 * 3)
  {
    use_gpu_backend = true;
  }
  return use_gpu_backend;
}


int decode_images(const FileData &img_data, const std::vector<size_t> &img_len,
                  std::vector<nvjpegImage_t> &out, decode_params_t &params, ThreadPool &workers,
                  double &time) {
  cudaEvent_t startEvent = NULL, stopEvent = NULL;
  float loopTime = 0; 
  
  CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
  CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));

  CHECK_CUDA(cudaEventRecord(startEvent, params.global_stream));
  
  if(params.num_threads < 2)
  {
    auto& per_thread_params = params.nvjpeg_per_thread_data[0];
    int buffer_index = 0;
    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(per_thread_params.nvjpeg_decode_params, params.fmt));
    for (int i = 0; i < params.batch_size; i++) {
        CHECK_NVJPEG(
            nvjpegJpegStreamParse(params.nvjpeg_handle, (const unsigned char *)img_data[i].data(), img_len[i],
            0, 0, per_thread_params.jpeg_streams[buffer_index]));

        bool use_gpu_backend =  pick_gpu_backend(per_thread_params.jpeg_streams[buffer_index]);
      
        nvjpegJpegDecoder_t& decoder =   use_gpu_backend ?  per_thread_params.nvjpeg_dec_gpu: per_thread_params.nvjpeg_dec_cpu;
        nvjpegJpegState_t&   decoder_state = use_gpu_backend ? per_thread_params.dec_state_gpu:per_thread_params.dec_state_cpu;

        CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(decoder_state,
            per_thread_params.pinned_buffers[buffer_index]));

        CHECK_NVJPEG(nvjpegDecodeJpegHost(params.nvjpeg_handle, decoder, decoder_state,
            per_thread_params.nvjpeg_decode_params, per_thread_params.jpeg_streams[buffer_index]));

        CHECK_CUDA(cudaStreamSynchronize(per_thread_params.stream));

        CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, decoder, decoder_state,
            per_thread_params.jpeg_streams[buffer_index], per_thread_params.stream));

        CHECK_NVJPEG(nvjpegDecodeJpegDevice(params.nvjpeg_handle, decoder, decoder_state,
            &out[i], per_thread_params.stream));

        buffer_index = (buffer_index+1)%pipeline_stages; // switch pinned buffer in pipeline mode to avoid an extra sync

    }
    CHECK_CUDA(cudaStreamSynchronize(per_thread_params.stream))
  }
  else
  {
    std::vector<int> buffer_indices(params.num_threads, 0);
    
    for (int i = 0; i < params.batch_size; i++) {
        workers.enqueue(std::bind(
            [&params, &buffer_indices, &out, &img_data, &img_len](int iidx, int thread_idx)
                {
                  auto& per_thread_params = params.nvjpeg_per_thread_data[thread_idx];
                  
                  CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(per_thread_params.nvjpeg_decode_params, params.fmt));
                  CHECK_NVJPEG(nvjpegJpegStreamParse(params.nvjpeg_handle, (const unsigned char *)img_data[iidx].data(), img_len[iidx],
                    0, 0, per_thread_params.jpeg_streams[buffer_indices[thread_idx]]));
                  bool use_gpu_backend =  pick_gpu_backend(per_thread_params.jpeg_streams[buffer_indices[thread_idx]]);
      
                  nvjpegJpegDecoder_t& decoder =   use_gpu_backend ?  per_thread_params.nvjpeg_dec_gpu: per_thread_params.nvjpeg_dec_cpu;
                  nvjpegJpegState_t&   decoder_state = use_gpu_backend ? per_thread_params.dec_state_gpu:per_thread_params.dec_state_cpu;

                  CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(decoder_state,
                    per_thread_params.pinned_buffers[buffer_indices[thread_idx]]));

                  CHECK_NVJPEG(nvjpegDecodeJpegHost(params.nvjpeg_handle, decoder, decoder_state,
                    per_thread_params.nvjpeg_decode_params, per_thread_params.jpeg_streams[buffer_indices[thread_idx]]));

                  CHECK_CUDA(cudaStreamSynchronize(per_thread_params.stream));

                  CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, decoder, decoder_state,
                    per_thread_params.jpeg_streams[buffer_indices[thread_idx]], per_thread_params.stream));

                  CHECK_NVJPEG(nvjpegDecodeJpegDevice(params.nvjpeg_handle, decoder, decoder_state,
                    &out[iidx], per_thread_params.stream));

                  // switch pinned buffer in pipeline mode to avoid an extra sync
                  buffer_indices[thread_idx] = (buffer_indices[thread_idx]+1)%pipeline_stages;
                  return EXIT_SUCCESS; // the CHECK_ statements returns 1 on failure, so we need to return a value here too.
                }, i, std::placeholders::_1
                )
            );
    }
    workers.wait();
    for ( auto& per_thread_params : params.nvjpeg_per_thread_data) {
        CHECK_CUDA(cudaStreamSynchronize(per_thread_params.stream))
    }
  }
  
  
  
  
  CHECK_CUDA(cudaEventRecord(stopEvent, params.global_stream));
  CHECK_CUDA(cudaEventSynchronize(stopEvent));
  CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
  time = 0.001 * static_cast<double>(loopTime); // cudaEventElapsedTime returns milliseconds

  return EXIT_SUCCESS;
}

int write_images(std::vector<nvjpegImage_t> &iout, std::vector<int> &widths,
                 std::vector<int> &heights, decode_params_t &params,
                 FileNames &filenames) {
  for (int i = 0; i < params.batch_size; i++) {
    // Get the file name, without extension.
    // This will be used to rename the output file.
    size_t position = filenames[i].rfind("/");
    std::string sFileName =
        (std::string::npos == position)
            ? filenames[i]
            : filenames[i].substr(position + 1, filenames[i].size());
    position = sFileName.rfind(".");
    sFileName = (std::string::npos == position) ? sFileName
                                                : sFileName.substr(0, position);
    std::string fname(params.output_dir + "/" + sFileName + ".bmp");

    int err;
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

double process_images(FileNames &image_names, decode_params_t &params,
                      double &total) {
  // vector for storing raw files and file lengths
  FileData file_data(params.batch_size);
  std::vector<size_t> file_len(params.batch_size);
  FileNames current_names(params.batch_size);
  std::vector<int> widths(params.batch_size);
  std::vector<int> heights(params.batch_size);
  // we wrap over image files to process total_images of files
  FileNames::iterator file_iter = image_names.begin();

  int total_processed = 0;

  // output buffers
  std::vector<nvjpegImage_t> iout(params.batch_size);
  // output buffer sizes, for convenience
  std::vector<nvjpegImage_t> isz(params.batch_size);

  CHECK_CUDA(cudaStreamCreateWithFlags(&params.global_stream, cudaStreamNonBlocking));

  for (int i = 0; i < iout.size(); i++) {
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
      iout[i].channel[c] = NULL;
      iout[i].pitch[c] = 0;
      isz[i].pitch[c] = 0;
    }
  }
  ThreadPool workers(params.num_threads);
  
  double test_time = 0;
  int warmup = 0;
  while (total_processed < params.total_images) {
    if (read_next_batch(image_names, params.batch_size, file_iter, file_data,
                        file_len, current_names))
      return EXIT_FAILURE;

    if (prepare_buffers(file_data, file_len, widths, heights, iout, isz,
                        current_names, params))
      return EXIT_FAILURE;

    double time;
    if (decode_images(file_data, file_len, iout, params, workers, time))
      return EXIT_FAILURE;
    if (warmup < params.warmup) {
      warmup++;
    } else {
      total_processed += params.batch_size;
      test_time += time;
    }

    if (params.write_decoded)
      write_images(iout, widths, heights, params, current_names);
  }
  total = test_time;

  release_buffers(iout);

  CHECK_CUDA(cudaStreamDestroy(params.global_stream));

  return EXIT_SUCCESS;
}


int main(int argc, const char *argv[]) {
  int pidx;

  if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
      (pidx = findParamIndex(argv, argc, "--help")) != -1) {
    std::cout << "Usage: " << argv[0]
              << " -i images_dir [-b batch_size] [-t total_images] "
                 "[-w warmup_iterations] [-o output_dir] "
                 "[-pipelined] [-batched] [-fmt output_format]\n";
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
        << "\toutput_dir\t:\tWrite decoded images as BMPs to this directory"
        << std::endl;
    std::cout << "\tpipelined\t:\tUse decoding in phases" << std::endl;
    std::cout << "\tbatched\t\t:\tUse batched interface" << std::endl;
    std::cout << "\toutput_format\t:\tnvJPEG output format for decoding. One "
                 "of [rgb, rgbi, bgr, bgri, yuv, y, unchanged]"
              << std::endl;
    return EXIT_SUCCESS;
  }

  decode_params_t params;

  params.input_dir = "./";
  if ((pidx = findParamIndex(argv, argc, "-i")) != -1) {
    params.input_dir = argv[pidx + 1];
  } else {
    // Search in default paths for input images.
     int found = getInputDir(params.input_dir, argv[0]);
    if (!found)
    {
      std::cout << "Please specify input directory with encoded images"<< std::endl;
      return EXIT_FAILURE;
    }
  }

  params.batch_size = 1;
  if ((pidx = findParamIndex(argv, argc, "-b")) != -1) {
    params.batch_size = std::atoi(argv[pidx + 1]);
  }

  params.total_images = -1;
  if ((pidx = findParamIndex(argv, argc, "-t")) != -1) {
    params.total_images = std::atoi(argv[pidx + 1]);
  }

  params.warmup = 0;
  if ((pidx = findParamIndex(argv, argc, "-w")) != -1) {
    params.warmup = std::atoi(argv[pidx + 1]);
  }

  params.num_threads = 1;
  if ((pidx = findParamIndex(argv, argc, "-j")) != -1) {
    params.num_threads = std::atoi(argv[pidx + 1]);
  }

  params.fmt = NVJPEG_OUTPUT_RGB;
  if ((pidx = findParamIndex(argv, argc, "-fmt")) != -1) {
    std::string sfmt = argv[pidx + 1];
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
    else {
      std::cout << "Unknown format: " << sfmt << std::endl;
      return EXIT_FAILURE;
    }
  }

  params.write_decoded = false;
  if ((pidx = findParamIndex(argv, argc, "-o")) != -1) {
    params.output_dir = argv[pidx + 1];
    if (params.fmt != NVJPEG_OUTPUT_RGB && params.fmt != NVJPEG_OUTPUT_BGR &&
        params.fmt != NVJPEG_OUTPUT_RGBI && params.fmt != NVJPEG_OUTPUT_BGRI) {
      std::cout << "We can write ony BMPs, which require output format be "
                   "either RGB/BGR or RGBi/BGRi"
                << std::endl;
      return EXIT_FAILURE;
    }
    params.write_decoded = true;
  }

  params.nvjpeg_per_thread_data.resize(params.num_threads);
  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
  nvjpegPinnedAllocator_t pinned_allocator ={&host_malloc, &host_free};

  nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
                                &pinned_allocator,NVJPEG_FLAGS_DEFAULT,  &params.nvjpeg_handle);

  for(auto& nvjpeg_data : params.nvjpeg_per_thread_data)
    create_nvjpeg_data(params.nvjpeg_handle, nvjpeg_data);

  // read source images
  FileNames image_names;
  readInput(params.input_dir, image_names);

  if (params.total_images == -1) {
    params.total_images = image_names.size();
  } else if (params.total_images % params.batch_size) {
    params.total_images =
        ((params.total_images) / params.batch_size) * params.batch_size;
    std::cout << "Changing total_images number to " << params.total_images
              << " to be multiple of batch_size - " << params.batch_size
              << std::endl;
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

  for(auto& nvjpeg_data : params.nvjpeg_per_thread_data)
    destroy_nvjpeg_data(nvjpeg_data);

  CHECK_NVJPEG(nvjpegDestroy(params.nvjpeg_handle));

  return EXIT_SUCCESS;
}
