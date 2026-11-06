/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "nvjpeg2k_dec_pipelined.h"

int write_image(std::string output_path, std::string filename, const nvjpeg2kImage_t &imgdesc, int width, int height,
               uint32_t num_components, uint8_t precision, uint8_t sgn, bool verbose, bool is_nlt3)
{
    // Get the file name, without extension.
    // This will be used to rename the output file.
    // fs::path accepts both / and \ as separators on Windows.
    std::string sFileName = fs::path(filename).stem().string();

    int err = EXIT_DECODE_FAILURE;
    if (is_nlt3)
    {
        std::string fname(output_path + separator + sFileName + ".pfm");
        err = writePFM(fname.c_str(), imgdesc, width, height, num_components);
        if (err)
        {
            std::cout << "Cannot write output file: " << fname << std::endl;
        }
        return err;
    }

    // For single component image output as PGM
    if (num_components == 1)
    {
        std::string fname(output_path + separator + sFileName + ".pgm");
        if (imgdesc.pixel_type == NVJPEG2K_UINT8)
        {
            err = writePGM<unsigned char>(fname.c_str(), (unsigned char *)imgdesc.pixel_data[0], 
                imgdesc.pitch_in_bytes[0], width, height, precision, sgn);
        }
        else if (imgdesc.pixel_type == NVJPEG2K_UINT16)
        {
            err = writePGM<unsigned short>(fname.c_str(), (unsigned short *)imgdesc.pixel_data[0],
                 imgdesc.pitch_in_bytes[0], width, height, precision, sgn);
        }
        else if(imgdesc.pixel_type == NVJPEG2K_INT16)
        {
            err = writePGM<short>(fname.c_str(), (short *)imgdesc.pixel_data[0],
                imgdesc.pitch_in_bytes[0], width, height, precision, sgn);
        }
        if (err)
        {
            std::cout << "Cannot write output file: " << fname << std::endl;
        }
    }
    else if (num_components == 3 || num_components == 4)
    {
        if (imgdesc.pixel_type != NVJPEG2K_UINT8 && imgdesc.pixel_type != NVJPEG2K_UINT16)
        {
            std::cout << "Signed 3- and 4-component images cannot be written by this sample" << std::endl;
            return EXIT_UNSUPPORTED;
        }
        if(num_components == 4 && verbose)
        {
            std::cout<<"Discarding the alpha channel when writing to file"<<std::endl;
        }
        size_t img_size = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
        if(img_size < std::numeric_limits<uint32_t>::max())
        {
            std::string fname(output_path + separator + sFileName + ".bmp");
            if (imgdesc.pixel_type == NVJPEG2K_UINT8)
            {
                err = writeBMP<unsigned char>(fname.c_str(),
                        (unsigned char *)imgdesc.pixel_data[0], imgdesc.pitch_in_bytes[0],
                        (unsigned char *)imgdesc.pixel_data[1], imgdesc.pitch_in_bytes[1],
                        (unsigned char *)imgdesc.pixel_data[2], imgdesc.pitch_in_bytes[2],
                        width, height, precision, verbose);
            }
            else if (imgdesc.pixel_type == NVJPEG2K_UINT16)
            {
                err = writeBMP<unsigned short>(fname.c_str(),
                        (unsigned short *)imgdesc.pixel_data[0], imgdesc.pitch_in_bytes[0],
                        (unsigned short *)imgdesc.pixel_data[1], imgdesc.pitch_in_bytes[1],
                        (unsigned short *)imgdesc.pixel_data[2], imgdesc.pitch_in_bytes[2],
                        width, height, precision, verbose);
            }
            if (err)
            {
                std::cout << "Cannot write output file: " << fname << std::endl;
            }
        }
        else
        {
            if(verbose)
            {
                std::cout<<"uncompressed size > 4GB. Writing it as .ppm"<<std::endl;
            }
            std::string fname(output_path + separator + sFileName + ".ppm");
            if (imgdesc.pixel_type == NVJPEG2K_UINT8)
            {
                err = writePPM<unsigned char>(fname.c_str(),
                        (unsigned char *)imgdesc.pixel_data[0], imgdesc.pitch_in_bytes[0],
                        (unsigned char *)imgdesc.pixel_data[1], imgdesc.pitch_in_bytes[1],
                        (unsigned char *)imgdesc.pixel_data[2], imgdesc.pitch_in_bytes[2],
                        width, height, precision, verbose);
            }
            else if (imgdesc.pixel_type == NVJPEG2K_UINT16)
            {
                err = writePPM<unsigned short>(fname.c_str(),
                        (unsigned short *)imgdesc.pixel_data[0], imgdesc.pitch_in_bytes[0],
                        (unsigned short *)imgdesc.pixel_data[1], imgdesc.pitch_in_bytes[1],
                        (unsigned short *)imgdesc.pixel_data[2], imgdesc.pitch_in_bytes[2],
                        width, height, precision, verbose);
            }
            if (err)
            {
                std::cout << "Cannot write output file: " << fname << std::endl;
            }
        }
    }
    else
    {
        std::cout << "num channels not supported"<<std::endl;
        return EXIT_UNSUPPORTED;
    }
    
    return err;
}

int prepare_buffers(FileData &file_data, std::vector<size_t> &file_len,
                    std::vector<nvjpeg2kImageSample_t> &ibuf,
                    FileNames &current_names,
                    decode_params_t &params,
                    double& parse_time) {

    nvjpeg2kImageInfo_t image_info;
    nvjpeg2kImageComponentInfo_t image_comp_info[NUM_COMPONENTS];
    parse_time = 0;
    for (int i = 0; i < params.batch_size; i++) 
    {
        auto io_start = perfclock::now();
        CHECK_NVJPEG2K(nvjpeg2kStreamParse(params.nvjpeg2k_handle, (unsigned char*)file_data[i].data(), file_len[i],
                0, 0, params.jpeg2k_streams[i]));
        auto io_end = perfclock::now();
        parse_time += std::chrono::duration_cast<std::chrono::seconds>(io_end-io_start).count();
        
        CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(params.jpeg2k_streams[i], &image_info));

        if( image_info.num_components > NUM_COMPONENTS) 
        {
            std::cout<<"Num Components > "<< NUM_COMPONENTS<<"not supported by this sample"<<std::endl;
            return EXIT_UNSUPPORTED;
        }
        for (uint32_t c = 0; c < image_info.num_components; c++)
        {
            CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(params.jpeg2k_streams[i], &image_comp_info[c], c));
        }

        ibuf[i].num_comps = image_info.num_components;
        ibuf[i].is_nlt3 = false;
        uint8_t nlt_bit_depth = image_comp_info[0].precision;
#if NVJPEG2K_HAS_NLT3_DECODE
        nvjpeg2kNLTInfo_t nlt_info;
        if (nvjpeg2kStreamGetComponentNLTInfo(params.jpeg2k_streams[i], 0, &nlt_info) == NVJPEG2K_STATUS_SUCCESS)
        {
            ibuf[i].is_nlt3 = (nlt_info.type == NVJPEG2K_NLT_TYPE3);
            nlt_bit_depth = nlt_info.bit_depth;
        }
#endif

        if(ibuf[i].is_nlt3)
        {
#if NVJPEG2K_HAS_NLT3_DECODE
            if(nlt_bit_depth == 16)
            {
                ibuf[i].pixel_type = NVJPEG2K_FP16;
            }
            else if(nlt_bit_depth > 16 && nlt_bit_depth <= 32)
            {
                ibuf[i].pixel_type = NVJPEG2K_FP32;
            }
            else
            {
                std::cout<<"NLT bit depth "<< static_cast<int>(nlt_bit_depth) <<" not supported by this sample"<<std::endl;
                return EXIT_UNSUPPORTED;
            }
#else
            std::cout<<"NLT type 3 decode requires nvJPEG2000 0.11 or newer"<<std::endl;
            return EXIT_UNSUPPORTED;
#endif
        }
        else if( image_comp_info[0].precision <= 8)
        {
            ibuf[i].pixel_type = NVJPEG2K_UINT8;
        }
        else if(image_comp_info[0].precision <= MAX_PRECISION)
        {
            ibuf[i].pixel_type = image_comp_info[0].sgn ? NVJPEG2K_INT16 : NVJPEG2K_UINT16;
        }
        else
        {
            std::cout<<"Precision > "<< MAX_PRECISION<<" not supported by this sample"<<std::endl;
            return EXIT_UNSUPPORTED;
        }

        uint32_t bytes_per_element = bytes_per_sample(ibuf[i].pixel_type);
        // realloc output buffer if required
        for (uint32_t c = 0; c < image_info.num_components; c++)
        {
            // for JPEG 2000 bitstreams with 420/422 subsampling, this sample enables RGB output
            // we are allocating assuming that all component dimensions are the same
            size_t aw = bytes_per_element * image_info.image_width;
            size_t ah = image_info.image_height;
            size_t sz = aw * ah;
            ibuf[i].pitch_in_bytes[c] = aw;
            if (sz > ibuf[i].comp_sz[c]) 
            {
                if (ibuf[i].component[c]) 
                {
                    CHECK_CUDA(cudaFree(ibuf[i].component[c]));
                }
                CHECK_CUDA(cudaMalloc((void**)&ibuf[i].component[c], sz));
                ibuf[i].comp_sz[c] = sz;
            }
        }
    }
    return EXIT_SUCCESS;
}

int free_buffers(std::vector<nvjpeg2kImageSample_t> &ibuf)
{
    for(auto& buf: ibuf)
    {
        for(int c = 0; c < NUM_COMPONENTS; c++)
        {
            if(buf.component[c])
            {
                CHECK_CUDA(cudaFree(buf.component[c]));
            }
            buf.component[c] = nullptr;
            buf.comp_sz[c] = 0;
            buf.pitch_in_bytes[c] = 0;
        }
        buf.num_comps = 0;
        buf.is_nlt3 = false;
    }
    return EXIT_SUCCESS;
}

int decode_images(FileNames &current_names, std::vector<nvjpeg2kImageSample_t> &out,
                  decode_params_t &params, double &time)
{
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    float loopTime = 0;

    cudaEvent_t pipeline_events[PIPELINE_STAGES];

    for (int p = 0; p < PIPELINE_STAGES; p++)
    {
        CHECK_CUDA(cudaEventCreate(&pipeline_events[p]));
    }
    CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));
    nvjpeg2kDecodeParams_t decode_params;
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsCreate(&decode_params));

    // set RGB  output for the entire batch, applies only to images with 420/422 subsampling
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetRGBOutput(decode_params, 1));

    std::vector<nvjpeg2kImage_t> nvjpeg2k_output;
    nvjpeg2k_output.resize(params.batch_size);

    for( int i = 0; i < params.batch_size; i++) 
    {
        nvjpeg2k_output[i].pixel_type = out[i].pixel_type;
        nvjpeg2k_output[i].num_components = out[i].num_comps;
        nvjpeg2k_output[i].pixel_data =  out[i].component;
        nvjpeg2k_output[i].pitch_in_bytes = out[i].pitch_in_bytes;
    }
    
    int buffer_index = 0;
    CHECK_CUDA(cudaEventRecord(startEvent, params.stream[0]));
    for( int i = 0; i < params.batch_size; i++)
    {
        if( i >= PIPELINE_STAGES)
        {
            // make sure that the previous stage are done
            CHECK_CUDA(cudaEventSynchronize(pipeline_events[buffer_index]));
        }

        CHECK_NVJPEG2K(nvjpeg2kDecodeImage(params.nvjpeg2k_handle, params.nvjpeg2k_decode_states[buffer_index],
            params.jpeg2k_streams[i], decode_params, &nvjpeg2k_output[i], params.stream[buffer_index]));

        CHECK_CUDA(cudaEventRecord(pipeline_events[buffer_index], params.stream[buffer_index]))

        buffer_index++;
        buffer_index = buffer_index%PIPELINE_STAGES;

    }
    for (int p = 0; p < PIPELINE_STAGES; p++)
    {
        CHECK_CUDA(cudaEventSynchronize(pipeline_events[p]));
    }

    CHECK_CUDA(cudaEventRecord(stopEvent, params.stream[0]));
    
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
    time += static_cast<double>(loopTime/1000.0); // loopTime is in milliseconds
    
    int result = EXIT_SUCCESS;
    if (params.write_decoded)
    {
        for( int i = 0; i < params.batch_size; i++)
        {
            nvjpeg2kImageInfo_t image_info;
            nvjpeg2kImageComponentInfo_t comp_info;
            CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(params.jpeg2k_streams[i], &image_info));
             // assume all components have the same precision
            CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(params.jpeg2k_streams[i], &comp_info, 0));

            int status = write_image(params.output_dir, current_names[i], nvjpeg2k_output[i], image_info.image_width,
                image_info.image_height, image_info.num_components, comp_info.precision,
                comp_info.sgn, params.verbose, out[i].is_nlt3);
            if(status)
            {
                result = status;
            }
        }
    }
    
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsDestroy(decode_params));
    for(int p = 0; p < PIPELINE_STAGES; p++)
    {
        CHECK_CUDA(cudaEventDestroy(pipeline_events[p]));
    }
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    
    return result;
}

int process_images(FileNames &image_names, decode_params_t &params,
                   double &total)
{
    // vector for storing raw files and file lengths
    FileData file_data(params.batch_size);
    std::vector<size_t> file_len(params.batch_size);
    FileNames current_names(params.batch_size);
    // we wrap over image files to process total_images of files
    FileNames::iterator file_iter = image_names.begin();
   // output buffers
   std::vector<nvjpeg2kImageSample_t> iout(params.batch_size);

    // stream for decoding
    for (int p =0; p < PIPELINE_STAGES; p++)
    {
        CHECK_CUDA(cudaStreamCreateWithFlags(&params.stream[p], cudaStreamNonBlocking));
    }

    int total_processed = 0;

    double test_time = 0;
    int warmup = 0;
    int result = EXIT_SUCCESS;
    while (total_processed < params.total_images)
    {
        if (read_next_batch(image_names, params.batch_size, file_iter, file_data,
                            file_len, current_names, params.verbose))
        {
            result = EXIT_MISSING_INPUT;
            break;
        }
        double parsetime = 0;
        // prepare_buffers and decode_images return EXIT_DECODE_FAILURE when decoding or writing fails, EXIT_FAILURE for CUDA errors
        int status = prepare_buffers(file_data, file_len, iout,
                        current_names, params, parsetime);
        if (status)
        {
            result = status;
            break;
        }

        double time = 0;
        status = decode_images(current_names, iout, params, time);
        if (status)
        {
            result = status;
            break;
        }
        if (warmup < params.warmup)
        {
            warmup++;
        }
        else
        {
            total_processed += params.batch_size;
            test_time += time + parsetime;
        }
    }
    total = test_time;
    for (int p = 0; p < PIPELINE_STAGES; p++)
    {
        CHECK_CUDA(cudaStreamDestroy(params.stream[p]));
    }

    if(free_buffers(iout))
        result = EXIT_FAILURE;

    return result;
}

int main(int argc, const char *argv[])
{
    int pidx;

    if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
        (pidx = findParamIndex(argv, argc, "--help")) != -1)
    {
        std::cout << "Usage: " << argv[0]
                  << " -i images_dir [-b batch_size] [-t total_images] "
                     "[-w warmup_iterations] [-o output_dir] [-v]"
                  << std::endl;
        std::cout << "Parameters: " << std::endl;
        std::cout << "\timages_dir\t:\tPath to a single image or a directory of images (searched recursively),\n"
                     "\t\t\t\tas .jp2 files or JPEG 2000 codestreams, Part 1 or HTJ2K"
                  << std::endl;
        std::cout << "\tbatch_size\t:\tDecode images from input by batches of "
                     "specified size"
                  << std::endl;
        std::cout << "\ttotal_images\t:\tDecode these many images, if there are "
                     "fewer images\n"
                  << "\t\t\t\tin the input than total images, decoder will loop "
                     "over the input"
                  << std::endl;
        std::cout << "\twarmup_iterations:\tRun these many batches first "
                     "without measuring performance"
                  << std::endl;
        std::cout << "\toutput_dir\t:\tWrite decoded images to this directory: 1-component images as .pgm,\n"
                     "\t\t\t\t3- and 4-component images as .bmp (alpha is dropped; .ppm above 4 GB),\n"
                     "\t\t\t\tand NLT type 3 images as .pfm;\n"
                     "\t\t\t\t4:2:0 and 4:2:2 images are converted to RGB"
                  << std::endl;
        std::cout << "\t-v\t\t:\tLog verbose messages to console"
                  << std::endl;
        std::cout << "Exit codes: 0 success, 2 invalid arguments, 3 missing input, 4 empty input directory,\n"
                     "\t5 image not supported by this sample, 6 decode or write failure, 1 any other error"
                  << std::endl;
        return EXIT_SUCCESS;
    }

    decode_params_t params;

    params.input_dir = "./";
    if ((pidx = findParamIndex(argv, argc, "-i")) != -1)
    {
        params.input_dir = paramValue(argv, argc, pidx);
    }
    else
    {
        // Search in default paths for input images.
        int found = getInputDir(params.input_dir, argv[0]);
        if (!found)
        {
            std::cout << "Please specify input directory with encoded images" << std::endl;
            return EXIT_INVALID_ARGUMENTS;
        }
    }

    params.batch_size = 1;
    if ((pidx = findParamIndex(argv, argc, "-b")) != -1)
    {
        params.batch_size = paramInt(argv, argc, pidx);
        if (params.batch_size <= 0)
        {
            std::cout << "Batch size must be a positive integer" << std::endl;
            return EXIT_INVALID_ARGUMENTS;
        }
    }

    params.total_images = -1;
    if ((pidx = findParamIndex(argv, argc, "-t")) != -1)
    {
        params.total_images = paramInt(argv, argc, pidx);
        if (params.total_images <= 0)
        {
            std::cout << "Total images must be a positive integer" << std::endl;
            return EXIT_INVALID_ARGUMENTS;
        }
    }

    params.warmup = 0;
    if ((pidx = findParamIndex(argv, argc, "-w")) != -1)
    {
        params.warmup = paramInt(argv, argc, pidx);
        if (params.warmup < 0)
        {
            std::cout << "Warmup iterations cannot be negative" << std::endl;
            return EXIT_INVALID_ARGUMENTS;
        }
    }

    params.write_decoded = false;
    if ((pidx = findParamIndex(argv, argc, "-o")) != -1)
    {
        params.output_dir = paramValue(argv, argc, pidx);
        params.write_decoded = true;
    }
    params.verbose = false;
    if ((pidx = findParamIndex(argv, argc, "-v")) != -1)
    {
        params.verbose = true;
    }

    if( params.verbose)
    {
        if(params.write_decoded) 
        {
            std::cout << "1-component images are written as .pgm, 3- and 4-component images as .bmp (.ppm above 4 GB), "
                         "and NLT type 3 images as .pfm"
                      << std::endl;
        }
        cudaDeviceProp props;
        int dev = 0;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&props, dev);
        std::cout<<"Using GPU - "<<props.name<<" with CC "<<props.major<<"."<<props.minor<<std::endl;
    }

    nvjpeg2kDeviceAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    nvjpeg2kPinnedAllocator_t pinned_allocator = {&host_malloc, &host_free};

    CHECK_NVJPEG2K(nvjpeg2kCreate(NVJPEG2K_BACKEND_DEFAULT, &dev_allocator,
                                  &pinned_allocator, &params.nvjpeg2k_handle));

    for(int p = 0; p < PIPELINE_STAGES; p++)
    {
        CHECK_NVJPEG2K(
        nvjpeg2kDecodeStateCreate(params.nvjpeg2k_handle, &params.nvjpeg2k_decode_states[p]));
    }
    
    params.jpeg2k_streams.resize(params.batch_size);

    for(auto& stream : params.jpeg2k_streams)
    {
        CHECK_NVJPEG2K(nvjpeg2kStreamCreate(&stream));
    }

    // read source images
    FileNames image_names;
    if (readInput(params.input_dir, image_names))
    {
        return EXIT_MISSING_INPUT;
    }
    if (image_names.empty())
    {
        std::cerr << "No images found in: " << params.input_dir << std::endl;
        return EXIT_NO_IMAGES;
    }

    if (params.total_images == -1)
    {
        params.total_images = image_names.size();
    }
    else if (params.total_images % params.batch_size)
    {
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
    int result = process_images(image_names, params, total);
    if (result)
        return result;
    std::cout << "Total decoding time: " << total << std::endl;
    std::cout << "Avg decoding time per image: " << total / params.total_images
              << std::endl;
    std::cout << "Avg decode speed  (in images per sec): " << params.total_images / total
              << std::endl;
    std::cout << "Avg decoding time per batch: "
              << total / ((params.total_images + params.batch_size - 1) /
                          params.batch_size)
              << std::endl;

    for(auto& stream : params.jpeg2k_streams)
    {
        CHECK_NVJPEG2K(nvjpeg2kStreamDestroy(stream));
    }
    for(int i =0; i < PIPELINE_STAGES; i++)
    {
        CHECK_NVJPEG2K(nvjpeg2kDecodeStateDestroy(params.nvjpeg2k_decode_states[i]));
    }
    CHECK_NVJPEG2K(nvjpeg2kDestroy(params.nvjpeg2k_handle));

    return EXIT_SUCCESS;
}
