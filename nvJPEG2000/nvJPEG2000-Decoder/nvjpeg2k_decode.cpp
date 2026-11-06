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


#include "nvjpeg2k_decode.h"

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

int free_output_buffers(nvjpeg2kImage_t& output_image)
{
    for(uint32_t c = 0; c < output_image.num_components;c++)
    {
         CHECK_CUDA(cudaFree(output_image.pixel_data[c]));
    }
    return EXIT_SUCCESS;
}

int allocate_output_buffers(nvjpeg2kImage_t& output_image, nvjpeg2kImageInfo_t& image_info, std::vector<nvjpeg2kImageComponentInfo_t> image_comp_info,
    int bytes_per_element, int rgb_output)
{
    output_image.num_components = image_info.num_components;
    if(rgb_output)
    {
        // for RGB output all component outputs dimensions are equal
        for(uint32_t c = 0; c < image_info.num_components;c++)
        {
            CHECK_CUDA(cudaMallocPitch(&output_image.pixel_data[c], &output_image.pitch_in_bytes[c], 
                image_info.image_width * bytes_per_element, image_info.image_height));
        }
    }
    else
    {
        for(uint32_t c = 0; c < image_info.num_components;c++)
        {
            CHECK_CUDA(cudaMallocPitch(&output_image.pixel_data[c], &output_image.pitch_in_bytes[c], 
                image_comp_info[c].component_width * bytes_per_element, image_comp_info[c].component_height));
        }
    }
    return EXIT_SUCCESS;
}

int decode_images(FileNames &current_names, const FileData &img_data, const std::vector<size_t> &img_len,
                  decode_params_t &params, double &time)
{
    CHECK_CUDA(cudaStreamSynchronize(params.stream));
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    float loopTime = 0;

    CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));
    nvjpeg2kDecodeParams_t decode_params;
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsCreate(&decode_params));

#if (NVJPEG2K_VER_MAJOR == 0 && NVJPEG2K_VER_MINOR >= 3) 
    // 420 and 422 subsampling are enabled in nvJPEG2k v 0.3.0
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetRGBOutput(decode_params, params.rgb_output));
#endif
    
    int bytes_per_element = 1;
    nvjpeg2kImage_t output_image;
    nvjpeg2kImageInfo_t image_info;
    std::vector<nvjpeg2kImageComponentInfo_t> image_comp_info;
    std::vector<unsigned short *> decode_output_u16;
    std::vector<short *> decode_output_s16;
    std::vector<unsigned char *> decode_output_u8;
#if NVJPEG2K_HAS_NLT3_DECODE
    std::vector<__half *> decode_output_h16;
    std::vector<float *> decode_output_f32;
#endif
    std::vector<size_t> decode_output_pitch;
    int result = EXIT_SUCCESS;
    for( int i =0; i < params.batch_size; i++)
    {
        auto io_start = perfclock::now();
        CHECK_NVJPEG2K(nvjpeg2kStreamParse(params.nvjpeg2k_handle, (unsigned char*)img_data[i].data(), img_len[i],
             0, 0, params.jpeg2k_stream));
        auto io_end = perfclock::now();
        double parse_time = std::chrono::duration_cast<std::chrono::seconds>(io_end-io_start).count();
        
        CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(params.jpeg2k_stream, &image_info));

        image_comp_info.resize(image_info.num_components);
        for (uint32_t c = 0; c < image_info.num_components; c++)
        {
            CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(params.jpeg2k_stream, &image_comp_info[c], c));
            //std::cout << "Component #" << c << " size: " << image_comp_info[c].component_width << " x "
            //<< image_comp_info[c].component_height << std::endl;
        }

        bool is_nlt3 = false;
        uint8_t nlt_bit_depth = image_comp_info[0].precision;
#if NVJPEG2K_HAS_NLT3_DECODE
        nvjpeg2kNLTInfo_t nlt_info;
        if (nvjpeg2kStreamGetComponentNLTInfo(params.jpeg2k_stream, 0, &nlt_info) == NVJPEG2K_STATUS_SUCCESS)
        {
            is_nlt3 = (nlt_info.type == NVJPEG2K_NLT_TYPE3);
            nlt_bit_depth = nlt_info.bit_depth;
        }
#endif

        decode_output_pitch.resize(image_info.num_components);
        output_image.pitch_in_bytes = decode_output_pitch.data();
#if NVJPEG2K_HAS_NLT3_DECODE
        if (is_nlt3 && nlt_bit_depth == 16)
        {
            decode_output_h16.resize(image_info.num_components);
            output_image.pixel_data = (void **)decode_output_h16.data();
            output_image.pixel_type = NVJPEG2K_FP16;
            bytes_per_element = 2;
        }
        else if (is_nlt3 && nlt_bit_depth > 16 && nlt_bit_depth <= 32)
        {
            decode_output_f32.resize(image_info.num_components);
            output_image.pixel_data = (void **)decode_output_f32.data();
            output_image.pixel_type = NVJPEG2K_FP32;
            bytes_per_element = 4;
        }
        else if (is_nlt3)
        {
            std::cout << "NLT bit depth " << static_cast<int>(nlt_bit_depth) << " not supported" << std::endl;
            return EXIT_UNSUPPORTED;
        }
        else
#endif
        if (image_comp_info[0].precision > 8 && image_comp_info[0].precision <= 16)
        {
            output_image.pixel_type = image_comp_info[0].sgn ? NVJPEG2K_INT16 : NVJPEG2K_UINT16;
            if(image_comp_info[0].sgn)
            {
                decode_output_s16.resize(image_info.num_components);
                output_image.pixel_data = (void **)decode_output_s16.data();
            }
            else
            {
                decode_output_u16.resize(image_info.num_components);
                output_image.pixel_data = (void **)decode_output_u16.data();
            }
            bytes_per_element = 2;
        }
        else if (image_comp_info[0].precision <= 8)
        {
            decode_output_u8.resize(image_info.num_components);
            output_image.pixel_data = (void **)decode_output_u8.data();
            output_image.pixel_type = NVJPEG2K_UINT8;
            bytes_per_element = 1;
        }
        else
        {
            std::cout << "Precision value " << static_cast<int>(image_comp_info[0].precision) << " not supported" << std::endl;
            return EXIT_UNSUPPORTED;
        }

        if(allocate_output_buffers(output_image, image_info, image_comp_info, bytes_per_element, params.rgb_output))
        {
            return EXIT_FAILURE;
        }
        CHECK_CUDA(cudaEventRecord(startEvent, params.stream));

        CHECK_NVJPEG2K(nvjpeg2kDecodeImage(params.nvjpeg2k_handle, params.nvjpeg2k_decode_state,
            params.jpeg2k_stream, decode_params, &output_image, params.stream));
        CHECK_CUDA(cudaEventRecord(stopEvent, params.stream));

        CHECK_CUDA(cudaEventSynchronize(stopEvent));
        CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
        time += static_cast<double>(loopTime/1000.0); // loopTime is in milliseconds
        time += parse_time;
        if (params.write_decoded)
        {
            bool skip_write = false;
            if(image_info.num_components == 3)
            {
                //check if the image is either 420 or 422
                if((float)image_comp_info[0].component_width/(float)image_comp_info[1].component_width > 1.0 && 
                    params.rgb_output == 0)
                {
                    std::cout<<"Unable to write 420/422 decode output to file. Use -rgb_output flag"<<std::endl;
                    skip_write = true;
                }
            }
            if(!skip_write)
            {
                result = write_image(params.output_dir, current_names[i], output_image, image_info.image_width,
                    image_info.image_height, image_info.num_components, image_comp_info[0].precision,
                    image_comp_info[0].sgn, params.verbose, is_nlt3);
            }
        }

        if(free_output_buffers(output_image))
        {
            return EXIT_FAILURE;
        }
        if(result)
        {
            break;
        }
    }

    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsDestroy(decode_params));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    return result;
}

int process_images(FileNames &image_names, decode_params_t &params,
                   double &total)
{
    // vector for storing raw files and file lengths
    FileData file_data(params.batch_size);
    std::vector<size_t> file_len(params.batch_size);
    FileNames current_names(params.batch_size);
    std::vector<int> widths(params.batch_size);
    std::vector<int> heights(params.batch_size);
    // we wrap over image files to process total_images of files
    FileNames::iterator file_iter = image_names.begin();

    // stream for decoding
    CHECK_CUDA(cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));

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

        double time = 0;
        int status = decode_images(current_names, file_data, file_len, params, time);
        if (status)
        {
            // decode_images returns EXIT_DECODE_FAILURE when decoding or writing fails, EXIT_FAILURE for CUDA errors
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
            test_time += time;
        }
    }
    total = test_time;

    CHECK_CUDA(cudaStreamDestroy(params.stream));

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
                     "[-w warmup_iterations] [-o output_dir] [-rgb_output] [-v]"
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
                     "\t\t\t\tand NLT type 3 images as .pfm"
                  << std::endl;
        std::cout << "\t-rgb_output\t:\tConvert 4:2:0 and 4:2:2 images to RGB, which is required to write them"
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

    params.rgb_output = 0;
    if ((pidx = findParamIndex(argv, argc, "-rgb_output")) != -1)
    {
        params.rgb_output = 1;
    }

    if(params.verbose)
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

    CHECK_NVJPEG2K(
        nvjpeg2kDecodeStateCreate(params.nvjpeg2k_handle, &params.nvjpeg2k_decode_state));

    CHECK_NVJPEG2K(nvjpeg2kStreamCreate(&params.jpeg2k_stream));

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
    std::cout << "Avg images per sec: " << params.total_images / total
              << std::endl;
    std::cout << "Avg decoding time per batch: "
              << total / ((params.total_images + params.batch_size - 1) /
                          params.batch_size)
              << std::endl;

    CHECK_NVJPEG2K(nvjpeg2kStreamDestroy(params.jpeg2k_stream));
    CHECK_NVJPEG2K(nvjpeg2kDecodeStateDestroy(params.nvjpeg2k_decode_state));
    CHECK_NVJPEG2K(nvjpeg2kDestroy(params.nvjpeg2k_handle));

    return EXIT_SUCCESS;
}
