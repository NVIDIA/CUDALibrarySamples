/*
 * Copyright (c) 2020 - 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "nvjpeg2000DecodeSample.h"

int write_image(std::string output_path, std::string filename, const nvjpeg2kImage_t &imgdesc, int width, int height,
               uint32_t num_components, uint8_t precision, uint8_t sgn, bool verbose)
{
    // Get the file name, without extension.
    // This will be used to rename the output file.
    size_t position = filename.rfind(separator);
    std::string sFileName =
        (std::string::npos == position)
            ? filename
            : filename.substr(position + 1, filename.size());
    position = sFileName.rfind(".");
    sFileName = (std::string::npos == position) ? sFileName
                                                : sFileName.substr(0, position);

    int err = EXIT_SUCCESS;
    
    // For single component image output as PGM channel
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
        if(num_components == 4 && verbose)
        {
            std::cout<<"Discarding the alpha channel and writing the 4 component image as a .bmp file"<<std::endl;
        }
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
        std::cout << "num channels not supported"<<std::endl;
        return EXIT_FAILURE;
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
    std::vector<unsigned char *> decode_output_u8;
    std::vector<size_t> decode_output_pitch;
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

        decode_output_pitch.resize(image_info.num_components);
        output_image.pitch_in_bytes = decode_output_pitch.data();
        if (image_comp_info[0].precision > 8 && image_comp_info[0].precision <= 16)
        {
            decode_output_u16.resize(image_info.num_components);
            output_image.pixel_data = (void **)decode_output_u16.data();
            output_image.pixel_type = image_comp_info[0].sgn ? NVJPEG2K_INT16 : NVJPEG2K_UINT16;
            bytes_per_element = 2;
        }
        else if (image_comp_info[0].precision == 8)
        {
            decode_output_u8.resize(image_info.num_components);
            output_image.pixel_data = (void **)decode_output_u8.data();
            output_image.pixel_type = NVJPEG2K_UINT8;
            bytes_per_element = 1;
        }
        else
        {
            std::cout << "Precision value " << image_comp_info[0].precision << " not supported" << std::endl;
            return EXIT_FAILURE;
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
            if(image_info.num_components == 3)
            {
                //check if the image is either 420 or 422
                if((float)image_comp_info[0].component_width/(float)image_comp_info[1].component_width > 1.0 && 
                    params.rgb_output == 0)
                {
                    if(params.verbose)
                    {
                        std::cout<<"Unable to write 420/422 decode output to file. Use -rgb_output flag"<<std::endl;
                    }
                    continue;
                }
            }
            write_image(params.output_dir, current_names[i], output_image, image_info.image_width, 
                image_info.image_height, image_info.num_components, image_comp_info[0].precision, 
                image_comp_info[0].sgn, params.verbose);
        }

        if(free_output_buffers(output_image))
        {
            return EXIT_FAILURE;
        }
    }

    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsDestroy(decode_params));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    return EXIT_SUCCESS;
}

double process_images(FileNames &image_names, decode_params_t &params,
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
    while (total_processed < params.total_images)
    {
        if (read_next_batch(image_names, params.batch_size, file_iter, file_data,
                            file_len, current_names, params.verbose))
            return EXIT_FAILURE;

        double time = 0;
        if (decode_images(current_names, file_data, file_len, params, time))
            return EXIT_FAILURE;
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

    return EXIT_SUCCESS;
}

int main(int argc, const char *argv[])
{
    int pidx;

    if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
        (pidx = findParamIndex(argv, argc, "--help")) != -1)
    {
        std::cout << "Usage: " << argv[0]
                  << " -i images_dir [-b batch_size] [-t total_images] "
                     "[-w warmup_iterations] [-o output_dir] [-v verbose] [-rgb_output]"
                  << std::endl;
        std::cout << "Parameters: " << std::endl;
        std::cout << "\timages_dir\t:\tPath to single image or directory of images"
                  << std::endl;
        std::cout << "\tbatch_size\t:\tDecode images from input by batches of "
                     "specified size"
                  << std::endl;
        std::cout << "\ttotal_images\t:\tDecode these many images, if there are "
                     "fewer images "<<std::endl
                  << "\t\t\t\tin the input than total images, decoder will loop "
                     "over the input"
                  << std::endl;
        std::cout << "\twarmup_iterations:\tRun these many batches first "
                     "without measuring performance"
                  << std::endl;
        std::cout << "\toutput_dir\t:\tWrite decoded images in BMP/PGM format to this directory"
                  << std::endl;   
        std::cout << "\trgb_output\t:\tUse this flag when decoding images with 420/422 subsampling"<<std::endl
                  << "\t\t\t\tsuch that the nvJPEG2000 library generates RGB output"
                  << std::endl;
        std::cout << "\tverbose\t\t:\tLog verbose messages to console"
                  << std::endl;
        return EXIT_SUCCESS;
    }

    decode_params_t params;

    params.input_dir = "./";
    if ((pidx = findParamIndex(argv, argc, "-i")) != -1)
    {
        params.input_dir = argv[pidx + 1];
    }
    else
    {
        // Search in default paths for input images.
        int found = getInputDir(params.input_dir, argv[0]);
        if (!found)
        {
            std::cout << "Please specify input directory with encoded images" << std::endl;
            return EXIT_FAILURE;
        }
    }

    params.batch_size = 1;
    if ((pidx = findParamIndex(argv, argc, "-b")) != -1)
    {
        params.batch_size = std::atoi(argv[pidx + 1]);
    }

    params.total_images = -1;
    if ((pidx = findParamIndex(argv, argc, "-t")) != -1)
    {
        params.total_images = std::atoi(argv[pidx + 1]);
    }

    params.warmup = 0;
    if ((pidx = findParamIndex(argv, argc, "-w")) != -1)
    {
        params.warmup = std::atoi(argv[pidx + 1]);
    }

    params.write_decoded = false;
    if ((pidx = findParamIndex(argv, argc, "-o")) != -1)
    {
        params.output_dir = argv[pidx + 1];

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
            std::cout << "3/4 channel images are written out as bmp files and 1 channels images are written out as .pgm files"
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
    readInput(params.input_dir, image_names);

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
    if (process_images(image_names, params, total))
        return EXIT_FAILURE;
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
