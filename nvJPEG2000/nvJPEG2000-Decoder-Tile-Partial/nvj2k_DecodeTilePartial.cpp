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

#include "nvj2k_DecodeTilePartial.h"

template<typename T>
int write_image(std::string output_path, std::string filename, const T &imgdesc, int width, int height,
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
            return EXIT_FAILURE;
        }
        for (uint32_t c = 0; c < image_info.num_components; c++) 
        {
            CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(params.jpeg2k_streams[i], &image_comp_info[c], c));
        }

        ibuf[i].num_comps = image_info.num_components;
        // realloc output buffer if required
        for (uint32_t c = 0; c < image_info.num_components; c++) 
        {
            uint32_t bytes_per_element = (image_comp_info[0].precision+7)/8;
            if( image_comp_info[0].precision <= 8)
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
                return EXIT_FAILURE;
            }
            // for JPEG 2000 bitstreams with 420/422 subsampling, this sample enables RGB output
            // we are allocating assuming that all component dimensions are the same
            uint32_t aw = bytes_per_element * image_info.image_width;
            uint32_t ah = image_info.image_height;
            uint32_t sz = aw * ah;
            ibuf[i].pitch_in_bytes[c] = aw;
            if (sz > ibuf[i].comp_sz[c]) 
            {
                if (ibuf[i].pixel_data[c]) 
                {
                    CHECK_CUDA(cudaFree(ibuf[i].pixel_data[c]));
                }
                CHECK_CUDA(cudaMalloc((void**)&ibuf[i].pixel_data[c], sz));
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
            if(buf.pixel_data[c])
            {
                CHECK_CUDA(cudaFree(buf.pixel_data[c]));
            }
            buf.pixel_data[c] = nullptr;
            buf.comp_sz[c] = 0;
            buf.pitch_in_bytes[c] = 0;
        }
        buf.num_comps = 0;
    }
    return EXIT_SUCCESS;
}

struct partial_decode_info
{
    unsigned int win_tilex0;
    unsigned int win_tilex1;
    unsigned int win_tiley0;
    unsigned int win_tiley1;
    unsigned int tile_id;
};

void determine_tiles_to_decode(const nvjpeg2kImageInfo_t& image_info, decode_params_t &params,
    std::vector<partial_decode_info>& tile_window_data)
{
    uint32_t tile_id = 0;
    for(uint32_t tile_y0 = 0; tile_y0 < image_info.image_height; tile_y0 += image_info.tile_height)
    {
        for(uint32_t tile_x0 = 0; tile_x0 < image_info.image_width; tile_x0 += image_info.tile_width)
        {
            // include min and max functions in braces for windows builds issues
            uint32_t tile_y1 = (std::min)(tile_y0 + image_info.tile_height, image_info.image_height);
            uint32_t tile_x1 = (std::min)(tile_x0 + image_info.tile_width, image_info.image_width);

            if( params.win_x0 < tile_x1 && params.win_x1 > tile_x0 &&
                params.win_y0 < tile_y1 && params.win_y1 > tile_y0)
            {
                partial_decode_info decode_data;
                decode_data.tile_id = tile_id;
                decode_data.win_tilex0 = (std::max)(tile_x0, params.win_x0);
                decode_data.win_tilex1 = (std::min)(tile_x1, params.win_x1);
                decode_data.win_tiley0 = (std::max)(tile_y0, params.win_y0);
                decode_data.win_tiley1 = (std::min)(tile_y1, params.win_y1);
                tile_window_data.push_back(decode_data);
                
            }
            tile_id++;
            
        }
    }
}

int decode_images_partial(FileNames &current_names, std::vector<nvjpeg2kImageSample> &out,
                  decode_params_t &params, double &time)
{
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    nvjpeg2kDecodeParams_t decode_params;
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsCreate(&decode_params));
    float loopTime = 0;

    cudaEvent_t pipeline_events[PIPELINE_STAGES];

    for (int p = 0; p < PIPELINE_STAGES; p++)
    {
        CHECK_CUDA(cudaEventCreate(&pipeline_events[p]));
        CHECK_CUDA(cudaEventRecord(pipeline_events[p], params.stream[p]));
    }
    CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));

    // set RGB  output for the entire batch
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetRGBOutput(decode_params, 1));

    CHECK_CUDA(cudaEventRecord(startEvent, params.stream[0]));
    int buffer_index = 0;
    for(int batch_id = 0; batch_id < params.batch_size; batch_id++)
    {
        nvjpeg2kImageInfo_t image_info;
        CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(params.jpeg2k_streams[batch_id], &image_info));

        std::vector<partial_decode_info> tile_window_data;
        determine_tiles_to_decode(image_info, params, tile_window_data);

        uint32_t out_width = 0;
        uint32_t out_height = 0;
        for( uint32_t i = 0; i < tile_window_data.size(); i++)
        {
            // make sure that the previous stage are done
            CHECK_CUDA(cudaEventSynchronize(pipeline_events[buffer_index]));
            nvjpeg2kImageSample_t tile_decode_out;
            
            auto& decode_data = tile_window_data[i];
            tile_decode_out.num_comps = out[batch_id].num_comps;
            tile_decode_out.pixel_type = out[batch_id].pixel_type;
            uint32_t bytes_per_comp = tile_decode_out.pixel_type == NVJPEG2K_UINT8 ? 1 : 2;
            for(uint32_t c = 0; c < out[batch_id].num_comps; c++)
            {
                size_t pitch_in_bytes = out[batch_id].pitch_in_bytes[c];
                tile_decode_out.pixel_data[c] = (uint8_t*)(out[batch_id].pixel_data[c]) + out_height * pitch_in_bytes + (out_width * bytes_per_comp);
                tile_decode_out.pitch_in_bytes[c] = out[batch_id].pitch_in_bytes[c];
            }
            out_width += decode_data.win_tilex1 - decode_data.win_tilex0;
                
            if( out_width == params.win_x1 - params.win_x0)
            {
                out_width = 0;
                out_height += decode_data.win_tiley1 - decode_data.win_tiley0;
            }

            nvjpeg2kImage_t nvjpeg2k_out;
            nvjpeg2k_out.num_components = tile_decode_out.num_comps;
            nvjpeg2k_out.pixel_data = (void**)tile_decode_out.pixel_data;
            nvjpeg2k_out.pitch_in_bytes = tile_decode_out.pitch_in_bytes;
            nvjpeg2k_out.pixel_type = tile_decode_out.pixel_type;

            CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetDecodeArea(decode_params, decode_data.win_tilex0, decode_data.win_tilex1,
                decode_data.win_tiley0, decode_data.win_tiley1));
            CHECK_NVJPEG2K(nvjpeg2kDecodeTile(params.nvjpeg2k_handle, params.nvjpeg2k_decode_states[buffer_index],
                params.jpeg2k_streams[batch_id], decode_params, decode_data.tile_id, 0,
                &nvjpeg2k_out, params.stream[buffer_index]));
            
            CHECK_CUDA(cudaEventRecord(pipeline_events[buffer_index], params.stream[buffer_index]));

            buffer_index++;
            buffer_index = buffer_index%PIPELINE_STAGES;

        }
    }
    for (int p = 0; p < PIPELINE_STAGES; p++)
    {
        CHECK_CUDA(cudaEventSynchronize(pipeline_events[p]));
    }

    CHECK_CUDA(cudaEventRecord(stopEvent, params.stream[0]));
    
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
    time += static_cast<double>(loopTime/1000.0); // loopTime is in milliseconds
    
    if (params.write_decoded)
    {
        for( int i = 0; i < params.batch_size; i++)
        {
            nvjpeg2kImageInfo_t image_info;
            nvjpeg2kImageComponentInfo_t comp_info;
            CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(params.jpeg2k_streams[i], &image_info));
            // assume all components have the same precision
            CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(params.jpeg2k_streams[i], &comp_info, 0));
            write_image(params.output_dir, current_names[i], out[i], params.win_x1 - params.win_x0, 
                params.win_y1 - params.win_y0, image_info.num_components, comp_info.precision,
                comp_info.sgn, params.verbose);
        }
    }

    for(int p = 0; p < PIPELINE_STAGES; p++)
    {
        CHECK_CUDA(cudaEventDestroy(pipeline_events[p]));
    }
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsDestroy(decode_params));
    
    return EXIT_SUCCESS;
}


int decode_images(FileNames &current_names, std::vector<nvjpeg2kImageSample> &out,
                  decode_params_t &params, double &time)
{
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    float loopTime = 0;

    cudaEvent_t pipeline_events[PIPELINE_STAGES];

    for (int p = 0; p < PIPELINE_STAGES; p++)
    {
        CHECK_CUDA(cudaEventCreate(&pipeline_events[p]));
        CHECK_CUDA(cudaEventRecord(pipeline_events[p], params.stream[p]));
    }
    CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));

    nvjpeg2kDecodeParams_t decode_params;
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsCreate(&decode_params));

    // set RGB  output for the entire batch
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetRGBOutput(decode_params, 1));

    CHECK_CUDA(cudaEventRecord(startEvent, params.stream[0]));
    int buffer_index = 0;
    for(int batch_id = 0; batch_id < params.batch_size; batch_id++)
    {
        nvjpeg2kImageInfo_t image_info;
        CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(params.jpeg2k_streams[batch_id], &image_info));

        uint32_t tile_id = 0;
        for(uint32_t tile_y0 = 0; tile_y0 < image_info.image_height; tile_y0 += image_info.tile_height)
        {
            for(uint32_t tile_x0 = 0; tile_x0 < image_info.image_width; tile_x0 += image_info.tile_width)
            {
                // make sure that the previous stage are done
                CHECK_CUDA(cudaEventSynchronize(pipeline_events[buffer_index]));
                nvjpeg2kImageSample tile_decode_out;
                tile_decode_out.num_comps = out[batch_id].num_comps;
                tile_decode_out.pixel_type = out[batch_id].pixel_type;
                uint32_t bytes_per_comp = tile_decode_out.pixel_type == NVJPEG2K_UINT8 ? 1 : 2;
                for(uint32_t c = 0; c < out[batch_id].num_comps; c++)
                {
                    size_t pitch_in_bytes = out[batch_id].pitch_in_bytes[c];
                    tile_decode_out.pixel_data[c] = (uint8_t*)(out[batch_id].pixel_data[c]) + tile_y0 * pitch_in_bytes + (tile_x0 * bytes_per_comp);
                    tile_decode_out.pitch_in_bytes[c] = out[batch_id].pitch_in_bytes[c];
                }
                // make sure that the previous stage are done before reusing
                CHECK_CUDA(cudaEventSynchronize(pipeline_events[buffer_index]));
            
                nvjpeg2kImage_t nvjpeg2k_out;
                nvjpeg2k_out.num_components = tile_decode_out.num_comps;
                nvjpeg2k_out.pixel_data = (void**)tile_decode_out.pixel_data;
                nvjpeg2k_out.pitch_in_bytes = tile_decode_out.pitch_in_bytes;
                nvjpeg2k_out.pixel_type = tile_decode_out.pixel_type;

                CHECK_NVJPEG2K(nvjpeg2kDecodeTile(params.nvjpeg2k_handle, params.nvjpeg2k_decode_states[buffer_index],
                    params.jpeg2k_streams[batch_id], decode_params, tile_id, 0,
                    &nvjpeg2k_out, params.stream[buffer_index]));
                
                CHECK_CUDA(cudaEventRecord(pipeline_events[buffer_index], params.stream[buffer_index]));

                buffer_index++;
                buffer_index = buffer_index%PIPELINE_STAGES;
                tile_id++;

            }
        }
    }
    for (int p = 0; p < PIPELINE_STAGES; p++)
    {
        CHECK_CUDA(cudaEventSynchronize(pipeline_events[p]));
    }

    CHECK_CUDA(cudaEventRecord(stopEvent, params.stream[0]));
    
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
    time += static_cast<double>(loopTime/1000.0); // loopTime is in milliseconds
    
    if (params.write_decoded)
    {
        for( int i = 0; i < params.batch_size; i++)
        {
            nvjpeg2kImageInfo_t image_info;
            nvjpeg2kImageComponentInfo_t comp_info;
            CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(params.jpeg2k_streams[i], &image_info));
            // assume all components have the same precision
            CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(params.jpeg2k_streams[i], &comp_info, 0));
            
            write_image(params.output_dir, current_names[i], out[i], image_info.image_width, 
                image_info.image_height, image_info.num_components, comp_info.precision,
                comp_info.sgn, params.verbose);
        }
    }

    for(int p = 0; p < PIPELINE_STAGES; p++)
    {
        CHECK_CUDA(cudaEventDestroy(pipeline_events[p]));
    }
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsDestroy(decode_params));

    return EXIT_SUCCESS;
}

double process_images(FileNames &image_names, decode_params_t &params,
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
    while (total_processed < params.total_images)
    {
        if (read_next_batch(image_names, params.batch_size, file_iter, file_data,
                            file_len, current_names, params.verbose))
            return EXIT_FAILURE;
        double parsetime = 0;
        if (prepare_buffers(file_data, file_len, iout,
                        current_names, params, parsetime))
            return EXIT_FAILURE;

        double time = 0;
        int ret_val = 0;
        if (params.partial_decode)
        {
            ret_val = decode_images_partial(current_names, iout, params, time);
        }
        else
        {
            ret_val = decode_images(current_names, iout, params, time);
        }
        if(ret_val)
        {
            return EXIT_FAILURE;
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
        EXIT_FAILURE;

    return EXIT_SUCCESS;
}

int parseDecodeCoordinates(const char* argv, decode_params_t& params)
{
    std::istringstream decode_area(argv);
    std::string temp;
    int idx = 0;
    while(getline(decode_area, temp,','))
    {
        if( idx == 0)
        {
            params.win_x0 = std::stoi(temp);
        }
        else if (idx == 1)
        {
            params.win_y0 = std::stoi(temp);
        }
        else if( idx == 2)
        {
            params.win_x1 = std::stoi(temp);
        }
        else if (idx == 3)
        {
            params.win_y1 = std::stoi(temp);
        }
        else
        {
            std::cout<<"invalid decode area here"<<std::endl;
            return EXIT_FAILURE;
        }
        idx++;
    }
    if (params.win_x0 >= params.win_x1 || params.win_y0 >= params.win_y1)
    {
        std::cout<<"invalid decode area here"<<std::endl;
        return EXIT_FAILURE;
    }
    params.partial_decode = true;
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
                     "[-w warmup_iterations] [-o output_dir] [-da decode_area_of_interest] [-v verbose]\n";
        std::cout << "Parameters: " << std::endl;
        std::cout << "\timages_dir\t:\tPath to single image or directory of images"
                  << std::endl;
        std::cout << "\tbatch_size\t:\tDecode images from input by batches of "
                     "specified size"
                  << std::endl;
        std::cout << "\ttotal_images\t:\tDecode these many images, if there are "
                     "fewer images \n"
                  << "\t\t\t\tin the input than total images, decoder will loop "
                     "over the input"
                  << std::endl;
        std::cout << "\twarmup_iterations:\tRun these many batches first "
                     "without measuring performance"
                  << std::endl;
        std::cout << "\tdecode_area_of_interest: Image coordinates specifying an area "
                  << "to be decoded"
                  << std::endl;
        std::cout
            << "\toutput_dir\t:\tWrite decoded images in BMP/PGM format to this directory"
            << std::endl;
        std::cout
            << "\tverbose\t\t:\tLog verbose messages to console"
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

    params.win_x0 = 0;
    params.win_y0 = 0;
    params.win_x1 = 0;
    params.win_y1 = 0;
    params.partial_decode = false;
    if ((pidx = findParamIndex(argv, argc, "-da")) != -1)
    {
        if(parseDecodeCoordinates(argv[pidx + 1], params))
        {
            return EXIT_SUCCESS;
        }
    }

    if( params.verbose)
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
    readInput(params.input_dir, image_names);

    if (params.total_images == -1)
    {
        params.total_images = static_cast<int>(image_names.size());
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
