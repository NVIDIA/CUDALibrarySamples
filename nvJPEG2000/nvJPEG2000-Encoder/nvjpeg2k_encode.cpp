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

#include "nvjpeg2k_encode.h"
#include <cmath>
#include <sstream>

struct bmpHeader
{
    char signature[2];
    uint32_t file_size;
    uint32_t reserved;
    uint32_t dataOffset;
};
struct bmpInfoHeader
{
    uint32_t Size;
    uint32_t Width;
    uint32_t Height;
    uint16_t Planes;
    uint16_t BitsPerPixel;
    uint32_t Compression;
    uint32_t ImageSize;
    uint32_t XPixelsPerM;
    uint32_t YPixelsPerM;
    uint32_t ColorsUsed;
    uint32_t ImportantColors;
};

int read_bmp_info_header(std::ifstream& file_input, bmpInfoHeader& info_header)
{
    file_input.read(reinterpret_cast<char*>(&info_header.Size), 4);
    if( info_header.Size != 40 )
    {
        std::cout<<"bmp file not supported"<<std::endl;
        return EXIT_FAILURE;
    }
    file_input.read(reinterpret_cast<char*>(&info_header.Width), 4);
    file_input.read(reinterpret_cast<char*>(&info_header.Height), 4);
    file_input.read(reinterpret_cast<char*>(&info_header.Planes), 2);
    file_input.read(reinterpret_cast<char*>(&info_header.BitsPerPixel), 2);
    file_input.read(reinterpret_cast<char*>(&info_header.Compression), 4);

    if( info_header.Compression != 0 )
    {
        std::cout<<"only raw bmp files are supported"<<std::endl;
        return EXIT_FAILURE;
    }
    file_input.read(reinterpret_cast<char*>(&info_header.ImageSize), 4);
    file_input.read(reinterpret_cast<char*>(&info_header.XPixelsPerM), 4);
    file_input.read(reinterpret_cast<char*>(&info_header.YPixelsPerM), 4);
    file_input.read(reinterpret_cast<char*>(&info_header.ColorsUsed), 4);
    file_input.read(reinterpret_cast<char*>(&info_header.ImportantColors), 4);

    return EXIT_SUCCESS;
}

int read_bmp(std::ifstream& file_input, Image& image)
{
    bmpHeader bmp_header;
    bmpInfoHeader bmp_info_header;
    nvjpeg2kImageInfo_t nvjpeg2k_info;
    std::vector<nvjpeg2kImageComponentInfo_t> nvjpeg2k_comp_info;
    
    file_input.read(bmp_header.signature, 2);
    if(bmp_header.signature[0] != 'B'|| bmp_header.signature[1]!= 'M')
    {
        std::cout<<"not a bmp file"<<std::endl;
        return EXIT_FAILURE;
    }
    file_input.read(reinterpret_cast<char*>(&bmp_header.file_size), 4);
    file_input.read(reinterpret_cast<char*>(&bmp_header.reserved), 4);
    file_input.read(reinterpret_cast<char*>(&bmp_header.dataOffset), 4);
    if(read_bmp_info_header(file_input, bmp_info_header))
    {
        return EXIT_FAILURE;
    }
    // bmp file data is 32 bit aligned
    size_t stride = (bmp_info_header.Width * bmp_info_header.BitsPerPixel + 31)/32  * 4;

    nvjpeg2kColorSpace_t color_space = NVJPEG2K_COLORSPACE_SRGB;
    nvjpeg2k_info.num_components = 3;
    nvjpeg2k_info.image_width = bmp_info_header.Width;
    nvjpeg2k_info.image_height = bmp_info_header.Height;
    nvjpeg2k_comp_info.resize(nvjpeg2k_info.num_components);
    
    for(auto& comp : nvjpeg2k_comp_info)
    {
        comp.component_width  = nvjpeg2k_info.image_width;
        comp.component_height = nvjpeg2k_info.image_height;
        comp.precision        = 8;
        comp.sgn              = 0;
    } 
    
    image.initialize(nvjpeg2k_info, nvjpeg2k_comp_info.data(), color_space);
    std::vector<uint8_t> raw_bmp;
    raw_bmp.resize(stride * bmp_info_header.Height);
    file_input.read(reinterpret_cast<char*>(raw_bmp.data()), stride * bmp_info_header.Height);
    
    auto& img = image.getImageHost();
    unsigned char*  r = reinterpret_cast<unsigned char*>(img.pixel_data[0]);
    unsigned char*  g = reinterpret_cast<unsigned char*>(img.pixel_data[1]);
    unsigned char*  b = reinterpret_cast<unsigned char*>(img.pixel_data[2]);
    
    uint32_t y_bmp = nvjpeg2k_info.image_height;
    for (uint32_t y = 0; y < nvjpeg2k_info.image_height; y++) 
    {
        y_bmp--; 
        for (uint32_t x = 0; x < nvjpeg2k_info.image_width; x++) 
        {
            r[y * img.pitch_in_bytes[0] + x] = raw_bmp[y_bmp * stride + (3 * x + 2)];
            g[y * img.pitch_in_bytes[1] + x] = raw_bmp[y_bmp * stride + (3 * x + 1)];
            b[y * img.pitch_in_bytes[2] + x] = raw_bmp[y_bmp * stride + (3 * x + 0)];
        }
    }
    return EXIT_SUCCESS;
}

int read_yuv(std::ifstream& file_input, Image& image, encode_params_t& params)
{
    image.initialize(params.img_info, params.comp_info, NVJPEG2K_COLORSPACE_SYCC);

    auto& nvjpeg2k_img = image.getImageHost();
    size_t total_bytes = 0;
    for(uint32_t c = 0; c < nvjpeg2k_img.num_components;c++)
    {
        char* buff = (char*)nvjpeg2k_img.pixel_data[c];
        for(uint32_t y = 0; y < params.comp_info[c].component_height; y++)
        {
            uint32_t bytes_to_read =  params.comp_info[c].component_width * ((params.comp_info[c].precision +7)/8);
            total_bytes += bytes_to_read;
            if(nvjpeg2k_img.pitch_in_bytes[c] != bytes_to_read)
            {
                std::cout<<"\n fail";
            }
            if( !file_input.read(buff, bytes_to_read))
            {
                std::cout<<"failed to read image from file"<<std::endl;
                return EXIT_FAILURE;
            }
            buff += nvjpeg2k_img.pitch_in_bytes[c];
        }
    }
    char extra;
    if(file_input.read(&extra,1))
    {
        std::cout<<"WARNING - failed to read the entire file"<<std::endl;
    }
    
    return EXIT_SUCCESS;
}

int read_pnm(std::ifstream&  file_input, Image& image)
{
    nvjpeg2kColorSpace_t color_space;
    std::string line;
    std::getline(file_input, line, '\n');
    
    nvjpeg2kImageInfo_t info;
    if( line == "P5")
    {
        info.num_components = 1;
        color_space = NVJPEG2K_COLORSPACE_GRAY;
    }
    else if( line == "P6")
    {
        info.num_components = 3;
        color_space = NVJPEG2K_COLORSPACE_SRGB;
    }
    else 
    {
        std::cout<<"pgm/ppm format not supported"<<std::endl;
        return EXIT_FAILURE;
    }
    
    std::vector<nvjpeg2kImageComponentInfo_t> comp_info(info.num_components);
    int idx = 0;
    uint32_t max_val = 0;
    while(std::getline(file_input, line, '\n'))
    {
        if(line[0] == '#')
        {
            continue;
        }
        else if(idx == 0)
        {
            size_t pos = line.find(" ");
            if (pos != std::string::npos)
            {
                info.image_width =  stoi(line.substr(0, pos));
                info.image_height = stoi(line.substr(pos, line.length()));
            }
            else
            {
                std::cout<<"\nInvalid pnm file"<<std::endl;
                return EXIT_FAILURE;
            }
            idx++;
        }
        else if(idx == 1)
        {
            max_val = stoi(line);
            if(max_val > ((1<<16) - 1))
            {
                std::cout<<"\nInvalid pgm prec value in pgm file"<<std::endl;
                return EXIT_FAILURE;
            }
            break;
        }
        else
        {
            std::cout<<"failed to read pgm file"<<std::endl;
            return EXIT_FAILURE;
        }
    }
    for(auto& comp : comp_info)
    {
        comp.component_width  = info.image_width;
        comp.component_height = info.image_height;
        comp.precision =  static_cast<uint8_t>(log2(max_val + 1));
        comp.sgn = 0;
    }
    
    image.initialize(info, comp_info.data(), color_space);
    auto& img = image.getImageHost();
    {
        if(img.pixel_type == NVJPEG2K_UINT16)
        {
            std::vector<uint16_t> temp_buffer(info.image_width * info.num_components);
            for(uint32_t y = 0; y < info.image_height;y++)
            {
                file_input.read((char*)temp_buffer.data(), info.image_width * info.num_components * sizeof(uint16_t));
                for(uint32_t x = 0; x < info.image_width; x++)
                {
                    for(uint32_t c = 0; c <info.num_components; c++)
                    { 
                        uint16_t* pix = reinterpret_cast<uint16_t*>(img.pixel_data[c]);
                        uint16_t value = temp_buffer[info.num_components * x + c];
                        pix[y * (img.pitch_in_bytes[0]/sizeof(uint16_t))+ x] = (value&0xff)<<8|value>>8;
                    }
                }
            }
        }
        else
        {
            std::vector<uint8_t> temp_buffer(info.image_width * info.num_components);
            for(uint32_t y = 0; y < info.image_height;y++)
            {
                file_input.read((char*)temp_buffer.data(), info.image_width * info.num_components);
                for(uint32_t x = 0; x < info.image_width; x++)
                {
                    for(uint32_t c = 0; c <info.num_components; c++)
                    {
                        uint8_t* pix = reinterpret_cast<uint8_t*>(img.pixel_data[c]);
                        pix[y * img.pitch_in_bytes[0]+ x] = temp_buffer[info.num_components*x + c];
                    }
                }
            }
        }
    }   
    char extra;
    if(file_input.read(&extra,1))
    {
        std::cout<<"WARNING - failed to read the entire file"<<std::endl;
    }
    return EXIT_SUCCESS;
}


int read_next_batch(FileNames &image_names, int batch_size,
    FileNames::iterator &cur_iter, Image* input_images, FileNames &current_names,
    encode_params_t& params)
{
    int counter = 0;

    while (counter < batch_size)
    {
        if (cur_iter == image_names.end())
        {
            // std::cerr << "Image list is too short to fill the batch, adding files "
            // from the beginning of the image list"
            // << std::endl;
            cur_iter = image_names.begin();
        }

        if (image_names.size() == 0)
        {
            std::cerr << "No valid images left in the input list, exit" << std::endl;
            return EXIT_FAILURE;
        }
        std::string ext = cur_iter->substr(cur_iter->find_last_of(".") + 1);
        std::ifstream image_file(cur_iter->c_str(), std::ios::in | std::ios::binary);

        if (!(image_file.is_open()))
        {
            std::cerr << "Cannot open image: " << *cur_iter
                      << ", removing it from image list" << std::endl;
            image_names.erase(cur_iter);
            continue;
        }

        if(ext == "pgm" || ext == "ppm")
        {
            if(read_pnm(image_file, input_images[counter]))
            {
                return EXIT_FAILURE;
            }
        }
        else if(ext == "bmp")
        {
            if(read_bmp(image_file, input_images[counter]))
            {
                return EXIT_FAILURE;
            }
        }
        else if( ext == "yuv")
        {
            if(!params.img_fmt_init)
            {
                std::cout << "For yuv files, pease provide -img_fmt width,height,num_comp,precision,chroma subsampling"<<std::endl;
                return EXIT_FAILURE;
            }
            read_yuv(image_file, input_images[counter], params);
        }
        else
        {
            std::cerr << "Cannot encode image: " << *cur_iter
                      << ", removing it from image list" << std::endl;
            image_file.close();
            image_names.erase(cur_iter);
            continue;
        }
        current_names[counter] = *cur_iter;

        input_images[counter].copyToDevice();

        counter++;
        cur_iter++;
        image_file.close();
    }
    return EXIT_SUCCESS;
}

void populate_encoderconfig(nvjpeg2kEncodeConfig_t& enc_config, Image& input_image, encode_params_t &params)
{
    memset(&enc_config, 0, sizeof(enc_config));
    enc_config.stream_type =  NVJPEG2K_STREAM_JP2;
    enc_config.color_space = input_image.getColorSpace();
    enc_config.image_width =  input_image.getnvjpeg2kImageInfo().image_width;
    enc_config.image_height = input_image.getnvjpeg2kImageInfo().image_height;
    enc_config.num_components = input_image.getnvjpeg2kImageInfo().num_components;
    enc_config.image_comp_info = input_image.getnvjpeg2kCompInfo();
    enc_config.code_block_w = (uint32_t)params.cblk_w;
    enc_config.code_block_h = (uint32_t)params.cblk_h;
    enc_config.irreversible = (uint32_t)params.irreversible;
    enc_config.mct_mode = enc_config.color_space == NVJPEG2K_COLORSPACE_SRGB ? 1 : 0;
    enc_config.prog_order = NVJPEG2K_LRCP;
    enc_config.num_resolutions = 6;
}

int encode_images(Image* input_images, encode_params_t &params, BitStreamData &bitstreams, double &time)
{
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    float loopTime = 0;
    
    CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));
    nvjpeg2kEncodeConfig_t enc_config;
    size_t bs_sz;

    CHECK_CUDA(cudaEventRecord(startEvent, params.stream));
    for(int batch_id = 0; batch_id < params.batch_size; batch_id++)
    {
        populate_encoderconfig(enc_config, input_images[batch_id], params);

        CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSetEncodeConfig(params.enc_params, &enc_config));
        CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSetQuality(params.enc_params, params.target_psnr));
        CHECK_NVJPEG2K(nvjpeg2kEncode(params.enc_handle, params.enc_state, params.enc_params, 
            &input_images[batch_id].getImageDevice(), params.stream));

        CHECK_NVJPEG2K(nvjpeg2kEncodeRetrieveBitstream(params.enc_handle, params.enc_state, NULL, &bs_sz, 
            params.stream));
        bitstreams[batch_id].resize(bs_sz);
        CHECK_NVJPEG2K(nvjpeg2kEncodeRetrieveBitstream(params.enc_handle, params.enc_state, bitstreams[batch_id].data(), &bs_sz, 
            params.stream));
        CHECK_CUDA(cudaStreamSynchronize(params.stream));
    }

    CHECK_CUDA(cudaEventRecord(stopEvent, params.stream));
    
    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
    time += static_cast<double>(loopTime/1000.0); // loopTime is in milliseconds

    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    
    return EXIT_SUCCESS;
}

int write_output(BitStreamData &bitstreams, FileNames &filenames, encode_params_t& params) 
{
    for (int batch_id = 0; batch_id < params.batch_size; batch_id++) 
    {
        // Get the file name, without extension.
        // This will be used to rename the output file.
        size_t position = filenames[batch_id].rfind("/");
        std::string sFileName =
            (std::string::npos == position)
                ? filenames[batch_id]
                : filenames[batch_id].substr(position + 1, filenames[batch_id].size());

        position = sFileName.rfind(".");
        sFileName = (std::string::npos == position) ? sFileName
                                                    : sFileName.substr(0, position);
        const char* wavelet_type []= {"revWavelet","irrevWavelet"};
        std::string fname(params.output_dir + "/" + sFileName 
                         + "_q" + std::to_string((int)params.target_psnr)
                         + "_"+ wavelet_type[params.irreversible]
                         + "_blksz"+ std::to_string((int)params.cblk_w)+"x"+std::to_string((int)params.cblk_h)
                         +".jp2");

        std::ofstream bitstream_file(fname,
                            std::ios::out | std::ios::binary);
        if (!(bitstream_file.is_open()))
        {
            std::cout << "Cannot open image: " <<fname<<std::endl;
            return EXIT_FAILURE;
            
        }
        bitstream_file.write((char*)bitstreams[batch_id].data(),bitstreams[batch_id].size());
        bitstream_file.close();

    }
    return EXIT_SUCCESS;
}

double process_images(FileNames &image_names, encode_params_t &params,
    double &total)
{
    std::vector<Image> input_images(params.batch_size);
    // we wrap over image files to process total_images of files
    FileNames::iterator file_iter = image_names.begin();
    
    BitStreamData bitsteam_output(params.batch_size);
    FileNames current_names(params.batch_size);
    CHECK_CUDA(cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));
    int total_processed = 0;

    double test_time = 0;
    int warmup = 0;

    while(total_processed < params.total_images)
    {
        if(read_next_batch(image_names, params.batch_size, file_iter, input_images.data(),current_names, params))
        {
            return EXIT_FAILURE;
        }
        double time = 0;
        if(encode_images(input_images.data(), params, bitsteam_output, time))
        {
            return EXIT_FAILURE;
        }   
        if(params.write_bitstream) 
        {
            if( write_output(bitsteam_output, current_names, params))
            {
                return EXIT_FAILURE;
            }
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

    for(auto& img : input_images)
    {
        img.tearDown();
    }
    total = test_time;

    CHECK_CUDA(cudaStreamDestroy(params.stream));
    

    return EXIT_SUCCESS;
}

int parse_blk_size(const char* blk_size, encode_params_t& params)
{
    std::istringstream img(blk_size);
    std::string temp;
    int idx = 0;
    std:: string sub_sampling = "unknown";
    while(getline(img, temp,','))
    {
        if( idx == 0)
        {
            params.cblk_w = std::stoi(temp);
        }
        else if (idx == 1)
        {
            params.cblk_h = std::stoi(temp);
        }
        idx++;
    }
    if(params.cblk_w != params.cblk_h)
    {
        std::cout<<"Only square code block sizes supported"<<std::endl;
        return EXIT_FAILURE;
    }

    if(params.cblk_w > 64 ||  params.cblk_h > 64)
    {
        std::cout<<"Invalid codeblock sizes"<<std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int parse_dimensions(const char* dims, encode_params_t& params)
{
    std::istringstream img(dims);
    std::string temp;
    int idx = 0;
    uint32_t  precision = 0;
    std:: string sub_sampling = "unknown";
    while(getline(img, temp,','))
    {
        if( idx == 0)
        {
            params.img_info.image_width = std::stoi(temp);
        }
        else if (idx == 1)
        {
            params.img_info.image_height = std::stoi(temp);
        }
        else if( idx == 2)
        {
            params.img_info.num_components = std::stoi(temp);
            if( params.img_info.num_components > 4)
            {
                std::cout<<"Max components cannot exceed 4"<<std::endl;
            }
        }
        else if (idx == 3)
        {
            precision = std::stoi(temp);
        }
        else if (idx == 4)
        {
            sub_sampling = temp;
        }
        else
        {
            std::cout<<"invalid image dims"<<std::endl;
            return EXIT_FAILURE;
        }
        idx++;
    }
    
    if( precision== 0 || precision > 16)
    {
        std::cout<<"invalid precision"<<std::endl;
        return EXIT_FAILURE;
    }
    
    if( sub_sampling == "unknown" || sub_sampling == "chroma444")
    {
        for(uint32_t c = 0; c < params.img_info.num_components; c++)
        {
            params.comp_info[c].component_width = params.img_info.image_width;
            params.comp_info[c].component_height = params.img_info.image_height;
            params.comp_info[c].precision = precision;
            params.comp_info[c].sgn = 0;
        }
    }
    else if(sub_sampling == "chroma420" || sub_sampling == "chroma422")
    {
        for(uint32_t c = 0; c < params.img_info.num_components; c++)
        {
            params.comp_info[c].precision = precision;
            params.comp_info[c].sgn = 0;
        }
        
        params.comp_info[0].component_width = params.img_info.image_width;
        params.comp_info[0].component_height = params.img_info.image_height;
        
        uint32_t chroma_w =  (params.img_info.image_width + 1)/2;
        uint32_t chroma_h =  sub_sampling == "chroma420" ?(params.img_info.image_height + 1)/2 : params.img_info.image_height;
        params.comp_info[1].component_width = chroma_w;
        params.comp_info[1].component_height = chroma_h;

        params.comp_info[2].component_width  = chroma_w;
        params.comp_info[2].component_height = chroma_h;

        for(uint32_t c = 3; c < params.img_info.num_components; c++)
        {
            params.comp_info[c].component_width = params.img_info.image_width;
            params.comp_info[c].component_height = params.img_info.image_height;
        }
    }
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
                  << "[-I] [-cblk cblk_w,cblk_h]"<<std::endl
                  << "\t[-w warmup_iterations] [-o output_dir] "<<std::endl
                  << "\t[-img_fmt img_w,img_h,num_comp,precision,chromaformat]"
                  << " (-img_fmt is mandatory for raw yuv files)"<<std::endl
                  << "\teg: for an 8 bit image of size 1920x1080 with 420 subsamling: "
                  << "-img-dims 1920,1080,3,8,chroma420"<<std::endl;
                  
        std::cout << "Parameters: " << std::endl;
        std::cout << "\timages_dir\t:\tPath to single image or directory of images"
                  << std::endl;
        std::cout << "\tbatch_size\t:\tEncode images from input by batches of "
                     "specified size"
                  << std::endl;
        std::cout << "\ttotal_images\t:\tEncode these many images, if there are "
                     "fewer images \n"
                  << "\t\t\t\tin the input than total images, encoder will loop "
                     "over the input"
                  << std::endl;
        std::cout<<"\t\t-I\t:\tEnable irreversible wavelet transform"<<std::endl;
        std::cout << "\tcblk_w,cblk_h\t:\tCode block width and code block height"<<std::endl
                  << "\t\t\t\tvalid values are 32,32 and 64,64 "<<std::endl;
        std::cout << "\twarmup_iterations:\tRun these many batches first "
                     "without measuring performance"
                  << std::endl;
        std::cout
            << "\toutput_dir\t:\tWrite compressed jpeg 2000 files  to this directory"
            << std::endl;
        return EXIT_SUCCESS;
    }

    encode_params_t params;

    params.input_dir = "./";
    if ((pidx = findParamIndex(argv, argc, "-i")) != -1)
    {
        params.input_dir = argv[pidx + 1];
        
    }
    else
    {
        std::cout << "Please specify input directory/file with encoded images" << std::endl;
        return EXIT_FAILURE;
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

    params.write_bitstream = false;
    if ((pidx = findParamIndex(argv, argc, "-o")) != -1)
    {
        params.output_dir = argv[pidx + 1];
        params.write_bitstream = true;
    }
    
    params.target_psnr = 0;
    if ((pidx = findParamIndex(argv, argc, "-q")) != -1)
    {
        params.target_psnr = atoi(argv[pidx + 1]);
    }
    params.img_fmt_init = false;
    if ((pidx = findParamIndex(argv, argc, "-img_fmt")) != -1)
    {
        if (parse_dimensions(argv[pidx + 1], params))
        {
            std::cout<<"\n -img_fmt is incorrectly set";
            return EXIT_FAILURE;
        }
        params.img_fmt_init = true;
    }

    params.irreversible = 0;
    if ((pidx = findParamIndex(argv, argc, "-I")) != -1)
    {
        params.irreversible = 1;
    }

    params.cblk_w = 64;
    params.cblk_h = 64;
    if ((pidx = findParamIndex(argv, argc, "-cblk")) != -1)
    {
        if (parse_blk_size(argv[pidx + 1], params))
        {
            std::cout<<"Invalid block size"<<std::endl;
            return EXIT_FAILURE;
        }
    }


    CHECK_NVJPEG2K(nvjpeg2kEncoderCreateSimple(&params.enc_handle));
    CHECK_NVJPEG2K(nvjpeg2kEncodeStateCreate(params.enc_handle, &params.enc_state));
    CHECK_NVJPEG2K(nvjpeg2kEncodeParamsCreate(&params.enc_params));

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

    std::cout << "Encoding images in directory: " << params.input_dir
              << ", total " << params.total_images << ", batchsize "
              << params.batch_size << std::endl;

    double total;
    if (process_images(image_names, params, total))
        return EXIT_FAILURE;
    
    std::cout << "Total encode time: " << total << std::endl;
    std::cout << "Avg encode time per image: " << total / params.total_images
              << std::endl;
    std::cout << "Avg encode speed  (in images per sec): " << params.total_images / total
              << std::endl;
    std::cout << "Avg encode time per batch: "
              << total / ((params.total_images + params.batch_size - 1) /
                          params.batch_size)
              << std::endl;

    CHECK_NVJPEG2K(nvjpeg2kEncodeParamsDestroy(params.enc_params));
    CHECK_NVJPEG2K(nvjpeg2kEncodeStateDestroy(params.enc_state));
    CHECK_NVJPEG2K(nvjpeg2kEncoderDestroy(params.enc_handle));
    
    return EXIT_SUCCESS;
}
