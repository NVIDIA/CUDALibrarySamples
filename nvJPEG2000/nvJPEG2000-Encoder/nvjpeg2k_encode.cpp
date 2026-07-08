/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "nvjpeg2k_encode.h"
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
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
        std::cout<<"WARNING - unexpected trailing data in file"<<std::endl;
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
        std::cout<<"WARNING - unexpected trailing data in file"<<std::endl;
    }
    return EXIT_SUCCESS;
}

std::string get_file_extension(const std::string& filename)
{
    std::string ext = fs::path(filename).extension().string();
    if(ext.empty() || ext == ".")
    {
        return "";
    }

    ext.erase(0, 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

bool read_pfm_token(std::ifstream& file_input, std::string& token)
{
    while(file_input >> std::ws && file_input.peek() == '#')
    {
        file_input.ignore((std::numeric_limits<std::streamsize>::max)(), '\n');
    }
    return static_cast<bool>(file_input >> token);
}

bool consume_pfm_header_separator(std::ifstream& file_input)
{
    char separator = '\0';
    if(!file_input.get(separator))
    {
        return false;
    }
    if(separator == '\r')
    {
        // Treat CRLF as a single header separator before reading binary raster data.
        if(file_input.peek() == '\n')
        {
            file_input.ignore(1);
        }
        return true;
    }
    return std::isspace(static_cast<unsigned char>(separator)) != 0;
}

bool checked_multiply_size(size_t a, size_t b, size_t& result)
{
    if(a != 0 && b > (std::numeric_limits<size_t>::max)() / a)
    {
        return false;
    }
    result = a * b;
    return true;
}

bool parse_pfm_dimension(const std::string& token, uint32_t& dimension)
{
    if(token.empty())
    {
        return false;
    }

    uint32_t value = 0;
    for(unsigned char c : token)
    {
        if(c < '0' || c > '9')
        {
            return false;
        }

        uint32_t digit = c - '0';
        if(value > ((std::numeric_limits<uint32_t>::max)() - digit) / 10)
        {
            return false;
        }
        value = value * 10 + digit;
    }

    if(value == 0)
    {
        return false;
    }
    dimension = value;
    return true;
}

bool parse_pfm_scale(const std::string& token, float& scale)
{
    if(token.empty())
    {
        return false;
    }

    char* end = nullptr;
    errno = 0;
    scale = std::strtof(token.c_str(), &end);
    if(end != token.c_str() + token.size() || errno == ERANGE || !std::isfinite(scale) || scale == 0.0f)
    {
        return false;
    }

    return true;
}

bool validate_pfm_sizes(const nvjpeg2kImageInfo_t& info, uint8_t pfm_precision,
    size_t& row_sample_count, size_t& row_bytes)
{
    if(info.image_width == 0 || info.image_height == 0 || info.num_components == 0)
    {
        return false;
    }

    size_t num_pixels = 0;
    if(!checked_multiply_size(static_cast<size_t>(info.image_width),
        static_cast<size_t>(info.image_height), num_pixels))
    {
        return false;
    }

    size_t num_samples = 0;
    if(!checked_multiply_size(num_pixels, static_cast<size_t>(info.num_components), num_samples))
    {
        return false;
    }

    size_t sample_bytes = 0;
    if(!checked_multiply_size(num_samples, sizeof(float), sample_bytes) || sample_bytes == 0)
    {
        return false;
    }

    if(!checked_multiply_size(static_cast<size_t>(info.image_width),
        static_cast<size_t>(info.num_components), row_sample_count))
    {
        return false;
    }

    if(!checked_multiply_size(row_sample_count, sizeof(float), row_bytes))
    {
        return false;
    }

    if(row_bytes > static_cast<size_t>((std::numeric_limits<std::streamsize>::max)()))
    {
        return false;
    }

    size_t output_pitch = 0;
    size_t output_component_bytes = 0;
    const size_t output_bytes_per_sample = pfm_precision == 16 ? sizeof(uint16_t) : sizeof(float);
    return checked_multiply_size(static_cast<size_t>(info.image_width), output_bytes_per_sample, output_pitch) &&
           checked_multiply_size(static_cast<size_t>(info.image_height), output_pitch, output_component_bytes) &&
           output_component_bytes != 0;
}

uint32_t swap_u32(uint32_t value)
{
    return ((value & 0x000000ffu) << 24) |
           ((value & 0x0000ff00u) << 8)  |
           ((value & 0x00ff0000u) >> 8)  |
           ((value & 0xff000000u) >> 24);
}

float swap_float(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits = swap_u32(bits);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

#if NVJPEG2K_HAS_NLT3_ENCODE
void write_pfm_sample(float sample, float& pixel)
{
    pixel = sample;
}

void write_pfm_sample(float sample, __half& pixel)
{
    pixel = __float2half(sample);
}

template <typename PixelType>
void copy_pfm_row_to_image(const float* row_samples, nvjpeg2kImage_t& img,
    const nvjpeg2kImageInfo_t& info, uint32_t dst_y, bool swap_endian)
{
    for(uint32_t c = 0; c < info.num_components; c++)
    {
        PixelType* dst = reinterpret_cast<PixelType*>(img.pixel_data[c]) +
            static_cast<size_t>(dst_y) * (img.pitch_in_bytes[c] / sizeof(PixelType));
        for(uint32_t x = 0; x < info.image_width; x++)
        {
            float sample = row_samples[static_cast<size_t>(x) * info.num_components + c];
            if(swap_endian)
            {
                sample = swap_float(sample);
            }
            write_pfm_sample(sample, dst[x]);
        }
    }
}

template <typename PixelType>
int read_pfm_rows(std::ifstream& file_input, nvjpeg2kImage_t& img,
    const nvjpeg2kImageInfo_t& info, size_t row_sample_count, size_t row_bytes, bool swap_endian)
{
    std::vector<float> row_samples;
    if(row_sample_count > row_samples.max_size())
    {
        std::cout << "Invalid pfm image dimensions" << std::endl;
        return EXIT_FAILURE;
    }
    row_samples.resize(row_sample_count);
    std::streamsize row_bytes_to_read = static_cast<std::streamsize>(row_bytes);

    for(uint32_t pfm_y = 0; pfm_y < info.image_height; pfm_y++)
    {
        if(!file_input.read(reinterpret_cast<char*>(row_samples.data()), row_bytes_to_read))
        {
            std::cout << "failed to read pfm image from file" << std::endl;
            return EXIT_FAILURE;
        }

        uint32_t dst_y = info.image_height - 1 - pfm_y;
        copy_pfm_row_to_image<PixelType>(row_samples.data(), img, info, dst_y, swap_endian);
    }

    return EXIT_SUCCESS;
}
#endif

int read_pfm(std::ifstream& file_input, Image& image, encode_params_t& params)
{
#if NVJPEG2K_HAS_NLT3_ENCODE
    std::string token;
    if(!read_pfm_token(file_input, token))
    {
        std::cout << "Invalid pfm file" << std::endl;
        return EXIT_FAILURE;
    }

    nvjpeg2kImageInfo_t info = {};
    nvjpeg2kColorSpace_t color_space;
    if(token == "Pf")
    {
        info.num_components = 1;
        color_space = NVJPEG2K_COLORSPACE_GRAY;
    }
    else if(token == "PF")
    {
        info.num_components = 3;
        color_space = NVJPEG2K_COLORSPACE_SRGB;
    }
    else
    {
        std::cout << "pfm format not supported" << std::endl;
        return EXIT_FAILURE;
    }

    if(!read_pfm_token(file_input, token))
    {
        std::cout << "Invalid pfm width" << std::endl;
        return EXIT_FAILURE;
    }
    if(!parse_pfm_dimension(token, info.image_width))
    {
        std::cout << "Invalid pfm width" << std::endl;
        return EXIT_FAILURE;
    }

    if(!read_pfm_token(file_input, token))
    {
        std::cout << "Invalid pfm height" << std::endl;
        return EXIT_FAILURE;
    }
    if(!parse_pfm_dimension(token, info.image_height))
    {
        std::cout << "Invalid pfm height" << std::endl;
        return EXIT_FAILURE;
    }

    if(!read_pfm_token(file_input, token))
    {
        std::cout << "Invalid pfm scale" << std::endl;
        return EXIT_FAILURE;
    }
    float scale = 0.0f;
    if(!parse_pfm_scale(token, scale))
    {
        std::cout << "Invalid pfm scale" << std::endl;
        return EXIT_FAILURE;
    }

    if(!consume_pfm_header_separator(file_input))
    {
        std::cout << "Invalid pfm scale" << std::endl;
        return EXIT_FAILURE;
    }

    if(params.pfm_precision != 16 && params.pfm_precision != 32)
    {
        std::cout << "Invalid pfm precision. Supported values are 16 and 32" << std::endl;
        return EXIT_FAILURE;
    }

    size_t row_sample_count = 0;
    size_t row_bytes = 0;
    if(!validate_pfm_sizes(info, params.pfm_precision, row_sample_count, row_bytes))
    {
        std::cout << "Invalid pfm image dimensions" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<nvjpeg2kImageComponentInfo_t> comp_info(info.num_components);
    for(auto& comp : comp_info)
    {
        comp.component_width  = info.image_width;
        comp.component_height = info.image_height;
        comp.precision = params.pfm_precision;
        comp.sgn = 0;
    }

    nvjpeg2kImageType_t pixel_type = params.pfm_precision == 16 ? NVJPEG2K_FP16 : NVJPEG2K_FP32;
    if(image.initialize(info, comp_info.data(), color_space, pixel_type))
    {
        return EXIT_FAILURE;
    }
    image.setNltType3(params.pfm_precision);

    bool file_little_endian = scale < 0.0f;
    const uint16_t endian_probe = 1;
    bool host_little_endian = *reinterpret_cast<const uint8_t*>(&endian_probe) == 1;
    bool swap_endian = file_little_endian != host_little_endian;

    auto& img = image.getImageHost();
    if(img.pixel_type == NVJPEG2K_FP16)
    {
        if(read_pfm_rows<__half>(file_input, img, info, row_sample_count, row_bytes, swap_endian))
        {
            return EXIT_FAILURE;
        }
    }
    else
    {
        if(read_pfm_rows<float>(file_input, img, info, row_sample_count, row_bytes, swap_endian))
        {
            return EXIT_FAILURE;
        }
    }

    char extra;
    if(file_input.read(&extra, 1))
    {
        std::cout << "WARNING - unexpected trailing data in file" << std::endl;
    }
    return EXIT_SUCCESS;
#else
    (void)file_input;
    (void)image;
    (void)params;
    std::cout << "PFM input requires nvJPEG2000 0.11 or newer" << std::endl;
    return EXIT_FAILURE;
#endif
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
        std::string ext = get_file_extension(*cur_iter);
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
        else if(ext == "pfm")
        {
            if(read_pfm(image_file, input_images[counter], params))
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
    enc_config = {}; // set to default
    enc_config.stream_type =  NVJPEG2K_STREAM_JP2;
    enc_config.color_space = input_image.getColorSpace();
    enc_config.image_width =  input_image.getnvjpeg2kImageInfo().image_width;
    enc_config.image_height = input_image.getnvjpeg2kImageInfo().image_height;
    enc_config.num_components = input_image.getnvjpeg2kImageInfo().num_components;
    enc_config.image_comp_info = input_image.getnvjpeg2kCompInfo();
    enc_config.code_block_w = (uint32_t)params.cblk_w;
    enc_config.code_block_h = (uint32_t)params.cblk_h;
    enc_config.irreversible = input_image.isNltType3() ? 0 : (uint32_t)params.is_irreversible;
    enc_config.mct_mode = (enc_config.color_space == NVJPEG2K_COLORSPACE_SRGB && !input_image.isNltType3()) ? 1 : 0;
    enc_config.prog_order = NVJPEG2K_LRCP;
    enc_config.num_resolutions = 6;

    if (params.use_ht || input_image.isNltType3()) {
        enc_config.rsiz = NVJPEG2K_RSIZ_HT;
        enc_config.encode_modes = NVJPEG2K_MODE_HT;
    }
}

bool is_128x32_code_block(const encode_params_t& params)
{
    return params.cblk_w == 128 && params.cblk_h == 32;
}

int validate_encoderconfig(Image& input_image, encode_params_t& params)
{
    if(input_image.isNltType3())
    {
        if(params.is_irreversible)
        {
            std::cout << "PFM input uses NLT type 3 and requires the reversible wavelet" << std::endl;
            return EXIT_FAILURE;
        }
        if(params.quality_value != 0)
        {
            std::cout << "Quality options are not supported with PFM input" << std::endl;
            return EXIT_FAILURE;
        }
    }
    else if(is_128x32_code_block(params))
    {
        std::cout << "128x32 code block size is only supported with PFM input" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int encode_images(Image* input_images, encode_params_t &params, BitStreamData &bitstreams, double &time)
{
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    float loopTime = 0;

    for(int batch_id = 0; batch_id < params.batch_size; batch_id++)
    {
        if(validate_encoderconfig(input_images[batch_id], params))
        {
            return EXIT_FAILURE;
        }
    }
    
    CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));
    nvjpeg2kEncodeConfig_t enc_config;
    size_t bs_sz;

    CHECK_CUDA(cudaEventRecord(startEvent, params.stream));
    for(int batch_id = 0; batch_id < params.batch_size; batch_id++)
    {
        populate_encoderconfig(enc_config, input_images[batch_id], params);

        CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSetEncodeConfig(params.enc_params, &enc_config));
#if NVJPEG2K_HAS_NLT3_ENCODE
        if(input_images[batch_id].isNltType3())
        {
            std::vector<nvjpeg2kNLTInfo_t> nlt_info(enc_config.num_components);
            for(auto& info : nlt_info)
            {
                info.type = NVJPEG2K_NLT_TYPE3;
                info.bit_depth = input_images[batch_id].getNltBitDepth();
            }
            CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSetNLTInfo(params.enc_params, nlt_info.data(), enc_config.num_components));
        }
        else
        {
            CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSetNLTInfo(params.enc_params, NULL, 0));
        }
#else
        if(input_images[batch_id].isNltType3())
        {
            std::cout << "NLT type 3 encoding requires nvJPEG2000 0.11 or newer" << std::endl;
            return EXIT_FAILURE;
        }
#endif
        if (params.quality_value != 0) {
            CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSpecifyQuality(params.enc_params, params.quality_type, params.quality_value));
        }
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

int write_output(BitStreamData &bitstreams, FileNames &filenames, Image* input_images, encode_params_t& params)
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
        const char* ht_type []= {"legacy", "HT"};
        int actual_irreversible = input_images[batch_id].isNltType3() ? 0 : params.is_irreversible;
        int actual_ht = (params.use_ht || input_images[batch_id].isNltType3()) ? 1 : 0;
        std::string fname(params.output_dir + "/" + sFileName 
                         + "_qt" + std::to_string(params.quality_type)
                         + "_qv" + std::to_string(params.quality_value)
                         + "_"+ wavelet_type[actual_irreversible]
                         + "_"+ ht_type[actual_ht]
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
            if( write_output(bitsteam_output, current_names, input_images.data(), params))
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
    std::string block_size(blk_size);
    for (char &c : block_size)
    {
        if(c == 'x' || c == 'X')
        {
            c = ',';
        }
    }

    std::istringstream img(block_size);
    std::string temp;
    int idx = 0;
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
    if(idx != 2 || !((params.cblk_w == 32 && params.cblk_h == 32) ||
                     (params.cblk_w == 64 && params.cblk_h == 64) ||
                     (params.cblk_w == 128 && params.cblk_h == 32)))
    {
        std::cout<<"Supported code block sizes are 32x32, 64x64 and 128x32"<<std::endl;
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
                  << "\t[-w warmup_iterations] [-o output_dir] [-ht] "<<std::endl
                  << "\t[-q_factor value] [-quantization value] [-psnr value]"<<std::endl
                  << "\t[-pfm_precision 16|32]"<<std::endl
                  << "\t[-img_fmt img_w,img_h,num_comp,precision,chromaformat]"
                  << " (-img_fmt is mandatory for raw yuv files)"<<std::endl
                  << "\teg: for an 8 bit image of size 1920x1080 with 420 subsampling: "
                  << "-img-dims 1920,1080,3,8,chroma420"<<std::endl;
                  
        std::cout << "Parameters: " << std::endl;
        std::cout << "\timages_dir\t:\tPath to single image or directory of images" << std::endl;
        std::cout << "\tbatch_size\t:\tEncode images from input by batches of specified size" << std::endl;
        std::cout << "\ttotal_images\t:\tEncode these many images, if there are fewer images \n"
                  << "\t\t\t\tin the input than total images, encoder will loop over the input" << std::endl;
        std::cout << "\t-ht\t\t:\tEnable High Throughput encoding"<<std::endl;
        std::cout << "\t-I\t\t:\tEnable irreversible wavelet transform. Must be set to use any quality option"<<std::endl;
        std::cout << "\t-q_factor\t:\tSet `value` as Q-Factor (jpeg-like) quality:\n"
                  << "\t\t\t\tfloating point from 1.0 (worst) to 100.0 (best)"<<std::endl;
        std::cout << "\t-quantization\t:\tSet `value` as quantization step (by how much pixel data will be divided):\n"
                  << "\t\t\t\tfloating point, bigger the value, the lower the result quality."<<std::endl;
        std::cout << "\t-psnr\t\t:\tSet `value` as target PSNR value: positive floating point.\n"
                  << "\t\t\t\tCannot be used with -ht option "<<std::endl;
        std::cout << "\t-pfm_precision\t:\tSet NLT bit depth for PFM input. Supported values are 16 and 32."
                  << "\n\t\t\t\tDefault is 32"<<std::endl;
        std::cout << "\tcblk_w,cblk_h\t:\tCode block width and code block height"<<std::endl
                  << "\t\t\t\tvalid values are 32x32 and 64x64. PFM input also supports 128x32"<<std::endl;
        std::cout << "\twarmup_iterations:\tRun these many batches first without measuring performance" << std::endl;
        std::cout << "\toutput_dir\t:\tWrite compressed jpeg 2000 files  to this directory" << std::endl;
        return EXIT_SUCCESS;
    }

    encode_params_t params = {};

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

    params.pfm_precision = 32;
    if ((pidx = findParamIndex(argv, argc, "-pfm_precision")) != -1)
    {
        int pfm_precision = std::atoi(argv[pidx + 1]);
        if(pfm_precision != 16 && pfm_precision != 32)
        {
            std::cout << "-pfm_precision must be 16 or 32" << std::endl;
            return EXIT_FAILURE;
        }
        params.pfm_precision = static_cast<uint8_t>(pfm_precision);
    }

    params.use_ht = 0;
    if ((pidx = findParamIndex(argv, argc, "-ht")) != -1)
    {
        params.use_ht = 1;
    }

    params.is_irreversible = 0;
    if ((pidx = findParamIndex(argv, argc, "-I")) != -1)
    {
        params.is_irreversible = 1;
    }

    // 0 means do not set quality, use default instead
    params.quality_value = 0;
    params.quality_type = NVJPEG2K_QUALITY_TYPE_TARGET_PSNR;
    if ((pidx = findParamIndex(argv, argc, "-q_factor")) != -1)
    {
        params.quality_value = atof(argv[pidx + 1]);
        params.quality_type = NVJPEG2K_QUALITY_TYPE_Q_FACTOR;
    }

    if ((pidx = findParamIndex(argv, argc, "-quantization")) != -1)
    {
        if (params.quality_value != 0)
        {
            std::cout << "At most one quality type can be set" << std::endl;
            return EXIT_FAILURE;
        }

        params.quality_value = atof(argv[pidx + 1]);
        params.quality_type = NVJPEG2K_QUALITY_TYPE_QUANTIZATION_STEP;
    }

    if ((pidx = findParamIndex(argv, argc, "-psnr")) != -1)
    {
        if (params.quality_value != 0)
        {
            std::cout << "At most one quality type can be set" << std::endl;
            return EXIT_FAILURE;
        }

        if (params.use_ht)
        {
            std::cout << "-psnr option cannot be used with -ht option" << std::endl;
            return EXIT_FAILURE;
        }

        params.quality_value = atof(argv[pidx + 1]);
        params.quality_type = NVJPEG2K_QUALITY_TYPE_TARGET_PSNR;
    }


    if (params.quality_value != 0 && !params.is_irreversible)
    {
        std::cout << "Quality can only be used if irreversible transform is enabled" << std::endl;
        return EXIT_FAILURE;
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

    // read source images
    FileNames image_names;
    if(readInput(params.input_dir, image_names))
    {
        return EXIT_FAILURE;
    }

    CHECK_NVJPEG2K(nvjpeg2kEncoderCreateSimple(&params.enc_handle));
    CHECK_NVJPEG2K(nvjpeg2kEncodeStateCreate(params.enc_handle, &params.enc_state));
    CHECK_NVJPEG2K(nvjpeg2kEncodeParamsCreate(&params.enc_params));

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
