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

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <windows.h>
#include <filesystem>
const std::string separator = "\\";
namespace fs = std::filesystem;
#else
#include <sys/time.h> // timings
#include <experimental/filesystem>
const std::string separator = "/";
namespace fs = std::experimental::filesystem::v1;
#endif


#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>

#define CHECK_CUDA(call)                                                                                          \
    {                                                                                                             \
        cudaError_t _e = (call);                                                                                  \
        if (_e != cudaSuccess)                                                                                    \
        {                                                                                                         \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                                  \
        }                                                                                                         \
    }

#define CHECK_NVJPEG2K(call)                                                                                \
    {                                                                                                       \
        nvjpeg2kStatus_t _e = (call);                                                                       \
        if (_e != NVJPEG2K_STATUS_SUCCESS)                                                                  \
        {                                                                                                   \
            std::cout << "NVJPEG failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                            \
        }                                                                                                   \
    }

typedef std::vector<std::vector<unsigned char>> BitStreamData;
typedef std::vector<std::string> FileNames;

constexpr int MAX_COMPONENTS = 4;

class Image
{
    std::vector<void *>  pixel_data_d_;
    std::vector<void *>  pixel_data_h_;
    std::vector<size_t> pitch_in_bytes_d_;
    std::vector<size_t> pitch_in_bytes_h_;
    
    std::vector<size_t> pixel_data_size_;
    uint32_t capacity_;
    
    nvjpeg2kImage_t image_d_;
    nvjpeg2kImage_t image_h_;
    
    nvjpeg2kImageInfo_t info_;
    std::vector<nvjpeg2kImageComponentInfo_t> comp_info_;
    nvjpeg2kColorSpace_t color_space_;

    void  clear()
    {
        image_d_.num_components = 0;
        image_d_.pixel_data = nullptr;
        image_d_.pitch_in_bytes = nullptr;
        image_d_.pixel_type = NVJPEG2K_UINT8;
        image_h_.num_components = 0;
        image_h_.pixel_data = nullptr;
        image_h_.pitch_in_bytes = nullptr;
        image_h_.pixel_type = NVJPEG2K_UINT8;
        capacity_ = 0;
    }

    public:
    
    nvjpeg2kColorSpace_t getColorSpace()
    {
        return color_space_;
    }

    nvjpeg2kImage_t& getImageHost()
    {
        return image_h_;
    }

    nvjpeg2kImage_t& getImageDevice()
    {
        return image_d_;
    }

    nvjpeg2kImageInfo_t& getnvjpeg2kImageInfo()
    {
        return info_;
    }

    nvjpeg2kImageComponentInfo_t* getnvjpeg2kCompInfo()
    {
        return comp_info_.data();
    }

    Image ()
    {
        image_d_.num_components = 0;
        image_d_.pixel_data = nullptr;
        image_d_.pitch_in_bytes = nullptr;
        image_d_.pixel_type = NVJPEG2K_UINT8;
        image_h_.num_components = 0;
        image_h_.pixel_data = nullptr;
        image_h_.pitch_in_bytes = nullptr;
        image_h_.pixel_type = NVJPEG2K_UINT8;
        capacity_ = 0;
    }

    int initialize(nvjpeg2kImageInfo_t& img_info, nvjpeg2kImageComponentInfo_t *img_comp_info, nvjpeg2kColorSpace_t color_space)
    {
        memcpy(&info_, &img_info, sizeof(img_info));
        comp_info_.resize(info_.num_components);
        color_space_ = color_space;
        for(uint32_t c = 0; c < info_.num_components; c++)
        {
            memcpy(&comp_info_[c], &img_comp_info[c], sizeof(comp_info_[c]));
        }
        if(info_.num_components >  capacity_ )
        {
            pixel_data_d_.resize(info_.num_components, nullptr);
            pitch_in_bytes_d_.resize(info_.num_components, 0);
            pixel_data_h_.resize(info_.num_components, nullptr);
            pitch_in_bytes_h_.resize(info_.num_components, 0);
            pixel_data_size_.resize(info_.num_components, 0);
            capacity_ = info_.num_components;
        }

        image_d_.pixel_data     = pixel_data_d_.data();
        image_d_.pitch_in_bytes = pitch_in_bytes_d_.data();
        image_h_.pixel_data     = pixel_data_h_.data();
        image_h_.pitch_in_bytes = pitch_in_bytes_h_.data();

        if(comp_info_[0].precision <= 8)
        {
            image_d_.pixel_type = NVJPEG2K_UINT8;
            image_h_.pixel_type = NVJPEG2K_UINT8;
        } 
        else if(comp_info_[0].precision > 8  && comp_info_[0].precision <= 16) 
        {
            image_d_.pixel_type = NVJPEG2K_UINT16;
            image_h_.pixel_type = NVJPEG2K_UINT16;
        }
        else
        {
            return EXIT_FAILURE;
        }
   
        image_d_.num_components = info_.num_components;
        image_h_.num_components = image_d_.num_components;
        int bytes_per_element = 1;
        if(image_d_.pixel_type == NVJPEG2K_UINT16)
        {
            bytes_per_element = 2;
        }

        for(uint32_t c = 0; c < info_.num_components;c++)
        {
            image_d_.pitch_in_bytes[c] = 
            image_h_.pitch_in_bytes[c] = comp_info_[c].component_width * bytes_per_element;
            size_t comp_size = comp_info_[c].component_height * image_d_.pitch_in_bytes[c];
            if( comp_size > pixel_data_size_[c])
            {
                if(image_d_.pixel_data[c])
                {
                    CHECK_CUDA(cudaFree(image_d_.pixel_data[c])); 
                }
                if( image_h_.pixel_data[c])
                {
                    free(image_h_.pixel_data[c]);
                }
                pixel_data_size_[c] = comp_size;
                CHECK_CUDA(cudaMalloc(&image_d_.pixel_data[c], comp_size));
                image_h_.pixel_data[c] = malloc(comp_size);
            }
        }
        return EXIT_SUCCESS;
    }


    int tearDown()
    {
        for(uint32_t c = 0; c < capacity_;c++)
        {
            if(image_d_.pixel_data[c])
            {
                CHECK_CUDA(cudaFree(pixel_data_d_[c]));
                pixel_data_d_[c] = nullptr;
            }
            if(image_h_.pixel_data[c])
            {
                free(pixel_data_h_[c]);
                pixel_data_h_[c] = nullptr;
            }
            pixel_data_size_[c] = 0;
        }
        return EXIT_SUCCESS;
    }
    
    int copyToDevice()
    {
        int bytes_per_comp = 1;
        if(image_d_.pixel_type == NVJPEG2K_UINT16)
        {
            bytes_per_comp = 2;
        }
        for (uint32_t c = 0; c < image_d_.num_components; c++)
        {
            CHECK_CUDA(cudaMemcpy2D(image_d_.pixel_data[c], image_d_.pitch_in_bytes[c], image_h_.pixel_data[c], image_h_.pitch_in_bytes[c], 
                comp_info_[c].component_width * bytes_per_comp,
                comp_info_[c].component_height, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        return EXIT_SUCCESS;
    }
};

struct encode_params_t
{
    std::string input_dir;
    int batch_size;
    int total_images;
    int dev;
    int warmup;
    int irreversible;
    int cblk_w;
    int cblk_h;
    double target_psnr;
    nvjpeg2kEncoder_t enc_handle;
    nvjpeg2kEncodeState_t enc_state;
    nvjpeg2kEncodeParams_t enc_params;
    nvjpeg2kImageComponentInfo_t comp_info[MAX_COMPONENTS]; // required when input is yuv
    nvjpeg2kImageInfo_t img_info; // required when input is yuv
    cudaStream_t stream;
    std::string output_dir;
    bool write_bitstream;
    bool img_fmt_init;
};

double Wtime(void)
{
#if defined(_WIN32)
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer)
    {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer)
    {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    }
    else
    {
        return (double)GetTickCount() / 1000.0;
    }
#else
    struct timespec tp;
    int rv = clock_gettime(CLOCK_MONOTONIC, &tp);

    if (rv)
        return 0;

    return tp.tv_nsec / 1.0E+9 + (double)tp.tv_sec;

#endif
}
// *****************************************************************************
// reading input directory to file list
// -----------------------------------------------------------------------------
int readInput(const std::string &sInputPath, std::vector<std::string> &filelist)
{
    if( fs::is_regular_file(sInputPath))
    {
        filelist.push_back(sInputPath);
    }
    else if (fs::is_directory(sInputPath))
    { 
        fs::recursive_directory_iterator iter(sInputPath);
        for(auto& p: iter)
        {
           if( fs::is_regular_file(p))
           {
                filelist.push_back(p.path().string());
           }
        }
    }
    else
    {
        std::cout<<"unable to open input"<<std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

// *****************************************************************************
// parse parameters
// -----------------------------------------------------------------------------
int findParamIndex(const char **argv, int argc, const char *parm)
{
    int count = 0;
    int index = -1;

    for (int i = 0; i < argc; i++)
    {
        if (strncmp(argv[i], parm, 100) == 0)
        {
            index = i;
            count++;
        }
    }

    if (count == 0 || count == 1)
    {
        return index;
    }
    else
    {
        std::cout << "Error, parameter " << parm
                  << " has been specified more than once, exiting\n"
                  << std::endl;
        return -1;
    }

    return -1;
}
