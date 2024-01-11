#pragma once
#include <vector>
#include <iostream>
#include <chrono>

#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>

float gpu_encode(unsigned char *image, int batch_size,
                 int *height, int *width, int dev);

typedef std::vector<std::vector<unsigned char>> BitStreamData;

// helper functions
inline void check_cuda(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        std::cout << "CUDA Runtime failure: '#" << status << "' at " << __FILE__ << ":" << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline int check_nvjpeg2k(nvjpeg2kStatus_t call)
{
    if (call != NVJPEG2K_STATUS_SUCCESS)
    {
        std::cout << "NVJPEG failure: '#" << call << "' at " << __FILE__ << ":" << __LINE__ << std::endl;
        return EXIT_FAILURE;
    }
    return call;
}

struct Image
{
    nvjpeg2kImage_t image_h_;
    nvjpeg2kImage_t image_d_;
    std::vector<nvjpeg2kImageComponentInfo_t> comp_info_;
    nvjpeg2kEncodeConfig_t enc_config;

    std::vector<void *> pixel_data_d_;
    std::vector<void *> pixel_data_h_;
    std::vector<size_t> pitch_in_bytes_d_;
    std::vector<size_t> pitch_in_bytes_h_;
    std::vector<size_t> pixel_data_size_;

    int bytes_per_element;

    Image(unsigned char *image, uint32_t image_width, uint32_t image_height, uint32_t num_components, nvjpeg2kImageType_t pixel_type)
    {
        pixel_data_h_.resize(num_components, nullptr);
        pixel_data_d_.resize(num_components, nullptr);
        pitch_in_bytes_h_.resize(num_components, 0);
        pitch_in_bytes_d_.resize(num_components, 0);
        pixel_data_size_.resize(num_components, 0);

        nvjpeg2kImageComponentInfo_t tmp_info;
        comp_info_.resize(num_components, tmp_info);

        image_d_.pixel_data = pixel_data_d_.data();
        image_d_.pitch_in_bytes = pitch_in_bytes_d_.data();
        image_h_.pixel_data = pixel_data_h_.data();
        image_h_.pitch_in_bytes = pitch_in_bytes_h_.data();

        image_d_.pixel_type = pixel_type;
        image_h_.pixel_type = pixel_type;

        image_d_.num_components = num_components;
        image_h_.num_components = num_components;

        bytes_per_element = pixel_type == NVJPEG2K_UINT8 ? 1 : 2;

        for (uint32_t c = 0; c < num_components; c++)
        {
            comp_info_[c].component_width = image_width;
            comp_info_[c].component_height = image_height;
            comp_info_[c].precision = pixel_type == NVJPEG2K_UINT8 ? 8 : 16;
            comp_info_[c].sgn = 0;
        }

        // allocate resources
        for (uint32_t c = 0; c < num_components; c++)
        {
            image_d_.pitch_in_bytes[c] =
                image_h_.pitch_in_bytes[c] = comp_info_[c].component_width * bytes_per_element * num_components;
            size_t comp_size = comp_info_[c].component_height * image_d_.pitch_in_bytes[c];
            if (comp_size > pixel_data_size_[c])
            {
                if (image_d_.pixel_data[c])
                {
                    check_cuda(cudaFree(image_d_.pixel_data[c]));
                }
                if (image_h_.pixel_data[c])
                {
                    free(image_h_.pixel_data[c]);
                }
                pixel_data_size_[c] = comp_size;
                check_cuda(cudaMalloc(&image_d_.pixel_data[c], comp_size));
                image_h_.pixel_data[c] = malloc(comp_size);
            }
        }

        // load MemoryView in host pixel rgb planars
        unsigned char *r = reinterpret_cast<unsigned char *>(image_h_.pixel_data[0]);
        unsigned char *g = reinterpret_cast<unsigned char *>(image_h_.pixel_data[1]);
        unsigned char *b = reinterpret_cast<unsigned char *>(image_h_.pixel_data[2]);

        auto start = std::chrono::steady_clock::now();
        for (unsigned int y = 0; y < image_height; y++)
        {
            for (unsigned int x = 0; x < image_width; x++)
            {
                r[y * image_h_.pitch_in_bytes[0] + x] = image[y * image_h_.pitch_in_bytes[0] + (3 * x + 0)];
                g[y * image_h_.pitch_in_bytes[1] + x] = image[y * image_h_.pitch_in_bytes[1] + (3 * x + 1)];
                b[y * image_h_.pitch_in_bytes[2] + x] = image[y * image_h_.pitch_in_bytes[2] + (3 * x + 2)];
            }
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

        // copy to device
        for (uint32_t c = 0; c < num_components; c++)
        {
            check_cuda(cudaMemcpy2D(image_d_.pixel_data[c], image_d_.pitch_in_bytes[c], image_h_.pixel_data[c], image_h_.pitch_in_bytes[c],
                                    comp_info_[c].component_width * bytes_per_element,
                                    comp_info_[c].component_height, cudaMemcpyHostToDevice));
        }

        // encode configurations
        memset(&enc_config, 0, sizeof(enc_config));
        enc_config.stream_type = NVJPEG2K_STREAM_JP2;
        enc_config.color_space = NVJPEG2K_COLORSPACE_SRGB;
        enc_config.image_width = image_width;
        enc_config.image_height = image_height;
        enc_config.num_components = num_components;
        enc_config.image_comp_info = comp_info_.data();
        enc_config.code_block_w = 64;
        enc_config.code_block_h = 64;
        enc_config.irreversible = 0;
        enc_config.mct_mode = 1;
        enc_config.prog_order = NVJPEG2K_RPCL;
        enc_config.num_resolutions = 6;
    }

    // release resources
    ~Image()
    {
        // std::cout << "released resources" << std::endl;
        for (auto &ptr : pixel_data_d_)
        {
            if (ptr)
            {
                cudaFree(ptr);
                ptr = nullptr;
            }
        }

        for (auto &ptr : pixel_data_h_)
        {
            if (ptr)
            {
                free(ptr);
                ptr = nullptr;
            }
        }
    }
};