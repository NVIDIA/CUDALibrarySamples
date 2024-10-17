/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <cuda_runtime.h>
#include <nvtiff.h>
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) || defined(_MSC_VER)
const std::string separator = "\\";
#else
const std::string separator = "/";
#endif

using perfclock = std::chrono::high_resolution_clock;

#define DIV_UP(a, b) (((a) + ((b)-1)) / (b))

#define CHECK_CUDA(call)                                                                                                \
    {                                                                                                                   \
        cudaError_t err = call;                                                                                         \
        if (cudaSuccess != err) {                                                                                       \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                         \
        }                                                                                                               \
    }

#define CHECK_NVTIFF(call)                                                                             \
    {                                                                                                  \
        nvtiffStatus_t _e = (call);                                                                    \
        if (_e != NVTIFF_STATUS_SUCCESS) {                                                             \
            fprintf(stderr, "nvtiff error code %d in file '%s' in line %i\n", _e, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    }

size_t getImageSize(const nvtiffImageInfo_t& info, uint32_t roi_width, uint32_t roi_height)
{
    size_t bits_per_pixel = info.bits_per_pixel;
    if (info.photometric_int == NVTIFF_PHOTOMETRIC_PALETTE) {
        bits_per_pixel = 3 * 16;
    }
    return DIV_UP((size_t)roi_width * (size_t)roi_height * bits_per_pixel, 8);
}

using std::chrono::high_resolution_clock;
template <typename T>
std::string format_duration(T dur)
{
    using namespace std::chrono;

    auto secs = duration_cast<seconds>(dur);
    dur -= secs;
    auto millis = duration_cast<milliseconds>(dur);
    dur -= millis;
    auto micros = duration_cast<microseconds>(dur);
    dur -= micros;

    if (secs.count() > 0) {
        return std::to_string(secs.count()) + "." + std::to_string(millis.count()) + "s ";
    } else if (millis.count() > 0) {
        return std::to_string(millis.count()) + "." + std::to_string(micros.count()) + "ms ";
    } else if (micros.count() > 0) {
        return std::to_string(micros.count()) + "us ";
    }
    return "0s";
}

// *****************************************************************************
// parse parameters
// -----------------------------------------------------------------------------
int findParamIndex(const char** argv, int argc, const char* parm)
{
    int count = 0;
    int index = -1;

    for (int i = 0; i < argc; i++) {
        if (strncmp(argv[i], parm, 100) == 0) {
            index = i;
            count++;
        }
    }

    if (count == 0 || count == 1) {
        return index;
    } else {
        std::cout << "Error, parameter " << parm << " has been specified more than once, exiting\n" << std::endl;
        return -1;
    }
    return -1;
}

// *****************************************************************************
// parse roi coordinates
// -----------------------------------------------------------------------------
int parseDecodeCoordinates(const char* argv, int32_t& offset_x, int32_t& offset_y, int32_t& roi_width, int32_t& roi_height)
{
    std::istringstream decode_area(argv);
    std::string temp;
    int idx = 0;
    while (getline(decode_area, temp, ',')) {
        if (idx == 0) {
            offset_x = std::stoi(temp);
        } else if (idx == 1) {
            offset_y = std::stoi(temp);
        } else if (idx == 2) {
            roi_width = std::stoi(temp);
        } else if (idx == 3) {
            roi_height = std::stoi(temp);
        } else {
            std::cout << "Invalid ROI" << std::endl;
            return EXIT_FAILURE;
        }
        idx++;
    }
    return EXIT_SUCCESS;
}

void write_pnm(const char* filename, unsigned char* chan, uint32_t ld, uint32_t width, uint32_t height, uint32_t BPP, uint32_t numcomp,
    uint32_t write_out_numcomp)
{
    std::ofstream rOutputStream(filename);
    if (!rOutputStream) {
        std::cerr << "Cannot open output file: " << filename << std::endl;
        return;
    }
    if (numcomp == 1) {
        rOutputStream << "P5\n";
    } else {
        rOutputStream << "P6\n";
    }

    rOutputStream << "#nvTIFF\n";
    rOutputStream << width << " " << height << "\n";
    rOutputStream << (1 << BPP) - 1 << "\n";
    if (BPP == 8) {
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                for (uint32_t c = 0; c < write_out_numcomp; c++) {
                    rOutputStream << chan[(y * ld + x) * numcomp + c];
                }
            }
        }
    } else {
        uint16_t* chan16 = reinterpret_cast<uint16_t*>(chan);
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                for (uint32_t c = 0; c < write_out_numcomp; c++) {
                    uint32_t pix_val = chan16[(y * ld + x) * numcomp + c];
                    rOutputStream << static_cast<unsigned char>((pix_val) >> 8) << static_cast<unsigned char>((pix_val)&0xff);
                }
            }
        }
    }
    return;
}

int write_image(std::string input_filename, nvtiffImageInfo& image_info, unsigned char* chan, uint32_t image_id, uint32_t width,
    uint32_t height, std::string& out_dir)
{
    // Get the file name, without extension.
    // This will be used to rename the output file.
    size_t position = input_filename.rfind(separator);
    std::string out_filename =
        (std::string::npos == position) ? input_filename : input_filename.substr(position + 1, input_filename.size());
    position = out_filename.rfind(".");
    out_filename = (std::string::npos == position) ? out_filename : out_filename.substr(0, position);
    out_filename += "_nvtiff_out_" + std::to_string(image_id);
    out_filename = out_dir + separator + out_filename;
    uint32_t num_samples = image_info.samples_per_pixel;
    uint32_t samples_written_to_file = image_info.samples_per_pixel;
    if (image_info.samples_per_pixel == 3 || image_info.samples_per_pixel == 4 ||
        image_info.photometric_int == NVTIFF_PHOTOMETRIC_PALETTE) {
        out_filename += ".ppm";
        samples_written_to_file = 3;
    } else if (image_info.samples_per_pixel == 1) {
        out_filename += ".pgm";
    } else {
        printf("Unable to write image with %d samples per pixel, continuing to the next image..\n", image_info.samples_per_pixel);
        return EXIT_SUCCESS;
    }
    uint32_t bits_per_sample = image_info.bits_per_sample[0];
    if (image_info.photometric_int == NVTIFF_PHOTOMETRIC_PALETTE) {
        num_samples = 3;
        samples_written_to_file = 3;
        bits_per_sample = 16;
    }

    if (bits_per_sample == 16 || bits_per_sample == 8) {
        write_pnm(out_filename.c_str(), chan, width, width, height, bits_per_sample, num_samples, samples_written_to_file);
    } else {
        printf("Unable to write to file for this set of tiff image, continuing to next image\n");
    }

    return EXIT_SUCCESS;
}

int decode_roi(const std::string& filename, int32_t x, int32_t y, int32_t width, int32_t height, uint32_t image_id, std::string& out_dir,
    bool write_output)
{
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    nvtiffStream_t tiff_stream;
    CHECK_NVTIFF(nvtiffStreamCreate(&tiff_stream));
    // use nvtiffStreamParse is file is already in host memory
    CHECK_NVTIFF(nvtiffStreamParseFromFile(filename.c_str(), tiff_stream));

    nvtiffImageInfo_t image_info;
    CHECK_NVTIFF(nvtiffStreamGetImageInfo(tiff_stream, image_id, &image_info));

    if (width == -1)
        width = image_info.image_width;

    if (height == -1)
        height = image_info.image_height;

    width = std::min((uint32_t)(image_info.image_width - (uint32_t)x), (uint32_t)width);
    height = std::min((uint32_t)(image_info.image_height - (uint32_t)y), (uint32_t)height);

    size_t size = getImageSize(image_info, width, height);

    void* image_out_d;
    CHECK_CUDA(cudaMalloc(&image_out_d, size));

    nvtiffDecoder_t decoder;
    CHECK_NVTIFF(nvtiffDecoderCreateSimple(&decoder, stream));

    nvtiffDecodeParams_t decode_params;
    CHECK_NVTIFF(nvtiffDecodeParamsCreate(&decode_params));
    CHECK_NVTIFF(nvtiffDecodeParamsSetROI(decode_params, x, y, width, height));

    if (image_info.photometric_int == NVTIFF_PHOTOMETRIC_PALETTE) {
        CHECK_NVTIFF(nvtiffDecodeParamsSetOutputFormat(decode_params, NVTIFF_OUTPUT_RGB_I_UINT16));
    } else if (image_info.photometric_int == NVTIFF_PHOTOMETRIC_YCBCR && image_info.compression == NVTIFF_COMPRESSION_JPEG) {
        CHECK_NVTIFF(nvtiffDecodeParamsSetOutputFormat(decode_params, NVTIFF_OUTPUT_RGB_I_UINT8));
    } else {
        CHECK_NVTIFF(nvtiffDecodeParamsSetOutputFormat(decode_params, NVTIFF_OUTPUT_UNCHANGED_I));
    }
    auto start = high_resolution_clock::now();
    CHECK_NVTIFF(nvtiffDecodeImage(tiff_stream, decoder, decode_params, image_id, image_out_d, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto end = high_resolution_clock::now();
    printf("ROI Decode took %s\n", format_duration(end - start).c_str());

    if (write_output) {
        std::vector<uint8_t> image_out_h(size);
        CHECK_CUDA(cudaMemcpy(image_out_h.data(), image_out_d, size, cudaMemcpyDeviceToHost));
        write_image(filename, image_info, image_out_h.data(), image_id, width, height, out_dir);
    }

    CHECK_CUDA(cudaFree(image_out_d));
    CHECK_NVTIFF(nvtiffDecodeParamsDestroy(decode_params));
    CHECK_NVTIFF(nvtiffStreamDestroy(tiff_stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    int pidx;

    if ((pidx = findParamIndex(argv, argc, "-h")) != -1 || (pidx = findParamIndex(argv, argc, "--help")) != -1) {
        std::cout << "Usage: " << argv[0]
                  << " -f <TIFF_FILE> [-image_id image_id]  [-o output_dir] [-roi offset_x, offset_y, roi_width, roi_height] " << std::endl;
        std::cout << "\t<TIFF_FILE>\t: TIFF file to decode. "
                  << std::endl;
        std::cout << "\timage_id\t: Image index(IFD location) within a TIFF file. Defaults to 0."
                  << std::endl;
        std::cout << "\toutput_dir\t: Write decoded images in pnm format to this directory"
                  << std::endl;
        std::cout<< "\toffset_x, offset_y, roi_width, roi_height : Region of interest coordinates for decoding."<<std::endl;
        return EXIT_SUCCESS;
    }
    std::string tiff_file;
    if ((pidx = findParamIndex(argv, argc, "-f")) != -1) {
        tiff_file = argv[pidx + 1];
    } else {
        std::cout << "Please specificy tiff file" << std::endl;
        EXIT_FAILURE;
    }
    uint32_t image_id = 0;
    if ((pidx = findParamIndex(argv, argc, "-image_id")) != -1) {
        image_id = std::atoi(argv[pidx + 1]);
    }

    int32_t offset_x = 0, offset_y = 0, roi_width = -1, roi_height = -1;
    if ((pidx = findParamIndex(argv, argc, "-roi")) != -1) {
        if (parseDecodeCoordinates(argv[pidx + 1], offset_x, offset_y, roi_width, roi_height)) {
            return EXIT_FAILURE;
        }
    }

    bool write_decoded = false;
    std::string output_dir;
    if ((pidx = findParamIndex(argv, argc, "-o")) != -1) {
        output_dir = argv[pidx + 1];
        write_decoded = true;
    }

    return decode_roi(tiff_file.c_str(), offset_x, offset_y, roi_width, roi_height, image_id, output_dir, write_decoded);
}