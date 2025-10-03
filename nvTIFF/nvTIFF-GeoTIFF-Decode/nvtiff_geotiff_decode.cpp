/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <string>
#include <vector>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) || defined(_MSC_VER)
    #define WINDOWS_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
    #include "getopt.h"
    #pragma warning(disable : 4819)
const std::string separator = "\\";
#else
    #include <getopt.h>
const std::string separator = "/";
#endif
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>

using perfclock = std::chrono::high_resolution_clock;

#include <cuda_runtime.h>
#include <nvtiff.h>

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

// clang-format off
std::map<nvtiffTag_t, std::string> tag2string = {
    {NVTIFF_TAG_MODEL_PIXEL_SCALE, "MODEL_PIXEL SCALE"},
    {NVTIFF_TAG_MODEL_TIE_POINT, "MODEL_TIEPOINT"},
    {NVTIFF_TAG_MODEL_TRANSFORMATION, "MODEL_TRANSFORMATION"}
};

std::map<nvtiffGeoKey_t, std::string> geokey2string = {
    { NVTIFF_GEOKEY_GT_MODEL_TYPE                , "GT_MODEL_TYPE"},
    { NVTIFF_GEOKEY_GT_RASTER_TYPE               , "GT_RASTER_TYPE"},
    { NVTIFF_GEOKEY_GT_CITATION                  , "GT_CITATION"},
    { NVTIFF_GEOKEY_GEODETIC_CRS                 , "GEODETIC_CRS"},
    { NVTIFF_GEOKEY_GEODETIC_CITATION            , "GEODETIC_CITATION"},
    { NVTIFF_GEOKEY_GEODETIC_DATUM               , "GEODETIC_DATUM"},
    { NVTIFF_GEOKEY_PRIME_MERIDIAN               , "PRIME_MERIDIAN"},
    { NVTIFF_GEOKEY_GEOG_LINEAR_UNITS            , "GEOG_LINEAR_UNITS"},
    { NVTIFF_GEOKEY_GEOG_LINEAR_UNIT_SIZE        , "GEOG_LINEAR_UNIT_SIZE"},
    { NVTIFF_GEOKEY_GEOG_ANGULAR_UNITS           , "GEOG_ANGULAR_UNITS"},
    { NVTIFF_GEOKEY_GEOG_ANGULAR_UNIT_SIZE       , "GEOG_ANGULAR_UNIT_SIZE"},
    { NVTIFF_GEOKEY_ELLIPSOID                    , "ELLIPSOID"},
    { NVTIFF_GEOKEY_ELLIPSOID_SEMI_MAJOR_AXIS    , "ELLIPSOID_SEMI_MAJOR_AXIS"},
    { NVTIFF_GEOKEY_ELLIPSOID_SEMI_MINOR_AXIS    , "ELLIPSOID_SEMI_MINOR_AXIS"},
    { NVTIFF_GEOKEY_ELLIPSOID_INV_FLATTENING     , "ELLIPSOID_INV_FLATTENING"},
    { NVTIFF_GEOKEY_GEOG_AZIMUTH_UNITS           , "GEOG_AZIMUTH_UNITS"},
    { NVTIFF_GEOKEY_PRIME_MERIDIAN_LONG          , "PRIME_MERIDIAN_LONG"},
    { NVTIFF_GEOKEY_PROJECTED_CRS                , "PROJECTED_CRS"},
    { NVTIFF_GEOKEY_PROJECTED_CITATION           , "PROJECTED_CITATION"},
    { NVTIFF_GEOKEY_PROJECTION                   , "PROJECTION"},
    { NVTIFF_GEOKEY_PROJ_METHOD                  , "PROJ_METHOD"},
    { NVTIFF_GEOKEY_PROJ_LINEAR_UNITS            , "PROJ_LINEAR_UNITS"},
    { NVTIFF_GEOKEY_PROJ_LINEAR_UNIT_SIZE        , "PROJ_LINEAR_UNIT_SIZE"},
    { NVTIFF_GEOKEY_PROJ_STD_PARALLEL1           , "PROJ_STD_PARALLEL1"},
    { NVTIFF_GEOKEY_PROJ_STD_PARALLEL            , "PROJ_STD_PARALLEL"},
    { NVTIFF_GEOKEY_PROJ_STD_PARALLEL2           , "PROJ_STD_PARALLEL2"},
    { NVTIFF_GEOKEY_PROJ_NAT_ORIGIN_LONG         , "PROJ_NAT_ORIGIN_LONG"},
    { NVTIFF_GEOKEY_PROJ_ORIGIN_LONG             , "PROJ_ORIGIN_LONG"},
    { NVTIFF_GEOKEY_PROJ_NAT_ORIGIN_LAT          , "PROJ_NAT_ORIGIN_LAT"},
    { NVTIFF_GEOKEY_PROJ_ORIGIN_LAT              , "PROJ_ORIGIN_LAT"},
    { NVTIFF_GEOKEY_PROJ_FALSE_EASTING           , "PROJ_FALSE_EASTING"},
    { NVTIFF_GEOKEY_PROJ_FALSE_NORTHING          , "PROJ_FALSE_NORTHING"},
    { NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_LONG       , "PROJ_FALSE_ORIGIN_LONG"},
    { NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_LAT        , "PROJ_FALSE_ORIGIN_LAT"},
    { NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_EASTING    , "PROJ_FALSE_ORIGIN_EASTING"},
    { NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_NORTHING   , "PROJ_FALSE_ORIGIN_NORTHING"},
    { NVTIFF_GEOKEY_PROJ_CENTER_LONG             , "PROJ_CENTER_LONG"},
    { NVTIFF_GEOKEY_PROJ_CENTER_LAT              , "PROJ_CENTER_LAT"},
    { NVTIFF_GEOKEY_PROJ_CENTER_EASTING          , "PROJ_CENTER_EASTING"},
    { NVTIFF_GEOKEY_PROJ_CENTER_NORTHING         , "PROJ_CENTER_NORTHING"},
    { NVTIFF_GEOKEY_PROJ_SCALE_AT_NAT_ORIGIN     , "PROJ_SCALE_AT_NAT_ORIGIN"},
    { NVTIFF_GEOKEY_PROJ_SCALE_AT_ORIGIN         , "PROJ_SCALE_AT_ORIGIN"},
    { NVTIFF_GEOKEY_PROJ_SCALE_AT_CENTER         , "PROJ_SCALE_AT_CENTER"},
    { NVTIFF_GEOKEY_PROJ_AZIMUTH_ANGLE           , "PROJ_AZIMUTH_ANGLE"},
    { NVTIFF_GEOKEY_PROJ_STRAIGHT_VERT_POLE_LONG , "PROJ_STRAIGHT_VERT_POLE_LONG"},
    { NVTIFF_GEOKEY_VERTICAL                     , "VERTICAL" },
    { NVTIFF_GEOKEY_VERTICAL_CITATION            , "VERTICAL_CITATION" },
    { NVTIFF_GEOKEY_VERTICAL_DATUM               , "VERTICAL_DATUM" },
    { NVTIFF_GEOKEY_VERTICAL_UNITS               , "VERTICAL_UNITS" },
    { NVTIFF_GEOKEY_BASE                         , "BASE" },
    { NVTIFF_GEOKEY_END                          , "END" }};
// clang-format on

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

int write_image(std::string input_filename, nvtiffImageInfo& image_info, unsigned char* chan, uint32_t image_id)
{
    // Get the file name, without extension.
    // This will be used to rename the output file.
    size_t position = input_filename.rfind(separator);
    std::string out_filename =
        (std::string::npos == position) ? input_filename : input_filename.substr(position + 1, input_filename.size());
    position = out_filename.rfind(".");
    out_filename = (std::string::npos == position) ? out_filename : out_filename.substr(0, position);
    out_filename += "_nvtiff_out_" + std::to_string(image_id);
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
        write_pnm(out_filename.c_str(), chan, image_info.image_width, image_info.image_width, image_info.image_height, bits_per_sample,
            num_samples, samples_written_to_file);
    } else {
        printf("Unable to write to file for this set of tiff image, continuing to next image\n");
    }

    return EXIT_SUCCESS;
}

void print_geotiff_metadata(nvtiffStream_t tiff_stream)
{
    constexpr std::array<nvtiffTag_t, 3> geo_tags = {
        NVTIFF_TAG_MODEL_PIXEL_SCALE, NVTIFF_TAG_MODEL_TIE_POINT, NVTIFF_TAG_MODEL_TRANSFORMATION};

    for (auto& tag : geo_tags) {

        nvtiffTagDataType_t tag_type;
        uint32_t count = 0, size = 0;
        nvtiffStatus_t status = nvtiffStreamGetTagInfo(tiff_stream, 0, tag, &tag_type, &size, &count);
        if (status == NVTIFF_STATUS_SUCCESS) {
            if (tag_type != NVTIFF_TAG_TYPE_DOUBLE) {
                std::cout << "tag type is not supported by the sample. skipping" << std::endl;
                continue;
            }
            std::vector<double> values(count);
            CHECK_NVTIFF(nvtiffStreamGetTagValue(tiff_stream, 0, tag, (void*)values.data(), count));
            std::cout << "TAG " << tag2string[tag] << ": ";
            for (auto& val : values) {
                std::cout << val << ", ";
            }
            std::cout << std::endl;
        }
    }

    uint32_t geokey_count;
    nvtiffStatus_t status = nvtiffStreamGetNumberOfGeoKeys(tiff_stream, nullptr, &geokey_count);
    if (status != NVTIFF_STATUS_SUCCESS) {
        return; // not a geotiff, return
    }

    std::vector<nvtiffGeoKey_t> geo_keys(geokey_count);
    CHECK_NVTIFF(nvtiffStreamGetNumberOfGeoKeys(tiff_stream, geo_keys.data(), &geokey_count));
    std::cout << std::endl;
    for (auto& geo_key : geo_keys) {

        uint32_t size = 0;
        uint32_t count = 0;
        nvtiffGeoKeyDataType_t geo_key_type;
        CHECK_NVTIFF(nvtiffStreamGetGeoKeyInfo(tiff_stream, geo_key, &size, &count, &geo_key_type));

        if (geo_key_type == NVTIFF_GEOKEY_TYPE_SHORT) {
            unsigned short val = 0;
            CHECK_NVTIFF(nvtiffStreamGetGeoKeySHORT(tiff_stream, geo_key, &val, 0, 1));
            std::cout << "GEOKEY " << geokey2string[geo_key] << ": " << val << std::endl;
        } else if (geo_key_type == NVTIFF_GEOKEY_TYPE_DOUBLE) {
            double val = 0;
            CHECK_NVTIFF(nvtiffStreamGetGeoKeyDOUBLE(tiff_stream, geo_key, &val, 0, 1));
            std::cout << "GEOKEY " << geokey2string[geo_key] << ": " << val << std::endl;
        } else if (geo_key_type == NVTIFF_GEOKEY_TYPE_ASCII) {
            std::vector<char> geotiffstring(count);
            CHECK_NVTIFF(nvtiffStreamGetGeoKeyASCII(tiff_stream, geo_key, geotiffstring.data(), count));
            std::cout << "GEOKEY " << geokey2string[geo_key] << ": " << geotiffstring.data() << std::endl;
        }
    }
    std::cout << std::endl;
}
static void usage(const char* pname)
{

    fprintf(stdout,
        "Usage:\n"
        "%s [options] -f|--file <TIFF_FILE>\n"
        "\n"
        "General options:\n"
        "\n"
        "\t-d DEVICE_ID\n"
        "\t--device DEVICE_ID\n"
        "\t\tSpecifies the GPU to use for images decoding.\n"
        "\t\tDefault: device 0 is used.\n"
        "\n"
        "\t-v\n"
        "\t--verbose\n"
        "\t\tPrints some information about the decoded TIFF file.\n"
        "\n"
        "\t-h\n"
        "\t--help\n"
        "\t\tPrints this help\n"
        "\n"
        "Decoding options:\n"
        "\n"
        "\t-f TIFF_FILE\n"
        "\t--file TIFF_FILE\n"
        "\t\tSpecifies the TIFF file to decode. The code supports both single and multi-image\n"
        "\n"
        "\t-b BEG_FRM\n"
        "\t--frame-beg BEG_FRM\n"
        "\t\tSpecifies the image id in the input TIFF file to start decoding from.  The image\n"
        "\t\tid must be a value between 0 and the total number of images in the file minus 1.\n"
        "\t\tValues less than 0 are clamped to 0.\n"
        "\t\tDefault: 0\n"
        "\n"
        "\t-e END_FRM\n"
        "\t--frame-end END_FRM\n"
        "\t\tSpecifies the image id in the input TIFF file to stop  decoding  at  (included).\n"
        "\t\tThe image id must be a value between 0 and the total number  of  images  in  the\n"
        "\t\tfile minus 1.  Values greater than num_images-1  are  clamped  to  num_images-1.\n"
        "\t\tDefault:  num_images-1.\n"
        "\n"
        "\t--decode-out NUM_OUT\n"
        "\t\tEnables the writing of selected images from the decoded  input  TIFF  file  into\n"
        "\t\tseparate PNM files for inspection.  If no argument is  passed,  only  the  first\n"
        "\t\timage is written to disk,  otherwise  the  first  NUM_OUT  images  are  written.\n"
        "\t\tOutput files are named <in_filename>_nvtiff_out_0.(ppm/pgm), \n"
        "\t\t<in_filename>_nvtiff_out_1.(ppm/pgm)....\n"
        "\t\tDefault: disabled.\n"
        "\n",
        pname);

    exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{
    int devId = 0;
    std::string fname;

    int verbose = 0;
    uint32_t decWriteOutN = 0;
    uint32_t frameBeg = std::numeric_limits<uint32_t>::min();
    uint32_t frameEnd = std::numeric_limits<uint32_t>::max();
    uint32_t dec_range = 0;

    int och;
    while (1) {
        int option_index = 0;
        static struct option long_options[] = {{"file", required_argument, 0, 'f'}, {"device", required_argument, 0, 'd'},
            {"decode-out", optional_argument, 0, 1}, {"frame-beg", required_argument, 0, 'b'}, {"frame-end", required_argument, 0, 'e'},
            {"verbose", no_argument, 0, 'v'}, {"help", no_argument, 0, 'h'}, {0, 0, 0, 0}};

        och = getopt_long(argc, argv, "f:d:vo::hb:e:m:cEr:s:", long_options, &option_index);
        if (och == -1)
            break;
        switch (och) {
        case 0: // handles long opts with non-NULL flag field
            break;
        case 'd':
            devId = atoi(optarg);
            break;
        case 'f':
            fname = optarg;
            break;
        case 'b':
            frameBeg = atoi(optarg);
            dec_range = 1;
            break;
        case 'e':
            frameEnd = atoi(optarg);
            dec_range = 1;
            break;
        case 'v':
            verbose++;
            break;
        case 1:
            decWriteOutN = 1;
            if (!optarg && argv[optind] != NULL && argv[optind][0] != '-') {

                decWriteOutN = atoi(argv[optind++]);
            }
            break;
        case 'h':
        case '?':
            usage(argv[0]);
        default:
            fprintf(stderr, "unknown option: %c\n", och);
            usage(argv[0]);
        }
    }

    if (fname.empty()) {
        fprintf(stderr, "Please specify a TIFF file with the -f option!\n");
        usage(argv[0]);
    }

    if (frameBeg > frameEnd) {
        fprintf(stderr, "Invalid frame range!\n");
        usage(argv[0]);
    }

    CHECK_CUDA(cudaSetDevice(devId));

    cudaDeviceProp props;

    printf("\nUsing GPU:\n");
    CHECK_CUDA(cudaGetDeviceProperties(&props, devId));
    printf("\t%2d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n", devId, props.name, props.multiProcessorCount,
        props.maxThreadsPerMultiProcessor, props.major, props.minor, props.ECCEnabled ? "on" : "off");
    printf("\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    nvtiffStream_t tiff_stream;
    nvtiffDecoder_t tiff_decoder;
    CHECK_NVTIFF(nvtiffStreamCreate(&tiff_stream));
    CHECK_NVTIFF(nvtiffDecoderCreate(&tiff_decoder, nullptr, nullptr, 0));
    auto parse_start = perfclock::now();
    CHECK_NVTIFF(nvtiffStreamParseFromFile(fname.c_str(), tiff_stream));
    auto parse_end = perfclock::now();
    double parse_time = std::chrono::duration<float>(parse_end - parse_start).count();
    printf("file parsing done in %lf secs\n", parse_time);

    print_geotiff_metadata(tiff_stream);

    uint32_t num_images = 0;
    CHECK_NVTIFF(nvtiffStreamGetNumImages(tiff_stream, &num_images));

    std::vector<nvtiffImageInfo_t> image_info(num_images);
    std::vector<uint8_t*> nvtiff_out(num_images);
    std::vector<size_t> nvtiff_out_size(num_images);
    for (uint32_t image_id = 0; image_id < num_images; image_id++) {

        CHECK_NVTIFF(nvtiffStreamGetImageInfo(tiff_stream, image_id, &image_info[image_id]));
        nvtiff_out_size[image_id] = DIV_UP((size_t)image_info[image_id].bits_per_pixel * image_info[image_id].image_width, 8) *
                                    (size_t)image_info[image_id].image_height;
        if (image_info[image_id].photometric_int == NVTIFF_PHOTOMETRIC_PALETTE) {
            nvtiff_out_size[image_id] = image_info[image_id].image_width * image_info[image_id].image_height * 3 * sizeof(uint16_t);
        }
        CHECK_CUDA(cudaMalloc(&nvtiff_out[image_id], nvtiff_out_size[image_id]));
    }

    frameBeg = std::max(frameBeg, 0u);
    frameEnd = std::min(frameEnd, num_images - 1);
    uint32_t nDecode = frameEnd - frameBeg + 1;

    printf("Decoding %d images: [%d, %d], from file %s... \n", nDecode, frameBeg, frameEnd, fname.c_str());

    auto io_start = perfclock::now();
    if (!dec_range) {
        CHECK_NVTIFF(nvtiffDecode(tiff_stream, tiff_decoder, nvtiff_out.data(), stream));
    } else {
        CHECK_NVTIFF(nvtiffDecodeRange(tiff_stream, tiff_decoder, frameBeg, nDecode, nvtiff_out.data(), stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto io_end = perfclock::now();
    double decode_time = std::chrono::duration<float>(io_end - io_start).count();

    printf("Decode done in %lf secs\n\n", decode_time);

    if (decWriteOutN) {

        const uint32_t nout = std::min(decWriteOutN, nDecode);

        printf("Writing images for the first %d subfile(s)...\n", nout);
        fflush(stdout);

        for (uint32_t image_id = 0; image_id < nout; image_id++) {

            auto& info = image_info[image_id];
            std::vector<uint8_t> imageOut_h(nvtiff_out_size[image_id]);
            CHECK_CUDA(cudaMemcpy(imageOut_h.data(), nvtiff_out[image_id], nvtiff_out_size[image_id], cudaMemcpyDeviceToHost));
            write_image(fname, info, imageOut_h.data(), image_id);
        }
    }

    // cleanup
    for (uint32_t i = 0; i < nDecode; i++) {
        CHECK_CUDA(cudaFree(nvtiff_out[i]));
    }

    if (tiff_stream) {
        CHECK_NVTIFF(nvtiffStreamDestroy(tiff_stream));
    }

    if (tiff_decoder) {
        CHECK_NVTIFF(nvtiffDecoderDestroy(tiff_decoder, stream));
    }

    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}