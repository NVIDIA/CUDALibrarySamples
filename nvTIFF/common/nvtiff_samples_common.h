/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Shared plumbing for the nvTIFF samples: status checking, output-format
// selection, encode metadata validation, and PNM preview writing. The nvTIFF
// API usage worth reading lives in the individual samples, not here.

#ifndef NVTIFF_SAMPLES_COMMON_H
#define NVTIFF_SAMPLES_COMMON_H

#include <cuda_runtime.h>
#include <nvtiff.h>

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

inline const char* nvtiff_status_string(nvtiffStatus_t status)
{
    switch (status) {
        case NVTIFF_STATUS_SUCCESS: return "NVTIFF_STATUS_SUCCESS";
        case NVTIFF_STATUS_NOT_INITIALIZED: return "NVTIFF_STATUS_NOT_INITIALIZED";
        case NVTIFF_STATUS_INVALID_PARAMETER: return "NVTIFF_STATUS_INVALID_PARAMETER";
        case NVTIFF_STATUS_BAD_TIFF: return "NVTIFF_STATUS_BAD_TIFF";
        case NVTIFF_STATUS_TIFF_NOT_SUPPORTED: return "NVTIFF_STATUS_TIFF_NOT_SUPPORTED";
        case NVTIFF_STATUS_ALLOCATOR_FAILURE: return "NVTIFF_STATUS_ALLOCATOR_FAILURE";
        case NVTIFF_STATUS_EXECUTION_FAILED: return "NVTIFF_STATUS_EXECUTION_FAILED";
        case NVTIFF_STATUS_ARCH_MISMATCH: return "NVTIFF_STATUS_ARCH_MISMATCH";
        case NVTIFF_STATUS_INTERNAL_ERROR: return "NVTIFF_STATUS_INTERNAL_ERROR";
        case NVTIFF_STATUS_NVCOMP_NOT_FOUND: return "NVTIFF_STATUS_NVCOMP_NOT_FOUND";
        case NVTIFF_STATUS_NVJPEG_NOT_FOUND: return "NVTIFF_STATUS_NVJPEG_NOT_FOUND";
        case NVTIFF_STATUS_TAG_NOT_FOUND: return "NVTIFF_STATUS_TAG_NOT_FOUND";
        case NVTIFF_STATUS_PARAMETER_OUT_OF_BOUNDS: return "NVTIFF_STATUS_PARAMETER_OUT_OF_BOUNDS";
        case NVTIFF_STATUS_NVJPEG2K_NOT_FOUND: return "NVTIFF_STATUS_NVJPEG2K_NOT_FOUND";
        case NVTIFF_STATUS_BATCH_INCOMPATIBLE: return "NVTIFF_STATUS_BATCH_INCOMPATIBLE";
    }
    return "unknown nvtiffStatus_t value";
}

inline const char* compression_string(nvtiffCompression_t compression)
{
    switch (compression) {
        case NVTIFF_COMPRESSION_NONE: return "uncompressed";
        case NVTIFF_COMPRESSION_LZW: return "LZW";
        case NVTIFF_COMPRESSION_JPEG: return "JPEG";
        case NVTIFF_COMPRESSION_ADOBE_DEFLATE: return "Adobe Deflate";
        case NVTIFF_COMPRESSION_PACKBITS: return "PackBits";
        case NVTIFF_COMPRESSION_DEFLATE: return "Deflate";
        case NVTIFF_COMPRESSION_APERIO_JP2000_YCC: return "Aperio JPEG2000 YCC";
        case NVTIFF_COMPRESSION_APERIO_JP2000_RGB: return "Aperio JPEG2000 RGB";
        case NVTIFF_COMPRESSION_JP2000: return "JPEG2000";
        case NVTIFF_COMPRESSION_UNKNOWN: return "unknown";
    }
    return "unknown";
}

inline const char* photometric_string(nvtiffPhotometricInt_t photometric)
{
    switch (photometric) {
        case NVTIFF_PHOTOMETRIC_MINISWHITE: return "grayscale (white is zero)";
        case NVTIFF_PHOTOMETRIC_MINISBLACK: return "grayscale";
        case NVTIFF_PHOTOMETRIC_RGB: return "RGB";
        case NVTIFF_PHOTOMETRIC_PALETTE: return "palette";
        case NVTIFF_PHOTOMETRIC_MASK: return "mask";
        case NVTIFF_PHOTOMETRIC_SEPARATED: return "separated (CMYK)";
        case NVTIFF_PHOTOMETRIC_YCBCR: return "YCbCr";
        case NVTIFF_PHOTOMETRIC_UNKNOWN: return "unknown";
    }
    return "unknown";
}

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error \"%s\" at %s:%d in\n  %s\n",         \
                cudaGetErrorString(err_), __FILE__, __LINE__, #call);             \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

#define CHECK_NVTIFF(call)                                                        \
    do {                                                                          \
        nvtiffStatus_t status_ = (call);                                          \
        if (status_ != NVTIFF_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "nvTIFF error %s at %s:%d in\n  %s\n",           \
                nvtiff_status_string(status_), __FILE__, __LINE__, #call);        \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

inline size_t div_up(size_t a, size_t b)
{
    return (a + b - 1) / b;
}

inline bool parse_positive_uint32(const char* text, uint32_t* value)
{
    const char* end = text;
    while (*end != '\0') {
        end++;
    }
    uint32_t parsed = 0;
    const auto result = std::from_chars(text, end, parsed);
    if (result.ec != std::errc() || result.ptr != end || parsed == 0) {
        return false;
    }
    *value = parsed;
    return true;
}

// How an image will be decoded: the requested nvTIFF output format plus the
// channel layout of the resulting (always interleaved) output buffer.
struct DecodeTarget
{
    nvtiffOutputFormat_t format;
    uint32_t channels;
    uint32_t bytes_per_channel;
};

// Picks a decode output format the PNM preview writer can consume where nvTIFF
// supports the conversion: palette images decode to 16-bit RGB, JPEG YCbCr to
// 8-bit RGB, everything else keeps its native layout.
inline DecodeTarget pick_decode_target(const nvtiffImageInfo_t& info)
{
    if (info.photometric_int == NVTIFF_PHOTOMETRIC_PALETTE) {
        return {NVTIFF_OUTPUT_RGB_I_UINT16, 3, 2};
    }
    if (info.photometric_int == NVTIFF_PHOTOMETRIC_YCBCR && info.compression == NVTIFF_COMPRESSION_JPEG) {
        return {NVTIFF_OUTPUT_RGB_I_UINT8, 3, 1};
    }
    const uint32_t bytes_per_channel = (uint32_t)div_up(info.bits_per_sample[0], 8);
    return {NVTIFF_OUTPUT_UNCHANGED_I, info.samples_per_pixel, bytes_per_channel};
}

// Dense row size in bytes for a decode of `width` pixels into `target`.
inline size_t decode_row_bytes(const nvtiffImageInfo_t& info, const DecodeTarget& target, uint32_t width)
{
    if (target.format == NVTIFF_OUTPUT_UNCHANGED_I) {
        return div_up((size_t)width * info.bits_per_pixel, 8);
    }
    return (size_t)width * target.channels * target.bytes_per_channel;
}

inline bool make_lzw_encode_info(
    const nvtiffImageInfo_t& input_info, const DecodeTarget& target, nvtiffImageInfo_t& encode_info)
{
    nvtiffImageInfo_t out = {};
    out.image_type = input_info.image_type;
    out.image_width = input_info.image_width;
    out.image_height = input_info.image_height;
    out.compression = NVTIFF_COMPRESSION_LZW;
    out.planar_config = NVTIFF_PLANARCONFIG_CONTIG;

    if (target.format != NVTIFF_OUTPUT_UNCHANGED_I) {
        if (target.channels != 3 || (target.bytes_per_channel != 1 && target.bytes_per_channel != 2)) {
            std::printf("  (cannot re-encode: converted decode output is not RGB8/RGB16)\n");
            return false;
        }
        out.photometric_int = NVTIFF_PHOTOMETRIC_RGB;
        out.samples_per_pixel = 3;
        for (uint32_t c = 0; c < out.samples_per_pixel; c++) {
            out.bits_per_sample[c] = (uint16_t)(8 * target.bytes_per_channel);
            out.sample_format[c] = NVTIFF_SAMPLEFORMAT_UINT;
            out.bits_per_pixel = (uint16_t)(out.bits_per_pixel + out.bits_per_sample[c]);
        }
        encode_info = out;
        return true;
    }

    if (input_info.samples_per_pixel == 1) {
        if (input_info.photometric_int != NVTIFF_PHOTOMETRIC_MINISBLACK &&
            input_info.photometric_int != NVTIFF_PHOTOMETRIC_UNKNOWN) {
            std::printf("  (cannot re-encode: LZW encode supports grayscale input only when 0 is black)\n");
            return false;
        }
        out.photometric_int = NVTIFF_PHOTOMETRIC_MINISBLACK;
    } else if (input_info.samples_per_pixel >= 3 && input_info.samples_per_pixel <= 5) {
        if (input_info.photometric_int != NVTIFF_PHOTOMETRIC_RGB &&
            input_info.photometric_int != NVTIFF_PHOTOMETRIC_UNKNOWN) {
            std::printf("  (cannot re-encode: LZW encode supports RGB photometric data for 3 or more samples)\n");
            return false;
        }
        out.photometric_int = NVTIFF_PHOTOMETRIC_RGB;
    } else {
        std::printf("  (cannot re-encode: LZW encode supports 1 channel, or 3 to 5 channels)\n");
        return false;
    }

    out.samples_per_pixel = input_info.samples_per_pixel;
    for (uint32_t c = 0; c < out.samples_per_pixel; c++) {
        const uint16_t bits = input_info.bits_per_sample[c];
        if (bits == 0 || bits % 8 != 0) {
            std::printf("  (cannot re-encode: LZW encode requires byte-aligned samples)\n");
            return false;
        }
        if (input_info.sample_format[c] != NVTIFF_SAMPLEFORMAT_UNKNOWN &&
            input_info.sample_format[c] != NVTIFF_SAMPLEFORMAT_UINT &&
            input_info.sample_format[c] != NVTIFF_SAMPLEFORMAT_IEEEFP) {
            std::printf("  (cannot re-encode: LZW encode does not support this sample format)\n");
            return false;
        }
        out.bits_per_sample[c] = bits;
        out.sample_format[c] = input_info.sample_format[c];
        out.bits_per_pixel = (uint16_t)(out.bits_per_pixel + bits);
    }

    encode_info = out;
    return true;
}

// Writes an interleaved 8- or 16-bit grayscale/RGB(A) buffer as PGM/PPM
// (MINISWHITE is inverted, alpha is dropped). Returns false for layouts this
// simple preview cannot represent.
inline bool write_pnm(const std::string& path, const uint8_t* data, size_t pitch_bytes, uint32_t width,
    uint32_t height, uint32_t channels, uint32_t bytes_per_channel, nvtiffPhotometricInt_t photometric,
    nvtiffSampleFormat_t sample_format)
{
    if (bytes_per_channel != 1 && bytes_per_channel != 2) {
        std::printf("  (no preview: %u byte samples are not representable as PNM)\n", bytes_per_channel);
        return false;
    }
    if (channels != 1 && channels != 3 && channels != 4) {
        std::printf("  (no preview: %u channel(s) are not representable as PNM)\n", channels);
        return false;
    }
    if (sample_format != NVTIFF_SAMPLEFORMAT_UINT) {
        std::printf("  (no preview: PNM previews preserve only unsigned integer samples)\n");
        return false;
    }
    if (channels == 1) {
        if (photometric != NVTIFF_PHOTOMETRIC_MINISBLACK && photometric != NVTIFF_PHOTOMETRIC_MINISWHITE &&
            photometric != NVTIFF_PHOTOMETRIC_MASK) {
            std::printf("  (no preview: unsupported grayscale photometric interpretation)\n");
            return false;
        }
    } else if (photometric != NVTIFF_PHOTOMETRIC_RGB) {
        std::printf("  (no preview: PNM color previews require RGB data)\n");
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "cannot open output file %s\n", path.c_str());
        return false;
    }
    const uint32_t out_channels = (channels == 1) ? 1 : 3;
    out << (out_channels == 1 ? "P5\n" : "P6\n") << width << " " << height << "\n"
        << ((bytes_per_channel == 1) ? 255 : 65535) << "\n";
    std::vector<uint8_t> row_buffer;
    if (bytes_per_channel == 1 && channels == out_channels && photometric != NVTIFF_PHOTOMETRIC_MINISWHITE) {
        for (uint32_t y = 0; y < height; y++) {
            const uint8_t* row = data + (size_t)y * pitch_bytes;
            out.write(reinterpret_cast<const char*>(row), (std::streamsize)((size_t)width * out_channels));
        }
        out.flush();
        return out.good();
    }
    row_buffer.resize((size_t)width * out_channels * bytes_per_channel);
    for (uint32_t y = 0; y < height; y++) {
        const uint8_t* row = data + (size_t)y * pitch_bytes;
        for (uint32_t x = 0; x < width; x++) {
            for (uint32_t c = 0; c < out_channels; c++) {
                if (bytes_per_channel == 1) {
                    uint8_t value = row[x * channels + c];
                    if (photometric == NVTIFF_PHOTOMETRIC_MINISWHITE) {
                        value = (uint8_t)(255 - value);
                    }
                    row_buffer[(size_t)x * out_channels + c] = value;
                } else {
                    // PNM stores 16-bit samples big-endian.
                    uint16_t value = reinterpret_cast<const uint16_t*>(row)[x * channels + c];
                    if (photometric == NVTIFF_PHOTOMETRIC_MINISWHITE) {
                        value = (uint16_t)(65535 - value);
                    }
                    const size_t out_offset = ((size_t)x * out_channels + c) * 2;
                    row_buffer[out_offset] = (uint8_t)(value >> 8);
                    row_buffer[out_offset + 1] = (uint8_t)(value & 0xff);
                }
            }
        }
        out.write(reinterpret_cast<const char*>(row_buffer.data()), (std::streamsize)row_buffer.size());
    }
    out.flush();
    return out.good();
}

// Copies a decoded plane back to the host and writes a PGM/PPM preview.
inline void write_decoded_preview(const std::string& path, const nvtiffImageInfo_t& info,
    const DecodeTarget& target, const void* plane_d, size_t pitch_bytes, uint32_t width, uint32_t height)
{
    if (target.format == NVTIFF_OUTPUT_UNCHANGED_I && info.bits_per_sample[0] % 8 != 0) {
        std::printf("  (no preview: %u bits per sample)\n", info.bits_per_sample[0]);
        return;
    }
    std::vector<uint8_t> host((size_t)height * pitch_bytes);
    CHECK_CUDA(cudaMemcpy(host.data(), plane_d, host.size(), cudaMemcpyDeviceToHost));
    const nvtiffPhotometricInt_t photometric =
        (target.format == NVTIFF_OUTPUT_UNCHANGED_I) ? info.photometric_int : NVTIFF_PHOTOMETRIC_RGB;
    const nvtiffSampleFormat_t sample_format =
        (target.format == NVTIFF_OUTPUT_UNCHANGED_I) ? info.sample_format[0] : NVTIFF_SAMPLEFORMAT_UINT;
    if (write_pnm(
            path, host.data(), pitch_bytes, width, height, target.channels, target.bytes_per_channel, photometric,
            sample_format)) {
        std::printf("  wrote preview %s\n", path.c_str());
    }
}

// "path/to/input.tif" -> "input"
inline std::string base_name_no_ext(const std::string& path)
{
    const size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const size_t dot = name.find_last_of('.');
    return (dot == std::string::npos) ? name : name.substr(0, dot);
}

#endif // NVTIFF_SAMPLES_COMMON_H
