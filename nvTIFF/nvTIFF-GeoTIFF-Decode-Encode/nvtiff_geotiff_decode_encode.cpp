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

// nvTIFF-GeoTIFF-Decode-Encode
// ----------------------------
// Preserves GeoTIFF metadata through a decode-encode roundtrip.

#include "nvtiff_samples_common.h"

#include <cstring>

// One geo key read from the input, kept for re-encoding.
struct GeoKeyValue
{
    nvtiffGeoKey_t key;
    nvtiffGeoKeyDataType_t type;
    uint32_t count;
    std::vector<unsigned short> shorts;
    std::vector<double> doubles;
    std::string ascii;
};

static constexpr uint16_t NVTIFF_TAG_RPC_COEFFICIENT = 50844;

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::printf("Usage: %s <input_geotiff.tif> <output.tif>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string input_file = argv[1];
    const std::string output_file = argv[2];

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    nvtiffStream_t tiff_stream;
    CHECK_NVTIFF(nvtiffStreamOpenFromFile(input_file.c_str(), &tiff_stream));

    // Read the geo keys (two-call pattern: count, then key IDs). Some
    // georeferenced TIFFs use plain TIFF tags such as RPC coefficients without
    // a GeoKeyDirectory, so zero geo keys is not an error by itself.
    uint32_t num_keys = 0;
    const nvtiffStatus_t geo_status = nvtiffStreamGetNumberOfGeoKeys(tiff_stream, nullptr, &num_keys);
    if (geo_status != NVTIFF_STATUS_SUCCESS && geo_status != NVTIFF_STATUS_TAG_NOT_FOUND &&
        geo_status != NVTIFF_STATUS_TIFF_NOT_SUPPORTED) {
        std::fprintf(stderr, "%s: failed to read GeoTIFF keys (%s)\n", input_file.c_str(),
            nvtiff_status_string(geo_status));
        CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));
        CHECK_CUDA(cudaStreamDestroy(stream));
        return EXIT_FAILURE;
    }

    std::vector<GeoKeyValue> geo_keys;
    if (geo_status == NVTIFF_STATUS_SUCCESS && num_keys > 0) {
        std::vector<nvtiffGeoKey_t> key_ids(num_keys);
        CHECK_NVTIFF(nvtiffStreamGetNumberOfGeoKeys(tiff_stream, key_ids.data(), &num_keys));
        std::printf("%s: %u geo key(s)\n", input_file.c_str(), num_keys);

        for (uint32_t i = 0; i < num_keys; i++) {
            GeoKeyValue value = {};
            value.key = key_ids[i];
            uint32_t size = 0;
            CHECK_NVTIFF(nvtiffStreamGetGeoKeyInfo(tiff_stream, value.key, &size, &value.count, &value.type));
            switch (value.type) {
                case NVTIFF_GEOKEY_TYPE_SHORT: {
                    value.shorts.resize(value.count);
                    CHECK_NVTIFF(
                        nvtiffStreamGetGeoKeySHORT(tiff_stream, value.key, value.shorts.data(), 0, value.count));
                    std::printf("  key %u (SHORT):", (unsigned)value.key);
                    for (unsigned short v : value.shorts) {
                        std::printf(" %u", v);
                    }
                    std::printf("\n");
                    break;
                }
                case NVTIFF_GEOKEY_TYPE_DOUBLE: {
                    value.doubles.resize(value.count);
                    CHECK_NVTIFF(
                        nvtiffStreamGetGeoKeyDOUBLE(tiff_stream, value.key, value.doubles.data(), 0, value.count));
                    std::printf("  key %u (DOUBLE):", (unsigned)value.key);
                    for (double v : value.doubles) {
                        std::printf(" %g", v);
                    }
                    std::printf("\n");
                    break;
                }
                case NVTIFF_GEOKEY_TYPE_ASCII: {
                    value.ascii.assign((size_t)size * value.count + 1, '\0');
                    CHECK_NVTIFF(nvtiffStreamGetGeoKeyASCII(
                        tiff_stream, value.key, &value.ascii[0], (uint32_t)value.ascii.size()));
                    value.ascii.resize(std::strlen(value.ascii.c_str()));
                    std::printf("  key %u (ASCII): %s\n", (unsigned)value.key, value.ascii.c_str());
                    break;
                }
                default:
                    std::printf("  key %u: unknown type, skipped\n", (unsigned)value.key);
                    continue;
            }
            geo_keys.push_back(value);
        }
    } else {
        std::printf("%s: 0 geo key(s)\n", input_file.c_str());
    }

    // Read the raster georeferencing tags (plain TIFF tags of DOUBLEs).
    const auto read_double_tag = [&](uint16_t tag, const char* label) {
        std::vector<double> values;
        nvtiffTagDataType_t type;
        uint32_t size = 0;
        uint32_t count = 0;
        const nvtiffStatus_t status = nvtiffStreamGetTagInfo(tiff_stream, 0, tag, &type, &size, &count);
        if (status == NVTIFF_STATUS_TAG_NOT_FOUND) {
            return values;
        }
        CHECK_NVTIFF(status);
        if (type != NVTIFF_TAG_TYPE_DOUBLE) {
            std::printf("  %s: non-DOUBLE tag skipped\n", label);
            return values;
        }
        values.resize(count);
        CHECK_NVTIFF(nvtiffStreamGetTagValue(tiff_stream, 0, tag, values.data(), count));
        std::printf("  %s:", label);
        for (double v : values) {
            std::printf(" %g", v);
        }
        std::printf("\n");
        return values;
    };

    const std::vector<double> pixel_scale = read_double_tag(NVTIFF_TAG_MODEL_PIXEL_SCALE, "ModelPixelScale");
    const std::vector<double> tie_points = read_double_tag(NVTIFF_TAG_MODEL_TIE_POINT, "ModelTiePoint");
    const std::vector<double> transform = read_double_tag(NVTIFF_TAG_MODEL_TRANSFORMATION, "ModelTransformation");
    const std::vector<double> rpc_coefficients = read_double_tag(NVTIFF_TAG_RPC_COEFFICIENT, "RPCCoefficient");

    if (geo_keys.empty() && pixel_scale.empty() && tie_points.empty() && transform.empty() &&
        rpc_coefficients.empty()) {
        std::fprintf(stderr, "%s has no supported GeoTIFF georeferencing metadata\n", input_file.c_str());
        CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));
        CHECK_CUDA(cudaStreamDestroy(stream));
        return EXIT_FAILURE;
    }

    // Decode the raster (same minimal path as nvTIFF-Decode-Encode).
    nvtiffImageInfo_t info;
    CHECK_NVTIFF(nvtiffStreamGetImageInfo(tiff_stream, 0, &info));
    std::printf("raster: %u x %u, %u channel(s), %s, %s\n", info.image_width, info.image_height,
        info.samples_per_pixel, photometric_string(info.photometric_int), compression_string(info.compression));

    const DecodeTarget target = pick_decode_target(info);
    nvtiffImageInfo_t encode_info = {};
    if (!make_lzw_encode_info(info, target, encode_info)) {
        CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));
        CHECK_CUDA(cudaStreamDestroy(stream));
        return EXIT_FAILURE;
    }

    nvtiffDecoder_t decoder;
    CHECK_NVTIFF(nvtiffDecoderCreateSimple(&decoder, stream));
    nvtiffDecodeParams_t decode_params;
    CHECK_NVTIFF(nvtiffDecodeParamsCreate(&decode_params));
    CHECK_NVTIFF(nvtiffDecodeParamsSetOutputFormat(decode_params, target.format));
    size_t pitch = decode_row_bytes(info, target, info.image_width);
    void* pixels_d = nullptr;
    CHECK_CUDA(cudaMalloc(&pixels_d, pitch * info.image_height));
    nvtiffImage_t image = {};
    image.plane_data = &pixels_d;
    image.plane_pitch_bytes = &pitch;
    image.num_planes = 1;
    CHECK_NVTIFF(nvtiffDecode(tiff_stream, decoder, decode_params, &image, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Encode a new LZW GeoTIFF with georeferencing re-attached.
    nvtiffEncoder_t encoder;
    CHECK_NVTIFF(nvtiffEncoderCreate(&encoder, nullptr, nullptr, stream));
    nvtiffEncodeParams_t encode_params;
    CHECK_NVTIFF(nvtiffEncodeParamsCreate(&encode_params));
    CHECK_NVTIFF(nvtiffEncodeParamsSetImageInfo(encode_params, &encode_info));
    uint8_t* encode_inputs[1] = {static_cast<uint8_t*>(pixels_d)};
    CHECK_NVTIFF(nvtiffEncodeParamsSetInputs(encode_params, encode_inputs, 1));

    // Geo keys: scalar SHORT/DOUBLE keys use the typed setters, ASCII keys
    // the string setter, multi-value keys the generic nvtiffEncodeParamsSetGeoKey.
    for (const GeoKeyValue& value : geo_keys) {
        switch (value.type) {
            case NVTIFF_GEOKEY_TYPE_SHORT:
                if (value.count == 1) {
                    CHECK_NVTIFF(nvtiffEncodeParamsSetGeoKeySHORT(encode_params, value.key, value.shorts[0], 1));
                } else {
                    CHECK_NVTIFF(nvtiffEncodeParamsSetGeoKey(encode_params, value.key, value.type,
                        (void*)value.shorts.data(), sizeof(unsigned short), value.count));
                }
                break;
            case NVTIFF_GEOKEY_TYPE_DOUBLE:
                if (value.count == 1) {
                    CHECK_NVTIFF(nvtiffEncodeParamsSetGeoKeyDOUBLE(encode_params, value.key, value.doubles[0], 1));
                } else {
                    CHECK_NVTIFF(nvtiffEncodeParamsSetGeoKey(encode_params, value.key, value.type,
                        (void*)value.doubles.data(), sizeof(double), value.count));
                }
                break;
            default: // ASCII
                CHECK_NVTIFF(nvtiffEncodeParamsSetGeoKeyASCII(
                    encode_params, value.key, value.ascii.c_str(), (uint32_t)value.ascii.size() + 1));
                break;
        }
    }
    // Raster georeferencing tags travel as plain TIFF tags.
    if (!pixel_scale.empty()) {
        CHECK_NVTIFF(nvtiffEncodeParamsSetTag(encode_params, NVTIFF_TAG_MODEL_PIXEL_SCALE, NVTIFF_TAG_TYPE_DOUBLE,
            (void*)pixel_scale.data(), (uint32_t)pixel_scale.size()));
    }
    if (!tie_points.empty()) {
        CHECK_NVTIFF(nvtiffEncodeParamsSetTag(encode_params, NVTIFF_TAG_MODEL_TIE_POINT, NVTIFF_TAG_TYPE_DOUBLE,
            (void*)tie_points.data(), (uint32_t)tie_points.size()));
    }
    if (!transform.empty()) {
        CHECK_NVTIFF(nvtiffEncodeParamsSetTag(encode_params, NVTIFF_TAG_MODEL_TRANSFORMATION,
            NVTIFF_TAG_TYPE_DOUBLE, (void*)transform.data(), (uint32_t)transform.size()));
    }
    if (!rpc_coefficients.empty()) {
        CHECK_NVTIFF(nvtiffEncodeParamsSetTag(encode_params, NVTIFF_TAG_RPC_COEFFICIENT, NVTIFF_TAG_TYPE_DOUBLE,
            (void*)rpc_coefficients.data(), (uint32_t)rpc_coefficients.size()));
    }

    CHECK_NVTIFF(nvtiffEncode(encoder, &encode_params, 1, stream));
    CHECK_NVTIFF(nvtiffEncodeFinalize(encoder, &encode_params, 1, stream));
    CHECK_NVTIFF(nvtiffWriteTiffFile(encoder, &encode_params, 1, output_file.c_str(), stream));
    std::printf("wrote %s with %zu geo key(s)%s preserved\n", output_file.c_str(), geo_keys.size(),
        rpc_coefficients.empty() ? "" : " and RPC tag");

    // Roundtrip check: the file we just wrote must report the same
    // number of geo keys.
    nvtiffStream_t check_stream;
    CHECK_NVTIFF(nvtiffStreamOpenFromFile(output_file.c_str(), &check_stream));
    uint32_t check_keys = 0;
    const nvtiffStatus_t check_geo_status = nvtiffStreamGetNumberOfGeoKeys(check_stream, nullptr, &check_keys);
    if (!geo_keys.empty()) {
        CHECK_NVTIFF(check_geo_status);
        if (check_keys != geo_keys.size()) {
            std::fprintf(stderr, "roundtrip check failed: output has %u geo key(s), expected %zu\n", check_keys,
                geo_keys.size());
            CHECK_NVTIFF(nvtiffStreamClose(check_stream));
            CHECK_NVTIFF(nvtiffEncodeParamsDestroy(encode_params, stream));
            CHECK_NVTIFF(nvtiffEncoderDestroy(encoder, stream));
            CHECK_NVTIFF(nvtiffDecodeParamsDestroy(decode_params));
            CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
            CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));
            CHECK_CUDA(cudaFree(pixels_d));
            CHECK_CUDA(cudaStreamDestroy(stream));
            return EXIT_FAILURE;
        }
    } else {
        if (check_geo_status != NVTIFF_STATUS_SUCCESS && check_geo_status != NVTIFF_STATUS_TAG_NOT_FOUND &&
            check_geo_status != NVTIFF_STATUS_TIFF_NOT_SUPPORTED) {
            CHECK_NVTIFF(check_geo_status);
        }
    }
    std::printf("roundtrip check: output has %u geo key(s)\n", check_keys);
    if (!rpc_coefficients.empty()) {
        nvtiffTagDataType_t check_type;
        uint32_t check_size = 0;
        uint32_t check_count = 0;
        CHECK_NVTIFF(nvtiffStreamGetTagInfo(
            check_stream, 0, NVTIFF_TAG_RPC_COEFFICIENT, &check_type, &check_size, &check_count));
        std::printf("roundtrip check: output has RPC tag with %u value(s)\n", check_count);
    }
    CHECK_NVTIFF(nvtiffStreamClose(check_stream));

    CHECK_NVTIFF(nvtiffEncodeParamsDestroy(encode_params, stream));
    CHECK_NVTIFF(nvtiffEncoderDestroy(encoder, stream));
    CHECK_NVTIFF(nvtiffDecodeParamsDestroy(decode_params));
    CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
    CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));
    CHECK_CUDA(cudaFree(pixels_d));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
