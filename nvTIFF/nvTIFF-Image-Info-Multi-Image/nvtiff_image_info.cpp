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

// nvTIFF-Image-Info-Multi-Image
// -----------------------------
// Prints IFD metadata, tags, and SubIFDs.

#include "nvtiff_samples_common.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Prints the value of an ASCII tag when the image has it.
static void print_ascii_tag(nvtiffStream_t tiff_stream, size_t ifd_offset, uint16_t tag, const char* label)
{
    nvtiffTagDataType_t type;
    uint32_t size = 0;
    uint32_t count = 0;
    const nvtiffStatus_t status = nvtiffStreamGetTagInfo(tiff_stream, ifd_offset, tag, &type, &size, &count);
    if (status == NVTIFF_STATUS_TAG_NOT_FOUND) {
        return;
    }
    CHECK_NVTIFF(status);
    if (type != NVTIFF_TAG_TYPE_ASCII) {
        return;
    }
    std::vector<char> value(count + 1, '\0');
    CHECK_NVTIFF(nvtiffStreamGetTagValue(tiff_stream, ifd_offset, tag, value.data(), count));
    std::printf("  %-18s %s\n", label, value.data());
}

static std::vector<size_t> get_subifd_offsets(nvtiffStream_t tiff_stream, size_t ifd_offset)
{
    nvtiffTagDataType_t type;
    uint32_t size = 0;
    uint32_t count = 0;

    const uint16_t TIFF_TAG_SUBIFD = 330;
    const nvtiffStatus_t status =
        nvtiffStreamGetTagInfo(tiff_stream, ifd_offset, TIFF_TAG_SUBIFD, &type, &size, &count);
    if (status == NVTIFF_STATUS_TAG_NOT_FOUND) {
        return {};
    }
    CHECK_NVTIFF(status);

    std::vector<size_t> offsets;
    offsets.reserve(count);
    if (size == sizeof(uint32_t)) {
        std::vector<uint32_t> values(count);
        CHECK_NVTIFF(nvtiffStreamGetTagValue(tiff_stream, ifd_offset, TIFF_TAG_SUBIFD, values.data(), count));
        for (uint32_t value : values) {
            offsets.push_back(value);
        }
    } else if (size == sizeof(uint64_t)) {
        std::vector<uint64_t> values(count);
        CHECK_NVTIFF(nvtiffStreamGetTagValue(tiff_stream, ifd_offset, TIFF_TAG_SUBIFD, values.data(), count));
        for (uint64_t value : values) {
            offsets.push_back(static_cast<size_t>(value));
        }
    } else {
        std::printf("  SubIFDs: unsupported tag type\n");
    }
    return offsets;
}

static void print_info(nvtiffStream_t tiff_stream, size_t ifd_offset, const std::string& label)
{
    nvtiffImageInfo_t info;
    CHECK_NVTIFF(nvtiffStreamGetImageInfo(tiff_stream, ifd_offset, &info));
    nvtiffImageGeometry_t geometry;
    CHECK_NVTIFF(nvtiffStreamGetImageGeometry(tiff_stream, ifd_offset, &geometry));

    std::printf("\n%s (IFD offset %zu):\n", label.c_str(), ifd_offset);
    std::printf("  %u x %u, %u channel(s), %u bits per pixel, %s, %s\n", info.image_width, info.image_height,
        info.samples_per_pixel, info.bits_per_pixel, photometric_string(info.photometric_int),
        compression_string(info.compression));
    // NewSubfileType flags mark pyramid levels, pages and masks.
    if (info.image_type & NVTIFF_IMAGETYPE_REDUCED_IMAGE) {
        std::printf("  reduced-resolution image (pyramid level)\n");
    }
    if (info.image_type & NVTIFF_IMAGETYPE_PAGE) {
        std::printf("  page of a multi-page document\n");
    }
    if (info.image_type & NVTIFF_IMAGETYPE_MASK) {
        std::printf("  transparency mask\n");
    }
    const bool tiled = (geometry.type == NVTIFF_IMAGE_TILED);
    std::printf("  layout: %u %s of %u x %u\n", geometry.num_striles, tiled ? "tile(s)" : "strip(s)",
        geometry.strile_width, geometry.strile_height);

    // All TIFF tags present in this IFD (two-call pattern).
    uint32_t num_tags = 0;
    CHECK_NVTIFF(nvtiffStreamGetNumberOfTags(tiff_stream, ifd_offset, nullptr, &num_tags));
    std::vector<uint16_t> tags(num_tags);
    CHECK_NVTIFF(nvtiffStreamGetNumberOfTags(tiff_stream, ifd_offset, tags.data(), &num_tags));
    std::printf("  %u tag(s):", num_tags);
    for (uint32_t t = 0; t < num_tags; t++) {
        std::printf(" %u", tags[t]);
    }
    std::printf("\n");
    print_ascii_tag(tiff_stream, ifd_offset, 270, "ImageDescription:");
    print_ascii_tag(tiff_stream, ifd_offset, 305, "Software:");

    // Some images contain "SubIFDs" associated with each IFD, often reduced resolution images
    // Here we print them for the sake of completeness, but most images would not have SubIFDs.

    const std::vector<size_t> subifd_offsets = get_subifd_offsets(tiff_stream, ifd_offset);
    if (!subifd_offsets.empty()) {
        std::printf("  SubIFDs:");
        for (size_t offset : subifd_offsets) {
            std::printf(" %zu", offset);
        }
        std::printf("\n");
    }

    // NOTE: SubIFDs might also form their own chains; these chains can be explored with the same
    // `nvtiffStreamGetNextIFDOffset` machinery as the root IFD chain.
    for (size_t i = 0; i < subifd_offsets.size(); i++) {
        size_t subifd_offset = subifd_offsets[i];
        print_info(tiff_stream, subifd_offset, label + " SubIFD " + std::to_string(i));
    }
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::printf("Usage: %s <input.tif>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string input_file = argv[1];

    nvtiffStream_t tiff_stream;
    CHECK_NVTIFF(nvtiffStreamOpenFromFile(input_file.c_str(), &tiff_stream));

    nvtiffStreamHeader_t header;
    CHECK_NVTIFF(nvtiffStreamGetHeader(tiff_stream, &header));
    std::vector<size_t> ifd_offsets;
    for (size_t offset = header.first_ifd_offset; offset != NVTIFF_NO_IMAGE;) {
        ifd_offsets.push_back(offset);
        CHECK_NVTIFF(nvtiffStreamGetNextIFDOffset(tiff_stream, offset, &offset));
    }
    std::printf("%s: %s, %zu image(s)\n", input_file.c_str(),
        header.variant == NVTIFF_BIG_TIFF ? "BigTIFF" : "TIFF", ifd_offsets.size());

    for (size_t i = 0; i < ifd_offsets.size(); i++) {
        print_info(tiff_stream, ifd_offsets[i], "image " + std::to_string(i));
    }

    CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));
    return EXIT_SUCCESS;
}
