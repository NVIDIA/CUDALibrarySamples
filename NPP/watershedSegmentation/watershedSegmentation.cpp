/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "watershedSegmentation.h"
#include <npp.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#define NUM_IMAGES 5

Npp8u *d_input[NUM_IMAGES], *h_input[NUM_IMAGES];
Npp8u *d_scratch[NUM_IMAGES], *d_segments[NUM_IMAGES], *h_segments[NUM_IMAGES];
Npp32u *d_labels[NUM_IMAGES], *h_labels[NUM_IMAGES];

const std::string base_path = "../images/";
const std::string input_files[] = {
    "Lena_512x512_8u_Gray.raw",
    "CT_skull_512x512_8u_Gray.raw",
    "Rocks_512x512_8u_Gray.raw",
    "coins_500x383_8u_Gray.raw",
    "coins_overlay_500x569_8u_Gray.raw"
};

void cleanup() {
    for (int i = 0; i < NUM_IMAGES; ++i) {
        cudaFree(d_input[i]);
        cudaFree(d_scratch[i]);
        cudaFree(d_segments[i]);
        cudaFree(d_labels[i]);
        free(h_input[i]);
        free(h_segments[i]);
        free(h_labels[i]);
    }
}

bool loadRaw(const std::string &file, Npp8u *buffer, int width, int height) {
    FILE *fp = fopen((base_path + file).c_str(), "rb");
    if (!fp) return false;
    size_t size = fread(buffer, 1, width * height, fp);
    fclose(fp);
    return size == width * height;
}

int main(int argc, char **argv) {
    NppStreamContext ctx = {};
    cudaGetDevice(&ctx.nCudaDeviceId);
    cudaStreamCreate(&ctx.hStream);

    NppiSize rois[NUM_IMAGES] = {{512,512}, {512,512}, {512,512}, {500,383}, {500,569}};

    for (int i = 0; i < NUM_IMAGES; ++i) {
        int size = rois[i].width * rois[i].height;

        cudaMalloc(&d_input[i], size);
        cudaMalloc(&d_segments[i], size);
        cudaMalloc(&d_labels[i], size * sizeof(Npp32u));

        h_input[i] = (Npp8u*)malloc(size);
        h_segments[i] = (Npp8u*)malloc(size);
        h_labels[i] = (Npp32u*)malloc(size * sizeof(Npp32u));

        if (!loadRaw(input_files[i], h_input[i], rois[i].width, rois[i].height)) {
            std::cerr << "Failed to load " << input_files[i] << "\n";
            cleanup();
            return -1;
        }

        cudaMemcpy2D(d_input[i], rois[i].width, h_input[i], rois[i].width,
                     rois[i].width, rois[i].height, cudaMemcpyHostToDevice);
        cudaMemcpy(d_segments[i], d_input[i], size, cudaMemcpyDeviceToDevice);

        size_t scratch_size;
        nppiSegmentWatershedGetBufferSize_8u_C1R(rois[i], &scratch_size);
        cudaMalloc(&d_scratch[i], scratch_size);

        nppiSegmentWatershed_8u_C1IR_Ctx(d_segments[i], rois[i].width,
                                         d_labels[i], rois[i].width * sizeof(Npp32u),
                                         nppiNormInf, NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE,
                                         rois[i], d_scratch[i], ctx);

        cudaMemcpy(h_segments[i], d_segments[i], size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_labels[i], d_labels[i], size * sizeof(Npp32u), cudaMemcpyDeviceToHost);

        std::string out_seg = base_path + "Segmented_" + input_files[i];
        FILE *fseg = fopen(out_seg.c_str(), "wb");
        fwrite(h_segments[i], 1, size, fseg); fclose(fseg);

        std::string out_label = base_path + "Labels_" + input_files[i];
        FILE *flab = fopen(out_label.c_str(), "wb");
        fwrite(h_labels[i], sizeof(Npp32u), size, flab); fclose(flab);

        std::cout << "Processed " << input_files[i] << "\n";
    }

    cleanup();
    cudaStreamDestroy(ctx.hStream);
    return 0;
}