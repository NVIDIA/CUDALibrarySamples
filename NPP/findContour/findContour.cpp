/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <fstream>

#include <npp.h>



#ifdef _WIN32

#define WINDOWS_LEAN_AND_MEAN

#define NOMINMAX

#include <windows.h>

#pragma warning(disable:4819)

#endif



// Input/output file paths

const std::string INPUT_PATH = "images/CircuitBoard_2048x1024_8u.raw";

const std::string OUTPUT_LABELS = "images/CircuitBoard_LabelMarkersUF_8Way_2048x1024_32u.raw";

const std::string OUTPUT_COMPRESSED = "images/CircuitBoard_CompressedMarkerLabelsUF_8Way_2048x1024_32u.raw";

const std::string OUTPUT_CONTOURS = "images/CircuitBoard_Contours_8Way_2048x1024_8u.raw";

const std::string OUTPUT_GEOMETRY = "images/CircuitBoard_ContoursReconstructed_8Way_2048x1024_8u.raw";



// Dimensions

constexpr int WIDTH = 2048;

constexpr int HEIGHT = 1024;



// Helper to load raw 8-bit image

int loadImage(Npp8u* dst, const std::string& path, int width, int height) {

    FILE* fp = fopen(path.c_str(), "rb");

    if (!fp) return -1;

    size_t read = fread(dst, 1, width * height, fp);

    fclose(fp);

    return (read == width * height) ? 0 : -1;

}



int main() {

    NppiSize roi = { WIDTH, HEIGHT };

    NppStreamContext ctx = {};

    cudaGetDevice(&ctx.nCudaDeviceId);

    cudaStreamCreate(&ctx.hStream);



    // Allocate device and host memory

    Npp8u* d_input = nullptr; cudaMalloc(&d_input, WIDTH * HEIGHT);

    Npp32u* d_labels = nullptr; cudaMalloc(&d_labels, WIDTH * HEIGHT * sizeof(Npp32u));

    Npp8u* h_input = (Npp8u*)malloc(WIDTH * HEIGHT);

    Npp32u* h_labels = (Npp32u*)malloc(WIDTH * HEIGHT * sizeof(Npp32u));



    if (loadImage(h_input, INPUT_PATH, WIDTH, HEIGHT) != 0) return -1;

    cudaMemcpy2DAsync(d_input, WIDTH, h_input, WIDTH, WIDTH, HEIGHT, cudaMemcpyHostToDevice, ctx.hStream);



    // Labeling

    int scratchSize = 0;

    nppiLabelMarkersUFGetBufferSize_32u_C1R(roi, &scratchSize);

    Npp8u* d_scratch = nullptr; cudaMalloc(&d_scratch, scratchSize);



    nppiLabelMarkersUF_8u32u_C1R_Ctx(d_input, WIDTH, d_labels, WIDTH * sizeof(Npp32u), roi, nppiNormInf, d_scratch, ctx);

    cudaMemcpy2DAsync(h_labels, WIDTH * sizeof(Npp32u), d_labels, WIDTH * sizeof(Npp32u), WIDTH * sizeof(Npp32u), HEIGHT, cudaMemcpyDeviceToHost, ctx.hStream);

    cudaStreamSynchronize(ctx.hStream);



    FILE* fp = fopen(OUTPUT_LABELS.c_str(), "wb");

    fwrite(h_labels, sizeof(Npp32u), WIDTH * HEIGHT, fp); fclose(fp);



    // Label compression

    int compScratchSize = 0;

    nppiCompressMarkerLabelsGetBufferSize_32u_C1R(WIDTH * HEIGHT, &compScratchSize);

    Npp8u* d_compScratch = nullptr; cudaMalloc(&d_compScratch, compScratchSize);

    int compressedCount = 0;

    nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(d_labels, WIDTH * sizeof(Npp32u), roi, WIDTH * HEIGHT, &compressedCount, d_compScratch, ctx);

    cudaMemcpy2DAsync(h_labels, WIDTH * sizeof(Npp32u), d_labels, WIDTH * sizeof(Npp32u), WIDTH * sizeof(Npp32u), HEIGHT, cudaMemcpyDeviceToHost, ctx.hStream);

    cudaStreamSynchronize(ctx.hStream);



    fp = fopen(OUTPUT_COMPRESSED.c_str(), "wb");

    fwrite(h_labels, sizeof(Npp32u), WIDTH * HEIGHT, fp); fclose(fp);



    // Contour extraction

    Npp8u* d_contours = nullptr; cudaMalloc(&d_contours, WIDTH * HEIGHT);

    NppiContourPixelDirectionInfo* d_directions = nullptr; cudaMalloc(&d_directions, WIDTH * HEIGHT * sizeof(NppiContourPixelDirectionInfo));

    Npp32u* d_counts = nullptr; cudaMalloc(&d_counts, (compressedCount + 4) * sizeof(Npp32u));

    Npp32u* d_found = nullptr; cudaMalloc(&d_found, (compressedCount + 4) * sizeof(Npp32u));

    Npp32u* d_offsets = nullptr; cudaMalloc(&d_offsets, (compressedCount + 4) * sizeof(Npp32u));

    NppiCompressedMarkerLabelsInfo* d_info = nullptr; cudaMalloc(&d_info, compressedCount * sizeof(NppiCompressedMarkerLabelsInfo));



    nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(d_labels, WIDTH * sizeof(Npp32u), roi, compressedCount, d_info, d_contours, WIDTH, d_directions, WIDTH * sizeof(NppiContourPixelDirectionInfo), nullptr, d_counts, nullptr, d_offsets, nullptr, ctx); 



    // Copy and write contours

    Npp8u* h_contours = (Npp8u*)malloc(WIDTH * HEIGHT);

    cudaMemcpy(h_contours, d_contours, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    fp = fopen(OUTPUT_CONTOURS.c_str(), "wb");

    fwrite(h_contours, 1, WIDTH * HEIGHT, fp); fclose(fp);



    // Clean up

    cudaFree(d_input); cudaFree(d_labels); cudaFree(d_scratch);

    cudaFree(d_compScratch); cudaFree(d_contours); cudaFree(d_directions);

    cudaFree(d_counts); cudaFree(d_found); cudaFree(d_offsets); cudaFree(d_info);

    free(h_input); free(h_labels); free(h_contours);

    cudaStreamDestroy(ctx.hStream);



    printf("Done. Compressed Labels: %d\n", compressedCount);

    return 0;

}