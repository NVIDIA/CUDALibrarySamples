/* Copyright 2021 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
* 
* NOTICE TO LICENSEE: 
* 
* The source code and/or documentation ("Licensed Deliverables") are 
* subject to NVIDIA intellectual property rights under U.S. and 
* international Copyright laws. 
* 
* The Licensed Deliverables contained herein are PROPRIETARY and 
* CONFIDENTIAL to NVIDIA and are being provided under the terms and 
* conditions of a form of NVIDIA software license agreement by and 
* between NVIDIA and Licensee ("License Agreement") or electronically 
* accepted by Licensee.  Notwithstanding any terms or conditions to 
* the contrary in the License Agreement, reproduction or disclosure 
* of the Licensed Deliverables to any third party without the express 
* written consent of NVIDIA is prohibited. 
* 
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE 
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE 
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND. 
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED 
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, 
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE. 
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY 
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY 
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, 
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS 
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE 
* OF THESE LICENSED DELIVERABLES. 
* 
* U.S. Government End Users.  These Licensed Deliverables are a 
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 
* 1995), consisting of "commercial computer software" and "commercial 
* computer software documentation" as such terms are used in 48 
* C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government 
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and 
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all 
* U.S. Government End Users acquire the Licensed Deliverables with 
* only those rights set forth herein. 
* 
* Any use of the Licensed Deliverables in individual and commercial 
* software must include, in the user documentation and internal 
* comments to the code, the above Disclaimer and U.S. Government End 
* Users Notice. 
*
* This sample demonstrates usage of NPP library APIs for GPU-accelerated 
* image labeling, compression, and contour geometry extraction using 
* Union-Find (UF) methods. The example processes grayscale images and 
* outputs various labeled representations for further image analysis.
*
*
* Requirements:
*   - NVIDIA GPU with CUDA support
*   - CUDA Toolkit (tested with >= 11.5)
*   - NPP Library with UF and contour support
*
* Output:
*   - Labeled and compressed marker images
*   - Contour geometry and direction outputs
*   - Binary image files for visualization with RAW viewers like ImageJ
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
    cudaGetDeviceProperties(&ctx.nMultiProcessorCount, ctx.nCudaDeviceId);

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
