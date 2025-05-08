/* Copyright 2020 NVIDIA Corporation.  All rights reserved.
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