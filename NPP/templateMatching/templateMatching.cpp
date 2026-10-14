/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Template matching with normalized cross-correlation (NCC) in NPP.
//
// A template is cut out of a circuit-board image and deliberately degraded (gain, offset and noise), then located
// again in the full image:
//
//     nppiCrossCorrValid_NormLevel_8u32f_C1R_Ctx   NCC score for every template position
//     nppiMaxIndx_32f_C1R_Ctx                      position of the best score, found on the GPU
//
// The "NormLevel" score is the zero-mean normalized correlation coefficient (Pearson), in [-1, 1].  Because the mean and
// the energy of both patches are removed, the score does not change if the template has a different brightness (offset)
// or contrast (gain) than the image, which makes it robust to illumination changes in inspection setups.
//
// The sample checks the GPU scores against a CPU computation, reports how clearly the best match stands out from the
// next best one, writes an annotated copy of the image, and times the GPU search.

#include <cuda_runtime.h>
#include <npp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                                      \
    do {                                                                                      \
        cudaError_t e_ = (call);                                                              \
        if (e_ != cudaSuccess) {                                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    } while (0)

#define CHECK_NPP(call)                                                                       \
    do {                                                                                      \
        NppStatus s_ = (call);                                                                \
        if (s_ != NPP_SUCCESS) {                                                              \
            fprintf(stderr, "NPP error %d at %s:%d\n", (int)s_, __FILE__, __LINE__);          \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    } while (0)

static const int kImageWidth = 2048, kImageHeight = 1024;

static NppStreamContext makeStreamContext(cudaStream_t stream)
{
    NppStreamContext ctx = {}; // zero-initialize: hWorkspace (formerly nReserved0) must be 0 unless a workspace is created
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    ctx.hStream                            = stream;
    ctx.nCudaDeviceId                      = dev;
    ctx.nMultiProcessorCount               = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor       = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock                = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock                 = prop.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    CHECK_CUDA(cudaStreamGetFlags(stream, &ctx.nStreamFlags));
    return ctx;
}

// Zero-mean normalized cross-correlation of the template against the image patch at (px, py).
static double cpuNCC(const std::vector<Npp8u>& img, int px, int py, const std::vector<Npp8u>& tpl, int tw, int th)
{
    double sa = 0, sb = 0;
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x) {
            sa += img[(size_t)(py + y) * kImageWidth + px + x];
            sb += tpl[(size_t)y * tw + x];
        }
    const double ma = sa / (tw * th), mb = sb / (tw * th);
    double sab = 0, saa = 0, sbb = 0;
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x) {
            const double a = img[(size_t)(py + y) * kImageWidth + px + x] - ma, b = tpl[(size_t)y * tw + x] - mb;
            sab += a * b; saa += a * a; sbb += b * b;
        }
    return sab / std::sqrt(saa * sbb);
}

int main(int argc, char** argv)
{
    std::string imagePath = "../images/CircuitBoard_2048x1024_8u.raw";
    int    tx = 1300, ty = 850, tw = 96, th = 64, iterations = 50;
    double gain = 0.6, offset = 30.0;
    int    noise = 3;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : (char*)"0"; };
        if (a == "-x") tx = atoi(next());
        else if (a == "-y") ty = atoi(next());
        else if (a == "-tw") tw = atoi(next());
        else if (a == "-th") th = atoi(next());
        else if (a == "-g") gain = atof(next());
        else if (a == "-o") offset = atof(next());
        else if (a == "-noise") noise = atoi(next());
        else if (a == "-n") iterations = atoi(next());
        else if (a == "-i") imagePath = next();
        else {
            printf("Usage: %s [-x tpl-x] [-y tpl-y] [-tw tpl-width] [-th tpl-height] [-g gain] [-o offset] [-noise amplitude] "
                   "[-n timing-iterations] [-i image.raw]\n"
                   "  The template is cut from the image at (x,y), then degraded: pixel = gain*pixel + offset + noise.\n"
                   "  The image must be %dx%d 8-bit gray RAW.\n", argv[0], kImageWidth, kImageHeight);
            return 0;
        }
    }
    // Compare with subtraction so that huge values cannot overflow int and slip past the bounds check.
    if (tx < 0 || ty < 0 || tw < 8 || th < 8 || tw > kImageWidth || th > kImageHeight || tx > kImageWidth - tw ||
        ty > kImageHeight - th) {
        fprintf(stderr, "Template must lie inside the %dx%d image and be at least 8x8\n", kImageWidth, kImageHeight);
        return EXIT_FAILURE;
    }
    if (noise < 0 || noise > 255) {
        fprintf(stderr, "-noise must be between 0 and 255\n");
        return EXIT_FAILURE;
    }
    if (iterations < 0) {
        fprintf(stderr, "-n must not be negative\n");
        return EXIT_FAILURE;
    }

    const int W = kImageWidth, H = kImageHeight;
    std::vector<Npp8u> image((size_t)W * H);
    {
        std::ifstream f(imagePath, std::ios::binary);
        if (!f || !f.read(reinterpret_cast<char*>(image.data()), image.size())) {
            fprintf(stderr, "Cannot read %zu bytes from %s (run from the build directory)\n", image.size(), imagePath.c_str());
            return EXIT_FAILURE;
        }
    }

    // Template: crop, then change its brightness/contrast and add noise.
    srand(1234);
    std::vector<Npp8u> tpl((size_t)tw * th);
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x) {
            double v = gain * image[(size_t)(ty + y) * W + tx + x] + offset;
            if (noise > 0) v += (rand() % (2 * noise + 1)) - noise;
            tpl[(size_t)y * tw + x] = (Npp8u)std::min(255.0, std::max(0.0, v + 0.5));
        }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    NppStreamContext ctx = makeStreamContext(stream);

    const NppiSize srcSize = {W, H}, tplSize = {tw, th};
    const int      outW = W - tw + 1, outH = H - th + 1; // "Valid" mode: template fully inside the image
    const size_t   outBytes = (size_t)outW * outH * sizeof(Npp32f);

    Npp8u *d_src, *d_tpl, *d_corrBuf, *d_maxBuf;
    Npp32f *d_score, *d_max;
    int *d_maxX, *d_maxY;
    size_t corrBufSize = 0, maxBufSize = 0;
    CHECK_NPP(nppiValidNormLevelGetBufferHostSize_8u32f_C1R_Ctx(srcSize, &corrBufSize, ctx));
    CHECK_NPP(nppiMaxIndxGetBufferHostSize_32f_C1R_Ctx({outW, outH}, &maxBufSize, ctx));
    CHECK_CUDA(cudaMalloc(&d_src, image.size()));
    CHECK_CUDA(cudaMalloc(&d_tpl, tpl.size()));
    CHECK_CUDA(cudaMalloc(&d_score, outBytes));
    CHECK_CUDA(cudaMalloc(&d_corrBuf, std::max<size_t>(corrBufSize, 1))); // never pass NPP a null buffer
    CHECK_CUDA(cudaMalloc(&d_maxBuf, std::max<size_t>(maxBufSize, 1)));
    CHECK_CUDA(cudaMalloc(&d_max, sizeof(Npp32f)));
    CHECK_CUDA(cudaMalloc(&d_maxX, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_maxY, sizeof(int)));
    CHECK_CUDA(cudaMemcpyAsync(d_src, image.data(), image.size(), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_tpl, tpl.data(), tpl.size(), cudaMemcpyHostToDevice, stream));

    auto search = [&]() {
        CHECK_NPP(nppiCrossCorrValid_NormLevel_8u32f_C1R_Ctx(d_src, W, srcSize, d_tpl, tw, tplSize, d_score,
                                                             outW * (int)sizeof(Npp32f), d_corrBuf, ctx));
        CHECK_NPP(nppiMaxIndx_32f_C1R_Ctx(d_score, outW * (int)sizeof(Npp32f), {outW, outH}, d_maxBuf, d_max, d_maxX,
                                          d_maxY, ctx));
    };

    search();
    float bestScore = 0.f;
    int   bestX = 0, bestY = 0;
    CHECK_CUDA(cudaMemcpyAsync(&bestScore, d_max, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(&bestX, d_maxX, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(&bestY, d_maxY, sizeof(int), cudaMemcpyDeviceToHost, stream));
    std::vector<Npp32f> score((size_t)outW * outH);
    CHECK_CUDA(cudaMemcpyAsync(score.data(), d_score, outBytes, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("Image %dx%d, template %dx%d cut from (%d,%d), degraded with gain %.2f, offset %.1f, noise +/-%d\n", W, H, tw, th,
           tx, ty, gain, offset, noise);
    printf("Best match at (%d,%d), NCC score %.5f   [expected (%d,%d)]\n", bestX, bestY, bestScore, tx, ty);

    // How clearly does the best match stand out?  Best score outside a window around the peak.
    float second = -2.f;
    for (int y = 0; y < outH; ++y)
        for (int x = 0; x < outW; ++x)
            if (std::abs(x - bestX) >= tw / 2 || std::abs(y - bestY) >= th / 2) second = std::max(second, score[(size_t)y * outW + x]);
    printf("Best score outside the peak neighbourhood: %.5f  (margin %.5f)\n", second, bestScore - second);

    // Verify against the CPU at the match, at the true location, and at a few other positions.
    double maxDiff = 0.0;
    const int checks[][2] = {{bestX, bestY}, {tx, ty}, {0, 0}, {outW - 1, outH - 1}, {outW / 2, outH / 3}, {outW / 5, outH / 2}};
    for (const auto& p : checks) {
        const double ref = cpuNCC(image, p[0], p[1], tpl, tw, th);
        maxDiff = std::max(maxDiff, std::fabs(ref - score[(size_t)p[1] * outW + p[0]]));
    }
    printf("Max |GPU - CPU| NCC over %zu checked positions: %.2e  [%s]\n", sizeof(checks) / sizeof(checks[0]), maxDiff,
           maxDiff < 1e-3 ? "PASS" : "FAIL");
    const bool located = (std::abs(bestX - tx) <= 1 && std::abs(bestY - ty) <= 1);
    printf("Located the template within 1 pixel: %s\n", located ? "YES" : "NO (template may be ambiguous in this image)");

    // Annotated output: draw a box around the match.
    {
        std::vector<Npp8u> out(image);
        for (int x = bestX; x < bestX + tw; ++x) { out[(size_t)bestY * W + x] = 255; out[(size_t)(bestY + th - 1) * W + x] = 255; }
        for (int y = bestY; y < bestY + th; ++y) { out[(size_t)y * W + bestX] = 255; out[(size_t)y * W + bestX + tw - 1] = 255; }
        std::ofstream pgm("match_result.pgm", std::ios::binary);
        pgm << "P5\n" << W << " " << H << "\n255\n";
        pgm.write(reinterpret_cast<const char*>(out.data()), out.size());
    }

    // Time the GPU search (scores for all positions + argmax). -n 0 skips the timing.
    if (iterations == 0) {
        printf("Timing skipped (-n 0)\n");
    } else {
        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));
        CHECK_CUDA(cudaEventRecord(t0, stream));
        for (int i = 0; i < iterations; ++i) search();
        CHECK_CUDA(cudaEventRecord(t1, stream));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        printf("GPU search over %d x %d positions: %.3f ms per match (average of %d runs)\n", outW, outH, ms / iterations, iterations);

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
    cudaFree(d_src); cudaFree(d_tpl); cudaFree(d_score); cudaFree(d_corrBuf); cudaFree(d_maxBuf);
    cudaFree(d_max); cudaFree(d_maxX); cudaFree(d_maxY);
    cudaStreamDestroy(stream);
    return (maxDiff < 1e-3 && located) ? EXIT_SUCCESS : EXIT_FAILURE;
}
