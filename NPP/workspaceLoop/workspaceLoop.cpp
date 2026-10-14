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

// NPP workspace in a per-frame loop (CUDA Toolkit 13.4)
// ======================================================
//
// A video-style loop runs the same NPP call on every frame, and the frames do not all have the
// same size. Three ways of providing the scratch buffer that NPP needs are compared:
//
//   1. per-frame alloc : GetBufferSize + cudaMalloc + call + cudaFree on every frame
//   2. preallocated    : one buffer, sized once for the largest frame, passed on every call
//   3. workspace       : nppiWorkspaceCreate once, pass NULL as the buffer; NPP grows a
//                        high-water-mark buffer on demand and reuses it afterwards
//   4. external        : workspace backed by memory the application owns (nppiWorkspaceSetExternal),
//                        for frameworks that must route every allocation through their own allocator
//
// The NPP call is nppiFilterCannyBorder_8u_C1R_Ctx. Input frames are resident on the device, so the
// timings show the cost of the NPP calls and of buffer handling only. All four modes must produce
// identical edge maps; the sample checks that.

#include <cuda_runtime.h>
#include <npp.h>
#include <nppworkspace.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

struct FrameSize {
    int w, h;
};

// The frame sizes the loop cycles through.
static const FrameSize kSizes[] = {{640, 480}, {1280, 720}, {1920, 1080}, {1280, 720}};
static const int kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);

static NppStreamContext makeStreamContext(cudaStream_t stream)
{
    NppStreamContext ctx = {}; // zero-initialize: hWorkspace (formerly nReserved0) must be 0 unless a workspace is created
    int dev = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&dev));
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

// Gray test frame: smooth gradient, a few rectangles and a disc, plus a little noise.
static std::vector<Npp8u> makeFrame(int w, int h)
{
    std::vector<Npp8u> img((size_t)w * h);
    unsigned seed = 12345u;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int v = 60 + (80 * x) / w;
            if (x > w / 8 && x < w / 3 && y > h / 6 && y < h / 2) v = 200;
            if (x > w / 2 && x < 3 * w / 4 && y > h / 2 && y < 5 * h / 6) v = 30;
            int dx = x - 3 * w / 4, dy = y - h / 4, r = h / 8;
            if (dx * dx + dy * dy < r * r) v = 230;
            seed = seed * 1664525u + 1013904223u;
            v += (int)((seed >> 24) % 7) - 3;
            img[(size_t)y * w + x] = (Npp8u)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    }
    return img;
}

struct Frame {
    FrameSize size;
    Npp8u *src, *dst;
    std::vector<Npp8u> result; // edge map copied back from the last run, used to compare the modes
};

enum Mode { kAlloc, kPrealloc, kWorkspace, kExternal, kNumModes };
static const char* kModeName[kNumModes] = {"per-frame alloc", "preallocated", "workspace (NULL)", "workspace + external"};

static const int kCannyLow = 60, kCannyHigh = 120;

static void cannyCall(const Frame& f, Npp8u* buf, const NppStreamContext& ctx)
{
    NppiSize roi = {f.size.w, f.size.h};
    CHECK_NPP(nppiFilterCannyBorder_8u_C1R_Ctx(f.src, f.size.w, roi, {0, 0}, f.dst, f.size.w, roi, NPP_FILTER_SOBEL,
                                               NPP_MASK_SIZE_3_X_3, (Npp16s)kCannyLow, (Npp16s)kCannyHigh, nppiNormL2,
                                               NPP_BORDER_REPLICATE, buf, ctx));
}

// Processes `frames` frames and returns the wall-clock time per frame in microseconds.
static double runLoop(Mode mode, std::vector<Frame>& all, int frames, NppStreamContext& ctx, Npp8u* sharedBuf)
{
    cudaStream_t stream = ctx.hStream;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; ++i) {
        Frame& f = all[i % kNumSizes];
        NppiSize roi = {f.size.w, f.size.h};
        if (mode == kAlloc) {
            int size = 0;
            Npp8u* buf = nullptr;
            CHECK_NPP(nppiFilterCannyBorderGetBufferSize(roi, &size));
            CHECK_CUDA(cudaMalloc(&buf, size > 0 ? size : 1)); // never hand NPP a null buffer when it is not using the workspace
            cannyCall(f, buf, ctx);
            CHECK_CUDA(cudaFree(buf)); // implicit device synchronization
        } else if (mode == kPrealloc) {
            cannyCall(f, sharedBuf, ctx);
        } else {
            cannyCall(f, nullptr /* scratch from workspace */, ctx);
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / frames;
}

int main(int argc, char** argv)
{
    int frames = 2000;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            frames = atoi(argv[++i]);
        } else {
            printf("Usage: %s [-n frames]\n", argv[0]);
            return 1;
        }
    }
    if (frames < kNumSizes) frames = kNumSizes;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Device frames (one per entry of kSizes) and the size of the largest scratch requirement.
    std::vector<Frame> all(kNumSizes);
    int maxScratch = 1;
    for (int i = 0; i < kNumSizes; ++i) {
        Frame& f = all[i];
        f.size = kSizes[i];
        std::vector<Npp8u> img = makeFrame(f.size.w, f.size.h);
        CHECK_CUDA(cudaMalloc(&f.src, img.size()));
        CHECK_CUDA(cudaMalloc(&f.dst, img.size()));
        CHECK_CUDA(cudaMemcpy(f.src, img.data(), img.size(), cudaMemcpyHostToDevice));
        int size = 0;
        NppiSize roi = {f.size.w, f.size.h};
        CHECK_NPP(nppiFilterCannyBorderGetBufferSize(roi, &size));
        if (size > maxScratch) maxScratch = size;
    }

    printf("Canny on %d frames cycling through", frames);
    for (int i = 0; i < kNumSizes; ++i) printf(" %dx%d", kSizes[i].w, kSizes[i].h);
    printf("\nLargest scratch requirement: %.1f MB\n\n", maxScratch / 1048576.0);

    Npp8u* sharedBuf = nullptr;
    CHECK_CUDA(cudaMalloc(&sharedBuf, maxScratch));

    std::vector<std::vector<Npp8u>> results(kNumModes);
    double usPerFrame[kNumModes] = {};
    size_t wsCapacity[kNumModes] = {};

    for (int m = 0; m < kNumModes; ++m) {
        NppStreamContext ctx = makeStreamContext(stream);
        void* external = nullptr;
        if (m == kWorkspace || m == kExternal) {
            CHECK_NPP(nppiWorkspaceCreate(&ctx));
            if (m == kExternal) {
                CHECK_CUDA(cudaMalloc(&external, maxScratch));
                CHECK_NPP(nppiWorkspaceSetExternal(&ctx, external, maxScratch));
            }
        }

        // Warm-up pass over every frame size. With the workspace this is also where it grows to its high-water mark.
        runLoop((Mode)m, all, kNumSizes, ctx, sharedBuf);
        // Median of five timed runs.
        double t[5];
        for (int r = 0; r < 5; ++r) t[r] = runLoop((Mode)m, all, frames, ctx, sharedBuf);
        for (int a = 0; a < 5; ++a)
            for (int b = a + 1; b < 5; ++b)
                if (t[b] < t[a]) { double x = t[a]; t[a] = t[b]; t[b] = x; }
        usPerFrame[m] = t[2];

        if (m == kWorkspace || m == kExternal) CHECK_NPP(nppiWorkspaceGetCapacity(&ctx, &wsCapacity[m]));

        // Keep the 1080p result of this mode for the comparison.
        Frame& big = all[2];
        results[m].resize((size_t)big.size.w * big.size.h);
        CHECK_CUDA(cudaMemcpy(results[m].data(), big.dst, results[m].size(), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemset(big.dst, 0, results[m].size()));

        if (m == kWorkspace || m == kExternal) CHECK_NPP(nppiWorkspaceDestroy(&ctx));
        if (external) CHECK_CUDA(cudaFree(external));
    }

    bool same = true;
    for (int m = 1; m < kNumModes; ++m) same = same && results[m] == results[0];
    size_t edges = 0;
    for (Npp8u v : results[0]) edges += v != 0;

    printf("%-24s %14s %10s\n", "mode", "us per frame", "vs alloc");
    for (int m = 0; m < kNumModes; ++m) printf("%-24s %14.1f %9.2fx\n", kModeName[m], usPerFrame[m], usPerFrame[kAlloc] / usPerFrame[m]);
    printf("\nWorkspace capacity after the loop: %.1f MB (internal), %.1f MB (external)\n", wsCapacity[kWorkspace] / 1048576.0,
           wsCapacity[kExternal] / 1048576.0);
    printf("1080p edge pixels: %zu, results identical across modes: %s\n", edges, same ? "YES" : "NO");

    for (Frame& f : all) {
        cudaFree(f.src);
        cudaFree(f.dst);
    }
    cudaFree(sharedBuf);
    cudaStreamDestroy(stream);
    printf("%s\n", same && edges > 0 ? "PASS" : "FAIL");
    return same && edges > 0 ? 0 : 1;
}
