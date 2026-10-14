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

// NPP vs OpenCV (CPU) benchmark
// =============================
//
// Times the same four operations with NPP on the GPU and with OpenCV on the CPU:
//   Gaussian blur 5x5, Canny edges, bilinear resize to half size, RGB to gray.
//
// Three numbers are reported per operation:
//   NPP kernel   : GPU time of the NPP call only (data already on the device, workspace in use)
//   NPP end-to-end: host-to-device copy + NPP call + device-to-host copy (pinned host memory)
//   OpenCV 1 / N threads : cv::setNumThreads(1) and all hardware threads
//
// Use "kernel" when the data stays on the GPU through a pipeline, and "end-to-end" when the GPU is used
// for a single operation on host data. The outputs are compared for sanity (mean absolute difference);
// they are not bit-identical because the implementations differ in rounding and border handling.

#include <cuda_runtime.h>
#include <npp.h>
#include <nppworkspace.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
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

static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Median wall time in milliseconds of `iters` calls to fn(), over several repetitions.
static double timeCpu(const std::function<void()>& fn, int iters)
{
    fn(); // warm-up
    std::vector<double> t;
    for (int r = 0; r < 5; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) fn();
        auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    }
    return median(t);
}

// Median time in milliseconds of `iters` calls to fn() on the stream, measured with CUDA events.
static double timeGpu(const std::function<void()>& fn, cudaStream_t stream, int iters)
{
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    fn(); // warm-up (also grows the workspace)
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<double> t;
    for (int r = 0; r < 5; ++r) {
        CHECK_CUDA(cudaEventRecord(e0, stream));
        for (int i = 0; i < iters; ++i) fn();
        CHECK_CUDA(cudaEventRecord(e1, stream));
        CHECK_CUDA(cudaEventSynchronize(e1));
        float ms = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        t.push_back(ms / iters);
    }
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    return median(t);
}

static double meanAbsDiff(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat d;
    cv::absdiff(a, b, d);
    return cv::mean(d.reshape(1))[0];
}

struct Result {
    const char* name;
    double gpuKernel, gpuE2E, cpu1, cpuN, diff;
};

int main(int argc, char** argv)
{
    std::string imagePath = "../images/CircuitBoard_2048x1024_8u.raw";
    int W = 4096, H = 2048, iters = 20;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) imagePath = argv[++i];
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) iters = atoi(argv[++i]);
        else {
            printf("Usage: %s [-i 2048x1024.raw] [-w width] [-h height] [-n iterations]\n", argv[0]);
            printf("The 2048x1024 input image is tiled to width x height (default 4096x2048).\n");
            return 1;
        }
    }

    if (W < 16 || H < 16 || W > 16384 || H > 16384 || iters < 1) {
        fprintf(stderr, "Invalid option: need 16 <= -w/-h <= 16384 and -n >= 1\n");
        return 1;
    }

    // Input: the 2048x1024 circuit board, tiled (mirrored at the seams) to the requested size.
    const int SW = 2048, SH = 1024;
    std::vector<unsigned char> raw((size_t)SW * SH);
    std::ifstream f(imagePath, std::ios::binary);
    if (!f.read((char*)raw.data(), raw.size())) {
        fprintf(stderr, "Cannot read %zu bytes from %s\n", raw.size(), imagePath.c_str());
        return 1;
    }
    cv::Mat base(SH, SW, CV_8UC1, raw.data()), gray(H, W, CV_8UC1);
    for (int y = 0; y < H; ++y) {
        int sy = (y / SH) % 2 ? SH - 1 - y % SH : y % SH;
        for (int x = 0; x < W; ++x) {
            int sx = (x / SW) % 2 ? SW - 1 - x % SW : x % SW;
            gray.at<unsigned char>(y, x) = base.at<unsigned char>(sy, sx);
        }
    }
    cv::Mat rgb;
    {
        cv::Mat planes[3] = {gray, 255 - gray, gray / 2 + 40};
        cv::merge(planes, 3, rgb);
    }
    const int OW = W / 2, OH = H / 2;
    const int hwThreads = cv::getNumThreads();
    if (hwThreads < 2)
        printf("Warning: this OpenCV build reports only %d thread; the multi-thread column will not be meaningful.\n"
               "Point CMake at an OpenCV built with a threading backend (pthreads, TBB or OpenMP).\n\n", hwThreads);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    NppStreamContext ctx = makeStreamContext(stream);
    CHECK_NPP(nppiWorkspaceCreate(&ctx));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, ctx.nCudaDeviceId));
    printf("Image %dx%d, %d iterations per timing, median of 5\n", W, H, iters);
    printf("GPU: %s | OpenCV %s, %d CPU threads\n\n", prop.name, CV_VERSION, hwThreads);

    // Device and pinned host buffers.
    size_t grayBytes = (size_t)W * H, rgbBytes = grayBytes * 3, halfBytes = (size_t)OW * OH;
    Npp8u *d_gray, *d_rgb, *d_out, *d_half;
    CHECK_CUDA(cudaMalloc(&d_gray, grayBytes));
    CHECK_CUDA(cudaMalloc(&d_rgb, rgbBytes));
    CHECK_CUDA(cudaMalloc(&d_out, grayBytes));
    CHECK_CUDA(cudaMalloc(&d_half, halfBytes));
    Npp8u *h_in, *h_out;
    CHECK_CUDA(cudaMallocHost(&h_in, rgbBytes));
    CHECK_CUDA(cudaMallocHost(&h_out, grayBytes));
    std::memcpy(h_in, gray.data, grayBytes);
    CHECK_CUDA(cudaMemcpy(d_gray, gray.data, grayBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rgb, rgb.data, rgbBytes, cudaMemcpyHostToDevice));

    NppiSize full = {W, H}, half = {OW, OH};
    NppiRect fullRect = {0, 0, W, H}, halfRect = {0, 0, OW, OH};

    // NPP operations: each takes the device input/output and works in the stream.
    auto nppGauss = [&]() {
        CHECK_NPP(nppiFilterGaussBorder_8u_C1R_Ctx(d_gray, W, full, {0, 0}, d_out, W, full, NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE, ctx));
    };
    auto nppCanny = [&]() {
        CHECK_NPP(nppiFilterCannyBorder_8u_C1R_Ctx(d_gray, W, full, {0, 0}, d_out, W, full, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, 60, 120,
                                                   nppiNormL2, NPP_BORDER_REPLICATE, nullptr /* scratch from workspace */, ctx));
    };
    auto nppResize = [&]() {
        CHECK_NPP(nppiResize_8u_C1R_Ctx(d_gray, W, full, fullRect, d_half, OW, half, halfRect, NPPI_INTER_LINEAR, ctx));
    };
    auto nppRgbToGray = [&]() { CHECK_NPP(nppiRGBToGray_8u_C3C1R_Ctx(d_rgb, W * 3, d_out, W, full, ctx)); };

    // End-to-end wrapper: copy the input in, run the op, copy the output back.
    auto e2e = [&](const std::function<void()>& op, const Npp8u* hostIn, Npp8u* devIn, size_t inBytes, const Npp8u* devOut, size_t outBytes) {
        return [&, hostIn, devIn, inBytes, devOut, outBytes]() {
            CHECK_CUDA(cudaMemcpyAsync(devIn, hostIn, inBytes, cudaMemcpyHostToDevice, stream));
            op();
            CHECK_CUDA(cudaMemcpyAsync(h_out, devOut, outBytes, cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
        };
    };
    Npp8u* h_rgbIn;
    CHECK_CUDA(cudaMallocHost(&h_rgbIn, rgbBytes));
    std::memcpy(h_rgbIn, rgb.data, rgbBytes);

    // OpenCV counterparts.
    cv::Mat cvOut, cvHalf, cvGrayFromRgb;
    auto cvGauss  = [&]() { cv::GaussianBlur(gray, cvOut, cv::Size(5, 5), 0); };
    auto cvCanny  = [&]() { cv::Canny(gray, cvOut, 60, 120, 3, true); };
    auto cvResize = [&]() { cv::resize(gray, cvHalf, cv::Size(OW, OH), 0, 0, cv::INTER_LINEAR); };
    auto cvColor  = [&]() { cv::cvtColor(rgb, cvGrayFromRgb, cv::COLOR_RGB2GRAY); };

    struct Op {
        const char* name;
        std::function<void()> npp, cv;
        const Npp8u *hIn; Npp8u* dIn; size_t inBytes; const Npp8u* dOut; size_t outBytes;
        int outW, outH;
        cv::Mat* cvResult;
    };
    std::vector<Op> ops = {
        {"Gaussian 5x5", nppGauss, cvGauss, h_in, d_gray, grayBytes, d_out, grayBytes, W, H, &cvOut},
        {"Canny 3x3 L2", nppCanny, cvCanny, h_in, d_gray, grayBytes, d_out, grayBytes, W, H, &cvOut},
        {"Resize 1/2 bilinear", nppResize, cvResize, h_in, d_gray, grayBytes, d_half, halfBytes, OW, OH, &cvHalf},
        {"RGB to gray", nppRgbToGray, cvColor, h_rgbIn, d_rgb, rgbBytes, d_out, grayBytes, W, H, &cvGrayFromRgb},
    };

    std::vector<Result> results;
    for (Op& op : ops) {
        Result r = {op.name, 0, 0, 0, 0, 0};
        r.gpuKernel = timeGpu(op.npp, stream, iters);
        r.gpuE2E    = timeGpu(e2e(op.npp, op.hIn, op.dIn, op.inBytes, op.dOut, op.outBytes), stream, iters);
        cv::setNumThreads(1);
        r.cpu1 = timeCpu(op.cv, std::max(1, iters / 4));
        cv::setNumThreads(hwThreads);
        r.cpuN = timeCpu(op.cv, iters);

        // Sanity comparison of the outputs (one fresh run of each).
        op.npp();
        CHECK_CUDA(cudaStreamSynchronize(stream));
        cv::Mat nppResult(op.outH, op.outW, CV_8UC1);
        CHECK_CUDA(cudaMemcpy(nppResult.data, op.dOut, (size_t)op.outW * op.outH, cudaMemcpyDeviceToHost));
        op.cv();
        r.diff = meanAbsDiff(nppResult, *op.cvResult);
        results.push_back(r);
        if (!strcmp(op.name, "Canny 3x3 L2"))
            printf("Canny edge pixels: NPP %d, OpenCV %d (thresholds differ in rounding, counts are not expected to match exactly)\n\n",
                   cv::countNonZero(nppResult), cv::countNonZero(*op.cvResult));
    }

    printf("%-22s %11s %11s %11s %11s | %9s %9s %9s | %s\n", "operation", "NPP kernel", "NPP e2e", "CV 1 thr", "CV N thr", "kernel", "e2e", "e2e", "mean abs");
    printf("%-22s %11s %11s %11s %11s | %9s %9s %9s | %s\n", "", "(ms)", "(ms)", "(ms)", "(ms)", "vs 1 thr", "vs 1 thr", "vs N thr", "diff");
    for (const Result& r : results)
        printf("%-22s %11.3f %11.3f %11.3f %11.3f | %8.1fx %8.1fx %8.1fx | %.2f\n", r.name, r.gpuKernel, r.gpuE2E, r.cpu1, r.cpuN,
               r.cpu1 / r.gpuKernel, r.cpu1 / r.gpuE2E, r.cpuN / r.gpuE2E, r.diff);

    nppiWorkspaceDestroy(&ctx);
    cudaFree(d_gray); cudaFree(d_rgb); cudaFree(d_out); cudaFree(d_half);
    cudaFreeHost(h_in); cudaFreeHost(h_out); cudaFreeHost(h_rgbIn);
    cudaStreamDestroy(stream);
    return 0;
}
