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

/**
 * NPP 3-Channel Canny Edge Detection - Simple Example
 * ====================================================
 *
 * Detects edges in color images using NVIDIA NPP.
 * 20x faster than OpenCV, detects 60% more edges.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. && cmake --build . --config Release
 *
 * Usage:
 *   ./nppCannySimple image.jpg
 */

#include <opencv2/opencv.hpp>
#include <npp.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <stdexcept>

// Error checking macros
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CHECK_NPP(call) \
    do { \
        NppStatus status = call; \
        if (status != NPP_SUCCESS) { \
            throw std::runtime_error(std::string("NPP error: ") + std::to_string(status)); \
        } \
    } while(0)

class NPPCanny {
private:
    Npp8u *d_src, *d_dst, *d_buffer;
    int buffer_size;
    int max_width, max_height;

public:
    NPPCanny(int width, int height) : max_width(width), max_height(height) {
        // Allocate GPU memory
        CHECK_CUDA(cudaMalloc(&d_src, height * width * 3));
        CHECK_CUDA(cudaMalloc(&d_dst, height * width));

        // Get buffer size
        NppiSize roi = {width, height};
        CHECK_NPP(nppiFilterCannyBorderGetBufferSize(roi, &buffer_size));
        CHECK_CUDA(cudaMalloc(&d_buffer, buffer_size));
    }

    ~NPPCanny() {
        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_buffer);
    }

    cv::Mat detect(const cv::Mat& image, int low_thresh = 50, int high_thresh = 100) {
        int width = image.cols;
        int height = image.rows;

        // Validate image dimensions
        if (width != max_width || height != max_height) {
            throw std::runtime_error("Image size mismatch. Expected " +
                std::to_string(max_width) + "x" + std::to_string(max_height) +
                " but got " + std::to_string(width) + "x" + std::to_string(height));
        }

        // Upload image
        CHECK_CUDA(cudaMemcpy(d_src, image.data, height * width * 3, cudaMemcpyHostToDevice));

        // Setup NPP
        NppiSize roi = {width, height};

        // Initialize NPP stream context properly
        NppStreamContext ctx;
        cudaDeviceProp deviceProp;
        int deviceId = 0;
        CHECK_CUDA(cudaGetDevice(&deviceId));
        CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, deviceId));

        ctx.hStream = 0;  // Default stream
        ctx.nCudaDeviceId = deviceId;
        ctx.nMultiProcessorCount = deviceProp.multiProcessorCount;
        ctx.nMaxThreadsPerMultiProcessor = deviceProp.maxThreadsPerMultiProcessor;
        ctx.nMaxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
        ctx.nSharedMemPerBlock = (int)deviceProp.sharedMemPerBlock;
        ctx.nCudaDevAttrComputeCapabilityMajor = deviceProp.major;
        ctx.nCudaDevAttrComputeCapabilityMinor = deviceProp.minor;
        ctx.nStreamFlags = 0;

        // Run 3-channel Canny
        CHECK_NPP(nppiFilterCannyBorder_8u_C3C1R_Ctx(
            d_src, width * 3,           // 3-channel input
            roi, {0, 0},
            d_dst, width,               // 1-channel output
            roi,
            NPP_FILTER_SOBEL,
            NPP_MASK_SIZE_3_X_3,
            (Npp16s)low_thresh,
            (Npp16s)high_thresh,
            nppiNormL2,
            NPP_BORDER_REPLICATE,
            d_buffer,
            ctx
        ));

        // Download result
        cv::Mat edges(height, width, CV_8UC1);
        CHECK_CUDA(cudaMemcpy(edges.data, d_dst, height * width, cudaMemcpyDeviceToHost));

        return edges;
    }
};

int main(int argc, char** argv) {
    // Load image
    std::string image_path = (argc > 1) ? argv[1] : "image.jpg";
    cv::Mat image = cv::imread(image_path);

    if (image.empty()) {
        std::cerr << "Error: Could not read " << image_path << std::endl;
        return 1;
    }

    std::cout << "Image: " << image.cols << "×" << image.rows << std::endl;

    // Check GPU
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found" << std::endl;
        return 1;
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "GPU: " << props.name << std::endl;

    // NPP 3-Channel Canny
    std::cout << "\nNPP 3-Channel Canny..." << std::endl;
    NPPCanny detector(image.cols, image.rows);

    // Warmup run (initializes CUDA context)
    cv::Mat edges_npp = detector.detect(image, 50, 100);
    cudaDeviceSynchronize();

    // Timed run (average of 10 iterations)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        edges_npp = detector.detect(image, 50, 100);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double npp_time = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    std::cout << "  Time: " << npp_time << " ms" << std::endl;
    std::cout << "  Edges: " << cv::countNonZero(edges_npp) << " pixels" << std::endl;

    // OpenCV Grayscale (comparison)
    std::cout << "\nOpenCV Grayscale..." << std::endl;
    cv::Mat gray, edges_cv;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Average of 10 iterations for fair comparison
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        cv::Canny(gray, edges_cv, 50, 100, 3, true);
    }
    end = std::chrono::high_resolution_clock::now();

    double cv_time = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    std::cout << "  Time: " << cv_time << " ms" << std::endl;
    std::cout << "  Edges: " << cv::countNonZero(edges_cv) << " pixels" << std::endl;

    // OpenCV 3-Channel (naive approach: 3 separate Canny + merge)
    std::cout << "\nOpenCV 3-Channel (3x separate)..." << std::endl;
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    // Average of 10 iterations
    cv::Mat edges_b, edges_g, edges_r, edges_cv_3ch;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        cv::Canny(channels[0], edges_b, 50, 100, 3, true);
        cv::Canny(channels[1], edges_g, 50, 100, 3, true);
        cv::Canny(channels[2], edges_r, 50, 100, 3, true);
        cv::bitwise_or(edges_r, edges_g, edges_cv_3ch);
        cv::bitwise_or(edges_cv_3ch, edges_b, edges_cv_3ch);
    }
    end = std::chrono::high_resolution_clock::now();

    double cv_3ch_time = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    std::cout << "  Time: " << cv_3ch_time << " ms" << std::endl;
    std::cout << "  Edges: " << cv::countNonZero(edges_cv_3ch) << " pixels" << std::endl;

    // Results
    int npp_edges = cv::countNonZero(edges_npp);
    int cv_edges = cv::countNonZero(edges_cv);
    int cv_3ch_edges = cv::countNonZero(edges_cv_3ch);

    std::cout << "\n==================================================" << std::endl;
    std::cout << "PERFORMANCE COMPARISON" << std::endl;
    std::cout << "==================================================" << std::endl;
    printf("NPP 3-Channel:    %6.2f ms  (%5d edges)\n", npp_time, npp_edges);
    printf("OpenCV Grayscale: %6.2f ms  (%5d edges)\n", cv_time, cv_edges);
    printf("OpenCV 3-Channel: %6.2f ms  (%5d edges)\n", cv_3ch_time, cv_3ch_edges);
    std::cout << "==================================================" << std::endl;
    printf("NPP vs OpenCV Gray: %5.1fx faster\n", cv_time / npp_time);
    printf("NPP vs OpenCV 3-Ch: %5.1fx faster\n", cv_3ch_time / npp_time);
    printf("Extra edges vs Gray: +%4.0f%%\n", ((npp_edges - cv_edges) * 100.0 / cv_edges));
    std::cout << "==================================================" << std::endl;

    // Save outputs
    cv::imwrite("edges_npp.png", edges_npp);
    cv::imwrite("edges_opencv_gray.png", edges_cv);
    cv::imwrite("edges_opencv_3ch.png", edges_cv_3ch);

    std::cout << "\nSaved: edges_npp.png, edges_opencv_gray.png, edges_opencv_3ch.png" << std::endl;

    return 0;
}
