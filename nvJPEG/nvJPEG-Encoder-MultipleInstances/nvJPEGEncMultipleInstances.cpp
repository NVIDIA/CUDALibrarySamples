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

#include <algorithm>
#include <array>
#include <barrier>
#include <cassert>
#include <cfloat>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#define CHECK_CUDA(call) \
do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) \
    { \
        std::cerr << "CUDA Runtime failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_NVJPEG(call) \
do { \
    nvjpegStatus_t _e = (call); \
    if (_e != NVJPEG_STATUS_SUCCESS) \
    { \
        std::cerr << "NVJPEG failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK(condition, message) \
do { \
    if (!(condition)) { \
        std::cerr << "Error: " << message << std::endl; \
        std::exit(1); \
    } \
} while(0)

constexpr int NUM_BACKENDS = 3;

using Clock = std::chrono::steady_clock;
using namespace std::chrono;

struct per_gpu;
struct per_thread;
struct per_thread_encode_slot;

// Parameters provided to the application.
struct params
{
    bool backend_enabled[NUM_BACKENDS] = {};
    int only_backend = -1;
    std::string input_dir = "";
    std::string output_dir = ".";
    int total_images = 0;
    int num_states = 0;
    int num_threads = 0;
    int num_runs = 1;
    int quality = 80;
    bool preallocate = false;
    bool write_output = false;
    bool download_bitstream = false;
    bool optimized_huffman = false;
    bool subsampling_420 = true;
};

struct img_t;

// Keep track of throughput and latency for each backend.
struct backend_t
{
    nvjpegEncoderState_t template_state = nullptr; // Used only for hardware backend info query
    std::atomic<int> num_jobs_started = 0;
    std::atomic<int> num_jobs_done = 0;
    std::atomic<double> pixels_started = 0;
    std::atomic<double> pixels_done = 0;
    nvjpegEncBackend_t backend = NVJPEG_ENC_BACKEND_DEFAULT;
    Clock::time_point start;
    float elapsed_time = 0;
    float throughput = 0;
    float max_throughput = 0.0f;
    int stable_counter = 0;
    std::atomic<float> total_latency = 0;
    std::atomic<bool> calibration_done = false;
    std::mutex update_mutex;
    bool enabled = false;

    bool isSupported(const img_t& img, const params& p) const;
    void update(per_thread_encode_slot& es);
};

struct per_thread_encode_slot
{
    per_thread* pt = nullptr;
    img_t* img = nullptr;
    nvjpegEncBackend_t backend = NVJPEG_ENC_BACKEND_DEFAULT;
    nvjpegEncoderState_t encode_state = nullptr;
};

// Keep all the data relevant to a source image for encoding.
struct img_t
{
    std::string file_name;
    unsigned char* input_host_buf = nullptr;
    int input_host_buf_size = 0;
    unsigned char* output_host_buf = nullptr;
    int output_host_buf_size = 0;
    nvjpegImage_t dev_img = {};
    int padded_dev_buf_size = 0;
    int dev_buf_size = 0;
    int jpeg_size = 0;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int img_id = 0;
    nvjpegChromaSubsampling_t ss = NVJPEG_CSS_444;
    Clock::time_point start;
    Clock::time_point stop;

    void copyFrom(const img_t& src);

    void readFromBmpFile(const std::string& fileName, const per_gpu& pg);
    void uploadToDevice(per_thread& pt);

    void allocateInputDeviceBuf(per_gpu& pg, per_thread& pt);
    void deallocateInputDeviceBuf(per_gpu& pg, per_thread& pt);
    void allocateInputHostBuf(int size);
    void deallocateInputHostBuf();
    void allocateOutputHostBuf(int size);
    void deallocateOutputHostBuf();

    void encodeHost(per_thread_encode_slot& es, const params& p);
    void encodeDevice(per_thread_encode_slot& es, const params& p);
    void retrieveBitStreamToHost(per_thread_encode_slot& es);
    void writeJpegToFile(const params& p);

    double getTotalPixels() const;

    static int padSize(int size);
    static int padDimension(const per_gpu& pg, int dim);
};

// Each per_thread object is guaranteed to not be shared among host threads.
struct per_thread
{
    per_gpu* pg = nullptr;
    std::array<per_thread_encode_slot, NUM_BACKENDS> slots;
    cudaStream_t stream = nullptr;
    img_t img[2]; // double-buffered
    int img_slot = -1;

    int findEncodeSlot(const params& p, per_gpu& pg, const img_t& img, int sid);
    int nextImageSlot() { return (img_slot + 1) % 2; }
    per_thread_encode_slot* prevEs = nullptr;
};

// Main "mother" struct.
struct per_gpu
{
    std::string device_name;
    nvjpegHandle_t handle = nullptr;
    std::array<backend_t, NUM_BACKENDS> backends = {};
    std::array<int, NUM_BACKENDS> backend_order = {};
    nvjpegEncoderParams_t eparams = nullptr;
    std::vector<per_thread> per_threads;
    int device_id = 0;
    int num_engines = 0;
    int async_malloc_supported = 0;
    int pitch_alignment = 256;
    std::atomic<bool> calibration_done = false;

    void create(params& p);
    void destroy();
    void preAllocateBuffers(const std::string& fileName, const params& p, const per_gpu& pg);
};

// Pad the size of a bitstream to multiples of 1 MB.
int img_t::padSize(int size)
{
    constexpr int mb = 1 * 1024 * 1024;
    int paddedSize = mb * ((size + mb - 1) / mb);
    return paddedSize;
}

// Pad a dimension of an image to multiples of 512.
int img_t::padDimension(const per_gpu& pg, int dim)
{
    const int pad = std::max(((512 / pg.pitch_alignment) * pg.pitch_alignment), pg.pitch_alignment);
    int paddedDim = pad * ((dim + pad - 1) / pad);
    return paddedDim;
}

double img_t::getTotalPixels() const
{
    static constexpr float scale[] = {
        3.0f,  // 444
        2.0f,  // 422
        1.5f,  // 420
        2.0f,  // 440
        1.5f,  // 411
        1.25f, // 410
        1.0f,  // GRAY
    };
    return widths[0] * heights[0] * scale[ss];
}

// Copy the host content of the source image.
void img_t::copyFrom(const img_t& src)
{
    file_name = src.file_name;
    if (src.input_host_buf && src.input_host_buf_size > 0) {
        int requiredSize = src.input_host_buf_size;
        if (input_host_buf_size < requiredSize) {
            deallocateInputHostBuf();
            allocateInputHostBuf(requiredSize);
        }
        std::memcpy(input_host_buf, src.input_host_buf, src.input_host_buf_size);
    }
    for (int i = 0; i < NVJPEG_MAX_COMPONENT; ++i) {
        widths[i] = src.widths[i];
        heights[i] = src.heights[i];
    }
}

// Allocate the input host buffer, storing the raw RGB data.
void img_t::allocateInputHostBuf(int size)
{
    int paddedSize = padSize(size);
    if (input_host_buf_size < paddedSize) {
        deallocateInputHostBuf();
        input_host_buf_size = paddedSize;
        CHECK_CUDA(cudaMallocHost((void**)&input_host_buf, input_host_buf_size));
    }
}

// Deallocate the input host buffer.
void img_t::deallocateInputHostBuf()
{
    if (input_host_buf) {
        CHECK_CUDA(cudaFreeHost(input_host_buf));
        input_host_buf = nullptr;
        input_host_buf_size = 0;
    }
}

// Allocate the output host buffer, storing the JPEG bitstream.
void img_t::allocateOutputHostBuf(int size)
{
    int paddedSize = padSize(size);
    if (output_host_buf_size < paddedSize) {
        deallocateOutputHostBuf();
        output_host_buf_size = paddedSize;
        CHECK_CUDA(cudaMallocHost((void**)&output_host_buf, output_host_buf_size));
    }
}

// Deallocate the output host buffer.
void img_t::deallocateOutputHostBuf()
{
    if (output_host_buf) {
        CHECK_CUDA(cudaFreeHost(output_host_buf));
        output_host_buf = nullptr;
        output_host_buf_size = 0;
    }
}

// Allocate the input device buffer, storing the raw image before encoding.
void img_t::allocateInputDeviceBuf(per_gpu& pg, per_thread& pt)
{
    int w = widths[0], h = heights[0];
    int paddedPitch = padDimension(pg, w);
    int planeSize = paddedPitch * h;
    int totalSize = planeSize * 3;

    if (totalSize > padded_dev_buf_size) {
        deallocateInputDeviceBuf(pg, pt);
        if (pg.async_malloc_supported)
            CHECK_CUDA(cudaMallocAsync((void**)(&(dev_img.channel[0])), totalSize, pt.stream));
        else
            CHECK_CUDA(cudaMalloc((void**)(&(dev_img.channel[0])), totalSize));
        padded_dev_buf_size = totalSize;
    }

    dev_buf_size = totalSize;

    for (int c = 0; c < 3; ++c) {
        dev_img.pitch[c] = paddedPitch;
        dev_img.channel[c] = dev_img.channel[0] + c * planeSize;
    }
}

// Deallocate the input device buffer.
void img_t::deallocateInputDeviceBuf(per_gpu& pg, per_thread& pt)
{
    if (dev_img.channel[0]) {
        if (pg.async_malloc_supported)
            CHECK_CUDA(cudaFreeAsync(dev_img.channel[0], pt.stream));
        else
            CHECK_CUDA(cudaFree(dev_img.channel[0]));
        dev_img.channel[0] = nullptr;
        padded_dev_buf_size = 0;
    }
}

// Read a BMP file to memory in planar RGB format with padded pitch.
void img_t::readFromBmpFile(const std::string& fileName, const per_gpu& pg)
{
    if (file_name == fileName)
        return;
    file_name = fileName;

    std::ifstream file(fileName, std::ios::binary | std::ios::ate);
    CHECK(file.is_open(), "Error: Could not open BMP file: " + fileName);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> fileData(fileSize);
    CHECK(file.read(reinterpret_cast<char*>(fileData.data()), fileSize),
        "Error: Failed to read BMP file: " + fileName);

    CHECK(fileSize >= 54, "Error: BMP file too small: " + fileName);
    CHECK(fileData[0] == 'B' && fileData[1] == 'M', "Error: Not a valid BMP file: " + fileName);

    int dataOffset = *reinterpret_cast<int*>(&fileData[10]);
    int imgWidth = *reinterpret_cast<int*>(&fileData[18]);
    int imgHeight = *reinterpret_cast<int*>(&fileData[22]);
    int bpp = *reinterpret_cast<short*>(&fileData[28]);

    CHECK(bpp == 24, "Error: Only 24-bit BMP files are supported. This file is " + std::to_string(bpp) + "-bit.\n");

    widths[0] = imgWidth;
    heights[0] = imgHeight;

    int paddedPitch = padDimension(pg, imgWidth);
    int planeSize = paddedPitch * imgHeight;
    int totalSize = planeSize * 3;
    allocateInputHostBuf(totalSize);

    int bytesPerPixel = bpp / 8;
    int rowStride = imgWidth * bytesPerPixel;
    int rowPadding = (4 - (rowStride % 4)) % 4;
    int paddedRowSize = rowStride + rowPadding;

    unsigned char* rPtr = input_host_buf;
    unsigned char* gPtr = input_host_buf + planeSize;
    unsigned char* bPtr = input_host_buf + 2 * planeSize;

    const unsigned char* srcData = fileData.data() + dataOffset;

    for (int y = 0; y < imgHeight; y++) {
        int destRow = imgHeight - 1 - y; // BMP is bottom-up
        const unsigned char* rowPtr = srcData + y * paddedRowSize;
        int destOffset = destRow * paddedPitch;

        for (int x = 0; x < imgWidth; x++) {
            const unsigned char* pixel = rowPtr + x * 3;
            rPtr[destOffset + x] = pixel[2]; // BGR -> RGB
            gPtr[destOffset + x] = pixel[1];
            bPtr[destOffset + x] = pixel[0];
        }
    }
}

// Do nothing, just start the clock for timing.
void img_t::encodeHost(per_thread_encode_slot& es, const params& p)
{
    backend_t& b = es.pt->pg->backends[es.backend];
    int num_jobs_started = b.num_jobs_started.fetch_add(1);
    if (num_jobs_started == 0)
        b.start = Clock::now();
    start = Clock::now();
    ss = p.subsampling_420 ? NVJPEG_CSS_420 : NVJPEG_CSS_444;
    auto pixels = getTotalPixels();
    b.pixels_started += pixels;
}

// Upload the raw image buffer to the device.
void img_t::uploadToDevice(per_thread& pt)
{
    CHECK_CUDA(cudaMemcpyAsync(dev_img.channel[0], input_host_buf, dev_buf_size, cudaMemcpyHostToDevice, pt.stream));
}

// Encode the image on the device.
void img_t::encodeDevice(per_thread_encode_slot& es, const params& p)
{
    // Setting input chroma subsampling to CSS_UNKNOWN since it doesn't matter as we are using RGB input
    // The actual desired subsampling in the output JPEG bitstream was set using nvjpegEncoderParamsSetSamplingFactors
    CHECK_NVJPEG(nvjpegEncode(es.pt->pg->handle, es.encode_state, es.pt->pg->eparams, &dev_img,
        NVJPEG_CSS_UNKNOWN, NVJPEG_INPUT_RGB, widths[0], heights[0], es.pt->stream));
}

// Retrieve JPEG bitstream to host.
void img_t::retrieveBitStreamToHost(per_thread_encode_slot& es)
{
    size_t size = 0;
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(es.pt->pg->handle, es.encode_state, nullptr, &size, es.pt->stream));
    allocateOutputHostBuf((int)size);
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(es.pt->pg->handle, es.encode_state, output_host_buf, &size, es.pt->stream));
    jpeg_size = size;
}

// Write encoded JPEG to a file.
void img_t::writeJpegToFile(const params& p)
{
    std::string fileName = p.output_dir + "/" + std::to_string(img_id) + ".jpg";
    std::ofstream output(fileName, std::ios::binary);
    CHECK(output, "Error: Failed to open file " + fileName);
    output.write(reinterpret_cast<const char*>(output_host_buf), jpeg_size);
    CHECK(output, "Error: Failed to write " + std::to_string(jpeg_size) + " bytes to file " + fileName);
}

// Return true if the backend supports the given image.
bool backend_t::isSupported(const img_t& img, const params& p) const
{
    bool progressive = false; // This is hardcoded for now but in the future we may want to set progressive in params
    if (backend == NVJPEG_ENC_BACKEND_HARDWARE) {
        int w = img.widths[0], h = img.heights[0];
        if (!p.subsampling_420 || progressive || w > 16000 || h > 16000 || w < 16 || h < 16 || (w % 2) || (h % 2))
            return false;
    }
    return true;
}

// Update the latest throughput and latency.
void backend_t::update(per_thread_encode_slot& es)
{
    assert(backend == es.backend);
    pixels_done += es.img->getTotalPixels();
    ++num_jobs_done;
    es.img->stop = Clock::now();
    float latency = duration<float, std::milli>(es.img->stop - es.img->start).count();
    total_latency += latency;
    float time = duration<float, std::milli>(es.img->stop - start).count();

    std::lock_guard<std::mutex> lock(update_mutex);

    if (stable_counter < 8) {
        float current_throughput = pixels_done / time;
        if (current_throughput > max_throughput * 1.0001f) { // small epsilon for noise
            max_throughput = current_throughput;
            stable_counter = 0;
        }
        else {
            ++stable_counter;
        }
    }

    if (time > elapsed_time) {
        elapsed_time = time;
        throughput = pixels_done / time;
    }
}

// Find a backend to encode the next image.
int per_thread::findEncodeSlot(const params& p, per_gpu& pg, const img_t& img, int sid)
{
    if (p.only_backend > 0)
        return p.only_backend;

    // Phase 1: calibration, determine the best throughput for each backend
    if (!pg.calibration_done) {
        // return the first enabled backend that is not yet fully estimated
        for (int b = 1; b < NUM_BACKENDS; ++b) {
            backend_t& be = pg.backends[b];
            if (be.enabled && be.stable_counter < 8 && be.isSupported(img, p))
                return b;
        }
        // if we reach here, all enabled backends are calibrated
        // sort and mark calibration_done once
        bool expected = false;
        if (pg.calibration_done.compare_exchange_strong(expected, true)) {
            float a[2] = {
                pg.backends[1].max_throughput,
                pg.backends[2].max_throughput
            };
            int imx = (a[0] >= a[1]) ? 0 : 1; // max
            int imn = 1 - imx; // min
            pg.backend_order[0] = imx + 1;
            pg.backend_order[1] = imn + 1;
        }
    }

    // Phase 2: calibration is done, now try to allocate backend
    // from the best throughput to the worst. If the "backlog"
    // is greater than twice the number of threads, "spill-over"
    // to the next backend.
    int nextBackend = 0;
    for (int i = 0; i < NUM_BACKENDS - 1; ++i) {
        int b = pg.backend_order[i];
        backend_t& be = pg.backends[b];
        if (be.enabled && (be.num_jobs_started - be.num_jobs_done < p.num_threads * 2) && be.isSupported(img, p)) {
            nextBackend = b;
            break;
        }
    }

    // If all backends have long backlogs, pick the backend
    // with the least current drain time
    if (nextBackend == 0) {
        double pixels = img.getTotalPixels();
        float minDrainTime = FLT_MAX;
        for (int b = 1; b < NUM_BACKENDS; ++b) {
            backend_t& be = pg.backends[b];
            if (be.enabled && be.isSupported(img, p)) {
                float drainTime = (be.pixels_started - be.pixels_done + pixels) / be.throughput;
                if (drainTime < minDrainTime) {
                    nextBackend = b;
                    minDrainTime = drainTime;
                }
            }
        }
    }

    return nextBackend;
}

// Create all "global" states and handles.
void per_gpu::create(params& p)
{
    CHECK_CUDA(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    device_name = prop.name;
    CHECK_CUDA(cudaDeviceGetAttribute(&async_malloc_supported, cudaDevAttrMemoryPoolsSupported, device_id));
    CHECK_CUDA(cudaDeviceGetAttribute(&pitch_alignment, cudaDevAttrTexturePitchAlignment, device_id));

    CHECK_NVJPEG(nvjpegCreateSimple(&handle));
    CHECK_NVJPEG(nvjpegSetDeviceMemoryPadding(1024 * 1024, handle));

    unsigned int nEngines = 0;
    CHECK_NVJPEG(nvjpegGetHardwareEncoderInfo(handle, &nEngines));
    num_engines = nEngines;

    if (p.num_states == 0)
        p.num_states = std::max({ 1, num_engines * 2, p.num_threads });
    if (p.num_threads == 0)
        p.num_threads = p.num_states;
    p.total_images = std::max(p.total_images, p.num_states);

    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle, &eparams, nullptr));
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(eparams, p.quality, nullptr));
    CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(eparams, p.optimized_huffman, nullptr));
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(eparams, p.subsampling_420 ? NVJPEG_CSS_420 : NVJPEG_CSS_444, nullptr));
    CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(eparams, NVJPEG_ENCODING_BASELINE_DCT, nullptr));

    for (int b = 0; b < NUM_BACKENDS; ++b) {
        if (p.backend_enabled[b]) {
            backends[b].enabled = true;
            backends[b].backend = nvjpegEncBackend_t(b);
        }
    }

    per_threads.resize(p.num_states);
    for (int s = 0; s < p.num_states; ++s) {
        per_thread& pt = per_threads[s];
        pt.pg = this;
        CHECK_CUDA(cudaStreamCreateWithFlags(&pt.stream, cudaStreamNonBlocking));

        for (int b = 0; b < NUM_BACKENDS; ++b) {
            if (backends[b].enabled) {
                per_thread_encode_slot& es = pt.slots[b];
                es.pt = &pt;
                es.backend = nvjpegEncBackend_t(b);
                // Sometimes the user requests for a backend but it's not available.
                // So we check for that and potentially override the setting here.
                bool be = NVJPEG_STATUS_SUCCESS == nvjpegEncoderStateCreateWithBackend(handle, &es.encode_state, nvjpegEncBackend_t(b), pt.stream);
                backends[b].enabled = p.backend_enabled[b] = be;
            }
        }
    }

    // The hardware backend can get stuck if we don't retrieve bitstream
    if (backends[NVJPEG_ENC_BACKEND_HARDWARE].enabled)
        p.download_bitstream = true;
}

// Destroy all handles and buffers.
void per_gpu::destroy()
{
    for (per_thread& pt : per_threads) {
        for (int i = 0; i < 2; ++i) {
            pt.img[i].deallocateInputDeviceBuf(*this, pt);
            pt.img[i].deallocateInputHostBuf();
            pt.img[i].deallocateOutputHostBuf();
        }
        CHECK_CUDA(cudaStreamDestroy(pt.stream));
        for (int b = 0; b < NUM_BACKENDS; ++b) {
            if (backends[b].enabled && pt.slots[b].encode_state)
                CHECK_NVJPEG(nvjpegEncoderStateDestroy(pt.slots[b].encode_state));
        }
    }
    if (eparams)
        CHECK_NVJPEG(nvjpegEncoderParamsDestroy(eparams));
    if (handle)
        CHECK_NVJPEG(nvjpegDestroy(handle));
}

// Preallocate all host/device buffers (good for benchmarking purposes).
void per_gpu::preAllocateBuffers(const std::string& fileName, const params& p, const per_gpu& pg)
{
    img_t templateImg;
    templateImg.readFromBmpFile(fileName, pg);

    for (per_thread& pt : per_threads) {
        for (int i = 0; i < 2; ++i) {
            pt.img[i].copyFrom(templateImg);
            pt.img[i].allocateInputDeviceBuf(*this, pt);
            pt.img[i].allocateOutputHostBuf(pt.img[i].padded_dev_buf_size);
            pt.img[i].uploadToDevice(pt);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    templateImg.deallocateInputHostBuf();
}

// Compare two strings using natural sorting order (e.g., 1 < 2 < 10 < 20 < 100).
bool naturalCompare(std::string_view a, std::string_view b)
{
    auto ai = a.begin(), bi = b.begin();
    while (ai != a.end() && bi != b.end()) {
        if (std::isdigit(*ai) && std::isdigit(*bi)) {
            while (ai != a.end() && *ai == '0') ++ai;
            while (bi != b.end() && *bi == '0') ++bi;
            auto aj = ai, bj = bi;
            while (aj != a.end() && std::isdigit(*aj)) ++aj;
            while (bj != b.end() && std::isdigit(*bj)) ++bj;
            if (aj - ai != bj - bi) return (aj - ai) < (bj - bi);
            while (ai != aj) {
                if (*ai != *bi) return *ai < *bi;
                ++ai; ++bi;
            }
        }
        else {
            if (*ai != *bi) return *ai < *bi;
            ++ai; ++bi;
        }
    }
    return ai == a.end() && bi != b.end();
}

// Read the input directory to gather all the BMP file names.
std::vector<std::string> getFileNames(params& p)
{
    std::vector<std::string> results;
    results.reserve(128);
    if (!p.input_dir.empty()) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(p.input_dir)) {
            auto ext = entry.path().extension().string();
            if (ext == ".bmp" || ext == ".BMP")
                results.emplace_back(entry.path().string());
        }
        std::sort(results.begin(), results.end(), naturalCompare);
        if (p.total_images > 0 && results.size() > p.total_images)
            results.resize(p.total_images);
        if (p.total_images == 0)
            p.total_images = results.size();
    }
    p.preallocate = results.size() == 1;
    return results;
}

struct encode_result
{
    std::string device_name;
    int num_engines = 0;
    float latency[NUM_BACKENDS] = {};
    float throughput[NUM_BACKENDS] = {};
    float percentage[NUM_BACKENDS] = {};
    float total_throughput = 0;
    int num_threads = 1;
    int num_states = 1;
    unsigned int width = 0;
    unsigned int height = 0;
};

// Encode all images.
encode_result encodeAllImages(params& p)
{
    encode_result result;

    std::vector<std::string> fileNames = getFileNames(p);
    CHECK(!fileNames.empty(), "No BMP files in input directory.");
    per_gpu pg;
    pg.create(p);
    if (p.preallocate)
        pg.preAllocateBuffers(fileNames[0], p, pg);

    std::barrier sync_point(p.num_threads + 1);
    std::vector<std::thread> threads;
    threads.reserve(p.num_threads);

    for (int tid = 0; tid < p.num_threads; ++tid) {
        threads.emplace_back([&p, &pg, &fileNames, &sync_point, tid]() {
            sync_point.arrive_and_wait();

            int base = p.total_images / p.num_threads;
            int extra = p.total_images % p.num_threads;
            int numImagesPerThread = base + (tid < extra ? 1 : 0);
            int startImg = tid * base + std::min(tid, extra);
            int endImg = startImg + numImagesPerThread;

            for (int i = startImg, sid = tid; i < endImg; ++i, sid += p.num_threads) {
                sid = sid >= p.num_states ? tid : sid;
                per_thread& pt = pg.per_threads[sid];
                pt.img_slot = pt.nextImageSlot();
                img_t& img = pt.img[pt.img_slot];
                img.img_id = i;

                const auto& fileName = fileNames[i % fileNames.size()];
                img.readFromBmpFile(fileName, pg);
                img.allocateInputDeviceBuf(pg, pt);
                int encodeSlot = pt.findEncodeSlot(p, pg, img, sid);
                per_thread_encode_slot& es = pt.slots[encodeSlot];
                img.encodeHost(es, p);
                CHECK_CUDA(cudaStreamSynchronize(pt.stream));
                if (pt.prevEs) {
                    pt.pg->backends[pt.prevEs->backend].update(*pt.prevEs);
                    if (p.download_bitstream)
                        pt.prevEs->img->retrieveBitStreamToHost(*pt.prevEs);
                    if (p.write_output) {
                        CHECK_CUDA(cudaStreamSynchronize(pt.stream));
                        pt.prevEs->img->writeJpegToFile(p);
                    }
                }
                es.img = &img;
                pt.prevEs = &es;
                if (!p.preallocate)
                    img.uploadToDevice(pt);
                img.encodeDevice(es, p);
            }

            for (int sid = tid; sid < p.num_states; sid += p.num_threads) {
                per_thread& pt = pg.per_threads[sid];
                CHECK_CUDA(cudaStreamSynchronize(pt.stream));
                if (pt.prevEs) {
                    pt.pg->backends[pt.prevEs->backend].update(*pt.prevEs);
                    if (p.download_bitstream)
                        pt.prevEs->img->retrieveBitStreamToHost(*pt.prevEs);
                    if (p.write_output) {
                        CHECK_CUDA(cudaStreamSynchronize(pt.stream));
                        pt.prevEs->img->writeJpegToFile(p);
                    }
                }
            }
            });
    }

    sync_point.arrive_and_wait();
    auto start = Clock::now();
    for (auto& thread : threads)
        thread.join();
    cudaDeviceSynchronize();
    auto end = Clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    for (int b = 0; b < NUM_BACKENDS; ++b) {
        result.throughput[b] = pg.backends[b].throughput;
        if (pg.backends[b].num_jobs_done > 0)
            result.latency[b] = pg.backends[b].total_latency / pg.backends[b].num_jobs_done;
        result.percentage[b] = pg.backends[b].num_jobs_done * 100.f / p.total_images;
    }
    result.total_throughput = (p.total_images) * 1000.0 / duration.count();
    result.num_states = p.num_states;
    result.num_threads = p.num_threads;
    result.device_name = pg.device_name;
    result.num_engines = pg.num_engines;

    pg.destroy();

    return result;
}

// Find the index of a param in the command used to launch this program (e.g., "-s").
int findParamIndex(const char** argv, int argc, const char* param)
{
    int count = 0;
    int index = -1;

    for (int i = 0; i < argc; i++) {
        if (strncmp(argv[i], param, 100) == 0) {
            index = i;
            count++;
        }
    }

    if (count == 0 || count == 1) {
        return index;
    }
    else {
        std::cerr << "Error, parameter " << param
            << " has been specified more than once, exiting\n";
        return -1;
    }

    return -1;
}

// Example usage:
// Encode all images from directory "img"
// -i img
// Encode all images from directory "img" using 1 thread 
// -i img -j 1
// Encode 4000 images (with potential repetitions) from directory "img" (add -n)
// For benchmarking purposes, it is recommended to put just one image in the "img" directory.
// This image will be encoded 4000 times.
// -i img -n 4000
// Write JPEG outputs to the current directory (add -o)
// -i img -o .
// Set JPEG quality (add -q), default is 80
// -i img -q 70
// Enable optimized Huffman (add -f), default is off
// -i img -f 
// Disable chroma subsampling (add -p), default is on (use 4:2:0)
// -i img -p
// Use only the hardware backend (add -b) (NOTE: this only works on Thor)
// -i img -b hardware
// Use both gpu and hardware backends (with automatic load-balancing) (NOTE: this only works on Thor)
// -i img -b gpu hardware
// Use all supported backends (the same as not providing -b)
// -i img
// Use as many threads as CPU hardware threads (-j 0)
// -i img -j 0
// Use two threads and 8 states (add -s)
// -i img -j 2 -s 8
// Use one thread and twice as many states as hardware JPEG encode engines (provide -s 0 or just omit it)
// -i img -j 1
// -i img -j 1 -s 0
// Automatically detect the best number of states and CPU threads to use (omit -j)
// -i img
// Automatically detect the best number of CPU threads to use but start with 8 states
// -i img -s 8
// Benchmark individual backends (add -r to run multiple times and pick the best)
// -i img -n 4000 -b gpu -r 4
// -i img -n 4000 -b hardware -r 4
// Benchmark automatic load-balancing among backends
// -i img -n 4000 -b gpu hardware -r 4
int main(int argc, const char* argv[])
{
    int pidx;
    if (argc < 2 || (pidx = findParamIndex(argv, argc, "-h")) != -1) {
        std::cerr << "Usage: " << argv[0] << " -i indir [-n nimages] [-s nstates] [-j nthreads] [-b backends] [-r nruns] [-d] [-o outdir] [-q quality] [-f] [-p]\n";
        std::cerr << "Parameters:\n";
        std::cerr << "  (OPTIONAL)  -i indir: Directory to take BMP images from.\n";
        std::cerr << "  (OPTIONAL)  -n nimages: Number of images to encode.\n"
                  << "               If not provided, encode all images in the input directory.\n"
                  << "               Will be adjusted to be at least the number of states.\n";
        std::cerr << "  (OPTIONAL)  -s nstates: Number of states.\n"
                  << "               If not provided, use twice the number of hardware encode engines, if present.\n"
                  << "               Will be adjusted to be at least the number of threads.\n";
        std::cerr << "  (OPTIONAL)  -j nthreads: Number of CPU threads.\n"
                  << "               If not provided, automatically find the best number of threads to use.\n"
                  << "               Use 0 to set to the number of CPU hardware threads on the system.\n";
        std::cerr << "  (OPTIONAL)  -b backends: any of gpu/hardware/gpu hardware\n";
        std::cerr << "  (OPTIONAL)  -r nruns: Run this many times and pick the one with the maximum throughput.\n";
        std::cerr << "  (OPTIONAL)  -o outdir: Directory to write encoded JPEG images.\n";
        std::cerr << "  (OPTIONAL)  -d: Download JPEG bitstream to the host.\n";
        std::cerr << "  (OPTIONAL)  -q quality: JPEG quality 1-100 (default 80).\n";
        std::cerr << "  (OPTIONAL)  -f: Enable optimized Huffman.\n";
        std::cerr << "  (OPTIONAL)  -p: Disable chroma subsampling.\n";
        return 0;
    }

    params p;

    bool autoThreads = false;
    if ((pidx = findParamIndex(argv, argc, "-j")) != -1) {
        CHECK(pidx + 1 < argc, "Number of threads not provided after -j\n");
        p.num_threads = std::atoi(argv[pidx + 1]);
        if (p.num_threads == 0)
            p.num_threads = std::thread::hardware_concurrency();
    }
    else {
        autoThreads = true;
    }

    if ((pidx = findParamIndex(argv, argc, "-s")) != -1) {
        CHECK(pidx + 1 < argc, "Need num states provided after -s\n");
        p.num_states = std::atoi(argv[pidx + 1]);
    }
    CHECK(p.num_states >= 0, "Need non-negative total number of states\n");

    if ((pidx = findParamIndex(argv, argc, "-n")) != -1) {
        CHECK(pidx + 1 < argc, "No total images provided after -n\n");
        p.total_images = std::max(0, std::atoi(argv[pidx + 1]));
    }

    if ((pidx = findParamIndex(argv, argc, "-r")) != -1) {
        CHECK(pidx + 1 < argc, "No runs provided after -r\n");
        p.num_runs = std::max(1, std::atoi(argv[pidx + 1]));
    }

    if ((pidx = findParamIndex(argv, argc, "-i")) != -1) {
        CHECK(pidx + 1 < argc, "No input dir provided after -i\n");
        p.input_dir = argv[pidx + 1];
    }
    CHECK(pidx != -1, "Need to provide the input directory with -i\n");

    p.download_bitstream = findParamIndex(argv, argc, "-d") != -1;

    if ((pidx = findParamIndex(argv, argc, "-o")) != -1) {
        CHECK(pidx + 1 < argc, "No output dir provided after -o\n");
        p.output_dir = argv[pidx + 1];
        p.write_output = true;
        p.download_bitstream = true;
    }

    if ((pidx = findParamIndex(argv, argc, "-q")) != -1) {
        CHECK(pidx + 1 < argc, "No quality provided after -q\n");
        p.quality = std::max(1, std::min(100, std::atoi(argv[pidx + 1])));
    }

    p.optimized_huffman = findParamIndex(argv, argc, "-f") != -1;
    p.subsampling_420 = findParamIndex(argv, argc, "-p") == -1;

    p.backend_enabled[0] = true;
    if ((pidx = findParamIndex(argv, argc, "-b")) != -1) {
        CHECK(pidx + 1 < argc, "No backends provided after -b\n");

        for (int i = pidx + 1; i < argc && argv[i][0] != '-'; ++i) {
            const char* backend = argv[i];
            if (strcmp(backend, "gpu") == 0)
                p.backend_enabled[NVJPEG_ENC_BACKEND_GPU] = true;
            else if (strcmp(backend, "hardware") == 0)
                p.backend_enabled[NVJPEG_ENC_BACKEND_HARDWARE] = true;
            else {
                std::cerr << "Unknown backend: " << backend << "\n";
                return 1;
            }
        }
    }
    else {
        p.backend_enabled[NVJPEG_ENC_BACKEND_GPU] = true;
        p.backend_enabled[NVJPEG_ENC_BACKEND_HARDWARE] = true;
    }

    int numBackends = 0;
    int onlyBackend = -1;
    for (int b = 1; b < NUM_BACKENDS; ++b) {
        if (p.backend_enabled[b]) {
            onlyBackend = b;
            ++numBackends;
        }
    }
    if (numBackends == 1)
        p.only_backend = onlyBackend;

    if (p.write_output)
        std::filesystem::create_directories(p.output_dir);

    CHECK_CUDA(cudaDeviceReset());

    encode_result bestResult;
    int stage = 0;
    if (autoThreads) {
        p.num_threads = std::max(p.num_states, p.num_threads);
        if (p.num_states != 0)
            p.num_states = std::max(p.num_states, p.num_threads);
        while (true) {
            encode_result currentBest;
            for (int t = 0; t < p.num_runs; ++t) {
                encode_result result = encodeAllImages(p);
                if (result.total_throughput > currentBest.total_throughput)
                    currentBest = result;
            }

            if (currentBest.total_throughput < bestResult.total_throughput) {
                p.num_states = bestResult.num_states;
                p.num_threads = bestResult.num_threads;
                if (++stage >= 2)
                    break;
            }
            else {
                bestResult = currentBest;
            }
            p.num_threads += (stage == 0) ? p.num_threads : 1;
            p.num_states = std::max(p.num_states, p.num_threads);
        }
    }
    for (int i = 0; i < p.num_runs; ++i) {
        encode_result result = encodeAllImages(p);
        if (result.total_throughput > bestResult.total_throughput)
            bestResult = result;
    }
    CHECK_CUDA(cudaDeviceReset());

    std::cout << "------------------------------------------------\n";
    std::cout << "GPU: " << bestResult.device_name << "\n";
    std::cout << "Num hardware encode engines: " << bestResult.num_engines << "\n";
    std::cout << "Input: " << p.input_dir << "\n";
    std::cout << "Enabled backends:";
    if (p.backend_enabled[1])
        std::cout << " gpu";
    if (p.backend_enabled[2])
        std::cout << " hardware";
    std::cout << "\n";
    std::cout << "Subsampling 420: " << (p.subsampling_420 ? "on" : "off") << "\n";
    std::cout << "Optimized Huffman: " << (p.optimized_huffman ? "on" : "off") << "\n";
    std::cout << "Quality: " << p.quality << "\n";
    std::cout << "Throughput: " << bestResult.total_throughput << " images/s\n";
    std::cout << "Latency: " << bestResult.latency[1] << ", " << bestResult.latency[2] << " ms\n";
    std::cout << "Percentage: " << bestResult.percentage[1] << ", " << bestResult.percentage[2] << " %\n";
    std::cout << "Num threads: " << bestResult.num_threads << "\n";
    std::cout << "Num states: " << bestResult.num_states << "\n";
    std::cout << "Num runs: " << p.num_runs << "\n";

    return 0;
}