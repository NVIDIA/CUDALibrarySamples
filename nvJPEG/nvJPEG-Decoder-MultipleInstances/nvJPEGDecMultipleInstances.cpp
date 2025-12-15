/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
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

constexpr int NUM_BACKENDS = 4;

// Parameters provided to the application.
struct params 
{
    bool backend_enabled[NUM_BACKENDS] = {};
    int only_backend = 0;
    nvjpegBackend_t backend = NVJPEG_BACKEND_DEFAULT;
    nvjpegOutputFormat_t fmt = NVJPEG_OUTPUT_RGBI;
    std::string input_dir = "";
    std::string output_dir = ".";
    int total_images = 0;
    int num_states = 0;
    int num_threads = 0;
    int num_runs = 1;
    bool write_output = false;
};

struct per_gpu;
struct per_thread;
struct per_thread_decode_slot;

struct per_gpu;
struct per_thread;

using Clock = std::chrono::steady_clock;
using namespace std::chrono;

// Keep all the data revelant to a JPEG image.
// Here, we assume the decode output format is interleaved RGB (NVJPEG_OUTPUT_RGBI).
struct img_t 
{
    std::string file_name;
    nvjpegJpegStream_t jpeg = nullptr;
    int jpeg_bit_stream_size = 0;
    unsigned char* input_host_buf = nullptr;
    int input_host_buf_size = 0;
    unsigned char* output_host_buf = nullptr;
    int output_host_buf_size = 0;
    int dev_buf_size = 0;
    int padded_dev_buf_size = 0;
    nvjpegImage_t dev_img = {}; // device image buffer
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    nvjpegChromaSubsampling_t ss = NVJPEG_CSS_444;
    nvjpegJpegEncoding_t encoding = NVJPEG_ENCODING_BASELINE_DCT;
    int img_id = 0;
    Clock::time_point start;
    Clock::time_point stop;

    void copyFrom(const img_t& src);

    static int padSize(int size);
    static int padDimension(const per_gpu& pg, int dim);

    double getTotalPixels() const;
    
    // Decode
    void readFromJpegFile(const std::string& fileName);
    void parseStream(per_gpu& pg);
    void decodeHost(per_thread_decode_slot& ds);
    void transferToDevice(per_thread_decode_slot& ds);
    void decodeDevice(per_thread_decode_slot& ds);
    void optionalDownload(const params& p, per_thread& pt);
    void optionalWriteBmp(const params& p);
    
    // Allocation
    void allocateDeviceOutputBuf(per_gpu& pg, nvjpegOutputFormat_t fmt, per_thread& pt);
    void deallocateDeviceOutputBuf(per_gpu& pg, per_thread& pt);
    void allocateInputHostBuf(int size);
    void deallocateInputHostBuf();
    void optionalAllocateHostOutputBuf(const params& p, int size);
    void deallocateOutputHostBuf();
};

struct per_thread_decode_slot 
{
    per_thread* pt = nullptr;
    img_t* img = nullptr;
    nvjpegBackend_t backend = NVJPEG_BACKEND_DEFAULT;
    nvjpegJpegDecoder_t decoder = nullptr;
    nvjpegJpegState_t jpeg_state = nullptr;
};

// Each per_thread object is guaranteed to not be shared among host threads.
// However, one thread can use multiple per_thread objects.
struct per_thread 
{
    per_gpu* pg = nullptr;
    std::array<per_thread_decode_slot, NUM_BACKENDS> slots;
    cudaStream_t stream = nullptr;
    nvjpegBufferDevice_t device_buffer = nullptr;
    nvjpegBufferPinned_t pinned_buffer[2] = {}; // double-buffered
    img_t img[2]; // double-buffered
    int img_slot = -1;

    int findDecodeSlot(const params& p, per_gpu& pg, const img_t& img, int sid);
    int nextImageSlot();

    per_thread_decode_slot* prevDs = nullptr;
};

// Keep track of throughput and latency for each backend.
struct backend_t 
{
    nvjpegJpegDecoder_t decoder = nullptr;
    std::atomic<int> num_jobs_started = 0;
    std::atomic<int> num_jobs_done = 0;
    std::atomic<double> pixels_started = 0;
    std::atomic<double> pixels_done = 0;
    nvjpegBackend_t backend = NVJPEG_BACKEND_DEFAULT;
    Clock::time_point start;
    float elapsed_time = 0;
    float throughput = 0; // (num_jobs_done) / (stop - start)
    float max_throughput = 0.0f; // best throughput (pixels/s) found during calibration
    int stable_counter = 0; // #images since last improvement of throughput
    float drain_time = FLT_MAX; // (num_jobs_start - num_jobs_done + 1) / throughput;
    std::atomic<float> total_latency = 0;
    std::atomic<bool> calibration_done = false;
    std::mutex update_mutex;
    bool enabled = false;

    bool isSupported(const img_t& img) const;
    void update(per_thread_decode_slot& ds);
};

// Main "mother" struct.
struct per_gpu {
    std::string device_name;
    nvjpegHandle_t handle = nullptr;
    std::array<backend_t, NUM_BACKENDS> backends = {};
    std::array<int, NUM_BACKENDS> backend_order = {};
    nvjpegDecodeParams_t decode_params = nullptr;
    std::vector<per_thread> per_threads;
    int device_id = 0;
    int async_malloc_supported = 0;
    int pitch_alignment = 256;
    int num_engines = 0;
    float min_drain_time = FLT_MAX;
    int min_backend = 0;
    std::atomic<bool> calibration_done = false;

    void create(params& p);
    void destroy();
    void preAllocateOutputBuffers(const std::string& fileName, const params& p);
};

// Allocate all buffers and handles.
void per_gpu::create(params& p) 
{
    CHECK_CUDA(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    device_name = prop.name;
    CHECK_CUDA(cudaDeviceGetAttribute(&async_malloc_supported, cudaDevAttrMemoryPoolsSupported, device_id));
    //std::cout << "Async malloc supported: " << async_malloc_supported << "\n";
    CHECK_CUDA(cudaDeviceGetAttribute(&pitch_alignment, cudaDevAttrTexturePitchAlignment, device_id));
    //std::cout << "Pitch alignment: " << pitch_alignment << "\n";
    CHECK_NVJPEG(nvjpegCreateSimple(&handle));
    CHECK_NVJPEG(nvjpegSetDeviceMemoryPadding(1024 * 1024, handle));
    CHECK_NVJPEG(nvjpegSetPinnedMemoryPadding(1024 * 1024, handle));
    unsigned int nEngines = 1, nCoresPerEngine = 1;
    CHECK_NVJPEG(nvjpegGetHardwareDecoderInfo(handle, &nEngines, &nCoresPerEngine));
    num_engines = nEngines;
    if (p.num_states == 0)
        p.num_states = std::max({1, num_engines * 2, p.num_threads});
    if (p.num_threads == 0)
        p.num_threads = p.num_states;
    p.total_images = std::max(p.total_images, p.num_states);
    //std::cout << "Number of hardware engines: " << nEngines << "\n";
    for (int b = 0; b < NUM_BACKENDS; ++b) {
        if (p.backend_enabled[b]) {
            // Sometimes the user can request a backend but it's not available on the platform
            // So we check that and potentially override the setting here
            bool be = NVJPEG_STATUS_SUCCESS == nvjpegDecoderCreate(handle, nvjpegBackend_t(b), &backends[b].decoder);
            p.backend_enabled[b] = backends[b].enabled = be;
        }
    }
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle, &decode_params));
    per_threads.resize(p.num_states);
    for (int s = 0; s < p.num_states; ++s) {
        per_thread& pt = per_threads[s];
        pt.pg = this;
        for (int i = 0; i < 2; ++i) {
            CHECK_NVJPEG(nvjpegBufferPinnedCreate(handle, nullptr, &pt.pinned_buffer[i]));
            CHECK_NVJPEG(nvjpegJpegStreamCreate(handle, &pt.img[i].jpeg));
        }
        CHECK_NVJPEG(nvjpegBufferDeviceCreate(handle, nullptr, &pt.device_buffer));
        CHECK_CUDA(cudaStreamCreateWithFlags(&pt.stream, cudaStreamNonBlocking));
            
        for (int b = 0; b < NUM_BACKENDS; ++b) {
            if (backends[b].enabled) {
                per_thread_decode_slot& ds = pt.slots[b];
                ds.pt = &pt;
                ds.decoder = backends[b].decoder;
                ds.backend = backends[b].backend = nvjpegBackend_t(b);
                CHECK_NVJPEG(nvjpegDecoderStateCreate(handle, ds.decoder, &ds.jpeg_state));
                CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(ds.jpeg_state, pt.device_buffer));
            }
        }
    }

    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(decode_params, p.fmt));
}

// Deallocate all buffers and handles.
void per_gpu::destroy() 
{
    for (per_thread& pt : per_threads) {
        for (int i = 0; i < 2; ++i) {
            pt.img[i].deallocateDeviceOutputBuf(*this, pt);
            pt.img[i].deallocateInputHostBuf();
            pt.img[i].deallocateOutputHostBuf();
            CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pt.pinned_buffer[i]));
            CHECK_NVJPEG(nvjpegJpegStreamDestroy(pt.img[i].jpeg));
        }
        CHECK_NVJPEG(nvjpegBufferDeviceDestroy(pt.device_buffer));
        CHECK_CUDA(cudaStreamDestroy(pt.stream));
        for (int b = 0; b < NUM_BACKENDS; ++b) {
            if (backends[b].enabled) {
                per_thread_decode_slot& ls = pt.slots[b];
                CHECK_NVJPEG(nvjpegJpegStateDestroy(ls.jpeg_state));
            }
        }
    }
    CHECK_NVJPEG(nvjpegDecodeParamsDestroy(decode_params));
    for (int b = 1; b < NUM_BACKENDS; ++b) {
        if (backends[b].enabled)
            CHECK_NVJPEG(nvjpegDecoderDestroy(backends[b].decoder));
    }
    CHECK_NVJPEG(nvjpegDestroy(handle));
}

// Preallocate all image buffers when there's only one source image.
void per_gpu::preAllocateOutputBuffers(const std::string& fileName, const params& p) 
{
    assert(p.fmt == NVJPEG_OUTPUT_RGBI);

    //std::cerr << "Pre-allocate image buffers: ON\n";
    img_t templateImg;
    templateImg.readFromJpegFile(fileName);
    int ncomps = 0;
    nvjpegChromaSubsampling_t ss = NVJPEG_CSS_444;
    CHECK_NVJPEG(nvjpegGetImageInfo(handle, templateImg.input_host_buf, templateImg.jpeg_bit_stream_size, 
                                    &ncomps, &ss, templateImg.widths, templateImg.heights));
    for (per_thread& pt : per_threads) {
        for (int i = 0; i < 2; ++i) {
            img_t& img = pt.img[i];
            img.copyFrom(templateImg);
            img.allocateDeviceOutputBuf(*this, p.fmt, pt);
            img.optionalAllocateHostOutputBuf(p, img.padded_dev_buf_size);
        }
    }
    templateImg.deallocateInputHostBuf();
}

// Update the latest throughput and latency.
// This function is called after each image is decoded.
void backend_t::update(per_thread_decode_slot& ds)
{
    assert(backend == ds.backend);
    //std::cerr << "backend " << backend << "\n";
    pixels_done += ds.img->getTotalPixels();
    ++num_jobs_done;
    ds.img->stop = Clock::now();
    float latency = duration<float, std::milli>(ds.img->stop - ds.img->start).count();
    total_latency += latency;
    float time = duration<float, std::milli>(ds.img->stop - start).count();

    std::lock_guard<std::mutex> lock(update_mutex);

    if (stable_counter < 8) {
        float current_throughput = pixels_done / time;
        //std::cerr << "max throughput = " << max_throughput <<"\n";
        if (current_throughput > max_throughput * 1.0001f) { // small epsilon for noise
            max_throughput = current_throughput;
            stable_counter = 0;
        } else {
            ++stable_counter;
        }
    }

    if (time > elapsed_time) {
        elapsed_time = time;
        float avgLatency = total_latency / num_jobs_done;
        throughput = pixels_done / time;
    }
}

// Return true if the backend supports the given image.
bool backend_t::isSupported(const img_t& img) const
{
    if (backend == NVJPEG_BACKEND_HARDWARE) {
        if (img.widths[0] >= 16384 || img.heights[0] >= 16384 ||
            (img.encoding != NVJPEG_ENCODING_BASELINE_DCT && 
             img.encoding != NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN) ||
            (img.ss != NVJPEG_CSS_444 && img.ss != NVJPEG_CSS_422 && img.ss != NVJPEG_CSS_420 && 
             img.ss != NVJPEG_CSS_440 && img.ss != NVJPEG_CSS_GRAY))
        {
            return false;
        }
    }
    return true;
}

// Return the next image slot.
int per_thread::nextImageSlot()
{
    return (img_slot + 1) % 2;
}

// Find a backend to decode the given image.
int per_thread::findDecodeSlot(const params& p, per_gpu& pg, const img_t& img, int sid)
{
    if (p.only_backend)
        return p.only_backend;

    // Phase 1: calibration, determine the best throughput for each backend
    if (!pg.calibration_done) {
        // return the first enabled backend that is not yet fully estimated
        for (int b = 1; b < NUM_BACKENDS; ++b) {
            backend_t& be = pg.backends[b];
            if (be.enabled && be.stable_counter < 8 && be.isSupported(img))
                return b;
        }
        // if we reach here, all enabled backends are calibrated
        // sort and mark calibration_done once
        bool expected = false;
        if (pg.calibration_done.compare_exchange_strong(expected, true)) {
            float a[3] = {
                pg.backends[1].max_throughput,
                pg.backends[2].max_throughput,
                pg.backends[3].max_throughput
            };
            //std::cerr << "estimated throughputs: " << a[0] << " " << a[1] << " " << a[2] << "\n";
            int imx = (a[0] >= a[1]) ? (a[0] >= a[2] ? 0 : 2) : (a[1] >= a[2] ? 1 : 2); // max
            int imn = (a[0] < a[1]) ? (a[0] < a[2] ? 0 : 2) : (a[1] <= a[2] ? 1 : 2); // min
            int imm = 3 - imx - imn; // middle
            pg.backend_order[0] = imx + 1;
            pg.backend_order[1] = imm + 1;
            pg.backend_order[2] = imn + 1;
        }
    }

    // Phase 2: calibration is done, now try to allocate backend
    // from the best throughput to the worst. If the "backlog"
    // is greater than twice the number of threads, "spill-over"
    // to the next backend.
    int nextBackend = 0;
    for (int i = 0; i < NUM_BACKENDS; ++i) {
        int b = pg.backend_order[i];
        backend_t& be = pg.backends[b];
        if (be.enabled && (be.num_jobs_done - be.num_jobs_started < p.num_threads * 2) && be.isSupported(img)) {
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
            if (be.enabled && be.isSupported(img)) {
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

#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t type = 0x4D42;  // "BM"
    uint32_t fileSize;
    uint32_t reserved = 0;
    uint32_t offsetData = 54;
};

struct BMPInfoHeader {
    uint32_t size = 40;
    int32_t width;
    int32_t height;
    uint16_t planes = 1;
    uint16_t bitCount = 24;
    uint32_t compression = 0;
    uint32_t sizeImage;
    int32_t xPelsPerMeter = 0;
    int32_t yPelsPerMeter = 0;
    uint32_t clrUsed = 0;
    uint32_t clrImportant = 0;
};
#pragma pack(pop)

// Write BMP data to a file on disk.
static void writeBmp(const char* fileName, unsigned char* data, int pitchInBytes, uint32_t width, uint32_t height) 
{
    FILE* fp = fopen(fileName, "wb");
    if (!fp)
        return;

    uint32_t extraBytes = (4 - ((width * 3) % 4)) % 4;
    uint32_t rowSize = width * 3 + extraBytes;
    uint32_t paddedSize = rowSize * height;
    uint32_t fileSize = paddedSize + 54;

    BMPFileHeader fileHeader;
    fileHeader.fileSize = fileSize;

    BMPInfoHeader infoHeader;
    infoHeader.width = width;
    infoHeader.height = height;
    infoHeader.sizeImage = paddedSize;

    fwrite(&fileHeader, sizeof(fileHeader), 1, fp);
    fwrite(&infoHeader, sizeof(infoHeader), 1, fp);

    std::vector<unsigned char> rowBuffer(rowSize);

    for (uint32_t y = height; y-- > 0;) {
        for (uint32_t x = 0; x < width; x++) {
            int red = data[0 + y * pitchInBytes + 3 * x];
            int green = data[1 + y * pitchInBytes + 3 * x];
            int blue = data[2 + y * pitchInBytes + 3 * x];
            if (red > 255) red = 255; if (red < 0) red = 0;
            if (green > 255) green = 255; if (green < 0) green = 0;
            if (blue > 255) blue = 255; if (blue < 0) blue = 0;
            rowBuffer[x * 3] = blue;
            rowBuffer[x * 3 + 1] = green;
            rowBuffer[x * 3 + 2] = red;
        }

        for (int i = 0; i < extraBytes; i++)
            rowBuffer[width * 3 + i] = 0;
        fwrite(rowBuffer.data(), 1, rowSize, fp);
    }

    fclose(fp);
}

// Copy data from the source image.
void img_t::copyFrom(const img_t& src) 
{
    file_name = src.file_name;
    if (src.input_host_buf && src.input_host_buf_size > 0) {
        int requiredSize = src.input_host_buf_size;
        if (input_host_buf_size < requiredSize) {
            if (input_host_buf)
                CHECK_CUDA(cudaFreeHost(input_host_buf));
            CHECK_CUDA(cudaMallocHost((void**)&input_host_buf, requiredSize));
            input_host_buf_size = requiredSize;
        }
        std::memcpy(input_host_buf, src.input_host_buf, src.input_host_buf_size);
    }
    jpeg_bit_stream_size = src.jpeg_bit_stream_size;
    
    for (int i = 0; i < 3; ++i) {
        widths[i] = src.widths[i];
        heights[i] = src.heights[i];
    }
}

// Pad the size of a JPEG bitstream to multiples of 1 MB.
int img_t::padSize(int size)
{
    constexpr int oneMb = 1 * 1024 * 1024;
    int paddedSize = oneMb * ((size + oneMb - 1) / oneMb);
    return paddedSize;
}

// Pad the dimension of an image to multiples of 512.
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

// Read a JPEG file from disk.
void img_t::readFromJpegFile(const std::string& fileName) 
{
    if (file_name != fileName) {
        file_name = fileName;
        FILE* inputStream = fopen(fileName.c_str(), "rb");
        fseek(inputStream, 0, SEEK_END);
        jpeg_bit_stream_size = ftell(inputStream);
        fseek(inputStream, 0, SEEK_SET);
        allocateInputHostBuf(jpeg_bit_stream_size);
        fread(input_host_buf, jpeg_bit_stream_size, 1, inputStream);
        fclose(inputStream);
    }
}

// Allocate the device buffer to store the decompressed image.
void img_t::allocateDeviceOutputBuf(per_gpu& pg, nvjpegOutputFormat_t fmt, per_thread& pt)
{
    assert(fmt == NVJPEG_OUTPUT_RGBI);

    int ncomps = 0;
    nvjpegChromaSubsampling_t ss = NVJPEG_CSS_444;
    CHECK_NVJPEG(nvjpegGetImageInfo(pg.handle, input_host_buf, jpeg_bit_stream_size, &ncomps, &ss, widths, heights));
    
    int w = widths[0], h = heights[0];
    int paddedPitch = padDimension(pg, w) * 3;
    int imageSize = paddedPitch * h;

    if (imageSize > padded_dev_buf_size) {
        deallocateDeviceOutputBuf(pg, pt);
        if (pg.async_malloc_supported)
            CHECK_CUDA(cudaMallocAsync((void**)&dev_img.channel[0], imageSize, pt.stream));
        else
            CHECK_CUDA(cudaMalloc((void**)&dev_img.channel[0], imageSize));
        padded_dev_buf_size = imageSize;
    }

    dev_buf_size = imageSize;
    dev_img.pitch[0] = paddedPitch;
}

// Deallocate the device buffer storing the decompressed image.
void img_t::deallocateDeviceOutputBuf(per_gpu& pg, per_thread& pt)
{
    if (dev_img.channel[0]) {
        if (pg.async_malloc_supported)
            CHECK_CUDA(cudaFreeAsync(dev_img.channel[0], pt.stream));
        else
            CHECK_CUDA(cudaFree(dev_img.channel[0]));
    }
}

// Allocate host buffer to store the input JPEG bitstream.
void img_t::allocateInputHostBuf(int size)
{
    if (input_host_buf_size < padSize(size)) {
        deallocateInputHostBuf();
        input_host_buf_size = padSize(size);
        CHECK_CUDA(cudaMallocHost((void**)&input_host_buf, input_host_buf_size));
    }
}

// Deallocate the host buffer storing the input JPEG bitstream.
void img_t::deallocateInputHostBuf()
{
    if (input_host_buf) {
        CHECK_CUDA(cudaFreeHost(input_host_buf));
        input_host_buf = nullptr;
    }
}

// Optionally allocate host output buffer to store the decoded image, prior to writing.
void img_t::optionalAllocateHostOutputBuf(const params& p, int size)
{
    if (p.write_output && output_host_buf_size < padSize(size)) {
        deallocateOutputHostBuf();
        output_host_buf_size = padSize(size);
        CHECK_CUDA(cudaMallocHost((void**)&output_host_buf, output_host_buf_size));
    }
}

// Deallocate the host buffer storing the decoded image.
void img_t::deallocateOutputHostBuf()
{
    if (output_host_buf) {
        CHECK_CUDA(cudaFreeHost(output_host_buf));
        output_host_buf = nullptr;
    }
}

// Parse the input JPEG bit stream for metadata.
void img_t::parseStream(per_gpu& pg)
{
    CHECK_NVJPEG(nvjpegJpegStreamParse(pg.handle, input_host_buf, input_host_buf_size, 0, 0, jpeg));
    CHECK_NVJPEG(nvjpegJpegStreamGetChromaSubsampling(jpeg, &ss));
    CHECK_NVJPEG(nvjpegJpegStreamGetJpegEncoding(jpeg, &encoding));
}

// Partially decode the image on the host.
void img_t::decodeHost(per_thread_decode_slot& ds)
{
    backend_t& be = ds.pt->pg->backends[ds.backend];
    int num_jobs_started = be.num_jobs_started.fetch_add(1);
    if (num_jobs_started == 0)
        be.start = Clock::now();
    start = Clock::now();
    auto pixels = getTotalPixels();
    be.pixels_started += pixels;
    CHECK_NVJPEG(nvjpegDecodeJpegHost(ds.pt->pg->handle, ds.decoder, ds.jpeg_state, ds.pt->pg->decode_params, jpeg));
}

// (ASYNC) Transfer the partially decoded results to the device to continue decoding.
void img_t::transferToDevice(per_thread_decode_slot& ds)
{
    CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(ds.pt->pg->handle, ds.decoder, ds.jpeg_state, jpeg, ds.pt->stream));
}

// (ASYNC) Decode on the device.
void img_t::decodeDevice(per_thread_decode_slot& ds)
{
    CHECK_NVJPEG(nvjpegDecodeJpegDevice(ds.pt->pg->handle, ds.decoder, ds.jpeg_state, &dev_img, ds.pt->stream));
}

// Optionally download the decoded image.
void img_t::optionalDownload(const params& p, per_thread& pt)
{
    if (p.write_output)
        CHECK_CUDA(cudaMemcpyAsync(output_host_buf, dev_img.channel[0], dev_buf_size, cudaMemcpyDeviceToHost, pt.stream));
}

// Optionally write the decoded output as BMP file.
void img_t::optionalWriteBmp(const params& p)
{
    if (p.write_output) {
        namespace fs = std::filesystem;
        auto outputPath = fs::path(p.output_dir) / (std::to_string(img_id) + ".bmp");
        writeBmp(outputPath.string().c_str(), output_host_buf, dev_img.pitch[0], widths[0], heights[0]);
    }
}

// Compare two strings using natural sorting order (e.g., 1 < 2 < 10 < 20 < 100).
bool naturalCompare(std::string_view a, std::string_view b) 
{
    auto ai = a.begin(), bi = b.begin();
    while (ai != a.end() && bi != b.end()) {
        if (std::isdigit(*ai) && std::isdigit(*bi)) {
            // skip leading zeros
            while (ai != a.end() && *ai == '0') ++ai;
            while (bi != b.end() && *bi == '0') ++bi;
            // find end of each number
            auto aj = ai, bj = bi;
            while (aj != a.end() && std::isdigit(*aj)) ++aj;
            while (bj != b.end() && std::isdigit(*bj)) ++bj;
            // compare by length first, then lexicographically
            if (aj - ai != bj - bi) return (aj - ai) < (bj - bi);
            while (ai != aj) {
                if (*ai != *bi) return *ai < *bi;
                ++ai; ++bi;
            }
        } else {
            if (*ai != *bi) return *ai < *bi;
            ++ai; ++bi;
        }
    }
    return ai == a.end() && bi != b.end();
}

// Read the input directory to gather all the JPEG file names.
std::vector<std::string> getFileNames(params& p)
{
    std::vector<std::string> results;
    results.reserve(128);
    if (!p.input_dir.empty()) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(p.input_dir)) {
            auto ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG")
                results.emplace_back(entry.path().string());
        }

        //std::random_device rd;
        //std::mt19937 g(rd());
        //std::shuffle(results.begin(), results.end(), g);
        std::sort(results.begin(), results.end(), naturalCompare);

        if (p.total_images > 0 && results.size() > p.total_images)
            results.resize(p.total_images);
        if (p.total_images == 0)
            p.total_images = results.size();
    }
    return results;
}

struct decode_result
{
    std::string device_name;
    int num_engines = 0;
    float latency[NUM_BACKENDS] = {};
    float throughput[NUM_BACKENDS] = {};
    float percentage[NUM_BACKENDS] = {};
    float total_throughput = 0;
    int num_threads = 1;
    int num_states = 1;
};

// Decode all images.
decode_result decodeAllImages(params& p)
{
    decode_result result;

    std::vector<std::string> fileNames = getFileNames(p);
    CHECK(!fileNames.empty(), "No BMP files in input directory.");
    per_gpu pg;
    pg.create(p);
    if (fileNames.size() == 1) 
        pg.preAllocateOutputBuffers(fileNames[0], p);

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
                const auto& fileName = fileNames[i % fileNames.size()];
                img.img_id = i;
                img.readFromJpegFile(fileName);
                img.parseStream(pg);
                img.allocateDeviceOutputBuf(pg, p.fmt, pt);
                img.optionalAllocateHostOutputBuf(p, img.padded_dev_buf_size);
                per_thread_decode_slot& ds = pt.slots[pt.findDecodeSlot(p, pg, img, sid)];
                CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(ds.jpeg_state, pt.pinned_buffer[pt.img_slot]));
                img.decodeHost(ds);
                CHECK_CUDA(cudaStreamSynchronize(pt.stream));
                if (pt.prevDs) {
                    pt.pg->backends[pt.prevDs->backend].update(*pt.prevDs);
                    pt.prevDs->img->optionalWriteBmp(p);
                }
                ds.img = &img;
                pt.prevDs = &ds;
                img.transferToDevice(ds);
                img.decodeDevice(ds);
                img.optionalDownload(p, pt);
            }

            for (int sid = tid; sid < p.num_states; sid += p.num_threads) {
                per_thread& pt = pg.per_threads[sid];
                CHECK_CUDA(cudaStreamSynchronize(pt.stream));
                if (pt.prevDs) {
                    pt.pg->backends[pt.prevDs->backend].update(*pt.prevDs);
                    pt.prevDs->img->optionalWriteBmp(p);
                }
            }
        });
    }
    
    // Start all threads
    sync_point.arrive_and_wait();
    auto start = Clock::now();
    for (auto& thread : threads)
        thread.join();
    cudaDeviceSynchronize();
    auto end = Clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    // Gather results
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
    
    //std::cout << "num threads " << result.num_threads << " num states " << result.num_states << " throughput " << result.total_throughput << "\n";
    return result;
}

// Find the index of a param in the command used to launch this program (e.g., "-s").
int findParamIndex(const char** argv, int argc, const char* param) {
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
    } else {
        std::cerr << "Error, parameter " << param
            << " has been specified more than once, exiting\n";
        return -1;
    }

    return -1;
}

// Example usage:
// Decode all images from directory "img"
// -i img
// Decode all images from directory "img" using 1 thread 
// -i img -j 1
// Decode 4000 images (with potential repetitions) from directory "img" (add -n)
// For benchmarking purposes, it is recommended to put just one image in the "img" directory.
// This image will be decoded 4000 times.
// -i img -n 4000
// Write outputs to the current directory (add -o)
// -i img -o .
// Use only the hardware backend (add -b)
// -i img -b hardware
// Use both gpu and hardware backends (with automatic load-balancing)
// -i img -b gpu hardware
// Use cpu, gpu and hardware backends (with automatic load-balancing)
// -i img -b cpu gpu hardware
// Use all supported backends (omit -b)
// -i img
// Use as many threads as CPU hardware threads (-j 0)
// -i img -j 0
// Use two threads and 8 states (add -s)
// -i img -j 2 -s 8
// Use one thread and twice as many states as hardware JPEG decode engines (provide -s 0 or just omit it)
// -i img -j 1
// -i img -j 1 -s 0
// Use one thread with all available hardware JPEG decode engines
// -i img -j 1 -b hardware
// Automatically detect the best number of CPU threads to use (omit -j)
// -i img
// Automatically detect the best number of CPU threads to use but start with 8 states
// -i img -s 8
// Benchmark individual backends (add -r to run multiple times and pick the best)
// -i img -n 4000 -b cpu -r 4
// -i img -n 4000 -b gpu -r 4
// -i img -n 4000 -b hardware -r 4
// Benchmark automatic load-balancing among backends
// -i img -n 4000 -b cpu gpu hardware -r 4
int main(int argc, const char* argv[]) 
{
    int pidx;
    if (argc < 2 || (pidx = findParamIndex(argv, argc, "-h")) != -1) {
        std::cerr << "Usage: " << argv[0] << "-i indir [-n nimages] [-s nstates] [-j nthreads] [-b backends] [-r nruns] [-o outdir]\n";
        std::cerr << "Parameters:\n";
        std::cerr << "  (REQUIRED)  -i indir: Directory to take JPEG images from.\n";
        std::cerr << "  (OPTIONAL)  -n nimages: Number of images to decode.\n"
                  << "               If not provided, decode all images in the input directory.\n"
                  << "               Will be automatically adjusted to be at least number of states.\n";
        std::cerr << "  (OPTIONAL)  -s nstates: Number of states.\n"
                  << "               If not provided, use twice the number of hardware engines.\n"
                  << "               Will be automatically adjusted to be at least number of threads.\n";
        std::cerr << "  (OPTIONAL)  -j nthreads: Number of CPU threads.\n"
                  << "               If not provided, automatically find the best number of threads to use.\n"
                  << "               Use 0 to set to the number of CPU cores on the system.\n";
        std::cerr << "  (OPTIONAL)  -b backends: any of\n"
                  << "               cpu/gpu/hardware/cpu gpu/cpu hardware/gpu hardware/cpu gpu hardware\n";
        std::cerr << "  (OPTIONAL)  -r nruns: Run this many times and pick the one with the maximum throughput.\n";
        std::cerr << "  (OPTIONAL)  -o outdir: Directory to write decoded images in BMP format.\n";
        return 0;
    }

    params p;
    // (OPTIONAL) Get number of threads
    bool autoThreads = false;
    if ((pidx = findParamIndex(argv, argc, "-j")) != -1) {
        CHECK(pidx + 1 < argc, "Number of threads not provided after -j (e.g., -j 1)\n");
        p.num_threads = std::atoi(argv[pidx + 1]);
        if (p.num_threads == 0)
            p.num_threads = std::thread::hardware_concurrency();
    } else { // -j is not provided, automatically find the best number of threads 
        autoThreads = true;
    }

    // (OPTIONAL) Get number of states
    if ((pidx = findParamIndex(argv, argc, "-s")) != -1) {
        CHECK(pidx + 1 < argc, "Need num states provided after -s (e.g., -s 1)\n");
        p.num_states = std::atoi(argv[pidx + 1]);
    }
    CHECK(p.num_states >= 0, "Need non-negative total number of states\n");
  
    // (OPTIONAL) Get number of images
    if ((pidx = findParamIndex(argv, argc, "-n")) != -1) {
        CHECK(pidx + 1 < argc, "No total images provided after -n\n");
        p.total_images = std::max(0, std::atoi(argv[pidx + 1]));
    }

    // (OPTIONAL) Get the number of runs
    if ((pidx = findParamIndex(argv, argc, "-r")) != -1) {
        CHECK(pidx + 1 < argc, "No runs provided after -r\n");
        p.num_runs = std::max(0, std::atoi(argv[pidx + 1]));
    }

    // (REQUIRED) Get input directory
    CHECK((pidx = findParamIndex(argv, argc, "-i")) != -1, "No input directory provided with -i\n");
    CHECK(pidx + 1 < argc, "No input dir provided after -i\n");
    p.input_dir = argv[pidx + 1];
    //std::cout << "Input: " << p.input_dir << "\n";
    
    // (OPTIONAL) Get the output directory
    if ((pidx = findParamIndex(argv, argc, "-o")) != -1) {
        CHECK(pidx + 1 < argc, "No output dir provided after -o\n");
        p.output_dir = argv[pidx + 1];
        p.write_output = true;
        //std::cout << "Output: " << p.output_dir << "\n";
    }
    
    // (OPTIONAL) Get the backends
    if ((pidx = findParamIndex(argv, argc, "-b")) != -1) {
        CHECK(pidx + 1 < argc, "No backends provided after -b\n");
        
        for (int i = pidx + 1; i < argc && argv[i][0] != '-'; ++i) {
            const char* backend = argv[i];
            if (strcmp(backend, "cpu") == 0)
                p.backend_enabled[NVJPEG_BACKEND_HYBRID] = true;
            else if (strcmp(backend, "gpu") == 0)
                p.backend_enabled[NVJPEG_BACKEND_GPU_HYBRID] = true;
            else if (strcmp(backend, "hardware") == 0)
                p.backend_enabled[NVJPEG_BACKEND_HARDWARE] = true;
            else {
                std::cerr << "Unknown backend: " << backend << "\n";
                return 1;
            }
        }
    } else { // enable all backends
        p.backend_enabled[NVJPEG_BACKEND_HYBRID] = true;
        p.backend_enabled[NVJPEG_BACKEND_GPU_HYBRID] = true;
        p.backend_enabled[NVJPEG_BACKEND_HARDWARE] = true;
    }
    int numBackends = 0;
    int onlyBackend = 0;
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

    decode_result bestResult;
    // If -j is not provided, we find the best number of threads to use
    int stage = 0;
    if (autoThreads) {
        p.num_threads = std::max(p.num_states, p.num_threads); // start with same number of threads as number of states
        if (p.num_states != 0)
            p.num_states = std::max(p.num_states, p.num_threads);
        while (true) {
            decode_result currentBest;
            for (int t = 0; t < p.num_runs; ++t) {
                decode_result result = decodeAllImages(p);
                if (result.total_throughput > currentBest.total_throughput)
                    currentBest = result;
            }
    
            if (currentBest.total_throughput < bestResult.total_throughput) {
                // revert
                p.num_states = bestResult.num_states;
                p.num_threads = bestResult.num_threads;
                if (++stage >= 2)
                    break;
            } else {
                bestResult = currentBest;
            }
            p.num_threads += (stage == 0) ? p.num_threads : 1;
            p.num_states = std::max(p.num_states, p.num_threads);
        }
    }
    for (int i = 0; i < p.num_runs; ++i) {
        decode_result result = decodeAllImages(p);
        if (result.total_throughput > bestResult.total_throughput)
            bestResult = result;
    }
    CHECK_CUDA(cudaDeviceReset());
    
    std::cout << "------------------------------------------------\n";
    std::cout << "GPU: " << bestResult.device_name << "\n";
    std::cout << "Num hardware decode engines: " << bestResult.num_engines << "\n";
    std::cout << "Input: " << p.input_dir << "\n";
    std::cout << "Enabled backends:";
    if (p.backend_enabled[NVJPEG_BACKEND_HYBRID])
        std::cout << " cpu";
    if (p.backend_enabled[NVJPEG_BACKEND_GPU_HYBRID])
        std::cout << " gpu";
    if (p.backend_enabled[NVJPEG_BACKEND_HARDWARE])
        std::cout << " hardware";
    std::cout << "\n";
    std::cout << "Throughput: " << bestResult.total_throughput << " images/s\n";
    std::cout << "Latency: " << bestResult.latency[1] << ", " << bestResult.latency[2] << ", " << bestResult.latency[3] << " ms\n";
    std::cout << "Throughput " << bestResult.throughput[1] << ", " << bestResult.throughput[2] << ", " << bestResult.throughput[3] << " ms\n";
    std::cout << "Percentage: " << bestResult.percentage[1] << ", " << bestResult.percentage[2] << ", " << bestResult.percentage[3] << " %\n";
    std::cout << "Num threads: " << bestResult.num_threads << "\n";
    std::cout << "Num states: " << bestResult.num_states << "\n";
    std::cout << "Num runs: " << p.num_runs << "\n";
    
    return 0;
}

