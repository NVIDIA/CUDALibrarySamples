/*
 * Copyright 2025 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <fstream>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>

class ThreadPool {
public:
    ThreadPool(int threads);
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(int, Args...)>::type>
    {
        auto task = std::make_shared< std::packaged_task<int(int)> >(
            std::bind(std::forward<F>(f), std::placeholders::_1, std::forward<Args>(args)...)
        );
                
        std::future<int> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            // don't allow enqueueing after stopping the pool
            if(stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            tasks.emplace([task](int tid){ (*task)(tid); });
        }
        condition.notify_one();
        return res;
    }
    void wait()
    {
        std::unique_lock<std::mutex> lock(this->queue_mutex);
        completed.wait(lock, [this]{return this->in_flight == 0 && this->tasks.empty();});
    }
    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers)
            worker.join();
    }
private:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    std::queue< std::function<void(int)> > tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    std::condition_variable completed;
    int in_flight = 0;
    bool stop = false;
};
 
// the constructor just launches some amount of workers
ThreadPool::ThreadPool(int threads) : workers(threads)
{
    for (int i = 0; i<threads; ++i)
        workers[i] = std::thread(
            [this, i]
            {
                for(;;)
                {
                    std::function<void(int)> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                                             [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                        in_flight++;
                    }
                    task(i);
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    in_flight--;
                    if ((this->in_flight == 0) && this->tasks.empty())
                        completed.notify_one();
                }
            }
    );
}

#define CHECK_CUDA(call) \
    { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) \
        { \
            std::cerr << "CUDA Runtime failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

#define CHECK_NVJPEG(call) \
    { \
        nvjpegStatus_t _e = (call); \
        if (_e != NVJPEG_STATUS_SUCCESS) \
        { \
            std::cerr << "NVJPEG failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

#define CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Error: " << message << std::endl; \
            std::exit(1); \
        } \
    } while(0)

int dev_malloc(void** p, size_t s) { return (int)cudaMalloc(p, s); }
int dev_free(void* p) { return (int)cudaFree(p); }
int host_malloc(void** p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }
int host_free(void* p) { return (int)cudaFreeHost(p); }

size_t pad(size_t size)
{
    // Round up to nearest multiple of twoMegabytes
    constexpr size_t twoMegabytes = 2 * 1024 * 1024;
    size_t paddedSize = twoMegabytes * ((size + twoMegabytes - 1) / twoMegabytes);
    return paddedSize;
}

struct params {
    nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_RGBI;
    std::string input_dir = "";
    std::string output_dir = ".";
    unsigned int total_images = 1;
    unsigned int num_states = 1;
    bool write_output = false;
    // Encode params
    nvjpegEncoderParams_t encode_params = nullptr;
    int encoding_mode = 0;
    unsigned int width = 3840;
    unsigned int height = 2160;
};

struct nvjpeg_local_states {
    cudaStream_t stream = nullptr;
    int img = -1; // the image that this stream is processing
    nvjpegEncoderState_t encode_state = nullptr;
};

struct nvjpeg_states {
    nvjpegHandle_t nvjpeg_handle = nullptr;
    cudaStream_t global_stream = nullptr;
    nvjpegDevAllocator_t dev_allocator = { &dev_malloc, &dev_free };
    std::vector<nvjpeg_local_states> local_states;

    void createEncode(int nStates, nvjpegEncoderParams_t& encode_params) {
        if (!nvjpeg_handle) {
            CHECK_NVJPEG(nvjpegCreateSimple(&nvjpeg_handle));
        }
        CHECK_CUDA(cudaStreamCreateWithFlags(&global_stream, cudaStreamNonBlocking));
        CHECK_NVJPEG(nvjpegEncoderParamsCreate(nvjpeg_handle, &encode_params, global_stream));
        { // These are optional
            CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(encode_params, 80, global_stream));
            CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(encode_params, 0, global_stream));
            CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_420, global_stream));
            CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(encode_params, NVJPEG_ENCODING_BASELINE_DCT, global_stream));
        }
        local_states.resize(nStates);
        std::cerr << "Creating " << nStates << " local states" << std::endl;
        for (auto& ls : local_states) {
            CHECK_CUDA(cudaStreamCreateWithFlags(&ls.stream, cudaStreamNonBlocking));
            CHECK_NVJPEG(nvjpegEncoderStateCreateWithBackend(nvjpeg_handle, &ls.encode_state, NVJPEG_ENC_BACKEND_DEFAULT, ls.stream));
        }
    }

    int createEncode(nvjpegEncoderParams_t& encode_params, const params& p) {
        CHECK_NVJPEG(nvjpegCreateSimple(&nvjpeg_handle));
        unsigned int nEngines = 0;
        CHECK_NVJPEG(nvjpegGetHardwareEncoderInfo(nvjpeg_handle, &nEngines));
        std::cerr << "Number of hardware engines = " << nEngines << "\n";
        createEncode(p.num_states, encode_params);
        return p.num_states;
    }

    void destroy() {
        for (auto& ls : local_states) {
            if (ls.encode_state)
                CHECK_NVJPEG(nvjpegEncoderStateDestroy(ls.encode_state));
            CHECK_CUDA(cudaStreamDestroy(ls.stream));
            ls.encode_state = nullptr;
            ls.stream = nullptr;
        }
        CHECK_CUDA(cudaStreamDestroy(global_stream));
        if (nvjpeg_handle)
            CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));
        global_stream = nullptr;
        nvjpeg_handle = nullptr;
    }
};

int divUp(int n, int m) {
    return (n + m - 1) / m;
}

struct img_t {
    unsigned char* host_buf = nullptr;
    size_t host_buf_size = 0;
    nvjpegImage_t dev_img = {};
    size_t jpeg_bit_stream_size = 0;
    size_t padded_size = 0;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int ncomps = 0;
    nvjpegInputFormat_t input_format;
    nvjpegChromaSubsampling_t input_subsampling;

    int& getWidth() { return widths[0]; }
    int& getHeight() { return heights[0]; }
    int pitch() { return getWidth() * 3; }
    int bytes() { return getWidth() * getHeight() * ncomps; }
    unsigned char* devData() { return dev_img.channel[0]; }
    
    std::random_device rd;
    std::mt19937 gen;

    img_t(): rd(), gen(rd()) {}

    ~img_t() {
        deallocateDeviceData();
        deallocateHostData();
    }

    void writeJpegToFile(const std::string& outputDir, int i) {
        std::filesystem::path outputPath(outputDir);
    
        std::error_code ec;
        if (!std::filesystem::exists(outputPath)) {
            std::filesystem::create_directories(outputPath, ec);
            CHECK(!ec, "Error: Failed to create directory " + outputPath.string() + ": " << ec.message());
        }
    
        std::string fileName = (outputPath / (std::to_string(i) + ".jpg")).string();
        std::ofstream output(fileName, std::ios::binary);
        CHECK(output, "Error: Failed to open file " + fileName);
    
        output.write(reinterpret_cast<const char*>(host_buf), jpeg_bit_stream_size);
        CHECK(output, "Error: Failed to write " + std::to_string(jpeg_bit_stream_size) + " bytes to file " + fileName);
    }

    void allocateInputImgRandom(int width, int height, size_t totalSize) {
        if (width > getWidth() || height > getHeight()) {
            deallocateDeviceData();
            CHECK_CUDA(cudaMalloc((void**)(&(dev_img.channel[0])), totalSize));
            std::cerr << "Allocating new image\n";
            // Fill with random values (0-255 for image data)
            std::vector<unsigned char> randomData(totalSize);
            std::uniform_int_distribution<> dis(0, 255);  // Create new distribution each time
            for (size_t i = 0; i < totalSize; i++) {
                randomData[i] = static_cast<unsigned char>(dis(gen));
            }
            CHECK_CUDA(cudaMemcpy(dev_img.channel[0], randomData.data(), totalSize, cudaMemcpyHostToDevice));

            getWidth() = width;
            getHeight() = height;
        }
    }

    // Simple function to read RGB data from a BMP file
    void allocateInputImgFromBmp(const char* bmpFilePath, int& width, int& height) {
        std::vector<unsigned char> rgbData;

        std::ifstream file(bmpFilePath, std::ios::binary);
        CHECK(file.is_open(), "Error: Could not open BMP file: " + std::string(bmpFilePath));

        unsigned char header[54];
        file.read(reinterpret_cast<char*>(header), 54);

        CHECK(header[0] == 'B' && header[1] == 'M', "Error: Not a valid BMP file: " + std::string(bmpFilePath));

        int dataOffset = *reinterpret_cast<int*>(&header[10]);
        int imgWidth = *reinterpret_cast<int*>(&header[18]);
        int imgHeight = *reinterpret_cast<int*>(&header[22]);
        int bpp = *reinterpret_cast<short*>(&header[28]);

        CHECK(bpp == 24, "Error: Only 24-bit BMP files are supported. This file is " + std::to_string(bpp) + "-bit.\n");
        
        size_t totalSize = size_t(imgWidth) * imgHeight * 3;

        if (imgWidth > getWidth() || imgHeight > getHeight()) {
            deallocateDeviceData();
            CHECK_CUDA(cudaMalloc((void**)(&(dev_img.channel[0])), totalSize));
            std::cerr << "Allocating new image " << imgWidth << " x " << imgHeight << "\n";
            getWidth() = width = imgWidth;
            getHeight() = height = imgHeight;
        }

        int bytesPerPixel = bpp / 8;
        int rowPadding = (4 - (imgWidth * bytesPerPixel) % 4) % 4;

        file.seekg(dataOffset, std::ios::beg);
        rgbData.resize(imgWidth * imgHeight * 3);
        unsigned char* rPtr = rgbData.data() + 0 * imgWidth * imgHeight;
        unsigned char* gPtr = rgbData.data() + 1 * imgWidth * imgHeight;
        unsigned char* bPtr = rgbData.data() + 2 * imgWidth * imgHeight;
        for (int y = 0; y < imgHeight; y++) {
            int destRow = imgHeight - 1 - y;
            for (int x = 0; x < imgWidth; x++) {
                unsigned char pixel[3];
                file.read(reinterpret_cast<char*>(pixel), 3);
                rPtr[destRow * imgWidth + x] = pixel[2]; // R
                gPtr[destRow * imgWidth + x] = pixel[1]; // G
                bPtr[destRow * imgWidth + x] = pixel[0]; // B
            }
            // Skip padding bytes
            file.seekg(rowPadding, std::ios::cur);
        }

        CHECK_CUDA(cudaMemcpy(dev_img.channel[0], rgbData.data(), rgbData.size(), cudaMemcpyHostToDevice));
    }

    // Allocate the device buffer to store the input raw image before encoding.
    void allocateInputOnDevice(const char* bmpFilePath, nvjpegHandle_t nvjpegHandle, nvjpegInputFormat_t inputFormat, 
                               nvjpegChromaSubsampling_t inputSubsampling, int width, int height)
    {
        input_format = inputFormat;
        input_subsampling = inputSubsampling;
        int dx = 1, dy = 1;
        if      (inputSubsampling == NVJPEG_CSS_422) { dx = 2; dy = 1; }
        else if (inputSubsampling == NVJPEG_CSS_420) { dx = 2; dy = 2; }
        else if (inputSubsampling == NVJPEG_CSS_440) { dx = 1; dy = 2; }
        else if (inputSubsampling == NVJPEG_CSS_411) { dx = 4; dy = 1; }
        else if (inputSubsampling == NVJPEG_CSS_410) { dx = 4; dy = 2; }
        if (inputFormat == NVJPEG_INPUT_RGB) {
            allocateInputImgFromBmp(bmpFilePath, width, height);
            for (int c = 0; c < 3; ++c) {
                dev_img.pitch[c] = width;
                dev_img.channel[c] = devData() + c * dev_img.pitch[c] * height;
            }
        } else if (inputFormat == NVJPEG_INPUT_YUV) {
            size_t totalSize = 0;
            for (int c = 0; c < 3; ++c) {
                int w = c == 0 ? width  : divUp(width , dx);
                int h = c == 0 ? height : divUp(height, dy);
                dev_img.pitch[c] = w;
                totalSize += dev_img.pitch[c] * h;
            }
            allocateInputImgRandom(width, height, totalSize);
            for (int c = 0; c < 3; ++c) {
                int h = c == 0 ? height : divUp(height, dy);
                dev_img.channel[c] = devData() + c * dev_img.pitch[c] * h;
            }
        } else if (inputFormat == NVJPEG_INPUT_NV12) {
            size_t totalSize = 0;
            for (int c = 0; c < 2; ++c) {
                int w = c == 0 ? width  : divUp(width , dx);
                int h = c == 0 ? height : divUp(height, dy);
                dev_img.pitch[c] = c == 0 ? w : w * 2; // since U and V are in the same channel
                totalSize += dev_img.pitch[c] * h;
            }
            allocateInputImgRandom(width, height, totalSize);
            for (int c = 0; c < 2; ++c) {
                int h = c == 0 ? height : divUp(height, dy);
                dev_img.channel[c] = devData() + c * dev_img.pitch[c] * h;
            }
        } else { // BGR/RGBI/BGRI
            std::cerr << "Not supported for now\n";
            exit(1);
        }
    }
    
    void deallocateDeviceData()
    {
        if (devData())
            CHECK_CUDA(cudaFree(devData()));
    }

    void deallocateHostData()
    {
        if (host_buf)
            CHECK_CUDA(cudaFreeHost(host_buf));
    }

    void encode(nvjpegHandle_t nvjpegHandle, nvjpegEncoderState_t encodeState, 
                nvjpegEncoderParams_t encodeParams, cudaStream_t stream)
    {
        CHECK_NVJPEG(nvjpegEncode(nvjpegHandle, encodeState, encodeParams, &dev_img, 
                                  input_subsampling, input_format, getWidth(), getHeight(), stream));
    }

    void allocateHostBuf(size_t size) {
        if (host_buf_size < pad(size)) {
            deallocateHostData();
            host_buf_size = pad(size);
            CHECK_CUDA(cudaMallocHost((void**)&host_buf, host_buf_size));
        }
    }

    // If encoding has finished successfully, this returns NVJPEG_STATUS_SUCCESS. Also, the bitstream is retrieved.
    // If encoding is still going on, this returns NVJPEG_STATUS_INCOMPLETE_BITSTREAM.
    // If encoding has finished unsuccessfully, returns NVJPEG_STATUS_EXECUTION_FAILED.
    nvjpegStatus_t retrieveBitStreamToHost(nvjpegHandle_t nvjpegHandle, nvjpegEncoderState_t encodeState, 
                                           cudaStream_t stream)
    {
        size_t size = 0;
        nvjpegStatus_t status = nvjpegEncodeRetrieveBitstream(nvjpegHandle, encodeState, nullptr, &size, stream);
        if (status == NVJPEG_STATUS_SUCCESS) {
            allocateHostBuf(size);
            CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nvjpegHandle, encodeState, host_buf, &size, stream));
            jpeg_bit_stream_size = size;
        }
        return status;
    }

    nvjpegStatus_t retrieveBitStreamToHostBlocking(nvjpegHandle_t nvjpegHandle, 
                                                   nvjpegEncoderState_t encodeState, cudaStream_t stream)
    {
        nvjpegStatus_t result;
        CHECK_CUDA(cudaStreamSynchronize(stream));
        result = retrieveBitStreamToHost(nvjpegHandle, encodeState, stream);
        return result;
    }
};

// Get the file names in p.input_dir into a vector
std::vector<std::string> getFileNames(params& p) {
    std::vector<std::string> results;
    if (!p.input_dir.empty()) {
        std::filesystem::recursive_directory_iterator dirIter(p.input_dir), endIter;
        for (unsigned int i = 0; dirIter != endIter && (p.total_images == 0 || i < p.total_images);) {
            const auto& entry = *dirIter;
            const std::string& fileName = entry.path().string();
            size_t l = fileName.length();
            if (fileName[l - 4] == '.' && fileName[l - 3] == 'b' && fileName[l - 2] == 'm' && fileName[l - 1] == 'p') {
                results.push_back(fileName);
                ++i;
            }
            ++dirIter;
        }
        if (p.total_images == 0) {
            p.total_images = (unsigned int)results.size();
            p.num_states = p.total_images;
        }
    }
    return results;
}

// Simple version where we synchronize at the end of each image.
void encodeSingleThreadedBlocking(params& p)
{
    std::vector<std::string> fileNames = getFileNames(p);

    nvjpeg_states gs; // global states
    int nStates = gs.createEncode(p.encode_params, p);
    std::cerr << "Encode single threaded blocking, " << p.total_images << " images, " << nStates << " states\n";
    std::vector<img_t> imgs(nStates);
   
    int fileId = 0;
    for (img_t& img : imgs) {
        if (fileNames.empty()) { // generate random images
            img.allocateInputOnDevice("", gs.nvjpeg_handle, NVJPEG_INPUT_YUV, NVJPEG_CSS_420, p.width, p.height);
        } else { // encode BMP images
            img.allocateInputOnDevice(fileNames[fileId % fileNames.size()].data(), gs.nvjpeg_handle, NVJPEG_INPUT_RGB, NVJPEG_CSS_420, p.width, p.height);
            ++fileId;
        }
    }

    auto start = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < p.total_images; ++i) {
        img_t& img = imgs[i % nStates];
        auto& ls = gs.local_states[i % nStates];
        img.encode(gs.nvjpeg_handle, ls.encode_state, p.encode_params, ls.stream);
        nvjpegStatus_t result = img.retrieveBitStreamToHostBlocking(gs.nvjpeg_handle, ls.encode_state, ls.stream);
        if (result == NVJPEG_STATUS_SUCCESS && p.write_output) {
            CHECK_CUDA(cudaStreamSynchronize(ls.stream));
            img.writeJpegToFile(p.output_dir, i);
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cerr << "Encoding loop took " << duration.count() << " ms" << std::endl;
    std::cerr << "Throughput = " << (size_t(p.width) * p.height * p.total_images) / (1000.0 * duration.count()) << " mpix/s\n";

    cudaDeviceSynchronize();
    gs.destroy();
}

// Use a single thread but doesn't block at the end of each image
void encodeSingleThreadedNonBlocking(params& p)
{
    std::vector<std::string> fileNames = getFileNames(p);

    nvjpeg_states gs; // global states
    int nStates = gs.createEncode(p.encode_params, p);
    std::cerr << "Encode single threaded nonblocking, " << p.total_images << " images, " << nStates << " states\n";
    std::vector<img_t> imgs(nStates);

    int fileId = 0;
    for (img_t& img : imgs) {
        if (fileNames.empty()) { // generate random images
            img.allocateInputOnDevice("", gs.nvjpeg_handle, NVJPEG_INPUT_YUV, NVJPEG_CSS_420, p.width, p.height);
        } else { // encode BMP images
            img.allocateInputOnDevice(fileNames[fileId % fileNames.size()].data(), gs.nvjpeg_handle, NVJPEG_INPUT_RGB, NVJPEG_CSS_420, p.width, p.height);
            ++fileId;
        }
    }
    
    unsigned int i = 0;
    int state = 0;
    auto start = std::chrono::steady_clock::now();
    while (i < p.total_images) {
        auto& ls = gs.local_states[state % nStates];
        
        if (ls.img >= 0) {
            nvjpegStatus_t status = NVJPEG_STATUS_SUCCESS;
            status = imgs[ls.img % nStates].retrieveBitStreamToHost(gs.nvjpeg_handle, ls.encode_state, ls.stream);
            if (status == NVJPEG_STATUS_SUCCESS) {
                if (p.write_output) {
                    CHECK_CUDA(cudaStreamSynchronize(ls.stream));
                    imgs[ls.img % nStates].writeJpegToFile(p.output_dir, i);        
                }
            } else if (status == NVJPEG_STATUS_INCOMPLETE_BITSTREAM) {
                // this state is still encoding, move to the next state (but keep the image)
                state = (state + 1) % nStates;
                continue;
            }
        }

        imgs[i % nStates].encode(gs.nvjpeg_handle, ls.encode_state, p.encode_params, ls.stream);
        ls.img = i;

        ++i;
        state = (state + 1) % nStates;
    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cerr << "Encoding loop took " << duration.count() << " ms" << std::endl;
    std::cerr << "Throughput = " << (size_t(p.width) * p.height * p.total_images) / (1000.0 * duration.count()) << " mpix/s\n";

    cudaDeviceSynchronize();
    gs.destroy();
}

// Use multiple CPU threads
void encodeMultiThreaded(params& p)
{
    std::vector<std::string> fileNames = getFileNames(p);

    std::cerr << "Encode multithreaded " << p.num_states << " threads, " << p.total_images << " images\n";
    int nStates = p.num_states;
    nvjpeg_states gs; // global states
    gs.createEncode(nStates, p.encode_params);
    std::vector<img_t> imgs(nStates);
    ThreadPool workers(nStates);

    int fileId = 0;
    for (img_t& img : imgs) {
        if (fileNames.empty()) { // generate random images
            img.allocateInputOnDevice("", gs.nvjpeg_handle, NVJPEG_INPUT_YUV, NVJPEG_CSS_420, p.width, p.height);
        } else { // encode BMP images
            img.allocateInputOnDevice(fileNames[fileId % fileNames.size()].data(), gs.nvjpeg_handle, NVJPEG_INPUT_RGB, NVJPEG_CSS_420, p.width, p.height);
            ++fileId;
        }
    }
    
    auto start = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < p.total_images; ++i) {
        workers.enqueue(std::bind([&p, &imgs, &gs, nStates] (int i, int tid)
        {
            img_t& img = imgs[i % nStates];
            auto& ls = gs.local_states[tid];
            CHECK_CUDA(cudaStreamSynchronize(ls.stream));
            img.encode(gs.nvjpeg_handle, ls.encode_state, p.encode_params, ls.stream);
            nvjpegStatus_t result = img.retrieveBitStreamToHostBlocking(gs.nvjpeg_handle, ls.encode_state, ls.stream);
            if (result == NVJPEG_STATUS_SUCCESS && p.write_output) {
                CHECK_CUDA(cudaStreamSynchronize(ls.stream));
                img.writeJpegToFile(p.output_dir, i);
            }
            return 0;
        }, i, std::placeholders::_1));
    }
    workers.wait();
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cerr << "Encoding loop took " << duration.count() << " ms" << std::endl;
    std::cerr << "Throughput = " << (size_t(p.width) * p.height * p.total_images) / (1000.0 * duration.count()) << " mpix/s\n";

    cudaDeviceSynchronize();
    gs.destroy();
}

int findParamIndex(const char** argv, int argc, const char* parm) {
    int count = 0;
    int index = -1;

    for (int i = 0; i < argc; i++) {
        if (strncmp(argv[i], parm, 100) == 0) {
            index = i;
            count++;
        }
    }

    if (count == 0 || count == 1) {
        return index;
    }
    else {
        std::cerr << "Error, parameter " << parm
            << " has been specified more than once, exiting\n"
            << std::endl;
        return -1;
    }

    return -1;
}

int main(int argc, const char* argv[]) {
    int pidx;
    if (argc < 2 || (pidx = findParamIndex(argv, argc, "-h")) != -1) {
        std::cerr << "Usage: " << argv[0] << "[-n nimages] [-m mode] [-j nstates] [-w width] [-o outdir]\n";
        std::cerr << "Parameters:\n";
        std::cerr << "    -n nimages: Encode this many images\n";
        std::cerr << "    -m mode:\n";
        std::cerr << "        0 (single threaded, blocking)\n";
        std::cerr << "        1 (single threaded, nonblocking)\n"; 
        std::cerr << "        2 (multithreaded)\n";
        std::cerr << "    -j nstates: Create this many encoder states (also the number of threads in multithreaded mode)\n";
        std::cerr << "    -w width: Set the width of each image (the height will be set automatically assuming aspect ratio 16:9)\n";
        std::cerr << "    -i indir: Encode BMP images from this directory\n";
        std::cerr << "    -o outdir: Write encoded images to this directory\n";
        return 0;
    }

    params p;
    if ((pidx = findParamIndex(argv, argc, "-j")) != -1) {
        CHECK(pidx + 1 < argc, "No num states provided after -j\n");
        p.num_states = std::max(1, std::atoi(argv[pidx + 1]));
    }
  
    bool numStatesProvided = pidx != -1;
    if ((pidx = findParamIndex(argv, argc, "-n")) != -1) {
        CHECK(pidx + 1 < argc, "No total images provided after -n\n");
        p.total_images = std::max(0, std::atoi(argv[pidx + 1]));
    }

    if (numStatesProvided && p.total_images < p.num_states) {
        std::cerr << "Num images < num states, using " << p.num_states << " images instead\n";
        p.total_images = p.num_states;
    }

    if ((pidx = findParamIndex(argv, argc, "-i")) != -1) {
        CHECK(pidx + 1 < argc, "No input dir provided after -i\n");
        p.input_dir = argv[pidx + 1];
        p.write_output = true;
        std::cerr << "Reading input BMP images from " << p.input_dir << "\n";
    } else {
        std::cerr << "Writing random images\n";
        if ((pidx = findParamIndex(argv, argc, "-w")) != -1) {
            CHECK(pidx + 1 < argc, "No width provided after -w\n");
            p.width = std::max(16, std::atoi(argv[pidx + 1]) / 2 * 2); // the / 2 * 2 makes sure it's even
            p.height = (p.width * 9 / 16) / 2 * 2;
        } 
        std::cerr << "Width = " << p.width << ", height = " << p.height << "\n";
    }
    
    if ((pidx = findParamIndex(argv, argc, "-o")) != -1) {
        CHECK(pidx + 1 < argc, "No output dir provided after -o\n");
        p.output_dir = argv[pidx + 1];
        p.write_output = true;
        std::cerr << "Writing output images to " << p.output_dir << "\n";
    } else {
        std::cerr << "Not writing output images\n";
    }

    if ((pidx = findParamIndex(argv, argc, "-m")) != -1) {
        CHECK(pidx + 1 < argc, "No encoding mode provided after -m\n");
        p.encoding_mode = std::atoi(argv[pidx + 1]);
    }

    if (p.encoding_mode == 0)
        encodeSingleThreadedBlocking(p);
    else if (p.encoding_mode == 1)
        encodeSingleThreadedNonBlocking(p);
    else if (p.encoding_mode == 2)
        encodeMultiThreaded(p);
    else
        std::cerr << "Encoding mode can only be 0, 1, or 2\n";
    
    std::cerr << "Finished without errors\n";

    return 0;
}

