#include <cutensorMg.h>
#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <chrono>

bool CHECK_success(cudaError_t status)
{
    return status == cudaSuccess;
}

const char* CHECK_pretty(cudaError_t status)
{
    return cudaGetErrorName(status);
}

bool CHECK_success(cutensorStatus_t status)
{
    return status == CUTENSOR_STATUS_SUCCESS;
}

const char* CHECK_pretty(cutensorStatus_t status)
{
    return cutensorGetErrorString(status);
}

template<typename T>
T product(const std::vector<T> &values)
{
    T result = 1;
    for (auto& value : values)
    {
        result *= value;
    }
    return result;
}

template<typename T, typename U>
std::vector<T> multiply(const std::vector<T> &lhs, const std::vector<U> &rhs)
{
    std::vector<T> result;
    assert(lhs.size() == rhs.size() || lhs.empty() || rhs.empty());
    for (size_t i = 0; i < lhs.size(); i++)
    {
        result.push_back((lhs.empty() ? 1 : lhs[i]) * (rhs.empty() ? 1 : rhs[i]));
    }
    return result;
}

template<typename T, typename U>
std::vector<T> discretize(const std::vector<T> &in, const std::vector<U> &block)
{
    if (in.empty())
    {
        return in;
    }
    if (block.empty())
    {
        return in;
    }

    std::vector<T> result;
    assert(in.size() == block.size());
    for (size_t i = 0; i < in.size(); i++)
    {
        U b = block[i];
        result.push_back(b * ((in[i] + b - 1) / b));
    }
    return result;
}

#define CHECK(x) do { auto CHECK_err = (x); if (! CHECK_success(CHECK_err)) { \
  printf("Error (%s:%d): \"%s\" returned %s (%d)\n", __FILE__, __LINE__, \
    #x, CHECK_pretty(CHECK_err), CHECK_err); exit(-1);} } while(0)

template<typename K, typename V, typename K2>
std::vector<V> collect(const std::unordered_map<K, V> &map, const std::vector<K2> &index)
{
    std::vector<V> result;
    for (auto& elem : index)
    {
        result.push_back(map.at(elem));
    }
    return result;
}

void printDeviceInfo(int deviceId)
{
    struct cudaDeviceProp prop;
    int currentDeviceId = 0;
    CHECK(cudaGetDevice(&currentDeviceId));
    CHECK(cudaSetDevice(deviceId));
    CHECK(cudaGetDeviceProperties(&prop, deviceId));
    printf( "device %d (%s): SMs %2d  Capabilities %d.%d, SmClock %.1f Mhz, MemSize (MB) %d, MemClock %.1f Mhz\n",
            deviceId,
            prop.name,
            prop.multiProcessorCount, prop.major, prop.minor,
            (float)prop.clockRate*1e-3,
            (int)(prop.totalGlobalMem/(1024*1024)),
            (float)prop.memoryClockRate*1e-3);
    CHECK(cudaSetDevice(currentDeviceId));
}

int main(int argc, char** argv)
{
    printf("This sample uses the following GPUs:\n");
    std::vector<int32_t> devices;
    if (argc == 1)
    {
        int numDevices;
        CHECK(cudaGetDeviceCount(&numDevices));
        for (int i = 0; i < numDevices; i++)
        {
            printDeviceInfo(i);
            devices.push_back(i);
        }
    }
    else
    {
        for (int i = 1; i < argc; i++)
        {
            const int deviceId = atoi(argv[i]);
            printDeviceInfo(deviceId);
            devices.push_back(deviceId);
        }
    }
    cutensorMgHandle_t handle;
    printf("Initializing cutensorMg handle ... ");
    CHECK(cutensorMgCreate(&handle, devices.size(), devices.data()));
    printf("done.\n");
    
    std::unordered_map<int32_t, int64_t> extent;
    extent['i'] = 4096;
    extent['j'] = 4096;
    extent['k'] = 4096;

    std::unordered_map<int32_t, int64_t> blocksize;
    blocksize['i'] = 2048;
    blocksize['j'] = 2048;
    blocksize['k'] = 2048;

    std::unordered_map<int32_t, int32_t> deviceCount;
    deviceCount['i'] = 2;
    deviceCount['j'] = 2;
    deviceCount['k'] = 2;

    std::vector<int32_t> modesA {'i', 'k'};
    std::vector<int32_t> modesB {'k', 'j'};
    std::vector<int32_t> modesC {'i', 'j'};

    cudaDataType_t kDataType = CUDA_R_32F;
    const int64_t kElementSize = 4;

    printf("Creating distributed tensor descriptors ... ");

    auto fillUp = [](const std::vector<int32_t> &devices, const int32_t n)
    {
        std::vector<int32_t> ret; 
        int32_t numDevices = devices.size();
        for(int i=0; i < n; ++i)
        {
            ret.push_back(devices[i%numDevices]);
        }
        return ret;
    };

    cutensorMgTensorDescriptor_t descA;
    std::vector<int64_t> extentA = collect(extent, modesA);
    std::vector<int64_t> blocksizeA = collect(blocksize, modesA);
    std::vector<int32_t> deviceCountA = collect(deviceCount, modesA);
    std::vector<int32_t> devicesA = fillUp(devices, product(deviceCountA));
    assert(product(deviceCountA) == devicesA.size());
    CHECK(cutensorMgCreateTensorDescriptor(handle, &descA, modesA.size(),
        extentA.data(), NULL, blocksizeA.data(), NULL,
        deviceCountA.data(), devicesA.size(), devicesA.data(), kDataType));

    cutensorMgTensorDescriptor_t descB;
    std::vector<int64_t> extentB = collect(extent, modesB);
    std::vector<int64_t> blocksizeB = collect(blocksize, modesB);
    std::vector<int32_t> deviceCountB = collect(deviceCount, modesB);
    std::vector<int32_t> devicesB = fillUp(devices, product(deviceCountB));
    assert(product(deviceCountB) == devicesB.size());
    CHECK(cutensorMgCreateTensorDescriptor(handle, &descB, modesB.size(),
        extentB.data(), NULL, blocksizeB.data(), NULL,
        deviceCountB.data(), devicesB.size(), devicesB.data(), kDataType));

    cutensorMgTensorDescriptor_t descC;
    std::vector<int64_t> extentC = collect(extent, modesC);
    std::vector<int64_t> blocksizeC = collect(blocksize, modesC);
    std::vector<int32_t> deviceCountC = collect(deviceCount, modesC);
    std::vector<int32_t> devicesC = fillUp(devices, product(deviceCountC));
    assert(product(deviceCountC) == devicesC.size());
    CHECK(cutensorMgCreateTensorDescriptor(handle, &descC, modesC.size(),
        extentC.data(), NULL, blocksizeC.data(), NULL,
        deviceCountC.data(), devicesC.size(), devicesC.data(), kDataType));

    printf("done.\n");

    printf("Querying workspace size (per GPU) ... ");

    const cutensorComputeType_t kComputeType = CUTENSOR_COMPUTE_32F;
    const cutensorWorksizePreference_t kWorksizePreference = 
        CUTENSOR_WORKSPACE_DEFAULT;

    cutensorMgContractionDescriptor_t contractionDesc;
    CHECK(cutensorMgCreateContractionDescriptor(handle, &contractionDesc,
                descA, modesA.data(),
                descB, modesB.data(),
                descC, modesC.data(),
                descC, modesC.data(),
                kComputeType));

    cutensorMgContractionFind_t contractionFind;
    CHECK(cutensorMgCreateContractionFind(handle, &contractionFind,
                CUTENSORMG_ALGO_DEFAULT));

    std::vector<int64_t> workspaceSize(devices.size());
    int64_t workspaceHostSize;
    CHECK(cutensorMgContractionGetWorkspace(handle,
        contractionDesc, contractionFind, kWorksizePreference, workspaceSize.data(), &workspaceHostSize));

    printf("done.\n");

    printf("Initializing contraction plan ... \n");
 
    cutensorMgContractionPlan_t plan;
    CHECK(cutensorMgCreateContractionPlan(handle, &plan,
                contractionDesc, contractionFind, workspaceSize.data(), workspaceHostSize));

    printf("done.\n");

    printf("Allocating data ... ");

    int64_t elementsA = product(discretize(extentA, multiply(deviceCountA, blocksizeA))) / product(deviceCountA);
    std::vector<void*> memoryA;
    for (auto& device : devicesA)
    {
        void* memory;
        CHECK(cudaSetDevice(device));
        CHECK(cudaMalloc(&memory, elementsA * kElementSize));
        memoryA.push_back(memory);
    }

    int64_t elementsB = product(discretize(extentB, multiply(deviceCountB, blocksizeB))) / product(deviceCountB);
    std::vector<void*> memoryB;
    for (auto& device : devicesB)
    {
        void* memory;
        CHECK(cudaSetDevice(device));
        CHECK(cudaMalloc(&memory, elementsB * kElementSize));
        memoryB.push_back(memory);
    }

    int64_t elementsC = product(discretize(extentC, multiply(deviceCountC, blocksizeC))) / product(deviceCountC);
    std::vector<void*> memoryC;
    for (auto& device : devicesC)
    {
        void* memory;
        CHECK(cudaSetDevice(device));
        CHECK(cudaMalloc(&memory, elementsC * kElementSize));
        memoryC.push_back(memory);
    }

    std::vector<cudaStream_t> streams;
    for (auto& device : devices)
    {
        cudaStream_t stream;
        CHECK(cudaSetDevice(device));
        CHECK(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }

    /*
     * Allocate workspace
     */
    // host
    void* workspaceHost = nullptr;
    CHECK(cudaMallocHost(&workspaceHost, workspaceHostSize));

    // devices
    std::vector<void*> workspace;
    for (int i = 0; i < devices.size(); i++)
    {
        void* memory;
        CHECK(cudaSetDevice(devices[i]));
        CHECK(cudaMalloc(&memory, workspaceSize[i]));
        workspace.push_back(memory);
    }

    printf("done.\n");

    printf("Performing distributed tensor contraction ...\n");

    float kAlpha = 1;
    float kBeta = 0;


    int currentDeviceId = -1;
    CHECK(cudaGetDevice(&currentDeviceId));

    float minElapsed = 0;
    const int nRep = 3; // for stable timings
    for (int rep = 0; rep < nRep; rep++)
    {
        const auto start = std::chrono::steady_clock::now();
        CHECK(cutensorMgContraction(handle, plan, &kAlpha,
            const_cast<const void**>(memoryA.data()),
            const_cast<const void**>(memoryB.data()), &kBeta, 
            const_cast<const void**>(memoryC.data()), memoryC.data(),
            workspace.data(), workspaceHost, streams.data()));

        for (auto& deviceId : devices)
        {
            CHECK(cudaSetDevice(deviceId));
            CHECK(cudaDeviceSynchronize());
        }

        const auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> dur = end - start;
        if (minElapsed == 0 || minElapsed > dur.count()) {
            minElapsed = dur.count();
        }
    }
    CHECK(cudaSetDevice(currentDeviceId));

    printf("execution took: %.2e millisec.\n", minElapsed);

    printf("Free resources ...\n");

    for (auto& stream : streams)
    {
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaStreamDestroy(stream));
    }

    for (auto& memory : memoryA)
    {
        CHECK(cudaFree(memory));
    }

    for (auto& memory : memoryB)
    {
        CHECK(cudaFree(memory));
    }

    for (auto& memory : memoryC)
    {
        CHECK(cudaFree(memory));
    }

    CHECK(cudaFreeHost(workspaceHost));

    CHECK(cutensorMgDestroyContractionDescriptor(contractionDesc));
    CHECK(cutensorMgDestroyContractionFind(contractionFind));
    CHECK(cutensorMgDestroyContractionPlan(plan));

    CHECK(cutensorMgDestroyTensorDescriptor(descA));
    CHECK(cutensorMgDestroyTensorDescriptor(descB));
    CHECK(cutensorMgDestroyTensorDescriptor(descC));

    CHECK(cutensorMgDestroy(handle));
    printf("Done: everything has completed successfully.\n");
}
