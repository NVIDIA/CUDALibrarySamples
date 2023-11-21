#include <cutensorMg.h>
#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <cstdint>
#include <cinttypes>
#include <unordered_map>
#include <chrono>
#include <numeric>
#include <cmath>

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

#define CHECK(x) do { auto CHECK_err = (x); if (! CHECK_success(CHECK_err)) { \
  printf("\nError (%s:%d): \"%s\" returned %s (%d)\n", __FILE__, __LINE__, \
    #x, CHECK_pretty(CHECK_err), CHECK_err); exit(-1);} } while(0)

/**
 * \brief Create a cuTENSORMg tensor descriptor and corresponding memory buffers
 * \details Distributes the tensor across devices as evenly as possible
 * \param[out] desc cuTENSORMg descriptor for the tensor
 * \param[out] memory Per-device memory buffers for the tensor
 * \param[in] modes Modes of the tensor, indexing into extentMap and blocksizeMap
 * \param[in] extentMap Contains the extent for each mode
 * \param[in] blocksizeMap Contains the block size for each mode
 * \param[in] handle cuTENSORMg handle
 * \param[in] numDevices Number of devices to distribute the tensor over
 **/
void createTensorDescriptor(cutensorMgTensorDescriptor_t &desc,
        std::vector<void*> &memory,
        const std::vector<int32_t> &modes,
        const std::unordered_map<int32_t, int64_t> &extentMap,
        const std::unordered_map<int32_t, int64_t> &blocksizeMap,
        cutensorMgHandle_t handle, int numDevices)
{
    const int kElementSize = 4;
    const cudaDataType_t kDataType = CUDA_R_32F;

    int32_t numModes = modes.size();

    std::vector<int64_t> extent;
    std::vector<int64_t> blocksize;
    std::vector<int32_t> deviceCount(numModes, 1);

    for (auto mode : modes)
    {
        extent.push_back(extentMap.at(mode));
        blocksize.push_back(blocksizeMap.at(mode));
    }

    std::vector<int32_t> devices(numDevices);
    std::iota(devices.begin(), devices.end(), 0);

    int remainingDevices = numDevices;
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (int i = modes.size() - 1; i >= 0 && remainingDevices > 1; i = i - 1)
        {
            int32_t maxDeviceCount = extentMap.at(modes.at(i)) / blocksizeMap.at(modes.at(i));
            if (deviceCount[i] < maxDeviceCount)
            {
                deviceCount[i] *= 2;
                remainingDevices /= 2;
		changed = true;
            }
        }
    }

    int64_t elements = 1;
    for (int i = 0; i < numModes; i++)
    {
        int64_t numBlocks = (extent[i] + blocksize[i] - 1) / blocksize[i];
        int64_t numBlocksPerDevice = (numBlocks + deviceCount[i] - 1) / deviceCount[i];
        elements *= numBlocksPerDevice * blocksize[i];
    }

    printf("Elements=%" PRId64 "\n", elements);

    for (int i = 0; i < numDevices; i++)
    {
        CHECK(cudaSetDevice(i));
        void* ptr;
        CHECK(cudaMalloc(&ptr, elements * kElementSize));
        memory.push_back(ptr);
    }

    CHECK(cutensorMgCreateTensorDescriptor(handle, &desc, numModes,
            extent.data(), NULL, blocksize.data(), NULL,
            deviceCount.data(), numDevices / remainingDevices, devices.data(), kDataType));
}

int main(int argc, char** argv)
{
    if (argc == 1)
    {
        printf("%s <numDevices> <scaling>\n", argv[0]);
        printf("  Simple example of cuTENSORMg across device counts and scales\n");
        printf("  Parameters:\n");
        printf("  <numDevices> Number of devices to run on\n");
        printf("  <scaling>    Scaling factor for the problem size\n\n");
        return EXIT_FAILURE;
    }
    int numDevices = argc >= 2 ? atoi(argv[1]) : 1;
    assert((numDevices & (numDevices - 1)) == 0); // power of two
    assert(numDevices >= 1);

    int scaling = argc >= 3 ? atoi(argv[2]) : 2;
    assert(scaling >= 1);

    std::vector<int32_t> devices(numDevices);
    std::iota(devices.begin(), devices.end(), 0);

    cutensorMgHandle_t handle;
    printf("Initializing cutensorMg handle ... ");
    CHECK(cutensorMgCreate(&handle, devices.size(), devices.data()));
    printf("done.\n");
    
    int32_t M0 = 0, M1 = 1, M2 = 2, N0 = 3, N1 = 4, N2 = 5, K0 = 6, K1 = 7, K2 = 8;

    std::unordered_map<int32_t, int64_t> extent;
    extent[M0] = 16;
    extent[M1] = 8 * scaling;
    extent[M2] = 8;
    extent[N0] = 16;
    extent[N1] = 8 * scaling;
    extent[N2] = 8;
    extent[K0] = 16;
    extent[K1] = 32;
    extent[K2] = 8;

    int numDevicesM = numDevices >= 4 ? numDevices / 2 : numDevices;
    int numDevicesN = numDevices / numDevicesM;
    int M = extent[M0] * extent[M1] * extent[M2];
    int N = extent[N0] * extent[N1] * extent[N2];

    std::unordered_map<int32_t, int64_t> blocksize;
    blocksize[M0] = 16;
    blocksize[M1] = ceil(ceil(M / ceil(M / 4096.0 / numDevicesM)) / numDevicesM / extent[M0] / extent[M2]);
    blocksize[M2] = 8;
    blocksize[N0] = 16;
    blocksize[N1] = ceil(ceil(N / ceil(N / 4096.0 / numDevicesN)) / numDevicesN / extent[N0] / extent[N1]);
    blocksize[N2] = 8;
    blocksize[K0] = 16;
    blocksize[K1] = 16;
    blocksize[K2] = 8;

    std::vector<int32_t> modesA {K0, M0, M1, K1, M2, K2};
    std::vector<int32_t> modesB {K0, N0, K1, N1, K2, N2};
    std::vector<int32_t> modesC {M0, N0, M1, N1, M2, N2};

    printf("Creating distributed tensor descriptors and allocating memory ... ");

    cutensorMgTensorDescriptor_t descA;
    std::vector<void*> memoryA;
    createTensorDescriptor(descA, memoryA, modesA, extent, blocksize, handle, numDevices);

    cutensorMgTensorDescriptor_t descB;
    std::vector<void*> memoryB;
    createTensorDescriptor(descB, memoryB, modesB, extent, blocksize, handle, numDevices);

    cutensorMgTensorDescriptor_t descC;
    std::vector<void*> memoryC;
    createTensorDescriptor(descC, memoryC, modesC, extent, blocksize, handle, numDevices);

    printf("done.\n");

    printf("Creating distributed contraction descriptors ... ");

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

    printf("Allocating workspace memory ... ");

    std::vector<cudaStream_t> streams;
    for (auto& device : devices)
    {
        cudaStream_t stream;
        CHECK(cudaSetDevice(device));
        CHECK(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }

    void* workspaceHost = nullptr;
    CHECK(cudaMallocHost(&workspaceHost, workspaceHostSize));

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

        for (auto& stream : streams)
        {
            CHECK(cudaStreamSynchronize(stream));
        }

        const auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> dur = end - start;
        printf("rep %d: %f ms\n", rep, dur.count());
        if (rep == 0 || minElapsed > dur.count()) {
            minElapsed = dur.count();
        }
    }

    double flops = 2.0;
    for (auto elem : extent)
    {
        flops *= elem.second;
    }
    flops /= (minElapsed * 1e-3); // FLOPS/s

    flops *= 1e-9; // GFLOPS/s

    printf("execution time: %.2e ms.\n", minElapsed);

    printf("execution perf: %.2e GFLOPS/s.\n", flops);

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
