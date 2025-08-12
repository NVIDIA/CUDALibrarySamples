/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include <random>
#include <memory>
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cutensor.h>

// Handle cuTENSOR errors
#define HANDLE_ERROR(x)                                                           \
{                                                                                 \
    const cutensorStatus_t err = (x);                                             \
    if ( err != CUTENSOR_STATUS_SUCCESS )                                         \
    { throw std::runtime_error { std::string { cutensorGetErrorString(err) } }; } \
};

// Handle CUDA errors.
#define HANDLE_CUDA_ERROR(x)                                                  \
{                                                                             \
    const cudaError_t err = (x);                                              \
    if ( err != cudaSuccess )                                                 \
    { throw std::runtime_error { std::string { cudaGetErrorString(err) } }; } \
};

template <typename T>
using cuda_ptr = std::unique_ptr<T,decltype(&cudaFree)>;

template <typename T>
cuda_ptr<T> cuda_alloc( size_t count )
{
    void* result;
    cudaError_t err = cudaMalloc( &result, sizeof(T)*count );
    if ( err != cudaSuccess ) throw std::bad_alloc {};
    else return cuda_ptr<T> { reinterpret_cast<T*>(result), &cudaFree };
}

// Useful for automatic clearing of resources.
template <typename F>
class Guard
{
    F f;
    bool invoke;
public:
    explicit Guard(F x) noexcept: f(std::move(x)), invoke(true) {}
    Guard(Guard &&g) noexcept: f(std::move(g.f)), invoke(g.invoke) { g.invoke=false; }

    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;

    ~Guard() noexcept
    {
        if (invoke) f();
    }
};

template <class F>
inline Guard<F> finally(const F& f) noexcept
{
    return Guard<F>(f);
}

template <class F>
inline Guard<F> finally(F&& f) noexcept
{
    return Guard<F>(std::forward<F>(f));
}



std::mt19937 get_seeded_random_engine() 
{
    using     rand_type = std::random_device::result_type;
    using mersenne_type = std::mt19937::result_type;
    constexpr size_t N  = std::mt19937::state_size * sizeof(mersenne_type);
    constexpr size_t M  =  1 + (N-1)/sizeof(rand_type);

    rand_type random_data[M];
    std::random_device source;
    std::generate(random_data,random_data+M,std::ref(source));
    std::seed_seq seed(random_data,random_data+M);

    return std::mt19937( seed );
}


int main()
try
{
    using          ModeType = int32_t;
    using        ExtentType = int32_t;
    using        StrideType = int64_t;
    using SectionExtentType = int64_t;

    // Random number generator.
    std::mt19937 eng = get_seeded_random_engine();
    std::uniform_real_distribution<double> dist(0.,1.);
    auto rand = [&eng,&dist]() { return dist(eng); };

    // Initialise the library.
    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));
    auto guardHandle = finally( [&handle]() { cutensorDestroy(handle); } );

    //////////////////////////////////////
    // Example:                         //
    // We compute C_i = A_{kil}B_{kl}   //
    //////////////////////////////////////

    std::vector<ModeType> modeA {'k','i','l'};
    std::vector<ModeType> modeB {'k','l'};
    std::vector<ModeType> modeC {'i'};

    std::unordered_map<ModeType, std::vector<SectionExtentType>> sectionExtents;
    sectionExtents['k'] = {10, 10, 15};
    sectionExtents['i'] = {20, 20, 25};
    sectionExtents['l'] = {30, 30, 35};
   
    // Helper-λ to allocate and initialise block-sparse tensors with random
    // data. In this example we use 64-bit double precision numbers.
    cutensorDataType_t dataType = CUTENSOR_R_64F;
    auto initTensor = [handle,&sectionExtents,&rand,dataType]
    (
      const std::vector<ModeType>   &modes,
      const std::vector<ExtentType> &nonZeroCoordinates,
      cutensorBlockSparseTensorDescriptor_t &desc, 
      std::vector<void*> &dev
    ) -> cuda_ptr<double>
    {
        uint32_t numModes         = modes.size();
        uint64_t numNonZeroBlocks = nonZeroCoordinates.size() / numModes;
        std::vector<uint32_t>     numSections;
        std::vector<SectionExtentType>   extents;
        for ( ModeType mode: modes )
        {
            const std::vector<SectionExtentType> &modeExtents = sectionExtents.at(mode);

            numSections.push_back(modeExtents.size());
            extents.insert(extents.end(),modeExtents.begin(),modeExtents.end());
        }

        // We assume packed contiguous storage, column-major order.
        // This means that we may pass nullptr for the strides array later.
        // The offets are used to set the pointers in the dev vector.
        std::vector<StrideType> offsets(numNonZeroBlocks+1); offsets[0]=0;
        for ( uint64_t i = 0; i < numNonZeroBlocks; ++i )
        {
           StrideType size = 1;
           for ( uint32_t j = 0; j < numModes; ++j )
              size *= sectionExtents.at( modes[j] ).at( nonZeroCoordinates[i*numModes+j] ); 
           offsets[i+1]=offsets[i]+size;
        }
        const StrideType totalSize { offsets[numNonZeroBlocks] };

        cuda_ptr<double> buf { cuda_alloc<double>(totalSize) };
        std::vector<double> tmp( totalSize );
        std::generate( tmp.begin(), tmp.end(), rand );
        HANDLE_CUDA_ERROR(cudaMemcpy(buf.get(),tmp.data(),totalSize*sizeof(double),cudaMemcpyHostToDevice));
        tmp.clear();

        dev.resize(numNonZeroBlocks);
        for ( uint64_t i = 0; i < numNonZeroBlocks; ++i )
            dev[i] = buf.get() + offsets[i];

        HANDLE_ERROR(cutensorCreateBlockSparseTensorDescriptor
        (
            handle, &desc,
            numModes, numNonZeroBlocks, numSections.data(), extents.data(),
            nonZeroCoordinates.data(), nullptr, dataType
        ));

        return buf;
    };
                                         

    //////////////
    // Tensor A //
    //////////////

    std::vector<void*> devA;
    cutensorBlockSparseTensorDescriptor_t descA = nullptr;

    // Order-3 Tensor ("box"). E.g., one block in each corner.
    const std::vector<ExtentType> nonZeroCoordinatesA
    {
        0, 0, 0, // Block 0.
        2, 0, 0, // Block 1.
        0, 2, 0, // Block 2.
        2, 2, 0, // Block 3.
        0, 0, 2, // Block 4.
        2, 0, 2, // Block 5.
        0, 2, 2, // Block 6.
        2, 2, 2  // Block 7.
    };

    cuda_ptr<double> bufA = initTensor(modeA,nonZeroCoordinatesA,descA,devA);
    auto guardDescA = finally( [&descA]() { cutensorDestroyBlockSparseTensorDescriptor(descA); } );

    //////////////
    // Tensor B //
    //////////////

    std::vector<void*> devB;
    cutensorBlockSparseTensorDescriptor_t descB = nullptr;

    // Order-2 Tensor ("matrix"), two blocks
    const std::vector<ExtentType> nonZeroCoordinatesB =
    {
        0, 0, // Block 0. 
        1, 2  // Block 1. 
    }; 

    cuda_ptr<double> bufB = initTensor(modeB,nonZeroCoordinatesB,descB,devB);
    auto guardDescB = finally( [&descB]() { cutensorDestroyBlockSparseTensorDescriptor(descB); } );

    //////////////
    // Tensor C //
    //////////////

    std::vector<void*> devC;
    cutensorBlockSparseTensorDescriptor_t descC = nullptr;

    // Order-1 Tensor ("vector"), three blocks
    // Actually not sparse, we specify full.
    const std::vector<ExtentType> nonZeroCoordinatesC =
    {
        0, // Block 0. 
        1, // Block 1. 
        2  // Block 2. 
    }; 

    cuda_ptr<double> bufC = initTensor(modeC,nonZeroCoordinatesC,descC,devC);
    auto guardDescC = finally( [&descC]() { cutensorDestroyBlockSparseTensorDescriptor(descC); } );

    ///////////////////////////////
    // Block-sparse Contraction. //
    ///////////////////////////////

    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateBlockSparseContraction(handle, &desc,
                descA, modeA.data(), CUTENSOR_OP_IDENTITY,
                descB, modeB.data(), CUTENSOR_OP_IDENTITY,
                descC, modeC.data(), CUTENSOR_OP_IDENTITY,
                descC, modeC.data(),
                CUTENSOR_COMPUTE_DESC_64F));
    auto guardOpDesc = finally( [&desc]() { cutensorDestroyOperationDescriptor(desc); } );

    // Currently, block-sparse contraction plans only support default settings.
    cutensorPlanPreference_t planPref = nullptr;

    // Query workspace estimate. For block-sparse contraction plans, this is estimate is exact.
    uint64_t workspaceSize = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,desc,planPref,workspacePref,&workspaceSize));

    cuda_ptr<char> work = cuda_alloc<char>(workspaceSize);

    // Create Contraction Plan
    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,&plan,desc,planPref,workspaceSize));
    auto guardPlan = finally( [&plan]() { cutensorDestroyPlan(plan); } );

    // Execute
    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
    auto guardStream = finally( [&stream]() { cudaStreamDestroy(stream); } );

    double alpha = 1., beta = 0.;
    HANDLE_ERROR(cutensorBlockSparseContract(handle, plan,
                (void*) &alpha, (const void *const *) devA.data(), (const void *const *) devB.data(),
                (void*) &beta,  (const void *const *) devC.data(), (      void *const *) devC.data(), 
                (void*) work.get(), workspaceSize, stream));

    return EXIT_SUCCESS;
}
catch ( std::exception &ex )
{
    std::cerr << "Exception caught! Exiting." << std::endl;
    std::cerr << ex.what() << std::endl;
    return EXIT_FAILURE;
}
catch ( ... )
{
    std::cerr << "Unknown exception caught! Exiting." << std::endl;
    return EXIT_FAILURE;
}

