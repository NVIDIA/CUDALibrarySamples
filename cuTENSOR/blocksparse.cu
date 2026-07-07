/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include <cutensor.h>

#include "utils.cuh"


int main()
try
{
    using          ModeType = int32_t;
    using        ExtentType = int32_t;
    using        StrideType = int64_t;
    using SectionExtentType = int64_t;

    randomgen<double> rand;

    // Initialise the library.
    cutensorHandle_t handle;
    handle_error(cutensorCreate(&handle));
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

        auto buf = cuda_alloc     <double>(totalSize);
        auto tmp = cuda_host_alloc<double>(totalSize);
        std::generate( tmp.get(), tmp.get() + totalSize, rand );
        handle_error(cudaMemcpy(buf.get(),tmp.get(),totalSize*sizeof(double),cudaMemcpyHostToDevice));

        dev.resize(numNonZeroBlocks);
        for ( uint64_t i = 0; i < numNonZeroBlocks; ++i )
            dev[i] = buf.get() + offsets[i];

        handle_error(cutensorCreateBlockSparseTensorDescriptor
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
    handle_error(cutensorCreateBlockSparseContraction(handle, &desc,
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
    handle_error(cutensorEstimateWorkspaceSize(handle,desc,planPref,workspacePref,&workspaceSize));

    cuda_ptr<char> work = cuda_alloc<char>(workspaceSize);

    // Create Contraction Plan
    cutensorPlan_t plan;
    handle_error(cutensorCreatePlan(handle,&plan,desc,planPref,workspaceSize));
    auto guardPlan = finally( [&plan]() { cutensorDestroyPlan(plan); } );

    // Execute
    cudaStream_t stream;
    handle_error(cudaStreamCreate(&stream));
    auto guardStream = finally( [&stream]() { cudaStreamDestroy(stream); } );

    double alpha = 1., beta = 0.;
    handle_error(cutensorBlockSparseContract(handle, plan,
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

