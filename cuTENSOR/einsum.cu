/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutensor.h>

#include "utils.cuh"

template<typename U>
struct CuTensorTypeTraits;

template<>
struct CuTensorTypeTraits<double> {
  static cudaDataType_t getDataType() {return CUDA_R_64F;}
  static cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_64F;}
  typedef double ScalarType;
};

template<>
struct CuTensorTypeTraits<float> {
  static cudaDataType_t getDataType() {return CUDA_R_32F;}
  static cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_32F;}
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<__half> {
  static cudaDataType_t getDataType() {return CUDA_R_16F;}
  static cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_16F;}
  typedef float ScalarType;
};

template<typename ComputeType,
         typename IntType, int kMaxNumModes_>
struct Einsum
{
    static const std::vector<IntType> emptyVec;

    Einsum(const std::string &equation,
           const std::vector<IntType> &A_shape,
           const std::vector<IntType> &B_shape = emptyVec) :
        numModesA_(A_shape.size()),
        numModesB_(B_shape.size()),
        numModesC_(0),
        isInitialized_(false)
    {
        const auto arrow_pos = equation.find("->");
        const auto comma_pos = equation.find(",");
        const auto dots = equation.find("...");
        const bool isBroadcast = (dots != std::string::npos);
        const bool isImplicit = (arrow_pos == std::string::npos);
        if (isBroadcast) // TODO
        {
            return;
        }
        const bool usesB = (comma_pos != std::string::npos);

        size_t a_start = 0;
        size_t a_end = isImplicit ? ((comma_pos == std::string::npos) ? equation.size() : comma_pos) : 
                                    ((comma_pos == std::string::npos) ? arrow_pos : comma_pos);
        size_t b_start = usesB ? comma_pos + 1 : 0;
        size_t b_end   = usesB ? (isImplicit ? equation.size() : arrow_pos) : 0;
        size_t c_start = isImplicit ? equation.size() : arrow_pos + 2;
        size_t c_end = equation.size();


        char modeA[kMaxNumModes_ + 2];
        uint32_t numModesA = 0;
        for (int i = a_start; i < a_end && numModesA < kMaxNumModes_ + 2; ++i){
            if (equation.at(i) != ' ') // skip spaces
            {
                modeA[numModesA++] = equation.at(i);
            }
        }

        char modeB[kMaxNumModes_ + 2];
        uint32_t numModesB = 0;
        for (int i = b_start; i < b_end && numModesB < kMaxNumModes_ + 2; ++i){
            if (equation.at(i) != ' ') // skip spaces
            {
                modeB[numModesB++] = equation.at(i);
            }
        }

        char modeC[kMaxNumModes_ + 2];
        uint32_t numModesC = 0;
        for (int i = c_start; i < c_end && numModesC < kMaxNumModes_ + 2; ++i){
            if (equation.at(i) != ' ') // skip spaces
            {
                modeC[numModesC++] = equation.at(i);
            }
        }

        if ((numModesA != numModesA_) || (numModesB != numModesB_))
        {
            // substring size and shape don't match
            return;
        }
        if (numModesA_ > kMaxNumModes_ || numModesB_ > kMaxNumModes_)
        {
            // too many modes
            return;
        }

        /**
         * Copy all modes from modeA to modeC if they don't appear in modeB
         */
        auto copyModesIf = [](const char* modeA, uint32_t numModesA,
                const char* modeB, uint32_t numModesB,
                char* modeC, uint32_t &numModesC)
        {
            for (uint32_t i = 0; i < numModesA; i++)
            {
                auto mode = modeA[i];
                bool found = false;
                for(uint32_t j=0; j < numModesB; ++j){
                    if(mode == modeB[j])
                    {
                        found = true;
                        break;
                    }
                }

                if (!found) // is non-contracted mode
                {
                    modeC[numModesC++] = mode;
                    if (numModesC > kMaxNumModes_)
                    {
                        // too many modes
                        return false;
                    }
                }
            }
            return true;
        };


        std::array<char, kMaxNumModes_+1> implicitModeC;
        char* redirectModeC;
        if (isImplicit)
        {
            // we have to copy all non-contracted modes from A over to C
            if (copyModesIf(modeA, numModesA_, modeB, numModesB_, implicitModeC.data(), numModesC_) == false)
            {
                return;
            }
            // we have to copy all non-contracted modes from B over to C
            if (copyModesIf(modeB, numModesB_, modeA, numModesA_, implicitModeC.data(), numModesC_) == false)
            {
                return;
            }
            std::sort(implicitModeC.begin(), std::next(implicitModeC.begin(), numModesC_)); // modes are sorted w.r.t. lexical order
            implicitModeC[numModesC_] = '\0';
            redirectModeC = implicitModeC.data();
        }
        else
        {
            redirectModeC = modeC;
            numModesC_ = numModesC;
        }

        for (uint32_t i = 0; i < numModesA_; i++)
        {
            modesA_[i] = modeA[numModesA_ - i - 1];
            extentA_[i] = A_shape[numModesA_ - i - 1];
        }

        for (uint32_t i = 0; i < numModesB_; i++)
        {
            modesB_[i] = modeB[numModesB_ - i - 1];
            extentB_[i] = B_shape[numModesB_ - i - 1];
        }

        for (uint32_t i = 0; i < numModesC_; i++)
        {
            const auto mode = redirectModeC[numModesC_ - i - 1];
            modesC_[i] = mode;
            bool found = false;
            for (uint32_t j=0; j < numModesA_; ++j)
            {
                if (modesA_[j] == mode)
                {
                    extentC_[i] = extentA_[j];
                    found = true;
                    break;
                }
            }
            for (uint32_t j=0; !found && j < numModesB_; ++j)
            {
                if (modesB_[j] == mode)
                {
                    extentC_[i] = extentB_[j];
                    break;
                }
            }
        }

        isInitialized_ = true;
    }

    size_t getWorksize() const { return kWorksize_; }

    std::vector<IntType> getOutputShape() const
    {
        if (!isInitialized_) return {};
        std::vector<IntType> extentC(numModesC_);
        for (int i=0; i < numModesC_; ++i)
        {
            extentC[i] = extentC_.at(numModesC_ - i - 1);
        }

        return extentC;
    }

    /**
     * Computes the einsum call A,B->C
     *
     * \param[in] A_raw device pointer of A
     * \param[in] B_raw device pointer of B
     * \param[out] C_raw device pointer of C
     * \param[out] wor_raw device pointer to the scratchpad memory
     * Dispatch to contraction
     */
    bool execute(const cutensorHandle_t handle,
                 const void* A_raw,
                 const void* B_raw,
                 void* C_raw,
                 void *work_raw, cudaStream_t stream) const
    {
        if (!isInitialized_) return false;

        cudaDataType_t cutensorType = CuTensorTypeTraits<ComputeType>::getDataType();
        const cutensorComputeDescriptor_t descCompute = CuTensorTypeTraits<ComputeType>::getComputeDesc();

        const uint32_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
        assert(uintptr_t(A_raw) % kAlignment == 0);
        assert(uintptr_t(B_raw) % kAlignment == 0);
        assert(uintptr_t(C_raw) % kAlignment == 0);

        cutensorTensorDescriptor_t descA;
        handle_error(cutensorCreateTensorDescriptor(handle,
                    &descA,
                    numModesA_,
                    extentA_.data(),
                    NULL,/*stride*/
                    cutensorType, kAlignment));
        auto guardDescA = finally( [&descA]() { cutensorDestroyTensorDescriptor(descA); } );

        cutensorTensorDescriptor_t descC;
        handle_error(cutensorCreateTensorDescriptor(handle,
                    &descC,
                    numModesC_,
                    extentC_.data(),
                    NULL,/*stride*/
                    cutensorType, kAlignment));
        auto guardDescC = finally( [&descC]() { cutensorDestroyTensorDescriptor(descC); } );


        /**************************
         * Set the algorithm to use
         ***************************/
        cutensorPlanPreference_t planPref;
        handle_error(cutensorCreatePlanPreference(
                    handle,
                    &planPref,
                    CUTENSOR_ALGO_DEFAULT,
                    CUTENSOR_JIT_MODE_NONE));
        auto guardPlanPref = finally( [&planPref]() { cutensorDestroyPlanPreference(planPref); } );

        if (numModesB_ > 0)
        {
            // dispatch to contraction
            cutensorTensorDescriptor_t descB;
            handle_error(cutensorCreateTensorDescriptor(handle,
                    &descB,
                    numModesB_,
                    extentB_.data(),
                    NULL,/*stride*/
                    cutensorType, kAlignment));
            auto guardDescB = finally( [&descB]() { cutensorDestroyTensorDescriptor(descB); } );

            /*******************************
             * Create Contraction Descriptor
             *******************************/

            cutensorOperationDescriptor_t desc;
            handle_error(cutensorCreateContraction(handle, 
                        &desc,
                        descA, modesA_.data(), /* unary operator A*/CUTENSOR_OP_IDENTITY,
                        descB, modesB_.data(), /* unary operator B*/CUTENSOR_OP_IDENTITY,
                        descC, modesC_.data(), /* unary operator C*/CUTENSOR_OP_IDENTITY,
                        descC, modesC_.data(),
                        descCompute));
            auto guardDesc = finally( [&desc]() { cutensorDestroyOperationDescriptor(desc); } );

            /**************************
             * Create Contraction Plan
             **************************/
            cutensorPlan_t plan;
            handle_error(cutensorCreatePlan(handle,
                        &plan,
                        desc,
                        planPref,
                        kWorksize_));
            auto guardPlan = finally( [&plan]() { cutensorDestroyPlan(plan); } );

            typename CuTensorTypeTraits<ComputeType>::ScalarType alpha = 1;
            typename CuTensorTypeTraits<ComputeType>::ScalarType beta = 0;

            handle_error(cutensorContract(handle,
                               plan,
                               (void*) &alpha, A_raw, B_raw,
                               (void*) &beta,  C_raw, C_raw, 
                               work_raw, kWorksize_, stream));
        }
        else
        {
            /*******************************
             * Create Contraction Descriptor
             *******************************/
            cutensorOperationDescriptor_t desc;
            handle_error(cutensorCreateReduction(
                 handle, &desc,
                 descA, modesA_.data(), CUTENSOR_OP_IDENTITY,
                 descC, modesC_.data(), CUTENSOR_OP_IDENTITY,
                 descC, modesC_.data(),
                 CUTENSOR_OP_ADD, descCompute));
            auto guardDesc = finally( [&desc]() { cutensorDestroyOperationDescriptor(desc); } );

            /**************************
             * Create Contraction Plan
             **************************/
            cutensorPlan_t plan;
            handle_error(cutensorCreatePlan(handle,
                        &plan,
                        desc,
                        planPref,
                        kWorksize_));
            auto guardPlan = finally( [&plan]() { cutensorDestroyPlan(plan); } );

            // dispatch to reduction
            typename CuTensorTypeTraits<ComputeType>::ScalarType alpha = 1;
            typename CuTensorTypeTraits<ComputeType>::ScalarType beta = 0;

            handle_error(cutensorReduce(handle, plan,
                    (const void*)&alpha, A_raw,
                    (const void*)&beta,  C_raw, 
                    C_raw, work_raw, kWorksize_, stream));
        }
        return true;
    }

    bool isInitialized() const { return isInitialized_; }

    private:
    static const size_t kWorksize_ = 1024ULL * 1024ULL * 8ULL * 128ULL;
    uint32_t numModesA_;
    uint32_t numModesB_;
    uint32_t numModesC_;
    bool isInitialized_;
    std::array<int, kMaxNumModes_> modesA_;
    std::array<int, kMaxNumModes_> modesB_;
    std::array<int, kMaxNumModes_> modesC_;
    std::array<int64_t, kMaxNumModes_> extentA_;
    std::array<int64_t, kMaxNumModes_> extentB_;
    std::array<int64_t, kMaxNumModes_> extentC_;
};

void einsum(cutensorHandle_t handle,
            const std::vector<int> &A_shape,
            const std::vector<int> &B_shape,
            const std::string &subscripts)
{
    constexpr int kMaxNumModes_ = 40; // maximal number of modes supported by cuTENSOR
    typedef float Compute;

    Einsum<Compute, int, kMaxNumModes_> myEinsum(subscripts, A_shape, B_shape);
    if (!myEinsum.isInitialized()) {
        return;
    }

    size_t totalElementsA = 1;
    for (const auto e : A_shape) {
        totalElementsA *= e;
    }
    size_t totalElementsB = 1;
    for (const auto e : B_shape) {
        totalElementsB *= e;
    }
    auto C_shape = myEinsum.getOutputShape();
    size_t totalElementsC = 1;
    for (const auto e : C_shape) {
        totalElementsC *= e;
    }

    auto         A_raw = cuda_alloc<Compute>(totalElementsA);
    auto         B_raw = cuda_alloc<Compute>(totalElementsB);
    auto    output_raw = cuda_alloc<Compute>(totalElementsC);
    auto workspace_raw = cuda_alloc<char>   (myEinsum.getWorksize());
    
    auto ret = myEinsum.execute(handle, A_raw.get(), B_raw.get(), output_raw.get(), workspace_raw.get(), 0);

    if (!ret) {
        printf("%s: not supported\n", subscripts.c_str());
    }else{
        printf("%s: succeeded\n", subscripts.c_str());
    }
}

int main()
try
{
    cutensorHandle_t handle;
    handle_error( cutensorCreate(&handle) );
    auto guardHandle = finally( [&handle]() { cutensorDestroy(handle); } );

    /**********************
     * Setup planCache (optional)
     **********************/
    constexpr int32_t numCachelines = 1024;
    handle_error( cutensorHandleResizePlanCache(handle, numCachelines) );
  
    einsum(handle, {2, 4, 5}, {4, 8, 7}, "ijn,jmk->inkm"); // contraction (explict)
    einsum(handle, {2, 4, 5}, {4, 8, 7}, "ijn,jmk"); // contraction (implicit)
    einsum(handle, {2, 4, 5}, {}, "nij");  // permutation (implicit)
    einsum(handle, {2, 4, 5}, {}, "nij->ijn");  // permutation (same as previous example, but explicit)
    einsum(handle, {2, 4, 5}, {}, "nij->ji"); // reduction

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

