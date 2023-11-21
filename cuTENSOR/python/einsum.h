/*  
        issert tgt != ""
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 * 
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name(s) of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */  

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>
#include <array>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuComplex.h>
#include "cutensor.h"

#define HANDLE_ERROR(x) { const auto err = x;\
    if (err == CUTENSOR_STATUS_NOT_SUPPORTED) { return false; }\
    if (err != CUTENSOR_STATUS_SUCCESS) {printf("cutensor: Error %s in line %d\n", cutensorGetErrorString(err), __LINE__); return false; } }

template<typename U>
struct CuTensorTypeTraits;

template<>
struct CuTensorTypeTraits<double> {
  static cutensorDataType_t getDataType() {return CUTENSOR_R_64F;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_64F;}
  typedef double ScalarType;
};

template<>
struct CuTensorTypeTraits<float> {
  static cutensorDataType_t getDataType() {return CUTENSOR_R_32F;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_32F;}
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<cuDoubleComplex> {
  static cutensorDataType_t getDataType() {return CUTENSOR_C_64F;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_64F;}
  typedef cuDoubleComplex ScalarType;
};

template<>
struct CuTensorTypeTraits<cuComplex> {
  static cutensorDataType_t getDataType() {return CUTENSOR_C_32F;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_32F;}
  typedef cuComplex ScalarType;
};

template<>
struct CuTensorTypeTraits<__half> {
  static cutensorDataType_t getDataType() {return CUTENSOR_R_16F;}
  static const cutensorComputeDescriptor_t getComputeDesc() {return CUTENSOR_COMPUTE_DESC_16F;}
  typedef float ScalarType;
};


template<typename ComputeType,
         typename IntType, int kMaxNumModes_>
struct Einsum
{
    static const std::vector<IntType> emptyVec;

    Einsum(const std::string &equation,
           const std::vector<IntType> &A_shape,
           const std::vector<IntType> &B_shape = emptyVec,
           const cutensorOperator_t opA = CUTENSOR_OP_IDENTITY,
           const cutensorOperator_t opB = CUTENSOR_OP_IDENTITY
           ) :
        numModesA_(A_shape.size()),
        numModesB_(B_shape.size()),
        numModesC_(0),
        opA_(opA),
        opB_(opB),
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
        if (! usesB)
        {
            numModesB_ = 0;
        }

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

    uint64_t getWorksize() const { return kWorksize_; }

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
     * create the contraction plan according to the workspace size provided
     *
     * \param[in] handle cuTensor handle
     * \param[in] workspaceSizeProvided size of the workspace provided by the user
     */
    bool plan(const cutensorHandle_t handle, const uint64_t workspaceSizeProvided)
    {
        if (!isInitialized_) return false;

        cutensorDataType_t cutensorType = CuTensorTypeTraits<ComputeType>::getDataType();
        const cutensorComputeDescriptor_t descCompute = CuTensorTypeTraits<ComputeType>::getComputeDesc();

        const uint32_t kAlignment = 128;
        cutensorTensorDescriptor_t descA;
        HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                    &descA,
                    numModesA_,
                    extentA_.data(),
                    NULL,/*stride*/
                    cutensorType, kAlignment));

        cutensorTensorDescriptor_t descC;
        HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                    &descC,
                    numModesC_,
                    extentC_.data(),
                    NULL,/*stride*/
                    cutensorType, kAlignment));

        cutensorOperationDescriptor_t desc;
        cutensorTensorDescriptor_t descB = nullptr;

        if (numModesB_ > 0)
        {
            // dispatch to contraction

            HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                    &descB,
                    numModesB_,
                    extentB_.data(),
                    NULL,/*stride*/
                    cutensorType, kAlignment));

            /*******************************
             * Create Contraction Descriptor
             *******************************/

            HANDLE_ERROR(cutensorCreateContraction(handle, 
                        &desc,
                        descA, modesA_.data(), /* unary operator A*/opA_,
                        descB, modesB_.data(), /* unary operator B*/opB_,
                        descC, modesC_.data(), /* unary operator C*/CUTENSOR_OP_IDENTITY,
                        descC, modesC_.data(),
                        descCompute));
        }
        else
        {
            /*******************************
             * Create Contraction Descriptor
             *******************************/
            HANDLE_ERROR(cutensorCreateReduction(
                 handle, &desc,
                 descA, modesA_.data(), CUTENSOR_OP_IDENTITY,
                 descC, modesC_.data(), CUTENSOR_OP_IDENTITY,
                 descC, modesC_.data(),
                 CUTENSOR_OP_ADD, descCompute));
        }

        /**************************
         * Set the algorithm to use
         ***************************/
        cutensorPlanPreference_t planPref;
        HANDLE_ERROR(cutensorCreatePlanPreference(
                    handle,
                    &planPref,
                    CUTENSOR_ALGO_DEFAULT,
                    CUTENSOR_JIT_MODE_NONE));

        /**********************
         * Query workspace estimate
         **********************/
        uint64_t workspaceSizeEstimate = 0;
        const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
        HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
                                                  desc,
                                                  planPref,
                                                  workspacePref,
                                                  &workspaceSizeEstimate));

        /**************************
         * Create Contraction Plan
         **************************/
        HANDLE_ERROR(cutensorCreatePlan(handle,
                    &plan_,
                    desc,
                    planPref,
                    workspaceSizeProvided));

        uint64_t actualWorkspaceSize = 0;
        HANDLE_ERROR(cutensorPlanGetAttribute(handle,
                                             plan_,
                                             CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                                             &actualWorkspaceSize,
                                             sizeof(actualWorkspaceSize)));

        // At this point the user knows exactly how much memory is need by the operation and
        // only the smaller actual workspace needs to be allocated
        assert(actualWorkspaceSize <= workspaceSizeEstimate);
        assert(actualWorkspaceSize <= workspaceSizeProvided);
        kWorksize_ = std::max(actualWorkspaceSize, static_cast<uint64_t>(4ULL * 1024ULL * 1024ULL));

        HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
        HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
        HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
        HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
        return true;
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
        const uint32_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
        assert(uintptr_t(A_raw) % kAlignment == 0);
        assert(uintptr_t(B_raw) % kAlignment == 0);
        assert(uintptr_t(C_raw) % kAlignment == 0);

        typename CuTensorTypeTraits<ComputeType>::ScalarType alpha = 1;
        typename CuTensorTypeTraits<ComputeType>::ScalarType beta = 0;

        if (numModesB_ > 0)
        {
            HANDLE_ERROR(cutensorContract(handle,
                               plan_,
                               (void*) &alpha, A_raw, B_raw,
                               (void*) &beta,  C_raw, C_raw, 
                               work_raw, kWorksize_, stream));
        }
        else
        {
            // dispatch to reduction
            HANDLE_ERROR(cutensorReduce(handle, plan_,
                    (const void*)&alpha, A_raw,
                    (const void*)&beta,  C_raw, 
                    C_raw, work_raw, kWorksize_, stream));
        }
        return true;
    }

    bool isInitialized() const { return isInitialized_; }

    ~Einsum()
    {
        if (isInitialized_)
        {
            cutensorDestroyPlan(plan_);
        }
    }

    private:
    uint64_t kWorksize_ = 1024ULL * 1024ULL * 1024ULL;
    cutensorPlan_t plan_;
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
    cutensorOperator_t opA_ = CUTENSOR_OP_IDENTITY;
    cutensorOperator_t opB_ = CUTENSOR_OP_IDENTITY;
};

inline cutensorHandle_t CreateCuTensorHandle() {
  cutensorHandle_t handle;
  cutensorCreate(&handle);
  return handle;
}

inline cutensorHandle_t GetCuTensorHandle() {
  static thread_local cutensorHandle_t handle = CreateCuTensorHandle();
  return handle;
}

