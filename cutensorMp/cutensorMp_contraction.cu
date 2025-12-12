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

#include "cutensorMp.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cuComplex.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <numeric>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

#define CUDA_CHECK(x)                                                                 \
    {                                                                                 \
        const auto err = x;                                                           \
        if (err != cudaSuccess)                                                       \
        {                                                                             \
            printf("CUDA error: %s in line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(-1);                                                                 \
        }                                                                             \
    }

#define MPI_CHECK(x)                                                          \
    {                                                                         \
        int status = x;                                                       \
        if (status != MPI_SUCCESS)                                            \
        {                                                                     \
            char errstr[MPI_MAX_ERROR_STRING];                                \
            int errlen = 0;                                                   \
            MPI_Error_string(status, errstr, &errlen);                        \
            printf("MPI error: %.*s in line %d\n", errlen, errstr, __LINE__); \
            exit(-1);                                                         \
        }                                                                     \
    }

#define NCCL_CHECK(x)                                                                    \
    {                                                                                    \
        ncclResult_t status = x;                                                         \
        if (status != ncclSuccess)                                                       \
        {                                                                                \
            printf("NCCL error: %s in line %d\n", ncclGetErrorString(status), __LINE__); \
            exit(-1);                                                                    \
        }                                                                                \
    }

#define CUTENSOR_CHECK(x)                                                            \
    {                                                                                \
        const auto err = x;                                                          \
        if (err != CUTENSOR_STATUS_SUCCESS)                                          \
        {                                                                            \
            printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); \
            exit(-1);                                                                \
        }                                                                            \
    }

inline int getLocalDevice()
{
    int localRank;
    MPI_Comm localComm;

    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm));
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&localComm));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    return localRank % deviceCount;
}

void comm_init(ncclComm_t* comm, int* rank, int* nranks, int* local_device)
{
    MPI_CHECK(MPI_Init(nullptr, nullptr));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, nranks));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, rank));
    *local_device = getLocalDevice();
    CUDA_CHECK(cudaSetDevice(*local_device));
    CUDA_CHECK(cudaFree(nullptr));

    // Initialize NCCL communicator
    ncclUniqueId ncclId;
    if (*rank == 0)
    {
        NCCL_CHECK(ncclGetUniqueId(&ncclId));
    }
    MPI_CHECK(MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCL_CHECK(ncclCommInitRank(comm, *nranks, ncclId, *rank));
    if (*rank == 0)
    {
        printf("NCCL initialized\n");
    }
}

void comm_barrier(ncclComm_t /*comm*/, cudaStream_t stream) { CUDA_CHECK(cudaStreamSynchronize(stream)); }

// Simple helpers to keep this example standalone
static inline int64_t ceil_div_int64(int64_t a, int64_t b) { return (a + b - 1) / b; }

static inline size_t accumulate_size(std::vector<int64_t> const& extents)
{
    size_t prod = 1;
    for (auto e : extents)
        prod *= static_cast<size_t>(e);
    return prod;
}

static inline std::vector<int64_t> get_nranks_per_mode(std::vector<int32_t> const& modes,
                                                       std::vector<int32_t> const& distributed_modes,
                                                       std::vector<int64_t> const& extents)
{
    std::vector<int64_t> nranks_per_mode(modes.size(), 1);
    for (size_t i = 0; i < modes.size(); ++i)
    {
        if (std::find(distributed_modes.begin(), distributed_modes.end(), modes[i]) != distributed_modes.end())
        {
            nranks_per_mode[i] = extents[i];
        }
    }
    return nranks_per_mode;
}

static inline std::vector<int64_t> calc_local_extents(std::vector<int64_t> const& global,
                                                      std::vector<int64_t> const& nranksPerMode)
{
    std::vector<int64_t> local(global.size(), 0);
    for (size_t i = 0; i < global.size(); ++i)
    {
        int64_t const div = std::max<int64_t>(1, nranksPerMode[i]);
        local[i] = ceil_div_int64(global[i], div);
    }
    return local;
}

template <typename TT>
static inline void fast_generate_random_tensor_device(std::vector<int64_t> const& extents, TT* d_ptr, int seed)
{
    // Simple memset-based filler to keep this example standalone
    size_t const num_bytes = accumulate_size(extents) * sizeof(TT);
    unsigned char const byte = static_cast<unsigned char>(seed & 0xFF);
    CUDA_CHECK(cudaMemset(d_ptr, byte, num_bytes));
}

static inline void print_performance_metrics(std::vector<int32_t> const& modeA, std::vector<int32_t> const& modeB,
                                             std::vector<int32_t> const& modeC, std::vector<int64_t> const& extentA,
                                             std::vector<int64_t> const& extentB, std::vector<int64_t> const& extentC,
                                             double const min_ms)
{
    int num_k_modes = (modeA.size() + modeB.size() - modeC.size()) / 2;
    int num_unique_modes = num_k_modes + modeC.size();
    size_t totalA = accumulate_size(extentA);
    size_t totalB = accumulate_size(extentB);
    size_t totalC = accumulate_size(extentC);
    double gb_per_rank =
        (static_cast<double>(totalA + totalB + totalC) * sizeof(cuComplex)) / (1024.0 * 1024.0 * 1024.0);
    double gflops_per_rank = (std::pow(2, num_k_modes) * totalC * 8.0) / (1024.0 * 1024.0 * 1024.0);
    double gb_per_rank_per_sec = gb_per_rank / (min_ms / 1000.0);
    double gflops_per_rank_per_sec = gflops_per_rank / (min_ms / 1000.0);
    printf(
        "Performance Summary: min_time: %.3f ms, data size: ~%.3f GB/rank, gflops: %.3f GFLOPS/rank, "
        "bandwidth: ~%.3f GB/rank/s, gflops: ~%.3f GFLOPS/rank/s\n",
        min_ms, gb_per_rank, gflops_per_rank, gb_per_rank_per_sec, gflops_per_rank_per_sec);
}

static inline void comm_destroy(ncclComm_t comm)
{
    ncclCommDestroy(comm);
}

std::vector<std::vector<char>> split_equation(std::string const& eq)
{
    // Split equation into left-hand side and right-hand side at "->"
    auto const arrow_pos = eq.find("->");
    std::string lhs = eq.substr(0, arrow_pos);
    std::string rhs = eq.substr(arrow_pos + 2);

    // Remove spaces
    lhs.erase(remove_if(lhs.begin(), lhs.end(), isspace), lhs.end());
    rhs.erase(remove_if(rhs.begin(), rhs.end(), isspace), rhs.end());

    std::vector<std::vector<char>> results;

    // Split left-hand side into input tensor modes at ","
    std::stringstream lhs_stream(lhs);
    std::string tensor_modes;
    while (std::getline(lhs_stream, tensor_modes, ','))
    {
        std::vector<char> mode_vec;
        for (char c : tensor_modes)
        {
            mode_vec.push_back(c);
        }
        results.push_back(mode_vec);
    }

    // Add output tensor modes to results
    std::vector<char> output_modes;
    for (char c : rhs)
    {
        output_modes.push_back(c);
    }
    results.push_back(output_modes);

    return results;
}

std::vector<int64_t> get_extent(std::vector<int32_t> const& modes, int64_t const extent = 2)
{
    std::vector<int64_t> extents;
    for (int32_t mode : modes)
    {
        extents.push_back(extent);
    }
    return extents;
}

// only support 1 process per GPU
int main(int argc, char** argv)
{
    ncclComm_t comm;
    int rank, nranks, local_device;

    comm_init(&comm, &rank, &nranks, &local_device);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // wait for all processes to initialize
    comm_barrier(comm, stream);

    // Parse command line for optional controls
    // --eq <equation>
    // --distA <modes> --distB <modes> --distC <modes> (default: "")
    std::string eq = "abcdefghijEFGHIJKLMNOPQRSTUVWXYZ,abcdefghijABCD->EFGHIJKLMNOPQRSTUVWAXBYCZD";
    std::string distA, distB, distC;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a.rfind("--eq=", 0) == 0)
        {
            eq = a.substr(5);
        }
        else if (a == "--eq" && (i + 1) < argc)
        {
            eq = argv[++i];
        }
        else if (a.rfind("--distA=", 0) == 0)
        {
            distA = a.substr(9);
        }
        else if (a == "--distA" && (i + 1) < argc)
        {
            distA = argv[++i];
        }
        else if (a.rfind("--distB=", 0) == 0)
        {
            distB = a.substr(9);
        }
        else if (a == "--distB" && (i + 1) < argc)
        {
            distB = argv[++i];
        }
        else if (a.rfind("--distC=", 0) == 0)
        {
            distC = a.substr(9);
        }
        else if (a == "--distC" && (i + 1) < argc)
        {
            distC = argv[++i];
        }
    }

    // get the device count and local extent
    if (rank == 0)
    {
        printf("Getting the device count and local extent ...\n");
    }
    // get the equation parts
    std::vector<std::vector<char>> eqs = split_equation(eq);
    assert(eqs.size() == 3);
    if (rank == 0)
    {
        printf("split_equation: %s\n", eq.c_str());
        printf("modeA: ");
        for (auto& m : eqs[0])
        {
            printf("%c ", m);
        }
        printf("\n");
        printf("modeB: ");
        for (auto& m : eqs[1])
        {
            printf("%c ", m);
        }
        printf("\n");
        printf("modeC: ");
        for (auto& m : eqs[2])
        {
            printf("%c ", m);
        }
        printf("\n");
    }

    // get the mode and convert to int32_t
    std::vector<int32_t> modeA(eqs[0].begin(), eqs[0].end());
    std::vector<int32_t> modeB(eqs[1].begin(), eqs[1].end());
    std::vector<int32_t> modeC(eqs[2].begin(), eqs[2].end());

    // get the extent
    std::vector<int64_t> extentA = get_extent(modeA, 2);
    std::vector<int64_t> extentB = get_extent(modeB, 2);
    std::vector<int64_t> extentC = get_extent(modeC, 2);

    if (rank == 0)
    {
        printf("extentA: ");
        for (auto& e : extentA)
        {
            printf("%ld ", e);
        }
        printf("\n");
        printf("extentB: ");
        for (auto& e : extentB)
        {
            printf("%ld ", e);
        }
        printf("\n");
        printf("extentC: ");
        for (auto& e : extentC)
        {
            printf("%ld ", e);
        }
        printf("\n");
    }

    // Determine distributed modes per tensor
    std::vector<int32_t> distributed_modes_A;
    std::vector<int32_t> distributed_modes_B;
    std::vector<int32_t> distributed_modes_C;

    auto build_dist = [](std::vector<int32_t> const& modes, std::string const& s)
    {
        std::string cleaned = s;
        cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ','), cleaned.end());
        std::vector<int32_t> out;
        for (auto m : modes)
        {
            if (!cleaned.empty() && cleaned.find(static_cast<char>(m)) != std::string::npos)
            {
                out.push_back(m);
            }
        }
        return out;
    };

    if (!distA.empty() || !distB.empty() || !distC.empty())
    {
        distributed_modes_A = build_dist(modeA, distA);
        distributed_modes_B = build_dist(modeB, distB);
        distributed_modes_C = build_dist(modeC, distC);
        if (rank == 0)
        {
            printf("distA='%s', distB='%s', distC='%s'\n", distA.c_str(), distB.c_str(), distC.c_str());
        }
    }

    // update the device count based on the distributed modes
    // omit the second input tensor since it is not distributed
    std::vector<int64_t> nranksPerModeA = get_nranks_per_mode(modeA, distributed_modes_A, extentA);
    std::vector<int64_t> nranksPerModeB = get_nranks_per_mode(modeB, distributed_modes_B, extentB);
    std::vector<int64_t> nranksPerModeC = get_nranks_per_mode(modeC, distributed_modes_C, extentC);

    if (rank == 0)
    {
        printf("nranksPerModeA: ");
        for (auto& c : nranksPerModeA)
        {
            printf("%ld ", c);
        }
        printf("\n");
        printf("nranksPerModeB: ");
        for (auto& c : nranksPerModeB)
        {
            printf("%ld ", c);
        }
        printf("\n");
        printf("nranksPerModeC: ");
        for (auto& c : nranksPerModeC)
        {
            printf("%ld ", c);
        }
        printf("\n");
    }

    // Calculate local extents for each tensor
    std::vector<int64_t> localExtentA = calc_local_extents(extentA, nranksPerModeA);
    std::vector<int64_t> localExtentB = calc_local_extents(extentB, nranksPerModeB);
    std::vector<int64_t> localExtentC = calc_local_extents(extentC, nranksPerModeC);

    if (rank == 0)
    {
        printf("localExtentA: ");
        for (auto& e : localExtentA)
        {
            printf("%ld ", e);
        }
        printf("\n");
        printf("localExtentB: ");
        for (auto& e : localExtentB)
        {
            printf("%ld ", e);
        }
        printf("\n");
        printf("localExtentC: ");
        for (auto& e : localExtentC)
        {
            printf("%ld ", e);
        }
        printf("\n");
    }
    if (rank == 0)
    {
        printf("Done: getting the input parameters.\n");
    }

    // initialize the tensors
    size_t total_size_A = accumulate_size(extentA);
    size_t total_size_B = accumulate_size(extentB);
    size_t total_size_C = accumulate_size(extentC);

    cuComplex* d_A = nullptr;
    cuComplex* d_B = nullptr;
    cuComplex* d_C = nullptr;

    size_t total_size_A_partial = accumulate_size(localExtentA);
    size_t total_size_B_partial = accumulate_size(localExtentB);
    size_t total_size_C_partial = accumulate_size(localExtentC);
    CUDA_CHECK(cudaMalloc(&d_A, total_size_A_partial * sizeof(cuComplex)));
    CUDA_CHECK(cudaMalloc(&d_B, total_size_B_partial * sizeof(cuComplex)));
    CUDA_CHECK(cudaMalloc(&d_C, total_size_C_partial * sizeof(cuComplex)));

    fast_generate_random_tensor_device(localExtentA, d_A, rank);
    fast_generate_random_tensor_device(localExtentB, d_B, rank * nranks + rank + 1);

    /***************************************** cuTensorMp compute start *****************************************/
    cutensorDataType_t const kDataType = CUTENSOR_C_32F;  // cuComplex
    cutensorComputeDescriptor_t const descCompute = CUTENSOR_COMPUTE_DESC_32F;

    // initialize cuTensorMp handle
    cutensorMpHandle_t handle;
    CUTENSOR_CHECK(cutensorMpCreate(&handle, comm, local_device, stream));

    cutensorMpTensorDescriptor_t descA;
    CUTENSOR_CHECK(cutensorMpCreateTensorDescriptor(handle, &descA, modeA.size(), extentA.data(), nullptr, nullptr,
                                                    nullptr, nranksPerModeA.data(), nranks, nullptr, kDataType));

    cutensorMpTensorDescriptor_t descB;
    CUTENSOR_CHECK(cutensorMpCreateTensorDescriptor(handle, &descB, modeB.size(), extentB.data(), nullptr, nullptr,
                                                    nullptr, nranksPerModeB.data(), nranks, nullptr, kDataType));

    cutensorMpTensorDescriptor_t descC;
    CUTENSOR_CHECK(cutensorMpCreateTensorDescriptor(handle, &descC, modeC.size(), extentC.data(), nullptr, nullptr,
                                                    nullptr, nranksPerModeC.data(), nranks, nullptr, kDataType));

    cutensorMpOperationDescriptor_t desc;
    CUTENSOR_CHECK(cutensorMpCreateContraction(handle, &desc, descA, modeA.data(), CUTENSOR_OP_IDENTITY, descB,
                                               modeB.data(), CUTENSOR_OP_IDENTITY, descC, modeC.data(),
                                               CUTENSOR_OP_IDENTITY, descC, modeC.data(), descCompute));

    cutensorMpPlanPreference_t planPref;
    // buget memory size for cuTensor and cuTensorMp excution buffer
    uint64_t cutensormp_workspace_device_budget = 10ULL * 1024ULL * 1024ULL * 1024ULL;
    uint64_t cutensormp_workspace_host_budget = 1024ULL;  // not used yet, just for test
    cutensorMpAlgo_t cutensormp_algo = CUTENSORMP_ALGO_DEFAULT;
    if (rank == 0)
    {
        printf("cuTensorMp using algo %d.\n", cutensormp_algo);
    }
    CUTENSOR_CHECK(cutensorMpCreatePlanPreference(
        handle, &planPref, cutensormp_algo, cutensormp_workspace_device_budget, cutensormp_workspace_host_budget));

    cutensorMpPlan_t plan;
    CUTENSOR_CHECK(cutensorMpCreatePlan(handle, &plan, desc, planPref));

    uint64_t cutensormp_workspace_device_actual = 0, cutensormp_workspace_host_actual = 0;
    CUTENSOR_CHECK(cutensorMpPlanGetAttribute(handle, plan, CUTENSORMP_PLAN_REQUIRED_WORKSPACE_DEVICE,
                                              &cutensormp_workspace_device_actual, sizeof(uint64_t)));
    CUTENSOR_CHECK(cutensorMpPlanGetAttribute(handle, plan, CUTENSORMP_PLAN_REQUIRED_WORKSPACE_HOST,
                                              &cutensormp_workspace_host_actual, sizeof(uint64_t)));

    if (rank == 0)
    {
        printf("cutensormp_workspace_device_actual: %ld\n", cutensormp_workspace_device_actual);
        printf("cutensormp_workspace_host_actual: %ld\n", cutensormp_workspace_host_actual);
    }

    void* workspace_device = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace_device, cutensormp_workspace_device_actual));
    void* workspace_host = nullptr;
    CUDA_CHECK(cudaMallocHost(&workspace_host, cutensormp_workspace_host_actual));

    cuComplex kAlpha = make_cuComplex(1.0f, 0.0f);
    cuComplex kBeta = make_cuComplex(0.0f, 0.0f);

    int const nRep = 5;  // for stable timings
    std::vector<double> times;
    for (int rep = 0; rep < nRep; rep++)
    {
        CUDA_CHECK(cudaMemset(d_C, 0, total_size_C_partial * sizeof(cuComplex)));
        comm_barrier(comm, stream);
        if (rep == 1)
        {
            CUDA_CHECK(cudaProfilerStart());
        }

        double const begin = MPI_Wtime();
        CUTENSOR_CHECK(
            cutensorMpContract(handle, plan, &kAlpha, d_A, d_B, &kBeta, d_C, d_C, workspace_device, workspace_host));

        comm_barrier(comm, stream);
        if (rep == nRep - 2)
        {
            CUDA_CHECK(cudaProfilerStop());
        }
        double const end = MPI_Wtime();
        double dur = (end - begin) * 1000.0;  // Convert to milliseconds
        if (rep > 0)
        {  // Skip first iteration for timing (warmup)
            times.push_back(dur);
        }
    }

    if (rank == 0)
    {
        if (times.size() > 0)
        {
            double total_time = 0;
            double min_time = times[0];
            for (double t : times)
            {
                total_time += t;
                if (t < min_time)
                    min_time = t;
            }
            double avg_time = total_time / times.size();
            printf("\ncuTensorMp - average time: %.3fms, min time: %.3fms\n", avg_time, min_time);

            // Calculate and print performance metrics
            print_performance_metrics(modeA, modeB, modeC, localExtentA, localExtentB, localExtentC, min_time);
            printf("\n");
        }
    }

    /***************************************** cuTensorMp compute end *****************************************/

    // Synchronize all processes before starting cleanup
    comm_barrier(comm, stream);

    if (rank == 0)
    {
        printf("\nFree resources ...\n");
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUTENSOR_CHECK(cutensorMpDestroyPlan(plan));
    CUTENSOR_CHECK(cutensorMpDestroyTensorDescriptor(descA));
    CUTENSOR_CHECK(cutensorMpDestroyTensorDescriptor(descB));
    CUTENSOR_CHECK(cutensorMpDestroyTensorDescriptor(descC));
    CUTENSOR_CHECK(cutensorMpDestroyOperationDescriptor(desc));
    CUTENSOR_CHECK(cutensorMpDestroyPlanPreference(planPref));
    CUTENSOR_CHECK(cutensorMpDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(workspace_device));
    CUDA_CHECK(cudaFreeHost(workspace_host));
    comm_destroy(comm);

    if (rank == 0)
    {
        printf("\nDone: everything has completed successfully.\n");
    }
    return 0;  // Added return statement
}
