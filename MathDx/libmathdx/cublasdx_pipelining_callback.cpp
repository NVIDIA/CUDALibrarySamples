/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <libcublasdx.h>
#include <nvJitLink.h>
#include <nvrtc.h>

#include <array>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "common_examples.hpp"
#include "macros.hpp"

using namespace examples;

int main() {

    long long int m = 512;
    long long int n = 512;
    long long int k = 512;
    long long int tile_m = 128;
    long long int tile_n = 128;
    long long int tile_k = 32;
    long long int num_threads = 128;

    arch_t dx_sm = maybe_accelerated_dx(get_dx_cc());
    arch_t target_sm = maybe_accelerated_target(get_target_cc());

    auto dx_sm_array = dx_sm.to_array();
    auto target_sm_array = target_sm.to_array();

    auto [nvrtc_major, nvrtc_minor] = get_nvrtc_version();
    if (nvrtc_major == 13 && nvrtc_minor == 0 && target_sm.cc >= cc_t { 10, 0 }) {
        printf("Pipeline examples on SM100+ requires NVRTC 13.1 or above.\n");
        return 0;
    }

    if (dx_sm.cc < cc_t { 7, 5 }) {
        printf("Pipeline examples require SM75 or above.\n");
        return 0;
    }

    /**
     * Create the cuBLASDx descriptor
     */
    cublasdxDescriptor h { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDescriptor(&h));

    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_FUNCTION, cublasdxFunction::CUBLASDX_FUNCTION_MM));
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK));
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_API, cublasdxApi::CUBLASDX_API_TENSORS));
    std::array<long long int, 3> prec = { COMMONDX_PRECISION_I8, COMMONDX_PRECISION_I8, COMMONDX_PRECISION_I32 };
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64s(h, cublasdxOperatorType::CUBLASDX_OPERATOR_PRECISION, prec.size(), prec.data()));
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64s(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_SM, dx_sm_array.size(), dx_sm_array.data()));
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_TYPE, cublasdxType::CUBLASDX_TYPE_REAL));
    std::array<long long int, 3> block_dim = { num_threads, 1, 1 };
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64s(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_BLOCK_DIM, block_dim.size(), block_dim.data()));
    std::array<long long int, 3> size = { tile_m, tile_n, tile_k };
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64s(h, cublasdxOperatorType::CUBLASDX_OPERATOR_SIZE, size.size(), size.data()));

    std::array<long long int, 3> arrangement = { CUBLASDX_ARRANGEMENT_COL_MAJOR,
                                                 CUBLASDX_ARRANGEMENT_COL_MAJOR,
                                                 CUBLASDX_ARRANGEMENT_COL_MAJOR };
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64s(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_ARRANGEMENT, arrangement.size(), arrangement.data()));

    std::array<long long int, 3> alignment = { 16, 16, 16 };
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64s(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_ALIGNMENT, alignment.size(), alignment.data()));

    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_ENABLE_INPUT_STREAMING, 1));

    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_WITH_PIPELINE, 1));

    LIBMATHDX_CHECK(cublasdxSetOptionStr(h, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, "matmul"));

    /**
     * Define the input and output tensors
     */

    int8_t* d_a {};
    int8_t* d_b {};
    int32_t* d_c {};
    CUDA_CHECK(cudaMallocManaged(&d_a, m * k * sizeof(int8_t)));
    CUDA_CHECK(cudaMallocManaged(&d_b, k * n * sizeof(int8_t)));
    CUDA_CHECK(cudaMallocManaged(&d_c, m * n * sizeof(int32_t)));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 100);
    for (int i = 0; i < m * k; i++) {
        d_a[i] = static_cast<int8_t>(dis(gen));
    }
    for (int i = 0; i < k * n; i++) {
        d_b[i] = static_cast<int8_t>(dis(gen));
    }
    for (int i = 0; i < m * n; i++) {
        d_c[i] = 0;
    }


    cublasdxTensor matrix_a { 0 };
    cublasdxTensor matrix_b { 0 };
    cublasdxTensor tile_gemm_c { 0 };
    std::vector<long long int> shape_a = { m, k };
    std::vector<long long int> shape_b = { k, n };
    // Importantly, the shape of the output tensor is the tile size, not the global size
    // the raw pointer must be offset for each CTA tile as shown in the kernel code
    std::vector<long long int> shape_c = { tile_m, tile_n };
    std::vector<long long int> strides_a = { 1, m };
    std::vector<long long int> strides_b = { 1, k };
    // while the shape of the tile gemm c is the tile size, the strides are the global size
    std::vector<long long int> strides_c = { 1, m };
    LIBMATHDX_CHECK(cublasdxCreateTensorStrided(CUBLASDX_MEMORY_SPACE_GMEM,
                                                COMMONDX_R_8I,
                                                nullptr,
                                                shape_a.size(),
                                                shape_a.data(),
                                                strides_a.data(),
                                                &matrix_a));
    LIBMATHDX_CHECK(cublasdxCreateTensorStrided(CUBLASDX_MEMORY_SPACE_GMEM,
                                                COMMONDX_R_8I,
                                                nullptr,
                                                shape_b.size(),
                                                shape_b.data(),
                                                strides_b.data(),
                                                &matrix_b));
    LIBMATHDX_CHECK(cublasdxCreateTensorStrided(CUBLASDX_MEMORY_SPACE_GMEM,
                                                COMMONDX_R_32I,
                                                nullptr,
                                                shape_c.size(),
                                                shape_c.data(),
                                                strides_c.data(),
                                                &tile_gemm_c));

    cublasdxTensor acc_c { 0 };

    LIBMATHDX_CHECK(cublasdxCreateTensor(h, CUBLASDX_TENSOR_SUGGESTED_ACCUMULATOR_C, &acc_c));

    cublasdxPipeline device_pipeline { 0 };
    cublasdxPipeline tile_pipeline { 0 };

    LIBMATHDX_CHECK(cublasdxCreateDevicePipeline(h,
                                                 CUBLASDX_DEVICE_PIPELINE_SUGGESTED,
                                                 LIBMATHDX_MAX_PIPELINE_DEPTH,
                                                 CUBLASDX_BLOCK_SIZE_STRATEGY_FIXED,
                                                 matrix_a,
                                                 matrix_b,
                                                 &device_pipeline));
    LIBMATHDX_CHECK(cublasdxCreateTilePipeline(h, CUBLASDX_TILE_PIPELINE_DEFAULT, device_pipeline, &tile_pipeline));

    std::array tensors = { matrix_a, matrix_b, tile_gemm_c, acc_c };
    std::array pipelines = { device_pipeline, tile_pipeline };
    LIBMATHDX_CHECK(cublasdxFinalize(tensors.size(), tensors.data(), pipelines.size(), pipelines.data()));


    // size and alignment of accumulator tensor
    long long int acc_size = 0;
    LIBMATHDX_CHECK(cublasdxGetTensorTraitInt64(acc_c, CUBLASDX_TENSOR_TRAIT_STORAGE_BYTES, &acc_size));
    long long int acc_alignment = 0;
    LIBMATHDX_CHECK(cublasdxGetTensorTraitInt64(acc_c, CUBLASDX_TENSOR_TRAIT_ALIGNMENT_BYTES, &acc_alignment));


    // size and alignment of the device pipeline
    long long int device_pipeline_storage_size = 0;
    LIBMATHDX_CHECK(cublasdxGetPipelineTraitInt64(
        device_pipeline, CUBLASDX_PIPELINE_TRAIT_STORAGE_BYTES, &device_pipeline_storage_size));
    long long int device_pipeline_storage_alignment = 0;
    LIBMATHDX_CHECK(cublasdxGetPipelineTraitInt64(
        device_pipeline, CUBLASDX_PIPELINE_TRAIT_STORAGE_ALIGNMENT_BYTES, &device_pipeline_storage_alignment));

    // size and alignment of the shared memory buffer (on sm_90a and above (accelerated arches with suffix 'a') should
    // be 128B aligned for TMA usage)
    long long int shared_memory_buffer_size = 0;
    LIBMATHDX_CHECK(cublasdxGetPipelineTraitInt64(
        device_pipeline, CUBLASDX_PIPELINE_TRAIT_BUFFER_SIZE, &shared_memory_buffer_size));
    long long int shared_memory_buffer_alignment = 0;
    LIBMATHDX_CHECK(cublasdxGetPipelineTraitInt64(
        device_pipeline, CUBLASDX_PIPELINE_TRAIT_BUFFER_ALIGNMENT_BYTES, &shared_memory_buffer_alignment));

    // block dimension for GEMM kernel launch (might be different than operator block dim)
    std::array<long long int, 3> device_pipeline_block_dim = { 0, 0, 0 };
    LIBMATHDX_CHECK(cublasdxGetPipelineTraitInt64s(
        device_pipeline, CUBLASDX_PIPELINE_TRAIT_BLOCK_DIM, 3, device_pipeline_block_dim.data()));

    // size and alignment of the tile pipeline
    long long int tile_pipeline_storage_size = 0;
    LIBMATHDX_CHECK(cublasdxGetPipelineTraitInt64(
        tile_pipeline, CUBLASDX_PIPELINE_TRAIT_STORAGE_BYTES, &tile_pipeline_storage_size));
    long long int tile_pipeline_storage_alignment = 0;
    LIBMATHDX_CHECK(cublasdxGetPipelineTraitInt64(
        tile_pipeline, CUBLASDX_PIPELINE_TRAIT_STORAGE_ALIGNMENT_BYTES, &tile_pipeline_storage_alignment));
    /**
     * Define a function operating on those input and output tensors.
     *
     * The device function output is an opaque and stateful accumulator.
     */
    cublasdxDeviceFunction init_acc { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunctionWithPipelines(
        h, CUBLASDX_DEVICE_FUNCTION_CREATE, 1, &acc_c, 1, &tile_pipeline, &init_acc));

    cublasdxDeviceFunction init_device_pipeline { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunctionWithPipelines(
        h, CUBLASDX_DEVICE_FUNCTION_CREATE, 0, nullptr, 1, &device_pipeline, &init_device_pipeline));

    cublasdxDeviceFunction init_tile_pipeline { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunctionWithPipelines(
        h, CUBLASDX_DEVICE_FUNCTION_CREATE, 0, nullptr, 1, &tile_pipeline, &init_tile_pipeline));

    cublasdxDeviceFunction destroy_acc { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunction(h, CUBLASDX_DEVICE_FUNCTION_DESTROY, 1, &acc_c, &destroy_acc));

    cublasdxDeviceFunction destroy_device_pipeline { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunctionWithPipelines(
        h, CUBLASDX_DEVICE_FUNCTION_DESTROY, 0, nullptr, 1, &device_pipeline, &destroy_device_pipeline));

    cublasdxDeviceFunction destroy_tile_pipeline { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunctionWithPipelines(
        h, CUBLASDX_DEVICE_FUNCTION_DESTROY, 0, nullptr, 1, &tile_pipeline, &destroy_tile_pipeline));

    cublasdxDeviceFunction execute_pipeline { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunctionWithPipelines(
        h, CUBLASDX_DEVICE_FUNCTION_EXECUTE, 1, &acc_c, 1, &tile_pipeline, &execute_pipeline));

    cublasdxDeviceFunction epilogue { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunctionWithPipelines(
        h, CUBLASDX_DEVICE_FUNCTION_EPILOGUE, 1, &acc_c, 1, &tile_pipeline, &epilogue));

    // set the callback function for the epilogue function
    LIBMATHDX_CHECK(
        cublasdxSetDeviceFunctionOptionStr(epilogue, CUBLASDX_DEVICE_FUNCTION_OPTION_CALLBACK, "epilogue_callback"));

    std::array copy_tensors = { acc_c, tile_gemm_c };
    cublasdxDeviceFunction copy_acc_big_gmem_c { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunction(
        h, CUBLASDX_DEVICE_FUNCTION_COPY, copy_tensors.size(), copy_tensors.data(), &copy_acc_big_gmem_c));

    std::vector<cublasdxDeviceFunction> functions = { execute_pipeline,
                                                      init_acc,
                                                      init_device_pipeline,
                                                      init_tile_pipeline,
                                                      destroy_acc,
                                                      destroy_device_pipeline,
                                                      destroy_tile_pipeline,
                                                      copy_acc_big_gmem_c,
                                                      epilogue };

    std::unordered_map<cublasdxDeviceFunction, std::string> function_symbols;

    for (auto f : functions) {
        size_t symbol_size { 0 };
        LIBMATHDX_CHECK(cublasdxGetDeviceFunctionTraitStrSize(f, CUBLASDX_DEVICE_FUNCTION_TRAIT_SYMBOL, &symbol_size));
        std::vector<char> symbol(symbol_size, '\0');
        LIBMATHDX_CHECK(
            cublasdxGetDeviceFunctionTraitStr(f, CUBLASDX_DEVICE_FUNCTION_TRAIT_SYMBOL, symbol.size(), symbol.data()));
        printf("Device function %lld: symbol: %s\n", static_cast<long long int>(f), symbol.data());
        function_symbols[f] = std::string(symbol.data());
    }

    /**
     * Compile the device functions to LTOIR
     */
    commondxCode code { 0 };
    LIBMATHDX_CHECK(commondxCreateCode(&code));
    LIBMATHDX_CHECK(
        commondxSetCodeOptionInt64s(code, COMMONDX_OPTION_TARGET_SM, target_sm_array.size(), target_sm_array.data()));
    LIBMATHDX_CHECK(cublasdxFinalizeDeviceFunctions(code, functions.size(), functions.data()));

    /**
     * Extract the LTOIR
     */
    size_t lto_size = 0;
    LIBMATHDX_CHECK(commondxGetCodeLTOIRSize(code, &lto_size));
    std::vector<char> lto(lto_size);
    LIBMATHDX_CHECK(commondxGetCodeLTOIR(code, lto_size, lto.data()));

    printf("Generated LTOIR for GEMM pipeline with epilogue callback, %zu bytes\n", lto.size());

    const char kernel_template[] = R"(
    struct my_user_data {
         void* ptr;
    };

    struct libmathdx_tensor_0s_0s { void* ptr; };
    struct libmathdx_pipeline { void* ptr; };
    
    #define M %d
    #define N %d
    #define K %d
    #define tile_m %d
    #define tile_n %d
    #define tile_k %d
    #define acc_name libmathdx_tensor_0s_0s
    #define ga_name libmathdx_tensor_0s_0s
    #define gb_name libmathdx_tensor_0s_0s
    #define gc_name libmathdx_tensor_0s_0s
    #define device_pipeline_name libmathdx_pipeline
    #define tile_pipeline_name libmathdx_pipeline
    
    constexpr unsigned acc_size = %lld;
    
    constexpr unsigned acc_alignment = %lld;
    
    constexpr unsigned block_size = %lld;
    constexpr unsigned smem_alignment = %lld;
    constexpr unsigned tile_pipeline_size = %lld;
    constexpr unsigned tile_pipeline_alignment = %lld;
    

    #define execute_pipeline_acc %s
    #define copy_acc_gc %s
    #define create_dev_pipe %s
    #define create_tile_pipe %s
    #define create_acc %s
    #define destroy_dev_pipe %s
    #define destroy_tile_pipe %s
    #define destroy_acc %s
    #define epilogue %s

    using C_VALUE_TYPE = signed int;
    
    extern "C" __device__ void execute_pipeline_acc(tile_pipeline_name, acc_name);
    extern "C" __device__ void copy_acc_gc(acc_name, gc_name);
    extern "C" __device__ void create_dev_pipe(device_pipeline_name, ga_name, gb_name);
    extern "C" __device__ void create_tile_pipe(device_pipeline_name, tile_pipeline_name, char*, int*, int*);
    extern "C" __device__ void create_acc(tile_pipeline_name, acc_name);
    extern "C" __device__ void destroy_dev_pipe(device_pipeline_name);
    extern "C" __device__ void destroy_tile_pipe(tile_pipeline_name);
    extern "C" __device__ void destroy_acc(acc_name);
    extern "C" __device__ void epilogue(tile_pipeline_name, acc_name, void* user_data);

    extern "C" __device__ void epilogue_callback(acc_name accumulator, my_user_data* user_data) {
        auto gc = gc_name { user_data->ptr };
        copy_acc_gc(accumulator, gc);
    }

    // Create the device pipeline
    extern "C" __global__ void create_device_pipeline(void* device_pipeline_ptr, void* ga_storage, void* gb_storage) {
        if(threadIdx.x == 0) {
            auto ga = ga_name { ga_storage };
            auto gb = gb_name { gb_storage };
            auto device_pipeline = device_pipeline_name { device_pipeline_ptr };
            create_dev_pipe(device_pipeline, ga, gb);
        }
    }

    // Perform the GEMM
    extern "C" __launch_bounds__(block_size, 1) __global__ void gemm(void* device_pipeline_ptr, void* gc_storage)
    {
    
        int row_id = blockIdx.x;
        int col_id = blockIdx.y;

        C_VALUE_TYPE* gmem_c = reinterpret_cast<C_VALUE_TYPE*>(gc_storage) + ((row_id * tile_m) + (col_id * tile_n) * M);

        // Allocate dynamic shared memory for tiles
        extern __shared__ __align__(smem_alignment) char smem[];

        // Allocate local memory 
        alignas(acc_alignment) char acc_storage[acc_size];
        alignas(tile_pipeline_alignment) char tile_pipeline_storage[tile_pipeline_size];

        // Create opaque types
        auto acc = acc_name { acc_storage };
        auto device_pipeline = device_pipeline_name { device_pipeline_ptr };
        auto tile_pipeline = tile_pipeline_name { tile_pipeline_storage };

        // Perform the GEMM
        create_tile_pipe(device_pipeline, tile_pipeline, smem, &row_id, &col_id);
        create_acc(tile_pipeline, acc);
        execute_pipeline_acc(tile_pipeline, acc);

        auto user_data = my_user_data { gmem_c };
        epilogue(tile_pipeline, acc, &user_data);

        // Destroy the accumulator and the tile pipeline
        destroy_acc(acc);
        destroy_tile_pipe(tile_pipeline);
    }

    // Destroy the device pipeline
    extern "C" __global__ void destroy_device_pipeline(void* device_pipeline_ptr) {
        if(threadIdx.x == 0) {
            auto device_pipeline = device_pipeline_name { device_pipeline_ptr };
            destroy_dev_pipe(device_pipeline);
        }
    }
    )";

    std::string cpp = strprintf(kernel_template,
                                m,
                                n,
                                k,
                                tile_m,
                                tile_n,
                                tile_k,
                                acc_size,
                                acc_alignment,
                                device_pipeline_block_dim[0],
                                shared_memory_buffer_alignment,
                                tile_pipeline_storage_size,
                                tile_pipeline_storage_alignment,
                                function_symbols[execute_pipeline].data(),
                                function_symbols[copy_acc_big_gmem_c].data(),
                                function_symbols[init_device_pipeline].data(),
                                function_symbols[init_tile_pipeline].data(),
                                function_symbols[init_acc].data(),
                                function_symbols[destroy_device_pipeline].data(),
                                function_symbols[destroy_tile_pipeline].data(),
                                function_symbols[destroy_acc].data(),
                                function_symbols[epilogue].data());

    std::vector<char> cubin = compile_and_link(cpp, lto, target_sm);

    CUmodule module {};
    CUfunction kernel, kernel2, kernel3 {};
    CUDA_CHECK(cudaSetDevice(0));
    CU_CHECK(cuModuleLoadDataEx(&module, cubin.data(), 0, 0, 0));
    CU_CHECK(cuModuleGetFunction(&kernel, module, "create_device_pipeline"));
    CU_CHECK(cuModuleGetFunction(&kernel2, module, "gemm"));
    CU_CHECK(cuModuleGetFunction(&kernel3, module, "destroy_device_pipeline"));

    // create the device pipeline using trait
    char* device_pipeline_ptr {};
    CUDA_CHECK(cudaMalloc(&device_pipeline_ptr, device_pipeline_storage_size));
    CUDA_CHECK(cudaMemset(device_pipeline_ptr, 0, device_pipeline_storage_size));


    {
        std::vector<void*> kernel_args = { reinterpret_cast<void*>(&device_pipeline_ptr),
                                           reinterpret_cast<void*>(&d_a),
                                           reinterpret_cast<void*>(&d_b) };
        CU_CHECK(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_args.data(), nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    {
        CU_CHECK(cuFuncSetAttribute(
            kernel2, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(shared_memory_buffer_size)));
        std::vector<void*> kernel_args = { reinterpret_cast<void*>(&device_pipeline_ptr),
                                           reinterpret_cast<void*>(&d_c) };
        CU_CHECK(cuLaunchKernel(kernel2,
                                static_cast<unsigned int>(m / tile_m),
                                static_cast<unsigned int>(n / tile_n),
                                1,
                                static_cast<unsigned int>(device_pipeline_block_dim[0]),
                                static_cast<unsigned int>(device_pipeline_block_dim[1]),
                                static_cast<unsigned int>(device_pipeline_block_dim[2]),
                                static_cast<unsigned int>(shared_memory_buffer_size),
                                nullptr,
                                kernel_args.data(),
                                nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    {
        std::vector<void*> kernel_args = { reinterpret_cast<void*>(&device_pipeline_ptr) };
        CU_CHECK(cuLaunchKernel(kernel3, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_args.data(), nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<int32_t> h_c_ref(m * n, 0);
    for (int l = 0; l < k; l++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                h_c_ref[i + m * j] += d_a[i + m * l] * d_b[l + k * j];
            }
        }
    }

    for (int i = 0; i < m * n; i++) {
        if (h_c_ref[i] != d_c[i]) {
            printf("Error at %d: h_c_ref[%d] = %d, d_c[%d] = %d\n", i, i, h_c_ref[i], i, d_c[i]);
            abort();
        }
    }
    printf("Successfully ran the kernel\n");

    /**
     * Destroy handles
     */
    CU_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cudaFree(device_pipeline_ptr));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    LIBMATHDX_CHECK(cublasdxDestroyTensor(matrix_a));
    LIBMATHDX_CHECK(cublasdxDestroyTensor(matrix_b));
    LIBMATHDX_CHECK(cublasdxDestroyTensor(acc_c));
    LIBMATHDX_CHECK(cublasdxDestroyTensor(tile_gemm_c));
    LIBMATHDX_CHECK(cublasdxDestroyPipeline(device_pipeline));
    LIBMATHDX_CHECK(cublasdxDestroyPipeline(tile_pipeline));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(init_acc));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(init_device_pipeline));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(init_tile_pipeline));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(destroy_acc));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(destroy_device_pipeline));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(destroy_tile_pipeline));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(execute_pipeline));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(epilogue));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(copy_acc_big_gmem_c));
    LIBMATHDX_CHECK(commondxDestroyCode(code));
    LIBMATHDX_CHECK(cublasdxDestroyDescriptor(h));
}
