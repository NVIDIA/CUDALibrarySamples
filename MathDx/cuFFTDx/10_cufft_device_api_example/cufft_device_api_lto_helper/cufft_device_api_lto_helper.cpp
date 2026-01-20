/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../../lto_helper/common_lto.hpp"

int main(int argc, char* argv[]) {

    if (!cufftdx::utils::check_cufft_device_api_version()) {
        return EXIT_FAILURE;
    }

    const int expected_args = 2;
    if (argc < expected_args) {
        std::cerr << "Error: Missing arguments.\n";
        std::cerr << "You provided " << argc - 1 << " argument(s), but "
                << expected_args - 1 << " are required.\n\n";
        return EXIT_FAILURE;
    }

    // Path where header + ltoir/fatbins should be placed
    std::string output_dir = argv[1];

    std::vector<int> cuda_archs = supported_cuda_architectures;
    if (argc > expected_args) {
        std::string arg = argv[2];
        cuda_archs.clear();
        if (arg.find("--CUDA_ARCHITECTURES=") == 0) {
            if (!parseCUDAArchitectures(arg, cuda_archs)) {
                return EXIT_FAILURE;
            }
        }
    }

    // Handles describing the properties of entries that should be included in the database
    std::vector<cufftDescriptionHandle> desc_handles;
    for (int cuda_arch : cuda_archs) {
        cufftdx_traits traits;
        traits.size = 128;
        traits.direction = cufftdx::fft_direction::forward;
        traits.set_execution_type(cufftdx::utils::execution_type::block);
        traits.sm = cuda_arch;
        cufftDescriptionHandle desc_handle = createDescriptionHandleWithTraits(traits);
        desc_handles.push_back(desc_handle);
    }

    cufftDeviceHandle device_handle;
    CUFFT_CHECK_AND_EXIT(cufftDeviceCreate(&device_handle, desc_handles.size(), desc_handles.data()));

    // For every description handle, check if it is valid / supported
    for (const auto& desc : desc_handles) {
        bool is_supported = false;
        CUFFT_CHECK_AND_EXIT(cufftDeviceIsSupported(device_handle, desc, &is_supported));
        if (!is_supported) {
            printDescriptionInfo(desc);
            std::cout << "Warning: Description handle (" << desc << ") is currently not supported.\n";
        }
    }

    size_t database_str_size = 0;
    CUFFT_CHECK_AND_EXIT(cufftDeviceGetDatabaseStrSize(device_handle, &database_str_size));

    if (database_str_size) {
        std::string database_header_name = "lto_database.hpp.inc";

        std::vector<char> database_str(database_str_size);
        CUFFT_CHECK_AND_EXIT(cufftDeviceGetDatabaseStr(device_handle, database_str.size(), database_str.data()));

        if (!writeFile((output_dir + "/" + database_header_name).data(), database_str.data(), database_str.size() - 1)) {
            return EXIT_FAILURE;
        }
    }
    else {
        std::cout << "No databases." << std::endl;
    }

    size_t count = 0;
    CUFFT_CHECK_AND_EXIT(cufftDeviceGetNumLTOIRs(device_handle, &count));

    std::vector<size_t> code_sizes(count);
    CUFFT_CHECK_AND_EXIT(cufftDeviceGetLTOIRSizes(device_handle, code_sizes.size(), code_sizes.data()));

    std::vector<std::vector<char>> code_ptrs_vec(count);
    std::vector<char*> code_ptrs(count);
    std::vector<cufftDeviceCodeContainer> code_containers(count);
    for (size_t n = 0; n < count; ++n) {
        code_ptrs_vec[n].resize(code_sizes[n]);
        code_ptrs[n] = code_ptrs_vec[n].data();
    }

    CUFFT_CHECK_AND_EXIT(cufftDeviceGetLTOIRs(device_handle, count, code_ptrs.data(), code_containers.data()));

    for (size_t n = 0; n < count; ++n) {
        std::string database_artifact_name = "cufft_generated_" + std::to_string(n);
        std::string ext = parseFileExtension(code_containers[n]);

        if (!writeFile((output_dir + "/" + database_artifact_name + "." + ext).data(), code_ptrs[n], code_sizes[n])) {
            return EXIT_FAILURE;
        }
    }

    CUFFT_CHECK_AND_EXIT(cufftDeviceDestroy(device_handle));
}
