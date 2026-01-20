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

#ifndef CUFFTDX_EXAMPLE_COMMON_LTO_HPP
#define CUFFTDX_EXAMPLE_COMMON_LTO_HPP

#include <cufft_device.h>
#include <cufftdx/utils.hpp>

#include <string>
#include <vector>
#include <regex>
#include <iostream>
#include <fstream>
#include <optional>


#ifndef CUFFT_CHECK_AND_EXIT
#    define CUFFT_CHECK_AND_EXIT(error)                                                 \
        {                                                                               \
            auto status = static_cast<cufftResult>(error);                              \
            if (status != CUFFT_SUCCESS) {                                              \
                std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                      \
            }                                                                           \
        }
#endif // CUFFT_CHECK_AND_EXIT

const std::vector<int> supported_cuda_architectures = {750,800,860,870,890,900,1000,1030,1100,1200,1210};

struct cufftdx_traits {
    cufftdx::utils::algorithm                     algorithm;
    std::optional<cufftdx::utils::execution_type> execution_type;
    std::optional<unsigned int>                   size;
    std::optional<cufftdx::fft_direction>         direction;
    std::optional<int>                            sm;
    cufftdx::fft_type                             type;
    cufftdx::precision                            precision;
    cufftdx::real_mode                            real_mode;
    unsigned int                                  elements_per_thread;
    cufftdx::experimental::code_type              code_type;
    // Set to default values from https://docs.nvidia.com/cuda/cufftdx/api/traits.html#description-traits
    cufftdx_traits():
        algorithm(cufftdx::utils::algorithm::ct),
        execution_type(std::nullopt),
        size(std::nullopt),
        direction(std::nullopt),
        sm(std::nullopt),
        type(cufftdx::fft_type::c2c),
        precision(cufftdx::precision::f32),
        real_mode(cufftdx::real_mode::normal),
        elements_per_thread(0),
        code_type(cufftdx::experimental::code_type::ltoir) {}

    void set_type(cufftdx::fft_type type) {
        this->type = type;
        if (type == cufftdx::fft_type::r2c) {
            direction = cufftdx::fft_direction::forward;
        } else if (type == cufftdx::fft_type::c2r) {
            direction = cufftdx::fft_direction::inverse;
        }
    }
    void set_execution_type(cufftdx::utils::execution_type execution_type) {
        this->execution_type = execution_type;
        if (execution_type == cufftdx::utils::execution_type::thread) {
            sm = 0; // set to dummy value
        }
    }
    bool is_complete() const {
        return execution_type.has_value() && size.has_value() && direction.has_value() && sm.has_value();
    }
};

// Function to parse the `--CUDA_ARCHITECTURES` argument and extract valid architectures (e.g., 50 -> 500)
bool parseCUDAArchitectures(const std::string& architectures_arg, std::vector<int>& architectures) {
    // Check if the argument has the correct format: "--CUDA_ARCHITECTURES=50;100"
    if (architectures_arg.find("--CUDA_ARCHITECTURES=") != 0) {
        std::cerr << "Error: Invalid argument. Expected format: --CUDA_ARCHITECTURES=XX;YY;...\n";
        return false;
    }

    // Extract the part after the equals sign
    std::string architectures_list = architectures_arg.substr(std::string("--CUDA_ARCHITECTURES=").length());

    // Split the string by semicolon and search for the number
    std::regex pattern(R"(^(\d+))"); // Match numeric prefix at the start of the string
    std::stringstream ss(architectures_list);
    std::string arch;

    while (std::getline(ss, arch, ';')) {
        std::smatch match;
        if (std::regex_search(arch, match, pattern)) {
            int value = std::stoi(match[1]) * 10; // Multiply by 10 for the representation
            architectures.push_back(value);
        } else {
            std::cout << "Warning: Skipping invalid architecture: '" << arch << "'\n";
        }
    }
    return true;
}

cufftDescriptionHandle createDescriptionHandleWithTraits(const cufftdx_traits& traits) {
    if (!traits.is_complete()) {
        std::cerr << "Error: Not all traits are set.\n";
        std::exit(1);
    }
    // Map frontend traits to backend traits
    const auto backend_traits = cufftdx::utils::frontend_to_backend(traits.algorithm,
                                                                    traits.execution_type.value(),
                                                                    traits.size.value(),
                                                                    traits.type,
                                                                    traits.direction.value(),
                                                                    traits.sm.value(),
                                                                    traits.real_mode,
                                                                    traits.elements_per_thread,
                                                                    0, /* block_dim_x */
                                                                    traits.code_type);

    // Create cufft description handle
    cufftDescriptionHandle desc_handle;
    CUFFT_CHECK_AND_EXIT(cufftDescriptionCreate(&desc_handle));

    // size
    CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_SIZE, backend_traits.size));
    // sm
    CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_SM, backend_traits.sm));
    // elements_per_thread
    CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_ELEMENTS_PER_THREAD, backend_traits.elements_per_thread));

    // type
    long long type_value = CUFFT_DESC_C2C;
    switch (backend_traits.type) {
        case cufftdx::fft_type::c2c: type_value = CUFFT_DESC_C2C; break;
        case cufftdx::fft_type::r2c: type_value = CUFFT_DESC_R2C; break;
        case cufftdx::fft_type::c2r: type_value = CUFFT_DESC_C2R; break;
    }
    CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_TYPE, type_value));

    // direction
    long long direction_value = CUFFT_DESC_FORWARD;
    switch (backend_traits.direction) {
        case cufftdx::fft_direction::forward: direction_value = CUFFT_DESC_FORWARD; break;
        case cufftdx::fft_direction::inverse: direction_value = CUFFT_DESC_INVERSE; break;
    }
    CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_DIRECTION, direction_value));

    // precision
    long long precision_value = CUFFT_DESC_SINGLE;
    switch (traits.precision) {
        case cufftdx::precision::f16: precision_value = CUFFT_DESC_HALF; break;
        case cufftdx::precision::f32: precision_value = CUFFT_DESC_SINGLE; break;
        case cufftdx::precision::f64: precision_value = CUFFT_DESC_DOUBLE; break;
    }
    CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_PRECISION, precision_value));
    return desc_handle;
}

void printDescriptionInfo(cufftDescriptionHandle desc) {
    long long int value;
    std::cout << "Description Handle Information:" << std::endl;
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_SIZE, &value));
    std::cout << "  CUFFT_DESC_TRAIT_SIZE: " << value << std::endl;
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_DIRECTION, &value));
    std::cout << "  CUFFT_DESC_TRAIT_DIRECTION: " << value << std::endl;
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_PRECISION, &value));
    std::cout << "  CUFFT_DESC_TRAIT_PRECISION: " << value << std::endl;
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_TYPE, &value));
    std::cout << "  CUFFT_DESC_TRAIT_TYPE: " << value << std::endl;
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_SM, &value));
    std::cout << "  CUFFT_DESC_TRAIT_SM: " << value << std::endl;
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_ELEMENTS_PER_THREAD, &value));
    std::cout << "  CUFFT_DESC_TRAIT_ELEMENTS_PER_THREAD: " << value << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
}

std::string parseFileExtension(cufftDeviceCodeContainer code_container) {
    switch (code_container) {
        case CUFFT_DEVICE_LTOIR : return "ltoir";
        case CUFFT_DEVICE_FATBIN : return "fatbin";
        default : return "";
    }
}

bool writeFile(const char* path, const char* data, const size_t size) {
    std::ofstream file(path);
    if (file.is_open() && data && size) {
        file.write(data, size);
    }
    else {
        std::cout << "Error: failed to open" << path << "\n";
        return false;
    }
    file.close();
    return true;
}

#endif
