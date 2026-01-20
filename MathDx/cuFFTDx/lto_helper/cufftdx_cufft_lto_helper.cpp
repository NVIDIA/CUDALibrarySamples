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

#include "common_lto.hpp"

std::vector<std::string> parseCSVHeader(const std::string &header_line, const std::vector<std::string> &expected_fields) {
    std::vector<std::string> header_map;
    std::stringstream ss(header_line);
    std::string column;
    int column_index = 0;

    // Parse the header line and populate header_map with valid fields
    while (std::getline(ss, column, ',')) {
        column = std::regex_replace(column, std::regex("^\\s+|\\s+$"), ""); // Trim leading/trailing spaces
        if (std::find(expected_fields.begin(), expected_fields.end(), column) != expected_fields.end()) {
            header_map.push_back(column);
        }
        column_index++;
    }
    return header_map;
}

std::optional<cufftdx_traits> parseLineToTraits(const std::string& line, int line_number, const std::vector<std::string>& header_map) {
    std::stringstream ss(line);
    std::string token;

    int column_index = 0;
    cufftdx_traits traits;
    while (std::getline(ss, token, ',')) {
        token = std::regex_replace(token, std::regex("^\\s+|\\s+$"), ""); // Trim spaces
        if (column_index >= header_map.size()) {
            std::cerr << "Error: Too many columns on line " << line_number << "\n";
            return std::nullopt;
        }
        const std::string &field = header_map.at(column_index);

        // Allow empty tokens
        if (token.empty()) {
            column_index++;
            continue; // Skip further processing for this line
        }
        if (field == "execution_type") {
            if (token == "execution_type::block") {
                traits.set_execution_type(cufftdx::detail::execution_type::block);
            } else if (token == "execution_type::thread") {
                traits.set_execution_type(cufftdx::detail::execution_type::thread);
            } else {
                std::cerr << "Error: Invalid value for 'execution_type' on line " << line_number << ": " << token << "\n Valid values are: execution_type::block, execution_type::thread.\n";
                return std::nullopt;
            }
        } else if (field == "size") {
            try {
                traits.size = std::stoul(token);
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: Invalid value for 'size' on line " << line_number << ": " << token << std::endl;
                return std::nullopt;
            }
        } else if (field == "type") {
            if (token == "fft_type::c2c") {
                traits.set_type(cufftdx::fft_type::c2c);
            } else if (token == "fft_type::r2c") {
                traits.set_type(cufftdx::fft_type::r2c);
            } else if (token == "fft_type::c2r") {
                traits.set_type(cufftdx::fft_type::c2r);
            } else {
                std::cerr << "Error: Invalid value for 'type' on line " << line_number << ": " << token << "\n Valid values are: fft_type::c2c, fft_type::r2c, fft_type::c2r\n";
                return std::nullopt;
            }
        } else if (field == "direction") {
            if (token == "fft_direction::forward") {
                traits.direction = cufftdx::fft_direction::forward;
            } else if (token == "fft_direction::inverse") {
                traits.direction = cufftdx::fft_direction::inverse;
            } else {
                std::cerr << "Error: Invalid value for 'direction' on line " << line_number << ": " << token << "\n Valid values are: fft_direction::forward, fft_direction::inverse\n";
                return std::nullopt;
            }
        } else if (field == "precision") {
            if (token == "__half") {
                traits.precision = cufftdx::precision::f16;
            } else if (token == "float") {
                traits.precision = cufftdx::precision::f32;
            } else if (token == "double") {
                traits.precision = cufftdx::precision::f64;
            } else {
                std::cerr << "Error: Invalid value for 'precision' on line " << line_number << ": " << token << "\n Valid values are: float, double\n";
                return std::nullopt;
            }
        } else if (field == "real_mode") {
            if (token == "real_mode::normal") {
                traits.real_mode = cufftdx::real_mode::normal;
            } else if (token == "real_mode::folded") {
                traits.real_mode = cufftdx::real_mode::folded;
            } else {
                std::cerr << "Error: Invalid value for 'real_mode' on line " << line_number << ": " << token << "\n Valid values are: real_mode::normal, real_mode::folded\n";
                return std::nullopt;
            }
        } else if (field == "elements_per_thread") {
            try {
                traits.elements_per_thread = std::stoul(token);
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: Invalid value for 'elements_per_thread' on line " << line_number << ": " << token << std::endl;
                return std::nullopt;
            }
        } else {
            std::cerr << "Error: Unexpected field '" << field << "' on line " << line_number << "\n";
            return std::nullopt;
        }
        column_index++;
    }
    return traits;
}

void showUsage(const std::string& program_name) {
    std::cout << "Usage: " << program_name << " <output_dir> <csv_file> [--CUDA_ARCHITECTURES=XX;YY;...]\n";
    std::cout << "    <output_dir>       Directory where the output files will be written.\n";
    std::cout << "    <csv_file>         Path to the input CSV file.\n";
    std::cout << "    --CUDA_ARCHITECTURES (Optional): List of CUDA architectures separted by semicolon (e.g., 50,70).\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " ./output ./data/input.csv --CUDA_ARCHITECTURES=80,90\n";
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {

    if (!cufftdx::utils::check_cufft_device_api_version()) {
        return EXIT_FAILURE;
    }

    const int expected_args = 3;
    if (argc < expected_args) {
        std::cerr << "Error: Missing arguments.\n";
        std::cerr << "You provided " << argc - 1 << " argument(s), but "
                << expected_args - 1 << " are required.\n\n";
        showUsage(argv[0]);
        return EXIT_FAILURE;
    }
    std::string output_dir = argv[1]; // First argument: Output directory
    std::string csv_file   = argv[2]; // Second argument: Path to the CSV file

    // Optional argument: CUDA architectures
    std::vector<int> cuda_architectures = supported_cuda_architectures;
    if (argc > 3) {
        std::string arg = argv[3];
        cuda_architectures.clear();
        if (arg.find("--CUDA_ARCHITECTURES=") == 0) {
            if (!parseCUDAArchitectures(arg, cuda_architectures)) {
                return EXIT_FAILURE; // Exit if the architectures argument is invalid
            }
        }
    }

    std::vector<cufftDescriptionHandle> desc_handles;

    std::ifstream file(csv_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << csv_file << std::endl;
        return EXIT_FAILURE;
    }

    std::string line;
    int line_number = 1; // Line number tracker

    std::string header;
    if (!std::getline(file, header)) {
        std::cerr << "Error: Empty or invalid CSV file." << std::endl;
        return EXIT_FAILURE; // Exit if the CSV file is empty
    }

    const std::vector<std::string> fields = {
        "execution_type",
        "size",
        "type",
        "direction",
        "precision",
        "real_mode",
        "elements_per_thread"
    };

    // Dynamically parse the header
    auto header_map = parseCSVHeader(header, fields);
    if (header_map.empty()) {
        std::cerr << "Invalid CSV header. Aborting.\n";
        return EXIT_FAILURE;
    }

    // Start reading lines from the CSV file
    while (std::getline(file, line)) {
        line_number++;
        auto frontend_traits = parseLineToTraits(line, line_number, header_map);
        if (frontend_traits.has_value()) {
            for (int arch : cuda_architectures) {
                frontend_traits.value().sm = arch;
                cufftDescriptionHandle desc = createDescriptionHandleWithTraits(frontend_traits.value());
                desc_handles.push_back(desc);
            }
        } else {
            std::cout << "Skipping invalid line " << line_number << "\n";
            continue;
        }
    }
    file.close(); // Close the CSV file

    // Create CUFFT device handle
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
    if (database_str_size > 0) {
        std::vector<char> database_str(database_str_size);
        CUFFT_CHECK_AND_EXIT(cufftDeviceGetDatabaseStr(device_handle, database_str.size(), database_str.data()));
        std::string header_name = std::string("lto_database.hpp.inc");

        if (!writeFile((output_dir + "/" + header_name).data(), database_str.data(), database_str.size() - 1)) {
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "No databases." << std::endl;
    }

    size_t count = 0;
    CUFFT_CHECK_AND_EXIT(cufftDeviceGetNumLTOIRs(device_handle, &count));

    if (count > 0) {
        std::vector<size_t> code_sizes(count);
        CUFFT_CHECK_AND_EXIT(cufftDeviceGetLTOIRSizes(device_handle, code_sizes.size(), code_sizes.data()));

        // Get all the chunks
        std::vector<std::vector<char>> codes;
        std::vector<char*> code_ptrs;
        for(auto size: code_sizes) {
            codes.push_back(std::vector<char>(size));
        }
        for(auto& code: codes) {
            code_ptrs.push_back(code.data());
        }
        std::vector<cufftDeviceCodeContainer> code_containers(count);
        CUFFT_CHECK_AND_EXIT(cufftDeviceGetLTOIRs(device_handle, code_ptrs.size(), code_ptrs.data(), code_containers.data()));
        for(unsigned i = 0; i < count; i++) {
            std::string name = "cufft_generated_" + std::to_string(i);
            std::string ext = parseFileExtension(code_containers[i]);

            if (!writeFile((output_dir + "/" + name + "." + ext).data(), code_ptrs[i], code_sizes[i])) {
                return EXIT_FAILURE;
            }
        }
    } else {
        std::cout << "No LTOIRs have been found. A valid configuration for the traits provided may not be available for the selected architecture."<< std::endl;
        return 0;
    }

    CUFFT_CHECK_AND_EXIT(cufftDeviceDestroy(device_handle));
    return EXIT_SUCCESS;
}
