/* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "common_lto.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <regex>
#include <unordered_map>

#include <cuda.h>
#include <cufft_device.h>

#ifndef CUFFT_CHECK_AND_EXIT
#    define CUFFT_CHECK_AND_EXIT(error)                                                 \
        {                                                                               \
            auto status = static_cast<cufftResult>(error);                              \
            if (status != CUFFT_SUCCESS) {                                              \
                std::cerr << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                      \
            }                                                                           \
        }
#endif // CUFFT_CHECK_AND_EXIT



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

std::pair<std::unordered_map<std::string, long long int>, bool> parseLineToTraits(const std::string& line, int line_number, const std::vector<std::string>& header_map) {
    std::unordered_map<std::string, long long int> line_map;
    std::stringstream ss(line);
    std::string token;

    // Initially mark the line as valid
    bool valid = true;

    int column_index = 0;
    while (std::getline(ss, token, ',')) {
        token = std::regex_replace(token, std::regex("^\\s+|\\s+$"), ""); // Trim spaces
        if (column_index >= header_map.size()) {
            std::cerr << "Error: Too many columns on line " << line_number << "\n";
            valid = false;
            return { line_map, valid };
        }
        const std::string &field = header_map.at(column_index);

        // Allow empty tokens
        if (token.empty()) {
            column_index++;
            continue; // Skip further processing for this line
        }

        if (field == "size" || field == "elements_per_thread") {
            try {
                line_map[field] = std::stoll(token);
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: Invalid value for " << field << " on line " << line_number << ": " << token << std::endl;
                valid = false;
            }
        } else if (field == "direction") {
            long long value = (token == "fft_direction::forward") ? CUFFT_DESC_FORWARD : (token == "fft_direction::inverse") ? CUFFT_DESC_INVERSE : -1;
            if (value != -1) {
                line_map[field] = value;
            } else {
                std::cerr << "Error: Invalid value for 'direction' on line " << line_number << ": " << token << "\n Valid values are: fft_direction::forward, fft_direction::inverse\n";
                valid = false;
            }
        } else if (field == "precision") {
            long long value = (token == "float") ? CUFFT_DESC_SINGLE : (token == "double") ? CUFFT_DESC_DOUBLE : -1;
            if (value != -1) {
                line_map[field] = value;
            } else {
                std::cerr << "Error: Invalid value for 'precision' on line " << line_number << ": " << token << "\n Valid values are: float, double\n";
                valid = false;
            }
        } else if (field == "type") {
            long long value = (token == "fft_type::c2c") ? CUFFT_DESC_C2C : (token == "fft_type::r2c") ? CUFFT_DESC_R2C : (token == "fft_type::c2r") ? CUFFT_DESC_C2R : -1;
            if (value != -1) {
                line_map[field] = value;
            } else {
                std::cerr << "Error: Invalid value for 'type' on line " << line_number << ": " << token << "\n Valid values are: fft_type::c2c, fft_type::r2c, fft_type::c2r\n";
                valid = false;
            }
        } else if (field == "real_mode") {
            long long value = (token == "real_mode::normal") ? CUFFT_DESC_NORMAL  : (token == "real_mode::folded") ? CUFFT_DESC_FOLDED : -1;
            if (value != -1) {
                line_map[field] = value;
            } else {
                std::cerr << "Error: Invalid value for 'real_mode' on line " << line_number << ": " << token << "\n Valid values are: real_mode::normal, real_mode::folded\n";
                valid = false;
            }
        } else if (field == "exec_op") {
            long long value = (token == "Block") ? CUFFT_DESC_BLOCK : (token == "Thread") ? CUFFT_DESC_THREAD : -1;
            if (value != -1) {
                line_map[field] = value;
            } else {
                std::cerr << "Error: Invalid value for 'exec_op' on line " << line_number << ": " << token << "\n Valid values are: Block, Thread.\n";
                valid = false;
            }
        } else {
            std::cerr << "Error: Unexpected field '" << field << "' on line " << line_number << "\n";
            valid = false;
            return { line_map, valid };
        }

        column_index++;
    }


    // Return traits object with valid set to true
    return { line_map, valid };
}

// Function to create a cufftDescriptionHandle with the given traits (already parsed)
cufftDescriptionHandle createDescriptionHandleWithTraits(const std::unordered_map<std::string, long long int>& traits) {
    cufftDescriptionHandle desc_handle;
    CUFFT_CHECK_AND_EXIT(cufftDescriptionCreate(&desc_handle));

    // Dynamically check for each trait and set it if present in the map
    if (traits.find("size") != traits.end()) {
        CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_SIZE, traits.at("size")));
    }
    if (traits.find("direction") != traits.end()) {
        CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_DIRECTION, traits.at("direction")));
    }
    if (traits.find("precision") != traits.end()) {
        CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_PRECISION, traits.at("precision")));
    }
    if (traits.find("type") != traits.end()) {
        CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_TYPE, traits.at("type")));
    }
    if (traits.find("sm") != traits.end()) {
        CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_SM, traits.at("sm")));
    }
    if (traits.find("real_mode") != traits.end()) {
        CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_REAL_MODE, traits.at("real_mode")));
    }
    if (traits.find("exec_op") != traits.end()) {
        CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_EXEC_OP, traits.at("exec_op")));
    }
    if (traits.find("elements_per_thread") != traits.end()) {
        CUFFT_CHECK_AND_EXIT(cufftDescriptionSetTraitInt64(desc_handle, CUFFT_DESC_TRAIT_ELEMENTS_PER_THREAD, traits.at("elements_per_thread")));
    }

    return desc_handle;
}

void printDescriptionInfo(cufftDescriptionHandle desc) {
    long long int value;

    std::cout << "Description Handle Information:" << std::endl;

    // Query CUFFT_DESC_TRAIT_SIZE
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_SIZE, &value));
    std::cout << "  CUFFT_DESC_TRAIT_SIZE: " << value << std::endl;

    // Query CUFFT_DESC_TRAIT_DIRECTION
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_DIRECTION, &value));
    std::cout << "  CUFFT_DESC_TRAIT_DIRECTION: " << value << std::endl;

    // Query CUFFT_DESC_TRAIT_PRECISION
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_PRECISION, &value));
    std::cout << "  CUFFT_DESC_TRAIT_PRECISION: " << value << std::endl;

    // Query CUFFT_DESC_TRAIT_TYPE
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_TYPE, &value));
    std::cout << "  CUFFT_DESC_TRAIT_TYPE: " << value << std::endl;

    // Query CUFFT_DESC_TRAIT_SM
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_SM, &value));
    std::cout << "  CUFFT_DESC_TRAIT_SM: " << value << std::endl;

    // Query CUFFT_DESC_TRAIT_REAL_MODE
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_REAL_MODE, &value));
    std::cout << "  CUFFT_DESC_TRAIT_REAL_MODE: " << value << std::endl;

    // Query CUFFT_DESC_TRAIT_EXEC_OP
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_EXEC_OP, &value));
    std::cout << "  CUFFT_DESC_TRAIT_EXEC_OP: " << value << std::endl;

    // Query CUFFT_DESC_TRAIT_ELEMENTS_PER_THREAD
    CUFFT_CHECK_AND_EXIT(cufftDescriptionGetTraitInt64(desc, CUFFT_DESC_TRAIT_ELEMENTS_PER_THREAD, &value));
    std::cout << "  CUFFT_DESC_TRAIT_ELEMENTS_PER_THREAD: " << value << std::endl;

    std::cout << "------------------------------------------------------" << std::endl;
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

    const int expected_args = 3;
    if (argc < expected_args) {
        std::cerr << "Error: Missing arguments.\n";
        std::cerr << "You provided " << argc - 1 << " argument(s), but "
                << expected_args - 1 << " are required.\n\n";
        showUsage(argv[0]);
        return EXIT_FAILURE;
    }
    std::string output_dir = argv[1]; // First argument: Output directory
    std::string csv_file = argv[2];  // Second argument: Path to the CSV file

    // Optional argument: CUDA architectures
    std::vector<int> cuda_architectures = {700,720,750,800,860,870,890,900};
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
        "size",
        "direction",
        "precision",
        "type",
        "real_mode",
        "exec_op",
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

        auto [traits, valid] = parseLineToTraits(line, line_number, header_map);
        if (!valid) {
            std::cout << "Skipping invalid line " << line_number << "\n";
            continue;
        }

        if (cuda_architectures.empty()) {
            cufftDescriptionHandle desc = createDescriptionHandleWithTraits(traits);
            desc_handles.push_back(desc);
        } else {
            for (int arch : cuda_architectures) {
                traits["sm"] = static_cast<long long int>(arch);
                cufftDescriptionHandle desc = createDescriptionHandleWithTraits(traits);
                desc_handles.push_back(desc);
            }
        }
    }

    file.close(); // Close the CSV file

    // Create CUFFT device handle
    cufftDeviceHandle device_handle;
    CUFFT_CHECK_AND_EXIT(cufftDeviceCreate(&device_handle, desc_handles.size(), desc_handles.data()));

    // For every description handle, check if it is valid / supported
    for (const auto& desc : desc_handles) {
        auto status = cufftDeviceCheckDescription(device_handle, desc);
        if (status != CUFFT_SUCCESS) {
            printDescriptionInfo(desc);
            if (status == CUFFT_NOT_SUPPORTED) {
                std::cout << "Warning: Description handle (" << desc << ") is currently not supported.\n";
            } else if (status == CUFFT_INVALID_VALUE) {
                std::cerr << "Error: Description handle (" << desc << ") is not valid / complete.\n";
                return EXIT_FAILURE;
            } else {
                std::cerr << "Error: cufftDeviceCheckDescription failed with status (" << status << ")\n";
                return EXIT_FAILURE;
            }
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
            codes.emplace_back(std::vector<char>(size));
            code_ptrs.emplace_back(codes.back().data());
        }
        std::vector<cufftDeviceCodeType> code_types(count);
        CUFFT_CHECK_AND_EXIT(cufftDeviceGetLTOIRs(device_handle, code_ptrs.size(), code_ptrs.data(), code_types.data()));
        for(unsigned i = 0; i < count; i++) {
            std::string name = "cufft_generated_" + std::to_string(i);
            std::string ext = parseFileExtension(code_types[i]);

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
