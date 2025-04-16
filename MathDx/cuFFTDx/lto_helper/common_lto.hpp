#ifndef CUFFTDX_EXAMPLE_COMMON_LTO_HPP
#define CUFFTDX_EXAMPLE_COMMON_LTO_HPP

#include <cufft_device.h>

#include <string>
#include <vector>
#include <regex>
#include <iostream>
#include <fstream>

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

cufftResult checkDescriptions(cufftDeviceHandle device_handle, size_t count, const cufftDescriptionHandle* descs) {
    // For every description handle, check if it is valid / supported
    for (size_t n = 0; n < count; ++n) {
        const cufftDescriptionHandle& desc = descs[n];

        cufftResult status = cufftDeviceCheckDescription(device_handle, desc);
        if (status != CUFFT_SUCCESS) {
            if (status == CUFFT_NOT_SUPPORTED) {
                std::cout << "Warning: Description handle (" << desc << ") is currently not supported.\n";
            } else if (status == CUFFT_INVALID_VALUE) {
                std::cerr << "Error: Description handle (" << desc << ") is not valid / complete.\n";
                return status;
            } else {
                std::cerr << "Error: cufftDeviceCheckDescription failed with status (" << status << ")\n";
                return status;
            }
        }
    }

    return CUFFT_SUCCESS;
}

std::string parseFileExtension(cufftDeviceCodeType code_type) {
    switch (code_type) {
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
