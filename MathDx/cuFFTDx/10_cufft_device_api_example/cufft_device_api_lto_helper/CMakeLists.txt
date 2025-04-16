cmake_minimum_required(VERSION 3.18)

# Define project
project(cufft_device_api_lto_helper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Possible variables that can be set externally
# cufft_ROOT: points to the cuFFT library root directory

find_package(cufft 11.5.0 EXACT REQUIRED CONFIG
    PATHS
        "${PROJECT_SOURCE_DIR}/../../../../../cufft"
        "/opt/cufft"
)

# Define the helper executable
add_executable(cufft_device_api_lto_helper cufft_device_api_lto_helper.cpp)
target_link_libraries(cufft_device_api_lto_helper
    PUBLIC
        cufft::cufft_static
)
