cmake_minimum_required(VERSION 3.18)

# Define project
project(lto_helper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Possible variables that can be set externally
# cufft_ROOT: points to the cuFFT library root directory

find_package(cufft 11.5.0 EXACT REQUIRED CONFIG
    PATHS
        "${PROJECT_SOURCE_DIR}/../../../../cufft"
        "/opt/cufft"
)

# Define the helper executable
add_executable(cufftdx_cufft_lto_helper cufftdx_cufft_lto_helper.cpp)
target_link_libraries(cufftdx_cufft_lto_helper
    PRIVATE
        cufft::cufft_static
)
