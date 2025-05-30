cmake_minimum_required(VERSION 3.26.0)

project(09_introduction_lto_example_introduction_lto_example LANGUAGES CXX CUDA)

if(NOT DEFINED CUFFTDX_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "CUFFTDX_CUDA_ARCHITECTURES is not defined")
endif()

# Enable testing only for selected architectures
add_compile_definitions(CUFFTDX_EXAMPLE_CMAKE)
foreach(CUDA_ARCH ${CUFFTDX_CUDA_ARCHITECTURES})
    # Extract SM from SM-real/SM-virtual
    string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
    list(GET CUDA_ARCH_LIST 0 ARCH)
    add_compile_definitions(CUFFTDX_EXAMPLE_ENABLE_SM_${ARCH})
endforeach()

include("${CMAKE_CURRENT_SOURCE_DIR}/../lto_helper/lto_helper.cmake")

# Generate and link LTO blobs
run_cufft_lto_helper(
    ${CMAKE_CURRENT_SOURCE_DIR}/../lto_helper
    ${CMAKE_CURRENT_BINARY_DIR}/../lto_helper
    09_introduction_lto_example_introduction_lto_example
    ${CMAKE_CURRENT_SOURCE_DIR}/introduction_lto_cases.csv
    "${CUFFTDX_CUDA_ARCHITECTURES}"
    TRUE # Force regenerate blobs
)

if(NOT CUFFT_LTO_HELPER_RESULT)
    message(FATAL_ERROR "run_cufft_lto_helper encountered errors for target 09_introduction_lto_example_introduction_lto_example.")
endif()

if(NOT TARGET cufftdx)
    find_package(cufftdx
        REQUIRED
        CONFIG
        PATHS
            "${PROJECT_SOURCE_DIR}/../../../../"
    )
endif()

add_executable(09_introduction_lto_example_introduction_lto_example
    introduction_lto_example.cu
)

target_link_libraries(09_introduction_lto_example_introduction_lto_example
    PRIVATE
        cufftdx::cufftdx
)

target_link_options(09_introduction_lto_example_introduction_lto_example
    PUBLIC
        $<DEVICE_LINK:$<TARGET_OBJECTS:09_introduction_lto_example_introduction_lto_example_lto_lib>>
)

target_include_directories(09_introduction_lto_example_introduction_lto_example
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/../lto_helper/09_introduction_lto_example_introduction_lto_example_artifacts
)

set_target_properties(09_introduction_lto_example_introduction_lto_example
    PROPERTIES
        CUDA_ARCHITECTURES "${CUFFTDX_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        INTERPROCEDURAL_OPTIMIZATION ON
)
