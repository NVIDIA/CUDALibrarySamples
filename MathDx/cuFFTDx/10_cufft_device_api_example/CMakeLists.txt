cmake_minimum_required(VERSION 3.26.0)

project(00_cufft_device_api_example LANGUAGES CXX CUDA)

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

include("${CMAKE_CURRENT_SOURCE_DIR}/cufft_device_api_lto_helper/cufft_device_api_lto_helper.cmake")

run_cufft_device_api_lto_helper(
    ${CMAKE_CURRENT_SOURCE_DIR}/cufft_device_api_lto_helper
    ${CMAKE_CURRENT_BINARY_DIR}/cufft_device_api_lto_helper
    00_cufft_device_api_example
    "${CUFFTDX_CUDA_ARCHITECTURES}"
)

if(NOT CUFFT_DEVICE_API_LTO_HELPER_RESULT)
    message(FATAL_ERROR "run_cufft_device_api_lto_helper encountered errors for target ${LTO_EXAMPLE}.")
endif()

if(NOT TARGET cufftdx)
    find_package(cufftdx
        REQUIRED
        CONFIG
        PATHS
            "${PROJECT_SOURCE_DIR}/../../../../"
    )
endif()

add_executable(00_cufft_device_api_example
    00_cufft_device_api_example.cu
)

target_link_libraries(00_cufft_device_api_example
    PRIVATE
        cufftdx::cufftdx
)

target_link_options(00_cufft_device_api_example
    PUBLIC
        $<DEVICE_LINK:$<TARGET_OBJECTS:00_cufft_device_api_example_lto_lib>>
)

target_include_directories(00_cufft_device_api_example
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/cufft_device_api_lto_helper/00_cufft_device_api_example_artifacts
)

set_target_properties(00_cufft_device_api_example
    PROPERTIES
        CUDA_ARCHITECTURES "${CUFFTDX_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        INTERPROCEDURAL_OPTIMIZATION ON
)
