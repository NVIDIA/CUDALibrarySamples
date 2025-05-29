cmake_minimum_required(VERSION 3.18.0)

# cuBLASDx Examples project
project(cublasdx_examples VERSION 0.4.0 LANGUAGES CXX CUDA)

# PROJECT_IS_TOP_LEVEL is available since 3.21
if(CMAKE_VERSION VERSION_LESS 3.21)
    get_directory_property(project_has_parent PARENT_DIRECTORY)
    if(project_has_parent)
        set(PROJECT_IS_TOP_LEVEL FALSE)
    else()
        set(PROJECT_IS_TOP_LEVEL TRUE)
    endif()
endif()

if(PROJECT_IS_TOP_LEVEL)
    enable_testing()

    if(NOT MSVC)
        set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} -Wall -Wextra")
        set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} -fno-strict-aliasing")
        set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} -Wno-deprecated-declarations")
        if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch64")
            set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} -m64")
        endif ()

        if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR (CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*"))
            # Print error/warnings numbers
            set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} --display_error_number")
        endif()

        if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR (CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*"))
            # if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 23.1.0)
                set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} --diag_suppress1")
                set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} --diag_suppress111")
                set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} --diag_suppress177")
                set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} --diag_suppress941")
            # endif()
        endif()

        # because -Wall is passed, all diagnostic ignores of -Wunkown-pragmas
        # are ignored, which leads to unlegible CuTe compilation output
        # fixed in GCC13 https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53431
        # but still not widely adopted
        set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} -Wno-unknown-pragmas")
        # CUTLASS/CuTe workarounds
        set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} -Wno-switch")
        set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} -Wno-unused-but-set-parameter")
        set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} -Wno-sign-compare")

        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0.0)
            # Ignore NVCC warning #940-D: missing return statement at end of non-void function
            # This happens in cuBLASDx test sources, not in cuBLASDx headers
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress 940")
            # Ignore NVCC warning warning #186-D: pointless comparison of unsigned integer with zero
            # cutlass/include/cute/algorithm/gemm.hpp(658)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress 186")
        endif()
    else()
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
        add_definitions(-D_CRT_NONSTDC_NO_WARNINGS)
        add_definitions(-D_SCL_SECURE_NO_WARNINGS)
        add_definitions(-DNOMINMAX)
        set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} /W3") # Warning level
        set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} /WX") # All warnings are errors
        set(CUBLASDX_CUDA_CXX_FLAGS "${CUBLASDX_CUDA_CXX_FLAGS} /Zc:__cplusplus") # Enable __cplusplus macro
    endif()

    # Global CXX flags/options
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUBLASDX_CUDA_CXX_FLAGS}")

    # Global CUDA CXX flags/options
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all") # Compress all fatbins
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number") # Show error/warning numbers
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CUBLASDX_CUDA_CXX_FLAGS}\"")

    # Targeted CUDA Architectures, see https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
    set(CUBLASDX_CUDA_ARCHITECTURES 70-real;80-real CACHE
        STRING "List of targeted cuBLASDx CUDA architectures, for example \"70-real;75-real;80\""
    )
    # Remove unsupported architectures
    list(REMOVE_ITEM CUBLASDX_CUDA_ARCHITECTURES 30;32;35;37;50;52;53)
    list(REMOVE_ITEM CUBLASDX_CUDA_ARCHITECTURES 30-real;32-real;35-real;37-real;50-real;52-real;53-real;)
    list(REMOVE_ITEM CUBLASDX_CUDA_ARCHITECTURES 30-virtual;32-virtual;35-virtual;37-virtual;50-virtual;52-virtual;53-virtual)
    message(STATUS "Targeted cuBLASDx CUDA Architectures: ${CUBLASDX_CUDA_ARCHITECTURES}")
endif()

if(PROJECT_IS_TOP_LEVEL OR CUBLASDX_TEST_PACKAGE)
    find_package(mathdx REQUIRED CONFIG
        PATHS
            "${PROJECT_SOURCE_DIR}/../.." # example/cublasdx
            "/opt/nvidia/mathdx/25.06"
    )
endif()


# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Enable testing only for selected architectures
foreach(CUDA_ARCH ${CUBLASDX_CUDA_ARCHITECTURES})
    # Extract SM from SM-real/SM-virtual
    string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
    list(GET CUDA_ARCH_LIST 0 ARCH)
    # Remove "a"
    string(REPLACE "a" "" ARCH "${ARCH}")
    # Remove "f"
    string(REPLACE "f" "" ARCH "${ARCH}")

    add_compile_definitions(CUBLASDX_EXAMPLE_ENABLE_SM_${ARCH})
endforeach()

###############################################################
# Common correctness checking objects
# ###############################################################
if (NOT TARGET common_correctness_objects)
    set(common_correctness_objects_sources reference/check_error.cu reference/naive_reference.cu)
    set_source_files_properties("${common_correctness_objects_sources}" PROPERTIES LANGUAGE CUDA)
    add_library(common_correctness_objects STATIC "${common_correctness_objects_sources}")
    target_link_libraries(common_correctness_objects PRIVATE mathdx::cublasdx)
    set_target_properties(common_correctness_objects
        PROPERTIES
            CUDA_ARCHITECTURES "${CUBLASDX_CUDA_ARCHITECTURES}"
    )
endif()

# ###############################################################
# add_cufftdx_cublasdx_example
# ###############################################################
function(add_cufftdx_cublasdx_example GROUP_TARGET EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)

    # Get the absolute path and filename
    get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
    get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
     # Compose target name: dir_filename
    set(EXAMPLE_TARGET "${EXAMPLE_DIR}_${EXAMPLE_FILENAME}")

    set(EXAMPLE_NAME "cuBLASDx.example.${EXAMPLE_DIR}.${EXAMPLE_FILENAME}")

     # Compose output directory: <build_dir>/<subdir>
    set(EXAMPLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_DIR}")

    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            mathdx::cublasdx
            CUDA::cublas
            cufftdx::cufftdx
            CUDA::cufft
            common_correctness_objects
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})

    # Set the output directory and executable name
    set_target_properties(${EXAMPLE_TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_DIR}"
        OUTPUT_NAME "${EXAMPLE_FILENAME}"
    )

    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${CUBLASDX_CUDA_ARCHITECTURES}"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
            # Required to support std::tuple in device code
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
    )
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# add_cublasdx_example
# ###############################################################
function(add_cublasdx_example2 GROUP_TARGET ENABLE_RELAXED_CONSTEXPR EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)

    # Get the absolute path and filename
    get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
    get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
     # Compose target name: dir_filename
    set(EXAMPLE_TARGET "${EXAMPLE_DIR}_${EXAMPLE_FILENAME}")

    set(EXAMPLE_NAME "cuBLASDx.example.${EXAMPLE_DIR}.${EXAMPLE_FILENAME}")

     # Compose output directory: <build_dir>/<subdir>
    set(EXAMPLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_DIR}")

    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            mathdx::cublasdx
            CUDA::cublas
            common_correctness_objects
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})

    # Set the output directory and executable name
    set_target_properties(${EXAMPLE_TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_DIR}"
        OUTPUT_NAME "${EXAMPLE_FILENAME}"
    )

    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${CUBLASDX_CUDA_ARCHITECTURES}"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
            # Required to support std::tuple in device code
            "$<$<BOOL:${ENABLE_RELAXED_CONSTEXPR}>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>>"
    )
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

function(add_cublasdx_example GROUP_TARGET EXAMPLE_SOURCES)
    add_cublasdx_example2("${GROUP_TARGET}" True "${EXAMPLE_SOURCES}")
endfunction()

# ###############################################################
# add_cublasdx_nvrtc_example
# ###############################################################
function(add_cublasdx_nvrtc_example GROUP_TARGET EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)

    # Get the absolute path and filename
    get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
    get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
    # Compose target name: dir_filename
    set(EXAMPLE_TARGET "${EXAMPLE_DIR}_${EXAMPLE_FILENAME}")
    
    set(EXAMPLE_NAME "cuBLASDx.example.${EXAMPLE_DIR}.${EXAMPLE_FILENAME}")
     # Compose output directory: <build_dir>/<subdir>
    set(EXAMPLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_DIR}")

    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            $<IF:$<TARGET_EXISTS:mathdx::cublasdx>,mathdx::cublasdx,cublasdx::cublasdx>
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nvrtc
            common_correctness_objects
    )
    target_compile_definitions(${EXAMPLE_TARGET}
        PRIVATE
            CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
            CUTLASS_INCLUDE_DIR="${cublasdx_cutlass_INCLUDE_DIR}"
            COMMONDX_INCLUDE_DIR="${cublasdx_commondx_INCLUDE_DIR}"
            CUBLASDX_INCLUDE_DIRS="${cublasdx_INCLUDE_DIRS}"
    )
    # Disable NVRTC examples for NVC++ compiler, because of libcu++/NRTC/NVC++ bug
    if(NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR (CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*")))
        add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
        set_tests_properties(${EXAMPLE_NAME}
            PROPERTIES
                LABELS "EXAMPLE"
        )
    endif()

    # Set the output directory and executable name
    set_target_properties(${EXAMPLE_TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_DIR}"
        OUTPUT_NAME "${EXAMPLE_FILENAME}"
    )
    
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${CUBLASDX_CUDA_ARCHITECTURES}"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# cuBLASDx Examples
# ###############################################################

add_custom_target(cublasdx_examples)

# cuBLASDx NVRTC examples
add_cublasdx_nvrtc_example(cublasdx_examples 15_gemm_nvrtc/nvrtc_gemm.cpp)

# cuBLASDx introduction examples
add_cublasdx_example(cublasdx_examples 01_gemm_introduction/introduction_example.cu)

# cuBLASDx simple examples
add_cublasdx_example(cublasdx_examples 02_gemm_precisions/simple_gemm_mixed_precision.cu)
add_cublasdx_example(cublasdx_examples 02_gemm_precisions/simple_gemm_fp32.cu)
add_cublasdx_example(cublasdx_examples 02_gemm_precisions/simple_gemm_int8_int8_int32.cu)
add_cublasdx_example(cublasdx_examples 08_gemm_decoupled_io_and_compute/simple_gemm_fp32_decoupled.cu)
add_cublasdx_example(cublasdx_examples 03_gemm_complex/simple_gemm_cfp16.cu)
add_cublasdx_example(cublasdx_examples 02_gemm_precisions/simple_gemm_fp8.cu)
add_cublasdx_example(cublasdx_examples 03_gemm_complex/simple_gemm_std_complex_fp32.cu)
add_cublasdx_example(cublasdx_examples 06_gemm_leading_dimension/simple_gemm_leading_dimensions.cu)
add_cublasdx_example(cublasdx_examples 09_gemm_custom_layout/simple_gemm_custom_layout.cu)
add_cublasdx_example(cublasdx_examples 07_gemm_transform/simple_gemm_transform.cu)
add_cublasdx_example(cublasdx_examples 09_gemm_custom_layout/simple_gemm_aat.cu)
add_cublasdx_example(cublasdx_examples 11_gemm_device_performance/device_gemm_performance.cu)
add_cublasdx_example(cublasdx_examples 12_gemm_device_partial_sums/gemm_device_partial_sums.cu)

# cuBLASDx advanced examples
add_cublasdx_example(cublasdx_examples 14_gemm_fused/gemm_fusion.cu)
add_cublasdx_example(cublasdx_examples 04_gemm_blockdim/blockdim_gemm_fp16.cu)
add_cublasdx_example(cublasdx_examples 05_gemm_batched/batched_gemm_fp64.cu)

# cuBLASDx performance examples
# Examples which measure performance are enabled only for CUDA >=11.8.0 because of an NVCC bug
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8.0)
    add_cublasdx_example(cublasdx_examples 10_gemm_block_performance/single_gemm_performance.cu)
    add_cublasdx_example(cublasdx_examples 14_gemm_fused/fused_gemm_performance.cu)
endif()

# cuBLASDx/cuFFTDx examples
if(cufftdx_FOUND)
    # Examples which measure performance are enabled only for CUDA >=11.8.0 because of an NVCC bug
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8.0)
        add_cufftdx_cublasdx_example(cublasdx_examples 13_gemm_fft/gemm_fft.cu)
        add_cufftdx_cublasdx_example(cublasdx_examples 13_gemm_fft/gemm_fft_fp16.cu)
        add_cufftdx_cublasdx_example(cublasdx_examples 13_gemm_fft/gemm_fft_performance.cu)
    endif()
endif()
