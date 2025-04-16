cmake_minimum_required(VERSION 3.26.0)

# cuFFTDx Examples project
project(cufftdx_examples LANGUAGES CXX CUDA)

# Find CUDA Toolkit packaged, required for NVRTC sample
find_package(CUDAToolkit)

# Project options
option(USE_MATHDX_PACKAGE "Use mathDx package to find cuFFTDx" ON)
option(USE_CUFFTDX_PACKAGE "Use cuFFTDx package to find cuFFTDx" OFF)

if(DEFINED cufftdx_ROOT OR DEFINED ENV{cufftdx_ROOT})
    SET(USE_CUFFTDX_PACKAGE ON CACHE BOOL "Use cuFFTDx package to find cuFFTDx" FORCE)
    SET(USE_MATHDX_PACKAGE OFF CACHE BOOL "Use mathDx package to find cuFFTDx" FORCE)
endif()

if(DEFINED mathdx_ROOT OR DEFINED ENV{mathdx_ROOT})
    SET(USE_CUFFTDX_PACKAGE OFF CACHE BOOL "Use cuFFTDx package to find cuFFTDx" FORCE)
    SET(USE_MATHDX_PACKAGE ON CACHE BOOL "Use mathDx package to find cuFFTDx" FORCE)
endif()

if(NOT TARGET cufftdx)
    #Setting the flag for the targets inside package
    set(cufftdx_SEPARATE_TWIDDLES_CUDA_ARCHITECTURES ${CUFFTDX_CUDA_ARCHITECTURES})
    if(USE_MATHDX_PACKAGE)
        message(STATUS "Using mathDx package to find cuFFTDx")
        # Find mathDx and cuFFTDx (mathDx's component)
        # Default path: "/opt/nvidia/mathdx/25.01.1", path to mathDx can be passed cmake in mathdx_ROOT variable
        find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG
            PATHS
                "${PROJECT_SOURCE_DIR}/../.." # example/cufftdx
                "${PROJECT_SOURCE_DIR}/../../.." # include/cufftdx/example
                "/opt/nvidia/mathdx/25.01.1"
        )
    elseif(USE_CUFFTDX_PACKAGE)
        message(STATUS "Using cuFFTDx package to find cuFFTDx")
        # Find cuFFTDx
        # Default path: "/opt/nvidia/mathdx/25.01.1/", path to cuFFTDx can be passed cmake in cufftdx_ROOT variable
        find_package(cufftdx REQUIRED CONFIG
            PATHS
                "/opt/nvidia/mathdx/25.01.1/"
                "${PROJECT_SOURCE_DIR}/../../"
        )
    else()
        message(FATAL_ERROR "No cuFFTDx package found")
    endif()
endif()

if((NOT TARGET cufftdx) AND (NOT CUFFTDX_TEST_RELEASED_PACKAGE) AND (NOT MATHDX_TEST_RELEASED_PACKAGE))
    # Targeted CUDA Architectures, see https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
    if(CUFFTDX_TARGET_ARCHS)
        set(CUFFTDX_TARGET_ARCHS 70;75;80 CACHE
            STRING "[LEGACY] List of targeted cuFFTDx Example CUDA architectures (compute capabilities), for example \"70;75\". Can't be older than 70."
        )
        list(SORT CUFFTDX_TARGET_ARCHS)
        # Remove unsupported architectures
        list(REMOVE_ITEM CUFFTDX_TARGET_ARCHS 30;32;35;37;50;52;53;60;61;62)

        # Translate legacy option CUFFTDX_TARGET_ARCHS into CUFFTDX_CUDA_ARCHITECTURES
        set(CUFFTDX_TARGET_ARCHS_TRANSLATED)
        foreach(ARCH ${CUFFTDX_TARGET_ARCHS})
            list(APPEND CUFFTDX_TARGET_ARCHS_TRANSLATED ${ARCH}-real)
        endforeach()
        set(CUFFTDX_CUDA_ARCHITECTURES ${CUFFTDX_TARGET_ARCHS_TRANSLATED} CACHE
            STRING "List of targeted cuFFTDx CUDA architectures, for example \"70-real;75-real;80\""
        )
    else()
        set(CUFFTDX_CUDA_ARCHITECTURES 70-real;75-real;80-real CACHE
            STRING "List of targeted cuFFTDX CUDA architectures, for example \"70-real;75-real;80\""
        )
        # Remove unsupported architectures
        list(REMOVE_ITEM CUFFTDX_CUDA_ARCHITECTURES 30;32;35;37;50;52;53;60;61;62)
        list(REMOVE_ITEM CUFFTDX_CUDA_ARCHITECTURES 30-real;32-real;35-real;37-real;50-real;52-real;53-real;60-real;61-real;62-real)
        list(REMOVE_ITEM CUFFTDX_CUDA_ARCHITECTURES 30-virtual;32-virtual;35-virtual;37-virtual;50-virtual;52-virtual;53-virtual;60-virtual;61-virtual;62-virtual)
    endif()
    message(STATUS "Targeted cuFFTDx Examples CUDA Architectures: ${CUFFTDX_CUDA_ARCHITECTURES}")

    # Global CXX/CUDA flags
    if(NOT MSVC)
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} -Wall -Wextra -Wno-deprecated-declarations")
    else()
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
        add_definitions(-D_CRT_NONSTDC_NO_WARNINGS)
        add_definitions(-D_SCL_SECURE_NO_WARNINGS)
        add_definitions(-DNOMINMAX)
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} /W3") # Warning level
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} /WX") # All warnings are errors
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} /Zc:__cplusplus") # Enable __cplusplus macro
    endif()

    # Global CXX flags/options
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUFFTDX_CUDA_CXX_FLAGS}")

    # Global CUDA CXX flags/options
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CUFFTDX_CUDA_CXX_FLAGS}\"")

    # Clang
    if(BUILD_CUFFTDX)
        if(CMAKE_CUDA_HOST_COMPILER MATCHES ".*clang.*")
            # clang complains about unused function in CUDA system headers
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unused-function")
        endif()
    endif()

    # CUDA Architectures
    set(CMAKE_CUDA_ARCHITECTURES OFF)

    # Enable testing (ctest)
    enable_testing()
endif()

# ###############################################################
# add_cufftdx_example
# ###############################################################
function(add_cufftdx_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            $<IF:$<TARGET_EXISTS:mathdx::cufftdx>,mathdx::cufftdx,cufftdx::cufftdx>
    )
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${CUFFTDX_CUDA_ARCHITECTURES}"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "CUFFTDX_EXAMPLE"
    )
    # LTO example
    string(FIND ${EXAMPLE_NAME} "lto" LTO_EXAMPLE)
    if(${LTO_EXAMPLE} GREATER -1)
        include(lto_helper/lto_helper.cmake)
        # Get directory of test sources file
        get_filename_component(EXAMPLE_FILE_DIR ${EXAMPLE_MAIN_SOURCE} REALPATH)
        get_filename_component(EXAMPLE_FILE_DIR ${EXAMPLE_FILE_DIR} DIRECTORY)
        # Generate and link LTO blobs
        run_cufft_lto_helper(
            ${CMAKE_CURRENT_SOURCE_DIR}/lto_helper/
            ${CMAKE_CURRENT_BINARY_DIR}/lto_helper/
            ${EXAMPLE_TARGET}
            ${EXAMPLE_FILE_DIR}/${EXAMPLE_TARGET}_cases.csv
            "${CUFFTDX_CUDA_ARCHITECTURES}"
            TRUE # Force regenerate blobs
        )
        if(NOT CUFFT_LTO_HELPER_RESULT)
            message(FATAL_ERROR "run_cufft_lto_helper encountered errors for target ${LTO_EXAMPLE}.")
        endif()
        target_link_options(${EXAMPLE_TARGET} PUBLIC $<DEVICE_LINK:$<TARGET_OBJECTS:${EXAMPLE_TARGET}_lto_lib>>)
        target_include_directories(${EXAMPLE_TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/lto_helper) # Include LTO database file
        set_target_properties(${EXAMPLE_TARGET}
            PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                INTERPROCEDURAL_OPTIMIZATION ON
        )
    endif()
    if(CUFFTDX_SANTIZERS_BUILD)
        # See https://github.com/google/sanitizers/issues/629 for why
        set_tests_properties(${EXAMPLE_NAME} PROPERTIES ENVIRONMENT "ASAN_OPTIONS=protect_shadow_gap=0")
    endif()
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# add_cufft_and_cufftdx_example
# ###############################################################
function(add_cufft_and_cufftdx_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            $<IF:$<TARGET_EXISTS:mathdx::cufftdx>,mathdx::cufftdx,cufftdx::cufftdx>
    )
    if(CUFFTDX_EXAMPLES_CUFFT_CALLBACK)
        if(TARGET cufft)
            target_link_libraries(${EXAMPLE_TARGET} PRIVATE cufft_static)
        else()
            target_link_libraries(${EXAMPLE_TARGET} PRIVATE CUDA::cufft_static)
        endif()
        set_target_properties(${EXAMPLE_TARGET}
            PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
        )
        target_compile_definitions(${EXAMPLE_TARGET} PRIVATE CUFFTDX_EXAMPLES_CUFFT_CALLBACK)
    else()
        if(TARGET cufft)
            target_link_libraries(${EXAMPLE_TARGET} PRIVATE cufft)
        else()
            target_link_libraries(${EXAMPLE_TARGET} PRIVATE CUDA::cufft)
        endif()
    endif()
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${CUFFTDX_CUDA_ARCHITECTURES}"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "CUFFTDX_EXAMPLE"
    )
    if(CUFFTDX_SANTIZERS_BUILD)
        # See https://github.com/google/sanitizers/issues/629 for why
        set_tests_properties(${EXAMPLE_NAME} PROPERTIES ENVIRONMENT "ASAN_OPTIONS=protect_shadow_gap=0")
    endif()
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# add_cufftdx_nvrtc_example
# ###############################################################
function(add_cufftdx_nvrtc_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            $<IF:$<TARGET_EXISTS:mathdx::cufftdx>,mathdx::cufftdx,cufftdx::cufftdx>
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nvrtc
            CUDA::cuda_driver
    )
    target_compile_definitions(${EXAMPLE_TARGET}
        PRIVATE
            CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
            CUTLASS_INCLUDE_DIR="${cufftdx_cutlass_INCLUDE_DIR}"
            COMMONDX_INCLUDE_DIR="${cufftdx_commondx_INCLUDE_DIR}"
            CUFFTDX_INCLUDE_DIRS="${cufftdx_INCLUDE_DIRS}"
    )
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${CUFFTDX_CUDA_ARCHITECTURES}"
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "CUFFTDX_EXAMPLE"
    )
    # LTO example
    string(FIND ${EXAMPLE_NAME} "lto" LTO_EXAMPLE)
    if(${LTO_EXAMPLE} GREATER -1)
        find_package(cufft 11.5.0 EXACT REQUIRED CONFIG
            PATHS
                "${PROJECT_SOURCE_DIR}/../../../cufft"
                "/opt/cufft"
        )
        target_compile_definitions(${EXAMPLE_TARGET} PRIVATE CUFFTDX_ENABLE_CUFFT_DEPENDENCY) # To include cufftdx::utils::get_database_and_ltoir helper function
        target_link_libraries(${EXAMPLE_TARGET}
            PRIVATE
                cufft::cufft_static
                CUDA::nvJitLink
        )
    endif()
    if(CUFFTDX_SANTIZERS_BUILD)
        # See https://github.com/google/sanitizers/issues/629 for why
        set_tests_properties(${EXAMPLE_NAME} PROPERTIES ENVIRONMENT "ASAN_OPTIONS=protect_shadow_gap=0")
    endif()
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# add_standalone_example
# ###############################################################
function(add_standalone_example EXAMPLE_DIR EXAMPLE_TARGET)
    add_subdirectory(${EXAMPLE_DIR})
    add_test(NAME "${EXAMPLE_TARGET}" COMMAND ${EXAMPLE_TARGET})
    set_tests_properties("${EXAMPLE_TARGET}"
        PROPERTIES
            LABELS "CUFFTDX_EXAMPLE"
    )
endfunction()

# Enable testing only for selected architectures
add_compile_definitions(CUFFTDX_EXAMPLE_CMAKE)
foreach(CUDA_ARCH ${CUFFTDX_CUDA_ARCHITECTURES})
    # Extract SM from SM-real/SM-virtual
    string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
    list(GET CUDA_ARCH_LIST 0 ARCH)
    add_compile_definitions(CUFFTDX_EXAMPLE_ENABLE_SM_${ARCH})
endforeach()

# ###############################################################
# cuFFTDx Examples
# ###############################################################

add_custom_target(cufftdx_examples)

# CUFFTDX_EXAMPLES_CUFFT_CALLBACK
option(CUFFTDX_EXAMPLES_CUFFT_CALLBACK "Build cuFFTDx convolution_performance example with cuFFT callback" OFF)

# CUFFTDX_EXAMPLES_LTO
# cuFFTDx LTO examples are currently supported on EA (cuFFTDx 1.4.0) ahead of productization
if((cufftdx_FOUND AND (cufftdx_VERSION VERSION_EQUAL 1.4.0)) OR
   ((NOT CUFFTDX_TEST_RELEASED_PACKAGE) AND (CUFFTDX_PROJECT_VERSION VERSION_EQUAL 1.4.0)))
    option(CUFFTDX_EXAMPLES_LTO "Build cuFFTDx LTO examples" ON)
else()
    option(CUFFTDX_EXAMPLES_LTO "Build cuFFTDx LTO examples" OFF)
endif()

# CTK requirement for cuFFTDx LTO examples
if(CUFFTDX_EXAMPLES_LTO AND NOT ${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "12.8.0")
    message(FATAL_ERROR "cuFFTDx LTO examples (version ${cufftdx_VERSION}) requires CUDAToolkit version 12.8 and above.")
endif()

add_cufftdx_example(cufftdx_examples "cuFFTDx.example.introduction_example" 00_introduction_example/00_introduction_example.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_thread" 01_simple_fft_thread/00_simple_fft_thread.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_thread_fp16" 01_simple_fft_thread/01_simple_fft_thread_fp16.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block" 02_simple_fft_block/00_simple_fft_block.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_shared" 02_simple_fft_block/01_simple_fft_block_shared.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_std_complex" 02_simple_fft_block/02_simple_fft_block_std_complex.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_half2" 02_simple_fft_block/03_simple_fft_block_half2.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_fp16" 02_simple_fft_block/04_simple_fft_block_fp16.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_c2r" 02_simple_fft_block/05_simple_fft_block_c2r.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_r2c" 02_simple_fft_block/06_simple_fft_block_r2c.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_c2r_fp16" 02_simple_fft_block/07_simple_fft_block_c2r_fp16.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_r2c_fp16" 02_simple_fft_block/08_simple_fft_block_r2c_fp16.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_cub_io" 02_simple_fft_block/09_simple_fft_block_cub_io.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.block_fft_performance" 03_block_fft_performance/00_block_fft_performance.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.block_fft_performance_many" 03_block_fft_performance/01_block_fft_performance_many.cu)
add_cufftdx_nvrtc_example(cufftdx_examples "cuFFTDx.example.nvrtc_fft_thread" 04_nvrtc_fft/00_nvrtc_fft_thread.cu)
add_cufftdx_nvrtc_example(cufftdx_examples "cuFFTDx.example.nvrtc_fft_block" 04_nvrtc_fft/01_nvrtc_fft_block.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.fft_2d" 05_fft_Xd/00_fft_2d.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.fft_2d_single_kernel" 05_fft_Xd/01_fft_2d_single_kernel.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.fft_2d_r2c_c2r" 05_fft_Xd/02_fft_2d_r2c_c2r.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.fft_3d" 05_fft_Xd/03_fft_3d.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.fft_3d_box_single_block" 05_fft_Xd/04_fft_3d_box_single_block.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.fft_3d_cube_single_block" 05_fft_Xd/05_fft_3d_cube_single_block.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.convolution" 06_convolution/00_convolution.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.convolution_padded" 06_convolution/01_convolution_padded.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.convolution_performance" 06_convolution/02_convolution_performance.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.convolution_r2c_c2r" 06_convolution/03_convolution_r2c_c2r.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.convolution_3d" 07_convolution_3d/00_convolution_3d.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.convolution_3d_c2r" 07_convolution_3d/01_convolution_3d_c2r.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.convolution_3d_r2c" 07_convolution_3d/02_convolution_3d_r2c.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.convolution_3d_padded" 07_convolution_3d/03_convolution_3d_padded.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.convolution_3d_padded_r2c" 07_convolution_3d/04_convolution_3d_padded_r2c.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.mixed_precision_fft_1d" 08_mixed_precision/00_mixed_precision_fft_1d.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.mixed_precision_fft_2d" 08_mixed_precision/01_mixed_precision_fft_2d.cu)
if(CUFFTDX_EXAMPLES_LTO)
    add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_thread_lto" 01_simple_fft_thread/02_simple_fft_thread_lto.cu)
    add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_c2r_lto" 02_simple_fft_block/10_simple_fft_block_c2r_lto.cu)
    add_cufftdx_example(cufftdx_examples "cuFFTDx.example.block_fft_lto_ptx_performance" 03_block_fft_performance/02_block_fft_lto_ptx_performance.cu)
    add_standalone_example(09_introduction_lto_example 00_introduction_lto_example)
    add_standalone_example(10_cufft_device_api_example 00_cufft_device_api_example)
    if (NOT CMAKE_CROSSCOMPILING)
        add_cufftdx_nvrtc_example(cufftdx_examples "cuFFTDx.example.nvrtc_fft_thread_lto" 04_nvrtc_fft/02_nvrtc_fft_thread_lto.cu)
        add_cufftdx_nvrtc_example(cufftdx_examples "cuFFTDx.example.nvrtc_fft_block_lto" 04_nvrtc_fft/03_nvrtc_fft_block_lto.cu)
    endif()
endif()
