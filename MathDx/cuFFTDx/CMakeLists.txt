cmake_minimum_required(VERSION 3.26)

# cuFFTDx project
project(cufftdx_examples VERSION 1.5.0 LANGUAGES CXX CUDA)

# Project options
option(USE_MATHDX_PACKAGE "Use mathDx package to find cuFFTDx" ON)
option(USE_CUFFTDX_PACKAGE "Use cuFFTDx package to find cuFFTDx" OFF)

if(USE_CUFFTDX_PACKAGE)
    SET(USE_MATHDX_PACKAGE OFF CACHE BOOL "Use mathDx package to find cuFFTDx" FORCE)
endif()

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
        # Default path: "/opt/nvidia/mathdx/25.06", path to mathDx can be passed cmake in mathdx_ROOT variable
        find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG
            PATHS
                "${PROJECT_SOURCE_DIR}/../.." # example/cufftdx
                "${PROJECT_SOURCE_DIR}/../../.." # include/cufftdx/example
                "/opt/nvidia/mathdx/25.06"
        )
    elseif(USE_CUFFTDX_PACKAGE)
        message(STATUS "Using cuFFTDx package to find cuFFTDx")
        # Find cuFFTDx
        # Default path: "/opt/nvidia/mathdx/25.06/", path to cuFFTDx can be passed cmake in cufftdx_ROOT variable
        find_package(cufftdx REQUIRED CONFIG PATHS "/opt/nvidia/mathdx/25.06/" "${PROJECT_SOURCE_DIR}/../../")
    else()
        message(FATAL_ERROR "No cuFFTDx package found")
    endif()
endif()

if(PROJECT_IS_TOP_LEVEL)
    # Enable running examples with CTest
    enable_testing()

    set(DX_PROJECT_CMAKE_VAR_PREFIX "CUFFTDX")
    set(DX_PROJECT_FULL_NAME "cuFFTDx")

    # ******************************************
    # cuFFTDx Install Dirs
    # ******************************************
    include(GNUInstallDirs)
    set(CUFFTDX_EXAMPLES_BIN_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}/example/")

    # Include custom CMake modules/scripts
    list(APPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/
    )

    if(NOT MSVC)
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} -Wall -Wextra")
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} -fno-strict-aliasing")
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} -Wno-deprecated-declarations")
        if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch64")
            set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} -m64")
        endif ()

        if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR (CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*"))
            # Print error/warnings numbers
            set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} --display_error_number")
        endif()

        if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR (CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*"))
            # if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 23.1.0)
                set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} --diag_suppress1")
                set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} --diag_suppress111")
                set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} --diag_suppress177")
                set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} --diag_suppress941")
            # endif()
        endif()

        # Clang
        if(CMAKE_CUDA_HOST_COMPILER MATCHES ".*clang.*")
            # clang complains about unused function in CUDA system headers
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unused-function")
        endif()


        # because -Wall is passed, all diagnostic ignores of -Wunkown-pragmas
        # are ignored, which leads to unlegible CuTe compilation output
        # fixed in GCC13 https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53431
        # but still not widely adopted
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} -Wno-unknown-pragmas")
        # CUTLASS/CuTe workarounds
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} -Wno-switch")
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} -Wno-unused-but-set-parameter")
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} -Wno-sign-compare")

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
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all") # Compress all fatbins
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number") # Show error/warning numbers
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CUFFTDX_CUDA_CXX_FLAGS}\"")

    # Targeted CUDA Architectures, see https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
    set(CUFFTDX_CUDA_ARCHITECTURES 70-real;80-real CACHE
        STRING "List of targeted cuFFTDx CUDA architectures, for example \"70-real;75-real;80\""
    )
    # Remove unsupported architectures
    list(REMOVE_ITEM CUFFTDX_CUDA_ARCHITECTURES 30;32;35;37;50;52;53)
    list(REMOVE_ITEM CUFFTDX_CUDA_ARCHITECTURES 30-real;32-real;35-real;37-real;50-real;52-real;53-real;)
    list(REMOVE_ITEM CUFFTDX_CUDA_ARCHITECTURES 30-virtual;32-virtual;35-virtual;37-virtual;50-virtual;52-virtual;53-virtual)
    message(STATUS "Targeted cuFFTDx CUDA Architectures: ${CUFFTDX_CUDA_ARCHITECTURES}")

    # CUDA Architectures
    set(CMAKE_CUDA_ARCHITECTURES OFF)

endif()

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

add_compile_definitions(CUFFTDX_EXAMPLE_CMAKE)
# Enable testing only for selected architectures
foreach(CUDA_ARCH ${CUFFTDX_CUDA_ARCHITECTURES})
    # Extract SM from SM-real/SM-virtual
    string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
    list(GET CUDA_ARCH_LIST 0 ARCH)
    # Remove "a"
    string(REPLACE "a" "" ARCH "${ARCH}")
    # Remove "f"
    string(REPLACE "f" "" ARCH "${ARCH}")
    add_compile_definitions(CUFFTDX_EXAMPLE_ENABLE_SM_${ARCH})
endforeach()


# ###############################################################
# install_example
# ###############################################################
function(install_example EXAMPLE_TARGET)
    install(TARGETS ${EXAMPLE_TARGET}
        RUNTIME DESTINATION ${CUFFTDX_EXAMPLES_BIN_INSTALL_DIR}/
        COMPONENT Examples
    )
endfunction()


# ###############################################################
# add_cufftdx_example
# ###############################################################
function(add_cufftdx_example GROUP_TARGET EXAMPLE_SOURCES)

    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    # Get the absolute path and filename
    get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
    get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
     # Compose target name: dir_filename
    set(EXAMPLE_TARGET "${EXAMPLE_DIR}_${EXAMPLE_FILENAME}")

    #Compute the test name also. It starts with cuFFTDx.example.+subdirectory.+test source name
    set(EXAMPLE_NAME "cuFFTDx.example.${EXAMPLE_DIR}.${EXAMPLE_FILENAME}")

     # Compose output directory: <build_dir>/<subdir>
    set(EXAMPLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_DIR}")

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

    # Set the output directory and executable name
    set_target_properties(${EXAMPLE_TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_DIR}"
        OUTPUT_NAME "${EXAMPLE_FILENAME}"
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
            ${EXAMPLE_FILE_DIR}/${EXAMPLE_FILENAME}_cases.csv
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
    if(CUFFTDX_SANITIZERS_BUILD)
        # See https://github.com/google/sanitizers/issues/629 for why
        set_tests_properties(${EXAMPLE_NAME} PROPERTIES ENVIRONMENT "ASAN_OPTIONS=protect_shadow_gap=0")
    endif()
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# add_cufft_and_cufftdx_example
# ###############################################################
function(add_cufft_and_cufftdx_example GROUP_TARGET EXAMPLE_SOURCES)

    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    # Get the absolute path and filename
    get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
    get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
    # Compose target name: dir_filename
    set(EXAMPLE_TARGET "${EXAMPLE_DIR}_${EXAMPLE_FILENAME}")

    #Compute the test name also. It starts with cuFFTDx.example.+subdirectory.+test source name
    set(EXAMPLE_NAME "cuFFTDx.example.${EXAMPLE_DIR}.${EXAMPLE_FILENAME}")

     # Compose output directory: <build_dir>/<subdir>
    set(EXAMPLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_DIR}")

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

    # Set the output directory and executable name
    set_target_properties(${EXAMPLE_TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_DIR}"
        OUTPUT_NAME "${EXAMPLE_FILENAME}"
    )

    if(CUFFTDX_SANITIZERS_BUILD)
        # See https://github.com/google/sanitizers/issues/629 for why
        set_tests_properties(${EXAMPLE_NAME} PROPERTIES ENVIRONMENT "ASAN_OPTIONS=protect_shadow_gap=0")
    endif()
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# add_cufftdx_nvrtc_example
# ###############################################################
function(add_cufftdx_nvrtc_example GROUP_TARGET EXAMPLE_SOURCES)

    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    # Get the absolute path and filename
    get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
    get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
     # Compose target name: dir_filename
    set(EXAMPLE_TARGET "${EXAMPLE_DIR}_${EXAMPLE_FILENAME}")

    #Compute the test name also. It starts with cuFFTDx.example.+subdirectory.+test source name
    set(EXAMPLE_NAME "cuFFTDx.example.${EXAMPLE_DIR}.${EXAMPLE_FILENAME}")

     # Compose output directory: <build_dir>/<subdir>
    set(EXAMPLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_DIR}")

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

    # Set the output directory and executable name
    set_target_properties(${EXAMPLE_TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_DIR}"
        OUTPUT_NAME "${EXAMPLE_FILENAME}"
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

    if(CUFFTDX_SANITIZERS_BUILD)
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

    #Compute the test name also. It starts with cuFFTDx.example.+subdirectory.+test source name
    set(EXAMPLE_NAME "cuFFTDx.example.${EXAMPLE_DIR}.${EXAMPLE_TARGET}")

    add_test(NAME "${EXAMPLE_NAME}" COMMAND ${EXAMPLE_TARGET})
    set_tests_properties("${EXAMPLE_NAME}"
        PROPERTIES
            LABELS "CUFFTDX_EXAMPLE"
    )
endfunction()


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
add_cufftdx_example(cufftdx_examples 00_introduction_example/introduction_example.cu)
add_cufftdx_example(cufftdx_examples 01_simple_fft_thread/simple_fft_thread.cu)
add_cufftdx_example(cufftdx_examples 01_simple_fft_thread/simple_fft_thread_fp16.cu)
add_cufftdx_example(cufftdx_examples 02_simple_fft_block/simple_fft_block.cu)
add_cufftdx_example(cufftdx_examples 02_simple_fft_block/simple_fft_block_shared.cu)
add_cufftdx_example(cufftdx_examples 02_simple_fft_block/simple_fft_block_std_complex.cu)
add_cufftdx_example(cufftdx_examples 02_simple_fft_block/simple_fft_block_half2.cu)
add_cufftdx_example(cufftdx_examples 02_simple_fft_block/simple_fft_block_fp16.cu)
add_cufftdx_example(cufftdx_examples 02_simple_fft_block/simple_fft_block_c2r.cu)
add_cufftdx_example(cufftdx_examples 02_simple_fft_block/simple_fft_block_r2c.cu)
add_cufftdx_example(cufftdx_examples 02_simple_fft_block/simple_fft_block_c2r_fp16.cu)
add_cufftdx_example(cufftdx_examples 02_simple_fft_block/simple_fft_block_r2c_fp16.cu)
add_cufftdx_example(cufftdx_examples 02_simple_fft_block/simple_fft_block_cub_io.cu)
add_cufftdx_example(cufftdx_examples 03_block_fft_performance/block_fft_performance.cu)
add_cufftdx_example(cufftdx_examples 03_block_fft_performance/block_fft_performance_many.cu)
add_cufftdx_nvrtc_example(cufftdx_examples 04_nvrtc_fft/nvrtc_fft_thread.cu)
add_cufftdx_nvrtc_example(cufftdx_examples 04_nvrtc_fft/nvrtc_fft_block.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 05_fft_Xd/fft_2d.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 05_fft_Xd/fft_2d_single_kernel.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 05_fft_Xd/fft_2d_r2c_c2r.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 05_fft_Xd/fft_3d.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 05_fft_Xd/fft_3d_box_single_block.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 05_fft_Xd/fft_3d_cube_single_block.cu)
add_cufftdx_example(cufftdx_examples 06_convolution/convolution.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 06_convolution/convolution_padded.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 06_convolution/convolution_performance.cu)
add_cufftdx_example(cufftdx_examples 06_convolution/convolution_r2c_c2r.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 07_convolution_3d/convolution_3d.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 07_convolution_3d/convolution_3d_c2r.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 07_convolution_3d/convolution_3d_r2c.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 07_convolution_3d/convolution_3d_padded.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 07_convolution_3d/convolution_3d_padded_r2c.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 08_mixed_precision/mixed_precision_fft_1d.cu)
add_cufft_and_cufftdx_example(cufftdx_examples 08_mixed_precision/mixed_precision_fft_2d.cu)

if(CUFFTDX_EXAMPLES_LTO)
    add_cufftdx_example(cufftdx_examples 01_simple_fft_thread/simple_fft_thread_lto.cu)
    add_cufftdx_example(cufftdx_examples 02_simple_fft_block/simple_fft_block_c2r_lto.cu)
    add_cufftdx_example(cufftdx_examples 03_block_fft_performance/block_fft_lto_ptx_performance.cu)
    add_standalone_example(09_introduction_lto_example 09_introduction_lto_example_introduction_lto_example)
    add_standalone_example(10_cufft_device_api_example 10_cufft_device_api_example_cufft_device_api_example)
    if (NOT CMAKE_CROSSCOMPILING)
        add_cufftdx_nvrtc_example(cufftdx_examples 04_nvrtc_fft/nvrtc_fft_thread_lto.cu)
        add_cufftdx_nvrtc_example(cufftdx_examples 04_nvrtc_fft/nvrtc_fft_block_lto.cu)
    endif()
endif()
