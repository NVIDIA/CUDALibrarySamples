cmake_minimum_required(VERSION 3.23.0)

# cuRANDDx example project
project(curanddx_example VERSION ${CMAKE_PROJECT_VERSION} LANGUAGES CXX CUDA)

if(PROJECT_IS_TOP_LEVEL)
    if(NOT MSVC)
        set(CURANDDX_CUDA_CXX_FLAGS "${CURANDDX_CUDA_CXX_FLAGS} -Wall -Wextra")
        set(CURANDDX_CUDA_CXX_FLAGS "${CURANDDX_CUDA_CXX_FLAGS} -fno-strict-aliasing")

        if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch64")
            set(CURANDDX_CUDA_CXX_FLAGS "${CURANDDX_CUDA_CXX_FLAGS} -m64")
        endif()

        if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR(CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*"))
            # Print error/warnings numbers
            set(CURANDDX_CUDA_CXX_FLAGS "${CURANDDX_CUDA_CXX_FLAGS} --display_error_number")
        endif()

        if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR(CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*"))
            # set(CURANDDX_CUDA_CXX_FLAGS "${CURANDDX_CUDA_CXX_FLAGS} --diag_suppress=1,111,177,185,941")
        endif()
    else()
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
        add_definitions(-D_CRT_NONSTDC_NO_WARNINGS)
        add_definitions(-D_SCL_SECURE_NO_WARNINGS)
        add_definitions(-DNOMINMAX)
        set(CURANDDX_CUDA_CXX_FLAGS "${CURANDDX_CUDA_CXX_FLAGS} /W3") # Warning level
        set(CURANDDX_CUDA_CXX_FLAGS "${CURANDDX_CUDA_CXX_FLAGS} /WX") # All warnings are errors
        set(CURANDDX_CUDA_CXX_FLAGS "${CURANDDX_CUDA_CXX_FLAGS} /Zc:__cplusplus") # Enable __cplusplus macro
    endif()

    # Global CXX flags/options
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CURANDDX_CUDA_CXX_FLAGS}")

    # Global CUDA CXX flags/options
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all") # Compress all fatbins
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number") # Show error/warning numbers
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CURANDDX_CUDA_CXX_FLAGS}\"")

    # Targeted CUDA Architectures, see https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
    set(CURANDDX_CUDA_ARCHITECTURES 70-real;80-real CACHE
        STRING "List of targeted cuRANDDx CUDA architectures, for example \"70-real;75-real;80\""
    )

    # Remove unsupported architectures
    list(REMOVE_ITEM CURANDDX_CUDA_ARCHITECTURES 30;32;35;37;50;52;53)
    list(REMOVE_ITEM CURANDDX_CUDA_ARCHITECTURES 30-real;32-real;35-real;37-real;50-real;52-real;53-real;)
    list(REMOVE_ITEM CURANDDX_CUDA_ARCHITECTURES 30-virtual;32-virtual;35-virtual;37-virtual;50-virtual;52-virtual;53-virtual)
    message(STATUS "Targeted cuRANDDx CUDA Architectures: ${CURANDDX_CUDA_ARCHITECTURES}")
endif()

# ******************************************
# Tests as standalone project to enable testing release package
# ******************************************
if(PROJECT_IS_TOP_LEVEL)
    find_package(mathdx REQUIRED CONFIG
        PATHS
            "${PROJECT_SOURCE_DIR}/../.." # example/curanddx
            "/opt/nvidia/mathdx/25.06"
    )
endif()

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Enable testing only for selected architectures
foreach(CUDA_ARCH ${CURANDDX_CUDA_ARCHITECTURES})
    # Extract SM from SM-real/SM-virtual
    string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
    list(GET CUDA_ARCH_LIST 0 ARCH)
    # Remove "a"
    string(REPLACE "a" "" ARCH "${ARCH}")
    # Remove "f"
    string(REPLACE "f" "" ARCH "${ARCH}")
    add_compile_definitions(CURANDDX_EXAMPLE_ENABLE_SM_${ARCH})
endforeach()

if(CURANDDX_EXAMPLE_NVPL_RAND_AVAILABLE)
    # -Dnvpl_root=<NVPL install root path>
    find_package(nvpl REQUIRED COMPONENTS rand)
endif()

# ###############################################################
# add_curanddx_example_nvpl_rand
# ###############################################################
function(add_curanddx_example2_nvpl_rand GROUP_TARGET EXAMPLE_NAME ENABLE_RELAXED_CONSTEXPR EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})

    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            mathdx::curanddx
            CUDA::curand
    )
    if(CURANDDX_EXAMPLE_NVPL_RAND_AVAILABLE)
        target_link_libraries(${EXAMPLE_TARGET} PRIVATE nvpl::rand_mt)
        target_compile_definitions(${EXAMPLE_TARGET} PRIVATE -DCURANDDX_EXAMPLE_NVPL_RAND_AVAILABLE)
    endif()

    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
        CUDA_ARCHITECTURES "${CURANDDX_CUDA_ARCHITECTURES}"
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

function(add_curanddx_example_nvpl_rand GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    add_curanddx_example2_nvpl_rand("${GROUP_TARGET}" "${EXAMPLE_NAME}" True "${EXAMPLE_SOURCES}")
endfunction()

# ###############################################################
# add_curanddx_example
# ###############################################################
function(add_curanddx_example2 GROUP_TARGET EXAMPLE_NAME ENABLE_RELAXED_CONSTEXPR EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})

    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
        mathdx::curanddx
        CUDA::curand
    )

    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
        CUDA_ARCHITECTURES "${CURANDDX_CUDA_ARCHITECTURES}"
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

function(add_curanddx_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    add_curanddx_example2("${GROUP_TARGET}" "${EXAMPLE_NAME}" True "${EXAMPLE_SOURCES}")
endfunction()

# ###############################################################
# add_curanddx_nvrtc_example
# ###############################################################
function(add_curanddx_nvrtc_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})

    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            mathdx::curanddx
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nvrtc
    )
    target_compile_definitions(${EXAMPLE_TARGET}
        PRIVATE
            CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
            COMMONDX_INCLUDE_DIR="${curanddx_commondx_INCLUDE_DIR}"
            CURANDDX_INCLUDE_DIRS="${curanddx_INCLUDE_DIRS}"
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
        LABELS "EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()


# ###############################################################
# cuRANDDx Examples
# ###############################################################
add_custom_target(curanddx_examples)

# cuRANDDx examples using thread API
add_curanddx_example_nvpl_rand(curanddx_examples "cuRANDDx.example.introduction_simple_pcg_thread_api" 02_thread_api/pcg_thread_api.cu)

add_curanddx_example(curanddx_examples "cuRANDDx.example.philox_thread_api" 01_introduction/philox_thread_api.cu)
add_curanddx_example(curanddx_examples "cuRANDDx.example.xorwow_init_and_generate_thread_api" 02_thread_api/xorwow_init_and_generate_thread_api.cu)
add_curanddx_example(curanddx_examples "cuRANDDx.example.sobol_thread_api" 02_thread_api/sobol_thread_api.cu)
add_curanddx_example(curanddx_examples "cuRANDDx.example.mrg_two_distributions_thread_api" 02_thread_api/mrg_two_distributions_thread_api.cu)

add_curanddx_nvrtc_example(curanddx_examples "cuRANDDx.example.nvrtc_pcg_thread_api" 03_nvrtc/nvrtc_pcg_thread_api.cpp)
