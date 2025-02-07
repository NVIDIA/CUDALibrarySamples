cmake_minimum_required(VERSION 3.18)

# cuSOLVERDx project
project(cusolverdx_examples VERSION 0.1.0 LANGUAGES CXX CUDA)

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
    # Enable running examples with CTest
    enable_testing()

    set(DX_PROJECT_CMAKE_VAR_PREFIX "CUSOLVERDX")
    set(DX_PROJECT_FULL_NAME "cuSOLVERDx")

    # ******************************************
    # cuSOLVERDx Install Dirs
    # ******************************************
    include(GNUInstallDirs)
    set(CUSOLVERDX_EXAMPLES_BIN_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}/example/")

    # Include custom CMake modules/scripts
    list(APPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/
    )

    if(NOT MSVC)
        set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} -Wall -Wextra")
        set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} -fno-strict-aliasing")
        set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} -Wno-deprecated-declarations")

        if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch64")
            set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} -m64")
        endif()

        if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR(CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*"))
            # Print error/warnings numbers
            set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} --display_error_number")
        endif()

        if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR(CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*"))
            # if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 23.1.0)
            set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} --diag_suppress1")
            set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} --diag_suppress111")
            set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} --diag_suppress177")
            set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} --diag_suppress941")

            # endif()
        endif()

        # because -Wall is passed, all diagnostic ignores of -Wunkown-pragmas
        # are ignored, which leads to unlegible CuTe compilation output
        # fixed in GCC13 https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53431
        # but still not widely adopted
        set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} -Wno-unknown-pragmas")

        # CUTLASS/CuTe workarounds
        set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} -Wno-switch")
        set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} -Wno-unused-but-set-parameter")
        set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} -Wno-sign-compare")

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
        set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} /W3") # Warning level
        set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} /WX") # All warnings are errors
        set(CUSOLVERDX_CUDA_CXX_FLAGS "${CUSOLVERDX_CUDA_CXX_FLAGS} /Zc:__cplusplus") # Enable __cplusplus macro
    endif()

    # Global CXX flags/options
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUSOLVERDX_CUDA_CXX_FLAGS}")

    # Global CUDA CXX flags/options
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all") # Compress all fatbins
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number") # Show error/warning numbers
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CUSOLVERDX_CUDA_CXX_FLAGS}\"")

    # Targeted CUDA Architectures, see https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
    set(CUSOLVERDX_CUDA_ARCHITECTURES 70-real;80-real CACHE
        STRING "List of targeted cuSOLVERDx CUDA architectures, for example \"70-real;75-real;80\""
    )

    # Remove unsupported architectures
    list(REMOVE_ITEM CUSOLVERDX_CUDA_ARCHITECTURES 30;32;35;37;50;52;53)
    list(REMOVE_ITEM CUSOLVERDX_CUDA_ARCHITECTURES 30-real;32-real;35-real;37-real;50-real;52-real;53-real;)
    list(REMOVE_ITEM CUSOLVERDX_CUDA_ARCHITECTURES 30-virtual;32-virtual;35-virtual;37-virtual;50-virtual;52-virtual;53-virtual)
    message(STATUS "Targeted cuSOLVERDx CUDA Architectures: ${CUSOLVERDX_CUDA_ARCHITECTURES}")
endif()

# ******************************************
# Tests as standalone project to enable testing release package
# ******************************************
if(PROJECT_IS_TOP_LEVEL OR CUSOLVERDX_TEST_PACKAGE)
    # Project options
    message(CHECK_START "Example: Looking for mathDx package")
    find_package(mathdx QUIET COMPONENTS cusolverdx cublasdx CONFIG
        PATHS
            "${PROJECT_SOURCE_DIR}/../.." # example/cusolverdx
            "${PROJECT_SOURCE_DIR}/../../.." # include/cusolverdx/example
            "/opt/nvidia/mathdx/25.01"
    )

    if(mathdx_FOUND)
        message(CHECK_PASS "found, dependencies resolved")
    else()
        message(CHECK_FAIL "failed, looking separately")
        message(CHECK_START "Example: Looking for cusolverdx")
        find_package(cusolverdx REQUIRED CONFIG HINTS ${cusolverdx_DIR}
            PATHS
                "${cusolverdx_ROOT}"
                "/opt/nvidia/mathdx/25.01/include/cusolverdx"
                "${PROJECT_SOURCE_DIR}/../../cusolverdx"
        )
        message(CHECK_PASS "found, dependencies resolved")
    endif()
endif()

# cuBLASDx
if(NOT mathdx_FOUND)
    if(USE_MATHDX_PACKAGE)
        find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG
            PATHS
                "${PROJECT_SOURCE_DIR}/../.." # example/cublasdx
                "${PROJECT_SOURCE_DIR}/../../.." # include/cublasdx/example
                "opt/nvidia/mathdx/25.01"
        )
    else()
        # Setup cuSolverDx+cuBLASDx examples
        find_package(cublasdx CONFIG HINTS ${cublasdx_DIR} PATHS "/opt/nvidia/mathdx/25.01")
    endif()
endif()
if(cublasdx_FOUND)
    message(STATUS "Examples: cuBLASDx found (${cublasdx_INCLUDE_DIRS}), cuBLASDx+cuSolverDx examples enabled")
else()
    message(STATUS "Examples: cuBLASDx NOT found, cuBLASDx+cuSolverDx examples disabled")
endif()

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Enable testing only for selected architectures
foreach(CUDA_ARCH ${CUSOLVERDX_CUDA_ARCHITECTURES})
    # Extract SM from SM-real/SM-virtual
    string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
    list(GET CUDA_ARCH_LIST 0 ARCH)
    if(${ARCH} STREQUAL "90a")
        set(ARCH 90)
    endif()
    add_compile_definitions(CUSOLVERDX_EXAMPLE_ENABLE_SM_${ARCH})
endforeach()

# ###############################################################
# install_example
# ###############################################################
function(install_example EXAMPLE_TARGET)
    install(TARGETS ${EXAMPLE_TARGET}
        RUNTIME DESTINATION ${CUSOLVERDX_EXAMPLES_BIN_INSTALL_DIR}/
        COMPONENT Examples
    )
endfunction()

# ##############################################################
# Common example objects
# #############################################################
set(common_example_objects_sources common/error_checking.cpp)
set_source_files_properties("${common_example_objects_sources}" PROPERTIES LANGUAGE CUDA)
add_library(common_example_objects STATIC "${common_example_objects_sources}")
target_link_libraries(common_example_objects
    PRIVATE
        $<IF:$<TARGET_EXISTS:mathdx::cusolverdx>,mathdx::cusolverdx,cusolverdx::cusolverdx>
)

target_compile_definitions(common_example_objects PRIVATE CUSOLVERDX_TEST_NO_GTEST)
set_target_properties(common_example_objects
    PROPERTIES
        CUDA_ARCHITECTURES "${CUSOLVERDX_CUDA_ARCHITECTURES}"
)

# ###############################################################
# add_cublasdx_cusolverdx_example
# ###############################################################
function(add_cublasdx_cusolverdx_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            $<IF:$<TARGET_EXISTS:mathdx::cusolverdx>,mathdx::cusolverdx,cusolverdx::cusolverdx>
            $<IF:$<TARGET_EXISTS:mathdx::cublasdx>,mathdx::cublasdx,cublasdx::cublasdx>
            CUDA::cusolver
            common_example_objects
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            INTERPROCEDURAL_OPTIMIZATION ON # This is basically "$<DEVICE_LINK:-dlto>" but it requires CMake 3.25+
            CUDA_ARCHITECTURES "${CUSOLVERDX_CUDA_ARCHITECTURES}"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
            # Required to support std::tuple in device code
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
    )
    if(CMAKE_VERSION VERSION_LESS 3.25)
        # Workaround for LTO support in CMAKE <3.25
        target_link_options(${EXAMPLE_TARGET}
            PRIVATE
                $<DEVICE_LINK:-dlto>
                $<DEVICE_LINK:$<TARGET_FILE:$<IF:$<TARGET_EXISTS:mathdx::cusolverdx>,mathdx::cusolverdx,cusolverdx::cusolverdx>>>
        )
        target_compile_options(${EXAMPLE_TARGET}
            PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-rdc=true>"
        )
        foreach(CUDA_ARCH ${CUSOLVERDX_CUDA_ARCHITECTURES})
            # Extract SM from SM-real/SM-virtual
            string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
            list(GET CUDA_ARCH_LIST 0 ARCH)
            if(${ARCH} STREQUAL "90a")
                set(ARCH 90)
            endif()
            target_compile_options(${EXAMPLE_TARGET}
                PRIVATE
                    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--generate-code arch=compute_${ARCH},code=lto_${ARCH}>"
            )
        endforeach()
    endif()
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# add_cusolverdx_example_link_to_lto_lib
# ###############################################################
function(add_cusolverdx_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            INTERPROCEDURAL_OPTIMIZATION ON # This is basically "$<DEVICE_LINK:-dlto>" but it requires CMake 3.25+
            CUDA_ARCHITECTURES "${CUSOLVERDX_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            $<IF:$<TARGET_EXISTS:mathdx::cusolverdx>,mathdx::cusolverdx,cusolverdx::cusolverdx>
            CUDA::cusolver
            common_example_objects
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )
    if(CMAKE_VERSION VERSION_LESS 3.25)
        # Workaround for LTO support in CMAKE <3.25
        target_link_options(${EXAMPLE_TARGET}
            PRIVATE
                $<DEVICE_LINK:-dlto>
                $<DEVICE_LINK:$<TARGET_FILE:$<IF:$<TARGET_EXISTS:mathdx::cusolverdx>,mathdx::cusolverdx,cusolverdx::cusolverdx>>>
        )
        target_compile_options(${EXAMPLE_TARGET}
            PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-rdc=true>"
        )
        foreach(CUDA_ARCH ${CUSOLVERDX_CUDA_ARCHITECTURES})
            # Extract SM from SM-real/SM-virtual
            string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
            list(GET CUDA_ARCH_LIST 0 ARCH)
            if(${ARCH} STREQUAL "90a")
                set(ARCH 90)
            endif()
            target_compile_options(${EXAMPLE_TARGET}
                PRIVATE
                    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--generate-code arch=compute_${ARCH},code=lto_${ARCH}>"
            )
        endforeach()
    endif()
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
    install_example(${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# add_cusolverdx_nvrtc_example
# ###############################################################
function(add_cusolverdx_nvrtc_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nvrtc
            $<IF:$<TARGET_EXISTS:CUDA::nvJitLink>,CUDA::nvJitLink,${CUDAToolkit_LIBRARY_DIR}/libnvJitLink.so>
    )
    target_compile_definitions(${EXAMPLE_TARGET}
        PRIVATE
            CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
            CUTLASS_INCLUDE_DIR="${cusolverdx_cutlass_INCLUDE_DIR}"
            COMMONDX_INCLUDE_DIR="${cusolverdx_commondx_INCLUDE_DIR}"
            CUSOLVERDX_INCLUDE_DIRS="${cusolverdx_INCLUDE_DIRS}"
            CUSOLVERDX_LIBRARY="$<TARGET_FILE:$<IF:$<TARGET_EXISTS:mathdx::cusolverdx>,mathdx::cusolverdx,cusolverdx::cusolverdx>>"
    )
    # Use fatbin for CTK < 12.1 Update 1 (bug workaround), and for arm/aarch64 (.a file is x86-64 library)
    if((NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)") OR (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.1.105))
        target_compile_definitions(${EXAMPLE_TARGET}
            PRIVATE
                CUSOLVERDX_FATBIN="${cusolverdx_FATBIN}"
        )
    endif()
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
        LABELS "EXAMPLE"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
    install_example(${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# cuSOLVERDx Examples
# ###############################################################
add_custom_target(cusolverdx_examples)

# cuSOLVERDx NVRTC examples
add_cusolverdx_nvrtc_example(cusolverdx_examples "cusolverdx.example.nvrtc_potrs" nvrtc_potrs.cpp)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    # cuSOLVERDx introduction examples
    add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.simple_potrf" simple_potrf.cu)
    add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.potrf_runtime_ld" potrf_runtime_ld.cu)
    add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.posv_batched" posv_batched.cu)
    add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.getrf_wo_pivot" getrf_wo_pivot.cu)
    add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.gesv_batched_wo_pivot" gesv_batched_wo_pivot.cu)

    # cuBLASDx/cuSolverDx examples
    if(cublasdx_FOUND)
        message(STATUS "cuBLASDx was found, examples requiring cuBLASDx enabled")
    else()
        message(STATUS "cuBLASDx was not found, examples requiring cuBLASDx disabled")
    endif()

    if(cublasdx_FOUND)
        # Example for handling larger matrices directly out of global memory
        add_cublasdx_cusolverdx_example(cusolverdx_examples "cusolverdx.example.blocked_potrf" blocked_potrf.cu)
    endif()
endif()
