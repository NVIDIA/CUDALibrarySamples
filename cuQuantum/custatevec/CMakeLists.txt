#  
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  - Neither the name(s) of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

cmake_minimum_required(VERSION 3.13.0 FATAL_ERROR)
if(CMAKE_VERSION VERSION_GREATER 3.18)
  cmake_policy(SET CMP0104 OLD)
  set(CUDA_ARCHITECTURES FALSE)
endif()

# ---[ Project specification.
project(custatevec_example LANGUAGES C CXX CUDA)

include(GNUInstallDirs)

# ##########################################
# cuStateVec_example build mode
# ##########################################

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# ##########################################
# cuStateVec_example dependencies
# ##########################################

find_package(CUDA 10.1 REQUIRED)

if (NOT CUSTATEVEC_ROOT)
  set(CUSTATEVEC_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
endif()

include_directories(${CUSTATEVEC_ROOT}/include)
link_directories(${CUSTATEVEC_ROOT}/lib64)

# ##########################################
# cuStateVec_example building flags
# ##########################################

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ##########################################
# cuStateVec_example target
# ##########################################

function(add_custatevec_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_include_directories(${EXAMPLE_TARGET}
        PUBLIC
            ${CUDA_INCLUDE_DIRS}
            ${CUSTATEVEC_ROOT}/include
    )
    target_link_libraries(${EXAMPLE_TARGET}
        PUBLIC custatevec cudart cublas cublasLt
    )
    # Install example
    install(
        TARGETS ${EXAMPLE_TARGET}
        RUNTIME
        DESTINATION ${CUSTATEVEC_EXAMPLE_BINARY_INSTALL_DIR}
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

add_custom_target(custatevec_examples)

add_custatevec_example(custatevec_examples "cuStateVec.example.gate_application"  gate_application.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.exponential_pauli" exponential_pauli.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.expectation"       expectation.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.sampler"           sampler.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.measure_zbasis"    measure_zbasis.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.batch_measure"     batch_measure.cu)
