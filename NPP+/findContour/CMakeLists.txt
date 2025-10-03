# SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CMAKE_MINIMUM_REQUIRED(VERSION 3.10.0 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

# ---[ Project specIFication.
SET(PROJECT_NAME findContour)

PROJECT(${PROJECT_NAME} LANGUAGES CUDA CXX)


find_package(CUDAToolkit REQUIRED)

#export NPP_PLUS_PATH=/usr/lib/x86_64-linux-gnu/libnpp_plus/12
#export NPP_PLUS_INCLUDE_DIR=/usr/include/libnpp_plus/12
#cmake .. -DNPP_PLUS_PATH=/usr/lib/x86_64-linux-gnu 
add_definitions(-DNPP_PLUS_ON)
find_library(NPP_PLUS_IF_LIBRARY
     NAMES npp_plus_if
     PATHS ${NPP_PLUS_PATH}/12)

if(NOT NPP_PLUS_IF_LIBRARY)
  message(FATAL_ERROR, " NPP_PLUS_IF_LIBRARY library not found. Make sure the library is installed. If using the tar.xz/zip packages set -DNPP_PLUS_PATH")
else()
  message(INFO, " NPP_PLUS_IF_LIBRARY library found here: ${NPP_PLUS_IF_LIBRARY}")
endif()

find_library(NPP_PLUS_C_LIBRARY
     NAMES npp_plus_c
     PATHS ${NPP_PLUS_PATH}/12)

if(NOT NPP_PLUS_C_LIBRARY)
  message(FATAL_ERROR, " NPP_PLUS_IF_LIBRARY library not found. Make sure the library is installed. If using the tar.xz/zip packages set -DNPP_PLUS_PATH")
else()
  message(INFO, " NPP_PLUS_C_LIBRARY library found here: ${NPP_PLUS_C_LIBRARY}")
endif()

find_library(NPP_PLUS_ISU_LIBRARY
     NAMES npp_plus_isu
     PATHS ${NPP_PLUS_PATH}/12)

if(NOT NPP_PLUS_ISU_LIBRARY)
  message(FATAL_ERROR, " NPP_PLUS_IF_LIBRARY library not found. Make sure the library is installed. If using the tar.xz/zip packages set -DNPP_PLUS_PATH")
else()
  message(INFO, " NPP_PLUS_ISU_LIBRARY library found here: ${NPP_PLUS_ISU_LIBRARY}")
endif()

find_path(NPP_HEADER_PATH
    NAMES npp.h
    PATHS /usr/include/libnpp_plus/12
)

if (NPP_HEADER_PATH)
    message(STATUS "Found npp.h in: ${NPP_HEADER_PATH}")
else()
    message(FATAL_ERROR "Could not find npp.h")
endif()

find_path(NPP_PLUS_HEADER_PATH
    NAMES nppPlus.h
    PATHS /usr/local/include /usr/include /usr/include/nppPlus
)

if (NPP_PLUS_HEADER_PATH)
    message(STATUS "Found nppPlus.h in: ${NPP_PLUS_HEADER_PATH}")
else()
    message(FATAL_ERROR "Could not find nppPlus.h")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include_directories(
 ${NPP_HEADER_PATH}
 ${NPP_PLUS_HEADER_PATH}
 SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 

link_directories(${CMAKE_CUDA_INCLUDE_DIRS})

if (UNIX)
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(NPP_PLUS_IF_LIBRARY npp_plus_if ${NPP_PLUS_LIB})
    find_library(NPP_PLUS_C_LIBRARY npp_plus_c ${NPP_PLUS_LIB})
    find_library(NPP_PLUS_ISU_LIBRARY npp_plus_isu ${NPP_PLUS_LIB})
    find_library(CULIBOS culibos ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif (UNIX)


if (MSVC OR WIN32 OR MSYS)
    MESSAGE( STATUS "CMAKE_CUDA_COMPILER:         " ${CMAKE_CUDA_COMPILER} )
    set(CUDA_LIBRARY_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64")
    set(CUDART_LIBRARY ${CUDA_LIBRARY_PATH}/cudart.lib)
    set(NPP_PLUS_IF_LIBRARY ${NPP_PLUS_LIB}/npp_plus_if.lib)
    set(NPP_PLUS_C_LIBRARY ${NPP_PLUS_LIB}/npp_plus_c.lib)
    set(NPP_PLUS_ISU_LIBRARY ${NPP_PLUS_LIB}/npp_plus_isu.lib)
    set(CULIBOS ${CUDA_LIBRARY_PATH}/culibos.lib)
endif()

PROJECT(${PROJECT_NAME} C CXX CUDA)

# ---[ Use the default installation path if not set.
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# ---[ Build type
IF(NOT CMAKE_BUILD_TYPE) 
    SET(CMAKE_BUILD_TYPE Develop)
ENDIF(NOT CMAKE_BUILD_TYPE)

SET(EXAMPLES_DESCRIPTOR_SOURCES "findContourNPPPlus.cpp")
ADD_EXECUTABLE(${PROJECT_NAME} ${EXAMPLES_DESCRIPTOR_SOURCES})
SET_SOURCE_FILES_PROPERTIES(${EXAMPLES_DESCRIPTOR_SOURCES} PROPERTIES LANGUAGE CUDA)
TARGET_COMPILE_FEATURES(${PROJECT_NAME} PUBLIC cxx_std_11)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE ${NPP_HEADER_PATH})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE ${NPP_PLUS_HEADER_PATH})
if (UNIX)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC ${CUDART_LIBRARY} ${NPP_PLUS_IF_LIBRARY} ${NPP_PLUS_C_LIBRARY} ${NPP_PLUS_ISU_LIBRARY} ${CULIBOS})
endif (UNIX)

if(MSVC OR WIN32 OR MSYS)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC ${CUDART_LIBRARY} ${NPP_PLUS_IF_LIBRARY} ${NPP_PLUS_C_LIBRARY} ${NPP_PLUS_ISU_LIBRARY} ${CULIBOS})
endif(MSVC OR WIN32 OR MSYS)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET ${PROJECT_NAME} 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif(APPLE)

INSTALL(TARGETS ${PROJECT_NAME} DESTINATION bin)
