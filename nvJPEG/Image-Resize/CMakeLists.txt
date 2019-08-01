# 
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 
# ---[ Check cmake version.
CMAKE_MINIMUM_REQUIRED(VERSION 3.10.0 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

# ---[ Project specIFication.
SET(PROJECT_NAME imageResize)
PROJECT(${PROJECT_NAME} C CXX CUDA)

# ---[ Use the default installation path if not set.
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# ---[ Build type
IF(NOT CMAKE_BUILD_TYPE) 
    SET(CMAKE_BUILD_TYPE Develop)
ENDIF(NOT CMAKE_BUILD_TYPE)

SET(EXAMPLES_DESCRIPTOR_SOURCES "imageResize.cu")

ADD_EXECUTABLE(${PROJECT_NAME} ${EXAMPLES_DESCRIPTOR_SOURCES})
TARGET_COMPILE_FEATURES(${PROJECT_NAME} PUBLIC cxx_std_11)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC cudart nvjpeg nppig_static nppc_static culibos)

INSTALL(TARGETS ${PROJECT_NAME} DESTINATION bin)

