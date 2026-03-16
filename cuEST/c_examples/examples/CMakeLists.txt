# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

function(
    ADD_EXAMPLE
    EXAMPLE_NAME
    SOURCE_FILE
    )

    add_executable(
        ${EXAMPLE_NAME}
        ${SOURCE_FILE}
        )
    target_include_directories(
        ${EXAMPLE_NAME}
        PRIVATE
            ${CUEST_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/common
        )
    target_link_libraries(
        ${EXAMPLE_NAME}
        PRIVATE
            ${CUEST_LIBRARY}
            CUDA::cudart
            CUDA::cublas
            CUDA::cusolver
            Threads::Threads
            m
        )
endfunction()



set(CMAKE_FOLDER "0_context")
add_subdirectory(0_context)

set(CMAKE_FOLDER "1_basic_data_structures")
add_subdirectory(1_basic_data_structures)

set(CMAKE_FOLDER "2_one_electron_integrals")
add_subdirectory(2_one_electron_integrals)

set(CMAKE_FOLDER "3_density_fitting")
add_subdirectory(3_density_fitting)

set(CMAKE_FOLDER "4_exchange_correlation")
add_subdirectory(4_exchange_correlation)

set(CMAKE_FOLDER "5_effective_core_potentials")
add_subdirectory(5_effective_core_potentials)
