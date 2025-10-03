# SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

function(add_curand_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCES})
set_property(TARGET ${EXAMPLE_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
target_include_directories(${EXAMPLE_NAME}
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${EXAMPLE_NAME}
    PUBLIC
		CUDA::cudart
		CUDA::curand
)
set_target_properties(${EXAMPLE_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Install example
install(
    TARGETS ${EXAMPLE_NAME}
    RUNTIME
    DESTINATION ${curand_examples_BINARY_INSTALL_DIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
)

add_dependencies(${GROUP_TARGET} ${EXAMPLE_NAME})
endfunction()

add_custom_target(curand_examples)