# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Installable cuest python_api examples — mirrors c_api/examples in shape.
# Each per-example CMakeLists.txt calls add_python_example() to generate a
# tiny Makefile that drives `python run.py <args>`.

# Maps input tags to filenames under examples/inputs/ (shared with c_api).
set(CUEST_PYTHON_EXAMPLE_INPUT_xyz       "h2o.xyz")
set(CUEST_PYTHON_EXAMPLE_INPUT_xyz_heavy "ch2i2.xyz")
set(CUEST_PYTHON_EXAMPLE_INPUT_gbs       "def2-svp.gbs")
set(CUEST_PYTHON_EXAMPLE_INPUT_aux       "def2-universal-jkfit.gbs")
set(CUEST_PYTHON_EXAMPLE_INPUT_ecp       "def2-svp-ecp.gbs")

function(ADD_PYTHON_EXAMPLE EXAMPLE_NAME SOURCE_FILE)
    # Keyword args:
    #   INPUTS    <tag> ...   input tags (xyz, xyz_heavy, gbs, aux, ecp)
    #   MIN_GPUS  <N>         skip the run target if fewer GPUs are visible
    cmake_parse_arguments(ADD_PYE "" "MIN_GPUS" "INPUTS" ${ARGN})
    if (NOT DEFINED ADD_PYE_MIN_GPUS OR ADD_PYE_MIN_GPUS STREQUAL "")
        set(ADD_PYE_MIN_GPUS 1)
    endif()

    if (INSTALL_TESTS)
        set(EXAMPLE_ARGS "")
        foreach(tag IN LISTS ADD_PYE_INPUTS)
            if (NOT DEFINED CUEST_PYTHON_EXAMPLE_INPUT_${tag})
                message(FATAL_ERROR
                    "add_python_example(${EXAMPLE_NAME}): unknown input tag '${tag}' "
                    "(expected one of: xyz, xyz_heavy, gbs, aux, ecp)")
            endif()
            if (EXAMPLE_ARGS STREQUAL "")
                set(EXAMPLE_ARGS "\$(INPUTS_DIR)/${CUEST_PYTHON_EXAMPLE_INPUT_${tag}}")
            else()
                set(EXAMPLE_ARGS "${EXAMPLE_ARGS} \$(INPUTS_DIR)/${CUEST_PYTHON_EXAMPLE_INPUT_${tag}}")
            endif()
        endforeach()
        set(EXAMPLE_MIN_GPUS "${ADD_PYE_MIN_GPUS}")

        file(RELATIVE_PATH REL_DIR
             "${CMAKE_SOURCE_DIR}/python_api/examples"
             "${CMAKE_CURRENT_SOURCE_DIR}")
        set(INSTALL_DIR "examples/python_api/${REL_DIR}")

        install(FILES ${SOURCE_FILE} DESTINATION ${INSTALL_DIR})

        set(GENERATED_MAKEFILE
            "${CMAKE_CURRENT_BINARY_DIR}/Makefile.${EXAMPLE_NAME}.install")
        configure_file(
            "${CMAKE_SOURCE_DIR}/python_api/examples/common/Makefile.in"
            "${GENERATED_MAKEFILE}"
            @ONLY
            )
        install(
            FILES "${GENERATED_MAKEFILE}"
            DESTINATION ${INSTALL_DIR}
            RENAME Makefile
            )
    endif()
endfunction()

add_subdirectory(0_context)
add_subdirectory(1_basic_data_structures)
add_subdirectory(2_one_electron_integrals)
add_subdirectory(3_density_fitting)
add_subdirectory(4_exchange_correlation)
add_subdirectory(5_effective_core_potentials)
add_subdirectory(6_pcm)

if (INSTALL_TESTS)
    install(
        DIRECTORY common/
        DESTINATION examples/python_api/common
        PATTERN "Makefile.in" EXCLUDE
        )
    install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/python_api/helpers/
        DESTINATION examples/python_api/helpers
        PATTERN "__pycache__" EXCLUDE
        )
    install(
        PROGRAMS run.sh
        DESTINATION examples/python_api
        )
    # Inputs live under examples/inputs/ — shared with c_api. Install them
    # here too so the python tree works even when BUILD_C_API_TESTS is off.
    install(
        FILES
        ${CMAKE_SOURCE_DIR}/c_api/tests/data/geometry/h2o.xyz
        ${CMAKE_SOURCE_DIR}/c_api/tests/data/geometry/ch2i2.xyz
        ${CMAKE_SOURCE_DIR}/c_api/tests/data/basis_set/def2-svp.gbs
        ${CMAKE_SOURCE_DIR}/c_api/tests/data/basis_set/def2-universal-jkfit.gbs
        ${CMAKE_SOURCE_DIR}/c_api/tests/data/basis_set/def2-svp-ecp.gbs
        DESTINATION
        examples/inputs
        )
endif()
