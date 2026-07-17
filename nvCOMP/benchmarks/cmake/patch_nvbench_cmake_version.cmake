# TODO: Remove once we can use CMake > 3.30.4 Relax cmake_minimum_required() in
# NVBench sources so it builds with the running CMake version.  The 3.30.4 floor
# is imposed by rapids-cmake, which we pre-populate separately, so it is safe to
# lower.
#
# Only patch when the host CMake is actually below the required version; on
# systems that already satisfy the requirement (e.g. Windows CI) we keep the
# original version to preserve correct policy behaviour.

set(required_version "3.30.4")

if(NVCOMP_CMAKE_VERSION VERSION_GREATER_EQUAL "${required_version}")
  return()
endif()

# Use the larger of the running version and nvcomp's own minimum so we never
# produce a cmake_minimum_required below what the project itself requires.
if(NVCOMP_CMAKE_VERSION VERSION_GREATER "${NVCOMP_CMAKE_MIN_VERSION}")
  set(target_version "${NVCOMP_CMAKE_VERSION}")
else()
  set(target_version "${NVCOMP_CMAKE_MIN_VERSION}")
endif()

foreach(rel_path "CMakeLists.txt" "cmake/RAPIDS.cmake")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${rel_path}")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${rel_path}" contents)
    string(
      REGEX
      REPLACE
        "cmake_minimum_required\\((VERSION )?[0-9]+\\.[0-9]+(\\.[0-9]+)?([^\n]*)"
        "cmake_minimum_required(VERSION ${target_version}\\3"
        contents
        "${contents}")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/${rel_path}" "${contents}")
  endif()
endforeach()

# Also clamp cmake_policy(VERSION X.Y...) calls that exceed the running version.
# These appear in rapids-cmake modules and prevent older CMake from loading
# them.
file(GLOB_RECURSE all_cmake_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cmake")
foreach(cmake_file ${all_cmake_files})
  file(READ "${cmake_file}" contents)
  string(REGEX MATCHALL "cmake_policy\\(VERSION [0-9]+\\.[0-9]+[^\n)]*\\)"
               policy_calls "${contents}")
  if(NOT policy_calls)
    continue()
  endif()
  set(modified FALSE)
  foreach(call ${policy_calls})
    string(REGEX MATCH "[0-9]+\\.[0-9]+" policy_ver "${call}")
    if(policy_ver VERSION_GREATER "${NVCOMP_CMAKE_VERSION}")
      string(REPLACE "${call}" "cmake_policy(VERSION ${target_version})"
                     contents "${contents}")
      set(modified TRUE)
    endif()
  endforeach()
  if(modified)
    file(WRITE "${cmake_file}" "${contents}")
  endif()
endforeach()
