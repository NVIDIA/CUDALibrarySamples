# Function to find headers and libraries with fallback logic
function(find_dependency_with_fallback
  DEP_NAME # Name for messages (e.g., "MPI", "NCCL")
  ENV_VAR_NAME # Environment variable name (e.g., "MPI_HOME", "NCCL_HOME")
  HEADER_NAME # Header file name (e.g., "mpi.h", "nccl.h")
  LIBRARY_NAMES # Library names to search for (e.g., "mpi;mpi_cxx")
  RESULT_INCLUDE_VAR # Output variable for include directory
  RESULT_LIB_VARS # Output variable names for libraries (list)
)
  set(ENV_HOME $ENV{${ENV_VAR_NAME}})

  # Initialize local tracking variables
  set(found_include_dir "")
  set(found_libraries "")

  # Try environment variable path first
  if(DEFINED ENV{${ENV_VAR_NAME}} AND NOT "$ENV{${ENV_VAR_NAME}}" STREQUAL "")
    # Check header
    if(EXISTS "${ENV_HOME}/include/${HEADER_NAME}")
      set(found_include_dir "${ENV_HOME}/include")
      set(${RESULT_INCLUDE_VAR} "${ENV_HOME}/include" PARENT_SCOPE)
      message(STATUS "Found ${DEP_NAME} header: ${ENV_HOME}/include/${HEADER_NAME}")
    endif()

    # Check libraries
    list(LENGTH RESULT_LIB_VARS num_lib_vars)
    list(LENGTH LIBRARY_NAMES num_lib_names)
    math(EXPR max_libs "${num_lib_vars} - 1")

    foreach(i RANGE ${max_libs})
      if(i LESS ${num_lib_names})
        list(GET LIBRARY_NAMES ${i} lib_name)
        list(GET RESULT_LIB_VARS ${i} lib_var)

        find_library(${lib_var}_TEMP NAMES ${lib_name} PATHS ${ENV_HOME}/lib ${ENV_HOME}/lib64 NO_DEFAULT_PATH)

        if(${lib_var}_TEMP)
          set(${lib_var} ${${lib_var}_TEMP} PARENT_SCOPE)
          list(APPEND found_libraries ${lib_var})
          message(STATUS "Found ${DEP_NAME} library: ${${lib_var}_TEMP}")
        endif()
      endif()
    endforeach()
  endif()

  # Fallback to system directories if not found
  if(NOT found_include_dir)
    find_path(${RESULT_INCLUDE_VAR}_TEMP ${HEADER_NAME}
      PATHS /usr/include /usr/local/include /usr/include/${DEP_NAME} /usr/local/include/${DEP_NAME}
      PATH_SUFFIXES ${DEP_NAME}
    )

    if(${RESULT_INCLUDE_VAR}_TEMP)
      set(found_include_dir ${${RESULT_INCLUDE_VAR}_TEMP})
      set(${RESULT_INCLUDE_VAR} ${${RESULT_INCLUDE_VAR}_TEMP} PARENT_SCOPE)
      message(STATUS "Found system ${DEP_NAME} header: ${${RESULT_INCLUDE_VAR}_TEMP}/${HEADER_NAME}")
    elseif(EXISTS "/usr/include/${HEADER_NAME}")
      set(found_include_dir "/usr/include")
      set(${RESULT_INCLUDE_VAR} "/usr/include" PARENT_SCOPE)
      message(STATUS "Found system ${DEP_NAME} header: /usr/include/${HEADER_NAME}")
    endif()
  endif()

  # Search for libraries in system directories
  list(LENGTH RESULT_LIB_VARS num_lib_vars)
  list(LENGTH LIBRARY_NAMES num_lib_names)
  math(EXPR max_libs "${num_lib_vars} - 1")

  foreach(i RANGE ${max_libs})
    if(i LESS ${num_lib_names})
      list(GET LIBRARY_NAMES ${i} lib_name)
      list(GET RESULT_LIB_VARS ${i} lib_var)

      # Check if this library was already found
      list(FIND found_libraries ${lib_var} lib_found_index)

      if(lib_found_index EQUAL -1)
        find_library(${lib_var}_TEMP NAMES ${lib_name}
          PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib64 /lib/x86_64-linux-gnu
        )

        if(${lib_var}_TEMP)
          set(${lib_var} ${${lib_var}_TEMP} PARENT_SCOPE)
          list(APPEND found_libraries ${lib_var})
          message(STATUS "Found system ${DEP_NAME} library: ${${lib_var}_TEMP}")
        endif()
      endif()
    endif()
  endforeach()

  # Final validation
  if(NOT found_include_dir)
    message(FATAL_ERROR "${DEP_NAME} header (${HEADER_NAME}) not found. Please set ${ENV_VAR_NAME} environment variable or install ${DEP_NAME} in system directories.")
  endif()

  # Validate that at least one library was found
  if(NOT found_libraries)
    message(FATAL_ERROR "${DEP_NAME} library not found. Please set ${ENV_VAR_NAME} environment variable or install ${DEP_NAME} in system directories.")
  endif()
endfunction()
