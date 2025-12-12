# FindNCCL.cmake
#
# Find the NVIDIA Collective Communications Library (NCCL)
#
# This module defines the following variables:
#   NCCL_FOUND          - True if NCCL is found
#   NCCL_INCLUDE_DIR    - Include directory for NCCL headers
#   NCCL_LIBRARY        - NCCL library path
#   NCCL_VERSION        - Version of NCCL (if available)
#
# This module also defines the following imported target:
#   NCCL::NCCL          - The NCCL library target
#
# The following variables can be used to guide the search:
#   NCCL_HOME           - Root directory of NCCL installation
#   NCCL_ROOT           - Alternative to NCCL_HOME
#   NCCL_INCLUDE_DIR    - Directory containing nccl.h
#   NCCL_LIBRARY        - Path to the NCCL library
#
# Environment variables:
#   NCCL_HOME           - Root directory of NCCL installation

# Use NCCL_ROOT if NCCL_HOME is not set
if(NOT NCCL_HOME AND DEFINED ENV{NCCL_HOME})
    set(NCCL_HOME $ENV{NCCL_HOME})
endif()

if(NOT NCCL_HOME AND NCCL_ROOT)
    set(NCCL_HOME ${NCCL_ROOT})
endif()

if(NOT NCCL_HOME AND DEFINED ENV{NCCL_ROOT})
    set(NCCL_HOME $ENV{NCCL_ROOT})
endif()

# Set search paths
set(_NCCL_SEARCH_PATHS)
if(NCCL_HOME)
    list(APPEND _NCCL_SEARCH_PATHS ${NCCL_HOME})
endif()

# Add standard system paths
list(APPEND _NCCL_SEARCH_PATHS
    /usr/local/nccl
    /usr/local
    /usr
    /opt/nccl
    /opt/local
)

# Find the header file
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATHS ${_NCCL_SEARCH_PATHS}
    PATH_SUFFIXES include
    DOC "Path to NCCL include directory"
)

# Find the library
find_library(NCCL_LIBRARY
    NAMES nccl
    PATHS ${_NCCL_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
    DOC "Path to NCCL library"
)

# Extract version information if header is found
if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
    file(READ "${NCCL_INCLUDE_DIR}/nccl.h" _NCCL_HEADER_CONTENT)
    
    # Extract major version
    string(REGEX MATCH "#define NCCL_MAJOR[ \t]+([0-9]+)" _NCCL_MAJOR_MATCH "${_NCCL_HEADER_CONTENT}")
    if(_NCCL_MAJOR_MATCH)
        set(NCCL_VERSION_MAJOR ${CMAKE_MATCH_1})
    endif()
    
    # Extract minor version
    string(REGEX MATCH "#define NCCL_MINOR[ \t]+([0-9]+)" _NCCL_MINOR_MATCH "${_NCCL_HEADER_CONTENT}")
    if(_NCCL_MINOR_MATCH)
        set(NCCL_VERSION_MINOR ${CMAKE_MATCH_1})
    endif()
    
    # Extract patch version
    string(REGEX MATCH "#define NCCL_PATCHLEVEL[ \t]+([0-9]+)" _NCCL_PATCH_MATCH "${_NCCL_HEADER_CONTENT}")
    if(_NCCL_PATCH_MATCH)
        set(NCCL_VERSION_PATCH ${CMAKE_MATCH_1})
    endif()
    
    # Construct full version string
    if(NCCL_VERSION_MAJOR AND NCCL_VERSION_MINOR)
        if(NCCL_VERSION_PATCH)
            set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
        else()
            set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}")
        endif()
    endif()
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
    REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
    VERSION_VAR NCCL_VERSION
    HANDLE_COMPONENTS
)

# Create imported target
if(NCCL_FOUND AND NOT TARGET NCCL::NCCL)
    add_library(NCCL::NCCL UNKNOWN IMPORTED)
    set_target_properties(NCCL::NCCL PROPERTIES
        IMPORTED_LOCATION "${NCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
    )
    
    # Add CUDA dependency if available
    if(TARGET CUDA::cudart)
        set_target_properties(NCCL::NCCL PROPERTIES
            INTERFACE_LINK_LIBRARIES "CUDA::cudart"
        )
    endif()
endif()

# Mark variables as advanced
mark_as_advanced(
    NCCL_INCLUDE_DIR
    NCCL_LIBRARY
)

# Provide status messages
if(NCCL_FOUND)
    if(NOT NCCL_FIND_QUIETLY)
        message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
        if(NCCL_VERSION)
            message(STATUS "NCCL version: ${NCCL_VERSION}")
        endif()
        message(STATUS "NCCL include dir: ${NCCL_INCLUDE_DIR}")
    endif()
else()
    if(NCCL_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find NCCL library. Please set NCCL_HOME environment variable or install NCCL in a standard location.")
    elseif(NOT NCCL_FIND_QUIETLY)
        message(STATUS "NCCL not found. Set NCCL_HOME environment variable or install NCCL in a standard location.")
    endif()
endif()
