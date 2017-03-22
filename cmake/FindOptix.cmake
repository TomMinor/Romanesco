# Try to find OptiX headers, libraries (and DLLs only if on Windows)

# Outputs
unset(OPTIX_LIBRARIES CACHE)
unset(OPTIX_INCLUDE_DIRS CACHE)

# Find the Optix directory from CMAKE_PREFIX_PATH
find_path( OPTIX_INCLUDE_DIR optix.h )
if ( OPTIX_INCLUDE_DIR )
  set (OPTIX_ROOT_DIR ${OPTIX_INCLUDE_DIR}/../ )
else()
  message(WARNING "
    OPTIX not found. 
    The Optix folder containg the following should be added to CMAKE_PREFIX_PATH:
    /lib64: containing optix[64_]*.lib or *.so
    /include: containing the header files"
  )
endif()
# Get absolute path of relative path
get_filename_component(OPTIX_ROOT_DIR "${OPTIX_ROOT_DIR}"
                       REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")

# Cmake likes to grab the libs from /lib instead of /lib64, so disable default paths
find_library(OPTIX_LIBRARY NAMES optix optix.1 PATHS "${OPTIX_ROOT_DIR}" NO_DEFAULT_PATH PATH_SUFFIXES lib64)
find_library(OPTIXU_LIBRARY NAMES optixu optixu.1 PATHS "${OPTIX_ROOT_DIR}" NO_DEFAULT_PATH PATH_SUFFIXES lib64)
find_library(OPTIXPRIME_LIBRARY NAMES optix_prime optix_prime.1 PATHS "${OPTIX_ROOT_DIR}" NO_DEFAULT_PATH PATH_SUFFIXES lib64)

message(${OPTIX_LIBRARY})

find_path( OPTIX_INCLUDE_DIR optix.h ${OPTIX_INCLUDE_DIR} )

# Find the DLLs so we can copy them over in a build step if necessary
IF (WIN32)
  find_file(OPTIX_DLL NAMES optix.dll optix.1.dll ${OPTIX_ROOT_DIR} PATH_SUFFIXES bin)
  find_file(OPTIXU_DLL NAMES optixu.dll optixu.1.dll ${OPTIX_ROOT_DIR} PATH_SUFFIXES bin)
  find_file(OPTIXPRIME_DLL NAMES optix_prime.dll optix_prime.1.dll ${OPTIX_ROOT_DIR} PATH_SUFFIXES bin)

  set(OPTIX_DLLS ${OPTIX_DLL} ${OPTIXU_DLL} ${OPTIXPRIME_DLL})
ENDIF()


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set OPTIX_FOUND to TRUE
find_package_handle_standard_args(Optix  DEFAULT_MSG
                                  OPTIX_LIBRARY OPTIXU_LIBRARY OPTIXPRIME_LIBRARY OPTIX_INCLUDE_DIR)

mark_as_advanced( OPTIX_INCLUDE_DIR 
                  OPTIX_LIBRARY OPTIXU_LIBRARY OPTIXPRIME_LIBRARY
                  OPTIX_DLL OPTIXU_DLL OPTIXPRIME_DLL )

set(OPTIX_LIBRARIES ${OPTIX_LIBRARY} ${OPTIXU_LIBRARY} ${OPTIXPRIME_LIBRARY})
set(OPTIX_INCLUDE_DIRS ${OPTIX_INCLUDE_DIR})

