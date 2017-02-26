cmake_minimum_required(VERSION 3.7)

project(romanesco)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Find at least a 4.0 version of CUDA.
find_package(CUDA 4.0 REQUIRED)
find_package(OPTIX REQUIRED)

# set(SDL2_ROOT ${CMAKE_SOURCE_DIR}/thirdparty/SDL2-2.0.5)
# set(GLM_ROOT ${CMAKE_SOURCE_DIR}/thirdparty/glm)

include_directories(include)
# include_directories(${SDL2_ROOT}/include)
# include_directories(${GLM_ROOT})

# link_directories(${SDL2_ROOT}/lib/x86)

# file(GLOB_RECURSE CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)
# file(GLOB_RECURSE C_SOURCES ${CMAKE_SOURCE_DIR}/src/*.c)
# file(GLOB_RECURSE HEADERS ${CMAKE_SOURCE_DIR}/include/*.h)
# file(GLOB_RECURSE VERTEX_SOURCES ${CMAKE_SOURCE_DIR}/shaders/*.frag)
# file(GLOB_RECURSE FRAGMENT_SOURCES ${CMAKE_SOURCE_DIR}/shaders/*.vert)
# file(GLOB_RECURSE KERNEL_SOURCES ${CMAKE_SOURCE_DIR}/kernels/*.cl)

# add_executable(romanesco ${CPP_SOURCES} ${C_SOURCES} ${HEADERS} ${VERTEX_SOURCES} ${FRAGMENT_SOURCES} ${KERNEL_SOURCES})

# source_group("Shaders"	FILES ${VERTEX_SOURCES})
# source_group("Shaders"	FILES ${FRAGMENT_SOURCES})
# source_group("Kernels"	FILES ${KERNEL_SOURCES})

# target_link_libraries(romanesco SDL2 opengl32 OpenCL::OpenCL)