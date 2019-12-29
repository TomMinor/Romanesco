#Todo: Detect 32bit build and react accordingly (no nvrtc, etc)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(OpenGL)

# Find at least a 4.0 version of CUDA.
find_package(CUDA REQUIRED)
include_directories( ${CUDA_INCLUDE_DIRS} )
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})

if(${CUDA_VERSION_MAJOR} VERSION_EQUAL 7} OR ${CUDA_VERSION_MAJOR} VERSION_GREATER 7})
    find_library(CUDA_nvrtc_LIBRARY nvrtc ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64 )
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_nvrtc_LIBRARY})
    message( "Using libnvrtc for runtime compilation. (${CUDA_nvrtc_LIBRARY})" )
    
    add_definitions(-DNVRTC_AVAILABLE)
else()
    message( "CUDA version is ${CUDA_VERSION_MAJOR}, at least 7.0 is required for libnvrtc. Using system nvcc for runtime compilation." )
endif()

find_package(Optix REQUIRED)
if(OPTIX_FOUND)
    Message(STATUS "--> using package OptiX")
    add_definitions(-DUSEOPTIX)
    include_directories(${OPTIX_INCLUDE_DIRS})

    configure_file(optixresources.h.in optixresources.h)
    include_directories( ${CMAKE_BINARY_DIR} )

    # The OptiX math libraries require this be set so windows.h includes min/max properly
    if(WIN32)
        add_definitions(-DNOMINMAX)
    endif()
endif()

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
find_package(Qt5Widgets REQUIRED core opengl widgets gui )
include_directories( ${Qt5Widgets_INCLUDE_DIRS} )
include_directories( ${Qt5Widgets_LIBRARIES} )
# We need to add -DQT_WIDGETS_LIB when using QtWidgets in Qt 5.
add_definitions(${Qt5Widgets_DEFINITIONS})

find_package(Boost)
if(Boost_FOUND)
    include_directories( ${Boost_INCLUDE_DIRS} )
    link_directories( ${Boost_LIBRARY_DIRS} )
endif(Boost_FOUND)

# ToDo modify this so it finds OpenEXR in standard Linux paths (/usr/lib64)
find_package(OpenEXR)
if(OpenEXR_FOUND)
    include_directories(${OPENEXR_INCLUDE_DIRS} )
    link_directories( ${OPENEXR_LIBRARY_DIR} )
        add_definitions(-DOPENEXR_AVAILABLE)

  find_package(ZLIB)
  get_filename_component(ZLIB_ROOT_DIR "${ZLIB_LIBRARIES}" DIRECTORY)
  get_filename_component(ZLIB_ROOT_DIR "${ZLIB_ROOT_DIR}/.." ABSOLUTE)
endif(OpenEXR_FOUND)

include_directories( ${CMAKE_SOURCE_DIR}/dependencies/glm )

add_definitions(-DDEBUG_GL)

file(GLOB CPP_CORE
    ${CMAKE_CURRENT_LIST_DIR}/Core/*
)

file(GLOB CPP_GUI
    ${CMAKE_CURRENT_LIST_DIR}/GUI/*
)

file(GLOB CPP_SDFOPS
    ${CMAKE_CURRENT_LIST_DIR}/SDFOps/*
)

file(GLOB CPP_OPTIX
    ${CMAKE_CURRENT_LIST_DIR}/Core/*
)

file(GLOB CPP_OPTIX
    ${CMAKE_CURRENT_LIST_DIR}/Optix/*
)


file(GLOB CPP_SOURCES 
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CPP_CORE} ${CPP_GUI} ${CPP_SDFOPS} ${CPP_OPTIX}
)


file(GLOB VERTEX_SOURCES ${CMAKE_CURRENT_LIST_DIR}/shaders/*.frag)
file(GLOB FRAGMENT_SOURCES ${CMAKE_CURRENT_LIST_DIR}/shaders/*.vert)

file(GLOB KERNEL_HEADERS ${CMAKE_CURRENT_LIST_DIR}/kernel/*.h)
file(GLOB KERNEL_SOURCES ${CMAKE_CURRENT_LIST_DIR}/kernel/*.cu)


file(GLOB HEADERS 
    ${HEADERS_CORE}
    ${HEADERS_GUI}
    ${HEADERS_SDFOPS}
    ${CMAKE_CURRENT_LIST_DIR}/kernel/*.h
  )

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/Core
    ${CMAKE_CURRENT_LIST_DIR}/GUI
    ${CMAKE_CURRENT_LIST_DIR}/GUI/nodes
    # ${CMAKE_CURRENT_LIST_DIR}/GUI/nodegraph
    ${CMAKE_CURRENT_LIST_DIR}/SDFOps
    ${CMAKE_CURRENT_LIST_DIR}/SDFOps/DistDeformer
    ${CMAKE_CURRENT_LIST_DIR}/SDFOps/DistOp
    ${CMAKE_CURRENT_LIST_DIR}/SDFOps/DomainOp
    ${CMAKE_CURRENT_LIST_DIR}/SDFOps/DomainDeformer
    ${CMAKE_CURRENT_LIST_DIR}/SDFOps/Primitive
    ${CMAKE_CURRENT_LIST_DIR}/sutil

    ${CMAKE_CURRENT_LIST_DIR}/kernel
  )


CUDA_INCLUDE_DIRECTORIES(${OPTIX_INCLUDE_DIRS})
#CUDA_INCLUDE_DIRECTORIES("${CMAKE_CURRENT_LIST_DIR}/dependencies/glm")

CUDA_COMPILE_PTX(
    CUDA_PTX_FILES
    ${KERNEL_SOURCES}
    OPTIONS --use_fast_math
)

set(CUDA_CU_FILES)
foreach(_file ${CUDA_PTX_FILES})
    # Fix the weird generated name from findCUDA
    STRING(REGEX REPLACE "cuda_compile_ptx_[0-9]+_+generated_" "ptx/" cuda_file ${_file})

    set(CUDA_CU_FILES ${CUDA_CU_FILES} ${cuda_file})
endforeach()

add_custom_target(ptx ALL
    DEPENDS ${CUDA_PTX_FILES} ${CUDA_CU_FILES}
    SOURCES ${KERNEL_SOURCES}
)

# set(CUDA_CU_FILES)
foreach(_file ${CUDA_PTX_FILES})
    # Fix the weird generated name from findCUDA
    STRING(REGEX REPLACE "cuda_compile_ptx_[0-9]+_+generated_" "ptx/" cuda_file ${_file})

    get_filename_component(PTXDIR ${cuda_file} DIRECTORY)

    add_custom_command(TARGET ptx POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PTXDIR}
        COMMAND ${CMAKE_COMMAND} -E rename ${_file} ${cuda_file} DEPENDS ${_file}
        COMMENT "Moving ${_file} -> ${cuda_file}\n" VERBATIM 
    )
endforeach()


add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${HEADERS} ${VERTEX_SOURCES} ${FRAGMENT_SOURCES})

source_group("Shaders\\Vertex"      FILES ${VERTEX_SOURCES})
source_group("Shaders\\Fragment"    FILES ${FRAGMENT_SOURCES})
source_group("Kernels"              FILES ${KERNEL_SOURCES})
source_group("Kernels\\include"     FILES ${KERNEL_HEADERS})

source_group("Core"   FILES ${CPP_CORE} )
source_group("Optix"   FILES ${CPP_OPTIX} )
source_group("GUI"    FILES ${CPP_GUI} )
source_group("SDFOps" FILES ${CPP_SDFOPS} )

# target_link_libraries(romanesco SDL2 opengl32 OpenCL::OpenCL)
target_link_libraries(${PROJECT_NAME}
    ${Qt5Widgets_LIBRARIES}
    ${OPENGL_gl_LIBRARY}
    ${CUDA_LIBRARIES}
    ${OPTIX_LIBRARIES}
    ${OPENEXR_LIBRARIES}
    ${Boost_LIBRARIES}
)

# Windows specific build steps
if(WIN32)
    # Run winddeployqt if it can be found
    find_program(WINDEPLOYQT_EXECUTABLE NAMES windeployqt HINTS ${QTDIR} ENV QTDIR PATH_SUFFIXES bin)
    message( "Configuring Qt deployment with ${WINDEPLOYQT_EXECUTABLE}" )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${WINDEPLOYQT_EXECUTABLE} $<TARGET_FILE:${PROJECT_NAME}>)

    # Install Optix DLLs
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${OPTIX_ROOT_DIR}/bin64"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)

    # Install OpenEXR/IlmBase/ZLib DLLs
    if(OpenEXR_FOUND)
      message(${OPENEXR_ROOT_DIR})
      add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${OPENEXR_ROOT_DIR}/lib"
          $<TARGET_FILE_DIR:${PROJECT_NAME}>)

      add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${ZLIB_ROOT_DIR}/bin"
          $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    endif(OpenEXR_FOUND)

    # ToDo this doesn't automatically copy every build, but good enough if you're not doing kernel development
    add_custom_target(copy-runtime-files ALL
        COMMAND cmake -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/styles $<TARGET_FILE_DIR:${PROJECT_NAME}>/styles
        COMMAND cmake -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/shaders $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
        COMMAND cmake -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/kernel $<TARGET_FILE_DIR:${PROJECT_NAME}>/kernel
        COMMAND cmake -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/images $<TARGET_FILE_DIR:${PROJECT_NAME}>/images
        COMMAND cmake -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/scenes $<TARGET_FILE_DIR:${PROJECT_NAME}>/scenes
    )
else()
    # Copy resources to build folder (styles/, shaders/, kernel/, images/, scenes/)
    # ToDo this doesn't automatically copy every build, but good enough if you're not doing kernel development
    add_custom_target(copy-runtime-files ALL
        COMMAND cmake -E copy_directory "${CMAKE_CURRENT_LIST_DIR}/styles" "${CMAKE_BINARY_DIR}/styles"
        COMMAND cmake -E copy_directory "${CMAKE_CURRENT_LIST_DIR}/shaders" "${CMAKE_BINARY_DIR}/shaders"
        COMMAND cmake -E copy_directory "${CMAKE_CURRENT_LIST_DIR}/kernel" "${CMAKE_BINARY_DIR}/kernel"
        COMMAND cmake -E copy_directory "${CMAKE_CURRENT_LIST_DIR}/images" "${CMAKE_BINARY_DIR}/images"
        COMMAND cmake -E copy_directory "${CMAKE_CURRENT_LIST_DIR}/scenes" "${CMAKE_BINARY_DIR}/scenes"
    )
    # install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR}/bin)
endif()
