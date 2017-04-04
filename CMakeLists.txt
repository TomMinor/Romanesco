cmake_minimum_required(VERSION 2.8.11)

project(romanesco)

#Todo: Detect 32bit build and react accordingly (no nvrtc)

# Force C++11
if(UNIX)
  SET(CMAKE_CXX_FLAGS "-std=c++0x")
endif()

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

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
endif()



find_package(Qt5Widgets REQUIRED core opengl widgets gui )
include_directories( ${Qt5Widgets_INCLUDE_DIRS} )
include_directories( ${Qt5Widgets_LIBRARIES} )

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
endif(OpenEXR_FOUND)

# find_package(glm REQUIRED)
include_directories( "${CMAKE_CURRENT_LIST_DIR}/thirdparty/glm" )

# We need add -DQT_WIDGETS_LIB when using QtWidgets in Qt 5.
add_definitions(${Qt5Widgets_DEFINITIONS})

add_definitions(-DDEBUG_GL)

file(GLOB CPP_SOURCES 
		${CMAKE_SOURCE_DIR}/src/main.cpp
 	)


file(GLOB CPP_SOURCES_CORE
    ${CMAKE_SOURCE_DIR}/src/Core/ImageWriter.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/NodeParser.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/OpenGlWindow.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/OptixScene.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/OptixSceneAdaptive.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/PinholeCamera.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/RenderMath.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/RuntimeCompiler.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/command.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/inputhandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/keyframecomponent.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/rendercontext.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/stringutilities.cpp
  )

file(GLOB CPP_SOURCES_GUI
    ${CMAKE_SOURCE_DIR}/src/GUI/gridscene.cpp
    ${CMAKE_SOURCE_DIR}/src/GUI/mainwindow.cpp
    ${CMAKE_SOURCE_DIR}/src/GUI/qframebuffer.cpp
    ${CMAKE_SOURCE_DIR}/src/GUI/qtimelinewidget.cpp
    ${CMAKE_SOURCE_DIR}/src/GUI/highlighter.cpp
    ${CMAKE_SOURCE_DIR}/src/GUI/QColorRampEditor.cpp
    ${CMAKE_SOURCE_DIR}/src/GUI/qrenderwindow.cpp
    ${CMAKE_SOURCE_DIR}/src/GUI/testglwidget.cpp

    ${CMAKE_SOURCE_DIR}/src/GUI/nodes/distanceopnode.cpp
    ${CMAKE_SOURCE_DIR}/src/GUI/nodes/terminatenode.cpp
    
    # ${CMAKE_SOURCE_DIR}/src/GUI/nodegraph/*.cpp
  )

file(GLOB CPP_SOURCES_SDFOPS
    ${CMAKE_SOURCE_DIR}/src/SDFOps/Base_SDFOP.cpp
    
    ${CMAKE_SOURCE_DIR}/src/SDFOps/DistDeformer/Blend_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/DistDeformer/Displace_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/DistOp/DistOpInterface_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/DistOp/Intersection_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/DistOp/Subtraction_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/DistOp/Union_SDFOP.cpp

    ${CMAKE_SOURCE_DIR}/src/SDFOps/DomainDeformer/Bend_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/DomainDeformer/Twist_SDFOP.cpp
    
    ${CMAKE_SOURCE_DIR}/src/SDFOps/DomainOp/Repetition_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/DomainOp/Scale_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/DomainOp/Transform_SDFOP.cpp

    ${CMAKE_SOURCE_DIR}/src/SDFOps/Primitive/Box_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/Primitive/Cone_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/Primitive/Mandelbulb_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/Primitive/Sphere_SDFOP.cpp   
    ${CMAKE_SOURCE_DIR}/src/SDFOps/Primitive/Capsule_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/Primitive/Cylinder_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/Primitive/Menger_SDFOP.cpp
    ${CMAKE_SOURCE_DIR}/src/SDFOps/Primitive/Torus_SDFOP.cpp

  )

file(GLOB CPP_SOURCES_SUTIL
    ${CMAKE_SOURCE_DIR}/include/sutil/*.cpp
  )


file(GLOB CPP_SOURCES 
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CPP_SOURCES_CORE} ${CPP_SOURCES_GUI} ${CPP_SOURCES_SDFOPS} ${CPP_SOURCES_SUTIL}
  )

file(GLOB HEADERS_CORE
    ${CMAKE_SOURCE_DIR}/include/Core/ImageWriter.h
    ${CMAKE_SOURCE_DIR}/include/Core/NodeParser.h
    ${CMAKE_SOURCE_DIR}/include/Core/OpenGlWindow.h
    ${CMAKE_SOURCE_DIR}/include/Core/OptixHeaders.h 
    ${CMAKE_SOURCE_DIR}/include/Core/OptixScene.h
    ${CMAKE_SOURCE_DIR}/include/Core/OptixSceneAdaptive.h
    ${CMAKE_SOURCE_DIR}/include/Core/PinholeCamera.h
    ${CMAKE_SOURCE_DIR}/include/Core/RenderMath.h
    ${CMAKE_SOURCE_DIR}/include/Core/RuntimeCompiler.h
    ${CMAKE_SOURCE_DIR}/include/Core/command.h
    ${CMAKE_SOURCE_DIR}/include/Core/commonStructs.h
    ${CMAKE_SOURCE_DIR}/include/Core/inputhandler.h
    ${CMAKE_SOURCE_DIR}/include/Core/keyframecomponent.h
    ${CMAKE_SOURCE_DIR}/include/Core/macrohelpers.h
    ${CMAKE_SOURCE_DIR}/include/Core/rendercontext.h
    ${CMAKE_SOURCE_DIR}/include/Core/stringutilities.h
  )

file(GLOB HEADERS_GUI 
  ${CMAKE_SOURCE_DIR}/include/GUI/QColorRampEditor.h
    ${CMAKE_SOURCE_DIR}/include/GUI/gridscene.h
    ${CMAKE_SOURCE_DIR}/include/GUI/highlighter.h
    ${CMAKE_SOURCE_DIR}/include/GUI/mainwindow.h
    ${CMAKE_SOURCE_DIR}/include/GUI/qframebuffer.h
    ${CMAKE_SOURCE_DIR}/include/GUI/qrenderwindow.h
    ${CMAKE_SOURCE_DIR}/include/GUI/qtimelineanimated.h
    ${CMAKE_SOURCE_DIR}/include/GUI/testglwidget.h

        ${CMAKE_SOURCE_DIR}/include/GUI/nodes/distanceopnode.h
        ${CMAKE_SOURCE_DIR}/include/GUI/nodes/terminatenode.h

        # ${CMAKE_SOURCE_DIR}/include/GUI/nodegraph/*.h
  )

file(GLOB HEADERS_SDFOPS 
    ${CMAKE_SOURCE_DIR}/include/SDFOps/Base_SDFOP.h
    
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DistDeformer/Blend_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DistDeformer/Displace_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DistOp/DistOpInterface_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DistOp/Intersection_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DistOp/Subtraction_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DistOp/Union_SDFOP.h

    ${CMAKE_SOURCE_DIR}/include/SDFOps/DomainDeformer/Bend_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DomainDeformer/Twist_SDFOP.h
    
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DomainOp/Repetition_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DomainOp/Scale_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DomainOp/Transform_SDFOP.h

    ${CMAKE_SOURCE_DIR}/include/SDFOps/Primitive/Box_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/Primitive/Cone_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/Primitive/Mandelbulb_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/Primitive/Sphere_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/Primitive/Capsule_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/Primitive/Cylinder_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/Primitive/Menger_SDFOP.h
    ${CMAKE_SOURCE_DIR}/include/SDFOps/Primitive/Torus_SDFOP.h
  )


file(GLOB VERTEX_SOURCES ${CMAKE_SOURCE_DIR}/shaders/*.frag)
file(GLOB FRAGMENT_SOURCES ${CMAKE_SOURCE_DIR}/shaders/*.vert)

file(GLOB KERNEL_HEADERS ${CMAKE_SOURCE_DIR}/kernel/*.h)
file(GLOB KERNEL_SOURCES ${CMAKE_SOURCE_DIR}/kernel/*.cu)


file(GLOB HEADERS 
    ${HEADERS_CORE}
    ${HEADERS_GUI}
    ${HEADERS_SDFOPS}
    ${CMAKE_SOURCE_DIR}/kernel/*.h
  )

include_directories(
    ${CMAKE_SOURCE_DIR}/include/Core
    ${CMAKE_SOURCE_DIR}/include/GUI
    ${CMAKE_SOURCE_DIR}/include/GUI/nodes
    # ${CMAKE_SOURCE_DIR}/include/GUI/nodegraph
    ${CMAKE_SOURCE_DIR}/include/SDFOps
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DistDeformer
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DistOp
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DomainOp
    ${CMAKE_SOURCE_DIR}/include/SDFOps/DomainDeformer
    ${CMAKE_SOURCE_DIR}/include/SDFOps/Primitive
    ${CMAKE_SOURCE_DIR}/include/sutil

    ${CMAKE_SOURCE_DIR}/kernel
  )


CUDA_INCLUDE_DIRECTORIES(${OPTIX_INCLUDE_DIRS})

CUDA_COMPILE_PTX( CUDA_PTX_FILES ${KERNEL_SOURCES}
                  OPTIONS --use_fast_math
)

add_custom_target(ptx ALL
    DEPENDS ${CUDA_PTX_FILES}
    SOURCES ${KERNEL_SOURCES} )

set(CUDA_CU_FILES)
foreach(_file ${CUDA_PTX_FILES})
  # Fix the weird generated name from findCUDA
  STRING(REGEX REPLACE "cuda_compile_ptx_*generated_" "ptx/" cuda_file ${_file})
  set(CUDA_CU_FILES ${CUDA_CU_FILES} ${cuda_file})
  
  get_filename_component(PTXDIR ${cuda_file} DIRECTORY)

  add_custom_command(TARGET ptx POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${PTXDIR}
      COMMAND ${CMAKE_COMMAND} -E rename ${_file} ${cuda_file} DEPENDS ${_file}
      COMMENT "Moving ${_file} -> ${cuda_file}\n" VERBATIM 
  )
endforeach()

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${HEADERS} ${VERTEX_SOURCES} ${FRAGMENT_SOURCES})

source_group("Shaders\\Vertex"  FILES ${VERTEX_SOURCES})
source_group("Shaders\\Fragment"  FILES ${FRAGMENT_SOURCES})
source_group("Kernels"  FILES ${KERNEL_SOURCES})
source_group("Kernels\\include" FILES ${KERNEL_HEADERS})

source_group("Core"   FILES ${CPP_SOURCES_CORE} )
source_group("GUI"    FILES ${CPP_SOURCES_GUI} )
source_group("SDFOps" FILES ${CPP_SOURCES_SDFOPS} )
source_group("Core\\include"    FILES ${HEADERS_CORE} )
source_group("GUI\\include"   FILES ${HEADERS_GUI} )
source_group("SDFOps\\include"  FILES ${HEADERS_SDFOPS} )

# target_link_libraries(romanesco SDL2 opengl32 OpenCL::OpenCL)
target_link_libraries(${PROJECT_NAME}
                    ${Qt5Widgets_LIBRARIES}
                    ${OPENGL_gl_LIBRARY}
                    ${CUDA_LIBRARIES}
                    ${OPTIX_LIBRARIES}
                    ${OPENEXR_LIBRARIES}
                    ${Boost_LIBRARIES}
									  )

# @ToDo Setup this deployment stuff for Optix/CUDA too
# http://stackoverflow.com/questions/36981942/how-to-use-add-custom-command-to-copy-dlls-using-generator-expressions

# Windows specific build steps
if(WIN32)
    # Run winddeployqt if it can be found
    find_program(WINDEPLOYQT_EXECUTABLE NAMES windeployqt HINTS ${QTDIR} ENV QTDIR PATH_SUFFIXES bin)
    message( "Configuring Qt deployment with ${WINDEPLOYQT_EXECUTABLE}" )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${WINDEPLOYQT_EXECUTABLE} $<TARGET_FILE:${PROJECT_NAME}>)
endif()

# install(TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_BINARY_BIN}/bin)