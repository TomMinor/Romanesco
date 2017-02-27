cmake_minimum_required(VERSION 3.7)

project(romanesco)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Find at least a 4.0 version of CUDA.
find_package(CUDA 4.0 REQUIRED)
find_package(OPTIX REQUIRED)
find_package(Qt5Widgets REQUIRED)

include_directories( ${Qt5Widgets_INCLUDE_DIRS} )
include_directories( ${Qt5Widgets_LIBRARIES} )

# We need add -DQT_WIDGETS_LIB when using QtWidgets in Qt 5.
add_definitions(${Qt5Widgets_DEFINITIONS})

# Executables fail to build with Qt 5 in the default configuration
# without -fPIE. We add that here.
set(CMAKE_CXX_FLAGS "${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")


include_directories( ${OPTIX_INCLUDE_DIR} )
link_directories( ${OPTIX_LIB} )



# set(SDL2_ROOT ${CMAKE_SOURCE_DIR}/thirdparty/SDL2-2.0.5)
# set(GLM_ROOT ${CMAKE_SOURCE_DIR}/thirdparty/glm)

# include_directories(${SDL2_ROOT}/include)
# include_directories(${GLM_ROOT})

# link_directories(${SDL2_ROOT}/lib/x86)

file(GLOB_RECURSE CPP_SOURCES_CORE
		${CMAKE_SOURCE_DIR}/src/Core/*.cpp
	)

file(GLOB_RECURSE CPP_SOURCES_GUI
		${CMAKE_SOURCE_DIR}/src/GUI/*.cpp
		${CMAKE_SOURCE_DIR}/src/GUI/nodes/*.cpp
		${CMAKE_SOURCE_DIR}/src/GUI/nodegraph/*.cpp
	
	)

file(GLOB_RECURSE CPP_SOURCES_SDFOPS
		${CMAKE_SOURCE_DIR}/src/SDFOps/*.cpp
		${CMAKE_SOURCE_DIR}/src/SDFOps/DistDeformer/*.cpp
		${CMAKE_SOURCE_DIR}/src/SDFOps/DistOp/*.cpp
		${CMAKE_SOURCE_DIR}/src/SDFOps/DomainOp/*.cpp
		${CMAKE_SOURCE_DIR}/src/SDFOps/DomainDeformer/*.cpp
		${CMAKE_SOURCE_DIR}/src/SDFOps/Primitive/*.cpp
	)

file(GLOB_RECURSE CPP_SOURCES 
		${CMAKE_SOURCE_DIR}/src/main.cpp
 		${CPP_SOURCES_CORE} ${CPP_SOURCES_GUI} ${CPP_SOURCES_SDFOPS}
 	)

file(GLOB_RECURSE HEADERS_CORE
		${CMAKE_SOURCE_DIR}/include/Core/*.h
	)

file(GLOB_RECURSE HEADERS_GUI 
        ${CMAKE_SOURCE_DIR}/include/GUI/*.h
        ${CMAKE_SOURCE_DIR}/include/GUI/nodes/*.h
        ${CMAKE_SOURCE_DIR}/include/GUI/nodegraph/*.h
	)

file(GLOB_RECURSE HEADERS_SDFOPS 
        ${CMAKE_SOURCE_DIR}/include/SDFOps/*.h
        ${CMAKE_SOURCE_DIR}/include/SDFOps/DistDeformer/*.h
        ${CMAKE_SOURCE_DIR}/include/SDFOps/DistOp/*.h
        ${CMAKE_SOURCE_DIR}/include/SDFOps/DomainOp/*.h
        ${CMAKE_SOURCE_DIR}/include/SDFOps/DomainDeformer/*.h
        ${CMAKE_SOURCE_DIR}/include/SDFOps/Primitive/*.h
	)

file(GLOB_RECURSE HEADERS 
		${CMAKE_SOURCE_DIR}/include/*.h
 		${HEADERS_CORE} ${HEADERS_GUI} ${HEADERS_SDFOPS}
 	)

file(GLOB_RECURSE VERTEX_SOURCES ${CMAKE_SOURCE_DIR}/shaders/*.frag)
file(GLOB_RECURSE FRAGMENT_SOURCES ${CMAKE_SOURCE_DIR}/shaders/*.vert)

file(GLOB_RECURSE KERNEL_HEADERS ${CMAKE_SOURCE_DIR}/kernel/*.h)
file(GLOB_RECURSE KERNEL_SOURCES ${CMAKE_SOURCE_DIR}/kernel/*.cu)

include_directories(${HEADERS})

add_executable(romanesco ${CPP_SOURCES} ${HEADERS})

source_group("Shaders\\Vertex"	FILES ${VERTEX_SOURCES})
source_group("Shaders\\Fragment"	FILES ${FRAGMENT_SOURCES})
source_group("Kernels"	FILES ${KERNEL_SOURCES})
source_group("Kernels\\include"	FILES ${KERNEL_HEADERS})

source_group("Core"		FILES ${CPP_SOURCES_CORE} )
source_group("GUI"		FILES ${CPP_SOURCES_GUI} )
source_group("SDFOps"	FILES ${CPP_SOURCES_SDFOPS} )
source_group("Core\\include"		FILES ${HEADERS_CORE} )
source_group("GUI\\include"		FILES ${HEADERS_GUI} )
source_group("SDFOps\\include"	FILES ${HEADERS_SDFOPS} )

# target_link_libraries(romanesco SDL2 opengl32 OpenCL::OpenCL)
target_link_libraries(romanesco ${Qt5Widgets_LIBRARIES})