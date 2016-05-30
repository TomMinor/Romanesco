TARGET=romanesco
OBJECTS_DIR=obj

# as I want to support 4.8 and 5 this will set a flag for some of the mac stuff
# mainly in the types.h file for the setMacVisual which is native in Qt5
isEqual(QT_MAJOR_VERSION, 5) {
        cache()
        DEFINES +=QT5BUILD
}


UI_DIR=ui
MOC_DIR=moc

CONFIG += c++11
CONFIG += debug_and_releas
CONFIG -= app_bundle

# Force debug for now
QMAKE_CXXFLAGS += -g

QT+= core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets


#SOURCES += \
#src/main.cpp \
#\
#src/Core/OptixScene.cpp \
#src/Core/NodeParser.cpp \
#src/Core/OptixSceneAdaptive.cpp \
#src/Core/PinholeCamera.cpp \
#src/Core/RuntimeCompiler.cpp \
#src/Core/ViewportWindow.cpp \
#src/Core/RenderMath.cpp \
#src/Core/ImageWriter.cpp \
#src/Core/OpenGlWindow.cpp \
#src/Core/stringutilities.cpp \
#\
#src/GUI/gridscene.cpp \
#src/GUI/mainwindow.cpp \
#src/GUI/qtimelinewidget.cpp \
#src/GUI/testglwidget.cpp \
#src/GUI/qtimeslider.cpp \
#\
#src/GUI/nodegraph/qneblock.cpp \
#src/GUI/nodegraph/qneconnection.cpp \
#src/GUI/nodegraph/qneport.cpp \
#src/GUI/nodegraph/qnodeseditor.cpp \
#\
#src/GUI/nodes/distanceopnode.cpp \
#src/GUI/nodes/terminatenode.cpp \
#\
#src/SDFOps/Base_SDFOP.cpp \
#src/SDFOps/DistDeformer/Blend_SDFOP.cpp \
#src/SDFOps/DistDeformer/Displace_SDFOP.cpp \
#src/SDFOps/DistOp/DistOpInterface_SDFOP.cpp \
#src/SDFOps/DistOp/Intersection_SDFOP.cpp \
#src/SDFOps/DistOp/Subtraction_SDFOP.cpp \
#src/SDFOps/DistOp/Union_SDFOP.cpp \
#\
#src/SDFOps/DomainDeformer/Twist_SDFOP.cpp \
#src/SDFOps/DomainDeformer/Bend_SDFOP.cpp \
#\
#src/SDFOps/Primitive/Bend_SDFOP.cpp \
#src/SDFOps/Primitive/Box_SDFOP.cpp \
#src/SDFOps/Primitive/Capsule_SDFOP.cpp \
#src/SDFOps/Primitive/Cone_SDFOP.cpp \
#src/SDFOps/Primitive/Cylinder_SDFOP.cpp \
#src/SDFOps/Primitive/Mandelbulb_SDFOP.cpp \
#src/SDFOps/Primitive/Menger_SDFOP.cpp \
#src/SDFOps/Primitive/Sphere_SDFOP.cpp \
#src/SDFOps/Primitive/Torus_SDFOP.cpp \
#src/SDFOps/Primitive/Twist_SDFOP.cpp

#HEADERS += \
#include/Core/OptixScene.h \
#include/Core/NodeParser.h \
#include/Core/OptixSceneAdaptive.h \
#include/Core/PinholeCamera.h \
#include/Core/RuntimeCompiler.h \
#include/Core/ViewportWindow.h \
#include/Core/RenderMath.h \
#include/Core/ImageWriter.h \
#include/Core/OpenGlWindow.h \
#include/Core/stringutilities.h \
#\
#include/GUI/gridscene.h \
#include/GUI/mainwindow.h \
#include/GUI/testglwidget.h \
#include/GUI/qtimeslider.h \
#include/GUI/qtimelineanimated.h \
#\
#include/GUI/nodegraph/qneblock.h \
#include/GUI/nodegraph/qneconnection.h \
#include/GUI/nodegraph/qneport.h \
#include/GUI/nodegraph/qnodeseditor.h \
#\
#include/GUI/nodes/distanceopnode.h \
#include/GUI/nodes/terminatenode.h \
#\
#include/SDFOps/Base_SDFOP.h \
#include/SDFOps/DistDeformer/Blend_SDFOP.h \
#include/SDFOps/DistDeformer/Displace_SDFOP.h \
#include/SDFOps/DistOp/DistOpInterface_SDFOP.h \
#include/SDFOps/DistOp/Intersection_SDFOP.h \
#include/SDFOps/DistOp/Subtraction_SDFOP.h \
#include/SDFOps/DistOp/Union_SDFOP.h \
#\
#include/SDFOps/DomainDeformer/Twist_SDFOP.h \
#include/SDFOps/DomainDeformer/Bend_SDFOP.h \
#\
#include/SDFOps/Primitive/Bend_SDFOP.h \
#include/SDFOps/Primitive/Box_SDFOP.h \
#include/SDFOps/Primitive/Capsule_SDFOP.h \
#include/SDFOps/Primitive/Cone_SDFOP.h \
#include/SDFOps/Primitive/Cylinder_SDFOP.h \
#include/SDFOps/Primitive/Mandelbulb_SDFOP.h \
#include/SDFOps/Primitive/Menger_SDFOP.h \
#include/SDFOps/Primitive/Sphere_SDFOP.h \
#include/SDFOps/Primitive/Torus_SDFOP.h \
#include/SDFOps/Primitive/Twist_SDFOP.h

SOURCES += src/main.cpp \
           src/Core/*.cpp \
           src/GUI/*.cpp \
           src/GUI/nodes/*.cpp \
           src/GUI/nodegraph/*.cpp \
           src/SDFOps/*.cpp \
           src/SDFOps/DistDeformer/*.cpp \
           src/SDFOps/DistOp/*.cpp \
           src/SDFOps/DomainOp/*.cpp \
           src/SDFOps/DomainDeformer/*.cpp \
           src/SDFOps/Primitive/*.cpp

HEADERS += include/Core/*.h \
                       include/GUI/*.h \
                       include/GUI/nodes/*.h \
                       include/GUI/nodegraph/*.h \
                       include/SDFOps/*.h \
                       include/SDFOps/DistDeformer/*.h \
                       include/SDFOps/DistOp/*.h \
                       include/SDFOps/DomainOp/*.h \
                       include/SDFOps/DomainDeformer/*.h \
                       include/SDFOps/Primitive/*.h \
                      kernel/*.h \
    kernel/cutil_matrix.h \
    kernel/romanescomath.h \
    kernel/romanescocore.h
OTHER_FILES += shaders/*

RESOURCES += \
    romanesco.qrc

INCLUDEPATH += ./include/Core
INCLUDEPATH += ./include/GUI
INCLUDEPATH += ./include/SDFOps
INCLUDEPATH += ./include/SDFOps/DistDeformer
INCLUDEPATH += ./include/SDFOps/DomainDeformer
INCLUDEPATH += ./include/SDFOps/DistOp
INCLUDEPATH += ./include/SDFOps/DomainOp
INCLUDEPATH += ./include/SDFOps/Primitive
INCLUDEPATH += ./include
INCLUDEPATH += /opt/local/include
INCLUDEPATH += ./kernel

DESTDIR=./

CONFIG += console
CONFIG -= app_bundle

# Link to OpenImageIO/OpenEXR
LIBS += -lpthread -lIlmImf -lHalf -lOpenImageIO -lOpenImageIO_Util

macx:INCLUDEPATH+=/usr/local/include/
unix:LIBS += -L/usr/local/lib
unix:LIBS += -L/home/i7245143/local/lib
LIBS += -lboost_system

#LIBS += -lSeExpr


CUDA_SOURCES += kernel/*.cu
OTHER_FILES += kernel/*.h
OTHER_FILES += scenes/*.cu

# Setup CUDA paths
linux:CUDA_DIR = $$system( dirname $(dirname $(which nvcc)) )
linux:CUDA_INCLUDE = $$CUDA_DIR/include
linux:CUDA_SDK = $$CUDA_DIR/samples
linux:CUDA_VERSION = $$system($$CUDA_DIR/bin/nvcc --version | grep release | grep -o -E '[0-9]\.[0-9]' | head -1)
INCLUDEPATH += $$CUDA_DIR/include
INCLUDEPATH += $$CUDA_DIR/common/inc/
INCLUDEPATH += $$CUDA_DIR/../shared/inc/

DEFINES += CUDA_INCLUDE_PATH=$$CUDA_INCLUDE

# Setup Optix paths
linux:OPTIX_DIR = $(HOME)/src/optix
PTX_DIR = ./ptx

linux:INCLUDEPATH += $$OPTIX_DIR/SDK/sutil
linux:INCLUDEPATH += $$OPTIX_DIR/SDK
linux:INCLUDEPATH += $$OPTIX_DIR/include

QMAKE_LIBDIR += $$CUDA_SDK/common/lib
QMAKE_LIBDIR += $$CUDA_DIR/lib64
QMAKE_LIBDIR += $$OPTIX_DIR/lib64
QMAKE_LIBDIR += $$OPTIX_DIR/build/lib

#Add our cuda and optix libraries
LIBS += -lcudart  -loptix -loptixu -lglut -lsutil

# nvcc flags (ptxas option verbose is always useful), add the PTX flags to compile optix files
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -ptx

# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

warning($$CUDA_INC)

lessThan(CUDA_VERSION, 7.0) {
    warning( "CUDA version is $$CUDA_VERSION, at least 7.0 is required for libnvrtc. Using system nvcc for runtime compilation." )

    #todo: Fix this so it's delimited properly
    #DEFINES += CUDA_INC=$$CUDA_INC
    DEFINES += CUDA_EXE=$$CUDA_DIR/bin/nvcc
} else {
    message( "Using libnvrtc for runtime compilation." )
    DEFINES += NVRTC_AVAILABLE
    LIBS += -lnvrtc
}

# Tweak arch according to  your GPU's compute capability
# Either run your device query in cuda/samples or look in section 6 here #http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#axzz3OzHV3KTV
#for optix you can only have one architechture when using the PTX flags when using the -ptx flag you dont want to have the -c flag for compiling
optix.input = CUDA_SOURCES
optix.output = $$PTX_DIR/${QMAKE_FILE_BASE}.cu.ptx
optix.commands = $$CUDA_DIR/bin/nvcc -m64 -gencode arch=compute_20,code=sm_20 $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
#use this line for debug code
#optix.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -gencode arch=compute_50,code=sm_50 $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
optix.CONFIG = target_predeps
optix.CONFIG += no_link
optix.dependency_type = TYPE_C
QMAKE_EXTRA_UNIX_COMPILERS += optix
