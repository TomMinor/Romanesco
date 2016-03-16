TARGET=optixqt
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

QT+=gui opengl core

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

SOURCES += src/*.cpp \
           src/gui/*.cpp \
           src/gui/nodes/*.cpp \
           src/SDFOps/*.cpp \
           src/SDFOps/DistDeformer/*.cpp \
           src/SDFOps/DistOp/*.cpp \
           src/SDFOps/DomainOp/*.cpp \
           src/SDFOps/DomainDeformer/*.cpp \
           src/SDFOps/Primitive/*.cpp

 
HEADERS += include/*.h \
           include/gui/*.h \
           include/gui/nodes/*.h \
           include/SDFOps/*.h \
           include/SDFOps/DistDeformer/*.h \
           include/SDFOps/DistOp/*.h \
           include/SDFOps/DomainOp/*.h \
           include/SDFOps/DomainDeformer/*.h \
           include/SDFOps/Primitive/*.h

OTHER_FILES += shaders/*


INCLUDEPATH += ./include
INCLUDEPATH += ./include/gui
INCLUDEPATH += ./include/SDFOps
INCLUDEPATH += ./include/SDFOps/DistDeformer
INCLUDEPATH += ./include/SDFOps/DomainDeformer
INCLUDEPATH += ./include/SDFOps/DistOp
INCLUDEPATH += ./include/SDFOps/DomainOp
INCLUDEPATH += ./include/SDFOps/Primitive

INCLUDEPATH += /opt/local/include
DESTDIR=./

CONFIG += console
CONFIG -= app_bundle
 
LIBS += -lpthread -lIlmImf -lHalf

macx:INCLUDEPATH+=/usr/local/include/
unix:LIBS += -L/usr/local/lib


CUDA_SOURCES += kernel/*.cu
OTHER_FILES += kernel/*.h

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
